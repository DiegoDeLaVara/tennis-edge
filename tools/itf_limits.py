r"""Resting limit orders at the ITF open: instead of paying the ask, post a bid at our price and wait.

ITF prices drift toward our model after the open (tools/itf_openers, itf_early2), but the drift (1-2 pts) is smaller
than the opening spread (~6c), so taker bets lose. A resting maker order pays no spread and no fee (Kalshi ITF series
charge makers nothing), at the cost of adverse selection: you get filled more often when the price moves against you.

Simulation per match and side, placed at the first tradable quote (open):
  fair = model+open blend (fitted per experience band on the OTHER half -> out of sample both ways)
  "improve": bid 1c above the open bid, only if fair beats it by the edge threshold and it stays below the open ask
  "deep":    bid at fair / (1 + edge), only if that is below the open ask
  fill:      a later hourly candle (before start - 30 min) has that side's ask low <= our bid (improve: we are first in
             the queue) / <= bid - 1c (deep, conservative: price traded through us). Unfilled orders cancel at start.
Return per filled order: (1 - L) / L if won else -1. CLV = closing mid - L.

  .venv\Scripts\python.exe -m tools.itf_limits --fetch      (hourly candles -> work/itf_hourly.jsonl, resumable)
  .venv\Scripts\python.exe -m tools.itf_limits

Result (2026-11-07, 6,637 matches): every variant lost in both halves. Bids 1c above the open bid filled 16-29% of the
time and lost 4-21% (closing-line value -0.9 to -1.6c); deeper bids at fair minus 3-10% filled 11-26% and lost 7-14%
(CLV -2.2 to -3.0c). Adverse selection: the orders that get filled are the ones the price moves against.
"""
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from edge import sources as src
from tools.itf_early2 import BANDS, X, data

OUT = Path("work/itf_hourly.jsonl")
HIST_CUT = 0


def fl(x, k):
    v = (x or {}).get(k + "_dollars", (x or {}).get(k))
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def fetch_one(e, p):
    tail = e["event"].split("-")[-1]
    codes = sorted({tail[-6:-3], tail[-3:]})
    if len(codes) != 2:
        return None
    hist = p["start"] < HIST_CUT
    tk = f"{e['event']}-{codes[0]}"
    path = f"/historical/markets/{tk}/candlesticks" if hist else f"/series/{e['series']}/markets/{tk}/candlesticks"
    d = src.kalshi(path, {"start_ts": int(p["open"]["ts"] - 3600), "end_ts": int(p["start"]), "period_interval": 60})
    C = []
    for x in d.get("candlesticks") or []:
        b, a = x.get("yes_bid"), x.get("yes_ask")
        C.append([x["end_period_ts"], fl(b, "low"), fl(b, "high"), fl(b, "close"), fl(a, "low"), fl(a, "high"), fl(a, "close")])
    return {"event": e["event"], "c": C}


def fetch():
    global HIST_CUT
    cut = src.kalshi("/historical/cutoff")
    HIST_CUT = datetime.fromisoformat(cut.get("market_settled_ts", "2026-09-08T00:00:00Z").replace("Z", "+00:00")).timestamp()
    src.KALSHI_PACE.gap = 1 / 3.0
    done = {json.loads(l)["event"] for l in OUT.open()} if OUT.exists() else set()
    P = {json.loads(l)["event"]: json.loads(l) for l in open("work/itf_paths.jsonl")}
    want = set(data().event)
    ev = [json.loads(l) for l in open("work/kalshi_events.jsonl")]
    ev = [e for e in ev if e["event"] in want and e["event"] not in done and P.get(e["event"], {}).get("open")]
    ev = list({e["event"]: e for e in ev}.values())
    print(f"to fetch: {len(ev)} ({len(done)} cached)", flush=True)
    with OUT.open("a") as fh, ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(fetch_one, e, P[e["event"]]) for e in ev]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                if r:
                    fh.write(json.dumps(r) + "\n")
            except Exception as err:  # noqa
                print("err", err, flush=True)
            if i % 250 == 0:
                fh.flush(); print(f"  {i}/{len(ev)}", flush=True)


def analyse():
    D = data()
    H = {json.loads(l)["event"]: json.loads(l)["c"] for l in OUT.open()}
    P = {json.loads(l)["event"]: json.loads(l) for l in open("work/itf_paths.jsonl")}
    D = D[D.event.isin(H)].reset_index(drop=True)
    D["half"] = (D.index >= len(D) // 2).astype(int)
    D["p_bl"] = np.nan
    for _, lo, hi in BANDS:
        b = (D.nmin >= lo) & (D.nmin < hi)
        for tr, te in ((0, 1), (1, 0)):
            mtr, mte = b & (D.half == tr), b & (D.half == te)
            m = LogisticRegression(C=100).fit(X(D[mtr]), D.y[mtr])
            D.loc[mte, "p_bl"] = m.predict_proba(X(D[mte]))[:, 1]
    print(f"{len(D):,} ITF matches with hourly pre-match candles; fair = out-of-sample model+open blend")
    cents = lambda x: math.floor(x * 100 + 1e-9) / 100
    for mode in ("improve", "deep"):
        for thr in (0.0, 0.03, 0.06, 0.10):
            if mode == "deep" and thr == 0.0:
                continue
            parts, rate_parts = [], []
            for h in (0, 1):
                rets, clv, tried = [], [], 0
                for r in D[D.half == h].itertuples():
                    op = P[r.event]["open"]; t0 = op["ts"]; t_end = P[r.event]["start"] - 1800
                    C = [c for c in H[r.event] if t0 < c[0] <= t_end]
                    for side in (0, 1):
                        fair = r.p_bl if side == 0 else 1 - r.p_bl
                        bid = op["bid"] if side == 0 else 1 - op["ask"]
                        ask = op["ask"] if side == 0 else 1 - op["bid"]
                        if mode == "improve":
                            L = round(bid + 0.01, 2)
                            if L >= ask or fair / L - 1 < max(thr, 0.005):
                                continue
                        else:
                            L = cents(fair / (1 + thr))
                            if L >= ask:
                                continue
                        if not 0.08 <= L <= 0.92:
                            continue
                        tried += 1
                        need = L if mode == "improve" else L - 0.01
                        filled = False
                        for c in C:
                            alo = c[4] if side == 0 else (None if c[2] is None else 1 - c[2])
                            if alo is not None and alo <= need + 1e-9:
                                filled = True; break
                        if not filled:
                            continue
                        won = r.y == (1 if side == 0 else 0)
                        rets.append((1 - L) / L if won else -1.0)
                        clv.append(((r.p_close if side == 0 else 1 - r.p_close) - L) * 100)
                x = np.array(rets)
                if len(x) < 5:
                    parts.append(f"half {h}: fills {len(x)}/{tried}"); continue
                parts.append(f"half {h}: filled {len(x):>4}/{tried:<5} ({len(x) / max(tried, 1):4.0%}) ROI {x.mean():+6.1%} ± {x.std(ddof=1) / math.sqrt(len(x)):4.1%} CLV {np.mean(clv):+.1f}c")
            print(f"   {mode:<8} edge >= {thr:>4.0%}   " + "   ".join(parts))


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        analyse()
