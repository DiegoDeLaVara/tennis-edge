r"""ITF opening lines: is Kalshi's first real ITF price beatable by our model, even though the closing price isn't?

Step 1 (--fetch): for every collected ITF event (work/kalshi_events.jsonl), download the hourly price path and keep
  the first tradable quote (two-sided, spread <= 15c) as the OPEN, plus quotes ~12h and ~3h before the start.
  Saved to work/itf_paths.jsonl (resumable).
Step 2: compare open vs close vs our model (log-loss, both halves), check whether the price drifts toward our
  model between open and close, and test betting our model's edge at the opening ask (ROI + closing-line value).

  .venv\Scripts\python.exe -m tools.itf_openers --fetch
  .venv\Scripts\python.exe -m tools.itf_openers
"""
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from edge import sources as src
from tools.kalshi_backtest import fee

PATHS = Path("work/itf_paths.jsonl")


def quote(c):
    b, a = c.get("bid"), c.get("ask")
    if b is None or a is None or not (0 < b < a < 1) or a - b > 0.15:
        return None
    return {"bid": b, "ask": a, "ts": c["end_period_ts"]}


def fetch_one(e):
    # the 'a' side market: tickers are EVENT-XXX; kalshi_collect sorted the pair by ticker and stored a = first
    # tickers are EVENT-<3-letter code>; the event name ends with both codes, and side 'a' is the first alphabetically
    tail = e["event"].split("-")[-1]
    codes = sorted({tail[-6:-3], tail[-3:]})
    C = []
    if len(codes) == 2:
        C = src.kalshi_candles(e["series"], f"{e['event']}-{codes[0]}", e["open"] - 3600, e["start"] + 1800, 60, historical=e["start"] < HIST_CUT)
    if not C:                                           # unusual codes: look the market up
        ms = src.kalshi("/markets", {"event_ticker": e["event"], "limit": 10}).get("markets") or []
        if len(ms) != 2:
            ms = src.kalshi("/historical/markets", {"event_ticker": e["event"], "limit": 10}).get("markets") or []
        ms = sorted(ms, key=lambda m: m["ticker"])
        if len(ms) != 2:
            return None
        C = src.kalshi_candles(e["series"], ms[0]["ticker"], e["open"] - 3600, e["start"] + 1800, 60, historical=e["start"] < HIST_CUT)
    Q = [(c["end_period_ts"], quote(c)) for c in C if c["end_period_ts"] <= e["start"]]
    Q = [(t, q) for t, q in Q if q]
    if not Q:
        return None
    def at(hours):
        cand = [q for t, q in Q if t <= e["start"] - hours * 3600]
        return cand[-1] if cand else None
    return {"event": e["event"], "open": Q[0][1], "q12": at(12), "q3": at(3), "close": Q[-1][1], "start": e["start"]}


HIST_CUT = 0


def fetch():
    global HIST_CUT
    cut = src.kalshi("/historical/cutoff")
    from datetime import datetime
    HIST_CUT = datetime.fromisoformat(cut.get("trades_created_ts", "2026-09-03T00:00:00Z").replace("Z", "+00:00")).timestamp()
    src.KALSHI_PACE.gap = 1 / 3.0
    done = {json.loads(l)["event"] for l in PATHS.open()} if PATHS.exists() else set()
    ev = [json.loads(l) for l in open("work/kalshi_events.jsonl")]
    from tools.itf_study import load_itf
    rated = set(load_itf().event)                       # only matches our model rated
    ev = [e for e in ev if "ITF" in e["series"] and e["event"] in rated and e["event"] not in done and e.get("open") and e.get("start")]
    print(f"to fetch: {len(ev)}", flush=True)
    with PATHS.open("a") as fh, ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(fetch_one, e) for e in ev]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                if r:
                    fh.write(json.dumps(r) + "\n")
            except Exception as err:  # noqa
                pass
            if i % 250 == 0:
                fh.flush(); print(f"  {i}/{len(ev)}", flush=True)


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def analyse():
    from tools.itf_study import load_itf
    D = load_itf()
    P = {}
    for l in PATHS.open():
        r = json.loads(l)
        P[r["event"]] = r
    D = D[D.event.isin(P)].copy().reset_index(drop=True)
    mid = lambda q: (q["bid"] + q["ask"]) / 2 if q else np.nan
    D["p_open"] = [mid(P[e]["open"]) for e in D.event]
    D["p_q12"] = [mid(P[e]["q12"]) for e in D.event]
    D["p_close"] = [mid(P[e]["close"]) for e in D.event]
    D["open_ask_a"] = [P[e]["open"]["ask"] for e in D.event]
    D["open_bid_a"] = [P[e]["open"]["bid"] for e in D.event]
    D["lead_h"] = [(P[e]["start"] - P[e]["open"]["ts"]) / 3600 for e in D.event]
    D = D.sort_values("date").reset_index(drop=True)
    D["half"] = (D.index >= len(D) // 2).astype(int)
    print(f"ITF matches with an opening quote and our model: {len(D):,}; open is a median {D.lead_h.median():.0f} h before the start")
    print("\n== Accuracy (log-loss, lower is better)")
    for h in (0, 1):
        g = D[D.half == h]
        g12 = g[g.p_q12.notna()]
        print(f"   half {h}: open {ll(g.y, g.p_open):.4f} | 12h before {ll(g12.y, g12.p_q12):.4f} (n={len(g12)}) | close {ll(g.y, g.p_close):.4f} | our model {ll(g.y, g.pm):.4f}")
    from sklearn.linear_model import LogisticRegression
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    X = np.c_[lg(D.pm.values), lg(D.p_open.values)]
    tr, te = (D.half == 0).values, (D.half == 1).values
    m = LogisticRegression(C=100).fit(X[tr], D.y.values[tr])
    pb = m.predict_proba(X[te])[:, 1]
    print(f"   model + open, fitted on half 0: weights model {m.coef_[0][0]:.2f}, open {m.coef_[0][1]:.2f}; half 1 log-loss {ll(D.y[te], pb):.4f} "
          f"vs open alone {ll(D.y[te], D.p_open[te]):.4f} vs close {ll(D.y[te], D.p_close[te]):.4f}")
    D["p_blend"] = m.predict_proba(X)[:, 1]

    print("\n== Does the price move toward our model between open and close?")
    for gap in (0.05, 0.10, 0.20):
        for h in (0, 1):
            g = D[(D.half == h) & ((D.pm - D.p_open).abs() >= gap)]
            mv = (g.p_close - g.p_open) * np.sign(g.pm - g.p_open)
            print(f"   model differs from open by {gap:.0%}+  half {h}: n={len(g):>5}  moved toward model {(mv > 0.005).mean():.0%}, away {(mv < -0.005).mean():.0%}, "
                  f"avg {mv.mean() * 100:+.1f} pts")

    print("\n== Bet our edge at the OPENING ask (fee included); CLV = closing mid minus entry")
    for name, col in (("model alone", "pm"), ("model + open blend", "p_blend")):
        for thr in (0.03, 0.06, 0.10):
            parts = []
            for h in (0, 1):
                g = D[D.half == h]
                pa = g[col].values
                ask_a, ask_b = g.open_ask_a.values, 1 - g.open_bid_a.values
                ev_a = pa / (ask_a + fee(ask_a)) - 1
                ev_b = (1 - pa) / (ask_b + fee(ask_b)) - 1
                side_a = ev_a >= ev_b
                ev = np.where(side_a, ev_a, ev_b)
                price = np.where(side_a, ask_a, ask_b)
                sel = (ev >= thr) & (price >= 0.08) & (price <= 0.92)
                won = np.where(side_a, g.y.values == 1, g.y.values == 0)[sel]
                c = price[sel] + fee(price[sel])
                r = np.where(won, (1 - c) / c, -1.0)
                clv = (np.where(side_a, g.p_close.values, 1 - g.p_close.values)[sel] - price[sel]) * 100
                parts.append(f"half {h}: n={sel.sum():>4} ROI {r.mean() if len(r) else 0:+6.1%} ± {r.std(ddof=1) / math.sqrt(max(len(r), 2)):4.1%} CLV {clv.mean() if len(clv) else 0:+.1f}c")
            print(f"   {name:<20} edge >= {thr:.0%}   " + "   ".join(parts))


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        analyse()
