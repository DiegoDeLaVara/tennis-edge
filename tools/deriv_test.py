"""Are Kalshi's total-games and game-spread markets consistent with Kalshi's own match price? Fair value comes from
the match mid (the market's own win probability) plus each pair's serve strength, through the calibrated
set/game engine (edge.markov, day-form mixture). Bets are taken at the real ask about an hour before the start.

  .venv\\Scripts\\python.exe -m tools.deriv_test [--series KXATPGTOTAL KXWTAGTOTAL KXATPGSPREAD]
Caches candles in work/deriv_candles.jsonl.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from edge import markov, sources as src
from tools.kalshi_backtest import load, fee, ll

W = Path(__file__).resolve().parents[1] / "work"
CF = W / "deriv_candles.jsonl"
SLAMS = ("australian open", "roland garros", "french open", "wimbledon", "us open")


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def settled(series):
    out = src.kalshi_markets(series, status="settled", max_pages=20)
    cur = ""
    for _ in range(30):
        p = {"series_ticker": series, "limit": 1000}
        if cur:
            p["cursor"] = cur
        d = src.kalshi("/historical/markets", p)
        ms = d.get("markets") or []
        out += ms
        cur = d.get("cursor") or ""
        if not cur or not ms:
            break
    return [m for m in out if src.fnum(m, "volume_fp") >= 50 and m.get("result") in ("yes", "no")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=["KXATPGTOTAL", "KXWTAGTOTAL", "KXATPGSPREAD"])
    ap.add_argument("--max", type=int, default=0, help="most recent N markets per series")
    args = ap.parse_args()
    src.KALSHI_PACE.gap = 1 / 8.0
    K = load()
    ev = {json.loads(l)["event"]: json.loads(l) for l in open(W / "kalshi_events.jsonl")}
    P = pd.read_csv(W / "recent_preds.csv", low_memory=False, usecols=["a", "b", "date", "pa", "pb"])
    tot = {}
    for r in P.itertuples():
        tot[(r.a, r.b)] = (r.date, r.pa + r.pb)
        tot[(r.b, r.a)] = (r.date, r.pa + r.pb)
    match = {}
    for r in K.itertuples():
        match[r.event.split("-", 1)[1]] = r
    hist_cut = ts(src.kalshi("/historical/cutoff").get("trades_created_ts", "2026-09-03T00:00:00Z"))
    jobs = []
    for s in args.series:
        ms = settled(s)
        n0 = len(jobs)
        for m in ms:
            suf = m["event_ticker"].split("-")[1]
            if suf in match and match[suf].qts:
                jobs.append((s, m))
        if args.max and len(jobs) - n0 > args.max:
            jobs = jobs[:n0] + sorted(jobs[n0:], key=lambda j: -ts(j[1].get("close_time") or j[1]["expected_expiration_time"]))[:args.max]
        print(s, "settled with volume", len(ms), "linked to a priced match", len(jobs) - n0, flush=True)
    have = {}
    if CF.exists():
        for l in CF.open():
            d = json.loads(l)
            have[d["t"]] = d["q"]
    todo = [(s, m) for s, m in jobs if m["ticker"] not in have]
    print("candles to fetch", len(todo), flush=True)

    def fetch(job):
        s, m = job
        r = match[m["event_ticker"].split("-")[1]]
        end = ts(m.get("close_time") or m["expected_expiration_time"])
        c = src.kalshi_candles(s, m["ticker"], r.qts - 2 * 86400, r.qts + 1, 60, historical=end < hist_cut)
        q = None
        for x in c:
            if x["end_period_ts"] <= r.qts and x["bid"] is not None and x["ask"] is not None and 0 <= x["bid"] < x["ask"] <= 1:
                q = [x["bid"], x["ask"]]
        return {"t": m["ticker"], "q": q}
    with CF.open("a") as fh, ThreadPoolExecutor(8) as ex:
        for i, d in enumerate(ex.map(fetch, todo)):
            fh.write(json.dumps(d) + "\n")
            have[d["t"]] = d["q"]
            if i % 500 == 0:
                fh.flush(); print(i, flush=True)

    rows = []
    for s, m in jobs:
        q = have.get(m["ticker"])
        strike = m.get("floor_strike")
        if not q or q[1] - q[0] > 0.10 or (strike is None and ("GTOTAL" in s or "GSPREAD" in s)):
            continue
        r = match[m["event_ticker"].split("-")[1]]
        e = ev.get(r.event, {})
        t0 = tot.get((r.ka, r.kb))
        total = t0[1] if t0 and abs((pd.Timestamp(t0[0]) - pd.Timestamp(r.date)).days) <= 7 else (1.25 if r.tour == "atp" else 1.12)
        bo = 5 if (r.tour == "atp" and any(x in (e.get("tourney") or "").lower() for x in SLAMS) and e.get("round") != "Q") else 3
        sd = markov.DAY_SD[r.tour]
        pa, pb = markov.solve_serve_mix(r.px, total, bo, sd)
        D = markov.mix_dist(pa, pb, bo, sd)
        if "SETWINNER" in s or "EXACTMATCH" in s:
            from edge.refresh import deriv_prob
            pr = deriv_prob(s, m, D, {"names": [e.get("a", ""), e.get("b", "")]})
            if pr is None:
                continue
            fair = pr[0]
            kind = "exact score" if "EXACT" in s else "set " + m["event_ticker"].rsplit("-", 1)[-1] + " winner"
        elif "GTOTAL" in s:
            fair = markov.prob(D, lambda sa, sb, t, d, *_: t > strike)
            kind = "total"
        else:
            sub = (m.get("yes_sub_title") or "").lower()
            la, lb = e.get("a", "").split()[-1].lower() if e.get("a") else "", e.get("b", "").split()[-1].lower() if e.get("b") else ""
            if la and la in sub and lb not in sub:
                w = 0
            elif lb and lb in sub and la not in sub:
                w = 1
            else:
                continue
            fair = markov.prob(D, lambda sa, sb, t, d, *_: (d if w == 0 else -d) > strike)
            kind = "spread"
        rows.append({"kind": kind, "series": s, "date": r.date, "y": 1 if m["result"] == "yes" else 0, "fair": fair,
                     "bid": q[0], "ask": q[1], "mid": (q[0] + q[1]) / 2, "strike": strike})
    R = pd.DataFrame(rows).sort_values("date")
    cut = R.date.iloc[int(len(R) * 0.5)]
    print(f"\npriced derivative markets: {len(R)}  (split at {cut})")
    for kind, G in R.groupby("kind"):
        y = G.y.values
        print(f"\n== {kind}: n={len(G)}  log-loss Kalshi mid {ll(y, G.mid.clip(0.01, 0.99).values):.4f}  our fair {ll(y, G.fair.clip(0.01, 0.99).values):.4f}"
              f"  average {ll(y, ((G.mid + G.fair) / 2).clip(0.01, 0.99).values):.4f}")
        print(f"   calibration: mean fair {G.fair.mean():.3f}  mean Kalshi mid {G.mid.mean():.3f}  actual {G.y.mean():.3f}")
        for part, H in (("first half", G[G.date < cut]), ("second half", G[G.date >= cut])):
            for th in (0.03, 0.06, 0.10):
                res = []
                for x in H.itertuples():
                    for side in (1, 0):
                        p = x.fair if side else 1 - x.fair
                        price = x.ask if side else 1 - x.bid
                        if 0.05 <= price <= 0.95 and p / (price + fee(price)) - 1 >= th:
                            c = price + fee(price)
                            win = x.y == side
                            res.append((1 - c) / c if win else -1.0)
                if res:
                    a = np.array(res)
                    print(f"   {part:11s} edge >= {int(th * 100):2d}%: n={len(a):5d} ROI {a.mean() * 100:+6.1f}% ± {a.std() / np.sqrt(len(a)) * 100:4.1f}")


if __name__ == "__main__":
    main()
