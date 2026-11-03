"""Is the early Kalshi line beatable at prices you could actually take? For each held-out match, read the hourly
bid/ask path of player A's market and back the model at the real ask at several points before the start.

  .venv\\Scripts\\python.exe -m tools.early_test [--n 4000]
Caches candles in work/early_candles.jsonl.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from edge import sources as src
from tools.kalshi_backtest import load, fee, ll

W = Path(__file__).resolve().parents[1] / "work"
C = W / "early_candles.jsonl"


def fetch(e, hist_cut):
    ms = src.kalshi("/historical/markets", {"event_ticker": e["event"]}).get("markets") or \
        src.kalshi("/markets", {"event_ticker": e["event"]}).get("markets") or []
    if len(ms) != 2:
        return {"event": e["event"], "c": []}
    a = sorted(ms, key=lambda m: m["ticker"])[0]
    c = src.kalshi_candles(e["series"], a["ticker"], e["open"] - 3600, e["start"] + 1, 60, historical=e["end"] < hist_cut)
    return {"event": e["event"], "c": [[x["end_period_ts"], x["bid"], x["ask"], x["volume"]] for x in c]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    K = load().sort_values("date")
    K = K[K.p_model.notna()]
    cut = K.date.iloc[int(len(K) * 0.6)]
    V = K[K.date >= cut]
    ev = {json.loads(l)["event"]: json.loads(l) for l in open(W / "kalshi_events.jsonl")}
    have = {json.loads(l)["event"] for l in C.open()} if C.exists() else set()
    todo = [ev[x] for x in V.event if x in ev and x not in have and ev[x].get("open")][:args.n]
    print("held-out matches", len(V), "to fetch", len(todo), flush=True)
    hist_cut = src.datetime.fromisoformat(src.kalshi("/historical/cutoff").get("trades_created_ts", "2026-09-03T00:00:00Z").replace("Z", "+00:00")).timestamp()
    with C.open("a") as fh, ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(lambda e: fetch(e, hist_cut), todo)):
            fh.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                fh.flush(); print(i, flush=True)
    candles = {}
    for l in C.open():
        d = json.loads(l)
        candles[d["event"]] = d["c"]

    def quote_at(c, t, max_spread):
        q = None
        for ts, b, a, v in c:
            if ts > t:
                break
            if b is not None and a is not None and 0 < b < a <= 1 and a - b <= max_spread:
                q = (b, a, ts)
        return q

    def first_quote(c, max_spread):
        for ts, b, a, v in c:
            if b is not None and a is not None and 0 < b < a <= 1 and a - b <= max_spread:
                return (b, a, ts)
        return None

    rows = []
    for r in V.itertuples():
        e, c = ev.get(r.event), candles.get(r.event)
        if not e or not c:
            continue
        rec = {"y": r.y, "p": r.p_model, "mid": r.px, "start": e["start"], "open": e["open"]}
        for name, fn in (("first", lambda: first_quote(c, 0.06)),
                         ("t-24h", lambda: quote_at(c, e["start"] - 24 * 3600, 0.06)),
                         ("t-12h", lambda: quote_at(c, e["start"] - 12 * 3600, 0.06)),
                         ("t-6h", lambda: quote_at(c, e["start"] - 6 * 3600, 0.06)),
                         ("t-2h", lambda: quote_at(c, e["start"] - 2 * 3600, 0.06))):
            q = fn()
            if q and q[2] < e["start"] - 1800:
                rec[name] = q
        rows.append(rec)
    print(f"\nheld-out matches with an hourly book: {len(rows)}  (verify period from {cut})")
    for name in ("first", "t-24h", "t-12h", "t-6h", "t-2h"):
        R = [x for x in rows if name in x]
        if not R:
            continue
        y = np.array([x["y"] for x in R])
        mids = np.array([(x[name][0] + x[name][1]) / 2 for x in R])
        pm = np.array([x["p"] for x in R])
        hrs = np.median([(x["start"] - x[name][2]) / 3600 for x in R])
        line = f"  {name:6s} n={len(R):5d} (median {hrs:4.1f}h before start)  log-loss: that mid {ll(y, mids):.4f}  model {ll(y, pm):.4f}  pre-match mid {ll(y, np.array([x['mid'] for x in R])):.4f}"
        print(line)
        for th in (0.03, 0.06, 0.10):
            res, closer = [], []
            for x in R:
                b, a, _ = x[name]
                for side in (0, 1):
                    ps = x["p"] if side == 0 else 1 - x["p"]
                    price = a if side == 0 else 1 - b
                    if not (0.05 <= price <= 0.95) or ps / (price + fee(price)) - 1 < th:
                        continue
                    c_ = price + fee(price)
                    win = (x["y"] == 1) if side == 0 else (x["y"] == 0)
                    res.append((1 - c_) / c_ if win else -1.0)
                    close = x["mid"] if side == 0 else 1 - x["mid"]
                    closer.append(close - price)
            if res:
                a_ = np.array(res)
                print(f"      back model edge >= {int(th * 100):2d}%: n={len(a_):5d} ROI {a_.mean() * 100:+6.1f}% ± {a_.std() / np.sqrt(len(a_)) * 100:4.1f}"
                      f"   CLV (pre-match mid - price paid) {np.mean(closer) * 100:+.1f}c")


if __name__ == "__main__":
    main()
