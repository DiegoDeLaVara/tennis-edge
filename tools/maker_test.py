"""How did the resting side of real Kalshi tennis trades do? Every fill has a taker and a maker; the maker bought the
opposite side at the traded price, paid no fee (or a quarter fee on tour-level series) and suffered whatever adverse
selection there was. If makers buying favourites (or selling longshots) earn money, a patient limit-order strategy can.

  .venv\\Scripts\\python.exe -m tools.maker_test [--n 2500]
Uses pre-match trades only (before the detected start). Caches per-match aggregates in work/maker.jsonl.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from edge import sources as src
from tools.kalshi_collect import trades

W = Path(__file__).resolve().parents[1] / "work"
OUT = W / "maker.jsonl"
MAKER_FEE = {"KXATPMATCH": 0.0175, "KXWTAMATCH": 0.0175}          # others: makers pay nothing
BUCKETS = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def one(e, hist_cut):
    ms = src.kalshi("/historical/markets", {"event_ticker": e["event"]}).get("markets") or \
        src.kalshi("/markets", {"event_ticker": e["event"]}).get("markets") or []
    if len(ms) != 2 or any(m.get("result") not in ("yes", "no") for m in ms):
        return None
    is_hist = e["end"] < hist_cut
    agg = {}                        # bucket -> [cost, pnl, contracts]
    mf = MAKER_FEE.get(e["series"], 0.0)
    for m in ms:
        won_yes = m["result"] == "yes"
        for t in trades(m["ticker"], is_hist, e["start"]):
            cnt = float(t.get("count_fp") or 0)
            yes = float(t.get("yes_price_dollars") or 0)
            if cnt <= 0 or not 0 < yes < 1:
                continue
            if t.get("taker_side") == "yes":       # maker sold yes = bought no
                q, win = 1 - yes, not won_yes
            else:
                q, win = yes, won_yes
            pnl = cnt * ((1 - q) if win else -q) - cnt * mf * q * (1 - q)
            b = int(np.digitize(q, BUCKETS))
            a = agg.setdefault(b, [0.0, 0.0, 0.0])
            a[0] += cnt * q; a[1] += pnl; a[2] += cnt
    return {"event": e["event"], "series": e["series"], "end": e["end"], "agg": agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    args = ap.parse_args()
    hist_cut = ts(src.kalshi("/historical/cutoff").get("trades_created_ts", "2026-09-03T00:00:00Z"))
    ev = [json.loads(l) for l in open(W / "kalshi_events.jsonl")]
    ev = [e for e in ev if e.get("quote") and e.get("start") and e.get("end")]
    ev.sort(key=lambda e: -e["end"])
    done = {json.loads(l)["event"] for l in OUT.open()} if OUT.exists() else set()
    per = {}
    for e in ev:
        per.setdefault(e["series"], [])
        if len(per[e["series"]]) < args.n // 3:
            per[e["series"]].append(e)
    todo = [e for L in per.values() for e in L if e["event"] not in done]
    print("matches to fetch", len(todo), flush=True)
    with OUT.open("a") as fh, ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(lambda e: one(e, hist_cut), todo)):
            if r:
                fh.write(json.dumps(r) + "\n")
            if i % 250 == 0:
                fh.flush(); print(i, flush=True)
    rows = [json.loads(l) for l in OUT.open()]
    print(f"\nmatches with pre-match trades: {len(rows)}")
    for series in sorted({r["series"] for r in rows}):
        R = [r for r in rows if r["series"] == series]
        print(f"\n== {series} ({len(R)} matches), maker fee {MAKER_FEE.get(series, 0)}: what makers earned buying at each price")
        for b in range(1, len(BUCKETS)):
            cost = sum(r["agg"].get(str(b), [0, 0, 0])[0] for r in R)
            pnl = sum(r["agg"].get(str(b), [0, 0, 0])[1] for r in R)
            # match-level spread of returns, for a rough standard error
            per_match = [r["agg"][str(b)][1] / r["agg"][str(b)][0] for r in R if str(b) in r["agg"] and r["agg"][str(b)][0] > 0]
            if cost < 500:
                continue
            se = np.std(per_match) / np.sqrt(len(per_match)) if len(per_match) > 1 else 0
            print(f"   {BUCKETS[b - 1]:.2f}-{BUCKETS[b]:.2f}: ${cost:>12,.0f} risked over {len(per_match):4d} matches  maker ROI {pnl / cost * 100:+6.2f}%  (match-level ± {se * 100:.1f})")


if __name__ == "__main__":
    main()
