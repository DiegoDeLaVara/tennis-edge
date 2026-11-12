r"""Backtest of the STAKED "Kalshi cheaper than Polymarket" strategy (config price_gap), never tested before.

1. Closed Polymarket tennis match markets (gamma API) paired with Kalshi events (work/kalshi_events.jsonl) by both
   players' names and date.
2. Polymarket's price at the moment of Kalshi's last pre-match quote, from the public CLOB price history
   (cache work/poly_hist.jsonl, resumable).
3. The live rule: Kalshi ask + fee at least 3c below Polymarket's price, Kalshi spread <= 4c, ask 10-90c,
   Polymarket volume >= $2,000 -> buy on Kalshi. Scored in two time halves, plus accuracy (which market is sharper)
   and the reverse question (when they disagree, which one is right?).

  .venv\Scripts\python.exe -m tools.poly_gap [--fetch] [--pages DAYS_BACK]

Result (2026-11-11, 12,265 Jun-Nov 2026 matches on both markets): Kalshi mid 0.5854, Polymarket 0.5859, the AVERAGE of
both 0.5843 (better than either: each market knows something the other doesn't). The live rule (Kalshi ask+fee 3c+
below Polymarket, Poly vol >= $2k, spread <= 4c) returned +12.4% / +16.9% in the two halves (pooled n=204, +14.7% +-
11.1%); 69% of its picks are underdogs (won 43-44% vs ~41% paid). 6 of 8 threshold variants positive in both halves.
Consistent and mechanism-based but not yet statistically proven - keep it at its small stake and watch the record.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from edge import smart
from edge.names import fold
from tools.kalshi_backtest import fee

CACHE = Path("work/poly_hist.jsonl")
CLOB = "https://clob.polymarket.com/prices-history"


def key2(a, b):
    la, lb = (fold(a) or [""])[-1], (fold(b) or [""])[-1]
    return tuple(sorted((la, lb)))


def fetch(pages):
    done = {json.loads(l)["kalshi"] for l in CACHE.open()} if CACHE.exists() else set()
    K = {}
    for l in open("work/kalshi_events.jsonl"):
        e = json.loads(l)
        if e.get("quote") and e.get("y") is not None:
            K.setdefault(key2(e["a"], e["b"]), []).append(e)
    # Polymarket lists ~300 tennis markets a day, so walk day by day over the Kalshi history (end-date windows)
    ks = sorted(e["start"] for v in K.values() for e in v)
    d0 = max(ks[0], time.time() - pages * 86400)
    PE = []
    day = d0
    while day < ks[-1] + 86400:
        a, b = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(day)), time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(day + 86400))
        for off in range(0, 2000, 100):
            ev = smart.get(smart.G + "/events", {"tag_slug": "tennis", "closed": "true", "limit": 100, "offset": off,
                                                 "end_date_min": a, "end_date_max": b}) or []
            for e in ev:
                mt = smart.VS.match(e.get("title") or "")
                if not mt or "/" in e.get("title", ""):
                    continue
                ml = [m for m in e.get("markets", []) if m.get("sportsMarketType") == "moneyline"] or                      [m for m in e.get("markets", []) if (m.get("question") or "").strip() == e["title"].strip()]
                if not ml:
                    continue
                m = ml[0]
                try:
                    outcomes = json.loads(m["outcomes"]); tokens = json.loads(m.get("clobTokenIds") or "[]")
                except (KeyError, ValueError, TypeError):
                    continue
                PE.append({"a": outcomes[0], "b": outcomes[1], "tokens": tokens, "volume": float(m.get("volume") or 0),
                           "start": m.get("gameStartTime") or e.get("startDate")})
            if len(ev) < 100:
                break
        day += 86400
    print(f"closed Polymarket tennis singles matches {time.strftime('%Y-%m-%d', time.gmtime(d0))} on: {len(PE)}", flush=True)
    n = 0
    with CACHE.open("a") as fh:
        for pe in PE:
            if not pe["tokens"] or pe["volume"] < 500:
                continue
            st = smart.parse_t(pe["start"]) if pe["start"] else None
            cands = [e for e in K.get(key2(pe["a"], pe["b"]), []) if st is None or abs(e["start"] - st) < 2 * 86400]
            if len(cands) != 1 or cands[0]["event"] in done:
                continue
            e = cands[0]
            ts = e["quote"]["ts"]
            h = smart.get(CLOB, {"market": pe["tokens"][0], "startTs": int(ts - 6 * 3600), "endTs": int(ts), "fidelity": 5}) or {}
            pts = [x for x in (h.get("history") or []) if x.get("t") and x["t"] <= ts]
            if not pts:
                continue
            pp = float(pts[-1]["p"])
            # Polymarket outcome 0 -> which Kalshi side?
            a_is_0 = (fold(pe["a"]) or [""])[-1] == (fold(e["a"]) or [""])[-1]
            fh.write(json.dumps({"kalshi": e["event"], "start": e["start"], "y": e["y"], "series": e["series"], "q": e["quote"],
                                 "poly_a": pp if a_is_0 else 1 - pp, "poly_age": ts - pts[-1]["t"], "poly_vol": pe["volume"]}) + "\n")
            done.add(e["event"]); n += 1
            if n % 100 == 0:
                fh.flush(); print("  matched", n, flush=True)
            time.sleep(0.15)
    print("new matched:", n)


def analyse():
    R = [json.loads(l) for l in CACHE.open()]
    R = [r for r in R if r["poly_age"] <= 1800]          # Polymarket price no older than 30 min at Kalshi's quote
    R.sort(key=lambda r: r["start"])
    n = len(R)
    print(f"{n} matches on both markets with a fresh Polymarket price at Kalshi's last pre-match quote")
    if n < 50:
        return
    ll = lambda y, p: -math.log(min(max(p if y else 1 - p, 1e-4), 1))
    km = [(r["q"]["bid"] + r["q"]["ask"]) / 2 for r in R]
    lk = np.mean([ll(r["y"], p) for r, p in zip(R, km)]); lp = np.mean([ll(r["y"], r["poly_a"]) for r in R])
    print(f"accuracy (log-loss, lower is better): Kalshi mid {lk:.4f} | Polymarket {lp:.4f}")
    half = np.arange(n) >= n // 2
    for gap in (0.02, 0.03, 0.05, 0.08):
        for vmin in (0, 2000):
            res = {0: [], 1: []}
            for k, r in enumerate(R):
                q = r["q"]
                if q["ask"] - q["bid"] > 0.04 or r["poly_vol"] < vmin:
                    continue
                for side in (0, 1):
                    ask = q["ask"] if side == 0 else 1 - q["bid"]
                    pp = r["poly_a"] if side == 0 else 1 - r["poly_a"]
                    if not 0.10 <= ask <= 0.90:
                        continue
                    c = ask + fee(ask)
                    if pp - c >= gap:
                        won = (r["y"] == 1) == (side == 0)
                        res[int(half[k])].append(((1 - c) / c if won else -1.0, won, c, pp))
            cells = []
            for h in (0, 1):
                v = res[h]
                if len(v) < 5:
                    cells.append(f"half {h}: n={len(v)}"); continue
                x = np.array([z[0] for z in v])
                cells.append(f"half {h}: n={len(v):>4} won {np.mean([z[1] for z in v]):5.1%} (paid {np.mean([z[2] for z in v]):.1%}, Poly said {np.mean([z[3] for z in v]):.1%}) ROI {x.mean():+6.1%} ± {x.std(ddof=1) / math.sqrt(len(x)):4.1%}")
            print(f"   Kalshi ask+fee {int(gap * 100)}c+ below Polymarket, Poly vol >= ${vmin:,}:  " + "   ".join(cells))


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch(int(sys.argv[sys.argv.index("--pages") + 1]) if "--pages" in sys.argv else 60)
    analyse()
