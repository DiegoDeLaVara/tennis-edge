r""""If a player just pulled off an upset, are they more likely to upset again next time they're the underdog?"

For every match where a player is the underdog, look at their previous match: did they win it as the underdog (an
upset)? Then compare how often they win now with what the price expected, and the return of backing them.
Variants: big upset (they were under 30%), the next round of the same event (within 4 days), two upsets in a row,
and the reverse (a favourite who was just upset). Data: Pinnacle closing odds 2010-24 (tennis-data, exact 50/50 prices
dropped) and Kalshi 2025-26 (real ask + fee, previous match must have finished before this one started).
Two time halves + luck runs (results re-drawn from the price, histories rebuilt).

  .venv\Scripts\python.exe -m tools.upset_momentum [--null N]

Result (2026-11-11): no momentum. Upset winners won 25.7% / 29.6% next time as the dog vs 26.5% / 29.3% priced
(Pinnacle), 28.4% / 27.7% vs 28.3% / 26.5% (Kalshi); backing them lost 3-12%. Two upsets in a row did slightly WORSE
than priced (Pinnacle 23.3% / 25.7% vs 25.4% / 28.2%): the market over-rates a hot underdog a little. 0 rules held
(luck runs 0-1).
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

from edge.names import fold
from tools.kalshi_backtest import fee


def pinnacle():
    from tools.motivation import load
    O = load()
    O["i"] = np.arange(len(O))
    O = O.sort_values(["date", "i"], kind="stable")
    rows = []
    for r in O.itertuples():
        if not (r.psw and r.psl and math.isfinite(r.psw) and math.isfinite(r.psl)):
            continue
        fw, fl = 1 / r.psw, 1 / r.psl
        pw = fw / (fw + fl)
        if abs(pw - 0.5) < 1e-6:
            continue
        t = r.date.toordinal()
        # (start, end, player a, player b, p_a, a_won, return backing a, return backing b)
        a_first = (len(rows) % 2 == 0)
        if a_first:
            rows.append((t, t, f"{r.tour}|{r.w}", f"{r.tour}|{r.l}", pw, 1, r.psw - 1, r.psl - 1))
        else:
            rows.append((t, t, f"{r.tour}|{r.l}", f"{r.tour}|{r.w}", 1 - pw, 0, r.psl - 1, r.psw - 1))
    return rows, 1                         # time unit: days


def kalshi():
    rows = []
    for l in open("work/kalshi_events.jsonl"):
        e = json.loads(l)
        q = e.get("quote")
        if not q or e.get("y") is None or not (0 < q["bid"] < q["ask"] < 1) or q["ask"] - q["bid"] > 0.12:
            continue
        tour = "wta" if ("WTA" in e["series"] or "ITFW" in e["series"]) else "atp"
        ka, kb = f"{tour}|{' '.join(sorted(fold(e['a'])))}", f"{tour}|{' '.join(sorted(fold(e['b'])))}"
        ca, cb = q["ask"] + fee(q["ask"]), (1 - q["bid"]) + fee(1 - q["bid"])
        rows.append((e["start"], e.get("end") or e["start"] + 10800, ka, kb, (q["bid"] + q["ask"]) / 2, int(e["y"]), (1 - ca) / ca, (1 - cb) / cb))
    rows.sort(key=lambda r: r[0])
    return rows, 86400


def evaluate(rows, ys, unit, quiet=False):
    last = {}                                  # player -> list of (end, was_dog, won, p)
    hist = defaultdict(list)
    R = defaultdict(lambda: {0: [], 1: []})
    half_at = rows[len(rows) // 2][0]
    import heapq
    pend = []
    for (t0, t1, a, b, pa, _, ra, rb), y in zip(rows, ys):
        while pend and pend[0][0] <= t0 - (0 if unit == 86400 else 0.5):
            end, pl, rec = heapq.heappop(pend)
            hist[pl].append(rec)
        h = int(t0 >= half_at)
        for pl, p, won, r in ((a, pa, y, ra), (b, 1 - pa, 1 - y, rb)):
            H = hist[pl]
            if not H or p >= 0.5:
                continue
            last_ = H[-1]
            gap_days = (t0 - last_[0]) / unit
            if gap_days > 30:
                continue
            ret = r if won else -1.0
            prev_upset = last_[1] and last_[2]
            if prev_upset:
                R["won last match as the underdog (an upset)"][h].append((ret, won, p))
                if last_[3] < 0.30:
                    R["...a BIG upset (they were under 30%)"][h].append((ret, won, p))
                if gap_days <= 4:
                    R["...in the previous round (within 4 days)"][h].append((ret, won, p))
                if len(H) >= 2 and H[-2][1] and H[-2][2] and (last_[0] - H[-2][0]) / unit <= 30:
                    R["...two upsets in a row"][h].append((ret, won, p))
            elif last_[1] and not last_[2]:
                R["(compare) lost last match as the underdog"][h].append((ret, won, p))
            elif not last_[1] and last_[2]:
                R["(compare) won last match as the favourite"][h].append((ret, won, p))
            elif not last_[1] and not last_[2]:
                R["(compare) was UPSET last match (lost as favourite)"][h].append((ret, won, p))
        e1 = t1 if t1 > t0 else t0 + (unit if unit == 1 else 1)
        heapq.heappush(pend, (e1, a, (t0, pa < 0.5, y == 1, pa)))
        heapq.heappush(pend, (e1, b, (t0, pa >= 0.5, y == 0, 1 - pa)))
    passed = []
    for name in ["won last match as the underdog (an upset)", "...a BIG upset (they were under 30%)", "...in the previous round (within 4 days)",
                 "...two upsets in a row", "(compare) lost last match as the underdog", "(compare) won last match as the favourite",
                 "(compare) was UPSET last match (lost as favourite)"]:
        cells, ok = [], True
        for h in (0, 1):
            v = R[name][h]
            if len(v) < 60:
                cells.append(f"n={len(v):>5} (few)"); ok = False; continue
            x = np.array([z[0] for z in v])
            cells.append(f"n={len(v):>5} dog won {np.mean([z[1] for z in v]):5.1%} vs {np.mean([z[2] for z in v]):5.1%} back {x.mean():+6.1%} ± {x.std(ddof=1) / math.sqrt(len(x)):4.1%}")
            ok &= x.mean() > 0.01
        if ok and not name.startswith("(compare)"):
            passed.append(name)
        if not quiet:
            print(f"   {name:<52} " + "   ".join(cells) + ("  <-- held" if ok and not name.startswith("(compare)") else ""))
    return passed


def main():
    nulls = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else 0
    for loader, title in ((pinnacle, "Pinnacle closing odds 2010-24 (tour)"), (kalshi, "Kalshi 2025-26 (all levels, ask + fee)")):
        rows, unit = loader()
        ys = [r[5] for r in rows]
        print(f"\n===== {title}: {len(rows):,} matches; the player is the UNDERDOG now, previous match within 30 days")
        evaluate(rows, ys, unit)
        if nulls:
            rng = np.random.default_rng(0); c = []
            for _ in range(nulls):
                c.append(len(evaluate(rows, [int(rng.random() < r[4]) for r in rows], unit, quiet=True)))
            print(f"   luck runs: {c}")


if __name__ == "__main__":
    main()
