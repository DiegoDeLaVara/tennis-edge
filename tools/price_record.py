r""""Has this player been priced like this before, and how did they do?" For every match, each player's past record
against the market's price: overall, as a favourite, and at a price close to today's (within 7.5 pts). Does a player
who beat (or fell short of) their price at this level do it again?

Rules are scored like every other study: back / fade, two time halves, luck runs where results are re-drawn from the
price (the player histories are rebuilt from those fake results, so the luck runs carry the same selection effects).

  .venv\Scripts\python.exe -m tools.price_record [--null N]
Data: Pinnacle closing odds 2010-24 (tennis-data, tools.motivation.load) and Kalshi 2025-26 (work/kalshi_events.jsonl,
all levels, real ask + fee).

Result (2026-11-08): 0 rules held in both halves on Pinnacle (47k), Kalshi tour (9.4k) or Kalshi ITF. Favourites who had
beaten their price at this level by 10+ pts won 62.4% / 64.8% vs 63.8% / 64.1% priced; players who fell short didn't
keep falling short. A player's record against the price does not carry forward. Kalshi-trained "safe favourite" models
(price + ranks, + our model, + ages/round) also lost on their top picks in both halves (-3% to -11%).
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

from edge.names import fold
from tools.kalshi_backtest import fee

BAND = 0.075


def pinnacle_rows():
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
        t = r.date.toordinal()
        rows.append((t, t, f"{r.tour}|{r.w}", f"{r.tour}|{r.l}", pw, 1, r.psw - 1, r.psl - 1, "tour"))
    # put the winner/loser in a fixed order so 'a' is not always the winner
    out = []
    for k, (t0, t1, w, l, pw, _, rw, rl, lvl) in enumerate(rows):
        if k % 2:
            out.append((t0, t1, w, l, pw, 1, rw, rl, lvl))
        else:
            out.append((t0, t1, l, w, 1 - pw, 0, rl, rw, lvl))
    return out, "Pinnacle closing odds, ATP 2010-24 / WTA 2020-24"


def kalshi_rows():
    out = []
    for l in open("work/kalshi_events.jsonl"):
        e = json.loads(l)
        q = e.get("quote")
        if not q or e.get("y") is None or not (0 < q["bid"] < q["ask"] < 1) or q["ask"] - q["bid"] > 0.12:
            continue
        tour = "wta" if ("WTA" in e["series"] or "ITFW" in e["series"]) else "atp"
        ka, kb = f"{tour}|{' '.join(sorted(fold(e['a'])))}", f"{tour}|{' '.join(sorted(fold(e['b'])))}"
        ca, cb = q["ask"] + fee(q["ask"]), (1 - q["bid"]) + fee(1 - q["bid"])
        lvl = "itf" if "ITF" in e["series"] else "tour"
        out.append((e["start"], e.get("end") or e["start"] + 3 * 3600, ka, kb, (q["bid"] + q["ask"]) / 2, int(e["y"]), (1 - ca) / ca, (1 - cb) / cb, lvl))
    out.sort(key=lambda r: r[0])
    return out, "Kalshi 2025-26, all levels (real ask + fee)"


def features(rows, ys):
    """Per row, for side a and b: past (n, sum(won - price)) overall / as favourite / at a price within BAND of today's.
    Only matches that FINISHED before this one started count."""
    H = defaultdict(list)                      # player -> [(end_time, p, won)]
    pending = []                               # (end, player, p, won) not yet finished
    import heapq
    F = []
    for (t0, t1, a, b, pa, _, _, _, _), y in zip(rows, ys):
        while pending and pending[0][0] <= t0:
            end, pl, p, w = heapq.heappop(pending)
            H[pl].append((p, w))
        f = []
        for pl, p in ((a, pa), (b, 1 - pa)):
            h = H[pl]
            all_n = len(h); all_s = sum(w - q for q, w in h)
            fv = [(q, w) for q, w in h if q >= 0.5]
            bd = [(q, w) for q, w in h if abs(q - p) <= BAND]
            f.append((all_n, all_s, len(fv), sum(w - q for q, w in fv), len(bd), sum(w - q for q, w in bd)))
        F.append(f)
        end = t1 if t1 > t0 else t0 + 1        # Pinnacle rows: dates (ordinals) -> available from the next day
        heapq.heappush(pending, (end, a, pa, y))
        heapq.heappush(pending, (end, b, 1 - pa, 1 - y))
    return F


def rules():
    # (name, side test on (my features, opp features, my price)) -> back that side
    def per(n, s, k):
        return s / n if n >= k else None
    R = {}
    for lo, hi, bn in ((0.55, 0.75, "55-75% fav"), (0.5, 1.0, "any fav")):
        def inb(p, lo=lo, hi=hi):
            return lo <= p < hi
        R[f"{bn}: beat the price at THIS price level by 10+ pts (6+ times priced here) -> back"] = \
            lambda m, o, p, inb=inb: p >= 0.5 and inb(p) and (per(m[4], m[5], 6) or -9) >= 0.10
        R[f"{bn}: fell 10+ pts short at THIS price level (6+ times) -> back the underdog"] = \
            lambda m, o, p, inb=inb: p < 0.5 and inb(1 - p) and (per(o[4], o[5], 6) or 9) <= -0.10
        R[f"{bn}: beat the price as a favourite by 8+ pts (10+ times) -> back"] = \
            lambda m, o, p, inb=inb: p >= 0.5 and inb(p) and (per(m[2], m[3], 10) or -9) >= 0.08
        R[f"{bn}: fell 8+ pts short as a favourite (10+ times) -> back the underdog"] = \
            lambda m, o, p, inb=inb: p < 0.5 and inb(1 - p) and (per(o[2], o[3], 10) or 9) <= -0.08
        R[f"{bn}: beats the price overall by 5+ pts (20+ matches) -> back"] = \
            lambda m, o, p, inb=inb: p >= 0.5 and inb(p) and (per(m[0], m[1], 20) or -9) >= 0.05
        R[f"{bn}: falls 5+ pts short overall (20+ matches) -> back the underdog"] = \
            lambda m, o, p, inb=inb: p < 0.5 and inb(1 - p) and (per(o[0], o[1], 20) or 9) <= -0.05
        R[f"{bn}: underdog has beaten its price as an underdog-at-this-level by 10+ pts -> back the underdog"] = \
            lambda m, o, p, inb=inb: p < 0.5 and inb(1 - p) and (per(m[4], m[5], 6) or -9) >= 0.10
    return R


def score(rows, ys, quiet=False, levels=("tour", "itf")):
    F = features(rows, ys)
    half = np.arange(len(rows)) >= len(rows) // 2
    passed = []
    for name, fn in rules().items():
        res = {0: [], 1: []}
        for k, ((t0, t1, a, b, pa, _, ra, rb, lvl), y, (fa, fb)) in enumerate(zip(rows, ys, F)):
            if lvl not in levels:
                continue
            for m, o, p, won, r in ((fa, fb, pa, y, ra), (fb, fa, 1 - pa, 1 - y, rb)):
                if fn(m, o, p):
                    res[int(half[k])].append((r if won else -1.0, won, p))
        cells, ok = [], True
        for h in (0, 1):
            v = res[h]
            if len(v) < 60:
                cells.append(f"n={len(v):>5} (few)"); ok = False; continue
            x = np.array([z[0] for z in v])
            ok &= x.mean() > 0.01
            cells.append(f"n={len(v):>5} won {np.mean([z[1] for z in v]):5.1%} vs {np.mean([z[2] for z in v]):5.1%} ROI {x.mean():+6.1%} ± {x.std(ddof=1) / math.sqrt(len(x)):4.1%}")
        if ok:
            passed.append(name)
        if not quiet:
            print(f"   {name:<96} " + "   ".join(cells) + ("  <-- both halves" if ok else ""))
    return passed


def main():
    nulls = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else 0
    for loader, levels_list in ((pinnacle_rows, [("tour",)]), (kalshi_rows, [("tour",), ("itf",)])):
        rows, title = loader()
        ys = [r[5] for r in rows]
        for levels in levels_list:
            print(f"\n===== {title} - {'/'.join(levels)} ({sum(r[8] in levels for r in rows):,} matches)")
            passed = score(rows, ys, levels=levels)
            print(f"   {len(passed)} rule(s) held in both halves")
            if nulls:
                rng = np.random.default_rng(0)
                cnt = []
                for _ in range(nulls):
                    yf = [int(rng.random() < r[4]) for r in rows]
                    cnt.append(len(score(rows, yf, quiet=True, levels=levels)))
                print(f"   luck runs (results drawn from the price): {cnt} -> mean {np.mean(cnt):.1f}")


if __name__ == "__main__":
    main()
