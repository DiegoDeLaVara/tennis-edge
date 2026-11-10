r"""Everything-and-anything search for "locks" among 70-85% favourites.

All features from every study merged into one table (storylines, travel, retirements, book gaps, entry status,
head-to-head detail, playing-style profiles, rankings, ages, round, tier, surface), restricted to favourites Pinnacle
priced 70-85% (exact 50/50 prices are irrelevant here but dropped upstream anyway).

Search: every single condition (each numeric feature at its 10/25/75/90% cut, both directions; each 0/1 flag), then
every PAIR of the 80 best single conditions. A rule must, in each of three time periods - discover (oldest 50%),
validate (next 25%), test (newest 25%) - have 40/20/20+ bets, win at least 3 pts more than priced, and make money
backing the favourite at Pinnacle's closing odds. The whole search is repeated on luck runs (results re-drawn from
the price): real rules have to outnumber what luck produces.

  .venv\Scripts\python.exe -m tools.lock70 [--luck N]

Result (2026-11-10): 14,459 favourites priced 70-85% (won 77.6% vs 76.8%, backing all -1.5%). 3,826 rules tested,
106 held in all three periods vs 21-110 in 10 luck runs (mean 58.5) -> not beyond luck. Survivors were mostly two
themes: (A) fav won <=41% of games in their last meeting yet still 70-85%; (B) dog lost a final at its last event.
Kalshi 2025-26 (tools/lock70_kalshi): A FAILED (71.1% vs 75.3%, -8.3%; leak-proof post-archive -6.4%); B can't be
checked (archive ends June 2026). No new 70% locks.
"""
import itertools
import math
import sys

import numpy as np
import pandas as pd

KEYS = ["date", "tour", "ps_f", "ps_d"]


def frame():
    M = pd.read_pickle("work/motivation.pkl")
    T = pd.read_pickle("work/travel_study.pkl")
    R = pd.read_pickle("work/trap_study.pkl")
    D = M.copy()
    for X in (T, R):
        for c in X.columns:
            if c not in D:
                D[c] = X[c].values
    D = D[np.abs(D.p - 0.5) > 1e-6]
    for f, cols in (("entry_study", ["entry_f", "entry_d", "age_f", "age_d"]),
                    ("h2h_detail", None), ("style_matchup", None)):
        X = pd.read_pickle(f"work/{f}.pkl").drop_duplicates(KEYS)
        keep = cols or [c for c in X.columns if c not in D.columns or c in KEYS]
        keep = list(dict.fromkeys(KEYS + [c for c in keep if c not in KEYS]))
        D = D.merge(X[keep], on=KEYS, how="left")
    for s in ("Q", "WC", "LL", "PR"):
        D[f"e{s}_f"] = (D.entry_f == s).astype(float); D[f"e{s}_d"] = (D.entry_d == s).astype(float)
    D["derby"] = [float(a is not None and a == b) for a, b in zip(D.nat_f, D.nat_d)]
    for c, vals in (("series", None), ("round", None), ("surf", None), ("court", None)):
        for v in D[c].dropna().unique():
            if (D[c] == v).mean() >= 0.03:
                D[f"is_{c}_{v}"] = (D[c] == v).astype(float)
    D["rank_ratio"] = np.log((D.rank_d.fillna(2000) + 1) / (D.rank_f.fillna(2000) + 1))
    D["age_gap"] = D.age_f - D.age_d
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    return D


SKIP = {"date", "tour", "y", "p", "ps_f", "ps_d", "series", "round", "surf", "court", "nat_f", "nat_d", "entry_f", "entry_d", "done",
        "max_f", "max_d"}


def conditions(B):
    out = []
    for c in [c for c in B.columns if c not in SKIP]:
        v = pd.to_numeric(B[c], errors="coerce").values.astype(float)
        ok = np.isfinite(v)
        if ok.sum() < 300:
            continue
        u = np.unique(v[ok])
        if len(u) <= 2:
            for x in u:
                out.append((f"{c} = {x:g}", ok & (v == x)))
        else:
            for q in np.unique(np.nanquantile(v[ok], [0.1, 0.25, 0.75, 0.9])):
                out.append((f"{c} >= {q:.3g}", ok & (v >= q)))
                out.append((f"{c} <= {q:.3g}", ok & (v <= q)))
    return out


def search(B, y, conds, quiet=True):
    n = len(B)
    split = np.zeros(n, int); split[n // 2:] = 1; split[(3 * n) // 4:] = 2
    r = np.where(y == 1, B.ps_f.values - 1, -1.0)
    p = B.p.values
    need = {0: 40, 1: 20, 2: 20}

    def ok_in(m, s):
        g = m & (split == s)
        k = g.sum()
        if k < need[s]:
            return False, k, 0, 0, 0
        won, pr, roi = y[g].mean(), p[g].mean(), r[g].mean()
        return (won - pr >= 0.03 and roi > 0), k, won, pr, roi
    # stage 1: single conditions, ranked on the discover period only
    singles = []
    for name, m in conds:
        g = m & (split == 0)
        if g.sum() >= 40:
            z = (y[g].mean() - p[g].mean()) / math.sqrt(max((p[g] * (1 - p[g])).mean() / g.sum(), 1e-9))
            singles.append((z, name, m))
    singles.sort(key=lambda t: -t[0])
    cands = [(nm, m) for _, nm, m in singles] + \
            [(f"{a} AND {b}", ma & mb) for (_, a, ma), (_, b, mb) in itertools.combinations(singles[:80], 2)
             if a.split(" ")[0] != b.split(" ")[0]]
    held = []
    for name, m in cands:
        res = [ok_in(m, s) for s in (0, 1, 2)]
        if all(x[0] for x in res):
            held.append((name, res))
    return held, len(cands)


def main():
    luck = int(sys.argv[sys.argv.index("--luck") + 1]) if "--luck" in sys.argv else 10
    D = frame()
    B = D[(D.p >= 0.70) & (D.p < 0.85)].reset_index(drop=True)
    print(f"{len(D):,} Pinnacle-priced matches with all features; {len(B):,} favourites priced 70-85% "
          f"(won {B.y.mean():.1%} vs {B.p.mean():.1%} priced; backing all of them {np.where(B.y == 1, B.ps_f - 1, -1).mean():+.1%})")
    conds = conditions(B)
    held, nc = search(B, B.y.values, conds)
    print(f"tested {nc:,} rules ({len(conds)} single conditions + pairs of the best 80); held in all 3 periods: {len(held)}")
    held.sort(key=lambda h: -min(x[2] - x[3] for x in h[1]))
    for name, res in held[:30]:
        print("   " + name[:95].ljust(95) + "  " + "  ".join(f"n={k:>4} won {w:5.1%} vs {pr:5.1%} ROI {roi:+5.1%}" for _, k, w, pr, roi in res))
    rng = np.random.default_rng(7)
    counts = []
    for _ in range(luck):
        fake = (rng.random(len(B)) < B.p.values).astype(int)
        counts.append(len(search(B, fake, conds)[0]))
    print(f"luck runs (same search, results drawn from the price): {counts} -> mean {np.mean(counts):.1f}, max {max(counts)}")
    pd.to_pickle({"held": held, "luck": counts}, "work/lock70.pkl")


if __name__ == "__main__":
    main()
