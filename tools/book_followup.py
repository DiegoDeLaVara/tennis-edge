r"""Follow-ups to tools/book_study on the same joined bookmaker data (cached in work/book_sides.pkl):
form over-reaction (does the market over-rate a hot underdog?), line shopping by odds band, devig methods,
tired favourite in both halves, parlays of +EV legs, and a games-total calibration for the Markov engine.

  .venv\Scripts\python.exe -m tools.book_followup [--games]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tools.book_study import load_odds, join, sides, roi, ll, logit_v

CACHE = Path("work/book_sides.pkl")


def data():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    S = sides(join(load_odds()))
    S = S[S.p_pin.between(0.5, 0.995)].reset_index(drop=True)
    for c in ("max_f", "max_d", "avg_f", "avg_d", "b365_f", "b365_d"):
        S[c] = pd.to_numeric(S[c], errors="coerce")
    S.to_pickle(CACHE)
    return S


def ok(x):
    return x is not None and np.isfinite(x) and x > 1


def line(name, bets):
    n, m, se = roi(bets)
    print(f"   {name:<62} n={n:>6}  ROI {m:+6.1%} ± {se:4.1%}")


def power_fair(o1, o2):
    i1, i2 = 1 / o1, 1 / o2
    lo, hi = 0.5, 3.0
    for _ in range(50):
        k = (lo + hi) / 2
        if i1 ** k + i2 ** k > 1:
            lo = k
        else:
            hi = k
    a, b = i1 ** k, i2 ** k
    return a / (a + b)


def main():
    if "--games-only" in sys.argv:
        S = data()
        return games_calib(S[S.year >= 2018])
    S = data()
    halves = {"2010-17": S[S.year < 2018], "2018-24": S[S.year >= 2018]}
    print(f"{len(S):,} matches with Pinnacle closing prices")

    print("\n== Does the market over-rate a hot underdog? (form = results beyond rating, 45-day; back the FAVOURITE)")
    for gap in (0.05, 0.10, 0.15, 0.20):
        for nm, H in halves.items():
            b1 = [(r.ps_f, r.y == 1) for r in H.itertuples() if r.form_d - r.form_f >= gap]
            b2 = [(r.max_f, r.y == 1) for r in H.itertuples() if r.form_d - r.form_f >= gap and ok(r.max_f)]
            n, m1, s1 = roi(b1); _, m2, s2 = roi(b2)
            print(f"   dog hotter by >= {gap:.2f}  {nm}  n={n:>5}  fav at Pinnacle {m1:+6.1%} ± {s1:3.1%}   at best price {m2:+6.1%} ± {s2:3.1%}")
    print("   ...and the mirror: favourite hotter by >= 0.10, back the favourite")
    for nm, H in halves.items():
        line(f"fav hotter >= 0.10, {nm}, at Pinnacle", [(r.ps_f, r.y == 1) for r in H.itertuples() if r.form_f - r.form_d >= 0.10])
    print("   calibration: Pinnacle fair vs actual, by form gap (dog - fav), 2010-24")
    S["fg"] = pd.cut(S.form_d - S.form_f, [-1, -0.15, -0.05, 0.05, 0.15, 1])
    for k, g in S.groupby("fg", observed=True):
        print(f"     gap {str(k):<16} n={len(g):>6}  Pinnacle said fav {g.p_pin.mean():.3f}  fav won {g.y.mean():.3f}  ({g.y.mean()-g.p_pin.mean():+.3f})")

    print("\n== Tired favourite (10-day minutes fav >= 400, dog <= 150), both halves")
    for nm, H in halves.items():
        line(f"{nm} at Pinnacle", [(r.ps_f, r.y == 1) for r in H.itertuples() if r.load_f >= 400 and r.load_d <= 150])
        line(f"{nm} at best price", [(r.max_f, r.y == 1) for r in H.itertuples() if r.load_f >= 400 and r.load_d <= 150 and ok(r.max_f)])

    print("\n== Line shopping by odds band (best price >= 2% over Pinnacle fair), both halves")
    for lo, hi in ((1.0, 1.5), (1.5, 2.2), (2.2, 3.0), (3.0, 5.0), (5.0, 50)):
        for nm, H in halves.items():
            bets = []
            for r in H.itertuples():
                for p, m, w in ((r.p_pin, r.max_f, r.y == 1), (1 - r.p_pin, r.max_d, r.y == 0)):
                    if ok(m) and lo <= m < hi and p * m - 1 >= 0.02:
                        bets.append((m, w))
            line(f"odds {lo}-{hi}  {nm}", bets)

    print("\n== Fair price from Pinnacle: which devig is best calibrated? (log-loss, all years)")
    pp = np.array([power_fair(a, b) for a, b in zip(S.ps_f, S.ps_d)])
    print(f"   proportional {ll(S.y, S.p_pin):.5f}   power {ll(S.y, pp):.5f}")
    tr, te = halves["2010-17"], halves["2018-24"]
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=100).fit(logit_v(tr.p_pin).reshape(-1, 1), tr.y)
    pst = lr.predict_proba(logit_v(te.p_pin).reshape(-1, 1))[:, 1]
    print(f"   stretch fitted on 2010-17: logit x {lr.coef_[0][0]:.3f} + {lr.intercept_[0]:+.3f}; 2018-24: proportional {ll(te.y, te.p_pin):.5f} "
          f"power {ll(te.y, pp[S.year >= 2018]):.5f} stretched {ll(te.y, pst):.5f}")

    print("\n== Parlays, 2018-24, priced at Bet365 (one book) vs Pinnacle")
    by_day = defaultdict(list)
    for r in te.itertuples():
        if ok(r.b365_f) and 1.1 <= r.ps_f < 1.6:
            by_day[r.date].append((r.p_pin, r.b365_f, r.y == 1, r.ps_f))
    for legs in (1, 2, 3, 4):
        res, resp = [], []
        for d, L in by_day.items():
            L = sorted(L, key=lambda x: -x[0])[:legs]
            if len(L) < legs:
                continue
            w = all(x[2] for x in L)
            res.append((float(np.prod([x[1] for x in L])), w)); resp.append((float(np.prod([x[3] for x in L])), w))
        n, m, se = roi(res); _, m2, se2 = roi(resp)
        print(f"   {legs}-leg favourites  n={n:>5}  at Bet365 {m:+6.1%} ± {se:4.1%}   at Pinnacle {m2:+6.1%} ± {se2:4.1%}")
    print("   parlays of line-shopping legs (best price >= 3% over Pinnacle fair, odds <= 3), 2 per day, at those prices")
    by_day = defaultdict(list)
    for r in te.itertuples():
        for p, m, w in ((r.p_pin, r.max_f, r.y == 1), (1 - r.p_pin, r.max_d, r.y == 0)):
            if ok(m) and m <= 3 and p * m - 1 >= 0.03:
                by_day[r.date].append((p * m - 1, m, w))
    for legs in (1, 2, 3):
        res = []
        for d, L in by_day.items():
            L = sorted(L, key=lambda x: -x[0])[:legs]
            if len(L) == legs:
                res.append((float(np.prod([x[1] for x in L])), all(x[2] for x in L)))
        line(f"{legs}-leg +EV parlay", res)

    if "--games" in sys.argv:
        games_calib(te)


def games_calib(te):
    """Fit a serve-strength shift per (tour, best-of, surface) so the engine's expected games match reality."""
    from edge import markov
    G = te[te.done & (te.games > 0)]
    G = G.sample(frac=1, random_state=2).groupby(["tour", "bo", "surface"]).head(500)
    out = {}
    print("\n== Games calibration: shift to the combined serve-win rate that makes expected games match (2018-24)")
    for k, g in G.groupby(["tour", "bo", "surface"]):
        if len(g) < 150:
            continue
        res = {}
        for delta in (-0.02, 0.0, 0.02):
            e = []
            for r in g.itertuples():
                pa, pb = markov.solve_serve_mix(r.p_pin, r.spw_sum + delta, r.bo, markov.DAY_SD[r.tour])
                D = markov.mix_dist(pa, pb, r.bo, markov.DAY_SD[r.tour])
                e.append(sum(kk[2] * v for kk, v in D.items()))
            res[delta] = float(np.mean(e))
        act = float(g.games.mean())
        slope = (res[0.02] - res[-0.02]) / 0.04
        shift = max(-0.03, min(0.03, (act - res[0.0]) / slope)) if slope > 0 else 0.0
        key = f"{k[0]}|{int(k[1])}|{k[2]}"
        out[key] = round(shift, 4)
        print(f"   {key:<16} n={len(g):>4}  actual {act:5.2f}  engine {res[0.0]:5.2f}  -> serve shift {shift:+.4f}")
    Path("model/games_adj.json").write_text(json.dumps(out, indent=1))
    print("   written model/games_adj.json")


if __name__ == "__main__":
    main()
