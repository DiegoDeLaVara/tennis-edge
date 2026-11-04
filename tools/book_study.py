r"""Strategy lab at real bookmaker prices. Joins tennis-data.co.uk odds (Pinnacle, best-of-market "Max", average,
Bet365; ATP 2010-24, WTA 2020-24, downloaded to work/bookodds from the public DanielSzakacs/atp_data and wta_data
mirrors) to our walk-forward features, then tests:

  1. model vs Pinnacle (log-loss) and a model+Pinnacle blend
  2. flat strategies: favourites/underdogs by price, line shopping (best price beats Pinnacle's fair), model edges
  3. ladders: restake part of the run's profit on the next bet, reset after a loss or N wins
  4. same-day parlays of favourites
  5. total games: our Markov engine anchored to Pinnacle's price vs the games actually played
  6. upsets by level (Challengers etc.) from the 850k-match history, and Kalshi prices by series

Rules are judged on the newest half of matches (2018+), never on the years they were picked from.

  .venv\Scripts\python.exe -m tools.book_study
"""
import glob
import math
import random
import sys
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from edge import markov
from edge.model import Model, logit, sig
from edge.names import fold

ROOT = "work"
rng = random.Random(7)


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def num(x):
    try:
        v = float(x)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- load + join
def load_odds():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/bookodds/*.csv")):
        tour = "atp" if "atp_" in f else "wta"
        D = pd.read_csv(f, encoding="latin-1", low_memory=False)
        for r in D.to_dict("records"):
            try:
                d = pd.to_datetime(r["Date"]).date()
            except Exception:
                continue
            g = 0
            for i in range(1, 6):
                a, b = r.get(f"W{i}"), r.get(f"L{i}")
                try:
                    if pd.notna(a) and pd.notna(b):
                        g += int(float(a)) + int(float(b))
                except (TypeError, ValueError):
                    pass
            rows.append({"tour": tour, "date": d, "w": str(r["Winner"]), "l": str(r["Loser"]), "surface": r.get("Surface"),
                         "series": r.get("Series", r.get("Tier")), "round": r.get("Round"), "bo": int(r["Best of"]) if pd.notna(r.get("Best of")) else 3,
                         "done": r.get("Comment") == "Completed", "games": g,
                         "psw": num(r.get("PSW")), "psl": num(r.get("PSL")), "maxw": num(r.get("MaxW")), "maxl": num(r.get("MaxL")),
                         "avgw": num(r.get("AvgW")), "avgl": num(r.get("AvgL")), "b365w": num(r.get("B365W")), "b365l": num(r.get("B365L"))})
    return pd.DataFrame(rows)


def td_tokens(name):
    """'Auger-Aliassime F.' -> (['auger','aliassime'], 'f')"""
    parts = name.replace(".", ". ").split()
    if len(parts) < 2:
        return None
    sur = [p for p in parts if not p.endswith(".")] or parts[:-1]
    ini = [p for p in parts if p.endswith(".")]
    toks = [t for s in sur for t in fold(s)]
    i0 = fold(ini[0])[0][0] if ini and fold(ini[0]) else ""
    return toks, i0


def matches(key, tt):
    ks = key.split(":", 1)[1].split()
    toks, i0 = tt
    if not toks or not all(t in ks for t in toks):
        return False
    rest = [k for k in ks if k not in toks]
    return not i0 or not rest or any(k.startswith(i0) for k in rest)


def join(O):
    F = pd.read_csv(f"{ROOT}/features.csv", low_memory=False)
    F = F[F.date >= "2009-12-01"].reset_index(drop=True)
    F["d"] = pd.to_datetime(F.date).dt.date
    idx = defaultdict(list)
    for i, (tour, a, b, d) in enumerate(zip(F.tour, F.a, F.b, F.d)):
        for side, k in ((0, a), (1, b)):
            for t in k.split(":", 1)[1].split():
                idx[(tour, t)].append((d, i, side))
    M = Model()
    out = []
    for r in O.itertuples():
        tw, tl = td_tokens(r.w), td_tokens(r.l)
        if not tw or not tl or not tw[0]:
            continue
        hit = None
        for (d, i, side) in idx.get((r.tour, tw[0][-1]), []):
            if not (timedelta(0) <= r.date - d <= timedelta(16)):
                continue
            ka, kb = F.at[i, "a"], F.at[i, "b"]
            kw, kl = (ka, kb) if side == 0 else (kb, ka)
            if matches(kw, tw) and matches(kl, tl):
                won_a = F.at[i, "y"] == 1
                if (side == 0) == won_a:
                    hit = (i, side)
                    break
        if hit is None:
            continue
        i, side = hit
        f = F.iloc[i]
        pa, _ = M.prob(f, f.tour, f.level, int(f.best_of))
        pw = pa if side == 0 else 1 - pa                      # model prob for the actual winner
        d = r._asdict()
        d.update({"p_model_w": pw, "level": f.level, "form_w": f.form_a if side == 0 else f.form_b,
                  "form_l": f.form_b if side == 0 else f.form_a, "spw_sum": f.pa + f.pb,
                  "load_w": f.load_a if side == 0 else f.load_b, "load_l": f.load_b if side == 0 else f.load_a})
        out.append(d)
    J = pd.DataFrame(out)
    print(f"odds rows {len(O):,} | joined to model features {len(J):,} ({len(J)/len(O):.0%})")
    return J


# ---------------------------------------------------------------- per-player view
def sides(J):
    """Two rows per match (one per player) in a random but fixed orientation-free form; we pick ONE row per match:
    side 'a' = the Pinnacle favourite. y = favourite won."""
    rows = []
    for r in J.itertuples():
        if not (r.psw and r.psl):
            continue
        fw, fl = 1 / r.psw, 1 / r.psl
        s = fw + fl
        fair_w = fw / s
        fav_is_w = fair_w >= 0.5
        def pick(w, l):
            return (w, l) if fav_is_w else (l, w)
        pf = fair_w if fav_is_w else 1 - fair_w
        pm = r.p_model_w if fav_is_w else 1 - r.p_model_w
        rows.append({"date": r.date, "year": r.date.year, "tour": r.tour, "series": r.series, "level": r.level, "bo": r.bo,
                     "surface": r.surface, "round": r.round, "done": r.done, "games": r.games, "spw_sum": r.spw_sum,
                     "y": 1 if fav_is_w else 0, "p_pin": pf, "p_model": pm,
                     "ps_f": pick(r.psw, r.psl)[0], "ps_d": pick(r.psw, r.psl)[1],
                     "max_f": pick(r.maxw, r.maxl)[0], "max_d": pick(r.maxw, r.maxl)[1],
                     "avg_f": pick(r.avgw, r.avgl)[0], "avg_d": pick(r.avgw, r.avgl)[1],
                     "b365_f": pick(r.b365w, r.b365l)[0], "b365_d": pick(r.b365w, r.b365l)[1],
                     "form_f": pick(r.form_w, r.form_l)[0], "form_d": pick(r.form_w, r.form_l)[1],
                     "load_f": pick(r.load_w, r.load_l)[0], "load_d": pick(r.load_w, r.load_l)[1]})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def roi(bets):
    """bets: list of (odds, won). Flat 1u."""
    if not bets:
        return 0, 0.0, 0.0
    r = np.array([(o - 1) if w else -1.0 for o, w in bets])
    return len(r), float(r.mean()), float(r.std(ddof=1) / math.sqrt(len(r))) if len(r) > 1 else 0.0


def show(name, bets):
    n, m, se = roi(bets)
    print(f"   {name:<58} n={n:>6}  ROI {m:+6.1%} ± {se:4.1%}")


def stream(S, side_fn):
    """side_fn(row) -> list of (odds, won) for that match."""
    out = []
    for r in S.itertuples():
        out.extend(side_fn(r))
    return out


# ---------------------------------------------------------------- ladder
def ladder(bets, f=0.5, steps=99, base=1.0):
    """Stake = base + f * profit banked in the current run. A run ends after a loss (the run's profit that was
    restaked is lost with it) or after `steps` wins (profit kept). Returns per-bet P&L and stakes."""
    run, k, pnl, stakes = 0.0, 0, [], []
    for o, w in bets:
        st = base + f * run
        stakes.append(st)
        if w:
            g = st * (o - 1)
            pnl.append(g)
            run += g
            k += 1
            if k >= steps:
                run, k = 0.0, 0
        else:
            pnl.append(-st)
            run, k = 0.0, 0
    return np.array(pnl), np.array(stakes)


def dd(pnl):
    c = np.cumsum(pnl)
    return float(np.max(np.maximum.accumulate(np.concatenate([[0], c]))[1:] - c)) if len(c) else 0.0


def ladder_table(name, bets, sims=400):
    print(f"\n  {name}  ({len(bets)} bets, in date order; units: base stake 1u)")
    print(f"   {'plan':<34} {'profit':>9} {'staked':>9} {'ROI/staked':>10} {'max drawdown':>13} {'paths ahead':>12}")
    plans = [("flat 1u", 0.0, 99), ("restake 25% of run profit", 0.25, 99), ("restake 50% of run profit", 0.5, 99),
             ("restake 100% (let it ride)", 1.0, 99), ("let it ride, bank after 3 wins", 1.0, 3),
             ("let it ride, bank after 5 wins", 1.0, 5)]
    for label, f, steps in plans:
        pnl, st = ladder(bets, f, steps)
        ahead = 0
        for _ in range(sims):
            b = bets[:]
            rng.shuffle(b)
            ahead += ladder(b, f, steps)[0].sum() > 0
        print(f"   {label:<34} {pnl.sum():>+9.1f} {st.sum():>9.0f} {pnl.sum()/st.sum():>+10.2%} {dd(pnl):>13.1f} {ahead/sims:>12.0%}")


# ---------------------------------------------------------------- main
def main():
    O = load_odds()
    J = join(O)
    S = sides(J)
    S = S[S.p_pin.between(0.5, 0.995)]
    test = S[S.year >= 2018]
    train = S[S.year < 2018]
    print(f"matches with Pinnacle prices: {len(S):,}  (train 2010-17 {len(train):,}, test 2018-24 {len(test):,})")

    # 1. accuracy
    print("\n== 1. Who predicts better? (2018-24, log-loss; lower is better)")
    from sklearn.linear_model import LogisticRegression
    X = lambda D: np.c_[logit_v(D.p_model), logit_v(D.p_pin)]
    lr = LogisticRegression(C=100).fit(X(train), train.y)
    pb = lr.predict_proba(X(test))[:, 1]
    print(f"   Pinnacle closing (no-vig) {ll(test.y, test.p_pin):.4f}  | our model {ll(test.y, test.p_model):.4f}  | "
          f"blend {ll(test.y, pb):.4f}  (weights model {lr.coef_[0][0]:.3f}, Pinnacle {lr.coef_[0][1]:.3f})")
    for lv in ["atp", "wta"]:
        T = test[test.tour == lv]
        if len(T):
            print(f"   {lv}: n={len(T):,} Pinnacle {ll(T.y, T.p_pin):.4f} model {ll(T.y, T.p_model):.4f}")
    test = test.assign(p_blend=pb)

    # 2. flat strategies
    print("\n== 2. Flat 1u bets, 2018-24")
    print("  Favourite-longshot: backing every player in a price band, at Pinnacle / at the best price in the market")
    bands = [(1.0, 1.25), (1.25, 1.5), (1.5, 1.8), (1.8, 2.2), (2.2, 3.0), (3.0, 5.0), (5.0, 50)]
    for lo, hi in bands:
        bp = stream(test, lambda r: [(o, w) for o, w in ((r.ps_f, r.y == 1), (r.ps_d, r.y == 0)) if lo <= o < hi])
        bm = stream(test, lambda r: [(m, w) for o, m, w in ((r.ps_f, r.max_f, r.y == 1), (r.ps_d, r.max_d, r.y == 0))
                                     if lo <= o < hi and m])
        n1, m1, s1 = roi(bp)
        n2, m2, s2 = roi(bm)
        print(f"   odds {lo:>4}-{hi:<4}  n={n1:>6}  at Pinnacle {m1:+6.1%} ± {s1:3.1%}   at best price {m2:+6.1%} ± {s2:3.1%}")

    print("  Line shopping: bet when the best price beats Pinnacle's no-vig fair odds by k")
    for k in (0.0, 0.02, 0.05, 0.10):
        bets = stream(test, lambda r: [(m, w) for p, m, w in ((r.p_pin, r.max_f, r.y == 1), (1 - r.p_pin, r.max_d, r.y == 0))
                                       if m and p * m - 1 >= k])
        show(f"best price edge >= {k:.0%}", bets)
        bets = stream(test, lambda r: [(m, w) for p, m, w in ((r.p_pin, r.b365_f, r.y == 1), (1 - r.p_pin, r.b365_d, r.y == 0))
                                       if m and p * m - 1 >= k])
        show(f"   Bet365 alone edge >= {k:.0%}", bets)
    print("  Model edges (blend of model + Pinnacle), at the best price")
    for k in (0.02, 0.04, 0.07):
        bets = stream(test, lambda r: [(m, w) for p, m, w in ((r.p_blend, r.max_f, r.y == 1), (1 - r.p_blend, r.max_d, r.y == 0))
                                       if m and p * m - 1 >= k])
        show(f"blend edge >= {k:.0%}", bets)
        bets = stream(test, lambda r: [(m, w) for p, m, w in ((r.p_model, r.ps_f, r.y == 1), (1 - r.p_model, r.ps_d, r.y == 0))
                                       if m and p * m - 1 >= k])
        show(f"   model alone edge >= {k:.0%} at Pinnacle", bets)
    print("  Situations, at Pinnacle")
    show("hot underdog (form gap >= 0.10)", stream(test, lambda r: [(r.ps_d, r.y == 0)] if r.form_d - r.form_f >= 0.10 else []))
    show("cold favourite fade (fav form - dog form <= -0.10)", stream(test, lambda r: [(r.ps_d, r.y == 0)] if r.form_f - r.form_d <= -0.10 else []))
    show("tired favourite (10-day mins >= 400, dog <= 150)", stream(test, lambda r: [(r.ps_f, r.y == 1)] if r.load_f >= 400 and r.load_d <= 150 else []))
    show("tired underdog fade -> back fav (dog 10-day mins >= 400, fav <= 150)", stream(test, lambda r: [(r.ps_f, r.y == 1)] if r.load_d >= 400 and r.load_f <= 150 else []))
    for bo in (3, 5):
        show(f"favourites in best-of-{bo}, odds 1.2-1.6", stream(test, lambda r: [(r.ps_f, r.y == 1)] if r.bo == bo and 1.2 <= r.ps_f < 1.6 else []))

    # 3. ladders
    print("\n== 3. Ladders (2018-24, real order of matches)")
    favs = stream(test, lambda r: [(r.ps_f, r.y == 1)] if 1.2 <= r.ps_f < 1.5 else [])
    ladder_table("A. Favourites at Pinnacle odds 1.20-1.50", favs)
    heavy = stream(test, lambda r: [(r.max_f, r.y == 1)] if r.max_f and 1.08 <= r.ps_f < 1.25 else [])
    ladder_table("B. Heavy favourites (1.08-1.25) at the best price", heavy)
    shop = stream(test, lambda r: [(m, w) for p, m, w in ((r.p_pin, r.max_f, r.y == 1), (1 - r.p_pin, r.max_d, r.y == 0))
                                   if m and p * m - 1 >= 0.03 and m <= 3.0])
    ladder_table("C. Line shopping: best price >= 3% over Pinnacle fair, odds <= 3.0", shop)

    # 4. parlays
    print("\n== 4. Same-day parlays of the day's biggest Pinnacle favourites, priced at Bet365 (one book), 2018-24")
    by_day = defaultdict(list)
    for r in test.itertuples():
        if r.b365_f and 1.1 <= r.ps_f < 1.6:
            by_day[r.date].append((r.p_pin, r.b365_f, r.y == 1, r.ps_f))
    for legs in (1, 2, 3, 4):
        res, resp = [], []
        for d, L in by_day.items():
            L = sorted(L, key=lambda x: -x[0])[:legs]
            if len(L) < legs:
                continue
            o = np.prod([x[1] for x in L]); op = np.prod([x[3] for x in L])
            w = all(x[2] for x in L)
            res.append((o, w)); resp.append((op, w))
        n, m, se = roi(res)
        n2, m2, se2 = roi(resp)
        print(f"   {legs}-leg  n={n:>5}  hit {np.mean([w for _, w in res]):.0%}  ROI at Bet365 {m:+6.1%} ± {se:4.1%}   at Pinnacle {m2:+6.1%} ± {se2:4.1%}")

    # 5. total games
    print("\n== 5. Total games: Markov engine anchored to Pinnacle's price vs games played (completed matches, 2018-24)")
    G = test[test.done & (test.games > 0)].sample(min(6000, int((test.done & (test.games > 0)).sum())), random_state=1)
    rows = []
    for r in G.itertuples():
        pa, pb = markov.solve_serve_mix(r.p_pin, r.spw_sum, r.bo, markov.DAY_SD[r.tour])
        D = markov.mix_dist(pa, pb, r.bo, markov.DAY_SD[r.tour])
        tot = defaultdict(float)
        for k, v in D.items():
            tot[k[2]] += v
        mean = sum(t * v for t, v in tot.items())
        ts = sorted(tot)
        cdf, med = 0, None
        for t in ts:
            cdf += tot[t]
            if cdf >= 0.5:
                med = t
                break
        line = med + 0.5 if med is not None else round(mean) + 0.5     # a book-style half line near the median
        p_over = sum(v for t, v in tot.items() if t > line)
        rows.append({"tour": r.tour, "bo": r.bo, "p": r.p_pin, "surface": r.surface, "exp": mean, "act": r.games,
                     "line": line, "p_over": p_over, "over": r.games > line})
    T = pd.DataFrame(rows)
    T["band"] = pd.cut(T.p, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    print(f"   all: expected {T.exp.mean():.2f} games, actual {T.act.mean():.2f}; over the model's median line "
          f"{T.over.mean():.1%} vs model said {T.p_over.mean():.1%}")
    for keys in (["tour", "bo"], ["band"], ["surface"]):
        g = T.groupby(keys, observed=True).agg(n=("act", "size"), exp=("exp", "mean"), act=("act", "mean"), over=("over", "mean"), p_over=("p_over", "mean"))
        for k, v in g.iterrows():
            if v.n >= 150:
                print(f"   {str(k):<28} n={int(v.n):>5}  expected {v.exp:5.2f}  actual {v.act:5.2f}  ({v.act-v.exp:+.2f})   over hit {v.over:.1%} vs model {v.p_over:.1%}")

    # 6. upsets by level
    print("\n== 6. Upsets by level (all matches 2015-26 in our history; favourite by our pre-match rating)")
    F = pd.read_csv(f"{ROOT}/features.csv", low_memory=False, usecols=["date", "tour", "level", "y", "p_elo"])
    F = F[F.date >= "2015-01-01"]
    F["pf"] = np.maximum(F.p_elo, 1 - F.p_elo)
    F["fav_won"] = np.where(F.p_elo >= 0.5, F.y == 1, F.y == 0)
    names = {"G": "Grand Slams", "M": "Masters 1000", "A": "ATP/WTA 250-500", "F": "Tour finals", "C": "Challengers / WTA125",
             "S": "ITF / Futures", "Q": "Qualifying", "D": "Davis/BJK Cup"}
    for lv, g in F.groupby("level"):
        if len(g) < 2000:
            continue
        close = g[g.pf.between(0.6, 0.75)]
        print(f"   {names.get(lv, lv):<22} n={len(g):>7}  favourite won {g.fav_won.mean():.1%} (rating said {g.pf.mean():.1%}, {g.fav_won.mean()-g.pf.mean():+.1%})"
              f"   | fav 60-75%: won {close.fav_won.mean():.1%} vs {close.pf.mean():.1%}")
    try:
        from tools.kalshi_backtest import load as kload, fee
        K = kload()
        K["fav_a"] = K.px >= 0.5
        print("\n   Kalshi, real ask + fee, by series (all settled matches we have):")
        for s, g in K.groupby("series"):
            if len(g) < 150:
                continue
            fav, dog, big = [], [], []
            for r in g.itertuples():
                pa, pbk = r.ask_a, r.ask_b
                fa = r.px >= 0.5
                fp, dp = (pa, pbk) if fa else (pbk, pa)
                fw = (r.y == 1) == fa
                for price, won, lst in ((fp, fw, fav), (dp, not fw, dog)):
                    if 0.03 <= price <= 0.97:
                        c = price + fee(price)
                        lst.append((1 - c) / c if won else -1.0)
                if fp >= 0.75 and 0.03 <= dp:
                    c = dp + fee(dp)
                    big.append((1 - c) / c if not fw else -1.0)
            fmt = lambda L: f"{np.mean(L):+6.1%} ± {np.std(L)/math.sqrt(len(L)):4.1%}" if L else "  n/a"
            mid = np.mean([max(r.px, 1 - r.px) for r in g.itertuples()])
            fwr = np.mean([(r.y == 1) == (r.px >= 0.5) for r in g.itertuples()])
            print(f"   {s:<24} n={len(g):>5}  fav priced {mid:.1%} won {fwr:.1%}  | back fav {fmt(fav)}  back dog {fmt(dog)}  big dogs (fav>=75c) {fmt(big)}")
    except Exception as e:  # noqa
        print("   (Kalshi part skipped:", e, ")")


def logit_v(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    main()
