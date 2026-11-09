r"""Everything at once: instead of testing one rule at a time, give a gradient-boosted model EVERY feature we have built
(motivation, travel, entry status, retirements, book disagreement, fatigue, form, age, home, money flow, our model)
plus the market price, and ask whether it predicts results - and upsets in particular - better than the price alone.

Two data sets, each scored out of sample both ways (fit on one time half, test on the other):
  Pinnacle: 47k ATP/WTA tour matches 2010-24 (feature caches from tools.motivation, travel_study, entry_study,
            trap_study - run those first), returns at Pinnacle closing odds.
  Kalshi:   8k tour/Challenger matches 2025-26 (work/locks_df.pkl = tools.upset_risk.kalshi(), see tools.locks70),
            returns at the real ask + fee; includes our own model's number.
Then "upset finder": the matches where the model most expects an upset beyond the price - do they happen, and does
backing those underdogs pay?

  .venv\Scripts\python.exe -m tools.upset_ml [--pinnacle | --kalshi]

Result (2026-11-08, CORRECTED same day): the first run included 189 exact 50/50 Pinnacle prices where the source
studies label the winner as "favourite" (y always 1) - a leak the model exploited. With them dropped (pinnacle_frame
now does): all-features model = Pinnacle close (log-loss -0.0003 / +0.0006); its "safest" 5% of favourites flip
(+6.7% / -4.9%); walk-forward price+ranks table -0.1 / -0.2 / -5.4 / -2.5% (all periods -2.4% +- 1.8). Upset finder:
backing its top-20% dogs +3.0% / +1.6% (+-2.2) - marginal, and it failed on Kalshi. Kalshi: worse than the price.
Conclusion: no safe-favourite or upset model beats the market. Lesson: drop exact 50/50 prices before labelling sides.
"""
import math
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from tools.kalshi_backtest import fee


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gbm():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=400, l2_regularization=5.0,
                                          min_samples_leaf=80, early_stopping=True, validation_fraction=0.2,
                                          n_iter_no_change=30, random_state=0)


def pinnacle_frame():
    M = pd.read_pickle("work/motivation.pkl")
    T = pd.read_pickle("work/travel_study.pkl")
    R = pd.read_pickle("work/trap_study.pkl")
    assert (M.ps_f.values == T.ps_f.values).all() and len(R) == len(M) and (np.isclose(R.ps_f.values, M.ps_f.values)).all()
    D = M.copy()
    for c in T.columns:
        if c not in D:
            D[c] = T[c].values
    for c in R.columns:
        if c not in D:
            D[c] = R[c].values
    E = pd.read_pickle("work/entry_study.pkl")
    E = E.drop_duplicates(["date", "ps_f", "ps_d", "tour"])
    D = D.merge(E[["date", "ps_f", "ps_d", "tour", "entry_f", "entry_d", "age_f", "age_d"]], on=["date", "ps_f", "ps_d", "tour"], how="left")
    D["derby"] = [float(a is not None and a == b) for a, b in zip(D.nat_f, D.nat_d)]
    for s in ("Q", "WC", "LL", "PR"):
        D[f"e{s}_f"] = (D.entry_f == s).astype(float)
        D[f"e{s}_d"] = (D.entry_d == s).astype(float)
    for c in ("series", "round", "surf", "court"):
        D[c + "_c"] = D[c].astype("category").cat.codes
    D["month"] = pd.to_datetime(D.date).dt.month
    D["lp"] = lg(D.p)
    drop = {"date", "y", "p", "ps_f", "ps_d", "tour", "series", "round", "surf", "court", "nat_f", "nat_d", "entry_f", "entry_d", "done", "max_f", "max_d"}
    feats = [c for c in D.columns if c not in drop]
    # exact 50/50 prices: the source studies label the winner as "favourite" there, so y is always 1 (a leak) - drop them
    D = D[np.abs(D.p.values - 0.5) > 1e-6]
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    X = D[feats].astype(float)
    return D, X, D.p.values, D.ps_f.values - 1, D.ps_d.values - 1


def kalshi_frame():
    D = pd.read_pickle("work/locks_df.pkl").sort_values("date", kind="stable").reset_index(drop=True)
    fa = D.px >= 0.5
    D["ask_f"] = np.where(fa, D.ask_a, D.ask_b)
    D["ask_d"] = np.where(fa, D.ask_b, D.ask_a)
    D["y"] = D.fw
    D["lp"] = lg(D.pf)
    D["lpm"] = lg(D.pm_f.fillna(D.pf))
    D["series_c"] = D.series.astype("category").cat.codes
    D["round_c"] = D["round"].astype("category").cat.codes
    keep = [c for c in D.columns if c.endswith("_f") or c.endswith("_d")]
    keep = [c for c in keep if c not in ("ask_f", "ask_d", "pm_f") and D[c].dtype != object]
    keep += ["lp", "lpm", "series_c", "round_c", "month", "spread", "vol", "n_trades", "steam", "big_net"]
    X = D[keep].apply(pd.to_numeric, errors="coerce").astype(float)
    cf, cd = D.ask_f + fee(D.ask_f), D.ask_d + fee(D.ask_d)
    return D, X, D.pf.values, ((1 - cf) / cf).values, ((1 - cd) / cd).values


def evaluate(name, D, X, p, win_f, win_d):
    y = D.y.values
    half = (np.arange(len(D)) >= len(D) // 2).astype(int)
    q = np.full(len(D), np.nan)
    imp = []
    for tr, te in ((0, 1), (1, 0)):
        m = gbm().fit(X[half == tr], y[half == tr])
        q[half == te] = m.predict_proba(X[half == te])[:, 1]
        imp.append(m)
    print(f"\n===== {name}: {len(D):,} matches, {X.shape[1]} features")
    for h in (0, 1):
        s = half == h
        print(f"   half {h} (scored by the model fitted on the other half): log-loss market {ll(y[s], p[s]):.4f}  "
              f"all-features model {ll(y[s], q[s]):.4f}  ({ll(y[s], q[s]) - ll(y[s], p[s]):+.4f})")
    # upset finder: model's upset chance beyond the price
    extra = (1 - q) - (1 - p)
    print("   Upset finder - matches ranked by how much MORE the model expects an upset than the price does:")
    for top in (0.02, 0.05, 0.10, 0.20):
        parts = []
        for h in (0, 1):
            s = half == h
            thr = np.quantile(extra[s], 1 - top)
            g = s & (extra >= thr)
            up = (y[g] == 0)
            r = np.where(up, win_d[g], -1.0)
            parts.append(f"half {h}: n={g.sum():>5} upsets {up.mean():5.1%} vs priced {(1 - p[g]).mean():5.1%}  back dog {r.mean():+6.1%} ± {r.std(ddof=1) / math.sqrt(g.sum()):4.1%}")
        print(f"     top {int(top * 100):>2}%   " + "   ".join(parts))
    print("   ...and the matches the model thinks are SAFEST (back the favourite):")
    for top in (0.05, 0.10, 0.20):
        parts = []
        for h in (0, 1):
            s = half == h
            thr = np.quantile(extra[s], top)
            g = s & (extra <= thr)
            r = np.where(y[g] == 1, win_f[g], -1.0)
            parts.append(f"half {h}: n={g.sum():>5} fav won {(y[g] == 1).mean():5.1%} vs priced {p[g].mean():5.1%}  back fav {r.mean():+6.1%} ± {r.std(ddof=1) / math.sqrt(g.sum()):4.1%}")
        print(f"     top {int(top * 100):>2}%   " + "   ".join(parts))
    try:
        from sklearn.inspection import permutation_importance
        s = half == 1
        pi = permutation_importance(imp[0], X[s], y[s], scoring="neg_log_loss", n_repeats=3, random_state=0)
        order = np.argsort(-pi.importances_mean)[:10]
        print("   Features the half-0 model leaned on most (log-loss lost when shuffled, on half 1): " +
              ", ".join(f"{X.columns[i]} {pi.importances_mean[i] * 1000:+.2f}" for i in order))
    except Exception as e:  # noqa
        print("   (importance failed:", e, ")")


def main():
    if "--kalshi" not in sys.argv:
        evaluate("Pinnacle closing odds, tour 2010-24", *pinnacle_frame())
    if "--pinnacle" not in sys.argv:
        evaluate("Kalshi tour/Challenger 2025-26 (real ask + fee)", *kalshi_frame())


if __name__ == "__main__":
    main()
