r"""Market ratings: learn each player's strength from past closing prices instead of results alone, then predict the
NEXT match before its own price exists. If that gets close to the closing line, we can price matches the market
hasn't listed yet (later rounds of a draw, futures, early lines) far better than results-only Elo.

Walk-forward on Pinnacle-priced matches (ATP 2010-24, WTA 2020-24): each match is predicted from ratings built
only on earlier matches, then the ratings are nudged toward this match's closing price (and its result).

  .venv\Scripts\python.exe -m tools.market_elo
"""
import math
from collections import defaultdict

import numpy as np
import pandas as pd


def lg(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sg(x):
    return 1 / (1 + math.exp(-x))


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run(J, lr=0.25, lr_res=0.0, decay_days=365):
    m = defaultdict(float)
    n = defaultdict(int)
    last = {}
    rows = []
    for r in J.itertuples():
        iw, il = 1 / r.psw, 1 / r.psl
        pw = iw / (iw + il)
        kw, kl = r.kw, r.kl
        for k in (kw, kl):                                    # shrink stale ratings toward 0 (unknown)
            if k in last:
                gap = (r.date - last[k]).days
                if gap > 30:
                    m[k] *= math.exp(-(gap - 30) / decay_days)
        pred = m[kw] - m[kl]
        rows.append((r.date.year, pw, sg(pred), r.pe_w, min(n[kw], n[kl]), r.tour))
        err = lg(pw) - pred                                  # learn the market's view of this match
        res = (1 - sg(pred))                                 # and a little from the result (winner = kw)
        for k, sgn in ((kw, 1), (kl, -1)):
            step = lr / (1 + 0.05 * n[k]) + 0.05
            m[k] += sgn * (step * err / 2 + lr_res * res)
            n[k] += 1
            last[k] = r.date
    return pd.DataFrame(rows, columns=["year", "pin", "mkt", "elo", "nmin", "tour"])


def main():
    J = pd.read_pickle("work/rules_lab.pkl")
    J = J[np.isfinite(J.psw.astype(float)) & np.isfinite(J.psl.astype(float))].sort_values("date").reset_index(drop=True)
    print(f"{len(J):,} priced matches; scores on 2018-24, only where both players have 10+ earlier priced matches\n")
    import sys
    for lr in ([float(x) for x in sys.argv[1:]] or (0.15, 0.3, 0.5)):
        R = run(J, lr=lr)
        T = R[(R.year >= 2018) & (R.nmin >= 10)]
        y = np.ones(len(T))                                  # rows are winner-first: y = 1 for the winner
        # log-loss is orientation-free, so score as "probability given to the eventual winner"
        b = lambda p: float(-np.mean(np.log(np.clip(p, 1e-4, 1))))
        X = np.c_[[lg(x) for x in T.mkt], [lg(x) for x in T.elo]]
        # blend weights fitted on 2010-17 (symmetrised so the winner-first order can't leak)
        Tr = R[(R.year < 2018) & (R.nmin >= 10)]
        Xr = np.c_[[lg(x) for x in Tr.mkt], [lg(x) for x in Tr.elo]]
        Xs = np.r_[Xr, -Xr]
        ys = np.r_[np.ones(len(Xr)), np.zeros(len(Xr))]
        from sklearn.linear_model import LogisticRegression
        L = LogisticRegression(C=100, fit_intercept=False).fit(Xs, ys)
        pb = L.predict_proba(X)[:, 1]
        print(f"lr {lr}: n={len(T):,}  Pinnacle close {b(T.pin.values):.4f} | results-only Elo {b(T.elo.values):.4f} | "
              f"market ratings {b(T.mkt.values):.4f} | blend {b(pb):.4f} (weights market {L.coef_[0][0]:.2f}, Elo {L.coef_[0][1]:.2f})")
        for t in ("atp", "wta"):
            g = T[T.tour == t]
            if len(g) > 300:
                print(f"     {t}: n={len(g):,} Pinnacle {b(g.pin.values):.4f}  Elo {b(g.elo.values):.4f}  market ratings {b(g.mkt.values):.4f}")


if __name__ == "__main__" and "--write" not in __import__("sys").argv:
    main()


def write_seed():
    """Seed state/market_ratings.json.gz: Pinnacle closing prices 2010-24, then Kalshi pre-match mids 2025-26.
    The Kalshi stretch doubles as an out-of-sample check of the live blend (edge/mrating weights)."""
    from datetime import date as _d
    from edge.mrating import MR, SEED, _lg
    from tools.kalshi_backtest import load
    J = pd.read_pickle("work/rules_lab.pkl")
    J = J[np.isfinite(J.psw.astype(float)) & np.isfinite(J.psl.astype(float))].sort_values("date").reset_index(drop=True)
    M = MR()
    for r in J.itertuples():
        iw, il = 1 / r.psw, 1 / r.psl
        M.update(r.kw, r.kl, iw / (iw + il), r.date.toordinal())
    K = load().sort_values(["date", "qts"]).reset_index(drop=True)
    rows = []
    for r in K.itertuples():
        t = _d.fromisoformat(r.date).toordinal()
        x = M.logit(r.ka, r.kb, t)
        if x is not None and r.p_model == r.p_model:
            rows.append((r.y, r.px, sg(_lg(r.p_model)), sg(M.blend(_lg(r.p_model), r.ka, r.kb, t)), sg(x)))
        M.update(r.ka, r.kb, r.px, t)
    T = pd.DataFrame(rows, columns=["y", "mkt", "model", "blend", "mr"])
    print(f"Kalshi check (matches where both players already had market ratings): n={len(T):,}")
    print(f"   Kalshi price {ll(T.y, T.mkt):.4f} | our model {ll(T.y, T.model):.4f} | market ratings {ll(T.y, T.mr):.4f} | model + market ratings {ll(T.y, T.blend):.4f}")
    M.save(SEED)
    print(f"wrote {SEED} ({len(M.m):,} players)")


if __name__ == "__main__" and "--write" in __import__("sys").argv:
    write_seed()
