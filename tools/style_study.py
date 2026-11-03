r"""Do head-to-head and style-matchup records improve predictions?

1. History, walk-forward by year: the fitted model's features with and without the matchup features
   (vs-style record beyond expectation, head-to-head beyond expectation).
2. Kalshi: do the same features predict the winner beyond Kalshi's pre-match price?

  .venv\Scripts\python.exe -m tools.style_study
"""
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from edge.model import design
from tools.kalshi_backtest import load, fee, logit, ll


def main():
    F = pd.read_csv("work/features.csv", low_memory=False)
    S = pd.read_csv("work/style.csv", low_memory=False)
    assert len(F) == len(S), (len(F), len(S))
    F = pd.concat([F.reset_index(drop=True), S[["style_a", "style_b", "vs_a", "vs_b", "vs_n_a", "vs_n_b", "h2h_res"]].reset_index(drop=True)], axis=1)
    F = F[(F.n_a >= 3) & (F.n_b >= 3)].copy()
    F["year"] = F.date.str[:4].astype(int)
    print("rows", len(F), "| share with a style for both players:", round((F.style_a.notna() & F.style_b.notna()).mean(), 3))
    # matchup features
    F["m_vs"] = F.vs_a - F.vs_b
    F["m_h2h"] = F.h2h_res
    F["lefty_vs"] = 0.0
    print("\nstyle mix:", F.style_a.value_counts(normalize=True).round(3).to_dict())

    # 1. walk-forward on history: base design (p_stats approximated by p_elo to keep this fast) +/- matchup
    recs = F.to_dict("records")
    Xb = np.array([design(r, r["p_elo"], r["tour"], r["level"], r["best_of"]) for r in recs])
    Xm = np.c_[Xb, F[["m_vs", "m_h2h"]].values]
    y, yr = F.y.values, F.year.values
    rows = []
    for test in range(2016, 2027):
        tr, te = yr < test, yr == test
        if te.sum() == 0:
            continue
        pb = LogisticRegression(C=1.0, max_iter=500).fit(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1]
        lm = LogisticRegression(C=1.0, max_iter=500).fit(Xm[tr], y[tr])
        pm = lm.predict_proba(Xm[te])[:, 1]
        rows.append((test, te.sum(), ll(y[te], pb), ll(y[te], pm), np.mean((pb > .5) == y[te]), np.mean((pm > .5) == y[te])))
    print("\n== walk-forward on all matches (lower log-loss is better)")
    print(" year      n   base ll  +matchup ll   base acc  +matchup acc")
    for r in rows:
        print(f" {r[0]}  {r[1]:6d}   {r[2]:.4f}     {r[3]:.4f}      {r[4]:.4f}     {r[5]:.4f}")
    tot_b = np.average([r[2] for r in rows], weights=[r[1] for r in rows])
    tot_m = np.average([r[3] for r in rows], weights=[r[1] for r in rows])
    print(f" overall: base {tot_b:.4f}  +matchup {tot_m:.4f}  (difference {tot_b - tot_m:+.4f})")
    full = LogisticRegression(C=1.0, max_iter=500).fit(Xm[yr >= 2012], y[yr >= 2012])
    print(f" matchup coefficients (logit units): vs-style {full.coef_[0][-2]:+.3f}, h2h-beyond-rating {full.coef_[0][-1]:+.3f}")
    # where matchup records are deep enough to matter
    deep = (F.vs_n_a >= 10) & (F.vs_n_b >= 10)
    for test_rows, label in ((deep, "both players 10+ matches vs the other's style"), (F.h2h_n >= 3 if "h2h_n" in F else deep, "3+ previous meetings")):
        pass

    # 2. Kalshi: beyond the market price
    K = load()
    idx = {}
    for r in F[F.year >= 2025].itertuples():
        idx.setdefault((r.a, r.b), []).append(r)
    rows = []
    for r in K.itertuples():
        best, flip, gap = None, False, 99
        for (a, b, fl) in ((r.ka, r.kb, False), (r.kb, r.ka, True)):
            for c in idx.get((a, b), []):
                g = abs((pd.Timestamp(c.date) - pd.Timestamp(r.date)).days)
                if g <= 7 and g < gap:
                    best, flip, gap = c, fl, g
        if best is None:
            continue
        s = -1 if flip else 1
        rows.append({"date": r.date, "y": r.y, "px": r.px, "ask_a": r.ask_a, "ask_b": r.ask_b,
                     "m_vs": s * best.m_vs, "m_h2h": s * best.m_h2h, "deep": int(best.vs_n_a >= 10 and best.vs_n_b >= 10)})
    D = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    cut = int(len(D) * 0.6)
    Dd, Dv = D.iloc[:cut], D.iloc[cut:]
    X = lambda d: np.c_[logit(d.px.values), d[["m_vs", "m_h2h"]].values]
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X(Dd), Dd.y)
    pv = lr.predict_proba(X(Dv))[:, 1]
    print(f"\n== Kalshi ({len(D)} matches): effect beyond the price, fit on older 60%: vs-style {lr.coef_[0][1]:+.3f}, h2h {lr.coef_[0][2]:+.3f}")
    print(f"   newest 40% log-loss: market {ll(Dv.y.values, Dv.px.values):.4f}  market+matchup {ll(Dv.y.values, pv):.4f}")

    def bet(d, p, th):
        out = []
        for r, q in zip(d.itertuples(), p):
            for side in (0, 1):
                ps = q if side == 0 else 1 - q
                price = r.ask_a if side == 0 else r.ask_b
                if 0.05 <= price <= 0.95 and ps / (price + fee(price)) - 1 >= th:
                    c = price + fee(price)
                    w = (r.y == 1) if side == 0 else (r.y == 0)
                    out.append((1 - c) / c if w else -1.0)
        a = np.array(out)
        return f"n={len(a):5d} ROI {a.mean() * 100:+6.1f}% ± {a.std() / math.sqrt(len(a)) * 100:4.1f}" if len(a) else "n=0"
    for th in (0.02, 0.04, 0.07):
        print(f"   bet market+matchup edge >= {int(th * 100)}% (newest 40%): {bet(Dv, pv, th)}")
    # simple rule: strong matchup signal both ways
    for thr in (0.06, 0.10):
        sub = Dv[Dv.m_vs.abs() >= thr]
        out = []
        for r in sub.itertuples():
            side = 0 if r.m_vs > 0 else 1
            price = r.ask_a if side == 0 else r.ask_b
            if 0.05 <= price <= 0.95:
                c = price + fee(price)
                w = (r.y == 1) if side == 0 else (r.y == 0)
                out.append((1 - c) / c if w else -1.0)
        if out:
            a = np.array(out)
            print(f"   back the player with the better style matchup (|gap| >= {thr}): n={len(a)} ROI {a.mean() * 100:+.1f}% ± {a.std() / math.sqrt(len(a)) * 100:.1f}")


if __name__ == "__main__":
    main()
