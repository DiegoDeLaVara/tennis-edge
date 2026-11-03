r"""Does recent form predict winners — alone, on top of ratings, and on top of Kalshi's price?

Form definitions (all from matches BEFORE the one being predicted):
  last5 / last10 / last20  win % over the last N matches
  win30                     win % in the last 30 days (needs 3+ matches)
  beat10                    results beyond the rating's expectation, last 10 matches (positive = playing above level)
  form45                    the model's own form term (decayed results beyond expectation, 45-day half-life)
  quality10                 last-10 wins weighted by opponent rating (beating good players counts more)

  .venv\Scripts\python.exe -m tools.form_study
"""
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tools.kalshi_backtest import load, fee, logit, ll


def build():
    F = pd.read_csv("work/features.csv", low_memory=False, usecols=["date", "tour", "level", "a", "b", "y", "p_elo", "elo_a", "elo_b", "form_a", "form_b", "n_a", "n_b"])
    F["t"] = pd.to_datetime(F.date).map(pd.Timestamp.toordinal)
    hist = defaultdict(lambda: deque(maxlen=20))     # key -> (t, won, expected, opp_elo)
    cols = {c: np.zeros(len(F)) for c in ("l5_a", "l5_b", "l10_a", "l10_b", "l20_a", "l20_b", "w30_a", "w30_b", "b10_a", "b10_b", "q10_a", "q10_b")}
    ok = np.zeros(len(F), dtype=bool)

    def feats(h, t):
        L = list(h)
        if len(L) < 5:
            return None
        w = [x[1] for x in L]
        l30 = [x for x in L if t - x[0] <= 30]
        b10 = [x[1] - x[2] for x in L[-10:]]
        q10 = [(x[1] - 0.5) * (x[3] / 1500.0) for x in L[-10:]]
        return (np.mean(w[-5:]), np.mean(w[-10:]), np.mean(w), np.mean([x[1] for x in l30]) if len(l30) >= 3 else np.mean(w[-5:]),
                np.mean(b10), np.mean(q10))
    for i, r in enumerate(F.itertuples(index=False)):
        fa, fb = feats(hist[r.a], r.t), feats(hist[r.b], r.t)
        if fa and fb:
            ok[i] = True
            for j, nm in enumerate(("l5", "l10", "l20", "w30", "b10", "q10")):
                cols[nm + "_a"][i], cols[nm + "_b"][i] = fa[j], fb[j]
        ya = r.y
        hist[r.a].append((r.t, ya, r.p_elo, r.elo_b))
        hist[r.b].append((r.t, 1 - ya, 1 - r.p_elo, r.elo_a))
    for c, v in cols.items():
        F[c] = v
    F = F[ok & (F.n_a >= 10) & (F.n_b >= 10)].copy()
    for nm in ("l5", "l10", "l20", "w30", "b10", "q10"):
        F["d_" + nm] = F[nm + "_a"] - F[nm + "_b"]
    F["d_form45"] = F.form_a - F.form_b
    F["year"] = F.date.str[:4].astype(int)
    return F


DEFS = {"d_l5": "win % last 5", "d_l10": "win % last 10", "d_l20": "win % last 20", "d_w30": "win % last 30 days",
        "d_b10": "beyond expectation, last 10", "d_form45": "model's form term (45-day)", "d_q10": "quality-weighted last 10"}


def main():
    F = build()
    tr, te = F[F.year.between(2012, 2021)], F[F.year >= 2022]
    y = te.y.values
    print(f"matches: train 2012-21 {len(tr):,}, test 2022-26 {len(te):,}\n")
    le = logit(tr.p_elo.values), logit(te.p_elo.values)
    base = LogisticRegression(max_iter=500).fit(le[0][:, None], tr.y).predict_proba(le[1][:, None])[:, 1]
    print(f"== Predicting the winner on 2022-26 (log-loss, lower = better; coin flip = 0.6931)")
    print(f"   ratings (Elo) alone ............................ {ll(y, base):.4f}   accuracy {np.mean((base > .5) == y):.3f}")
    for c, name in DEFS.items():
        alone = LogisticRegression(max_iter=500).fit(tr[[c]].values, tr.y).predict_proba(te[[c]].values)[:, 1]
        both = LogisticRegression(max_iter=500).fit(np.c_[le[0], tr[c].values], tr.y).predict_proba(np.c_[le[1], te[c].values])[:, 1]
        print(f"   {name:30s} alone {ll(y, alone):.4f} (acc {np.mean((alone > .5) == y):.3f})   ratings + it {ll(y, both):.4f}  gain {ll(y, base) - ll(y, both):+.4f}")
    allc = list(DEFS)
    both = LogisticRegression(max_iter=500).fit(np.c_[le[0], tr[allc].values], tr.y).predict_proba(np.c_[le[1], te[allc].values])[:, 1]
    print(f"   ratings + every form measure ................... {ll(y, both):.4f}  gain {ll(y, base) - ll(y, both):+.4f}")

    # does a hot player beat a cold one more than ratings say? simple buckets
    print("\n== Hot vs cold (2022-26): favourite by rating, by form gap (win % last 10)")
    te2 = te.assign(fav_hot=np.where(te.p_elo >= 0.5, te.d_l10, -te.d_l10), fav_won=np.where(te.p_elo >= 0.5, te.y, 1 - te.y), fav_p=np.maximum(te.p_elo, 1 - te.p_elo))
    for lo, hi, lab in ((-1, -0.3, "favourite much colder"), (-0.3, -0.1, "favourite colder"), (-0.1, 0.1, "similar form"), (0.1, 0.3, "favourite hotter"), (0.3, 1.01, "favourite much hotter")):
        g = te2[(te2.fav_hot >= lo) & (te2.fav_hot < hi)]
        print(f"   {lab:24s} n={len(g):6d}  rating said {g.fav_p.mean():.3f}  favourite actually won {g.fav_won.mean():.3f}  ({(g.fav_won.mean() - g.fav_p.mean()) * 100:+.1f} pts)")

    # Kalshi: beyond the market price
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
        d = {"date": r.date, "y": r.y, "px": r.px, "ask_a": r.ask_a, "ask_b": r.ask_b}
        for c in allc:
            d[c] = s * getattr(best, c)
        rows.append(d)
    D = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    cut = int(len(D) * 0.6)
    Dd, Dv = D.iloc[:cut], D.iloc[cut:]
    yv = Dv.y.values
    print(f"\n== Kalshi ({len(D)} matches, newest 40% = test): does form add anything to the market price?")
    print(f"   Kalshi price alone ............................. {ll(yv, Dv.px.values):.4f}")
    for c, name in DEFS.items():
        lr = LogisticRegression(max_iter=1000).fit(np.c_[logit(Dd.px.values), Dd[c].values], Dd.y)
        p = lr.predict_proba(np.c_[logit(Dv.px.values), Dv[c].values])[:, 1]
        print(f"   price + {name:30s} {ll(yv, p):.4f}   coefficient {lr.coef_[0][1]:+.3f}")
    print("\n== Betting the hotter player when Kalshi makes them the underdog (newest 40%)")
    for thr in (0.2, 0.3, 0.4):
        out = []
        for r in Dv.itertuples():
            if abs(r.d_l10) < thr:
                continue
            hot = 0 if r.d_l10 > 0 else 1
            hot_p = r.px if hot == 0 else 1 - r.px
            if hot_p >= 0.5:
                continue
            price = r.ask_a if hot == 0 else r.ask_b
            if 0.05 <= price <= 0.95:
                c = price + fee(price)
                w = (r.y == 1) if hot == 0 else (r.y == 0)
                out.append((1 - c) / c if w else -1.0)
        if out:
            a = np.array(out)
            print(f"   form gap >= {int(thr * 100)} pts (last 10): n={len(a)} ROI {a.mean() * 100:+.1f}% ± {a.std() / math.sqrt(len(a)) * 100:.1f}")


if __name__ == "__main__":
    main()
