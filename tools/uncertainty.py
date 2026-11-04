r"""Uncertainty-aware ratings. A player with few rated matches, or back from a long layoff, has a rating we know less
well, so a prediction involving them should sit closer to 50/50 (the Glicko idea). We measure that uncertainty per
player and shrink the model's logit:

    u      = 1/(n_a + k) + 1/(n_b + k) + c * (min(days_a, 365) + min(days_b, 365)) / 365
    logit' = logit / sqrt(1 + a * u)

a, c, k are fitted on 2016-20 walk-forward predictions and judged on 2021-26; then checked on Kalshi (does the
shrunk model blend better with the market, and is the market itself worse calibrated on uncertain matches?).
--write stores the parameters in model/coef.json ("uncertainty").

  .venv\Scripts\python.exe -m tools.uncertainty [--write]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from edge.model import design
from tools.fit import stats_grid, ll

ROOT = Path(__file__).resolve().parents[1]


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def unc(n_a, n_b, d_a, d_b, c, k):
    return 1 / (n_a + k) + 1 / (n_b + k) + c * (np.minimum(d_a, 365) + np.minimum(d_b, 365)) / 365


def walk_forward():
    cache = ROOT / "work" / "wf_preds.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = pd.read_csv(ROOT / "work" / "features.csv", low_memory=False)
    df = df[(df.n_a >= 3) & (df.n_b >= 3)].copy()
    df["year"] = df.date.str[:4].astype(int)
    g3, g5 = stats_grid(3), stats_grid(5)
    pts = np.c_[(df.pa + df.pb).clip(0.9, 1.6), (df.pa - df.pb).clip(-0.4, 0.4)]
    df["p_stats"] = np.where(df.best_of == 5, g5(pts), g3(pts)).clip(0.003, 0.997)
    X = np.array([design(r, r["p_stats"], r["tour"], r["level"], r["best_of"]) for r in df.to_dict("records")])
    y, yrs = df.y.values, df.year.values
    df["p_model"] = np.nan
    for test in range(2016, 2027):
        tr, te = yrs < test, yrs == test
        if te.sum():
            df.loc[te, "p_model"] = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    out = df[df.year >= 2016][["date", "year", "tour", "level", "a", "b", "y", "p_model", "p_elo", "p_elo_all", "p_stats", "n_a", "n_b", "days_a", "days_b", "stat_w"]]
    out.to_pickle(cache)
    return out


def apply(D, a, c, k):
    u = unc(D.n_a.values, D.n_b.values, D.days_a.fillna(0).values, D.days_b.fillna(0).values, c, k)
    return 1 / (1 + np.exp(-lg(D.p_model.values) / np.sqrt(1 + a * u)))


def main():
    D = walk_forward()
    tr, te = D[D.year <= 2020], D[D.year >= 2021]
    print(f"walk-forward predictions: {len(D):,} matches (fit 2016-20: {len(tr):,}, test 2021-26: {len(te):,})")
    base_tr, base_te = ll(tr.y.values, tr.p_model.values), ll(te.y.values, te.p_model.values)
    best = None
    for a, c, k in itertools.product((0, 2, 5, 10, 20, 40), (0, 0.02, 0.05, 0.1, 0.2), (2, 5, 10, 20)):
        v = ll(tr.y.values, apply(tr, a, c, k))
        if best is None or v < best[0]:
            best = (v, a, c, k)
    v, a, c, k = best
    print(f"best on 2016-20: a={a} c={c} k={k}  log-loss {base_tr:.5f} -> {v:.5f}")
    pt = apply(te, a, c, k)
    print(f"2021-26 (unseen): log-loss {base_te:.5f} -> {ll(te.y.values, pt):.5f}  ({(base_te - ll(te.y.values, pt)) * 1e4:+.1f} bp)")
    u = unc(te.n_a.values, te.n_b.values, te.days_a.fillna(0).values, te.days_b.fillna(0).values, c, k)
    te = te.assign(u=u, p_new=pt)
    q = np.quantile(te.u, [0.5, 0.8, 0.95])
    print("\n   by uncertainty (2021-26):            n     log-loss before -> after   favourite won vs model said (before / after)")
    for name, m in (("low (bottom half)", te.u <= q[0]), ("medium", (te.u > q[0]) & (te.u <= q[1])), ("high (top 20%)", (te.u > q[1]) & (te.u <= q[2])),
                    ("very high (top 5%)", te.u > q[2])):
        g = te[m]
        pf = np.maximum(g.p_model, 1 - g.p_model); fw = np.where(g.p_model >= 0.5, g.y, 1 - g.y)
        pn = np.where(g.p_model >= 0.5, g.p_new, 1 - g.p_new)
        print(f"   {name:<30} {len(g):>7}   {ll(g.y.values, g.p_model.values):.4f} -> {ll(g.y.values, g.p_new.values):.4f}      "
              f"{fw.mean():.3f} vs {pf.mean():.3f} / {pn.mean():.3f}")
    for lv, g in te.groupby("level"):
        if len(g) > 5000:
            print(f"   level {lv:<3} n={len(g):>7}  {ll(g.y.values, g.p_model.values):.4f} -> {ll(g.y.values, g.p_new.values):.4f}")

    # ---- Kalshi: does it help next to the market?
    try:
        from tools.kalshi_backtest import load
        K = load()
        P = pd.read_csv(ROOT / "work" / "recent_preds.csv", low_memory=False)
        idx = {}
        for r in P.itertuples():
            idx.setdefault((r.a, r.b), []).append(r)
        rows = []
        for r in K.itertuples():
            hit = None
            for (x, z, flip) in ((r.ka, r.kb, False), (r.kb, r.ka, True)):
                for cnd in idx.get((x, z), []):
                    if abs((pd.Timestamp(cnd.date) - pd.Timestamp(r.date)).days) <= 7:
                        hit = (cnd, flip)
            if hit is None or r.p_model != r.p_model:
                continue
            cnd, flip = hit
            na, nb, da, db = (cnd.n_b, cnd.n_a, cnd.days_b, cnd.days_a) if flip else (cnd.n_a, cnd.n_b, cnd.days_a, cnd.days_b)
            rows.append({"y": r.y, "px": r.px, "p_model": r.p_model, "n_a": na, "n_b": nb, "days_a": da, "days_b": db, "date": r.date})
        Kd = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        Kd["p_new"] = apply(Kd, a, c, k)
        Kd["u"] = unc(Kd.n_a.values, Kd.n_b.values, Kd.days_a.fillna(0).values, Kd.days_b.fillna(0).values, c, k)
        half = len(Kd) // 2
        print(f"\nKalshi ({len(Kd):,} matches): model {ll(Kd.y.values, Kd.p_model.values):.4f} -> uncertainty-aware {ll(Kd.y.values, Kd.p_new.values):.4f}; "
              f"market {ll(Kd.y.values, Kd.px.values):.4f}")
        for nm, col in (("model", "p_model"), ("uncertainty-aware", "p_new")):
            Xa = np.c_[lg(Kd[col].values), lg(Kd.px.values)]
            m = LogisticRegression(C=100).fit(Xa[:half], Kd.y.values[:half])
            print(f"   blend with market, fitted on older half, scored on newer: {nm:<18} {ll(Kd.y.values[half:], m.predict_proba(Xa[half:])[:, 1]):.4f}"
                  f"   (market alone {ll(Kd.y.values[half:], Kd.px.values[half:]):.4f})")
        qk = np.quantile(Kd.u, 0.8)
        for nm, g in (("certain 80%", Kd[Kd.u <= qk]), ("uncertain top 20%", Kd[Kd.u > qk])):
            pf = np.maximum(g.px, 1 - g.px); fw = np.where(g.px >= 0.5, g.y, 1 - g.y)
            print(f"   market on {nm:<20} n={len(g):>5}: favourite won {fw.mean():.3f} vs priced {pf.mean():.3f}")
    except Exception as e:  # noqa
        print("Kalshi check skipped:", e)

    if "--write" in sys.argv:
        f = ROOT / "model" / "coef.json"
        d = json.loads(f.read_text())
        d["uncertainty"] = {"a": a, "c": c, "k": k, "note": "logit / sqrt(1 + a*u), u = 1/(n_a+k) + 1/(n_b+k) + c*(days_a+days_b)/365 (tools/uncertainty)"}
        f.write_text(json.dumps(d, indent=1))
        print("written to model/coef.json")


if __name__ == "__main__":
    main()
