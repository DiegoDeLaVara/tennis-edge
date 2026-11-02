"""Fit the pre-match model walk-forward on work/features.csv and write model/coef.json.

  .venv\\Scripts\\python.exe -m tools.fit
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LogisticRegression

from edge import markov
from edge.model import design, FEATURES

ROOT = Path(__file__).resolve().parents[1]


def stats_grid(best_of):
    S = np.round(np.arange(0.90, 1.601, 0.01), 3)
    D = np.round(np.arange(-0.40, 0.401, 0.01), 3)
    G = np.zeros((len(S), len(D)))
    for i, s in enumerate(S):
        for j, d in enumerate(D):
            G[i, j] = markov.fast_p_match(min(max(s / 2 + d / 2, 0.02), 0.98), min(max(s / 2 - d / 2, 0.02), 0.98), best_of)
    return RegularGridInterpolator((S, D), G, bounds_error=False, fill_value=None)


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    df = pd.read_csv(ROOT / "work" / "features.csv", low_memory=False)
    df = df[(df.n_a >= 3) & (df.n_b >= 3)].copy()
    df["year"] = df.date.str[:4].astype(int)
    print("rows", len(df))
    g3, g5 = stats_grid(3), stats_grid(5)
    pts = np.c_[(df.pa + df.pb).clip(0.9, 1.6), (df.pa - df.pb).clip(-0.4, 0.4)]
    df["p_stats"] = np.where(df.best_of == 5, g5(pts), g3(pts)).clip(0.003, 0.997)
    recs = df.to_dict("records")
    X = np.array([design(r, r["p_stats"], r["tour"], r["level"], r["best_of"]) for r in recs])
    y = df.y.values
    yrs = df.year.values
    rows = []
    for test in range(2016, 2027):
        tr, te = yrs < test, yrs == test
        if te.sum() == 0:
            continue
        lr = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], y[tr])
        p = lr.predict_proba(X[te])[:, 1]
        pe = df.p_elo.values[te]
        rows.append({"year": test, "n": int(te.sum()), "ll_elo": ll(y[te], pe), "ll_model": ll(y[te], p),
                     "acc_elo": float(np.mean((pe > .5) == y[te])), "acc_model": float(np.mean((p > .5) == y[te]))})
        df.loc[te, "p_model"] = p
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    tot = df[df.year >= 2016]
    print("overall 2016+: ll elo %.4f model %.4f | acc elo %.4f model %.4f" % (ll(tot.y.values, tot.p_elo.values), ll(tot.y.values, tot.p_model.values),
          np.mean((tot.p_elo > .5) == tot.y), np.mean((tot.p_model > .5) == tot.y)))
    for name, sub in tot.groupby(["tour", "level"]):
        if len(sub) > 2000:
            print(name, len(sub), "ll %.4f acc %.3f" % (ll(sub.y.values, sub.p_model.values), np.mean((sub.p_model > .5) == sub.y)))
    # calibration check
    tot = tot.assign(bin=(tot.p_model * 10).astype(int).clip(0, 9))
    print(tot.groupby("bin").agg(n=("y", "size"), pred=("p_model", "mean"), actual=("y", "mean")).round(3).to_string())
    final = LogisticRegression(C=1.0, max_iter=500).fit(X[yrs >= 2012], y[yrs >= 2012])
    out = {"w": [round(float(v), 6) for v in final.coef_[0]], "b": round(float(final.intercept_[0]), 6), "features": FEATURES,
           "trained_rows": int((yrs >= 2012).sum()), "walk_forward": rows}
    old = ROOT / "model" / "coef.json"
    if old.exists():
        prev = json.loads(old.read_text())
        if "market" in prev:
            out["market"] = prev["market"]
    old.parent.mkdir(exist_ok=True)
    old.write_text(json.dumps(out, indent=1))
    print(dict(zip(FEATURES, out["w"])), out["b"])
    df[df.year >= 2025].to_csv(ROOT / "work" / "recent_preds.csv", index=False)


if __name__ == "__main__":
    main()
