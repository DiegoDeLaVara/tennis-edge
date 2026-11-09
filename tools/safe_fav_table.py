r"""NEGATIVE RESULT - NOT DEPLOYED. With the 50/50-price leak removed (see tools/upset_ml) the walk-forward table
version returned -0.1 / -0.2 / -5.4 / -2.5% (all periods -2.4%), so the board does not use it.

Export the Pinnacle "safe favourite" model (tools/upset_ml: gradient boosting on Pinnacle's no-vig price + both
players' rankings) as a lookup table the cloud refresh can read without scikit-learn.

Model: trained on all 47k Pinnacle-priced tour matches 2010-24. Walk-forward it picked favourites (top 5% of
"model chance minus Pinnacle chance") that returned +7..+22% at Pinnacle's own close in every later period, but it LOST
when applied to Kalshi prices (-9% / -2%), so it is only used with a live Pinnacle line, and only on paper.

Table: q[i_price][i_rank_fav][i_rank_dog] = model chance the favourite wins, price 0.50..0.95 (step 0.01), rankings on
a log grid (None = unranked). `thr` = the gap (q - price) at the 95th percentile over the training matches.

  .venv\Scripts\python.exe -m tools.safe_fav_table      -> model/safe_fav.json

"""
import json
from pathlib import Path

import numpy as np

from tools.upset_ml import gbm, pinnacle_frame

ROOT = Path(__file__).resolve().parents[1]
PRICES = [round(0.50 + 0.01 * i, 2) for i in range(46)]
RANKS = sorted({int(round(x)) for x in np.exp(np.linspace(0, np.log(2000), 48))})      # 42 log-spaced points


def snap_rank(r):
    if r is None or not np.isfinite(r) or r <= 0:
        return len(RANKS)                                   # last slot = unranked
    lr = np.log(r)
    return int(np.argmin([abs(lr - np.log(x)) for x in RANKS]))


def table_q(T, p, rf, rd):
    i = int(round((min(max(p, 0.5), 0.95) - 0.5) * 100))
    return T[i][snap_rank(rf)][snap_rank(rd)]


def main():
    D, X, p, wf, wd = pinnacle_frame()
    cs = ["lp", "rank_f", "rank_d"]
    m = gbm().fit(X[cs], D.y.values)
    rv = [float(r) for r in RANKS] + [np.nan]
    grid = np.array([[np.log(pp / (1 - pp)), a, b] for pp in PRICES for a in rv for b in rv])
    q = m.predict_proba(grid)[:, 1].reshape(len(PRICES), len(rv), len(rv))
    T = np.round(q, 4).tolist()
    # threshold and a check that the table reproduces the model
    qt = np.array([table_q(T, pp, a, b) for pp, a, b in zip(p, D.rank_f.values, D.rank_d.values)])
    qm = m.predict_proba(X[cs])[:, 1]
    gap = qt - p
    thr = float(np.quantile(gap, 0.95))
    sel = gap >= thr
    y = D.y.values
    r = np.where(y[sel] == 1, wf[sel], -1.0)
    print(f"table vs model: corr {np.corrcoef(qt, qm)[0, 1]:.4f}, mean abs diff {np.abs(qt - qm).mean():.4f}")
    print(f"threshold gap {thr:.4f}; in-sample top 5%: n={sel.sum()} won {(y[sel] == 1).mean():.1%} vs {p[sel].mean():.1%} ROI {r.mean():+.1%}")
    out = {"prices": PRICES, "ranks": RANKS, "q": T, "thr": round(thr, 4),
           "note": "Pinnacle price + rankings GBM (tools/upset_ml, tools/safe_fav_table). Paper only; use with a live Pinnacle line."}
    (ROOT / "model" / "safe_fav.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote model/safe_fav.json", (ROOT / "model" / "safe_fav.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
