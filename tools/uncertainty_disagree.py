r"""Variant of tools/uncertainty: measure uncertainty as disagreement between our own sub-models (overall Elo,
surface Elo, serve/return model) instead of match counts, and shrink the logit when they disagree.

  .venv\Scripts\python.exe -m tools.uncertainty_disagree
"""
import itertools

import numpy as np

from tools.fit import ll
from tools.uncertainty import walk_forward, lg


def main():
    D = walk_forward()
    D["has"] = (D.stat_w >= 400).astype(float)
    D["dis"] = np.abs(lg(D.p_elo.values) - lg(D.p_elo_all.values)) + D.has * np.abs(lg(D.p_elo.values) - lg(D.p_stats.values))
    tr, te = D[D.year <= 2020], D[D.year >= 2021]
    best = None
    for a in (0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6):
        p = 1 / (1 + np.exp(-lg(tr.p_model.values) / np.sqrt(1 + a * tr.dis.values ** 2)))
        v = ll(tr.y.values, p)
        if best is None or v < best[0]:
            best = (v, a)
    a = best[1]
    pt = 1 / (1 + np.exp(-lg(te.p_model.values) / np.sqrt(1 + a * te.dis.values ** 2)))
    print(f"best a on 2016-20: {a}; 2021-26 log-loss {ll(te.y.values, te.p_model.values):.5f} -> {ll(te.y.values, pt):.5f}")
    q = np.quantile(te.dis, [0.5, 0.8, 0.95])
    for name, m in (("sub-models agree (bottom half)", te.dis <= q[0]), ("some disagreement", (te.dis > q[0]) & (te.dis <= q[1])),
                    ("big disagreement (top 20%)", (te.dis > q[1]) & (te.dis <= q[2])), ("huge (top 5%)", te.dis > q[2])):
        g = te[m]
        pf = np.maximum(g.p_model, 1 - g.p_model); fw = np.where(g.p_model >= 0.5, g.y, 1 - g.y)
        print(f"   {name:<34} n={len(g):>7}  favourite won {fw.mean():.3f} vs model said {pf.mean():.3f} ({(fw.mean() - pf.mean()) * 100:+.1f} pts)")


if __name__ == "__main__":
    main()
