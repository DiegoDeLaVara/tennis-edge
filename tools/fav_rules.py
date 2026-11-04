r"""ROI at Kalshi's real ask + fee for the 'favourite under-priced' situations found by tools/upset_risk, in both
halves of the data. Also the 'stay away' situations, to see what backing either side would have done.

  .venv\Scripts\python.exe -m tools.fav_rules
"""
import math

import numpy as np

from tools.kalshi_backtest import fee
from tools.upset_risk import kalshi


def roi(g, side):
    """side 'f' backs the favourite at its ask, 'd' the underdog."""
    fa = g.px >= 0.5
    if side == "f":
        price = np.where(fa, g.ask_a, g.ask_b); won = g.fw == 1
    else:
        price = np.where(fa, g.ask_b, g.ask_a); won = g.fw == 0
    ok = (price >= 0.05) & (price <= 0.95)
    price, won = price[ok], won[ok]
    c = price + 0.07 * price * (1 - price)
    r = np.where(won, (1 - c) / c, -1.0)
    return len(r), r.mean() if len(r) else 0, r.std(ddof=1) / math.sqrt(len(r)) if len(r) > 1 else 0


def main():
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        D = kalshi()
    rules = [
        ("all favourites", "f", lambda D: D.pf > 0),
        ("veteran favourite (32+)", "f", lambda D: D.age_f >= 32),
        ("veteran favourite (32+), priced 60-90%", "f", lambda D: (D.age_f >= 32) & D.pf.between(0.6, 0.9)),
        ("young underdog (20 or under): back fav", "f", lambda D: D.age_d <= 20),
        ("heavy favourite 80-93%", "f", lambda D: D.pf.between(0.8, 0.93)),
        ("veteran fav OR young dog, fav 60-93%", "f", lambda D: ((D.age_f >= 32) | (D.age_d <= 20)) & D.pf.between(0.6, 0.93)),
        ("model disagrees 12+ (model lower): back fav", "f", lambda D: D.pm_f <= D.pf - 0.12),
        ("model disagrees 12+ (model lower): back dog", "d", lambda D: D.pm_f <= D.pf - 0.12),
        ("fav retired last match: back dog", "d", lambda D: D.retired_last_f == 1),
        ("fav 4+ matches in 7 days: back dog", "d", lambda D: D.m7_f >= 4),
    ]
    print(f"Kalshi, real ask + taker fee. half 0 = older, half 1 = newer ({len(D):,} matches)")
    for name, side, f in rules:
        m = f(D).fillna(False).astype(bool)
        out = []
        for h in (0, 1):
            n, mu, se = roi(D[m & (D.half == h)], side)
            out.append(f"half {h}: n={n:>5} ROI {mu:+6.1%} ± {se:4.1%}")
        print(f"   {name:<46} " + "   ".join(out))


if __name__ == "__main__":
    main()
