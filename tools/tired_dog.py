r"""Tired underdogs: the mirror of the tired-favourite rule. Underdog played a lot in the last 3 days (or 10 days at
Pinnacle) and the favourite is fresh: is the underdog over-punished by the price (back the dog) or under-punished
(back the favourite)? Kalshi tour/Challenger and Pinnacle (10-day minutes), both halves, real prices.

  .venv\Scripts\python.exe -m tools.tired_dog
"""
import contextlib
import io
import math

import numpy as np
import pandas as pd

from tools.kalshi_backtest import fee


def k_roi(g, side):
    price = g.ask_f if side == "f" else g.ask_d
    won = g.fw == 1 if side == "f" else g.fw == 0
    ok = price.between(0.05, 0.95)
    c = price[ok] + fee(price[ok])
    r = np.where(won[ok], (1 - c) / c, -1.0)
    return len(r), r.mean() if len(r) else 0, r.std(ddof=1) / math.sqrt(len(r)) if len(r) > 1 else 0


def main():
    from tools.upset_risk import kalshi
    with contextlib.redirect_stdout(io.StringIO()):
        K = kalshi()
    fa = K.px >= 0.5
    K["ask_f"] = np.where(fa, K.ask_a, K.ask_b); K["ask_d"] = np.where(fa, K.ask_b, K.ask_a)
    print(f"== Kalshi tour + Challenger ({len(K):,} matches; minutes in the last 3 days)")
    rules = [("tired dog 200+ min, fresh fav < 90", (K.mins3_d >= 200) & (K.mins3_f < 90)),
             ("tired dog 150+ min, fav < 60", (K.mins3_d >= 150) & (K.mins3_f < 60)),
             ("dog played 100+ more minutes than fav", (K.mins3_d - K.mins3_f) >= 100),
             ("both tired (200+ each)", (K.mins3_d >= 200) & (K.mins3_f >= 200)),
             ("reference: tired fav 200+, fresh dog < 90", (K.mins3_f >= 200) & (K.mins3_d < 90))]
    for name, m in rules:
        for side in ("d", "f"):
            parts = []
            for h in (0, 1):
                g = K[m.fillna(False) & (K.half == h)]
                n, mu, se = k_roi(g, side)
                gap = (g.fw.mean() - g.pf.mean()) * 100 if len(g) else 0
                parts.append(f"half {h}: n={n:>4} fav won {gap:+5.1f} pts vs price  back {'dog' if side == 'd' else 'fav'} {mu:+6.1%} ± {se:4.1%}")
            print(f"   {name:<42} " + "   ".join(parts))

    print("\n== Pinnacle closing odds, ATP 2010-24 / WTA 2020-24 (minutes in the last 10 days)")
    S = pd.read_pickle("work/book_sides.pkl")
    S["half"] = (S.year >= 2018).astype(int)
    for name, m in (("tired dog 400+ min, fresh fav <= 150", (S.load_d >= 400) & (S.load_f <= 150)),
                    ("dog played 250+ more minutes than fav", (S.load_d - S.load_f) >= 250),
                    ("reference: tired fav 400+, fresh dog <= 150", (S.load_f >= 400) & (S.load_d <= 150))):
        parts = []
        for h in (0, 1):
            g = S[m & (S.half == h)]
            rd = np.where(g.y == 0, g.ps_d - 1, -1.0); rf = np.where(g.y == 1, g.ps_f - 1, -1.0)
            bd = np.where(g.y == 0, g.max_d - 1, -1.0)
            parts.append(f"{'2010-17' if h == 0 else '2018-24'}: n={len(g):>4} fav won {(g.y.mean() - g.p_pin.mean()) * 100:+5.1f} pts  "
                         f"back dog {rd.mean():+6.1%} (best price {np.nanmean(bd):+6.1%})  back fav {rf.mean():+6.1%}")
        print(f"   {name:<42} " + "   ".join(parts))


if __name__ == "__main__":
    main()
