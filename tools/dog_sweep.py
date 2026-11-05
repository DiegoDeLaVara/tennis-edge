r"""Underdog value sweep: back the underdog (priced 15-50c) when our model rates them at least G pts above the market.
Every cell is ROI at the real Kalshi ask after the fee, shown for the older and newer half separately. A cell is
only worth acting on if both halves are positive.

  .venv\Scripts\python.exe -m tools.dog_sweep
"""
import contextlib
import io
import math

import numpy as np

from tools.kalshi_backtest import fee


def sweep(D, title):
    print(f"\n== {title} ({len(D):,} matches)")
    print(f"   {'dog price':<10} {'model edge':>10}   {'older half':>28}   {'newer half':>28}")
    for lo, hi in ((0.15, 0.30), (0.30, 0.40), (0.40, 0.50), (0.15, 0.50)):
        for g in (0.10, 0.15, 0.20, 0.25, 0.30):
            parts, both = [], True
            for h in (0, 1):
                x = D[(D.half == h) & D.ask_d.between(lo, hi) & ((D.pm_d - (1 - D.pf)) >= g)]
                c = x.ask_d + fee(x.ask_d)
                r = np.where(x.fw == 0, (1 - c) / c, -1.0)
                if len(r) < 25:
                    parts.append(f"n={len(r):>4}   (few)           "); both = False
                    continue
                parts.append(f"n={len(r):>4} ROI {r.mean():+6.1%} ± {r.std(ddof=1) / math.sqrt(len(r)):4.1%}")
                both &= r.mean() > 0
            print(f"   {lo:.0%}-{hi:.0%}    {g:>9.0%}    {parts[0]:>28}   {parts[1]:>28}{'   <-- both halves +' if both else ''}")


def main():
    from tools.itf_study import load_itf
    I = load_itf()
    I["pm_d"] = 1 - I.pm_f
    sweep(I, "Kalshi ITF")
    from tools.upset_risk import kalshi
    with contextlib.redirect_stdout(io.StringIO()):
        K = kalshi()
    fa = K.px >= 0.5
    K["ask_d"] = np.where(fa, K.ask_b, K.ask_a)
    K["pm_d"] = 1 - K.pm_f
    sweep(K, "Kalshi ATP/WTA tour + Challenger")


if __name__ == "__main__":
    main()
