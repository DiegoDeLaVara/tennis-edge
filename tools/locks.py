r"""How close to "certain" can a pick get? Filters Kalshi favourites down to the safest-looking ones (market and our
model both agree it's a big favourite, no risk flags) and shows how often they actually won and what they returned
at the real ask after the fee, in both halves of the data.

  .venv\Scripts\python.exe -m tools.locks
"""
import contextlib
import io
import math

import numpy as np

from tools.upset_risk import kalshi


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        D = kalshi()
    fa = D.px >= 0.5
    D["ask_f"] = np.where(fa, D.ask_a, D.ask_b)
    clean = (D.retired_last_f.fillna(0) == 0) & (D.retired_last_d.fillna(0) == 0) & (D.m7_f.fillna(0) < 4) & (D.days_f.fillna(0) < 45)
    agree = (D.pm_f - D.pf).abs() < 0.10
    tour = ~D.series.str.contains("ITF")
    tiers = [
        ("market 85%+", D.pf >= 0.85),
        ("market 85%+ and model 85%+", (D.pf >= 0.85) & (D.pm_f >= 0.85)),
        ("market 85%+, model agrees, no risk flags", (D.pf >= 0.85) & agree & clean),
        ("market 90%+, model 90%+, no risk flags", (D.pf >= 0.90) & (D.pm_f >= 0.90) & clean),
        ("market 90%+, model 90%+, no flags, best-of-5 or veteran fav", (D.pf >= 0.90) & (D.pm_f >= 0.90) & clean & (D.age_f >= 30)),
        ("market 80-90%, model agrees, no flags", D.pf.between(0.80, 0.90) & agree & clean),
    ]
    print(f"{len(D):,} Kalshi matches. 'won' = how often the pick won; ROI at the real ask after the fee, flat 1u.\n")
    for name, m in tiers:
        out = []
        for h in (0, 1):
            g = D[m & (D.half == h) & D.ask_f.between(0.05, 0.99)]
            if len(g) < 20:
                out.append(f"half {h}: n={len(g):>4} (few)")
                continue
            c = g.ask_f + 0.07 * g.ask_f * (1 - g.ask_f)
            r = np.where(g.fw == 1, (1 - c) / c, -1.0)
            out.append(f"half {h}: n={len(g):>4} won {g.fw.mean():5.1%} (priced {g.pf.mean():4.1%}) ROI {r.mean():+5.1%} ± {r.std(ddof=1) / math.sqrt(len(r)):3.1%}")
        print(f"   {name:<58} " + "   ".join(out))
        print(f"   {'':<58} longest losing gap: lose about 1 in {1 / max(1e-9, 1 - D[m].fw.mean()):.0f}")


if __name__ == "__main__":
    main()
