r"""Close matches at Pinnacle (favourite priced 50-58%, ATP 2010-24 / WTA 2020-24): does any signal pick the winner
better than the price, consistently in 2010-17 and 2018-24? Flat 1u at Pinnacle and at the best price.

  .venv\Scripts\python.exe -m tools.close_books
"""
import math

import numpy as np
import pandas as pd


def main():
    A = pd.read_pickle("work/book_sides.pkl")          # model, form, load (tools/book_followup)
    R = pd.read_pickle("work/rules_lab_S.pkl")         # reputation, upsets, scoreline, qualifier, rank, surface (tools/rules_lab)
    for S, sigs in ((A, {"our model": lambda S: S.p_model - S.p_pin, "form": lambda S: S.form_f - S.form_d,
                         "fresher (10-day minutes)": lambda S: S.load_d - S.load_f}),
                    (R, {"market reputation": lambda S: S.mres_f - S.mres_d, "tournament record": lambda S: S.tres_f - S.tres_d,
                         "last-match dominance": lambda S: S.share_f - S.share_d, "ranking vs Elo": lambda S: -S.rank_gap,
                         "surface specialist": lambda S: S.surf_gap, "came off upset win": lambda S: S.upsw_f - S.upsw_d})):
        p = S.p_pin if "p_pin" in S else S.p
        C = S[(p <= 0.58)].copy()
        C["pp"] = p[p <= 0.58]
        half = (C.year >= 2018).astype(int)
        base = [(C.y[half == h].mean(), C.pp[half == h].mean()) for h in (0, 1)]
        print(f"close matches: {len(C):,}; favourite won {base[0][0]:.1%}/{base[1][0]:.1%} vs priced {base[0][1]:.1%}/{base[1][1]:.1%}")
        for name, f in sigs.items():
            v = f(C).astype(float).values
            out = []
            for h in (0, 1):
                m = (half.values == h) & np.isfinite(v) & (v != 0)
                g = C[m]; pick = v[m] > 0
                won = np.where(pick, g.y == 1, g.y == 0)
                o = np.where(pick, g.ps_f, g.ps_d); ob = np.where(pick, g.max_f, g.max_d)
                r = np.where(won, o - 1, -1.0); rb = np.where(won, ob - 1, -1.0)
                rb = rb[np.isfinite(ob)]
                out.append(f"{'2010-17' if h == 0 else '2018-24'} n={len(g):>5} right {won.mean():.1%} Pinnacle {r.mean():+5.1%} ±{r.std() / math.sqrt(len(r)):.1%} best {rb.mean():+5.1%}")
            print(f"   {name:<26} " + "  |  ".join(out))


if __name__ == "__main__":
    main()
