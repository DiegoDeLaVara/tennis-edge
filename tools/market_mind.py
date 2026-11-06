r"""Think like the market: the house wins on the public's habits. For each habit, define the PUBLIC side (what casual
money likes) and the CONTRARIAN side, then measure both at Kalshi's real ask after the fee, separately for toss-ups
(favourite < 58%), leans (58-70%) and confident matches (70%+), in the older and newer half.

  .venv\Scripts\python.exe -m tools.market_mind
"""
import contextlib
import io
import math

import numpy as np
import pandas as pd

from tools.kalshi_backtest import fee

BANDS = [("toss-ups (<58%)", 0.50, 0.58), ("leans (58-70%)", 0.58, 0.70), ("confident (70%+)", 0.70, 0.97)]


def side_returns(D):
    rf, rd = [], []
    for price, won, out in ((D.ask_f.values, D.fw.values == 1, rf), (D.ask_d.values, D.fw.values == 0, rd)):
        c = price + fee(price)
        out.append(np.where((price >= 0.05) & (price <= 0.95), np.where(won, (1 - c) / c, -1.0), np.nan))
    return rf[0], rd[0]


def main():
    from tools.close_matches import data
    with contextlib.redirect_stdout(io.StringIO()):
        D = data()
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    D["h"] = (D.index >= len(D) // 2).astype(int)
    fa = D.px >= 0.5
    rf, rd = side_returns(D)
    # public side indicator: +1 = public likes the FAVOURITE, -1 = public likes the UNDERDOG, 0 = no view
    vol_hi = D.vol2 >= D.vol2.quantile(0.75)
    late = D["round"].astype(str).str.contains("QF|SF|F$|Final|Semi|Quarter", regex=True)
    H = {
        "fame: better-ranked player": np.sign(D.rank_d - D.rank_f),
        "fame: much better-ranked (3x+)": np.where(D.rank_d >= 3 * D.rank_f, 1, np.where(D.rank_f >= 3 * D.rank_d, -1, 0)),
        "attention: high-volume match -> favourite": np.where(vol_hi, 1, 0),
        "crowd: side with most of the money": np.sign(D.s_flow.fillna(0)),
        "big tickets: side the $250+ bets took": np.sign(D.s_bigmoney.fillna(0)),
        "hot streak: player on the longer win streak (3+)": np.where((D.streak_f >= 3) & (D.streak_f > D.streak_d), 1,
                                                                      np.where((D.streak_d >= 3) & (D.streak_d > D.streak_f), -1, 0)),
        "home crowd: home player": np.sign(D.home_f.fillna(0) - D.home_d.fillna(0)),
        "youth hype: younger by 4+ years": np.where(D.age_d - D.age_f >= 4, 1, np.where(D.age_f - D.age_d >= 4, -1, 0)),
        "chasing: side the price moved toward (3c+)": np.where(D.s_move >= 0.03, 1, np.where(D.s_move <= -0.03, -1, 0)),
        "big stage: QF or later -> favourite": np.where(late, 1, 0),
    }
    print(f"{len(D):,} Kalshi tour + Challenger matches ({D.date.min()} .. {D.date.max()}); split at {D.date.iloc[len(D) // 2]}")
    print("ROI of backing the PUBLIC side vs the CONTRARIAN side (real ask + fee). * = contrarian ahead and positive in BOTH halves\n")
    for name, s in H.items():
        s = np.asarray(s, float)
        print(f"-- {name}")
        for bname, lo, hi in BANDS:
            band = D.pf.between(lo, hi, inclusive="left").values
            cells, contra_ok = [], True
            for h in (0, 1):
                m = band & (D.h.values == h) & (s != 0)
                pub = np.where(s[m] > 0, rf[m], rd[m]); con = np.where(s[m] > 0, rd[m], rf[m])
                pub, con = pub[np.isfinite(pub)], con[np.isfinite(con)]
                if len(con) < 40:
                    cells.append(f"n={len(con):>4}  (few)                     "); contra_ok = False
                    continue
                se = con.std(ddof=1) / math.sqrt(len(con))
                cells.append(f"n={len(con):>4} public {pub.mean():+6.1%} | contrarian {con.mean():+6.1%} ± {se:4.1%}")
                contra_ok &= con.mean() > 0 and con.mean() > pub.mean()
            print(f"   {bname:<17} " + "    ".join(cells) + ("   *" if contra_ok else ""))


if __name__ == "__main__":
    main()
