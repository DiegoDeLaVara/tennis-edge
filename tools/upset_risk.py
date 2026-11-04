r"""Which matches are upset-prone or unpredictable *relative to the price*? For each situation we compare how often the
favourite won with what the price said (upset-prone = favourite wins less than priced), and the "surprise" of the
result beyond what the price implies (excess log-loss > 0 = more chaotic than the price admits). Each signal is
checked separately on the older and newer half of the data; only ones that agree in both halves are worth a warning.

Data: Kalshi (real pre-match mid, ~7.4k matches with context) and bookmaker closing prices (Pinnacle, 27k matches).

  .venv\Scripts\python.exe -m tools.upset_risk
"""
import math

import numpy as np
import pandas as pd

from tools.kalshi_backtest import load
from tools.upset_study import join as ctx_join


def ent(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def table(D, segs, title):
    """D needs: pf (favourite's fair prob), fw (favourite won 0/1), half (0 older / 1 newer)."""
    print(f"\n== {title}  (n={len(D):,}; favourite won {D.fw.mean():.1%}, priced {D.pf.mean():.1%})")
    print(f"   {'situation':<52} {'half':>5} {'n':>6} {'priced':>7} {'won':>6} {'gap':>7} {'± se':>6} {'excess surprise':>16}")
    for name, mask in segs:
        try:
            m = mask(D).fillna(False).astype(bool)
        except Exception as e:  # noqa
            print("   ", name, "skipped:", e)
            continue
        for h in (0, 1):
            g = D[m & (D.half == h)]
            if len(g) < 60:
                print(f"   {name:<52} {h:>5} {len(g):>6}   (too few)")
                continue
            gap = g.fw.mean() - g.pf.mean()
            se = math.sqrt((g.pf * (1 - g.pf)).sum()) / len(g)
            ll = -(g.fw * np.log(g.pf) + (1 - g.fw) * np.log(1 - g.pf)).mean()
            xs = ll - ent(g.pf).mean()
            flag = " <-- upset-prone" if gap < -2 * se else (" <-- fav under-priced" if gap > 2 * se else "")
            print(f"   {name:<52} {h:>5} {len(g):>6} {g.pf.mean():>7.1%} {g.fw.mean():>6.1%} {gap:>+7.1%} {se:>6.1%} {xs:>+16.3f}{flag}")


def kalshi():
    D = ctx_join()
    K = load()[["event", "spread", "vol", "n_trades", "round", "steam", "big_net"]]
    D = D.merge(K, on="event", how="left")
    D = D.sort_values("date").reset_index(drop=True)
    D["half"] = (D.index >= len(D) // 2).astype(int)
    fa = D.px >= 0.5
    D["pf"] = np.where(fa, D.px, 1 - D.px).clip(0.5, 0.995)
    D["fw"] = np.where(fa, D.y == 1, D.y == 0).astype(int)
    pick = lambda c: np.where(fa, D[c + "_a"], D[c + "_b"])
    other = lambda c: np.where(fa, D[c + "_b"], D[c + "_a"])
    for c in ("days", "retired_last", "won_by_ret", "m7", "m30", "mins3", "streak", "age", "rank", "home", "switch", "wins10"):
        if c + "_a" in D:
            D[c + "_f"], D[c + "_d"] = pick(c), other(c)
    pm = D.p_model
    D["pm_f"] = np.where(fa, pm, 1 - pm)
    segs = [
        ("all", lambda D: D.pf > 0),
        ("thin market: spread >= 4c", lambda D: D.spread >= 0.04),
        ("thin market: volume < $300", lambda D: D.vol < 300),
        ("deep market: volume >= $5k", lambda D: D.vol >= 5000),
        ("model disagrees: model fav prob 12+ pts lower", lambda D: D.pm_f <= D.pf - 0.12),
        ("model disagrees: model fav prob 12+ pts higher", lambda D: D.pm_f >= D.pf + 0.12),
        ("favourite retired in their last match", lambda D: D.retired_last_f == 1),
        ("underdog retired in their last match", lambda D: D.retired_last_d == 1),
        ("favourite back from 45+ day layoff", lambda D: D.days_f >= 45),
        ("underdog back from 45+ day layoff", lambda D: D.days_d >= 45),
        ("favourite played 4+ matches in last 7 days", lambda D: D.m7_f >= 4),
        ("favourite on a 6+ match win streak", lambda D: D.streak_f >= 6),
        ("favourite aged 32+", lambda D: D.age_f >= 32),
        ("underdog aged 20 or younger", lambda D: D.age_d <= 20),
        ("underdog at home", lambda D: D.home_d == 1),
        ("favourite switched surface", lambda D: D.switch_f == 1),
        ("favourite ranked outside top 100", lambda D: D.rank_f > 100),
        ("ITF series", lambda D: D.series.str.contains("ITF")),
        ("Challenger series", lambda D: D.series.str.contains("CHALLENGER")),
        ("WTA tour", lambda D: D.series == "KXWTAMATCH"),
        ("ATP tour", lambda D: D.series == "KXATPMATCH"),
        ("qualifying round", lambda D: D["round"].astype(str).str.startswith("Q")),
        ("late season (Oct-Dec)", lambda D: D.month >= 10),
        ("close match: favourite priced 50-60%", lambda D: D.pf < 0.6),
        ("heavy favourite 80%+", lambda D: D.pf >= 0.8),
    ]
    table(D, segs, "Kalshi pre-match price")
    return D


def books():
    S = pd.read_pickle("work/book_sides.pkl")
    S = S.sort_values("date").reset_index(drop=True)
    S["half"] = (S.year >= 2018).astype(int)
    S["pf"] = S.p_pin.clip(0.5, 0.995)
    S["fw"] = S.y.astype(int)
    S["month"] = pd.to_datetime(S.date).dt.month
    segs = [
        ("all", lambda D: D.pf > 0),
        ("model disagrees: model fav prob 12+ pts lower", lambda D: D.p_model <= D.pf - 0.12),
        ("model disagrees: model fav prob 12+ pts higher", lambda D: D.p_model >= D.pf + 0.12),
        ("underdog hotter form (gap >= 0.15)", lambda D: D.form_d - D.form_f >= 0.15),
        ("favourite much hotter (gap >= 0.15)", lambda D: D.form_f - D.form_d >= 0.15),
        ("favourite 400+ min in 10 days", lambda D: D.load_f >= 400),
        ("underdog 400+ min in 10 days", lambda D: D.load_d >= 400),
        ("Masters 1000", lambda D: D.series.astype(str).str.contains("Masters|WTA1000", regex=True)),
        ("Grand Slam", lambda D: D.series.astype(str).str.contains("Grand Slam")),
        ("250-level events", lambda D: D.series.astype(str).str.contains("250")),
        ("1st round", lambda D: D["round"].astype(str).str.startswith("1st")),
        ("quarterfinal or later", lambda D: D["round"].astype(str).str.contains("Quarter|Semi|The Final", regex=True)),
        ("best of 5", lambda D: D.bo == 5),
        ("grass", lambda D: D.surface == "Grass"),
        ("clay", lambda D: D.surface == "Clay"),
        ("late season (Oct-Nov)", lambda D: D.month >= 10),
        ("close match: favourite priced 50-60%", lambda D: D.pf < 0.6),
        ("heavy favourite 80%+", lambda D: D.pf >= 0.8),
    ]
    table(S, segs, "Pinnacle closing price (2010-17 = half 0, 2018-24 = half 1)")


if __name__ == "__main__":
    D = kalshi()
    books()
