r"""Are there "underdog days"? Tests whether upsets cluster by day / by tournament-day beyond chance, and whether
earlier results today (or yesterday) predict later matches against the price.

1. Clustering: for each day, surprise = sum(favourite won - price). If matches were independent given the price,
   its variance would be sum p(1-p). Ratio > 1 = clustering. Compared with shuffled days to show the noise band.
2. Prediction (Kalshi, real start times): favourite's result vs price given how many upsets already happened
   earlier the same day (whole tour, and same tournament), and given yesterday's upset surplus.
3. Rule check: back the favourite / underdog after an upset-heavy morning, at the real ask + fee.

  .venv\Scripts\python.exe -m tools.day_effects
"""
import math

import numpy as np
import pandas as pd

from tools.kalshi_backtest import load, fee

rng = np.random.default_rng(3)


def dispersion(D, key, label):
    g = D.groupby(key)
    s = g.apply(lambda x: (x.fw - x.pf).sum(), include_groups=False)
    v = g.apply(lambda x: (x.pf * (1 - x.pf)).sum(), include_groups=False)
    n = g.size()
    ok = n >= 4
    ratio = float((s[ok] ** 2).sum() / v[ok].sum())
    sims = []
    for _ in range(200):
        Dp = D.copy()
        Dp["fw"] = (rng.random(len(D)) < Dp.pf).astype(int)          # simulated independent results at the price
        gp = Dp.groupby(key)
        sp = gp.apply(lambda x: (x.fw - x.pf).sum(), include_groups=False)
        sims.append(float((sp[ok] ** 2).sum() / v[ok].sum()))
    lo, hi = np.percentile(sims, [2.5, 97.5])
    print(f"   {label:<38} groups={int(ok.sum()):>5}  dispersion {ratio:.3f}   (independent matches: {lo:.3f}-{hi:.3f})"
          f"  {'<-- clustering' if ratio > hi else ''}")


def main():
    # ---- Pinnacle: day and tournament-day clustering
    J = pd.read_pickle("work/rules_lab.pkl")
    J = J[np.isfinite(J.psw.astype(float)) & np.isfinite(J.psl.astype(float))].copy()
    iw, il = 1 / J.psw, 1 / J.psl
    pw = iw / (iw + il)
    J["pf"] = np.maximum(pw, 1 - pw)
    J["fw"] = (pw >= 0.5).astype(int)
    J["day"] = J.date.astype(str) + "|" + J.tour
    J["tday"] = J.day + "|" + J.tournament.astype(str)
    print("== 1. Do upsets cluster? (Pinnacle closing prices, ATP 2010-24 / WTA 2020-24)")
    dispersion(J, "day", "whole tour, same day")
    dispersion(J, "tday", "same tournament, same day")
    J["wk"] = pd.to_datetime(J.date).dt.strftime("%G-%V") + "|" + J.tournament.astype(str)
    dispersion(J, "wk", "same tournament, same week")

    # ---- Kalshi: real start times
    K = load()
    K = K[K.qts.notna()].copy()
    K["start"] = K.qts.astype(float)
    fa = K.px >= 0.5
    K["pf"] = np.where(fa, K.px, 1 - K.px)
    K["fw"] = np.where(fa, K.y == 1, K.y == 0).astype(int)
    K["ask_f"] = np.where(fa, K.ask_a, K.ask_b)
    K["ask_d"] = np.where(fa, K.ask_b, K.ask_a)
    K["day"] = pd.to_datetime(K.start, unit="s").dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.date.astype(str)
    K["tday"] = K.day + "|" + K.series
    print("\n== 1b. Kalshi: same day / same series-day")
    dispersion(K, "day", "same day (all series)")
    dispersion(K, "tday", "same series, same day")

    # ---- 2. earlier today predicts later?
    K = K.sort_values("start").reset_index(drop=True)
    surplus_day, surplus_same, n_day = [], [], []
    by_day = {d: g for d, g in K.groupby("day")}
    for r in K.itertuples():
        g = by_day[r.day]
        prior = g[g.start <= r.start - 3 * 3600]                       # matches that started 3+ h earlier (likely finished)
        n_day.append(len(prior))
        surplus_day.append(float((prior.pf - prior.fw).sum()))          # + = more upsets than the prices expected
        same = prior[prior.series == r.series]
        surplus_same.append(float((same.pf - same.fw).sum()))
    K["up_day"], K["up_same"], K["n_prior"] = surplus_day, surplus_same, n_day
    days = sorted(by_day)
    yest = {d: float((by_day[p].pf - by_day[p].fw).sum()) for p, d in zip(days[:-1], days[1:])}
    K["up_yday"] = K.day.map(yest)
    K["half"] = (K.index >= len(K) // 2).astype(int)
    print("\n== 2. Do earlier upsets predict the next match? (Kalshi; favourite won minus price, pts)")
    for col, label in (("up_day", "upset surplus earlier today (all)"), ("up_same", "upset surplus earlier today (same series)"),
                       ("up_yday", "upset surplus yesterday")):
        x = K[col]
        print(f"   {label}")
        for nm, m in (("upset-heavy (surplus >= +1.5)", x >= 1.5), ("normal", x.between(-1.5, 1.5, inclusive="neither")), ("favourite-heavy (<= -1.5)", x <= -1.5)):
            parts = []
            for h in (0, 1):
                g = K[m & (K.half == h)]
                if len(g) < 40:
                    parts.append(f"half {h}: n={len(g):>4}   (few)")
                    continue
                gap = (g.fw - g.pf).mean()
                se = math.sqrt((g.pf * (1 - g.pf)).sum()) / len(g)
                parts.append(f"half {h}: n={len(g):>4} {gap * 100:+5.1f} ± {se * 100:3.1f}")
            print(f"      {nm:<32} " + "   ".join(parts))
        c = np.corrcoef(x.fillna(0), K.fw - K.pf)[0, 1]
        print(f"      correlation with the next match's result vs price: {c:+.3f}")

    print("\n== 3. Betting after an upset-heavy day so far (real ask + fee)")
    for nm, m, side in (("back underdogs (surplus >= 1.5)", K.up_day >= 1.5, "d"), ("back favourites (surplus >= 1.5)", K.up_day >= 1.5, "f"),
                        ("back favourites after favourite-heavy day (<= -1.5)", K.up_day <= -1.5, "f"),
                        ("back underdogs after favourite-heavy day (<= -1.5)", K.up_day <= -1.5, "d")):
        parts = []
        for h in (0, 1):
            g = K[m & (K.half == h)]
            price = g.ask_f if side == "f" else g.ask_d
            won = g.fw == 1 if side == "f" else g.fw == 0
            ok = price.between(0.05, 0.95)
            c = price[ok] + fee(price[ok])
            r = np.where(won[ok], (1 - c) / c, -1.0)
            parts.append(f"half {h}: n={len(r):>4} ROI {r.mean():+6.1%} ± {r.std(ddof=1) / math.sqrt(len(r)):4.1%}" if len(r) > 30 else f"half {h}: few")
        print(f"   {nm:<52} " + "   ".join(parts))


if __name__ == "__main__":
    main()
