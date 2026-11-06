r"""Think like the market, round 2: habits specific to Kalshi's public (US residents on a US app).
Hypotheses (public side -> contrarian side):
  American player (US bettors back their own) | US prime time (7-11 pm ET, casual money) -> favourite |
  weekend -> favourite | fallen star (peak rating far above today's) | giant-killer hype (just beat a much
  stronger player) | left-hander (often underrated, so here the "public" side is the right-hander)
Same layout as tools/market_mind: ROI of public vs contrarian side at the real ask + fee, by confidence band and half.

  .venv\Scripts\python.exe -m tools.market_mind2
"""
import bisect
import contextlib
import gzip
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from tools.market_mind import BANDS, side_returns


def history():
    """Per player: sorted dates with pre-match Elo, and the result / opponent-strength gap of each match."""
    F = pd.read_csv("work/features.csv", low_memory=False, usecols=["date", "a", "b", "y", "elo_a", "elo_b"])
    H = defaultdict(list)
    for d, a, b, y, ea, eb in zip(F.date.values, F.a.values, F.b.values, F.y.values, F.elo_a.values, F.elo_b.values):
        H[a].append((d, ea, y == 1, eb - ea, b))
        H[b].append((d, eb, y == 0, ea - eb, a))
    out = {}
    for k, L in H.items():
        L.sort(key=lambda x: x[0])
        dates = [x[0] for x in L]
        peak, run = [], -1e9
        for x in L:
            run = max(run, x[1]); peak.append(run)
        out[k] = (dates, L, peak)
    return out


def before(Hk, date, opp):
    """Last match strictly before this one. Some sources date matches by the tournament's first day, so skip any row
    against today's opponent in the last 21 days (that row can be this very match, result included)."""
    dates, L, peak = Hk
    i = bisect.bisect_right(dates, date) - 1
    while i >= 0 and L[i][4] == opp and (pd.Timestamp(date) - pd.Timestamp(L[i][0])).days <= 21:
        i -= 1
    while i >= 0 and L[i][0] >= date:
        i -= 1
    return (L[i], peak[i]) if i >= 0 else (None, None)


def main():
    from tools.close_matches import data
    from tools.kalshi_backtest import load
    with contextlib.redirect_stdout(io.StringIO()):
        D = data()
    K = load()[["event", "qts"]].drop_duplicates("event")
    D = D.drop(columns=[c for c in ("qts",) if c in D]).merge(K, on="event", how="left")
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    D["h"] = (D.index >= len(D) // 2).astype(int)
    fa = (D.px >= 0.5).values
    rf, rd = side_returns(D)
    st = json.load(gzip.open("state/ratings.json.gz", "rt"))
    ioc = {k: v[-5] for k, v in st["players"].items()}
    Hh = history()
    io_f, io_d, fall_f, fall_d, gk_f, gk_d = [], [], [], [], [], []
    for r in D.itertuples():
        kf, kd = (r.ka, r.kb) if r.px >= 0.5 else (r.kb, r.ka)
        io_f.append(ioc.get(kf)); io_d.append(ioc.get(kd))
        for k, opp, fall, gk in ((kf, kd, fall_f, gk_f), (kd, kf, fall_d, gk_d)):
            last, pk = before(Hh[k], r.date, opp) if k in Hh else (None, None)
            if last is None:
                fall.append(np.nan); gk.append(0); continue
            fall.append(pk - last[1])                                         # how far below their own peak
            recent = (pd.Timestamp(r.date) - pd.Timestamp(last[0])).days <= 14
            gk.append(1 if (recent and last[2] and last[3] >= 150) else 0)    # last match: beat someone 150+ Elo stronger
    usa_f = np.array([x == "USA" for x in io_f]); usa_d = np.array([x == "USA" for x in io_d])
    fall_f, fall_d = np.array(fall_f, float), np.array(fall_d, float)
    gk_f, gk_d = np.array(gk_f), np.array(gk_d)
    et = pd.to_datetime(D.qts, unit="s", utc=True).dt.tz_convert("US/Eastern") + pd.Timedelta(hours=1)   # quote is ~1h before start
    hour, wd = et.dt.hour.values, et.dt.day_of_week.values
    lefty_f = np.where(fa, D.lefty_a, D.lefty_b); lefty_d = np.where(fa, D.lefty_b, D.lefty_a)
    Hy = {
        "American player (vs non-American)": np.where(usa_f & ~usa_d, 1, np.where(usa_d & ~usa_f, -1, 0)),
        "US prime time (7-11 pm ET) -> favourite": np.where((hour >= 19) & (hour <= 23), 1, 0),
        "weekend -> favourite": np.where(wd >= 5, 1, 0),
        "fallen star (300+ Elo below own peak)": np.where((fall_f >= 300) & ~(fall_d >= 300), 1, np.where((fall_d >= 300) & ~(fall_f >= 300), -1, 0)),
        "giant-killer hype (last match beat a 150+ stronger player)": np.where((gk_f == 1) & (gk_d == 0), 1, np.where((gk_d == 1) & (gk_f == 0), -1, 0)),
        "right-hander vs left-hander (public underrates lefties)": np.where((lefty_f == 0) & (lefty_d == 1), 1, np.where((lefty_d == 0) & (lefty_f == 1), -1, 0)),
    }
    print(f"{len(D):,} Kalshi tour + Challenger matches; American players found in {int((usa_f | usa_d).sum()):,} matches")
    print("ROI of the PUBLIC side vs the CONTRARIAN side (real ask + fee). * = contrarian positive and ahead in BOTH halves\n")
    for name, s in Hy.items():
        s = np.asarray(s, float)
        print(f"-- {name}")
        for bname, lo, hi in BANDS:
            band = D.pf.between(lo, hi, inclusive="left").values
            cells, ok = [], True
            for h in (0, 1):
                m = band & (D.h.values == h) & (s != 0)
                pub = np.where(s[m] > 0, rf[m], rd[m]); con = np.where(s[m] > 0, rd[m], rf[m])
                pub, con = pub[np.isfinite(pub)], con[np.isfinite(con)]
                if len(con) < 40:
                    cells.append(f"n={len(con):>4}  (few)                                "); ok = False
                    continue
                se = con.std(ddof=1) / math.sqrt(len(con))
                cells.append(f"n={len(con):>4} public {pub.mean():+6.1%} | contrarian {con.mean():+6.1%} ± {se:4.1%}")
                ok &= con.mean() > 0 and con.mean() > pub.mean()
            print(f"   {bname:<17} " + "    ".join(cells) + ("   *" if ok else ""))


if __name__ == "__main__":
    main()
