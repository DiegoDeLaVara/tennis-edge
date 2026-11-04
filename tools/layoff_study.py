r"""Players who haven't played in a while: does the time off (favourite's or underdog's) move results beyond the price?
Three markets, each split into older/newer halves:
  - Kalshi ATP/WTA tour + Challenger (real ask + fee)
  - Kalshi ITF (real ask + fee)
  - Pinnacle closing odds, ATP 2010-24 / WTA 2020-24 (Pinnacle and best price)
Days since last match come from our match history (any level), walk-forward.

  .venv\Scripts\python.exe -m tools.layoff_study
"""
import contextlib
import io
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from tools.kalshi_backtest import fee

BUCKETS = [("active (played in last 14 days)", 0, 14), ("15-30 days off", 15, 30), ("31-60 days off", 31, 60),
           ("61-120 days off", 61, 120), ("over 120 days off (comeback)", 121, 10 ** 6)]


def kalshi_roi(g, side):
    price = g.ask_f if side == "f" else g.ask_d
    won = g.fw == 1 if side == "f" else g.fw == 0
    ok = price.between(0.04, 0.96)
    c = price[ok] + fee(price[ok])
    r = np.where(won[ok], (1 - c) / c, -1.0)
    return len(r), (r.mean() if len(r) else 0)


def book_roi(g, side):
    o = g.ps_f if side == "f" else g.ps_d
    won = g.fw == 1 if side == "f" else g.fw == 0
    r = np.where(won, o - 1, -1.0)
    return len(r), (r.mean() if len(r) else 0)


def table(D, title, roi_fn, who):
    print(f"\n== {title}: {len(D):,} matches")
    for side_name, col in (("FAVOURITE", "days_f"), ("UNDERDOG", "days_d")):
        print(f"   {side_name} layoff               half   n      fav won vs priced    back fav   back dog")
        for name, lo, hi in BUCKETS:
            parts = []
            for h in (0, 1):
                g = D[D[col].between(lo, hi) & (D.half == h)]
                if len(g) < 40:
                    parts.append(f"{h}: n={len(g):>5}  (few)")
                    continue
                gap = (g.fw.mean() - g.pf.mean()) * 100
                se = math.sqrt((g.pf * (1 - g.pf)).sum()) / len(g) * 100
                _, rf = roi_fn(g, "f"); _, rd = roi_fn(g, "d")
                flag = " *" if abs(gap) > 2 * se else ""
                parts.append(f"{h}: n={len(g):>5}  {gap:+5.1f} ± {se:3.1f} pts{flag:<2}  {rf:+6.1%}  {rd:+6.1%}")
            print(f"   {name:<34} " + "   |   ".join(parts))


def pinnacle():
    J = pd.read_pickle("work/rules_lab.pkl")
    J = J[np.isfinite(J.psw.astype(float)) & np.isfinite(J.psl.astype(float))].reset_index(drop=True)
    W = pd.read_pickle("work/wf_preds.pkl")[["date", "a", "b", "y", "days_a", "days_b"]].copy()
    W["d"] = pd.to_datetime(W.date).dt.date
    idx = {}
    for r in W.itertuples():
        idx.setdefault((r.a, r.b), []).append(r)
    dw, dl = [], []
    for r in J.itertuples():
        hit = None
        for (x, z, fl) in ((r.kw, r.kl, False), (r.kl, r.kw, True)):
            for c in idx.get((x, z), []):
                if timedelta(0) <= r.date - c.d <= timedelta(16):
                    hit = (c, fl)
        if hit is None:
            dw.append(np.nan); dl.append(np.nan)
            continue
        c, fl = hit
        dw.append(c.days_b if fl else c.days_a); dl.append(c.days_a if fl else c.days_b)
    J["days_w"], J["days_l"] = dw, dl
    iw, il = 1 / J.psw, 1 / J.psl
    pw = iw / (iw + il)
    fa = pw >= 0.5
    D = pd.DataFrame({"pf": np.where(fa, pw, 1 - pw), "fw": fa.astype(int), "ps_f": np.where(fa, J.psw, J.psl), "ps_d": np.where(fa, J.psl, J.psw),
                      "days_f": np.where(fa, J.days_w, J.days_l), "days_d": np.where(fa, J.days_l, J.days_w),
                      "date": pd.to_datetime(J.date)})
    D = D.dropna(subset=["days_f", "days_d"]).sort_values("date").reset_index(drop=True)
    D["half"] = (D.index >= len(D) // 2).astype(int)              # even split by date
    print(f"Pinnacle split date: {D.date.iloc[len(D) // 2].date()}")
    return D


def main():
    from tools.upset_risk import kalshi
    with contextlib.redirect_stdout(io.StringIO()):
        K = kalshi()
    fa = K.px >= 0.5
    K["ask_f"] = np.where(fa, K.ask_a, K.ask_b); K["ask_d"] = np.where(fa, K.ask_b, K.ask_a)
    table(K, "Kalshi ATP/WTA tour + Challenger", kalshi_roi, "k")
    try:
        from tools.itf_study import load_itf
        I = load_itf()
        table(I, "Kalshi ITF", kalshi_roi, "i")
    except Exception as e:  # noqa
        print("ITF skipped:", e)
    table(pinnacle(), "Pinnacle closing odds (2016-24, split evenly by date)", book_roi, "p")
    print("\n* = favourite won more (+) or less (-) than priced by over 2 standard errors in that half")


if __name__ == "__main__":
    main()
