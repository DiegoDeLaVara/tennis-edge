r"""Kalshi check of two rules found on 15 years of bookmaker prices (tools/book_followup):
  - fade the hot underdog: back the favourite when the underdog's form (results beyond rating, 45-day) beats the
    favourite's by a gap
  - tired favourite, 10-day minutes version
Real pre-match ask + taker fee, newest 40% shown separately from the oldest 60%.

  .venv\Scripts\python.exe -m tools.kalshi_form_fade
"""
import math

import numpy as np
import pandas as pd

from tools.kalshi_backtest import load, fee


def main():
    K = load()
    P = pd.read_csv("work/recent_preds.csv", low_memory=False)
    idx = {}
    for r in P.itertuples():
        idx.setdefault((r.a, r.b), []).append(r)
    rows = []
    for r in K.itertuples():
        best, flip, gap = None, False, 99
        for (a, b, fl) in ((r.ka, r.kb, False), (r.kb, r.ka, True)):
            for c in idx.get((a, b), []):
                g = abs((pd.Timestamp(c.date) - pd.Timestamp(r.date)).days)
                if g <= 7 and g < gap:
                    best, flip, gap = c, fl, g
        if best is None:
            continue
        fa, fb = (best.form_b, best.form_a) if flip else (best.form_a, best.form_b)
        la, lb = (best.load_b, best.load_a) if flip else (best.load_a, best.load_b)
        fav_a = r.px >= 0.5
        rows.append({"date": r.date, "series": r.series, "fav_won": (r.y == 1) == fav_a,
                     "ask_f": r.ask_a if fav_a else r.ask_b, "form_f": fa if fav_a else fb, "form_d": fb if fav_a else fa,
                     "load_f": la if fav_a else lb, "load_d": lb if fav_a else la})
    D = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    cut = int(len(D) * 0.6)
    print(f"Kalshi matches joined to form/load: {len(D):,}")

    def run(name, mask):
        for nm, H in (("oldest 60%", D.iloc[:cut]), ("newest 40%", D.iloc[cut:])):
            h = H[mask(H) & H.ask_f.between(0.05, 0.95)]
            c = h.ask_f + h.ask_f.map(fee)
            ret = np.where(h.fav_won, (1 - c) / c, -1.0)
            se = ret.std(ddof=1) / math.sqrt(len(ret)) if len(ret) > 1 else 0
            print(f"   {name:<46} {nm}  n={len(ret):>5}  ROI {ret.mean() if len(ret) else 0:+6.1%} ± {se:4.1%}")

    run("all favourites", lambda H: H.ask_f > 0)
    for g in (0.05, 0.10, 0.15):
        run(f"fade hot dog: dog form - fav form >= {g:.2f}", lambda H, g=g: (H.form_d - H.form_f) >= g)
    run("tired fav: 10-day mins fav >= 400, dog <= 150", lambda H: (H.load_f >= 400) & (H.load_d <= 150))


if __name__ == "__main__":
    main()
