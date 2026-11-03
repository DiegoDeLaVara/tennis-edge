r"""What predicts upsets beyond the Kalshi price? Joins work/context.csv (home, schedule, streaks, retirements, surface
switch, ranking, age, hand) to the Kalshi matches with real pre-match quotes. Rules are found on the older 60% and
checked on the newest 40%; every bet is at the real ask with Kalshi's fee.

  .venv\Scripts\python.exe -m tools.upset_study
"""
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tools.kalshi_backtest import load, fee, logit, ll


def join():
    K = load()
    C = pd.read_csv("work/context.csv", low_memory=False)
    idx = {}
    for r in C.itertuples(index=False):
        idx.setdefault((r.a, r.b), []).append(r)
    cols = [c for c in C.columns if c.endswith("_a")]
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
        d = {"event": r.event, "date": r.date, "series": r.series, "y": r.y, "px": r.px, "ask_a": r.ask_a, "ask_b": r.ask_b,
             "tourney": best.tourney, "home_country": best.home_country, "p_model": r.p_model}
        for c in cols:
            base = c[:-2]
            va, vb = getattr(best, base + "_a"), getattr(best, base + "_b")
            d[base + "_a"], d[base + "_b"] = (vb, va) if flip else (va, vb)
        rows.append(d)
    D = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    D["month"] = pd.to_datetime(D.date).dt.month
    return D


def bet(D, side_fn):
    """side_fn(row) -> 0 (back A), 1 (back B) or None. Returns per-bet returns at the ask."""
    out = []
    for r in D.itertuples():
        s = side_fn(r)
        if s is None:
            continue
        price = r.ask_a if s == 0 else r.ask_b
        if not (0.05 <= price <= 0.95):
            continue
        c = price + fee(price)
        win = (r.y == 1) if s == 0 else (r.y == 0)
        out.append((1 - c) / c if win else -1.0)
    return np.array(out)


def fmt(a):
    if len(a) == 0:
        return "n=    0"
    return f"n={len(a):5d} ROI {a.mean() * 100:+6.1f}% ± {a.std() / math.sqrt(len(a)) * 100:4.1f}"


def main():
    D = join()
    print(f"Kalshi matches with context + real quotes: {len(D)} ({D.date.min()} to {D.date.max()})")
    cut = int(len(D) * 0.6)
    disc, ver = D.iloc[:cut], D.iloc[cut:]
    fav = lambda r: 0 if r.px >= 0.5 else 1                                   # market favourite side
    dog = lambda r: 1 - fav(r)
    a_or_b = lambda r, s: (getattr(r, s + "_a"), getattr(r, s + "_b"))

    def side_val(r, s, side):
        va, vb = a_or_b(r, s)
        return va if side == 0 else vb

    print("\nBase rate: favourites at the ask", fmt(bet(D, fav)), "| underdogs at the ask", fmt(bet(D, dog)))
    rules = {
        "Underdog playing at home (fav not)": lambda r: dog(r) if side_val(r, "home", dog(r)) == 1 and side_val(r, "home", fav(r)) != 1 else None,
        "Favourite playing at home": lambda r: fav(r) if side_val(r, "home", fav(r)) == 1 and side_val(r, "home", dog(r)) != 1 else None,
        "Fade fav on a 6+ match win streak (back dog)": lambda r: dog(r) if (side_val(r, "streak", fav(r)) or 0) >= 6 else None,
        "Back dog on a 5+ match win streak": lambda r: dog(r) if (side_val(r, "streak", dog(r)) or 0) >= 5 else None,
        "Fav played 3+ matches in last 7 days, dog <=1": lambda r: dog(r) if (side_val(r, "m7", fav(r)) or 0) >= 3 and (side_val(r, "m7", dog(r)) or 0) <= 1 else None,
        "Fav played 200+ min in last 3 days, dog <90": lambda r: dog(r) if (side_val(r, "mins3", fav(r)) or 0) >= 200 and (side_val(r, "mins3", dog(r)) or 0) < 90 else None,
        "Fav retired in their last match (back dog)": lambda r: dog(r) if side_val(r, "retired_last", fav(r)) == 1 else None,
        "Fav won last match by opponent retiring": lambda r: dog(r) if side_val(r, "won_by_ret", fav(r)) == 1 else None,
        "Fav switching surface, dog not": lambda r: dog(r) if side_val(r, "switch", fav(r)) == 1 and side_val(r, "switch", dog(r)) == 0 else None,
        "Fav back after 45+ days off": lambda r: dog(r) if 45 <= (side_val(r, "days", fav(r)) or 0) < 900 else None,
        "Fav won 4+ matches in last 10 days (title run)": lambda r: dog(r) if (side_val(r, "wins10", fav(r)) or 0) >= 4 else None,
        "Lefty underdog vs righty favourite": lambda r: dog(r) if side_val(r, "lefty", dog(r)) == 1 and side_val(r, "lefty", fav(r)) == 0 else None,
        "Fav aged 31+, dog under 24": lambda r: dog(r) if (side_val(r, "age", fav(r)) or 0) >= 31 and 0 < (side_val(r, "age", dog(r)) or 99) < 24 else None,
        "Dog ranked higher than fav (market trusts form)": lambda r: dog(r) if (side_val(r, "rank", dog(r)) or 9999) < (side_val(r, "rank", fav(r)) or 9999) else None,
        "Late season (Oct-Nov) favourites": lambda r: fav(r) if r.month in (10, 11) else None,
        "Late season (Oct-Nov) underdogs": lambda r: dog(r) if r.month in (10, 11) else None,
        "Top-50 fav at a Challenger/ITF (motivation?)": lambda r: dog(r) if 0 < (side_val(r, "rank", fav(r)) or 9999) <= 50 and r.series in ("KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH", "KXITFMATCH") else None,
    }
    print(f"\n{'rule (older 60% | newest 40%)':52s} {'discover':30s} {'verify':30s}")
    for name, fn in rules.items():
        print(f"  {name:50s} {fmt(bet(disc, fn)):30s} {fmt(bet(ver, fn)):30s}")

    # all factors at once, beyond the price
    X_cols = []
    E = D.copy()
    E["lm"] = logit(E.px.values)

    def diff(s, f=lambda x: x):
        a, b = E[s + "_a"].astype(float), E[s + "_b"].astype(float)
        return f(a).fillna(0) - f(b).fillna(0)
    E["d_home"] = diff("home"); E["d_streak"] = diff("streak", lambda x: np.log1p(x.clip(0, 20)))
    E["d_m7"] = diff("m7"); E["d_mins3"] = diff("mins3", lambda x: np.log1p(x.clip(0, 600)))
    E["d_ret"] = diff("retired_last"); E["d_wbr"] = diff("won_by_ret"); E["d_switch"] = diff("switch")
    E["d_layoff"] = diff("days", lambda x: (x.clip(0, 900) >= 45).astype(float))
    E["d_wins10"] = diff("wins10"); E["d_lefty"] = diff("lefty")
    E["d_age"] = diff("age", lambda x: x.clip(15, 45)); E["d_rank"] = diff("rank", lambda x: -np.log(x.clip(1, 2000)))
    X_cols = ["d_home", "d_streak", "d_m7", "d_mins3", "d_ret", "d_wbr", "d_switch", "d_layoff", "d_wins10", "d_lefty", "d_age", "d_rank"]
    Ed, Ev = E.iloc[:cut], E.iloc[cut:]
    lr = LogisticRegression(C=0.5, max_iter=2000).fit(np.c_[Ed.lm, Ed[X_cols].values], Ed.y)
    print("\nAll factors together, effect beyond the Kalshi price (fit on older 60%; + = helps that player more than the market thinks):")
    sd = Ed[X_cols].std()
    for c, w in zip(X_cols, lr.coef_[0][1:]):
        print(f"   {c:10s} {w * sd[c]:+.3f}")
    pv = lr.predict_proba(np.c_[Ev.lm, Ev[X_cols].values])[:, 1]
    print(f"   newest 40% log-loss: market {ll(Ev.y.values, Ev.px.values):.4f}  market+context {ll(Ev.y.values, pv):.4f}")
    Ev = Ev.assign(p=pv)
    for th in (0.03, 0.06, 0.10):
        def fn(r, th=th):
            for s in (0, 1):
                ps = r.p if s == 0 else 1 - r.p
                price = r.ask_a if s == 0 else r.ask_b
                if 0.05 <= price <= 0.95 and ps / (price + fee(price)) - 1 >= th:
                    return s
            return None
        print(f"   bet market+context edge >= {int(th * 100):2d}% (newest 40%): {fmt(bet(Ev, fn))}")


if __name__ == "__main__":
    main()
