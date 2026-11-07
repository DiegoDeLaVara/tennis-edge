r"""Is the line "real", and will this player actually perform? Tests on 15 years of Pinnacle closing odds.

1. Body / commitment: players who recently retired mid-match or gave a walkover (injury, low commitment), players
   who retire often, players who reached this match through an opponent's retirement.
2. Line credibility: when the rest of the market (average book, Bet365) disagrees with Pinnacle's line, is Pinnacle
   the real line, or does the consensus know something? (If the consensus side wins more than Pinnacle says, the
   Pinnacle line was "not real".)

Every rule is split into leans (favourite 58-70%) and stronger favourites (70%+), checked in two time halves at
Pinnacle close, with luck runs (--null N: results drawn from the closing price) to count false passes.

  .venv\Scripts\python.exe -m tools.trap_study [--null N]
"""
import glob
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

BANDS = [("toss-up 50-58%", 0.50, 0.58), ("lean 58-70%", 0.58, 0.70), ("favourite 70%+", 0.70, 1.01)]


def num(x):
    try:
        v = float(x)
        return v if v > 1.0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def build():
    out = []
    for f in sorted(glob.glob("work/bookodds/*.csv")):
        D = pd.read_csv(f, encoding="latin-1", low_memory=False)
        D["date"] = pd.to_datetime(D["Date"], errors="coerce", format="mixed")
        D = D[D.date.notna()]
        g = lambda c: [num(x) for x in D[c]] if c in D else [np.nan] * len(D)
        out.append(pd.DataFrame({"tour": "atp" if "atp_" in f else "wta", "date": D.date.values,
                                 "w": D.Winner.astype(str).values, "l": D.Loser.astype(str).values,
                                 "comment": D.Comment.astype(str).str.strip().values,
                                 "psw": g("PSW"), "psl": g("PSL"), "avgw": g("AvgW"), "avgl": g("AvgL"),
                                 "b365w": g("B365W"), "b365l": g("B365L"), "maxw": g("MaxW"), "maxl": g("MaxL")}))
    O = pd.concat(out, ignore_index=True)
    O["i"] = np.arange(len(O))
    O = O.sort_values(["date", "i"], kind="stable").reset_index(drop=True)
    H = defaultdict(list)                      # player -> [(date, retired_here, walkover_given, won_by_ret)]
    rows = []
    for r in O.itertuples():
        ret = r.comment.lower().startswith("retired")
        wo = r.comment.lower().startswith("walkover")
        ok = np.isfinite(r.psw) and np.isfinite(r.psl)
        if ok:
            fw, fl = 1 / r.psw, 1 / r.psl
            pw = fw / (fw + fl)
            fav_w = pw >= 0.5
            F, Dg = (r.w, r.l) if fav_w else (r.l, r.w)
            pick = lambda a, b: (a, b) if fav_w else (b, a)
            row = {"date": r.date, "tour": r.tour, "y": int(fav_w), "p": max(pw, 1 - pw), "done": not (ret or wo)}
            row["ps_f"], row["ps_d"] = pick(r.psw, r.psl)
            row["max_f"], row["max_d"] = pick(r.maxw, r.maxl)
            for k, (a, b) in (("avg", (r.avgw, r.avgl)), ("b365", (r.b365w, r.b365l))):
                if np.isfinite(a) and np.isfinite(b):
                    x = (1 / a) / (1 / a + 1 / b)
                    row[f"{k}_gap"] = (x if fav_w else 1 - x) - row["p"]          # consensus fav prob minus Pinnacle
                else:
                    row[f"{k}_gap"] = np.nan
            for side, p in (("f", F), ("d", Dg)):
                h = [x for x in H[p] if (r.date - x[0]).days <= 365]
                last = H[p][-1] if H[p] else None
                recent = last is not None and (r.date - last[0]).days <= 30
                row[f"ret_last_{side}"] = recent and last[1]
                row[f"wo_last_{side}"] = recent and last[2]
                row[f"wonret_last_{side}"] = recent and last[3]
                prev10 = H[p][-11:-1]
                row[f"ret_prev10_{side}"] = any(x[1] or x[2] for x in prev10)                  # earlier, not the last one
                row[f"ret_year_{side}"] = sum(x[1] or x[2] for x in h)
                row[f"n_year_{side}"] = len(h)
            rows.append(row)
        H[r.w].append((r.date, False, False, ret or wo))
        H[r.l].append((r.date, ret, wo, False))
    D = pd.DataFrame(rows)
    D.to_pickle("work/trap_study.pkl")
    return D


def rules(D):
    def pair(a, b):
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        return np.where(a & ~b, 1, np.where(b & ~a, -1, 0))
    prone_f = (D.ret_year_f >= 2).values; prone_d = (D.ret_year_d >= 2).values
    return {
        "retired mid-match in last match (within 30 days)": pair(D.ret_last_f, D.ret_last_d),
        "gave a walkover in last match (within 30 days)": pair(D.wo_last_f, D.wo_last_d),
        "retired/withdrew in the 10 matches before the last": pair(D.ret_prev10_f, D.ret_prev10_d),
        "retires often: 2+ retirements/walkovers in a year": pair(prone_f, prone_d),
        "got through last match by opponent retiring": pair(D.wonret_last_f, D.wonret_last_d),
        "consensus rates favourite 2+ pts above Pinnacle -> favourite": np.where(D.avg_gap >= 0.02, 1, np.where(D.avg_gap <= -0.02, -1, 0)),
        "Bet365 rates favourite 3+ pts above Pinnacle -> favourite": np.where(D.b365_gap >= 0.03, 1, np.where(D.b365_gap <= -0.03, -1, 0)),
    }


def run(D, quiet=False):
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    half = (np.arange(len(D)) >= len(D) // 2).astype(int)
    rf = np.where(D.y == 1, D.ps_f - 1, -1.0); rd = np.where(D.y == 0, D.ps_d - 1, -1.0)
    passed = []
    for name, s in rules(D).items():
        if not quiet:
            print(f"-- {name}")
        for bname, lo, hi in BANDS:
            band = D.p.between(lo, hi, inclusive="left").values
            for role, sign in (("story side is FAVOURITE", 1), ("story side is UNDERDOG", -1)):
                m0 = band & (s == sign)
                cells, back_ok, fade_ok = [], True, True
                for h in (0, 1):
                    m = m0 & (half == h)
                    back = rf[m] if sign == 1 else rd[m]
                    fade = rd[m] if sign == 1 else rf[m]
                    if len(back) < 60:
                        cells.append(f"n={len(back):>5} (few)                                    "); back_ok = fade_ok = False
                        continue
                    exp = D.p.values[m] if sign == 1 else 1 - D.p.values[m]
                    won = (back > 0).mean()
                    se = back.std(ddof=1) / math.sqrt(len(back))
                    cells.append(f"n={len(back):>5} won {won:5.1%} (priced {exp.mean():5.1%}) back {back.mean():+6.1%} ± {se:4.1%} fade {fade.mean():+6.1%}")
                    back_ok &= back.mean() > 0.01; fade_ok &= fade.mean() > 0.01
                tag = "  <-- BACK held" if back_ok else ("  <-- FADE held" if fade_ok else "")
                if back_ok or fade_ok:
                    passed.append((name, bname, role, "back" if back_ok else "fade"))
                if not quiet and m0.sum() >= 120:
                    print(f"   {bname:<15} {role:<24} " + "   ".join(cells) + tag)
    return passed


def main():
    try:
        D = pd.read_pickle("work/trap_study.pkl")
    except FileNotFoundError:
        D = build()
    if "--null" in sys.argv:
        k = int(sys.argv[sys.argv.index("--null") + 1]); c = []
        for sd in range(k):
            N = D.copy(); N["y"] = (np.random.default_rng(sd).random(len(N)) < N.p.values).astype(int)
            c.append(len(run(N, quiet=True)))
        print(f"luck runs: passes per run = {c} (mean {np.mean(c):.1f})")
        return
    print(f"{len(D):,} matches with Pinnacle closing odds ({D.date.min():%Y}-{D.date.max():%Y})\n")
    p = run(D)
    print(f"\n{len(p)} passed both halves: {p}")


if __name__ == "__main__":
    main()
