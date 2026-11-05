r"""Rule scan on 15 years of Pinnacle closing odds (ATP 2010-24, WTA 2020-24): same discover (oldest 50%) /
validate (next 25%) / test (newest 25%) design and luck check as tools/rule_scan, with far more matches.
Returns at Pinnacle's closing odds (and the best book price for reference).

  .venv\Scripts\python.exe -m tools.rule_scan_pin [--null SEED]
"""
import math
import sys

import numpy as np
import pandas as pd


def stat(r, m):
    x = r[m]; x = x[np.isfinite(x)]
    return (len(x), float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 1 else (len(x), 0.0, 1.0)


def scan(D, V, label):
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    n = len(D)
    split = np.zeros(n, int); split[n // 2:] = 1; split[(3 * n) // 4:] = 2
    disc, val, test = split == 0, split == 1, split == 2
    if "--null" in sys.argv:
        rng = np.random.default_rng(int(sys.argv[sys.argv.index("--null") + 1]))
        D["y"] = (rng.random(n) < D.p.values).astype(int)
    R = {"fav": np.where(D.y == 1, D.ps_f - 1, -1.0), "dog": np.where(D.y == 0, D.ps_d - 1, -1.0)}
    B = {"fav": np.where(D.y == 1, D.max_f - 1, -1.0), "dog": np.where(D.y == 0, D.max_d - 1, -1.0)}
    bands = {"any price": np.ones(n, bool), "fav 50-62%": D.p.between(0.5, 0.62).values, "fav 62-78%": D.p.between(0.62, 0.78).values,
             "fav 78-95%": D.p.between(0.78, 0.95).values}
    if "--lean" in sys.argv:                       # leans only: favourite priced 58-70%
        bands = {"lean 58-70%": D.p.between(0.58, 0.70).values}
    rules = []
    for name, f in V.items():
        v = np.asarray(f(D), float); ok = np.isfinite(v)
        if ok[disc].sum() < 300:
            continue
        uniq = np.unique(v[disc & ok])
        if len(uniq) <= 3:
            conds = [(f"{name} = {u:g}", ok & (v == u)) for u in uniq]
        else:
            qs = np.unique(np.nanquantile(v[disc & ok], [0.1, 0.25, 0.75, 0.9]))
            conds = [(f"{name} >= {q:.3g}", ok & (v >= q)) for q in qs if q > np.nanmin(v[disc & ok])] + \
                    [(f"{name} <= {q:.3g}", ok & (v <= q)) for q in qs if q < np.nanmax(v[disc & ok])]
        for cname, cm in conds:
            for bname, bm in bands.items():
                m = cm & bm
                for side in ("fav", "dog"):
                    k, mu, se = stat(R[side], m & disc)
                    if k >= 150:
                        rules.append((cname, bname, side, m, k, mu, se))
    good = sorted([r for r in rules if r[5] >= 0.02 and r[5] / max(r[6], 1e-9) >= 1.0], key=lambda r: -r[5] / r[6])[:80]
    surv = []
    for cname, bname, side, m, k, mu, se in good:
        kv, mv, _ = stat(R[side], m & val)
        if kv >= 60 and mv > 0:
            kt, mt, st = stat(R[side], m & test)
            kb, mb, _ = stat(B[side], m & test)
            surv.append((f"back {side.upper()} when {cname}" + ("" if bname == "any price" else f" and {bname}"), (k, mu), (kv, mv), (kt, mt, st), mb))
    held = [s for s in surv if s[3][0] >= 60 and s[3][1] > 0]
    print(f"{label}: {n:,} matches, {len(rules):,} rules; kept {len(good)}; validated {len(surv)}; held in all three: {len(held)}")
    if "--null" not in sys.argv:
        print(f"   {'rule':<86} {'discover':>14} {'validate':>14} {'TEST at Pinnacle':>24} {'TEST best price':>16}")
        for r, d, v, t, mb in surv:
            print(f"   {r[:86]:<86} {d[0]:>5} {d[1]:+6.1%}  {v[0]:>5} {v[1]:+6.1%}  {t[0]:>5} {t[1]:+6.1%} ± {t[2]:4.1%}   {mb:+6.1%}"
                  f"{'  <-- all three' if t[0] >= 60 and t[1] > 0 else ''}")


def main():
    A = pd.read_pickle("work/book_sides.pkl").rename(columns={"p_pin": "p"})
    VA = {"model edge on fav": lambda D: D.p_model - D.p, "fav form": lambda D: D.form_f, "dog form": lambda D: D.form_d,
          "fav-dog form": lambda D: D.form_f - D.form_d, "fav mins 10d": lambda D: D.load_f, "dog mins 10d": lambda D: D.load_d,
          "dog-fav mins 10d": lambda D: D.load_d - D.load_f, "best-of-5": lambda D: (D.bo == 5).astype(float),
          "WTA": lambda D: (D.tour == "wta").astype(float), "fav best price over Pinnacle": lambda D: D.max_f / D.ps_f - 1,
          "dog best price over Pinnacle": lambda D: D.max_d / D.ps_d - 1}
    scan(A, VA, "Pinnacle set A (model, form, minutes, format)")
    S = pd.read_pickle("work/rules_lab_S.pkl")
    VS = {"fav market reputation": lambda D: D.mres_f, "dog market reputation": lambda D: D.mres_d,
          "fav tournament record": lambda D: D.tres_f, "dog tournament record": lambda D: D.tres_d,
          "fav last-match share": lambda D: D.share_f, "dog last-match share": lambda D: D.share_d,
          "fav upset win last": lambda D: D.upsw_f, "dog upset win last": lambda D: D.upsw_d,
          "fav upset loss last": lambda D: D.upsl_f, "dog upset loss last": lambda D: D.upsl_d,
          "fav qualifier": lambda D: D.qual_f, "dog qualifier": lambda D: D.qual_d,
          "ranking vs Elo": lambda D: D.rank_gap if "rank_gap" in D else np.nan, "surface specialist": lambda D: D.surf_gap,
          "fav tier record": lambda D: D.tier_f, "dog tier record": lambda D: D.tier_d, "pre-Slam week": lambda D: D.preslam,
          "fav post-Slam": lambda D: D.postslam_f}
    scan(S, VS, "Pinnacle set B (history, upsets, scheduling)")


if __name__ == "__main__":
    main()
