r"""Rule scan: test thousands of "back this player when..." rules built from every variable we have, without fooling
ourselves. Matches are split by date into DISCOVER (oldest 50%), VALIDATE (next 25%) and TEST (newest 25%).
Rules are ranked only on DISCOVER; the best ones are checked on VALIDATE; only those are reported on TEST.

Rule = one condition on a variable (favourite's value, underdog's value, or the difference; above/below a
quantile), optionally combined with a price band, and a side (back the favourite or the underdog). Returns are
at Kalshi's real pre-match ask after the taker fee.

  .venv\Scripts\python.exe -m tools.rule_scan            # tour + Challenger
  .venv\Scripts\python.exe -m tools.rule_scan --itf      # ITF
"""
import contextlib
import io
import json
import math
import sys

import numpy as np
import pandas as pd

from tools.kalshi_backtest import fee


def returns(D):
    out = {}
    for side, price, won in (("fav", D.ask_f.values, D.fw.values == 1), ("dog", D.ask_d.values, D.fw.values == 0)):
        ok = (price >= 0.05) & (price <= 0.95)
        c = price + fee(price)
        r = np.where(won, (1 - c) / c, -1.0)
        out[side] = np.where(ok, r, np.nan)
    return out


def tour_data():
    from tools.close_matches import data
    with contextlib.redirect_stdout(io.StringIO()):
        D = data()
    fa = D.px >= 0.5
    D["form_f"] = np.where(fa, D.form_a, D.form_b); D["form_d"] = np.where(fa, D.form_b, D.form_a)
    D["load_f"] = np.where(fa, D.load_a, D.load_b); D["load_d"] = np.where(fa, D.load_b, D.load_a)
    D["edge_f"] = D.pm_f - D.pf
    V = {  # name -> (values, description)
        "model edge on fav (model - price)": D.edge_f, "price of fav": D.pf, "sub-model disagreement": D.dis,
        "market volume $": D.vol2, "spread": D.spread,
        "line move toward fav since open": D.s_move, "big-money net toward fav": D.big_net.fillna(0) * np.where(fa, 1, -1),
        "flow share toward fav": D.s_flow, "steam toward fav": D.s_steam,
        "fav mins last 3d": D.mins3_f, "dog mins last 3d": D.mins3_d, "dog-fav mins last 3d": D.mins3_d - D.mins3_f,
        "fav mins last 10d": D.load_f, "dog mins last 10d": D.load_d,
        "fav matches last 7d": D.m7_f, "dog matches last 7d": D.m7_d, "fav matches last 30d": D.m30_f, "dog matches last 30d": D.m30_d,
        "fav days since last": D.days_f, "dog days since last": D.days_d,
        "fav form": D.form_f, "dog form": D.form_d, "fav-dog form": D.form_f - D.form_d,
        "fav wins last 10": D.wins10_f, "dog wins last 10": D.wins10_d,
        "fav win streak": D.streak_f, "dog win streak": D.streak_d,
        "fav age": D.age_f, "dog age": D.age_d, "fav rank": D.rank_f, "dog rank": D.rank_d,
        "log rank ratio dog/fav": np.log(D.rank_d.clip(1, 3000) / D.rank_f.clip(1, 3000)),
        "h2h toward fav": D.s_h2h, "serve/return model toward fav": D.s_stats, "surface elo edge toward fav": D.s_surface,
        "fav at home": D.home_f, "dog at home": D.home_d, "fav switched surface": D.switch_f, "dog switched surface": D.switch_d,
    }
    return D, V


def itf_data():
    from tools.itf_study import load_itf
    D = load_itf()
    D["edge_f"] = D.pm_f - D.pf
    V = {"model edge on fav (model - price)": D.edge_f, "price of fav": D.pf, "fav rated matches": D.n_f, "dog rated matches": D.n_d,
         "fav days since last": D.days_f, "dog days since last": D.days_d, "fav age": D.age_f, "dog age": D.age_d,
         "fav rank": D.rank_f, "dog rank": D.rank_d, "log rank ratio dog/fav": np.log(D.rank_d.clip(1, 3000) / D.rank_f.clip(1, 3000))}
    return D, V


def stat(r, m):
    x = r[m]
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return len(x), 0.0, 1.0
    return len(x), float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def main():
    itf = "--itf" in sys.argv
    D, V = itf_data() if itf else tour_data()
    D = D.sort_values("date").reset_index(drop=True)
    n = len(D)
    split = np.zeros(n, int); split[n // 2:] = 1; split[(3 * n) // 4:] = 2
    disc, val, test = split == 0, split == 1, split == 2
    print(f"{'ITF' if itf else 'Tour + Challenger'}: {n:,} matches | discover {D.date[disc].min()}..{D.date[disc].max()} | "
          f"validate ..{D.date[val].max()} | test ..{D.date[test].max()}")
    if "--null" in sys.argv:                       # luck check: outcomes drawn from the market's own price (no edge exists)
        rng = np.random.default_rng(int(sys.argv[sys.argv.index("--null") + 1]))
        D["fw"] = (rng.random(n) < D.pf.values).astype(int)
    R = returns(D)
    bands = {"any price": np.ones(n, bool), "fav 50-62%": D.pf.between(0.5, 0.62).values, "fav 62-78%": D.pf.between(0.62, 0.78).values,
             "fav 78-95%": D.pf.between(0.78, 0.95).values}
    rules = []
    for name, v in V.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        if ok[disc].sum() < 200:
            continue
        uniq = np.unique(v[disc & ok])
        if len(uniq) <= 3:                          # binary-ish variable
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
                    if k >= 80:
                        rules.append((cname, bname, side, m, k, mu, se))
    print(f"rules tested on DISCOVER: {len(rules):,}")
    # rank on discover by t-stat, require a real ROI
    good = [r for r in rules if r[5] >= 0.03 and r[5] / max(r[6], 1e-9) >= 1.0]
    good.sort(key=lambda r: -r[5] / r[6])
    good = good[:80]
    print(f"top DISCOVER rules kept (ROI >= +3%, t >= 1, best 80): {len(good)}")
    survivors = []
    for cname, bname, side, m, k, mu, se in good:
        kv, mv, sv = stat(R[side], m & val)
        if kv >= 30 and mv > 0:
            kt, mt, st = stat(R[side], m & test)
            survivors.append({"rule": f"back {side.upper()} when {cname}" + ("" if bname == "any price" else f" and {bname}"),
                              "discover": (k, mu), "validate": (kv, mv), "test": (kt, mt, st)})
    print(f"also positive on VALIDATE (n >= 30): {len(survivors)}\n")
    print(f"   {'rule':<92} {'discover':>16} {'validate':>16} {'TEST (unseen)':>24}")
    for s in survivors[:60]:
        d, v, t = s["discover"], s["validate"], s["test"]
        flag = "  <-- holds in all three" if t[0] >= 30 and t[1] > 0 else ""
        print(f"   {s['rule'][:92]:<92} {d[0]:>5} {d[1]:+7.1%}  {v[0]:>5} {v[1]:+7.1%}  {t[0]:>5} {t[1]:+7.1%} ± {t[2]:4.1%}{flag}")
    held = [s for s in survivors if s["test"][0] >= 30 and s["test"][1] > 0]
    print(f"\nheld up in all three periods: {len(held)} of {len(survivors)} validated rules")
    if "--null" in sys.argv:
        return
    json.dump([{k: v for k, v in s.items()} for s in held], open(f"work/rule_scan_{'itf' if itf else 'tour'}.json", "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
