r"""Mine the LIVE match archive for anything that marks upsets beyond Kalshi's price.

Every graded archive snapshot (taken by the board before the match: prices, our model's parts, player context, money
flow, order book, risk flags, verdict, entry status) is turned into ~80 features from the favourite's side. Target:
did the favourite lose (upset)? Measure: upsets minus what the closing price expected.

1. Single conditions (each numeric feature at its 10/25/75/90% cut both ways, each flag) and pairs of the best 60:
   a rule needs 40+ matches in each time half and +3 pts more upsets than priced in BOTH halves.
2. Combined model: logistic regression of the upset on the price plus all features, fitted on one half and scored on
   the other (both ways), vs the price alone.
3. Luck runs: the whole scan repeated with upsets re-drawn from the price.

  .venv\Scripts\python.exe -m tools.upset_live [--luck N]

Result (2026-11-15, 1,592 live matches, 103 features): upsets 28.7% vs 28.5% priced. 318 single/pair rules gave +3 pts
more upsets in both halves, but luck runs gave 56-581 (mean 267) -> nothing beyond chance. Price + all features was
far worse than the price alone (log-loss 0.605 vs 0.558; 0.600 vs 0.549), and its "most upset-prone" 10% were not
upset more (26% vs 31%, 35% vs 34%). From 2026-11-15 the archive also logs Polymarket's price (the one upset signal
that held in backtests), so this scan can include it once enough rows exist.
"""
import itertools
import math
import sys

import numpy as np
import pandas as pd

from tools.archive_study import load


def frame():
    out = []
    for e in load():
        pc = e.get("p_close")
        if e.get("y") is None or pc is None or abs(pc - 0.5) < 1e-6:
            continue
        fa = pc >= 0.5
        i, j = (0, 1) if fa else (1, 0)
        sf, sd = ("_a", "_b") if fa else ("_b", "_a")
        sg = 1 if fa else -1
        md, cx, fl, q, rk = e.get("model") or {}, e.get("ctx") or {}, e.get("flow") or {}, e.get("q") or [{}, {}], e.get("risk") or {}
        pf = pc if fa else 1 - pc
        o = {"start": e["start"], "pf": pf, "upset": int(e["y"] != (1 if fa else 0)),
             "itf": int("ITF" in e["series"]), "chall": int("CHALLENGER" in e["series"]), "wta": int(e.get("tour") == "wta"),
             "bo5": int(e.get("bo") == 5), "clay": int(e.get("surface") == "Clay"), "grass": int(e.get("surface") == "Grass"),
             "open_to_close_fav": (pc - (e.get("p_open") or pc)) * sg,
             "hours_listed": ((e.get("last_seen") or 0) - (e.get("first_seen") or 0)) / 3600,
             "verdict_" + str(e.get("verdict")): 1, "risk_" + str(rk.get("level")): 1, "risk_net": rk.get("net"),
             "n_flags": len(rk.get("flags") or []), "n_signals": len(e.get("signals") or []), "staked": int(bool(e.get("staked")))}
        for k in ("p_model", "p_elo", "p_stats", "p_mrate", "p"):
            v = md.get(k)
            if v is not None:
                o[f"{k}_minus_price"] = (v if fa else 1 - v) - pf
        o["dis"] = md.get("dis")
        if md.get("form") and None not in md["form"]:
            o["form_fav"], o["form_dog"] = md["form"][i], md["form"][j]; o["form_gap"] = o["form_fav"] - o["form_dog"]
        if md.get("load"):
            o["load_fav"], o["load_dog"] = md["load"][i], md["load"][j]; o["load_gap"] = (md["load"][i] or 0) - (md["load"][j] or 0)
        if md.get("h2h"):
            o["h2h_fav"], o["h2h_dog"] = md["h2h"][i], md["h2h"][j]
        if md.get("pa") is not None:
            o["serve_fav"], o["serve_dog"] = (md["pa"], md["pb"]) if fa else (md["pb"], md["pa"])
        o["stat_w"] = md.get("stat_w")
        for k in ("mins3", "m7", "m30", "days", "age", "rank", "streak", "wins10", "home", "lefty", "ht", "switch", "retired_last", "won_by_ret"):
            if k + "_a" in cx:
                o[f"fav_{k}"], o[f"dog_{k}"] = cx.get(k + sf), cx.get(k + sd)
        if o.get("fav_rank") and o.get("dog_rank"):
            o["log_rank_ratio"] = math.log(o["dog_rank"] / o["fav_rank"])
        if o.get("fav_age") and o.get("dog_age"):
            o["age_gap"] = o["fav_age"] - o["dog_age"]
        for k in ("n", "vol", "big_vol", "steam"):
            o[f"flow_{k}"] = fl.get(k)
        for k in ("net_share", "big_net", "late_net"):
            if fl.get(k) is not None:
                o[f"flow_{k}_to_fav"] = fl[k] * sg
        if fl.get("px_6h") is not None and fl.get("px_last") is not None:
            o["move_6h_to_fav"] = (fl["px_last"] - fl["px_6h"]) * sg
        if fl.get("px_24h") is not None and fl.get("px_last") is not None:
            o["move_24h_to_fav"] = (fl["px_last"] - fl["px_24h"]) * sg
        qf, qd = q[i] or {}, q[j] or {}
        if qf.get("ask") is not None and qf.get("bid") is not None:
            o["spread_fav"] = qf["ask"] - qf["bid"]
        o["vol_fav"], o["vol_dog"], o["oi_fav"], o["oi_dog"] = qf.get("vol"), qd.get("vol"), qf.get("oi"), qd.get("oi")
        if o["vol_fav"] is not None and o["vol_dog"] is not None and (o["vol_fav"] + o["vol_dog"]) > 0:
            o["vol_share_dog"] = o["vol_dog"] / (o["vol_fav"] + o["vol_dog"])
        pl = e.get("poly") or {}
        if pl.get("poly_price") is not None:
            o["poly_minus_price_fav"] = (pl["poly_price"] if fa else 1 - pl["poly_price"]) - pf     # logged from 2026-11-15
        en = e.get("entry") or [None, None]
        o["entry_fav_" + str(en[i])] = 1; o["entry_dog_" + str(en[j])] = 1
        for t in e.get("tags") or []:
            o["tag_" + t] = 1
        out.append(o)
    D = pd.DataFrame(out).sort_values("start", kind="stable").reset_index(drop=True)
    for c in D.columns:
        if c.startswith(("verdict_", "risk_low", "risk_good", "risk_warn", "risk_stay", "risk_normal", "risk_None", "entry_", "tag_")):
            D[c] = D[c].fillna(0)
    return D


SKIP = {"start", "pf", "upset"}


def conditions(D):
    out = []
    for c in [c for c in D.columns if c not in SKIP]:
        v = pd.to_numeric(D[c], errors="coerce").values.astype(float)
        ok = np.isfinite(v)
        if ok.sum() < 120:
            continue
        u = np.unique(v[ok])
        if len(u) <= 2:
            for x in u:
                out.append((f"{c} = {x:g}", ok & (v == x)))
        else:
            for q in np.unique(np.nanquantile(v[ok], [0.1, 0.25, 0.75, 0.9])):
                out.append((f"{c} >= {q:.3g}", ok & (v >= q)))
                out.append((f"{c} <= {q:.3g}", ok & (v <= q)))
    return out


def scan(D, up, conds):
    half = (np.arange(len(D)) >= len(D) // 2)
    ex = 1 - D.pf.values

    def extra(m):
        r = []
        for h in (False, True):
            g = m & (half == h)
            if g.sum() < 40:
                return None
            r.append((g.sum(), up[g].mean() - ex[g].mean()))
        return r
    singles = []
    for name, m in conds:
        r = extra(m)
        if r:
            singles.append((min(x[1] for x in r), name, m, r))
    singles.sort(key=lambda t: -t[0])
    held = [(n, r) for s, n, m, r in singles if s >= 0.03]
    for (_, a, ma, _), (_, b, mb, _) in itertools.combinations(singles[:60], 2):
        if a.split(" ")[0] == b.split(" ")[0]:
            continue
        r = extra(ma & mb)
        if r and min(x[1] for x in r) >= 0.03:
            held.append((f"{a} AND {b}", r))
    return held


def model_test(D):
    from sklearn.linear_model import LogisticRegression
    X = D[[c for c in D.columns if c not in SKIP]].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().mean() > 0.3]
    X = X.fillna(X.median())
    lp = np.log(np.clip(D.pf, 1e-4, 1 - 1e-4) / (1 - np.clip(D.pf, 1e-4, 1 - 1e-4)))
    y = D.upset.values
    half = (np.arange(len(D)) >= len(D) // 2)
    ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1))))
    for tr, te in ((False, True), (True, False)):
        mu, sd = X[half == tr].mean(), X[half == tr].std().replace(0, 1)
        Z = np.c_[-lp, ((X - mu) / sd).values]
        m = LogisticRegression(C=0.05, max_iter=5000).fit(Z[half == tr], y[half == tr])
        q = m.predict_proba(Z[half == te])[:, 1]
        print(f"   fit half {int(tr)} -> test half {int(te)}: log-loss price {ll(y[half == te], 1 - D.pf.values[half == te]):.4f}  "
              f"price + {X.shape[1]} features {ll(y[half == te], q):.4f}")
        # do the matches it calls most upset-prone have more upsets than priced?
        top = q >= np.quantile(q - (1 - D.pf.values[half == te]), 0.0) if False else None
        gap = q - (1 - D.pf.values[half == te])
        sel = gap >= np.quantile(gap, 0.9)
        print(f"      its top 10% 'upset-prone' (beyond price): n={sel.sum()} upsets {y[half == te][sel].mean():.1%} vs priced {(1 - D.pf.values[half == te])[sel].mean():.1%}")


def main():
    luck = int(sys.argv[sys.argv.index("--luck") + 1]) if "--luck" in sys.argv else 10
    D = frame()
    print(f"{len(D):,} graded live matches; upsets {D.upset.mean():.1%} vs priced {(1 - D.pf).mean():.1%}; {D.shape[1] - 3} features")
    conds = conditions(D)
    held = scan(D, D.upset.values, conds)
    held.sort(key=lambda h: -min(x[1] for x in h[1]))
    print(f"\n== {len(conds)} single conditions + pairs of the best 60: {len(held)} gave +3 pts more upsets than priced in BOTH halves")
    for name, r in held[:25]:
        print(f"   {name[:100]:<100}  " + "   ".join(f"n={n:>4} {100 * d:+5.1f} pts" for n, d in r))
    rng = np.random.default_rng(3)
    c = [len(scan(D, (rng.random(len(D)) < 1 - D.pf.values).astype(int), conds)) for _ in range(luck)]
    print(f"\n== luck runs (upsets re-drawn from the price, same scan): {c} -> mean {np.mean(c):.1f}, max {max(c)}")
    print("\n== combined model")
    model_test(D)


if __name__ == "__main__":
    main()
