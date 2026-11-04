r"""Can we pick winners of close matches (market favourite 50-58%)? Every signal we have is turned into a pick
(this side or that side) and scored on accuracy and on ROI at Kalshi's real ask after the fee, separately in the
older and newer half. Then all signals together in a logistic model fitted on the older half, scored on the newer.
Also: does low model confidence (sub-models disagree) weaken the Lock tier?

  .venv\Scripts\python.exe -m tools.close_matches
"""
import contextlib
import io
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tools.kalshi_backtest import load, fee
from tools.upset_risk import kalshi


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def data():
    with contextlib.redirect_stdout(io.StringIO()):
        D = kalshi()
    K = load()[["event", "px_first", "net_share", "late_net", "whale_dir", "vol"]].drop_duplicates("event").rename(columns={"vol": "vol2"})
    D = D.drop_duplicates("event")
    D = D.merge(K, on="event", how="left")
    P = pd.read_csv("work/recent_preds.csv", low_memory=False)
    idx = {}
    for r in P.itertuples():
        idx.setdefault((r.a, r.b), []).append(r)
    KK = load()[["event", "ka", "kb"]].drop_duplicates("event")
    D = D.merge(KK, on="event", how="left")
    extra = []
    for r in D.itertuples():
        hit = None
        for (x, z, flip) in ((r.ka, r.kb, False), (r.kb, r.ka, True)):
            for c in idx.get((x, z), []):
                if abs((pd.Timestamp(c.date) - pd.Timestamp(r.date)).days) <= 7:
                    hit = (c, flip)
        if hit is None:
            extra.append({})
            continue
        c, fl = hit
        dis = abs(lg(c.p_elo) - lg(c.p_elo_all)) + (abs(lg(c.p_elo) - lg(c.p_stats)) if c.stat_w >= 400 else 0)
        s = -1 if fl else 1                      # orient to Kalshi side a
        extra.append({"dis": float(dis), "form_a": c.form_b if fl else c.form_a, "form_b": c.form_a if fl else c.form_b,
                      "load_a": c.load_b if fl else c.load_a, "load_b": c.load_a if fl else c.load_b,
                      "h2h_a": s * c.h2h, "pst_a": float(lg(c.p_stats)) * s if c.stat_w >= 400 else 0.0,
                      "pelo_a": float(lg(c.p_elo)) * s, "pall_a": float(lg(c.p_elo_all)) * s})
    E = pd.DataFrame(extra)
    D = pd.concat([D.reset_index(drop=True), E], axis=1)
    fa = D.px >= 0.5
    sgn = np.where(fa, 1, -1)                     # +1 = favourite is side a
    D["ask_f"] = np.where(fa, D.ask_a, D.ask_b)
    D["ask_d"] = np.where(fa, D.ask_b, D.ask_a)
    # everything below is "for the favourite" (positive = points to the favourite)
    D["s_model"] = D.pm_f - 0.5
    D["s_move"] = (D.px - D.px_first) * sgn
    D["s_bigmoney"] = np.sign(D.big_net.fillna(0)) * sgn
    D["s_flow"] = D.net_share.fillna(0) * sgn
    D["s_late"] = np.sign(D.late_net.fillna(0)) * sgn
    D["s_steam"] = D.steam.fillna(0) * sgn
    D["s_form"] = (D.form_a - D.form_b) * sgn
    D["s_wins10"] = D.wins10_f - D.wins10_d
    D["s_rank"] = np.log(D.rank_d.clip(1, 3000) / D.rank_f.clip(1, 3000))
    D["s_younger"] = D.age_d - D.age_f
    D["s_fresh"] = D.mins3_d.fillna(0) - D.mins3_f.fillna(0)
    D["s_rest"] = D.days_f.clip(0, 60) - D.days_d.clip(0, 60)
    D["s_streak"] = D.streak_f.fillna(0) - D.streak_d.fillna(0)
    D["s_home"] = D.home_f.fillna(0) - D.home_d.fillna(0)
    D["s_h2h"] = D.h2h_a * sgn
    D["s_stats"] = D.pst_a * sgn
    D["s_surface"] = (D.pelo_a - D.pall_a) * sgn
    D["s_load"] = (D.load_b - D.load_a) * sgn
    return D


def score(g, pick_fav):
    """pick_fav: bool array (True = back the favourite). Accuracy and ROI at the ask after the fee."""
    won = np.where(pick_fav, g.fw == 1, g.fw == 0)
    price = np.where(pick_fav, g.ask_f, g.ask_d)
    ok = (price > 0.05) & (price < 0.95)
    c = price[ok] + 0.07 * price[ok] * (1 - price[ok])
    r = np.where(won[ok], (1 - c) / c, -1.0)
    return ok.sum(), won[ok].mean() if ok.sum() else 0, r.mean() if ok.sum() else 0, r.std(ddof=1) / math.sqrt(ok.sum()) if ok.sum() > 1 else 0


def main():
    D = data()
    print(f"{len(D):,} Kalshi matches")

    # ---- locks by model confidence
    lock = (D.pf >= 0.85) & ((D.pm_f - D.pf).abs() < 0.10) & (D.retired_last_f.fillna(0) == 0) & (D.m7_f.fillna(0) < 4)
    print("\n== Locks split by model confidence (sub-model disagreement)")
    for nm, m in (("high/medium confidence (< 0.6)", D.dis < 0.6), ("low confidence (>= 0.6)", D.dis >= 0.6)):
        for h in (0, 1):
            g = D[lock & m & (D.half == h)]
            n, acc, roi, se = score(g, np.ones(len(g), bool))
            print(f"   {nm:<34} half {h}: n={n:>4} won {acc:.1%} (priced {g.pf.mean():.1%}) ROI {roi:+.1%} ± {se:.1%}")

    C = D[(D.pf <= 0.58) & D.ask_f.between(0.3, 0.7)].copy()
    print(f"\n== Close matches (market favourite 50-58%): {len(C):,}. Baseline: the favourite won {C.fw.mean():.1%} (priced {C.pf.mean():.1%})")
    print(f"   {'signal (pick the side it points to)':<40} {'half':>5} {'n':>5} {'picked right':>12} {'ROI':>7} {'± se':>6}")
    sigs = [c for c in C.columns if c.startswith("s_")]
    for s in ["fav"] + sigs:
        for h in (0, 1):
            g = C[C.half == h]
            if s == "fav":
                pick = np.ones(len(g), bool)
            else:
                v = g[s].values.astype(float)
                use = np.isfinite(v) & (v != 0)
                g, pick = g[use], v[use] > 0
            if len(g) < 60:
                continue
            n, acc, roi, se = score(g, pick)
            print(f"   {s:<40} {h:>5} {n:>5} {acc:>12.1%} {roi:>+7.1%} {se:>6.1%}")

    # ---- all signals together, fitted on the older half
    X = C[sigs].astype(float).fillna(0).values
    mu, sd = X[C.half.values == 0].mean(0), X[C.half.values == 0].std(0) + 1e-9
    Xs = (X - mu) / sd
    off = lg(C.pf.values)
    Xf = np.c_[off, Xs]
    tr, te = C.half.values == 0, C.half.values == 1
    m = LogisticRegression(C=0.3, max_iter=2000).fit(Xf[tr], C.fw.values[tr])
    p = m.predict_proba(Xf[te])[:, 1]
    g = C[te]
    print("\n== All signals combined (logistic, fitted on the older half, scored on the newer)")
    ll = lambda y, q: float(-np.mean(y * np.log(np.clip(q, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-4, 1))))
    print(f"   log-loss: market {ll(g.fw.values, g.pf.values):.4f}  combined {ll(g.fw.values, p):.4f}")
    for thr in (0.0, 0.03, 0.06):
        sel = np.abs(p - g.pf.values) >= thr
        n, acc, roi, se = score(g[sel], (p >= g.pf.values)[sel] if thr else (p >= 0.5)[sel])
        print(f"   bet the side the combined model prefers, edge >= {thr:.0%}: n={n} picked right {acc:.1%} ROI {roi:+.1%} ± {se:.1%}")
    w = dict(zip(["market"] + sigs, np.round(m.coef_[0], 3)))
    print("   weights:", {k: v for k, v in w.items() if abs(v) >= 0.02})


if __name__ == "__main__":
    main()
