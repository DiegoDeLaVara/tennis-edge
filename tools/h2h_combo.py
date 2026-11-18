r"""Head-to-head COMBINED with other factors: does the H2H record mean more when it agrees (or clashes) with recent form,
serve-vs-return matchup, who's better on this surface, and the H2H on THIS surface?

Data: Sackmann tour-level history (results, surfaces) joined to Pinnacle closing odds 2010-24 (as tools/h2h_detail),
plus serve/return and surface profiles from tools/style_matchup (work/style_matchup.pkl). Exact 50/50 prices dropped.
Only meetings and form from BEFORE the target's tournament count (tournament-start dates, so same-event rows are out).

Each scenario: back the favourite / back the underdog at Pinnacle close, two time halves; --null N luck runs; and a
logistic model (price + all these features + interactions) fitted on one half and scored on the other.

  .venv\Scripts\python.exe -m tools.h2h_combo [--null N]

Result (2026-11-18, 43,176 matches): 4 of 21 scenarios held in both halves vs 1.7 per luck run (max 5). Best: "fav leads
H2H but dog in better form (last 10, +20 pts) -> back dog" +20.7% / +14.1% (140/134). Price + all features: -0.0006 /
-0.0004 log-loss (tiny). Fresh Kalshi check (Jul-Nov 2026, H2H from archive + Kalshi meetings, form from Kalshi results):
that scenario LOST 42.5% (21 matches, favourites won 76% vs 60% priced); "dog busier + leads H2H" +3.7% +- 20.5% (34).
Not added to the board.
"""
import math
import sys
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from tools.entry_study import td_last
from tools.h2h_detail import last_tok, sack
from tools.motivation import load as load_td

KEYS = ["date", "tour", "ps_f", "ps_d"]


def build():
    S = sack()
    by_pair = defaultdict(list)
    hist = defaultdict(list)                     # player -> [(d0, won)]
    for i, r in enumerate(S.itertuples()):
        by_pair[frozenset((r.w, r.l))].append(i)
        hist[r.w].append((r.d0, 1)); hist[r.l].append((r.d0, 0))
    idx = defaultdict(list)
    for i, r in enumerate(S.itertuples()):
        if r.d0.year >= 2009:
            idx[(r.tour,) + tuple(sorted((last_tok(r.wn), last_tok(r.ln))))].append(i)
    O = load_td()
    out = []
    for r in O.itertuples():
        if not (r.psw and r.psl and math.isfinite(r.psw) and math.isfinite(r.psl)):
            continue
        fw, fl = 1 / r.psw, 1 / r.psl
        pw = fw / (fw + fl)
        if abs(pw - 0.5) < 1e-6:
            continue
        lw, ll_ = td_last(r.w), td_last(r.l)
        cand = [i for i in idx.get((r.tour,) + tuple(sorted((lw, ll_))), []) if S.d0[i] - timedelta(3) <= r.date <= S.d0[i] + timedelta(17)]
        if len(cand) != 1:
            continue
        t = S.iloc[cand[0]]
        win_id, lose_id = (t.w, t.l) if last_tok(t.wn) == lw else (t.l, t.w)
        fav_w = pw >= 0.5
        F, Dg = (win_id, lose_id) if fav_w else (lose_id, win_id)
        prior = [j for j in by_pair[frozenset((F, Dg))] if S.d0[j] < t.d0]
        o = {"date": r.date, "tour": r.tour, "y": int(fav_w), "p": max(pw, 1 - pw), "ps_f": r.psw if fav_w else r.psl,
             "ps_d": r.psl if fav_w else r.psw, "surf": t.surf, "n_meet": len(prior)}
        if prior:
            P = S.iloc[prior]
            fwon = (P.w == F).values
            o["h2h"] = fwon.mean()
            same = (P.surf == t.surf).values
            o["n_surf"] = int(same.sum())
            o["h2h_surf"] = fwon[same].mean() if same.any() else np.nan
            o["h2h_other"] = fwon[~same].mean() if (~same).any() else np.nan
        for tag, pl in (("f", F), ("d", Dg)):
            h = [x for x in hist[pl] if x[0] < t.d0][-10:]
            o[f"form10_{tag}"] = np.mean([x[1] for x in h]) if len(h) >= 5 else np.nan
            o[f"m30_{tag}"] = sum(1 for x in hist[pl] if timedelta(0) < t.d0 - x[0] <= timedelta(30))
        out.append(o)
    D = pd.DataFrame(out)
    St = pd.read_pickle("work/style_matchup.pkl").drop_duplicates(KEYS)
    cols = [c for c in ("sv_f", "rt_f", "sv_d", "rt_d", "style_edge", "surface_lift_f", "surface_lift_d", "ssv_f", "srt_f", "ssv_d", "srt_d") if c in St]
    D = D.merge(St[KEYS + cols], on=KEYS, how="left")
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    D.to_pickle("work/h2h_combo.pkl")
    return D


def rules(D):
    lead = (D.n_meet >= 2) & (D.h2h >= 0.6)
    trail = (D.n_meet >= 2) & (D.h2h <= 0.4)
    fform = (D.form10_f - D.form10_d) >= 0.2
    dform = (D.form10_d - D.form10_f) >= 0.2
    serve_edge = D.style_edge >= D.style_edge.quantile(0.75)
    serve_bad = D.style_edge <= D.style_edge.quantile(0.25)
    f_surf = D.surface_lift_f - D.surface_lift_d >= 0.02
    d_surf = D.surface_lift_d - D.surface_lift_f >= 0.02
    s_lead = (D.n_surf >= 2) & (D.h2h_surf >= 0.67)
    s_trail = (D.n_surf >= 2) & (D.h2h_surf <= 0.33)
    return {
        "fav leads H2H AND fav in better form (last 10, +20 pts)": lead & fform,
        "fav leads H2H BUT dog in better form": lead & dform,
        "fav trails H2H AND dog in better form": trail & dform,
        "fav trails H2H BUT fav in better form": trail & fform,
        "fav leads H2H AND better serve-vs-return matchup": lead & serve_edge,
        "fav leads H2H BUT worse serve-vs-return matchup": lead & serve_bad,
        "fav trails H2H AND worse serve-vs-return matchup": trail & serve_bad,
        "fav trails H2H BUT better serve-vs-return matchup": trail & serve_edge,
        "fav leads H2H AND better on this surface": lead & f_surf,
        "fav trails H2H AND dog better on this surface": trail & d_surf,
        "fav leads H2H ON THIS SURFACE (2+ meetings here)": s_lead,
        "fav trails H2H ON THIS SURFACE (2+ meetings here)": s_trail,
        "fav leads overall H2H BUT trails on this surface": lead & s_trail,
        "fav trails overall H2H BUT leads on this surface": trail & s_lead,
        "fav leads on this surface AND better on this surface": s_lead & f_surf,
        "fav trails on this surface AND dog better on this surface": s_trail & d_surf,
        "fav leads H2H here, better form AND better surface": s_lead & fform & f_surf,
        "dog leads H2H here, better form AND better surface": s_trail & dform & d_surf,
        "fav leads H2H AND form AND serve matchup all agree": lead & fform & serve_edge,
        "dog leads H2H AND form AND serve matchup all agree": trail & dform & serve_bad,
        "dog busier (more matches in 30 days) AND leads H2H": trail & (D.m30_d - D.m30_f >= 3),
    }


def run_rules(D, quiet=False):
    half = np.arange(len(D)) >= len(D) // 2
    rf = np.where(D.y == 1, D.ps_f - 1, -1.0); rd = np.where(D.y == 0, D.ps_d - 1, -1.0)
    passed = []
    for name, m in rules(D).items():
        m = np.asarray(m.fillna(False) if hasattr(m, "fillna") else m, bool)
        cells, b_ok, f_ok = [], True, True
        for h in (False, True):
            g = m & (half == h)
            if g.sum() < 60:
                cells.append(f"n={g.sum():>4} (few)"); b_ok = f_ok = False; continue
            cells.append(f"n={g.sum():>4} fav won {D.y[g].mean():5.1%} vs {D.p[g].mean():5.1%} | fav {rf[g].mean():+5.1%} dog {rd[g].mean():+5.1%}")
            b_ok &= rf[g].mean() > 0.01; f_ok &= rd[g].mean() > 0.01
        if b_ok or f_ok:
            passed.append(name)
        if not quiet:
            print(f"   {name:<58} " + "   ".join(cells) + ("  <-- back fav held" if b_ok else "  <-- back dog held" if f_ok else ""))
    return passed


def model_test(D):
    from sklearn.linear_model import LogisticRegression
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(p, 1e-4, 1 - 1e-4))))
    X = pd.DataFrame({"lp": lg(D.p), "h2h": D.h2h.fillna(0.5) - 0.5, "met": (D.n_meet > 0).astype(float), "lognm": np.log1p(D.n_meet),
                      "h2h_surf": D.h2h_surf.fillna(0.5) - 0.5, "nsurf": np.log1p(D.n_surf.fillna(0)),
                      "form_gap": (D.form10_f - D.form10_d).fillna(0), "style": D.style_edge.fillna(0),
                      "surf_gap": (D.surface_lift_f - D.surface_lift_d).fillna(0)})
    for a in ("form_gap", "style", "surf_gap", "h2h_surf"):
        X[f"h2h_x_{a}"] = X.h2h * X[a]
    y, p = D.y.values, D.p.values
    half = np.arange(len(D)) >= len(D) // 2
    print("\n== Price + H2H, H2H on this surface, form, serve matchup, surface edge and their interactions")
    for tr, te in ((False, True), (True, False)):
        mu, sd = X[half == tr].mean(), X[half == tr].std().replace(0, 1)
        Z = (X - mu) / sd
        m = LogisticRegression(C=0.5, max_iter=3000).fit(Z[half == tr], y[half == tr])
        q = m.predict_proba(Z[half == te])[:, 1]
        print(f"   fit half {int(tr)} -> test half {int(te)}: log-loss price {ll(y[half == te], p[half == te]):.4f}  price + all {ll(y[half == te], q):.4f}  "
              f"({ll(y[half == te], q) - ll(y[half == te], p[half == te]):+.4f})")


def main():
    try:
        D = pd.read_pickle("work/h2h_combo.pkl")
    except FileNotFoundError:
        D = build()
    if "--null" in sys.argv:
        k = int(sys.argv[sys.argv.index("--null") + 1]); rng = np.random.default_rng(0); c = []
        for _ in range(k):
            N = D.copy(); N["y"] = (rng.random(len(N)) < N.p.values).astype(int); c.append(len(run_rules(N, True)))
        print(f"luck runs: {c} -> mean {np.mean(c):.1f}, max {max(c)}"); return
    print(f"{len(D):,} Pinnacle-priced matches; met before {int((D.n_meet >= 1).sum()):,}; met 2+ times on this surface "
          f"{int((D.n_surf >= 2).sum()):,}; form known for both {int((D.form10_f.notna() & D.form10_d.notna()).sum()):,}")
    print("\n== H2H combined with form, serve matchup, surface (Pinnacle close, two time halves; > +1% in both to count)")
    p = run_rules(D)
    print(f"   {len(p)} held both halves: {p}")
    model_test(D)


if __name__ == "__main__":
    main()
