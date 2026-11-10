r"""Style MATCHUPS across the whole tour: does a kind of player have an edge over another kind (big server vs strong
returner, lefty vs righty, tall vs short, clay-raised player on grass...) that Pinnacle's price misses?

Different from tools/style_study (each player's personal record vs three style buckets): here every match feeds the
same population-level patterns, so each pattern has thousands of examples.

Profiles: each player's serve and return points won, ace and double-fault rates, decayed (half-life 1 year) from
their earlier tour-level matches in the Sackmann archive (1991-2024), overall and on the match surface, plus
height, hand and nation. Targets: matches 2010-24 (ATP) / 2020-24 (WTA) joined to Pinnacle closing odds, exact 50/50
prices dropped (they leak the result). Both players need 300+ decayed serve points.

Tests: (1) single matchup rules, back fav / back dog at Pinnacle close, two halves, luck runs; (2) price + matchup
features vs price alone (logistic, fitted on one half, scored on the other, both ways); (3) gradient boosting
walk-forward (train on earlier years only) - top 5% "safest favourites" and "likeliest upsets".

  .venv\Scripts\python.exe -m tools.style_matchup [--null N]

Result (2026-11-10, 40,625 matches): 1 of 19 rules held in both halves vs 1.4 per luck run (max 5) -> no better than
luck overall. Price + matchup features: +0.0008 / +0.0002 log-loss (worse); walk-forward GBM flips by period.
The one survivor, "both players big servers (ace rate 1+ SD above tour) -> back the underdog" (+7.5% / +17.3%,
stronger the bigger the servers, ATP and WTA, hard and grass; tiebreak-lottery logic), did NOT confirm on Kalshi
2025-26 (tools/bigserve_kalshi: 74 matches, -5.9% +- 18%; 1.5 SD: -38% on 21). Watch item only.
"""
import math
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tools.entry_study import td_last
from tools.h2h_detail import last_tok, parse_score
from tools.motivation import load as load_td

SACK = Path(r"C:\TENNIS\archive\sack")
HL = 365.0
CLAY_NATIONS = {"ESP", "ARG", "ITA", "CHI", "BRA", "URU", "PER", "COL", "ECU", "BOL", "PAR", "POR"}


class Acc:
    __slots__ = ("v", "t")

    def __init__(self):
        self.v = [0.0] * 6          # svpt, svwon, ace, df, rtpt, rtwon
        self.t = None

    def decay(self, t):
        if self.t is not None and t > self.t:
            f = 0.5 ** ((t - self.t).days / HL)
            self.v = [x * f for x in self.v]
        if self.t is None or t > self.t:
            self.t = t

    def add(self, t, sv, sw, ace, df, rt, rw):
        self.decay(t)
        for i, x in enumerate((sv, sw, ace, df, rt, rw)):
            self.v[i] += x

    def snap(self, t):
        if self.t is None:
            return None
        f = 0.5 ** (max((t - self.t).days, 0) / HL)
        sv, sw, ace, df, rt, rw = [x * f for x in self.v]
        if sv < 1 or rt < 1:
            return None
        return {"n": sv, "sv": sw / sv, "ace": ace / sv, "df": df / sv, "rt": rw / rt}


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def build():
    rows = []
    for tour in ("atp", "wta"):
        for y in range(1991, 2025):
            f = SACK / tour / f"{tour}_matches_{y}.csv"
            if f.exists():
                D = pd.read_csv(f, low_memory=False)
                D["tour"] = tour
                rows.append(D)
    S = pd.concat(rows, ignore_index=True)
    S["d0"] = pd.to_datetime(S.tourney_date.astype(int).astype(str), format="%Y%m%d").dt.date
    S = S.sort_values(["d0", "match_num"], kind="stable").reset_index(drop=True)
    A, AS = defaultdict(Acc), defaultdict(Acc)          # overall / per surface
    info = {}
    prof = []
    for r in S.itertuples():
        w, l, t, surf = f"{r.tour}:{r.winner_id}", f"{r.tour}:{r.loser_id}", r.d0, r.surface
        info[w] = (r.winner_hand, num(r.winner_ht), r.winner_ioc)
        info[l] = (r.loser_hand, num(r.loser_ht), r.loser_ioc)
        prof.append((A[w].snap(t), A[l].snap(t), AS[(w, surf)].snap(t), AS[(l, surf)].snap(t), info[w], info[l]))
        wsv, lsv = num(r.w_svpt), num(r.l_svpt)
        if wsv and lsv and parse_score(r.score) is not None:
            ww = (num(r.w_1stWon) or 0) + (num(r.w_2ndWon) or 0)
            lw = (num(r.l_1stWon) or 0) + (num(r.l_2ndWon) or 0)
            for acc in (A[w], AS[(w, surf)]):
                acc.add(t, wsv, ww, num(r.w_ace) or 0, num(r.w_df) or 0, lsv, lsv - lw)
            for acc in (A[l], AS[(l, surf)]):
                acc.add(t, lsv, lw, num(r.l_ace) or 0, num(r.l_df) or 0, wsv, wsv - ww)
    # join Pinnacle
    idx = defaultdict(list)
    for i, r in enumerate(S.itertuples()):
        if r.d0.year >= 2009:
            idx[(r.tour,) + tuple(sorted((last_tok(r.winner_name), last_tok(r.loser_name))))].append(i)
    O = load_td()
    out = []
    for r in O.itertuples():
        if not (r.psw and r.psl and math.isfinite(r.psw) and math.isfinite(r.psl)):
            continue
        fw, fl = 1 / r.psw, 1 / r.psl
        pw = fw / (fw + fl)
        if abs(pw - 0.5) < 1e-6:
            continue
        lw_, ll_ = td_last(r.w), td_last(r.l)
        cand = [i for i in idx.get((r.tour,) + tuple(sorted((lw_, ll_))), []) if S.d0[i] - timedelta(3) <= r.date <= S.d0[i] + timedelta(17)]
        if len(cand) != 1:
            continue
        i = cand[0]
        pW, pL, sW, sL, iW, iL = prof[i]
        winner_is_tdw = last_tok(S.winner_name[i]) == lw_
        # profiles of the tennis-data winner / loser
        P_w, P_l = (pW, pL) if winner_is_tdw else (pL, pW)
        S_w, S_l = (sW, sL) if winner_is_tdw else (sL, sW)
        I_w, I_l = (iW, iL) if winner_is_tdw else (iL, iW)
        if not P_w or not P_l or P_w["n"] < 300 or P_l["n"] < 300:
            continue
        fav_w = pw >= 0.5
        Pf, Pd, Sf, Sd, If, Id = (P_w, P_l, S_w, S_l, I_w, I_l) if fav_w else (P_l, P_w, S_l, S_w, I_l, I_w)
        surf = S.surface[i]
        o = {"date": r.date, "tour": r.tour, "surf": surf, "bo": num(S.best_of[i]) or 3, "y": int(fav_w), "p": max(pw, 1 - pw),
             "ps_f": r.psw if fav_w else r.psl, "ps_d": r.psl if fav_w else r.psw}
        for k in ("sv", "rt", "ace", "df"):
            o[k + "_f"], o[k + "_d"] = Pf[k], Pd[k]
            o["s" + k + "_f"] = Sf[k] if Sf and Sf["n"] >= 150 else np.nan
            o["s" + k + "_d"] = Sd[k] if Sd and Sd["n"] >= 150 else np.nan
        o["lefty_f"], o["lefty_d"] = int(If[0] == "L"), int(Id[0] == "L")
        o["ht_f"], o["ht_d"] = If[1] if If[1] else np.nan, Id[1] if Id[1] else np.nan
        o["clay_f"], o["clay_d"] = int(If[2] in CLAY_NATIONS), int(Id[2] in CLAY_NATIONS)
        out.append(o)
    D = pd.DataFrame(out).sort_values("date", kind="stable").reset_index(drop=True)
    # tour-relative styles (z-scores within tour)
    for k in ("sv", "rt", "ace"):
        for s in ("f", "d"):
            c = f"{k}_{s}"
            D[c + "_z"] = D.groupby("tour")[c].transform(lambda x: (x - x.mean()) / x.std())
    D["fav_serve_vs_dog_return"] = D.sv_f - D.rt_d          # how many serve points the fav should win
    D["dog_serve_vs_fav_return"] = D.sv_d - D.rt_f
    D["style_edge"] = (D.sv_f - D.rt_d) - (D.sv_d - D.rt_f)  # fav's serve-vs-return matchup minus the dog's
    D["surface_lift_f"] = (D.ssv_f + D.srt_f) - (D.sv_f + D.rt_f)
    D["surface_lift_d"] = (D.ssv_d + D.srt_d) - (D.sv_d + D.rt_d)
    D["ht_diff"] = D.ht_f - D.ht_d
    D.to_pickle("work/style_matchup.pkl")
    return D


def rules(D):
    big_f, big_d = D.ace_f_z >= 1, D.ace_d_z >= 1
    ret_f, ret_d = D.rt_f_z >= 1, D.rt_d_z >= 1
    fast, clay = D.surf.isin(["Grass", "Carpet"]), D.surf == "Clay"
    return {
        "big server fav vs strong-returner dog": big_f & ret_d,
        "  ...on grass": big_f & ret_d & fast,
        "  ...on clay": big_f & ret_d & clay,
        "strong-returner fav vs big-server dog": ret_f & big_d,
        "  ...on grass": ret_f & big_d & fast,
        "  ...on clay": ret_f & big_d & clay,
        "big server vs big server (tiebreak lottery)": big_f & big_d,
        "returner vs returner": ret_f & ret_d,
        "lefty fav vs righty dog": (D.lefty_f == 1) & (D.lefty_d == 0),
        "righty fav vs lefty dog": (D.lefty_f == 0) & (D.lefty_d == 1),
        "fav 10+ cm taller, on grass": (D.ht_diff >= 10) & fast,
        "fav 10+ cm shorter, on grass": (D.ht_diff <= -10) & fast,
        "fav 10+ cm taller, on clay": (D.ht_diff >= 10) & clay,
        "clay-nation fav on grass vs non-clay dog": (D.clay_f == 1) & (D.clay_d == 0) & fast,
        "clay-nation dog on clay vs non-clay fav": (D.clay_f == 0) & (D.clay_d == 1) & clay,
        "fav much better on THIS surface than overall (+3 pts)": D.surface_lift_f >= 0.03,
        "dog much better on THIS surface than overall (+3 pts)": D.surface_lift_d >= 0.03,
        "fav's serve-vs-return matchup clearly better (style edge top 10%)": D.style_edge >= D.style_edge.quantile(0.9),
        "dog's serve-vs-return matchup better despite being the dog": D.style_edge <= 0,
    }


def run_rules(D, quiet=False):
    half = (np.arange(len(D)) >= len(D) // 2)
    rf = np.where(D.y == 1, D.ps_f - 1, -1.0); rd = np.where(D.y == 0, D.ps_d - 1, -1.0)
    passed = []
    for name, m in rules(D).items():
        m = np.asarray(m.fillna(False) if hasattr(m, "fillna") else m, bool)
        cells, b_ok, f_ok = [], True, True
        for h in (False, True):
            g = m & (half == h)
            if g.sum() < 80:
                cells.append(f"n={g.sum():>5} (few)"); b_ok = f_ok = False; continue
            cells.append(f"n={g.sum():>5} fav won {D.y[g].mean():5.1%} vs {D.p[g].mean():5.1%}  back fav {rf[g].mean():+6.1%}  back dog {rd[g].mean():+6.1%}")
            b_ok &= rf[g].mean() > 0.01; f_ok &= rd[g].mean() > 0.01
        if b_ok or f_ok:
            passed.append(name)
        if not quiet:
            print(f"   {name:<66} " + "   ".join(cells) + ("  <-- back fav held" if b_ok else "  <-- back dog held" if f_ok else ""))
    return passed


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


FEATS = ["sv_f", "rt_f", "ace_f", "df_f", "sv_d", "rt_d", "ace_d", "df_d", "ssv_f", "srt_f", "ssv_d", "srt_d", "lefty_f", "lefty_d",
         "ht_f", "ht_d", "clay_f", "clay_d", "fav_serve_vs_dog_return", "dog_serve_vs_fav_return", "style_edge", "surface_lift_f",
         "surface_lift_d", "ht_diff"]


def model_tests(D):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    X = D[FEATS].copy()
    X["lp"] = lg(D.p); X["grass"] = (D.surf == "Grass").astype(int); X["clay"] = (D.surf == "Clay").astype(int); X["bo5"] = (D.bo == 5).astype(int)
    for a, b in (("fav_serve_vs_dog_return", "grass"), ("dog_serve_vs_fav_return", "grass"), ("style_edge", "grass"), ("style_edge", "clay"), ("ht_diff", "grass")):
        X[f"{a}_x_{b}"] = X[a] * X[b]
    y = D.y.values; p = D.p.values
    half = (np.arange(len(D)) >= len(D) // 2)
    Xf = X.fillna(X.median())
    print("\n== Price + matchup features (logistic) vs price alone, fitted on one half, scored on the other")
    for tr, te in ((False, True), (True, False)):
        mu, sd = Xf[half == tr].mean(), Xf[half == tr].std().replace(0, 1)
        m = LogisticRegression(C=0.3, max_iter=3000).fit(((Xf - mu) / sd)[half == tr], y[half == tr])
        q = m.predict_proba(((Xf - mu) / sd)[half == te])[:, 1]
        print(f"   fit half {int(tr)} -> test half {int(te)}: price {ll(y[half == te], p[half == te]):.4f}  price+matchups {ll(y[half == te], q):.4f}  ({ll(y[half == te], q) - ll(y[half == te], p[half == te]):+.4f})")
    print("\n== Gradient boosting, walk-forward (trained on earlier years only); returns at Pinnacle close")
    yr = pd.to_datetime(D.date).dt.year.values
    rf = np.where(y == 1, D.ps_f - 1, -1.0); rd = np.where(y == 0, D.ps_d - 1, -1.0)
    for tr_end, te in ((2014, (2015, 2016, 2017)), (2017, (2018, 2019, 2020)), (2020, (2021, 2022)), (2022, (2023, 2024))):
        trm, tsm = yr <= tr_end, np.isin(yr, te)
        g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=400, l2_regularization=5.0, min_samples_leaf=80,
                                           early_stopping=True, validation_fraction=0.2, n_iter_no_change=30, random_state=0).fit(X[trm], y[trm])
        q = g.predict_proba(X[tsm])[:, 1]; gap = q - p[tsm]
        safe = gap >= np.quantile(gap, 0.95); upset = gap <= np.quantile(gap, 0.05)
        print(f"   test {te}: n={tsm.sum()} log-loss {ll(y[tsm], q) - ll(y[tsm], p[tsm]):+.4f} vs price | safest 5% back fav {rf[tsm][safe].mean():+6.1%} "
              f"(won {y[tsm][safe].mean():.1%} vs {p[tsm][safe].mean():.1%}) | likeliest upsets 5% back dog {rd[tsm][upset].mean():+6.1%} "
              f"(dog won {1 - y[tsm][upset].mean():.1%} vs {1 - p[tsm][upset].mean():.1%})")


def main():
    try:
        D = pd.read_pickle("work/style_matchup.pkl")
    except FileNotFoundError:
        D = build()
    if "--null" in sys.argv:
        k = int(sys.argv[sys.argv.index("--null") + 1]); rng = np.random.default_rng(0); c = []
        for _ in range(k):
            N = D.copy(); N["y"] = (rng.random(len(N)) < N.p.values).astype(int); c.append(len(run_rules(N, True)))
        print(f"luck runs: {c} -> mean {np.mean(c):.1f}"); return
    print(f"{len(D):,} Pinnacle-priced matches with style profiles for both players (exact 50/50 prices dropped)")
    print("\n== Matchup rules (Pinnacle close, two time halves; > +1% in both to count)")
    p = run_rules(D)
    print(f"   {len(p)} held both halves: {p}")
    model_tests(D)


if __name__ == "__main__":
    main()
