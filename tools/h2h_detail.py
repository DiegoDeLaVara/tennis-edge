r"""Head-to-head DETAIL: not just who won before, but how - straight sets or a split, share of games, breaks of serve
each way, tiebreaks, how recent, same surface. Does any of it predict the next meeting better than Pinnacle's price?

History: Sackmann archive tour-level matches 1991-2024 (scores + serve stats), keyed by player IDs. Targets: those
matches 2010-24 (ATP) / 2020-24 (WTA) that join to a tennis-data Pinnacle closing price (same join as tools.entry_study),
with exact 50/50 prices dropped. Only meetings BEFORE the target's tournament count.

Tests, all from the favourite's side: (1) a logistic model price + H2H-detail features, fitted on one half and scored
on the other (both ways) vs the price alone; (2) single rules, back / fade, two halves, with luck runs.

  .venv\Scripts\python.exe -m tools.h2h_detail [--null N]

Result (2026-11-09): 43,176 Pinnacle-priced matches, 20,729 with an earlier meeting (7,509 with 3+), break stats on
20,500. 0 of 15 rules held in both halves (dominant last win, straight-sets loss, split, lucky/unlucky last result,
break counts, games vs wins, tiebreak record, recency, same surface). Price + all H2H detail vs price alone, fitted on
one half and scored on the other: +0.0000 / -0.0004 log-loss. How they played each other is already in the price.
"""
import math
import re
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from edge.names import fold
from tools.motivation import load as load_td
from tools.entry_study import td_last

SACK = Path(r"C:\TENNIS\archive\sack")
SET = re.compile(r"(\d+)-(\d+)(?:\((\d+)\))?")


def parse_score(s):
    """[(games_w, games_l, tiebreak?)] or None for retirements / walkovers / junk."""
    s = str(s)
    if any(x in s.upper() for x in ("RET", "W/O", "DEF", "ABD", "WALKOVER", "UNK")):
        return None
    sets = []
    for a, b, tb in SET.findall(s):
        a, b = int(a), int(b)
        if a > 7 or b > 7:                    # match tiebreak / long final set: count as a set, games capped
            a, b = (1, 0) if a > b else (0, 1)
        sets.append((a, b, tb != "" or (max(a, b) == 7 and min(a, b) == 6)))
    return sets or None


def last_tok(name):
    t = fold(str(name))
    return t[-1] if t else ""


def sack():
    rows = []
    for tour, years in (("atp", range(1991, 2025)), ("wta", range(1991, 2025))):
        for y in years:
            f = SACK / tour / f"{tour}_matches_{y}.csv"
            if not f.exists():
                continue
            D = pd.read_csv(f, low_memory=False)
            for r in D.itertuples():
                sc = parse_score(r.score)
                if sc is None:
                    continue
                def num(x):
                    try:
                        v = float(x); return v if math.isfinite(v) else None
                    except (TypeError, ValueError):
                        return None
                bw = None if num(r.l_bpFaced) is None or num(r.l_bpSaved) is None else num(r.l_bpFaced) - num(r.l_bpSaved)   # breaks BY the winner
                bl = None if num(r.w_bpFaced) is None or num(r.w_bpSaved) is None else num(r.w_bpFaced) - num(r.w_bpSaved)
                rows.append({"tour": tour, "d0": pd.to_datetime(str(int(r.tourney_date)), format="%Y%m%d").date(), "surf": r.surface,
                             "w": f"{tour}:{r.winner_id}", "l": f"{tour}:{r.loser_id}", "wn": r.winner_name, "ln": r.loser_name,
                             "sets_w": sum(a > b for a, b, _ in sc), "sets_l": sum(b > a for a, b, _ in sc),
                             "gw": sum(a for a, _, _ in sc), "gl": sum(b for _, b, _ in sc), "tb": sum(t for _, _, t in sc),
                             "tbw": sum(1 for a, b, t in sc if t and a > b), "bw": bw, "bl": bl, "bo": int(r.best_of) if num(r.best_of) else 3})
    S = pd.DataFrame(rows).sort_values("d0", kind="stable").reset_index(drop=True)
    return S


def build():
    S = sack()
    by_pair = defaultdict(list)
    for i, r in enumerate(S.itertuples()):
        by_pair[frozenset((r.w, r.l))].append(i)
    # join Pinnacle targets to Sackmann rows (tour + both surnames + tournament week)
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
        f = {"date": r.date, "tour": r.tour, "y": int(fav_w), "p": max(pw, 1 - pw), "ps_f": r.psw if fav_w else r.psl,
             "ps_d": r.psl if fav_w else r.psw, "n_meet": len(prior), "surf": t.surf}
        if prior:
            P = S.iloc[prior]
            fwon = (P.w == F).values
            g_f = np.where(fwon, P.gw, P.gl); g_d = np.where(fwon, P.gl, P.gw)
            s_f = np.where(fwon, P.sets_w, P.sets_l); s_d = np.where(fwon, P.sets_l, P.sets_w)
            b_f = np.where(fwon, P.bw, P.bl).astype(float); b_d = np.where(fwon, P.bl, P.bw).astype(float)
            tb_f = np.where(fwon, P.tbw, P.tb - P.tbw)
            last = P.iloc[-1]; lw_ = bool(fwon[-1])
            f.update({"h2h_win_share": fwon.mean(), "h2h_games_share": g_f.sum() / max(g_f.sum() + g_d.sum(), 1),
                      "h2h_sets_share": s_f.sum() / max(s_f.sum() + s_d.sum(), 1),
                      "h2h_games_minus_wins": g_f.sum() / max(g_f.sum() + g_d.sum(), 1) - fwon.mean(),
                      "h2h_tb_share": tb_f.sum() / P.tb.sum() if P.tb.sum() else np.nan,
                      "last_won": int(lw_), "last_straight": int((s_d[-1] if lw_ else s_f[-1]) == 0),
                      "last_split": int(min(s_f[-1], s_d[-1]) > 0), "last_games_share": g_f[-1] / max(g_f[-1] + g_d[-1], 1),
                      "last_days": (t.d0 - last.d0).days, "last_same_surface": int(last.surf == t.surf)})
            ok = np.isfinite(b_f) & np.isfinite(b_d)
            if ok.any():
                f["h2h_break_share"] = b_f[ok].sum() / max(b_f[ok].sum() + b_d[ok].sum(), 1)
                f["h2h_breaks_diff_per_match"] = (b_f[ok] - b_d[ok]).mean()
                if ok[-1]:
                    f["last_breaks_diff"] = b_f[-1] - b_d[-1]
        out.append(f)
    D = pd.DataFrame(out).sort_values("date", kind="stable").reset_index(drop=True)
    D.to_pickle("work/h2h_detail.pkl")
    return D


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


FEATS = ["h2h_win_share", "h2h_games_share", "h2h_sets_share", "h2h_games_minus_wins", "h2h_tb_share", "last_won", "last_straight",
         "last_split", "last_games_share", "h2h_break_share", "h2h_breaks_diff_per_match", "last_breaks_diff"]


def model_test(D):
    from sklearn.linear_model import LogisticRegression
    M = D[D.n_meet >= 1].reset_index(drop=True)
    half = (np.arange(len(M)) >= len(M) // 2)
    X = np.c_[lg(M.p), M[FEATS].fillna(M[FEATS].median()).values, np.log1p(M.n_meet)]
    print(f"\n== Model test on {len(M):,} matches where they had met before (price + H2H detail vs price alone)")
    for tr, te in ((False, True), (True, False)):
        m = LogisticRegression(C=0.5, max_iter=2000).fit(X[half == tr], M.y[half == tr])
        q = m.predict_proba(X[half == te])[:, 1]
        y, p = M.y[half == te].values, M.p[half == te].values
        print(f"   fit on half {int(tr)}, test half {int(te)}: log-loss price {ll(y, p):.4f}  price+H2H {ll(y, q):.4f}  ({ll(y, q) - ll(y, p):+.4f})")


def rules(D):
    has = D.n_meet >= 1
    return {
        "fav dominated the last meeting (straight sets, 60%+ of games)": has & (D.last_won == 1) & (D.last_straight == 1) & (D.last_games_share >= 0.6),
        "fav LOST the last meeting in straight sets": has & (D.last_won == 0) & (D.last_straight == 1),
        "last meeting was a split (went the distance)": has & (D.last_split == 1),
        "fav won last meeting but lost the games count (lucky win)": has & (D.last_won == 1) & (D.last_games_share < 0.5),
        "fav lost last meeting but won more games (unlucky loss)": has & (D.last_won == 0) & (D.last_games_share > 0.5),
        "fav broke serve more than dog in their meetings (+1/match)": has & (D.h2h_breaks_diff_per_match >= 1),
        "dog broke serve more than fav in their meetings (+1/match)": has & (D.h2h_breaks_diff_per_match <= -1),
        "fav's H2H games share 5+ pts above their H2H win share (better than record)": has & (D.h2h_games_minus_wins >= 0.05),
        "fav's H2H games share 5+ pts below their win share (record flatters them)": has & (D.h2h_games_minus_wins <= -0.05),
        "fav wins most tiebreaks vs this dog (65%+, 3+ TBs)": has & (D.h2h_tb_share >= 0.65),
        "fav loses most tiebreaks vs this dog (35%-)": has & (D.h2h_tb_share <= 0.35),
        "fav trails H2H (win share under 50%, 3+ meetings)": has & (D.h2h_win_share < 0.5) & (D.n_meet >= 3),
        "last meeting within 60 days": has & (D.last_days <= 60),
        "last meeting on the same surface, fav won it": has & (D.last_same_surface == 1) & (D.last_won == 1),
        "last meeting on the same surface, fav lost it": has & (D.last_same_surface == 1) & (D.last_won == 0),
    }


def run_rules(D, quiet=False):
    half = (np.arange(len(D)) >= len(D) // 2)
    rf = np.where(D.y == 1, D.ps_f - 1, -1.0); rd = np.where(D.y == 0, D.ps_d - 1, -1.0)
    passed = []
    for name, m in rules(D).items():
        m = np.asarray(m, bool)
        cells, back_ok, fade_ok = [], True, True
        for h in (False, True):
            g = m & (half == h)
            if g.sum() < 80:
                cells.append(f"n={g.sum():>5} (few)"); back_ok = fade_ok = False; continue
            b, f = rf[g], rd[g]
            cells.append(f"n={g.sum():>5} fav won {D.y[g].mean():5.1%} vs {D.p[g].mean():5.1%}  back fav {b.mean():+6.1%}  back dog {f.mean():+6.1%}")
            back_ok &= b.mean() > 0.01; fade_ok &= f.mean() > 0.01
        tag = "  <-- back fav held" if back_ok else "  <-- back dog held" if fade_ok else ""
        if back_ok or fade_ok:
            passed.append(name)
        if not quiet:
            print(f"   {name:<74} " + "   ".join(cells) + tag)
    return passed


def main():
    try:
        D = pd.read_pickle("work/h2h_detail.pkl")
    except FileNotFoundError:
        D = build()
    if "--null" in sys.argv:
        k = int(sys.argv[sys.argv.index("--null") + 1]); rng = np.random.default_rng(0); c = []
        for _ in range(k):
            N = D.copy(); N["y"] = (rng.random(len(N)) < N.p.values).astype(int); c.append(len(run_rules(N, True)))
        print(f"luck runs: {c} -> mean {np.mean(c):.1f}"); return
    print(f"{len(D):,} Pinnacle-priced matches; {int((D.n_meet >= 1).sum()):,} with an earlier meeting "
          f"({int((D.n_meet >= 3).sum()):,} with 3+); break stats on {int(D.h2h_break_share.notna().sum()):,}")
    print("\n== Rules (Pinnacle close, two time halves; > +1% in both to count)")
    p = run_rules(D)
    print(f"   {len(p)} held both halves: {p}")
    model_test(D)


if __name__ == "__main__":
    main()
