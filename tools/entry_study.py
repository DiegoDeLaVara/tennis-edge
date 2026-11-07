r"""Late news / entry status: lucky losers (got in hours before after losing in qualifying), wildcards, protected or
special rankings (players returning from long injury), special exempts and alternates. Does the market misprice them?

Entry status comes from the Sackmann archive (C:\TENNIS\archive\sack, ATP 2010-24, WTA 2020-24), joined to
tennis-data Pinnacle closing odds by tour + both surnames + tournament week. Same scoring as tools.travel_study:
back / fade the story player at Pinnacle close, by favourite/underdog role, two time halves, --null luck runs.

  .venv\Scripts\python.exe -m tools.entry_study [--null N] [--rebuild]
  .venv\Scripts\python.exe -m tools.entry_study --kalshi     (fresh check on Kalshi 2025-26 tour matches)

Result (2026-11-07, 44,949 Pinnacle matches): 5 cells held both halves vs 3.3 per luck run. Best: back QUALIFIERS who
are the favourite (won 67.8/65.6% vs 62.6/62.2% priced, +5.7%/+2.5%, ~1,000 bets per half). Kalshi 2025-26 check:
same direction but unproven (116 bets, won 67.2% vs 63.7%, +2.1% ± 6.8% vs -3.3% for all favourites). Lucky-loser,
late/special-entry and wildcard-30+ rules failed on Kalshi. Watch item only. Medical timeouts: no data (the
observer score feed only marks suspended/retired).
"""
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from edge.names import fold
from tools.motivation import load as load_td

SACK = Path(r"C:\TENNIS\archive\sack")
PKL = Path("work/entry_study.pkl")
NORM = {"SR": "PR", "ALT": "Alt", "A": "Alt", "SE": "SE", "LL": "LL", "WC": "WC", "Q": "Q", "PR": "PR", "Alt": "Alt"}


def last_tok(name):
    t = fold(str(name))
    return t[-1] if t else ""


def td_last(name):
    """'Del Potro J.M.' -> 'potro'; 'Auger-Aliassime F.' -> last surname token."""
    parts = str(name).strip().split(" ")
    sur = [p for p in parts if not (p.endswith(".") and len(p.replace(".", "")) <= 3)]
    return last_tok(" ".join(sur) if sur else name)


def sack_rows():
    out = defaultdict(list)            # (tour, lastA, lastB sorted) -> [(start, end, {last: (entry, age)})]
    for tour, years in (("atp", range(2009, 2025)), ("wta", range(2019, 2025))):
        for y in years:
            f = SACK / tour / f"{tour}_matches_{y}.csv"
            if not f.exists():
                continue
            D = pd.read_csv(f, low_memory=False, usecols=["tourney_date", "winner_name", "loser_name", "winner_entry", "loser_entry", "winner_age", "loser_age"])
            for r in D.itertuples():
                d0 = pd.to_datetime(str(int(r.tourney_date)), format="%Y%m%d").date()
                lw, ll_ = last_tok(r.winner_name), last_tok(r.loser_name)
                info = {lw: (NORM.get(str(r.winner_entry), None), r.winner_age), ll_: (NORM.get(str(r.loser_entry), None), r.loser_age)}
                out[(tour,) + tuple(sorted((lw, ll_)))].append((d0 - timedelta(3), d0 + timedelta(17), info))
    return out


def build():
    O = load_td()
    S = sack_rows()
    rows, hit = [], 0
    O = O.sort_values("date", kind="stable").reset_index(drop=True)
    for r in O.itertuples():
        if not (r.psw and r.psl and math.isfinite(r.psw) and math.isfinite(r.psl)):
            continue
        lw, ll_ = td_last(r.w), td_last(r.l)
        cand = [c for c in S.get((r.tour,) + tuple(sorted((lw, ll_))), []) if c[0] <= r.date <= c[1]]
        if len(cand) != 1:
            continue
        hit += 1
        info = cand[0][2]
        fw, fl = 1 / r.psw, 1 / r.psl
        pw = fw / (fw + fl)
        fav_w = pw >= 0.5
        lf, ld = (lw, ll_) if fav_w else (ll_, lw)
        rows.append({"date": r.date, "tour": r.tour, "series": r.series, "round": r.round, "y": int(fav_w), "p": max(pw, 1 - pw),
                     "ps_f": r.psw if fav_w else r.psl, "ps_d": r.psl if fav_w else r.psw,
                     "entry_f": info[lf][0], "entry_d": info[ld][0], "age_f": info[lf][1], "age_d": info[ld][1],
                     "rank_f": r.wr if fav_w else r.lr, "rank_d": r.lr if fav_w else r.wr})
    D = pd.DataFrame(rows)
    print(f"joined {hit:,} Pinnacle matches to Sackmann entry status")
    D.to_pickle(PKL)
    return D


def hypotheses(D):
    def pair(a, b):
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        return np.where(a & ~b, 1, np.where(b & ~a, -1, 0))
    ef, ed = D.entry_f.values, D.entry_d.values
    early = D["round"].isin(["1st Round", "2nd Round"]).values
    r1 = (D["round"] == "1st Round").values
    H = {}
    for s, label in (("LL", "lucky loser (got in late after losing qualifying)"), ("WC", "wildcard"),
                     ("PR", "protected / special ranking (back from injury)"), ("SE", "special exempt"),
                     ("Alt", "alternate"), ("Q", "qualifier")):
        H[label] = pair(ef == s, ed == s)
        H[label + ", first round"] = pair((ef == s) & r1, (ed == s) & r1)
    H["wildcard aged 30+ (returning name)"] = pair((ef == "WC") & (D.age_f.values >= 30), (ed == "WC") & (D.age_d.values >= 30))
    H["wildcard aged under 21 (prospect)"] = pair((ef == "WC") & (D.age_f.values < 21), (ed == "WC") & (D.age_d.values < 21))
    H["wildcard ranked outside 150"] = pair((ef == "WC") & (D.rank_f.values > 150), (ed == "WC") & (D.rank_d.values > 150))
    H["protected ranking, early rounds"] = pair((ef == "PR") & early, (ed == "PR") & early)
    H["protected ranking ranked outside 200 (long absence)"] = pair((ef == "PR") & (D.rank_f.values > 200), (ed == "PR") & (D.rank_d.values > 200))
    H["any late or special entry (LL/PR/SE/Alt)"] = pair(np.isin(ef, ["LL", "PR", "SE", "Alt"]), np.isin(ed, ["LL", "PR", "SE", "Alt"]))
    return H


def run(D, quiet=False):
    import tools.travel_study as ts
    old = ts.hypotheses
    ts.hypotheses = hypotheses
    try:
        return ts.run(D, quiet)
    finally:
        ts.hypotheses = old


def main():
    D = build() if ("--rebuild" in sys.argv or not PKL.exists()) else pd.read_pickle(PKL)
    if "--null" in sys.argv:
        seeds = int(sys.argv[sys.argv.index("--null") + 1])
        counts = []
        for sd in range(seeds):
            rng = np.random.default_rng(sd)
            N = D.copy(); N["y"] = (rng.random(len(N)) < N.p.values).astype(int)
            counts.append(len(run(N, quiet=True)))
        print(f"luck runs: rules passing both halves per run = {counts} (mean {np.mean(counts):.1f})")
        return
    print(f"{len(D):,} matches; entry status counts (either player):")
    for s in ("LL", "WC", "PR", "SE", "Alt", "Q"):
        print(f"   {s:<4} {int(((D.entry_f == s) | (D.entry_d == s)).sum()):>6}")
    print("ROI of backing / fading the story player at Pinnacle close, two time halves. A rule needs > +1% in BOTH.\n")
    passed = run(D)
    print(f"\n{len(passed)} rule(s) held in both halves: {passed}")


if __name__ == "__main__" and "--kalshi" not in sys.argv:
    main()


def kalshi_check():
    """Fresh-data check: Kalshi tour + Challenger matches 2025-26 (real ask + fee) joined to Sackmann entry status."""
    import json
    from tools.kalshi_backtest import fee
    S = defaultdict(list)
    for tour, files in (("atp", ["atp_matches_2025", "atp_matches_2026", "atp_matches_qual_chall_2025", "atp_matches_qual_chall_2026"]),
                        ("wta", ["wta_matches_2025", "wta_matches_2026"])):
        for f in files:
            D = pd.read_csv(SACK / tour / f"{f}.csv", low_memory=False)
            D = D[~D["round"].astype(str).str.startswith("Q")]
            for r in D.itertuples():
                d0 = pd.to_datetime(str(int(r.tourney_date)), format="%Y%m%d").date()
                lw, ll_ = last_tok(r.winner_name), last_tok(r.loser_name)
                info = {lw: (NORM.get(str(r.winner_entry)), r.winner_age), ll_: (NORM.get(str(r.loser_entry)), r.loser_age)}
                S[(tour,) + tuple(sorted((lw, ll_)))].append((d0 - timedelta(3), d0 + timedelta(17), info))
    rows = []
    for l in open("work/kalshi_events.jsonl"):
        e = json.loads(l)
        if e["series"] not in ("KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH") or not e.get("quote") or e.get("round") == "Q":
            continue
        q = e["quote"]
        if q["ask"] - q["bid"] > 0.12 or not 0 < q["bid"] < q["ask"] < 1:
            continue
        tour = "wta" if e["series"] == "KXWTAMATCH" else "atp"
        d = date.fromtimestamp(e["start"])
        la, lb = last_tok(e["a"]), last_tok(e["b"])
        cand = [c for c in S.get((tour,) + tuple(sorted((la, lb))), []) if c[0] <= d <= c[1]]
        if len(cand) != 1:
            continue
        info = cand[0][2]
        px = (q["bid"] + q["ask"]) / 2
        fa = px >= 0.5
        lf, ld = (la, lb) if fa else (lb, la)
        rows.append({"date": d, "series": e["series"], "fw": int(e["y"] == (1 if fa else 0)), "pf": px if fa else 1 - px,
                     "ask_f": q["ask"] if fa else 1 - q["bid"], "ask_d": 1 - q["bid"] if fa else q["ask"],
                     "entry_f": info[lf][0], "entry_d": info[ld][0], "age_f": info[lf][1], "age_d": info[ld][1]})
    K = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    K["half"] = (K.index >= len(K) // 2).astype(int)
    print(f"{len(K):,} Kalshi tour/Challenger matches 2025-26 joined to entry status "
          f"({K.date.min()} .. {K.date.max()}); qualifiers in {int(((K.entry_f == 'Q') | (K.entry_d == 'Q')).sum())}")
    ret = lambda won, a: np.where(won, (1 - (a + fee(a))) / (a + fee(a)), -1.0)
    rules = [("qualifier favourite (vs non-qualifier) -> back", (K.entry_f == "Q") & (K.entry_d != "Q"), "f"),
             ("qualifier favourite, Challenger only", (K.entry_f == "Q") & (K.entry_d != "Q") & (K.series == "KXATPCHALLENGERMATCH"), "f"),
             ("qualifier favourite, tour only", (K.entry_f == "Q") & (K.entry_d != "Q") & (K.series != "KXATPCHALLENGERMATCH"), "f"),
             ("lucky loser favourite -> back", (K.entry_f == "LL") & (K.entry_d != "LL"), "f"),
             ("any late/special entry favourite (LL/PR/SE/Alt) -> back", K.entry_f.isin(["LL", "PR", "SE", "Alt"]) & ~K.entry_d.isin(["LL", "PR", "SE", "Alt"]), "f"),
             ("wildcard aged 30+ favourite -> back the underdog", (K.entry_f == "WC") & (K.age_f >= 30) & (K.entry_d != "WC"), "d"),
             ("qualifier underdog (reference) -> back", (K.entry_d == "Q") & (K.entry_f != "Q"), "d"),
             ("all favourites (baseline)", K.pf > 0, "f")]
    for name, m, side in rules:
        out = []
        for h in (0, 1, None):
            g = K[m & ((K.half == h) if h is not None else True)]
            if len(g) < 10:
                out.append(f"{'all' if h is None else 'half ' + str(h)}: n={len(g)}"); continue
            won = g.fw.values == (1 if side == "f" else 0)
            a = (g.ask_f if side == "f" else g.ask_d).values
            r = ret(won, a)
            pr = (g.pf if side == "f" else 1 - g.pf).mean()
            out.append(f"{'all' if h is None else 'half ' + str(h)}: n={len(g):>4} won {won.mean():5.1%} vs {pr:5.1%} ROI {r.mean():+6.1%} ± {r.std(ddof=1) / math.sqrt(len(r)):4.1%}")
        print(f"   {name:<58} " + "   ".join(out))


if __name__ == "__main__" and "--kalshi" in sys.argv:
    kalshi_check()
