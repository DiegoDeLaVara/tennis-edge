r"""Narrative / motivation study: do "incentive" stories beat Pinnacle's closing price?
Ideas from tipster-style reasoning (motivation, rivalries, defending titles, ranking pressure, letdowns) turned into
rules we can measure on 15 years of Pinnacle closing odds (tennis-data: ATP 2010-24, WTA 2020-24).

Each hypothesis marks a "story" player in some matches. We report the ROI of backing that player at Pinnacle close,
split by whether the story player is the favourite or the underdog, in two time halves. A rule only counts if it
makes money in BOTH halves; the --null runs (results drawn from the closing price, so no real edge exists) show how
many rules pass by luck alone.

  .venv\Scripts\python.exe -m tools.motivation [--null SEEDS]

Result (2026-11-06, 47,294 matches): 4 of ~50 cells held in both halves vs 5.2 per luck run -> storylines are priced.
Watch item: top-10 favourites in Masters/1000 R3/R16 (back the dog +3.4% / +9.6% Pinnacle, +23.8% on 89 Kalshi
matches 2025-26) - not staked; re-check in tools/archive_study once the archive is large.
"""
import gzip
import json
import math
import sys
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from tools.book_study import matches, td_tokens

SLAM = {"Grand Slam"}
SMALL = {"ATP250", "ATP500", "International", "WTA250", "WTA500", "Premier"}
BIG = {"Grand Slam", "Masters 1000", "WTA1000"}
EARLY = {"1st Round", "2nd Round"}


def nation_lookup():
    st = json.load(gzip.open("state/ratings.json.gz", "rt"))["players"]
    idx = defaultdict(list)
    for k, v in st.items():
        ioc = v[-5] if isinstance(v[-5], str) else None
        for t in k.split(":", 1)[1].split():
            idx[(k.split(":")[0], t)].append((k, ioc))
    cache = {}

    def get(tour, name):
        if (tour, name) in cache:
            return cache[(tour, name)]
        tt = td_tokens(name)
        out = None
        if tt and tt[0]:
            found = {ioc for k, ioc in idx.get((tour, tt[0][-1]), []) if matches(k, tt) and ioc}
            out = found.pop() if len(found) == 1 else None
        cache[(tour, name)] = out
        return out
    return get


def num(x):
    try:
        v = float(x)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


def load():
    import glob
    out = []
    for f in sorted(glob.glob("work/bookodds/*.csv")):
        D = pd.read_csv(f, encoding="latin-1", low_memory=False)
        D["date"] = pd.to_datetime(D["Date"], errors="coerce", format="mixed")
        D = D[D.date.notna()]
        out.append(pd.DataFrame({"tour": "atp" if "atp_" in f else "wta", "date": D.date.dt.date.values,
                                 "w": D.Winner.astype(str).values, "l": D.Loser.astype(str).values,
                                 "series": D.get("Series", D.get("Tier")).astype(str).values, "round": D.Round.astype(str).values,
                                 "psw": [num(x) for x in D.PSW], "psl": [num(x) for x in D.PSL],
                                 "tourn": D.Tournament.astype(str).values,
                                 "wr": pd.to_numeric(D.WRank, errors="coerce").values,
                                 "lr": pd.to_numeric(D.LRank, errors="coerce").values}))
    return pd.concat(out, ignore_index=True)


def build():
    O = load()
    O["i"] = np.arange(len(O))
    O = O.sort_values(["date", "i"], kind="stable").reset_index(drop=True)
    O["tid"] = O.tour + "|" + O.tourn.astype(str) + "|" + pd.to_datetime(O.date).dt.year.astype(str)
    start = O.groupby("tid").date.min()
    O["tstart"] = O.tid.map(start)

    bigstarts = defaultdict(list)
    for tid, (tour, ser, d0) in O.groupby("tid").agg(tour=("tour", "first"), ser=("series", "first"), d0=("date", "min")).iterrows():
        if ser in BIG:
            bigstarts[tour].append(d0)
    for k in bigstarts:
        bigstarts[k].sort()

    def big_soon(tour, d):
        return any(timedelta(1) <= b - d <= timedelta(8) for b in bigstarts[tour])
    nat = nation_lookup()
    champ = {}                                               # (tour, tournament, year) -> champion name
    for r in O[O["round"] == "The Final"].itertuples():
        champ[(r.tour, str(r.tourn), r.date.year)] = r.w

    hist = defaultdict(list)          # player -> list of (date, tid, won, opp, rank)
    tres = {}                         # (player, tid) -> (start, reached final, won title, matches won, lost first)
    rows = []
    for r in O.itertuples():
        if not (r.psw and r.psl and math.isfinite(r.psw) and math.isfinite(r.psl)):
            for p, won, opp, rk in ((r.w, 1, r.l, r.wr), (r.l, 0, r.w, r.lr)):
                hist[p].append((r.date, r.tid, won, opp, rk, np.nan, r.round))
            continue
        fw, fl = 1 / r.psw, 1 / r.psl
        pw = fw / (fw + fl)
        fav_w = pw >= 0.5
        F, Dg = (r.w, r.l) if fav_w else (r.l, r.w)
        rf, rd = (r.wr, r.lr) if fav_w else (r.lr, r.wr)
        feats = {}
        for side, p, opp, rk in (("f", F, Dg, rf), ("d", Dg, F, rd)):
            H = hist[p]
            prior = [h for h in H if h[1] != r.tid]                         # other tournaments only (no leak)
            this = [h for h in H if h[1] == r.tid]
            feats[f"first_{side}"] = len(this) == 0
            # previous tournament result
            prev_t = None
            if prior:
                last_tid = prior[-1][1]
                pt = [h for h in prior if h[1] == last_tid]
                prev_t = pt
            gap = (r.date - prior[-1][0]).days if prior else 999
            feats[f"title_last_{side}"] = bool(prev_t) and gap <= 10 and champ.get(tuple([r.tour] + prev_t[-1][1].split("|")[1:2] + [int(prev_t[-1][1].split("|")[2])])) == p
            feats[f"lostfinal_last_{side}"] = bool(prev_t) and gap <= 10 and len(prev_t) >= 3 and prev_t[-1][2] == 0 and \
                champ.get((r.tour, prev_t[-1][1].split("|")[1], int(prev_t[-1][1].split("|")[2]))) == prev_t[-1][3]
            feats[f"early_last_{side}"] = bool(prev_t) and gap <= 14 and len(prev_t) == 1 and prev_t[0][2] == 0
            feats[f"defending_{side}"] = champ.get((r.tour, str(r.tourn), r.date.year - 1)) == p
            feats[f"opener_{side}"] = not any(h[0].year == r.date.year for h in H)
            ranks = [h[4] for h in prior if np.isfinite(h[4])]
            feats[f"careerhigh_{side}"] = len(ranks) >= 30 and np.isfinite(rk) and rk < min(ranks)
            old = [h[4] for h in prior if 80 <= (r.date - h[0]).days <= 120 and np.isfinite(h[4])]
            feats[f"slump_{side}"] = bool(old) and np.isfinite(rk) and rk >= 1.6 * np.median(old) and np.median(old) <= 100
            feats[f"rising_{side}"] = bool(old) and np.isfinite(rk) and rk <= 0.6 * np.median(old)
            meet = [h for h in prior if h[3] == opp]
            feats[f"revenge_{side}"] = bool(meet) and meet[-1][2] == 0 and (r.date - meet[-1][0]).days <= 120
            feats[f"bubble_{side}"] = np.isfinite(rk) and 85 <= rk <= 115 and r.date.month in (9, 10, 11)
            feats[f"nat_{side}"] = nat(r.tour, p)
            here = [h for h in prior if h[1].split("|")[1] == str(r.tourn) and np.isfinite(h[5])]
            feats[f"tres_{side}"] = sum(h[2] - h[5] for h in here) if len(here) >= 6 else np.nan       # wins above price here
            er = [h for h in prior if h[6] in EARLY and np.isfinite(h[5])][-40:]
            feats[f"eres_{side}"] = float(np.mean([h[2] - h[5] for h in er])) if len(er) >= 15 else np.nan
            lr = [h for h in prior if h[6] not in EARLY and np.isfinite(h[5])][-40:]
            feats[f"lres_{side}"] = float(np.mean([h[2] - h[5] for h in lr])) if len(lr) >= 15 else np.nan
        rows.append({"date": r.date, "tour": r.tour, "series": str(r.series), "round": r.round,
                     "y": int(fav_w), "p": max(pw, 1 - pw), "ps_f": r.psw if fav_w else r.psl, "ps_d": r.psl if fav_w else r.psw,
                     "rank_f": rf, "rank_d": rd, "big_soon": big_soon(r.tour, r.date), **feats})
        for p, won, opp, rk, pp in ((r.w, 1, r.l, r.wr, pw), (r.l, 0, r.w, r.lr, 1 - pw)):
            hist[p].append((r.date, r.tid, won, opp, rk, pp, r.round))
    D = pd.DataFrame(rows)
    D.to_pickle("work/motivation.pkl")
    return D


def hypotheses(D):
    def pair(a, b):                                        # player trait: +1 fav has it alone, -1 dog alone
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        return np.where(a & ~b, 1, np.where(b & ~a, -1, 0))
    nf, nd = D.nat_f.values, D.nat_d.values
    derby = np.array([x is not None and x == y for x, y in zip(nf, nd)])
    late_small = D.date.map(lambda d: d.month in (10, 11)).values & D.series.isin(SMALL).values
    top20f = (D.rank_f <= 20).values
    H = {
        "defending champion": pair(D.defending_f, D.defending_d),
        "won a title last week (first match since)": pair(D.title_last_f & D.first_f, D.title_last_d & D.first_d),
        "lost a final last week (first match since)": pair(D.lostfinal_last_f & D.first_f, D.lostfinal_last_d & D.first_d),
        "lost first match last event (bounce-back)": pair(D.early_last_f & D.first_f, D.early_last_d & D.first_d),
        "season opener (first match of the year)": pair(D.opener_f, D.opener_d),
        "at best ranking we have seen (belief)": pair(D.careerhigh_f, D.careerhigh_d),
        "ranking slumping (60% worse than 3-4 months ago)": pair(D.slump_f, D.slump_d),
        "ranking rising fast (40% better than 3-4 months ago)": pair(D.rising_f, D.rising_d),
        "revenge: lost last meeting within 120 days": pair(D.revenge_f, D.revenge_d),
        "ranked 85-115 in Sep-Nov (fighting for Slam entry)": pair(D.bubble_f, D.bubble_d),
        "compatriot derby -> favourite": np.where(derby, 1, 0),
        "Oct-Nov small event, top-20 favourite -> favourite": np.where(late_small & top20f, 1, 0),
        "Slam, top-20 favourite (max motivation) -> favourite": np.where(D.series.isin(SLAM).values & top20f, 1, 0),
        "LOOK-AHEAD: top-20 fav at small event, Masters/Slam starts within 8 days -> favourite":
            np.where(D.big_soon.values & D.series.isin(SMALL).values & top20f, 1, 0),
        "LOOK-AHEAD: same, QF or later (would have to stay all week) -> favourite":
            np.where(D.big_soon.values & D.series.isin(SMALL).values & top20f & D["round"].isin(["Quarterfinals", "Semifinals", "The Final"]).values, 1, 0),
        "player beats the price at THIS tournament (+1 win or more, 6+ matches)": pair(D.tres_f >= 1, D.tres_d >= 1),
        "player trails the price at THIS tournament (-1 win or worse, 6+ matches)": pair(D.tres_f <= -1, D.tres_d <= -1),
        "slow starter: trails price by 8+ pts in R1/R2, early round now": pair((D.eres_f <= -0.08) & D["round"].isin(EARLY), (D.eres_d <= -0.08) & D["round"].isin(EARLY)),
        "fast starter: beats price by 8+ pts in R1/R2, early round now": pair((D.eres_f >= 0.08) & D["round"].isin(EARLY), (D.eres_d >= 0.08) & D["round"].isin(EARLY)),
        "fades late: trails price by 8+ pts in QF+, QF or later now": pair((D.lres_f <= -0.08) & ~D["round"].isin(EARLY), (D.lres_d <= -0.08) & ~D["round"].isin(EARLY)),
    }
    for tier_name, tiers in (("Slam", {"Grand Slam"}), ("Masters/1000", {"Masters 1000", "WTA1000"}), ("250/500", SMALL)):
        for rname, rounds in (("R1", {"1st Round"}), ("R2", {"2nd Round"}), ("R3/R16", {"3rd Round", "4th Round"}),
                              ("QF", {"Quarterfinals"}), ("SF/F", {"Semifinals", "The Final"})):
            m = D.series.isin(tiers).values & D["round"].isin(rounds).values
            H[f"top-10 favourite, {tier_name} {rname} -> favourite"] = np.where(m & (D.rank_f <= 10).values, 1, 0)
    return H


def run(D, quiet=False):
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    half = (np.arange(len(D)) >= len(D) // 2).astype(int)
    rf = np.where(D.y == 1, D.ps_f - 1, -1.0)
    rd = np.where(D.y == 0, D.ps_d - 1, -1.0)
    passed = []
    for name, s in hypotheses(D).items():
        if not quiet:
            print(f"-- {name}")
        for role, mask_fn in (("story player is FAVOURITE", lambda s: s == 1), ("story player is UNDERDOG", lambda s: s == -1)):
            m0 = mask_fn(s)
            if name.endswith("-> favourite") and role.endswith("UNDERDOG"):
                continue
            cells, back_ok, fade_ok = [], True, True
            for h in (0, 1):
                m = m0 & (half == h)
                back = np.where(s[m] == 1, rf[m], rd[m]); fade = np.where(s[m] == 1, rd[m], rf[m])
                if len(back) < 80:
                    cells.append(f"n={len(back):>5} (few)                     "); back_ok = fade_ok = False
                    continue
                se = back.std(ddof=1) / math.sqrt(len(back))
                won = (back > 0).mean()
                cells.append(f"n={len(back):>5} won {won:5.1%} back {back.mean():+6.1%} ± {se:4.1%} fade {fade.mean():+6.1%}")
                back_ok &= back.mean() > 0.01
                fade_ok &= fade.mean() > 0.01
            tag = "  <-- BACK held both halves" if back_ok else ("  <-- FADE held both halves" if fade_ok else "")
            if back_ok or fade_ok:
                passed.append((name, role, "back" if back_ok else "fade"))
            if not quiet:
                print(f"   {role:<26} " + "   ".join(cells) + tag)
    return passed


def main():
    try:
        D = pd.read_pickle("work/motivation.pkl")
    except FileNotFoundError:
        D = build()
    if "--null" in sys.argv:
        seeds = int(sys.argv[sys.argv.index("--null") + 1])
        counts = []
        for sd in range(seeds):
            rng = np.random.default_rng(sd)
            N = D.copy(); N["y"] = (rng.random(len(N)) < N.p.values).astype(int)
            counts.append(len(run(N, quiet=True)))
        print(f"luck runs: rules passing both halves per run = {counts} (mean {np.mean(counts):.1f})")
        return
    nat_ok = D.nat_f.notna() & D.nat_d.notna()
    print(f"{len(D):,} matches with Pinnacle closing odds; nationality found for both players in {nat_ok.mean():.0%}")
    print("ROI of backing / fading the story player at Pinnacle close, two time halves. A rule needs > +1% in BOTH.\n")
    passed = run(D)
    print(f"\n{len(passed)} rule(s) held in both halves: {passed}")


if __name__ == "__main__":
    main()
