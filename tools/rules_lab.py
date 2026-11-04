r"""Rules lab: can past matches and past upsets predict anything the closing price misses?

Every feature is computed walk-forward (only matches before the one being priced). Each is judged on matches with
Pinnacle closing odds (ATP 2010-24, WTA 2020-24): fitted with the price on 2010-17, scored on 2018-24, and broken
into quintiles so we can see whether favourites beat or miss their price in a consistent direction in BOTH halves.

Features (all from the favourite's side minus the underdog's unless noted):
  mres      - market reputation: how much a player has beaten (+) or missed (-) their own Pinnacle odds before
  tres      - the same, at this tournament only
  ups_win   - player's previous match was an upset win (won as a <=35% underdog by the market), within 21 days
  ups_loss  - previous match was an upset loss (lost as a >=70% favourite), within 21 days
  share     - share of games won in the previous match (blowout vs grind), any level
  qual      - came through qualifying into this event
  rank_gap  - what the ATP/WTA ranking says minus what our Elo says (does the market lean on ranking?)
  surf_gap  - surface Elo minus overall Elo (surface specialist)

  .venv\Scripts\python.exe -m tools.rules_lab
"""
import math
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tools.book_study import load_odds, td_tokens, matches

CACHE = Path("work/rules_lab.pkl")
RORD = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q": 3, "R128": 4, "R64": 5, "R32": 6, "R16": 7, "RR": 7, "QF": 8, "SF": 9, "BR": 9, "F": 10}


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def history(F):
    """Per feature row, pre-match info for side a and b from each player's previous match (any level)."""
    order = np.lexsort((F["round"].map(RORD).fillna(5).values, F.d.values))
    last = {}
    n = len(F)
    out = {k: np.full(n, np.nan) for k in ("share_a", "share_b", "qual_a", "qual_b")}
    A, Bk, Y, PE = F.a.values, F.b.values, F.y.values, F.p_elo.values
    GA, GB, RD, D = F.games_a.values, F.games_b.values, np.array([str(x) if isinstance(x, str) else "" for x in F["round"]], dtype=object), F.d.values
    for i in order:
        for side, k in ((0, A[i]), (1, Bk[i])):
            s = "a" if side == 0 else "b"
            L = last.get(k)
            if L is not None and (D[i] - L[0]).days <= 60:
                out["share_" + s][i] = L[1]
                out["qual_" + s][i] = 1.0 if (L[2].startswith("Q") and not RD[i].startswith("Q") and (D[i] - L[0]).days <= 10) else 0.0
            ga, gb = (GA[i], GB[i]) if side == 0 else (GB[i], GA[i])
            share = ga / (ga + gb) if ga + gb > 0 else np.nan
            last[k] = (D[i], share, RD[i])
    return out


def build():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    import glob
    O = load_odds()
    # load_odds skips rows with bad dates, so re-read the extra columns the same way to keep them aligned
    rows = []
    for f in sorted(glob.glob("work/bookodds/*.csv")):
        D = pd.read_csv(f, encoding="latin-1", low_memory=False)
        for r in D.to_dict("records"):
            try:
                pd.to_datetime(r["Date"])
            except Exception:
                continue
            rows.append((r.get("Tournament"), pd.to_numeric(r.get("WRank"), errors="coerce"), pd.to_numeric(r.get("LRank"), errors="coerce")))
    O["tournament"] = [x[0] for x in rows]
    O["wrank"] = [x[1] for x in rows]
    O["lrank"] = [x[2] for x in rows]

    F = pd.read_csv("work/features.csv", low_memory=False)
    F = F[F.date >= "2008-01-01"].reset_index(drop=True)
    F["d"] = pd.to_datetime(F.date).dt.date
    H = history(F)
    idx = defaultdict(list)
    for i, (tour, a, b, d) in enumerate(zip(F.tour, F.a, F.b, F.d)):
        if d.year < 2009:
            continue
        for side, k in ((0, a), (1, b)):
            for t in k.split(":", 1)[1].split():
                idx[(tour, t)].append((d, i, side))
    out = []
    for r in O.itertuples():
        if not (r.psw and r.psl) or r.psw != r.psw or r.psl != r.psl:      # NaN-safe
            continue
        tw, tl = td_tokens(r.w), td_tokens(r.l)
        if not tw or not tl or not tw[0]:
            continue
        hit = None
        for (d, i, side) in idx.get((r.tour, tw[0][-1]), []):
            if not (timedelta(0) <= r.date - d <= timedelta(16)):
                continue
            ka, kb = F.at[i, "a"], F.at[i, "b"]
            kw, kl = (ka, kb) if side == 0 else (kb, ka)
            if matches(kw, tw) and matches(kl, tl) and ((side == 0) == (F.at[i, "y"] == 1)):
                hit = (i, side)
                break
        if hit is None:
            continue
        i, side = hit
        w, l = ("a", "b") if side == 0 else ("b", "a")
        pe = F.at[i, "p_elo"] if side == 0 else 1 - F.at[i, "p_elo"]
        pea = F.at[i, "p_elo_all"] if side == 0 else 1 - F.at[i, "p_elo_all"]
        out.append({"date": r.date, "tour": r.tour, "tournament": r.tournament, "kw": F.at[i, w], "kl": F.at[i, l],
                    "psw": r.psw, "psl": r.psl, "maxw": r.maxw, "maxl": r.maxl, "wrank": r.wrank, "lrank": r.lrank,
                    "pe_w": pe, "pea_w": pea, "share_w": H["share_" + w][i], "share_l": H["share_" + l][i],
                    "qual_w": H["qual_" + w][i], "qual_l": H["qual_" + l][i], "series": str(r.series), "round": str(r.round)})
    J = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    J.to_pickle(CACHE)
    return J


def market_features(J):
    """Walk-forward market reputation and upset memory over the Pinnacle-priced matches, in date order."""
    res = defaultdict(lambda: [0.0, 0])
    tres = defaultdict(lambda: [0.0, 0])
    tier_res = defaultdict(lambda: [0.0, 0])
    deep = {}                                  # player -> date of their last Slam QF-or-better
    slam_first = {}
    for r in J.itertuples():
        if tier(r.series) == "slam":
            k = (r.date.year, r.tournament)
            slam_first[k] = min(slam_first.get(k, r.date), r.date)
    last = {}
    cols = defaultdict(list)
    for r in J.itertuples():
        iw, il = 1 / r.psw, 1 / r.psl
        pw = iw / (iw + il)
        for k, s in ((r.kw, "w"), (r.kl, "l")):
            a, n = res[k]
            cols["mres_" + s].append(a / (n + 25))
            cols["mn_" + s].append(n)
            a, n = tres[(k, r.tournament)]
            cols["tres_" + s].append(a / (n + 4))
            L = last.get(k)
            recent = L is not None and (r.date - L[0]).days <= 21
            cols["upsw_" + s].append(1.0 if recent and L[1] == "upset_win" else 0.0)
            cols["upsl_" + s].append(1.0 if recent and L[1] == "upset_loss" else 0.0)
            a, n = tier_res[(k, tier(r.series))]
            cols["tier_" + s].append(a / (n + 10))
            cols["tiern_" + s].append(n)
            dd = deep.get(k)
            cols["postslam_" + s].append(1.0 if (dd is not None and tier(r.series) != "slam" and 0 < (r.date - dd).days <= 14) else 0.0)
        # update after the match
        for k, won, p in ((r.kw, 1, pw), (r.kl, 0, 1 - pw)):
            res[k][0] += won - p; res[k][1] += 1
            tres[(k, r.tournament)][0] += won - p; tres[(k, r.tournament)][1] += 1
            tag = "upset_win" if (won and p <= 0.35) else "upset_loss" if (not won and p >= 0.70) else None
            tier_res[(k, tier(r.series))][0] += won - p; tier_res[(k, tier(r.series))][1] += 1
            if tier(r.series) == "slam" and any(x in str(r.round) for x in ("Quarter", "Semi", "The Final")):
                deep[k] = r.date
            last[k] = (r.date, tag)
    for c, v in cols.items():
        J[c] = v
    # week before a Slam: non-Slam matches within 8 days of a Slam's first match
    starts = sorted(slam_first.values())
    J["preslam"] = [1.0 if tier(sr) != "slam" and any(0 < (st - d).days <= 8 for st in starts) else 0.0 for d, sr in zip(J.date, J.series)]
    J["tier"] = [tier(x) for x in J.series]
    return J


def tier(series):
    s = str(series)
    if "Grand Slam" in s:
        return "slam"
    if "Masters" in s or "1000" in s or "Premier Mandatory" in s:
        return "1000"
    if "500" in s or "Premier 5" in s:
        return "500"
    return "250"


def orient(J):
    """Favourite-oriented table: y = favourite won, x_* = favourite minus underdog (or the favourite's own flag)."""
    iw, il = 1 / J.psw, 1 / J.psl
    pw = iw / (iw + il)
    fw = pw >= 0.5
    pick = lambda w, l: np.where(fw, J[w], J[l])
    J = J[np.isfinite(J.psw.astype(float)) & np.isfinite(J.psl.astype(float))].reset_index(drop=True)
    iw, il = 1 / J.psw, 1 / J.psl
    pw = iw / (iw + il)
    fw = pw >= 0.5
    S = pd.DataFrame({"date": J.date, "year": pd.to_datetime(J.date).dt.year, "tour": J.tour, "y": fw.astype(int),
                      "p": np.where(fw, pw, 1 - pw), "ps_f": pick("psw", "psl"), "ps_d": pick("psl", "psw"),
                      "max_f": pick("maxw", "maxl"), "max_d": pick("maxl", "maxw")})
    S["half"] = (S.year >= 2018).astype(int)
    S["preslam"], S["tier"] = J.preslam.values, J.tier.values
    for f in ("mres", "tres", "upsw", "upsl", "share", "qual", "tier", "tiern", "postslam"):
        S[f + "_f"], S[f + "_d"] = pick(f + "_w", f + "_l"), pick(f + "_l", f + "_w")
    S["mn_f"], S["mn_d"] = pick("mn_w", "mn_l"), pick("mn_l", "mn_w")
    pe = np.where(fw, J.pe_w, 1 - J.pe_w)
    pea = np.where(fw, J.pea_w, 1 - J.pea_w)
    rf, rd = pick("wrank", "lrank"), pick("lrank", "wrank")
    S["rank_lr"] = np.log(np.clip(rd.astype(float), 1, 3000) / np.clip(rf.astype(float), 1, 3000))
    S["pe"], S["pea"] = pe, pea
    S["surf_gap"] = lg(pe) - lg(pea)
    return S


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_offset(x, y, off, iters=50):
    """Logistic y ~ off + a + b*x (Newton). Returns a, b, se_b."""
    X = np.c_[np.ones(len(x)), x]
    w = np.zeros(2)
    for _ in range(iters):
        z = off + X @ w
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (y - p)
        Hm = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        w += np.linalg.solve(Hm, g)
    se = math.sqrt(np.linalg.inv(Hm)[1, 1])
    return w[0], w[1], se


def roi(odds, won):
    r = np.where(won, odds - 1, -1.0)
    return len(r), (r.mean() if len(r) else 0), (r.std(ddof=1) / math.sqrt(len(r)) if len(r) > 1 else 0)


def test(S, name, x, note=""):
    ok = np.isfinite(x)
    tr = ok & (S.half == 0).values
    te = ok & (S.half == 1).values
    off = lg(S.p.values)
    print(f"\n-- {name}  {note}")
    out = {}
    for h, m in (("2010-17", tr), ("2018-24", te)):
        if m.sum() < 300:
            print(f"   {h}: too few ({m.sum()})")
            continue
        a, b, se = fit_offset(x[m], S.y.values[m], off[m])
        out[h] = (b, se)
        print(f"   {h}: n={m.sum():>6}  effect beyond the price {b:+.3f} (z = {b / se:+.1f})")
    if "2010-17" in out:
        a, b, se = fit_offset(x[tr], S.y.values[tr], off[tr])
        p0 = 1 / (1 + np.exp(-off[te]))
        p1 = 1 / (1 + np.exp(-(off[te] + a + b * x[te])))
        p0b = 1 / (1 + np.exp(-(off[te] + a)))
        print(f"   fitted on 2010-17, scored on 2018-24: log-loss price {ll(S.y.values[te], p0b):.5f} -> with feature {ll(S.y.values[te], p1):.5f} "
              f"({(ll(S.y.values[te], p0b) - ll(S.y.values[te], p1)) * 1e4:+.1f} bp)")
    # quintiles (cut points from the older half)
    try:
        qs = np.unique(np.nanquantile(x[tr], [0.2, 0.4, 0.6, 0.8]))
        if len(qs) >= 2:
            bins = np.digitize(x, qs)
            line = []
            for h, m in (("old", tr), ("new", te)):
                cells = []
                for k in range(len(qs) + 1):
                    g = m & (bins == k)
                    cells.append(f"{(S.y.values[g].mean() - S.p.values[g].mean()) * 100:+.1f}" if g.sum() >= 80 else "  . ")
                line.append(f"{h}: " + " ".join(f"{c:>5}" for c in cells))
            print("   fav won minus priced (pts), low -> high feature:  " + "   |   ".join(line))
    except Exception:
        pass
    return out


def rule(S, name, mask, side):
    """Flat 1u on the favourite (side 'f') or underdog ('d') at Pinnacle and at the best price, both halves."""
    parts = []
    for h in (0, 1):
        g = S[mask & (S.half == h)]
        if side == "f":
            a = roi(g.ps_f.values, g.y.values == 1); b = roi(g.max_f.dropna().values, g.y.values[g.max_f.notna().values] == 1)
        else:
            a = roi(g.ps_d.values, g.y.values == 0); b = roi(g.max_d.dropna().values, g.y.values[g.max_d.notna().values] == 0)
        parts.append(f"{'2010-17' if h == 0 else '2018-24'} n={a[0]:>5} Pinnacle {a[1]:+6.1%} ± {a[2]:4.1%} best {b[1]:+6.1%}")
    print(f"   RULE {name:<50} " + "  |  ".join(parts))


def main():
    J = build()
    J = J[np.isfinite(J.psw.astype(float)) & np.isfinite(J.psl.astype(float))].sort_values("date").reset_index(drop=True)
    J = market_features(J)
    S = orient(J)
    print(f"{len(S):,} matches with Pinnacle closing prices; favourite won {S.y.mean():.1%}, priced {S.p.mean():.1%}")

    # rank vs Elo: rank-implied prob fitted on 2010-17
    tr = (S.half == 0) & np.isfinite(S.rank_lr)
    a, b, _ = fit_offset(S.rank_lr.values[tr], S.y.values[tr], np.zeros(tr.sum()))
    S["rank_gap"] = (a + b * S.rank_lr) - lg(S.pe)

    seasoned = (S.mn_f >= 15) & (S.mn_d >= 15)
    test(S, "market reputation (mres fav - dog)", np.where(seasoned, S.mres_f - S.mres_d, np.nan), "players with 15+ priced matches each")
    test(S, "market reputation of the favourite only", np.where(S.mn_f >= 15, S.mres_f, np.nan))
    test(S, "market reputation of the underdog only", np.where(S.mn_d >= 15, S.mres_d, np.nan))
    test(S, "tournament reputation (tres fav - dog)", (S.tres_f - S.tres_d).values)
    test(S, "favourite's last match was an upset LOSS", S.upsl_f.values.astype(float))
    test(S, "underdog's last match was an upset WIN", S.upsw_d.values.astype(float))
    test(S, "favourite's last match was an upset WIN", S.upsw_f.values.astype(float))
    test(S, "underdog's last match was an upset LOSS", S.upsl_d.values.astype(float))
    test(S, "last-match game share (fav - dog)", (S.share_f - S.share_d).values)
    test(S, "favourite is a qualifier", S.qual_f.values.astype(float))
    test(S, "underdog is a qualifier", S.qual_d.values.astype(float))
    test(S, "ranking says more than Elo (rank_gap)", S.rank_gap.values)
    test(S, "surface specialist (surface Elo - overall Elo)", S.surf_gap.values)
    tiered = (S.tiern_f >= 10) & (S.tiern_d >= 10)
    test(S, "effort by event tier: record vs odds at this tier (fav - dog)", np.where(tiered, S.tier_f - S.tier_d, np.nan), "10+ priced matches at the tier each")
    test(S, "favourite's record vs odds at this tier", np.where(S.tiern_f >= 10, S.tier_f, np.nan))
    test(S, "week before a Slam", S.preslam.values.astype(float))
    test(S, "favourite's first event after a deep Slam run", S.postslam_f.values.astype(float))
    test(S, "underdog's first event after a deep Slam run", S.postslam_d.values.astype(float))
    for t in ("250", "500", "1000", "slam"):
        g = S[S.tier == t]
        print(f"   tier {t:<5} n={len(g):>6}  favourite won {g.y.mean():.1%} vs priced {g.p.mean():.1%} ({(g.y.mean() - g.p.mean()) * 100:+.1f} pts)")

    print("\n== Candidate rules (flat 1u)")
    qt = np.nanquantile((S.tier_f - S.tier_d)[tiered & (S.half == 0)], [0.1, 0.9])
    rule(S, "fav coasts at this tier (bottom 10%): back dog", tiered & ((S.tier_f - S.tier_d) <= qt[0]), "d")
    rule(S, "fav shows up at this tier (top 10%): back fav", tiered & ((S.tier_f - S.tier_d) >= qt[1]), "f")
    rule(S, "week before a Slam: back dog", S.preslam == 1, "d")
    rule(S, "week before a Slam: back fav", S.preslam == 1, "f")
    rule(S, "fav after deep Slam run: back dog", S.postslam_f == 1, "d")
    rule(S, "back fav whose last match was an upset loss", S.upsl_f == 1, "f")
    rule(S, "fade dog coming off an upset win (back fav)", S.upsw_d == 1, "f")
    rule(S, "back dog coming off an upset win", S.upsw_d == 1, "d")
    q = np.nanquantile((S.mres_f - S.mres_d)[seasoned & (S.half == 0)], [0.1, 0.9])
    rule(S, "fav with much better market reputation (top 10%)", seasoned & ((S.mres_f - S.mres_d) >= q[1]), "f")
    rule(S, "dog with much better market reputation (bottom 10%)", seasoned & ((S.mres_f - S.mres_d) <= q[0]), "d")
    qr = np.nanquantile(S.rank_gap[S.half == 0], [0.1, 0.9])
    rule(S, "ranking flatters the fav (top 10% rank_gap): back dog", S.rank_gap >= qr[1], "d")
    rule(S, "ranking under-sells the fav (bottom 10%): back fav", S.rank_gap <= qr[0], "f")
    qs_ = np.nanquantile(S.surf_gap[S.half == 0], [0.1, 0.9])
    rule(S, "fav is the surface specialist (top 10%)", S.surf_gap >= qs_[1], "f")
    rule(S, "dog is the surface specialist (bottom 10%)", S.surf_gap <= qs_[0], "d")
    S.to_pickle("work/rules_lab_S.pkl")


if __name__ == "__main__":
    main()
