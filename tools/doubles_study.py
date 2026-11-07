r"""Doubles on Kalshi: is there anything to bet, and can we price them better than the market?
Data: work/kalshi_doubles.jsonl (tools.kalshi_collect --series KX*DOUBLES --out ...), singles schedule from
work/kalshi_events.jsonl, singles ratings from state/ratings.json.gz.

Part 1  price accuracy by band and level, wide spreads.
Part 2  rating models, walk-forward (fit on one half, test on the other): singles Elo of the pair, and a doubles Elo
        learned only from earlier doubles results. Log-loss vs the market and betting where they disagree 10+ pts.
Part 3  situational rules (back / fade the side with the trait, two halves, real ask + fee):
        regular partnership vs new pair, singles star in the pair, played singles earlier the same day, same-country
        pair, recent doubles form, later rounds. --null N reruns part 3 on results drawn from the price (luck check).

  .venv\Scripts\python.exe -m tools.doubles_study [--null N]
"""
import gzip
import json
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from edge.names import fold
from tools.kalshi_backtest import fee, ll

GROUP = {"KXATPDOUBLES": "tour", "KXWTADOUBLES": "tour", "KXATPCHALLENGERDOUBLES": "challenger",
         "KXITFDOUBLES": "itf", "KXITFWDOUBLES": "itf"}
TOUR = {"KXATPDOUBLES": "atp", "KXWTADOUBLES": "wta", "KXATPCHALLENGERDOUBLES": "atp", "KXITFDOUBLES": "atp", "KXITFWDOUBLES": "wta"}
LEVELS = ("tour", "challenger", "itf")


def pkey(name):
    return " ".join(sorted(t for t in fold(name) if t not in ("jr", "sr")))


def players(side):
    return [pkey(p) for p in side.split("/") if p.strip()]


def load():
    rows = []
    for l in open("work/kalshi_doubles.jsonl"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        q = r.get("quote")
        if not q or q["ask"] - q["bid"] > 0.15 or not (0.02 <= q["ask"] <= 0.98):
            continue
        rows.append({"event": r["event"], "series": r["series"], "g": GROUP[r["series"]], "tour": TOUR[r["series"]],
                     "start": r["start"], "end": r["end"], "a": r["a"], "b": r["b"], "y": r["y"], "bid": q["bid"], "ask": q["ask"],
                     "round": r.get("round") or ""})
    D = pd.DataFrame(rows).sort_values("start", kind="stable").reset_index(drop=True)
    D["mid"] = (D.bid + D.ask) / 2
    D["spread"] = D.ask - D.bid
    D["half"] = (np.arange(len(D)) >= len(D) // 2).astype(int)
    D["cost_a"] = D.ask + fee(D.ask)
    D["cost_b"] = (1 - D.bid) + fee(1 - D.bid)                                  # buy NO on A
    D["r_a"] = np.where(D.y == 1, 1 / D.cost_a - 1, -1.0)
    D["r_b"] = np.where(D.y == 0, 1 / D.cost_b - 1, -1.0)
    fa = D.mid >= 0.5
    D["pf"] = np.where(fa, D.mid, 1 - D.mid)
    D["fw"] = np.where(fa, D.y, 1 - D.y)
    D["r_fav"] = np.where(fa, D.r_a, D.r_b)
    D["r_dog"] = np.where(fa, D.r_b, D.r_a)
    return D


def returns(D, y):
    r_a = np.where(y == 1, 1 / D.cost_a.values - 1, -1.0)
    r_b = np.where(y == 0, 1 / D.cost_b.values - 1, -1.0)
    return r_a, r_b


# ---------------------------------------------------------------- features (strictly from earlier matches)
def singles_state():
    st = json.load(gzip.open("state/ratings.json.gz", "rt"))["players"]
    out = {}
    for k, v in st.items():
        tour, nm = k.split(":", 1)
        kk = (tour, " ".join(sorted(nm.split())))
        if kk not in out or v[2] > out[kk][1]:
            out[kk] = (v[1], v[2], v[-5] if isinstance(v[-5], str) else None)
    return out


def singles_schedule():
    S = defaultdict(list)                      # player key -> singles start times
    for l in open("work/kalshi_events.jsonl"):
        r = json.loads(l)
        for nm in (r["a"], r["b"]):
            S[pkey(nm)].append(r["start"])
    for k in S:
        S[k].sort()
    return S


def features(D):
    ss, sched = singles_state(), singles_schedule()
    elo = defaultdict(lambda: 1500.0); nd = defaultdict(int)
    pair_n = defaultdict(int)
    form = defaultdict(list)                   # player -> recent doubles results (1/0), price-adjusted
    pending = []                               # matches started but not yet finished: (end, row index)
    F = defaultdict(list)
    ends = D.end.values
    order = np.argsort(D.start.values, kind="stable")
    done_upto = []
    fin = sorted(range(len(D)), key=lambda i: ends[i])
    fi = 0
    for i in order:
        r = D.iloc[i]
        # apply every match that finished before this one started
        while fi < len(fin) and ends[fin[fi]] < r.start:
            j = fin[fi]; fi += 1
            rj = D.iloc[j]
            pa, pb = players(rj.a), players(rj.b)
            ea, eb = np.mean([elo[p] for p in pa]), np.mean([elo[p] for p in pb])
            exp = 1 / (1 + 10 ** ((eb - ea) / 400))
            for p in pa:
                elo[p] += 32 * (rj.y - exp); nd[p] += 1; form[p].append(rj.y - rj.mid)
            for p in pb:
                elo[p] -= 32 * (rj.y - exp); nd[p] += 1; form[p].append((1 - rj.y) - (1 - rj.mid))
            pair_n[tuple(sorted(pa))] += 1; pair_n[tuple(sorted(pb))] += 1
        for side, nm in (("a", r.a), ("b", r.b)):
            ps = players(nm)
            F[f"delo_{side}"].append(np.mean([elo[p] for p in ps]))
            F[f"dn_{side}"].append(min(nd[p] for p in ps) if ps else 0)
            F[f"pair_{side}"].append(pair_n[tuple(sorted(ps))])
            sv = [ss.get((r.tour, p)) for p in ps]
            ok = [v for v in sv if v and v[1] >= 10]
            F[f"selo_{side}"].append(np.mean([v[0] for v in ok]) if len(ok) == len(ps) == 2 else np.nan)
            F[f"star_{side}"].append(max([v[0] for v in ok], default=np.nan))
            nat = [v[2] for v in sv if v]
            F[f"samenat_{side}"].append(len(nat) == 2 and nat[0] == nat[1])
            busy = 0
            for p in ps:
                t = sched.get(p, [])
                busy += any(r.start - 14 * 3600 <= x < r.start - 1800 for x in t[-50:])   # singles earlier that day
            F[f"busy_{side}"].append(busy)
            fr = [x for p in ps for x in form[p][-4:]]
            F[f"form_{side}"].append(np.mean(fr) if len(fr) >= 4 else np.nan)
    for k, v in F.items():
        arr = np.empty(len(D), dtype=object if k.startswith("samenat") else float)
        arr[order] = v
        D[k] = arr
    D["samenat_a"] = D.samenat_a.astype(bool); D["samenat_b"] = D.samenat_b.astype(bool)
    return D


# ---------------------------------------------------------------- part 1
def part1(D):
    print("PART 1. PRICE ACCURACY: favourite won vs priced; ROI at the real ask + fee (half 1 | half 2)")
    bands = [("50-60c", 0.5, 0.6), ("60-70c", 0.6, 0.7), ("70-80c", 0.7, 0.8), ("80-90c", 0.8, 0.9), ("90c+", 0.9, 1.01)]
    for g in LEVELS:
        G = D[D.g == g]
        if not len(G):
            continue
        print(f"  {g}: n={len(G)}  favourites won {G.fw.mean():.1%} priced {G.pf.mean():.1%}; back every fav {G.r_fav.mean():+.1%}, every dog {G.r_dog.mean():+.1%}")
        for bn, lo, hi in bands:
            cells = []
            for h in (0, 1):
                x = G[(G.pf >= lo) & (G.pf < hi) & (G.half == h)]
                if len(x) < 30:
                    cells.append(f"n={len(x):>4} (few)" + " " * 45); continue
                se = x.r_fav.std(ddof=1) / math.sqrt(len(x))
                cells.append(f"n={len(x):>4} won {x.fw.mean():5.1%} vs {x.pf.mean():5.1%} fav {x.r_fav.mean():+6.1%}±{se:4.1%} dog {x.r_dog.mean():+6.1%}")
            print(f"     {bn:<7} " + " | ".join(cells))
    print("\n  Wide spreads (ask - bid >= 5c):")
    for g in LEVELS:
        for lab, m in (("tight <5c", D.spread < 0.05), ("wide >=5c", D.spread >= 0.05)):
            x = D[(D.g == g) & m]
            if len(x) >= 30:
                cells = []
                for h in (0, 1):
                    xh = x[x.half == h]
                    cells.append(f"fav won {xh.fw.mean():5.1%} vs {xh.pf.mean():5.1%} fav {xh.r_fav.mean():+6.1%} dog {xh.r_dog.mean():+6.1%}")
                print(f"   {g:<10} {lab:<10} n={len(x):>5}  " + " | ".join(cells))


# ---------------------------------------------------------------- part 2
def part2(D):
    print("\nPART 2. RATING MODELS vs THE MARKET (fit on one half, scored on the other)")
    lm_all = np.log(D.mid / (1 - D.mid)).values
    for name, da, db, okm in (("singles Elo of the pair", D.selo_a, D.selo_b, D.selo_a.notna() & D.selo_b.notna()),
                              ("doubles Elo (from earlier doubles)", D.delo_a, D.delo_b, (D.dn_a >= 3) & (D.dn_b >= 3))):
        for g in LEVELS:
            m = (okm & (D.g == g)).values
            if m.sum() < 150:
                print(f"  {name:<36} {g:<10} only {m.sum()} matches"); continue
            diff = ((da - db).values[m]) / 400 * math.log(10)
            lm = lm_all[m]; y = D.y.values[m]; h = D.half.values[m]; mid = D.mid.values[m]
            res = []
            for tr, te in ((h == 0, h == 1), (h == 1, h == 0)):
                if tr.sum() < 50 or te.sum() < 50:
                    continue
                f = lambda w, s: ll(y[s], 1 / (1 + np.exp(-(w[0] * lm[s] + w[1] * diff[s]))))
                w = minimize(f, [1.0, 0.0], args=(tr,)).x
                c = minimize(lambda c: ll(y[tr], 1 / (1 + np.exp(-c[0] * diff[tr]))), [1.0]).x
                res.append(f"market {ll(y[te], mid[te]):.4f} | +rating {f(w, te):.4f} (w {w[1]:+.2f}) | rating alone {ll(y[te], 1 / (1 + np.exp(-c[0] * diff[te]))):.4f}")
            print(f"  {name:<36} {g:<10} n={m.sum():>5}  " + "   ||   ".join(res))
            pe = 1 / (1 + np.exp(-diff))
            gap = pe - mid
            ra, rb = D.r_a.values[m], D.r_b.values[m]
            cells = []
            for h0 in (0, 1):
                k = (np.abs(gap) >= 0.10) & (h == h0)
                if k.sum() >= 30:
                    r = np.where(gap[k] > 0, ra[k], rb[k])
                    cells.append(f"half {h0 + 1}: n={k.sum()} {r.mean():+6.1%} ± {r.std(ddof=1) / math.sqrt(k.sum()):4.1%}")
            if cells:
                print(f"  {'':<36} {'':<10} back the side the rating prefers by 10+ pts: " + "   ".join(cells))


# ---------------------------------------------------------------- part 3
def rules(D):
    def pair(a, b):
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        return np.where(a & ~b, 1, np.where(b & ~a, -1, 0))
    star_gap = (D.star_a - D.star_b).values
    late = D["round"].isin(["QF", "SF", "F"]).values
    return {
        "regular partnership (2+ earlier matches together) vs new pair": pair(D.pair_a >= 2, D.pair_b >= 2),
        "pair has a singles star (best singles Elo 150+ higher)": np.where(star_gap >= 150, 1, np.where(star_gap <= -150, -1, 0)),
        "played singles earlier the same day": pair(D.busy_a >= 1, D.busy_b >= 1),
        "same-country pair vs mixed pair": pair(D.samenat_a, D.samenat_b),
        "hot doubles form (beat their prices by 15+ pts lately)": pair(D.form_a >= 0.15, D.form_b >= 0.15),
        "cold doubles form (trailed their prices by 15+ pts lately)": pair(D.form_a <= -0.15, D.form_b <= -0.15),
        "favourite in QF/SF/F -> favourite": np.where(late, np.where(D.mid >= 0.5, 1, -1), 0),
    }


def part3(D, y=None, quiet=False):
    y = D.y.values if y is None else y
    r_a, r_b = returns(D, y)
    fa = D.mid.values >= 0.5
    passed = []
    if not quiet:
        print("\nPART 3. SITUATIONS: ROI of backing / fading the side with the trait (half 1 | half 2)")
    for name, s in rules(D).items():
        if not quiet:
            print(f"-- {name}")
        for g in LEVELS + ("all",):
            lv = np.ones(len(D), bool) if g == "all" else (D.g == g).values
            for role, want_fav in (("trait side is FAVOURITE", True), ("trait side is UNDERDOG", False)):
                trait_is_a = s == 1
                is_fav = np.where(trait_is_a, fa, ~fa)
                m0 = lv & (s != 0) & (is_fav == want_fav)
                back = np.where(trait_is_a, r_a, r_b); fade = np.where(trait_is_a, r_b, r_a)
                cells, bok, fok = [], True, True
                for h in (0, 1):
                    m = m0 & (D.half.values == h)
                    if m.sum() < 40:
                        cells.append(f"n={m.sum():>4} (few)" + " " * 28); bok = fok = False; continue
                    b, f = back[m], fade[m]
                    cells.append(f"n={m.sum():>4} back {b.mean():+6.1%} ± {b.std(ddof=1) / math.sqrt(m.sum()):4.1%} fade {f.mean():+6.1%}")
                    bok &= b.mean() > 0.02; fok &= f.mean() > 0.02
                if bok or fok:
                    passed.append((name, g, role, "back" if bok else "fade"))
                if not quiet and m0.sum() >= 80:
                    print(f"   {g:<10} {role:<24} " + " | ".join(cells) + ("  <-- BACK held" if bok else "  <-- FADE held" if fok else ""))
    return passed


def main():
    D = features(load())
    if "--null" in sys.argv:
        k = int(sys.argv[sys.argv.index("--null") + 1]); c = []
        for sd in range(k):
            y = (np.random.default_rng(sd).random(len(D)) < D.mid.values).astype(int)
            c.append(len(part3(D, y, quiet=True)))
        print(f"luck runs (part 3): rules passing both halves per run = {c} (mean {np.mean(c):.1f})")
        return
    print(f"{len(D):,} settled doubles matches with a pre-match quote "
          f"({pd.to_datetime(D.start.min(), unit='s'):%Y-%m-%d} .. {pd.to_datetime(D.start.max(), unit='s'):%Y-%m-%d}); "
          + ", ".join(f"{g} {int((D.g == g).sum())}" for g in LEVELS) + "\n")
    part1(D)
    part2(D)
    p = part3(D)
    print(f"\n{len(p)} rule cells held in both halves: {p}")


if __name__ == "__main__":
    main()
