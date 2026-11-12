r"""Scenario batch (2026-11-11): situations with a physical or market reason that earlier studies did not cover.

Pinnacle (tennis-data 2010-24, set scores, exact 50/50 prices dropped):
  last match was draining for the favourite / underdog: won a deciding-set tiebreak, came back from a set down, went
  the full distance; or very easy (opponent won 4 games or fewer); and "the bye effect": the favourite plays its
  first match at the event (bye) against an underdog who already won a round there.
Kalshi 2025-26 (real ask + fee):
  rest in HOURS from the previous match's end (Kalshi settlement time) to this start; thin markets (low volume);
  wide closing spreads.
Each rule: back the favourite / back the underdog, two time halves; --null N luck runs (current results re-drawn
from the price).

  .venv\Scripts\python.exe -m tools.scenarios [--null N]

Result (2026-11-11): 0 of 11 Pinnacle and 0 of 10 Kalshi scenarios held in both halves (luck runs mean 0.7 / 0.2).
Deciding-set tiebreaks, comebacks, full-distance or easy last matches, the bye effect, rest in hours, thin markets
and wide spreads are all priced. Thin ITF markets: backing the underdog lost 10-15% (avoid, as the board already does).
"""
import glob
import json
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from edge.names import fold
from tools.kalshi_backtest import fee


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def pin_rows():
    out = []
    for f in sorted(glob.glob("work/bookodds/*.csv")):
        D = pd.read_csv(f, encoding="latin-1", low_memory=False)
        D["date"] = pd.to_datetime(D["Date"], errors="coerce", format="mixed")
        D = D[D.date.notna()].copy()
        D["tour"] = "atp" if "atp_" in f else "wta"
        out.append(D)
    O = pd.concat(out, ignore_index=True)
    O["i"] = np.arange(len(O))
    O = O.sort_values(["date", "i"], kind="stable").reset_index(drop=True)
    last = {}                                    # player -> dict of last-match facts
    played_here = defaultdict(set)               # (tourney key) -> players who already played there
    rows = []
    for r in O.itertuples():
        tkey = (r.tour, r.Tournament, r.date.year)
        w, l = f"{r.tour}|{r.Winner}", f"{r.tour}|{r.Loser}"
        sets = []
        for k in range(1, 6):
            a, b = num(getattr(r, f"W{k}", None)), num(getattr(r, f"L{k}", None))
            if a is not None and b is not None:
                sets.append((a, b))
        completed = str(getattr(r, "Comment", "")) == "Completed" and sets
        psw, psl = num(r.PSW), num(r.PSL)
        if psw and psl and psw > 1 and psl > 1:
            fw, fl = 1 / psw, 1 / psl
            pw = fw / (fw + fl)
            if abs(pw - 0.5) > 1e-6:
                fav_w = pw >= 0.5
                F, Dg = (w, l) if fav_w else (l, w)
                lf, ld = last.get(F), last.get(Dg)
                ok = lambda x: x is not None and 0 < (r.date - x["date"]).days <= 14
                rows.append({"date": r.date, "y": int(fav_w), "p": max(pw, 1 - pw), "ps_f": psw if fav_w else psl, "ps_d": psl if fav_w else psw,
                             "f_last": lf if ok(lf) else None, "d_last": ld if ok(ld) else None,
                             "f_first_here": F not in played_here[tkey], "d_first_here": Dg not in played_here[tkey],
                             "round": r.Round})
        if completed:
            bo = 5 if len(sets) > 3 or (num(getattr(r, "Best of", 3)) or 3) == 5 else 3
            n_sets = len(sets)
            deciding = n_sets == (3 if bo == 3 else 5)
            dtb = deciding and max(sets[-1]) == 7 and min(sets[-1]) == 6
            comeback = sets[0][0] < sets[0][1]
            games_l = sum(b for _, b in sets)
            last[w] = {"date": r.date, "won": 1, "dtb": dtb, "comeback": comeback, "distance": deciding, "easy": games_l <= 4, "games_opp": games_l}
            last[l] = {"date": r.date, "won": 0, "dtb": dtb, "comeback": False, "distance": deciding, "easy": False, "games_opp": sum(a for a, _ in sets)}
        played_here[tkey].update((w, l))
    return pd.DataFrame(rows)


def pin_rules(D):
    g = lambda col, k: np.array([bool(x and x.get(k)) for x in D[col]])
    won = lambda col: np.array([bool(x and x["won"]) for x in D[col]])
    has = lambda col: np.array([x is not None for x in D[col]])
    return {
        "fav won a DECIDING-SET TIEBREAK last match": won("f_last") & g("f_last", "dtb"),
        "dog won a deciding-set tiebreak last match": won("d_last") & g("d_last", "dtb"),
        "fav came back from a set down last match": won("f_last") & g("f_last", "comeback"),
        "dog came back from a set down last match": won("d_last") & g("d_last", "comeback"),
        "fav's last match went the distance": won("f_last") & g("f_last", "distance"),
        "dog's last match went the distance": won("d_last") & g("d_last", "distance"),
        "fav's last win was easy (opp won <=4 games)": won("f_last") & g("f_last", "easy"),
        "dog's last win was easy (opp won <=4 games)": won("d_last") & g("d_last", "easy"),
        "fav drained (distance) AND dog easy last win": won("f_last") & g("f_last", "distance") & won("d_last") & g("d_last", "easy"),
        "BYE effect: fav's first match here, dog already won a round here": D.f_first_here.values & ~D.d_first_here.values & has("d_last") & won("d_last"),
        "reverse: dog's first match here, fav already won a round": D.d_first_here.values & ~D.f_first_here.values & has("f_last") & won("f_last"),
    }


def kalshi_rows():
    ev = []
    for l in open("work/kalshi_events.jsonl"):
        e = json.loads(l)
        q = e.get("quote")
        if not q or e.get("y") is None or not (0 < q["bid"] < q["ask"] < 1) or q["ask"] - q["bid"] > 0.15 or not e.get("end"):
            continue
        ev.append(e)
    ev.sort(key=lambda e: e["start"])
    last_end = {}
    rows = []
    for e in ev:
        tour = "wta" if ("WTA" in e["series"] or "ITFW" in e["series"]) else "atp"
        ka, kb = f"{tour}|{' '.join(sorted(fold(e['a'])))}", f"{tour}|{' '.join(sorted(fold(e['b'])))}"
        q = e["quote"]; px = (q["bid"] + q["ask"]) / 2; fa = px >= 0.5
        if abs(px - 0.5) < 1e-6:
            continue
        F, Dg = (ka, kb) if fa else (kb, ka)
        hr = lambda k: (e["start"] - last_end[k]) / 3600 if k in last_end and last_end[k] < e["start"] else None
        ask_f = q["ask"] if fa else 1 - q["bid"]; ask_d = 1 - q["bid"] if fa else q["ask"]
        cf, cd = ask_f + fee(ask_f), ask_d + fee(ask_d)
        fl = e.get("flow") or {}
        rows.append({"date": e["start"], "y": int((e["y"] == 1) == fa), "p": px if fa else 1 - px, "win_f": (1 - cf) / cf, "win_d": (1 - cd) / cd,
                     "rest_f": hr(F), "rest_d": hr(Dg), "vol": fl.get("vol"), "spread": q["ask"] - q["bid"], "itf": "ITF" in e["series"],
                     "ok_f": 0.05 <= ask_f <= 0.95, "ok_d": 0.05 <= ask_d <= 0.95})
        last_end[ka] = last_end[kb] = e["end"]
    return pd.DataFrame(rows)


def k_rules(K):
    rf, rd = K.rest_f, K.rest_d
    vq = K.vol.quantile([0.25, 0.75]).values
    return {
        "fav short turnaround (<20 h since last match ended), dog rested 40 h+": (rf < 20) & (rd >= 40),
        "dog short turnaround (<20 h), fav rested 40 h+": (rd < 20) & (rf >= 40),
        "fav played within 30 h (any dog)": rf < 30,
        "dog played within 30 h (any fav)": rd < 30,
        "both played within 30 h": (rf < 30) & (rd < 30),
        "thin market (bottom 25% volume)": K.vol <= vq[0],
        "busy market (top 25% volume)": K.vol >= vq[1],
        "wide closing spread (4c+)": K.spread >= 0.04,
        "thin market, tour/Challenger only": (K.vol <= vq[0]) & ~K.itf,
        "thin market, ITF only": (K.vol <= vq[0]) & K.itf,
    }


def score(D, rules, rf, rd, okf=None, okd=None, quiet=False):
    half = np.arange(len(D)) >= len(D) // 2
    passed = []
    for name, m in rules.items():
        m = np.asarray(pd.Series(m).fillna(False), bool)
        cells, b_ok, f_ok = [], True, True
        for h in (False, True):
            g = m & (half == h)
            gf = g & (okf if okf is not None else True); gd = g & (okd if okd is not None else True)
            if g.sum() < 80:
                cells.append(f"n={g.sum():>5} (few)"); b_ok = f_ok = False; continue
            cells.append(f"n={g.sum():>5} fav won {D.y.values[g].mean():5.1%} vs {D.p.values[g].mean():5.1%}  back fav {rf[gf].mean():+6.1%}  back dog {rd[gd].mean():+6.1%}")
            b_ok &= rf[gf].mean() > 0.01; f_ok &= rd[gd].mean() > 0.01
        if b_ok or f_ok:
            passed.append(name)
        if not quiet:
            print(f"   {name:<72} " + "   ".join(cells) + ("  <-- back fav held" if b_ok else "  <-- back dog held" if f_ok else ""))
    return passed


def main():
    nulls = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else 0
    P = pin_rows()
    print(f"===== Pinnacle closing odds: {len(P):,} matches")
    R = pin_rules(P)
    ret = lambda D, y: (np.where(y == 1, D.ps_f - 1, -1.0), np.where(y == 0, D.ps_d - 1, -1.0))
    rf, rd = ret(P, P.y.values)
    score(P, R, rf, rd)
    K = kalshi_rows()
    print(f"\n===== Kalshi 2025-26: {len(K):,} matches (all levels, ask + fee); rest known for fav in {K.rest_f.notna().mean():.0%}")
    RK = k_rules(K)
    kf = lambda K, y: (np.where(y == 1, K.win_f, -1.0), np.where(y == 0, K.win_d, -1.0))
    a, b = kf(K, K.y.values)
    score(K, RK, a, b, K.ok_f.values, K.ok_d.values)
    if nulls:
        rng = np.random.default_rng(1); cp, ck = [], []
        for _ in range(nulls):
            yp = (rng.random(len(P)) < P.p.values).astype(int); x, z = ret(P, yp); cp.append(len(score(P.assign(y=yp), R, x, z, quiet=True)))
            yk = (rng.random(len(K)) < K.p.values).astype(int); x, z = kf(K, yk); ck.append(len(score(K.assign(y=yk), RK, x, z, K.ok_f.values, K.ok_d.values, quiet=True)))
        print(f"\nluck runs - Pinnacle: {cp} (mean {np.mean(cp):.1f}); Kalshi: {ck} (mean {np.mean(ck):.1f})")


if __name__ == "__main__":
    main()
