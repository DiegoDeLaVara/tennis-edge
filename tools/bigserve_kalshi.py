r"""Fresh check of the style_matchup survivor on Kalshi 2025-26: both players big servers -> back the underdog.
Result (2026-11-10): 1 SD: n=74 -5.9% +- 18%; 0.5 SD: +7.0% +- 8.8% (n=353); 1.5 SD: -38% (n=21). Not confirmed.

  .venv\Scripts\python.exe -m tools.bigserve_kalshi
"""
import json, math
from collections import defaultdict
from datetime import date
import numpy as np, pandas as pd
from edge.names import fold
from tools.style_matchup import Acc, num, SACK
from tools.h2h_detail import parse_score
from tools.kalshi_backtest import fee

P = pd.read_pickle("work/style_matchup.pkl")
Z = {}
for t in ("atp", "wta"):
    v = pd.concat([P.loc[P.tour == t, "ace_f"], P.loc[P.tour == t, "ace_d"]])
    Z[t] = (v.mean(), v.std())
rows = []
for tour in ("atp", "wta"):
    for y in range(2015, 2027):
        for suf in ("", "_qual_chall") if tour == "atp" else ("",):
            f = SACK / tour / f"{tour}_matches{suf}_{y}.csv"
            if f.exists():
                D = pd.read_csv(f, low_memory=False); D["tour"] = tour; rows.append(D)
S = pd.concat(rows, ignore_index=True)
S["d0"] = pd.to_datetime(S.tourney_date.astype(int).astype(str), format="%Y%m%d").dt.date
S = S.sort_values(["d0"], kind="stable")
name2id = {}
for r in S.itertuples():
    for nm, pid in ((r.winner_name, r.winner_id), (r.loser_name, r.loser_id)):
        name2id[(r.tour, " ".join(sorted(fold(str(nm)))))] = f"{r.tour}:{pid}"
ev = []
for l in open("work/kalshi_events.jsonl"):
    e = json.loads(l)
    if e["series"] not in ("KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH") or not e.get("quote") or e.get("y") is None or e.get("round") == "Q":
        continue
    q = e["quote"]
    if q["ask"] - q["bid"] > 0.12 or not 0 < q["bid"] < q["ask"] < 1:
        continue
    ev.append(e)
ev.sort(key=lambda e: e["start"])
A = defaultdict(Acc)
srows = list(S.itertuples()); i = 0; out = []
for e in ev:
    d = date.fromtimestamp(e["start"])
    while i < len(srows) and srows[i].d0 < d:
        r = srows[i]; i += 1
        wsv, lsv = num(r.w_svpt), num(r.l_svpt)
        if wsv and lsv and parse_score(r.score) is not None:
            ww = (num(r.w_1stWon) or 0) + (num(r.w_2ndWon) or 0); lw = (num(r.l_1stWon) or 0) + (num(r.l_2ndWon) or 0)
            A[f"{r.tour}:{r.winner_id}"].add(r.d0, wsv, ww, num(r.w_ace) or 0, num(r.w_df) or 0, lsv, lsv - lw)
            A[f"{r.tour}:{r.loser_id}"].add(r.d0, lsv, lw, num(r.l_ace) or 0, num(r.l_df) or 0, wsv, wsv - ww)
    tour = "wta" if e["series"] == "KXWTAMATCH" else "atp"
    ia, ib = name2id.get((tour, " ".join(sorted(fold(e["a"]))))), name2id.get((tour, " ".join(sorted(fold(e["b"])))))
    if not ia or not ib:
        continue
    pa, pb = A[ia].snap(d), A[ib].snap(d)
    if not pa or not pb or pa["n"] < 300 or pb["n"] < 300:
        continue
    q = e["quote"]; px = (q["bid"] + q["ask"]) / 2; fa = px >= 0.5
    m, s = Z[tour]
    za, zb = (pa["ace"] - m) / s, (pb["ace"] - m) / s
    ask_d = (1 - q["bid"]) if fa else q["ask"]
    dog_won = (e["y"] == 0) if fa else (e["y"] == 1)
    c = ask_d + fee(ask_d)
    out.append({"date": d, "series": e["series"], "za": za, "zb": zb, "pd": 1 - px if fa else px, "ask_d": ask_d, "dog_won": int(dog_won),
                "r": (1 - c) / c if dog_won else -1.0, "last_stats": srows[i - 1].d0 if i else None})
K = pd.DataFrame(out)
print(f"Kalshi matches with style profiles for both: {len(K)} ({K.date.min()} .. {K.date.max()}); profiles use archive stats up to {S.d0.max()}")
for th in (0.5, 1.0, 1.5):
    m = (K.za >= th) & (K.zb >= th) & K.ask_d.between(0.05, 0.95)
    g = K[m]
    if len(g) >= 10:
        print(f"  both ace z>={th}: n={len(g)} dog won {g.dog_won.mean():.1%} vs {g.pd.mean():.1%} priced, back dog at ask+fee {g.r.mean():+.1%} ± {g.r.std(ddof=1)/math.sqrt(len(g)):.1%}")
        for ser, gg in g.groupby("series"):
            print(f"      {ser}: n={len(gg)} dog won {gg.dog_won.mean():.1%} vs {gg.pd.mean():.1%} ROI {gg.r.mean():+.1%}")
base = K[K.ask_d.between(0.05, 0.95)]
print(f"  all Kalshi underdogs (baseline): n={len(base)} ROI {base.r.mean():+.1%}")
