r"""Fresh check of the two themes the lock70 search found, on Kalshi 2025-26 (real ask + fee), 70-85% favourites:
  A. the favourite won 41% or fewer of the games in their last meeting with this opponent (still priced 70-85%)
  B. the underdog lost a FINAL at its previous tournament (within 10 days) and this is its first match since
History (meetings, finals) from the Sackmann archive 2015-26 (tour + Challenger/qualifying; ends mid-2026).

  .venv\Scripts\python.exe -m tools.lock70_kalshi

Result (2026-11-10): A n=128 won 71.1% vs 75.3% priced, -8.3% (post-archive leak-proof n=75: -6.4%). B: no cases.
LEAK NOTE: Sackmann dates matches by tournament start - the first run counted the match being predicted as the
"last meeting" (54.6% vs 75.3%). Always exclude the last ~16 days of archive meetings.
"""
import json
import math
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from edge.names import fold
from tools.h2h_detail import parse_score
from tools.kalshi_backtest import fee
from tools.style_matchup import SACK


def main():
    rows = []
    for tour in ("atp", "wta"):
        for y in range(2015, 2027):
            for suf in (("", "_qual_chall") if tour == "atp" else ("", "_qual_itf")):
                f = SACK / tour / f"{tour}_matches{suf}_{y}.csv"
                if f.exists():
                    D = pd.read_csv(f, low_memory=False, usecols=["tourney_id", "tourney_date", "winner_name", "loser_name", "score", "round"])
                    D["tour"] = tour; rows.append(D)
    S = pd.concat(rows, ignore_index=True)
    S["d0"] = pd.to_datetime(S.tourney_date.astype(int).astype(str), format="%Y%m%d").dt.date
    S = S.sort_values("d0", kind="stable")
    nk = lambda t, n: (t, " ".join(sorted(fold(str(n)))))
    meets = defaultdict(list)            # pair -> [(d0, games_a_share for first key)]
    last_t = {}                          # player -> list of (d0, tourney_id, round, won)
    hist = defaultdict(list)
    for r in S.itertuples():
        w, l = nk(r.tour, r.winner_name), nk(r.tour, r.loser_name)
        sc = parse_score(r.score)
        if sc:
            gw, gl = sum(a for a, _, _ in sc), sum(b for _, b, _ in sc)
            meets[frozenset((w, l))].append((r.d0, w, gw / max(gw + gl, 1)))
        hist[w].append((r.d0, r.tourney_id, r.round, 1)); hist[l].append((r.d0, r.tourney_id, r.round, 0))
    end = S.d0.max()
    out = []
    for l in open("work/kalshi_events.jsonl"):
        e = json.loads(l)
        if e["series"] not in ("KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH") or not e.get("quote") or e.get("y") is None:
            continue
        q = e["quote"]
        if q["ask"] - q["bid"] > 0.12 or not 0 < q["bid"] < q["ask"] < 1:
            continue
        px = (q["bid"] + q["ask"]) / 2; fa = px >= 0.5; pf = px if fa else 1 - px
        if not 0.70 <= pf < 0.85:
            continue
        tour = "wta" if e["series"] == "KXWTAMATCH" else "atp"
        d = date.fromtimestamp(e["start"])
        F, Dg = (nk(tour, e["a"]), nk(tour, e["b"])) if fa else (nk(tour, e["b"]), nk(tour, e["a"]))
        # the archive dates matches by TOURNAMENT START: skip anything from the last 16 days so the match being predicted
        # (or a later round of it) can never count as "the last meeting"
        prev = [m for m in meets.get(frozenset((F, Dg)), []) if m[0] < d - timedelta(16)]
        A = None
        if prev:
            d0, k, sh = prev[-1]
            A = (sh if k == F else 1 - sh) <= 0.407
        hd = [h for h in hist.get(Dg, []) if h[0] < d - timedelta(5)]
        Bf = None
        if hd and d - hd[-1][0] <= timedelta(17) and not any(d - timedelta(5) <= h[0] < d for h in hist.get(Dg, [])):
            lt = hd[-1]
            Bf = (lt[2] == "F" and lt[3] == 0)
        ask_f = q["ask"] if fa else 1 - q["bid"]
        c = ask_f + fee(ask_f); won = (e["y"] == 1) == fa
        out.append({"date": d, "series": e["series"], "pf": pf, "won": int(won), "r": (1 - c) / c if won else -1.0, "A": A, "B": Bf,
                    "in_archive": d <= end + timedelta(10)})
    K = pd.DataFrame(out)
    post = K[K.date > end]
    g = post[post.A == True]  # noqa: E712
    print(f"   leak-proof check (matches after the archive ends, {len(post)}): theme A n={len(g)}"
          + (f" won {g.won.mean():.1%} vs {g.pf.mean():.1%} priced, ROI {g.r.mean():+.1%}" if len(g) else ""))
    print(f"Kalshi 70-85% favourites: {len(K)} ({K.date.min()} .. {K.date.max()}); archive history ends {end}")
    print(f"   all of them: won {K.won.mean():.1%} vs {K.pf.mean():.1%} priced, ROI {K.r.mean():+.1%}")
    for name, col in (("A. fav won <=41% of games in their last meeting", "A"), ("B. dog lost a final at its last event", "B")):
        g = K[K[col] == True]  # noqa: E712
        h = K[K[col].notna()]
        if len(g):
            print(f"   {name}: n={len(g)} won {g.won.mean():.1%} vs {g.pf.mean():.1%} priced, ROI {g.r.mean():+.1%} ± {g.r.std(ddof=1) / math.sqrt(max(len(g), 2)):.1%}"
                  f"   (rule could be checked on {len(h)}; the rest {h[h[col] == False].won.mean():.1%} vs {h[h[col] == False].pf.mean():.1%})")  # noqa: E712
        else:
            print(f"   {name}: no cases")


if __name__ == "__main__":
    main()
