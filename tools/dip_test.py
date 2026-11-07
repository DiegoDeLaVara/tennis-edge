r"""Buy the dip: when a player's live Kalshi price falls well below their pre-match price, buy, then either hold to the
end or sell into a bounce. Every variant at real prices: buy at the ask, sell at the bid, Kalshi taker fee both ways.

Data: the tennis-observer recording (every order-book tick of ~900 live matches, 22-29 Sep 2026, with the score).
Triggers: price drop of D cents below pre-match, or score events (favourite loses set 1 / goes a break down in set 1).
Exits: hold to settlement, take profit at +T, take profit +T / stop loss -S.
Filters: who dipped (pre-match favourite, big favourite, underdog, anyone), level (tour, Challenger, ITF).

Each cell is scored in two halves (matches ordered by start). The grid is large, so a few cells will look good by
luck; the honest test is --validate on fresh matches (Kalshi 1-minute candles from other dates).

  .venv\Scripts\python.exe -m tools.dip_test [--db PATH]
"""
import argparse
import itertools
import math
import sqlite3
from collections import defaultdict

import numpy as np

from edge.names import fold

DB = "C:/Users/vara7/Documents/tennis-observer/data/observer.sqlite"
DROPS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
EXITS = [("hold", None, None), ("TP+5", 0.05, None), ("TP+10", 0.10, None), ("TP+15", 0.15, None), ("TP+20", 0.20, None),
         ("TP+10/SL-10", 0.10, 0.10), ("TP+15/SL-15", 0.15, 0.15)]
WHO = {"anyone": lambda p0, pm: True, "pre-match fav (55%+)": lambda p0, pm: p0 >= 0.55, "big fav (75%+)": lambda p0, pm: p0 >= 0.75,
       "underdog (<45%)": lambda p0, pm: p0 < 0.45,
       # players WE thought would win (pm = model's pre-match chance; observer model, or our current model if known)
       "model fav (60%+)": lambda p0, pm: pm is not None and pm >= 0.60,
       "model fav (70%+)": lambda p0, pm: pm is not None and pm >= 0.70,
       "model 5+ pts above market": lambda p0, pm: pm is not None and pm - p0 >= 0.05,
       "model 10+ pts above market": lambda p0, pm: pm is not None and pm - p0 >= 0.10,
       "model AND market fav (60%+/55%+)": lambda p0, pm: pm is not None and pm >= 0.60 and p0 >= 0.55}


def fee(p):
    return 0.07 * p * (1 - p)


def our_model():
    """event -> (our model's chance for side 'a', a's name), from the Kalshi backtest set (tools.close_matches)."""
    import contextlib, io, json
    from tools.close_matches import data
    with contextlib.redirect_stdout(io.StringIO()):
        D = data()
    names = {}
    for l in open("work/kalshi_events.jsonl"):
        r = json.loads(l); names[r["event"]] = r["a"]
    return {e: (p, names[e]) for e, p in zip(D.event, D.p_model) if e in names and p == p}


def load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    M = {}
    ours = our_model()
    for mid, grp, pre, w, st, mp, ev, n1 in c.execute("select milestone_id, grp, pre_p1, winner_side, start_ts, model_pre_p1, event_ticker, name1 "
                                                       "from matches where winner_side in (1, 2)"):
        pm = mp
        if ev in ours:                                  # prefer our current model when it priced this event
            pa, a_name = ours[ev]
            pm = pa if fold(a_name) == fold(n1) else 1 - pa
        M[mid] = {"grp": grp, "pre": pre, "w": w, "start": st, "pm": pm, "ticks": []}
    pre_last = {}
    for mid, b1, a1 in c.execute("select milestone_id, bid1, ask1 from ticks where phase = 'pre' order by ts"):
        if mid in M and b1 and a1 and 0 < b1 < a1 < 1 and a1 - b1 <= 0.06:
            pre_last[mid] = (b1 + a1) / 2
    q = ("select milestone_id, ts, sets1, sets2, games1, games2, server, bid1, ask1, bid2, ask2 from ticks "
         "where phase = 'live' order by milestone_id, ts")
    for r in c.execute(q):
        if r[0] in M:
            M[r[0]]["ticks"].append(r[1:])
    out = []
    for mid, m in M.items():
        p0 = m["pre"] if m["pre"] else pre_last.get(mid)
        if p0 is None or len(m["ticks"]) < 20 or not 0.03 < p0 < 0.97:
            continue
        m["p0"] = p0
        out.append(m)
    out.sort(key=lambda m: m["start"] or 0)
    return out


def side_path(m, s):
    """(ts, bid, ask, my_sets, opp_sets, my_games, opp_games) for side s (1 or 2) using that side's own book."""
    P = []
    for ts, s1, s2, g1, g2, srv, b1, a1, b2, a2 in m["ticks"]:
        b, a = (b1, a1) if s == 1 else (b2, a2)
        if b is None or a is None or not (0 <= b < a <= 1) or a - b > 0.15:
            continue
        P.append((ts, b, a, (s1, s2)[s - 1] or 0, (s2, s1)[s - 1] or 0, (g1, g2)[s - 1] or 0, (g2, g1)[s - 1] or 0))
    return P


def trade(P, i, won, tp, sl):
    """Buy at P[i] ask; exit by rule; return per $ staked."""
    a = P[i][2]
    cost = a + fee(a)
    if tp is not None or sl is not None:
        for t in P[i + 1:]:
            b = t[1]
            if (tp is not None and b >= a + tp) or (sl is not None and b <= a - sl):
                return (b - fee(b)) / cost - 1
    return (1.0 if won else 0.0) / cost - 1


def triggers(P, p0):
    """Index of the first tick that fires each trigger (or None)."""
    T = {}
    for d in DROPS:
        T[f"price {int(d * 100)}c below pre-match"] = next((i for i, t in enumerate(P) if t[2] <= p0 - d), None)
    T["lost set 1 (buy at first tick of set 2)"] = next((i for i, t in enumerate(P) if t[4] == 1 and t[3] == 0), None)
    T["down a break in set 1 (2+ games behind)"] = next((i for i, t in enumerate(P) if t[3] == 0 and t[4] == 0 and t[6] - t[5] >= 2), None)
    T["down a set and a break"] = next((i for i, t in enumerate(P) if t[4] - t[3] >= 1 and t[6] - t[5] >= 2), None)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()
    ms = load(args.db)
    print(f"{len(ms)} recorded live matches with a pre-match price and a result")
    half = {id(m): int(i >= len(ms) // 2) for i, m in enumerate(ms)}
    R = defaultdict(list)                      # (trigger, who, level, exit) -> [(half, ret)]
    base = defaultdict(list)
    for m in ms:
        for s in (1, 2):
            P = side_path(m, s)
            if len(P) < 10:
                continue
            p0 = m["p0"] if s == 1 else 1 - m["p0"]
            pm = None if m["pm"] is None else (m["pm"] if s == 1 else 1 - m["pm"])
            won = m["w"] == s
            lvl = {"main": "tour", "challenger": "challenger", "itf": "itf"}.get(m["grp"], "other")
            # random-entry baseline: buy at a random live tick, same exits
            j = (hash((id(m), s)) % len(P))
            for en, tp, sl in EXITS:
                base[en].append(trade(P, j, won, tp, sl))
            for tn, i in triggers(P, p0).items():
                if i is None or i >= len(P) - 1:
                    continue
                for wn, wf in WHO.items():
                    if not wf(p0, pm):
                        continue
                    for lv in ("all", lvl):
                        for en, tp, sl in EXITS:
                            R[(tn, wn, lv, en)].append((half[id(m)], trade(P, i, won, tp, sl)))
    print("\nBaseline (buy at a random live moment, same exits):  " +
          "  ".join(f"{en} {np.mean(v):+.1%}" for en, v in base.items()))
    rows = []
    for k, v in R.items():
        h = np.array([x[0] for x in v]); r = np.array([x[1] for x in v])
        if (h == 0).sum() < 25 or (h == 1).sum() < 25:
            continue
        r0, r1 = r[h == 0].mean(), r[h == 1].mean()
        rows.append((k, len(r), r.mean(), r.std(ddof=1) / math.sqrt(len(r)), r0, r1, (h == 0).sum(), (h == 1).sum()))
    both = [x for x in rows if x[4] > 0.02 and x[5] > 0.02]
    print(f"\n{len(rows)} strategy cells with 25+ trades in each half; {len(both)} made 2%+ in BOTH halves")
    print("\nAll cells for the broadest version (anyone who dips, all levels):")
    for k, n, mu, se, r0, r1, n0, n1 in sorted([x for x in rows if x[0][1] == "anyone" and x[0][2] == "all"], key=lambda x: (x[0][0], x[0][3])):
        print(f"   {k[0]:<42} {k[3]:<12} n={n:>4}  {mu:+6.1%} ± {se:4.1%}   halves {r0:+6.1%} / {r1:+6.1%}")
    print("\nPlayers the MODEL backed, all levels (hold = keep to the end):")
    for k, n, mu, se, r0, r1, n0, n1 in sorted([x for x in rows if x[0][1].startswith("model") and x[0][2] == "all" and x[0][3] in ("hold", "TP+10")],
                                               key=lambda x: (x[0][1], x[0][0], x[0][3])):
        print(f"   {k[1]:<34} {k[0]:<42} {k[3]:<6} n={n:>4}  {mu:+6.1%} ± {se:4.1%}   halves {r0:+6.1%} / {r1:+6.1%}")
    print("\nCells that made 2%+ in both halves (candidates, NOT proven - check with fresh data):")
    for k, n, mu, se, r0, r1, n0, n1 in sorted(both, key=lambda x: -x[2] / x[3])[:40]:
        print(f"   {k[0]:<42} {k[1]:<22} {k[2]:<10} {k[3]:<12} n={n:>4} ({n0}/{n1})  {mu:+6.1%} ± {se:4.1%}   halves {r0:+6.1%} / {r1:+6.1%}")


if __name__ == "__main__":
    main()
