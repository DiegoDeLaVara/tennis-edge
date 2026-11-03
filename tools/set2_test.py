r"""In-play set-2 markets, from the tennis-observer recording. At the first quote after set 1 ends:
  1. back the set-1 loser to win set 2 (all, and when the loser was the pre-match favourite);
  2. consistency: in best-of-3 the set-1 loser must win sets 2 and 3, so P(match) = P(set 2) x P(set 3 | won set 2).
     With sets roughly alike, P(set 2) ~ sqrt(P(match)). Bet the set-2 market when it strays from that.
Bets at the real ask (with Kalshi's taker fee); one bet per match.

  .venv\Scripts\python.exe -m tools.set2_test
"""
import argparse
import math
import sqlite3

import numpy as np


def fee(p):
    return 0.07 * p * (1 - p)


def roi(lst):
    a = np.array(lst)
    return f"n={len(a):4d} ROI {a.mean() * 100:+6.1f}% ± {a.std() / math.sqrt(len(a)) * 100:4.1f}" if len(a) else "n=   0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=r"C:\Users\vara7\Documents\tennis-observer\data\observer.sqlite")
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    res = dict(c.execute("select ticker, result from outcomes"))
    M = {r[0]: r for r in c.execute("select milestone_id, best_of, pre_p1, winner_side, s2_ticker1, s2_ticker2 from matches "
                                    "where run_id >= 3 and best_of = 3 and s2_ticker1 is not null and winner_side in (1, 2)")}
    print("best-of-3 matches with set-2 markets:", len(M))
    snap = {}
    q = ("select milestone_id, ts, sets1, sets2, games1, games2, bid1, ask1, s2b1, s2a1, s2b2, s2a2 from ticks "
         "where run_id >= 3 and phase = 'live' and sets1 + sets2 = 1 order by milestone_id, ts")
    for mid, ts, s1, s2, g1, g2, b1, a1, x1, y1, x2, y2 in c.execute(q):
        if mid not in M or mid in snap:
            continue
        if None in (b1, a1, x1, y1, x2, y2) or (g1 or 0) + (g2 or 0) > 1:
            continue
        if not (0 < b1 < a1 < 1) or a1 - b1 > 0.05 or not (0 <= x1 < y1 <= 1) or not (0 <= x2 < y2 <= 1):
            continue
        snap[mid] = (s1, s2, (b1 + a1) / 2, y1, y2, x1, x2)
    print("matches with a clean quote right after set 1:", len(snap))
    loser_all, loser_fav, loser_dog, gap = [], [], [], {0.03: [], 0.06: [], 0.10: []}
    for mid, (s1, s2, pm1, ask1, ask2, bid1, bid2) in snap.items():
        _, bo, pre, win, t1, t2 = M[mid]
        r1, r2 = res.get(t1), res.get(t2)
        if r1 not in ("yes", "no"):
            continue
        p1_won_set2 = r1 == "yes"
        loser = 1 if s2 == 1 else 2                         # player who lost set 1
        ask = ask1 if loser == 1 else ask2
        won = p1_won_set2 if loser == 1 else not p1_won_set2
        if 0.05 <= ask <= 0.95:
            cst = ask + fee(ask)
            r = (1 - cst) / cst if won else -1.0
            loser_all.append(r)
            fav = pre is not None and (pre >= 0.5) == (loser == 1)
            (loser_fav if fav else loser_dog).append(r)
        # consistency with the match market (player 1's view)
        pm = pm1 if loser == 1 else 1 - pm1                 # loser's match probability
        implied = math.sqrt(max(pm, 1e-4))                  # loser's set-2 probability if sets 2 and 3 are alike
        for side_is_loser in (True, False):
            p = implied if side_is_loser else 1 - implied
            a = (ask1 if loser == 1 else ask2) if side_is_loser else (ask2 if loser == 1 else ask1)
            w = won if side_is_loser else not won
            if not (0.05 <= a <= 0.95):
                continue
            ev = p / (a + fee(a)) - 1
            for th in gap:
                if ev >= th:
                    cst = a + fee(a)
                    gap[th].append((1 - cst) / cst if w else -1.0)
    print("\nback the set-1 loser to win set 2:          ", roi(loser_all))
    print("   ...when the loser was the pre-match favourite", roi(loser_fav))
    print("   ...when the loser was the underdog          ", roi(loser_dog))
    for th, L in gap.items():
        print(f"set-2 price off its match-market value by {int(th * 100):2d}%+:", roi(L))


if __name__ == "__main__":
    main()
