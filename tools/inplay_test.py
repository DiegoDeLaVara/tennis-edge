"""In-play: does Kalshi's live price drift from what the score says? Uses the tennis-observer recording (live ticks with
score, server and book). At each game boundary the fair probability is the exact scoring model from the current score,
with serve-point rates solved from the pre-match market price. Bets at the real ask; one bet per match per set.

  .venv\\Scripts\\python.exe -m tools.inplay_test --db C:\\Users\\vara7\\Documents\\tennis-observer\\data\\observer.sqlite
"""
import argparse
import sqlite3
from functools import lru_cache

import numpy as np

from edge import markov

FEE_FREE_SPREAD = 0.04


def fee(p):
    return 0.07 * p * (1 - p)


@lru_cache(maxsize=None)
def set_from(pa, pb, ga, gb, a_srv):
    if (ga >= 6 and ga - gb >= 2) or ga == 7:
        return 1.0
    if (gb >= 6 and gb - ga >= 2) or gb == 7:
        return 0.0
    if ga == 6 and gb == 6:
        return markov._tb(0, 0, pa, pb, a_srv)
    w = markov.hold(pa) if a_srv else 1 - markov.hold(pb)
    return w * set_from(pa, pb, ga + 1, gb, not a_srv) + (1 - w) * set_from(pa, pb, ga, gb + 1, not a_srv)


def match_from(s, bo, sa, sb):
    need = bo // 2 + 1
    if sa >= need:
        return 1.0
    if sb >= need:
        return 0.0
    return s * match_from(s, bo, sa + 1, sb) + (1 - s) * match_from(s, bo, sa, sb + 1)


@lru_cache(maxsize=None)
def live_p(pa, pb, bo, sa, sb, ga, gb, a_srv):
    """P(A wins) at the start of a game (0-0 points), A serving if a_srv."""
    s_future = (set_from(pa, pb, 0, 0, True) + set_from(pa, pb, 0, 0, False)) / 2
    ps = set_from(pa, pb, ga, gb, a_srv)
    return ps * match_from(s_future, bo, sa + 1, sb) + (1 - ps) * match_from(s_future, bo, sa, sb + 1)


def live_p_mix(pa, pb, bo, sa, sb, ga, gb, a_srv, sd, prior_games):
    """Day-form mixture with node weights updated by how the games so far went (crude Bayesian update)."""
    tot = 0.0
    acc = 0.0
    for a, b, w in markov._nodes(pa, pb, sd):
        ha, hb = markov.hold(a), markov.hold(b)
        # likelihood of A having won `prior_games[0]` of the games and B `prior_games[1]` under this node
        ga_, gb_ = prior_games
        n = ga_ + gb_
        pg = (ha + (1 - hb)) / 2            # A's chance of winning a random game (half on serve)
        lik = pg ** ga_ * (1 - pg) ** gb_ if n else 1.0
        acc += w * lik * live_p(a, b, bo, sa, sb, ga, gb, a_srv)
        tot += w * lik
    return acc / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=r"C:\Users\vara7\Documents\tennis-observer\data\observer.sqlite")
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    M = {r[0]: r for r in c.execute("select milestone_id, tour, best_of, pre_p1, winner_side, grp from matches "
                                    "where run_id >= 3 and pre_p1 is not null and winner_side in (1, 2)")}
    print("matches with pre-match price and a winner:", len(M))
    rows = []
    q = ("select milestone_id, ts, sets1, sets2, games1, games2, server, in_tb, pts1, pts2, bid1, ask1, bid2, ask2, quote_age "
         "from ticks where run_id >= 3 and phase = 'live' and feed_changed = 1 order by milestone_id, ts")
    last_state = {}
    for (mid, ts, s1, s2, g1, g2, srv, tb, p1, p2, b1, a1, b2, a2, age) in c.execute(q):
        m = M.get(mid)
        if not m or tb or srv not in (1, 2) or (p1 or 0) != 0 or (p2 or 0) != 0:
            continue
        need = int(m[2] or 3) // 2 + 1
        if None in (s1, s2, g1, g2) or not (0 <= g1 <= 6 and 0 <= g2 <= 6) or (g1 == 6 and g2 == 6)                 or abs(g1 - g2) > 5 or (max(g1, g2) == 6 and abs(g1 - g2) >= 2) or s1 >= need or s2 >= need:
            continue
        if None in (b1, a1, b2, a2) or not (0 < b1 < a1 < 1) or a1 - b1 > FEE_FREE_SPREAD or (age is not None and age > 15):
            continue
        state = (s1, s2, g1, g2)
        if last_state.get(mid) == state:
            continue
        last_state[mid] = state
        rows.append((mid, ts, s1, s2, g1, g2, srv, b1, a1, b2, a2))
    print("game-boundary snapshots:", len(rows))

    out = []
    for (mid, ts, s1, s2, g1, g2, srv, b1, a1, b2, a2) in rows:
        _, tour, bo, pre, win, grp = M[mid]
        tour = "wta" if str(tour).upper() == "WTA" else "atp"
        bo = int(bo or 3)
        total = 1.25 if tour == "atp" else 1.12
        sd = markov.DAY_SD[tour]
        pa, pb = markov.solve_serve_mix(min(max(pre, 0.03), 0.97), total, bo, sd)
        p_iid = live_p(round(pa, 3), round(pb, 3), bo, s1, s2, g1, g2, srv == 1)
        p_mix = live_p_mix(pa, pb, bo, s1, s2, g1, g2, srv == 1, sd, (g1, g2))
        out.append({"mid": mid, "ts": ts, "set": s1 + s2, "y": 1 if win == 1 else 0, "mkt": (b1 + a1) / 2,
                    "ask1": a1, "ask2": a2, "iid": p_iid, "mix": p_mix, "pre": pre, "grp": grp,
                    "games": g1 + g2 + 12 * (s1 + s2)})
    y = np.array([x["y"] for x in out])

    def ll(p):
        p = np.clip(p, 1e-3, 1 - 1e-3)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"log-loss at game boundaries: market {ll(np.array([x['mkt'] for x in out])):.4f}  "
          f"score model (iid) {ll(np.array([x['iid'] for x in out])):.4f}  score model (day-form) {ll(np.array([x['mix'] for x in out])):.4f}")
    mids = sorted({x["mid"] for x in out})
    half = set(mids[: len(mids) // 2])
    for col in ("iid", "mix"):
        for th in (0.03, 0.06, 0.10):
            for part, keep in (("first half", lambda x: x["mid"] in half), ("second half", lambda x: x["mid"] not in half)):
                taken = set()
                res = []
                for x in out:
                    if not keep(x):
                        continue
                    for side in (0, 1):
                        p = x[col] if side == 0 else 1 - x[col]
                        price = x["ask1"] if side == 0 else x["ask2"]
                        if not (0.05 <= price <= 0.95) or p / (price + fee(price)) - 1 < th:
                            continue
                        k = (x["mid"], x["set"], side)
                        if k in taken:
                            continue
                        taken.add(k)
                        cst = price + fee(price)
                        won = (x["y"] == 1) if side == 0 else (x["y"] == 0)
                        res.append((1 - cst) / cst if won else -1.0)
                if res:
                    a = np.array(res)
                    print(f"  back {col:3s} model, edge >= {int(th * 100):2d}%, {part:11s}: n={len(a):5d} ROI {a.mean() * 100:+6.1f}% ± {a.std() / np.sqrt(len(a)) * 100:4.1f}")
    # simple momentum check without a model: after the market moves 10c+ between two game boundaries, does it revert?
    by = {}
    for x in out:
        by.setdefault(x["mid"], []).append(x)
    rev = []
    for mid, L in by.items():
        for i in range(1, len(L)):
            mv = L[i]["mkt"] - L[i - 1]["mkt"]
            if abs(mv) >= 0.10:
                side = 1 if mv > 0 else 0          # fade the move: buy the side that just got cheaper
                price = L[i]["ask1"] if side == 0 else L[i]["ask2"]
                if 0.05 <= price <= 0.95:
                    cst = price + fee(price)
                    won = (L[i]["y"] == 1) if side == 0 else (L[i]["y"] == 0)
                    rev.append((1 - cst) / cst if won else -1.0)
    if rev:
        a = np.array(rev)
        print(f"  fade any 10c+ move between games: n={len(a)} ROI {a.mean() * 100:+.1f}% ± {a.std() / np.sqrt(len(a)) * 100:.1f}")
    for thr in (0.10, 0.20):
        fol = []
        for mid, L in by.items():
            for i in range(1, len(L)):
                mv = L[i]["mkt"] - L[i - 1]["mkt"]
                if abs(mv) >= thr:
                    side = 0 if mv > 0 else 1      # follow the move: buy the side that just got more expensive
                    price = L[i]["ask1"] if side == 0 else L[i]["ask2"]
                    if 0.05 <= price <= 0.95:
                        cst = price + fee(price)
                        won = (L[i]["y"] == 1) if side == 0 else (L[i]["y"] == 0)
                        fol.append((1 - cst) / cst if won else -1.0)
        if fol:
            a = np.array(fol)
            print(f"  follow any {int(thr * 100)}c+ move between games: n={len(a)} ROI {a.mean() * 100:+.1f}% ± {a.std() / np.sqrt(len(a)) * 100:.1f}")


if __name__ == "__main__":
    main()
