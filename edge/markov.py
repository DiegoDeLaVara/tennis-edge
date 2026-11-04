"""Exact tennis scoring maths from two serve-point probabilities.

pa = P(A wins a point on A's serve), pb = P(B wins a point on B's serve). Everything below is exact under the
usual i.i.d.-points assumption; the final set is played with an ordinary tiebreak (Slams use a 10-point one,
a small difference). Gives match odds, set scores, total games, game spreads, set winners and tiebreaks, all
from one joint distribution so combos price correctly.
"""
from functools import lru_cache


def hold(p):
    """P(server wins a standard game)."""
    q = 1 - p
    deuce = p * p / (p * p + q * q)
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * deuce


@lru_cache(maxsize=200000)
def _tb(a, b, pa, pb, a_serves_first):
    """P(A wins tiebreak from a-b points). Point n (0-based) is served by the first server if n%4 in (0,3)."""
    if a >= 7 and a - b >= 2:
        return 1.0
    if b >= 7 and b - a >= 2:
        return 0.0
    if a >= 6 and b >= 6 and a == b:
        # from level beyond 6-6 the next two points are one on each serve
        w = pa * (1 - pb)          # A wins both
        l = (1 - pa) * pb          # B wins both
        return w / (w + l) if w + l > 0 else 0.5
    n = a + b
    first_serves = n % 4 in (0, 3)
    a_serving = first_serves == a_serves_first
    p = pa if a_serving else 1 - pb
    return p * _tb(a + 1, b, pa, pb, a_serves_first) + (1 - p) * _tb(a, b + 1, pa, pb, a_serves_first)


@lru_cache(maxsize=20000)
def set_dist(pa, pb, a_first):
    """Distribution of one set: {(a_games, b_games): prob} (7-6 / 6-7 are tiebreak sets)."""
    ha, hb = hold(pa), hold(pb)
    out = {}
    st = {(0, 0): 1.0}
    while st:
        nxt = {}
        for (ga, gb), pr in st.items():
            if ga == 6 and gb == 6:
                tb = _tb(0, 0, pa, pb, a_first)          # 13th game: first server of the set serves first
                out[(7, 6)] = out.get((7, 6), 0) + pr * tb
                out[(6, 7)] = out.get((6, 7), 0) + pr * (1 - tb)
                continue
            a_serving = ((ga + gb) % 2 == 0) == a_first
            pw = ha if a_serving else 1 - hb
            for (na, nb), pp in (((ga + 1, gb), pw), ((ga, gb + 1), 1 - pw)):
                if (na >= 6 and na - nb >= 2) or (nb >= 6 and nb - na >= 2) or na == 7 or nb == 7:
                    out[(na, nb)] = out.get((na, nb), 0) + pr * pp
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0) + pr * pp
        st = nxt
    return out


def _r(x):
    return round(min(max(x, 0.02), 0.98), 3)


@lru_cache(maxsize=4000)
def match_dist(pa, pb, best_of=3):
    """Joint distribution over final states. Key: (sets_a, sets_b, total_games, a_minus_b_games, set1_a, set2_a, any_tb)."""
    pa, pb = _r(pa), _r(pb)
    need = best_of // 2 + 1
    final = {}
    for a_first0 in (True, False):
        st = {(0, 0, a_first0, 0, 0, -1, -1, 0): 0.5}
        while st:
            nxt = {}
            for (sa, sb, af, tot, diff, s1, s2, tb), pr in st.items():
                for (ga, gb), pp in set_dist(pa, pb, af).items():
                    a_won = ga > gb
                    n_sa, n_sb = sa + a_won, sb + (not a_won)
                    k = sa + sb
                    n_s1 = int(a_won) if k == 0 else s1
                    n_s2 = int(a_won) if k == 1 else s2
                    games = ga + gb
                    n_af = af if (games if games < 13 else 13) % 2 == 0 else not af
                    key_ = (n_sa, n_sb, n_af, tot + games, diff + ga - gb, n_s1, n_s2, tb or int(games == 13))
                    if n_sa == need or n_sb == need:
                        fk = (n_sa, n_sb, tot + games, diff + ga - gb, n_s1, n_s2, key_[7])
                        final[fk] = final.get(fk, 0) + pr * pp
                    else:
                        nxt[key_] = nxt.get(key_, 0) + pr * pp
            st = nxt
    return final


def p_match(pa, pb, best_of=3):
    return sum(v for k, v in match_dist(pa, pb, best_of).items() if k[0] > k[1])


def prob(dist, pred):
    return sum(v for k, v in dist.items() if pred(*k))


def fast_p_match(pa, pb, best_of=3):
    """Match win prob from the average set win prob (ignores who serves first in later sets; within ~0.2%)."""
    s = sum(v for (ga, gb), v in set_dist(round(pa, 4), round(pb, 4), True).items() if ga > gb)
    s = (s + sum(v for (ga, gb), v in set_dist(round(pa, 4), round(pb, 4), False).items() if ga > gb)) / 2
    if best_of == 5:
        return s ** 3 * (1 + 3 * (1 - s) + 6 * (1 - s) ** 2)
    return s * s * (3 - 2 * s)


def solve_serve(p_win, total, best_of=3):
    """Serve-point probs (pa, pb) with pa + pb = total that give A a match win probability of p_win."""
    p_win = min(max(p_win, 0.005), 0.995)
    lo, hi = -0.4, 0.4
    for _ in range(24):
        d = (lo + hi) / 2
        if fast_p_match(total / 2 + d / 2, total / 2 - d / 2, best_of) < p_win:
            lo = d
        else:
            hi = d
    d = (lo + hi) / 2
    return _r(total / 2 + d / 2), _r(total / 2 - d / 2)


def summary(pa, pb, best_of=3):
    """Everything the board shows for one match, from A's point of view."""
    D = match_dist(pa, pb, best_of)
    need = best_of // 2 + 1
    tot = {}
    for (sa, sb, t, d, s1, s2, tb), v in D.items():
        tot[t] = tot.get(t, 0) + v
    mean_games = sum(t * v for t, v in tot.items())
    return {
        "p": prob(D, lambda sa, sb, *_: sa > sb),
        "sets": {f"{sa}-{sb}": round(v, 4) for (sa, sb), v in _agg(D, lambda k: (k[0], k[1])).items()},
        "straight_a": prob(D, lambda sa, sb, *_: sa == need and sb == 0),
        "straight_b": prob(D, lambda sa, sb, *_: sb == need and sa == 0),
        "set1_a": prob(D, lambda *k: k[4] == 1),
        "set2_a": prob(D, lambda *k: k[5] == 1),
        "tiebreak": prob(D, lambda *k: k[6] == 1),
        "games_mean": mean_games,
        "games": {t: round(v, 5) for t, v in sorted(tot.items())},
        "hold_a": hold(pa), "hold_b": hold(pb),
    }


def _agg(D, f):
    out = {}
    for k, v in D.items():
        kk = f(k)
        out[kk] = out.get(kk, 0) + v
    return out


# ---------------------------------------------------------------- day-form mixture
# Real sets are correlated: a player has a good or bad day. Model it as a random shift of the serve-point gap,
# d ~ Normal(d0, sd), averaged with Gauss-Hermite nodes. Without it the maths over-predicts three-setters
# (45% vs 35% observed) and total games (+1.7 games).
_GH = [(-2.651961356835233, 0.000971781245099519), (-1.673551628767471, 0.05451558281912703), (-0.8162878828589647, 0.4256072526101278),
       (0.0, 0.8102646175568073), (0.8162878828589647, 0.4256072526101278), (1.673551628767471, 0.05451558281912703),
       (2.651961356835233, 0.000971781245099519)]
_GHW = sum(w for _, w in _GH)
DAY_SD = {"atp": 0.073, "wta": 0.073}     # fitted on 2025-26 results: matches 3-set rate and total games
_ADJ = None


def games_shift(tour, best_of, surface):
    """Shift to the combined serve-win rate so expected total games match reality (tools/book_followup --games,
    fitted on 2018-24 matches anchored to Pinnacle's price; e.g. grass plays longer, best-of-3 a bit shorter)."""
    global _ADJ
    if _ADJ is None:
        import json
        from pathlib import Path
        f = Path(__file__).resolve().parents[1] / "model" / "games_adj.json"
        try:
            _ADJ = json.loads(f.read_text())
        except (OSError, ValueError):
            _ADJ = {}
    return _ADJ.get(f"{tour}|{int(best_of)}|{surface}", 0.0)


def _nodes(pa, pb, sd):
    s, d0 = pa + pb, pa - pb
    for z, w in _GH:
        d = d0 + sd * z * 1.4142135623730951
        yield _r(s / 2 + d / 2), _r(s / 2 - d / 2), w / _GHW


def mix_p_match(pa, pb, best_of=3, sd=0.06):
    return sum(w * fast_p_match(a, b, best_of) for a, b, w in _nodes(pa, pb, sd))


def mix_dist(pa, pb, best_of=3, sd=0.06):
    out = {}
    for a, b, w in _nodes(pa, pb, sd):
        for k, v in match_dist(a, b, best_of).items():
            out[k] = out.get(k, 0) + w * v
    return out


def solve_serve_mix(p_win, total, best_of=3, sd=0.06):
    p_win = min(max(p_win, 0.005), 0.995)
    lo, hi = -0.45, 0.45
    for _ in range(22):
        d = (lo + hi) / 2
        if mix_p_match(total / 2 + d / 2, total / 2 - d / 2, best_of, sd) < p_win:
            lo = d
        else:
            hi = d
    d = (lo + hi) / 2
    return round(total / 2 + d / 2, 4), round(total / 2 - d / 2, 4)


def summary_dist(D, pa, pb, best_of=3):
    need = best_of // 2 + 1
    tot = {}
    for (sa, sb, t, d, s1, s2, tb), v in D.items():
        tot[t] = tot.get(t, 0) + v
    return {
        "p": prob(D, lambda sa, sb, *_: sa > sb),
        "sets": {f"{sa}-{sb}": round(v, 4) for (sa, sb), v in _agg(D, lambda k: (k[0], k[1])).items()},
        "straight_a": prob(D, lambda sa, sb, *_: sa == need and sb == 0),
        "straight_b": prob(D, lambda sa, sb, *_: sb == need and sa == 0),
        "set1_a": prob(D, lambda *k: k[4] == 1),
        "set2_a": prob(D, lambda *k: k[5] == 1),
        "tiebreak": prob(D, lambda *k: k[6] == 1),
        "games_mean": sum(t * v for t, v in tot.items()),
        "games": {t: round(v, 5) for t, v in sorted(tot.items())},
        "hold_a": hold(_r(pa)), "hold_b": hold(_r(pb)),
    }
