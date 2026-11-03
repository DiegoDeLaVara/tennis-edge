"""Exact tournament odds from a draw: the probability each player reaches every round and wins the title, given a
pairwise win-probability function. No Monte Carlo: each slot's distribution of occupants is built bottom-up."""
import math


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sig(x):
    return 1 / (1 + math.exp(-max(min(x, 30), -30)))


def simulate(draw, pwin, use_results=True):
    """Returns {player: [P(occupy level 0), P(level 1), ..., P(champion)]}.
    pwin(a, b) -> P(a beats b). Unknown first-round slots (None) are treated as an anonymous qualifier 'Qualifier#j'."""
    depth = len(draw.levels) - 1
    dist = []
    lv0 = []
    for j, nm in enumerate(draw.levels[0]):
        if nm == "BYE":
            lv0.append(None)
        else:
            lv0.append({nm or f"Qualifier#{j}": 1.0})
    dist.append(lv0)
    for k in range(depth):
        nxt = []
        for j in range(len(draw.levels[k + 1])):
            a, b = dist[k][2 * j], dist[k][2 * j + 1]
            known = draw.levels[k + 1][j] if use_results else None
            if known and known != "BYE":
                nxt.append({known: 1.0})
                continue
            if a is None or b is None:
                nxt.append(a if b is None else b)
                continue
            d = {}
            for x, px in a.items():
                if px <= 0:
                    continue
                for y, py in b.items():
                    if py <= 0:
                        continue
                    w = pwin(x, y)
                    d[x] = d.get(x, 0.0) + px * py * w
                    d[y] = d.get(y, 0.0) + px * py * (1 - w)
            nxt.append(d)
        dist.append(nxt)
    out = {}
    for k, L in enumerate(dist):
        for d in L:
            if not d:
                continue
            for x, p in d.items():
                out.setdefault(x, [0.0] * (depth + 1))[k] += p
    return out


def round_names(depth):
    """Labels for levels, last = champion: ... 'R16', 'QF', 'SF', 'F', 'W'."""
    tail = ["W", "F", "SF", "QF", "R16", "R32", "R64", "R128"]
    return [tail[depth - k] if depth - k < len(tail) else f"L{k}" for k in range(depth + 1)]
