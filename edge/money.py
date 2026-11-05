"""Money-flow signals from Kalshi's public trade tape.

Kalshi lists one market per player, so money toward player A is: A-yes taker buys + B-no taker buys, and against A
is A-no + B-yes. "Sharp" here means big tickets: bettors who risk $250+ on one fill are, on average, better informed
than $5 fills (this is a hypothesis the backtest checks, not a given). "Steam" is a fast price move on volume.
"""
from datetime import datetime

BIG = 250.0          # $ notional for a "big ticket"


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def tape(trades_a, trades_b):
    """Unified list of (ts, dollars, direction toward A, implied price of A) sorted by time."""
    out = []
    for tr, side in ((trades_a, 1), (trades_b, -1)):
        for t in tr:
            yes = _f(t.get("yes_price_dollars"))
            cnt = _f(t.get("count_fp"))
            taker_yes = t.get("taker_side") == "yes"
            dollars = cnt * (yes if taker_yes else 1 - yes)
            d = (1 if taker_yes else -1) * side
            pa = yes if side == 1 else 1 - yes
            out.append((_ts(t["created_time"]), dollars, d, pa))
    out.sort()
    return out


def detect_start(T, bucket=600, lo=2500.0, lo2=1500.0):
    """First 10-minute bucket where trading surges and stays high: in-play has begun. None if not yet."""
    if not T:
        return None
    t0 = T[0][0]
    vols = {}
    for t, d, *_ in T:
        k = int((t - t0) // bucket)
        vols[k] = vols.get(k, 0) + d
    n = max(vols) + 1
    v = [vols.get(i, 0.0) for i in range(n)]
    for i in range(n - 1):
        prior = sorted(x for x in v[max(0, i - 36):i] if x > 0)
        med = prior[len(prior) // 2] if prior else 0.0
        thr = max(lo, 8 * med)
        if v[i] >= thr and v[i + 1] >= max(lo2, 3 * med):
            return t0 + i * bucket
    return None


def flow_features(trades_a, trades_b, start_ts=None, lo=2500.0, lo2=1500.0):
    """Pre-match money flow. With start_ts None, the start is detected from the tape (in-play surge)."""
    T = tape(trades_a, trades_b)
    started = detect_start(T, lo=lo, lo2=lo2)
    if start_ts is None:
        start_ts = (started - 600) if started else float("inf")
    T = [x for x in T if x[0] <= start_ts]
    if not T:
        return {"n": 0, "vol": 0.0, "started": started}
    if start_ts == float("inf"):
        start_ts = T[-1][0]
    vol = sum(x[1] for x in T)
    net = sum(x[1] * x[2] for x in T)
    big = [x for x in T if x[1] >= BIG]
    late = [x for x in T if x[0] >= start_ts - 3 * 3600]
    whale = max(T, key=lambda x: x[1])

    def price_at(t):
        p = None
        for x in T:
            if x[0] <= t:
                p = x[3]
            else:
                break
        return p
    # steam: largest 60-minute move in A's price that came with at least $500 traded
    steam = 0.0
    j = 0
    win = 0.0
    for i in range(len(T)):
        win += T[i][1]
        while T[i][0] - T[j][0] > 3600:
            win -= T[j][1]
            j += 1
        if win >= 500:
            mv = T[i][3] - T[j][3]
            if abs(mv) > abs(steam):
                steam = mv
    return {"started": started, "t_last": T[-1][0], "n": len(T), "vol": round(vol, 2), "net": round(net, 2), "net_share": round(net / vol, 4) if vol else 0,
            "big_n": len(big), "big_vol": round(sum(x[1] for x in big), 2), "big_net": round(sum(x[1] * x[2] for x in big), 2),
            "late_vol": round(sum(x[1] for x in late), 2), "late_net": round(sum(x[1] * x[2] for x in late), 2),
            "whale": round(whale[1], 2), "whale_dir": whale[2],
            "px_first": round(T[0][3], 4), "px_last": round(T[-1][3], 4), "px_6h": price_at(start_ts - 6 * 3600),
            "px_24h": price_at(start_ts - 24 * 3600), "steam": round(steam, 4)}


def signals(flow, fair_a=None):
    """Plain-English money signals for player A (positive = toward A)."""
    out = []
    if not flow or flow.get("n", 0) < 5:
        return out
    if flow.get("big_vol", 0) >= 750 and abs(flow.get("big_net", 0)) >= 0.6 * flow["big_vol"]:
        out.append(("sharp", 1 if flow["big_net"] > 0 else -1, f"Big tickets: ${abs(flow['big_net']):,.0f} net on one side ({flow['big_n']} trades of ${BIG:.0f}+)"))
    if abs(flow.get("steam", 0)) >= 0.05:
        out.append(("steam", 1 if flow["steam"] > 0 else -1, f"Steam move: price moved {flow['steam'] * 100:+.0f}c inside an hour on volume"))
    if flow.get("px_first") is not None and flow.get("px_last") is not None and flow["vol"] >= 1000:
        mv = flow["px_last"] - flow["px_first"]
        if abs(mv) >= 0.06:
            out.append(("drift", 1 if mv > 0 else -1, f"Line moved {mv * 100:+.0f}c since open"))
    if flow.get("vol", 0) >= 2000 and abs(flow.get("net_share", 0)) >= 0.35:
        out.append(("flow", 1 if flow["net_share"] > 0 else -1, f"{abs(flow['net_share']) * 100:.0f}% of ${flow['vol']:,.0f} traded is one-way"))
    return out
