r"""Do proven Polymarket tennis wallets predict results beyond the price? Walk-forward, at real prices.

Collect (--fetch): closed Polymarket tennis singles markets (gamma, day windows) with volume >= MIN_VOL; for each,
every PRE-MATCH fill (v2 trades, maker + taker rows, timestamp before the listed start) reduced to each wallet's net
shares on each player and net cash spent; plus the last pre-match Polymarket price and, when the same match is on
Kalshi (work/kalshi_events.jsonl, names + date), Kalshi's last pre-match quote. Cache: work/wallets.jsonl (resumable).

Test: for each month, rank wallets ONLY on earlier months (profit if held to settlement on their pre-match positions,
shrunk ROI, 15+ matches), then follow their combined pre-match position in that month's matches: back the side they
hold 75%+ of $200+ on. Scored at Polymarket's last pre-match price and at Kalshi's ask + fee. Baseline: the same rule
with wallets picked at random from equally active ones (luck check).

  .venv\Scripts\python.exe -m tools.wallet_study --fetch [--since 2026-08-01] [--workers 4]
  .venv\Scripts\python.exe -m tools.wallet_study

Result (2026-11-13; 11,634 markets Jul 15 - Nov 12, 33,651 wallets, 6,686 also on Kalshi): ranked "smart" wallets
(walk-forward by month) = random equally active wallets. Following them: Polymarket -0.5% +- 1.6% (random -0.9%),
Kalshi -2.6% +- 1.9% (random -2.1%). 13 stricter variants (top 25/100, 40+ matches with 10%/20% ROI, $1k/$5k+ positions,
fading the worst): mostly flip between halves; best "40+ matches, ROI 10%+, $5k+ on one side" +4.6% on 512 (~1 SE) =
within luck for 13 tries. Wallet money is information, not a proven signal. NOTE: Polymarket Data API v1 was retired
(410) in Nov 2026; edge/smart.py now uses /v2 (cursor paging, condition=, up to 500 holders per outcome).
"""
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from edge import smart
from edge.names import fold
from tools.kalshi_backtest import fee

CACHE = Path("work/wallets.jsonl")
MIN_VOL = 5000


def key2(a, b):
    return tuple(sorted(((fold(a) or [""])[-1], (fold(b) or [""])[-1])))


def events(since_ts, until_ts):
    out = []
    day = since_ts
    while day < until_ts:
        a, b = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(day)), time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(day + 86400))
        cur = 0
        while True:
            ev = smart.get(smart.G + "/events", {"tag_slug": "tennis", "closed": "true", "limit": 100, "offset": cur,
                                                 "end_date_min": a, "end_date_max": b}) or []
            for e in ev:
                if not smart.VS.match(e.get("title") or "") or "/" in e.get("title", ""):
                    continue
                ml = [m for m in e.get("markets", []) if m.get("sportsMarketType") == "moneyline"]
                if not ml:
                    continue
                m = ml[0]
                try:
                    outcomes, prices = json.loads(m["outcomes"]), [float(x) for x in json.loads(m["outcomePrices"])]
                except (KeyError, ValueError, TypeError):
                    continue
                vol = float(m.get("volume") or 0)
                st = smart.parse_t(m.get("gameStartTime") or e.get("startDate") or "")
                if vol < MIN_VOL or not st or max(prices) < 0.98:
                    continue
                out.append({"cid": m["conditionId"], "a": outcomes[0], "b": outcomes[1], "start": st, "win": 0 if prices[0] > 0.5 else 1,
                            "vol": vol, "title": e["title"]})
            if len(ev) < 100:
                break
            cur += 100
        day += 86400
    return out


def one(e, K):
    T = smart.trades(e["cid"], max_n=25000)
    pre = [t for t in T if t["timestamp"] and t["timestamp"] < e["start"] and t["proxyWallet"]]
    if not pre:
        return None
    W = {}
    for t in pre:
        sz, px = float(t["size"] or 0), float(t["price"] or 0)
        sign = 1 if t["side"] == "BUY" else -1
        d = W.setdefault(t["proxyWallet"], [0.0, 0.0, 0.0])
        d[int(t["outcomeIndex"] or 0)] += sign * sz
        d[2] += sign * sz * px
    pre.sort(key=lambda t: t["timestamp"])
    lt = pre[-1]
    p0 = float(lt["price"]) if lt["outcomeIndex"] == 0 else 1 - float(lt["price"])
    k = None
    for ke in K.get(key2(e["a"], e["b"]), []):
        if abs(ke["start"] - e["start"]) < 2 * 86400:
            q = ke["quote"]
            a0 = (fold(e["a"]) or [""])[-1] == (fold(ke["a"]) or [""])[-1]
            k = {"bid": q["bid"], "ask": q["ask"], "a_is_0": a0}
            break
    return {"cid": e["cid"], "start": e["start"], "win": e["win"], "vol": e["vol"], "p0": round(p0, 4), "n_trades": len(T),
            "kalshi": k, "w": {w: [round(v[0], 2), round(v[1], 2), round(v[2], 2)] for w, v in W.items() if abs(v[2]) >= 1}}


def fetch(since, workers):
    done = {json.loads(l)["cid"] for l in CACHE.open()} if CACHE.exists() else set()
    K = {}
    for l in open("work/kalshi_events.jsonl"):
        e = json.loads(l)
        if e.get("quote") and e.get("y") is not None:
            K.setdefault(key2(e["a"], e["b"]), []).append(e)
    E = [e for e in events(since, time.time() - 86400) if e["cid"] not in done]
    print(f"to fetch: {len(E)} markets ({len(done)} cached)", flush=True)
    with CACHE.open("a") as fh, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, e, K) for e in E]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                if r:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            except Exception as err:  # noqa
                print("err", err, flush=True)
            if i % 200 == 0:
                fh.flush(); print(f"  {i}/{len(E)}", flush=True)


def rank(R, min_n=15):
    S = defaultdict(lambda: [0.0, 0.0, 0])          # wallet -> pnl, cost (gross buys), n
    for r in R:
        for w, (s0, s1, cash) in r["w"].items():
            pnl = (s0 if r["win"] == 0 else s1) - cash
            gross = max(cash, 0) + 1e-9
            s = S[w]; s[0] += pnl; s[1] += abs(cash); s[2] += 1
    out = {}
    for w, (pnl, cost, n) in S.items():
        if n >= min_n and cost >= 300:
            roi = pnl / cost
            out[w] = (roi * math.sqrt(n / (n + 30)), n, roi)
    return out


def follow(R, wallets, min_usd=200, share=0.75):
    res = []
    for r in R:
        side = [0.0, 0.0]
        for w, (s0, s1, cash) in r["w"].items():
            if w in wallets:
                side[0] += max(s0, 0) * r["p0"]; side[1] += max(s1, 0) * (1 - r["p0"])
        tot = sum(side)
        if tot < min_usd or max(side) < share * tot:
            continue
        pick = 0 if side[0] > side[1] else 1
        won = r["win"] == pick
        pp = r["p0"] if pick == 0 else 1 - r["p0"]
        pm = (1 - pp) / pp if won else -1.0 if 0.03 < pp < 0.97 else None
        km = None
        k = r.get("kalshi")
        if k and k["ask"] - k["bid"] <= 0.06:
            a_side = (pick == 0) == k["a_is_0"]
            ask = k["ask"] if a_side else 1 - k["bid"]
            if 0.05 <= ask <= 0.95:
                c = ask + fee(ask); km = (1 - c) / c if won else -1.0
        res.append((r["start"], won, pp, pm, km))
    return res


def analyse():
    R = [json.loads(l) for l in CACHE.open()]
    R.sort(key=lambda r: r["start"])
    print(f"{len(R):,} markets with pre-match trades ({sum(1 for r in R if r.get('kalshi')):,} also on Kalshi); "
          f"{len({w for r in R for w in r['w']}):,} wallets")
    months = sorted({datetime.fromtimestamp(r["start"], timezone.utc).strftime("%Y-%m") for r in R})
    rng = np.random.default_rng(0)
    tot = {"smart": [], "rand": []}
    for m in months[1:]:
        train = [r for r in R if datetime.fromtimestamp(r["start"], timezone.utc).strftime("%Y-%m") < m]
        test = [r for r in R if datetime.fromtimestamp(r["start"], timezone.utc).strftime("%Y-%m") == m]
        if len(train) < 500:
            continue
        rk = rank(train)
        smart_w = {w for w, (sc, n, roi) in rk.items() if sc > 0.03}
        pool = [w for w in rk]
        rand_w = set(rng.choice(pool, size=min(len(smart_w), len(pool)), replace=False)) if pool and smart_w else set()
        for name, W in (("smart", smart_w), ("rand", rand_w)):
            res = follow(test, W)
            tot[name] += res
            pm = [x[3] for x in res if x[3] is not None]; km = [x[4] for x in res if x[4] is not None]
            print(f"  {m} {name:<5} wallets {len(W):>4}: signals {len(res):>4}  won {np.mean([x[1] for x in res]) if res else 0:5.1%} vs "
                  f"{np.mean([x[2] for x in res]) if res else 0:5.1%} priced | Polymarket ROI {np.mean(pm) if pm else 0:+6.1%} (n={len(pm)}) | "
                  f"Kalshi ROI {np.mean(km) if km else 0:+6.1%} (n={len(km)})")
    for name, res in tot.items():
        pm = np.array([x[3] for x in res if x[3] is not None]); km = np.array([x[4] for x in res if x[4] is not None])
        se = lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0
        print(f"ALL {name}: signals {len(res)}; Polymarket ROI {pm.mean() if len(pm) else 0:+.1%} ± {se(pm):.1%} (n={len(pm)}); "
              f"Kalshi ROI {km.mean() if len(km) else 0:+.1%} ± {se(km):.1%} (n={len(km)})")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else "2026-08-01"
        fetch(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp(),
              int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4)
    analyse()
