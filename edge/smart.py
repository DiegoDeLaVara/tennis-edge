"""Smart money from Polymarket, where every wallet's trades are public.

1. rank_wallets(): replay recently settled tennis match markets, credit each wallet with its realised profit, and
   keep wallets that won on many separate matches (shrunk ROI, so one lucky hit is not "smart").
2. positions(): for open matches, how much those wallets hold on each player right now.
Kalshi's tape is anonymous, so this is the only venue where "who is betting" can be measured.
"""
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

G = "https://gamma-api.polymarket.com"
D = "https://data-api.polymarket.com/v2"       # v1 was retired (410) in Nov 2026
S = requests.Session()
S.headers["User-Agent"] = "TennisEdge/1.0 (https://github.com/DiegoDeLaVara/tennis-edge)"
VS = re.compile(r"^(?:(.+?):\s*)?(.+?)\s+vs\.?\s+(.+?)$")


def get(url, params=None):
    for a in range(4):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After") or 2 * (a + 1))); continue
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(1 + a)
    return None


def match_events(closed, pages=10, since_iso=None):
    """Tennis head-to-head events with their moneyline market."""
    out = []
    for off in range(0, pages * 100, 100):
        p = {"tag_slug": "tennis", "closed": str(closed).lower(), "limit": 100, "offset": off, "order": "endDate", "ascending": "false"}
        if since_iso and closed:
            p["end_date_min"] = since_iso
        ev = get(G + "/events", p) or []
        for e in ev:
            mt = VS.match(e.get("title") or "")
            if not mt:
                continue
            ml = [m for m in e.get("markets", []) if m.get("sportsMarketType") == "moneyline"]
            if not ml:
                ml = [m for m in e.get("markets", []) if (m.get("question") or "").strip() == e["title"].strip()]
            if not ml:
                continue
            m = ml[0]
            try:
                outcomes = json.loads(m["outcomes"])
                prices = [float(x) for x in json.loads(m["outcomePrices"])]
            except (KeyError, ValueError, TypeError):
                continue
            try:
                tokens = json.loads(m.get("clobTokenIds") or "[]")
            except ValueError:
                tokens = []
            out.append({"event": e["title"], "tourney": mt.group(1), "a": outcomes[0], "b": outcomes[1], "cid": m["conditionId"], "tokens": tokens,
                        "prices": prices, "volume": float(m.get("volume") or 0), "start": m.get("gameStartTime") or e.get("startDate"),
                        "closed": bool(m.get("closed"))})
        if len(ev) < 100:
            break
    return out


def trades(cid, max_n=20000, taker_only=False):
    """All fills of one market (v2, cursor-paged), in the old v1 row shape. taker_only=False also returns the maker
    side of each fill, so every wallet's own buys and sells are there."""
    out, cur = [], None
    while len(out) < max_n:
        p = {"condition": cid, "taker_only": str(taker_only).lower()}
        if cur:
            p["cursor"] = cur
        else:
            p["limit"] = 1000
        j = get(D + "/trades", p) or {}
        for t in j.get("data") or []:
            out.append({"proxyWallet": t.get("proxy_wallet"), "side": t.get("side"), "size": t.get("size"), "price": t.get("price"),
                        "outcomeIndex": t.get("outcome_index"), "timestamp": t.get("timestamp"),
                        "name": t.get("name") or t.get("pseudonym")})
        cur = (j.get("pagination") or {}).get("next_cursor")
        if not cur:
            break
    return out


def parse_t(s):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00").replace(" ", "T")).timestamp()
    except ValueError:
        return None


def rank_wallets(prev=None, days=75, min_markets=15, top=400, max_new=1500, k_shrink=20, half_life=120):
    """Rank Polymarket tennis wallets by CLOSING-LINE VALUE (tools/wallet_study, 2026-11-18): how far the last pre-match
    price moved in their favour after they bought, dollar-weighted. Skill shows up there long before profit does
    (profit-ranked "pros" did no better than random wallets; CLV-ranked ones, combined with a Kalshi-vs-Polymarket gap,
    returned +49% vs +20% for the gap alone, both halves).
    Incremental: `prev` (last smart.json) carries each wallet's running tally (decayed, half-life `half_life` days);
    only newly closed matches are processed, so history grows past the API's recent window."""
    prev = prev or {}
    agg = prev.get("agg", {})                         # wallet -> [sum clv*usd, sum usd, n, name, pnl, t_last]
    done = set(prev.get("done", []))
    now = time.time()
    since = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(now - days * 86400))
    evs = match_events(True, pages=40, since_iso=since)
    evs = [e for e in evs if e["cid"] not in done and e["volume"] >= 500 and max(e["prices"]) > 0.98 and "/" not in e["a"]]
    evs = sorted(evs, key=lambda e: -e["volume"])[:max_new]

    def one(e):
        win = 0 if e["prices"][0] > 0.5 else 1
        st = parse_t(e["start"]) or 0
        per, last = {}, None
        for t in trades(e["cid"], max_n=8000):
            w = t.get("proxyWallet")
            ts = t.get("timestamp") or 0
            if not w or (st and ts >= st):            # pre-match positions only
                continue
            sz, px = float(t.get("size") or 0), float(t.get("price") or 0)
            o = int(t.get("outcomeIndex") or 0)
            sign = 1 if t.get("side") == "BUY" else -1
            d = per.setdefault(w, [0.0, 0.0, 0.0, t.get("name") or t.get("pseudonym") or w[:8]])
            d[o] += sign * sz
            d[2] += sign * sz * px
            if last is None or ts > last[0]:
                last = (ts, px if o == 0 else 1 - px)
        if last is None:
            return e["cid"], {}
        p0 = last[1]
        out = {}
        for w, (s0, s1, cash, name) in per.items():
            if s0 > 0 >= s1 and cash > 0:
                side_p, sh, won = p0, s0, win == 0
            elif s1 > 0 >= s0 and cash > 0:
                side_p, sh, won = 1 - p0, s1, win == 1
            else:
                continue
            avg = cash / sh
            if not 0.02 < avg < 0.98:
                continue
            usd = min(cash, 5000.0)
            out[w] = ((side_p - avg) * usd, usd, name, (sh if won else 0.0) - cash)
        return e["cid"], out
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, evs))
    # decay the running tally, then add the new matches
    dt = max(now - prev.get("built", now), 0) / 86400
    f = 0.5 ** (dt / half_life)
    for w, a in agg.items():
        a[0] *= f; a[1] *= f; a[2] *= f; a[4] *= f
    for cid, out in res:
        done.add(cid)
        for w, (cu, usd, name, pnl) in out.items():
            a = agg.setdefault(w, [0.0, 0.0, 0.0, name, 0.0, now])
            a[0] += cu; a[1] += usd; a[2] += 1; a[3] = name or a[3]; a[4] += pnl; a[5] = now
    agg = {w: a for w, a in agg.items() if a[2] >= 0.5 or now - a[5] < 30 * 86400}       # forget long-gone small wallets
    ranked = []
    for w, (cu, usd, n, name, pnl, _) in agg.items():
        if n < min_markets or usd < 500:
            continue
        clv = cu / usd
        score = clv * n / (n + k_shrink)
        if score > 0.02:
            ranked.append({"wallet": w, "name": name, "n": round(n), "clv": round(clv, 4), "roi": round(pnl / usd, 4), "score": round(score, 4)})
    ranked.sort(key=lambda x: -x["score"])
    return {"built": now, "events": len(evs), "tracked": len(agg), "wallets": ranked[:top],
            "agg": {w: [round(a[0], 2), round(a[1], 2), round(a[2], 3), a[3], round(a[4], 2), round(a[5])] for w, a in agg.items()},
            "done": sorted(done)[-20000:], "method": "clv"}


def positions(cid, smart, prices=None):
    """$ value held by ranked wallets on each outcome (v2 returns up to 500 holders per outcome; v1 gave only 20).
    Holder amounts are shares; with `prices` they are valued at the current price of that outcome."""
    j = get(D + "/holders", {"condition": cid, "limit": 500}) or {}
    side = [0.0, 0.0]
    who = [[], []]
    for tok in j.get("data") or []:
        for x in tok.get("holders", []):
            w = smart.get(x.get("proxy_wallet"))
            if w:
                i = int(x.get("outcome_index", 0))
                sh = float(x.get("amount") or 0)
                side[i] += sh * (prices[i] if prices else 1.0)
                tag = f"beats the closing price by {w['clv'] * 100:+.1f}c" if w.get("clv") is not None else f"{w.get('roi', 0) * 100:+.0f}% ROI"
                who[i].append(f"{w['name']} ({w['n']} matches, {tag})")
    return side, who
