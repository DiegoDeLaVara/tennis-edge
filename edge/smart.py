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


def rank_wallets(days=75, min_markets=20, top=250, max_events=900, until_days=0, events=None):
    since = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(time.time() - days * 86400))
    evs = events if events is not None else match_events(True, pages=40, since_iso=since)
    cut = time.time() - until_days * 86400
    evs = [e for e in evs if e["volume"] >= 500 and max(e["prices"]) > 0.98 and (parse_t(e["start"]) or 0) < cut]
    evs = sorted(evs, key=lambda e: -e["volume"])[:max_events]
    stats = {}

    def one(e):
        win = 0 if e["prices"][0] > 0.5 else 1
        per = {}
        st = parse_t(e["start"]) or 0
        for t in trades(e["cid"], max_n=8000):
            w = t.get("proxyWallet")
            if not w or (st and (t.get("timestamp") or 0) >= st):      # pre-match positions only: that's the skill we follow
                continue
            sz, px = float(t.get("size") or 0), float(t.get("price") or 0)
            pay = sz if t.get("outcomeIndex") == win else 0.0
            cost = sz * px
            sign = 1 if t.get("side") == "BUY" else -1
            d = per.setdefault(w, [0.0, 0.0, t.get("name") or t.get("pseudonym") or w[:8]])
            d[0] += sign * (pay - cost)       # profit if held to settlement (sells reverse it)
            d[1] += cost if sign > 0 else 0
        return per
    with ThreadPoolExecutor(6) as ex:
        for per in ex.map(one, evs):
            for w, (pnl, cost, name) in per.items():
                s = stats.setdefault(w, {"pnl": 0.0, "cost": 0.0, "n": 0, "wins": 0, "name": name})
                s["pnl"] += pnl; s["cost"] += cost; s["n"] += 1; s["wins"] += pnl > 0
    ranked = []
    for w, s in stats.items():
        if s["n"] < min_markets or s["cost"] < 300:
            continue
        roi = s["pnl"] / s["cost"]
        score = roi * math.sqrt(s["n"] / (s["n"] + 30))     # shrink small samples
        if score > 0.03:
            ranked.append({"wallet": w, "name": s["name"], "n": s["n"], "roi": round(roi, 4), "pnl": round(s["pnl"], 2),
                           "hit": round(s["wins"] / s["n"], 3), "score": round(score, 4)})
    ranked.sort(key=lambda x: -x["score"])
    return {"built": time.time(), "events": len(evs), "wallets": ranked[:top]}


def positions(cid, smart):
    """$ value held by ranked wallets on each outcome (v2 returns up to 500 holders per outcome; v1 gave only 20)."""
    j = get(D + "/holders", {"condition": cid, "limit": 500}) or {}
    side = [0.0, 0.0]
    who = [[], []]
    for tok in j.get("data") or []:
        for x in tok.get("holders", []):
            w = smart.get(x.get("proxy_wallet"))
            if w:
                i = int(x.get("outcome_index", 0))
                side[i] += float(x.get("amount") or 0)
                who[i].append(f"{w['name']} ({w['n']} matches, {w['roi'] * 100:+.0f}% ROI)")
    return side, who
