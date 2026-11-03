"""Sportsbook line shopping for tennis: Pinnacle's no-vig price is the fair line, and we flag US sportsbook prices (and
Kalshi asks) that pay more than it. The same approach tested +1.4% to +3.1% per bet in trading_bot_V2's backtests.

Credits: the-odds-api charges 1 credit per tennis tournament per scan (h2h, <= 10 bookmakers). The budget here is
deliberately small and stops above a reserve so the user's other scanners on the same key keep working.
"""
import json
import os
import time
from datetime import datetime

import requests

API = "https://api.the-odds-api.com/v4/sports"
US_BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet", "fanatics", "hardrockbet", "ballybet"]
NICE = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars", "betrivers": "BetRivers",
        "espnbet": "ESPN BET", "fanatics": "Fanatics", "hardrockbet": "Hard Rock", "ballybet": "Bally Bet", "pinnacle": "Pinnacle"}


def devig(prices):
    """Decimal odds -> fair probabilities (power method, as in trading_bot_V2 bets.scan)."""
    imp = [1 / p for p in prices]
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if sum(i ** k for i in imp) > 1:
            lo = k
        else:
            hi = k
    out = [i ** k for i in imp]
    t = sum(out)
    return [o / t for o in out]


def american(d):
    return f"+{(d - 1) * 100:.0f}" if d >= 2 else f"-{100 / (d - 1):.0f}"


def scan(data_dir, cfg, notes, key=None):
    """Refresh tennis odds when the budget allows; always return the latest snapshot (possibly from an earlier run)."""
    key = key or os.environ.get("ODDS_API_KEY")
    f = data_dir / "odds.json"
    try:
        snap = json.loads(f.read_text()) if f.exists() else {}
    except ValueError:
        snap = {}
    usage = snap.get("usage", {})
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if usage.get("day") != today:
        usage = {"day": today, "credits": 0}
    bc = cfg.get("books", {})
    budget, reserve = bc.get("daily_credits", 4), bc.get("reserve", 100)
    min_gap = bc.get("rescan_hours", 6) * 3600
    if not key:
        snap["status"] = "no key"
        return snap
    if time.time() - snap.get("ts", 0) < min_gap or usage["credits"] >= budget:
        snap["usage"] = usage
        return snap
    try:
        r = requests.get(API, params={"apiKey": key}, timeout=30)           # free
        remaining = int(r.headers.get("x-requests-remaining") or 0)
        sports = [s["key"] for s in r.json() if s["key"].startswith("tennis_") and s.get("active")]
    except (requests.RequestException, ValueError) as e:
        notes.append(f"odds api: {e}")
        return snap
    events = []
    for sk in sports:
        if usage["credits"] >= budget or remaining <= reserve:
            break
        ev = requests.get(f"{API}/{sk}/events", params={"apiKey": key}, timeout=30)      # free
        soon = [e for e in (ev.json() if ev.status_code == 200 else [])
                if 0 < datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).timestamp() - time.time() < 36 * 3600]
        if not soon:
            continue
        r = requests.get(f"{API}/{sk}/odds", params={"apiKey": key, "markets": "h2h", "oddsFormat": "decimal",
                                                     "bookmakers": ",".join(["pinnacle"] + US_BOOKS)}, timeout=40)
        usage["credits"] += 1
        remaining = int(r.headers.get("x-requests-remaining") or remaining)
        if r.status_code != 200:
            notes.append(f"odds api {sk}: HTTP {r.status_code}")
            continue
        for e in r.json():
            events.append({"sport": sk, "id": e["id"], "start": e["commence_time"], "home": e["home_team"], "away": e["away_team"],
                           "books": {b["key"]: {o["name"]: o["price"] for m in b["markets"] if m["key"] == "h2h" for o in m["outcomes"]}
                                     for b in e.get("bookmakers", [])}})
    if events or not snap.get("events"):
        snap = {"ts": time.time(), "events": events, "sports": sports}
    snap["usage"] = usage
    snap["remaining"] = remaining
    snap["status"] = "ok"
    f.write_text(json.dumps(snap))
    return snap


def value_bets(snap, cfg):
    """For each event with a Pinnacle line: fair probabilities and every US-book price that beats them."""
    bc = cfg.get("books", {})
    min_ev, max_ev = bc.get("min_ev", 0.02), bc.get("max_ev", 0.12)
    out = []
    for e in snap.get("events", []):
        pin = e["books"].get("pinnacle")
        if not pin or len(pin) != 2:
            continue
        names = list(pin)
        fair = dict(zip(names, devig([pin[n] for n in names])))
        offers = []
        for bk, prices in e["books"].items():
            if bk == "pinnacle":
                continue
            for n, price in prices.items():
                if n not in fair:
                    continue
                ev = fair[n] * price - 1
                offers.append({"book": NICE.get(bk, bk), "player": n, "price": price, "us": american(price), "ev": round(ev, 4)})
        best = {}
        for o in offers:
            if o["player"] not in best or o["price"] > best[o["player"]]["price"]:
                best[o["player"]] = o
        flagged = [o for o in offers if min_ev <= o["ev"] <= max_ev]
        out.append({"id": e["id"], "sport": e["sport"], "start": e["start"], "players": names, "fair": {n: round(p, 4) for n, p in fair.items()},
                    "pinnacle": pin, "best": best, "value": sorted(flagged, key=lambda o: -o["ev"])})
    return out
