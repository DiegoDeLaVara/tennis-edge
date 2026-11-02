"""Data sources. Everything here is free and public: the Sackmann archive (history, local only), TennisMyLife
(ATP tour results with serve stats, updated daily), ESPN's scoreboard (ATP/WTA tour results and schedule) and
Kalshi's public market data (prices, trades, settlements; no key)."""
import csv
import io
import json
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from .names import key, fold

S = requests.Session()
S.headers["User-Agent"] = "tennis-edge/1.0 (personal research)"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "R128": 4, "R64": 5, "R32": 6, "R16": 7, "RR": 7, "ER": 6,
               "QF": 8, "SF": 9, "BR": 9, "F": 10}


class _Pace:
    """Kalshi's public API allows roughly 10 reads a second; space calls across threads so we never trip 429s."""

    def __init__(self, per_sec):
        self.gap = 1.0 / per_sec
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


KALSHI_PACE = _Pace(8)


def get(url, params=None, tries=6, timeout=40):
    for a in range(tries):
        if url.startswith(KALSHI):
            KALSHI_PACE.wait()
        try:
            r = S.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2.0 * (a + 1)); continue
            if r.status_code >= 500:
                time.sleep(1 + a); continue
            return r
        except requests.RequestException:
            time.sleep(1 + a)
    return None


def day_offset(rnd, draw, level):
    """Rough match day within the event, so fatigue windows see Monday-start events spread over the week."""
    slam = level == "G"
    steps = {"Q1": -3, "Q2": -2, "Q3": -1, "Q4": -1, "R128": 1, "R64": 2 if slam else 1, "R32": 4 if slam else 2,
             "R16": 6 if slam else 3, "RR": 3, "QF": 9 if slam else 4, "SF": 11 if slam else 5, "BR": 5, "F": 13 if slam else 6}
    return steps.get(rnd, 2)


def norm_level(src, lvl, rnd):
    lvl = (lvl or "").strip()
    if rnd and rnd.startswith("Q"):
        return "Q"
    if src == "futures" or lvl.isdigit() or lvl in ("S", "ITF"):
        return "S"
    if lvl in ("250", "500", "A", "P", "PM", "I", "T1", "T2", "T3", "T4", "T5", "W"):
        return "A"
    return lvl or "A"


# ---------------------------------------------------------------- archive (local history)
def archive_rows(root, tour, families, years):
    root = Path(root)
    for fam in families:
        for y in years:
            f = root / tour / f"{tour}_matches{fam}_{y}.csv"
            if not f.exists():
                continue
            src = "futures" if "futures" in fam else ("qual" if "qual" in fam else "tour")
            with open(f, encoding="utf-8", errors="replace", newline="") as fh:
                for r in csv.DictReader(fh):
                    m = std_row(r, tour, src)
                    if m:
                        yield m


def std_row(r, tour, src):
    try:
        d0 = date(int(r["tourney_date"][:4]), int(r["tourney_date"][4:6]), int(r["tourney_date"][6:8]))
    except (ValueError, KeyError, TypeError):
        return None
    rnd = r.get("round") or ""
    level = norm_level(src, r.get("tourney_level"), rnd)
    try:
        draw = int(r.get("draw_size") or 32)
    except ValueError:
        draw = 32
    d = d0 + timedelta(days=day_offset(rnd, draw, r.get("tourney_level")))
    m = {"tour": tour, "date": d.toordinal(), "surface": r.get("surface") or None, "level": level,
         "w": key(r["winner_name"], tour), "l": key(r["loser_name"], tour), "w_name": r["winner_name"], "l_name": r["loser_name"],
         "score": r.get("score") or "", "best_of": int(r.get("best_of") or 3), "round": rnd, "minutes": r.get("minutes"),
         "tourney": r.get("tourney_name"), "indoor": r.get("indoor"),
         "uid": f"{tour}|{r.get('tourney_id')}|{r.get('match_num')}|{key(r['winner_name'], tour)}",
         "sort": (d.toordinal(), ROUND_ORDER.get(rnd, 5))}
    for side in ("w", "l"):
        for c in ("svpt", "1stWon", "2ndWon"):
            v = r.get(f"{side}_{c}")
            if v not in (None, ""):
                m[f"{side}_{c}"] = v
        nm = "winner" if side == "w" else "loser"
        for c in ("hand", "ht", "ioc", "age", "rank"):
            v = r.get(f"{nm}_{c}")
            if v not in (None, ""):
                m[f"{side}_{c}"] = v
    return m


# ---------------------------------------------------------------- TennisMyLife (ATP tour, current year)
def tml_rows(year):
    out = []
    for path in (f"{year}.csv", "ongoing_tourneys.csv"):
        r = get(f"https://stats.tennismylife.org/data/{path}")
        if r is None or r.status_code != 200:
            continue
        for row in csv.DictReader(io.StringIO(r.content.decode("utf-8", "replace"))):
            m = std_row(row, "atp", "tour")
            if m:
                m["uid"] = f"tml|{row.get('tourney_id')}|{key(row['winner_name'], 'atp')}|{key(row['loser_name'], 'atp')}"
                out.append(m)
    return out


# ---------------------------------------------------------------- ESPN (ATP/WTA tour scoreboard)
def espn_day(tour, d):
    r = get(f"https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard", {"dates": d.strftime("%Y%m%d")})
    return r.json() if r is not None and r.status_code == 200 else {}


ESPN_ROUND = {"1st round": "R64", "2nd round": "R32", "3rd round": "R16", "4th round": "R16", "round of 128": "R128",
              "round of 64": "R64", "round of 32": "R32", "round of 16": "R16", "quarterfinal": "QF", "quarterfinals": "QF",
              "semifinal": "SF", "semifinals": "SF", "final": "F", "round robin": "RR"}


def espn_matches(tour, payload):
    """Singles competitions from one scoreboard payload: finished results and scheduled/live matches."""
    out = []
    for e in payload.get("events", []):
        tname = e.get("name") or ""
        major = bool(e.get("major"))
        for g in e.get("groupings") or []:
            slug = g.get("grouping", {}).get("slug", "")
            if "singles" not in slug:
                continue
            for c in g.get("competitions", []):
                comp = c.get("competitors") or []
                if len(comp) != 2:
                    continue
                names = [x.get("athlete", {}).get("displayName") or x.get("athlete", {}).get("fullName") for x in comp]
                if not all(names):
                    continue
                st = c.get("status", {}).get("type", {})
                rd = (c.get("round") or {}).get("displayName", "")
                rnd = "Q" if "qualif" in rd.lower() else ESPN_ROUND.get(rd.lower(), "R32")
                sets = [[int(l.get("value") or 0) for l in x.get("linescores", [])] for x in comp]
                out.append({"id": c.get("id"), "tour": tour, "tourney": tname, "major": major, "round": rnd,
                            "start": c.get("date"), "state": st.get("state"), "completed": st.get("completed"),
                            "detail": st.get("detail") or st.get("shortDetail"), "names": names,
                            "winner": [bool(x.get("winner")) for x in comp], "sets": sets,
                            "venue": (c.get("venue") or {}).get("fullName")})
    return out


def espn_result_row(c, surface_of):
    if not c["completed"] or not any(c["winner"]):
        return None
    wi = 0 if c["winner"][0] else 1
    li = 1 - wi
    sw, sl = c["sets"][wi], c["sets"][li]
    score = " ".join(f"{a}-{b}" for a, b in zip(sw, sl))
    if "retire" in (c.get("detail") or "").lower():
        score += " RET"
    if "walkover" in (c.get("detail") or "").lower():
        score = "W/O"
    d = datetime.fromisoformat(c["start"].replace("Z", "+00:00")).date()
    tour = c["tour"]
    return {"tour": tour, "date": d.toordinal(), "surface": surface_of(c["tourney"], tour), "level": "G" if c["major"] else ("Q" if c["round"] == "Q" else "A"),
            "w": key(c["names"][wi], tour), "l": key(c["names"][li], tour), "w_name": c["names"][wi], "l_name": c["names"][li],
            "score": score, "best_of": 5 if (c["major"] and tour == "atp") else 3, "round": c["round"], "tourney": c["tourney"],
            "uid": f"espn|{c['id']}", "sort": (d.toordinal(), ROUND_ORDER.get(c["round"], 5))}


# ---------------------------------------------------------------- Kalshi
MATCH_SERIES = {"KXATPMATCH": ("atp", "A"), "KXWTAMATCH": ("wta", "A"), "KXATPCHALLENGERMATCH": ("atp", "C"),
                "KXWTACHALLENGERMATCH": ("wta", "C"), "KXITFMATCH": ("atp", "S"), "KXITFWMATCH": ("wta", "S"),
                "KXCHALLENGERMATCH": ("atp", "C")}
RULE_RE = re.compile(r"professional tennis match in the (\d{4}) (.+?) (Round [Oo]f \d+|Quarterfinals?|Semifinals?|Finals?|Qualif\w*.*?|Round Robin|First Round|Second Round|Third Round)\b")


def kalshi(path, params=None):
    r = get(KALSHI + path, params)
    if r is None or r.status_code != 200:
        return {}
    return r.json()


def kalshi_markets(series, status="open", max_pages=20, min_close_ts=None):
    out, cur = [], ""
    for _ in range(max_pages):
        p = {"series_ticker": series, "status": status, "limit": 1000}
        if cur:
            p["cursor"] = cur
        if min_close_ts:
            p["min_close_ts"] = min_close_ts
        d = kalshi("/markets", p)
        ms = d.get("markets") or []
        out += ms
        cur = d.get("cursor") or ""
        if not cur or not ms:
            break
    return out


def parse_rules(m):
    mt = RULE_RE.search(m.get("rules_primary") or "")
    if not mt:
        return None, None
    tname = mt.group(2)
    rd = mt.group(3).lower()
    rnd = {"quarterfinal": "QF", "quarterfinals": "QF", "semifinal": "SF", "semifinals": "SF", "final": "F", "finals": "F",
           "round robin": "RR", "first round": "R32", "second round": "R16", "third round": "R16"}.get(rd)
    if rnd is None:
        n = re.search(r"\d+", rd)
        rnd = f"R{n.group()}" if n and "round" in rd else "Q"
    return tname, rnd


def group_events(markets):
    """Kalshi lists one market per player; pair them into matches keyed by event_ticker."""
    ev = {}
    for m in markets:
        ev.setdefault(m["event_ticker"], []).append(m)
    return {k: v for k, v in ev.items() if len(v) == 2}


def px(m, f):
    v = m.get(f + "_dollars")
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def fnum(m, f):
    try:
        return float(m.get(f) or 0)
    except ValueError:
        return 0.0


def kalshi_result_rows(series, markets, surface_of):
    tour, level = MATCH_SERIES[series]
    out = []
    for et, (a, b) in group_events(markets).items():
        if a.get("result") not in ("yes", "no") or b.get("result") not in ("yes", "no") or a["result"] == b["result"]:
            continue
        w, l = (a, b) if a["result"] == "yes" else (b, a)
        tname, rnd = parse_rules(a)
        when = a.get("occurrence_datetime") or a.get("expected_expiration_time") or a.get("close_time")
        d = datetime.fromisoformat(when.replace("Z", "+00:00")).date()
        out.append({"tour": tour, "date": d.toordinal(), "surface": surface_of(tname or "", tour), "level": "Q" if rnd == "Q" else level,
                    "w": key(w["yes_sub_title"], tour), "l": key(l["yes_sub_title"], tour), "w_name": w["yes_sub_title"], "l_name": l["yes_sub_title"],
                    "score": "", "best_of": 3, "round": rnd, "tourney": tname, "uid": f"kalshi|{et}", "sort": (d.toordinal(), ROUND_ORDER.get(rnd, 5))})
    return out


def kalshi_trades(ticker, min_ts=None, max_pages=10):
    out, cur = [], ""
    for _ in range(max_pages):
        p = {"ticker": ticker, "limit": 1000}
        if cur:
            p["cursor"] = cur
        if min_ts:
            p["min_ts"] = int(min_ts)
        d = kalshi("/markets/trades", p)
        tr = d.get("trades") or []
        out += tr
        cur = d.get("cursor") or ""
        if not cur or not tr:
            break
    return out


def kalshi_candles(series, ticker, start_ts, end_ts, period=60):
    d = kalshi(f"/series/{series}/markets/{ticker}/candlesticks", {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": period})
    return d.get("candlesticks") or []


# ---------------------------------------------------------------- surfaces
class Surfaces:
    """Tournament name -> surface, learned from history plus a few manual entries, matched on name tokens."""
    STOP = {"atp", "wta", "challenger", "ch", "open", "itf", "m15", "m25", "w15", "w35", "w50", "w75", "w100", "125", "125k",
            "tour", "the", "de", "of", "cup", "presented", "by", "international", "masters", "1000", "500", "250", "championships"}

    def __init__(self, table=None):
        self.t = dict(table or {})

    def toks(self, name):
        return [t for t in fold(name) if t not in self.STOP and not t.isdigit() and len(t) > 2]

    def learn(self, name, surface, tour):
        if surface in ("Hard", "Clay", "Grass", "Carpet") and name:
            for t in self.toks(name):
                self.t[f"{tour}|{t}"] = surface
                self.t[f"*|{t}"] = surface

    def __call__(self, name, tour):
        for t in self.toks(name or ""):
            for k in (f"{tour}|{t}", f"*|{t}"):
                if k in self.t:
                    return self.t[k]
        return None
