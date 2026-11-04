"""Sports +EV bet finder (all sports). The market is the model: Pinnacle's no-vig line is the fair price,
and we look for US sportsbooks paying more than fair. Backtests elsewhere in this repo found this (line
shopping) to be the only approach that beat the market (+1.4% to +3.1% per bet before limits); picks from
models or smart money did not.

  python -m bets.scan                 scan now within the daily credit budget, log bets
  python -m bets.scan --settle        update closing-line value and settle finished games (free, no credits)
  python -m bets.scan --status        credits, record

Odds API cost: bookmakers <= 10 counts as one region, so a sport costs one credit per market. The free events
endpoint tells us which sports have games soon, so credits only go where there are games.
"""
import argparse, gzip, json, math, os, re, sqlite3, statistics, time, unicodedata, datetime as dt
from pathlib import Path
import requests

# Cloud copy of trading_bot_V2/bets/scan.py: same scanning, budget and settlement logic, but the data lives in the
# published folder (rolling state on gh-pages), the key comes from the job's ODDS_API_KEY secret, and only bets
# with a Pinnacle line are taken (bets priced off the other books' median had -4% CLV on the local record).
D = Path(__file__).resolve().parents[1] / "out"
DB = D / "sports.db"
CONFIG = Path(__file__).resolve().parents[1] / "config.json"


def use_dir(path):
    global D, DB, ODDS
    D = Path(path)
    DB = D / "sports.db"
    ODDS = D / "odds"


def env_key():
    return os.environ.get("ODDS_API_KEY")
API = "https://api.the-odds-api.com/v4/sports"
US_BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet", "fanatics", "hardrockbet", "ballybet"]
BOOKS = ["pinnacle"] + US_BOOKS                       # 10 bookmakers = one region of credits
NICE = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
        "betrivers": "BetRivers", "espnbet": "ESPN BET", "fanatics": "Fanatics", "hardrockbet": "Hard Rock",
        "ballybet": "Bally Bet", "pinnacle": "Pinnacle"}
US_MAIN = ("americanfootball_nfl", "americanfootball_ncaaf", "basketball_nba", "basketball_ncaab", "basketball_wnba",
           "icehockey_nhl", "baseball_mlb")
ODDS = D / "odds"                                     # raw snapshots: shared with soccer/books.py, and our own odds history
DEFAULTS = {"bankroll": 1000.0, "unit_pct": 0.01, "min_ev": 0.02, "min_ev_consensus": 0.045, "max_ev_clean": 0.12, "daily_credits": 12, "close_credits": 4,
            "reserve": 40, "rescan_hours": 6, "horizon_hours": 30, "max_units": 3.0, "max_odds": 6.0, "books_i_use": US_BOOKS}


def cfg():
    c = dict(DEFAULTS)
    try:
        c.update(json.loads(CONFIG.read_text()).get("sportsbook", {}))
    except (OSError, ValueError):
        pass
    return c


def db():
    D.mkdir(exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""create table if not exists bets(id integer primary key, key text unique, ts real, sport text,
        event_id text, game text, start real, market text, point real, outcome text, book text, price real,
        fair real, ev real, units real, stake real, flag text, close_fair real, close_price real, clv real,
        result text, pnl_units real)""")
    c.execute("create table if not exists spend(day text, sport text, credits int, ts real)")
    c.execute("create table if not exists scans(sport text primary key, ts real)")
    c.execute("create table if not exists kv(k text primary key, v text)")
    return c


def devig(prices):
    """Decimal odds -> fair probs, power method."""
    imp = [1 / p for p in prices]
    s = sum(imp)
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if sum(i ** k for i in imp) > 1:
            lo = k
        else:
            hi = k
    out = [i ** k for i in imp]
    t = sum(out)
    return [o / t for o in out] if abs(t - 1) < 1e-3 else [i / s for i in imp]


def american(d):
    return f"+{(d - 1) * 100:.0f}" if d >= 2 else f"-{100 / (d - 1):.0f}"


def api(path, **p):
    r = requests.get(f"{API}{path}", params={"apiKey": env_key(), **p}, timeout=40)
    return r


def credits_left(db_):
    row = db_.execute("select v from kv where k='remaining'").fetchone()
    return int(row[0]) if row else 500


def spent_today(db_, close=None):
    """Credits spent today; close=True only closing-line checks, close=False only scans."""
    q = "select coalesce(sum(credits),0) from spend where day=?"
    if close is not None:
        q += " and sport %s like 'close:%%'" % ("" if close else "not")
    return db_.execute(q, (time.strftime("%Y-%m-%d"),)).fetchone()[0]


def upcoming_sports(hours):
    """{sport_key: n_events in the next `hours`} from the free events endpoint."""
    sp = [s["key"] for s in api("").json() if s.get("active") and not s.get("has_outrights")]
    now = time.time()
    out = {}
    for k in sp:
        try:
            r = api(f"/{k}/events")
            if r.status_code != 200:
                continue
            n = 0
            for e in r.json():
                t = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).timestamp()
                if now + 1800 < t < now + hours * 3600:
                    n += 1
            if n:
                out[k] = n
        except requests.RequestException:
            continue
    return out


PRIORITY = ["americanfootball_nfl", "basketball_nba", "icehockey_nhl", "americanfootball_ncaaf", "basketball_ncaab",
            "baseball_mlb", "basketball_wnba", "soccer_epl", "soccer_usa_mls", "soccer_uefa_champs_league", "soccer_spain_la_liga",
            "soccer_italy_serie_a", "soccer_germany_bundesliga", "soccer_france_ligue_one", "soccer_uefa_europa_league",
            "mma_mixed_martial_arts", "tennis_atp", "tennis_wta", "boxing_boxing"]


def rank(sport):
    if sport.startswith("tennis") and cfg().get("tennis_first"):
        return -1                                  # this is a tennis board: tennis gets the budget first
    for i, k in enumerate(PRIORITY):
        if sport == k or (k.startswith("tennis_") and sport.startswith(k)):
            return i
    return 99


def markets_for(sport):
    if sport.startswith("tennis"):
        return cfg().get("tennis_markets", "h2h")      # games totals and games spreads too: Pinnacle prices both
    if sport in US_MAIN:
        return "h2h,spreads,totals"
    if sport.startswith("soccer"):
        return "h2h,totals"
    return "h2h"


def fair_for(event):
    """{(market, point): {outcome: prob}} using Pinnacle (else median of 4+ books), plus source tag and #books."""
    out = {}
    per = {}
    for b in event.get("bookmakers", []):
        for m in b["markets"]:
            groups = {}
            for o in m["outcomes"]:
                groups.setdefault(abs(o["point"]) if m["key"] == "spreads" and "point" in o else o.get("point"), []).append(o)
            for pt, outs in groups.items():
                if len(outs) < 2:
                    continue
                per.setdefault((m["key"], pt), []).append((b["key"], {o["name"] + ("" if "point" not in o else f"|{o['point']}"): o["price"] for o in outs}))
    for key, lst in per.items():
        pin = [x for k, x in lst if k == "pinnacle"]
        src = None
        if pin:
            names = list(pin[0])
            fair = dict(zip(names, devig([pin[0][n] for n in names])))
            src = "Pinnacle"
        else:
            if len(lst) < 4:
                continue
            names = list(lst[0][1])
            probs = []
            for _, x in lst:
                if set(x) == set(names):
                    probs.append(dict(zip(names, devig([x[n] for n in names]))))
            if len(probs) < 4:
                continue
            fair = {n: statistics.median(p[n] for p in probs) for n in names}
            src = "consensus"
        out[key] = (fair, src, len(lst))
    return out


def find_bets(event, sport, c):
    """All +EV outcomes at the books the user can bet at, best price per outcome."""
    fairs = fair_for(event)
    start = dt.datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00")).timestamp()
    game = f'{event["away_team"]} @ {event["home_team"]}'
    best = {}
    for b in event.get("bookmakers", []):
        if b["key"] not in c["books_i_use"]:
            continue
        for m in b["markets"]:
            for o in m["outcomes"]:
                pt = o.get("point")
                fk = (m["key"], abs(pt) if m["key"] == "spreads" and pt is not None else pt)
                if fk not in fairs:
                    continue
                fair, src, nb = fairs[fk]
                name = o["name"] + ("" if pt is None else f"|{pt}")
                if name not in fair:
                    continue
                k = (m["key"], pt, o["name"])
                if k not in best or o["price"] > best[k][0]:
                    best[k] = (o["price"], b["key"], fair[name], src, nb)
    res = []
    for (mk, pt, name), (price, book, p, src, nb) in best.items():
        ev = p * price - 1
        # without Pinnacle the fair line is a median of soft books; those bets had negative CLV, so ask for more
        # 95k-match soccer study (soccer/edge_tests.py): beating the sharp line early held up in every odds band except
        # longshots above 6.0, where the holdout CLV shrank to +1.5% and results lost 7%
        if src == "consensus" and c.get("require_pinnacle", True):
            continue
        if price > (c.get("tennis_max_odds", c["max_odds"]) if sport.startswith("tennis") else c["max_odds"]) or ev < (c["min_ev_consensus"] if src == "consensus" else c["min_ev"]):
            continue
        kelly = ev / (price - 1)
        units = max(0.5, min(c["max_units"], round(0.25 * kelly / c["unit_pct"] * 2) / 2))
        if ev < 0.04:
            units = min(units, 1.0)       # thin edge: a 1% error in the fair price would erase it
        elif ev < 0.07:
            units = min(units, 2.0)
        flag = []
        if ev > c["max_ev_clean"]:
            flag.append("check line (very high EV: stale, limited or different number)")
            units = min(units, 0.5)
        if src == "consensus":
            flag.append("no Pinnacle line (weaker fair)")
            units = min(units, 1.0)
        if nb < 4:
            flag.append("few books")
        res.append({"sport": sport, "event_id": event["id"], "game": game, "start": start, "market": mk, "point": pt,
                    "outcome": name, "book": book, "price": price, "fair": p, "ev": ev, "units": units,
                    "stake": units * c["bankroll"] * c["unit_pct"], "flag": "; ".join(flag)})
    return res


def log_bets(c_db, bets, c):
    new = 0
    for b in bets:
        key = f'{b["event_id"]}|{b["market"]}|{b["point"]}|{b["outcome"]}|{b["book"]}'
        if c_db.execute("select 1 from bets where key=?", (key,)).fetchone():
            continue
        c_db.execute("insert into bets(key,ts,sport,event_id,game,start,market,point,outcome,book,price,fair,ev,units,stake,flag)"
                     " values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (key, time.time(), b["sport"], b["event_id"], b["game"], b["start"], b["market"], b["point"], b["outcome"],
                      b["book"], b["price"], b["fair"], b["ev"], b["units"], b["stake"], b["flag"]))
        new += 1
    c_db.commit()
    return new


def scan(force_budget=None):
    c = cfg()
    d = db()
    budget = force_budget if force_budget is not None else c["daily_credits"] - spent_today(d, close=False)
    left = credits_left(d)
    print(f"credits: ~{left} left, daily budget remaining {budget}, reserve {c['reserve']}")
    sports = upcoming_sports(c["horizon_hours"])
    order = sorted(sports, key=lambda s: (rank(s), -sports[s]))
    allbets = []
    for sp in order:
        cost = len(markets_for(sp).split(","))
        row = d.execute("select ts from scans where sport=?", (sp,)).fetchone()
        if row and time.time() - row[0] < c["rescan_hours"] * 3600:
            continue
        if cost > budget or left - cost < c["reserve"]:
            print(f"skip {sp} ({sports[sp]} games): cost {cost}, budget {budget}, left {left}")
            continue
        r = api(f"/{sp}/odds", bookmakers=",".join(BOOKS), markets=markets_for(sp), oddsFormat="decimal", dateFormat="iso")
        if r.status_code != 200:
            print(sp, "HTTP", r.status_code, r.text[:100])
            continue
        used = int(float(r.headers.get("x-requests-last", cost)))
        left = int(float(r.headers.get("x-requests-remaining", left - used)))
        budget -= used
        d.execute("insert into spend values(?,?,?,?)", (time.strftime("%Y-%m-%d"), sp, used, time.time()))
        d.execute("insert or replace into scans values(?,?)", (sp, time.time()))
        d.execute("insert or replace into kv values('remaining',?)", (str(left),))
        d.commit()
        events = r.json()
        save_snapshot(sp, events)
        # snapshots: keep latest fair per outcome for CLV
        for ev_ in events:
            update_close(d, ev_)
            allbets += [b for b in find_bets(ev_, sp, c) if b["start"] > time.time() + 600]
        print(f"{sp:30} {sports[sp]:3d} games  cost {used}  credits left {left}")
    n = log_bets(d, allbets, c)
    print(f"{len(allbets)} +EV prices found, {n} new bets logged")
    return allbets


def save_snapshot(sport, events):
    """Latest odds per sport (read by soccer/books.py instead of spending its own credits), plus an append-only
    history so opening-vs-closing line studies can be backtested on our own data later."""
    ODDS.mkdir(exist_ok=True)
    now = time.time()
    tmp = ODDS / f"{sport}.json.tmp"
    tmp.write_text(json.dumps({"ts": now, "events": events}), encoding="utf-8")
    tmp.replace(ODDS / f"{sport}.json")


def close_check(window_min=75):
    """True closing line: re-price events with pending Pinnacle-fair bets shortly before they start (one credit per
    event and market). Without this, CLV compares against a line that may be a day old."""
    c, d = cfg(), db()
    try:
        d.execute("alter table bets add column close_ts real")
    except sqlite3.OperationalError:
        pass
    budget = c["close_credits"] - spent_today(d, close=True)
    now = time.time()
    rows = d.execute("select distinct sport, event_id, market from bets where result is null and close_ts is null "
                     "and flag not like '%no Pinnacle%' and start between ? and ? order by start",
                     (now + 300, now + window_min * 60)).fetchall()
    done = 0
    for sport, eid, mk in rows:
        if budget <= 0 or credits_left(d) - 1 < c["reserve"]:
            break
        try:
            r = api(f"/{sport}/events/{eid}/odds", bookmakers=",".join(BOOKS), markets=mk, oddsFormat="decimal", dateFormat="iso")
        except requests.RequestException:     # offline (PC just woke up): try again next run
            break
        if r.status_code != 200:
            continue
        used = int(float(r.headers.get("x-requests-last", 1)))
        budget -= used
        d.execute("insert into spend values(?,?,?,?)", (time.strftime("%Y-%m-%d"), f"close:{sport}", used, time.time()))
        if r.headers.get("x-requests-remaining"):
            d.execute("insert or replace into kv values('remaining',?)", (str(int(float(r.headers["x-requests-remaining"]))),))
        ev = r.json()
        update_close(d, ev)
        d.execute("update bets set close_ts=? where event_id=? and market=?", (time.time(), eid, mk))
        d.commit()
        done += 1
    return done


def update_close(d, event):
    """Refresh the closing fair line / price of logged bets for this event (latest scan before the start)."""
    rows = d.execute("select id,market,point,outcome,book from bets where event_id=? and result is null", (event["id"],)).fetchall()
    if not rows:
        return
    start = dt.datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00")).timestamp()
    if time.time() > start:
        return
    fairs = fair_for(event)
    for id_, mk, pt, out, book in rows:
        fk = (mk, abs(pt) if mk == "spreads" and pt is not None else pt)
        if fk not in fairs:
            continue
        fair = fairs[fk][0]
        name = out + ("" if pt is None else f"|{pt}")
        if name in fair:
            row = d.execute("select price from bets where id=?", (id_,)).fetchone()
            d.execute("update bets set close_fair=?, clv=? where id=?", (fair[name], row[0] * fair[name] - 1, id_))
    d.commit()


# ----------------------------------------------------------------------------- settlement via ESPN (free)
ESPN = {"americanfootball_nfl": "football/nfl", "americanfootball_ncaaf": "football/college-football",
        "basketball_nba": "basketball/nba", "basketball_ncaab": "basketball/mens-college-basketball",
        "basketball_wnba": "basketball/wnba", "icehockey_nhl": "hockey/nhl", "baseball_mlb": "baseball/mlb",
        "soccer_epl": "soccer/eng.1", "soccer_usa_mls": "soccer/usa.1", "soccer_spain_la_liga": "soccer/esp.1",
        "soccer_italy_serie_a": "soccer/ita.1", "soccer_germany_bundesliga": "soccer/ger.1",
        "soccer_france_ligue_one": "soccer/fra.1", "soccer_uefa_champs_league": "soccer/uefa.champions",
        "soccer_uefa_europa_league": "soccer/uefa.europa", "soccer_uefa_europa_conference_league": "soccer/uefa.europa.conf",
        "soccer_italy_serie_b": "soccer/ita.2", "soccer_spain_segunda_division": "soccer/esp.2",
        "soccer_germany_bundesliga2": "soccer/ger.2", "soccer_france_ligue_two": "soccer/fra.2", "soccer_efl_champ": "soccer/eng.2",
        "soccer_netherlands_eredivisie": "soccer/ned.1", "soccer_portugal_primeira_liga": "soccer/por.1",
        "soccer_mexico_ligamx": "soccer/mex.1", "soccer_brazil_campeonato": "soccer/bra.1",
        "soccer_argentina_primera_division": "soccer/arg.1", "soccer_turkey_super_league": "soccer/tur.1",
        "soccer_greece_super_league": "soccer/gre.1", "soccer_belgium_first_div": "soccer/bel.1",
        "soccer_spl": "soccer/sco.1", "mma_mixed_martial_arts": "mma/ufc"}
GROUPS = {"basketball/mens-college-basketball": "50", "football/college-football": "80"}   # all of D-I, not just top 25
STOP = {"fc", "cf", "sc", "the", "de", "st", "state", "united", "city", "university", "of"}


def toks(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()     # Atlético == Atletico
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 1}


def same_team(a, b):
    x, y = toks(a), toks(b)
    return bool(x and y and (x <= y or y <= x or len(x & y) >= max(1, min(len(x), len(y)) - 0)))


def espn_games(sport, start_ts):
    path = ESPN.get(sport)
    if not path:
        return []
    out = []
    et = dt.datetime.fromtimestamp(start_ts, dt.timezone.utc) - dt.timedelta(hours=5)
    for delta in (0, 1):
        day = (et + dt.timedelta(days=delta)).strftime("%Y%m%d")
        try:
            j = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard",
                             params={"dates": day, "limit": 500, **({"groups": GROUPS[path]} if path in GROUPS else {})},
                             timeout=30).json()
        except Exception:
            continue
        for e in j.get("events", []):
            if path.startswith("mma"):     # one competition per fight, winner flag instead of a score, no home/away
                for comp in e["competitions"]:
                    cs = comp["competitors"]
                    if not comp["status"]["type"].get("completed") or len(cs) != 2:
                        continue
                    f = [((c.get("athlete") or {}).get("displayName", ""), 1.0 if c.get("winner") else 0.0) for c in cs]
                    if f[0][1] + f[1][1] == 1:
                        out += [{"home": f[0], "away": f[1]}, {"home": f[1], "away": f[0]}]
                continue
            comp = e["competitions"][0]
            if not comp["status"]["type"].get("completed"):
                continue
            teams = {c["homeAway"]: (c["team"]["displayName"], float(c.get("score") or 0)) for c in comp["competitors"]}
            if len(teams) == 2:
                out.append(teams)
    return out


TENNIS = []        # recent tennis results from refresh (w_name, l_name, date ordinal, score)


def tennis_games(sport, start_ts):
    """Tennis has no ESPN team scoreboard here: grade from the results the refresh pulled in (TennisMyLife, ESPN,
    Kalshi). Returned in the same home/away shape, scored as sets won, plus total games when the score is known."""
    day = dt.datetime.fromtimestamp(start_ts, dt.timezone.utc).date().toordinal()
    out = []
    for m in TENNIS:
        if not -14 <= m["date"] - day <= 3:        # some sources date a match by its tournament's first day
            continue
        gw, gl = 0, 0
        for a, b in re.findall(r"(\d+)-(\d+)", re.sub(r"\(\d+\)", "", m.get("score") or "")):
            a, b = int(a), int(b)
            gw += a; gl += b
        ret = bool(re.search(r"RET|W/O|DEF|ABD", m.get("score") or "", re.I))
        games = (gw, gl) if gw + gl and not ret else None
        out.append({"home": (m["w_name"], 1.0), "away": (m["l_name"], 0.0), "games": games})
        out.append({"home": (m["l_name"], 0.0), "away": (m["w_name"], 1.0), "games": games[::-1] if games else None})
    return out


def settle():
    d = db()
    rows = d.execute("select id,sport,game,start,market,point,outcome,price from bets where result is null and start < ?",
                     (time.time() - 3 * 3600,)).fetchall()
    cache = {}
    n = 0
    for id_, sport, game, start, mk, pt, out, price in rows:
        away, home = game.split(" @ ")
        k = (sport, int(start // 86400))
        if k not in cache:
            cache[k] = tennis_games(sport, start) if sport.startswith("tennis") else espn_games(sport, start)
        g = next((t for t in cache[k] if same_team(t["away"][0], away) and same_team(t["home"][0], home)), None)
        if not g:
            if time.time() - start > 4 * 86400:
                d.execute("update bets set result='void' where id=?", (id_,))
            continue
        hs, as_ = g["home"][1], g["away"][1]
        if sport.startswith("tennis") and mk in ("totals", "spreads"):
            if not g.get("games"):                         # retirement or no score: books void these
                if time.time() - start > 4 * 86400:
                    d.execute("update bets set result='void' where id=?", (id_,))
                continue
            hs, as_ = g["games"]                           # games won by the "home" / "away" player
        if mk == "h2h":
            winner = "Draw" if hs == as_ else (home if hs > as_ else away)
            res = "win" if (out == "Draw" and winner == "Draw") or (out != "Draw" and winner != "Draw" and same_team(out, winner)) else "loss"
        elif mk == "totals":
            tot = hs + as_
            res = "push" if tot == pt else ("win" if (tot > pt) == (out.split("|")[0] == "Over") else "loss")
        elif mk == "spreads":
            margin = (hs - as_) if same_team(out, home) else (as_ - hs)
            adj = margin + pt
            res = "push" if adj == 0 else ("win" if adj > 0 else "loss")
        else:
            continue
        pnl = (price - 1) if res == "win" else (-1.0 if res == "loss" else 0.0)
        unit_row = d.execute("select units from bets where id=?", (id_,)).fetchone()[0]
        d.execute("update bets set result=?, pnl_units=? where id=?", (res, pnl * unit_row, id_))
        n += 1
    d.commit()
    return n


def status():
    d = db()
    print("credits left (approx):", credits_left(d), "| spent today:", spent_today(d))
    for r in d.execute("select result,count(*),round(sum(units),1),round(coalesce(sum(pnl_units),0),2),round(avg(clv)*100,2) from bets group by result"):
        print(r)


def build_json():
    """sportsbook.json for the phone page: open bets, the record, CLV and credits."""
    c, d = cfg(), db()
    now = time.time()
    cols = ["game", "sport", "start", "market", "point", "outcome", "book", "price", "fair", "ev", "units", "stake", "flag", "ts", "clv"]
    open_ = [dict(zip(cols, r)) for r in d.execute(f"select {','.join(cols)} from bets where result is null and start>? order by ev desc", (now,))]
    for b in open_:
        b["book"] = NICE.get(b["book"], b["book"])
        b["us"] = american(b["price"])
        b["min_odds"] = round(1 / b["fair"] * 1.01, 3)
        b["min_us"] = american(b["min_odds"])
    settled = [dict(zip(cols + ["result", "pnl_units"], r)) for r in d.execute(
        f"select {','.join(cols)},result,pnl_units from bets where result in ('win','loss','push') order by start desc")]

    def agg(rows):
        risk = sum(r["units"] for r in rows if r["result"] in ("win", "loss"))
        pnl = sum(r["pnl_units"] or 0 for r in rows)
        clv = [r["clv"] for r in rows if r["clv"] is not None]
        return {"n": len(rows), "w": sum(r["result"] == "win" for r in rows), "l": sum(r["result"] == "loss" for r in rows),
                "units": round(risk, 2), "pnl": round(pnl, 2), "roi": round(pnl / risk, 4) if risk else 0,
                "clv": round(sum(clv) / len(clv), 4) if clv else None, "clv_n": len(clv)}
    by_sport, by_book = {}, {}
    for r in settled:
        by_sport.setdefault(r["sport"], []).append(r)
        by_book.setdefault(NICE.get(r["book"], r["book"]), []).append(r)
    all_clv = [r[0] for r in d.execute("select clv from bets where clv is not null")]
    last = d.execute("select max(ts) from scans").fetchone()[0]
    for r in settled:
        r["book"] = NICE.get(r["book"], r["book"])
        r["us"] = american(r["price"])
    out = {"generated": now, "has_key": bool(env_key()), "bankroll": c["bankroll"], "unit_pct": c["unit_pct"],
           "credits_left": credits_left(d), "spent_today": spent_today(d), "daily_budget": c["daily_credits"] + c["close_credits"],
           "last_scan": last, "open": open_, "record": agg(settled), "by_sport": {k: agg(v) for k, v in by_sport.items()},
           "by_book": {k: agg(v) for k, v in by_book.items()},
           "clv_all": {"n": len(all_clv), "avg": round(sum(all_clv) / len(all_clv), 4) if all_clv else None,
                       "beat": round(sum(1 for x in all_clv if x > 0) / len(all_clv), 3) if all_clv else None},
           "recent": settled[:40]}
    (D / "sportsbook.json").write_text(json.dumps(out, separators=(",", ":")))
    return out


def run(data_dir, notes, tennis_rows=None):
    """One cloud cycle: scan within budget, re-price bets about to start, settle finished ones, write the page data."""
    use_dir(data_dir)
    TENNIS[:] = tennis_rows or []
    if env_key():
        for step in (scan, close_check, settle):
            try:
                step()
            except Exception as e:
                notes.append(f"sportsbook {step.__name__}: {e}")
    else:
        notes.append("sportsbook: waiting for the ODDS_API_KEY secret")
    build_json()
