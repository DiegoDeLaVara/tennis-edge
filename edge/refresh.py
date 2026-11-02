"""One refresh of the board. Run by the cloud job every ~20 minutes (and runnable locally).

  python -m edge.refresh --data out
Reads/writes in --data: ratings.json.gz (rolling state), log.json (paper record of every pick), smart.json
(ranked Polymarket wallets) and board.json (what the page shows).
"""
import argparse
import gzip
import json
import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import markov, money, smart
from . import sources as src
from .model import Model, logit, sig
from .names import Index, key
from .ratings import Ratings

ROOT = Path(__file__).resolve().parents[1]
SLAMS = ("australian open", "roland garros", "french open", "wimbledon", "us open")
NOW = time.time()


def fee(p):
    return 0.07 * p * (1 - p)


def ev_at(p, price):
    """Expected return per $ for buying YES at `price` (taker fee included)."""
    if price is None or price <= 0 or price >= 1:
        return None
    return p / (price + fee(price)) - 1


def max_price(p, edge):
    best = None
    for c in range(1, 100):
        x = c / 100
        if p / (x + fee(x)) - 1 >= edge:
            best = c
    return best


def units(p, price, cfg):
    """Fractional Kelly, in units of cfg.unit_pct of bankroll."""
    cost = price + fee(price)
    b = (1 - cost) / cost
    k = (p * b - (1 - p)) / b
    if k <= 0:
        return 0.0
    u = k * cfg["kelly_frac"] / cfg["unit_pct"]
    return round(min(max(u, cfg["min_units"]), cfg["max_units"]) * 4) / 4


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def tsf(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


# ------------------------------------------------------------------ state
def load_state(data):
    f = data / "ratings.json.gz"
    if not f.exists():
        f = ROOT / "state" / "ratings.json.gz"
    with gzip.open(f, "rt") as fh:
        st = json.load(fh)
    R = Ratings.load(st)
    surfaces = src.Surfaces(st.get("surfaces"))
    seen = {}
    for w, l, d in st.get("seen_pairs", []):
        seen[(min(w, l), max(w, l), d)] = 1
    return R, surfaces, seen, st


def save_state(data, R, surfaces, seen, st):
    out = R.dump(min_last=date.today().toordinal() - 900)
    out["built"] = st.get("built")
    out["updated"] = iso(NOW)
    cut = date.today().toordinal() - 40
    out["seen_pairs"] = [[w, l, d] for (w, l, d) in seen if d >= cut]
    out["surfaces"] = surfaces.t
    tmp = data / "ratings.json.gz.tmp"
    with gzip.open(tmp, "wt") as fh:
        json.dump(out, fh, separators=(",", ":"))
    tmp.replace(data / "ratings.json.gz")


def is_seen(seen, m):
    """Sources date the same match differently (TennisMyLife estimates the day), so a pair within 7 days is a repeat."""
    a, b = min(m["w"], m["l"]), max(m["w"], m["l"])
    return any((a, b, d) in seen for d in range(m["date"] - 7, m["date"] + 8))


def update_results(R, surfaces, seen, cfg, log):
    """Add results we have not seen: TennisMyLife, ESPN (last 4 days), Kalshi settlements (last 4 days)."""
    since = date.today().toordinal() - 4
    rows = []
    try:
        rows += [m for m in src.tml_rows(date.today().year) if m["date"] >= since - 10]
    except Exception as e:
        log.append(f"tml failed: {e}")
    for d in range(0, 4):
        day = date.today() - timedelta(days=d)
        for tour in ("wta",):
            try:
                for c in src.espn_matches(tour, src.espn_day(tour, day)):
                    m = src.espn_result_row(c, surfaces)
                    if m and m["date"] >= since:
                        rows.append(m)
            except Exception as e:
                log.append(f"espn {tour} failed: {e}")
    min_close = int(NOW - 4 * 86400)
    for s in cfg["match_series"]:
        try:
            ms = src.kalshi_markets(s, status="settled", max_pages=5, min_close_ts=min_close)
            rows += src.kalshi_result_rows(s, ms, surfaces)
        except Exception as e:
            log.append(f"kalshi results {s} failed: {e}")
    rows.sort(key=lambda m: m["sort"])
    added = 0
    for m in rows:
        if is_seen(seen, m):
            continue
        R.add(m)
        seen[(min(m["w"], m["l"]), max(m["w"], m["l"]), m["date"])] = 1
        added += 1
    return added


# ------------------------------------------------------------------ markets
def pair_markets(markets):
    ev = {}
    for m in markets:
        ev.setdefault(m["event_ticker"], []).append(m)
    return ev


def suffix(event_ticker):
    parts = event_ticker.split("-")
    return parts[1] if len(parts) > 1 else event_ticker


def side_quote(m):
    return {"ticker": m["ticker"], "bid": src.px(m, "yes_bid"), "ask": src.px(m, "yes_ask"), "last": src.px(m, "last_price"),
            "vol": src.fnum(m, "volume_fp"), "vol24": src.fnum(m, "volume_24h_fp"), "oi": src.fnum(m, "open_interest_fp")}


def mid(q):
    b, a = q.get("bid"), q.get("ask")
    if b is not None and a is not None and a > 0 and a - b <= 0.12:
        return (a + b) / 2
    return None


def market_prob(qa, qb):
    ma, mb = mid(qa), mid(qb)
    if ma is not None and mb is not None:
        return (ma + 1 - mb) / 2
    if ma is not None:
        return ma
    if mb is not None:
        return 1 - mb
    return None


def player_card(R, k, t):
    p = R.P.get(k)
    if not p:
        return {"known": False}
    form = [{"d": date.fromordinal(d).isoformat(), "w": w, "opp": o, "s": sc, "surf": sf} for (d, mn, w, o, sf, sc) in reversed(p.recent[-8:])]
    return {"known": True, "elo": round(p.elo), "selo": {s: round(v) for s, v in p.selo.items() if p.sn[s] >= 5}, "n": p.n,
            "rank": p.rank, "hand": p.hand, "ht": p.ht, "ioc": p.ioc, "age": round((t - p.born) / 365.25, 1) if p.born else None,
            "wl": [p.wins, p.losses], "form": form, "days_off": t - p.last if p.last else None}


def run(data, cfg, do_smart=True, verbose=True):
    data.mkdir(parents=True, exist_ok=True)
    notes = []
    R, surfaces, seen, st = load_state(data)
    added = update_results(R, surfaces, seen, cfg, notes)
    t_today = date.today().toordinal()
    model = Model()
    idx = Index(R.P.keys())

    # ---- open match markets
    matches = []
    for s in cfg["match_series"]:
        tour, level = src.MATCH_SERIES[s]
        ms = src.kalshi_markets(s, status="open", max_pages=5)
        for et, pair in pair_markets(ms).items():
            if len(pair) != 2:
                continue
            a, b = sorted(pair, key=lambda m: m["ticker"])
            start = tsf(a.get("occurrence_datetime") or a["expected_expiration_time"])
            if start - NOW > cfg["horizon_hours"] * 3600:
                continue
            tname, rnd = src.parse_rules(a)
            tname = tname or (a.get("title") or "")
            lvl = "Q" if rnd == "Q" else level
            if tour == "atp" and level == "A" and any(x in tname.lower() for x in SLAMS):
                lvl = "G"
            if level == "A" and "finals" in tname.lower():
                lvl = "F"
            bo = 5 if (lvl == "G" and tour == "atp") else 3
            surf = surfaces(tname, tour) or "Hard"
            matches.append({"id": suffix(et), "event": et, "series": s, "tour": tour, "level": lvl, "tourney": tname, "round": rnd,
                            "surface": surf, "surface_known": surfaces(tname, tour) is not None, "bo": bo, "start": start,
                            "names": [a["yes_sub_title"], b["yes_sub_title"]], "q": [side_quote(a), side_quote(b)]})

    # ---- trade tapes (money flow) for matches with some trading
    def tape_job(m):
        if max(m["q"][0]["vol"], m["q"][1]["vol"]) < cfg["trade_fetch_min_volume"]:
            return None
        ta = src.kalshi_trades(m["q"][0]["ticker"], max_pages=6)
        tb = src.kalshi_trades(m["q"][1]["ticker"], max_pages=6)
        return money.flow_features(ta, tb)
    with ThreadPoolExecutor(10) as ex:
        flows = list(ex.map(tape_job, matches))

    # ---- Polymarket smart money
    smart_db = {}
    sf = data / "smart.json"
    if sf.exists():
        try:
            smart_db = json.loads(sf.read_text())
        except ValueError:
            smart_db = {}
    if do_smart and NOW - smart_db.get("built", 0) > cfg["smart_refresh_hours"] * 3600:
        try:
            smart_db = smart.rank_wallets()
            sf.write_text(json.dumps(smart_db))
        except Exception as e:
            notes.append(f"smart wallet ranking failed: {e}")
    sw = {w["wallet"]: w for w in smart_db.get("wallets", [])}
    poly = {}
    if do_smart:
        try:
            for e in smart.match_events(False, pages=6):
                ka, kb = idx.resolve(e["a"], "atp"), idx.resolve(e["b"], "atp")
                kwa, kwb = idx.resolve(e["a"], "wta"), idx.resolve(e["b"], "wta")
                for x, y in ((ka, kb), (kwa, kwb)):
                    poly[(x, y)] = (e, False)
                    poly[(y, x)] = (e, True)
        except Exception as e:
            notes.append(f"polymarket failed: {e}")

    # ---- derivatives
    derivs = {}
    for s in cfg["deriv_series"]:
        try:
            for m in src.kalshi_markets(s, status="open", max_pages=5):
                derivs.setdefault((s[2:5].lower(), suffix(m["event_ticker"])), []).append((s, m))
        except Exception as e:
            notes.append(f"deriv {s} failed: {e}")

    # ---- score
    board = []
    smart_jobs = []
    for m, fl in zip(matches, flows):
        tour = m["tour"]
        ka, kb = idx.resolve(m["names"][0], tour), idx.resolve(m["names"][1], tour)
        m["keys"] = [ka, kb]
        m["players"] = [player_card(R, ka, t_today), player_card(R, kb, t_today)]
        qa, qb = m["q"]
        p_mkt = market_prob(qa, qb)
        m["p_mkt"] = p_mkt
        status = "pre"
        if fl and fl.get("started"):
            status = "live"
        elif NOW > m["start"] + 4 * 3600:
            status = "live"
        m["status"] = status
        m["flow"] = fl
        m["signals"] = [{"k": k, "dir": d, "txt": t} for k, d, t in money.signals(fl)] if fl else []
        f = R.features(ka, kb, tour, m["surface"], t_today, m["bo"], m["level"]) if (ka in R.P and kb in R.P) else None
        if f and R.P[ka].n >= 3 and R.P[kb].n >= 3:
            p_model, p_st = model.prob(f, tour, m["level"], m["bo"])
            p_final = model.blend(p_model, p_mkt)
            total = f["pa"] + f["pb"]
            sd = markov.DAY_SD[tour]
            pa, pb = markov.solve_serve_mix(p_final, total, m["bo"], sd)
            D = markov.mix_dist(pa, pb, m["bo"], sd)
            summ = markov.summary_dist(D, pa, pb, m["bo"])
            m["model"] = {"p_elo": round(f["p_elo"], 4), "p_stats": round(p_st, 4) if f["stat_w"] >= 400 else None, "p_model": round(p_model, 4),
                          "p": round(p_final, 4), "pa": pa, "pb": pb, "stat_w": round(f["stat_w"]), "h2h": [R.h2h.get((ka, kb), 0), R.h2h.get((kb, ka), 0)],
                          "load": [round(f["load_a"]), round(f["load_b"])], "form": [round(f["form_a"], 3), round(f["form_b"], 3)]}
            m["sum"] = {"sets": summ["sets"], "straight": [round(summ["straight_a"], 4), round(summ["straight_b"], 4)],
                        "set1": round(summ["set1_a"], 4), "tb": round(summ["tiebreak"], 4), "games": round(summ["games_mean"], 1),
                        "hold": [round(summ["hold_a"], 3), round(summ["hold_b"], 3)],
                        "gdist": {k: v for k, v in summ["games"].items() if v >= 0.002}}
            m["_D"] = D
        else:
            m["model"] = None
        pe = poly.get((ka, kb))
        if pe:
            smart_jobs.append((m, pe))
        board.append(m)

    def smart_job(item):
        m, (e, flipped) = item
        side, who = smart.positions(e["cid"], sw) if sw else ([0, 0], [[], []])
        if flipped:
            side, who = side[::-1], who[::-1]
        pr = e["prices"][::-1] if flipped else e["prices"]
        return m, {"poly_price": round(pr[0], 3), "poly_vol": round(e["volume"]), "smart": [round(x) for x in side], "who": who}
    if smart_jobs:
        with ThreadPoolExecutor(6) as ex:
            for m, res in ex.map(smart_job, smart_jobs):
                m["poly"] = res
                tot = sum(res["smart"])
                if tot >= 200 and max(res["smart"]) >= 0.75 * tot:
                    d = 1 if res["smart"][0] > res["smart"][1] else -1
                    m["signals"].append({"k": "smart", "dir": d, "txt": f"Proven Polymarket tennis winners hold ${max(res['smart']):,.0f} on {m['names'][0 if d > 0 else 1]}"})

    # ---- bets
    strat = cfg["strategies"]
    for m in board:
        m["bets"] = []
        if m["status"] != "pre" or not m.get("model"):
            continue
        p = m["model"]["p"]
        nmin = min(R.P[m["keys"][0]].n, R.P[m["keys"][1]].n)
        vol = max(m["q"][0]["vol"], m["q"][1]["vol"])
        pm = m["model"]["p_model"]
        for i, q in enumerate(m["q"]):
            pi = p if i == 0 else 1 - p
            pmi = pm if i == 0 else 1 - pm
            ask = q.get("ask")
            ev = ev_at(pi, ask)
            if ev is None:
                continue
            ev_m = ev_at(pmi, ask)
            sig_dir = sum(s["dir"] for s in m["signals"]) * (1 if i == 0 else -1)
            tags, u = [], 0.0
            for name in ("value", "model_edge", "value_sharp"):
                sc = strat[name]
                e_, p_ = (ev_m, pmi) if sc.get("basis") == "model" else (ev, pi)
                if (e_ >= sc["min_ev"] and sc["min_price"] <= ask <= sc["max_price"] and nmin >= sc["min_matches"]
                        and m["level"] in sc["levels"] and vol >= sc["min_vol"] and (not sc.get("need_signal") or sig_dir > 0)):
                    tags.append(name)
                    u = max(u, units(p_, ask, cfg) * sc.get("kelly_mult", 1.0))
            u = round(min(max(u, cfg["min_units"]), cfg["max_units"]) * 4) / 4 if tags else 0
            m["bets"].append({"ticker": q["ticker"], "label": f"{m['names'][i]} to win", "kind": "match", "p": round(pi, 4), "p_model": round(pmi, 4),
                              "ask": ask, "ev": round(ev, 4), "ev_model": round(ev_m, 4), "take_at": max_price(pi, 0.03),
                              "units": u, "tags": tags, "money": sig_dir})
        D = m.pop("_D", None)
        for s, dm in derivs.get((m["tour"], m["id"]), []):
            pr = deriv_prob(s, dm, D, m)
            if pr is None:
                continue
            pi, label = pr
            ask = src.px(dm, "yes_ask")
            ev = ev_at(pi, ask)
            if ev is None:
                continue
            sc = strat["deriv"]
            tags = ["deriv"] if (ev >= sc["min_ev"] and sc["min_price"] <= ask <= sc["max_price"] and nmin >= sc["min_matches"] and m["level"] in sc["levels"]) else []
            m["bets"].append({"ticker": dm["ticker"], "label": label, "kind": "deriv", "p": round(pi, 4), "ask": ask, "bid": src.px(dm, "yes_bid"),
                              "ev": round(ev, 4), "take_at": max_price(pi, 0.05), "units": units(pi, ask, cfg) if tags else 0, "tags": tags})
    for m in board:
        m.pop("_D", None)

    # ---- combos: same-match pairs and the best cross-match pairs
    combos = build_combos(board, cfg)

    # ---- paper record
    logf = data / "log.json"
    log = json.loads(logf.read_text()) if logf.exists() else {"bets": {}}
    record_picks(log, board, combos)
    settle(log)
    logf.write_text(json.dumps(log))
    rec = summarize(log)

    picks = sorted([dict(b, match=m["id"], names=m["names"], tourney=m["tourney"], start=m["start"], tour=m["tour"], level=m["level"])
                    for m in board for b in m.get("bets", []) if b["tags"]], key=lambda b: -b["ev"])
    for m in board:
        m["bets"] = sorted(m.get("bets", []), key=lambda b: -(b["ev"] or -9))[:14]
    meta = json.loads((ROOT / "model" / "coef.json").read_text())
    out = {"generated": NOW, "added_results": added, "notes": notes, "n_matches": len(board),
           "config": {k: cfg[k] for k in ("bankroll", "unit_pct", "kelly_frac", "max_units")} | {"strategies": strat},
           "model_meta": {k: meta.get(k) for k in ("walk_forward", "market", "kalshi_backtest", "features")},
           "smart_wallets": len(sw), "matches": sorted(board, key=lambda m: m["start"]), "picks": picks, "combos": combos,
           "record": rec, "results": settled_map(data, cfg, notes)}
    (data / "board.json").write_text(json.dumps(out, separators=(",", ":"), default=lambda o: None))
    save_state(data, R, surfaces, seen, st)
    if verbose:
        print(f"matches {len(board)}, with model {sum(1 for m in board if m.get('model'))}, picks {len(picks)}, combos {len(combos)}, "
              f"new results {added}, notes {notes}")
    return out


def deriv_prob(series, dm, D, m):
    """Fair YES probability for a derivative market from the joint match distribution."""
    if D is None:
        return None
    na, nb = m["names"]
    sub = dm.get("yes_sub_title") or ""
    strike = dm.get("floor_strike")
    def who(s):
        s = s.lower()
        la, lb = na.split()[-1].lower(), nb.split()[-1].lower()
        if la in s and lb not in s:
            return 0
        if lb in s and la not in s:
            return 1
        return None
    if "GTOTAL" in series and strike is not None:
        return markov.prob(D, lambda sa, sb, t, d, *_: t > strike), f"Over {strike} games"
    if "GSPREAD" in series and strike is not None:
        w = who(sub)
        if w is None:
            return None
        return markov.prob(D, lambda sa, sb, t, d, *_: (d if w == 0 else -d) > strike), f"{m['names'][w]} -{strike} games"
    if "EXACTMATCH" in series:
        w = who(sub)
        import re
        sc = re.search(r"(\d)-(\d)", sub)
        if w is None or not sc:
            return None
        x, y = int(sc.group(1)), int(sc.group(2))
        if w == 0:
            return markov.prob(D, lambda sa, sb, *_: sa == x and sb == y), f"{m['names'][0]} wins {x}-{y}"
        return markov.prob(D, lambda sa, sb, *_: sb == x and sa == y), f"{m['names'][1]} wins {x}-{y}"
    if "SETWINNER" in series:
        w = who(sub)
        n = dm["event_ticker"].rsplit("-", 1)[-1]
        if w is None or n not in ("1", "2"):
            return None
        idx = 4 if n == "1" else 5
        pa = markov.prob(D, lambda *k: k[idx] == 1) / max(markov.prob(D, lambda *k: k[idx] >= 0), 1e-9)
        return (pa if w == 0 else 1 - pa), f"{m['names'][w]} wins set {n}"
    return None


def build_combos(board, cfg):
    out = []
    # same match: winner + games total, using the exact joint distribution
    for m in board:
        if m["status"] != "pre" or not m.get("model"):
            continue
        D = markov.mix_dist(m["model"]["pa"], m["model"]["pb"], m["bo"], markov.DAY_SD[m["tour"]])
        fav = 0 if m["model"]["p"] >= 0.5 else 1
        line = round(m["sum"]["games"]) + 0.5
        for over in (True, False):
            j = markov.prob(D, lambda sa, sb, t, d, *_: ((sa > sb) if fav == 0 else (sb > sa)) and ((t > line) if over else (t < line)))
            pw = m["model"]["p"] if fav == 0 else 1 - m["model"]["p"]
            pt = markov.prob(D, lambda sa, sb, t, *_: (t > line) if over else (t < line))
            lift = j / (pw * pt) - 1 if pw * pt > 0 else 0
            if 0.25 <= j <= 0.7 and abs(lift) >= 0.08:
                out.append({"kind": "same", "match": m["id"], "names": m["names"], "tourney": m["tourney"], "start": m["start"],
                            "legs": [f"{m['names'][fav]} to win", f"{'Over' if over else 'Under'} {line} games"], "p": round(j, 4),
                            "fair_c": round(j * 100), "take_at": max_price(j, cfg["strategies"]["combo"]["min_ev"]), "lift": round(lift, 3)})
    out.sort(key=lambda c: -c["lift"])
    same = out[:10]
    # two matches: pair the best straight picks (independent legs)
    legs = []
    for m in board:
        for b in m.get("bets", []):
            if b["kind"] == "match" and b["tags"]:
                legs.append((m, b))
    legs.sort(key=lambda x: -x[1]["ev"])
    pairs = []
    for i in range(min(len(legs), 8)):
        for k in range(i + 1, min(len(legs), 8)):
            (m1, b1), (m2, b2) = legs[i], legs[k]
            if m1["id"] == m2["id"]:
                continue
            j = b1["p"] * b2["p"]
            price = b1["ask"] * b2["ask"]
            pairs.append({"kind": "parlay", "legs": [b1["label"], b2["label"]], "tickers": [b1["ticker"], b2["ticker"]],
                          "names": [m1["names"], m2["names"]], "p": round(j, 4), "fair_c": round(j * 100),
                          "take_at": max_price(j, cfg["strategies"]["combo"]["min_ev"]), "ev_at_asks": round(j / (price + fee(price)) - 1, 4),
                          "start": min(m1["start"], m2["start"])})
    pairs.sort(key=lambda c: -c["ev_at_asks"])
    return same + pairs[:6]


# ------------------------------------------------------------------ paper record
def record_picks(log, board, combos):
    B = log.setdefault("bets", {})
    for m in board:
        started = m["status"] != "pre"
        for b in m.get("bets", []):
            k = b["ticker"]
            e = B.get(k)
            if e is None and b["tags"] and not started:
                B[k] = e = {"ticker": k, "label": b["label"], "names": m["names"], "tourney": m["tourney"], "tour": m["tour"], "level": m["level"],
                            "kind": b["kind"], "tags": b["tags"], "first_ts": NOW, "entry": b["ask"], "p": b["p"], "ev": b["ev"], "units": b["units"],
                            "close": b["ask"], "close_p": b["p"], "start": m["start"], "result": None}
            if e is not None and e["result"] is None and not started:
                e["close"] = b["ask"]               # last pre-match ask = closing line
                e["close_p"] = b["p"]
                e["start"] = m["start"]
                for t in b["tags"]:
                    if t not in e["tags"]:
                        e["tags"].append(t)


def settle(log):
    B = log.get("bets", {})
    open_ = [k for k, e in B.items() if e["result"] is None and NOW > e["start"] - 3600]
    for i in range(0, len(open_), 50):
        chunk = open_[i:i + 50]
        d = src.kalshi("/markets", {"tickers": ",".join(chunk), "limit": 100})
        for mk in d.get("markets", []):
            e = B.get(mk["ticker"])
            if not e:
                continue
            r = mk.get("result")
            if r in ("yes", "no"):
                e["result"] = r
                c = e["entry"] + fee(e["entry"])
                e["pnl_u"] = round(e["units"] * ((1 - c) / c if r == "yes" else -1), 3) if e["units"] else 0
                e["flat_ret"] = round((1 - c) / c if r == "yes" else -1, 4)
                e["settled_ts"] = NOW
            elif mk.get("status") in ("settled", "finalized") and r in ("", None, "void"):
                e["result"] = "void"
    # keep the file bounded: drop settled entries older than a year
    for k in [k for k, e in B.items() if e.get("settled_ts") and NOW - e["settled_ts"] > 365 * 86400]:
        del B[k]


def summarize(log):
    out = {}
    for e in log.get("bets", {}).values():
        if e["result"] not in ("yes", "no"):
            continue
        clv = (e["close"] - e["entry"]) if e.get("close") is not None and e.get("entry") is not None else 0.0
        for t in e["tags"] + ["all"]:
            s = out.setdefault(t, {"n": 0, "w": 0, "units_risked": 0.0, "pnl_u": 0.0, "flat": 0.0, "clv": 0.0})
            s["n"] += 1; s["w"] += e["result"] == "yes"
            s["units_risked"] += e["units"] or 0; s["pnl_u"] += e.get("pnl_u") or 0
            s["flat"] += e["flat_ret"]; s["clv"] += clv
    for s in out.values():
        s["roi_flat"] = round(s["flat"] / s["n"], 4) if s["n"] else 0
        s["roi_units"] = round(s["pnl_u"] / s["units_risked"], 4) if s["units_risked"] else 0
        s["clv_c"] = round(100 * s["clv"] / s["n"], 2) if s["n"] else 0
        s["pnl_u"] = round(s["pnl_u"], 2)
    pend = [e for e in log.get("bets", {}).values() if e["result"] is None]
    last = sorted([e for e in log.get("bets", {}).values() if e["result"] in ("yes", "no")], key=lambda e: -e.get("settled_ts", 0))[:40]
    return {"by_strategy": out, "pending": len(pend), "recent": last}


def recent_results(log):
    return {k: e["result"] for k, e in log.get("bets", {}).items() if e["result"]}


def settled_map(data, cfg, notes):
    """ticker -> yes/no for every tennis market settled in the last 21 days, so the page can grade your own bets."""
    f = data / "settled.json"
    try:
        old = json.loads(f.read_text()) if f.exists() else {}
    except ValueError:
        old = {}
    min_close = int(NOW - 2 * 86400) if old else int(NOW - 21 * 86400)
    for s in cfg["match_series"] + cfg["deriv_series"]:
        try:
            for mk in src.kalshi_markets(s, status="settled", max_pages=10, min_close_ts=min_close):
                if mk.get("result") in ("yes", "no"):
                    old[mk["ticker"]] = [mk["result"], tsf(mk.get("close_time") or iso(NOW))]
        except Exception as e:
            notes.append(f"settled {s} failed: {e}")
    old = {k: v for k, v in old.items() if NOW - v[1] < 21 * 86400}
    f.write_text(json.dumps(old))
    return {k: v[0] for k, v in old.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "out"))
    ap.add_argument("--no-smart", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text())
    run(Path(a.data), cfg, do_smart=not a.no_smart)


if __name__ == "__main__":
    main()
