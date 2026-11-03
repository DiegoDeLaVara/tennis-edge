"""Backtest tournament-winner markets: simulate each settled ATP/WTA draw from ratings as of its first match, compare
with Kalshi's winner-market prices at that moment, and bet the gaps at the real ask.

  .venv\\Scripts\\python.exe -m tools.futures_test
Needs work/elo_snaps.json.gz (tools.build_history), work/kalshi_events.jsonl and model/coef.json.
"""
import gzip
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from edge import draws, futures, sources as src
from edge.model import Model
from edge.names import Index, key
from tools.kalshi_backtest import load

W = Path(__file__).resolve().parents[1] / "work"


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def fee(p):
    return 0.07 * p * (1 - p)


def lg(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def all_markets(series):
    out = src.kalshi_markets(series, status="settled", max_pages=10)
    cur = ""
    for _ in range(20):
        p = {"series_ticker": series, "limit": 1000}
        if cur:
            p["cursor"] = cur
        d = src.kalshi("/historical/markets", p)
        ms = d.get("markets") or []
        out += ms
        cur = d.get("cursor") or ""
        if not cur or not ms:
            break
    return out


def main():
    import argparse, copy
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="start", choices=["start", "r16", "qf"])
    args = ap.parse_args()
    snaps = json.load(gzip.open(W / "elo_snaps.json.gz", "rt"))
    coef = Model().w
    K = load()
    ev_rows = [json.loads(l) for l in open(W / "kalshi_events.jsonl")]
    hist_cut = ts(src.kalshi("/historical/cutoff").get("trades_created_ts", "2026-09-03T00:00:00Z"))
    events = {}
    for series, women in (("KXATP", False), ("KXWTA", True)):
        for m in all_markets(series):
            if m.get("result") in ("yes", "no"):
                events.setdefault((series, women, m["event_ticker"]), []).append(m)
    print("settled winner events", len(events))
    rows, evsum = [], []
    for (series, women, et), ms in sorted(events.items()):
        e = src.kalshi("/events/" + et).get("event", {})
        sub = e.get("sub_title") or e.get("title") or ""
        mt = re.match(r"(\d{4})\s+(.+)", sub)
        if not mt:
            continue
        year, tname = int(mt.group(1)), re.sub(r"\s+Winner$", "", mt.group(2)).strip()
        tour = "wta" if women else "atp"
        # first match of this tournament in our Kalshi match data
        city = re.sub(r"^(ATP|WTA)\s+", "", tname).lower()
        starts = [x["start"] for x in ev_rows if x.get("tourney") and city in x["tourney"].lower() and str(year) in
                  datetime.fromtimestamp(x["start"], timezone.utc).strftime("%Y") and
                  x["series"] == ("KXWTAMATCH" if women else "KXATPMATCH")]
        if not starts:
            continue
        T = min(starts)
        if args.stage != "start":
            want = "QF" if args.stage == "qf" else "R16"
            st2 = [x["start"] for x in ev_rows if x.get("tourney") and city in x["tourney"].lower() and x.get("round") == want and
                   x["series"] == ("KXWTAMATCH" if women else "KXATPMATCH") and x["start"] >= T and x["start"] - T < 20 * 86400]
            if not st2:
                continue
            T = min(st2)
        page = draws.find_page(year, tname, women)
        D = draws.parse_draw(draws.wikitext(page)) if page else None
        if not D or sum(1 for x in D.levels[0] if x and x != "BYE") < 16:
            print("  skip (no draw)", et, tname, page)
            continue
        use_res = False
        if args.stage != "start":
            depth = len(D.levels) - 1
            lvl = depth - (3 if args.stage == "qf" else 4)     # level holding the 8 (or 16) players left
            if lvl < 1 or sum(1 for x in D.levels[lvl] if x) < len(D.levels[lvl]) * 0.75:
                continue
            D = copy.deepcopy(D)
            for k in range(lvl + 1, depth + 1):
                D.levels[k] = [None] * len(D.levels[k])
            use_res = True
        day = str(date.fromtimestamp(T).toordinal())
        S = snaps.get(day)
        if not S:
            continue
        idx = Index(S.keys())
        surf = "Clay" if any(x in tname.lower() for x in ("roland", "french", "rome", "madrid", "hamburg", "gstaad", "bastad", "kitzbuhel", "umag", "bucharest", "rabat", "strasbourg", "geneva", "lyon", "palermo", "prague", "iasi", "parma", "warsaw")) \
            else "Grass" if any(x in tname.lower() for x in ("wimbledon", "halle", "queen", "eastbourne", "mallorca", "hertogenbosch", "berlin", "bad homburg", "nottingham", "newport", "stuttgart")) and year == 2026 and T < datetime(2026, 7, 20, tzinfo=timezone.utc).timestamp() else "Hard"
        bo = 5 if (not women and any(x in tname.lower() for x in ("open", "wimbledon")) and any(x in tname.lower() for x in ("us open", "australian", "french", "wimbledon"))) else 3

        @lru_cache(None)
        def kk(n):
            return idx.resolve(n, tour)

        def base(a, b):
            A, B = S.get(kk(a)), S.get(kk(b))
            if not A or not B:
                return 0.0
            ea = 0.5 * A[0] + 0.5 * A[1].get(surf, A[0])
            eb = 0.5 * B[0] + 0.5 * B[1].get(surf, B[0])
            pb_ = 1 / (1 + 10 ** ((eb - ea) / 400))
            pa_ = 1 / (1 + 10 ** ((B[0] - A[0]) / 400))
            z = coef[0] * lg(pb_) + coef[1] * lg(pa_) + coef[4] * (A[3] - B[3]) + (coef[9] * lg(pb_) if women else 0)
            if bo == 5:
                z *= 1.15                         # longer matches favour the better player
            return z

        def unknown_adj(a, b, z):
            if kk(a) not in S and kk(b) in S:
                return -1.5
            if kk(b) not in S and kk(a) in S:
                return 1.5
            return z

        # market nudges from first-round Kalshi prices already quoted at T
        delta = {}
        for r in K.itertuples():
            if r.series != ("KXWTAMATCH" if women else "KXATPMATCH") or not r.qts or r.qts > T + 600 or abs(r.qts - T) > 1.5 * 86400:
                continue
            ka, kb = idx.resolve(r.ka.split(":", 1)[1], tour), idx.resolve(r.kb.split(":", 1)[1], tour)
            if ka not in S or kb not in S:
                continue
            zm = base(r.ka.split(":", 1)[1], r.kb.split(":", 1)[1])
            d = lg(r.px) - zm
            delta[ka] = delta.get(ka, 0) + d / 2
            delta[kb] = delta.get(kb, 0) - d / 2

        def p_model(a, b):
            return 1 / (1 + math.exp(-unknown_adj(a, b, base(a, b))))

        def p_mkt(a, b):
            z = unknown_adj(a, b, base(a, b)) + delta.get(kk(a), 0) - delta.get(kk(b), 0)
            return 1 / (1 + math.exp(-z))

        Rm = futures.simulate(D, p_model, use_results=use_res)
        Rk = futures.simulate(D, p_mkt, use_results=use_res)
        wkey = {kk(n): n for n in Rm}

        def quote(m):
            end = ts(m.get("close_time") or m["expected_expiration_time"])
            c = src.kalshi_candles(series, m["ticker"], T - 3 * 86400, T, 60, historical=end < hist_cut)
            q = None
            for x in c:
                if x["end_period_ts"] <= T and x["ask"] is not None and x["bid"] is not None and 0 <= x["bid"] < x["ask"] <= 1:
                    q = (x["bid"], x["ask"])
            return m, q
        with ThreadPoolExecutor(8) as ex:
            qs = list(ex.map(quote, ms))
        champ = None
        for m, q in qs:
            nm = wkey.get(kk(m.get("yes_sub_title") or ""))
            y = 1 if m["result"] == "yes" else 0
            if y:
                champ = nm
            if not nm or not q:
                continue
            rows.append({"event": et, "date": T, "player": nm, "y": y, "bid": q[0], "ask": q[1], "fm": Rm[nm][-1], "fk": Rk[nm][-1]})
        evsum.append((et, tname, page, len(D.levels[0]), champ, len(delta)))
        print(f"  {et:20s} {tname:28s} draw {len(D.levels[0]):3d}  priced players {sum(1 for r in rows if r['event'] == et):3d}  nudged {len(delta):3d}  champ {champ}", flush=True)
    json.dump(rows, open(W / f"futures_rows_{args.stage}.json", "w"))
    report(rows)


def report(rows):
    import pandas as pd
    R = pd.DataFrame(rows)
    R = R[(R.ask > 0)]
    two = R[(R.bid > 0) & (R.ask - R.bid <= 0.10)]
    two = two.assign(mid=(two.bid + two.ask) / 2)
    y = two.y.values

    def ll(p):
        p = np.clip(p, 0.002, 0.998)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"\nplayer markets with a two-sided quote: {len(two)} across {two.event.nunique()} tournaments")
    print(f"  log-loss  Kalshi mid {ll(two.mid.values):.4f}  model {ll(two.fm.values):.4f}  model+R1 market {ll(two.fk.values):.4f}")
    for col, lab in (("fm", "model"), ("fk", "model + R1 market")):
        for th in (0.05, 0.15, 0.30):
            res = []
            for r in R.itertuples():
                f = getattr(r, col)
                if 0.02 <= r.ask <= 0.95 and f / (r.ask + fee(r.ask)) - 1 >= th:
                    c = r.ask + fee(r.ask)
                    res.append((1 - c) / c if r.y else -1.0)
                if r.bid > 0.02:
                    pn = 1 - r.bid
                    if pn <= 0.98 and (1 - f) / (pn + fee(pn)) - 1 >= th:
                        c = pn + fee(pn)
                        res.append((1 - c) / c if not r.y else -1.0)
            if res:
                a = np.array(res)
                print(f"  bet {lab:18s} edge >= {int(th * 100):2d}%: n={len(a):4d} ROI {a.mean() * 100:+7.1f}% ± {a.std() / np.sqrt(len(a)) * 100:5.1f}   (wins {int((a > 0).sum())})")


if __name__ == "__main__":
    main()
