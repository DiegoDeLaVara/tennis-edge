"""Collect settled Kalshi tennis matches with their pre-match price path and trade tape, for backtesting the
model and the money-flow signals against the prices you can actually trade.

  .venv\\Scripts\\python.exe -m tools.kalshi_collect [--since 2025-06-01]
Writes work/kalshi_events.jsonl (resumable).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from edge import sources as src
from edge.money import flow_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "kalshi_events.jsonl"
SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXWTACHALLENGERMATCH", "KXATPCHALLENGERMATCH"]


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def historical(series, since_ts, max_pages=60):
    out, cur = [], ""
    for _ in range(max_pages):
        p = {"series_ticker": series, "limit": 1000}
        if cur:
            p["cursor"] = cur
        d = src.kalshi("/historical/markets", p)
        ms = d.get("markets") or []
        out += ms
        cur = d.get("cursor") or ""
        if not cur or not ms or ts(ms[-1]["close_time"]) < since_ts:
            break
    return out


def trades(ticker, historical_, max_ts):
    path = "/historical/trades" if historical_ else "/markets/trades"
    out, cur = [], ""
    for _ in range(10):
        p = {"ticker": ticker, "limit": 1000, "max_ts": int(max_ts)}
        if cur:
            p["cursor"] = cur
        d = src.kalshi(path, p)
        tr = d.get("trades") or []
        out += tr
        cur = d.get("cursor") or ""
        if not cur or not tr:
            break
    return out


def start_from_candles(c):
    """Start of the first hour where volume surges (in-play began); None if never."""
    v = [(x["end_period_ts"] - 3600, float(x.get("volume_fp") or 0)) for x in c]
    for i in range(len(v) - 1):
        prior = sorted(x for _, x in v[max(0, i - 24):i] if x > 0)
        med = prior[len(prior) // 2] if prior else 0.0
        if v[i][1] >= max(2500.0, 8 * med) and v[i + 1][1] >= max(1500.0, 3 * med):
            return v[i][0]
    return None


def quote_before(c, start):
    """A's yes bid/ask at the close of the last full hour before the start (what you could actually trade)."""
    pre = [x for x in c if x["end_period_ts"] <= start]
    for x in reversed(pre):
        try:
            b, a = float(x["yes_bid"]["close_dollars"]), float(x["yes_ask"]["close_dollars"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 < a <= 1 and 0 <= b < a:
            return {"bid": b, "ask": a, "ts": x["end_period_ts"]}
    return None


def one(series, et, ms, hist_cut):
    a, b = sorted(ms, key=lambda m: m["ticker"])
    end = ts(a.get("close_time") or a["expected_expiration_time"])
    opened = ts(a["open_time"])
    c = src.kalshi_candles(series, a["ticker"], opened - 3600, end + 3600, 60)
    start = start_from_candles(c)
    if start is None:                      # thin market: treat the last 2 hours before close as in-play
        start = end - 2 * 3600
    q = quote_before(c, start)
    is_hist = start < hist_cut
    ta = trades(a["ticker"], is_hist, start)
    tb = trades(b["ticker"], is_hist, start)
    tname, rnd = src.parse_rules(a)
    fl = flow_features(ta, tb, start)
    return {"event": et, "series": series, "end": end, "open": opened, "start": start, "a": a["yes_sub_title"], "b": b["yes_sub_title"],
            "y": 1 if a.get("result") == "yes" else 0, "tourney": tname, "round": rnd, "flow": fl, "quote": q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2025-06-01")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    since_ts = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp()
    cut = src.kalshi("/historical/cutoff")
    hist_cut = ts(cut.get("trades_created_ts", "2026-09-03T00:00:00Z"))
    done = set()
    if OUT.exists():
        done = {json.loads(l)["event"] for l in OUT.open()}
    jobs = []
    for s in SERIES:
        ms = src.kalshi_markets(s, status="settled", max_pages=40) + historical(s, since_ts)
        ev = src.group_events(ms)
        n0 = len(jobs)
        for et, pair in ev.items():
            if et in done or any(m.get("result") not in ("yes", "no") for m in pair):
                continue
            if ts(pair[0].get("close_time") or pair[0]["expected_expiration_time"]) < since_ts:
                continue
            jobs.append((s, et, pair))
        print(s, "events", len(ev), "new jobs", len(jobs) - n0, flush=True)
    with OUT.open("a") as fh, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, s, et, pair, hist_cut) for s, et, pair in jobs]
        for i, f in enumerate(as_completed(futs)):
            try:
                fh.write(json.dumps(f.result()) + "\n")
            except Exception as e:  # keep going; a bad market should not stop the run
                print("err", e)
            if i % 250 == 0:
                fh.flush(); print(i, "/", len(jobs), flush=True)


if __name__ == "__main__":
    main()
