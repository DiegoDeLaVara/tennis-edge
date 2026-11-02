"""Add the pre-match bid/ask (from hourly candles) for events collected before the collector saved quotes.
Appends to work/quotes.jsonl as it goes, so it can be stopped and resumed."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edge import sources as src
from tools.kalshi_collect import quote_before

W = Path(__file__).resolve().parents[1] / "work"
F, Q = W / "kalshi_events.jsonl", W / "quotes.jsonl"


def fix(e):
    ms = src.kalshi("/markets", {"event_ticker": e["event"]}).get("markets") or \
        src.kalshi("/historical/markets", {"event_ticker": e["event"]}).get("markets") or []
    if len(ms) != 2:
        return {"event": e["event"], "quote": None}
    a = sorted(ms, key=lambda m: m["ticker"])[0]
    c = src.kalshi_candles(e["series"], a["ticker"], e["start"] - 3 * 86400, e["start"] + 3600, 60)
    return {"event": e["event"], "quote": quote_before(c, e["start"])}


def main():
    src.KALSHI_PACE.gap = 1 / 5.0
    done = {json.loads(l)["event"] for l in Q.open()} if Q.exists() else set()
    todo = [json.loads(l) for l in F.open()]
    todo = [e for e in todo if "quote" not in e and e["event"] not in done and e.get("start")]
    print("to fill", len(todo), flush=True)
    with Q.open("a") as fh, ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(fix, todo)):
            fh.write(json.dumps(r) + "\n")
            if i % 100 == 0:
                fh.flush(); print(i, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
