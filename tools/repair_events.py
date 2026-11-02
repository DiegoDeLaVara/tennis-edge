"""Re-collect events whose pre-match quote is missing (mostly markets moved to Kalshi's /historical store),
rewriting work/kalshi_events.jsonl in place at the end and checkpointing to work/repair.jsonl as it goes."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edge import sources as src
from tools.kalshi_collect import one, ts

W = Path(__file__).resolve().parents[1] / "work"
F, R = W / "kalshi_events.jsonl", W / "repair.jsonl"


def job(e, hist_cut):
    ms = src.kalshi("/historical/markets", {"event_ticker": e["event"]}).get("markets") or \
        src.kalshi("/markets", {"event_ticker": e["event"]}).get("markets") or []
    if len(ms) != 2 or any(m.get("result") not in ("yes", "no") for m in ms):
        return {"event": e["event"], "drop": True}
    return one(e["series"], e["event"], ms, hist_cut)


def main():
    series = set(sys.argv[1:]) or {"KXATPMATCH", "KXWTAMATCH"}
    hist_cut = ts(src.kalshi("/historical/cutoff").get("trades_created_ts", "2026-09-03T00:00:00Z"))
    done = {json.loads(l)["event"] for l in R.open()} if R.exists() else set()
    ev = [json.loads(l) for l in F.open()]
    todo = [e for e in ev if e["series"] in series and not e.get("quote") and e["event"] not in done]
    print("to repair", len(todo), flush=True)
    with R.open("a") as fh, ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(lambda e: job(e, hist_cut), todo)):
            fh.write(json.dumps(r) + "\n")
            if i % 200 == 0:
                fh.flush(); print(i, flush=True)
    fixed = {}
    for l in R.open():
        d = json.loads(l)
        fixed[d["event"]] = d
    out = []
    for e in ev:
        d = fixed.get(e["event"])
        if d is None:
            out.append(e)
        elif not d.get("drop"):
            out.append(d)
    F.write_text("".join(json.dumps(e) + "\n" for e in out))
    print("done", sum(1 for e in out if e.get("quote")), "with quotes of", len(out), flush=True)


if __name__ == "__main__":
    main()
