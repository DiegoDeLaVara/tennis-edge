r"""Seed for the board's "priced like this before" line: every settled Kalshi match we collected (work/kalshi_events.jsonl,
2025-26, all levels) as per-player [start, pre-match price, won], keyed by rating key. The cloud refresh rolls it forward
(edge/refresh.update_price_history) from matches it saw before the start, adding only matches after seed_end.

  .venv\Scripts\python.exe -m tools.price_history_seed      -> state/price_history.json.gz
Testing (tools/price_record) found this record does NOT predict results; the board shows it as history only.
"""
import gzip
import json
from pathlib import Path

from edge.names import Index

ROOT = Path(__file__).resolve().parents[1]


def main():
    st = json.load(gzip.open(ROOT / "state" / "ratings.json.gz", "rt"))
    idx = Index(st["players"].keys())
    P, seen, end, miss = {}, set(), 0, 0
    for l in open(ROOT / "work" / "kalshi_events.jsonl"):
        e = json.loads(l)
        q = e.get("quote")
        if e["event"] in seen or not q or e.get("y") is None or not (0 < q["bid"] < q["ask"] < 1) or q["ask"] - q["bid"] > 0.15:
            continue
        seen.add(e["event"])
        tour = "wta" if ("WTA" in e["series"] or "ITFW" in e["series"]) else "atp"
        ka, kb = idx.resolve(e["a"], tour), idx.resolve(e["b"], tour)
        if not (ka and kb):
            miss += 1
            continue
        p = round((q["bid"] + q["ask"]) / 2, 3)
        s = int(e["start"])
        P.setdefault(ka, []).append([s, p, int(e["y"])])
        P.setdefault(kb, []).append([s, round(1 - p, 3), 1 - int(e["y"])])
        end = max(end, s)
    for k in P:
        P[k] = sorted(P[k])[-200:]
    H = {"seed_end": end, "p": P, "pending": {}}
    out = ROOT / "state" / "price_history.json.gz"
    out.write_bytes(gzip.compress(json.dumps(H, separators=(",", ":")).encode()))
    print(f"{len(seen) - miss:,} matches, {len(P):,} players ({miss} unmatched names) -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
