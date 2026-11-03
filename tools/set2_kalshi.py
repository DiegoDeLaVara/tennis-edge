r"""Back the set-1 loser to win set 2, on Kalshi's settled set-winner markets. The set-1 market closes when set 1
ends, so the set-2 ask a minute or two later is a price you could actually have taken. Large-sample version of
tools.set2_test (which only had the observer's recording).

  .venv\Scripts\python.exe -m tools.set2_kalshi [--max 2500]
"""
import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from edge import sources as src
from edge.names import key
from tools.kalshi_backtest import load

W = Path(__file__).resolve().parents[1] / "work"
OUT = W / "set2_rows.jsonl"


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def fee(p):
    return 0.07 * p * (1 - p)


def settled(series):
    out = src.kalshi_markets(series, status="settled", max_pages=30)
    cur = ""
    for _ in range(40):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=2500)
    args = ap.parse_args()
    hist_cut = ts(src.kalshi("/historical/cutoff").get("trades_created_ts", "2026-09-03T00:00:00Z"))
    K = load()
    pre = {r.event.split("-")[1]: r for r in K.itertuples()}
    done = {json.loads(l)["key"] for l in OUT.open()} if OUT.exists() else set()
    jobs = []
    for series in ("KXATPSETWINNER", "KXWTASETWINNER"):
        ms = [m for m in settled(series) if m.get("result") in ("yes", "no")]
        sets = {}
        for m in ms:
            base, n = m["event_ticker"].rsplit("-", 1)
            sets.setdefault(base, {}).setdefault(n, []).append(m)
        cand = []
        for base, d in sets.items():
            if "1" not in d or "2" not in d or len(d["1"]) != 2 or len(d["2"]) != 2:
                continue
            w1 = next((m for m in d["1"] if m["result"] == "yes"), None)
            if not w1:
                continue
            end1 = ts(w1.get("close_time") or w1["expected_expiration_time"])
            loser_name = next(m["yes_sub_title"] for m in d["1"] if m is not w1)
            l2 = next((m for m in d["2"] if m["yes_sub_title"] == loser_name), None)
            if not l2:
                continue
            cand.append((series, base, end1, l2))
        cand.sort(key=lambda x: -x[2])
        jobs += [c for c in cand[:args.max] if c[1] not in done]
        print(series, "matches with set 1 and set 2 markets", len(cand), flush=True)
    print("to fetch", len(jobs), flush=True)

    def one(job):
        series, base, end1, l2 = job
        end2 = ts(l2.get("close_time") or l2["expected_expiration_time"])
        c = src.kalshi_candles(series, l2["ticker"], end1 - 120, end1 + 900, 1, historical=end2 < hist_cut)
        q = None
        for x in c:
            if x["end_period_ts"] >= end1 + 60 and x["ask"] is not None and x["bid"] is not None and 0 < x["ask"] <= 1:
                q = (x["bid"], x["ask"], x["end_period_ts"] - end1)
                break
        suf = base.split("-")[1]
        r = pre.get(suf)
        return {"key": base, "series": series, "loser": l2["yes_sub_title"], "won": l2["result"] == "yes", "q": q,
                "pre_loser": (None if r is None else (r.px if r.ka == key(l2["yes_sub_title"], "wta" if "WTA" in series else "atp") else 1 - r.px)),
                "date": end1}
    with OUT.open("a") as fh, ThreadPoolExecutor(8) as ex:
        for i, d in enumerate(ex.map(one, jobs)):
            fh.write(json.dumps(d) + "\n")
            if i % 500 == 0:
                fh.flush(); print(i, flush=True)
    report()


def report():
    rows = [json.loads(l) for l in OUT.open()]
    rows = [r for r in rows if r["q"] and r["q"][1] - r["q"][0] <= 0.10 and 0.03 <= r["q"][1] <= 0.97]
    rows.sort(key=lambda r: r["date"])
    print(f"\nset-1 losers with a set-2 quote within 15 min: {len(rows)}")

    def run(sub, label):
        out = []
        for r in sub:
            a = r["q"][1]
            c = a + fee(a)
            out.append((1 - c) / c if r["won"] else -1.0)
        a = np.array(out)
        if len(a):
            wins = np.mean([r["won"] for r in sub])
            price = np.mean([r["q"][1] for r in sub])
            print(f"  {label:44s} n={len(a):5d} won {wins:.3f} vs avg ask {price:.3f}  ROI {a.mean() * 100:+6.1f}% ± {a.std() / math.sqrt(len(a)) * 100:4.1f}")
    half = len(rows) // 2
    for part, R in (("all", rows), ("older half", rows[:half]), ("newer half", rows[half:])):
        print(f" -- {part}")
        run(R, "back every set-1 loser in set 2")
        run([r for r in R if r["pre_loser"] is not None and r["pre_loser"] >= 0.6], "...loser was the favourite (60%+ pre-match)")
        run([r for r in R if r["pre_loser"] is not None and r["pre_loser"] <= 0.4], "...loser was the underdog (40%- pre-match)")
        for s in ("KXATPSETWINNER", "KXWTASETWINNER"):
            run([r for r in R if r["series"] == s], f"...{s[2:5]} only")
        for lo, hi in ((0.03, 0.3), (0.3, 0.5), (0.5, 0.97)):
            run([r for r in R if lo <= r["q"][1] < hi], f"...set-2 ask {int(lo * 100)}-{int(hi * 100)}c")


if __name__ == "__main__":
    main()
