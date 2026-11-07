r"""Fresh-data check for tools/dip_test candidates: Kalshi 1-minute candles (bid/ask closes) of Challenger matches played
AFTER the observer recording (30 Sep onward), from work/kalshi_events.jsonl. Same rules: pre-match favourite (55%+)
whose ask falls D cents below the pre-match price -> buy at that ask (+fee), hold to settlement or take profit.

  .venv\Scripts\python.exe -m tools.dip_validate [--n 600] [--rate 1]
Caches candles in work/dip_candles.jsonl (resumable).
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from edge import sources as src

CACHE = Path("work/dip_candles.jsonl")
SERIES = ("KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH")


def fee(p):
    return 0.07 * p * (1 - p)


def a_ticker(event):
    code = event.split("-")[-1][7:]                     # 26NOV01 + codes
    if len(code) != 6:
        return None
    return f"{event}-{min(code[:3], code[3:])}"


def fetch(args):
    cut = src.kalshi("/historical/cutoff")
    hist_cut = datetime.fromisoformat(cut.get("trades_created_ts", "2026-09-03T00:00:00Z").replace("Z", "+00:00")).timestamp()
    done = {json.loads(l)["event"] for l in CACHE.open()} if CACHE.exists() else set()
    rows = [json.loads(l) for l in open("work/kalshi_events.jsonl")]
    since = datetime(2026, 9, 30, tzinfo=timezone.utc).timestamp()
    rows = [r for r in rows if r["series"] in SERIES and r["start"] >= since and r.get("quote") and r["event"] not in done]
    rows = sorted(rows, key=lambda r: -r["start"])[:max(0, args.n - len(done))]
    src.KALSHI_PACE.gap = 1 / args.rate
    print(f"fetching 1-min candles for {len(rows)} Challenger matches ({len(done)} cached)", flush=True)
    with CACHE.open("a") as fh:
        for i, r in enumerate(rows):
            tk = a_ticker(r["event"])
            if not tk:
                continue
            try:
                c = src.kalshi_candles(r["series"], tk, r["start"] - 120, r["end"] + 60, 1, historical=r["end"] < hist_cut)
            except Exception as e:  # noqa
                print("err", r["event"], e); continue
            q = r["quote"]
            fh.write(json.dumps({"event": r["event"], "start": r["start"], "y": r["y"], "p0": (q["bid"] + q["ask"]) / 2,
                                 "c": [(x["end_period_ts"], x["bid"], x["ask"]) for x in c if x["bid"] is not None and x["ask"] is not None]}) + "\n")
            if i % 100 == 0:
                fh.flush(); print(i, flush=True)


def path_for(r, fav_is_a):
    out = []
    for ts, b, a in r["c"]:
        if not (0 <= b < a <= 1) or a - b > 0.15:
            continue
        out.append((ts, b, a) if fav_is_a else (ts, 1 - a, 1 - b))   # B's book = mirror of A's
    return out


def trade(P, i, won, tp):
    a = P[i][2]; cost = a + fee(a)
    if tp is not None:
        for _, b, _ in P[i + 1:]:
            if b >= a + tp:
                return (b - fee(b)) / cost - 1
    return (1.0 if won else 0.0) / cost - 1


def evaluate():
    R = [json.loads(l) for l in CACHE.open()]
    R = [r for r in R if len(r["c"]) >= 10 and 0.03 < r["p0"] < 0.97]
    print(f"\n{len(R)} fresh Challenger matches with in-play 1-minute prices")
    from tools.dip_test import our_model
    ours = our_model()
    print(f"our model priced {sum(r['event'] in ours for r in R)} of them")
    WHO = [("anyone", lambda p0, pm: True), ("pre-match fav (55%+)", lambda p0, pm: p0 >= 0.55),
           ("model fav (60%+)", lambda p0, pm: pm is not None and pm >= 0.60),
           ("model fav (70%+)", lambda p0, pm: pm is not None and pm >= 0.70),
           ("model 5+ pts above market", lambda p0, pm: pm is not None and pm - p0 >= 0.05),
           ("model AND market fav", lambda p0, pm: pm is not None and pm >= 0.60 and p0 >= 0.55)]
    for who, wf in WHO:
        for d in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            for en, tp in (("hold", None), ("TP+15", 0.15), ("TP+20", 0.20)):
                rets = []
                for r in R:
                    for fav_is_a in (True, False):
                        p0 = r["p0"] if fav_is_a else 1 - r["p0"]
                        pm = ours[r["event"]][0] if r["event"] in ours else None
                        pm = None if pm is None else (pm if fav_is_a else 1 - pm)
                        if not wf(p0, pm):
                            continue
                        P = path_for(r, fav_is_a)
                        i = next((k for k, t in enumerate(P) if t[2] <= p0 - d), None)
                        if i is None or i >= len(P) - 1:
                            continue
                        won = r["y"] == (1 if fav_is_a else 0)
                        rets.append(trade(P, i, won, tp))
                if len(rets) >= 20:
                    x = np.array(rets)
                    print(f"   {who:<22} drop {int(d * 100):>2}c  {en:<6} n={len(x):>4}  ROI {x.mean():+6.1%} ± {x.std(ddof=1) / math.sqrt(len(x)):4.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if not args.eval_only:
        fetch(args)
    evaluate()


if __name__ == "__main__":
    main()
