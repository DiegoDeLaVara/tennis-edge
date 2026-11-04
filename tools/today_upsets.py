r"""How upset-heavy was a day? Settled Kalshi tennis matches that closed today (US Eastern), each with its real
pre-match mid from hourly candles, compared with how many upsets those prices expected.

  .venv\Scripts\python.exe -m tools.today_upsets [YYYY-MM-DD]
"""
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from edge import sources as src

SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH", "KXITFMATCH", "KXITFWMATCH"]
ET = timezone(timedelta(hours=-5))


def pre_mid(series, mk):
    close = datetime.fromisoformat(mk["close_time"].replace("Z", "+00:00")).timestamp()
    C = src.kalshi_candles(series, mk["ticker"], close - 36 * 3600, close)
    C = [c for c in C if c.get("bid") is not None and c.get("ask") is not None]
    if len(C) < 3:
        return None
    vmax = max(c["volume"] or 0 for c in C)
    for i, c in enumerate(C):
        if (c["volume"] or 0) >= 0.25 * vmax and i > 0:            # trading surge = match under way
            p = C[i - 1]
            if 0 < p["ask"] - p["bid"] <= 0.12:
                return (p["bid"] + p["ask"]) / 2
            return None
    return None


def main():
    day = datetime.fromisoformat(sys.argv[1]).date() if len(sys.argv) > 1 else datetime.now(ET).date()
    lo = datetime(day.year, day.month, day.day, tzinfo=ET).timestamp()
    hi = lo + 86400
    jobs = []
    for s in SERIES:
        seen = set()
        for mk in src.kalshi_markets(s, status="settled", max_pages=5, min_close_ts=int(lo)):
            ct = datetime.fromisoformat(mk["close_time"].replace("Z", "+00:00")).timestamp()
            if not (lo <= ct < hi) or mk.get("result") not in ("yes", "no") or mk["event_ticker"] in seen:
                continue
            seen.add(mk["event_ticker"])
            jobs.append((s, mk))
    with ThreadPoolExecutor(6) as ex:
        mids = list(ex.map(lambda j: pre_mid(*j), jobs))
    rows = []
    for (s, mk), p in zip(jobs, mids):
        if p is None:
            continue
        y = 1 if mk["result"] == "yes" else 0
        pf = max(p, 1 - p)
        fw = y if p >= 0.5 else 1 - y
        rows.append((s, mk.get("yes_sub_title"), pf, fw))
    print(f"{day}: {len(rows)} settled matches with a clean pre-match price (of {len(jobs)})")
    tot_exp = tot_up = tot_var = 0
    for s in SERIES + ["ALL"]:
        R = [r for r in rows if s == "ALL" or r[0] == s]
        if not R:
            continue
        exp_up = sum(1 - r[2] for r in R)
        up = sum(1 - r[3] for r in R)
        var = sum(r[2] * (1 - r[2]) for r in R)
        z = (up - exp_up) / math.sqrt(var) if var else 0
        print(f"   {s:<22} n={len(R):>3}  upsets {up:>3} vs {exp_up:5.1f} expected  ({z:+.1f} sd)  big upsets (fav 75%+ lost): "
              f"{sum(1 for r in R if r[2] >= 0.75 and r[3] == 0)} vs {sum(1 - r[2] for r in R if r[2] >= 0.75):.1f} expected")


if __name__ == "__main__":
    main()
