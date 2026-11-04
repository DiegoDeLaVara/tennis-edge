r"""Post-mortem for one day: could the upsets have been seen coming? Re-scores every Kalshi tennis match that settled
that day with the ratings as they stood BEFORE the day (state/ratings.json.gz seed, or --state <file>), the market
ratings seed, and the board's risk flags, then checks whether any of them pointed at the upsets.

  .venv\Scripts\python.exe -m tools.postmortem [YYYY-MM-DD]
"""
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from edge import context, sources as src
from edge.model import Model, logit, sig
from edge.mrating import MR
from edge.names import Index
from edge.refresh import load_state, risk_flags
from tools.today_upsets import SERIES, ET, pre_mid

LEVEL = {"KXATPMATCH": "A", "KXWTAMATCH": "A", "KXATPCHALLENGERMATCH": "C", "KXWTACHALLENGERMATCH": "C", "KXITFMATCH": "S", "KXITFWMATCH": "S"}


def main():
    day = datetime.fromisoformat(sys.argv[1]).date() if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else datetime.now(ET).date()
    lo = datetime(day.year, day.month, day.day, tzinfo=ET).timestamp()
    R, surfaces, seen, st = load_state(Path("no-such-dir"))          # forces the pre-day seed
    print(f"ratings seed covers results through {st.get('updated') or st.get('generated') or 'the seed build'}")
    idx = Index(R.P.keys())
    M = Model()
    MRT = MR.load(Path("no-such-dir"))
    t = day.toordinal()
    events = defaultdict(list)
    for s in SERIES:
        for mk in src.kalshi_markets(s, status="settled", max_pages=5, min_close_ts=int(lo)):
            ct = datetime.fromisoformat(mk["close_time"].replace("Z", "+00:00")).timestamp()
            if lo <= ct < lo + 86400 and mk.get("result") in ("yes", "no"):
                events[(s, mk["event_ticker"])].append(mk)
    jobs = [(s, mks[0], mks) for (s, ev), mks in events.items() if len(mks) == 2]
    with ThreadPoolExecutor(6) as ex:
        mids = list(ex.map(lambda j: pre_mid(j[0], j[1]), jobs))
    rows = []
    for (s, mk, mks), p in zip(jobs, mids):
        if p is None:
            continue
        a = mk.get("yes_sub_title"); b = next(x for x in mks if x is not mk).get("yes_sub_title")
        tour = "wta" if "WTA" in s or s == "KXITFWMATCH" else "atp"
        ka, kb = idx.resolve(a, tour), idx.resolve(b, tour)
        y = 1 if mk["result"] == "yes" else 0
        m = {"names": [a, b], "p_mkt": p, "bo": 3, "series": s, "model": {}, "ctx": None}
        pm = pmr = None
        if ka in R.P and kb in R.P and R.P[ka].n >= 3 and R.P[kb].n >= 3:
            f = R.features(ka, kb, tour, "Hard", t, 3, LEVEL[s])
            if f:
                pm = M.prob(f, tour, LEVEL[s], 3)[0]
                pmr = sig(MRT.blend(logit(pm), ka, kb, t))
                m["model"] = {"p_model": pm, "form": [f["form_a"], f["form_b"]]}
                m["ctx"] = context.match_context(R, ka, kb, "", "Hard", t)
        r = risk_flags(m) or {"level": "normal", "stay": [], "warn": [], "good": []}
        fi = 0 if p >= 0.5 else 1
        rows.append({"s": s, "fav": m["names"][fi], "dog": m["names"][1 - fi], "pf": max(p, 1 - p), "fw": y if fi == 0 else 1 - y,
                     "pm_f": None if pm is None else (pm if fi == 0 else 1 - pm),
                     "pmr_f": None if pmr is None else (pmr if fi == 0 else 1 - pmr), "risk": r})
    n = len(rows)
    ups = [x for x in rows if x["fw"] == 0]
    exp = sum(1 - x["pf"] for x in rows)
    print(f"\n{day}: {n} settled matches with a clean pre-match price; upsets {len(ups)} vs {exp:.1f} expected from the prices\n")
    print("Upsets (favourite lost), biggest first:")
    for x in sorted(ups, key=lambda x: -x["pf"]):
        mm = "  -   " if x["pm_f"] is None else f"{x['pm_f']:.0%}"
        fl = x["risk"]["level"]
        print(f"   {x['s']:<21} {x['dog'][:22]:<22} beat {x['fav'][:22]:<22} market fav {x['pf']:.0%}  our model fav {mm:>4}  flag {fl}")

    def grp(label, sel):
        g = [x for x in rows if sel(x)]
        if not g:
            return
        u = sum(1 - x["fw"] for x in g); e = sum(1 - x["pf"] for x in g)
        print(f"   {label:<58} n={len(g):>3}  upsets {u:>3} vs {e:5.1f} expected  ({(u - e) / max(1, len(g)) * 100:+.0f} pts per match)")
    print("\nDid any signal point at them?")
    grp("our model rated the market favourite LOWER by 5+ pts", lambda x: x["pm_f"] is not None and x["pm_f"] <= x["pf"] - 0.05)
    grp("our model agreed or rated the favourite higher", lambda x: x["pm_f"] is not None and x["pm_f"] > x["pf"] - 0.05)
    grp("our model picked the underdog outright", lambda x: x["pm_f"] is not None and x["pm_f"] < 0.5)
    grp("flagged stay-away", lambda x: x["risk"]["level"] == "stay")
    grp("flagged upset risk", lambda x: x["risk"]["level"] == "warn")
    grp("flagged steadier than priced", lambda x: x["risk"]["level"] == "good")
    grp("no flag", lambda x: x["risk"]["level"] == "normal")
    sc = [x for x in rows if x["pm_f"] is not None]
    if sc:
        L = lambda pr: -sum(math.log(pr(x) if x["fw"] else 1 - pr(x)) for x in sc) / len(sc)
        print(f"\nlog-loss today on {len(sc)} rated matches: Kalshi price {L(lambda x: x['pf']):.4f} | our model {L(lambda x: x['pm_f']):.4f} | "
              f"model + market ratings {L(lambda x: x['pmr_f']):.4f}   (lower = better)")


if __name__ == "__main__":
    main()
