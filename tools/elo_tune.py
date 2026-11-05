r"""Tune Elo settings for ITF with a fast replay of the full match history (1985-2026, same rows as build_history).
Every prediction is made before the match updates the ratings (walk-forward). Settings are chosen on ITF matches
2016-21 and judged on 2022-26 (ITF and all levels, so nothing else gets worse).

Tunable: K = K0 / (n + off)^shape (newcomers move faster), level weights for ITF (S) / Challenger (C) / qualifying (Q),
starting rating for players whose first match is at ITF level, and the scoreline (margin) multiplier strength.

  .venv\Scripts\python.exe -m tools.elo_tune
"""
import math
import pickle
import time
from datetime import date
from pathlib import Path

import numpy as np

from edge.ratings import parse_score

CACHE = Path("work/allrows.pkl")


def rows():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    from edge import sources as src
    from tools.build_history import recent_rows, dedupe
    years = range(1985, 2027)
    R = []
    R += src.archive_rows(r"C:\TENNIS\archive\sack", "atp", ["", "_qual_chall", "_futures"], list(years))
    R += src.archive_rows(r"C:\TENNIS\archive\sack", "wta", ["", "_qual_itf"], years)
    R = [m for m in R if not (m["uid"].startswith("atp|2026-") and m["level"] in ("G", "M", "A", "F", "D") and m["round"] and not m["round"].startswith("Q"))]
    surfaces = src.Surfaces()
    for m in R:
        surfaces.learn(m.get("tourney"), m.get("surface"), m["tour"])
    end = {t: max(m["date"] for m in R if m["tour"] == t) for t in ("atp", "wta")}
    rec = recent_rows(min(end.values()) - 10, surfaces)
    rec = [m for m in rec if m["date"] > end[m["tour"]] - 10 or m["uid"].startswith("tml|")]
    allr = dedupe(sorted(R, key=lambda m: m["sort"]) + sorted(rec, key=lambda m: m["sort"]))
    allr.sort(key=lambda m: m["sort"])
    out = []
    for m in allr:
        sc = m.get("score") or ""
        gw, gl, sets, tbs, ret = parse_score(sc)
        if "W/O" in sc or (ret and gw + gl < 6):
            continue
        lvl = m.get("level") or "A"
        if (m.get("round") or "").startswith("Q") and lvl in ("A", "G", "M"):
            lvl = "Q"
        out.append((m["date"], m["tour"], lvl, m["w"], m["l"], gw / (gw + gl) if gw + gl else 0.5, bool(ret)))
    CACHE.write_bytes(pickle.dumps(out))
    return out


def replay(R, P, eval_from, split):
    """Returns per-level log-loss sums for the train window [eval_from, split) and test window [split, ...)."""
    K0, off, shape = P["K0"], P["off"], P["shape"]
    lw = {"S": P["wS"], "C": P["wC"], "Q": P["wQ"], "D": 0.85, "O": 0.85}
    elo, n = {}, {}
    acc = {"tr": [0.0, 0], "te": [0.0, 0], "tr_all": [0.0, 0], "te_all": [0.0, 0]}
    marg = P["marg"]
    initS = P["initS"]
    for d, tour, lvl, w, l, share, ret in R:
        ew = elo.get(w); el = elo.get(l)
        if ew is None:
            ew = elo[w] = initS if lvl == "S" else 1500.0; n[w] = 0
        if el is None:
            el = elo[l] = initS if lvl == "S" else 1500.0; n[l] = 0
        e = 1.0 / (1.0 + 10 ** ((el - ew) / 400.0))
        nw, nl = n[w], n[l]
        if d >= eval_from and nw >= 3 and nl >= 3:
            # log-loss of the eventual winner's probability, but symmetric: use the favourite's view
            v = -math.log(max(e, 1e-6))
            k = "te" if d >= split else "tr"
            acc[k + "_all"][0] += v; acc[k + "_all"][1] += 1
            if lvl == "S":
                acc[k][0] += v; acc[k][1] += 1
        weight = lw.get(lvl, 1.0) * (0.5 if ret else 1.0)
        mult = 1.0 + marg * (share - 0.75) / 0.25 * 0.4 if not ret else 1.0      # marg=1 -> 0.6..1.4 like the live model
        elo[w] = ew + K0 / (nw + off) ** shape * weight * mult * (1 - e)
        elo[l] = el - K0 / (nl + off) ** shape * weight * mult * (1 - e)
        n[w] = nw + 1; n[l] = nl + 1
    return {k: v[0] / max(v[1], 1) for k, v in acc.items()}, {k: v[1] for k, v in acc.items()}


def main():
    t0 = time.time()
    R = rows()
    print(f"{len(R):,} matches ({time.time() - t0:.0f}s)")
    eval_from, split = date(2016, 1, 1).toordinal(), date(2022, 1, 1).toordinal()
    base = {"K0": 250.0, "off": 5.0, "shape": 0.4, "wS": 0.85, "wC": 0.9, "wQ": 0.9, "initS": 1500.0, "marg": 1.0}
    res, cnt = replay(R, base, eval_from, split)
    print(f"counts: ITF train {cnt['tr']:,}  ITF test {cnt['te']:,}  all test {cnt['te_all']:,}")
    print(f"current settings: ITF train {res['tr']:.5f}  ITF test {res['te']:.5f}  all-levels test {res['te_all']:.5f}  ({time.time() - t0:.0f}s)")
    grid = {"K0": [200, 250, 320, 400], "off": [2, 5, 10], "shape": [0.3, 0.4, 0.5, 0.6], "wS": [0.7, 0.85, 1.0, 1.15],
            "wC": [0.8, 0.9, 1.0], "wQ": [0.8, 0.9, 1.0], "initS": [1350, 1400, 1450, 1500], "marg": [0.5, 1.0, 1.5]}
    best, best_v = dict(base), res["tr"]
    for sweep in range(2):
        for k, vals in grid.items():
            for v in vals:
                if v == best[k]:
                    continue
                P = dict(best); P[k] = v
                r, _ = replay(R, P, eval_from, split)
                if r["tr"] < best_v - 1e-6:
                    best_v, best = r["tr"], P
                    print(f"   better: {k}={v}  ITF train {r['tr']:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    r, _ = replay(R, best, eval_from, split)
    print(f"\nbest settings: {best}")
    print(f"ITF train {res['tr']:.5f} -> {r['tr']:.5f} | ITF TEST {res['te']:.5f} -> {r['te']:.5f} | all-levels TEST {res['te_all']:.5f} -> {r['te_all']:.5f}")
    Path("work/elo_tune_best.json").write_text(__import__("json").dumps(best))


if __name__ == "__main__":
    main()
