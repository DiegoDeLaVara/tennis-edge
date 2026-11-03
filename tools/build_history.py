"""Replay all match history through the ratings, write the seed state the cloud refresher starts from, and dump
pre-match feature rows for model fitting and backtests.

  .venv\\Scripts\\python.exe -m tools.build_history --archive C:\\TENNIS\\archive\\sack
"""
import argparse
import csv
import gzip
import json
import random
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from edge import sources as src
from edge.ratings import Ratings
from edge.names import key

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
WORK = ROOT / "work"


def dedupe(rows):
    """Same pair within 7 days from a *different* source is the same match (source dates are approximate):
    keep the first, so pass richer sources first. Within one source, distinct ids are distinct matches."""
    seen, out = {}, []
    for m in rows:
        pk = tuple(sorted((m["w"], m["l"])))
        srcn = m["uid"].split("|")[0]
        hit = False
        for d in range(m["date"] - 7, m["date"] + 8):
            s0 = seen.get((pk, d))
            if s0 is not None and s0 != srcn:
                hit = True
                break
        if hit:
            continue
        seen[(pk, m["date"])] = srcn
        out.append(m)
    return out


def recent_rows(after, surfaces, today=None):
    """Results after the archive ends: TML (ATP tour), ESPN (WTA tour, plus ATP as backup), Kalshi settlements."""
    today = today or date.today()
    rows = []
    tml = [m for m in src.tml_rows(today.year) if m["date"] > after]
    print("tml", len(tml))
    rows += tml
    espn = []
    d = date.fromordinal(after) - timedelta(days=7)
    seen_ids = set()
    while d <= today:
        for tour in ("wta",):
            for c in src.espn_matches(tour, src.espn_day(tour, d)):
                if c["id"] in seen_ids:
                    continue
                seen_ids.add(c["id"])
                m = src.espn_result_row(c, surfaces)
                if m and m["date"] > after:
                    espn.append(m)
        d += timedelta(days=3)
    print("espn", len(espn))
    rows += espn
    kal = []
    for s in src.MATCH_SERIES:
        ms = src.kalshi_markets(s, status="settled", max_pages=40)
        kal += [m for m in src.kalshi_result_rows(s, ms, surfaces) if m["date"] > after]
    print("kalshi", len(kal))
    rows += kal
    return dedupe(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=r"C:\TENNIS\archive\sack")
    ap.add_argument("--start", type=int, default=1985)
    ap.add_argument("--feat-from", type=int, default=2010)
    args = ap.parse_args()
    STATE.mkdir(exist_ok=True); WORK.mkdir(exist_ok=True)
    t0 = time.time()
    years = range(args.start, 2027)
    rows = []
    rows += src.archive_rows(args.archive, "atp", ["", "_qual_chall", "_futures"], [y for y in years])
    rows += src.archive_rows(args.archive, "wta", ["", "_qual_itf"], years)
    # archive's 2026 ATP tour file is superseded by TennisMyLife's complete 2026
    rows = [m for m in rows if not (m["uid"].startswith("atp|2026-") and m["level"] in ("G", "M", "A", "F", "D") and m["round"] and not m["round"].startswith("Q"))]
    print("archive rows", len(rows), round(time.time() - t0, 1), "s")
    surfaces = src.Surfaces()
    for m in rows:
        surfaces.learn(m.get("tourney"), m.get("surface"), m["tour"])
    arch_end = {t: max(m["date"] for m in rows if m["tour"] == t) for t in ("atp", "wta")}
    print("archive ends", {t: date.fromordinal(d).isoformat() for t, d in arch_end.items()})
    rec = recent_rows(min(arch_end.values()) - 10, surfaces)
    rec = [m for m in rec if m["date"] > arch_end[m["tour"]] - 10 or m["uid"].startswith("tml|")]
    allrows = dedupe(sorted(rows, key=lambda m: m["sort"]) + sorted(rec, key=lambda m: m["sort"]))
    allrows.sort(key=lambda m: m["sort"])
    print("total rows", len(allrows), "recent", len(rec))

    R = Ratings(w_surface=0.5, margin=True)
    feat_from = date(args.feat_from, 1, 1).toordinal()
    rnd = random.Random(7)
    out = open(WORK / "features.csv", "w", newline="")
    cols = ["date", "tour", "level", "surface", "best_of", "round", "src", "a", "b", "y", "p_elo", "p_elo_all", "elo_a", "elo_b",
            "n_a", "n_b", "pa", "pb", "stat_w", "mean_spw", "form_a", "form_b", "days_a", "days_b", "load_a", "load_b", "h2h", "h2h_n",
            "games_a", "games_b", "sets_a", "sets_b", "ret"]
    wr = csv.writer(out)
    wr.writerow(cols)
    # daily rating snapshots for draw backtests (players active in the last 2 years only)
    snap_from = date(2026, 4, 20).toordinal()
    snaps, next_snap = {}, snap_from
    for i, m in enumerate(allrows):
        while m["date"] >= next_snap and next_snap <= date.today().toordinal():
            snaps[next_snap] = {k: [round(p.elo, 1), {s: round(v, 1) for s, v in p.selo.items() if p.sn[s] >= 3}, p.n,
                                    round(p.form.at(next_snap, 45.0)[0] / (p.form.at(next_snap, 45.0)[1] + 2), 4)]
                                for k, p in R.P.items() if p.last >= next_snap - 730 and p.n >= 5}
            next_snap += 1
        if m["date"] >= feat_from and "W/O" not in m["score"]:
            a, b, y = (m["w"], m["l"], 1) if rnd.random() < 0.5 else (m["l"], m["w"], 0)
            f = R.features(a, b, m["tour"], m.get("surface") or "Hard", m["date"], m.get("best_of", 3), m["level"])
            if f:
                gw, gl, sets, tbs, ret = src_score(m["score"])
                ga, gb = (gw, gl) if y else (gl, gw)
                sw = sets_won(m["score"])
                sa, sb = (sw[0], sw[1]) if y else (sw[1], sw[0])
                wr.writerow([date.fromordinal(m["date"]).isoformat(), m["tour"], m["level"], m.get("surface") or "", m.get("best_of", 3),
                             m.get("round"), m["uid"].split("|")[0], a, b, y] + [round(f[c], 5) if isinstance(f[c], float) else f[c] for c in cols[10:28]]
                            + [ga, gb, sa, sb, int(ret)])
        R.add(m)
        if i % 200000 == 0:
            print(i, round(time.time() - t0, 1), "s")
    out.close()
    with gzip.open(WORK / "elo_snaps.json.gz", "wt") as fh:
        json.dump(snaps, fh, separators=(",", ":"))
    cut = (date.today() - timedelta(days=900)).toordinal()
    st = R.dump(min_last=cut)
    st["built"] = datetime.now(timezone.utc).isoformat()
    st["seen_pairs"] = [[m["w"], m["l"], m["date"]] for m in allrows if m["date"] > date.today().toordinal() - 40]
    st["surfaces"] = surfaces.t
    with gzip.open(STATE / "ratings.json.gz", "wt") as fh:
        json.dump(st, fh, separators=(",", ":"))
    print("players kept", len(st["players"]), "done", round(time.time() - t0, 1), "s")


def src_score(s):
    from edge.ratings import parse_score
    return parse_score(s)


def sets_won(s):
    import re
    a = b = 0
    for x, y in __import__("re").findall(r"(\d+)-(\d+)", re.sub(r"\(\d+\)", "", s or "")):
        x, y = int(x), int(y)
        if x > y:
            a += 1
        elif y > x:
            b += 1
    return a, b


if __name__ == "__main__":
    main()
