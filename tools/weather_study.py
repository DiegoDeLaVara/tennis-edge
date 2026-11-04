r"""Does the weather move tennis results beyond the price? Joins every outdoor ATP (2010-24) / WTA (2020-24) match with
Pinnacle closing odds to that day's weather at the venue (Open-Meteo historical archive: free, no key), then tests:
  - heat, wind, gusts, humidity, rain: does the favourite win more or less than priced?
  - total games: do conditions change the number of games beyond what the price and format imply?
Each effect is checked separately in 2010-17 and 2018-24.

  .venv\Scripts\python.exe -m tools.weather_study [--fetch-only]
Cache: work/weather_cache.json (geocodes + daily weather per tournament-week), resumable.
"""
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path("work/weather_cache.json")
UA = {"User-Agent": "TennisEdge/1.0 (https://github.com/DiegoDeLaVara/tennis-edge)"}
FIX = {"s-Hertogenbosch": "'s-Hertogenbosch", "Monte Carlo": "Monaco", "Estoril": "Cascais", "Queens Club": "London",
       "Gstaad": "Gstaad", "Nottingham": "Nottingham", "Delray Beach": "Delray Beach", "Los Cabos": "Cabo San Lucas",
       "Indian Wells": "Indian Wells", "Mallorca": "Palma", "Umag": "Umag", "Kitzbuhel": "Kitzbuehel", "Bastad": "Bastad"}
DAILY = "temperature_2m_max,apparent_temperature_max,wind_speed_10m_max,wind_gusts_10m_max,precipitation_sum,relative_humidity_2m_mean"


def load_raw():
    rows = []
    for f in sorted(glob.glob("work/bookodds/*.csv")):
        tour = "atp" if "atp_" in f else "wta"
        D = pd.read_csv(f, encoding="latin-1", low_memory=False)
        D["tour"] = tour
        rows.append(D)
    D = pd.concat(rows, ignore_index=True)
    D["d"] = pd.to_datetime(D.Date, errors="coerce", format="mixed")
    D = D[D.d.notna()]
    for c in ("PSW", "PSL", "MaxW", "MaxL"):
        D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D[(D.PSW > 1) & (D.PSL > 1)].copy()
    g = np.zeros(len(D))
    for i in range(1, 6):
        if f"W{i}" in D:
            g += pd.to_numeric(D[f"W{i}"], errors="coerce").fillna(0).values + pd.to_numeric(D[f"L{i}"], errors="coerce").fillna(0).values
    D["games"] = g
    D["bo"] = pd.to_numeric(D["Best of"], errors="coerce").fillna(3).astype(int)
    D["done"] = D.Comment.eq("Completed")
    return D


def fetch(D):
    C = json.loads(CACHE.read_text()) if CACHE.exists() else {"geo": {}, "wx": {}}
    out = D[D.Court.eq("Outdoor")]
    out = out.assign(wk=out.d.dt.strftime("%G-%V"))
    weeks = out.groupby(["Location", "wk"]).d.agg(["min", "max"]).reset_index()
    print(f"outdoor tournament-weeks: {len(weeks)}; cached {len(C['wx'])}")
    for loc in sorted(set(map(str, weeks.Location))):                 # geocode each venue once
        if loc in C["geo"]:
            continue
        try:
            j = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": FIX.get(loc, loc), "count": 1},
                             headers=UA, timeout=30).json()
            res = (j.get("results") or [None])[0]
            C["geo"][loc] = [res["latitude"], res["longitude"]] if res else None
        except Exception:
            pass
        time.sleep(0.15)
    todo = [r for r in weeks.itertuples() if C["geo"].get(str(r.Location)) and f"{r.Location}|{r.wk}" not in C["wx"]]
    print(f"to fetch: {len(todo)}")

    def one(r):
        g = C["geo"][str(r.Location)]
        for attempt in range(3):
            try:
                j = requests.get("https://archive-api.open-meteo.com/v1/archive", headers=UA, timeout=60, params={
                    "latitude": g[0], "longitude": g[1], "start_date": r.min.strftime("%Y-%m-%d"), "end_date": r.max.strftime("%Y-%m-%d"),
                    "daily": DAILY, "timezone": "auto", "wind_speed_unit": "kmh"}).json()
                if "daily" in j:
                    return f"{r.Location}|{r.wk}", j["daily"]
                time.sleep(5)
            except Exception:
                time.sleep(3)
        return None
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as ex:
        for i, res in enumerate(ex.map(one, todo), 1):
            if res:
                C["wx"][res[0]] = res[1]
            if i % 100 == 0:
                CACHE.write_text(json.dumps(C)); print(f"  {i} done")
    CACHE.write_text(json.dumps(C))
    return C


def attach(D, C):
    look = {}
    for k, d in C["wx"].items():
        loc = k.split("|")[0]
        for i, day in enumerate(d["time"]):
            look[(loc, day)] = {v: d[v][i] for v in d if v != "time"}
    W = [look.get((str(l), dd.strftime("%Y-%m-%d"))) for l, dd in zip(D.Location, D.d)]
    for v in DAILY.split(","):
        D[v] = [w.get(v) if w else np.nan for w in W]
    return D


def main():
    D = load_raw()
    C = fetch(D)
    if "--fetch-only" in sys.argv:
        return
    D = attach(D, C)
    D = D[D.Court.eq("Outdoor") & D.temperature_2m_max.notna()].copy()
    iw, il = 1 / D.PSW, 1 / D.PSL
    pw = iw / (iw + il)
    D["pf"] = np.maximum(pw, 1 - pw)
    D["fw"] = (pw >= 0.5).astype(int)
    D["ps_f"] = np.where(pw >= 0.5, D.PSW, D.PSL); D["ps_d"] = np.where(pw >= 0.5, D.PSL, D.PSW)
    D["half"] = (D.d.dt.year >= 2018).astype(int)
    print(f"\noutdoor matches with weather: {len(D):,}")
    print("weather spread: tmax p10/p50/p90 = %.0f/%.0f/%.0f C, wind max = %.0f/%.0f/%.0f km/h" % (
        *np.nanpercentile(D.temperature_2m_max, [10, 50, 90]), *np.nanpercentile(D.wind_speed_10m_max, [10, 50, 90])))

    # games baseline: mean games by format x surface x price band, from the same half
    D["band"] = pd.cut(D.pf, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    G = D[D.done & (D.games > 0)].copy()
    G["base"] = G.groupby(["half", "bo", "Surface", "band"], observed=True).games.transform("mean")
    G["gres"] = G.games - G.base

    conds = [("hot (max 32C+)", lambda X: X.temperature_2m_max >= 32), ("very hot (feels 35C+)", lambda X: X.apparent_temperature_max >= 35),
             ("cool (max under 15C)", lambda X: X.temperature_2m_max < 15),
             ("windy (max wind 30+ km/h)", lambda X: X.wind_speed_10m_max >= 30), ("gusty (gusts 50+ km/h)", lambda X: X.wind_gusts_10m_max >= 50),
             ("calm (max wind under 12 km/h)", lambda X: X.wind_speed_10m_max < 12),
             ("humid (mean RH 75%+)", lambda X: X.relative_humidity_2m_mean >= 75), ("dry (mean RH under 45%)", lambda X: X.relative_humidity_2m_mean < 45),
             ("rain 5mm+ (delays, heavy balls)", lambda X: X.precipitation_sum >= 5)]
    print("\n== Favourite won minus priced (pts), and total games vs the format/price baseline")
    print(f"   {'condition':<34} {'half':>5} {'n':>6} {'fav gap':>8} {'± se':>6}   {'n games':>7} {'games +/-':>9} {'± se':>6}")
    for name, f in conds:
        for h in (0, 1):
            g = D[f(D) & (D.half == h)]
            gg = G[f(G) & (G.half == h)]
            if len(g) < 80:
                print(f"   {name:<34} {h:>5} {len(g):>6}  (few)")
                continue
            gap = (g.fw - g.pf).mean(); se = math.sqrt((g.pf * (1 - g.pf)).sum()) / len(g)
            gm = gg.gres.mean(); gse = gg.gres.std(ddof=1) / math.sqrt(len(gg)) if len(gg) > 1 else 0
            print(f"   {name:<34} {h:>5} {len(g):>6} {gap * 100:>+8.1f} {se * 100:>6.1f}   {len(gg):>7} {gm:>+9.2f} {gse:>6.2f}"
                  f"{'  <-- fav' if abs(gap) > 2 * se else ''}{'  <-- games' if abs(gm) > 2 * gse else ''}")
    print("\n== Betting the favourite / underdog by condition (Pinnacle closing odds, flat 1u)")
    for name, f in conds:
        parts = []
        for side in ("f", "d"):
            for h in (0, 1):
                g = D[f(D) & (D.half == h)]
                o = g.ps_f if side == "f" else g.ps_d
                won = g.fw == 1 if side == "f" else g.fw == 0
                r = np.where(won, o - 1, -1.0)
                parts.append(f"{'fav' if side == 'f' else 'dog'} {h}: {r.mean():+5.1%}" if len(r) >= 80 else f"{'fav' if side == 'f' else 'dog'} {h}:   .  ")
        print(f"   {name:<34} " + "  ".join(parts))


if __name__ == "__main__":
    main()
