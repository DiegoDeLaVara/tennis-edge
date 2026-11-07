r"""Travel / body-clock study: do long trips, time-zone jumps, surface or indoor/outdoor switches, and altitude make a
player under- or over-perform Pinnacle's closing price?

Fatigue (minutes played in the last days) is the one situational factor that held up (tired favourites). This tests
its relatives. Data: tennis-data ATP 2010-24 / WTA 2020-24 main-tour matches with Pinnacle closing odds; each venue
geocoded once (Open-Meteo geocoding: latitude, longitude, elevation, IANA time zone; cache work/travel_geo.json).

A player's "trip" = previous tour-level event (last match there within 21 days) -> this event. Travel features are
only applied to the player's FIRST or SECOND match at the new event (jet lag fades). Each rule marks a "story"
player; we report ROI of backing / fading them at Pinnacle close by favourite/underdog role, in two time halves.
A rule only counts with > +1% in BOTH halves; --null N runs (results drawn from the price) show how many pass by luck.

  .venv\Scripts\python.exe -m tools.travel_study [--null N] [--rebuild]

Result (2026-11-07, 47,294 matches): 0 of 37 cells held in both halves vs 2.3 per luck run -> travel is priced.
Long-haul travellers (5,000+ km, first match) won ~1.0 pt MORE than priced in both halves (market slightly
over-worries about jet lag) - not enough to beat the margin. Back-to-back favourites ran -3.5 / -0.8 pts (not significant).
"""
import glob
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

CACHE = Path("work/travel_geo.json")
PKL = Path("work/travel_study.pkl")
UA = {"User-Agent": "TennisEdge/1.0 (https://github.com/DiegoDeLaVara/tennis-edge)"}
# venue -> (search name, country code) where the plain name is ambiguous or not a city
FIX = {"'s-Hertogenbosch": ("'s-Hertogenbosch", "NL"), "Monte Carlo": ("Monaco", "MC"), "Estoril": ("Cascais", "PT"),
       "Queens Club": ("London", "GB"), "London": ("London", "GB"), "Los Cabos": ("Cabo San Lucas", "MX"),
       "Mallorca": ("Palma", "ES"), "Kitzbuhel": ("Kitzbühel", "AT"), "Marrakech": ("Marrakesh", "MA"),
       "Halle": ("Halle (Westf.)", "DE"), "Newport": ("Newport", "US"), "Memphis": ("Memphis", "US"),
       "Washington": ("Washington", "US"), "Paris": ("Paris", "FR"), "Santiago": ("Santiago", "CL"),
       "Cordoba": ("Córdoba", "AR"), "Valencia": ("Valencia", "ES"), "Granby": ("Granby", "CA"), "Lexington": ("Lexington", "US"),
       "Cleveland": ("Cleveland", "US"), "San Jose": ("San Jose", "US"), "Dallas": ("Dallas", "US"), "Austin": ("Austin", "US"),
       "Atlanta": ("Atlanta", "US"), "Houston": ("Houston", "US"), "Charleston": ("Charleston", "US"),
       "Birmingham": ("Birmingham", "GB"), "Nottingham": ("Nottingham", "GB"), "Merida": ("Mérida", "MX"),
       "Guadalajara": ("Guadalajara", "MX"), "Monterrey": ("Monterrey", "MX"), "Toronto": ("Toronto", "CA"),
       "Sydney": ("Sydney", "AU"), "Melbourne": ("Melbourne", "AU"), "Brisbane": ("Brisbane", "AU"), "Hobart": ("Hobart", "AU"),
       "Adelaide": ("Adelaide", "AU"), "Florence": ("Florence", "IT"), "Napoli": ("Naples", "IT"), "Parma": ("Parma", "IT"),
       "Palermo": ("Palermo", "IT"), "Cagliari": ("Cagliari", "IT"), "Sardinia": ("Cagliari", "IT"), "Rome": ("Rome", "IT"),
       "Turin": ("Turin", "IT"), "Nice": ("Nice", "FR"), "Moscow": ("Moscow", "RU"), "St. Petersburg": ("Saint Petersburg", "RU"),
       "Costa Do Sauipe": ("Mata de São João", "BR"), "Gstaad": ("Gstaad", "CH"), "Courmayeur": ("Courmayeur", "IT"),
       "Marbella": ("Marbella", "ES"), "Gijon": ("Gijón", "ES"), "Tenerife": ("Santa Cruz de Tenerife", "ES"),
       "Nur-Sultan": ("Astana", "KZ"), "Johannesburg": ("Johannesburg", "ZA"), "Oeiras": ("Oeiras", "PT"),
       "Cancun": ("Cancún", "MX"), "Miami": ("Miami", "US"), "New York": ("New York", "US"), "Los Angeles": ("Los Angeles", "US"),
       "San Diego": ("San Diego", "US"), "Chicago": ("Chicago", "US"), "Indian Wells": ("Indian Wells", "US"),
       "Cincinnati": ("Mason", "US"), "Delray Beach": ("Delray Beach", "US"), "Fort Worth": ("Fort Worth", "US"),
       "Winston-Salem": ("Winston-Salem", "US"), "New Haven": ("New Haven", "US"), "Montreal": ("Montreal", "CA"),
       "Bastad": ("Båstad", "SE"), "Umag": ("Umag", "HR"), "Linz": ("Linz", "AT"), "Metz": ("Metz", "FR"),
       "Lyon": ("Lyon", "FR"), "Rouen": ("Rouen", "FR"), "Strasbourg": ("Strasbourg", "FR"), "Montpellier": ("Montpellier", "FR"),
       "Marseille": ("Marseille", "FR"), "Merida ": ("Mérida", "MX"), "Rabat": ("Rabat", "MA"), "Casablanca": ("Casablanca", "MA"),
       "Monastir": ("Monastir", "TN"), "Bogota": ("Bogotá", "CO"), "Quito": ("Quito", "EC"), "Antalya": ("Antalya", "TR"),
       "Istanbul": ("Istanbul", "TR"), "Hamburg": ("Hamburg", "DE"), "Berlin": ("Berlin", "DE"), "Stuttgart": ("Stuttgart", "DE"),
       "Munich": ("Munich", "DE"), "Cologne": ("Cologne", "DE"), "Dusseldorf": ("Düsseldorf", "DE"), "Bad Homburg": ("Bad Homburg", "DE"),
       "Eastbourne": ("Eastbourne", "GB"), "Barcelona": ("Barcelona", "ES"), "Madrid": ("Madrid", "ES"),
       "Buenos Aires": ("Buenos Aires", "AR"), "Rio de Janeiro": ("Rio de Janeiro", "BR"), "Sao Paulo": ("São Paulo", "BR"),
       "Vina del Mar": ("Viña del Mar", "CL"), "Seoul": ("Seoul", "KR"), "Tokyo": ("Tokyo", "JP"), "Osaka": ("Osaka", "JP")}


# hand-pinned venues where the geocoder picks the wrong town (checked 2026-11-07)
OVERRIDE = {"Halle": {"lat": 52.06, "lon": 8.36, "elev": 130, "tz": "Europe/Berlin", "cc": "DE", "name": "Halle (Westf.)"},
            "Cincinnati": {"lat": 39.36, "lon": -84.31, "elev": 250, "tz": "America/New_York", "cc": "US", "name": "Mason, OH"},
            "Newport": {"lat": 41.49, "lon": -71.31, "elev": 10, "tz": "America/New_York", "cc": "US", "name": "Newport, RI"},
            "Monte Carlo": {"lat": 43.75, "lon": 7.44, "elev": 50, "tz": "Europe/Monaco", "cc": "MC", "name": "Roquebrune-Cap-Martin"}}


def fetch_geo(locs):
    C = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for loc in sorted(locs):
        if loc in C:
            continue
        name, cc = FIX.get(loc.strip(), (loc.strip(), None))
        p = {"name": name, "count": 10}
        try:
            res = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=p, headers=UA, timeout=30).json().get("results") or []
        except Exception as e:  # noqa
            print("geo err", loc, e); continue
        if cc:
            res = [r for r in res if r.get("country_code") == cc] or res
        r = res[0] if res else None
        C[loc] = None if r is None else {"lat": r["latitude"], "lon": r["longitude"], "elev": r.get("elevation", 0) or 0,
                                         "tz": r.get("timezone"), "cc": r.get("country_code"), "name": r.get("name")}
        time.sleep(0.2)
    CACHE.write_text(json.dumps(C, indent=0, ensure_ascii=False), encoding="utf-8")
    C.update(OVERRIDE)
    return C


def num(x):
    try:
        v = float(x)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


def load():
    out = []
    for f in sorted(glob.glob("work/bookodds/*.csv")):
        D = pd.read_csv(f, encoding="latin-1", low_memory=False)
        D["date"] = pd.to_datetime(D["Date"], errors="coerce", format="mixed")
        D = D[D.date.notna()]
        out.append(pd.DataFrame({"tour": "atp" if "atp_" in f else "wta", "date": D.date.dt.date.values,
                                 "w": D.Winner.astype(str).values, "l": D.Loser.astype(str).values,
                                 "series": D.get("Series", D.get("Tier")).astype(str).values, "round": D.Round.astype(str).values,
                                 "psw": [num(x) for x in D.PSW], "psl": [num(x) for x in D.PSL],
                                 "tourn": D.Tournament.astype(str).values, "loc": D.Location.astype(str).values,
                                 "court": D.Court.astype(str).values, "surf": D.Surface.astype(str).values}))
    return pd.concat(out, ignore_index=True)


def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def offset(g, d):
    try:
        return datetime(d.year, d.month, d.day, 12, tzinfo=ZoneInfo(g["tz"])).utcoffset().total_seconds() / 3600
    except Exception:  # noqa
        return None


def build():
    O = load()
    G = fetch_geo(set(O["loc"]))
    bad = [k for k, v in G.items() if not v]
    print(f"{len(G)} venues geocoded, missing: {bad}")
    O["i"] = np.arange(len(O))
    O = O.sort_values(["date", "i"], kind="stable").reset_index(drop=True)
    O["tid"] = O.tour + "|" + O.tourn + "|" + pd.to_datetime(O.date).dt.year.astype(str)
    meta = O.groupby("tid").agg(loc=("loc", "first"), court=("court", "first"), surf=("surf", "first")).to_dict("index")
    hist = defaultdict(list)                     # player -> [(date, tid)]
    rows = []
    for r in O.itertuples():
        ok = r.psw and r.psl and math.isfinite(r.psw) and math.isfinite(r.psl)
        if ok:
            fw, fl = 1 / r.psw, 1 / r.psl
            pw = fw / (fw + fl)
            fav_w = pw >= 0.5
            F, Dg = (r.w, r.l) if fav_w else (r.l, r.w)
            g_here = G.get(r.loc)
            feats = {}
            for side, p in (("f", F), ("d", Dg)):
                H = hist[p]
                this = [h for h in H if h[1] == r.tid]
                prior = [h for h in H if h[1] != r.tid]
                nth = len(this) + 1                                # 1 = first match at this event
                feats[f"nth_{side}"] = nth
                feats[f"rest_{side}"] = np.nan
                for k in ("km", "tz", "east", "surfsw", "courtsw", "fromalt"):
                    feats[f"{k}_{side}"] = np.nan
                if prior and nth <= 2:
                    pd_, ptid = prior[-1]
                    gap = (r.date - pd_).days
                    pm = meta[ptid]
                    g_prev = G.get(pm["loc"])
                    if gap <= 21 and g_here and g_prev:
                        feats[f"rest_{side}"] = gap                 # days since last match at previous event
                        feats[f"km_{side}"] = hav(g_prev, g_here)
                        o1, o2 = offset(g_prev, pd_), offset(g_here, r.date)
                        if o1 is not None and o2 is not None:
                            dt = o2 - o1
                            if dt > 12: dt -= 24
                            if dt < -12: dt += 24
                            feats[f"tz_{side}"] = abs(dt)
                            feats[f"east_{side}"] = dt                # + = flew east (body clock behind)
                        feats[f"surfsw_{side}"] = float(pm["surf"] != r.surf)
                        feats[f"courtsw_{side}"] = float(pm["court"] != r.court)
                        feats[f"fromalt_{side}"] = g_prev["elev"]
            # altitude and its history: matches this player played at 1,000m+ (any time before)
            for side, p in (("f", F), ("d", Dg)):
                feats[f"altexp_{side}"] = sum(1 for h in hist[p] if h[1] != r.tid and (G.get(meta[h[1]]["loc"]) or {}).get("elev", 0) >= 1000)
            rows.append({"date": r.date, "tour": r.tour, "series": r.series, "round": r.round, "surf": r.surf, "court": r.court,
                         "elev": (g_here or {}).get("elev", np.nan), "y": int(fav_w), "p": max(pw, 1 - pw),
                         "ps_f": r.psw if fav_w else r.psl, "ps_d": r.psl if fav_w else r.psw, **feats})
        for p in (r.w, r.l):
            hist[p].append((r.date, r.tid))
    D = pd.DataFrame(rows)
    D.to_pickle(PKL)
    return D


def hypotheses(D):
    def pair(a, b):                                        # +1 favourite has the trait alone, -1 underdog alone
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        return np.where(a & ~b, 1, np.where(b & ~a, -1, 0))

    def g(c, side):
        return D[f"{c}_{side}"].values

    def tr(side, cond):
        return np.nan_to_num(cond(side), nan=0).astype(bool)
    first = lambda s: g("nth", s) == 1
    H = {
        "long trip 5,000+ km, first match": lambda s: (g("km", s) >= 5000) & first(s),
        "long trip 8,000+ km, first match": lambda s: (g("km", s) >= 8000) & first(s),
        "long trip 5,000+ km, first or second match": lambda s: g("km", s) >= 5000,
        "time-zone jump 5+ h, first match": lambda s: (g("tz", s) >= 5) & first(s),
        "time-zone jump 8+ h, first match": lambda s: (g("tz", s) >= 8) & first(s),
        "flew EAST 5+ h (harder jet lag), first match": lambda s: (g("east", s) >= 5) & first(s),
        "flew WEST 5+ h, first match": lambda s: (g("east", s) <= -5) & first(s),
        "time-zone jump 5+ h with 3 or fewer days to adjust": lambda s: (g("tz", s) >= 5) & (g("rest", s) <= 3),
        "time-zone jump 5+ h with 6+ days to adjust": lambda s: (g("tz", s) >= 5) & (g("rest", s) >= 6) & first(s),
        "back-to-back: played last event 2 or fewer days ago, first match": lambda s: (g("rest", s) <= 2) & first(s),
        "back-to-back AND 2,000+ km trip, first match": lambda s: (g("rest", s) <= 2) & (g("km", s) >= 2000) & first(s),
        "same city / short hop under 500 km, first match": lambda s: (g("km", s) < 500) & first(s),
        "surface switch (first match on new surface)": lambda s: (g("surfsw", s) == 1) & first(s),
        "surface switch with 3 or fewer days": lambda s: (g("surfsw", s) == 1) & (g("rest", s) <= 3) & first(s),
        "indoor/outdoor switch, first match": lambda s: (g("courtsw", s) == 1) & first(s),
        "came down from altitude (prev 1,000m+, now under 500m)": lambda s: (g("fromalt", s) >= 1000) & (D.elev.values < 500) & first(s),
    }
    out = {}
    for name, fn in H.items():
        out[name] = pair(tr("f", fn), tr("d", fn))
    alt = (D.elev.values >= 1000)
    out["at altitude (1,000m+), no prior altitude matches vs player with 3+"] = pair(
        alt & (D.altexp_f.values == 0) & (D.altexp_d.values >= 3), alt & (D.altexp_d.values == 0) & (D.altexp_f.values >= 3))
    out["at altitude (1,000m+), first match there, no prior altitude matches"] = pair(
        alt & (D.altexp_f.values == 0) & (D.nth_f.values == 1), alt & (D.altexp_d.values == 0) & (D.nth_d.values == 1))
    out["altitude venue (1,000m+) -> favourite"] = np.where(alt, 1, 0)
    return out


def run(D, quiet=False):
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    half = (np.arange(len(D)) >= len(D) // 2).astype(int)
    rf = np.where(D.y == 1, D.ps_f - 1, -1.0)
    rd = np.where(D.y == 0, D.ps_d - 1, -1.0)
    passed = []
    for name, s in hypotheses(D).items():
        if not quiet:
            print(f"-- {name}")
        for role, want in (("story player is FAVOURITE", 1), ("story player is UNDERDOG", -1)):
            if name.endswith("-> favourite") and want == -1:
                continue
            cells, back_ok, fade_ok = [], True, True
            for h in (0, 1):
                m = (s == want) & (half == h)
                back = rf[m] if want == 1 else rd[m]
                fade = rd[m] if want == 1 else rf[m]
                won = (D.y.values[m] == (1 if want == 1 else 0)).mean() if m.any() else np.nan
                pr = (D.p.values[m] if want == 1 else 1 - D.p.values[m]).mean() if m.any() else np.nan
                if len(back) < 80:
                    cells.append(f"n={len(back):>5} (few)                                    "); back_ok = fade_ok = False
                    continue
                se = back.std(ddof=1) / math.sqrt(len(back))
                cells.append(f"n={len(back):>5} won {won:5.1%} vs {pr:5.1%} back {back.mean():+6.1%} ± {se:4.1%} fade {fade.mean():+6.1%}")
                back_ok &= back.mean() > 0.01
                fade_ok &= fade.mean() > 0.01
            tag = "  <-- BACK held both halves" if back_ok else ("  <-- FADE held both halves" if fade_ok else "")
            if back_ok or fade_ok:
                passed.append((name, role, "back" if back_ok else "fade"))
            if not quiet:
                print(f"   {role:<26} " + "   ".join(cells) + tag)
    return passed


def main():
    if "--rebuild" in sys.argv or not PKL.exists():
        D = build()
    else:
        D = pd.read_pickle(PKL)
    if "--null" in sys.argv:
        seeds = int(sys.argv[sys.argv.index("--null") + 1])
        counts = []
        for sd in range(seeds):
            rng = np.random.default_rng(sd)
            N = D.copy(); N["y"] = (rng.random(len(N)) < N.p.values).astype(int)
            counts.append(len(run(N, quiet=True)))
        print(f"luck runs: rules passing both halves per run = {counts} (mean {np.mean(counts):.1f})")
        return
    trip = D.km_f.notna() | D.km_d.notna()
    print(f"{len(D):,} matches with Pinnacle closing odds; a known trip for at least one player in {trip.mean():.0%}")
    print("ROI of backing / fading the story player at Pinnacle close, two time halves. A rule needs > +1% in BOTH.\n")
    passed = run(D)
    print(f"\n{len(passed)} rule(s) held in both halves: {passed}")


if __name__ == "__main__":
    main()
