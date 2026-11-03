"""Situational context for a match: home country, schedule and fatigue, streaks, coming off a retirement, surface
switch, ranking, age, handedness. Used to test whether any of it predicts upsets beyond the market price."""
import math
import re

# tournament city / name fragment -> IOC country code (tour, Challenger and ITF venues that recur on Kalshi)
CITY = {
    "paris": "FRA", "lyon": "FRA", "marseille": "FRA", "montpellier": "FRA", "metz": "FRA", "rennes": "FRA", "orleans": "FRA",
    "roland garros": "FRA", "french open": "FRA", "brest": "FRA", "bordeaux": "FRA", "aix": "FRA", "nice": "FRA", "rouen": "FRA",
    "poitiers": "FRA", "limoges": "FRA", "strasbourg": "FRA", "saint-tropez": "FRA", "st. tropez": "FRA", "biarritz": "FRA", "quimper": "FRA",
    "london": "GBR", "wimbledon": "GBR", "queen": "GBR", "eastbourne": "GBR", "nottingham": "GBR", "birmingham": "GBR", "ilkley": "GBR",
    "glasgow": "GBR", "surbiton": "GBR", "bath": "GBR", "manchester": "GBR",
    "new york": "USA", "us open": "USA", "indian wells": "USA", "miami": "USA", "cincinnati": "USA", "washington": "USA", "atlanta": "USA",
    "winston": "USA", "delray": "USA", "dallas": "USA", "houston": "USA", "charleston": "USA", "austin": "USA", "san diego": "USA",
    "cleveland": "USA", "charlottesville": "USA", "knoxville": "USA", "champaign": "USA", "tiburon": "USA", "fairfield": "USA",
    "chicago": "USA", "newport": "USA", "memphis": "USA", "columbus": "USA", "sarasota": "USA", "tallahassee": "USA", "little rock": "USA",
    "orlando": "USA", "lexington": "USA", "rome, ga": "USA", "tyler": "USA", "phoenix": "USA", "monterey": "USA", "las vegas": "USA",
    "palm": "USA", "naples": "USA", "boca": "USA", "los angeles": "USA", "san francisco": "USA", "cary": "USA", "bloomington": "USA",
    "toronto": "CAN", "montreal": "CAN", "vancouver": "CAN", "granby": "CAN", "calgary": "CAN", "drummondville": "CAN", "saguenay": "CAN",
    "melbourne": "AUS", "australian open": "AUS", "sydney": "AUS", "brisbane": "AUS", "adelaide": "AUS", "perth": "AUS", "canberra": "AUS",
    "hobart": "AUS", "darwin": "AUS", "playford": "AUS", "burnie": "AUS", "traralgon": "AUS", "bendigo": "AUS",
    "auckland": "NZL", "rome": "ITA", "milan": "ITA", "turin": "ITA", "naples, it": "ITA", "napoli": "ITA", "florence": "ITA", "parma": "ITA",
    "bergamo": "ITA", "genoa": "ITA", "genova": "ITA", "bari": "ITA", "barletta": "ITA", "biella": "ITA", "cagliari": "ITA", "perugia": "ITA",
    "todi": "ITA", "verona": "ITA", "trieste": "ITA", "san marino": "SMR", "palermo": "ITA", "sardegna": "ITA", "pula": "ITA",
    "santa margherita": "ITA", "taranto": "ITA", "brescia": "ITA", "ortisei": "ITA", "olbia": "ITA", "vicenza": "ITA", "francavilla": "ITA",
    "padova": "ITA", "roma": "ITA", "cervia": "ITA", "sassuolo": "ITA", "bolzano": "ITA",
    "madrid": "ESP", "barcelona": "ESP", "valencia": "ESP", "mallorca": "ESP", "marbella": "ESP", "seville": "ESP", "sevilla": "ESP",
    "malaga": "ESP", "murcia": "ESP", "alicante": "ESP", "zaragoza": "ESP", "bilbao": "ESP", "oviedo": "ESP", "girona": "ESP",
    "monzon": "ESP", "segovia": "ESP", "vigo": "ESP", "gran canaria": "ESP", "tenerife": "ESP", "lleida": "ESP", "valladolid": "ESP",
    "manacor": "ESP", "denia": "ESP", "madrid open": "ESP",
    "hamburg": "GER", "munich": "GER", "stuttgart": "GER", "halle": "GER", "berlin": "GER", "bad homburg": "GER", "cologne": "GER",
    "heilbronn": "GER", "braunschweig": "GER", "ismaning": "GER", "oberstaufen": "GER", "augsburg": "GER", "lüdenscheid": "GER",
    "ludenscheid": "GER", "meerbusch": "GER", "koblenz": "GER", "eckental": "GER", "aachen": "GER", "nussloch": "GER", "essen": "GER",
    "vienna": "AUT", "kitzbuhel": "AUT", "kitzbühel": "AUT", "linz": "AUT", "salzburg": "AUT", "mauthausen": "AUT", "graz": "AUT",
    "basel": "SUI", "geneva": "SUI", "gstaad": "SUI", "lugano": "SUI", "biel": "SUI", "zug": "SUI",
    "rotterdam": "NED", "s-hertogenbosch": "NED", "hertogenbosch": "NED", "amersfoort": "NED", "rosmalen": "NED", "alphen": "NED",
    "brussels": "BEL", "antwerp": "BEL", "liege": "BEL", "mons": "BEL", "ostend": "BEL",
    "stockholm": "SWE", "bastad": "SWE", "båstad": "SWE", "copenhagen": "DEN", "helsinki": "FIN", "tampere": "FIN", "oslo": "NOR",
    "prague": "CZE", "ostrava": "CZE", "brno": "CZE", "prostejov": "CZE", "liberec": "CZE", "pilsen": "CZE",
    "bratislava": "SVK", "poprad": "SVK", "warsaw": "POL", "szczecin": "POL", "wroclaw": "POL", "poznan": "POL", "gdynia": "POL",
    "budapest": "HUN", "bucharest": "ROU", "cluj": "ROU", "iasi": "ROU", "sibiu": "ROU", "constanta": "ROU", "pitesti": "ROU", "slobozia": "ROU",
    "sofia": "BUL", "belgrade": "SRB", "novi sad": "SRB", "zagreb": "CRO", "umag": "CRO", "split": "CRO", "zadar": "CRO",
    "banja luka": "BIH", "sarajevo": "BIH", "ljubljana": "SLO", "portoroz": "SLO", "podgorica": "MNE", "skopje": "MKD",
    "athens": "GRE", "heraklion": "GRE", "istanbul": "TUR", "antalya": "TUR", "ankara": "TUR", "izmir": "TUR", "adana": "TUR",
    "monastir": "TUN", "hammamet": "TUN", "tunis": "TUN", "cairo": "EGY", "sharm": "EGY", "marrakech": "MAR", "rabat": "MAR",
    "casablanca": "MAR", "doha": "QAT", "dubai": "UAE", "abu dhabi": "UAE", "riyadh": "KSA", "jeddah": "KSA",
    "lisbon": "POR", "estoril": "POR", "porto": "POR", "oeiras": "POR", "braga": "POR", "maia": "POR", "guimaraes": "POR",
    "tokyo": "JPN", "osaka": "JPN", "kyoto": "JPN", "yokohama": "JPN", "matsuyama": "JPN", "kobe": "JPN", "toyota": "JPN",
    "beijing": "CHN", "shanghai": "CHN", "wuhan": "CHN", "ningbo": "CHN", "chengdu": "CHN", "zhuhai": "CHN", "hangzhou": "CHN",
    "guangzhou": "CHN", "shenzhen": "CHN", "zhangjiagang": "CHN", "jiujiang": "CHN", "tianjin": "CHN", "suzhou": "CHN", "wuxi": "CHN",
    "maanshan": "CHN", "zhengzhou": "CHN", "nanchang": "CHN", "hong kong": "HKG", "taipei": "TPE", "kaohsiung": "TPE",
    "seoul": "KOR", "busan": "KOR", "gwangju": "KOR", "chennai": "IND", "pune": "IND", "bengaluru": "IND", "bangalore": "IND",
    "mumbai": "IND", "delhi": "IND", "kolkata": "IND", "manila": "PHI", "bangkok": "THA", "nonthaburi": "THA", "singapore": "SGP",
    "jakarta": "INA", "kuala lumpur": "MAS", "almaty": "KAZ", "astana": "KAZ", "shymkent": "KAZ", "tashkent": "UZB", "fergana": "UZB",
    "samarkand": "UZB", "bukhara": "UZB", "nur-sultan": "KAZ", "tbilisi": "GEO", "yerevan": "ARM", "baku": "AZE",
    "buenos aires": "ARG", "cordoba": "ARG", "rosario": "ARG", "tucuman": "ARG", "mendoza": "ARG", "villa maria": "ARG",
    "santiago": "CHI", "concepcion": "CHI", "vina del mar": "CHI", "lima": "PER", "arequipa": "PER", "bogota": "COL", "cali": "COL",
    "medellin": "COL", "quito": "ECU", "guayaquil": "ECU", "salinas": "ECU", "montevideo": "URU", "asuncion": "PAR",
    "sao paulo": "BRA", "rio": "BRA", "florianopolis": "BRA", "campinas": "BRA", "brasilia": "BRA", "curitiba": "BRA",
    "porto alegre": "BRA", "costa do sauipe": "BRA", "mexico": "MEX", "acapulco": "MEX", "los cabos": "MEX", "monterrey": "MEX",
    "guadalajara": "MEX", "merida": "MEX", "puerto vallarta": "MEX", "morelos": "MEX", "san luis potosi": "MEX", "leon": "MEX",
    "tulum": "MEX", "santo domingo": "DOM", "punta cana": "DOM", "san jose": "CRC", "guatemala": "GUA",
}


def country_of(tourney):
    t = (tourney or "").lower()
    for c, code in sorted(CITY.items(), key=lambda kv: -len(kv[0])):
        if c in t:
            return code
    return None


def _rank(p):
    try:
        return int(float(p.rank)) if p.rank not in (None, "") else None
    except (TypeError, ValueError):
        return None


def player_context(p, t, surface, home):
    if p is None:
        return {}
    rec = sorted(p.recent, key=lambda x: x[0])
    streak = 0
    for d, mn, won, *_ in reversed(rec):
        if won:
            streak += 1
        else:
            break
    last = rec[-1] if rec else None
    retired_last = bool(last and not last[2] and re.search(r"RET|ABD", last[5] or ""))
    won_by_ret = bool(last and last[2] and re.search(r"RET|ABD", last[5] or ""))
    return {
        "home": int(bool(home and p.ioc == home)),
        "streak": streak,
        "m7": sum(1 for x in rec if 0 <= t - x[0] <= 7),
        "m30": sum(1 for x in rec if 0 <= t - x[0] <= 30),
        "wins10": sum(1 for x in rec if 0 <= t - x[0] <= 10 and x[2]),
        "mins3": sum((x[1] or 90) for x in rec if 0 <= t - x[0] <= 3),
        "retired_last": int(retired_last),
        "won_by_ret": int(won_by_ret),
        "days": (t - p.last) if p.last else 999,
        "switch": int(bool(last and last[4] and surface and last[4] != surface)),
        "age": round((t - p.born) / 365.25, 2) if p.born else None,
        "rank": _rank(p),
        "lefty": int(p.hand == "L"),
        "ht": float(p.ht) if p.ht not in (None, "") else None,
    }


FIELDS = ["home", "streak", "m7", "m30", "wins10", "mins3", "retired_last", "won_by_ret", "days", "switch", "age", "rank", "lefty", "ht"]


def match_context(R, ka, kb, tourney, surface, t):
    home = country_of(tourney)
    A = player_context(R.P.get(ka), t, surface, home)
    B = player_context(R.P.get(kb), t, surface, home)
    out = {"home_country": home}
    for f in FIELDS:
        out[f + "_a"] = A.get(f)
        out[f + "_b"] = B.get(f)
    return out
