"""Tournament draws from Wikipedia's bracket templates ("<year> <Tournament> – Singles" pages).

parse_draw(wikitext) -> Draw with `levels[k][j]` = occupant name (or None / "BYE") of slot j at level k, where level 0
holds the first-round slots, level 1 the second-round slots, and so on up to the champion. `played[k][j]` is True when
the match feeding slot j of level k+1 is finished.
"""
import re

import requests

S = requests.Session()
S.headers["User-Agent"] = "TennisEdge/1.0 (https://github.com/DiegoDeLaVara/tennis-edge)"
API = "https://en.wikipedia.org/w/api.php"


def wiki(params):
    for _ in range(3):
        try:
            r = S.get(API, params={**params, "format": "json", "formatversion": 2}, timeout=30)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
    return {}


SLAMS = {"australian open": "Australian Open", "french open": "French Open", "roland garros": "French Open",
         "wimbledon": "Wimbledon Championships", "us open": "US Open"}
SKIP = ("doubles", "qualifying", "wheelchair", "boys", "girls", "mixed", "legends", "junior")
# Kalshi uses the city; Wikipedia often uses the event's own name
ALIAS = {"basel": "Swiss Indoors", "halle": "Halle Open", "queen": "Queen's Club Championships", "indian wells": "BNP Paribas Open",
         "cincinnati": "Cincinnati Open", "toronto": "Canadian Open", "montreal": "Canadian Open", "rome": "Italian Open",
         "madrid": "Madrid Open", "monte carlo": "Monte-Carlo Masters", "monte-carlo": "Monte-Carlo Masters", "shanghai": "Shanghai Masters",
         "vienna": "Vienna Open", "stockholm": "Stockholm Open", "tokyo": "Japan Open", "beijing": "China Open", "dubai": "Dubai Tennis Championships",
         "doha": "Qatar Open", "bastad": "Swedish Open", "gstaad": "Swiss Open", "kitzbuhel": "Austrian Open", "hamburg": "Hamburg Open",
         "umag": "Croatia Open", "seoul": "Korea Open", "miami": "Miami Open", "barcelona": "Barcelona Open", "munich": "Bavarian International",
         "eastbourne": "Eastbourne International", "mallorca": "Mallorca Championships", "s-hertogenbosch": "Libéma Open",
         "berlin": "Berlin Tennis Open", "bad homburg": "Bad Homburg Open", "nottingham": "Nottingham Open", "riyadh": "Finals",
         "turin": "ATP Finals", "osaka": "Japan Women's Open", "guadalajara": "Guadalajara Open", "monterrey": "Monterrey Open",
         "auckland": "ASB Classic", "brussels": "European Open", "bucharest": "Tiriac Open", "houston": "Clay Court Championships",
         "london": "Queen's Club Championships", "montpellier": "Open Occitanie", "iasi": "Iasi Open", "rabat": "Rabat Grand Prix",
         "metz": "Moselle Open"}


def find_page(year, tournament, women=False):
    """Best-matching '<year> <event> – Singles' page for a Kalshi tournament name like 'ATP Paris' or 'US Open'."""
    name = re.sub(r"^(ATP|WTA)\s+(Challenger\s+)?", "", tournament).strip()
    slam = next((v for k, v in SLAMS.items() if k in name.lower()), None)
    alias = next((v for k, v in ALIAS.items() if k in name.lower()), None)
    if alias and not slam:
        name = f"{alias} {name}"
    q = f"{year} {slam or name} {'women' if (women and slam) else ''} singles"
    hits = wiki({"action": "query", "list": "search", "srsearch": q, "srlimit": 15}).get("query", {}).get("search", [])
    toks = [t for t in re.findall(r"[a-z]+", (slam or name).lower()) if len(t) > 2 and t not in ("open", "the", "tennis")]
    best, score = None, 0
    for h in hits:
        t = h["title"]
        tl = t.lower()
        if str(year) not in t or "singles" not in tl or any(x in tl for x in SKIP):
            continue
        if ("women" in tl and not women) or ("men's" in tl and "women" not in tl and women):
            continue
        if slam and ("women" in tl) != women:
            continue
        snip = (h.get("snippet") or "").lower()
        sc = sum(2 for x in toks if x in tl) + sum(1 for x in toks if x in snip and x not in tl) + (1 if tl.startswith(str(year)) else 0)
        if sc > score:
            best, score = t, sc
    return best if score >= 2 or (toks == [] and best) else None


def wikitext(title, at=None):
    """Page wikitext, or the revision in force at unix time `at`."""
    if at is None:
        d = wiki({"action": "parse", "page": title, "prop": "wikitext"})
        return (d.get("parse") or {}).get("wikitext")
    from datetime import datetime, timezone
    ts = datetime.fromtimestamp(at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    d = wiki({"action": "query", "prop": "revisions", "titles": title, "rvlimit": 1, "rvdir": "older", "rvstart": ts,
              "rvprop": "content", "rvslots": "main"})
    pages = d.get("query", {}).get("pages", [])
    try:
        return pages[0]["revisions"][0]["slots"]["main"]["content"]
    except (IndexError, KeyError):
        return None


def _templates(text):
    """Top-level {{...Bracket...}} templates in order, as raw strings."""
    out, i = [], 0
    while True:
        j = text.find("{{", i)
        if j < 0:
            break
        depth, k = 0, j
        while k < len(text):
            if text.startswith("{{", k):
                depth += 1; k += 2
            elif text.startswith("}}", k):
                depth -= 1; k += 2
                if depth == 0:
                    break
            else:
                k += 1
        body = text[j:k]
        head = body[2:60]
        if "Bracket" in head:
            out.append(body)
        i = k
    return out


def _params(body):
    """|RDr-teamNN=... values, split on top-level pipes only."""
    out, depth, cur, buf = {}, 0, None, []
    parts, i, start = [], 2, 2
    while i < len(body) - 2:
        if body.startswith("{{", i) or body.startswith("[[", i):
            depth += 1; i += 2; continue
        if body.startswith("}}", i) or body.startswith("]]", i):
            depth -= 1; i += 2; continue
        if body[i] == "|" and depth == 0:
            parts.append(body[start:i]); start = i + 1
        i += 1
    parts.append(body[start:len(body) - 2])
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


LINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


def _name(v):
    if not v:
        return None
    m = LINK.search(v)
    if not m:
        txt = re.sub(r"\{\{[^}]*\}\}|'''|''", "", v).strip()
        return txt or None
    return re.sub(r"\s*\((tennis|tennis player)\)", "", m.group(1)).strip()


def _won(v):
    return bool(v) and "'''" in v


def _teams(p):
    """{(round, slot): (name, bold)} for a bracket's params; slots are 0-based."""
    out = {}
    for k, v in p.items():
        m = re.match(r"RD(\d+)-team0*(\d+)$", k)
        if m:
            out[(int(m.group(1)), int(m.group(2)) - 1)] = (_name(v), _won(v))
    return out


class Draw:
    def __init__(self, levels, won):
        self.levels = levels          # list of lists of names (None = unknown, "BYE")
        self.won = won                # won[k][j]: occupant of levels[k][j] is known to have won that match

    @property
    def size(self):
        return len(self.levels[0])


def _draw_body(text):
    """The main '== Draw ==' section only (qualifying draws and later sections are cut off)."""
    mm = re.search(r"==\s*Draw\s*==", text)
    i = mm.start() if mm else -1
    body = text[i:] if i >= 0 else text
    m = re.search(r"\n==[^=]", body[8:])           # stop at the next top-level section (qualifying, etc.)
    return body[:8 + m.start()] if m else body


ENTRY = {"Q": "Q", "WC": "WC", "LL": "LL", "PR": "PR", "SR": "PR", "SE": "SE", "ALT": "Alt"}
ABBR = re.compile(r"\{\{\s*abbr\s*\|([^|}]+)[^}]*\}\}", re.I)
JUNK = re.compile(r"\{\{[^}]*\}\}|<[^>]+>|'''|''|&nbsp;")


def entries(text):
    """Main-draw entry status from the bracket's seed column: {player name: 'Q' | 'WC' | 'LL' | 'PR' | 'SE' | 'Alt'}
    (qualifier, wildcard, lucky loser, protected/special ranking, special exempt, alternate). Seeded and direct
    entries are left out. Used by the qual_fav paper strategy (tools/entry_study)."""
    if not text:
        return {}
    out = {}
    for t in _templates(_draw_body(text)):
        p = _params(t)
        for k, v in p.items():
            if not re.match(r"RD\d+-seed\d+$", k):
                continue
            tag = JUNK.sub(" ", ABBR.sub(r"\1", v)).strip().upper()
            st = ENTRY.get(tag) or next((ENTRY[x] for x in re.split(r"[/,\s]+", tag) if x in ENTRY), None)
            nm = _name(p.get(k.replace("seed", "team")))
            if st and nm and nm != "BYE":
                out.setdefault(nm, st)
    return out


def parse_draw(text):
    if not text:
        return None
    body = _draw_body(text)
    tm = _templates(body)
    if not tm:
        return None
    sections, finals = [], None
    for t in tm:
        p = _params(t)
        teams = _teams(p)
        if not teams:
            continue
        rounds = max(r for r, _ in teams)
        r1 = max(j for r, j in teams if r == 1) + 1 if any(r == 1 for r, _ in teams) else 0
        if r1 <= 8 and finals is None and not sections:
            finals = (teams, rounds)
        else:
            sections.append((teams, rounds))
    if not sections:
        return None
    sec_rounds = max(r for _, r in sections)
    per = 2 ** sec_rounds                         # first-round slots per section
    n = per * len(sections)
    depth = (n).bit_length() - 1                  # levels 0..depth (depth = champion)
    levels = [[None] * (n >> k) for k in range(depth + 1)]
    won = [[False] * (n >> k) for k in range(depth + 1)]
    for s, (teams, rounds) in enumerate(sections):
        for (r, j), (nm, b) in teams.items():
            k = r - 1
            width = per >> k
            if j >= width:
                continue
            levels[k][s * width + j] = nm
            won[k][s * width + j] = b
    if finals:
        teams, rounds = finals
        base = depth - rounds                     # level of the finals bracket's first round
        for (r, j), (nm, b) in teams.items():
            k = base + r - 1
            if 0 <= k < depth and j < len(levels[k]) and nm:
                levels[k][j] = nm
                won[k][j] = b
                if b and r == rounds:
                    levels[depth][0] = nm
    # byes: a level-1 player with both feeding level-0 slots empty
    for j, nm in enumerate(levels[1]):
        if nm and levels[0][2 * j] is None and levels[0][2 * j + 1] is None:
            levels[0][2 * j], levels[0][2 * j + 1] = nm, "BYE"
            won[0][2 * j] = True
    # winners recorded at the next level count as known results
    for k in range(depth):
        for j in range(len(levels[k + 1])):
            w = levels[k + 1][j]
            if w and w != "BYE":
                for c in (2 * j, 2 * j + 1):
                    if levels[k][c] == w:
                        won[k][c] = True
    return Draw(levels, won)
