"""Player-name keys that survive accents, word order ("You Xiaodi" vs "Xiaodi You") and hyphens."""
import re
import unicodedata

_SUFFIX = {"jr", "sr", "ii", "iii"}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]+", " ", s.lower().replace("-", " ").replace("'", "")).split()


def key(name, tour):
    toks = [t for t in fold(name) if t not in _SUFFIX]
    return f"{tour}:{' '.join(sorted(toks))}"


def surname_key(name, tour):
    """Fallback: last token + first initial, for sources that abbreviate first names."""
    toks = [t for t in fold(name) if t not in _SUFFIX]
    if not toks:
        return None
    return f"{tour}:{toks[-1]}:{toks[0][0]}"


class Index:
    """Resolve a display name to a known player key, tolerating small spelling differences."""

    def __init__(self, keys):
        self.keys = set(keys)
        self.by_tokens = {}
        for k in self.keys:
            tour, toks = k.split(":", 1)
            for t in toks.split():
                if len(t) > 2:
                    self.by_tokens.setdefault((tour, t), set()).add(k)

    def resolve(self, name, tour):
        k = key(name, tour)
        if k in self.keys:
            return k
        toks = [t for t in fold(name) if t not in _SUFFIX and len(t) > 2]
        cand = None
        for t in toks:
            s = self.by_tokens.get((tour, t), set())
            cand = s if cand is None else cand & s
        if cand and len(cand) == 1:
            return next(iter(cand))
        # same surname, compatible first names (e.g. "Alex" / "Alexander")
        if toks:
            sur = sorted(self.by_tokens.get((tour, toks[-1]), set()))
            first = toks[0]
            hits = [c for c in sur if any(w.startswith(first[:4]) or first.startswith(w[:4]) for w in c.split(":", 1)[1].split() if w != toks[-1])]
            if len(hits) == 1:
                return hits[0]
        return k
