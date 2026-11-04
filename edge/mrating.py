"""Market ratings: each player's strength as the betting market sees it, learned from past pre-match prices
(Pinnacle closing odds 2010-24, then Kalshi pre-match mids). They predict matches that have no price yet.

Walk-forward test (tools/market_elo, 11k ATP/WTA matches 2018-24, both players with 10+ priced matches):
results-only Elo log-loss 0.6080, market ratings 0.6074, blend 0.6024 (Pinnacle's own close: 0.5824).
"""
import gzip
import json
import math
from pathlib import Path

LR = 1.2
DECAY_DAYS = 365
W_MKT, W_MODEL = 0.43, 0.63          # blend fitted on 2010-17 (tools/market_elo, lr 1.2)
SEED = Path(__file__).resolve().parents[1] / "state" / "market_ratings.json.gz"


def _lg(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class MR:
    def __init__(self):
        self.m, self.n, self.last, self.pending = {}, {}, {}, {}

    # ---- persistence
    @classmethod
    def load(cls, data_dir):
        r = cls()
        for f in (Path(data_dir) / "market_ratings.json.gz", SEED):
            try:
                d = json.loads(gzip.decompress(f.read_bytes()))
                r.m, r.n, r.last, r.pending = d["m"], d["n"], d["last"], d.get("pending", {})
                return r
            except (OSError, ValueError, KeyError):
                continue
        return r

    def save(self, path):
        d = {"m": {k: round(v, 4) for k, v in self.m.items()}, "n": self.n, "last": self.last, "pending": self.pending}
        Path(path).write_bytes(gzip.compress(json.dumps(d, separators=(",", ":")).encode()))

    # ---- model
    def _cur(self, k, t):
        v = self.m.get(k)
        if v is None:
            return None
        gap = t - self.last.get(k, t)
        return v * math.exp(-(gap - 30) / DECAY_DAYS) if gap > 30 else v

    def logit(self, ka, kb, t, min_n=5):
        """Market-rating logit that ka beats kb on day ordinal t, or None when either player has too little history."""
        if self.n.get(ka, 0) < min_n or self.n.get(kb, 0) < min_n:
            return None
        return self._cur(ka, t) - self._cur(kb, t)

    def blend(self, model_logit, ka, kb, t):
        x = self.logit(ka, kb, t)
        return model_logit if x is None else W_MODEL * model_logit + W_MKT * x

    def update(self, ka, kb, p_mkt, t):
        """Nudge both players toward the market's pre-match price for ka."""
        for k in (ka, kb):
            v = self._cur(k, t)
            self.m[k] = 0.0 if v is None else v
        err = _lg(p_mkt) - (self.m[ka] - self.m[kb])
        for k, s in ((ka, 1), (kb, -1)):
            n = self.n.get(k, 0)
            self.m[k] += s * (LR / (1 + 0.05 * n) + 0.05) * err / 2
            self.n[k] = n + 1
            self.last[k] = t
