"""Pre-match win probability: a logistic blend of rating features, fitted walk-forward (tools/fit.py) and stored
as plain coefficients in model/coef.json so scoring needs no ML library. Then an optional blend with the market
price, also fitted, because closing prices carry information no ratings model has (injuries, motivation)."""
import json
import math
from pathlib import Path

from . import markov

COEF = Path(__file__).resolve().parents[1] / "model" / "coef.json"


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sig(x):
    return 1 / (1 + math.exp(-max(min(x, 30), -30)))


def p_stats(f, best_of=3):
    return markov.fast_p_match(f["pa"], f["pb"], best_of)


def design(f, ps, tour, level, best_of):
    """Feature vector (A-side). Must match tools/fit.py."""
    has = 1.0 if f["stat_w"] >= 400 else 0.0
    lay_a = 1.0 if f["days_a"] > 75 else 0.0
    lay_b = 1.0 if f["days_b"] > 75 else 0.0
    h2h = f["h2h"] / (f["h2h_n"] + 3)
    rookie = (1.0 if f["n_a"] < 20 else 0.0) - (1.0 if f["n_b"] < 20 else 0.0)
    return [
        logit(f["p_elo"]),
        logit(f["p_elo_all"]),
        has * logit(ps),
        has,
        f["form_a"] - f["form_b"],
        math.log1p(f["load_a"]) - math.log1p(f["load_b"]),
        lay_a - lay_b,
        h2h,
        rookie * logit(f["p_elo"]),
        (1.0 if tour == "wta" else 0.0) * logit(f["p_elo"]),
        (1.0 if level in ("S", "Q") else 0.0) * logit(f["p_elo"]),
    ]


FEATURES = ["elo_blend", "elo_all", "stats", "has_stats", "form", "fatigue", "layoff", "h2h", "rookie_x_elo", "wta_x_elo", "lowtier_x_elo"]


class Model:
    def __init__(self, path=COEF):
        d = json.loads(Path(path).read_text())
        self.w = d["w"]
        self.b = d["b"]
        self.mkt = d.get("market")          # {"a": model weight, "b": market weight, "c": intercept}
        self.meta = d

    def prob(self, f, tour, level, best_of=3):
        ps = p_stats(f, best_of)
        x = design(f, ps, tour, level, best_of)
        z = self.b + sum(w * v for w, v in zip(self.w, x))
        return sig(z), ps

    def blend(self, p_model, p_market):
        if p_market is None or not self.mkt:
            return p_model
        m = self.mkt
        return sig(m["c"] + m["a"] * logit(p_model) + m["b"] * logit(p_market))
