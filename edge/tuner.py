"""Self-learning loops (2026-11-09). Each one changes the board only when the live evidence clears a guard fixed in
advance, so a lucky streak can't fool it and accuracy can't get worse by more than noise.

1. Blend self-calibration - how much to trust our model next to Kalshi's price, per level (Tour / Challenger / ITF).
   Refit at most once a day from the ledger (clean rows only). The backtest weights in model/coef.json are the prior,
   worth PRIOR_N matches, so live data moves them slowly. A refit is ADOPTED only if, fitted without the newest 25% of
   matches, it beats the weights in use on those newest matches (out of sample). Otherwise the old weights stay.
2. Strategy promotion / demotion from the paper record (log.json):
   paper -> small stake: 150+ settled bets, 90% lower bound of flat ROI above 0, average closing-line value above 0.
   staked -> paper:      100+ settled bets, 90% upper bound of flat ROI below 0.
   Promoted strategies are sized from the LOWER bound of their live ROI (capped at +10%), times PROMOTE_MULT.
3. Auto-discovered rules (state/discovered.json, written monthly by .github/workflows/discover.yml from the leak-free
   archive, kept only when more rules held than in any luck run) are paper-tracked here and then go through loop 2.

State lives in <data>/tuner.json; every decision is logged there and shown on the Model record tab.
Pure Python (the cloud job installs only requests)."""
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX_TS = 1793908800                   # ledger rows from before the ITF in-play fix (5 Nov 2026 20:00 UTC) are not clean
PRIOR_N = 5516                        # Kalshi matches behind the backtest blend weights (model/coef.json "market")
MIN_LIVE = 150                        # live matches per level before a refit is even tried
RETUNE_EVERY = 86400
PROMOTE = {"n": 150, "z": 1.2816, "mult": 0.1, "max_edge": 0.10}
DEMOTE = {"n": 100, "z": 1.2816}
NEVER_STAKE = {"all", "combo"}


def kind(series):
    return "ITF" if "ITF" in series else "Challenger" if "CHALLENGER" in series else "Tour"


def _lg(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + math.exp(-max(min(z, 30), -30)))


def _ll(rows, w):
    s = 0.0
    for xm, xk, y in rows:
        p = min(max(_sig(w["c"] + w["a"] * xm + w["b"] * xk), 1e-4), 1 - 1e-4)
        s -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return s / len(rows)


def _solve3(H, g):
    A = [H[i][:] + [g[i]] for i in range(3)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(A[r][c]))
        A[c], A[piv] = A[piv], A[c]
        if abs(A[c][c]) < 1e-12:
            return [0.0, 0.0, 0.0]
        for r in range(3):
            if r != c:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [A[i][3] / A[i][i] for i in range(3)]


def fit_blend(rows, prior, prior_n=PRIOR_N):
    """MAP logistic y ~ c + a*logit(model) + b*logit(market) with a Gaussian prior centred on `prior`, as strong as
    prior_n matches with this design (per-parameter precision = prior_n * mean Fisher information at the prior)."""
    th0 = [prior["c"], prior["a"], prior["b"]]
    X = [(1.0, xm, xk) for xm, xk, _ in rows]
    Y = [y for _, _, y in rows]
    p0 = [_sig(sum(t * x for t, x in zip(th0, xi))) for xi in X]
    lam = [prior_n * sum(p * (1 - p) * xi[j] ** 2 for p, xi in zip(p0, X)) / len(X) for j in range(3)]
    th = th0[:]
    for _ in range(25):
        p = [_sig(sum(t * x for t, x in zip(th, xi))) for xi in X]
        g = [sum((y - pi) * xi[j] for y, pi, xi in zip(Y, p, X)) - lam[j] * (th[j] - th0[j]) for j in range(3)]
        H = [[sum(pi * (1 - pi) * xi[j] * xi[k] for pi, xi in zip(p, X)) + (lam[j] if j == k else 0) for k in range(3)] for j in range(3)]
        step = _solve3(H, g)
        th = [t + s for t, s in zip(th, step)]
        if max(abs(s) for s in step) < 1e-7:
            break
    return {"c": round(th[0], 4), "a": round(th[1], 4), "b": round(th[2], 4)}


def ledger_rows(data):
    try:
        L = json.loads((data / "ledger.json").read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for e in L.values():
        if e.get("y") is None or e.get("close") is None or e.get("model") is None or e.get("t0", 0) < FIX_TS:
            continue
        out.setdefault(kind(e["series"]), []).append((e.get("start", 0), _lg(e["model"]), _lg(e["close"]), int(e["y"])))
    for k in out:
        out[k].sort()
    return out


def tune_blend(T, data, prior, now):
    """Loop 1. T = tuner state (mutated)."""
    B = T.setdefault("blend", {})
    if now - T.get("blend_ts", 0) < RETUNE_EVERY:
        return
    T["blend_ts"] = now
    rows = ledger_rows(data)
    for lvl in ("Tour", "Challenger", "ITF"):
        cur = (B.get(lvl) or {}).get("w") or {k: prior[k] for k in ("a", "b", "c")}
        R = [(xm, xk, y) for _, xm, xk, y in rows.get(lvl, [])]
        st = {"w": cur, "n_live": len(R), "checked": round(now)}
        if len(R) < MIN_LIVE:
            st["status"] = f"waiting for {MIN_LIVE} clean live matches ({len(R)} so far); using the backtest weights"
            B[lvl] = st
            continue
        cut = int(len(R) * 0.75)
        cand_old = fit_blend(R[:cut], prior)                 # fitted without the newest 25%
        hold = R[cut:]
        ll_cur, ll_new, ll_mkt = _ll(hold, cur), _ll(hold, cand_old), _ll(hold, {"a": 0, "b": 1, "c": 0})
        st.update({"holdout_n": len(hold), "ll_current": round(ll_cur, 4), "ll_candidate": round(ll_new, 4), "ll_market": round(ll_mkt, 4)})
        if ll_new < ll_cur:
            new = fit_blend(R, prior)
            st["w"] = new
            st["status"] = (f"updated: model weight {cur['a']:.3f} -> {new['a']:.3f}, market {cur['b']:.3f} -> {new['b']:.3f} "
                            f"(beat the old weights on the newest {len(hold)} matches, {ll_cur:.4f} -> {ll_new:.4f})")
            if abs(new["a"] - cur["a"]) > 0.005 or abs(new["b"] - cur["b"]) > 0.005:
                T.setdefault("log", []).append({"ts": round(now), "what": f"Blend {lvl}: {st['status']}"})
        else:
            st["status"] = f"kept: a refit did not beat the current weights on the newest {len(hold)} matches ({ll_new:.4f} vs {ll_cur:.4f})"
        B[lvl] = st


def blend_weights(T):
    """Level -> {"a","b","c"} to use (None = backtest prior)."""
    return {k: v["w"] for k, v in (T.get("blend") or {}).items() if v.get("w")}


def strategy_stats(log):
    S = {}
    for e in log.get("bets", {}).values():
        if e.get("result") not in ("yes", "no") or e.get("flat_ret") is None:
            continue
        clv = (e["close"] - e["entry"]) if e.get("close") is not None and e.get("entry") is not None else 0.0
        for t in e.get("tags") or []:
            s = S.setdefault(t, {"r": [], "clv": []})
            s["r"].append(e["flat_ret"]); s["clv"].append(clv)
    out = {}
    for t, s in S.items():
        n = len(s["r"]); mu = sum(s["r"]) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in s["r"]) / (n - 1)) if n > 1 else 1.0
        out[t] = {"n": n, "roi": round(mu, 4), "se": round(sd / math.sqrt(n), 4), "clv": round(sum(s["clv"]) / n, 4)}
    return out


def review_strategies(T, log, strategies, now):
    """Loop 2: decide promotions/demotions from the paper record. Returns {tag: override dict}."""
    st = strategy_stats(log)
    O = T.setdefault("overrides", {})
    for t, s in st.items():
        if t in NEVER_STAKE or t not in strategies:
            continue
        base = strategies[t].get("kelly_mult") or 0
        lo, hi = s["roi"] - PROMOTE["z"] * s["se"], s["roi"] + DEMOTE["z"] * s["se"]
        o = O.get(t)
        staked_now = (o["kelly_mult"] if o else base) > 0
        if not staked_now and s["n"] >= PROMOTE["n"] and lo > 0 and s["clv"] > 0:
            O[t] = {"kelly_mult": PROMOTE["mult"], "edge": round(min(lo, PROMOTE["max_edge"]), 4), "ts": round(now),
                    "why": f"promoted: {s['n']} settled, ROI {s['roi']:+.1%} (90% floor {lo:+.1%}), CLV {100 * s['clv']:+.2f}c"}
            T.setdefault("log", []).append({"ts": round(now), "what": f"{strategies[t].get('label', t)}: {O[t]['why']}"})
        elif staked_now and s["n"] >= DEMOTE["n"] and hi < 0:
            O[t] = {"kelly_mult": 0, "ts": round(now), "why": f"demoted to paper: {s['n']} settled, ROI {s['roi']:+.1%} (90% ceiling {hi:+.1%})"}
            T.setdefault("log", []).append({"ts": round(now), "what": f"{strategies[t].get('label', t)}: {O[t]['why']}"})
    T["stats"] = st
    return O


def apply_overrides(strategies, O):
    for t, o in O.items():
        if t in strategies:
            strategies[t]["kelly_mult_base"] = strategies[t].get("kelly_mult")
            strategies[t]["kelly_mult"] = o["kelly_mult"]
            strategies[t]["tuner"] = o["why"]


# ---------------------------------------------------------------- loop 3: discovered rules on live matches
def flat_features(snap):
    """Same variables as tools/archive_study.flatten, from an archive-style snapshot (dict), in favourite/underdog terms.
    Pure Python so the cloud can evaluate discovered rules on live matches."""
    pa = snap.get("p_close")
    if pa is None:
        return None
    fa = pa >= 0.5
    sf, sd = ("_a", "_b") if fa else ("_b", "_a")
    md, cx, fl = snap.get("model") or {}, snap.get("ctx") or {}, snap.get("flow") or {}
    sgn = 1 if fa else -1
    o = {"pf": pa if fa else 1 - pa, "fa": fa,
         "open_move_to_fav": (pa - snap.get("p_open", pa)) * sgn,
         "model_edge_fav": None if md.get("p_model") is None else (md["p_model"] if fa else 1 - md["p_model"]) - (pa if fa else 1 - pa),
         "mrate_edge_fav": None if md.get("p_mrate") is None else (md["p_mrate"] if fa else 1 - md["p_mrate"]) - (pa if fa else 1 - pa),
         "sub_model_disagreement": md.get("dis"), "risk_net": (snap.get("risk") or {}).get("net"),
         "flow_share_to_fav": None if fl.get("net_share") is None else fl["net_share"] * sgn,
         "big_money_to_fav": None if fl.get("big_net") is None else (0 if fl["big_net"] == 0 else math.copysign(1, fl["big_net"])) * sgn,
         "volume": fl.get("vol"), "itf": int("ITF" in snap.get("series", ""))}
    if md.get("form"):
        f = md["form"]
        if f[0] is not None and f[1] is not None:
            o["form_gap_fav"] = (f[0] - f[1]) * sgn
    if md.get("load"):
        l = md["load"]
        o["fav_mins_10d"], o["dog_mins_10d"] = (l[0], l[1]) if fa else (l[1], l[0])
    for k in ("mins3", "m7", "m30", "days", "age", "rank", "streak", "wins10", "home", "lefty", "switch"):
        if k + "_a" in cx:
            o[f"fav_{k}"], o[f"dog_{k}"] = cx.get(k + sf), cx.get(k + sd)
    return o


BANDS = {"any price": (0.0, 1.01), "toss-up <58%": (0.0, 0.58), "lean 58-70%": (0.58, 0.70), "confident 70%+": (0.70, 1.01)}


def load_discovered():
    try:
        return json.loads((ROOT / "state" / "discovered.json").read_text()).get("rules", [])
    except (OSError, ValueError):
        return []


def rule_hits(rule, feats):
    """True when the discovered rule fires on this match. Returns the side to back: 0 = favourite, 1 = underdog."""
    v = feats.get(rule["col"])
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    lo, hi = BANDS.get(rule["band"], (0, 1.01))
    if not lo <= feats["pf"] < hi:
        return None
    ok = (v == rule["thr"]) if rule["op"] == "=" else (v >= rule["thr"]) if rule["op"] == ">=" else (v <= rule["thr"])
    return (0 if rule["side"] == "fav" else 1) if ok else None


def discover_meta():
    try:
        d = json.loads((ROOT / "state" / "discovered.json").read_text())
        return {k: v for k, v in d.items() if k != "rules"}
    except (OSError, ValueError):
        return None


def load_state(data):
    try:
        return json.loads((data / "tuner.json").read_text())
    except (OSError, ValueError):
        return {}


def save_state(data, T):
    T["log"] = (T.get("log") or [])[-60:]
    (data / "tuner.json").write_text(json.dumps(T, separators=(",", ":")))


def now():
    return time.time()
