r"""ITF (W15-W100 / M15-M25) at Kalshi prices: where our ratings and the market disagree, who's right?

Inspired by a live case (W35 Orlando, Mia Slama 73% on Kalshi vs Lavinia Tanasie, whom our Elo rated far higher):
  1. market calibration in ITF by price band
  2. market favourite that our model rates LOWER (by 10/20/30 pts, or as the underdog outright)
  3. newcomers / fast risers the market likes (few rated matches, much worse ranking than the opponent)
  4. model + market blend: weights fitted on the older half, log-loss on the newer
All bets at the real pre-match ask + taker fee; each rule shown in both halves.

  .venv\Scripts\python.exe -m tools.itf_study
"""
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from edge.names import key
from tools.kalshi_backtest import fee


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def load_itf():
    rows = []
    for l in open("work/kalshi_events.jsonl"):
        e = json.loads(l)
        if "ITF" not in e["series"] or not e.get("quote"):
            continue
        q = e["quote"]
        if q["ask"] - q["bid"] > 0.12 or q["bid"] <= 0 or q["ask"] >= 1:
            continue
        tour = "wta" if e["series"] == "KXITFWMATCH" else "atp"
        rows.append({"event": e["event"], "series": e["series"], "date": datetime.fromtimestamp(e["start"], timezone.utc).date().isoformat(),
                     "ka": key(e["a"], tour), "kb": key(e["b"], tour), "a": e["a"], "b": e["b"], "y": e["y"],
                     "px": (q["bid"] + q["ask"]) / 2, "ask_a": q["ask"], "ask_b": 1 - q["bid"], "tourney": e.get("tourney")})
    K = pd.DataFrame(rows).drop_duplicates("event")
    P = pd.read_csv("work/recent_preds.csv", low_memory=False)
    C = pd.read_csv("work/context.csv", low_memory=False, usecols=["date", "a", "b", "age_a", "age_b", "rank_a", "rank_b"])
    idx, cdx = {}, {}
    for r in P.itertuples():
        idx.setdefault((r.a, r.b), []).append(r)
    for r in C.itertuples():
        cdx.setdefault((r.a, r.b), []).append(r)
    out = []
    for r in K.itertuples():
        hit = None
        for (x, z, fl) in ((r.ka, r.kb, False), (r.kb, r.ka, True)):
            for c in idx.get((x, z), []):
                if abs((pd.Timestamp(c.date) - pd.Timestamp(r.date)).days) <= 7:
                    hit = (c, fl)
        if hit is None or hit[0].p_model != hit[0].p_model:
            continue
        c, fl = hit
        d = r._asdict()
        pm = c.p_model
        d.update({"pm": 1 - pm if fl else pm, "pe": 1 - c.p_elo if fl else c.p_elo,
                  "n_a": c.n_b if fl else c.n_a, "n_b": c.n_a if fl else c.n_b,
                  "days_a": c.days_b if fl else c.days_a, "days_b": c.days_a if fl else c.days_b})
        ag = None
        for (x, z, f2) in ((r.ka, r.kb, False), (r.kb, r.ka, True)):
            for cc in cdx.get((x, z), []):
                if abs((pd.Timestamp(cc.date) - pd.Timestamp(r.date)).days) <= 7:
                    ag = (cc, f2)
        if ag:
            cc, f2 = ag
            d.update({"age_a": cc.age_b if f2 else cc.age_a, "age_b": cc.age_a if f2 else cc.age_b,
                      "rank_a": cc.rank_b if f2 else cc.rank_a, "rank_b": cc.rank_a if f2 else cc.rank_b})
        out.append(d)
    D = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    fa = D.px >= 0.5
    D["pf"] = np.where(fa, D.px, 1 - D.px)
    D["fw"] = np.where(fa, D.y == 1, D.y == 0).astype(int)
    D["pm_f"] = np.where(fa, D.pm, 1 - D.pm)
    D["ask_f"] = np.where(fa, D.ask_a, D.ask_b)
    D["ask_d"] = np.where(fa, D.ask_b, D.ask_a)
    pick = lambda c: np.where(fa, D[c + "_a"], D[c + "_b"]) if c + "_a" in D else np.nan
    other = lambda c: np.where(fa, D[c + "_b"], D[c + "_a"]) if c + "_a" in D else np.nan
    for c in ("n", "age", "rank", "days"):
        D[c + "_f"], D[c + "_d"] = pick(c), other(c)
    D["half"] = (D.index >= len(D) // 2).astype(int)
    return D


def roi(g, side):
    price = g.ask_f if side == "f" else g.ask_d
    won = g.fw == 1 if side == "f" else g.fw == 0
    ok = price.between(0.04, 0.96)
    c = price[ok] + fee(price[ok])
    r = np.where(won[ok], (1 - c) / c, -1.0)
    return len(r), (r.mean() if len(r) else 0), (r.std(ddof=1) / math.sqrt(len(r)) if len(r) > 1 else 0)


def line(name, D, mask, side):
    parts = []
    for h in (0, 1):
        g = D[mask & (D.half == h)]
        if len(g) < 30:
            parts.append(f"half {h}: n={len(g):>4} (few)")
            continue
        n, m, se = roi(g, side)
        parts.append(f"half {h}: n={n:>4} fav won {g.fw.mean():5.1%} (priced {g.pf.mean():5.1%}) back {'fav' if side == 'f' else 'dog'} {m:+6.1%} ± {se:4.1%}")
    print(f"   {name:<52} " + "   ".join(parts))


def main():
    D = load_itf()
    print(f"ITF matches with a pre-match Kalshi quote and our model: {len(D):,} ({D.date.min()} to {D.date.max()})")
    print(f"log-loss: market {ll(D.y, D.px):.4f} | our model {ll(D.y, D.pm):.4f} | Elo only {ll(D.y, D.pe):.4f}")
    print("\n== 1. Is the ITF market calibrated?")
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)):
        line(f"favourite priced {lo:.0%}-{hi:.0%}", D, D.pf.between(lo, hi, inclusive="left"), "f")

    print("\n== 2. Market favourite that our model rates lower (the Slama case)")
    for gap in (0.10, 0.20, 0.30):
        line(f"model {gap * 100:.0f}+ pts below market on the fav: back fav", D, (D.pf - D.pm_f) >= gap, "f")
        line(f"model {gap * 100:.0f}+ pts below market on the fav: back dog", D, (D.pf - D.pm_f) >= gap, "d")
    line("our model has the market favourite as the UNDERDOG: back dog", D, D.pm_f < 0.5, "d")
    line("model 10+ pts ABOVE market on the fav: back fav", D, (D.pm_f - D.pf) >= 0.10, "f")

    print("\n== 3. Newcomers / risers the market likes")
    line("fav has < 40 rated matches, dog 100+: back dog", D, (D.n_f < 40) & (D.n_d >= 100), "d")
    line("fav has < 40 rated matches, dog 100+: back fav", D, (D.n_f < 40) & (D.n_d >= 100), "f")
    line("fav ranked 2x worse than dog: back dog", D, D.rank_f > 2 * D.rank_d, "d")
    line("fav ranked 2x worse than dog: back fav", D, D.rank_f > 2 * D.rank_d, "f")
    line("fav aged 18 or under: back dog", D, D.age_f <= 18, "d")
    line("fav aged 18 or under: back fav", D, D.age_f <= 18, "f")
    line("Slama case: fav < 40 matches AND model 20+ pts below: back dog", D, (D.n_f < 40) & ((D.pf - D.pm_f) >= 0.20), "d")

    print("\n== 4. How much should we trust our model next to the ITF market?")
    tr, te = D.half == 0, D.half == 1
    X = np.c_[lg(D.pm), lg(D.px)]
    m = LogisticRegression(C=100).fit(X[tr], D.y[tr])
    print(f"   weights (older half): model {m.coef_[0][0]:.2f}, market {m.coef_[0][1]:.2f}")
    print(f"   newer half log-loss: market {ll(D.y[te], D.px[te]):.4f}  blend {ll(D.y[te], m.predict_proba(X[te])[:, 1]):.4f}")


if __name__ == "__main__":
    main()
