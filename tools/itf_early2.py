r"""ITF early lines, round 2: trust our model only where it knows both players.

Finding that motivated it: on Kalshi ITF the model's log-loss gap to the market is 0.11 when a player has <10 known
matches but only 0.02-0.03 with 30+ (newcomers/juniors: the market knows things our results feed doesn't).
So: (1) does the open->close drift toward our model depend on experience? (2) a blend fitted per experience band
(model + open price), fitted on one half and tested on the other (both directions), (3) betting that blend's edge at
the opening ask and at the ~12h-before ask, fee included, with closing-line value (CLV).

  .venv\Scripts\python.exe -m tools.itf_early2
Data: work/itf_paths.jsonl (tools.itf_openers --fetch) + tools.itf_study.load_itf.
"""
import contextlib
import io
import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tools.itf_study import load_itf
from tools.kalshi_backtest import fee

BANDS = [("both 30+ matches", 30, 99999), ("both 10-29 (fewest)", 10, 30), ("someone under 10", 0, 10)]


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def data():
    with contextlib.redirect_stdout(io.StringIO()):
        D = load_itf()
    P = {json.loads(l)["event"]: json.loads(l) for l in open("work/itf_paths.jsonl")}
    D = D[D.event.isin(P)].copy()
    mid = lambda q: (q["bid"] + q["ask"]) / 2 if q else np.nan
    for k in ("open", "q12", "close"):
        D[f"p_{k}"] = [mid(P[e].get(k)) for e in D.event]
        D[f"bid_{k}"] = [(P[e].get(k) or {}).get("bid", np.nan) for e in D.event]
        D[f"ask_{k}"] = [(P[e].get(k) or {}).get("ask", np.nan) for e in D.event]
    D["nmin"] = np.minimum(D.n_a, D.n_b)
    D = D[D.p_open.notna() & D.p_close.notna() & D.pm.notna()].sort_values("date", kind="stable").reset_index(drop=True)
    D["half"] = (D.index >= len(D) // 2).astype(int)
    return D


def X(g):
    return np.c_[lg(g.pm), lg(g.p_open)]


def main():
    D = data()
    print(f"{len(D):,} ITF matches with open + close quotes and our model")
    print("\n== Spread by time (median, cents): open {:.0f}c | ~12h before {:.0f}c | close {:.0f}c".format(
        100 * np.nanmedian(D.ask_open - D.bid_open), 100 * np.nanmedian(D.ask_q12 - D.bid_q12), 100 * np.nanmedian(D.ask_close - D.bid_close)))

    print("\n== Drift open -> close toward our model, by how well we know the players (model 10+ pts off the open)")
    for name, lo, hi in BANDS:
        for h in (0, 1):
            g = D[(D.half == h) & (D.nmin >= lo) & (D.nmin < hi) & ((D.pm - D.p_open).abs() >= 0.10)]
            mv = (g.p_close - g.p_open) * np.sign(g.pm - g.p_open)
            print(f"   {name:<22} half {h}: n={len(g):>5}  toward {(mv > 0.005).mean():4.0%}  away {(mv < -0.005).mean():4.0%}  avg {100 * mv.mean():+.1f} pts")

    print("\n== Accuracy: blend (model + open) fitted per band on one half, tested on the other")
    D["p_bl"] = np.nan
    for name, lo, hi in BANDS:
        b = (D.nmin >= lo) & (D.nmin < hi)
        for tr, te in ((0, 1), (1, 0)):
            mtr, mte = b & (D.half == tr), b & (D.half == te)
            m = LogisticRegression(C=100).fit(X(D[mtr]), D.y[mtr])
            D.loc[mte, "p_bl"] = m.predict_proba(X(D[mte]))[:, 1]
            g = D[mte]
            print(f"   {name:<22} fit {tr} -> test {te}: weights model {m.coef_[0][0]:.2f} open {m.coef_[0][1]:.2f} | "
                  f"test log-loss blend {ll(g.y, g.p_bl):.4f} open {ll(g.y, g.p_open):.4f} close {ll(g.y, g.p_close):.4f}  (n={len(g)})")

    print("\n== Bet the out-of-sample blend's edge, fee included (each half scored by the blend fitted on the OTHER half)")
    for when in ("open", "q12"):
        for name, lo, hi in BANDS + [("all bands", 0, 99999)]:
            for thr in (0.03, 0.06, 0.10):
                parts = []
                for h in (0, 1):
                    g = D[(D.half == h) & (D.nmin >= lo) & (D.nmin < hi) & D[f"ask_{when}"].notna()]
                    pa = g.p_bl.values
                    ask_a, ask_b = g[f"ask_{when}"].values, 1 - g[f"bid_{when}"].values
                    ev_a, ev_b = pa / (ask_a + fee(ask_a)) - 1, (1 - pa) / (ask_b + fee(ask_b)) - 1
                    sa = ev_a >= ev_b
                    ev, price = np.where(sa, ev_a, ev_b), np.where(sa, ask_a, ask_b)
                    sel = (ev >= thr) & (price >= 0.08) & (price <= 0.92)
                    won = np.where(sa, g.y.values == 1, g.y.values == 0)[sel]
                    c = price[sel] + fee(price[sel])
                    r = np.where(won, (1 - c) / c, -1.0)
                    clv = (np.where(sa, g.p_close.values, 1 - g.p_close.values)[sel] - price[sel]) * 100
                    if len(r) < 2:
                        parts.append(f"half {h}: n={len(r):>4}"); continue
                    parts.append(f"half {h}: n={len(r):>4} ROI {r.mean():+6.1%} ± {r.std(ddof=1) / math.sqrt(len(r)):4.1%} CLV {clv.mean():+.1f}c")
                print(f"   at {when:<5} {name:<22} edge >= {thr:.0%}   " + "   ".join(parts))


if __name__ == "__main__":
    main()
