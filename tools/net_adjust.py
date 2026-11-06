r"""Fold the board's net signal score into the price. For each past Kalshi match, rebuild the same risk read-out the
live board uses (edge/refresh.risk_flags: veteran, young underdog, hot underdog, best-of-5, heavy favourite, busy
favourite, home player, tired underdog, underdog layoff, ITF model gap, teen ITF favourite), then adjust the
favourite's price by w x net points. Checks:
  1. accuracy: log-loss of the adjusted price vs Kalshi's own (w fitted on the older half, scored on the newer)
  2. betting: back the side the adjusted price says is cheap by 3%+ at the real ask after the fee
Caveat: several signal sizes were measured on this same Kalshi data, so the older half flatters; judge on the newer.

  .venv\Scripts\python.exe -m tools.net_adjust
"""
import contextlib
import io
import math

import numpy as np
import pandas as pd

from edge.refresh import risk_flags
from tools.kalshi_backtest import fee


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def nets(D, ctx_cols):
    out = []
    for r in D.itertuples(index=False):
        rd = r._asdict()
        ctx = {c: rd.get(c) for c in ctx_cols}
        for k, v in list(ctx.items()):
            if isinstance(v, float) and math.isnan(v):
                ctx[k] = None
        md = {"p_model": rd.get("p_model")}
        if rd.get("form_a") is not None and not (isinstance(rd.get("form_a"), float) and math.isnan(rd.get("form_a"))):
            md["form"] = [rd["form_a"], rd["form_b"]]
        if rd.get("load_a") is not None and not (isinstance(rd.get("load_a"), float) and math.isnan(rd.get("load_a"))):
            md["load"] = [int(rd["load_a"]), int(rd["load_b"])]
        m = {"p_mkt": rd["px"], "model": md, "ctx": ctx, "names": [rd.get("ka") or "A", rd.get("kb") or "B"], "bo": 3, "series": rd["series"]}
        try:
            r_ = risk_flags(m)
            out.append(0.0 if r_ is None or r_.get("stay") else r_.get("net", 0.0))
        except Exception:
            out.append(0.0)
    return np.array(out)


def evaluate(D, label):
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    D["h"] = (D.index >= len(D) // 2).astype(int)
    fa = D.px >= 0.5
    pf = np.where(fa, D.px, 1 - D.px)
    fw = np.where(fa, D.y == 1, D.y == 0).astype(int)
    net = D.net.values
    print(f"\n== {label}: {len(D):,} matches; matches with any signal: {(net != 0).mean():.0%}")
    tr, te = D.h.values == 0, D.h.values == 1
    best = None
    for w in (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        adj = np.clip(pf + w * net / 100, 0.02, 0.98)
        v = ll(fw[tr], adj[tr])
        if best is None or v < best[0]:
            best = (v, w)
    w = best[1]
    for nm, m in (("older half", tr), ("NEWER half", te)):
        sig = m & (net != 0)
        adj = np.clip(pf + w * net / 100, 0.02, 0.98); adj1 = np.clip(pf + net / 100, 0.02, 0.98)
        print(f"   {nm}: log-loss market {ll(fw[m], pf[m]):.5f} | adjusted (w={w}) {ll(fw[m], adj[m]):.5f} | adjusted (w=1) {ll(fw[m], adj1[m]):.5f}"
              f"   — on matches with a signal: market {ll(fw[sig], pf[sig]):.5f} vs adjusted {ll(fw[sig], adj1[sig]):.5f} (n={sig.sum()})")
    print(f"   betting where the w=1 adjusted price beats the ask by 3%+:")
    adj = np.clip(pf + net / 100, 0.02, 0.98)
    ask_f = np.where(fa, D.ask_a, D.ask_b); ask_d = np.where(fa, D.ask_b, D.ask_a)
    for nm, m in (("older half", tr), ("NEWER half", te)):
        cf, cd = ask_f + fee(ask_f), ask_d + fee(ask_d)
        ev_f, ev_d = adj / cf - 1, (1 - adj) / cd - 1
        bf = m & (ev_f >= 0.03) & (ask_f <= 0.95) & (ask_f >= 0.05)
        bd = m & (ev_d >= 0.03) & (ask_d <= 0.95) & (ask_d >= 0.05) & ~bf
        r = np.concatenate([np.where(fw[bf] == 1, (1 - cf[bf]) / cf[bf], -1.0), np.where(fw[bd] == 0, (1 - cd[bd]) / cd[bd], -1.0)])
        print(f"      {nm}: n={len(r):>4} ({bf.sum()} favourites, {bd.sum()} underdogs)  ROI {r.mean() if len(r) else 0:+6.1%} ± {r.std(ddof=1) / math.sqrt(max(len(r), 2)):4.1%}")


def main():
    from tools.close_matches import data
    with contextlib.redirect_stdout(io.StringIO()):
        D = data()
    ctx_cols = [c for c in D.columns if c.endswith("_a") or c.endswith("_b")]
    D["net"] = nets(D, ctx_cols)
    evaluate(D, "Kalshi tour + Challenger")
    from tools.itf_study import load_itf
    I = load_itf()
    I["p_model"] = I.pm
    I["net"] = nets(I, [c for c in I.columns if c.endswith("_a") or c.endswith("_b")])
    evaluate(I, "Kalshi ITF")


if __name__ == "__main__":
    main()
