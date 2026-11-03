"""Backtest against real Kalshi prices: model vs market, betting rules, and money-flow signals.

Needs work/recent_preds.csv (tools.fit) and work/kalshi_events.jsonl (tools.kalshi_collect).
  .venv\\Scripts\\python.exe -m tools.kalshi_backtest [--write]
--write stores the market-blend weights and the rule table in model/coef.json.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from edge.names import key
from edge.sources import MATCH_SERIES

ROOT = Path(__file__).resolve().parents[1]


def fee(p):
    return 0.07 * p * (1 - p)


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load():
    ev = [json.loads(l) for l in open(ROOT / "work" / "kalshi_events.jsonl")]
    qf = ROOT / "work" / "quotes.jsonl"
    extra = {}
    if qf.exists():
        for l in qf.open():
            try:
                d = json.loads(l)
                extra[d["event"]] = d["quote"]
            except ValueError:
                pass
    rows = []
    for e in ev:
        if "quote" not in e:
            e["quote"] = extra.get(e["event"])
        f = e["flow"]
        if not f or f.get("n", 0) < 3 or f.get("px_last") is None:
            continue
        tour = MATCH_SERIES[e["series"]][0]
        d = datetime.fromtimestamp(e["start"], timezone.utc).date().isoformat()
        q = e.get("quote")
        if not q or q["ask"] - q["bid"] > 0.12 or q["bid"] <= 0 or q["ask"] >= 1:
            continue
        rows.append({"event": e["event"], "series": e["series"], "tour": tour, "date": d, "ka": key(e["a"], tour), "kb": key(e["b"], tour),
                     "y": e["y"], "px": (q["bid"] + q["ask"]) / 2, "ask_a": q["ask"], "ask_b": 1 - q["bid"], "spread": q["ask"] - q["bid"], "qts": q.get("ts"),
                     "px_trade": f["px_last"], "px_first": f["px_first"], "vol": f["vol"], "net_share": f.get("net_share", 0),
                     "big_net": f.get("big_net", 0), "big_vol": f.get("big_vol", 0), "late_net": f.get("late_net", 0), "steam": f.get("steam", 0),
                     "whale": f.get("whale", 0), "whale_dir": f.get("whale_dir", 0), "n_trades": f["n"], "round": e.get("round")})
    K = pd.DataFrame(rows)
    P = pd.read_csv(ROOT / "work" / "recent_preds.csv", low_memory=False)
    P = P[P.p_model.notna()]
    idx = {}
    for r in P.itertuples():
        idx.setdefault((r.a, r.b), []).append((r.date, r.p_model, r.p_elo, r.level, r.n_a, r.n_b))
    pm, pe, lv, nmin = [], [], [], []
    for r in K.itertuples():
        hit, best = None, 99
        for (a, b, flip) in ((r.ka, r.kb, False), (r.kb, r.ka, True)):
            for (d, p, e_, l, na, nb) in idx.get((a, b), []):
                gap = abs((pd.Timestamp(d) - pd.Timestamp(r.date)).days)
                if gap <= 7 and gap < best:
                    best = gap
                    hit = (1 - p if flip else p, 1 - e_ if flip else e_, l, min(na, nb))
        pm.append(hit[0] if hit else np.nan); pe.append(hit[1] if hit else np.nan)
        lv.append(hit[2] if hit else None); nmin.append(hit[3] if hit else np.nan)
    K["p_model"], K["p_elo"], K["level"], K["nmin"] = pm, pe, lv, nmin
    return K


def roi_table(K, p_col, label, thresholds=(0.0, 0.03, 0.05, 0.08, 0.12)):
    out = []
    for side in (0, 1):
        pass
    for th in thresholds:
        bets = []
        for r in K.itertuples():
            p = getattr(r, p_col)
            for side in (0, 1):
                ps = p if side == 0 else 1 - p
                price = r.ask_a if side == 0 else r.ask_b               # the real ask an hour before the start
                if not (0.05 <= price <= 0.95):
                    continue
                ev = ps / (price + fee(price)) - 1
                if ev >= th:
                    win = (r.y == 1) if side == 0 else (r.y == 0)
                    c = price + fee(price)
                    bets.append((win, (1 - c) / c if win else -1.0, price))
        if bets:
            a = np.array([b[1] for b in bets])
            out.append({"rule": f"{label}: EV ≥ {int(th * 100)}%", "n": len(bets), "win": float(np.mean([b[0] for b in bets])),
                        "roi": float(a.mean()), "se": float(a.std() / np.sqrt(len(a)))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    K = load()
    print("kalshi events with prices", len(K), "matched to model", K.p_model.notna().sum())
    M = K[K.p_model.notna() & (K.nmin >= 10)].copy()
    M = M.sort_values("date")
    y = M.y.values
    print("\n== who predicts better (lower log-loss is better)")
    print(f"mean spread {M.spread.mean():.3f}; last-trade price ll {ll(y, M.px_trade.values):.4f}")
    print(f"market  {ll(y, M.px.values):.4f}  model {ll(y, M.p_model.values):.4f}  elo {ll(y, M.p_elo.values):.4f}   n={len(M)}")
    for s, g in M.groupby("series"):
        print(f"  {s:22s} n={len(g):5d} market {ll(g.y.values, g.px.values):.4f} model {ll(g.y.values, g.p_model.values):.4f}")
    # blend: fit on first 60% of dates, test on the rest
    cut = int(len(M) * 0.6)
    X = np.c_[logit(M.p_model.values), logit(M.px.values)]
    lr = LogisticRegression(C=10).fit(X[:cut], y[:cut])
    pb = lr.predict_proba(X[cut:])[:, 1]
    print(f"\nblend (fit on first 60%, test on last 40%): market {ll(y[cut:], M.px.values[cut:]):.4f}  blend {ll(y[cut:], pb):.4f}  "
          f"weights model {lr.coef_[0][0]:.3f} market {lr.coef_[0][1]:.3f} c {lr.intercept_[0]:.3f}")
    full = LogisticRegression(C=10).fit(X, y)
    M["p_blend"] = full.predict_proba(X)[:, 1]
    T = M.iloc[cut:].copy()
    T["p_blend"] = pb
    print("\n== betting rules on the held-out 40% (entry = real ask ~1h before start + Kalshi taker fee)")
    rows = []
    for p_col, lab in (("p_model", "Model alone"), ("p_blend", "Model + market blend")):
        for r in roi_table(T, p_col, lab):
            rows.append(r); print(f"  {r['rule']:38s} n={r['n']:5d} win {r['win']:.3f} ROI {r['roi'] * 100:+.1f}% ± {r['se'] * 100:.1f}")
    print("\n== by tier (held-out)")
    for s, g in T.groupby("series"):
        print(f"  {s}: n={len(g)} market ll {ll(g.y.values, g.px.values):.4f} model {ll(g.y.values, g.p_model.values):.4f} blend {ll(g.y.values, g.p_blend.values):.4f}")
        for p_col, lab in (("p_model", "model"), ("p_blend", "blend")):
            for r in roi_table(g, p_col, f"{s} {lab}", thresholds=(0.03, 0.06)):
                print(f"    {r['rule']:44s} n={r['n']:5d} win {r['win']:.3f} ROI {r['roi'] * 100:+.1f}% ± {r['se'] * 100:.1f}")
                rows.append(r)
    for p_col, lab in (("p_model", "model alone"), ("p_blend", "blend")):
        print(f"\n== by the price of the side bet, {lab} EV ≥ 5% (held-out)")
        for lo, hi in ((0.05, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.95)):
            res = []
            for r in T.itertuples():
                for side in (0, 1):
                    p = getattr(r, p_col)
                    ps = p if side == 0 else 1 - p
                    price = r.ask_a if side == 0 else r.ask_b
                    if not (lo <= price < hi) or ps / (price + fee(price)) - 1 < 0.05:
                        continue
                    win = (r.y == 1) if side == 0 else (r.y == 0)
                    c = price + fee(price)
                    res.append((1 - c) / c if win else -1.0)
            if res:
                a = np.array(res)
                print(f"  {lo:.2f}-{hi:.2f}  n={len(a):5d} win {np.mean(a > 0):.3f} ROI {a.mean() * 100:+.1f}% ± {a.std() / np.sqrt(len(a)) * 100:.1f}")
                rows.append({"rule": f"{lab.capitalize()} EV ≥ 5%, price {int(lo * 100)}-{int(hi * 100)}c", "n": int(len(a)),
                             "win": float(np.mean(a > 0)), "roi": float(a.mean())})
    # ---- money flow: does it add information beyond the price?
    print("\n== money flow vs outcome, controlling for price (all matched events)")
    F = K[K.vol >= 500].copy()
    F["drift"] = F.px - F.px_first
    F["big_dir"] = np.sign(F.big_net) * (F.big_vol >= 750)
    F["flow_dir"] = np.sign(F.net_share) * (F.net_share.abs() >= 0.35)
    for col in ("drift", "net_share", "big_dir", "flow_dir", "steam"):
        Xf = np.c_[logit(F.px.values), F[col].values]
        lr2 = LogisticRegression(C=10).fit(Xf, F.y.values)
        print(f"  {col:10s} coef {lr2.coef_[0][1]:+.3f}  (price coef {lr2.coef_[0][0]:.3f})  n={len(F)}")
    print("\n== following each money signal at the pre-match price")
    sig_rows = []
    for name, mask, d in (("Big tickets one way ($750+, 60%+)", (F.big_vol >= 750) & (F.big_net.abs() >= 0.6 * F.big_vol), np.sign(F.big_net)),
                          ("Line moved 6c+ since open", (F.drift.abs() >= 0.06) & (F.vol >= 1000), np.sign(F.drift)),
                          ("Steam 5c+ in an hour", F.steam.abs() >= 0.05, np.sign(F.steam)),
                          ("35%+ one-way flow", (F.vol >= 2000) & (F.net_share.abs() >= 0.35), np.sign(F.net_share))):
        G = F[mask]
        res = []
        for r, dd in zip(G.itertuples(), d[mask]):
            side = 0 if dd > 0 else 1
            price = r.ask_a if side == 0 else r.ask_b
            if not (0.05 <= price <= 0.95):
                continue
            win = (r.y == 1) if side == 0 else (r.y == 0)
            c = price + fee(price)
            res.append((1 - c) / c if win else -1.0)
        for fade in (False, True):
            if res:
                a = np.array(res)
                if fade:
                    continue
                print(f"  follow {name:36s} n={len(a):5d} ROI {a.mean() * 100:+.1f}% ± {a.std() / np.sqrt(len(a)) * 100:.1f}")
                sig_rows.append({"rule": f"Follow: {name}", "n": int(len(a)), "win": float(np.mean(a > 0)), "roi": float(a.mean())})
    # combined: model value AND money agrees
    print("\n== model value (EV ≥ 3%) when money flow agrees vs disagrees (held-out)")
    T2 = T.copy()
    T2["mdir"] = np.sign(T2.big_net.where(T2.big_vol >= 750, 0) + (T2.px - T2.px_first) * 10000 * (T2.vol >= 1000))
    for agree in (1, -1, 0):
        res = []
        for r in T2.itertuples():
            for side in (0, 1):
                ps = r.p_model if side == 0 else 1 - r.p_model
                price = r.ask_a if side == 0 else r.ask_b
                if not (0.05 <= price <= 0.95):
                    continue
                if ps / (price + fee(price)) - 1 < 0.03:
                    continue
                d = r.mdir if side == 0 else -r.mdir
                if (agree == 0 and d != 0) or (agree != 0 and d != agree):
                    continue
                win = (r.y == 1) if side == 0 else (r.y == 0)
                c = price + fee(price)
                res.append((1 - c) / c if win else -1.0)
        if res:
            a = np.array(res)
            lab = {1: "money agrees", -1: "money disagrees", 0: "no money signal"}[agree]
            print(f"  {lab:18s} n={len(a):5d} ROI {a.mean() * 100:+.1f}% ± {a.std() / np.sqrt(len(a)) * 100:.1f}")
            rows.append({"rule": f"Model EV ≥ 3% and {lab}", "n": int(len(a)), "win": float(np.mean(a > 0)), "roi": float(a.mean())})
    if args.write:
        f = ROOT / "model" / "coef.json"
        d = json.loads(f.read_text())
        d["market"] = {"a": round(float(full.coef_[0][0]), 4), "b": round(float(full.coef_[0][1]), 4), "c": round(float(full.intercept_[0]), 4), "n": int(len(M))}
        d["kalshi_backtest"] = {"period": f"{M.date.min()} to {M.date.max()}, {len(M)} matches", "rows": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "se"} for r in rows + sig_rows],
            "note": "Held-out = the most recent 40% of matches, never used to fit the blend. Entry at the real Kalshi ask about an hour before the start, plus the taker fee."}
        f.write_text(json.dumps(d, indent=1))
        print("wrote model/coef.json")


if __name__ == "__main__":
    main()
