"""Where, if anywhere, is Kalshi's tennis price beatable? Discover on the older 60% of matches, verify on the newest 40%.

  .venv\\Scripts\\python.exe -m tools.edge_hunt
A. Price-level bias (favourite-longshot) at the real ask.
B. Which factors predict the winner beyond the Kalshi price (residual regression), and segments where it misprices.
C. Early prices: is the first traded price softer than the pre-match price (and does the model know which way it moves)?
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tools.kalshi_backtest import load, fee, logit, ll


def bet_rows(K, side_mask_fn=None):
    """Both sides of every match as separate bets at the ask: price, win, return."""
    rows = []
    for r in K.itertuples():
        for side in (0, 1):
            price = r.ask_a if side == 0 else r.ask_b
            if not (0.02 <= price <= 0.98):
                continue
            win = int((r.y == 1) if side == 0 else (r.y == 0))
            c = price + fee(price)
            rows.append({"event": r.event, "side": side, "price": price, "win": win, "ret": (1 - c) / c if win else -1.0,
                         "series": r.series, "date": r.date})
    return pd.DataFrame(rows)


def show(lbl, a):
    a = np.asarray(a, float)
    if len(a) == 0:
        return f"{lbl:46s} n=    0"
    return f"{lbl:46s} n={len(a):5d} ROI {a.mean() * 100:+6.1f}% ± {a.std() / np.sqrt(len(a)) * 100:4.1f}"


def main():
    K = load()
    K = K.sort_values("date").reset_index(drop=True)
    cut = K.date.iloc[int(len(K) * 0.6)]
    print(f"{len(K)} matches with quotes; discover < {cut} <= verify\n")

    # ---------------- A. price-level bias
    print("== A. Buying every side at the ask, by price (fee included). Fair market = about -2..-4% (the fee + half spread)")
    Bt = bet_rows(K)
    bins = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
    Bt["bin"] = pd.cut(Bt.price, bins)
    for b, g in Bt.groupby("bin", observed=True):
        d, v = g[g.date < cut], g[g.date >= cut]
        print(f"  {str(b):14s} {show('discover', d.ret)}   {show('verify', v.ret)[0:0]}verify n={len(v):5d} ROI {v.ret.mean() * 100:+6.1f}%  win {g.win.mean():.3f} vs price {g.price.mean():.3f}")
    for s, g in Bt.groupby("series"):
        fav = g[g.price >= 0.7]
        dog = g[g.price <= 0.3]
        print(f"  {s:22s} {show('favourites 70c+', fav.ret)}  {show('dogs <=30c', dog.ret)}")

    # ---------------- B. residual: what beats the price?
    print("\n== B. Factors that predict the winner beyond the Kalshi mid (coefficients fit on the discover set)")
    M = K[K.p_model.notna()].copy()
    P = pd.read_csv("work/recent_preds.csv", low_memory=False)
    P["k1"] = P.a + "|" + P.b
    P["k2"] = P.b + "|" + P.a
    feat = ["form_a", "form_b", "days_a", "days_b", "load_a", "load_b", "h2h", "h2h_n", "n_a", "n_b", "stat_w", "pa", "pb", "surface", "round"]
    idx1 = P.drop_duplicates("k1").set_index("k1")[feat + ["date"]]
    rows = []
    for r in M.itertuples():
        k = r.ka + "|" + r.kb
        flip = False
        if k in idx1.index:
            f = idx1.loc[k]
        elif (r.kb + "|" + r.ka) in idx1.index:
            f = idx1.loc[r.kb + "|" + r.ka]; flip = True
        else:
            rows.append(None); continue
        if abs((pd.Timestamp(f["date"]) - pd.Timestamp(r.date)).days) > 7:
            rows.append(None); continue
        A, B_ = ("b", "a") if flip else ("a", "b")
        rows.append({"form": f[f"form_{A}"] - f[f"form_{B_}"],
                     "rest": np.clip(f[f"days_{A}"], 0, 60) - np.clip(f[f"days_{B_}"], 0, 60),
                     "layoff": int(f[f"days_{A}"] > 45) - int(f[f"days_{B_}"] > 45),
                     "fatigue": np.log1p(f[f"load_{A}"]) - np.log1p(f[f"load_{B_}"]),
                     "h2h": (f["h2h"] * (-1 if flip else 1)) / (f["h2h_n"] + 3),
                     "serve": (f["pa"] - f["pb"]) * (-1 if flip else 1),
                     "exp": np.log1p(f[f"n_{A}"]) - np.log1p(f[f"n_{B_}"]),
                     "surface": f["surface"]})
    F = pd.DataFrame([x if x else {} for x in rows], index=M.index)
    M = M.join(F)
    M = M[M.form.notna()].copy()
    M["lm"] = logit(M.px.values)
    M["lmod"] = logit(M.p_model.values) - M.lm
    cols = ["lmod", "form", "rest", "layoff", "fatigue", "h2h", "serve", "exp", "drift", "steam", "netflow"]
    M["drift"] = M.px - M.px_first
    M["netflow"] = M.net_share
    M = M.fillna({"steam": 0, "netflow": 0})
    D, V = M[M.date < cut], M[M.date >= cut]
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[D.lm, D[cols]], D.y)
    sd = D[cols].std()
    print("  standardised effect beyond price (positive = market underrates player A when this is high):")
    for c, w in zip(cols, lr.coef_[0][1:]):
        print(f"    {c:9s} {w * sd[c]:+.3f}")
    pv = lr.predict_proba(np.c_[V.lm, V[cols]])[:, 1]
    print(f"  verify log-loss: market {ll(V.y.values, V.px.values):.4f}  market+factors {ll(V.y.values, pv):.4f}  (n={len(V)})")
    V = V.assign(p=pv)
    for th in (0.02, 0.04, 0.07):
        res = []
        for r in V.itertuples():
            for side in (0, 1):
                ps = r.p if side == 0 else 1 - r.p
                price = r.ask_a if side == 0 else r.ask_b
                if 0.05 <= price <= 0.95 and ps / (price + fee(price)) - 1 >= th:
                    c = price + fee(price)
                    win = (r.y == 1) if side == 0 else (r.y == 0)
                    res.append((1 - c) / c if win else -1.0)
        print("  " + show(f"bet market+factors edge >= {int(th * 100)}% (verify)", res))

    print("\n== B2. Segments: does the market misprice these spots? (model-vs-market log-loss gap and ROI backing the model)")
    def seg(name, mask):
        for part, G in (("discover", D[mask(D)]), ("verify", V[mask(V)])):
            if len(G) < 60:
                continue
            res = []
            for r in G.itertuples():
                for side in (0, 1):
                    ps = r.p_model if side == 0 else 1 - r.p_model
                    price = r.ask_a if side == 0 else r.ask_b
                    if 0.05 <= price <= 0.95 and ps / (price + fee(price)) - 1 >= 0.05:
                        c = price + fee(price)
                        win = (r.y == 1) if side == 0 else (r.y == 0)
                        res.append((1 - c) / c if win else -1.0)
            gap = ll(G.y.values, G.px.values) - ll(G.y.values, G.p_model.values)
            print(f"  {name:30s} {part:8s} n={len(G):5d} model-better-by {gap:+.4f}  {show('back model edge>=5%', res)[46:]}")
    seg("Qualifying", lambda X: X["round"].astype(str).str.startswith("Q"))
    seg("Layoff 45d+ either side", lambda X: X.layoff.ne(0))
    seg("Big fatigue gap", lambda X: X.fatigue.abs() > 1.0)
    seg("Clay", lambda X: X.surface.eq("Clay"))
    seg("Grass", lambda X: X.surface.eq("Grass"))
    seg("Thin market (vol < $2k)", lambda X: X.vol < 2000)
    seg("Liquid market (vol >= $20k)", lambda X: X.vol >= 20000)
    seg("Wide spread (>= 3c)", lambda X: X.spread >= 0.03)

    # ---------------- C. early prices
    print("\n== C. First traded price vs pre-match price")
    E = K[K.px_first.notna() & K.p_model.notna()].copy()
    E = E[(E.px_first > 0.03) & (E.px_first < 0.97)]
    print(f"  log-loss: first trade {ll(E.y.values, E.px_first.values):.4f}  pre-match mid {ll(E.y.values, E.px.values):.4f}  model {ll(E.y.values, E.p_model.values):.4f}  n={len(E)}")
    E["dir_model"] = np.sign(E.p_model - E.px_first)
    E["moved"] = np.sign(E.px - E.px_first)
    agree = (E.dir_model == E.moved) & (E.moved != 0)
    print(f"  the line moved toward the model's side {agree.mean() * 100:.1f}% of the time it moved (n={int((E.moved != 0).sum())})")
    for th in (0.05, 0.10):
        res = []
        for r in E[E.date >= cut].itertuples():
            for side in (0, 1):
                ps = r.p_model if side == 0 else 1 - r.p_model
                price = (r.px_first if side == 0 else 1 - r.px_first) + 0.02   # early books are thin: assume 2c over
                if 0.05 <= price <= 0.95 and ps / (price + fee(price)) - 1 >= th:
                    c = price + fee(price)
                    win = (r.y == 1) if side == 0 else (r.y == 0)
                    res.append((1 - c) / c if win else -1.0)
        print("  " + show(f"back model at first price, edge >= {int(th * 100)}% (verify)", res))


if __name__ == "__main__":
    main()
