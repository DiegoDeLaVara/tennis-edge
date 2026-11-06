r"""Research on the match archive: exact pre-match snapshots logged live by the board (edge/refresh.update_archive),
so there is no leakage from history files. Downloads archive/*.jsonl from the site, flattens every snapshot into one
row of variables (prices, model parts, context for favourite and underdog, risk net, money flow, verdict), then runs
the same discover (oldest 50%) / validate (next 25%) / test (newest 25%) rule scan with luck checks as tools/rule_scan.

  .venv\Scripts\python.exe -m tools.archive_study [--local out] [--null SEED]
Re-run monthly: the archive grows by a few hundred matches a day.
"""
import gzip
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from tools.kalshi_backtest import fee

SITE = "https://diegodelavara.github.io/tennis-edge/"


def load():
    rows = []
    if "--local" in sys.argv:
        d = Path(sys.argv[sys.argv.index("--local") + 1]) / "archive"
        files = sorted(d.glob("*.jsonl*"))
        texts = [(gzip.decompress(f.read_bytes()) if f.suffix == ".gz" else f.read_bytes()).decode() for f in files]
    else:
        board = json.loads(urllib.request.urlopen(SITE + "board.json").read())
        names = (board.get("archive") or {}).get("files") or []
        raw = [urllib.request.urlopen(SITE + "archive/" + n).read() for n in names]
        texts = [(gzip.decompress(b) if n.endswith(".gz") else b).decode() for n, b in zip(names, raw)]
    for t in texts:
        rows += [json.loads(l) for l in t.splitlines() if l.strip()]
    return rows


def flatten(rows):
    out = []
    for e in rows:
        if e.get("y") is None or e.get("p_close") is None:
            continue
        pa = e["p_close"]
        fa = pa >= 0.5
        sf, sd = ("_a", "_b") if fa else ("_b", "_a")
        q = e["q"]
        ask_f, ask_d = (q[0]["ask"], q[1]["ask"]) if fa else (q[1]["ask"], q[0]["ask"])
        md, cx, fl = e.get("model") or {}, e.get("ctx") or {}, e.get("flow") or {}
        o = {"date": pd.Timestamp(e["start"], unit="s"), "series": e["series"], "pf": pa if fa else 1 - pa,
             "fw": int(e["y"] == 1) if fa else int(e["y"] == 0), "ask_f": ask_f, "ask_d": ask_d,
             "open_move_to_fav": (pa - e["p_open"]) * (1 if fa else -1),
             "model_edge_fav": None if md.get("p_model") is None else (md["p_model"] if fa else 1 - md["p_model"]) - (pa if fa else 1 - pa),
             "mrate_edge_fav": None if md.get("p_mrate") is None else (md["p_mrate"] if fa else 1 - md["p_mrate"]) - (pa if fa else 1 - pa),
             "sub_model_disagreement": md.get("dis"), "risk_net": (e.get("risk") or {}).get("net"),
             "flow_share_to_fav": None if fl.get("net_share") is None else fl["net_share"] * (1 if fa else -1),
             "big_money_to_fav": None if fl.get("big_net") is None else np.sign(fl["big_net"]) * (1 if fa else -1),
             "volume": fl.get("vol"), "itf": int("ITF" in e["series"])}
        if md.get("form"):
            f = md["form"]; o["form_gap_fav"] = (f[0] - f[1]) * (1 if fa else -1)
        if md.get("load"):
            l = md["load"]; o["fav_mins_10d"], o["dog_mins_10d"] = (l[0], l[1]) if fa else (l[1], l[0])
        for k in ("mins3", "m7", "m30", "days", "age", "rank", "streak", "wins10", "home", "lefty", "switch"):
            if k + "_a" in cx:
                o[f"fav_{k}"], o[f"dog_{k}"] = cx.get(k + sf), cx.get(k + sd)
        out.append(o)
    return pd.DataFrame(out)


def main():
    rows = load()
    D = flatten(rows)
    print(f"archived matches with results: {len(D):,}")
    if len(D) < 600:
        print("too few yet for a fair three-way split; the archive grows by a few hundred matches a day. Re-run in a few weeks.")
        return
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    n = len(D)
    if "--null" in sys.argv:
        rng = np.random.default_rng(int(sys.argv[sys.argv.index("--null") + 1]))
        D["fw"] = (rng.random(n) < D.pf.values).astype(int)
    split = np.zeros(n, int); split[n // 2:] = 1; split[(3 * n) // 4:] = 2
    R = {}
    for side, price, won in (("fav", D.ask_f.values.astype(float), D.fw.values == 1), ("dog", D.ask_d.values.astype(float), D.fw.values == 0)):
        c = price + fee(price)
        R[side] = np.where((price >= 0.05) & (price <= 0.95), np.where(won, (1 - c) / c, -1.0), np.nan)
    bands = {"any price": np.ones(n, bool), "toss-up <58%": (D.pf < 0.58).values, "lean 58-70%": D.pf.between(0.58, 0.70).values,
             "confident 70%+": (D.pf >= 0.70).values}
    stat = lambda r, m: (lambda x: (len(x), x.mean() if len(x) else 0, x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 1))(r[m][np.isfinite(r[m])])
    rules = []
    for col in [c for c in D.columns if c not in ("date", "series", "fw", "ask_f", "ask_d")]:
        v = pd.to_numeric(D[col], errors="coerce").values.astype(float); ok = np.isfinite(v)
        if ok[split == 0].sum() < 100:
            continue
        u = np.unique(v[(split == 0) & ok])
        conds = [(f"{col} = {x:g}", ok & (v == x)) for x in u] if len(u) <= 3 else \
            [(f"{col} >= {q:.3g}", ok & (v >= q)) for q in np.unique(np.nanquantile(v[(split == 0) & ok], [0.1, 0.25, 0.75, 0.9]))] + \
            [(f"{col} <= {q:.3g}", ok & (v <= q)) for q in np.unique(np.nanquantile(v[(split == 0) & ok], [0.1, 0.25, 0.75, 0.9]))]
        for cn, cm in conds:
            for bn, bm in bands.items():
                for side in ("fav", "dog"):
                    k, mu, se = stat(R[side], cm & bm & (split == 0))
                    if k >= 40:
                        rules.append((cn, bn, side, cm & bm, k, mu, se))
    good = sorted([r for r in rules if r[5] >= 0.03 and r[5] / max(r[6], 1e-9) >= 1.0], key=lambda r: -r[5] / r[6])[:60]
    surv = []
    for cn, bn, side, m, k, mu, se in good:
        kv, mv, _ = stat(R[side], m & (split == 1))
        if kv >= 20 and mv > 0:
            kt, mt, st = stat(R[side], m & (split == 2))
            surv.append((f"back {side.upper()} when {cn} ({bn})", (k, mu), (kv, mv), (kt, mt, st)))
    held = [s for s in surv if s[3][0] >= 20 and s[3][1] > 0]
    print(f"rules tested {len(rules):,}; validated {len(surv)}; held in all three periods: {len(held)}"
          + ("" if "--null" in sys.argv else "   (compare with --null 1..8 runs: real signals must beat that count)"))
    if "--null" not in sys.argv:
        for r, d, v, t in surv:
            print(f"   {r[:90]:<90} {d[0]:>4} {d[1]:+6.1%}  {v[0]:>4} {v[1]:+6.1%}  {t[0]:>4} {t[1]:+6.1%} ± {t[2]:4.1%}{'  <-- all three' if t[0] >= 20 and t[1] > 0 else ''}")


if __name__ == "__main__":
    main()
