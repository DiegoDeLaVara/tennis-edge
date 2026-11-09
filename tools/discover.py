r"""Monthly auto-discovery (the workflow checks weekly; a full scan runs when the last one is 28+ days old or the archive
has only just reached MIN_ROWS; --force to rescan by hand): scan the leak-free match archive for
single-variable rules, with the same discover (oldest 50%) / validate (next 25%) / test (newest 25%) split as
tools/archive_study, then repeat the whole scan on LUCK runs (results re-drawn from the closing price, so no rule is
real). Rules are kept only when more held in all three periods than in EVERY luck run; otherwise nothing is kept.

Kept rules go to state/discovered.json; the board paper-tracks them (edge/tuner.py) and they can only earn a stake
through the normal promotion guard (150+ live bets, 90% ROI floor above 0, positive closing-line value).

  .venv\Scripts\python.exe -m tools.discover [--local out] [--luck 8]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tools.archive_study import flatten, load
from tools.kalshi_backtest import fee

ROOT = Path(__file__).resolve().parents[1]
MIN_ROWS = 600
SKIP = ("date", "series", "fw", "ask_f", "ask_d")
BANDS = {"any price": (0.0, 1.01), "toss-up <58%": (0.0, 0.58), "lean 58-70%": (0.58, 0.70), "confident 70%+": (0.70, 1.01)}


def scan(D, fw):
    n = len(D)
    split = np.zeros(n, int); split[n // 2:] = 1; split[(3 * n) // 4:] = 2
    R = {}
    for side, price, won in (("fav", D.ask_f.values.astype(float), fw == 1), ("dog", D.ask_d.values.astype(float), fw == 0)):
        c = price + fee(price)
        R[side] = np.where((price >= 0.05) & (price <= 0.95), np.where(won, (1 - c) / c, -1.0), np.nan)
    pf = D.pf.values
    bm = {b: (pf >= lo) & (pf < hi) for b, (lo, hi) in BANDS.items()}

    def stat(r, m):
        x = r[m]; x = x[np.isfinite(x)]
        return len(x), (x.mean() if len(x) else 0.0), (x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 1.0)
    cands = []
    for col in [c for c in D.columns if c not in SKIP]:
        v = pd.to_numeric(D[col], errors="coerce").values.astype(float); ok = np.isfinite(v)
        if ok[split == 0].sum() < 100:
            continue
        u = np.unique(v[(split == 0) & ok])
        conds = [("=", x, ok & (v == x)) for x in u] if len(u) <= 3 else \
            [(">=", q, ok & (v >= q)) for q in np.unique(np.nanquantile(v[(split == 0) & ok], [0.1, 0.25, 0.75, 0.9]))] + \
            [("<=", q, ok & (v <= q)) for q in np.unique(np.nanquantile(v[(split == 0) & ok], [0.1, 0.25, 0.75, 0.9]))]
        for op, thr, cm in conds:
            for b, m_b in bm.items():
                for side in ("fav", "dog"):
                    k, mu, se = stat(R[side], cm & m_b & (split == 0))
                    if k >= 40 and mu >= 0.03 and mu / max(se, 1e-9) >= 1.0:
                        cands.append((col, op, float(thr), b, side, cm & m_b, k, mu, se))
    cands.sort(key=lambda r: -r[7] / r[8])
    held = []
    for col, op, thr, b, side, m, k, mu, se in cands[:60]:
        kv, mv, _ = stat(R[side], m & (split == 1))
        kt, mt, st = stat(R[side], m & (split == 2))
        if kv >= 20 and mv > 0 and kt >= 20 and mt > 0:
            held.append({"col": col, "op": op, "thr": round(thr, 4), "band": b, "side": side,
                         "discover": [k, round(mu, 4)], "validate": [kv, round(mv, 4)], "test": [kt, round(mt, 4), round(st, 4)]})
    return held, len(cands)


def main():
    luck = int(sys.argv[sys.argv.index("--luck") + 1]) if "--luck" in sys.argv else 8
    out = ROOT / "state" / "discovered.json"
    try:
        prev = json.loads(out.read_text())
    except (OSError, ValueError):
        prev = {}
    if prev.get("candidates") is not None and time.time() - prev.get("ts", 0) < 28 * 86400 and "--force" not in sys.argv:
        print("last full scan was under 28 days ago; nothing to do (the workflow checks weekly)"); return
    D = flatten(load())
    meta = {"ts": round(time.time()), "n_archive": len(D)}
    if len(D) < MIN_ROWS:
        meta["status"] = f"waiting: {len(D)} graded archive matches, need {MIN_ROWS} for a fair three-way split"
        out.write_text(json.dumps({**meta, "rules": []}, indent=1))
        print(meta["status"]); return
    D = D.sort_values("date", kind="stable").reset_index(drop=True)
    held, n_c = scan(D, D.fw.values)
    rng = np.random.default_rng(int(time.time()) % 100000)
    lucky = []
    for _ in range(luck):
        fake = (rng.random(len(D)) < D.pf.values).astype(int)
        lucky.append(len(scan(D, fake)[0]))
    meta.update({"candidates": n_c, "held": len(held), "luck_held": lucky})
    keep = held if len(held) > max(lucky) else []
    meta["status"] = (f"kept {len(keep)} rule(s): {len(held)} held in all three periods vs at most {max(lucky)} in {luck} luck runs"
                      if keep else f"nothing kept: {len(held)} held in all three periods vs up to {max(lucky)} in luck runs")
    rules = []
    for i, r in enumerate(keep[:10]):
        txt = f"back the {'favourite' if r['side'] == 'fav' else 'underdog'} when {r['col'].replace('_', ' ')} {r['op']} {r['thr']:g} ({r['band']})"
        rules.append({**r, "id": f"auto_{time.strftime('%Y%m%d')}_{i + 1}", "text": txt,
                      "why": f"Auto-found {time.strftime('%Y-%m-%d')} on {len(D)} archived matches: ROI {r['discover'][1]:+.1%} / "
                             f"{r['validate'][1]:+.1%} / {r['test'][1]:+.1%} in the three periods; more rules held than in any of "
                             f"{luck} luck runs. Paper only until 150+ live bets prove it."})
    out.write_text(json.dumps({**meta, "rules": rules}, indent=1))
    print(meta["status"])
    for r in rules:
        print("  ", r["id"], r["text"], r["discover"], r["validate"], r["test"])


if __name__ == "__main__":
    main()
