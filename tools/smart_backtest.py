r"""Out-of-sample test of Polymarket smart money: rank wallets on matches 30-120 days ago, then on the last 30 days
follow their net pre-match buying at the last pre-match price.  .venv\Scripts\python.exe -m tools.smart_backtest"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from edge import smart

ROOT = Path(__file__).resolve().parents[1]


def main():
    since = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(time.time() - 120 * 86400))
    evs = smart.match_events(True, pages=60, since_iso=since)
    print("closed match events", len(evs))
    R = smart.rank_wallets(min_markets=8, top=400, max_events=1500, until_days=4, events=evs)
    sw = {w["wallet"]: w for w in R["wallets"]}
    print("ranked on older matches:", R["events"], "events,", len(sw), "wallets")
    test = [e for e in evs if (smart.parse_t(e["start"]) or 0) >= time.time() - 4 * 86400 and e["volume"] >= 500 and max(e["prices"]) > 0.98]
    print("test events", len(test))

    def one(e):
        st = smart.parse_t(e["start"])
        tr = [t for t in smart.trades(e["cid"]) if st and t.get("timestamp", 0) < st]
        if not tr:
            return None
        tr.sort(key=lambda t: t["timestamp"])
        last_px = None
        net = [0.0, 0.0]
        for t in tr:
            i = int(t.get("outcomeIndex", 0))
            px = float(t.get("price") or 0)
            last_px = px if i == 0 else 1 - px
            if t.get("proxyWallet") in sw:
                v = float(t.get("size") or 0) * px * (1 if t.get("side") == "BUY" else -1)
                net[i] += v
        win = 0 if e["prices"][0] > 0.5 else 1
        return {"net": net, "px": last_px, "win": win, "vol": e["volume"]}
    with ThreadPoolExecutor(6) as ex:
        res = [r for r in ex.map(one, test) if r and r["px"] and 0.03 < r["px"] < 0.97]
    print("test events with pre-match trades", len(res))
    for th in (50, 200, 500):
        out = []
        for r in res:
            d = r["net"][0] - r["net"][1]
            if abs(d) < th:
                continue
            side = 0 if d > 0 else 1
            price = r["px"] if side == 0 else 1 - r["px"]
            price = min(price + 0.01, 0.99)
            out.append((1 - price) / price if r["win"] == side else -1.0)
        if out:
            a = np.array(out)
            print(f"follow smart wallets, net >= ${th}: n={len(a)} win {np.mean(a > 0):.3f} ROI {a.mean() * 100:+.1f}% ± {a.std() / np.sqrt(len(a)) * 100:.1f}")
    json.dump({"n_test": len(res)}, open(ROOT / "work" / "smart_bt.json", "w"))


if __name__ == "__main__":
    main()
