"""Which venue's pre-match price is sharper on the same matches: Polymarket or Kalshi? And does buying on Kalshi when
it is cheaper than Polymarket pay? Polymarket keeps ~8 days of closed tennis events, so the overlap is small.

  .venv\\Scripts\\python.exe -m tools.venue_test
"""
import numpy as np

from edge import smart
from edge.names import key
from tools.kalshi_backtest import load, fee, ll


def main():
    K = load()
    idx = {}
    for r in K.itertuples():
        idx[(r.ka, r.kb)] = (r, False)
        idx[(r.kb, r.ka)] = (r, True)
    evs = smart.match_events(True, pages=60)
    print("polymarket closed matches", len(evs))
    rows = []
    for e in evs:
        if max(e["prices"]) < 0.98 or e["volume"] < 500:
            continue
        st = smart.parse_t(e["start"])
        hit = None
        for tour in ("atp", "wta"):
            hit = idx.get((key(e["a"], tour), key(e["b"], tour)))
            if hit:
                break
        if not hit or not st:
            continue
        r, flip = hit
        if not e.get("tokens") or not r.qts:
            continue
        h = smart.get("https://clob.polymarket.com/prices-history", {"market": e["tokens"][0], "startTs": int(r.qts - 3 * 3600), "endTs": int(r.qts), "fidelity": 10})
        hist = (h or {}).get("history") or []
        if not hist:
            continue
        pa = float(hist[-1]["p"])
        poly_a = pa if not flip else 1 - pa                    # in Kalshi's A orientation
        rows.append({"y": r.y, "kalshi": r.px, "poly": poly_a, "ask_a": r.ask_a, "ask_b": r.ask_b, "vol": e["volume"]})
    print("matched", len(rows))
    if not rows:
        return
    y = np.array([x["y"] for x in rows])
    k = np.array([x["kalshi"] for x in rows])
    p = np.clip(np.array([x["poly"] for x in rows]), 0.01, 0.99)
    print(f"log-loss  kalshi {ll(y, k):.4f}  polymarket last trade {ll(y, p):.4f}  average {ll(y, (k + p) / 2):.4f}")
    for gap in (0.02, 0.04, 0.06):
        res = []
        for x in rows:
            for side in (0, 1):
                fair = x["poly"] if side == 0 else 1 - x["poly"]
                price = x["ask_a"] if side == 0 else x["ask_b"]
                if 0.05 <= price <= 0.95 and fair - (price + fee(price)) >= gap:
                    c = price + fee(price)
                    win = (x["y"] == 1) if side == 0 else (x["y"] == 0)
                    res.append((1 - c) / c if win else -1.0)
        if res:
            a = np.array(res)
            print(f"buy Kalshi when Polymarket is {int(gap * 100)}c+ higher (after fee): n={len(a)} ROI {a.mean() * 100:+.1f}% ± {a.std() / np.sqrt(len(a)) * 100:.1f}")


if __name__ == "__main__":
    main()
