r"""Seed the live Polymarket sharps ranking (edge/smart.rank_wallets, closing-line value) with the wallet study's
history (work/wallets.jsonl: every pre-match position on ~11,600 tennis markets, Jul-Nov 2026), so the board starts
with months of evidence instead of the API's last ~2 weeks.

Same rules as rank_wallets: one-sided pre-match positions only, closing-line value = last pre-match price minus the
wallet's average entry, dollar-weighted (capped $5,000 per market), decayed with a 120-day half-life to today.

  .venv\Scripts\python.exe -m tools.seed_sharps        -> state/sharps_seed.json.gz
"""
import gzip
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HALF_LIFE = 120.0


def main():
    now = time.time()
    agg, done = {}, []
    live = ROOT / "work" / "live_done.json"          # markets the live ranking already counted (avoid double-counting)
    skip = set(json.loads(live.read_text())) if live.exists() else set()
    for l in open(ROOT / "work" / "wallets.jsonl"):
        r = json.loads(l)
        if r["cid"] in skip:
            continue
        done.append(r["cid"])
        p0, win = r["p0"], r["win"]
        f = 0.5 ** (max(now - r["start"], 0) / 86400 / HALF_LIFE)
        for w, (s0, s1, cash) in r["w"].items():
            if s0 > 0 >= s1 and cash > 0:
                side_p, sh, won = p0, s0, win == 0
            elif s1 > 0 >= s0 and cash > 0:
                side_p, sh, won = 1 - p0, s1, win == 1
            else:
                continue
            avg = cash / sh
            if not 0.02 < avg < 0.98:
                continue
            usd = min(cash, 5000.0)
            a = agg.setdefault(w, [0.0, 0.0, 0.0, w[:8], 0.0, r["start"]])
            a[0] += (side_p - avg) * usd * f
            a[1] += usd * f
            a[2] += f
            a[4] += ((sh if won else 0.0) - cash) * f
            a[5] = max(a[5], r["start"])
    agg = {w: [round(a[0], 2), round(a[1], 2), round(a[2], 3), a[3], round(a[4], 2), round(a[5])] for w, a in agg.items() if a[2] >= 1}
    out = {"built": now, "agg": agg, "done": done, "source": "tools/wallet_study cache, Jul-Nov 2026"}
    p = ROOT / "state" / "sharps_seed.json.gz"
    with gzip.open(p, "wt") as fh:
        json.dump(out, fh, separators=(",", ":"))
    q = [a for a in agg.values() if a[2] >= 15 and a[1] >= 500 and a[0] / a[1] * a[2] / (a[2] + 20) > 0.02]
    print(f"{len(agg):,} wallets, {len(done):,} markets; {len(q)} would qualify as sharps -> {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
