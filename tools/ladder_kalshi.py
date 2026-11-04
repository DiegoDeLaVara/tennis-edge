r"""Ladders on Kalshi tennis at the real pre-match ask + taker fee, in match order. Compares flat 1u with restaking
part of the current run's profit (reset after a loss or after N wins), on a few bet streams.

  .venv\Scripts\python.exe -m tools.ladder_kalshi
"""
import numpy as np

from tools.book_study import ladder_table
from tools.kalshi_backtest import load, fee


def odds(price):
    c = price + fee(price)
    return 1 / c


def main():
    K = load().sort_values(["date", "qts"]).reset_index(drop=True)
    print(f"Kalshi matches with a real pre-match quote: {len(K):,}  ({K.date.min()} to {K.date.max()})")
    favs, heavy, chall, tour = [], [], [], []
    for r in K.itertuples():
        fa = r.px >= 0.5
        fp = r.ask_a if fa else r.ask_b
        won = (r.y == 1) == fa
        if 0.67 <= fp <= 0.83:
            favs.append((odds(fp), won))
            (chall if "CHALLENGER" in r.series else tour).append((odds(fp), won))
        if 0.83 < fp <= 0.93:
            heavy.append((odds(fp), won))
    ladder_table("A. Kalshi favourites priced 67-83c (decimal ~1.2-1.5)", favs)
    ladder_table("B. Kalshi heavy favourites 83-93c", heavy)
    ladder_table("C. Challenger favourites 67-83c", chall)
    ladder_table("D. Tour-level (ATP/WTA) favourites 67-83c", tour)


if __name__ == "__main__":
    main()
