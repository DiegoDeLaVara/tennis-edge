# Tennis Edge

A tennis betting board for Kalshi that refreshes itself every ~20 minutes in GitHub Actions and publishes a
phone-friendly page on GitHub Pages. No PC or Claude session needs to be running.

## What runs where

| Piece | Where | What |
|---|---|---|
| `tools/build_history.py` | local, once | Replays ~1.6M matches (Sackmann archive to 2026-06, then TennisMyLife, ESPN and Kalshi results) into `state/ratings.json.gz` |
| `tools/fit.py` | local | Walk-forward fit of the pre-match model -> `model/coef.json` |
| `tools/kalshi_collect.py`, `tools/kalshi_backtest.py` | local | Pre-match bid/ask and trade tape for settled Kalshi matches; tests model, market blend and money-flow rules against real prices (`--write` stores the blend) |
| `edge/refresh.py` | cloud, every 20 min | New results -> ratings; prices every open Kalshi tennis market (match, sets, games, set score); money flow; Polymarket smart wallets; picks with units; paper record |
| `site/index.html` | GitHub Pages | Picks, Matches, My bets (tracker, stored on the phone), Model record, How |

Rolling state (`ratings.json.gz`, `log.json`, `smart.json`, `settled.json`) lives on the `gh-pages` branch next to
`board.json`; the branch is force-pushed each run so it never grows.

## Model in one paragraph

Surface-blended, margin-weighted Elo + opponent-adjusted serve/return rates (decayed) + form residual, fatigue,
layoff, head-to-head, fitted as a logistic model (walk-forward 2016-26: log-loss 0.573 vs 0.578 Elo-only,
calibrated within ~0.5 pts per decile). Serve-point probabilities solved to match the final win probability feed an
exact Markov model of games/tiebreaks/sets with a "form on the day" random effect (fitted so three-set rate and
total games match reality), which prices set scores, totals, spreads, set winners and same-match combos.

## What the Kalshi backtest found (5,500 matches, June 2025 - Nov 2026)

Priced at the real ask about an hour before each match: Kalshi's price beat the ratings model on log-loss
(0.594 vs 0.612) and took ~90% of the weight in the fitted blend; betting model "edges" broke even. Tailing big
tickets or one-way flow lost ~3% after fees; line moves, steam and top Polymarket wallets broke even. So fair prices
lean on the market and every strategy is paper-tracked (half stakes or none) until its live record proves it.

## Local commands

```
.venv\Scripts\python.exe -m edge.refresh --data out          # one refresh into ./out
.venv\Scripts\python.exe -m tools.kalshi_backtest --write    # re-test and store the market blend
```

Not financial advice.
