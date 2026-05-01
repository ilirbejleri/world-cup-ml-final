# 2026 Forecast Diagnostics

## Knockout Draw Policy

All knockout-stage probability vectors are post-processed so P(draw)=0 and win/loss probabilities are renormalized. The calibrated score simulator samples from decisive score cells for knockout matches, with a fallback stochastic tiebreak if numerical rounding ever produces a tied score.

## Primary Forecast Simulator

Primary simulator: NN-original-18 calibrated score simulator. Match outcome probabilities come from the selected W/D/L model, while the Poisson score layer supplies calibrated scorelines for group goal-difference and goals-for tiebreakers.

## Spain Gap Against Betting Markets

Model Spain champion probability: 0.1486.
Model Spain champion rank: 1.
Model Spain Elo rank in configured field: 1.
Model Spain group advancement probability: 0.9494.

The final 2026-only forecast now updates Elo, the outcome model, and the Poisson score layer with supplemental international matches from 2022 through March 2026. Any remaining gap to market-implied probabilities should be interpreted as a limitation of the simplified knockout bracket and missing squad/injury/current-depth information rather than a claim that markets are wrong.