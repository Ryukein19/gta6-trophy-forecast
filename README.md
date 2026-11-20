[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ryukein19/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ryukein19/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase one week after launch. Entry for the make.fun bounty
[Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## The model

An earners / players decomposition, fitted the way the counts actually
behave:

- **Players tracked** `N(t)` are fitted with a saturating (logistic) curve
  in log-time. New players dilute every rate, so the denominator's growth
  is the dilution scale; the implied earners at the target are
  `rate * N`, which keeps the forecast count-consistent.
- **Each trophy's rate** is fitted with a saturating curve in log-time.
  The rate is the quantity scored, and it is the saturating curve:
  earners are players x progress, a product of two logistics, so fitting
  earners as a single logistic is misspecified and its extrapolation runs
  the wrong way. Fitting the rate directly captures the net of earning
  flow (players progressing push it up) against dilution (new players push
  it down).
- **Two guards keep a far extrapolation honest.** The rate's asymptote is
  capped: it cannot exceed the observed range by more than twice the
  observed movement (and never 100), so a trophy still rising steeply
  cannot be forecast at 99.99. And the extrapolation is damped toward
  the last observed rate in logit space, weighted by how far the target
  sits beyond the observed log-time range.
- **Quantiles** are centred on the forecast rate with a spread that grows
  with the case's horizon: a random-walk floor plus the fit's own
  residual uncertainty and the dilution uncertainty from the players
  curve, widened the further the target sits beyond the observed range.

## Run

Python 3.13, standard library only, no network.

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the
output directory: one row per trophy, five quantiles (5%, 25%, 50%, 75%,
95%) of its earn rate at the case's target.

## Validation

- Practice cases (`p00`, `p01`): both outputs valid (55 rows, five
  non-decreasing quantiles, 0–100, under the byte cap). Their history is
  flat, so the medians equal the last read rates, as they should.
- Synthetic backtest (realistic launch-week growth, 6,510-minute
  horizon): the model beats the house persistence baseline by ~51% on
  pinball loss in two scenarios (logistic-in-log-time truth and
  exponential-saturation truth), with a much smaller median error
  (1.7pp / 4.0pp vs 6.5pp / 10.3pp for persistence).
