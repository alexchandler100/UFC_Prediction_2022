# Robust Bayesian Kelly

This is a paper-only way to size moneyline bets while acknowledging that the
estimated win chance is uncertain. It does not decide which bets qualify and
cannot place a bet.

## Why ordinary Bayesian Kelly is not enough

For any fixed stake, Kelly's expected log growth is linear in the unknown win
chance. Averaging that growth over a Bayesian probability distribution gives
the same answer as inserting the distribution's average probability into the
ordinary Kelly formula. Uncertainty alone therefore does not reduce the stake.

The implemented rule makes the safety preference explicit:

1. Fit `logit(true chance) = slope * logit(market chance)`, with a prior
   centered on slope 1. There is no intercept, so swapping the fighters gives
   exactly the complementary probability.
2. Preserve 257 equally weighted values from the fitted slope distribution.
3. Recalculate the fight probability for every slope value.
4. Use the lower 10th-percentile probability for Kelly sizing. A positive
   stake therefore requires a positive edge even at that conservative point.
5. Cap one bet at 5% of the current bankroll. This separately limits exposure
   to errors not represented by the calibration and to related bets on one
   event.

The website shows the original probability, fitted average, middle 80% range,
chance of a positive edge, uncapped stake, final stake, and whether the cap
was applied.

## Data and later-data check

The frozen artifact uses 503 completed fights from 58 events, dated 2022-04-23
through 2026-04-04. Each row is a timestamped pre-fight multi-book consensus
with the actual winner.

The check fit the model on the earlier 363 fights from 46 events, through
2025-10-25, and evaluated it on the next 140 fights from 12 events, dated
2025-11-01 through 2026-04-04. Log loss moved from 0.55707 to 0.55584 and Brier
score from 0.18759 to 0.18714. Winner accuracy stayed 74.3%. Those small gains
support using calibration as a research option; they do not establish betting
profitability.

## Scope and limitations

- Only market-consensus moneyline probabilities are supported. Totals and
  method-of-victory markets need separate fitted uncertainty models.
- The fitted uncertainty describes how the consensus is calibrated across
  fights. It does not capture every matchup-specific unknown.
- The 5% cap is an additional risk limit, not a Bayesian result.
- Same-card bets are still sized independently before a final cash-availability
  scaling step. A future portfolio model should handle their shared risk.
- Full, half, and one-third Kelly remain in the interface as comparisons.
- Production execution remains disabled. Prospective paper results are needed
  before choosing any real-money policy.

Regenerate and validate the frozen calibration artifact with:

```bash
PYTHONPATH=src python src/fit_bayesian_kelly.py
python src/update_bet_performance.py
```
