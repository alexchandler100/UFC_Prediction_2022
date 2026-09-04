# Robust Bayesian Kelly

This is a paper-only way to size moneyline and fight-total bets while
acknowledging that the estimated chance is uncertain. It does not decide which
bets qualify and cannot place a bet.

## Why ordinary Bayesian Kelly is not enough

For any fixed stake, Kelly's expected log growth is linear in the unknown win
chance. Averaging that growth over a Bayesian probability distribution gives
the same answer as inserting the distribution's average probability into the
ordinary Kelly formula. Uncertainty alone therefore does not reduce the stake.

The implemented rule makes the safety preference explicit. For moneylines it
calibrates the market consensus. For totals it calibrates the duration model
separately at 0.5, 1.5, 2.5, 3.5, and 4.5 rounds:

1. Fit `logit(true chance) = slope * logit(original chance)`, with a prior
   centered on slope 1. There is no intercept. For totals, every Under draw is
   exactly one minus the matching Over draw.
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

The totals calibration uses 997 duration-model predictions from 81 events,
dated 2024-10-05 through 2026-08-29. Those fights occurred after the duration
model's training period. Within them, the check fitted calibration on the
earlier 787 fights and tested it on the following 210 fights. Probability error
improved at 1.5 rounds (log loss 0.73380 to 0.72412) and 2.5 rounds (0.71100 to
0.69759). The 3.5- and 4.5-round checks also improved, but each used only 14
later fights, which is far too small for a strong conclusion. Calibration at
0.5 rounds was slightly worse (0.40817 to 0.41198), so the system refuses to
use that adjustment for Kelly sizing.

## Scope and limitations

- Market-consensus moneylines and total-round lines that pass their later-fight
  calibration check are supported. Method-of-victory markets still need their
  own fitted uncertainty model.
- The totals posterior measures uncertainty in a Bayesian calibration of the
  existing duration model; it is not a full Bayesian refit of all of that
  model's hundreds of coefficients.
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
