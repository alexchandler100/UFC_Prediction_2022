# Nonlinear winner-feature experiment

The live winner model already uses 82 pre-fight differences, including age,
age squared, reach, logged fight counts, logged time since the last fight,
ratings, and career/recent striking and wrestling rates. This experiment asks
whether a few combinations of those inputs improve probability estimates.
It does not change the live model, betting recommendations, or simulator.

## November 21, 2026 results

Age with experience was the only family that improved both probability-error
scores in each selection year. Its improvement continued in both later years,
but the size of the gain was small and its uncertainty includes no improvement.
Keep the production model unchanged.

Selection results on 1,017 fights in 2023–24:

| Family | Log loss (lower is better) | Brier (lower is better) | AUC (higher is better) |
| --- | ---: | ---: | ---: |
| Baseline | 0.636969 | 0.223482 | 0.685557 |
| Age curve | 0.636458 | 0.223261 | 0.686351 |
| Age and experience | **0.636130** | **0.223094** | **0.687319** |
| Reach and activity | 0.637033 | 0.223507 | 0.685297 |
| Age and absorption | 0.636591 | 0.223294 | 0.686529 |

Age curve and age/absorption improved the combined score but worsened log
loss in 2024, so they did not qualify. AUC measures how well predictions rank
winners above losers; it does not measure probability calibration.

Final comparison on 1,014 fights across 82 cards, January 11, 2025 through
November 14, 2026:

| Measure | Baseline | Age and experience |
| --- | ---: | ---: |
| Log loss | 0.629742 | 0.629206 |
| Brier | 0.219569 | 0.219371 |
| AUC | 0.702370 | 0.703188 |
| Winner accuracy | 64.60% | 64.50% |
| Calibration slope (ideal 1) | 0.9717 | 0.9751 |
| Calibration intercept (ideal 0) | -0.0818 | -0.0830 |
| Ten-bin calibration error (lower is better) | 0.02054 | 0.02593 |

The candidate-minus-baseline log-loss difference is -0.000535, with a
95% whole-card resampling interval of -0.001557 to +0.000461. This interval
includes a small worsening. Calibration results are mixed, so this is not
evidence that the new model is uniformly better calibrated. Predictions move
by about 0.60 percentage points on average; the median change is 0.45 points.

The 1,003 discovery fights from 2021–22 showed noisy, non-monotonic error
patterns rather than a clear missing age or reach curve. The largest
relative-layoff bin overpredicted wins by about ten percentage points across
125 fights, but this is descriptive and earlier recovery-feature research
already found no convincing improvement. It should not trigger another
unrestricted search of layoff thresholds. Some threshold bins contained just
one or ten fights; the plots omit bins below 30 and display the remaining
fight counts to avoid treating those isolated outcomes as a pattern.

The most defensible follow-up is to keep this one-term age/experience
candidate frozen for future evaluation. More polynomial terms are not
justified by these results. Any later betting use still needs evidence
against contemporaneous market prices, not just against our baseline.

## What is tested

Four fixed families compete separately against the unchanged baseline:

| Family | Additional inputs per fighter |
| --- | --- |
| Age curve | Years above age 30 and above age 35 |
| Age and experience | Centered age multiplied by logged prior UFC fight count |
| Reach and activity | Centered reach multiplied by recent striking rate; centered reach multiplied by recent takedown rate |
| Age and absorption | Centered age multiplied by career strike absorption rate; centered age multiplied by career knockdown absorption rate |

Each expression is calculated for both fighters before taking the difference.
For example, `(age_A - 29) * experience_A - (age_B - 29) * experience_B`
reverses sign when fighters swap. Multiplying age difference by experience
difference would not reverse sign, and would violate the model's requirement
that the two fighters' win probabilities sum to one.

Unknown age or reach contributes zero to the relevant new terms. These are
UFC-record experience and activity measures, not complete career histories.
Absorption rates do not measure medical damage. The reach experiment is a
simple activity interaction, not a complete representation of opponent style.

## Evaluation protocol

1. Replay features before each fight and verify that all original features
   match the saved baseline. Count each fight once.
2. Use baseline predictions on 2021–22 fights to display descriptive error
   patterns. These plots do not establish statistical significance.
3. Compare four prespecified families on 2023–24. A family qualifies only if
   both log loss and Brier score improve in each year. Both scores measure
   probability error; lower is better. Among qualifying families choose the
   lowest combined log loss, preferring fewer terms for near ties.
4. Save that choice before evaluating 2025–26. Do not try the rejected
   families on these test years. If none qualifies, retain the baseline.
5. Report probability error, ranking accuracy, calibration, yearly results,
   and uncertainty from resampling entire UFC cards rather than pretending
   every fight on a card is independent.

For each evaluated year, training uses only the preceding ten years. The
existing model's chronological internal comparisons select regularization
and calibration using earlier fights. Imputation and scaling are fitted on
training data. The test-year outcomes never select that year's settings.
The 2026 evaluation is a rolling update and can train on completed 2025
fights; the feature-family choice stays fixed.

The later years are reserved within this experiment, but earlier project
experiments have already examined them. Current corrected source data and
profiles are also not exact archived snapshots of what was known then.
Consequently, this is historical research, not proof of future betting profit.
The next step after a convincing improvement would be prospective recording
and comparison against available market prices before changing live bets.

## Reproduction

In PowerShell at the repository root:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.venv/Scripts/python.exe src/evaluate_nonlinear_features.py
.venv/Scripts/python.exe -m unittest discover -s tests -p test_nonlinear_research.py
```

The compact report is saved as
`src/content/data/model_research/nonlinear_feature_comparison.json`.
Full prediction CSVs, source hashes, the frozen decision, residual bins,
and plots are saved under `artifacts/nonlinear_research/<source hash>/`.
Every cached CSV is digest-checked. Changing source data or experiment code
creates a new run directory. `residuals.png` displays errors in 2021–22;
`comparison.png` displays selection scores and final-test calibration.
