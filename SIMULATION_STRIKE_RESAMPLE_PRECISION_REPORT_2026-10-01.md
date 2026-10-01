# More training resamples recover the earlier loss, but do not pass the advance rule

Date: 2026-10-01. Research only; no production forecasts or betting rules changed.

The earlier 109/228 strike-leader result was sensitive to using only ten
resampled training histories. Increasing to 100 histories recovered most of
the loss: the two fixed random runs produced **119/228 and 123/228**, compared
with **120/228** for the original predictor fitted once to all earlier data.

The candidate still does not advance. The two runs disagree on the projected
strike leader in 14 decisive fights, giving 93.86% agreement against the fixed
95% requirement. One run also falls one correct selection below the required
120. No extra samples, replacement seeds or alternative averaging rules were
tried to make it pass.

All 229 development fights on 30 cards completed in 254 seconds. Each card used
two fixed sets of 100 training-history resamples. No new fight trajectories
were generated, and neither 2025 confirmation group was evaluated.

## Fixed design

The estimator is exactly the preceding `observation_bootstrap` candidate:
original observation contexts, equal-bout weighted fighter/opponent effects,
the original eight-prior-card shrinkage choices, and no extra age, experience
or layoff adjustment. Only the number of training-history resamples changes.

Run A extends the original seed sequence from ten to 100 members. Run B uses a
separate fixed seed sequence, also with 100 members. Both were declared before
scoring. The first ten members of A must reproduce the earlier candidate.
Four fixed prefix sizes (10, 25, 50, 100) are reported for diagnosis; only the
two 100-member results enter the decision. Pooling all 200 forecasts is
descriptive and cannot override a failed individual run.

Each 100-member run must preserve at least 120 correct strike-leader selections
and improve pace/accuracy probability error over both the original marginal
fighter baseline and the rejected v2 conditional predictor. The runs must also
agree on at least 95% of strike leaders, differ by at most one percentage point
in root-mean-square landing probability, and differ by at most 2% in relative
root-mean-square strike pace. These are numerical repeatability requirements,
not proof of a betting advantage.

## What changed with sample count

The following counts ask whether the fighter projected to land more
significant strikes actually won. They are not the simulator's winner forecasts
and do not use betting prices. The sample has 228 decisive fights and one
other result, which is excluded from this ordering comparison.

| Resampled histories averaged | Run A: original seed | Run B: separate seed |
| --- | ---: | ---: |
| 10 | 109 / 228 | 116 / 228 |
| 25 | 117 / 228 | 117 / 228 |
| 50 | 118 / 228 | 120 / 228 |
| 100 | 119 / 228 | 123 / 228 |

The pooled 200-member forecast produces 119/228. Averaging forecasts can change
which fighter leads; its correct-selection count is not the average of the
two runs' counts.

Run A recovers ten of the eleven selections lost in the earlier ten-member
comparison. Its improvement over that earlier result is 4.39 percentage
points, with a whole-card 95% interval of +0.85 to +8.19 points. Run B improves
by 6.14 points, interval +2.64 to +9.83.

Against the original 120/228 reference, neither 100-member run has a resolved
ordering advantage or disadvantage. A's difference is -0.44 points, interval
-3.04 to +2.39; B's is +1.32 points, interval -1.29 to +3.92. The one-fight
shortfall is not compelling evidence that the candidate is worse. It is still
a failure of the rule fixed before the experiment.

This revises the interpretation of the earlier handoff study: much of its
apparent ordering loss was sensitive to the small ten-member average. It does
not establish that resampling inherently destroys useful fighter information.

## Probability error and remaining instability

Lower is better for both error columns. Pace is evaluated at actual duration;
landing accuracy at actual attempts. Those observed quantities enter scoring
only, never forecast construction.

| Predictor | Pace error | Landing-accuracy error |
| --- | ---: | ---: |
| Original observation reference | 16.0173 | 4.4937 |
| Original marginal fighter baseline | 16.1790 | 4.6199 |
| Rejected v2 conditional predictor | 17.7106 | 4.5549 |
| Run A, 100 members | 16.1191 | 4.5092 |
| Run B, 100 members | 16.1189 | 4.5174 |
| Pooled 200, descriptive only | 16.1168 | 4.5120 |

Both final runs pass the declared probability-error comparisons with the
marginal and v2 baselines. They still have slightly worse pace error than the
original observation reference; both paired whole-card intervals for that
difference are above zero. Increasing numerical precision has not improved
on the original reference's conditional count predictions.

The two final runs differ by 0.51 percentage points in landing probability
and 1.39% in relative pace, meeting both numerical-value requirements. Their
projected leaders agree in 214 of 228 decisive fights, failing the separate
95% ordering requirement. Small differences can change the projected leader
when two fighters' estimated landed-strike rates are close.

Using the member-paired red-minus-blue strike rates, 40 of 229 fights in A and
36 in B have a projected difference within approximately two standard errors
of zero. This measures error from averaging a finite number of training
resamples, retaining the dependence between both fighters in the same member.
It is not total model uncertainty, future fight variability, or a confidence
interval for betting profit.

## Decision and next priority

Keep the research correction and its evidence, but do not run the next
simulation screen. Do not extend this study with more members or new seeds to
chase the missing selection or the agreement threshold. The fixed test was
completed, and the candidate did not satisfy all its requirements.

Defer further expansion of this simulator route. Make prospective obtainable
prices and selected-bet probability quality the next profit-research priority:
check actual early/decision/late captures, preserve missing observations, and
evaluate entry value against an independent late-market reference. The current
simulation findings do not establish an advantage over bookmaker prices.

## Reproduction and evidence

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m fight_sim.strike_resample_precision
.venv/Scripts/python.exe -m unittest discover -s tests -p test_fight_sim_strike_resample_precision.py
```

The [design](audit/simulation/2026-10-01/strike_resample_precision_design.json)
was written before running the study. The
[results](audit/simulation/2026-10-01/strike_resample_precision_results.json)
contain every fixed prefix, paired comparison, decision check, causal cutoff,
source commitment and output hash. Estimator code and handoff evidence are
pinned to `b19e6280`; original historical data to `59a16cd2`. Every cutoff's
eight-prior-card selections were recomputed and verified. Four ordinary Python
workers evaluated independent cards; forecasts are sorted deterministically
before storage and scoring.

The original reference and all ten original-seed members reproduce within the
declared 1e-9 tolerance. Local detailed outputs contain 91,600 member/side
predictions plus averaged predictions and per-fight precision diagnostics under
`artifacts/simulations/strike-resample-precision-20261001/`. Seed identifiers are
stored as exact decimal text to avoid rounding 64-bit integers through floating
point; all 60 card/stream seed-sequence hashes are verified.

Six new tests passed: seed-prefix preservation, whole-card sampling and causal
cutoffs, exact seed storage, paired-fighter uncertainty, mandatory passage of
both independent runs, and rejection of instability or meaningless numerical
improvements. Seven existing handoff tests also passed (13 total).

These are already studied development outcomes. Whole-card intervals use
4,000 resamples after averaging members within each fighter side. Scores use
mean predictions, and strike ordering uses the mean of member-specific
pace-times-accuracy products. Two fixed runs quantify repeatability here; they
do not prove convergence at arbitrarily large sample counts.
