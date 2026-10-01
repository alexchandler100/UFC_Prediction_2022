# Strike predictor handoff: correction tested, simulation screen withheld

Date: 2026-10-01. Research only. No production forecasts or betting rules changed.

Follow-up: the [fixed higher-precision check](SIMULATION_STRIKE_RESAMPLE_PRECISION_REPORT_2026-10-01.md)
recovered much of the ten-member ordering loss (119/228 and 123/228 at 100
members), but failed its separate stability/advance requirements. Read that
result before interpreting this earlier ten-member comparison as persistent
damage from resampling.

The proposed correction preserves the strike predictor through the engine, but
it fails the requirement to preserve useful fighter ordering. It therefore
does not advance to another fight simulation screen.

All 229 development fights across 30 cards were evaluated in 351 seconds,
using the ten saved training-card resamples for each fight. No new fight
trajectories were generated, and neither 2025 confirmation group was evaluated.

## What caused the earlier loss

The original predictor projects which fighter will land more significant
strikes. That projected strike leader was the eventual winner in 120 of 228
decisive fights. This is a diagnostic of fighter ordering, not a winner
probability forecast or a test of betting profit.

We fixed five comparisons before running the study:

| Predictor | Projected strike leader won | Strike-pace error | Landing-accuracy error |
| --- | ---: | ---: | ---: |
| Original observation predictor | 120 / 228 | 16.0173 | 4.4937 |
| Same method fitted to ten resampled sets of training cards; sole proposed correction | 109 / 228 | 16.1745 | 4.5549 |
| Resampled fits using the simulator's division baseline | 108 / 228 | 16.5846 | 4.5549 |
| Simulator baseline plus age, experience and layoff adjustments; reproduces rejected v2 | 106 / 228 | 17.7106 | 4.5549 |
| Original baseline plus those adjustments; diagnostic only | 105 / 228 | 17.0073 | 4.5549 |

Both error columns measure how poorly the predictions explain the observed
strike counts; lower is better. Pace is scored at the actual fight duration,
and landing accuracy at the actual number of attempts. Those outcomes are used
only for scoring and never to construct the forecasts.

**Most of the fighter-ordering drop appears at the resampling step.** Fitting
the original method on ten different, resampled sets of earlier cards and
averaging its predictions loses 11 of the original 120 correct strike leaders.
The difference is 4.82 percentage points worse, with a whole-card 95% interval
of 0.94 to 8.66 points worse. Landing-accuracy error also worsens, with an
interval excluding zero. The smaller pace-error deterioration is uncertain.

Changing to the simulator baseline then loses one more correct selection and
worsens pace error. Adding the existing age/experience/layoff adjustments loses
another two and worsens pace error further. The additional two-selection loss
is uncertain; it should not be interpreted as proof that these attributes are
intrinsically useless. Their current implementation does not help here. The
extra comparison using those adjustments with the original baseline also
worsens pace error and produces only 105 correct strike leaders.

These are fixed implementation comparisons on an already studied sample. They
do not establish that resampling generally harms forecasts. Ten retained
versions may be too few, and averaging nonlinear predictions need not reproduce
the prediction fitted to the full training history. This study does not
separate those two explanations.

## The mapping correction works as specified

The new research module fits the original observation context with the matching
card multiplicities, preserving each physical fighter-side bout as the unit
of evidence. It uses the same target-specific shrinkage choices selected from
eight strictly earlier cards. It excludes the extra covariate adjustment from
the sole proposed candidate.

For each matchup and parameter member, the mapping:

1. Combines the fighter's offensive accuracy and the opponent's vulnerability
   in the original log-odds units, so opponent effects are not weakened by the
   old defense conversion.
2. Preserves the full snapshot's relative phase strike rates, then normalizes
   exact engine intensity to the predicted pace under the declared context
   phase-time proxy. Ground time is split equally between top and bottom.
3. Solves for the engine accuracy input that reproduces the combined landing
   probability under the same phase/attempt weights.

The inverse calculations use only pre-fight predictions and fixed engine
formulas. They never optimize against a fight result. Existing engine caps,
phase modifiers, fatigue/damage mechanics, all non-strike parameters and random
seeds remain intact. Unreachable targets raise an error instead of being
silently clipped to a different forecast.

Across all 4,580 fighter/member combinations, the largest pace mismatch is
7.11e-15 attempts per minute and the largest accuracy mismatch is 2.22e-16.
The mapped candidate has the same scores and 109 correct strike leaders as
its input predictor. The engine handoff now preserves that particular
predictor, but that predictor still fails the required fighter-ordering check.
The preservation contract applies at neutral stamina/damage under the stated
occupancy proxy; it does not guarantee calibrated generated fight statistics.

## Decision

Keep the candidate research-only and withhold a new simulation screen. Its
pace error improves substantially against rejected v2, but landing-accuracy
error is effectively unchanged: the recorded numerical improvement is only
4.5e-9. Against the original marginal-fighter comparator, its pace and accuracy
point scores improve slightly, but both difference intervals include zero.
Its 109 correct strike leaders fail the predeclared minimum of 120.

No diagnostic arm was substituted after seeing the results, and no scalar,
threshold, or alternative mapping was tuned. The existing simulator remains
unchanged. This is a useful repair to the research implementation, not a
validated prediction improvement or a betting advantage.

If simulator work continues, the next inexpensive question is whether the
resampling loss persists at higher numerical precision. One fixed comparison
using more training-card resamples of this same estimator can distinguish a
noisy ten-member average from a persistent change in fighter ordering, without
simulating fights. Specify its sample count, seeds and decision rule before
running it; do not repeatedly increase the count until a threshold passes.
Prospective obtainable-price evidence remains the higher priority for the
profit project.

## Reproduction and verification

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m fight_sim.strike_handoff_audit
.venv/Scripts/python.exe -m unittest discover -s tests -p test_fight_sim_strike_handoff.py
```

The [frozen design](audit/simulation/2026-10-01/strike_handoff_design.json)
identifies the candidate and pass/fail rules before this run. The
[results](audit/simulation/2026-10-01/strike_handoff_results.json) contain all
scores, paired whole-card intervals, causal cutoffs, original source hashes,
selected shrinkage strengths, mapping/reproduction errors and hashes of the
local output files. The historical bridge evidence is pinned to commit
`2bc8e378`; original CSV content is recovered from `59a16cd2` and checked
against the saved run hashes.

The original observation predictions reproduce within 1.78e-15; the old v2
conditional predictions within 7.11e-15. All 27,480 arm/member/side rows and
4,580 mapping checks are retained under
`artifacts/simulations/strike-handoff-20261001/`. Only the 229 development
fights are evaluated. Whole-card intervals use 4,000 resamples after averaging
members within each fighter side; winner ordering averages member-specific
landed-rate products. These are scores of mean predictions, not complete
posterior mixture scores.

Seven new tests passed, covering reproduction of the original fit, card
multiplicity, strict training cutoffs, forecast independence from test-fight
outcomes/duration, exact two-sided mapping, preserved non-strike parameters,
fighter-swap symmetry and unreachable engine targets. The three existing
observation-audit tests and five bridge-audit tests also passed (15 total).
