# Why better strike estimates failed to improve the simulator

Date: 2026-10-01. Research only; no forecast, betting policy, or engine default changed.

The rejected v2 implementation loses useful fighter ordering before it reaches
the fight engine. Its rebuilding of the observation model is therefore the
first repair target. Two additional mapping errors deserve correction, but
neither establishes that the simulator can beat bookmaker prices.

The audit completed all 229 development fights on 30 cards, reconstructing ten
saved parameter members per fight, in 279 seconds of calculation. It ran no
new fight trajectories. Both 2025 confirmation groups remain unopened.

## What happened

The following table asks a deliberately narrow question: did the fighter
projected to land more significant strikes actually win? There were 228
decisive fights; the remaining fight is excluded from this comparison. These
are strike-leader comparisons, not the simulator's winner probabilities.

| Calculation | Strike leader also won |
| --- | ---: |
| Original observation audit, fighter history without opponent adjustment | 117 / 228 |
| Original observation audit, opponent adjusted | 120 / 228 |
| Rebuilt v2 observation estimate, before engine transforms | 106 / 228 |
| v2 snapshot phase-rate mapping, before mechanics | 98 / 228 |
| Neutral engine with its context phase-time proxy | 96 / 228 |
| Saved complete v2 simulations | 100 / 228 |
| Saved complete full-snapshot simulations | 119 / 228 |

Rebuilding the observation predictor loses 14 correct strike-leader selections,
or 6.14 percentage points. Resampling entire cards gives a 95% interval of
1.41 to 10.96 percentage points worse. Its strike-pace probability error also
worsens from 16.017 to 17.711, and accuracy error from 4.494 to 4.555. Lower is
better for both errors. Both deterioration intervals exclude zero.

This is not a faithful transfer of the earlier successful estimator. The
observation audit uses its own directly fitted division/era context. The v2
snapshot reconstruction instead uses the simulator's bootstrapped context,
phase-rate mixture, and age/experience/layoff rate adjustments. It reuses the
selected shrinkage strengths but refits the fighter effects against those
different baselines. This audit locates the loss at that boundary; it does not
isolate how much each changed ingredient contributes.

## Two additional translation problems

**Opponent accuracy effects are weakened.** The observation predictor adds the
actor's and opponent's effects to the log odds of landing a strike. The
snapshot converts opponent vulnerability into a defense probability, which the
engine applies through a bounded linear multiplier. At neutral distance, the
engine preserves about 97.8% of the actor effect but only 36.3% of the opponent
effect on average. Across fighter/member combinations the latter's middle 90%
range is 31.4% to 42.4%. The direction is correct; the strength is not preserved.
These are local sensitivities of log odds, not percentage changes in landing
probability. Opponent pace effects are deliberately excluded from prediction
in both implementations; that is not a newly discovered sign error.

**Strike composition is reused as time occupancy.** Fighter phase rates were
constructed using a parent context occupancy proxy. The v2 rescaling instead
normalizes those rates using the fighter's own strike-position shares. Those
shares describe where attempts occurred, not measured time standing, clinching,
or on the ground. Evaluating the resulting snapshot under the original context
proxy retains only 88.5% of the rebuilt conditional pace on average; the middle
90% range is 52.3% to 100%. Its overall attempt forecast falls by 12.74 attempts
per fighter at observed fight duration.

That reduction happens to improve pace error because the rebuilt predictor
already overestimates pace. It simultaneously loses eight more correct
strike-leader selections. A compensating error is not evidence that the
mapping is sound, and simply removing it may worsen count error unless the
earlier estimator mismatch is repaired too.

## Short fights explain only part of the strike shortage

The saved v2 simulations underestimate total attempts by 44.36 per fight.
Rescaling each saved member's counts to the observed duration leaves a deficit
of 29.36 attempts per fight. On this diagnostic calculation, about 15.00 missing
attempts, roughly one third of the shortage, are associated with the duration
rescaling; substantial underprediction remains. Average simulated duration was
56.2 seconds short on this 2024 cohort. This is a different sample and mechanics
profile from the much larger duration miss in the recent prospective forecasts.

The complete accounting below uses attempts **per fighter**, at observed
duration until the last row. Every row averages the same 458 fighter sides.

| Stage | Mean prediction minus actual attempts |
| --- | ---: |
| Rebuilt conditional observation prediction | +20.67 |
| Snapshot under context occupancy proxy | +7.93 |
| Neutral engine under context occupancy proxy | +51.28 |
| Neutral engine under saved simulated phase/top occupancy | +27.91 |
| Saved simulated counts rescaled to observed duration | -14.68 |
| Saved simulated counts at their generated duration | -22.18 |

Changing the occupancy assumption accounts for a 23.37-attempt reduction per
fighter between the two neutral-engine rows. The further gap to generated
counts includes fatigue, damage, changing state, and finite simulation noise.
These are descriptive differences, not isolated causal effects. Actual phase
time is unobserved, and the saved simulated occupancy is itself generated by
the model. Neither that row nor duration-standardized counts are available as
prospective betting predictors.

The audit reproduces the previous pooled distribution results: full versus v2
attempt distribution error is 77.354 versus 78.756, with attempt biases of
-37.687 versus -44.358 per fight. Landed-strike distribution error is 34.940
versus 34.680. Lower distribution error is better. This agrees with the earlier
screen; it does not rescue its worse winner forecasts.

## Decision and next experiment

Keep v2 rejected. Do not start another broad simulation search or open the
confirmation data. The next bounded task is to make the observation-to-snapshot
handoff preserve the tested estimator:

1. Reproduce its context baseline and fighter effects before adding simulator
   context, covariates, or bootstrap reconstruction. Isolate these ingredients
   with fixed diagnostic comparisons on the same open development cohort.
2. Specify one mapping that preserves conditional pace under the same explicit
   context occupancy proxy and combines actor/opponent accuracy effects in the
   same log-odds units. Verify signs and units algebraically and with tests;
   do not choose an attenuation factor to fit these fight outcomes.
3. Check conditional pace/accuracy probability error and strike-leader ordering
   before spending time on another simulation screen. A candidate that fails
   this inexpensive check should stop there. A survivor can receive the already
   bounded 229-fight development screen against full and reliability-weighted
   snapshots, with frozen mechanics and common seeds.

For the broader profit project, prospective obtainable-price evidence remains
the higher priority. The original observation strike leader won only 52.6% of
these fights, with uncertainty that includes chance. Correcting simulator
plumbing is useful research, not a demonstrated betting advantage.

## Reproduction and limits

In PowerShell from the repository root:

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m fight_sim.strike_bridge_audit
```

The runner requires the original local saved runs and cached parameter files;
it fails instead of silently fitting replacements. Historical CSVs are read
from revision `59a16cd2` and must reproduce the saved source hashes, allowing
only line-ending normalization. Engine, domain, parameter and opponent-audit
source files must match that revision. Each card uses its exact cached members,
original bootstrap seeds and eight strictly earlier selection cards. Training
cutoffs, artifact hashes, inputs, and full comparison intervals are recorded.

The [design](audit/simulation/2026-10-01/strike_bridge_design.json) and
[results](audit/simulation/2026-10-01/strike_bridge_results.json) are retained in
Git. Detailed 4,580 member/side rows and 458 averaged side rows remain under
`artifacts/simulations/strike-bridge-20261001/`; the member file's hash is sealed
in the results. The calculation is in
`src/fight_sim/strike_bridge_audit.py`.

Card intervals use 4,000 whole-card resamples after averaging members within
each physical fighter side/fight. Probability errors are evaluated at mean
predictions, not as full posterior mixture scores. Saved member means are
available, but the compact ledgers omit member-level count distributions;
pooled distributions remain available. Per-path rounded phase/control seconds
give approximate member duration and normalized top/bottom occupancy. There
are only ten saved paths per member, so member-level generated rates are noisy.

The original opponent-versus-marginal pace improvement was close to its
uncertainty boundary: its earlier 2,000-resample interval ended at -0.0007;
this fixed 4,000-resample interval ends at +0.0058. Do not describe that pace
advantage as robustly resolved. The accuracy improvement remains favorable.

Validation: 5 new diagnostic tests, 20 engine tests and 3 observation-audit
tests passed. The full 229-fight calculation completed and reproduced the
saved pooled attempt/landed distribution scores.
