# Outcome-engine v2 development report

Date: 2026-08-27

Status: `two_route_v2` with the v2.1 KD exposure correction passed every
predeclared development gate and is frozen for confirmation. It remains
candidate-only, paper-only, production-disabled, and execution-disabled.

## Why the legacy engine failed

The broad 229-fight audit exposed a structural coupling that scalar tuning
could not repair:

- each strike exchange contained three strikes and sampled one knockdown per
  landed strike, so a single exchange could create multiple official KDs;
- a marginal KD-per-landed rate was multiplied again by severity, opponent KO
  resistance, accumulated hurt, and damage;
- the same strike's hurt and damage increments were then reused in its
  immediate post-KD finish probability;
- KO/TKO was possible only after that latent knockdown, despite 29% of the
  pre-development KO/TKO wins having no official winner knockdown; and
- lowering the KD rate therefore also removed the engine's only KO route.

This explained why earlier KD/finish multiplier grids improved the KD
distribution while degrading joint side-by-method predictions.

## Implemented candidate

Legacy mechanics remain the default and retain their old serialized RNG
contract. `SimulatorConfig.outcome_mechanics_version="two_route_v2"` opts into:

1. an exchange-level official-KD hurdle,
   `P(KD)=1-exp(-rate_per_landed * landed * multiplier)`, capped at one KD per
   exchange;
2. no reuse of severity, hurt, damage, or KO resistance around a marginal KD
   rate that already contains those effects in observed data;
3. a separately calibrated KO-after-official-KD route;
4. a separately calibrated KO-without-official-KD rate per landed significant
   strike; and
5. path-level counters for landed exchanges, hurt events, KD recoveries,
   KD-route KOs, no-KD-route KOs, ground stoppages, and submission finishes.

The initial route probabilities used only physical fights from 2014-01-01
strictly before 2024-04-13: 5,062 fights, 2,184 official KDs, 1,133 KO/TKO wins
with a winner KD, 463 without one, and 416,387 landed significant strikes.
They yielded `post_knockdown_finish_probability=0.5187728938` and
`non_knockdown_ko_rate_per_landed=0.0011119463`.

## Frozen cohorts and compute

`SIMULATION_EXPERIMENT_COHORTS_V1.json` freezes exact card identities, input
hashes, the three-prior-UFC-fight exposure rule, eligible-fight counts, and a
checksum of every sorted eligible UFCStats fight ID:

- open development: 229 fights / 30 cards, 2024-04-13 through 2024-12-14;
- locked confirmation: 57 fights / 10 cards, 2025-01-11 through 2025-03-22;
- untouched final holdout: 72 fights / 10 cards, 2025-03-29 through 2025-06-14.

Both new development arms used 10 bootstrap members, exactly 100 paths per
fight, one seed, the conditional TD/control fit, and the same 229 fights as the
existing baseline. v2 completed in 1,170.7 seconds and v2.1 in 1,145.5 seconds;
combined new simulation compute was 38.6 minutes. Neither command approached
the 3,300-second hard cap.

## Results

| Metric | Legacy baseline | two-route v2 | two-route v2.1 |
|---|---:|---:|---:|
| Joint side/method log loss | 2.3845 | 2.1291 | **1.9869** |
| Method log loss | 1.2031 | 1.1819 | **1.1380** |
| Winner log loss | 0.7287 | **0.7143** | 0.7213 |
| Winner Brier | 0.2638 | **0.2579** | 0.2613 |
| Winner accuracy | 52.63% | 53.07% | 52.63% |
| Duration CRPS, seconds | 189.86 | 176.13 | **166.04** |
| Predicted KDs/fight | 0.847 | 0.558 | **0.410** |
| Observed KDs/fight | 0.358 | 0.358 | 0.358 |
| Predicted KO/TKO count | 98.65 | 88.78 | **72.65** |
| Observed KO/TKO count | 65 | 65 | 65 |
| Predicted decision count | 91.85 | 99.73 | **113.36** |
| Observed decision count | 125 | 125 | 125 |
| Duration mean bias, seconds | -128.50 | -99.24 | **-55.74** |

The first v2 arm proved the route split was sound: it predicted 21.69 no-KD
stoppages versus 21 observed, but still predicted 0.558 KDs/fight. v2.1 changed
only the KD hurdle multiplier to the development observed/predicted ratio,
`0.3580786 / 0.5580786 = 0.6416275430`.

Against legacy, v2.1 reduced absolute KD bias by 89.4%, KO-count bias by 77.3%,
decision-count bias by 64.9%, and duration bias by 56.6%. Joint log loss
improved by 0.3976, method log loss by 0.0650, duration CRPS by 23.81 seconds,
winner log loss by 0.0073, and Brier by 0.0026. The 2,000-replicate event-card
interval for candidate-minus-baseline joint log loss was [-0.8061, -0.0967].
All five protected action-distribution CRPS ratios were at or below 1.002.
Every development gate passed.

## Interpretation and next gate

This is strong evidence that the two-route outcome structure is more realistic
than the legacy coupled knockdown/finish mechanism. It is not evidence that the
simulator is yet a useful standalone winner predictor. v2.1 selected the same
52.63% of winners as legacy, its 0.7213 winner log loss remains worse than the
0.6931 constant-50/50 reference, and the KD scale was calibrated on this open
development cohort. The low 100-path screen also cannot precisely resolve rare
joint outcomes.

No more outcome-engine tuning may use these 30 cards. The frozen v2.1 profile
must next be compared with legacy on `confirmation_2025_a`, at materially higher
path count and at least two independent seeds. Only if it confirms should the
final holdout be opened. A favorable confirmation advances a research
candidate; it still cannot change the website, production model, stack, or any
betting decision.

After the outcome engine is confirmed or rejected, the next independent
structural work is phase/exposure calibration: model total engagement per alive
minute plus phase-relative action rates, and separate latent ground-top,
bottom, and controlled-clinch time from the UFCStats control observation. That
work must receive its own development/confirmation boundary.

## Reproduction

```bash
export PYTHONPATH="$PWD/src"

python -m fight_sim posterior-backtest \
  --cohort-manifest SIMULATION_EXPERIMENT_COHORTS_V1.json \
  --cohort-name development_2024 \
  --min-prior-ufc-fights 3 \
  --bootstrap-members 10 \
  --paths-per-matchup 100 \
  --seed-repeats 1 \
  --takedown-control-association \
  --simulator-config SIMULATION_MECHANICS_TWO_ROUTE_V2_1.json \
  --max-runtime-seconds 1800 \
  --workers 4 \
  --chunk-size 64 \
  --output-dir artifacts/simulations/two-route-v2-1-development-100paths-20260827

python -m fight_sim compare-outcome-mechanics \
  artifacts/simulations/broad-100paths-td-control-20260827 \
  artifacts/simulations/two-route-v2-1-development-100paths-20260827 \
  --minimum-balanced-events 30 \
  --output artifacts/simulations/two-route-v2-1-development-comparison-20260827.json
```
