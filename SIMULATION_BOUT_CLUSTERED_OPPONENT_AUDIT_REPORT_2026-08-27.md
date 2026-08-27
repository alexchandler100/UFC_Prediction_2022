# Bout-clustered opponent-adjustment audit

Date: 2026-08-27

Status: passed the pre-simulation observation gate. This authorizes one
bounded simulator development screen only. It does not validate a snapshot,
open either 2025 cohort, change the default simulator, update the website, or
authorize production/betting use.

## Why this audit came first

`opponent_adjusted_v1` made winner prediction decisively worse. Its conditional
shrinkage treated action counts as fighter-effect precision and assigned about
0.80--0.92 mean reliability to strike effects despite far fewer independent
bouts. The failed mode remains an explicit, non-default diagnostic for
reproducibility; normal simulations still use `full`.

The replacement experiment tests opponent adjustment on what UFCStats directly
observes before spending more Monte Carlo time. Each fighter-side bout is one
equal-weight unit when fitting effects. A 250-attempt fight therefore provides
one observation of fighter quality, not 250 independent observations.

## Frozen design

Five targets were evaluated:

1. significant-strike attempt pace, using conditional Poisson likelihood;
2. significant-strike accuracy, using conditional binomial likelihood;
3. takedown attempt pace, using conditional Poisson likelihood;
4. takedown accuracy, using conditional binomial likelihood; and
5. submission-attempt pace, using conditional Poisson likelihood.

Every outer 2024 card was predicted using only earlier fights. Context-only,
marginal-fighter, and two-way opponent-adjusted models shared the same causal
division/era parent. Ridge strengths `5, 10, 20, 40` were selected separately
for each target and model using next-card negative log likelihood on the eight
strictly preceding eligible cards. Exact selection ties preferred stronger
regularization. The outer comparison used the frozen 229-fight / 30-card
`development_2024` cohort and event-card block bootstrap intervals with 2,000
replicates.

For pace targets, the two-way opponent history debiases actor quality, while
the prediction uses the actor effect because the current simulator has no
opponent pace-vulnerability parameter. Accuracy targets use both adjusted
actor offense and opponent vulnerability. This matches the intended simulator
consumers rather than testing a richer model the engine cannot represent.

The audit completed in 103.6 seconds. It executed no simulated fight paths and
did not read either locked 2025 cohort.

## Results

Lower negative log likelihood is better.

| Target | Context NLL | Marginal NLL | Adjusted NLL | Adjusted / marginal | Paired adjusted-minus-marginal 95% interval |
|---|---:|---:|---:|---:|---:|
| Strike pace | 17.6002 | 16.1790 | **16.0173** | 0.9900 | `[-0.3320, -0.0007]` |
| Strike accuracy | 5.0573 | 4.6199 | **4.4937** | 0.9727 | `[-0.2126, -0.0326]` |
| Takedown pace | 2.8497 | 2.4359 | **2.4260** | 0.9959 | `[-0.0294, +0.0082]` |
| Takedown accuracy | 1.5321 | 1.5065 | **1.5003** | 0.9959 | `[-0.0270, +0.0134]` |
| Submission pace | 0.7963 | **0.7647** | 0.7691 | 1.0058 | `[+0.0018, +0.0072]` |

The marginal model beat context overall by 7.38% equal-target mean relative
loss, with event-card interval `[-8.88%, -5.89%]`. Opponent adjustment then
improved over the tuned marginal model by another 0.795%, with interval
`[-1.224%, -0.361%]` and 100% of bootstrap replicates favorable. Four of five
targets improved at the point estimate. Strike pace and accuracy had wholly
favorable target-specific intervals; both takedown intervals crossed zero.

Submission pace was a small but statistically resolved regression of 0.576%.
It remained inside the predeclared one-percent no-material-harm limit, so the
overall candidate passed. This does not license silently deleting the
submission result or tuning a special exception on these outer observations.
The simulator screen must preserve this limitation in its interpretation.

The selected ridge strengths were not fixed using outer-card observations.
Strike accuracy used adjusted ridge 5 on 27/30 cards and 10 on three. Strike
pace varied between 5, 10, and 20. Takedown accuracy often selected stronger
regularization, up to 40. Takedown pace selected 5 on all cards. This is
materially more conservative and causally selected than v1's action-count
reliability formula.

## Decision and next boundary

The bout-clustered model is eligible for one bounded simulator screen on the
same open 2024 development cohort. That implementation must:

- be versioned separately from rejected `opponent_adjusted_v1`;
- reconstruct each card's strictly prior eight-card ridge selection;
- retain equal-bout effect precision inside each card-bootstrap member;
- use common simulation seeds and the frozen v2.1 trajectory mechanics;
- compare against both `full` and `reliability_weighted` at 100 paths per
  fight, with a 3,300-second hard cap; and
- advance only if winner log loss and Brier improve without material joint,
  method, duration, or action-distribution harm.

Do not open `confirmation_2025_a` or `final_holdout_2025_b` merely because the
observation audit passed. If the simulator screen fails, retain the current
default and conclude that better conditional observation likelihood did not
translate into better fight-level outcome ranking.

## Reproduction and authority

```bash
export PYTHONPATH="$PWD/src"

python -m fight_sim opponent-adjustment-audit \
  --cohort-manifest SIMULATION_EXPERIMENT_COHORTS_V1.json \
  --cohort-name development_2024 \
  --min-prior-ufc-fights 3 \
  --inner-validation-events 8 \
  --minimum-training-fights 500 \
  --ridge-grid 5,10,20,40 \
  --bootstrap-replicates 2000 \
  --random-seed 52237 \
  --max-runtime-seconds 3300 \
  --output artifacts/simulations/opponent-adjustment-bout-clustered-audit-20260827.json \
  --predictions-output artifacts/simulations/opponent-adjustment-bout-clustered-predictions-20260827.csv
```

The report hash is
`e55f01f8d94ecf13d3ac59e5cc1a06abc69a542c28d11ed9ca56f7d8d1a8df29`.
The detailed prediction ledger remains ignored beneath `artifacts/`.
