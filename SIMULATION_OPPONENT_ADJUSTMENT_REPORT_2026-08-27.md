# Opponent-adjusted fighter-effect experiment

Date: 2026-08-27

Status: rejected on the open development cohort. The implementation remains
available as an explicitly named research diagnostic, but it must not be used
for upcoming forecasts, the website, stacking, production predictions, or
wagering. The locked 2025 confirmation cohorts remain unopened.

## Question and causal design

The previous snapshot ablation found weak directional signal in fighter
history, but neither the original fighter parameters nor an additional fixed
exposure-weighted shrinkage step beat constant 50/50 on winner log loss or
Brier score. The next predeclared question was whether strength-of-schedule
confounding was hiding useful signal.

`opponent_adjusted_v1` reconstructs each fitted bootstrap member's exact
event-card sample and fits two-way actor/opponent residuals around that
member's division/era context. It estimates supported effects for:

- significant-strike attempt pace;
- significant-strike accuracy and opponent vulnerability;
- takedown attempt pace;
- takedown accuracy and opponent vulnerability; and
- submission-attempt pace.

Rate residuals use stabilized log rates; accuracy residuals use stabilized
logits. Twelve alternating actor/opponent updates estimate between-fighter
variance from the training rows and shrink group means according to estimated
reliability. Effects are bounded to a factor of three. Historical strike-pace
residuals remove the fitted age, experience, and layoff adjustment so the
snapshot can apply the current causal covariates once. Phase mix and all rare
finish conversions or unsupported latent mechanics remain as previously fit.

Every event uses rows strictly before its cutoff. The held-out card never
affects an effect, variance, or reliability estimate. Both fighters use the
same bootstrap member, and corresponding candidate/baseline paths use common
random draws. Appending future fights is covered by an exact snapshot
invariance test.

The screen used the already-open `development_2024` cohort: 229 fights across
30 complete cards, at least three earlier UFCStats fights per side, ten
bootstrap members, 100 total paths per fight, one seed, the frozen v2.1 outcome
mechanics, and a 1,800-second hard cap. It completed all 22,900 paths in 1,106.1
seconds (18.4 minutes), with 30/30 causal fit-cache hits and no incomplete
fight.

## Results

One draw/no-contest was omitted only from the 228-fight binary winner metrics.

| Metric | Full fighter | Reliability weighted | Opponent adjusted v1 |
|---|---:|---:|---:|
| Winner accuracy | 52.63% | **55.70%** | 46.49% |
| Winner log loss | 0.72135 | **0.71470** | 0.82197 |
| Winner Brier | 0.26127 | **0.25863** | 0.30308 |
| Calibration intercept | -0.110 | -0.103 | -0.108 |
| Calibration slope | 0.224 | **0.260** | -0.192 |
| Joint side/method log loss | 1.98694 | **1.96508** | 2.31233 |
| Method log loss | **1.13801** | 1.14370 | 1.14341 |
| Duration CRPS, seconds | 166.04 | 164.26 | **163.70** |

The opponent-adjusted arm classified 106/228 winners correctly. Its Wilson
95% interval was 40.13%--52.97%, and its two-sided exact p-value against 50%
was 0.321. The negative calibration slope is especially damaging evidence:
the candidate's direction was inverted on this cohort, not merely too
confident.

Against `full`, candidate-minus-baseline winner log loss was +0.10062, with a
paired event-card 95% interval of `[+0.05421, +0.13443]`; Brier was +0.04060,
interval `[+0.02267, +0.05693]`; and joint log loss was +0.32540, interval
`[+0.07855, +0.63914]`. Against `reliability_weighted`, the corresponding
differences were +0.10727 (`[+0.05820, +0.14259]`), +0.04215
(`[+0.02429, +0.05967]`), and +0.34725 (`[+0.08827, +0.68209]`). All are
wholly harmful rather than unresolved Monte Carlo fluctuations.

The trajectory distributions did not fail broadly. Relative to reliability
weighting, duration CRPS improved by 0.57 seconds, method log loss improved by
0.00029, and protected action CRPS ratios ranged from 0.968 to 1.048. The
decisive regression is the fighter-side ranking injected by the new effects.

## Diagnosis

The estimator's apparent training reliability was far too high for the amount
of independent fighter evidence. On the final development cutoff, averaged
over ten bootstrap members, mean fitted reliability was:

| Effect family | Actor | Opponent |
|---|---:|---:|
| Strike pace | 0.908 | 0.920 |
| Strike accuracy | 0.800 | 0.811 |
| Takedown pace | 0.655 | 0.541 |
| Takedown accuracy | 0.482 | 0.584 |
| Submission pace | 0.222 | 0.256 |

The first implementation used action-count information as precision. Thousands
of attempts within repeated bouts therefore acted too much like independent
fighter evidence even though the effective independent units are bouts and
event cards. The event-card bootstrap preserves outer covariance, but it does
not repair an overconfident conditional shrinkage formula inside each member.
The model consequently replaced conservative marginal estimates with unstable
strength-of-schedule ranks.

This result rejects `opponent_adjusted_v1`. It does not establish that opponent
adjustment is intrinsically useless. A credible second attempt must estimate
uncertainty at the bout/card level and demonstrate out-of-training prediction
before its effects enter the simulator.

## Frozen next experiment

Do not weaken this failed model by tuning one multiplier on the same simulated
trajectories. The lowest-risk next screen is a **cross-fitted, bout-clustered
opponent-adjustment audit before simulation**:

1. fit the same supported observation targets on earlier cards only;
2. compare context-only, marginal fighter, and opponent-adjusted predictions
   on the next held-out card using count/logistic likelihoods;
3. calculate uncertainty and effective sample size by physical bout/card, not
   action count;
4. test a small predeclared regularization grid by nested chronological card
   splits, selecting on held-out observation likelihood rather than fight
   winner outcomes; and
5. run the expensive 229-fight simulator screen only if the opponent-adjusted
   observation model first beats the marginal model out of training.

Cap the audit at 3,300 seconds and keep `confirmation_2025_a` and
`final_holdout_2025_b` locked. If the audit cannot improve its directly
observed targets, retire opponent adjustment for this simulator version and
focus winner prediction work on combining the existing logistic model with
the simulator's more useful method/duration distributions.

## Reproduction and local authority

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
  --snapshot-parameter-mode opponent_adjusted_v1 \
  --max-runtime-seconds 1800 \
  --workers 4 \
  --chunk-size 64 \
  --output-dir artifacts/simulations/two-route-v2-1-opponent-adjusted-development-100paths-20260827
```

The population report hash is
`c26b1dffaea92164146002c3ca4cbf9d5c0e0d099457c36d9fcd344397bc759d`.
The comparisons against full and reliability-weighted have hashes
`9f44a77c0e90826fe00c6f6fd82b2b3a686a614e7815d79c1cff505296af641e`
and
`4b7a7ef619d92919327314d56581343d3bf64c517012105c24a8c9ae851d4bb0`.
Detailed ledgers and HTML remain ignored beneath `artifacts/`.
