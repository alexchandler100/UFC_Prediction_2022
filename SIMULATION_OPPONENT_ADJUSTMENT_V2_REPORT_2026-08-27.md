# Bout-clustered opponent-adjusted v2 simulator screen

Date: 2026-08-27

Status: rejected on the open development cohort. The mode remains an explicit
research diagnostic for reproduction only. It is not the default, does not
change the website or upcoming forecasts, and does not open either locked 2025
cohort.

## Question and frozen design

The preceding observation audit found that equal-bout, cross-fitted opponent
adjustment improved next-card significant-strike pace and accuracy likelihood.
This screen tested whether those improvements transfer through the simulator
to better fight-level predictions.

`opponent_adjusted_v2` is separate from rejected v1. For each outer-card
cutoff it selects target-specific ridge strengths on the eight strictly prior
eligible cards. Each parameter bootstrap member then fits two-way actor and
opponent effects using the matching event-card resample. A physical fighter-
side bout is one precision unit; action counts do not masquerade as independent
observations. Both fighters use the same bootstrap member.

Only strike pace and strike accuracy were replaced because they had wholly
favorable target-specific audit intervals. Takedown effects were inconclusive,
submission adjustment was harmful, and those parameters therefore remained on
the original full fighter snapshot. The frozen v2.1 mechanics and original
root seeds were reused.

The screen used all 229 eligible physical fights across the 30-card
`development_2024` cohort, 10 bootstrap members, and 100 total paths per fight.
It completed 22,900 paths in 1,206.7 seconds. This is an aggregate development
screen, not matchup-level probability precision.

## Outcome results

Lower loss and CRPS are better.

| Metric | Full snapshot | Reliability weighted | Opponent v2 strikes |
|---|---:|---:|---:|
| Winner accuracy | 52.63% | **55.70%** | 48.68% |
| Winner log loss | 0.72135 | **0.71470** | 0.75362 |
| Winner Brier | 0.26127 | **0.25863** | 0.27666 |
| Calibration slope | 0.224 | **0.260** | -0.084 |
| Joint side/method log loss | 1.98694 | **1.96508** | 2.13880 |
| Method log loss | **1.13801** | 1.14370 | 1.14902 |
| Method top-class accuracy | 49.34% | 49.34% | 48.91% |
| Duration CRPS, seconds | 166.04 | **164.26** | 166.77 |

Against `full`, v2 increased winner log loss by 0.03227 and Brier by about
0.015. The paired event-card Brier interval was wholly harmful at
`[+0.00112, +0.02857]`; the winner log-loss interval was
`[-0.00042, +0.06236]`. Joint log loss worsened by 0.15187 and method log loss
worsened by 0.01101.

Against the stronger reliability arm, rejection was decisive. The paired
event-card intervals were wholly harmful for winner log loss
`[+0.00347, +0.06746]`, Brier `[+0.00196, +0.03091]`, and joint side/method log
loss `[+0.01476, +0.42844]`.

Projected KO/TKO count bias moved from +7.65 to +6.09 fights and decision bias
moved from -11.64 to -11.12, but these small aggregate count changes did not
outweigh worse method probability loss or fight-level ranking.

## Posterior-predictive results

| Statistic | Full CRPS | v2 CRPS | Full mean bias | v2 mean bias |
|---|---:|---:|---:|---:|
| Significant-strike attempts | **77.35** | 78.76 | -37.69 | -44.36 |
| Significant strikes landed | 34.94 | **34.68** | -5.21 | -8.72 |
| Knockdowns | 0.2753 | **0.2709** | +0.052 | +0.043 |
| Takedowns | 1.2824 | **1.2739** | -0.832 | -0.822 |
| Submission attempts | **0.4204** | 0.4245 | -0.125 | -0.112 |
| Control seconds | **149.86** | 150.08 | -130.22 | -129.10 |

The intended strike transfer did not occur. Attempt CRPS worsened 1.8%, its
90% coverage remained only 69.9%, and mean underprediction grew by 6.67
attempts. Landed-strike CRPS improved only 0.7%, while its mean bias worsened by
3.52 landed strikes. Small knockdown/takedown changes are downstream trajectory
variation and do not rescue the failed primary outcomes.

## Decision and interpretation

Reject v2. Keep `full` as the operational default and preserve both opponent
modes only as explicitly named diagnostics. Do not attenuate or otherwise tune
v2 on these same outer outcomes, and do not spend either locked 2025 cohort on
it.

The experiment establishes an important negative result: improving a count
model conditional on the observed fight duration does not guarantee better
counts when duration, phase occupancy, damage, and termination are generated
endogenously. It also does not guarantee better winner ranking. The next step
is therefore not another Monte Carlo parameter sweep.

Before any v3 screen, run a fast causal bridge audit that compares the raw
observation prediction, the exact effective matchup intensity after the engine
transforms, and the existing endogenous trajectory distribution. It must
separate actor offense, opponent vulnerability, phase mapping, and generated
exposure, and test whether each implied strike differential ranks actual
winners. Another simulator screen is justified only if that audit identifies a
specific prospective mapping error or correction.

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
  --snapshot-parameter-mode opponent_adjusted_v2 \
  --max-runtime-seconds 3300 \
  --workers 4 \
  --chunk-size 64 \
  --output-dir artifacts/simulations/two-route-v2-1-opponent-adjusted-v2-strikes-development-100paths-20260827
```

The population report hash is
`5e901106f7d1e97e8d820554ea0b4528b1460ebed78eeeb8409cc5c6a670e3fc`.
The full and reliability comparison hashes are
`11e964afeb1ce2300944e4623b0f4ff7d6aef9727505a1f2ee1a1fe9ae7116a0`
and
`d072350cf6bdfbddacad46c30f261b38c0d631933866a0341c4af84a7a33dc8f`.
Detailed ledgers and reports remain ignored beneath `artifacts/`.
