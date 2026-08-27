# Conditional-control fit and broad low-path audit — 2026-08-27

## Scope

This report records two bounded, candidate-only experiments:

1. a paired development screen of a strongly pooled fighter-specific
   takedown-round control fit; and
2. a time-bounded chronological accuracy audit using exactly 100 Monte Carlo
   paths per fight.

Neither experiment changes the website, production forecasts, betting logic,
or the retained `mechanics-8ba01f34444f` simulator profile. The detailed
ledgers remain under ignored `artifacts/`.

## Candidate implementation

`fight_sim fit` and `fight_sim posterior-backtest` now accept the explicit
research flag `--takedown-control-association`. The candidate:

- pairs mirrored fighter-round rows without using later fights;
- treats same-round takedown and UFCStats `CTRL` values as interval-censored
  associations, not observed action order or top-position time;
- maps a fighter's own takedown-round credited-control share to coarse top
  retention;
- maps credited control conceded after an opponent takedown to coarse escape;
- uses global and division/era prior strength 25 and fighter prior strength 12;
  and
- fits global, context, and both fighter parameters inside the same bootstrap
  member so covariance is preserved.

The baseline artifact format and default fit are unchanged. The candidate has
its own parameter-model version and content-addressed cache identity. Compact
artifact replay preserves the flag and exact input commitment.

The posterior runner also accepts `--max-runtime-seconds` up to 3,300 seconds.
It checkpoints only complete fight/seed pairs and reports a partial,
candidate-only result when the deadline is reached.

## Five-card paired development screen

Both arms used the retained simulator mechanics, the same 31 eligible fights
on five recent cards, the same seeds, 16 bootstrap members, and 512 total paths
per fight. Thirty-two low-exposure fights were excluded. Baseline runtime was
14.1 minutes and candidate runtime was 16.8 minutes.

| Metric (lower is better) | Baseline | Conditional control | Difference |
| --- | ---: | ---: | ---: |
| Joint side × method log loss | 1.93147 | 1.90738 | -0.02408 |
| Winner log loss | 0.72506 | 0.72122 | -0.00384 |
| Winner Brier | 0.26549 | 0.26448 | -0.00101 |
| Method log loss | 1.25105 | 1.21776 | -0.03328 |
| Duration CRPS (seconds) | 259.55 | 257.17 | -2.38 |
| Total-control CRPS (seconds) | 148.66 | 145.13 | -3.54 |

Joint score and duration CRPS improved on four of five individual cards. The
screen therefore allowed the candidate to advance to the broader descriptive
audit. This was an opened development cohort, not promotion evidence.

## Broad 100-path chronological audit

The selector found 703 eligible fights among 100 recent cards and excluded 535
fights because at least one fighter had fewer than three strictly prior UFC
fights. The runner processed cards chronologically until its time budget:

- 229 fights on 30 complete event cards;
- April 13 through December 14, 2024;
- 10 bootstrap members × 10 inner paths = exactly 100 paths per fight;
- 22,900 total simulated trajectories;
- 56.3 minutes wall time; and
- a graceful `stopped_by_time_limit: true` report with every included fight
  fully checkpointed.

Fitting remained the largest cost (2,140 seconds); simulation took 1,112
seconds. The run can resume from its immutable manifest if more breadth is
needed later.

### Outcome accuracy

| Metric | All usable fights | Both fighters 5+ prior UFC fights |
| --- | ---: | ---: |
| Fights / decisive fights | 229 / 228 | 158 / 157 |
| Winner top-pick accuracy | 52.63% | 53.50% |
| Winner log loss | 0.72865 | 0.72169 |
| Winner Brier | 0.26382 | 0.26040 |
| Calibration intercept | -0.110 | +0.142 |
| Calibration slope | 0.210 | 0.253 |
| Method top-class accuracy | 49.34% | 50.63% |
| Joint side/method top-class accuracy | 28.82% | 31.65% |
| Joint side/method log loss | 2.38449 | 2.26024 |

A constant 50/50 winner forecast has log loss 0.69315 and Brier 0.25. The
simulator was worse on both. The paired event-card bootstrap estimate for
simulator-minus-50/50 winner log loss was +0.03550 with 95% interval
[+0.00017, +0.07077] across 30 cards. The Brier difference was +0.01382 with
interval [-0.00188, +0.02916]. The positive calibration slope shows some weak
directional signal, but the slope near 0.21 shows that the probabilities are
far too confident for that signal.

The primary joint score was also worse than division (1.69438) and population
(1.74569) frequency baselines. Five observed side/method outcomes received
zero paths, so the 100-path joint log loss is especially noisy and punitive;
that limitation does not explain the independently poor winner scores.

### What the simulated distributions got wrong

| Quantity | Observed / expected or bias | CRPS | 90% coverage |
| --- | ---: | ---: | ---: |
| Decision count | 125 observed / 91.85 expected | — | — |
| KO/TKO count | 65 observed / 98.65 expected | — | — |
| Submission count | 37 observed / 32.91 expected | — | — |
| Duration | -128.50 seconds mean bias | 189.86 sec | 96.1% |
| Total UFCStats control | -144.58 seconds mean bias | 156.06 sec | 73.4% |
| Significant-strike attempts | -55.23 mean bias | 82.51 | 69.0% |
| Significant strikes landed | -15.33 mean bias | 36.78 | 77.3% |
| Ground strikes landed | +2.63 mean bias | 7.04 | 95.2% |
| Knockdowns | 0.85 expected / 0.36 observed | 0.404 | 95.2% |
| Takedowns | 1.17 expected / 2.12 observed | 1.297 | 77.3% |
| Submission attempts | 0.45 expected / 0.62 observed | 0.428 | 92.6% |

Duration intervals over-covered while their mean was too short, indicating
broad but poorly centered distributions. Control, strike-attempt, and takedown
intervals under-covered and were systematically low. Ground-strike landings
were comparatively well centered. The new conditional-control fit moved the
five-card development point estimate in the desired direction, but it did not
solve the much larger population-level UFCStats-control deficit.

## Decision and next experiment

The current simulator is not useful as a standalone winner predictor and must
not influence production predictions or bets. The result is not proof that a
generative simulator can never help; it is evidence that the present mechanics
terminate fights too readily and produce overconfident winner probabilities.

The lowest-hanging coupled failure is excessive knockdowns/KO outcomes and
premature termination. Reducing that process may simultaneously increase
decision frequency, duration, standing attempts, takedowns, and control. It
should be addressed before independently scaling every underpredicted action,
which would risk double-correction.

The 30 completed cards are now development data. A next experiment may use
their cached candidate fits to predeclare a very small latent-knockdown grid,
with joint side/method score as the primary metric and duration, method,
knockdown, action, and control preservation gates. It must not call those same
cards validation. Later unrun cards beginning in January 2025 provide a
chronological confirmation window, and later cards must remain untouched until
the candidate and thresholds are frozen. Each command retains the 3,300-second
limit and common seeds. If a lower latent knockdown rate does not improve
headline probabilistic performance, retain the current mechanics and focus on
parameter calibration rather than adding more microphysics.

Reproduction:

```bash
export PYTHONPATH="$PWD/src"

python -m fight_sim posterior-backtest \
  --quick-screen \
  --simulator-config artifacts/simulations/mechanics-validated-finishing-final5-20260826.json \
  --takedown-control-association \
  --max-runtime-seconds 1200 \
  --workers 8 \
  --chunk-size 64 \
  --output-dir artifacts/simulations/td-control-screen-candidate-5-20260827

python -m fight_sim posterior-backtest \
  --last-events 100 \
  --min-prior-ufc-fights 3 \
  --bootstrap-members 10 \
  --paths-per-matchup 100 \
  --seed-repeats 1 \
  --simulator-config artifacts/simulations/mechanics-validated-finishing-final5-20260826.json \
  --takedown-control-association \
  --max-runtime-seconds 3300 \
  --workers 8 \
  --chunk-size 10 \
  --output-dir artifacts/simulations/broad-100paths-td-control-20260827
```
