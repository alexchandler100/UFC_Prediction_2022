# UFC simulation posterior-predictive tuning report — 2026-08-26

## Decision

Keep `mechanics-8ba01f34444f` unchanged. Reject the tested
`escape_hazard_multiplier = 0.67` candidate.

The candidate made simulated ground-control episodes longer. It improved the
control-time distribution, but the improvement was too small to explain the
official UFCStats control discrepancy and it degraded winner probabilities and
ground-strike distributions on the higher-precision confirmation cohort. The
website and upcoming-card forecasts were not changed.

## Experiment contract and runtime

- Fighters with fewer than three strictly prior UFCStats bouts were excluded.
- Screen: the newest ten completed cards, 65 eligible fights, 16 causal
  bootstrap members, 512 paths per fight, one seed, identical fighter snapshots
  and common seeds.
- Confirmation: the newest five cards, 31 eligible fights, 16 members, 1,024
  paths per fight and two seeds. This exactly matches the retained profile's
  existing 63,488-path holdout fidelity.
- The candidate value was predeclared from the middle-five-card control ratio:
  `125.24 / 186.38 = 0.672`, rounded to `0.67`.
- Current ten-card screen: 24.9 minutes, including ten causal cache misses.
- Candidate ten-card screen: 8.0 minutes, with ten cache hits.
- Candidate five-card confirmation: 7.7 minutes, with five cache hits.
- New computation: 130,048 paths in about 40.6 simulation/backtest minutes.

All results are candidate-only research. The screen uses one seed and is not
promotion evidence; the deciding comparison below uses the higher-precision
two-seed confirmation.

## How accurate the retained simulator currently is

These are strictly pre-fight predictions for 31 eligible fights from the newest
five completed cards. A probability score is more informative than raw accuracy,
but both are reported for interpretability.

| Target | Current result | Reference / interpretation |
| --- | ---: | --- |
| Winner accuracy | 41.9% (13/31) | Below 50%; small sample, but not encouraging |
| Winner log loss | 0.738 | Worse than constant 50/50 (`0.693`) |
| Winner Brier score | 0.272 | Worse than constant 50/50 (`0.250`) |
| Winner calibration | intercept 0.208; slope 0.206 | Far outside the predeclared ±0.05 / 0.85–1.15 gates |
| Exact side-and-method accuracy | 6.5% (2/31) | Very poor point classification |
| Joint side-and-method log loss | 1.951 | Worse than division `1.809` and population `1.837` baselines |
| Method accuracy | 32.3% (10/31) | Approximately chance-level among the major methods |
| Method log loss | 1.218 | Candidate-only; no demonstrated betting advantage |
| Goes-distance accuracy | 41.9% (13/31) | Not useful as a binary point prediction |
| Duration CRPS | 255.9 seconds | Full-distribution error; lower is better |
| Mean-duration absolute error | 382.8 seconds | About 6.4 minutes per fight |
| Median-duration absolute error | 421.8 seconds | About 7.0 minutes per fight |

The action-statistic picture is mixed rather than uniformly bad:

| Fight total | Observed mean | Predicted mean | Bias | Central 90% coverage |
| --- | ---: | ---: | ---: | ---: |
| Significant strikes landed | 80.65 | 82.31 | +1.67 | 74.2% |
| Significant-strike attempts | 156.06 | 144.84 | -11.23 | 67.7% |
| UFCStats control seconds | 268.19 | 144.57 | -123.63 | 80.6% |
| Takedowns landed | 1.90 | 1.30 | -0.61 | 80.6% |
| Submission attempts | 0.77 | 0.49 | -0.29 | 93.5% |
| Knockdowns | 0.45 | 0.78 | +0.33 | 96.8% |
| Ground significant strikes landed | 13.00 | 14.85 | +1.85 | 90.3% |

The mean for significant strikes is close, but its interval is too narrow.
Attempts, control, and takedowns are underpredicted; knockdowns are
overpredicted. Correct aggregate means alone therefore do not imply realistic
fight-level distributions.

## Tested control candidate

Candidate-minus-current differences are negative when the candidate improves a
loss/CRPS. Intervals use paired event-card block resampling on the 31-fight,
five-card confirmation cohort.

| Metric | Candidate minus current | Event-block 95% interval | Result |
| --- | ---: | ---: | --- |
| Joint side/method log loss | -0.0140 | [-0.0377, +0.0148] | Directionally better, inconclusive |
| Winner log loss | +0.0050 | [+0.0002, +0.0119] | Reliably worse |
| Method log loss | -0.0096 | [-0.0217, +0.0068] | Directionally better, inconclusive |
| Duration CRPS | -1.27 sec | [-4.07, +1.41] | Directionally better, inconclusive |
| Control-time CRPS | -3.58 sec | [-5.61, -1.30] | Reliably better, but small |
| Ground-strike CRPS | +0.178 | [+0.084, +0.260] | Reliably worse |
| Takedown CRPS | +0.0148 | [-0.0003, +0.0284] | Directionally worse |

Predicted control increased from 144.57 to 154.67 seconds, leaving a 113.53
second underprediction. Longer ground stays also changed finish opportunities
and ground actions, which is why fixing an observed statistic by changing the
escape process caused collateral predictive harm.

## Is the simulator worthwhile?

As a standalone winner, method, or totals betting model: **not currently**.
This cohort provides no evidence of a usable winner edge, and its probability
scores are worse than simple references. It must remain candidate-only and must
not influence production probabilities or wagers.

As a research framework: **potentially, but unproven**. Earlier chronological
tuning materially improved action moments, duration CRPS, and side/method loss,
and this experiment detected a reproducible control-distribution response in a
bounded 40-minute run. That shows the simulator is measurable and tunable. It
does not show that its detailed trajectories currently add predictive signal.

The practical conclusion is neither “working” nor “hopeless”: the architecture
is useful for falsifiable research, while the present parameterization is not a
competitive predictive model. The logistic model should remain authoritative
for winners. Simulation-derived betting views should remain research outputs
until they beat chronological baselines on materially larger holdouts.

## Next predeclared experiment

Do not keep lowering escape hazards. UFCStats control includes control semantics
that are broader than the simulator's current ground-top clock. The next change
should separate the latent fight dynamics from the observation model:

1. Preserve ground-top time as an internal simulated state statistic.
2. Add an explicit, globally fitted UFCStats-control observation projection,
   with separate ground and supported clinch-control contributions and a hard
   cap at fight duration. If controlling-side clinch state cannot be supported,
   do not invent it; use only the ground mapping first.
3. Fit the mapping on development cards, select it on intermediate cards, and
   evaluate once on later cards without changing finish or judging hazards.
4. Separately investigate winner discrimination, attempt-count dispersion,
   takedown volume, and excess knockdowns. Do not solve those deficiencies by
   one global multiplier sweep.

The new causal-fit cache reduces subsequent ten-card mechanics candidates from
about 25 minutes to about eight minutes, so these isolated tests can remain
bounded and interpretable.

## Follow-up low-hanging-fruit experiments

Three additional candidates were tested after the control experiment. None was
allowed to replace the retained profile.

### Winner temperature calibration

A nonnegative, zero-intercept temperature was fitted on the earlier five-card
selection cohort and evaluated once on the later five cards. The fitted
temperature was effectively zero (`5.3e-13`). It mapped every decisive winner
probability to 50/50 and improved holdout winner log loss from `0.738` to the
constant-reference `0.693`, but left winner accuracy unchanged. This is signal
deletion, not useful calibration, and was rejected. It is additional evidence
that the current simulated winner ranking is not predictive.

### Knockdown-rate-only candidate

The retained simulator predicted 1.038 total knockdowns per fight versus 0.588
observed on the middle-five selection cards. Applying that ratio to the retained
`0.800686` knockdown multiplier produced a predeclared `0.45` candidate.

On the 65-fight screen it improved winner log loss (`0.761` to `0.738`) and
knockdown CRPS (`0.395` to `0.337`), but worsened the primary joint loss
(`1.787` to `1.824`), method loss (`1.039` to `1.077`), duration CRPS (`201.83`
to `208.66` seconds), and landed-strike CRPS. Fewer knockdowns caused fewer
finishes and longer fights. It failed the screen and did not advance.

### Coupled knockdown/finish candidate

The next candidate preserved approximate KO finish intensity while reducing
repeated knockdowns: `knockdown_probability_multiplier = 0.45` and
`ko_tko_finish_probability_multiplier = 0.71`, since
`0.800686 * 0.40 / 0.45 = 0.7117`.

It passed the 65-fight screen and advanced to the two-seed, 63,488-path
confirmation. On those 31 later fights it produced these point estimates:

| Metric | Retained profile | Coupled candidate |
| --- | ---: | ---: |
| Winner accuracy | 41.9% | 45.2% |
| Winner log loss | 0.7383 | 0.7342 |
| Winner Brier | 0.2718 | 0.2704 |
| Exact side/method accuracy | 6.5% | 6.5% |
| Joint side/method log loss | 1.9509 | 1.9347 |
| Method log loss | 1.2178 | 1.2144 |
| Goes-distance accuracy | 41.9% | 48.4% |
| Duration CRPS | 255.88 sec | 254.04 sec |
| Predicted / observed knockdowns | 0.780 / 0.452 | 0.444 / 0.452 |
| Knockdown CRPS | 0.417 | 0.352 |

Paired five-event intervals crossed zero for joint, winner, method, duration,
and knockdown score changes; no headline metric showed reliable degradation.
Control CRPS improved reliably, while significant-strike and takedown changes
were small and inconclusive.

Despite broadly favorable point estimates, the formal predeclared validation
returned `rejected_baseline_fallback`. Mean duration bias moved from `+9.96` to
`+16.77` seconds, failing the required `absolute_duration_bias_improves` gate.
The validation artifact is
`artifacts/simulations/mechanics-validated-knockdown045-finish071-final5-20260826.json`
with hash
`3d44630bfede78e42ae76143ec8e0a8a79cd2ddbeac839b90619306d6dcf3a44`.

The retained profile therefore remains `mechanics-8ba01f34444f`. The coupled
candidate is promising evidence for a better parameterization, not a validated
replacement. Do not tune `0.71` again after observing this holdout. A future
round must predeclare a small coupled candidate set using earlier, previously
unused cards and reserve a fresh chronological confirmation cohort—or wait for
prospective fights—to avoid repeatedly optimizing the same five cards.
