# Predeclared knockdown/finish grid - 2026-08-27

## Purpose and boundary

Test whether reducing excess simulated knockdowns while increasing conditional
KO/TKO conversion can improve the simulator's primary joint side/method log
loss without materially damaging winner, duration, or action distributions.
This is candidate-only research. It cannot alter the website, production
predictions, or betting decisions.

The retained baseline is `mechanics-8ba01f34444f`, with knockdown multiplier
`0.8006864880387029` and conditional KO/TKO-finish multiplier `0.40`.

## Locked candidates

Each pair approximately preserves the baseline product
`0.8006864880387029 * 0.40 = 0.3202745952`:

| Label | Knockdown multiplier | KO/TKO conversion multiplier | Product |
| --- | ---: | ---: | ---: |
| `kd035-ko092` | 0.35 | 0.92 | 0.3220 |
| `kd045-ko071` | 0.45 | 0.71 | 0.3195 |
| `kd060-ko053` | 0.60 | 0.53 | 0.3180 |

No candidate value may be changed after a result from either locked cohort is
opened.

## Locked cohorts and fidelity

- Selection: five events from 2025-12-06 through 2026-02-07, expressed to the
  command as `--last-events 5 --skip-latest-events 25`.
- Confirmation: the immediately following five events from 2026-02-21 through
  2026-03-21, expressed as `--last-events 5 --skip-latest-events 20`.
- Require at least three strictly prior UFCStats fights for both fighters.
- Selection uses 16 bootstrap members, 512 paths per matchup, seed `2903`, and
  one seed repeat.
- Only the selected candidate advances. Confirmation uses 16 bootstrap members,
  1,024 paths per matchup, seed `2903`, and two seed repeats.
- Candidate and baseline runs use identical cohort, member, seed, worker, and
  chunk contracts. Causal parameter fits may be reused from the content-addressed
  cache, but aggregate simulations may not be substituted across configurations.

## Locked selection and validation rules

Selection uses `select-finishing --objective joint`. A candidate must improve
joint side/method log loss; remain within 0.02 of baseline method log loss,
winner log loss, and observable action error; remain within five seconds of
baseline duration CRPS; and keep absolute duration bias within 15 seconds of
baseline. Among eligible candidates, minimize joint log loss and use duration
CRPS only as the tie-breaker. If none is eligible, retain the baseline.

The one selected candidate is then evaluated once with `validate-finishing`.
It must pass every existing holdout gate, including improved duration CRPS and
improved absolute duration bias. A failure falls back to the baseline.

Even a passing historical confirmation creates only a named prospective shadow
candidate. The broader global mechanics were developed using later fights, so
this experiment is an incremental parameter comparison rather than a clean
end-to-end causal backtest. Promotion still requires untouched prospective
evidence and an explicit reviewed change.

## Locked follow-up observation experiment

The coupled grid returned `baseline_fallback`: all three candidates improved
the knockdown distribution, but none improved joint side/method log loss. The
development baseline observed 0.600 knockdowns per fight and predicted 1.015.
The direct ratio is `0.600 / 1.0147879464 = 0.5913`; before opening the locked
confirmation cohort, this is rounded to a single official-observation
probability of `0.59`.

This follow-up does not alter the latent knockdown, hurt, damage, effectiveness,
finish, judging, or termination processes. It only thins which latent
knockdowns are projected into the official UFCStats knockdown counter. The
default probability is `1.0`, preserving previous behavior. A separate named
random stream makes the observation draw unable to perturb any trajectory RNG
stream.

The confirmation cohort and fidelity remain locked as specified above. The
candidate passes only if:

1. joint, winner, method, duration CRPS, and duration-bias metrics are exactly
   unchanged;
2. every non-knockdown observable action error is exactly unchanged;
3. knockdown CRPS, absolute knockdown mean bias, and combined observable action
   error all improve; and
4. the observation probability is the only configuration difference.

A pass retains the projection only for prospective shadow research. It does
not promote the simulator or alter current forecasts.

## Results

The 35-fight development screen ran 71,680 paths. Its baseline and candidates
were:

| Profile | Joint log loss | Winner log loss | Method log loss | Duration CRPS | Knockdown CRPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1.8188 | 0.6040 | 1.1267 | 182.75 sec | 0.3717 |
| `kd035-ko092` | 1.8372 | 0.5959 | 1.1422 | 179.69 sec | 0.3399 |
| `kd045-ko071` | 1.8954 | 0.6049 | 1.1804 | 184.01 sec | 0.3337 |
| `kd060-ko053` | 1.8846 | 0.6015 | 1.1585 | 181.27 sec | 0.3349 |

All three improved knockdown CRPS but worsened the primary joint score. The
selector returned `baseline_fallback`, so no coupled mechanics candidate
advanced.

The official-observation projection then ran 126,976 paths on the locked
31-fight confirmation cohort. Every gate passed:

| Metric | Baseline | Observation 0.59 |
| --- | ---: | ---: |
| Joint side/method log loss | 1.683869 | 1.683869 |
| Winner log loss | 0.610680 | 0.610680 |
| Method log loss | 0.967728 | 0.967728 |
| Duration CRPS | 167.788 sec | 167.788 sec |
| Predicted / observed knockdowns | 0.879 / 0.290 | 0.522 / 0.290 |
| Knockdown CRPS | 0.3646 | 0.2280 |
| Combined observable-action error | 0.5619 | 0.4863 |

The paired five-event, 20,000-replicate block-bootstrap interval for the
candidate-minus-baseline knockdown CRPS change was `[-0.1976, -0.0782]`, around
a mean improvement of `-0.1366`; every resampled-card endpoint remained below
zero.

Every non-knockdown action error was exactly unchanged. The validation result
is `retained_for_prospective_shadow`, profile `mechanics-7dca94cd8d5b`, with
validation hash
`e99d2257a4c4a7aefc33ec319fccdc43188deb63e72fde30c080ddea99cad790`.
The candidate still overpredicts knockdowns, so `0.59` must not be retuned on
this opened cohort. The production/website profile remains
`mechanics-8ba01f34444f`.

Total new computation was 198,656 paths in about 53.2 minutes: 12.8 minutes
for the development baseline, 14.7 minutes for its three cached candidates,
16.9 minutes for the confirmation baseline, and 8.8 minutes for the cached
observation candidate.
