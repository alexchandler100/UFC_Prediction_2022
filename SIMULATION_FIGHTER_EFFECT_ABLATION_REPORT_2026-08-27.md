# Fighter-effect ablation report

Date: 2026-08-27

Status: the reliability-weighted snapshot is retained only as a research
diagnostic. It is not validated, must not affect the website or production
predictions, and does not yet justify opening the locked confirmation cohort.

## Question and frozen design

The v2.1 outcome engine made method and duration distributions substantially
more realistic, but its winner probabilities remained worse than constant
50/50 under log loss and Brier score. Its calibration slope of 0.224 suggested
that fighter-specific deviations might be much noisier or more extreme than
their predictive information supports.

Three snapshot policies were therefore compared on the already-open
`development_2024` cohort from `SIMULATION_EXPERIMENT_COHORTS_V1.json`:

1. `full`: the existing strongly pooled fighter parameters plus the global
   age, experience, and layoff rate adjustment;
2. `context_only`: division/era parameters with no fighter-specific deviation
   and no fighter-specific covariate adjustment; and
3. `reliability_weighted`: the fitted fighter vector receives a second causal,
   parameter-specific shrinkage step toward its division/era context.

The weighted policy uses fight minutes for action rates, strike/takedown/
submission opportunities for conditional parameters, observed composition
counts for phase and target mix, and eligible later rounds for pace decay.
Positive rates are interpolated geometrically, probabilities on the logit
scale, and compositions in normalized geometric space. Unsupported latent
mechanics receive no fighter deviation. These weights use only history strictly
before the event cutoff. They do not vary by simulation outcome or use any
2025 confirmation fight.

All arms used the same v2.1 mechanics, 229 fights / 30 complete cards, ten
bootstrap members, 100 total paths per fight, one seed, causal cutoff fits, and
same-member pairing. The named RNG contract does not include snapshot values,
so corresponding paths used common random draws. Context-only completed in
1,141.5 seconds and reliability-weighted in 1,141.9 seconds. Total new
simulation compute was 38.1 minutes for 45,800 paths, with no incomplete fights
or time-limit stop.

## Results

There were 228 decisive fights; one draw/no-contest outcome was omitted only
from binary winner metrics.

| Metric | Full fighter | Context only | Reliability weighted |
|---|---:|---:|---:|
| Winner accuracy | 52.63% | 48.25% | **55.70%** |
| Winner log loss | 0.72135 | **0.70886** | 0.71470 |
| Winner Brier | 0.26127 | **0.25768** | 0.25863 |
| Calibration intercept | -0.110 | -0.122 | **-0.103** |
| Calibration slope | 0.224 | -1.158 | **0.260** |
| Joint side/method log loss | 1.98694 | 1.98273 | **1.96508** |
| Method log loss | **1.13801** | 1.16825 | 1.14370 |
| Duration CRPS, seconds | 166.04 | **162.42** | 164.26 |
| Mean duration bias, seconds | -55.74 | +5.71 | -32.07 |
| Predicted knockdowns/fight | 0.4097 | 0.2731 | **0.3832** |
| Observed knockdowns/fight | 0.3581 | 0.3581 | 0.3581 |
| Predicted KO/TKO count | 72.65 | 54.64 | **69.25** |
| Observed KO/TKO count | 65 | 65 | 65 |
| Predicted decision count | 113.36 | 131.84 | **120.35** |
| Observed decision count | 125 | 125 | 125 |

Reliability weighting improved full-fighter winner log loss by 0.00665, Brier
by 0.00264, joint log loss by 0.02186, duration CRPS by 1.78 seconds, accuracy
by 3.07 percentage points, and the absolute KO and decision count biases. Its
method log loss worsened by 0.00569. Ground-strike CRPS improved by 10.3%; the
other protected action CRPS ratios were between 1.004 and 1.033, below the 5%
harm threshold.

None of the paired proper-score differences was statistically resolved across
cards. For reliability-weighted minus full, the event-card 95% interval was
`[-0.0825, 0.0348]` for joint log loss, `[-0.0233, 0.0146]` for winner log
loss, and `[-0.0102, 0.0071]` for winner Brier score. Context-only also had
wholly crossing-zero intervals against full. The generic outcome-mechanics
comparator labels both snapshot candidates `rejected` because its predeclared
mechanics gate requires method log loss to improve; that label is recorded but
is not repurposed as a fighter-effect selection rule.

The 55.70% weighted accuracy was 127/228. Its Wilson 95% interval was
49.21%--62.00%, and the exact two-sided binomial p-value against 50% was 0.098.
The five-prior-fights-per-side slice was directionally similar but smaller:
54.78% accuracy, 0.71525 winner log loss, and 0.25897 Brier across 157 decisive
fights.

## What the result means

Context-only is not a useful winner ranker. Its apparently better proper scores
come from collapsing toward 50/50; its accuracy fell below 50%, its calibration
slope became meaningless/negative, and its method counts lost useful fighter
heterogeneity. This is evidence that the fighter histories contain some
directional and method signal.

Reliability weighting is the best of the three diagnostic compromises. It
ranked more winners correctly, improved joint outcome and several realism point
metrics, and moved the calibration slope in the right direction. However, its
winner log loss of 0.71470 and Brier of 0.25863 are still worse than the 50/50
references of 0.69315 and 0.25000. Its accuracy interval includes chance and
all paired proper-score intervals include zero. It is therefore not evidence
of a standalone betting edge.

A symmetry-preserving zero-intercept temperature fit provides a useful
diagnostic, not a promotable correction. The full-fighter and weighted
probabilities preferred logit multipliers of 0.219 and 0.263 in sample. When
that one coefficient was refitted while leaving each event card out, weighted
log loss was 0.693205 and Brier was 0.249986--effectively identical to a coin
flip. Context-only selected a zero multiplier exactly. The apparent directional
signal is too weak for calibration alone to establish predictive value on this
sample.

## Next bounded experiment

Do not tune another scalar snapshot multiplier on these same trajectories and
do not blend this simulator into the production winner model. The next useful
change is in the parameter model:

1. jointly estimate opponent-adjusted offensive and defensive effects for the
   observable strike, takedown, and submission opportunity models instead of
   treating a fighter's raw history as independent of opponent strength;
2. estimate between-fighter variance and reliability from training data rather
   than applying another fixed pseudo-exposure rule;
3. preserve division/era parents, strict event cutoffs, same-card exclusion,
   bootstrap covariance, and the v2.1 trajectory mechanics; and
4. screen one predeclared joint-model candidate against both `full` and
   `reliability_weighted` on the 229-fight open development cohort at 100 paths
   per fight, with a 3,300-second hard cap.

The candidate advances only if winner log loss and Brier improve without a
material joint side/method, duration, or action-distribution regression. A
point accuracy increase alone is insufficient. The locked
`confirmation_2025_a` cohort remains unopened until one fighter-parameter
model survives this development gate. The separate requirement to confirm the
v2.1 outcome mechanics also remains in force.

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
  --snapshot-parameter-mode reliability_weighted \
  --max-runtime-seconds 1800 \
  --workers 4 \
  --chunk-size 64 \
  --output-dir artifacts/simulations/two-route-v2-1-reliability-development-100paths-20260827
```

The context-only report hash is
`d78c466a78811658000d45bccbbdf2ee2799c9a7d14477af1edb4dff90398c02`.
The reliability-weighted report hash is
`18bdbc77a591fb914b4b28918e35daf23e583252c98befa6b6a273378a1a43c8`.
The full-versus-weighted comparison hash is
`e3145fde34fb6b8e5b543b600daf120edf9d659be9be6b3415418f94f21be99c`.
Detailed ledgers and HTML reports remain ignored beneath `artifacts/`.
