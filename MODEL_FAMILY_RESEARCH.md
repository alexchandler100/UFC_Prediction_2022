# Winner model family research

## What was tested

`src/evaluate_model_families.py` compares materially different winner models
on the same point-in-time fight rows and the same 82 pre-fight variables. For
each evaluated calendar year, every model is fit only on earlier fights. Its
settings and probability calibration are selected using the latest earlier
year; the evaluated year is not used for either step.

The first comparison covers 1,877 UFC fights from January 2023 through August
2026:

| Model | Log loss | Accuracy | Brier score |
|---|---:|---:|---:|
| Current regularized logistic | **0.63138** | 64.04% | **0.22070** |
| Small neural network | 0.63601 | **64.47%** | 0.22244 |
| XGBoost | 0.63626 | 63.45% | 0.22288 |
| Fully Bayesian hierarchical probit | 0.63880 | 62.81% | 0.22413 |
| Random forest | 0.64082 | 63.35% | 0.22499 |
| Histogram gradient boosting | 0.64125 | 62.97% | 0.22499 |
| Gaussian Naive Bayes | 0.65335 | 60.90% | 0.23071 |

Lower log loss and Brier score are better. The neural network picked slightly
more winners, but its probabilities were worse. Because betting decisions
depend on probability quality rather than only selecting the favorite, it did
not beat the logistic model.

Whole-event resampling shows that Naive Bayes, random forest, histogram
boosting, and the first hierarchical Bayesian specification were reliably
worse than logistic regression on this period. The neural-network and
XGBoost differences were small enough that the uncertainty ranges still
include equal performance, but neither had the better point estimate.

Fixed 50/50 log-odds blends were also checked in case a weaker standalone
model contributed different information. None beat logistic regression:

| Logistic plus | Blend log loss |
|---|---:|
| XGBoost | 0.63207 |
| Neural network | 0.63208 |
| Random forest | 0.63316 |
| Hierarchical Bayesian | 0.63336 |
| Histogram boosting | 0.63378 |
| Naive Bayes | 0.63691 |

These are development results, not a permanent claim that logistic regression
will always be best. The period has now influenced model design and cannot be
described as fresh promotion evidence for a later variation.

## What “fully Bayesian” means here

The existing website Bayesian challenger is not a separate winner model. It
starts from the production logistic coefficients and uses a Laplace
approximation to show coefficient uncertainty.

The new `fight_predictor.hierarchical_bayes` model is genuinely different:

```text
P(fighter wins) = Phi(matchup variables × coefficients
                      + fighter ability - opponent ability)
```

It samples all 82 coefficients, every observed fighter ability, the population
variance controlling how strongly fighter abilities shrink toward average,
and a latent performance for every training fight. Two Albert-Chib Gibbs
chains use 300 warm-up and 300 retained draws each. A new fighter's unknown
ability is drawn from the learned population distribution in every posterior
draw rather than assigned an unjustified fixed rating.

This is a fully Bayesian model even though it does not use PyMC or Stan. Those
packages provide general-purpose samplers; they do not define the model. The
conditional distributions in this probit formulation can be sampled directly
with NumPy and SciPy. A small PyMC or Stan reproduction is still valuable as a
sampler-verification test later.

The two chains differed by about 1.2–1.5 percentage points on an average
forecast and roughly 2.8–4.1 points at the 95th percentile, depending on the
year. That is adequate to show that the observed performance deficit is much
larger than ordinary Monte Carlo jitter, but not adequate for publishing
production posterior intervals.

## Why the first Bayesian specification likely lost

- Fighter ability is static within each training period. It cannot represent
  development, aging, injuries, or abrupt decline.
- The 82 inputs already include several Elo ratings and historical summaries.
  Adding another static fighter ability can count similar evidence twice.
- One shared coefficient prior is unlikely to be appropriate for ratings,
  physical measurements, activity, striking, and grappling variables.
- The same 82-variable set was intentionally used for every family to isolate
  the algorithm. It is not necessarily the best Bayesian feature set.
- UFC outcomes alone provide limited information for estimating thousands of
  fighter effects plus 82 coefficients.

## Second Bayesian experiment

The planned redesign is now complete. Nine versions were compared on 1,953
fights from 2019 through 2022. They tested static versus changing fighter
skill, inclusion versus removal of Elo, ordinary versus feature-group priors,
and a model with no separate fighter-skill term. Each version used 300 warm-up
and 300 retained samples in each of two chains for every test year.

The best version kept all 82 variables, removed the extra fighter-skill term,
and applied tighter priors to the many related strike and grappling variables.
This matters: the data did not support an additional hidden fighter rating on
top of the point-in-time variables. Changing that rating after each fight was
also worse than keeping it static.

The selected version and a 25.5% Bayesian blend weight were frozen before the
later comparison:

| 2023-August 2026 model | Log loss | Accuracy | Brier score |
|---|---:|---:|---:|
| Current logistic | **0.63138** | **64.04%** | **0.22070** |
| Frozen logistic/Bayesian blend | 0.63164 | **64.04%** | 0.22078 |
| Improved fully Bayesian probit | 0.63471 | 63.35% | 0.22204 |
| First hierarchical Bayesian probit | 0.63880 | 62.81% | 0.22413 |

The redesign recovered about 55% of the first Bayesian model's log-loss gap,
but it did not beat logistic regression. Its point estimate was worse in every
evaluated year. Whole-event resampling put the Bayesian-minus-logistic
difference at `+0.00334`, with a 95% range from `-0.00029` to `+0.00707`.
The blend difference was `+0.00026`, with a range from `-0.00067` to
`+0.00122`. Those ranges include equal performance, so the new model is not
proven worse, but there is no evidence to promote or blend it.

This also gives a practical next decision. More elaborate hidden fighter-skill
states are not the lowest-cost path to better forecasts. If Bayesian work
continues, the bounded next test should use a logistic link and learn the
amount of shrinkage for each feature group from the earlier development data.
A small PyMC or Stan reproduction is still useful to verify the sampler, but
changing sampler software by itself is not expected to improve predictions.

## Third Bayesian experiment

The Bayesian logistic follow-up is complete. It keeps the logistic winner
likelihood but samples all 82 coefficients and the amount of coefficient
shrinkage. Five prior designs were compared on the same 1,953 fights from
2019–2022. The selected design learns separate shrinkage amounts for ratings,
physical attributes, activity/experience, records/results, striking, and
grappling.

The sampler uses Hamiltonian Monte Carlo with exact accept/reject correction,
plus exact conditional draws for the six shrinkage variances. It does not need
PyMC or Stan. On the four later yearly models, the two chains differed by only
0.13–0.19 percentage points on an average forecast. This is much steadier than
the earlier slice-sampling attempt and is adequate for the comparison.

The selected Bayesian model and its 54.6% share of a log-odds blend were frozen
before scoring 2023–August 2026:

| Model | Log loss | Accuracy | Brier score |
|---|---:|---:|---:|
| Frozen logistic/Bayesian blend | **0.63132** | 63.83% | **0.22062** |
| Current logistic | 0.63138 | **64.04%** | 0.22070 |
| Fully Bayesian logistic | 0.63181 | 63.83% | 0.22076 |
| Previous Bayesian probit | 0.63471 | 63.35% | 0.22204 |

The blend's log-loss improvement is only `0.00006`. Whole-event resampling
puts its 95% range from `-0.00128` to `+0.00118`, where negative favors the
blend. The Bayesian model and blend were worse in 2023–2024 and better in
2025–2026. Therefore this is an interesting point estimate, not reliable
evidence of an improvement. It does not justify changing production.

The learned standardized coefficient scales consistently gave physical
attributes the most freedom (roughly `0.16–0.24`), ratings about `0.10`, and
record/results the strongest shrinkage (roughly `0.06`). These are useful
model diagnostics, but they are not standalone measures of feature importance.

Further retrospective tuning on 2023–2026 would increasingly fit research
choices to known results. The useful next step is to freeze this exact blend
as paper-only and measure it on genuinely new fights. A production change
would still require a separate reviewed decision after enough prospective
evidence.

That prospective test is now implemented. The prior design and the
`0.5462639465757038` Bayesian share of the log-odds blend are constants in
code. On each scheduled data update, the Bayesian coefficients may learn from
newly completed fights under the same fixed procedure, just as the production
model is refreshed, but the model recipe and blend weight cannot be selected
again. The first pre-event forecast for a matchup is kept permanently in
`src/content/data/market/bayesian_logistic_shadow_forecasts.csv` and `.jsonl`.

After results arrive, the performance report compares the production
probability, Bayesian probability, and frozen blend on exactly the same fights.
It reports accuracy, log loss, Brier score, and a whole-card resampling range
for the paired log-loss difference. Formal review waits for at least 200 fights
across 20 events, and even a passing result only supports a separate reviewed
production change. Betting and automatic promotion remain disabled. A
date-only forecast made on the event's UTC date is refused rather than being
misrepresented as prospective; consequently the first record begins with the
next future card after this code is deployed, not the already-started August
29 card.

The nonlinear models also deserve one second-pass experiment using
family-specific variable selection and causally selected small blend weights.
That work should remain separate from the Bayesian redesign so a gain can be
attributed to the model rather than an uncontrolled combination of changes.

## Reproduction

Install the optional free research dependency and run the comparison locally:

```bash
python -m pip install -r requirements-research.txt
python src/evaluate_model_families.py \
  --years 2023 2024 2025 2026 \
  --workers 4 \
  --bayes-burn-in 300 \
  --bayes-draws 300 \
  --bayes-chains 2

python src/evaluate_dynamic_bayes.py \
  --development-years 2019 2020 2021 2022 \
  --evaluation-years 2023 2024 2025 2026 \
  --selection-burn-in 300 \
  --selection-draws 300 \
  --final-burn-in 300 \
  --final-draws 300 \
  --chains 2

python src/evaluate_bayesian_logistic.py \
  --development-years 2019 2020 2021 2022 \
  --evaluation-years 2023 2024 2025 2026 \
  --selection-burn-in 600 \
  --selection-draws 600 \
  --final-burn-in 1000 \
  --final-draws 1000 \
  --chains 2
```

The run took about 5.2 minutes on the development machine. The JSON report and
fight-level probabilities are stored under
`src/content/data/model_research/model_family_comparison.*`. Production model
artifacts, website predictions, and betting behavior are unchanged. The second
experiment took 13.8 minutes and wrote
`src/content/data/model_research/dynamic_bayes_*`. The third experiment took
5.4 minutes and wrote `src/content/data/model_research/bayesian_logistic_*`.
