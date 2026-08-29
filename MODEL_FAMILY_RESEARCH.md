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

## Next Bayesian experiment

The next version should be designed and tuned only on earlier development
years before it is rescored:

1. Replace static ability with a time-evolving fighter state whose uncertainty
   increases during inactivity and whose mean can change after each fight.
2. Partially pool fighter states by division and era while retaining one
   identity across division changes.
3. Remove Elo variables when latent fighter state is enabled, then compare
   that version with a coefficients-only Bayesian model to measure whether
   fighter effects add anything.
4. Give ratings, physical attributes, activity, striking, and grappling groups
   separate shrinkage priors. Perform prior-predictive checks before looking at
   fight outcomes.
5. Select prior scales and the Bayesian feature group using 2019–2022 only.
   Treat 2023–2026 as reused development evidence, then require prospective
   fights for any eventual promotion claim.
6. Reproduce a bounded subset in PyMC or Stan and compare posterior means and
   intervals with the direct sampler. This verifies inference; it is not
   expected to improve predictions by itself.

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
```

The run took about 5.2 minutes on the development machine. The JSON report and
fight-level probabilities are stored under
`src/content/data/model_research/model_family_comparison.*`. Production model
artifacts, website predictions, and betting behavior are unchanged.
