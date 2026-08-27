[![Update Data](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/update-data.yml/badge.svg)](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/update-data.yml)
[![Collect Market Snapshot](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/collect-market-snapshot.yml/badge.svg)](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/collect-market-snapshot.yml)

# UFC Prediction

This project collects UFCStats fight data and publishes a searchable fighter and matchup research site at [alexchandler100.github.io/UFC_Prediction_2022](https://alexchandler100.github.io/UFC_Prediction_2022/). The site exposes complete fighter profiles, career rates and totals, every recorded bout, matchup-specific comparisons, current model context, and book-by-book market research. The production weekly predictor remains the Python point-in-time winner model described below.

## Production pipeline

The weekly job:

1. Reconciles recent completed events from UFCStats and refreshes active fighter profiles.
2. Validates source IDs, mirrored fight rows, results, card order, timing metadata, and numeric domains.
3. Replays fights in causal bout order and builds one stable-ID feature row per physical W/L fight.
4. Tunes and evaluates a regularized logistic model with nested chronological folds, adds pre-bout Elo/state features, applies symmetric temperature calibration, and refits on the full ten-year window.
5. Saves and reloads a content-hashed JSON model artifact before forecasting the next card.
6. Builds a paper-only Bayesian logistic challenger around the same MAP coefficients, evaluates its posterior mean chronologically, and publishes probability intervals.
7. Builds a compact stable-ID fighter explorer publication with UFCStats performance plus linked Bellator/ONE history.
8. Adds timestamped no-vig multi-book consensus from The Odds API when valid lines are available and publishes the website JSON.

The 82-feature model is antisymmetric: swapping the fighters produces the complementary probability. Historical features use only information available before that bout; appending future fights cannot change an existing training row. Split decisions are valid W/L labels, while draws and no-contests are retained as state/history events but are not binary training labels.

Strict validation now rebuilds the full point-in-time matrix from raw fights and
fighter profiles and compares every one of its feature cells. The formula
contract, missing-data behavior, source invariants, and known coverage limits
are recorded in [DERIVED_DATA_AUDIT.md](DERIVED_DATA_AUDIT.md). The old
326-column `ufc_fights_reported_derived_doubled.csv` is retained only for
notebook compatibility; its overlapping composite scores and undefined-rate
handling are not used by the production model or website.

The regularization search includes `C=0.001` and `0.003`; an ablation found the old `0.01` lower boundary was masking stronger shrinkage. Recency weighting, decision-label weighting, and Glicko/RD were tested but remain unpromoted because their gains were small or inconsistent. Exact results are recorded in [AUDIT.md](AUDIT.md).

The Bayesian challenger does not replace that production probability. The L2
solution is treated as the MAP estimate under independent zero-mean Gaussian
coefficient priors, and the inverse penalized-likelihood Hessian supplies a
deterministic Laplace posterior. Each matchup therefore has a normal posterior
on its calibrated logit and a logit-normal posterior on win probability. The
site reports the posterior mean, 90% credible interval, expected-return
interval at the current best price, and posterior probability that EV is
positive. Calibration-slope, feature-measurement, and model-form uncertainty
are not included, so the interval is explicitly labeled a challenger estimate.
If either fighter has fewer than two prior bouts, the site may show the
coefficient-only interval for research but refuses to calculate a Bayesian EV
candidate; a zero feature vector must never appear falsely certain.
The predeclared shadow gate requires at least 5% posterior-mean EV and an 80%
posterior probability of positive EV; execution remains disabled pending
prospective CLV and return evidence.

The weekly publication also freezes the best available displayed book/price,
mean EV, 90% EV interval, probability of positive EV, and shadow pass/selection
under `bayesian-moneyline-shadow-v1`. After the card, those rows move into
prediction history and the performance report scores their log loss, Brier
score, flat-unit return, and whole-card bootstrap interval. This first-pass
monitor is timestamped but is not yet the immutable T-24/CLV ledger, so it can
never by itself enable execution.

A second policy now tests the Bayesian model as a conservative veto on the
existing moneyline policy rather than as an independent pick generator. At the
same immutable T-24 capture, the base leave-one-book-out policy first chooses
its target book, side, and price. The Bayesian filter may keep that exact side
only when posterior-mean EV is at least 5% and the posterior probability of
positive EV is at least 80%; otherwise it records a pass and never switches to
the opponent. The source Vegas publication hash, Bayesian artifact hash,
posterior, base decision ID, price, and filter reason are frozen in a separate
append-only ledger. Performance reporting compares base and filtered ROI on
the identical post-deployment cohort with a paired whole-card bootstrap and
separate filtered CLV. It remains paper-only until sample, positive-return,
positive-CLV, and improvement-over-base gates all pass.

Rebuild this challenger without network access using:

```console
python -B src/rebuild_bayesian_challenger.py
```

## External promotion history

The repository now has a source-attributed external-MMA collection layer for
Bellator, ONE, PFL, and regional histories. External bouts are replayed only as
fighter state: they may update overall Elo, record, activity, opponent strength,
and recency, but they can never become UFC training labels. Missing external
strike/takedown/control statistics remain unknown and do not become zeroes.

The preferred current-data path is a licensed export or API agreement with
[Combat Registry](https://www.combatreg.com/about/), the ABC's official MMA
record keeper. The checked-in bootstrap is the publisher-declared CC0
[All Pro MMA Fights](https://www.kaggle.com/datasets/binduvr/pro-mma-fights/versions/1)
snapshot: 10,448 UFC/Bellator/ONE bouts through August 11, 2021. It is valuable
for historical evaluation, not a current weekly feed. UFC rows in that snapshot
are excluded from replay and used only to crosswalk 1,983 external identities
from an exact date-and-matchup witness. Tapology and Sherdog are explicitly
blocked as scraper targets by the source registry unless written permission is
obtained; ONE's official site is reference-only under its current terms.

The importer preserves the source license, retrieval time, payload SHA-256,
stable source IDs, rejected-row audit, and canonical orientation. Raw provider
files can be retained content-addressed in private local storage without being
redistributed. The normalized ledger contains 10,448 bouts, and the candidate
state replay contains 4,236 non-UFC bouts. Its chronological A/B test left the
UFC label set unchanged and improved final-holdout log loss from `0.62284` to
`0.62121` (accuracy `64.82%` to `65.43%`); multi-year walk-forward log loss
improved from `0.63133` to `0.63071`. This is a small predictive gain, not
evidence of positive betting return.

The fighter website publishes the 925 non-UFC bouts that can be linked safely
to a UFCStats identity, covering 377 profiles. Each history row identifies its
promotion, event, dataset, and upstream source page. Bellator/ONE result,
method, round, and clock metadata contribute to the all-promotion record;
detailed striking/grappling rates remain explicitly UFCStats-only. The site
labels the 1,045 linked fighter perspectives with unavailable detailed stats
as metadata-only rather than showing fabricated zeroes.

Reproduce the collection and evaluation with:

```console
python -B src/collect_external_mma.py sources
python -B src/collect_external_mma.py import-kaggle path/to/pro_mma_fights.csv
python -B src/collect_external_mma.py crosswalk
python -B src/collect_external_mma.py build-auxiliary
python -B src/collect_external_mma.py validate
python -B src/evaluate_external_mma.py
```

Production use is deliberately controlled by
`src/content/data/external_mma/model_policy.json`. It remains disabled until an
approved model publication pins the exact auxiliary CSV hash, preventing a data
refresh from silently changing weekly forecasts. A future Combat Registry
export can use `import-canonical --source-key combat_registry_export
--license-confirmed`; its required columns and validation rules are enforced by
the adapter.

The auditable artifacts are:

- `src/content/data/processed/ufc_fights_point_in_time.csv`
- `src/content/data/processed/external_mma_auxiliary_doubled.csv`
- `src/content/data/external_mma/bouts.jsonl`
- `src/content/data/external_mma/snapshots.jsonl`
- `src/content/data/external_mma/evaluation_report.json`
- `src/content/data/external/winner_model.json`
- `src/content/data/external/bayesian_winner_challenger.json` (paper-only Laplace posterior)
- `src/content/data/external/vegas_odds.json`
- `src/content/data/external/fighter_explorer.json` (searchable career/profile index)
- `src/content/data/external/fighter_fights_*.json` (lazy-loaded complete fight logs)
- `src/content/data/market/quote_snapshots.jsonl`
- `src/content/data/market/quote_source_metadata.jsonl`
- `src/content/data/market/total_round_quote_snapshots.jsonl`
- `src/content/data/market/total_round_forecast_captures.jsonl`
- `src/content/data/market/total_round_paper_decisions.jsonl`
- `src/content/data/market/total_round_paper_settlements.jsonl`
- `src/content/data/market/paper_decisions.jsonl`
- `src/content/data/market/bayesian_filtered_paper_decisions.jsonl`
- `src/content/data/market/paper_settlements.jsonl`
- `src/content/data/market/performance_report.json`
- `src/content/data/external/outcome_model_evaluation.json` (candidate only)

The website treats fighter history as its primary research surface. When complete two-sided lines are available, it shows the no-vig market consensus and the independent model probability as separate supporting context. Betting recommendations are disabled until timestamped rolling tests demonstrate a repeatable market-relative edge.

## Expected-return research

Expected return is evaluated against sportsbook prices, not inferred from winner
accuracy. A separate, paper-only market tracker now stores each retrieval as an
immutable multi-book snapshot with stable event/fighter IDs, a fresh UTC
observation time, first-seen time, source-payload hash, and the exact frozen
model probability/model ID. Consensus probabilities never mix retrieval runs.
Native API captures also preserve the source event/book keys, the sportsbook's
own quote-update timestamp, scheduled commence time, and source-quote age.
When one book's offered price is evaluated, that book is excluded from the
consensus and at least three other books are required.
If more than one capture exists for a matchup, evaluation requires a
predeclared per-event UTC cutoff and deterministically selects the latest
capture at or before it; later snapshots cannot be chosen using outcomes.

The same API response now requests full-fight total-round markets alongside
moneylines. Complete Over/Under pairs are written to a separate immutable
ledger with the round line, both prices, no-vig probability, first-seen time,
source update time, stable matchup identity, and payload hash. Missing or
one-sided totals are omitted; they never alter the moneyline ledger.

A candidate discrete-time competing-risk model jointly predicts
fighter/opponent win by KO/TKO, submission, decision, or other outcome and a
coherent survival curve for round totals. On the untouched September
2024–August 2026 holdout its winner log loss was `0.6243`; joint-outcome log
loss was `1.6157` versus a `1.7200` development base-rate benchmark.
Method-only log loss was `1.0261` versus `1.0271`, while Over 1.5 and Over 2.5
improved their base-rate log loss by only about `0.009` each. These small gains
do not establish positive EV, so the model remains candidate-only. The weekly
update now freezes upcoming method and duration probabilities, and every odds
capture freezes the matching total-round probability in a separate immutable
forecast ledger. The website ranks positive estimated-return total prices with
the exact book, line, model probability, break-even probability, and price
timestamp, but labels them paper-only candidate signals rather than validated
recommendations. Method-of-victory probabilities are visible, but method EV is
not calculated without a real book-specific method price; the configured API
currently documents UFC winner and fight-total coverage, not method markets.
Reproduce the evaluation report with
`python -B src/evaluate_outcome_model.py`.

For a quoted side, candidate prop EV is calculated as
`model probability / offered break-even probability - 1`. Positive values are
shown in the combined market opportunity list; the existing 5% paper threshold
is also shown separately so a barely positive estimate is not mistaken for a
strong signal. No staking or wager execution is enabled.

Totals now also have a separate prospective T-24 decision and settlement
loop. For each matchup/line, it selects one target book, excludes that book
from a median consensus of at least two other fresh books, and freezes the
market, independent-model, and market-residual probabilities. The residual is
`logit(market) + weight * (logit(model) - logit(market))`. Its weight stays at
zero until at least 100 settled lines across 10 events exist and an event-block
bootstrap on the later 30% of cards shows at least a 0.002 log-loss improvement
over market-only after selecting the weight on the earlier 70%. The
predeclared 0%, 2.5%, 5%, 7.5%, and 10% EV thresholds are all reported in
shadow mode; the official paper policy remains the 5% residual threshold.
Settlements use UFCStats total fight time, void non-W/L outcomes and exact
line-boundary finishes, and publish same-book/same-line closing-price movement,
ROI, drawdown, calibration, and model-versus-market comparisons. These are
paper records, not executable wagers.

The prospective policy freezes at most one paper decision per matchup in a
predeclared T-24 window (20 to 28 hours before the card). It uses only source
quotes updated within 30 minutes, excludes the evaluated book from a consensus
of at least three other books, applies a fixed 5% expected-return threshold,
and currently locks the stats/market blend weight to zero (market-only). The
weekly updater later settles these immutable records from stable UFCStats IDs
and publishes paper ROI, drawdown, forecast scores, coverage, and a latest-
available same-book CLV proxy. This remains research only; no order, account,
bankroll, or wager-execution code exists.

The Bayesian-filtered moneyline challenger is evaluated only on new immutable
T-24 decisions made after its deployment; older paper decisions are not
retroactively labeled. Its promotion gate requires at least 500 paired settled
decisions across 40 events and 100 surviving selections, then requires the
95% whole-card bootstrap lower bounds for filtered return, filtered CLV, and
filtered-minus-base ROI to be positive. These requirements are deliberately
stricter than merely observing a higher point-estimate ROI.

A separate timing challenger now tests the handicapper rule "favorites early,
underdogs late" without enabling bets. The favorite is frozen from the first
eligible no-vig consensus and is never reclassified after a line flip. Three
predeclared shadow policies use the same 5% expected-return and leave-one-book-
out rules: first available for either side, fixed T-24, and an early favorite
with a late underdog fallback. Early is 32-144 hours before the card, T-24 is
20-28 hours, and late is 1-5 hours. The bounded performance report compares
best-price and same-book movement, latest-price CLV, flat-unit hypothetical
ROI, and deterministic whole-card bootstrap intervals. Capture/price selection
never uses outcomes, and all execution remains disabled.

The conservative Git-history reconstruction found 503 completed W/L fights
with at least three of the five core books. Only 230 also had a defensible
same-capture legacy model forecast, and 119 remained after prior-card warmup.
On those 119 fights, market-only log loss was `0.58713`, the legacy model was
`0.61856`, and the prior-card-selected blend was `0.58836`. The blend did not
beat the market.

A separate replay now compares the **current** 82-feature algorithm with the
market on all 503 recovered W/L fights. Each model probability is a nested,
whole-year out-of-fold prediction trained only on prior years; the fully fitted
current artifact is never applied backward. Consensus still won: market log
loss was `0.60152` versus `0.62356` for the model (accuracy `67.79%` versus
`65.61%`). After the 12-card/100-fight warmup, a prior-card-selected logit blend
scored `0.59926` versus `0.59937` for market-only on 399 fights. That apparent
`-0.00011` improvement is negligible and its event-block 95% interval
`[-0.00060, +0.00040]` crosses zero. The selected model weight was zero on 278
of 399 fights and never exceeded 10%.

A frozen style-matchup challenger now adds 30 causal features without changing
the production contract: head/body/leg and distance/clinch/ground attempt-share
differences, plus antisymmetric striking, takedown, control, power, and
offense-versus-vulnerability interactions. Across all 1,857 walk-forward fights
in the four market years, log loss moved from `0.63632` to `0.63590`; the
event-block 95% interval for the `-0.00042` change was
`[-0.00336, +0.00247]`. On the 503 market-matched fights, the challenger
improved the baseline from `0.62356` to `0.61838`, but consensus remained much
better at `0.60152`. The style/market blend scored `0.59891` on the 399 warmed-up
fights versus `0.59926` for the baseline blend and `0.59937` for market-only.
Neither paired interval excluded zero. This is a useful challenger signal, not
a promotion result; the weekly artifact remains the validated 82-feature model
and betting remains disabled.

This replay closes the old-model comparability gap, but remains development
evidence rather than a historical forecast record: feature engineering used
some evaluation-era outcomes, current reconciled source/profile corrections
can postdate a fight, Git timestamps are not provider quote timestamps, and
2024 odds are absent.

The locked exploratory paper rule required at least 5% predicted EV, used the
best listed core-book price, risked a hypothetical flat 1 unit, and never used
the target book in its own probability. It made 26 selections and finished
3-23, `-15.47u` (`-59.5%` hypothetical ROI). A fixed market-only comparator
using the identical fights, prices, and 5% rule made just 7 selections and lost
all 7 (`-7u`). This is not an executable return:
legacy commit times only bound when a quote was saved, availability was not
verified, 2024 is absent, and the sample is small. It is evidence to keep
betting disabled and collect prospective data, not a reason to loosen the
threshold or optimize the backtest.

Reproduce the read-only reconstruction or regenerate its content-addressed
outputs with:

```console
python -B src/backfill_market_history.py --dry-run
python -B src/backfill_market_history.py
python -B src/evaluate_current_model_vs_market.py --dry-run
python -B src/evaluate_current_model_vs_market.py
python -B src/evaluate_style_matchup_challenger.py --dry-run
python -B src/evaluate_style_matchup_challenger.py
```

The audit and ledgers are in `src/content/data/market_history_backfill/`.

## Evidence-first fight simulation research

The event-sourced Monte Carlo simulator is an independent, candidate-only
research challenger. It does not replace or blend with the production winner
model, place wagers, or expose a public arbitrary-matchup service. The website
may display a precomputed, read-only upcoming-card research publication, but
that publication is explicitly paper-only and has no production influence.
Its data, causal-fitting, deterministic RNG/replay, evaluation, and promotion
contracts are documented in
[SIMULATION_ARCHITECTURE.md](SIMULATION_ARCHITECTURE.md).

### Local CLI

The package lives below `src/`, so set the import path once when working from
the repository root. The research commands then share one interface:

```bash
export PYTHONPATH="$PWD/src"

python -m fight_sim backfill --help
python -m fight_sim fit --help
python -m fight_sim backtest --help
python -m fight_sim posterior-backtest --help
python -m fight_sim derive-mechanics --help
python -m fight_sim select-mechanics --help
python -m fight_sim validate-mechanics --help
python -m fight_sim select-finishing --help
python -m fight_sim validate-finishing --help
python -m fight_sim upcoming-card --help
python -m fight_sim run --help
python -m fight_sim replay --help
python -m fight_sim reduce --help
python -m fight_sim diff --help
python -m fight_sim analyze --help
python -m fight_sim validate-fight --help
python -m fight_sim gui --help
```

For example, these bounded commands checkpoint at most 25 UFCStats fight pages,
fit the current 200-member event-card bootstrap ensemble, and run a tractable
64-member historical screen with two worker processes:

```bash
mkdir -p artifacts/simulations/local
python -m fight_sim backfill \
  --max-fights 25 \
  --checkpoint-every 5 \
  --summary-output artifacts/simulations/local/round-backfill-summary.json
python -m fight_sim fit \
  --bootstrap-members 200 \
  --output artifacts/simulations/local/parameter_model.json.gz
python -m fight_sim backtest \
  --bootstrap-members 64 \
  --paths-per-matchup 4096 \
  --seed-repeats 2 \
  --first-test-year 2017 \
  --last-test-year 2026 \
  --max-fights 200 \
  --stack-min-training-fights 100 \
  --stack-l2-penalty 0.01 \
  --workers 2 \
  --chunk-size 64 \
  --output artifacts/simulations/local/backtest_report.json
```

The V1 parameter fitter uses strongly pooled empirical sufficient-statistic
ratios inside each event-card bootstrap member. It is not a simultaneous
opponent-adjusted offense/defense regression: accuracy comes from a fighter's
prior landed/attempted pairs, while defense comes from prior opponent
landed/attempted pairs. Head/body/leg values are landed-target composition, and
distance/clinch/ground values are strike-attempt composition rather than time
spent in each phase. They may scale strike hazards but never divide or inflate
whole-fight takedown/submission opportunity rates. Pace decay compares pooled
first and later rounds for each fighter; later-round observations are
survivor-selected, and V1 does not claim to correct that selection.

`--paths-per-matchup` is the total nested path count per independent seed and
must be divisible by the bootstrap-member count. Backtests use two independent
inner-process seeds by default, retain the first forecast as authoritative, and
hash their log-loss variation into `simulation_noise`. When the causal
production winner baseline is available, the backtest also evaluates a
zero-intercept, nonnegative logit stack of model and simulator probabilities.
Every test-year weight pair uses only earlier out-of-fold fights. The first 100
jointly covered fights are warmup, and fixed L2 regularization of 0.01 shrinks
unsupported simulation signal toward incumbent-only weights `(1, 0)`. Reports
include per-fold weights, same-fight log loss/Brier/calibration, event-card
paired intervals, and independent-seed stack stability. This remains candidate
research and cannot alter a production probability. For the exact default
repository study, a competing-risk, population, or division joint paired
interval that crosses zero automatically triggers one nonrecursive
16,384-path rerun using the same fitted fold artifacts.
`--skip-borderline-rerun` disables that escalation for a bounded diagnostic.
The default 2017--2026/200-fight command is a deliberately bounded screen, not
a claim of complete incumbent-horizon coverage. The backfill command updates
the normalized round CSV and reconciliation report in place; its checkpointing
is durable when run locally. Parameter fits accept only rows explicitly
reconciled as `matched` by default and cross-check their
bout/event/fighter/opponent identities against the causal doubled table;
unlabeled legacy rows are excluded. The backtest omits its detailed JSONL ledger
unless `--ledger-output` is supplied explicitly.

`run` writes `specs.json` and `convergence.json` before evaluating its gates.
When the gates fail it exits with status 3 and withholds the aggregate, traces,
and HTML report. `--allow-nonconverged-research` is available only for an
explicitly labeled local diagnostic; such output is not eligible for a shadow
publication. `replay`, `reduce`, and `diff` verify stored or regenerated event
streams, while `analyze` rebuilds a self-contained local report.

`validate-fight` compares one completed run with the mirrored official totals
for a physical fight and writes `validation.json` plus a self-contained
dark-mode `validation.html` beside the run. For every statistic available in
both the simulator and UFCStats it reports the observed value, simulated mean
and central intervals, discrete probability-integral-transform percentile,
inclusive two-sided predictive tail probability, standardized residual,
predictive point mass, and CRPS. Attempt-by-phase telemetry distinguishes
distance, clinch, and ground significant strikes. UFCStats does not distinguish
standing punches from kicks, and its control field is broader than the
simulator's ground top-position clock; both limitations are labeled rather
than silently treated as exact matches.

```bash
python -m fight_sim validate-fight \
  artifacts/simulations/<run-directory> \
  --fight-id <ufcstats-fight-id>
```

The single-fight report scores marginal distributions. It does not multiply
them into a false joint likelihood. Population parameter tuning must aggregate
strictly out-of-sample CRPS, interval coverage, and PIT calibration across
chronological fights rather than optimize one showcase bout.

`posterior-backtest` performs that population check on the newest complete
event cards. Every card gets its own pre-event parameter refit, so neither the
fight being scored nor a same-card fight enters its fighter snapshots or global
parameters. The primary cohort requires both fighters to have at least three
strictly prior UFCStats bouts; debuts and fighters with only one or two prior
UFC bouts are excluded rather than filled in from current career summaries.
The report repeats all metrics for the higher-information subset where both
fighters have at least five prior bouts. Exact path counts remain authoritative,
two independent seeds quantify inner Monte Carlo noise, and the output includes
a dark self-contained HTML report plus compressed local ledgers.
`--skip-latest-events` selects an earlier whole-card window, making it possible
to reserve intermediate selection cards and a final untouched holdout.

```bash
python -m fight_sim posterior-backtest \
  --last-events 20 \
  --min-prior-ufc-fights 3 \
  --bootstrap-members 64 \
  --paths-per-matchup 4096 \
  --seed-repeats 2 \
  --workers 4 \
  --output-dir artifacts/simulations/posterior-recent-20
```

Use the explicit fidelity presets while searching mechanics. They preserve the
same causal cutoffs and deterministic seeds, but avoid spending final-run
precision on candidates that will be discarded:

```bash
# Stage 1: screen all predeclared candidates on five development cards.
python -m fight_sim posterior-backtest \
  --quick-screen \
  --skip-latest-events 5 \
  --workers 8 \
  --output-dir artifacts/simulations/screens/candidate-name

# Stage 2: rerun only survivors at intermediate precision.
python -m fight_sim posterior-backtest \
  --confirmation-screen \
  --skip-latest-events 5 \
  --workers 8 \
  --output-dir artifacts/simulations/confirm/candidate-name
```

`--quick-screen` uses 5 cards, 16 bootstrap members, 512 total paths per
matchup, and one seed. `--confirmation-screen` uses 15 cards, 32 members, 2,048
paths, and one seed. Both outputs are labeled screening-only and cannot stand
in for the two-seed final evaluation. Candidates should share the same
`--random-seed` so their paired Monte Carlo noise is reduced by common random
numbers.

Every fight/seed pair is saved immediately beneath the ignored output
directory. Re-run the identical command with `--resume` after an interruption;
the run-contract hash rejects changed data or scientific settings. Materialized
event-cutoff fits are shared by default under
`artifacts/simulations/causal-fit-cache`, so mechanics candidates do not repeat
the same bootstrap fit. Use `--no-fit-cache` only for cache diagnostics. The
summary reports input fingerprint, fit/load, simulation, checkpoint, and cache
hit timings separately.

Measure a fixed run specification before and after performance changes with:

```bash
python -m fight_sim benchmark path/to/specs.json \
  --paths-per-member 256 \
  --workers 1,2,4,8 \
  --repeats 3 \
  --output artifacts/simulations/benchmark.json
```

The benchmark hard-fails if the aggregate changes with worker count and also
reports whether a local Numba or C++ prototype toolchain is available.

This command is local, candidate-only research. Its low-exposure exclusions are
reported per card, its nominal PIT uniformity p-values are labeled as
exploratory because they do not correct card clustering or multiple testing,
and it cannot change production predictions or betting decisions.

`derive-mechanics` estimates global observable-action corrections on development
cards only. `select-mechanics` applies predeclared winner/method/duration
preservation gates on an intermediate chronological window before choosing the
lowest observable moment error. `validate-mechanics` retains or rejects that
choice on the newest untouched cards. `select-finishing` and
`validate-finishing` repeat that selection/untouched-validation boundary for
global finish conversion only; a rejected finish candidate falls back to the
already validated action profile, not an unreviewed alternative.

The August 2026 study first simulated the newest 20 completed event cards with
both fighters required to have at least three strictly prior UFCStats bouts:
133 of 248 fights were eligible, and 272,384 total paths were retained. The
oldest ten cards supplied action-volume moments, the next five selected the
candidate, and the newest five were left untouched until final validation.
The retained action profile multiplied distance, clinch, and ground strike
hazards by 1.5344, 0.5399, and 0.7602; takedown and submission-attempt hazards
by 1.8891 and 5.2282; and knockdown conversion by 0.8007. A separately
predeclared screen on the middle five cards selected a 0.40 global KO/TKO
finish-after-knockdown multiplier.

On the final untouched five cards (31 eligible fights, 63,488 paths), adding
that finish adjustment improved joint side-by-method log loss from 2.0162 to
1.9509, method log loss from 1.2787 to 1.2178, winner log loss from 0.7534 to
0.7383, duration CRPS from 268.2 to 255.9 seconds, and the six-family observable
action error from 0.3213 to 0.2893. Mean predicted-minus-observed duration moved
from -87.4 to +10.0 seconds. These are useful held-out improvements over the
previous simulator configuration, not evidence that the simulator beats the
production winner model or warrants a wager. The 31-fight winner calibration
intercept (0.208) and slope (0.206), underpredicted UFCStats control time, and
wide parameter intervals remain explicit warnings and reasons to keep the
result candidate-only.

`upcoming-card` then fits or re-materializes one 200-member pre-event ensemble,
withholds any bout where either fighter has fewer than three prior UFCStats
bouts, stores every completed aggregate and convergence diagnostic under the
ignored run directory, and atomically writes the much smaller
`src/content/data/external/simulation_forecasts.json` website projection. A
maximum-path run that misses any convergence gate is withheld even though its
full aggregate remains available locally for diagnosis. `--parameter-artifact`
pins the exact fitted inputs and members for a rerun. The first recipe load
reconstructs those members and writes an ignored content-addressed materialized
cache; later card/candidate runs validate both commitments and load the member
columns directly. A newly fitted card also writes that cache immediately. Every
website object carries `candidate_only`, `paper_only`, `execution_enabled:
false`, and `production_influence: "none"`.

### Local simulation desktop explorer

The optional Qt desktop explorer reads a completed run directory directly. It
does not start a server, change website files, or require a paid service. Its
tabs cover winner/outcome summaries, every exact statistic PMF (including
duration, phase-specific strike attempts/landings, control, takedowns,
submissions, and knockdowns), totals and method-by-round markets, convergence
and uncertainty, plus selected single-path timelines and their event ledgers.
When `validation.json` is beside the aggregate, observed values and predictive
percentiles are overlaid automatically. Every plot has pan, zoom, reset, and
PNG export controls.

Install the optional desktop packages once, then open any completed run:

```powershell
python -m pip install -r requirements-gui.txt
$env:PYTHONPATH = "src"
python -m fight_sim gui artifacts/simulations/<run-directory>
```

To run a matchup and open it immediately, add `--launch-gui` to the normal
`fight_sim run` command. Keep `--max-traces` above zero (up to 32) when you want
the individual Monte Carlo trace explorer; aggregate plots do not depend on
full trace capture.

```powershell
python -m fight_sim run `
  --red-fighter-id <stable-id> `
  --blue-fighter-id <stable-id> `
  --division Lightweight `
  --max-traces 32 `
  --launch-gui
```

High-volume nested runs stream exact counters instead of retaining every path.
Detailed ledgers, traces, and local HTML reports belong under the ignored
`artifacts/simulations/` tree. Full shadow aggregates, including exact
per-bootstrap statistic histograms, are content-addressed gzip files below
`artifacts/simulations/shadow-authority/<event>/`. The immutable shadow
`compact_shadow_v1` projection omits only those large member histograms. The
website projection is narrower still: it keeps the overall outcome/duration
counts, survival and totals views, statistic summaries, winner uncertainty,
an omitted-field manifest, and the full aggregate SHA-256, while omitting
per-bootstrap outcome counts and statistic histograms that the browser never
uses. Locally that hash identifies the ignored authority file. On an ephemeral
Actions runner the file is not uploaded, so the hash is a deterministic replay
commitment. A compact
parameter or evaluation artifact is not production evidence merely because it
was generated successfully; failed or inferior evaluations remain valid
research results and never alter the weekly forecast.

### Frozen research bundle and shadow gate

The reviewed repository contract uses these exact paths:

```text
src/content/data/processed/ufc_fight_round_stats_doubled.csv
src/content/data/simulation/
  parameter_model.json.gz
  backtest_report.json
  research_status.json
  shadow_forecasts/<date>_<event>_<publication_sha256>.json
```

`parameter_model.json.gz` is a content-hashed
`ParameterEnsembleArtifact`. `backtest_report.json` is a content-hashed
`BacktestReport` with `candidate_only: true`, `production_enabled: false`, and
`execution_enabled: false`. Merely creating either file enables nothing. The
separately reviewed `research_status.json` must name both exact hashes and has
this schema:

```json
{
  "schema_version": 1,
  "candidate_only": true,
  "paper_only": true,
  "production_enabled": false,
  "execution_enabled": false,
  "integrity_gate_passed": true,
  "causal_backtest_gate_passed": true,
  "shadow_enabled": false,
  "parameter_artifact_sha256": "<64 lowercase hex characters>",
  "backtest_report_sha256": "<64 lowercase hex characters>"
}
```

The CLI and research workflow never create or change that status file. The two
gate fields mean the engine/data invariants and strictly causal evaluation were
reviewed; they do not claim that the simulator beat an incumbent. An enabled
status is also checked against measured evidence: exactly 200 ensemble members
using exclusively matched round rows; at least three chronological folds and
200 scored fights; at least two independently hashed seeds with 4,096 paths per
matchup; 99% population/division joint coverage; 90% production-winner and
competing-risk-joint coverage; and at least one timestamp-aligned moneyline and
full-fight-total comparison. After placing a reviewed bundle, validate the round
table, hashes, schemas, and status cross-links with:

```bash
python -B src/validate_data.py --allow-stale --require-simulation-artifact
```

No frozen bundle is checked in at present, so the dependent weekly shadow job
currently exits without generating a simulation shadow. A later explicit
review may set `shadow_enabled` to `true`. The production update is validated,
committed, and pushed before `simulation_shadow` starts. That separate job
checks out the exact SHA exported by the successful production job, uses only
the named frozen pair, appends a content-addressed card publication after every
matchup converges, and stages only immutable shadow JSON. It records that exact
source revision and refuses to push if the publication branch advances while it
runs. The shadow job has its own concurrency group, so its three-hour ceiling
does not retain the lock shared by the production updater and scheduled market
captures. A timeout, nonconvergence, or shadow failure therefore cannot prevent
or roll back the production update. Every shadow is marked
`candidate_only: true`, `paper_only: true`, `execution_enabled: false`, and
`production_influence: "none"`. This gate cannot blend probabilities, change
betting decisions, or satisfy the separate prospective production-promotion
gate.

### Manual no-extra-service workflow

The `simulation-research` GitHub Actions workflow is manual-only and uses a
standard `ubuntu-24.04` runner already used by this public repository. Its four
modes are tests, a bounded round-source check, parameter fitting, and a bounded
chronological backtest. The workflow hard-caps a backfill at 100 fights, a fit
at 200 bootstrap members, and a backtest at 64 members, 16,384 paths per
matchup, 500 matchups, and two worker processes. The six-hour job timeout
is a safety ceiling, not a guarantee that the largest permitted request will
finish. It has no schedule, large or self-hosted runner, separately provisioned
cloud runtime, database, object store, paid service, odds credential, push
permission, production publication step, or website step.

Workflow results remain inside the ephemeral runner by default. The explicit
`upload_compact_artifact` input may upload exactly one schema-checked parameter
artifact or compact summary, only after a 5 MiB hard cap passes, with three-day
retention. Parameter fits use a versioned self-contained gzip encoding; the
logical model hash excludes the creation timestamp. Fitted artifacts store
compact causal input frames plus the fit/bootstrap recipe and logical metadata,
rather than repeating every expanded member value. Loading reruns that frozen
recipe without consulting mutable repository files, must reproduce the exact
members and logical metadata, and validates the original content hash. A sealed
storage/input/member commitment wrapper lets the weekly pre-commit validator
and upload staging inspect integrity without running the fit; required research
validation and simulation execution still materialize every member. Legacy
row-oriented JSON and exact-columnar artifacts remain readable. The current
repository's measured 200-member gzip is under 1 MiB, so the workflow can
transfer the validated frozen artifact when the 5 MiB check passes. The
workflow never uploads detailed ledgers, trace populations, HTML
reports, the accumulated round table, or any production forecast. A workflow
round backfill is therefore useful for a bounded source/contract check; run the
same command locally when the accumulated CSV must persist across batches.
Review and commit any validated frozen bundle separately. An Actions upload is
only a short-lived transfer, not approval for
`research_status.json`; this workflow never commits, pushes, enables shadows,
or changes production.

## Running scripts locally from VS Code (Bash)

In VS Code, open **Terminal -> New Terminal** and select **Git Bash**.

### One-time setup

```bash
cd /c/Users/Alex/Projects/GitHub/UFC_Prediction_2022
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

### Update UFC data now

Copy this entire block into Git Bash. It prompts for the Odds API key without
showing or saving it. The GitHub repository secret is not available locally.

```bash
cd /c/Users/Alex/Projects/GitHub/UFC_Prediction_2022
source .venv/Scripts/activate

read -rsp "The Odds API key: " THE_ODDS_API_KEY && echo
export THE_ODDS_API_KEY
export MARKET_ODDS_SOURCE="the-odds-api"
export ODDS_API_REGIONS="us,us2"

python -B src/update_and_rebuild_model.py
python -B src/update_market_performance.py
python -B src/validate_data.py --require-model-artifact --require-market-data
```

The first Python command is the actual data/model updater. The second settles
paper results, and the third verifies the completed update. These commands
modify generated data files and retrieve sportsbook odds.

After it succeeds, inspect the files that changed:

```bash
git status --short
git diff --stat
```

### Test or validate without updating

```bash
python -B -m unittest discover -s tests -v
python -B src/validate_data.py --allow-stale
node --check script.js
```

### Capture one market snapshot manually

After setting the API variables shown above, run:

```bash
python -B src/capture_market_snapshot.py
python -B src/update_market_performance.py
python -B src/capture_market_snapshot.py --validate-only
```

This consumes Odds API credits and appends a new timestamped observation. It
does not place a wager.

From PowerShell, the equivalent current-card refresh is:

```powershell
cd C:\Users\Alex\Projects\GitHub\UFC_Prediction_2022
.\.venv\Scripts\Activate.ps1

$secureKey = Read-Host "The Odds API key" -AsSecureString
$env:THE_ODDS_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
$env:MARKET_ODDS_SOURCE = "the-odds-api"
$env:ODDS_API_REGIONS = "us,us2"

python -B src/capture_market_snapshot.py
python -B src/update_market_performance.py
python -B src/capture_market_snapshot.py --validate-only
python -B src/validate_data.py --require-model-artifact --require-market-data

Remove-Item Env:THE_ODDS_API_KEY
```

The refresh updates the local market ledgers and website publication. It uses
API credits and does not update the hosted site until the generated files are
committed and pushed. Do not paste the API key into an issue, commit, or chat.

### Preview the website locally

```bash
python -m http.server 8000
```

Open `http://localhost:8000`; press `Ctrl+C` to stop the server. For less common
research/import commands, run the relevant script with `--help` or use the
commands in the research section above.

## Weekly automation

`.github/workflows/update-data.yml` runs Sunday at 9:33 AM and 8:33 PM and
Wednesday at 9:33 PM America/Chicago, and can also be started manually. Sunday
morning publishes the next card after the prior event, Sunday evening retries
when UFCStats was not ready, and Wednesday is the bounded midweek refresh
before T-24. It uses pinned dependencies, tests before mutation, strict
post-build validation, a shallow checkout, scoped staging, a no-op commit
guard, and a starting-commit check so artifacts built from stale code are
never rebased onto newer code.

`.github/workflows/collect-market-snapshot.yml` runs separately Sunday at
10:17 AM and 9:17 PM; Monday at 11:17 PM; Tuesday through Thursday at 12:17 PM
and 6:17 PM; Friday at 12:17 PM, 6:17 PM, and 11:17 PM; and Saturday at 9:17 AM,
12:17 PM, 3:17 PM, and 6:17 PM (America/Chicago). Once a previously timed card
has commenced, a late retry exits successfully without spending another API
credit. Each run validates the frozen card/model publication, captures one
fresh MMA moneyline plus available full-fight total-round response from The
Odds API, appends separate validated quote/forecast/source-timing ledgers,
freezes any eligible T-24 paper decisions, and publishes a
bounded audit report, settles any newly completed moneyline and totals
decisions, and refreshes
the return/CLV report before strict revalidation. The authoritative update job
and collector share one publisher concurrency group and exact path allowlists;
the dependent paper-shadow job uses a separate group and cannot delay a price
capture. The collector creates no live wager.

The source's free Starter tier currently includes 500 request credits per
month. The configured `h2h,totals` request across `us,us2` costs up to four
credits, so the three updater runs plus the maximum fourteen scheduled captures use roughly 295 credits
in an average month (and post-commencement no-ops use none). Create a free key
at [The Odds API](https://the-odds-api.com/), then add
it to the repository under **Settings -> Secrets and variables -> Actions ->
New repository secret** with the exact name `THE_ODDS_API_KEY`. Never commit
the key. Both workflows fail early with a clear credential message when it is
missing, and the key is scoped only to the network capture step.
The provider's historical-odds endpoint is paid and is deliberately not used;
this project builds its own prospective history from the free current-odds feed.

After committing and pushing this upgrade, open **Actions -> Update UFC data** and use **Run workflow** once, followed by **Collect UFC market snapshot**, to verify the new report contract. If GitHub still marks a schedule as disabled, click **Enable workflow** once. Normal weekly runs should then require no local command.

Sportsbook prices remain optional enrichment for the authoritative model/data
update: an API outage, malformed schema, ambiguous matchup, or card mismatch
publishes model-only forecasts with an explicit status instead of discarding a
valid UFCStats/model rebuild. FightOdds remains available only as an explicit
local browser fallback; GitHub Actions does not depend on it.

After the first successful market capture, use the read-only and settlement
commands under [Capture one market snapshot manually](#capture-one-market-snapshot-manually)
to inspect its files locally.

## Evaluation and limitations

Accuracy from a random split was misleading because it mixed old and recent eras. Evaluation now uses expanding chronological folds and reports log loss, Brier score, calibration, AUC, accuracy, and coverage. The model artifact contains the exact current results; [AUDIT.md](AUDIT.md) explains the original 59.6% / 0.706 chronological failure, the corrected evaluation, data-integrity findings, and remaining work.

Historical prediction records span legacy model versions, so their aggregate accuracy is descriptive rather than a clean current-model backtest. The current website does not run a second prediction model in the browser; it reads the validated weekly artifacts and keeps model forecasts distinct from market probabilities. Follow [Preview the website locally](#preview-the-website-locally) to inspect those artifacts in a browser.
