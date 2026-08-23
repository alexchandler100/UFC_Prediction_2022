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
```

The audit and ledgers are in `src/content/data/market_history_backfill/`.

## Local setup and verification

Use Python 3.11, matching GitHub Actions:

```console
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -B -m unittest discover -s tests -v
python -B src/validate_data.py --allow-stale
node --check script.js
```

The checked-in snapshot can legitimately be stale before an update, which is why the preflight command uses `--allow-stale`.

To perform the networked update manually:

```console
cd src
python -B update_and_rebuild_model.py
python -B update_market_performance.py
cd ..
python -B src/validate_data.py --require-model-artifact --require-market-data
```

The final command verifies raw data, point-in-time lineage, model/state fingerprints, artifact dimensions, training cutoffs, and publication files together.

The old scripts `src/1-build_ufc_fights_reported_doubled.py` and `src/2-build_ufc_fights_reported_derived_doubled.py` remain for notebook compatibility. The weekly production path does not load or rebuild the 68 MB legacy derived table.

## Weekly automation

`.github/workflows/update-data.yml` runs Monday and Wednesday at 9:33 PM America/Chicago and can also be started manually. Monday publishes the next card early enough to observe fight-week movement; Wednesday is a bounded refresh/retry before T-24. It uses pinned dependencies, tests before mutation, strict post-build validation, a shallow checkout, scoped staging, a no-op commit guard, and a starting-commit check so artifacts built from stale code are never rebased onto newer code.

`.github/workflows/collect-market-snapshot.yml` runs separately Monday at
11:17 PM; Tuesday through Thursday at 12:17 PM and 6:17 PM; Friday at 12:17 PM,
6:17 PM, and 11:17 PM; and Saturday at 9:17 AM, 12:17 PM, 3:17 PM, and 6:17 PM
(America/Chicago). Once a previously timed card
has commenced, a late retry exits successfully without spending another API
credit. Each run validates the frozen card/model publication, captures one
fresh MMA moneyline plus available full-fight total-round response from The
Odds API, appends separate validated quote/forecast/source-timing ledgers,
freezes any eligible T-24 paper decisions, and publishes a
bounded audit report, settles any newly completed moneyline and totals
decisions, and refreshes
the return/CLV report before strict revalidation. The
two publishing workflows share one concurrency group and exact path allowlists.
The collector creates no live wager.

The source's free Starter tier currently includes 500 request credits per
month. The configured `h2h,totals` request across `us,us2` costs up to four
credits, so the two updater runs plus the maximum fourteen scheduled captures use roughly 280 credits
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

After the first successful market capture, its files can be checked locally
with:

```console
python -B src/capture_market_snapshot.py --validate-only
python -B src/validate_data.py --allow-stale --require-market-data
python -B src/update_market_performance.py
```

## Evaluation and limitations

Accuracy from a random split was misleading because it mixed old and recent eras. Evaluation now uses expanding chronological folds and reports log loss, Brier score, calibration, AUC, accuracy, and coverage. The model artifact contains the exact current results; [AUDIT.md](AUDIT.md) explains the original 59.6% / 0.706 chronological failure, the corrected evaluation, data-integrity findings, and remaining work.

Historical prediction records span legacy model versions, so their aggregate accuracy is descriptive rather than a clean current-model backtest. The current website does not run a second prediction model in the browser; it reads the validated weekly artifacts and keeps model forecasts distinct from market probabilities.

To inspect the website locally:

```console
python -m http.server 8000
```

Then open `http://localhost:8000`.
