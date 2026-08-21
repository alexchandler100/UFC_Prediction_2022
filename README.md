[![Update Data](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/update-data.yml/badge.svg)](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/update-data.yml)
[![Collect Market Snapshot](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/collect-market-snapshot.yml/badge.svg)](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/collect-market-snapshot.yml)

# UFC Prediction

This project collects UFCStats fight data and publishes weekly winner forecasts at [alexchandler100.github.io/UFC_Prediction_2022](https://alexchandler100.github.io/UFC_Prediction_2022/). The repository also contains older method-prediction and browser-model experiments; the production weekly predictor is the Python point-in-time winner model described below.

## Production pipeline

The weekly job:

1. Reconciles recent completed events from UFCStats and refreshes active fighter profiles.
2. Validates source IDs, mirrored fight rows, results, card order, timing metadata, and numeric domains.
3. Replays fights in causal bout order and builds one stable-ID feature row per physical W/L fight.
4. Tunes and evaluates a regularized logistic model with nested chronological folds, adds pre-bout Elo/state features, applies symmetric temperature calibration, and refits on the full ten-year window.
5. Saves and reloads a content-hashed JSON model artifact before forecasting the next card.
6. Adds timestamped no-vig multi-book consensus from The Odds API when valid lines are available and publishes the website JSON.

The 82-feature model is antisymmetric: swapping the fighters produces the complementary probability. Historical features use only information available before that bout; appending future fights cannot change an existing training row. Split decisions are valid W/L labels, while draws and no-contests are retained as state/history events but are not binary training labels.

The regularization search includes `C=0.001` and `0.003`; an ablation found the old `0.01` lower boundary was masking stronger shrinkage. Recency weighting, decision-label weighting, and Glicko/RD were tested but remain unpromoted because their gains were small or inconsistent. Exact results are recorded in [AUDIT.md](AUDIT.md).

The auditable artifacts are:

- `src/content/data/processed/ufc_fights_point_in_time.csv`
- `src/content/data/external/winner_model.json`
- `src/content/data/external/vegas_odds.json`
- `src/content/data/market/quote_snapshots.jsonl`
- `src/content/data/market/quote_source_metadata.jsonl`
- `src/content/data/market/paper_decisions.jsonl`
- `src/content/data/market/paper_settlements.jsonl`
- `src/content/data/market/performance_report.json`

When complete two-sided lines are available, the published primary forecast is the no-vig market consensus and the independent model probability remains visible. Betting recommendations are disabled until timestamped rolling tests demonstrate a repeatable market-relative edge.

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

The prospective policy freezes at most one paper decision per matchup in a
predeclared T-24 window (20 to 28 hours before the card). It uses only source
quotes updated within 30 minutes, excludes the evaluated book from a consensus
of at least three other books, applies a fixed 5% expected-return threshold,
and currently locks the stats/market blend weight to zero (market-only). The
weekly updater later settles these immutable records from stable UFCStats IDs
and publishes paper ROI, drawdown, forecast scores, coverage, and a latest-
available same-book CLV proxy. This remains research only; no order, account,
bankroll, or wager-execution code exists.

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

`.github/workflows/update-data.yml` runs each Wednesday at 9:33 PM America/Chicago and can also be started manually. It uses pinned dependencies, tests before mutation, strict post-build validation, a shallow checkout, scoped staging, a no-op commit guard, and a starting-commit check so artifacts built from stale code are never rebased onto newer code.

`.github/workflows/collect-market-snapshot.yml` runs separately at 12:17 PM and
6:17 PM Thursday; 12:17 PM, 6:17 PM, and 11:17 PM Friday; and 9:17 AM, 12:17 PM,
3:17 PM, and 6:17 PM Saturday (America/Chicago). Once a previously timed card
has commenced, a late retry exits successfully without spending another API
credit. Each run validates the frozen card/model publication, captures one
fresh MMA moneyline response from The Odds API, appends quote/forecast/source-
timing ledgers, freezes any eligible T-24 paper decisions, and publishes a
bounded audit report, settles any newly completed prior decisions, and refreshes
the return/CLV report before strict revalidation. The
two publishing workflows share one concurrency group and exact path allowlists.
The collector creates no live wager.

The source's free Starter tier currently includes 500 request credits per
month. The configured `h2h` request across `us,us2` costs two credits, so the
normal updater plus the maximum nine scheduled captures use roughly 87 credits
in an average month (and post-commencement no-ops use none). Create a free key
at [The Odds API](https://the-odds-api.com/), then add
it to the repository under **Settings -> Secrets and variables -> Actions ->
New repository secret** with the exact name `THE_ODDS_API_KEY`. Never commit
the key. Both workflows fail early with a clear credential message when it is
missing, and the key is scoped only to the network capture step.
The provider's historical-odds endpoint is paid and is deliberately not used;
this project builds its own prospective history from the free current-odds feed.

After committing and pushing this upgrade, open **Actions -> Update UFC data** and use **Run workflow** once to verify the backlog update. If GitHub still marks the inactivity-disabled schedule as disabled, click **Enable workflow** once. Normal weekly runs should then require no local command.

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

The browser fight simulator is clearly labeled experimental and is separate from the weekly artifact. Historical prediction records also span legacy model versions, so their aggregate accuracy is descriptive rather than a clean current-model backtest.

To inspect the website locally:

```console
python -m http.server 8000
```

Then open `http://localhost:8000`.
