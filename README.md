[![Update Data](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/update-data.yml/badge.svg)](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/update-data.yml)

# UFC Prediction

This project collects UFCStats fight data and publishes weekly winner forecasts at [alexchandler100.github.io/UFC_Prediction_2022](https://alexchandler100.github.io/UFC_Prediction_2022/). The repository also contains older method-prediction and browser-model experiments; the production weekly predictor is the Python point-in-time winner model described below.

## Production pipeline

The weekly job:

1. Reconciles recent completed events from UFCStats and refreshes active fighter profiles.
2. Validates source IDs, mirrored fight rows, results, card order, timing metadata, and numeric domains.
3. Replays fights in causal bout order and builds one stable-ID feature row per physical W/L fight.
4. Tunes and evaluates a regularized logistic model with nested chronological folds, adds pre-bout Elo/state features, applies symmetric temperature calibration, and refits on the full ten-year window.
5. Saves and reloads a content-hashed JSON model artifact before forecasting the next card.
6. Adds timestamped no-vig FightOdds consensus when valid lines are available and publishes the website JSON.

The 82-feature model is antisymmetric: swapping the fighters produces the complementary probability. Historical features use only information available before that bout; appending future fights cannot change an existing training row. Split decisions are valid W/L labels, while draws and no-contests are retained as state/history events but are not binary training labels.

The auditable artifacts are:

- `src/content/data/processed/ufc_fights_point_in_time.csv`
- `src/content/data/external/winner_model.json`
- `src/content/data/external/vegas_odds.json`

When complete two-sided lines are available, the published primary forecast is the no-vig market consensus and the independent model probability remains visible. Betting recommendations are disabled until timestamped rolling tests demonstrate a repeatable market-relative edge.

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
cd ..
python -B src/validate_data.py --require-model-artifact
```

The final command verifies raw data, point-in-time lineage, model/state fingerprints, artifact dimensions, training cutoffs, and publication files together.

The old scripts `src/1-build_ufc_fights_reported_doubled.py` and `src/2-build_ufc_fights_reported_derived_doubled.py` remain for notebook compatibility. The weekly production path does not load or rebuild the 68 MB legacy derived table.

## Weekly automation

`.github/workflows/update-data.yml` runs each Wednesday at 9:33 PM America/Chicago and can also be started manually. It uses pinned dependencies, tests before mutation, strict post-build validation, a shallow checkout, scoped staging, a no-op commit guard, and a starting-commit check so artifacts built from stale code are never rebased onto newer code.

After committing and pushing this upgrade, open **Actions -> Update UFC data** and use **Run workflow** once to verify the backlog update. If GitHub still marks the inactivity-disabled schedule as disabled, click **Enable workflow** once. Normal weekly runs should then require no local command.

FightOdds is optional enrichment: an outage, malformed schema, ambiguous matchup, or card mismatch publishes model-only forecasts with an explicit status instead of failing the UFCStats/model update.

## Evaluation and limitations

Accuracy from a random split was misleading because it mixed old and recent eras. Evaluation now uses expanding chronological folds and reports log loss, Brier score, calibration, AUC, accuracy, and coverage. The model artifact contains the exact current results; [AUDIT.md](AUDIT.md) explains the original 59.6% / 0.706 chronological failure, the corrected evaluation, data-integrity findings, and remaining work.

The browser fight simulator is clearly labeled experimental and is separate from the weekly artifact. Historical prediction records also span legacy model versions, so their aggregate accuracy is descriptive rather than a clean current-model backtest.

To inspect the website locally:

```console
python -m http.server 8000
```

Then open `http://localhost:8000`.
