# Free online data research

This experiment asks whether three no-cost public sources improve the existing
winner model. It does not change production predictions.

## Sources and local setup

The source files are intentionally kept below ignored `artifacts/`; the saved
report records a SHA-256 hash for every input.

- Historical UFC rankings: [UFC Rankings History](https://www.kaggle.com/datasets/jerzyszocik/ufc-rankings-history), declared CC0 by its publisher.
- Timestamped odds: [UFC Betting Odds Daily Dataset](https://www.kaggle.com/datasets/jerzyszocik/ufc-betting-odds-daily-dataset), declared CC0 by its publisher.
- Broader MMA records: [Database Complete MMA](https://github.com/LeandroIber/Database-complete-mma), published under the repository's MIT license. The database aggregates public records and is accepted only for research after stricter row-level checks.

Place the exports at:

```text
artifacts/free_data_audit/rankings/UFC_rankings_history.csv
artifacts/free_data_audit/odds/UFC_betting_odds.csv
artifacts/free_data_audit/mmastats/dataset_global_v3.duckdb
```

Run the source checks alone:

```console
python -B src/evaluate_online_data_challengers.py --audit-only
```

Run the full bounded comparison:

```console
python -B src/evaluate_online_data_challengers.py \
  --years 2022,2023,2024,2025,2026 \
  --max-runtime-minutes 55
```

The completed run took about 7.3 minutes on the development machine. The
command refuses a runtime setting above 60 minutes.

## Validation rules

Rankings use the latest clean weekly snapshot strictly before each fight.
Ten 2026 snapshots contained conflicting duplicated lists and were discarded
as a whole. Synthetic `Top Rank` categories were also discarded. A ranking
name must resolve to exactly one UFCStats fighter ID.

The broader MMA archive advertises 129,285 fight rows. Only 25,078 survive the
strict completed-date, decisive-result, unique-master-ID, duplicate-matchup,
and known-order checks. Source IDs are linked to UFCStats only through an exact
historical UFC date and two-fighter matchup. This produced 1,889 proven fighter
links and 6,644 non-UFC bouts that can directly enrich those fighters. These
bouts update prior history only; they never create UFC training labels or
invent missing fight statistics.

Odds count only when the recorded collection date precedes the event date and
at least three distinct books are available. The archive sometimes swaps the
two fighter URLs while leaving names and prices in the correct order. Sides are
therefore accepted only when the two names agree inside the stable UFCStats
fight ID; the bad URLs are counted in the audit. This leaves 45 fights across
five 2025 events. Older rows added to the archive in 2025 are not treated as
historical pre-fight odds.

## Results

All 2,383 test fights were predicted by models trained only on earlier years.
Lower log loss is better because it evaluates the full probability, not just
which fighter was placed above 50%.

| Input | Accuracy | Log loss | Change from current | 95% range for change |
|---|---:|---:|---:|---:|
| Current 82-variable model | 64.00% | 0.63404 | — | — |
| Ranking groups chosen from earlier training fights | 63.74% | 0.63382 | -0.00022 | [-0.00185, +0.00143] |
| All seven ranking variables | 63.70% | 0.63278 | -0.00126 | [-0.00357, +0.00117] |
| Expanded non-UFC history | 63.53% | 0.63320 | -0.00084 | [-0.00450, +0.00269] |
| Expanded history plus all rankings | 64.08% | 0.63181 | -0.00223 | [-0.00604, +0.00167] |

Negative change is better. The combined source produced the largest average
gain and a tiny accuracy gain, but its uncertainty range still includes no
improvement. It improved 2022-2024 and worsened 2025-2026. Expanded history
changed at least one model input for 2,349 of 2,383 test fights and gave 1,012
formerly one-sided UFC matchups history for both fighters, so the weak result
is not explained by negligible coverage.

On the much smaller odds sample:

| Input | Accuracy | Log loss |
|---|---:|---:|
| Current model | 66.67% | 0.59454 |
| Market consensus | 71.11% | 0.57790 |
| Fixed equal-weight model/market blend | 75.56% | 0.57029 |

The blend is encouraging, but 45 fights across only five events is far too
small to establish a repeatable advantage or learn a production blend weight.

## Decision and next test

Production remains unchanged. The most useful next step is to freeze the
expanded-history-plus-rankings candidate now and evaluate it on later untouched
events. In parallel, continue collecting the repository's timestamped market
snapshots; a properly time-aligned market history is more likely to produce a
first-order improvement than the narrow downloaded odds sample. Promotion
requires the improvement to repeat on new fights, without worsening Brier
score or calibration.

Exact results and per-fight probabilities are stored in
`online_data_challengers.json` and `online_data_challengers.csv` beside this
file.

