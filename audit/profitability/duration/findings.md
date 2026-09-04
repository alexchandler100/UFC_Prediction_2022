# Duration and totals profitability audit

The historical duration model learns an unrealistically durable five-round group because missing schedules are reconstructed from how fights ended. Correcting those source labels must precede any claim that the current large totals returns are real. This audit writes findings only; it does not change recommendations, models, or production ledgers.

## 1. Highest priority: schedules are inferred from outcomes

The reproduced development sample has 3,946 fights; 3,946 lack an explicit scheduled length. Of its 206 five-round fights, 163 are decisions and 43 are finishes; **0 end within the first three rounds**. 200/206 last over 3.5 rounds, reproducing the saved smoothed base rate of 96.63%.

In the later sample, only 29 fights have explicit five-round schedules and 14/29 (48.28%) last over 3.5 rounds. The inferred five-round subset instead has 36/38 (94.74%). These are differently selected groups, not a controlled estimate of the true change in UFC fight length.

Cause: [schedule inference](../../../src/fight_semantics.py#L158) assigns early finishes three rounds and late finishes five when true schedules are missing. [training](../../../src/fight_predictor/outcome_model.py#L185) and [evaluation](../../../src/fight_predictor/outcome_model.py#L379) both use this result-dependent input. [historical migration](../../../src/data_handler/data_handler.py#L366) creates blank schedules, while [fight-page extraction](../../../src/data_handler/data_handler.py#L660) already reads true schedules for new pages.

Consequence: scheduled five-round fights that finished early can be mislabeled as three rounds, biasing both groups. The live model shares duration and method predictions: 14 matchups have 56 method probabilities and 14 matchups have 44 totals probabilities requiring reevaluation. The current board has 11 totals across 9 fights, including 2 bets on five-round fights. Their separate stakes sum to 55% of bankroll. This does not measure corrected fair odds or prove every recommendation loses money.

Proposed correction: backfill scheduled lengths from independently recorded fight-page or event metadata, record its source, and exclude unresolved schedules from duration training and testing. Rebuild duration/method forecasts and their calibration only after that repair. Retain raw and calibrated estimates separately and calculate recommendations from the same probability used for stake decisions. A positive-slope-only calibration cannot move an over-50% estimate below 50%; calibration alone cannot fix missing early finishes.

Acceptance checks: an independently known five-round bout remains five rounds whether its result is an early knockout or a decision; a missing schedule plus finish round 1 remains unknown; both training and later evaluation include independently scheduled early finishes; development/holdout dates and identities stay separate; report results by schedule source and market line, with actual offered pre-fight prices before asserting profitability. The existing [unknown-schedule test](../../../tests/test_outcome_model.py#L66) omits finish round and therefore misses the real-data branch.

## 2. High priority: small later calibration checks can enable staking

The synthetic reproduction meets the current 40-fight/eight-event minimum but leaves only 2 later fights. Production returns line status **available** with check status **too_small**. [The rejection branch](../../../src/bayesian_total_calibration.py#L314) only rejects completed checks that worsen probability accuracy. Real long-total support is 3.5: 67 training fights and 14 later test fights; 4.5: 67 training fights and 14 later test fights; this is probability checking, not an odds-based demonstration of betting profit.

Proposed correction: require a complete, sufficiently supported later check before assigning positive stakes, and reassess the minimum sample after repairing schedules. The available data do not establish an optimal minimum or stake cap. Acceptance checks: incomplete/too-small checks produce unavailable sizing; available calibrations meet documented fight/event counts and later-period requirements; the corrected probabilities must beat simple and market-based comparisons before release.

## 3. High priority: the next totals archive cannot feed the replay builder

The existing archive contains 74 snapshots and its isolated replay builder result is **success**. It does not yet contain the current totals calibration policy. After the actual archiver appends the current board to a temporary copy, there are 86 snapshots and the actual replay builder returns **error**: `Bayesian Kelly assessment policy is invalid`.

This is a reachable next-capture compatibility failure, **not a claim that the saved website performance file is currently broken**. [The capture path](../../../src/capture_market_snapshot.py#L2514) archives the board; [archive conversion](../../../src/market_tracker/bankroll.py#L478) preserves totals assessments; [replay enrichment](../../../src/market_tracker/bankroll.py#L544) sends them to the moneyline validator before checking the category. The reproduction uses empty official inputs and does not run the live writer or modify any production archive.

Proposed correction: dispatch calibration validation by market/policy and preserve the exact assessment published before each fight. Acceptance checks: current totals pass archive-to-replay round-trip; historical unavailable assessments stay unavailable; delayed outcomes cannot replace a recorded stake; both current saved and newly appended archives remain valid.

## Reproduction and evidence

Run `python scripts/audit_profitability_duration.py --output-dir audit/profitability/duration` from the repository root. `summary.json` contains input/source SHA-256 hashes, split reproduction checks, counts, calibration support, current forecast identities, and both archive-builder results. `schedule_provenance.csv` summarizes groups; `schedule_fights.csv` lists every included fight; `current_forecasts.csv` and `current_total_bets.csv` identify affected published rows. No models are fit except a small synthetic calibration fixture held in memory. Production input hashes are verified unchanged before reports are written.
