# External MMA data contract

This directory holds source-attributed, canonical fight observations. Collection
is allowlisted by `source_registry.json`; a website listed as `reference_only`
or `prohibited_without_permission` cannot be imported by the collector.

`bouts.jsonl` is the normalized append-only ledger. `snapshots.jsonl` records
retrieval time, declared license, source page, payload hash, source row count,
and accepted/rejected counts. `rejections.jsonl` is the quarantine audit.
`identity_map.csv` contains only evidence-backed, approved mappings from a
provider fighter ID to a UFCStats fighter ID. Name-only guesses do not enter the
map.

Authorized recurring provider exports use `import-canonical`. Required columns
are:

- `source_bout_id`, `source_event_id`, and `source_url`
- `event_date`, `event_name`, and `promotion`
- `fighter_source_id`, `fighter_name`, `opponent_source_id`, and `opponent_name`
- `result` (`W`, `L`, `D`, or `NC`) and `method`

Optional columns are `source_bout_order`, `division`, `finish_round`,
`finish_clock_seconds`, `scheduled_rounds`, `discipline`, and `professional`.
One input row represents one physical bout; the importer canonicalizes the
participant orientation. Stable provider IDs are mandatory—display names are
never accepted as identities.

`model_policy.json` is the production circuit breaker. The model ignores the
generated auxiliary CSV unless the policy explicitly enables it and pins its
exact SHA-256. This lets collection and backtesting proceed without silently
changing the published model.
