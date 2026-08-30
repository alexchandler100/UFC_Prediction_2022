# Expert signal research

This is a prospective paper experiment, not a recommendation feed. Advertised
handicapper profit, ROI, or win rate is not treated as evidence because the
underlying pick history, prices, deletions, and timestamps may be incomplete.

The default registry is
`src/content/data/market/expert_source_registry.json`. Every current candidate
is disabled. A candidate can be enabled only after finding a free public source
that preserves a verifiable publication time and permits the intended access.
Paid Discord picks, live bets, parlays, and screenshots without stable public
URLs are excluded from the first contract.

The disabled research list also contains several more promising free sources:
Rob Brown's third-party BetMMA record, CageSide's public model picks, FightIQ's
expert consensus, Tapology's crowd consensus, and MMA Prophecy's aggregate
streams. None is enabled merely because a website claims a good record. Exact
pre-fight timestamp behavior and permitted automated access must be verified
first; model and crowd signals will be evaluated separately from human experts.

Validate the local ledger from Git Bash:

```bash
python src/import_expert_signals.py --validate-only
```

The default ledger stays outside Git at
`~/.ufc-data-lab/expert-signals/`. An import file may be CSV or JSONL and must
contain these fields:

```text
issued_at_utc,event_date,timing_precision,event_start_utc,analyst_id,source_url,source_record_id,source_text,event_id,matchup_id,selected_fighter_id,opponent_id,selected_fighter_name,opponent_name,market,posted_moneyline
```

`event_start_utc` is required for timestamp-precision events. Otherwise the
pick must have been observed before the UTC event date. `source_text` is hashed
and not retained; `source_text_sha256` may be supplied instead. The importer
sets the observation time itself, so a post-fight import cannot be backdated.

```bash
python src/import_expert_signals.py --input /path/to/new-picks.csv
```

The append-only ledger rejects changed copies of an existing pick and always
stores `paper_only=true` and `execution_enabled=false`.

Evaluation is deferred until real prospective coverage exists. The first
report requires at least 200 picks across 20 events. Production consideration
requires at least 500 picks across 40 events, positive closing-price movement,
and better probability scores than the timestamp-aligned market alone. Passing
those checks would still require an explicit reviewed production change.
