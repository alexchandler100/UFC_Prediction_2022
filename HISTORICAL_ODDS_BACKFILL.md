# Historical odds backfill

`src/backfill_bestfightodds_history.py` downloads the timestamped price history
behind BestFightOdds' public line-movement charts. It is local research only:
it does not modify production predictions or website data.

Method-of-victory history uses a separate database and runner. Keep this
winner-price backfill running; after it finishes or is cleanly paused, follow
`METHOD_ODDS_COLLECTION.md`. The method runner refuses concurrent bulk access
by default.

## Recommended Git Bash command

Start with the stronger period, where the audit verified multi-book history:

```bash
bash scripts/backfill_historical_odds.sh \
  --from-year 2021 \
  --to-year 2026 \
  --mode both \
  --delay-seconds 1 \
  --max-runtime-hours 6 \
  --max-requests 25000
```

The command is resumable. If it reaches its request, runtime, database-size, or
free-space limit, it commits the current event and exports everything collected
so far. Run the identical command again to continue. Completed pages and chart
series are not downloaded again.

The source's public robots policy allows the paths used by the script, but its
short published terms do not explicitly address automated bulk research reuse.
Running the wrapper acknowledges that uncertainty. Seeking clarification from
the source before a large backfill remains advisable.

No subscription, API key, cloud service, or paid artifact storage is used.

## Files and limits

The default database is outside the repository:

```text
~/.ufc-data-lab/historical-odds/bestfightodds/history.sqlite3
```

Exports are written beside it under `exports/`:

- `horizon_quotes.csv`: each bookmaker and the source's mean line;
- `market_consensus.csv`: average no-vig probability when at least three
  sportsbooks are available;
- `unmatched_matchups.csv`: canceled, replaced, misspelled, or otherwise
  unlinked source listings;
- `duplicate_source_matchups.csv`: UFC fights represented by multiple source
  matchup IDs and therefore withheld from consensus;
- `summary.json`: coverage, mapping, quote, request, and disk totals.

Defaults cap the database at 1 GiB and stop before free space falls below
5 GiB. Raw HTML is never stored. SQLite keeps exact timestamped decimal prices
and the CSV exports remain comparatively small.

## Honest time labels

Historical UFCStats data has an event date but not an authoritative start time.
Therefore `safe_t72`, `safe_t24`, and `safe_t6` are measured backward from
00:00 UTC on the source event date. Every export includes:

```text
cutoff_basis=source_event_calendar_date_at_00_utc
actual_event_start_time_known=False
```

These are conservative, repeatable research cutoffs—not claims that a quote
was exactly 24 hours before the first fight. Exact T-72/T-24/T-6 values should
be re-derived later if trustworthy historical start times are added.

`opening` is the first timestamp shared by both fighter sides.
`strict_latest_before_event_date` is the last shared timestamp before the
event calendar date begins in UTC. Same-day prices are excluded from that
strict view.

## Status and exports

Check progress without contacting the source:

```bash
bash scripts/backfill_historical_odds.sh --status-only --mode both
```

Rebuild CSV exports from the existing database without downloading odds:

```bash
bash scripts/backfill_historical_odds.sh --export-only --mode both
```

Analyze a read-only snapshot without stopping an active backfill:

```bash
python src/evaluate_bestfightodds_history.py
```

The evaluator writes its report and fight-level CSV outside Git under
`~/.ufc-data-lab/historical-odds/bestfightodds/analysis/`. It compares the
current model, three-or-more-book market consensus, and a fixed 50/50 log-odds
blend separately at opening, T-72, T-24, T-6, and the latest price before the
event date. It also compares market movement only on the same fights. The model
is retrained using earlier years only; the result remains retrospective
research and cannot change production predictions.

To add the weaker older mean/single-book period after the 2021+ run, reuse the
same database and extend the start year:

```bash
bash scripts/backfill_historical_odds.sh \
  --from-year 2012 \
  --to-year 2026 \
  --mode both \
  --delay-seconds 1 \
  --max-runtime-hours 6 \
  --max-requests 25000
```

Older mean or one/two-book rows must not be described as a three-book market
consensus.

## Verified pilot

The 2021 and 2022 pilots wrote the final resumable database, not temporary test
files. Together they produced:

- 2 parsed events;
- 20 completed UFCStats-matched fights out of 25 source listings;
- 5,378 timestamped price points;
- 473 horizon rows;
- 48 multi-book consensus rows;
- approximately 1.2 MiB of SQLite data after checkpointing.

The five unmatched 2022 listings were canceled or replaced bouts. A source
misspelling of Calvin Kattar as “Calvin Cattar” is linked only because the date
and opponent agree and the name differs by one character; this constrained
fallback is recorded separately from an exact match.
