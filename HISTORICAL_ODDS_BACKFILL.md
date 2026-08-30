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

A source page or chart series that fails twice is retained in the database as
unavailable, but no longer keeps every later run in a false `paused` state.
Raise `--max-source-attempts` if you deliberately want to retry those failures.
Missing series are never filled in or treated as prices.

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

Do not reuse the shorter 2023+ prediction file when the database contains
earlier years. The command above computes fresh point-in-time predictions for
every covered year. Then run the broad rolling profitability check:

```bash
python src/evaluate_rolling_moneyline_profitability.py
```

For each test year this fits only on earlier years, uses the latest 25% of the
earlier event dates to choose a value cutoff, and scores the next year. It
pools only those out-of-sample yearly results and also reports each year
separately. The default requires T-24 prices to be no more than 24 hours old.

When a causal walk-forward current-model CSV already exists, reuse it instead
of fitting the same yearly models again:

```bash
python src/evaluate_bestfightodds_history.py \
  --predictions-input src/content/data/model_research/model_family_comparison.csv
```

The reuse path validates stable identities, unique fights, dates, outcomes,
finite probabilities, and any supplied training cutoff before joining prices.
The report records the prediction file hash and covered date range.

The evaluator writes its report and fight-level CSV outside Git under
`~/.ufc-data-lab/historical-odds/bestfightodds/analysis/`. It compares the
current model, three-or-more-book market consensus, and a fixed 50/50 log-odds
blend separately at opening, T-72, T-24, T-6, and the latest price before the
event date. It also compares market movement only on the same fights. The model
is retrained using earlier years only; the result remains retrospective
research and cannot change production predictions.

After the winner-price backfill is complete, run the market-first experiment:

```bash
python src/evaluate_market_first_challenger.py
```

This starts with the three-or-more-book market probability and tests every
combination of four possible additions: disagreement with the point-in-time
UFC model, movement from the opening price, disagreement among sportsbooks,
and whether either fighter has fewer than three prior UFC fights. The earliest
60% of event dates fit the combinations, the next 20% choose one, and the
latest 20% are not examined until the final score. Results and fight-level
detail are written beside the database under `analysis/`, outside Git.

Running this command before the backfill finishes is useful for checking the
code, but its performance result is provisional and must be rerun on the
completed database. It is research-only and cannot change production odds,
predictions, or betting behavior.

## Provisional 2023-2026 research snapshot

On August 29, 2026, a read-only snapshot paired 1,643 fights across 147 events
with previously generated causal current-model probabilities. At the latest
price before the event date, log loss was:

- market: 0.58932;
- current model: 0.63011;
- fixed 50/50 log-odds blend: 0.59946.

Market-only also beat both alternatives at opening, T-72, T-24, and T-6.

The market-first experiment fit candidates on 976 T-24 fights through February
1, 2025, selected one on 309 fights from February 8 through September 27,
2025, and scored it once on 345 later fights from October 4, 2025 through June
20, 2026. The chosen adjustment used model disagreement and disagreement among
books. Its T-24 log loss was 0.58184 versus 0.58450 for market-only, a 0.00265
improvement. The whole-event 95% uncertainty interval for the difference was
-0.01456 to +0.01041, so no improvement remains a plausible explanation.

The same adjustment was worse at T-72, T-6, and the latest price. It is not a
production model or betting rule. The exact T-24 fit may be frozen for a new
future paper comparison, while market-only remains the production reference.
That snapshot has now been superseded by the broader rolling result below.

## Historical moneyline profitability check

Use the actual recorded sportsbook prices to test whether the T-24 probability
estimates would have made money:

```bash
python src/evaluate_historical_moneyline_profitability.py
```

For each offered price, the fair market probability excludes that sportsbook
and requires at least three other books. The evaluator takes no more than one
side per fight, risks one unit per selection, chooses an estimated-value cutoff
using the earlier selection period, and applies it once to the untouched later
period. It reports profit, ROI, maximum drawdown, movement toward the closing
market, and a whole-event bootstrap uncertainty interval. The default rejects
quotes more than 168 hours old. Freshness checks can be reproduced with
`--maximum-quote-age-hours 72` or `24` and separate report paths.

The August 29, 2026 provisional snapshot produced this later-period result for
the selected market-first adjustment:

| Maximum quote age | Earlier cutoff and result | Later result | Event-block 95% interval |
| --- | --- | --- | --- |
| 168 hours | 1%; 172 bets, +14.75 units (8.57%) | 194 bets, -5.27 units (-2.72%) | -14.63% to +8.62% ROI |
| 72 hours | 1%; 147 bets, +10.10 units (6.87%) | 152 bets, +1.25 units (0.82%) | -16.00% to +17.46% ROI |
| 24 hours | 2.5%; 41 bets, +4.87 units (11.87%) | 35 bets, +8.55 units (24.43%) | -8.26% to +52.75% ROI |

The 24-hour result is promising, and its average available movement toward the
later market was +1.32 probability points. It is not proof of an edge: there
were only 35 later bets across 22 events, the uncertainty interval includes a
loss, and this historical sample has already influenced research decisions.
The 72-hour result was essentially break-even and the week-old-price result was
negative. This may mean the adjustment is useful only when prices are actively
updating, or simply that stale chart prices were no longer genuinely available.

The current model and fixed 50/50 model/market blend did not find a profitable
cutoff in the earlier period and lost money at the documented 5% fallback in
all three freshness checks. Nothing from this retrospective test changes the
website, production probabilities, or betting behavior. The useful next test
is a frozen prospective paper ledger using prices captured before each event.
That ledger is now implemented as `market-first-t24-paper-v1`. It begins only
with captures at or after August 30, 2026 at 00:20:36 UTC, stores its decisions
and settlements separately under `src/content/data/market/`, and is updated by
both standard GitHub workflows. Run `python src/update_market_first_paper.py`
to update it locally or add `--validate-only` to inspect it without writing.

Historical T-24 remains relative to the source event date at 00:00 UTC, not an
exact card start time. The test also assumes access to the best recorded book,
with no stake limits, fees, slippage, or account restrictions. Reports and the
fight-level ledger remain outside Git under the `analysis/` directory described
above.

## Expanded rolling 2021-2026 result

The completed August 29, 2026 refresh generated new point-in-time model
probabilities for all covered years instead of reusing the shorter 2023+
convenience file. The main horizon comparison covers 2,610 fights across 235
events. At T-24, log loss was 0.59408 for the multi-book market, 0.63581 for
the UFC model, and 0.60515 for their fixed 50/50 log-odds blend.

The stricter rolling study required each recorded T-24 offer to be no more
than 24 hours old and required three other books for its leave-one-out fair
price. It scored 904 fights from 2022 through 2026. Each year used only prior
years for fitting and the latest 25% of the prior event dates for selecting a
betting cutoff:

| Probability source | Accuracy | Log loss |
| --- | ---: | ---: |
| Leave-one-book-out market | 66.98% | 0.59820 |
| Current UFC model | 63.83% | 0.63639 |
| Fixed 50/50 market/model blend | 67.92% | 0.60723 |
| Small market-first adjustment | 67.92% | 0.59790 |

The market-first adjustment's 0.00029 log-loss advantage over market-only is
too small to distinguish from chance: its whole-event 95% interval runs from
-0.00396 to +0.00354. The 50/50 blend's higher accuracy did not mean better
probabilities; its log-loss disadvantage was 0.00903, with a 95% interval of
+0.00047 to +0.01734.

Early folds did not contain enough qualifying prior bets to learn a stable
cutoff and therefore used the predeclared fixed 5% reference. Across all 113
market-first selections, including those fallback folds, profit was -6.66
units (-5.90%). Restricting the summary to years with enough earlier cutoff
examples left 35 bets across 27 events and was almost exactly break-even:
-0.03 units (-0.09%), with a 95% ROI interval from -35.01% to +36.31%.

This broader test does not support a profitable historical rule. It does
suggest that the small market-first probability adjustment is worth keeping
in the already-frozen prospective paper comparison, while market-only remains
the standard to beat. The reports and ledgers are stored outside Git under
`analysis/` with `rolling_` filenames.

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
