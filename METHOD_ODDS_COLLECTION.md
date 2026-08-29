# Method-of-victory odds collection

The project now collects real book-specific prices for six primary outcomes:
each fighter by KO/TKO, submission, or decision. Every available primary price
is retained. A coherent six-way no-vig distribution is calculated only when one
book lists all six; missing prices are never inferred. Exact-round props and
decision subtypes are intentionally excluded from this first contract.

## Keep the current winner backfill running

Do not stop an active winner-price history backfill just to start this one. Its
data is still useful. The method-history job uses a separate database and
refuses to begin while the winner job reports that it is running. This prevents
two large crawls from competing for the same source or writing to the same
database.

After the winner job finishes or is cleanly paused, run this from Git Bash:

```bash
bash scripts/backfill_historical_method_odds.sh \
  --from-year 2021 \
  --to-year 2026 \
  --mode mean \
  --delay-seconds 1 \
  --max-runtime-hours 6 \
  --max-requests 25000
```

The run is resumable. Its database and exports remain outside Git under
`~/.ufc-data-lab/historical-odds/bestfightodds/`. The default `mean` mode is the
lowest-request first pass. A later `--mode both` run can add per-book history
where it exists.

Check progress without downloading anything:

```bash
bash scripts/backfill_historical_method_odds.sh --status-only --mode mean
```

## Weekly collection

The existing market workflow now runs a bounded method collector after its
moneyline/totals capture. It checks only the public current-card page and stores
at most four immutable observations per fight and book: first available, T-72,
T-24, and T-6. A failure from this optional source does not discard a healthy
moneyline/totals capture.

Validated files are small and checked into the market-data directory:

- `method_market_snapshots.csv` and `.jsonl`: append-only source observations;
- `method_capture_report.json`: counts and integrity hashes;
- `current_method_markets.json`: bounded website view with real prices beside
  the candidate method model.

The Market tab labels model-versus-price EV as an unvalidated comparison. It is
not a betting recommendation. Paper settlement remains off until the exact
book rules for unusual outcomes such as disqualifications have been verified.
