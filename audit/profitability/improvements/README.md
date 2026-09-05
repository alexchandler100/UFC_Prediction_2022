# Profitability corrections implemented — September 4, 2026

The confirmed calculation and publication defects are corrected. **The saved prices currently produce zero qualifying paper bets.** This is the result of applying the stricter rules, not evidence that a profitable policy has been found. Betting execution remains disabled.

## Data and duration forecasts

- Recovered independently recorded schedules for **686 fights** from the existing round-statistics file: 1,372 raw side cells and 680 training-label cells. Every other CSV cell and the row order remained unchanged. [Repair record](schedule_repair.json)
- Duration training/evaluation now excludes unknown schedules. The repaired sample has **1,002 verified fights**, split into 792 earlier development fights and 210 later evaluation fights; 3,941 unresolved fights are excluded.
- Completed one duration-model rebuild using the existing three-value regularization check. The winner model was not retrained. New forecasts have their actual rebuild issuance timestamp; they are never attributed to an earlier odds capture.
- The five-round training group's smoothed Over-3.5 base rate changed from **96.63% to 45.45%**. Hooker–Parnasse's Over-3.5 estimate changed from **77.90% to 38.36%**. These are corrected model estimates, not known true probabilities.
- The later evaluation has only **20 five-round fights**. Five-round calibration is therefore unavailable. The repaired model also did not beat the simple five-round base-rate prediction. Totals require independent betting-performance evidence before receiving funded allocations, even when calibration becomes available.
- The new schedule contract prevents old, biased duration artifacts from generating current candidates. Legacy artifacts and captured predictions remain readable as historical evidence. Missing or insufficient verified history withholds duration output without stopping the winner updater.

## Recommendations and recorded performance

- Qualification and ranking use the same calibrated mean probability and require at least 5% expected return, a positive conservative return, and positive proposed stake. Raw estimates remain separate research fields.
- Schema 2 retains eligible offers from individual books. Changing selected books chooses an eligible accessible offer instead of merely hiding the globally best book.
- The displayed paper portfolio funds at most one selection per physical fight: maximum **1% per fight, 5% per card, and 10% across the current board**. This assumes no existing open bets; it is not an account-level outstanding-exposure monitor.
- Prices require exact source timestamps and known future event starts. The browser checks them on book selection and every minute, and rejects quotes more than 30 minutes old or timestamped in the future.
- Totals assessments now survive archive-to-settlement-to-performance validation under their own policy. New allocated stakes are preserved, and performance headlines distinguish funded bets from zero-stake records. Old snapshots are not rewritten.
- Method quotes remain visible when model estimates are unavailable. Earlier quotes cannot acquire a newly rebuilt prediction retroactively. Historical event-start revisions no longer break the method publication.
- The offline view refresh preserves original quote times and appends no forecast, price, decision, or bet ledger records. It refreshes the derived publications and their report hashes together.

## Research safeguards

Future rolling profitability evaluations select **no bets** when earlier evidence is insufficient or none of the earlier thresholds made a profit. This is a new version with separate output filenames; the original historical comparison remains intact. Imported predictions without complete training cutoffs are explicitly labeled unverified. These changes were tested without rerunning a broad strategy search.

## Reproduction and verification

```powershell
.venv/Scripts/python.exe -B src/repair_historical_schedules.py --apply --apply-pit --report audit/profitability/improvements/schedule_repair.json
.venv/Scripts/python.exe -B src/build_outcome_forecasts.py
.venv/Scripts/python.exe -B src/refresh_betting_publications.py
.venv/Scripts/python.exe -B src/build_fighter_explorer.py
.venv/Scripts/python.exe -B src/capture_market_snapshot.py --validate-only
.venv/Scripts/python.exe -B src/capture_method_market_snapshot.py --validate-only
.venv/Scripts/python.exe -B src/update_bet_performance.py --validate-only
node tests/test_upcoming_portfolio.cjs
```

The repair is repeatable: a second run proposes no cell changes. Rebuilding forecasts issues a new forecast at the actual current time and is only valid before the card date; do not backdate it. The refresh makes no API requests and never rewrites immutable evidence to look prospective.

After changing processed schedules, also rebuild the fighter explorer as shown
above. Its fight-history shards include `time_format`; changing the source CSV
without regenerating those shards causes the saved-data CI check to fail. This
offline rebuild updates the derived index and history files without retraining.

[Focused validation](validation.json) records **112 passing Python tests**, plus **7 JavaScript behavior checks** and the JavaScript syntax check. This includes early-finish schedule cases, rejected old duration artifacts, insufficient calibration, price expiry, accessible-book alternatives, allocation limits, saved totals assessment preservation, and method event-start changes.

The remaining uncertainty is economic: no strategy has demonstrated repeatable profit. Collect new decisions under fixed rules, retain the existing review counts and return/price-quality requirements, and evaluate any further changes against the obtainable market price. The next work should be additional verified historical schedules and new prospective results, not larger stakes.
