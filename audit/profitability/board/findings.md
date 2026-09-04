# Published betting-board audit

This is an offline snapshot audit as of 2026-09-04T17:26:56.686757Z. It changes no model,
published recommendation, price, calibration artifact, or staking default.

- 12 displayed bets: 11 totals and 1 moneylines.
- Published stakes total 55.00 units on a 100-unit bankroll.
- 1 qualified bets have zero recommended stake; 1 have negative estimated return after probability calibration.
- Exact source prices trace for 12/12 rows; saved forecasts trace for 12/12.
- 11 rows omit the source-update timestamp and 11 omit card start in the website board. The ledger can supply these where the trace matches.
- 0 rows remain fresh after 30 minutes if this snapshot does not update.
- Across single-book access scenarios, 32 qualifying book/selection alternatives are hidden by filtering only globally selected books.
- The illustrative capped allocation totals 5.00 units. It is a risk comparison, not an optimized or proven profitable strategy.

## Findings and proposed corrections

1. **Combined exposure is not limited.** `script.js:3033` acknowledges related bets; `script.js:3186` sizes all bets from the same card bankroll and only rescales above 100%. Add portfolio limits before presenting stakes as a combined strategy. Acceptance: one funded selection per fight, 1% fight / 5% card / 10% outstanding caps for this explicitly illustrative comparison; deterministic ties and unchanged entry odds.
2. **Qualification uses a different probability from staking.** `src/upcoming_bet_board.py:473` and `:508` select by unadjusted return, while `:492` attaches calibrated sizing. `script.js:3054` still displays zero-stake rows. Proposed correction: evaluate one declared calibrated selection policy prospectively and distinguish a rejected stake from an actionable recommendation. Acceptance: a nominally positive bet with negative calibrated return or zero stake cannot silently look actionable under that policy.
3. **Experimental totals are ranked with market-based moneylines.** `src/market_tracker/prop_opportunities.py:73` computes EV directly from the duration model. `src/bayesian_total_calibration.py:314` checks only whether calibration worsened probability errors, not betting profit. Require historical price-matched and prospective evidence before promotion; retain research comparisons. Acceptance: unsuccessful/missing performance evidence cannot become a promoted recommendation merely by exceeding raw EV.
4. **Stale prices remain displayed.** `script.js:3053` filters threshold and book without current-time/start checks; `src/update_and_rebuild_model.py:228` carries forward the prior capture time. `.github/workflows/collect-market-snapshot.yml:13` schedules separated captures. Proposed correction: explicit per-price expiry and card-start checks using preserved exact timestamps. Acceptance: stale, missing-timestamp and started-card cases are withheld; the stored audit is still available. Expiry scenarios here assume no intervening capture.
5. **Book filtering loses alternatives.** `src/upcoming_bet_board.py:500` and `src/market_tracker/prop_opportunities.py:144` retain only the globally best book; `script.js:3055` hides disallowed books. Proposed correction: retain qualifying per-book prices and select after the user's book choice, excluding each target from the comparison probability. Acceptance: removing the best book reveals an eligible second book, with its own price and recalculated moneyline probability; an ineligible price remains excluded.
6. **Method prices remain research comparisons.** `index.html:289` excludes methods from the qualified list and `script.js:2885` explains the unpassed performance check. `surface_trace.csv` records each method price/probability comparison, including incomplete book contracts. Preserve that distinction. Acceptance: method comparison rows do not enter the actionable board merely because their nominal EV is large.

## Reproduction and limits

Run `python scripts/audit_profitability_board.py --output-dir audit/profitability/board`.
Use `--as-of` with an explicit offset to test a later instant; the default is the
saved board timestamp for deterministic reproduction. `board_audit.json` contains
source hashes, each price/forecast trace, exposure, expiry and book-access scenarios.
CSV files expose the same calculations. No network calls or model fitting occur.
The browser check is direct code inspection and executable filter emulation, not
a browser automation test. Book access scenarios are hypothetical, not statements
about which accounts the user has. No profitability claim follows from the caps.
