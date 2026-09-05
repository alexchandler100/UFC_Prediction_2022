# Keep published recommendations visible

The Market page now includes Recorded recommendations under the fresh-price
list. It reads the saved publication archive plus the current board's published
bets. It does not recreate selections from expired offers or apply today's
calibration to an old recommendation.

Original prices, probabilities, EV and recorded stakes remain available. The
status changes to expired, awaiting result/review, or won/lost/void. A missing
start time is explicitly unavailable, never guessed. Results are joined by the
exact archived snapshot ID to the existing performance publication. Book filters
hide saved picks without retrospectively switching their bookmaker.

Repeated publications of the same selection at the same book are grouped with
every original timestamp/price accessible. Different totals selections remain
separate. Every group retains its odds-history plot. Older picks remain visible
after the current board rolls forward; earlier rules are labeled as historical.

Method paper reports likewise retain settled recommendations, with settlement
status and profit alongside the original selection. The existing settlement
ledger and all recommendation policies are unchanged. Fresh recommendations
still require eligible prices and pre-event timing.

Validation: 15 focused JavaScript tests, seven method paper tests, and 21 website
contract tests passed. The saved method publication validates without rewriting
any records. Regression coverage includes expiry, card start, board rollover,
settlement, duplicate publications, filtering, and immutable original odds/EV.

```powershell
node --test tests/test_upcoming_portfolio.cjs tests/test_candidate_diagnostics.cjs tests/test_bet_odds_history.cjs
.venv/Scripts/python.exe -m unittest discover -s tests -p test_method_paper.py
.venv/Scripts/python.exe -m unittest discover -s tests -p test_website_contract.py
.venv/Scripts/python.exe src/update_method_paper.py --validate-only
```
