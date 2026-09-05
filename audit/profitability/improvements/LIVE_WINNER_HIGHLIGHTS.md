# Inline winner highlights

The existing Matchups card list shows reported winners in gold, with a compact
KO/TKO, SUB, DEC or DQ label beside the name when the feed explicitly provides
that method. No panels, rows, columns or extra vertical spacing were added.
Winner names truncate with an ellipsis if needed to keep the label on the same
line; full names, source and last successful check are available in hover text.
Open matchup details are preserved because only the name elements update.

The browser requests ESPN's public current UFC scoreboard every two minutes
while Matchups is visible and a displayed card is within the fight-day window
(12 hours before the UTC event date through 48 hours after). Hidden tabs and
other views do not poll. Requests time out after ten seconds; failures back off
to ten minutes and retain any previously received results with a stale-data
tooltip. No API key, paid odds credits or extra GitHub Actions jobs are used.

Require a completed bout and exactly one named winner. Join both normalized
full fighter names and event date, accounting for Chicago/UTC midnight rollover.
Ambiguous duplicates are withheld. Draws, no contests and scheduled bouts do
not highlight a winner. Submission attempts and knockdowns never become finish
labels. Missing method information leaves the gold winner name without a guessed
method. A successful subsequent feed replaces prior results, allowing corrections.

These results affect display only: no training data, predictions, recommendation
records or paper settlements change.

Source inspected: https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard
The web reader returned completed September 5 bouts with winner flags and
explicit winner-method descriptions. Direct requests from this environment
returned HTTP access denied. Browser CORS/access and actual live delivery are
therefore unverified; this is not a confirmed working production feed. Failures
are exposed in existing-name/card-metadata hover text without adding layout.

Validation:

```powershell
node --check script.js
node --test tests/test_live_results.cjs
.venv/Scripts/python.exe -m unittest discover -s tests -p test_website_contract.py
```

Six result-parser/polling tests and 21 website contract tests passed. A deployed
browser check is still needed to verify ESPN access and visual sizing.
