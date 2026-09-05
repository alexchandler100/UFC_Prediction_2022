# Weekly monitoring and odds plots

Open Market > Are we collecting useful evidence? The report shows the most recent
collection, unique capture/fight counts over the past seven days, recorded
decisions and recommendations, pending results, and results still missing more
than two days after event start. Passes are separate from recommendations.
Experiments remain separate; their profits must not be added together.

The equal-stake winner comparison includes the existing strategy results and
200-fight/20-card first-review target. This is a review checkpoint, not proof of
profitability. No policy or model was retuned.

A report older than 48 hours displays a warning. The optional “Check latest
workflow runs” button reads the public GitHub API (20 recent runs per workflow)
to show failures independently of the last successfully published data. Network
or rate-limit errors display “unavailable,” never zero failures. Full run details
are linked. No credentials or paid services are required.

Expand a paper recommendation, then “Collected bookmaker odds over time.”
Moneyline and totals audit cards also expose this chart. All collected books are
available, regardless of the portfolio book filter. Choose all books or one book;
hover/focus/tap points for exact American odds, collection time and source-update
time, or expand the full table. The graph uses decimal odds to avoid the jump
between negative and positive American odds; higher means a higher winning payout.

Histories use stable fight/fighter IDs, exact method and exact full-fight totals
line. Different totals lines are never connected. Duplicate method-horizon copies
of one observation are collapsed. Nothing is averaged. Lines connect recorded
observations across gaps; they do not establish continuous price availability.
Method captures are bounded to opening and approximately 72/24/6 hours before the
event, and their source quote-update times are unknown. A single point cannot
establish a trend. Missing exact-selection histories display an explicit empty
state. History is fetched only when opened (initial publication about 1.24 MB).

Both scheduled workflows regenerate and validate the two publications after
their ledger/report updates. They do not collect extra odds or retrain anything.

```powershell
.venv/Scripts/python.exe src/update_research_monitor.py
.venv/Scripts/python.exe src/update_research_monitor.py --validate-only
.venv/Scripts/python.exe -m unittest discover -s tests -p test_research_monitor.py
node --test tests/test_bet_odds_history.cjs tests/test_candidate_diagnostics.cjs tests/test_upcoming_portfolio.cjs
```

Deployment inspection on September 5 found Pages run 33981147173 succeeded,
but market collection run 33980677834 and reliability run 33981147840 failed in
unit tests before collection. Two website-contract assertions still required
the removed phrase “Only one selection per fight is funded.” They now assert
the replacement hypothetical-stake wording. A third assertion required the old
timer syntax; it now verifies the combined callback still refreshes paper bets
every minute. GitHub annotations exposed only
the failing step, so local test results must be followed by a successful
scheduled run after pushing. Do not assume the live collection pipeline is
healthy just because Pages deployed.

The new experiments currently have no recorded fights. Existing ledger records
and isolated capture-to-settlement tests provide checks, but a complete live
cycle for the new experiment still needs fresh captures and completed events.
