# Prospective equal-stake moneyline comparison

Compare market consensus, the existing adjusted consensus **mean**, and the
production winner model. Each strategy risks one hypothetical unit when its
estimated return is at least 5%, selecting at most one side/book per fight.
No Kelly sizing or lower probability bound is used. A no-bet baseline starts
and stays at 100 units. This is a separate research experiment, not the website's
funded recommendation policy. It adds no service calls or model training.

The first run freezes the activation timestamp and a complete copy of the
current calibration artifact in `policy.json`. Previously collected quotes are
excluded. The calibration stays frozen even if the website's artifact changes.
Production forecasts retain their individual model IDs and training cutoffs;
model changes must be considered when interpreting pooled results.

Capture rules are fixed before results:

- First eligible capture per matchup, 20–28 hours before the event starts.
- Recording within five minutes of collection, and provider quotes no more than
  30 minutes old at both collection and recording; future timestamps rejected.
- Known event start, native model probability issued before collection, and
  at least three other fresh bookmakers for every target book's consensus.
- Both sides and all qualifying book inputs retained, including exact source
  timestamps, quote IDs, consensus quote IDs, forecast provenance, and policy hash.
- Highest expected return wins; ties use bookmaker, side, and quote ID in that
  order. Missing/ineligible captures are skipped, never backfilled later.

Results report all-book access as **hypothetical**, and separately reselect bets
for each observed bookmaker. Bookmakers without observations are unavailable;
their absence is not a zero-return result. These outputs do not assert that any
particular user can access a book or execute its recorded price. Market evidence
may include other books even when the betting comparison uses one book only.

Settlements use the existing UFCStats result index. Missing or ambiguous results
remain pending; recorded draws/no-contests void the paper bet. Old decision and
settlement values are preserved, duplicate matchups rejected, and each record is
hashed. Atomic file replacement and a writer lock protect the separate JSON
ledgers. No existing paper ledger is modified.

The report includes counts, pending and void bets, settled amounts risked, profit,
return per unit, and an illustrative 100-unit starting balance. Stakes remain
one unit rather than compounding. This balance is an accounting comparison, not
a funded portfolio simulation. Drawdown is measured at card ends and therefore
does not measure within-card declines or outstanding exposure. Profit intervals
resample whole cards 2,000 times with a fixed seed; fewer than two cards gives no
interval. Zero-risk bootstrap samples are omitted from the ROI interval.
The 2% and 5% reductions apply only to net winning payouts and are stress
scenarios, not measured execution costs.

Review after at least 200 settled fights across 20 cards; those counts alone do
not establish profitability. Compare actual bet counts and card-level uncertainty,
especially for longshots. Do not tune the 5% threshold on these results and then
call the same results independent evidence. Any revised strategy needs a new
version and a new future evaluation period. There is no automatic promotion to
real-money betting, and the existing totals restriction is unaffected.

Run from the repository root:

```powershell
.venv/Scripts/python.exe src/update_equal_stake_experiment.py
.venv/Scripts/python.exe src/update_equal_stake_experiment.py --validate-only
.venv/Scripts/python.exe -m unittest discover -s tests -p test_equal_stake_experiment.py
```

Outputs: `src/content/data/market/equal_stake_experiment/{policy,decisions,settlements,report}.json`.
Both existing data workflows include collection/settlement and validation steps.
The market workflow records immediately after capture to meet the five-minute
limit. Automation takes effect once these local changes are deployed; no new
scheduled job or paid data source is required.
