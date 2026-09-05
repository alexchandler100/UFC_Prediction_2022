# Method-of-victory paper recommendations

Methods now have a prospective recommendation and settlement experiment, like
the existing totals paper ledger. All stakes are hypothetical. The conservative
Kelly-based board and the fixed-stake experiments are different research policies;
neither places wagers. No prior demonstration of profitability is required to
enter this experiment.

Rules are frozen at initialization in `market/method_paper/policy.json`:

- Use only new captures after activation, recorded within 30 minutes of collection
  and before the event starts. BestFightOdds does not supply provider quote-update
  times in our stored method data; collection age is explicitly a proxy, not
  verification that a sportsbook price is still available.
- Require an exact same-capture matchup/horizon forecast, issued before collection,
  using the repaired duration-model version. Use the joint probability of the
  named fighter winning by KO/TKO, submission, or decision, not the unconditional
  probability that either fighter wins by that method.
- Freeze the first eligible capture for a fight, including a pass. Select the
  highest estimated EV across individual book/selection prices when EV is at
  least 5%. Risk one paper unit, with one selection per fight. No Kelly filter,
  averaging of executable prices, backfill, or later replacement of a pass.
- Preserve all offers, the exact selected quote, forecast, hashes, timestamps,
  source-price payload hash, and model lineage. These are separate JSON ledgers;
  existing forecasts, price histories, and other paper experiments are unchanged.

Settlement uses a **declared paper convention**, not verified bookmaker payouts:
standard KO/TKO, submission, and decision outcomes (including technical decisions)
are graded against the named fighter and method. Draws, no contests,
disqualifications, and changed scheduled round counts void the paper stake.
Missing schedules, conflicting or unrecognized results, cancellations, and
replaced fighters remain pending review. We do not infer cancellation from an
incomplete results feed. Stored settlements are never rewritten after later
result amendments. Profit uses each saved price and excludes pending/void stakes
from settled ROI; a no-bet baseline is zero profit.

The inspected [FanDuel New Jersey MMA rules](https://www.fanduel.com/fanduel-sportsbook-house-rules-nj)
classify technical decisions as decisions, void method bets on disqualification
or no contest, and void non-moneyline markets when scheduled rounds change.
Other jurisdictions/books can differ. These rules inform our explicit convention;
we do not assert that all captured bookmakers settle identically. Verbal
submissions or unusual result classifications may require review against the
official decision. Quoted EV uses the ordinary win/loss formula and does not
estimate the chance of a refunded stake. All prices and outcomes are research
evidence until bookmaker-specific execution and settlement are verified.

Open **Market > Method-of-victory paper recommendations**. Recently collected
prices appear first, sorted by estimated EV. Recorded selections stay visible
after quote expiry, marked as expired or awaiting results. Selecting books filters
the frozen picks; it does not retrospectively replace them with a different bet.
This is a separate experiment, so its one-unit stakes are not added to the other
policies' hypothetical portfolios.

```powershell
.venv/Scripts/python.exe src/update_method_paper.py
.venv/Scripts/python.exe src/update_method_paper.py --validate-only
.venv/Scripts/python.exe -m unittest discover -s tests -p test_method_paper.py
```

Both existing data workflows run the updater and validate/publish its outputs.
On the market workflow it runs immediately after optional method collection.
Failed or skipped collection cannot manufacture new recommendations from old
quotes. The command performs no network requests or training. Initializing it
with existing historical prices records zero recommendations, intentionally.
