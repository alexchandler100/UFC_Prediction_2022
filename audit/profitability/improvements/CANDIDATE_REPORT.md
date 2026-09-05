# Candidate explanations

Open **Market > Why are there no bets?** to compare every saved moneyline
side/book offer, plus available totals candidates. Announced fights without
matched quotes are included as unavailable. Select your sportsbooks above the
report to restrict the displayed offers.

The default ranking puts usable prices first, then moneylines with at least 5%
adjusted expected return and positive conservative expected return, ordered by
adjusted return. Both fights and offers within each fight use that ranking.
Choose **Highest independent-model EV** to inspect disagreements with the market,
or **Highest adjusted EV** to remove the preference for a positive conservative
estimate. Unusable prices remain below usable ones in every sort. These rankings
are diagnostic estimates, not validated profitability rankings.

Each row shows price, raw market probability/EV, adjusted probability/EV,
independent model probability/EV, and all applicable reasons it is not funded.
Market probabilities exclude the target book and require three other books that
met the existing capture-time consensus freshness rule. The report distinguishes
an adjustment taking EV below 5%, the conservative uncertainty rule, a zero
stake, missing data, price expiry, event start, totals restrictions, and portfolio
selection. Reasons overlap; counts are side/book offers, not independent bets.
Hover over a row for source update, forecast issuance, and capture-time reasons.
Expiry and ranking update every minute without a page reload.

This is a reconstruction from stored publications. A newer model forecast can
be compared with an older stored quote here, but this does not become a
prospective decision or a claim that the price remains obtainable. The separate
equal-stake experiment remains the source for future betting comparisons.
The funded recommendation rules and historical ledgers are unchanged.

Totals collection was verified separately: `_build_total_round_forecasts`
accepts verified duration forecasts without requiring betting-performance
approval, and the collector appends forecasts and prices before publication.
A regression test captures and round-trips an explicitly unfunded totals
forecast through the immutable store. Earlier invalid duration-model records
remain historical evidence and must not be pooled with the rebuilt model.
Method-of-victory prices remain in the existing research view.

Generate or check the reproducible JSON report:

```powershell
.venv/Scripts/python.exe src/update_candidate_report.py
.venv/Scripts/python.exe src/update_candidate_report.py --validate-only
.venv/Scripts/python.exe -m unittest discover -s tests -p test_candidate_report.py
node --test tests/test_candidate_diagnostics.cjs
```

The output is `src/content/data/market/candidate_report.json`. Both existing data
workflows regenerate and verify it before publishing. It uses local artifacts
only; no API calls, model training, or new paid services are involved.
