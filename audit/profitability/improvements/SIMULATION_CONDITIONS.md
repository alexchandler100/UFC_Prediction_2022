# When might a model add value?

Three fixed hypotheses now collect prospective evidence alongside the existing
simulation comparison. They do not change recommendations or train models.
Only comparisons created after this policy's actual activation qualify, and
indicators must be recorded before event start. Existing comparisons are excluded
from this new study, even if results have not yet been scored.

The thresholds below are illustrative research choices, not thresholds found to
be profitable. They are sealed in `market/simulation_conditions/policy.json`.

| Question | Fixed matching condition |
| --- | --- |
| Substantial relevant data? | Each fighter has at least 5 prior UFC fights, 100 distance strike attempts, 50 ground strike attempts, and 20 takedown attempts; all three statistics are present in at least 80% of prior fights. |
| Recent history? | Each fighter has at least 2 UFC fights in the 730 days before the simulation forecast. |
| Narrow simulation range? | Both reported red/blue win 95% parameter ranges are at most 20 percentage points wide, process sampling error is at most 2 percentage points each, and there are at least 32 parameter replicas. |

The first condition measures both fighters' own recorded attacking activity.
It does not measure time spent grappling or fully measure defensive experience.
The simulator's `ground_minutes` field counts fight exposure with available
ground statistics, so it is deliberately not labeled actual ground time here.
Missing/insufficient statistical coverage or uncertainty information is reported
as unavailable, separately from failing a condition. Low activity with complete
data is a nonmatch. The conditions are tested separately, with no combination
search or changing thresholds after seeing results.

History uses only fights before the simulation forecast's calendar date; same-day
records cannot establish earlier availability. We save the raw-data hash, each
fighter's counts/coverage, uncertainty ranges, the complete immutable comparison,
its source-publication/model lineage, and the exact reference quote and source
timing when usable. These indicators describe prior records visible when frozen;
they do not prove the exact contents of the original simulator's training inputs.
The uncertainty ranges are the published unconditional red/blue win ranges,
not recomputed ranges conditional on a decisive outcome, and can retain simulation
sampling noise. Forecast evaluation uses the existing decisive-outcome convention.

Each physical fight gets one record. A missing or replaced source publication
cannot substitute another simulation. New comparisons without matching indicators
remain visible in coverage counts. Existing records are not rewritten on reruns.
The new sidecar leaves existing comparison and settlement ledgers unchanged.

The report separates all fights, matches, nonmatches and unavailable evidence for
each condition and each simulation mechanics version. Within each group, market,
winner-model and simulation probabilities are scored on identical settled fights
using squared probability error (Brier score), log loss and winner accuracy.
Negative model-minus-market error favors that model. Uncertainty for the error
difference resamples whole cards 1,000 times with a fixed random seed.

Paper returns use the exact base decision's reference bookmaker, one unit per
fight per strategy, and at least 5% estimated EV. Quotes must match the capture
and fighter orientation, be observed no more than 5 minutes before the decision,
and have a provider update within 30 minutes. Missing prices yield unavailable
returns, not zero profit. Pending fights and void outcomes do not enter scored
returns. Recorded-payout and 2%/5% reductions in net winning payouts are reported
separately, with zero profit for the no-bet baseline. This is not all-book shopping
or verified execution. These hypothetical strategy portfolios are separate.

The 200-fight/20-card checkpoint applies to each group and is only a first review.
Groups overlap, and the descriptive intervals are not adjusted for examining
multiple hypotheses. Nothing automatically promotes a model. Historical searches
or future threshold changes remain development work and require fresh confirmation.

Both scheduled workflows update and validate this study after simulation
comparisons and settlements. Market > Are we collecting useful evidence? shows
the compact results and links to the complete report, including paper returns.

```powershell
.venv/Scripts/python.exe src/update_simulation_conditions.py
.venv/Scripts/python.exe src/update_simulation_conditions.py --validate-only
.venv/Scripts/python.exe src/update_research_monitor.py
.venv/Scripts/python.exe src/update_research_monitor.py --validate-only
.venv/Scripts/python.exe -m unittest discover -s tests -p test_simulation_conditions.py
```

No network requests, paid services, simulation reruns or historical optimization
are required. Initial publication has zero records by design.
