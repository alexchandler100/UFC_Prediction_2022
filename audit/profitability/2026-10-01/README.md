# UFC profitability audit and next plan — October 1, 2026

The best next investment is reliable price comparisons and complete pre-fight
records. Keep the simulator as a bounded research project. Current evidence does
not establish repeatable profit, and more detailed simulation has not earned a
larger role in bet selection.

This audit includes newer results than the previous handoff. The working checkout
is at `9933ae25`; public main was 51 commits ahead at
`e8444907cd409a2aadaa1ba9f04b6c9aa9c6d47b`. The GitHub comparison listed no changes
outside `src/content/data/`. I downloaded a separate, commit-pinned snapshot,
verified 116 file hashes, and recomputed the comparisons below. Recorded results
extend through September 26; the latest price capture is September 30 and the
weekly publication is October 1. These are paper records, not accepted wagers.

Production code, model settings, existing ledgers, and the working checkout's
data were not changed. New files are confined to this audit directory. The raw
download is locally retained under ignored `snapshot/`.

## What changed since the handoff

| Recorded experiment | Latest settled evidence | Meaning |
| --- | --- | --- |
| Main locked moneyline policy | 4 bets: 1 win, 3 losses; +9.48 units, +237% return on stakes | Sean Sharaf at +1248 returned +12.48 units; the other three bets lost. Every bet used market probability with zero winner-model weight. This is not evidence that the winner model works. |
| Small learned market adjustment | 18 bets across 3 cards; −2.34 units, −12.99% | The proposed market-based direction already exists, and this specific version has not earned promotion. Its probabilities were also worse than the market on all 31 scored fights. |
| Method-of-victory policy | 27 settled bets across 3 cards: 2 wins, 25 losses; −6.5 units, −24.07% | The original 0–12 is now followed by two profitable cards. Overall results still do not support betting it. Nine other recommendations are pending. |
| Bayesian weekly shadow selections | 29 bets across 6 cards; +6.93 units, +23.91% | Worth continuing unchanged. The reported return interval spans −15.59% to +87.83%, and this publication history lacks the full immutable decision/closing-price standard. |
| Locked totals policy | 1 bet, 1 loss; −1 unit | Almost no betting evidence. On 30 scored total lines, the independent duration model's probability error was worse than the market's. |
| Equal-stake moneyline comparison | Only 2 recorded fights | A timing-contract bug is excluding most usable fights. Fix measurement before interpreting the results. |

These experiments overlap and use different selection rules. Do not add their
profits or treat their sample sizes as independent evidence. The main policy's
reported return interval spans −100% to +1248%; four bets cannot establish a
durable advantage.

The broad historical audit still matters: across 904 fights and 189 events, the
winner model's 748 selected bets lost 42.59 units (−5.69%), the 50/50 blend's 633
bets lost 50.87 units (−8.04%), and the small market adjustment's 113 bets lost
6.66 units (−5.90%). Historical quotes and development choices have limitations;
those results are useful negative evidence, not an exact replay of today's rules.
See [the earlier history audit](../history/findings.md).

## Winner probabilities still need to justify departing from the market

Log loss measures the quality of stated probabilities, penalizing confident
mistakes heavily. Lower is better. These comparisons use identical fights within
each row; the two rows have different capture rules and should remain separate.

| Evidence set | Fights / cards | Market | Winner model | Fixed equal blend |
| --- | ---: | ---: | ---: | ---: |
| Latest weekly publications, Aug. 22–Sept. 26 | 68 / 6 | 0.64690 | 0.83648 | 0.71515 |
| Frozen approximately 24-hour comparisons since September began | 31 / 3 | 0.69701 | 0.88755 | 0.76698 |

The winner model did better on the latest seven paired September 26 fights, but
the six-card total still strongly favors the market. This is a reason to preserve
the complete comparison rather than choosing whichever card favors a model.
The 68 weekly comparisons include varying lead times and publication provenance;
they are not the immutable 24-hour betting experiment.

Another useful diagnosis comes from the broad historical selections: the winner
model assigned its chosen sides an average 50.86% chance of winning, but only
38.90% won. Average claimed return was +38.27%; realized return was −5.69%.
Selecting the largest apparent edge can concentrate probability errors. Global
winner accuracy and narrow coefficient-uncertainty intervals do not resolve that
problem.

The repository already tried richer model families, Bayesian combinations,
division-specific fits, stance, cardio, and opponent/style information. External
fight history produced a small uncertain improvement; most additions did not.
Repeating a broad model search is a low priority.

## Collection is working, but one experiment is losing evidence

The public collector is active. Of the last 40 inspected runs, 39 succeeded; the
one failure was September 14 before the optional-method fix. The September 30
17:35 UTC publication updated both moneyline and totals ledgers without changing
method ledgers. The evening publication updated all three. Together with the
rollback/skip tests, this supports the handoff's first objective without another
live dispatch. It is not a deliberate live source-outage test.

Evidence: [successful midday run](https://github.com/alexchandler100/UFC_Prediction_2022/actions/runs/36751579824),
[midday publication](https://github.com/alexchandler100/UFC_Prediction_2022/commit/baa4ce83206bb345ee80c8ab66f9dbfa7280ab87),
and [saved workflow evidence](workflow_evidence.json).

The equal-stake experiment has a separate, reproducible defect at
`src/market_tracker/equal_stake_experiment.py:66`: it requires the provider's
individual bout start to equal the common card start. The collector intentionally
retains these as separate values. For September 19 and 26, **204 of 231 quotes**
inside the experiment's 20–28-hour window fail that comparison. They cover **16
fights**, while same-start quotes cover only two fights. This explains the very
small experiment without interpreting missing records as deliberate passes.

The repair should retain both timestamps, anchor the decision window to the
declared card start, require collection/decision times before the relevant start,
and keep identity and freshness checks. Preserve rejected old captures as
diagnostic history; do not backfill newly reconstructed decisions as prospective
bets. Record a repair date/version and report the subsequent cohort separately.

Collection time also must remain distinct from the provider's quote-update time.
The provider exposes update timestamps separately in its
[official API documentation](https://the-odds-api.com/liveapi/guides/v4/).
Method records currently lack that provider-update timestamp, which limits claims
about whether the displayed price could actually have been obtained.

## The simulator deserves diagnosis before more tuning

On the 21 frozen fights with all three forecasts, the market picked 14 winners
correctly and the simulator picked 9. Their log losses were 0.66481 and 0.88758.
A half-market, half-simulator blend worsened the market score to 0.74893. The
winner-model/market blend happened to improve slightly on this smaller subset,
but that does not overturn its worse result on all 31 eligible comparisons.

I separately joined retained pre-event simulations to completed results by event
and fighter IDs. Taking the first recorded forecast per physical fight leaves
31 distinct fights across four cards. **One assumes the wrong scheduled length:**
Tsarukyan–Ruffy was simulated for three rounds, while its recorded schedule is
five. Forecasts need explicit invalidation/replacement when bout scheduling
changes; an early retained forecast alone is not enough.

Removing that mismatch leaves **30 fights across four cards**, all using
`mechanics-8ba01f34444f`:

- Expected decisions: **11.34**; observed: **18**.
- Expected KO/TKOs: **14.20**; observed: **9**.
- Expected submissions: **3.83**; observed: **3**.
- Average simulated duration: **154.37 seconds too short**.
- Winner accuracy: **13/30**, or 43.33%.

This is an explicitly defined broader sample, not a rerun of the prior agent's
6.3-versus-11 calculation. Including the wrong-schedule fight gives 11.45 expected
decisions versus 18 and a 147.33-second duration shortfall, so it does not explain
away the finish bias. One later record of an already-counted fight was
excluded; 61 records did not yet match completed results. Counts include only
one forecast per physical fight, with forecasts issued before the event date.
See [fight-level evidence](simulation_result_rows.csv).

The architectural concern is that good estimates of isolated actions do not
necessarily survive the simulator's phase, stamina, opponent-defense, and finish
transformations. The prior 229-fight opponent-adjustment test demonstrated this:
better conditional strike predictions produced worse simulated outcomes. Simply
reducing finish rates can improve average duration while damaging winner or
method probabilities. More simulated paths reduce numerical noise; they do not
repair those relationships.

There is already an appropriate next experiment in
[SIMULATION_TODO.md](../../../SIMULATION_TODO.md): compare predicted strike
rates at observed exposure, rates after the engine's transformations, and the
existing simulated distributions. Locate where the distortion enters before
running another parameter search. Keep the reserved 2025 confirmation groups
unopened until one candidate survives development.

Method paper bets use the separate outcome model, so a simulator repair will not
automatically repair the 2–25 method record. The current direct duration/method
model also needs work: its latest evaluation uses 831 development fights and 221
later evaluation fights, excluding 3,898 fights with unverified schedules.
The earlier schedule repair was essential, but did not demonstrate a profitable
totals strategy.

## Recommended work order

| Priority | Concrete work | What would justify moving forward |
| --- | --- | --- |
| 1 — next development session | Bring the development checkout up to the audited main revision; repair equal-stake timing; add a real different-card/bout-start regression case; expose exclusion counts and missed decision windows. Fix numerical report validation separately. | A fresh eligible capture records all qualifying fights; expected skips have reasons; a new offline replay preserves old decisions exactly. |
| 2 — same session / next card | Continue early and 24-hour prices; add a dependable final pre-card reference, with declared window and exact quote age. Record accessible books, full two-sided prices, passes, forecast/calibration versions, and a decision timestamp. | Every eligible fight has its intended observations or an explicit gap. Later-price comparisons identify their time window and missingness. No missing close is silently replaced by an hours-old quote. |
| 3 — next research session | Audit apparent value at selected prices, especially long shots. Keep the target sportsbook out of the comparison. Evaluate the existing calibrated-market and unadjusted-market policies on the same fights; test at most one earlier-selected alternative to house-margin removal or book weighting. | Improvement survives later cards, obtainable-book restrictions, and fixed worse-price scenarios. Historical discovery remains separate from new confirmation. |
| 4 — ongoing, fixed rules | Continue the existing early-favorite/late-underdog timing experiment, existing market adjustment, and Bayesian shadow. Publish a compact card-by-card evidence report. | Better prices on independently selected bets, then credible returns. No new thresholds chosen because the newest card looked favorable. |
| 5 — bounded simulator session | First enforce current schedule agreement and replace invalidated forecasts prospectively. Run the already-specified strike-to-engine diagnostic. Separately compare current direct duration forecasts, a simple earlier-data duration baseline, and the simulator by scheduled length/division. | Identify a concrete translation error or a later-tested improvement. Do not proceed to a broad mechanics search merely because average duration can be made closer. |

Price timing is presently the most interesting small positive lead. Across 55
paired fights on five cards, same-book early favorites improved the required
break-even win probability by about **0.70 percentage points**; late underdogs
improved it by about **0.83 points**. Those changes favor better entry prices, not
automatic bets on every favorite or underdog. The prices may come from books the
user cannot access, and five cards remain a very small sample. Keep the rules
fixed and continue collecting both sides even when no bet qualifies.

The practical objective should be net return at obtainable prices with tolerable
drawdowns. Require improved price quality, calibration on selected bets, and
returns together. A better probability score is useful supporting evidence; it
is neither necessary on every fight nor sufficient to prove a profitable
selection policy.

Retain the existing review counts and economic requirements: main policy 500
scored fights / 100 bets / 40 events; totals 300 scored lines / 100 bets / 30
events. The smaller 200-fight / 20-card comparisons are research checkpoints,
not automatic betting approval. Confidence ranges must still support positive
returns and better entry prices. With few cards, resampling those cards can
understate how much remains unknown.

I would spend most near-term effort on priorities 1–4, reserve one bounded
session for priority 5, and postpone new model families, larger simulation runs,
expert-feed expansion, and stake optimization. The project needs stronger
evidence of an advantage before more complexity or more capital.

## Verification and reproduction

- 15 method-collection tests passed, including rollback, skip, and publication
  status behavior.
- 9 equal-stake tests and 7 method-policy tests passed. The existing equal-stake
  suite does not cover different bout and card start times.
- Full local data validation reproduced all 714,138 feature cells and checked
  the local saved artifacts. It **failed four report-equality checks**. A targeted
  in-memory rebuild found only seven floating-point differences, at most
  `2.22e-16`, in those sections. These failures are numerical comparison
  brittleness, not changed returns or missing fights. See
  [exact differences](local_validation_differences.json). Use a tightly bounded
  tolerance for derived numeric metrics; keep IDs, ledgers, hashes, and decision
  contracts exact. Do not describe the full validation command as passing.
- Latest remote snapshot hashes were verified and its displayed probabilities,
  paper outcomes, and simulator joins were independently checked as described.
  A full latest-snapshot repository validation was not run locally; recent live
  workflows passed their validation steps.

Run the bounded read-only analysis after restoring the manifest's pinned files
under `snapshot/`:

```powershell
.venv/Scripts/python.exe -B audit/profitability/2026-10-01/analyze_snapshot.py
```

Outputs: [computed evidence](checkpoint_evidence.json),
[weekly paired forecasts](weekly_probability_rows.csv),
[simulation results](simulation_result_rows.csv), and
[source manifest](source_manifest.json). The script writes only audit outputs;
it makes no network requests, runs no new simulations, and changes no betting
policy.

## Implementation follow-up — October 1

The first implementation session repairs collection and measurement. It does
not change the betting thresholds, train a new model, or change simulator
mechanics. The checkout was fast-forwarded to the audited `e8444907` revision
before implementation. The earlier findings above describe the original audit;
this section records the subsequent repairs.

- **Equal-stake timing:** decisions use the card start for their 20–28 hour
  window. Provider estimates for later bouts are retained separately; both
  collection and decision must precede both starts. Four fresh books, native
  forecasts available before capture, exact identity agreement, and the
  five-minute decision limit still apply.
- **Historical proof without invented bets:** replaying the 231 quotes in the
  September 19 and 26 windows accepts 18 fights with the repaired rules versus
  2 with the original rules. The replay writes only its diagnostic report, no
  decisions or hypothetical profits. See [replay evidence](timing_repair_replay.json)
  and [reproduction script](replay_timing_repair.py).
- **Prospective boundary:** the sealed collection contract starts October 1 at
  15:30:13 UTC. The existing two decisions and settlements remain unchanged;
  zero repaired-contract decisions have been collected locally. Before/after
  results are separate. Latest-capture exclusions appear in the monitoring
  page. The original policy, decision values and settlements match the audited
  revision; repeat updates preserve their bytes, including Windows line endings.
  See [integrity evidence](repair_ledger_integrity.json).
- **Final prices:** fresh observations are measured in declared early
  (32–144 hour), day-before (20–28 hour), and final (15–90 minute) windows before
  each saved card start. Totals remain separate by line. Only two of the four
  settled main moneyline bets have a final same-book reference under these
  rules. The other two stay missing. Independent final probabilities require
  three other books and exclude the entry book; comparisons preserve fighter
  identity even when quote orientation reverses. These are pre-card references,
  not exact bout closing prices or proof of obtainable wagers.
- **Scheduling:** clock checks every half hour request odds only for an
  uncovered day-before or final window, based on the known card start. Regular
  collection continues. Clock-only jobs do not take the shared publication
  lock. Delayed workflow starts or source outages can still leave gaps, which
  remain visible. The additional schedule takes effect only after publication.
- **Numerical validation:** report comparisons tolerate only rounding-sized
  differences in derived floats. Current-data validation also exposed one
  saved pass with a replayed blend probability differing by `5.55e-17`; its
  largest derived difference was `2.22e-16`. See the [exact differences](current_decision_replay_differences.json).
  Both stored and rebuilt decisions must validate their own exact content
  hashes. Input probabilities, source IDs, timestamps, thresholds, stakes and
  selected actions must match exactly. Only derived calculations receive the
  tiny tolerance; stored records and IDs are never rewritten.

Verification completed: 541 tests in the full suite; 46 reliability tests after
the additional decision-replay repair; workflow YAML/dependency checks;
JavaScript syntax; report reproduction; and desktop/mobile browser checks.
The in-app browser tool failed before startup because its transport lacked
`sandboxPolicy`; a local headless Chrome/Selenium check succeeded. See
[UI evidence](repair_ui_validation.json). Strict model and market validation
passed, reproducing all 716,106 historical feature values. The optional
simulator research bundle is incomplete and was not included in that validation.
See [complete check results and remaining warnings](repair_validation.json).

Before relying on new collection, publish the code and contract and observe a
fresh eligible scheduled capture. Then inspect the repaired-record count and
gap report on the next card. Continue the fixed paper comparisons; the next
research session should audit selected prices and long shots against other
books. The simulator diagnostic remains a separate bounded task, as specified
in priority 5 above. No new evidence here establishes a profitable strategy.

## Selected-price audit — subsequent research

The collection repair was pushed in `0f2b6b9d`. The next research step used the
same pinned `e8444907` data so later collections cannot change this audit's
sample. The [design](selected_price_design.json) specifies one alternative
margin calculation and no threshold search. These outcomes were already known;
this is a sensitivity analysis, not a new prospective test. Reproduce it with:

```powershell
.venv/Scripts/python.exe -B audit/profitability/2026-10-01/selected_price_audit.py
```

The apparent long-shot edge is fragile. The ordinary calculation divides each
book's two implied win probabilities by their sum. The alternative raises
each implied probability to a common power until the pair sums to one; with
positive book margins this removes proportionally more probability from the
long shot. This is the power method described by
[Clarke, Kovalchik and Ingram (2017)](https://doi.org/10.11648/j.ajss.20170506.12),
whose evidence is not specific to UFC. Applying it to the same other-book
quotes makes all four original main selections fail the unchanged 5% entry
threshold:

| Recorded selection | Entry book / price | Expected return, ordinary market calculation | Expected return, power calculation |
| --- | --- | ---: | ---: |
| Jeisla Chaves | Bovada +450 | +6.52% | −7.88% |
| Terrance Chatman | BetMGM +750 | +7.89% | −18.42% |
| Sean Sharaf | BetOnline.ag +1248 | +13.65% | −36.29% |
| Vanessa Demopoulos | DraftKings +675 | +7.90% | −14.38% |

These expected returns are estimates, not observed profits. Sharaf still won
and the original four-bet ledger still made 9.48 units. A 5% reduction in winning
payout would leave 8.86 units, but that does not address uncertainty in the
estimated chance of winning. Excluding any one additional comparison book
preserves positive ordinary-market edges on all four, so no single comparison
book explains the finding. All selected quotes and consensus inputs pass the
strict source-time check. That rules out these two simple explanations within
the stored data; it does not establish that the prices were obtainable.

The market-adjustment model presents a different problem. It selected 18 bets
across three cards, won 11, and lost 2.34 units. It predicted about 12.70 wins
among those selections, versus 11.61 from the other books' ordinary market
probabilities. None of the 18 meets its original 2.5% threshold using the
ordinary market estimate alone, or the power market estimate alone. Their
qualification depends on the learned model adjustment, which has not earned
that confidence on this sample.

The later-price report also needs two distinct comparisons:

| Strategy | Bets with qualifying final same-book and independent prices | Mean same-book improvement | Mean advantage versus final other-book probabilities |
| --- | ---: | ---: | ---: |
| Main market-only policy | 2 of 4 | +0.83 percentage points | +1.06 percentage points |
| Market-adjustment policy | 9 of 18 | +1.35 percentage points | −1.98 percentage points |

A positive same-book move can coexist with an unattractive entry price after
the other books' margins are removed. Seven of the nine market-adjustment
entries have negative independent final advantage under ordinary margin
removal. Under power removal, the average independent advantage is −0.49
points and only three of nine are positive. The main policy's two available
references also turn negative under power removal (−2.17 points on average).
The monitoring page now reports both price measures, separate strategy counts,
missing references, fighter names and card dates. It respects the existing
selected-book filter. These comparisons are not bookmaker settlement evidence.

There is no reason to adopt the power method from this result alone. On the
same 31 fights across three cards, ordinary market probabilities still have
the lowest probability error:

| Estimate | Log loss — lower is better |
| --- | ---: |
| Ordinary market | 0.6970 |
| Existing frozen calibrated market | 0.7113 |
| Power market sensitivity | 0.7176 |
| Winner model | 0.8876 |

This error penalizes confident wrong forecasts. Three cards cannot reliably
rank small differences. The frozen calibration was available before these
31 decisions; its later creation makes it unavailable for the two August 22
selections in this audit. Those rows are explicitly missing that comparison.
Different sample counts must not be presented as a fair probability contest.

Keep all live selection rules fixed and paper-only. Prioritize new evidence
about independently favorable entry prices and calibration on the selections,
especially long shots. Keep the power calculation as a sensitivity check;
do not select it because it suppresses these bets. No new bookmaker weights,
margin alternatives, or thresholds were fitted. Book-specific results are in
the evidence, but access to a recorded book remains unverified until the user
supplies their usable books.

Detailed outputs: [computed evidence](selected_price_evidence.json),
[individual selections](selected_price_rows.csv), and
[reproducible code](selected_price_audit.py).

The [live verification run](https://github.com/alexchandler100/UFC_Prediction_2022/actions/runs/36886864389)
succeeded and published `a9b0d000`: 118 moneyline quotes, 44 total-round quotes,
12 winner forecasts and 14 total-line forecasts. Optional method-price output
validation was skipped, so it did not block the healthy core publication.
All 12 matchups were outside the 20–28 hour decision window; zero new
equal-stake decisions is expected. The follow-up report labels this as a timing
skip instead of a shortage of eligible books. See
[recorded live evidence](live_repair_verification.json).

The expanded monitoring report was rebuilt from that live data and reproduced
exactly. The policy and decision/settlement ledgers were unchanged. The full
545-test suite passed; the subsequent diagnostic-label adjustment passed all
15 equal-stake tests. Three independent audit tests check margin conversion,
fighter orientation, target-book exclusion, capture separation and minimum
book counts. Desktop/mobile and single-book filtering also passed; see
[browser evidence](selected_price_ui_validation.json) and
[follow-up validation](selected_price_validation.json).

The subsequent [simulator bridge audit](../../../SIMULATION_STRIKE_BRIDGE_AUDIT_REPORT_2026-10-01.md)
completed all 229 development fights without new trajectories. It identified
loss of fighter ordering when the observation predictor is rebuilt for the
simulator, plus phase-rate and opponent-defense mapping mismatches. The next
simulator task is to preserve that tested predictor before another expensive
screen. This reinforces the priority order: collect obtainable prospective
prices, test the selected bets' probabilities, and keep simulator repairs as
bounded research until they demonstrate a useful improvement.
