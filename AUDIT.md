# UFC Prediction Codebase Audit

Audit and implementation date: 2026-08-20 (research updated 2026-08-22)

## Executive conclusion

The original model's main weakness was not logistic regression itself. It was the combination of point-in-time feature errors, name-based identity joins, excluded split decisions, and a random train/test split that mixed eras. The random split reported 63.8% accuracy and 0.636 log loss, while the newest chronological 20% fell to 59.6% accuracy and 0.706 log loss. Since coin-flip log loss is 0.693, that future holdout was worse than always predicting 50/50.

The production path has been replaced with a stable-ID, pre-bout replay and nested chronological evaluation. The automation has also been rebuilt around validated, atomic publication. The current worktree no longer depends on manually running the updater each week, but the workflow must be committed, pushed, and manually dispatched once before that claim is verified on GitHub's runner.

Betting remains disabled. A conservative pre-event Git-history reconstruction
now confirms that the market is a stronger baseline than the legacy stats
forecast, while the first market/stats blend and 5%-EV paper policy failed their
promotion tests. A separate scheduled collector now starts the prospective,
timestamped track record needed to evaluate expected return honestly.

## What was implemented

### Point-in-time data and model

- One canonical row per physical `fight_url`; fighters and events are keyed by UFCStats URL IDs rather than fuzzy display names.
- Causal replay by date, event, and explicit bout order. A later tournament round can see an earlier same-event result, while appending future fights cannot change an existing feature row.
- All terminal W/L results, including split and majority decisions, are retained. Draws and no-contests advance state/history but are not forced into binary labels.
- Pre-bout global and division Elo, opponent strength, experience/reliability, recency/layoff, record/form, age/size, and observed performance-rate features.
- Missing source statistics are excluded from that statistic's exposure instead of being converted to zero. Sparse/debut matchups carry explicit history/status fields.
- Deterministic antisymmetric orientation and a zero-intercept model: `p(A, B) + p(B, A) = 1` by construction.
- Imputation, scaling, regularization selection, and calibration are fitted only on earlier folds. Evaluation uses expanding chronological folds; the final selected pipeline is then refit on the complete ten-year window.
- Symmetric temperature calibration rather than an unconstrained transformation that could break swapped-matchup symmetry.
- A portable content-hashed JSON artifact containing the exact ordered 82-feature contract, scaler, coefficients, calibration slope, hyperparameters, runtime versions, training window, temporal metrics, and training/state fingerprints.
- Forecasts are produced only after reloading and validating the saved artifact against the replayed source state.

### Data sanitation and lineage

- Central UFCStats client with timeouts, retries/backoff, rate limiting, HTTP/content checks, and handling for the site's HTTP-200 JavaScript proof-of-work page.
- All-or-nothing event ingestion. Missing fight links/details, unknown results, malformed timing, or source/parsed manifest mismatches defer the event instead of saving a partial card.
- Recent saved events are re-scraped so corrected statistics/results can propagate. A refresh that would remove an existing stable fight ID is quarantined for review.
- Active/recent fighter profiles are refreshed by URL ID, allowing later height/reach/DOB/name corrections to heal old missing values.
- Source card position and causal bout order are persisted, mirrored, range checked, required to be contiguous/inverse, and used in deterministic sorting.
- UFCStats `Time format:` metadata is parsed so early long rounds/overtimes are not treated as modern five-minute rounds. Unknown legacy durations remain missing rather than becoming false exposure; a newly scraped non-NC with unparseable duration is rejected.
- Draw/no-contest handling, complementary result checks, numeric finiteness/domains, landed <= attempted, control <= duration, unique identities, and exact two-sided mirror contracts are validated.
- The point-in-time validator requires the exact eligible raw fight-ID set and recomputes every lineage field and target. It also verifies cross-platform canonical training/state hashes.
- CSV/JSON writes use atomic replacement. The Git commit is the publication transaction: nothing is staged or pushed until strict cross-file validation passes.

### Forecasts, odds, and history

- The Odds API moneylines are converted to implied probabilities, de-vigged per book, and aggregated in probability space rather than averaging American odds.
- The prospective market ledger records a fresh observation time on every retrieval and a separate first-seen time for an unchanged quote. Capture IDs cannot mix runs, events, timestamps, or source payloads.
- Native API quotes have a companion immutable ledger containing the provider event/book keys, provider quote-update time, scheduled commence time, and quote age at retrieval without rewriting the existing quote IDs.
- Every captured quote carries stable event/fighter IDs, a nullable fight ID until settlement, the full parsed-source hash, and a frozen native model forecast with model/version/training/source-commit lineage.
- Market/stats blending is a one-parameter symmetric logit interpolation selected only from settled prior cards. Same-date cards cannot train each other.
- Multiple captures for one matchup require a predeclared event cutoff; selection deterministically takes the latest eligible capture and records its capture ID, while a matchup with no on-time capture fails closed.
- Paper price evaluation is leave-one-book-out: the target book is excluded from its own consensus, at least three other books are required, and the target/consensus must come from the same retrieval timestamp.
- Valid no-vig market consensus is the primary published forecast; the independent model probability, status, prior-fight counts, model ID, version, and trained-through date remain visible.
- The Odds API is optional enrichment for the authoritative weekly rebuild. Network, schema, duplicate-match, unmatched-card, and merge failures fall back to model-only forecasts with explicit status instead of discarding valid UFCStats/model output. FightOdds remains an explicit local-only browser fallback.
- New prediction-history rows carry stable fighter/fight IDs and actual result lineage. Forecast correctness and independent-model correctness are computed separately; draws, no-contests, ties, and missing forecasts are unscored rather than counted as losses or deleted.
- Betting recommendations and execution APIs are explicitly disabled. Paper records use fixed one-unit hypothetical risk only; there is no bankroll, account, order, or live-wager interface.
- A prospective T-24 policy freezes at most one paper decision per matchup, requires source quotes no more than 30 minutes old and at least three leave-one-book-out comparators, and keeps the blend market-only until prospective evidence supports otherwise. A versioned timing challenger separately freezes the first observed favorite and compares first-available, T-24, and causal early-favorite/late-underdog policies using identical 5% EV and leave-one-book-out rules. The updater settles completed records from stable UFCStats IDs and publishes bounded ROI/scoring/CLV/price-movement diagnostics with whole-card intervals.

### GitHub Actions and repository reliability

- Declared/pinned the previously missing Bokeh dependency and upgraded checkout/setup-python to their current v7 actions.
- Added `contents: write`, `workflow_dispatch`, Chicago-local scheduling, concurrency control, a timeout, dependency caching, pre-update tests, preflight validation, strict post-build artifact validation, and an Actions step summary.
- Uses a shallow checkout instead of cloning roughly 1 GiB of Git history every week.
- Stages only the known generated data/model publication paths, not `git add .`.
- The authoritative updater runs twice Sunday plus Wednesday, and a separate market-snapshot workflow makes fourteen bounded Monday-Saturday observations. The Sunday evening run is a source-readiness retry so the next card is normally published before Monday. They share the publication concurrency lock; the collector validates the frozen model/card first, stages only the quote/forecast/source-timing/decision ledgers and bounded report, and refuses to publish if the starting branch tip changed. Once a previously timed card commences, later attempts become successful no-ops without an API request.
- Cleanly exits when generated bytes are unchanged.
- Fetches the remote branch tip after generation and refuses to publish if code changed during the run; it never rebases old-code artifacts onto new code.
- Added offline push/PR CI for tests, structural validation, dependency checks, and JavaScript syntax.
- Removed 17 tracked bytecode files plus 35 generated Jupyter checkpoint/`.DS_Store` files (about 130 MB in the current tree) and ignores them going forward. These deletions remain recoverable from Git during review.

### Website

- The Upcoming Fights table distinguishes the primary forecast, market probability, independent model probability/status, and artifact provenance; missing or low-history data renders explicitly rather than as `NaN`.
- Betting-disabled status is visible, and the UI no longer links stale Bokeh explanations to a new model.
- The old browser-only 14-feature calculator is labeled experimental and separate. Its ISO-year/future-row date bugs were repaired, but it is not presented as the production artifact or given a current accuracy claim.
- Empty/updating cards and missing JSON now render safe states instead of crashing automatic fighter selection.

## Leakage-safe evaluation

The final model-v2 GitHub-equivalent disposable run passed in about 7 minutes
12 seconds (6 minutes 33 seconds for the updater plus tests and strict
validation). It
ingested 134 newly completed fights, producing 17,598 raw sides / 8,799 fights,
2,723 fighter IDs, and 8,644 eligible point-in-time W/L rows with 82 features.
The production model trained on 4,927 fights from 2016-08-20 through 2026-08-08
with `C=0.01` and calibration slope 1.13225. The reloaded, fingerprint-verified
artifact ID was `bbd8ee967ce46ab23de5`.

| Evaluation | Fights | Accuracy | AUC | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Nested 2023-2026 walk-forward | 1,852 | 64.15% | 0.6996 | 0.6297 | 0.2200 | 0.0260 |
| Newest-20% holdout | 995 | 64.92% | 0.7193 | 0.6194 | 0.2151 | 0.0537 |

The walk-forward folds selected `C=0.003` for 2023 and 2024 and `C=0.01` for
2025 and 2026. Relative to the same refreshed-data v1 run, the expanded grid
improved aggregate log loss from 0.630802 to 0.629725 and accuracy from 63.93%
to 64.15%. The newest holdout remained unchanged because its development
period still selected `C=0.01`.

For the August 22 card, all 13 matchups resolved to model probabilities (seven
normal-history and six low-history statuses) with zero abstentions. FightOdds
was serving the August 15 card, so its normal parser returned zero matching
lines; all 13 forecasts truthfully fell back to the stats model and all betting
statuses remained disabled. The incomplete same-day August 15 event was
deferred rather than partially ingested, leaving the validated source cutoff at
August 8 (seven days old).

These are historical estimates, not guaranteed future accuracy or evidence of profit. Log loss and Brier score are the primary promotion metrics because they penalize confident mistakes and evaluate probability quality; accuracy alone discards that information.

For context, the legacy 30-feature methodology scored about 59.1% accuracy and 0.708 log loss across comparable future-year folds. The large gap between its random and chronological results is why future cards, never randomly mixed fights, must be the outer evaluation unit.

### First-pass performance experiments

After the initial upgrade, fixed and strictly nested challengers were compared on
the identical 1,720-fight 2023-2026 walk-forward sample from the checked-in
snapshot ending 2026-05-09. Every original fold
selected `C=0.01`, the strongest regularization the code allowed. Extending the
grid to `(0.001, 0.003, 0.01, 0.03, 0.1)` showed that the first apparent
recency-weighting gain was mostly a proxy for missing stronger shrinkage.

| Challenger | Accuracy | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|
| Original regularization grid | 63.84% | 0.632912 | 0.221335 | 0.69602 |
| Expanded grid, uniform training | 64.07% | 0.631740 | 0.220906 | 0.69630 |
| Nested recency weighting | 63.90% | 0.631856 | 0.220968 | 0.69614 |
| Nested decision-reliability weighting | 64.19% | 0.632058 | 0.221051 | 0.69590 |
| Global Glicko rating + RD | 64.19% | 0.630488 | 0.220276 | 0.69906 |
| Five-year recency + Glicko | 64.42% | 0.630461 | 0.220221 | 0.69946 |

The expanded uniform grid improved aggregate log loss by 0.00117 and no yearly
fold worsened; 2023 and 2024 selected the now-interior `C=0.003`, while 2025 and
2026 retained `0.01`. It is the only promoted change because it fixes a search
boundary at almost no complexity cost.

The weighting grid tested half-lives of none, 8, 5, and 3 years and
split/majority multipliers of 1.0, 0.75, and 0.5. Weights were calculated only
inside each training fold, normalized to mean one, and applied to both scaling
and logistic fitting; calibration and scoring remained unweighted.

The best Glicko combination improved log loss by only 0.00128 versus the
properly retuned baseline. Its 90% event-block bootstrap interval for
candidate-minus-baseline log loss was `[-0.00305, +0.00050]`, 2026 worsened by
0.00039, and holdout accuracy fell. Recency added only 0.00003 beyond Glicko.
These challengers therefore remain unpromoted. Because these same historical
folds were used for model development, genuinely new events—not this table—are
the unbiased confirmation set.

The retained Glicko challenger specification is global rating plus rating
deviation, initial rating 1500, initial/maximum RD 350, minimum RD 50, a
four-year uncertainty-reset horizon, and the existing 0.75 split/majority
update weight. Ratings update simultaneously from pre-bout state; no-contests
inflate uncertainty and advance its clock without moving or contracting the
rating. Division Glicko and confidence-shrunk duplicates did not help.

## Market benchmark

On 312 completed historical forecasts with at least one usable two-sided book quote, the available no-vig consensus outperformed the old model on the same selected fights:

| Metric | Old stats model | Book consensus |
|---|---:|---:|
| Accuracy | 64.95% | 69.87% |
| Log loss | 0.641 | 0.587 |
| Brier score | 0.224 | 0.201 |
| AUC | 0.703 | 0.750 |

This is not a verified closing-line comparison: old records lack collection timestamps, only completed/matched fights are represented, and the sample is selected. It supports using market consensus as the required baseline, not claiming a durable betting advantage. A future residual design should be retained only if it improves rolling out-of-time log loss over raw no-vig market probability:

`logit(p_fight) = logit(p_no_vig_market) + residual(stats, form, matchup)`

### Expected-return first pass

The repository's main-line Git history was reconstructed conservatively using
the later of author/committer time as an upper bound, a strictly earlier UTC
calendar date than the event, stable-ID resolution to exactly one raw fight,
complete two-sided quotes, and per-book overround from 0.90 to 1.30. Same-day,
ambiguous, post-event, unmatched, and history-only cumulative quotes were
excluded. The result contains 803 strict fight snapshots across 58 cards and
3,718 immutable quotes from DraftKings, BetMGM, Caesars, BetRivers, and
FanDuel. The latest eligible snapshot produced 503 W/L fights and 9 draws/NCs;
2024 has no recoverable sample.

Only 230 of the 503 W/L fights had a same-capture, pre-event legacy forecast.
The prior-card evaluator required 12 completed cards and 100 prior fights,
leaving 119 untouched predictions:

| Forecast on identical fights | Accuracy | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|
| No-vig market | 71.43% | 0.58713 | 0.20096 | 0.75863 |
| Legacy stats model | 64.29% | 0.61856 | 0.21605 | 0.72781 |
| Prior-card-selected logit blend | 71.43% | 0.58836 | 0.20152 | 0.75891 |

The blend-minus-market log-loss difference was `+0.00123`; its card-block 95%
interval was `[-0.00690, +0.00903]`. The blend therefore did not establish any
incremental signal over market-only.

### Current-algorithm market replay

The legacy comparison did not answer whether the corrected production
algorithm adds information to consensus. A new stable-ID replay now generates
unrounded probabilities for the current 82-feature contract using nested
whole-year folds. Every test year uses a ten-year training window ending before
January 1, and regularization and symmetric calibration are selected within
that earlier window. Market orientation is reconciled by fighter IDs rather
than names.

All 503 recovered W/L market fights paired with an out-of-fold model forecast:

| Forecast on identical fights | Accuracy | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|
| No-vig market | 67.79% | 0.60152 | 0.20760 | 0.73672 |
| Current model algorithm | 65.61% | 0.62356 | 0.21716 | 0.71087 |

Model-minus-market log loss was `+0.02204`; its 58-card bootstrap 95% interval
was `[-0.00139, +0.04453]`. The market remains the stronger point estimate.
After the same 12-card/100-fight warmup used for prior-card weight selection,
399 fights remained. Their market log loss was `0.59937`; the selected blend
scored `0.59926`. The tiny `-0.00011` difference had a 42-card 95% interval of
`[-0.00060, +0.00040]`. Gamma was zero for 278 fights, 5% for 99, and 10% for
22; it never exceeded 10%. This does not establish incremental model signal.

The replay is stricter than applying today's artifact to old fights, but is
still development-only. The feature contract was designed with knowledge of
some evaluation-era outcomes, current reconciled facts/profile corrections can
postdate a fight, legacy commit time is not a provider timestamp, and 2024 is
missing. The immutable report and 503-row detail table are
`current_model_market_replay.json` and `current_model_market_replay.csv` under
`src/content/data/market_history_backfill/`.

### Style-matchup feature challenger

A single feature group was frozen before evaluation rather than selecting
individual features from test outcomes. It adds 30 columns to the unchanged
82-feature baseline: smoothed career attempt shares for head/body/leg and
distance/clinch/ground (offense and absorption), 12 nonlinear career/recent
striking, takedown, control, and power matchup terms, and six granular
offense-versus-absorption terms. Every addition is pre-bout and antisymmetric.
The challenger builder requires finite nonnegative granular counters, enforces
landed <= attempted, and verifies that both target and position partitions sum
exactly to the reported significant-strike totals. The current 17,622 raw rows
had zero missing granular cells and zero partition violations.

Both contracts were retuned and calibrated independently inside the same
nested whole-year folds. The baseline submatrix was also reproduced from raw
data before scoring, preventing an unnoticed data or feature-contract change:

| 2022, 2023, 2025, 2026 walk-forward | Fights | Accuracy | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|---:|
| Current 82-feature baseline | 1,857 | 63.70% | 0.63632 | 0.22309 | 0.68799 |
| 112-feature style challenger | 1,857 | 64.08% | 0.63590 | 0.22271 | 0.69088 |

The challenger-minus-baseline log-loss point difference was `-0.00042`; its
154-card 95% interval was `[-0.00336, +0.00247]`. On the 503 fights with market
history, the baseline, challenger, and market log losses were `0.62356`,
`0.61838`, and `0.60152`, respectively. The challenger's `-0.00518` improvement
over baseline had a 58-card interval of `[-0.01153, +0.00108]`, while it still
trailed market by `+0.01686` log loss.

After the prior-card warmup, the style/market blend scored `0.59891`, compared
with `0.59926` for the baseline blend and `0.59937` for market-only. The style
blend's differences versus baseline blend and market were `-0.00035`
(`[-0.00214, +0.00118]`) and `-0.00046`
(`[-0.00245, +0.00141]`). Neither interval excludes zero. The selector also
used style weights as high as 20%, making untouched prospective confirmation
especially important. The feature group is therefore retained, unchanged, as
a paper challenger; it does not replace the production artifact or enable
recommendations. Its immutable report/detail files are
`style_matchup_challenger.json` and `style_matchup_challenger.csv` beside the
baseline market replay.

The exploratory return policy excluded each target book from its consensus,
required three other books and predicted EV of at least 5%, selected one best
listed price per fight, and risked a hypothetical flat 1 unit. It made 26
selections: 3 wins, 23 losses, `-15.47u`, and `-59.5%` hypothetical ROI, with
maximum drawdown `16.12u`. The card-block 95% interval for profit per selection
was `[-100%, -11.73%]`. Orientation, result settlement, odds parsing, and EV
recomputation checks passed. The picks were underdog-heavy (average `+347`),
and the legacy model materially raised their probabilities above market.

A report-only fixed market-only comparator reused the same 117 eligible fights,
target-price universe, leave-one-book-out calculation, and 5% threshold. It
made 7 selections and lost all 7 (`-7u`, `-100%` quoted-price ROI). This tiny
sample does not establish that price shopping is intrinsically unprofitable;
it does show that neither the market-only nor blended rule passes a deployment
gate on the reconstructed history.

These are quoted-price counterfactuals, not realized or executable returns.
Git commit time is not a source quote timestamp, price availability/limits were
not verified, the legacy model probability was rounded, only matched completed
fights are represented, and the policy was not prospectively locked. The result
is still decision-useful: do not promote the blend, do not bet the apparent
legacy-model edges, and do not tune a threshold against these 26 outcomes.
Immutable outputs and all caveats are recorded under
`src/content/data/market_history_backfill/`.

## Highest-value remaining work

1. Rebuild the full legacy prediction history from immutable fight facts. Of 1,093 matched dated records, 51 stored `correct?` values disagreed with raw outcomes, and 397 of 1,491 rows have no date. New rows are fixed, but the legacy backfill still needs cancellation/rematch/alias review.
2. Remove the custom browser calculator or implement the exact artifact feature/scaler/calibration contract in a service or browser-compatible runtime. Do not maintain two user-facing models indefinitely.
3. Let the expanded Monday-Saturday collector accumulate native, timestamped multi-book snapshots. Evaluate market-only versus market-plus-stats and the locked early-favorite/late-underdog challenger on untouched future cards; report coverage, price movement, paper return, and CLV separately.
4. Preserve both the style-matchup v1 group and global Glicko rating/RD as fixed challengers, and reassess them only after genuinely new events accumulate. Test scheduled rounds/title status, weight-class changes, stance interactions, and line movement one at a time; the dataset is too small for complexity by default.
5. Add a slower periodic deep reconciliation beyond the recent-event window for very late overturns/corrections, and a bounded refresh cadence for inactive profiles.
6. Move large immutable datasets/model archives to releases, object storage, or Git LFS if desired. Removing generated files shrinks future checkouts, but the existing `.git` pack remains about 1 GiB unless history is deliberately rewritten.
7. Keep betting disabled until at least 500 eligible prospective fights across 40+ cards are settled, the blend beats timestamp-aligned market-only with a paired card-block interval below zero, and a predeclared paper policy has positive lower-bound return/price-quality evidence.

## Deployment checklist

1. Review the working-tree diff. Shared production CSV/JSON files were deliberately not overwritten during this implementation; all networked verification ran in disposable OS-temporary copies.
2. Commit and push the code/workflow/cleanup changes to the default branch.
3. Open **Actions -> Update UFC data**. Enable the workflow if GitHub still shows `disabled_inactivity`, then select **Run workflow** once.
4. Confirm the run passes the complete test suite, strict `--require-model-artifact` validation, and the starting-SHA publication guard.
5. Inspect the first bot commit: it should contain only the scoped processed/external data and model artifact paths.
6. Create a free The Odds API key and store it as the Actions repository secret `THE_ODDS_API_KEY`; never commit it. The maximum sixteen weekly requests use roughly 140 of the free tier's 500 monthly credits at the current two-region request cost.
7. Manually dispatch **Collect UFC market snapshot** once after the updater succeeds. Confirm it commits only the quote, forecast, source-timing, paper-decision mirrors and bounded report under `src/content/data/market/`.
8. Leave betting disabled and monitor both Actions summaries for raw/PIT counts, temporal metrics, card/model coverage, capture coverage, API credits remaining, and source-card mismatches.
