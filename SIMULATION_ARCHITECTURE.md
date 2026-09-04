# Evidence-first fight simulation

## Status and boundary

The fight simulator is an independent research challenger. It does not replace
or otherwise affect the production winner model, the candidate competing-risk
outcome model, or market decisions. Its frozen pre-event winner probability may
be included in predeclared paper-only comparisons with the model and market;
those comparisons cannot feed a prediction or decision. A read-only website tab may
show precomputed candidate distributions for an upcoming card, but it cannot run
arbitrary public simulations or feed any production decision. The simulator's
first job is to answer a scientific question: can one causally fitted generative
model produce better calibrated joint winner, method, duration, and fight-stat
distributions?

The boundary is explicit in each schema. Backtest and local-run status objects
carry `candidate_only: true`, `production_enabled: false`, and
`execution_enabled: false`; the reviewed research-status gate and every shadow
publication additionally carry `paper_only: true`. Parameter ensembles are
content-hashed model inputs and become eligible for a shadow only when a
separate reviewed status file names their exact hash. A negative evaluation is
still a useful result. No simulator output can influence production without a
separate reviewed code change after the promotion process in this document.

The implementation uses the repository, local Python, and standard runners in
the existing public GitHub repository. It introduces no hosted API, database,
object store, paid runner, new odds provider, or live wagering integration.

## Evidence and observability

UFCStats supplies completed-bout totals and per-round aggregates, not a sequence
of timestamped actions. The simulator therefore uses only coarse states and
strong pooling. A detailed narrative is never evidence that a probability is
well calibrated.

| Quantity | V1 evidence | V1 treatment |
| --- | --- | --- |
| Engagement and strike pace | Attempts and active fight minutes | Strongly pooled empirical rates plus a regularized age/experience/layoff pace multiplier |
| Strike target and broad phase | Head/body/leg landed counts and distance/clinch/ground attempt partitions | Strongly pooled composition; broad-phase partitions are not observed phase-time exposure |
| Accuracy and defense | Own and opponent landed/attempted pairs in each fighter's prior history | Strongly pooled empirical binomial ratios, not opponent-adjusted fighter effects |
| Takedowns | Attempts, completions, and whole-fight exposure | Pooled marginal attempt rate converted through the broad-phase composition proxy, plus empirical offense/defense resolution |
| Ground activity | Whole-fight control, ground strikes, submissions, reversals | Empirical pooled ratios feeding one coarse `GROUND` state; no invented positions or observed ground-time model |
| Power and durability | Knockdowns, KO/TKO result, landed strikes, exposure | Fighter knockdown/durability ratios with weak finish conversion kept at context/global level |
| Submission threat | Attempts and submission result | Attempt and finish/defense effects; no invented setup position |
| Pace decay | First-round and later-round attempt rates for fighters reaching those rounds | Strongly pooled empirical decay; later-round observations are survivor-selected and V1 does not correct that selection |
| Judging | Final decision result/type and simulated round effectiveness | Three latent judges with fixed global correlated-noise settings, not a fighter-specific or scorecard-fitted judging model |
| Injury or withdrawal | No reliable cancellation population | Excluded from V1 |
| Exact transitions or follow-up timing | Not observed; only interval-censored round totals | Never presented as observed causal conversion. Strongly pooled same-round associations may be audited as candidate predictors before a separate simulator-mechanics validation |

Official knockdowns are an observation as well as a latent fight mechanic.
`official_knockdown_observation_probability` may thin latent destabilization
events only when validated on a locked cohort. The latent event still drives
hurt, damage, effectiveness, finish conversion, judging, and termination; the
separate named observation RNG stream changes only the UFCStats-facing counter.
Its default is `1.0`, preserving previous trajectories and reports. This same
observation-layer pattern should be preferred for UFCStats control semantics
when changing latent phase dynamics would damage held-out outcomes.

Metadata-only external fights may update experience, activity, record, and rating
priors. They must not manufacture strike, takedown, control, or submission
parameters.

The current fitter is deliberately an empirical pooled model, not a fitted
hierarchical offense-versus-defense regression. Each event-card bootstrap member
recomputes global, division-era, and fighter sufficient-statistic ratios with
strong pseudo-exposure priors. A fighter's defense is estimated from what prior
opponents landed against that fighter, but V1 does not simultaneously estimate
or remove those opponents' offensive effects. Likewise, distance, clinch, and
ground strike partitions describe where recorded attempts occurred; UFCStats
does not provide time spent in those phases. V1 uses a pooled parent composition
only when scaling strike hazards. The stored shares are not measured occupancy
and are not themselves a fitted phase-transition policy. Whole-fight takedown
and submission attempt rates remain conservative marginal opportunity proxies;
they are never divided by strike composition.

The candidate-only `transition-audit` command evaluates whether strongly pooled
fighter and opponent histories improve a locked latest-card holdout over a
division/round context model. Its labels include `same_round_association` by
contract. Event-card block intervals must be wholly favorable before a target
can advance to a separate mechanics test; passing that audit does not itself
change a fitted parameter. The first 1,000-fight audit is documented in
`SIMULATION_TRANSITION_AUDIT_2026-08-27.md`.

The one transition that cleared that audit, same-round takedown/credited-control
association, is available only through the explicit
`--takedown-control-association` research fit. It maps own credited-control
share after a same-round takedown to coarse retention and opponent-conceded
share to coarse escape, with global/context/fighter pseudo-opportunities
25/25/12. This remains an interval-censored proxy, not measured action order or
top position. Its parameter-model version, fit input hash, compact recipe, and
materialized cache identity are distinct from the default model. The default
and production paths do not enable it. The paired mechanics screen and broad
accuracy audit are documented in
`SIMULATION_CONDITIONAL_CONTROL_AND_BREADTH_REPORT_2026-08-27.md`.

The snapshot layer also exposes research-only `full`, `context_only`,
`reliability_weighted`, `opponent_adjusted_v1`, and `opponent_adjusted_v2`
policies. Reliability
weighting applies a second parameter-specific causal exposure weight to the
already pooled fighter deviation on the parameter's natural scale.
`opponent_adjusted_v1` reconstructs each member's card bootstrap and estimates
two-way actor/opponent residuals for supported strike, takedown, and submission
observations. Neither policy mutates the shared bootstrap artifact.
The 229-fight ablation found weak directional fighter signal but no standalone
proper-score advantage over 50/50. It is documented in
`SIMULATION_FIGHTER_EFFECT_ABLATION_REPORT_2026-08-27.md`. The first opponent-
adjusted implementation then failed decisively because action-count precision
made fighter reliability far too high; it is documented in
`SIMULATION_OPPONENT_ADJUSTMENT_REPORT_2026-08-27.md`. It remains diagnostic
only. A second implementation must first pass a chronological, bout/card-
clustered observation-likelihood audit before spending another simulation
screen.

That audit is implemented by `opponent-adjustment-audit`. Fighter effects use
one equal-weight observation per fighter-side bout; ridge choices are selected
per target/model from the eight strictly preceding eligible cards; and Poisson
or binomial next-card likelihood is compared with physical event-card block
uncertainty. The first frozen audit passed overall and is documented in
`SIMULATION_BOUT_CLUSTERED_OPPONENT_AUDIT_REPORT_2026-08-27.md`. Passing permits
one bounded `opponent_adjusted_v2` development simulation screen, not production
or confirmation use. That v2 screen is complete and rejected. It retained
equal-bout precision and changed only the strike targets with favorable audit
intervals, but worsened winner, joint side/method, and strike-attempt
performance. Details are in
`SIMULATION_OPPONENT_ADJUSTMENT_V2_REPORT_2026-08-27.md`. No further opponent
simulation mode may run until a causal conditional-to-endogenous bridge audit
explains the failed parameter-to-engine translation.

## Data contract and causal fitting

When backfilled, the normalized round table has one row per physical bout,
fighter side, and round. It retains stable bout, event, fighter, and opponent
identities; round exposure; result context; and all supported UFCStats count
partitions. Round sums are reconciled with the existing doubled bout table.
Missing source rows or genuine UFCStats disagreements are explicit diagnostics,
never silent zeros. Fitting defaults to explicitly `matched` rows and verifies
their bout/event/fighter/opponent identity against the causal doubled bout
table. A programmatic `allow_legacy_unreconciled_rounds=True` override exists
only for isolated research migration; normal fit and shadow paths do not enable
it.

Local round backfill is both count-bounded and wall-clock-bounded. It atomically
checkpoints whole physical fights, resumes from stored stable IDs, and rejects a
`--max-runtime-seconds` value over 3,300 seconds. A source call still uses the
central client's finite connect/read timeouts; callers should leave headroom
between the command budget and the one-hour research ceiling.

Historical evaluation uses expanding calendar-year folds. Each fold fits one
artifact at January 1 from rows strictly before that cutoff, and every test bout
in that year uses the same frozen fold artifact. This is causal and avoids
same-card leakage, but it is deliberately conservative: an earlier event in the
same test year does not update a later test-year snapshot. The experience slice
is computed separately from each fighter's strictly earlier UFC event dates;
same-day bouts are excluded when order is ambiguous. Tests require that
appending future fight rows, with fixed profiles and provenance time, leaves an
earlier artifact and snapshot unchanged.

Each bootstrap member contains global, division-era, and fighter empirical
parameters. Round data currently affects the strongly pooled pace-decay ratio;
age, prior UFC experience, and layoff enter a regularized log-rate adjustment.
The base artifact has no simultaneous opponent-adjusted offense/defense fit.
The rejected `opponent_adjusted_v1` snapshot transform is explicitly separate
from that artifact and cannot be publication eligible. A publication-eligible
frozen artifact must contain 200 event-card block-bootstrap replicas.
Historical screening uses 64 replicas and reruns borderline primary comparisons
at higher precision. One outer draw selects the same replica for the global
model and both fighters so covariance is not broken.

Bootstrap percentiles are called **parameter/model uncertainty intervals**.
They are not Bayesian credible intervals. Inner-path Monte Carlo standard error
is calculated and reported separately.

## Engine contract

The engine has four phases: `DISTANCE`, `CLINCH`, `GROUND`, and `SCRAMBLE`.
Hurt, fatigue, and damage are bounded dynamic variables rather than additional
unobserved positions. The initial phase is distance.

Hazards are piecewise constant within five-second dynamics episodes. An action
receives a continuous timestamp from its waiting-time draw. A dynamics tick or
round bell preempts an action beyond that boundary. Round scoring, between-round
recovery, and termination are explicit immutable events. Official fight time
excludes rests even though recovery events remain in the trace.

Action policy, resolution, consequence, state reduction, scoring, and
termination are separate contracts. The supported action families are strikes,
clinch transitions, takedowns, coarse ground activity, escapes/reversals,
submission attempts, and recovery. The terminal contract preserves side by
KO/TKO, submission, decision, other, draw, and no-contest rather than
redistributing non-win outcomes.

The display run ID is the SHA-256 of the complete canonical run specification.
Named random streams use a separate common-random-number key containing the RNG
contract, root seed, matchup ID, parameter-artifact ID, bootstrap member,
simulation index, and stream name, and use NumPy `PCG64DXSM`. Consequently a
parameter change within the same named artifact can reuse the same underlying
draws, while changing worker count, chunk size, completion order, or telemetry
level cannot change a result. Global NumPy randomness and Python `hash()` are
forbidden.

Every state change in a full diagnostic trace is reducible from immutable
events. The normal `none`/`compact` execution path retains terminal result,
statistics, and final state hash but does not allocate individual events. A
selected `full` trace additionally retains event deltas, string-normalized
payload diagnostics, state hashes, probabilities, and named RNG calls. Its hash
chain detects mutation. Reducer replay reconstructs the final state from full
events; stochastic replay regenerates the same full trace from the run
specification and simulation index.

## Aggregates and telemetry

Exact conditional counts by bootstrap replica are authoritative. Publications
derive winner/method marginals, method-by-round matrices, duration histograms
and survival, valid half-round totals, decision types, and supported fight-stat
distributions from those counts.

For outer replica `j` and `m` inner paths, the marginal probability is the mean
of the conditional proportions. Process MCSE uses the nested conditional
variance rather than treating paths that share a parameter draw as independent.
The distribution of conditional replica probabilities supplies the separate
parameter/model interval.

Publication-eligible upcoming shadow runs begin with 200 replicas by 512 inner
paths (102,400 total). Inner paths double together, to at most 409,600, when
headline MCSE exceeds 0.2 percentage points, equal-sized recent batches disagree
beyond three combined standard errors, or parameter quantiles remain unstable.
A failed or incomplete shadow is withheld transactionally. A local `run` may
retain explicitly labeled nonconverged diagnostics only when the caller passes
`--allow-nonconverged-research`; that override can never make the output shadow-
eligible.

The nested runner reduces paths into mergeable exact counters as chunks finish.
Its only path-sized aggregate buffer is packed 64-bit fight duration; ordinary
high-volume runs do not retain all `SimulationPath` objects. A bounded candidate
pool supports deterministic trace selection, and the compatibility path API may
still retain paths for small diagnostics. At most 32 paths are selected across
populated outcome/round strata and rerun with full tracing. An invariant failure
is deterministically retried with full telemetry and aborts publication rather
than being converted into a partial result.

Population research uses three compute fidelities. A quick screen uses five
development cards, 16 bootstrap members, 512 total paths per matchup, and one
seed. Confirmation uses 15 cards, 32 members, 2,048 paths, and one seed. Only
the final 64-member, 4,096-path, repeated-seed run is selection-eligible. All
candidates in a stage retain common run seeds, and higher precision is spent
only on survivors or borderline comparisons.

Long posterior runs write one atomic, contract-hashed checkpoint per completed
fight/seed pair and can resume without changing worker or chunk invariance. A
shared ignored cache stores exact materialized member columns for each causal
event cutoff, data fingerprint, fit configuration, and parameter-model
version. Published parameter artifacts keep their self-contained causal fit
recipe; only the local performance cache chooses the faster materialized
representation. Phase timings separate input fingerprinting, causal fit/cache
loads, and simulation.

`posterior-backtest --max-runtime-seconds` is capped at 3,300 seconds. The
runner stops between complete fight/seed pairs, reports the planned and
completed cohort separately, and never folds a partial matchup into metrics.
A 100-total-path configuration may be used for broad descriptive accuracy when
the path count is divisible across bootstrap members. Such runs are always
screening-only: probabilities are quantized, rare joint outcomes may receive
zero paths, and a single seed cannot estimate end-to-end simulation noise.

Upcoming-card runs add a finer restart boundary because one 200-member matchup
can itself be expensive. After every adaptive doubling stage, all bootstrap
members have completed the same contiguous simulation-index range. At that
boundary the runner atomically stores exact integer counters, member-level
counts, canonical packed duration values, convergence history, the full
run/spec hash, engine and named-RNG contracts, and accumulator/aggregate hashes.
`upcoming-card --resume` validates the immutable card/input/scientific contract,
reuses durable matchup results, and continues only the next untouched index
range. Worker and chunk settings are operational rather than scientific and may
change. Half-completed batches are rerun, never merged as partial authority.
The website file is written only after all matchups are available or explicitly
withheld.

The Python implementation remains the authoritative reference. Its bulk path
uses cached immutable hazard bases, a direct lean-state clock/dynamics reducer,
and a seed-prefix implementation that is byte-identical to the named RNG
contract. `benchmark` verifies aggregate hashes across worker counts. A compiled
bulk backend is attempted only when a supported local compiler or JIT exists,
crosses the language boundary in batches, demonstrates a material speedup, and
passes deterministic/reference equivalence gates; Python fitting, telemetry,
replay, and analysis remain authoritative.

The full local aggregate includes exact per-bootstrap statistic histograms and
is written as content-addressed gzip under ignored
`artifacts/simulations/shadow-authority/<event>/`. A public
`compact_shadow_v1` projection omits only those large member-level statistic
histograms. It retains overall exact statistic histograms, per-bootstrap outcome
counts, outcome/duration/total distributions, and process and parameter/model
uncertainty. The compact object records both the omitted-field manifest and the
SHA-256 of the full aggregate. On a local run that hash points to the ignored
authority file; on a standard ephemeral Actions runner the file is neither
committed nor uploaded, so the hash is a deterministic replay commitment. The
immutable card JSON also hard-fails above 16 MiB. Historical seed ledgers use an
analogous compact projection and authority hash. No large ledger, full trace
population, or local authority file is retained as a paid workflow artifact.
Both full and compact objects are normalized through their actual JSON scalar
and mapping-key representation before hashing. This is required for numeric
histogram keys: the committed hash must be independently reproducible from the
stored JSON, not merely from the pre-serialization Python object.

## Evaluation and promotion

### Posterior-predictive fight validation

`python -m fight_sim validate-fight` compares one completed, causally fitted
simulation run with the repository's mirrored UFCStats bout totals. Exact
Monte Carlo count distributions are authoritative. Each aligned marginal
reports central interval coverage, a discrete PIT interval and midpoint,
inclusive one- and two-sided predictive tail probabilities, standardized
residual, predictive point mass, and CRPS. The simulator preserves significant
strike attempts and landed strikes by distance, clinch, and ground phase for
this purpose.

The observability boundary is explicit: UFCStats distance strikes combine
punches and kicks, while UFCStats control is broader than simulated ground
top-position time. These comparisons remain labeled as partial rather than
being silently treated as identical measurements. Marginal tail probabilities
are not multiplied into a pseudo joint likelihood because the current
streaming ledger does not retain the full correlated statistic vector.

Scaling this diagnostic must use chronological out-of-sample fights and score
predeclared families of statistics with aggregate CRPS, PIT calibration,
central-interval coverage, and event-card block uncertainty. Parameter or
mechanic changes are selected on training folds and retained only when they
improve held-out predictive checks without harming the primary side-by-method
metric. A single well-matched historical fight is never a tuning target by
itself.

The recent-event posterior-predictive study uses an event-date cutoff and
refits the complete parameter ensemble before every scored card. Its primary
cohort requires at least three strictly prior UFCStats bouts for each fighter;
UFC debuts and either-side histories of only one or two bouts are excluded.
They may later form a separately labeled prior-only study, but they cannot tune
fighter-specific mechanics. A five-prior-bouts-per-side sensitivity cohort is
reported from the same forecasts. Card selection happens before the exposure
filter so the manifest records every low-information exclusion on each of the
newest complete cards.

The population diagnostics retain deterministic randomized PIT values,
two-sided predictive tails, standardized residuals, CRPS, and 50/80/90/95%
central interval coverage for duration and every observable statistic. Coherent
path-level sums and red-minus-blue differentials are aggregated directly from
each trajectory rather than reconstructed from marginal distributions. KS and
Cramer-von Mises p-values are nominal iid diagnostics only; reports explicitly
warn that they neither establish a probability that reality came from the
simulator nor account for card clustering and multiple comparisons.

Outer evaluation uses expanding chronological calendar-year folds. The default
repository command is a bounded 200-fight screen distributed across 2017--2026,
not a claim of full incumbent-horizon coverage. Each fold refits its own causal
global model and snapshots. Historical screening uses roughly 4,096 paths per
matchup, repeats seeds to quantify simulation noise, and escalates borderline
joint comparisons to 16,384.

The predeclared primary selection score is joint side-by-method log loss.
Secondary scores are winner log loss/Brier/calibration, method log loss,
duration CRPS/integrated Brier, available totals log loss, and count-distribution
CRPS/coverage. Comparators are population/division joint baselines, the
production winner model, the candidate competing-risk winner and full
side-by-method forecasts, and timestamp-aligned moneyline and full-fight-total
market probabilities. Total-market comparisons score market and simulation on
the same settleable line/fights with log loss, Brier score, and paired card-block
intervals. The experience slice uses the less-experienced fighter's strictly
prior UFC fight count. Paired uncertainty resamples whole event cards.

A validated simulator may begin append-only pre-event shadow forecasting while
remaining execution-disabled. Shadow generation requires a separately reviewed
status file that names the exact frozen parameter and causal-backtest hashes,
preserves the candidate/paper-only flags, and explicitly sets `shadow_enabled`.
Those declarations cannot substitute for evidence: the loader also requires a
200-member ensemble based exclusively on matched round rows, at least three
chronological folds and 200 scored historical fights, at least two independently
hashed 4,096-path seeds, the declared primary metric, 99% population/division
joint coverage, 90% production-winner and competing-risk-joint coverage, and at
least one aligned moneyline and full-fight-total comparison. This gate permits
paper shadow collection; it does not assert that the simulator beat an incumbent
or satisfy production promotion. A symmetry-preserving stack may be frozen only
after retrospective improvement:

```text
logit(p_stack) = beta_model * logit(p_model)
               + beta_sim   * logit(p_sim)
```

The intercept is zero and both regularized coefficients are nonnegative. This
preserves complementary probabilities when fighter sides are swapped. The
implemented evaluator cross-fits weights by calendar year: only earlier
out-of-fold simulator and production-model predictions may train each test-year
stack. It requires 100 prior jointly covered fights and uses a predeclared L2
penalty of 0.01 centered on incumbent-only weights `(1, 0)`, so unsupported
simulation signal shrinks toward the existing model. Reports retain every
fold's coefficients, same-fight log loss/Brier/calibration, event-card block
intervals, and sensitivity to independent simulation seeds. A favorable
retrospective flag can freeze a research candidate only; it cannot enable
production or execution.

Beginning with events on or after 2026-09-01, a simpler prospective comparison
also freezes the simulator probability at the immutable market T-24 decision,
but only when that exact matchup's simulation already existed. It scores the
three inputs, their three fixed equal-weight pairings, and an equal-third
market/model/simulator pool in log-odds space. Withheld or late simulations are
never imputed. This measures whether the simulator adds information; it is not
the fitted stack above and has no production influence.

Production promotion requires at least 200 prospectively settled physical
fights across 20 events. At a predeclared checkpoint, the event-block 95%
interval for paired winner-log-loss improvement must be wholly below zero,
Brier score must not worsen, calibration intercept must remain within +/-0.05,
and calibration slope within 0.85--1.15. Passing creates a report; it never
changes production automatically.

## Local operation and analysis

### Versioned outcome mechanics

The default `legacy_v1` outcome mechanics remain replay-compatible. The opt-in
`two_route_v2` candidate models an official knockdown as an exchange-level
hurdle capped at one, then separates KO/TKO after an official KD from KO/TKO
without an official KD. The latter is necessary because official UFCStats data
contains a material population of KO/TKO wins with zero winner knockdowns.
Neither immediate route feeds the current strike's newly written hurt/damage
state back into its own finish probability. Route counts and phase times are
stored as path diagnostics and reduced into exact aggregate distributions.

`SIMULATION_EXPERIMENT_COHORTS_V1.json` seals chronological development,
confirmation, and final-holdout card identities plus the eligible fight-ID
checksum. `compare-outcome-mechanics` accepts only identical complete cards.
The v2.1 profile is development-selected and cannot become the default until it
passes both untouched cohorts. See
`SIMULATION_OUTCOME_ENGINE_V2_REPORT_2026-08-27.md`.

The research interface supports fitting, chronological backtesting, arbitrary
local runs, deterministic replay/reduction/diff, and generation of a
self-contained HTML analysis report. Reports cover convergence, process versus
parameter uncertainty, outcome/duration/stat distributions, calibration,
baseline comparisons, coverage warnings, and step-through trace inspection.

From the repository root, set `PYTHONPATH=src` and use these interfaces:

```text
python -m fight_sim backfill [--max-fights 1..100] [--checkpoint-every 1..25]
python -m fight_sim fit [--bootstrap-members 1..200] [--output PATH]
python -m fight_sim backtest [--bootstrap-members 1..64]
    [--paths-per-matchup 1..16384] [--max-fights 1..500]
    [--workers 1..64] [--chunk-size 1..4096] [--output PATH]
python -m fight_sim posterior-backtest [--last-events 1..100]
    [--skip-latest-events 0..99]
    [--min-prior-ufc-fights 0..100] [--bootstrap-members 1..64]
    [--paths-per-matchup 1..16384] [--seed-repeats 1..4]
    [--quick-screen | --confirmation-screen] [--resume]
    [--fit-cache-dir PATH | --no-fit-cache] [--output-dir PATH]
python -m fight_sim benchmark SPECS [--paths-per-member N]
    [--workers 1,2,4,8] [--repeats N] [--output PATH]
python -m fight_sim derive-mechanics POPULATION_RUN [--holdout-latest-events N]
python -m fight_sim select-mechanics BASELINE --candidate LABEL=REPORT ...
python -m fight_sim validate-mechanics BASELINE TUNED_HOLDOUT
python -m fight_sim select-finishing BASELINE --candidate LABEL=REPORT ...
python -m fight_sim validate-finishing BASELINE TUNED_HOLDOUT
python -m fight_sim upcoming-card [--simulator-config PROFILE]
    [--parameter-artifact PATH] [--bootstrap-members 1..200]
    [--minimum-prior-ufc-fights 0..100] [--website-output PATH] [--resume]
python -m fight_sim run --red-fighter-id ID --blue-fighter-id ID --division NAME ...
python -m fight_sim replay (--trace TRACE | --spec SPECS ...)
python -m fight_sim reduce TRACE
python -m fight_sim diff EXPECTED ACTUAL
python -m fight_sim analyze RUN_DIRECTORY_OR_AGGREGATE
python -m fight_sim validate-fight RUN_DIRECTORY --fight-id UFCSTATS_FIGHT_ID
python -m fight_sim gui RUN_DIRECTORY
```

The `gui` command is a local-only Qt/Matplotlib desktop explorer installed via
the separate `requirements-gui.txt`; ordinary fitting, backtests, shadow jobs,
and production updates do not install or import those dependencies. It consumes
the same exact aggregate counts, convergence records, validation diagnostics,
and bounded deterministic trace selection as the CLI/HTML tools. Thus charts
are views of the authoritative artifact rather than a second analysis
calculation. `run --launch-gui` is the convenience path for simulation followed
by interactive inspection. Pan/zoom and local image export are supplied by the
embedded Matplotlib navigation toolbar.

The checked-in research/publication contract is deliberately separate from the
ignored working tree:

```text
src/content/data/processed/ufc_fight_round_stats_doubled.csv
src/content/data/simulation/
  parameter_model.json.gz
  backtest_report.json
  research_status.json
  shadow_forecasts/<date>_<event>_<publication_sha256>.json
  upcoming_matchups/<simulation_input_sha256>.json
    # immutable automatic 4,096-path previews captured when a fight is discovered
src/content/data/external/simulation_forecasts.json
  # compact candidate/paper-only view of every currently announced event
artifacts/simulations/upcoming-card/
  # ignored exact aggregates and fitted pre-event ensemble
```

The status schema is version 1 and requires `candidate_only: true`,
`paper_only: true`, `production_enabled: false`, `execution_enabled: false`,
`integrity_gate_passed: true`, `causal_backtest_gate_passed: true`, an explicit
Boolean `shadow_enabled`, and exact `parameter_artifact_sha256` and
`backtest_report_sha256` values. The research CLI and workflow never manufacture
or edit this review decision. Strict bundle validation is available through
`python -B src/validate_data.py --allow-stale --require-simulation-artifact`.

The reviewed high-precision shadow bridge remains disabled when the status is
missing or `shadow_enabled` is false. It is separate from the automatic website
preview and from production predictions.

After each scheduled updater discovers all announced UFCStats cards, the
dependent `upcoming_simulations` job compares those stable matchup identities
with `simulation/upcoming_matchups/`. Eligible fights without a matching record
receive 64 fitted parameter replicas and 64 paths per replica (4,096 paths).
The shared parameter fit is performed once for the batch. Each completed fight
is written atomically to its own immutable, self-hashed record; later runs reuse
it. Low-history fights are reconsidered on every updater rather than receiving
fabricated statistics. A bounded run may leave work visibly queued, and the next
scheduled run continues with only the missing fights. The website file is then
rebuilt from the current announced schedule plus those records. Data-only commits
that arrive during simulation are rebased safely; any intervening code change
stops publication. This free standard-runner path never changes the production
model or enables wagering.

The manual `simulation-research` workflow uses only `ubuntu-24.04`, at most two
worker processes, bounded research inputs, and a six-hour ceiling. That ceiling
is a safety bound, not a guarantee that the largest 500-fight/16,384-path request
will finish. It has no schedule or write permission. Output stays on the
ephemeral runner unless the operator explicitly requests one validated file;
the transfer hard-fails above 5 MiB and expires after three days. Frozen
parameter ensembles use a versioned, self-contained gzip encoding. A fitted
artifact stores compact causal input frames, the fit/bootstrap recipe, and its
logical metadata instead of repeating every expanded member value. Loading
reruns that frozen recipe without consulting mutable repository files, must
reproduce the exact members and logical metadata, and validates the original
model-content SHA-256. A sealed wrapper separately commits to the physical
payload, embedded inputs, and exact logical member values. Weekly pre-commit
validation and upload staging inspect those commitments without refitting;
research execution and the required publication gate fully materialize them.
Legacy row-oriented JSON and exact-columnar artifacts
remain loadable. The current repository's measured 200-member gzip is under
1 MiB, so the workflow can transfer it when schema validation and the 5 MiB
cap pass. The workflow cannot upload ledgers, trace populations, HTML reports,
the accumulated round table, a status gate, or production forecasts. No paid
runner or separately provisioned cloud service is part of this path.

Compact frozen model, evaluation, and shadow artifacts may be reviewed and
committed under the exact bundle paths. Detailed ledgers and ad hoc reports live
under ignored `artifacts/simulations/` and can be regenerated from their run
specifications.

## Website boundary and deferred roadmap

The implemented website surface is intentionally narrow: a dark-mode Simulation
tab reads one precomputed, content-hashed JSON covering every announced event,
grouped chronologically with the main event first. It shows winner,
side-by-method, duration, total-round, method-by-round, decision-type, projected
statistic, process-error, and bootstrap-parameter distributions. Current
moneylines may be shown as raw research context, but method EV is never implied
without real synchronized method prices and settlement contracts. Matchups with
fewer than three prior UFCStats bouts on either side are displayed as withheld,
not simulated from fabricated detailed statistics. Newly discovered eligible
fights are displayed as queued until their atomic 4,096-path preview is complete.

The first population calibration used the newest 20 completed cards, retained
133 fights where both sides had three strictly prior UFCStats bouts, and split
the cards chronologically into ten action-development, five selection, and five
untouched validation cards. Observable action rates were adjusted only with
global multipliers. Finish conversion was screened separately on the selection
cards, where a 0.40 KO/TKO finish-after-knockdown multiplier minimized duration
CRPS while satisfying joint, method, winner, and action-preservation gates. On
the untouched 31-fight validation cohort it improved joint side-by-method log
loss 2.0162 -> 1.9509, method log loss 1.2787 -> 1.2178, winner log loss 0.7534
-> 0.7383, duration CRPS 268.2 -> 255.9 seconds, observable-action error 0.3213
-> 0.2893, and mean duration bias -87.4 -> +10.0 seconds. It was therefore
retained as mechanics profile `mechanics-8ba01f34444f`.

This validation compares simulator versions; it is not a production-promotion
result. Winner calibration remained weak on only 31 fights (intercept 0.208,
slope 0.206), mean UFCStats control time was materially underpredicted, and
bootstrap parameter intervals remained wide. Upcoming publications consequently
retain candidate/paper-only labels and a no-wager warning. Any future adjustment
to control/phase semantics or winner probabilities must repeat chronological
selection and untouched validation rather than fit the current upcoming card.

Still deferred are arbitrary visitor execution, arbitrary fighter selection,
division and three/five-round controls, progressive browser refinement, sampled
illustrative traces, and offline caching.

The settled no-extra-service direction is high-precision precomputation in
GitHub Actions plus on-device browser execution for custom fights, with a
service worker and IndexedDB for offline use. Browser execution cannot ship
until it passes the authoritative engine's golden trace, aggregate, swap, and
RNG parity tests. No Cloud Run, Firestore, hosted queue, or other paid runtime is
part of that roadmap.
