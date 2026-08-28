# Fight simulation continuation TODO

Status updated 2026-08-27 after the bout-clustered v2 simulator screen.
This file is the handoff contract for the next simulation session.

## Frozen next action: conditional-to-endogenous strike bridge audit

Do not run an `opponent_adjusted_v3` Monte Carlo screen yet. First implement a
fast, causal bridge audit on the open `development_2024` cohort that explains
why an opponent model that improved strike observation likelihood made the
simulator's strike-attempt distribution and fight-outcome ranking worse.

For every outer fight and bootstrap member, compare:

1. the audit's conditional-on-observed-duration strike pace/accuracy forecast;
2. the effective v2 snapshot matchup intensity after the exact engine offense,
   defense, phase, and mechanics transforms, still evaluated at observed
   exposure; and
3. the existing full and v2 endogenous simulated attempts/landed distributions.

Verify signs and magnitudes separately for actor offense and opponent
vulnerability, quantify whether phase allocation or the endogenous finish
clock causes the divergence, and measure whether each predicted strike
differential ranks the actual winner. Use only strictly earlier cards and
event-card uncertainty. This should execute no new fight trajectories and stay
well below one hour. Another simulator screen is authorized only if this audit
identifies a concrete translation error or a prospectively specified mapping
that improves both conditional strike likelihood and winner ranking.

## Completed: bout-clustered v2 simulator screen

The outcome-engine v2.1 work remains frozen in
`SIMULATION_MECHANICS_TWO_ROUTE_V2_1.json`. Its separate confirmation
requirement has not been waived. Before spending the locked 2025 cohort on a
fighter-parameter choice, the open 229-fight / 30-card development cohort was
used to compare full fighter snapshots, division/era context only, and a
second causal exposure-weighted shrinkage step. The design and results are in
`SIMULATION_FIGHTER_EFFECT_ABLATION_REPORT_2026-08-27.md`.

Reliability weighting raised point accuracy from 52.63% to 55.70%, improved
winner log loss from 0.72135 to 0.71470, Brier from 0.26127 to 0.25863, and
joint log loss from 1.98694 to 1.96508. It also preserved the broad v2.1
method/duration/action realism. These changes were not statistically resolved:
all event-card proper-score intervals crossed zero, the 55.70% accuracy Wilson
interval included 50%, and log loss/Brier remained worse than constant 50/50.
Context-only merely collapsed toward chance and lost useful method structure.
No snapshot mode is validated or production-eligible.

That opponent-adjusted screen is now complete and documented in
`SIMULATION_OPPONENT_ADJUSTMENT_REPORT_2026-08-27.md`. It finished all 229
fights in 18.4 minutes, but was decisively rejected: accuracy fell to 46.49%,
winner log loss rose to 0.82197, Brier rose to 0.30308, and calibration slope
became -0.192. Paired event-card intervals for winner log loss, Brier, and
joint log loss were wholly harmful against both `full` and
`reliability_weighted`. The trajectory's method, duration, and action
distributions were broadly preserved, isolating the failure to fighter-side
ranking.

The first estimator assigned about 0.80--0.92 mean reliability to strike
actor/opponent effects because action counts acted too much like independent
evidence. Do not tune a scalar attenuation on these same trajectories. Before
another simulator run, implement a bounded cross-fitted audit that scores
opponent-adjusted strike/takedown/submission observation models on next-card
likelihood, with uncertainty and effective sample size clustered by physical
bout/event. Select any small regularization grid only inside nested
chronological training splits. An opponent model may enter another 229-fight
simulation screen only if it first improves its directly observed held-out
targets over both context-only and marginal-fighter baselines.

That audit is now complete and documented in
`SIMULATION_BOUT_CLUSTERED_OPPONENT_AUDIT_REPORT_2026-08-27.md`. It scored all
229 fights in 103.6 seconds without simulation. The tuned marginal model beat
context by 7.38% equal-target relative held-out loss. Bout-clustered opponent
adjustment improved another 0.795%, with a wholly favorable event-card interval
of `[-1.224%, -0.361%]`. Strike pace and accuracy improved significantly;
takedown point estimates improved but intervals crossed zero; submission pace
worsened 0.576% but stayed inside the predeclared one-percent harm limit.

The audit therefore authorizes one `opponent_adjusted_v2` simulator screen on
the same open cohort. Reconstruct each cutoff's eight-prior-card target-specific
ridge selection, retain equal-bout precision inside the matching card-bootstrap
member, and reuse the v2.1 mechanics/common seeds. Compare against both `full`
and `reliability_weighted` at 100 paths/fight under a 3,300-second hard cap.
Do not reuse v1's action-count reliability and do not tune away its adverse
submission result using the outer observations.

That screen is now complete and rejected. The separately named v2 mode applied
only strike pace and accuracy because those were the two targets with wholly
favorable audit intervals; takedown and submission parameters stayed on the
full snapshot. All 229 fights / 30 cards completed at 100 paths per fight in
20.1 minutes. Accuracy fell to 48.68%, winner log loss rose to 0.75362, Brier
rose to 0.27666, calibration slope became -0.084, and joint side/method log
loss rose to 2.13880. Against `reliability_weighted`, event-card intervals were
wholly harmful for winner log loss, Brier, and joint loss. Strike-attempt CRPS
also worsened and underprediction grew to 44.36 attempts. Do not make v2 the
default, do not open either 2025 cohort, and do not tune its effect size on
these same outer fights. Full results are in
`SIMULATION_OPPONENT_ADJUSTMENT_V2_REPORT_2026-08-27.md`.

Keep `confirmation_2025_a` and `final_holdout_2025_b` unopened until a single
fighter-parameter model survives that development comparison. Then confirm the
combined frozen choice at materially more than 100 paths per fight and at least
two independent seeds. No website or production simulator profile changes are
authorized by this research.

## Completed and safe to use

- The 20-card posterior-predictive study, chronological tuning split, and final
  untouched five-card validation are complete.
- `mechanics-8ba01f34444f` is the retained candidate-only mechanics profile. It
  includes the action-rate calibration and the validated 0.40 global KO/TKO
  finish-after-knockdown multiplier.
- The website Simulation tab is implemented. It shows winner and model/market
  comparison, side by method, finish time, settled over/under probabilities,
  method by round, decision type, bootstrap uncertainty, Monte Carlo error, and
  projected fight-stat summaries.
- The current checked-in website publication uses the retained
  `mechanics-8ba01f34444f` profile. Four matchups are available, seven are
  withheld because at least one fighter has fewer than three prior UFCStats
  bouts, and Yan-Gomes plus Tsuruya-Borjas are withheld for maximum-path
  nonconvergence. Its publication hash is
  `f71326805d560c5d859b42a9ce2a87ae31fe0ba52fa868ff9868bba7d8fd6609`.
- The final-profile run uses the validated 200-member pre-event parameter
  artifact stored locally at:

  `artifacts/simulations/upcoming-2026-08-29-finish-200/parameter_model.json.gz`

## Completed Aug. 29 card refresh

The first legacy compact-artifact access took about 25 minutes to
deterministically re-materialize and populated an 89 MiB content-addressed
cache. The resumed card run is complete under
`artifacts/simulations/upcoming-2026-08-29-finish-200-rerun/`. It produced four
publishable forecasts: Nurmagomedov-Song and Aoriqileng-Asakura used 409,600
paths each, Perez-Sumudaerji used 102,400, and Jenkins-Woodson used 204,800.
Yan-Gomes and Tsuruya-Borjas each completed 409,600 paths and remain available
locally for diagnosis, but were correctly withheld from the website after
missing the parameter-quantile stability gate. No adaptive checkpoint remains.

## Next implementation work (in order)

1. **Completed:** content-addressed materialized parameter caches now preserve
   exact member commitments. A newly fitted card populates the cache directly;
   the first access to an older recipe artifact reconstructs it once and later
   accesses decode member columns without refitting.
2. **Completed:** the exact nested accumulator is atomically checkpointed after
   every member-balanced adaptive batch, including the run/spec and engine/RNG
   contracts, member counts, convergence history, and accumulator/aggregate
   hashes.
3. **Completed:** `upcoming-card --resume` validates an immutable run manifest,
   reuses completed matchup records, and resumes the next exact simulation-index
   range. Tests prove aggregate equality across direct/interrupted runs and
   changed worker/chunk layouts, plus checkpoint corruption rejection.
4. **Completed:** reran all six history-eligible Aug. 29 matchups with
   `mechanics-8ba01f34444f`, 200 bootstrap members, and 512 to 2,048 paths per
   member. The seven low-history fights and both maximum-path nonconvergences
   were withheld rather than publishing partial or unstable results.
5. **Completed scientifically:** atomically replaced
   `src/content/data/external/simulation_forecasts.json`, validated its hash and
   compact/full authority linkage, and reran website contract/mobile-overflow
   regression checks. JSON normalization now makes numeric mapping keys hash
   identically before and after serialization; all six local full aggregates
   match their content-addressed filenames and compact links. Live screenshot
   verification remains outstanding because the in-app browser sandbox was
   unavailable in this session; do not record that visual check as passed.
6. Continue chronological research on the known remaining deficiencies:
   winner calibration, UFCStats-control versus simulated-ground-top-control
   semantics, total strike-attempt dispersion, and wide fighter parameter
   intervals. First audit the parameter-quantile convergence rule on historical
   validation fights: measure retest stability and false-withhold frequency,
   then compare any candidate convergence statistic under the existing
   chronological selection/untouched split. Do not tune this or the mechanics
   on the upcoming card.
   The bounded 2026-08-26 control experiment is complete and documented in
   `SIMULATION_TUNING_REPORT_2026-08-26.md`. Reducing the escape hazard to
   `0.67` was rejected: it reliably improved control CRPS but reliably worsened
   winner log loss and ground-strike CRPS on the two-seed five-card
   confirmation. Keep `escape_hazard_multiplier = 1.0`. Next test a separate
   UFCStats-control observation model without changing latent fight dynamics.
   A follow-up low-hanging-fruit round also tested winner-temperature
   calibration, a `0.45` knockdown-only profile, and a coupled `0.45`
   knockdown / `0.71` conditional-finish profile. The temperature collapsed to
   50/50 and the knockdown-only profile failed the primary screen. The coupled
   profile improved all headline confirmation point estimates and nearly
   eliminated knockdown mean bias, but the formal validator returned
   `rejected_baseline_fallback` because absolute duration bias worsened from
   9.96 to 16.77 seconds. Do not tune it again on the same five-card holdout;
   use an earlier unused chronological split or prospective results.
   That split is now complete and documented in
   `SIMULATION_KNOCKDOWN_FINISH_GRID_2026-08-27.md`. All three coupled
   knockdown/finish profiles improved knockdown CRPS but worsened the primary
   joint side/method score, so the selector retained the baseline. A separate
   official-knockdown observation projection at `0.59` then passed every gate
   on the locked 31-fight confirmation: knockdown CRPS improved from 0.3646 to
   0.2280 and combined action error from 0.5619 to 0.4863, with exact outcome,
   duration, and non-knockdown invariance. Keep `mechanics-7dca94cd8d5b` as a
   prospective shadow candidate only; do not replace the current website
   profile or retune `0.59` on the opened cohort. The next analogous low-risk
   mechanics task remains a UFCStats-control observation projection that does
   not alter latent phase dynamics.
   The separate fighter-transition audit is complete on 1,000 recent fights / 81
   cards and is documented in `SIMULATION_TRANSITION_AUDIT_2026-08-27.md`.
   Fighter/opponent KD→KO, TD→submission-attempt, and TD→submission-win point
   estimates improved slightly, but every event-block 95% interval crossed
   zero; do not add those effects. Conditional TD-round credited-control MAE
   improved from 0.23690 to 0.23094 with a wholly favorable interval of
   −0.00976 to −0.00233. The mapping is now implemented behind
   `--takedown-control-association`; its paired five-card screen improved joint,
   winner, method, duration, and total-control point scores. The subsequent
   56.3-minute audit completed 229 fights / 30 cards at exactly 100 paths per
   fight and is documented in
   `SIMULATION_CONDITIONAL_CONTROL_AND_BREADTH_REPORT_2026-08-27.md`. Standalone
   winner accuracy was 52.63%; log loss 0.72865 and Brier 0.26382 were worse
   than 50/50, with excess knockdowns/KO outcomes and underpredicted duration,
   attempts, takedowns, and control. Keep this fit candidate-only. The next
   development experiment should test a very small predeclared latent-
   knockdown grid on the now-opened 30-card cohort using cached fits, common
   seeds, and preservation gates. Use later unrun cards beginning in January
   2025 for confirmation; do not call the opened cohort validation. No
   experiment command may exceed 3,300 seconds.
   The proposed latent-knockdown grid was superseded by the supported
   two-route outcome engine. v2.1 reduced KD, KO-count, decision-count, and
   duration absolute biases by 89%, 77%, 65%, and 57% and improved joint,
   method, duration, winner-log-loss, Brier, and protected action point metrics
   on the full opened development cohort. It is retained for confirmation, not
   production. The subsequent fighter-effect ablation is also complete.
   Context-only collapsed toward chance; exposure-weighted second-stage
   shrinkage reached 55.70% point accuracy and modestly improved full-fighter
   proper scores, but it remained worse than 50/50 and every paired card
   interval crossed zero. Treat it as a diagnostic base for the next
   opponent-adjusted hierarchical parameter experiment, not as a confirmed
   snapshot policy.
7. Profile materialized-artifact loading. Reconstruction is now cached, but
   validating and decoding the 200-member cache still takes roughly two to
   three minutes and peaks near 1.4 GiB. Optimize only with exact artifact and
   aggregate parity tests.
8. Prototype a compiled bulk kernel only after a supported local C++ compiler
   or Numba toolchain is installed. Keep Python telemetry/replay authoritative,
   require batched calls and at least a material measured speedup, and reject
   the backend unless deterministic/reference equivalence passes. The current
   machine has neither toolchain, so no untestable compiled path was added.

## Completed run resume/reproduction command

The existing directory below is complete. The same command with `--resume`
validates and reuses every durable result without rerunning simulations. To
reproduce from scratch, choose a different empty output directory and omit
`--resume` only for that first invocation:

```bash
export PYTHONPATH="$PWD/src"
python -m fight_sim upcoming-card \
  --simulator-config artifacts/simulations/mechanics-validated-finishing-final5-20260826.json \
  --parameter-artifact artifacts/simulations/upcoming-2026-08-29-finish-200/parameter_model.json.gz \
  --minimum-prior-ufc-fights 3 \
  --bootstrap-members 200 \
  --initial-paths-per-member 512 \
  --max-paths-per-member 2048 \
  --workers 8 \
  --chunk-size 64 \
  --output-dir artifacts/simulations/upcoming-2026-08-29-finish-200-rerun \
  --website-output src/content/data/external/simulation_forecasts.json \
  --resume
```

Worker and chunk settings may change without changing the run contract.

All outputs remain candidate-only, paper-only, execution-disabled, and have no
production influence. Do not blend them into the production model without the
predeclared retrospective and prospective promotion gates.
