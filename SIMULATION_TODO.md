# Fight simulation continuation TODO

Status recorded 2026-08-26 after deliberately stopping a long local card run.
This file is the handoff contract for the next simulation session.

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
- The current checked-in website publication is the earlier high-precision
  action-calibrated profile `mechanics-0aaba4c675ce`: five matchups available,
  seven withheld for fewer than three prior UFCStats bouts, and Tsuruya-Borjas
  withheld for nonconvergence. The UI explicitly labels the final
  finish-calibrated refresh as pending.
- The interrupted final-profile run reproduced and saved the validated
  200-member pre-event parameter artifact locally at:

  `artifacts/simulations/upcoming-2026-08-29-finish-200/parameter_model.json.gz`

## Why the final card refresh was stopped

The compact artifact took about 25 minutes to deterministically re-materialize.
Nurmagomedov-Song then completed 204,800 internal paths but missed the parameter
quantile convergence gate, causing a third 204,800-path batch to start. The run
was interrupted before that batch rather than holding the session open for the
remaining card. The current runner writes authority only after a matchup's
adaptive run returns, so no partial Nurmagomedov-Song forecast was published.

## Next implementation work (in order)

1. Add a materialized, content-verified parameter cache so a validated
   200-member artifact loads in under a minute without changing member values.
2. Checkpoint the exact nested aggregate after every member-balanced adaptive
   batch. The checkpoint must include the run/spec hash, member counts, named
   RNG contract, convergence history, and aggregate hash.
3. Add `upcoming-card --resume` so it reuses validated completed matchups and
   resumes the next exact simulation-index range. Prove direct-run/resume
   equality in tests across worker counts and chunk sizes.
4. Rerun all six history-eligible Aug. 29 matchups with
   `mechanics-8ba01f34444f`, 200 bootstrap members, and 512 to 2,048 paths per
   member. Continue withholding the seven low-history fights. Withhold any
   maximum-path nonconvergence rather than publishing partial results.
5. Atomically replace
   `src/content/data/external/simulation_forecasts.json`, validate its hash and
   compact/full authority linkage, retest the Simulation tab on desktop/mobile,
   and confirm that the pending-refresh label disappears.
6. Continue chronological research on the known remaining deficiencies:
   winner calibration, UFCStats-control versus simulated-ground-top-control
   semantics, total strike-attempt dispersion, and wide fighter parameter
   intervals. Do not tune these on the upcoming card.

## Current rerun command (before resume support exists)

Use a new empty output directory. This restarts the matchup simulations and the
current compact codec still spends time re-materializing the artifact:

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
  --website-output src/content/data/external/simulation_forecasts.json
```

All outputs remain candidate-only, paper-only, execution-disabled, and have no
production influence. Do not blend them into the production model without the
predeclared retrospective and prospective promotion gates.
