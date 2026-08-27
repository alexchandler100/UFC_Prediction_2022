# Fight simulation continuation TODO

Status updated 2026-08-26 after resuming and completing the bounded Aug. 29
card run. This file is the handoff contract for the next simulation session.

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
