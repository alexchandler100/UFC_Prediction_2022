"""Bounded-memory nested bootstrap/process Monte Carlo orchestration."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import math
from typing import Iterable, Iterator

import numpy as np

from .aggregate import ForecastAccumulator
from .domain import (
    AggregateForecast,
    SimulationEvent,
    SimulationPath,
    SimulationRunSpec,
    TraceManifest,
)
from .engine import SimulationInvariantError, simulate_fight
from .reducer import event_to_dict
from .telemetry import (
    build_trace_manifest,
    ensemble_run_id_for,
    select_trace_paths,
    trace_selection_hash,
    trace_stratum,
)


DEFAULT_PATH_RETENTION_LIMIT = 4096


@dataclass(frozen=True, slots=True)
class ConvergenceDiagnostics:
    paths_per_member: int
    total_paths: int
    winner_process_mcse: float
    split_estimate_difference: float
    split_combined_mcse: float
    parameter_quantile_max_shift: float
    mcse_within_target: bool
    headline_batches_stable: bool
    parameter_quantiles_stable: bool

    @property
    def converged(self) -> bool:
        return (
            self.mcse_within_target
            and self.headline_batches_stable
            and self.parameter_quantiles_stable
        )


@dataclass(frozen=True, slots=True)
class InvariantFailureRecord:
    bootstrap_member: int
    simulation_index: int
    failures: tuple[str, ...]
    events: tuple[SimulationEvent, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "bootstrap_member": self.bootstrap_member,
            "simulation_index": self.simulation_index,
            "failures": list(self.failures),
            "events": [event_to_dict(event) for event in self.events],
        }


class NestedSimulationBatchError(RuntimeError):
    """An incomplete nested run withheld after deterministic invariant failure."""

    def __init__(self, failures: Iterable[InvariantFailureRecord]) -> None:
        self.failures = tuple(
            sorted(
                failures,
                key=lambda item: (item.bootstrap_member, item.simulation_index),
            )
        )
        if not self.failures:
            raise ValueError("nested invariant error requires failure records")
        first = self.failures[0]
        message = first.failures[0] if first.failures else "unknown invariant failure"
        super().__init__(
            "nested simulation withheld after invariant failure at "
            f"member={first.bootstrap_member}, index={first.simulation_index}: {message}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "failed_invariant",
            "complete": False,
            "published": False,
            "failures": [item.to_dict() for item in self.failures],
        }


@dataclass(frozen=True, slots=True)
class CompactRunLedger:
    """Memory-accounting and failure ledger for one completed nested run."""

    total_paths: int
    retained_paths: int
    streaming: bool
    max_in_flight_paths: int
    packed_duration_bytes: int
    invariant_failure_count: int

    def __post_init__(self) -> None:
        if self.total_paths <= 0:
            raise ValueError("compact run ledger requires completed paths")
        if not 0 <= self.retained_paths <= self.total_paths:
            raise ValueError("retained path count is invalid")
        if self.max_in_flight_paths <= 0 or self.packed_duration_bytes < 0:
            raise ValueError("compact run memory accounting is invalid")
        if self.invariant_failure_count < 0:
            raise ValueError("invariant failure count must be nonnegative")


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    forecast: AggregateForecast
    paths: tuple[SimulationPath, ...]
    traces: tuple[SimulationPath, ...]
    trace_manifest: TraceManifest | None
    convergence: tuple[ConvergenceDiagnostics, ...]
    converged: bool
    ledger: CompactRunLedger | None = None
    invariant_failures: tuple[InvariantFailureRecord, ...] = ()

    @property
    def paths_retained(self) -> bool:
        return len(self.paths) == self.forecast.total_paths


def _validate_specs(specs: Iterable[SimulationRunSpec]) -> tuple[SimulationRunSpec, ...]:
    values = tuple(sorted(specs, key=lambda spec: spec.bootstrap_member))
    if not values:
        raise ValueError("nested simulation requires bootstrap specifications")
    members = [spec.bootstrap_member for spec in values]
    if len(set(members)) != len(members):
        raise ValueError("bootstrap member ids must be unique")
    first = values[0]
    for spec in values[1:]:
        if spec.bout != first.bout:
            raise ValueError("all nested specifications must describe the same bout")
        if str(spec.root_seed) != str(first.root_seed):
            raise ValueError("all nested specifications must share the root seed")
        if spec.parameter_artifact_id != first.parameter_artifact_id:
            raise ValueError("all nested specifications must share one parameter artifact")
        if spec.engine_version != first.engine_version or spec.rng_contract != first.rng_contract:
            raise ValueError("all nested specifications must share engine/RNG contracts")
    return values


class _TraceCandidatePool:
    """Keep only the globally relevant lowest-hash paths in each stratum."""

    def __init__(self, run_id: str, max_per_stratum: int) -> None:
        if max_per_stratum < 0:
            raise ValueError("max trace candidates must be nonnegative")
        self.run_id = run_id
        self.max_per_stratum = max_per_stratum
        self.groups: dict[str, list[tuple[str, SimulationPath]]] = {}

    def add(self, path: SimulationPath) -> None:
        if self.max_per_stratum == 0:
            return
        stratum = trace_stratum(path)
        candidate = (trace_selection_hash(self.run_id, path), path)
        group = self.groups.setdefault(stratum, [])
        group.append(candidate)
        group.sort(key=lambda item: item[0])
        if len(group) > self.max_per_stratum:
            group.pop()

    def extend(self, paths: Iterable[SimulationPath]) -> None:
        for path in paths:
            self.add(path)

    def paths(self) -> tuple[SimulationPath, ...]:
        return tuple(
            path
            for stratum in sorted(self.groups)
            for _, path in self.groups[stratum]
        )


@dataclass(frozen=True, slots=True)
class _ChunkJob:
    spec: SimulationRunSpec
    indices: tuple[int, ...]
    retain_paths: bool
    run_id: str
    max_trace_candidates: int


@dataclass(slots=True)
class _ChunkResult:
    accumulator: ForecastAccumulator
    retained_paths: tuple[SimulationPath, ...]
    trace_candidates: tuple[SimulationPath, ...]
    invariant_failures: tuple[InvariantFailureRecord, ...]
    simulated_paths: int


def _simulate_chunk(job: _ChunkJob) -> _ChunkResult:
    accumulator = ForecastAccumulator(job.spec.bout.scheduled_rounds)
    retained: list[SimulationPath] = []
    failures: list[InvariantFailureRecord] = []
    candidates = _TraceCandidatePool(job.run_id, job.max_trace_candidates)
    for index in job.indices:
        try:
            path = simulate_fight(job.spec, index)
        except SimulationInvariantError as error:
            failures.append(
                InvariantFailureRecord(
                    bootstrap_member=job.spec.bootstrap_member,
                    simulation_index=index,
                    failures=(str(error),),
                    events=error.events,
                )
            )
            # This chunk is incomplete and must never be merged into a
            # publishable aggregate. Return the durable full trace immediately.
            break
        accumulator.add_path(path)
        candidates.add(path)
        if job.retain_paths:
            retained.append(path)
        if path.invariant_failures:
            failures.append(
                InvariantFailureRecord(
                    bootstrap_member=path.bootstrap_member,
                    simulation_index=path.simulation_index,
                    failures=path.invariant_failures,
                )
            )
    return _ChunkResult(
        accumulator=accumulator,
        retained_paths=tuple(retained),
        trace_candidates=candidates.paths(),
        invariant_failures=tuple(failures),
        simulated_paths=accumulator.total_paths,
    )


def _chunks(start: int, stop: int, size: int) -> Iterator[tuple[int, ...]]:
    for offset in range(start, stop, size):
        yield tuple(range(offset, min(offset + size, stop)))


def _jobs(
    specs: tuple[SimulationRunSpec, ...],
    start_index: int,
    stop_index: int,
    *,
    chunk_size: int,
    retain_paths: bool,
    run_id: str,
    max_trace_candidates: int,
) -> Iterator[_ChunkJob]:
    for spec in specs:
        for indices in _chunks(start_index, stop_index, chunk_size):
            yield _ChunkJob(
                spec=spec,
                indices=indices,
                retain_paths=retain_paths,
                run_id=run_id,
                max_trace_candidates=max_trace_candidates,
            )


@dataclass(slots=True)
class _RangeResult:
    accumulator: ForecastAccumulator
    paths: tuple[SimulationPath, ...]
    trace_candidates: tuple[SimulationPath, ...]
    invariant_failures: tuple[InvariantFailureRecord, ...]
    max_in_flight_paths: int


def _run_range(
    specs: tuple[SimulationRunSpec, ...],
    start_index: int,
    stop_index: int,
    *,
    workers: int,
    chunk_size: int,
    retain_paths: bool,
    max_traces: int,
    run_id: str,
) -> _RangeResult:
    if start_index < 0 or stop_index < start_index:
        raise ValueError("invalid simulation index range")
    if workers <= 0 or chunk_size <= 0:
        raise ValueError("workers and chunk_size must be positive")
    accumulator = ForecastAccumulator(specs[0].bout.scheduled_rounds)
    retained: list[SimulationPath] = []
    failures: list[InvariantFailureRecord] = []
    candidates = _TraceCandidatePool(run_id, max_traces)

    def consume(block: _ChunkResult) -> None:
        if block.invariant_failures:
            failures.extend(block.invariant_failures)
            raise NestedSimulationBatchError(failures)
        accumulator.merge(block.accumulator)
        retained.extend(block.retained_paths)
        candidates.extend(block.trace_candidates)

    jobs = _jobs(
        specs,
        start_index,
        stop_index,
        chunk_size=chunk_size,
        retain_paths=retain_paths,
        run_id=run_id,
        max_trace_candidates=max_traces,
    )
    if workers == 1:
        for job in jobs:
            consume(_simulate_chunk(job))
        max_in_flight = min(chunk_size, max(stop_index - start_index, 1))
    else:
        # ProcessPoolExecutor.map eagerly retains all futures.  A fixed window
        # keeps both scheduled work and completed chunk results bounded.
        capacity = max(1, workers * 2)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = set()
            for _ in range(capacity):
                try:
                    pending.add(executor.submit(_simulate_chunk, next(jobs)))
                except StopIteration:
                    break
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    consume(future.result())
                    try:
                        pending.add(executor.submit(_simulate_chunk, next(jobs)))
                    except StopIteration:
                        pass
        max_in_flight = capacity * min(
            chunk_size, max(stop_index - start_index, 1)
        )
    retained.sort(key=lambda path: (path.bootstrap_member, path.simulation_index))
    failures.sort(key=lambda item: (item.bootstrap_member, item.simulation_index))
    return _RangeResult(
        accumulator=accumulator,
        paths=tuple(retained),
        trace_candidates=candidates.paths(),
        invariant_failures=tuple(failures),
        max_in_flight_paths=max_in_flight,
    )


def _trace_selected(
    specs: tuple[SimulationRunSpec, ...],
    candidates: tuple[SimulationPath, ...],
    max_traces: int,
) -> tuple[tuple[SimulationPath, ...], TraceManifest | None]:
    if max_traces <= 0:
        return (), None
    spec_by_member = {spec.bootstrap_member: spec for spec in specs}
    run_id = ensemble_run_id_for(specs)
    selected = select_trace_paths(candidates, run_id=run_id, max_traces=max_traces)
    compact_by_key = {
        (path.bootstrap_member, path.simulation_index): path for path in candidates
    }
    traces: list[SimulationPath] = []
    for member, index in selected:
        trace = simulate_fight(spec_by_member[member], index, telemetry="full")
        compact = compact_by_key[(member, index)]
        if (
            trace.result != compact.result
            or trace.red_stats != compact.red_stats
            or trace.blue_stats != compact.blue_stats
            or trace.final_state_hash != compact.final_state_hash
        ):
            raise RuntimeError("telemetry changed a selected simulation path")
        traces.append(trace)
    result = tuple(traces)
    return result, build_trace_manifest(specs, result)


def _probability_and_mcse(successes: int, paths: int) -> tuple[float, float]:
    if paths <= 0:
        return 0.0, math.inf
    probability = successes / paths
    return probability, math.sqrt(probability * (1.0 - probability) / paths)


def _convergence_from_accumulator(
    accumulator: ForecastAccumulator,
    forecast: AggregateForecast,
    *,
    winner_mcse_target: float,
    parameter_quantile_tolerance: float,
) -> ConvergenceDiagnostics:
    if winner_mcse_target <= 0 or parameter_quantile_tolerance < 0:
        raise ValueError("convergence tolerances must be positive/nonnegative")
    winner = next(item for item in forecast.uncertainty if item.metric == "red_win")
    split_totals = [
        sum(
            accumulator.split_paths[(member, parity)]
            for member in accumulator.member_paths
        )
        for parity in (0, 1)
    ]
    split_red = [
        sum(
            accumulator.split_red_wins[(member, parity)]
            for member in accumulator.member_paths
        )
        for parity in (0, 1)
    ]
    first_p, first_mcse = _probability_and_mcse(split_red[0], split_totals[0])
    second_p, second_mcse = _probability_and_mcse(split_red[1], split_totals[1])
    split_combined = math.sqrt(first_mcse**2 + second_mcse**2)
    split_difference = abs(first_p - second_p)

    member_probabilities: list[np.ndarray] = []
    for parity in (0, 1):
        values = [
            accumulator.split_red_wins[(member, parity)]
            / accumulator.split_paths[(member, parity)]
            for member in sorted(accumulator.member_paths)
            if accumulator.split_paths[(member, parity)] > 0
        ]
        member_probabilities.append(np.asarray(values, dtype=float))
    if len(member_probabilities[0]) and len(member_probabilities[1]):
        first_q = np.quantile(member_probabilities[0], [0.025, 0.5, 0.975])
        second_q = np.quantile(member_probabilities[1], [0.025, 0.5, 0.975])
        quantile_shift = float(np.max(np.abs(first_q - second_q)))
    else:
        quantile_shift = math.inf
    headline_stable = (
        split_difference == 0.0
        if split_combined == 0.0
        else split_difference <= 3.0 * split_combined
    )
    paths_per_member = min(item.paths for item in forecast.bootstrap_outcome_counts)
    return ConvergenceDiagnostics(
        paths_per_member=paths_per_member,
        total_paths=accumulator.total_paths,
        winner_process_mcse=winner.process_mcse,
        split_estimate_difference=split_difference,
        split_combined_mcse=split_combined,
        parameter_quantile_max_shift=quantile_shift,
        mcse_within_target=winner.process_mcse <= winner_mcse_target,
        headline_batches_stable=headline_stable,
        parameter_quantiles_stable=quantile_shift <= parameter_quantile_tolerance,
    )


def convergence_diagnostics(
    paths: Iterable[SimulationPath],
    forecast: AggregateForecast,
    *,
    winner_mcse_target: float = 0.002,
    parameter_quantile_tolerance: float = 0.01,
) -> ConvergenceDiagnostics:
    """Compatibility path API backed by the same exact streaming counters."""

    accumulator = ForecastAccumulator(forecast.scheduled_rounds)
    for path in paths:
        accumulator.add_path(path)
    if accumulator.total_paths != forecast.total_paths:
        raise ValueError("diagnostic paths disagree with forecast total")
    return _convergence_from_accumulator(
        accumulator,
        forecast,
        winner_mcse_target=winner_mcse_target,
        parameter_quantile_tolerance=parameter_quantile_tolerance,
    )


def _retain_paths(
    requested: bool | None,
    total_paths: int,
    path_retention_limit: int,
) -> bool:
    if path_retention_limit < 0:
        raise ValueError("path_retention_limit must be nonnegative")
    return bool(requested) if requested is not None else total_paths <= path_retention_limit


def _ledger(
    accumulator: ForecastAccumulator,
    *,
    retained_paths: int,
    streaming: bool,
    max_in_flight_paths: int,
    invariant_failures: int,
) -> CompactRunLedger:
    return CompactRunLedger(
        total_paths=accumulator.total_paths,
        retained_paths=retained_paths,
        streaming=streaming,
        max_in_flight_paths=max_in_flight_paths,
        packed_duration_bytes=(
            len(accumulator.duration_values_us)
            * accumulator.duration_values_us.itemsize
        ),
        invariant_failure_count=invariant_failures,
    )


def run_nested(
    specs: Iterable[SimulationRunSpec],
    paths_per_member: int,
    *,
    start_index: int = 0,
    workers: int = 1,
    chunk_size: int = 64,
    max_traces: int = 32,
    winner_mcse_target: float = 0.002,
    parameter_quantile_tolerance: float = 0.01,
    retain_paths: bool | None = None,
    path_retention_limit: int = DEFAULT_PATH_RETENTION_LIMIT,
) -> MonteCarloResult:
    values = _validate_specs(specs)
    if paths_per_member <= 0:
        raise ValueError("paths_per_member must be positive")
    keep_paths = _retain_paths(
        retain_paths, len(values) * paths_per_member, path_retention_limit
    )
    run_id = ensemble_run_id_for(values)
    result = _run_range(
        values,
        start_index,
        start_index + paths_per_member,
        workers=workers,
        chunk_size=chunk_size,
        retain_paths=keep_paths,
        max_traces=max_traces,
        run_id=run_id,
    )
    forecast = result.accumulator.forecast()
    diagnostics = _convergence_from_accumulator(
        result.accumulator,
        forecast,
        winner_mcse_target=winner_mcse_target,
        parameter_quantile_tolerance=parameter_quantile_tolerance,
    )
    traces, manifest = _trace_selected(values, result.trace_candidates, max_traces)
    return MonteCarloResult(
        forecast=forecast,
        paths=result.paths,
        traces=traces,
        trace_manifest=manifest,
        convergence=(diagnostics,),
        converged=diagnostics.converged,
        ledger=_ledger(
            result.accumulator,
            retained_paths=len(result.paths),
            streaming=not keep_paths,
            max_in_flight_paths=result.max_in_flight_paths,
            invariant_failures=len(result.invariant_failures),
        ),
        invariant_failures=result.invariant_failures,
    )


def run_adaptive_nested(
    specs: Iterable[SimulationRunSpec],
    *,
    initial_paths_per_member: int = 512,
    max_paths_per_member: int = 2048,
    start_index: int = 0,
    workers: int = 1,
    chunk_size: int = 64,
    max_traces: int = 32,
    winner_mcse_target: float = 0.002,
    parameter_quantile_tolerance: float = 0.01,
    retain_paths: bool | None = None,
    path_retention_limit: int = DEFAULT_PATH_RETENTION_LIMIT,
) -> MonteCarloResult:
    """Double inner paths while merging exact bounded-memory accumulators."""

    values = _validate_specs(specs)
    if initial_paths_per_member <= 0 or max_paths_per_member < initial_paths_per_member:
        raise ValueError("adaptive path bounds are invalid")
    keep_paths = _retain_paths(
        retain_paths, len(values) * max_paths_per_member, path_retention_limit
    )
    run_id = ensemble_run_id_for(values)
    cumulative = ForecastAccumulator(values[0].bout.scheduled_rounds)
    candidates = _TraceCandidatePool(run_id, max_traces)
    retained: list[SimulationPath] = []
    failures: list[InvariantFailureRecord] = []
    history: list[ConvergenceDiagnostics] = []
    max_in_flight_paths = 1
    current = min(initial_paths_per_member, max_paths_per_member)
    range_start = start_index
    while True:
        addition = _run_range(
            values,
            range_start,
            start_index + current,
            workers=workers,
            chunk_size=chunk_size,
            retain_paths=keep_paths,
            max_traces=max_traces,
            run_id=run_id,
        )
        cumulative.merge(addition.accumulator)
        candidates.extend(addition.trace_candidates)
        retained.extend(addition.paths)
        failures.extend(addition.invariant_failures)
        max_in_flight_paths = max(
            max_in_flight_paths, addition.max_in_flight_paths
        )
        forecast = cumulative.forecast()
        diagnostics = _convergence_from_accumulator(
            cumulative,
            forecast,
            winner_mcse_target=winner_mcse_target,
            parameter_quantile_tolerance=parameter_quantile_tolerance,
        )
        history.append(diagnostics)
        if diagnostics.converged or current >= max_paths_per_member:
            break
        next_count = min(current * 2, max_paths_per_member)
        range_start = start_index + current
        current = next_count
    retained.sort(key=lambda path: (path.bootstrap_member, path.simulation_index))
    failures.sort(key=lambda item: (item.bootstrap_member, item.simulation_index))
    traces, manifest = _trace_selected(values, candidates.paths(), max_traces)
    return MonteCarloResult(
        forecast=forecast,
        paths=tuple(retained),
        traces=traces,
        trace_manifest=manifest,
        convergence=tuple(history),
        converged=history[-1].converged,
        ledger=_ledger(
            cumulative,
            retained_paths=len(retained),
            streaming=not keep_paths,
            max_in_flight_paths=max_in_flight_paths,
            invariant_failures=len(failures),
        ),
        invariant_failures=tuple(failures),
    )
