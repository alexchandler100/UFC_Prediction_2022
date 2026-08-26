"""Deterministic diagnostic trace selection and serialization."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .domain import (
    DecisionType,
    EventType,
    FightResult,
    FighterStats,
    JudgeRoundScore,
    OutcomeMethod,
    Phase,
    RngDraw,
    SCHEMA_VERSION,
    Side,
    SimulationEvent,
    SimulationPath,
    SimulationRunSpec,
    StateDelta,
    TraceManifest,
)
from .reducer import event_to_dict, initial_state, reduce_events, state_hash
from .rng import run_id_for, sha256_hex


TRACE_SELECTION_ALGORITHM = "lowest_sha256_per_outcome_round_v1"


def ensemble_run_id_for(specs: Iterable[SimulationRunSpec]) -> str:
    values = tuple(specs)
    if not values:
        raise ValueError("ensemble run id requires specifications")
    run_ids = sorted(run_id_for(spec) for spec in values)
    return run_ids[0] if len(run_ids) == 1 else f"ensemble-{sha256_hex(run_ids)[:24]}"


def trace_stratum(path: SimulationPath) -> str:
    method = path.result.method.value
    if method in ("decision", "draw"):
        return f"{path.outcome_key}:distance"
    return f"{path.outcome_key}:round_{path.result.round_number}"


def trace_selection_hash(run_id: str, path: SimulationPath) -> str:
    """Stable ordering key used by batch and streaming trace selection."""

    return sha256_hex(
        {
            "algorithm": TRACE_SELECTION_ALGORITHM,
            "run_id": run_id,
            "bootstrap_member": path.bootstrap_member,
            "simulation_index": path.simulation_index,
            "stratum": trace_stratum(path),
        }
    )


def _selection_hash(run_id: str, path: SimulationPath) -> str:
    # Backward-compatible private alias for callers that imported internals.
    return trace_selection_hash(run_id, path)


def select_trace_paths(
    paths: Iterable[SimulationPath],
    *,
    run_id: str,
    max_traces: int = 32,
) -> tuple[tuple[int, int], ...]:
    """Select representative (bootstrap member, simulation index) pairs.

    Selection is independent of execution and input order.  Every represented
    outcome/round stratum receives one trace before a stratum receives two.
    """

    if max_traces < 0:
        raise ValueError("max_traces must be nonnegative")
    if max_traces == 0:
        return ()
    groups: dict[str, list[SimulationPath]] = defaultdict(list)
    for path in paths:
        groups[trace_stratum(path)].append(path)
    if not groups:
        return ()
    for stratum in groups:
        groups[stratum].sort(key=lambda path: _selection_hash(run_id, path))
    strata = sorted(
        groups,
        key=lambda value: sha256_hex(
            {"algorithm": TRACE_SELECTION_ALGORITHM, "run_id": run_id, "stratum": value}
        ),
    )
    selected: list[SimulationPath] = []
    depth = 0
    while len(selected) < max_traces:
        added = False
        for stratum in strata:
            if depth < len(groups[stratum]):
                selected.append(groups[stratum][depth])
                added = True
                if len(selected) == max_traces:
                    break
        if not added:
            break
        depth += 1
    return tuple((path.bootstrap_member, path.simulation_index) for path in selected)


def trace_digest(path: SimulationPath) -> str:
    if not path.events:
        raise ValueError("trace digest requires full telemetry events")
    return sha256_hex([event_to_dict(event) for event in path.events])


def build_trace_manifest(
    specs: Iterable[SimulationRunSpec],
    traced_paths: Iterable[SimulationPath],
) -> TraceManifest:
    specs_by_member = {spec.bootstrap_member: spec for spec in specs}
    paths = tuple(traced_paths)
    if not specs_by_member or not paths:
        raise ValueError("manifest requires specs and traced paths")
    first = specs_by_member[min(specs_by_member)]
    ensemble_run_id = ensemble_run_id_for(specs_by_member.values())
    for path in paths:
        if path.bootstrap_member not in specs_by_member:
            raise ValueError("traced path has no matching bootstrap specification")
        if not path.events:
            raise ValueError("manifest paths require full telemetry")
    return TraceManifest(
        run_id=ensemble_run_id,
        matchup_id=first.bout.matchup_id,
        root_seed=str(first.root_seed),
        engine_version=first.engine_version,
        rng_contract=first.rng_contract,
        parameter_artifact_id=first.parameter_artifact_id,
        selected_simulation_indices=tuple(path.simulation_index for path in paths),
        selected_bootstrap_members=tuple(path.bootstrap_member for path in paths),
        trace_hashes=tuple(
            (path.bootstrap_member, path.simulation_index, trace_digest(path))
            for path in paths
        ),
    )


def trace_to_dict(path: SimulationPath) -> dict[str, object]:
    if not path.events:
        raise ValueError("path does not contain full telemetry")
    return {
        "schema_version": SCHEMA_VERSION,
        "matchup_id": path.matchup_id,
        "scheduled_rounds": path.scheduled_rounds,
        "bootstrap_member": path.bootstrap_member,
        "simulation_index": path.simulation_index,
        "result": path.result.to_dict(),
        "red_stats": path.red_stats.to_dict(),
        "blue_stats": path.blue_stats.to_dict(),
        "final_state_hash": path.final_state_hash,
        "trace_hash": trace_digest(path),
        "events": [event_to_dict(event) for event in path.events],
    }


def _stats_from_dict(value: dict[str, object] | None) -> FighterStats | None:
    return None if value is None else FighterStats(**{key: int(item) for key, item in value.items()})


def _result_from_dict(value: dict[str, object] | None) -> FightResult | None:
    if value is None:
        return None
    return FightResult(
        winner=Side(str(value["winner"])) if value.get("winner") is not None else None,
        method=OutcomeMethod(str(value["method"])),
        round_number=int(value["round_number"]),
        fight_time_us=int(value["fight_time_us"]),
        round_time_us=int(value["round_time_us"]),
        reason=str(value["reason"]),
        decision_type=(
            DecisionType(str(value["decision_type"]))
            if value.get("decision_type") is not None
            else None
        ),
    )


def _round_score_from_dict(value: dict[str, object] | None) -> JudgeRoundScore | None:
    if value is None:
        return None
    return JudgeRoundScore(
        round_number=int(value["round_number"]),
        red_points=tuple(int(item) for item in value["red_points"]),  # type: ignore[arg-type]
        blue_points=tuple(int(item) for item in value["blue_points"]),  # type: ignore[arg-type]
        red_effectiveness=float(value["red_effectiveness"]),
        blue_effectiveness=float(value["blue_effectiveness"]),
    )


def _delta_from_dict(value: dict[str, object]) -> StateDelta:
    return StateDelta(
        fight_time_us=int(value["fight_time_us"]) if value.get("fight_time_us") is not None else None,
        round_time_us=int(value["round_time_us"]) if value.get("round_time_us") is not None else None,
        round_number=int(value["round_number"]) if value.get("round_number") is not None else None,
        phase=Phase(str(value["phase"])) if value.get("phase") is not None else None,
        clear_top_position=bool(value.get("clear_top_position", False)),
        top_position=Side(str(value["top_position"])) if value.get("top_position") is not None else None,
        red_stamina=float(value["red_stamina"]) if value.get("red_stamina") is not None else None,
        blue_stamina=float(value["blue_stamina"]) if value.get("blue_stamina") is not None else None,
        red_hurt=float(value["red_hurt"]) if value.get("red_hurt") is not None else None,
        blue_hurt=float(value["blue_hurt"]) if value.get("blue_hurt") is not None else None,
        red_damage=float(value["red_damage"]) if value.get("red_damage") is not None else None,
        blue_damage=float(value["blue_damage"]) if value.get("blue_damage") is not None else None,
        red_stats_delta=_stats_from_dict(value.get("red_stats_delta")),  # type: ignore[arg-type]
        blue_stats_delta=_stats_from_dict(value.get("blue_stats_delta")),  # type: ignore[arg-type]
        red_effectiveness_delta=float(value.get("red_effectiveness_delta", 0.0)),
        blue_effectiveness_delta=float(value.get("blue_effectiveness_delta", 0.0)),
        reset_round_effectiveness=bool(value.get("reset_round_effectiveness", False)),
        append_round_score=_round_score_from_dict(value.get("append_round_score")),  # type: ignore[arg-type]
        result=_result_from_dict(value.get("result")),  # type: ignore[arg-type]
    )


def _event_from_dict(value: dict[str, object]) -> SimulationEvent:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported simulation event schema_version")
    draws = tuple(
        RngDraw(
            stream=str(draw["stream"]),
            draw_index=int(draw["draw_index"]),
            distribution=str(draw["distribution"]),
            parameters=tuple(
                (str(key), str(item)) for key, item in sorted(draw["parameters"].items())
            ),
            value=str(draw["value"]),
        )
        for draw in value.get("rng_draws", [])
    )
    return SimulationEvent(
        sequence=int(value["sequence"]),
        event_type=EventType(str(value["event_type"])),
        fight_time_us=int(value["fight_time_us"]),
        round_number=int(value["round_number"]),
        actor=Side(str(value["actor"])) if value.get("actor") is not None else None,
        target=Side(str(value["target"])) if value.get("target") is not None else None,
        action=str(value.get("action", "")),
        phase_before=Phase(str(value["phase_before"])),
        phase_after=Phase(str(value["phase_after"])),
        delta=_delta_from_dict(value["delta"]),  # type: ignore[arg-type]
        rng_draws=draws,
        state_hash_before=str(value["state_hash_before"]),
        state_hash_after=str(value["state_hash_after"]),
        previous_event_hash=str(value["previous_event_hash"]),
        event_hash=str(value["event_hash"]),
        payload=tuple(
            (str(key), str(item)) for key, item in sorted(value.get("payload", {}).items())
        ),
        schema_version=str(value["schema_version"]),
    )


def trace_from_dict(value: dict[str, object], *, verify: bool = True) -> SimulationPath:
    """Load a serialized trace and, by default, verify every state/event hash."""

    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported simulation trace schema_version")
    events = tuple(_event_from_dict(item) for item in value.get("events", []))
    if not events:
        raise ValueError("serialized trace contains no events")
    path = SimulationPath(
        matchup_id=str(value["matchup_id"]),
        scheduled_rounds=int(value["scheduled_rounds"]),
        bootstrap_member=int(value["bootstrap_member"]),
        simulation_index=int(value["simulation_index"]),
        result=_result_from_dict(value["result"]),  # type: ignore[arg-type]
        red_stats=_stats_from_dict(value["red_stats"]),  # type: ignore[arg-type]
        blue_stats=_stats_from_dict(value["blue_stats"]),  # type: ignore[arg-type]
        final_state_hash=str(value["final_state_hash"]),
        events=events,
    )
    if path.result is None or path.red_stats is None or path.blue_stats is None:
        raise ValueError("serialized trace is missing final result/statistics")
    if verify:
        state = reduce_events(
            initial_state(path.matchup_id, path.scheduled_rounds),
            path.events,
            verify_hashes=True,
        )
        if state_hash(state) != path.final_state_hash:
            raise ValueError("serialized trace final state hash mismatch")
        if state.result != path.result or state.red_stats != path.red_stats or state.blue_stats != path.blue_stats:
            raise ValueError("serialized trace result/statistics mismatch")
        declared_trace_hash = value.get("trace_hash")
        if declared_trace_hash is not None and str(declared_trace_hash) != trace_digest(path):
            raise ValueError("serialized trace digest mismatch")
    return path
