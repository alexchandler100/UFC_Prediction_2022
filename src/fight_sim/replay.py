"""Reducer and stochastic replay diagnostics for full simulation traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .domain import SimulationEvent, SimulationPath, SimulationRunSpec, TelemetryLevel
from .engine import simulate_fight
from .reducer import event_to_dict, initial_state, reduce_events, state_hash


@dataclass(frozen=True, slots=True)
class TraceDifference:
    event_index: int | None
    field_path: str
    expected: str
    actual: str


def replay_trace(path: SimulationPath, *, scheduled_rounds: int | None = None) -> str:
    """Reduce stored events and verify their hash chain and declared final state."""

    if not path.events:
        raise ValueError("reducer replay requires full telemetry")
    rounds = path.scheduled_rounds if scheduled_rounds is None else scheduled_rounds
    if rounds != path.scheduled_rounds:
        raise ValueError("scheduled_rounds does not match trace contract")
    state = reduce_events(
        initial_state(path.matchup_id, rounds),
        path.events,
        verify_hashes=True,
    )
    digest = state_hash(state)
    if digest != path.final_state_hash:
        raise ValueError("replayed final state hash does not match path")
    if state.result != path.result or state.red_stats != path.red_stats or state.blue_stats != path.blue_stats:
        raise ValueError("replayed result/statistics do not match path")
    return digest


def _first_value_difference(expected: Any, actual: Any, prefix: str = "") -> tuple[str, Any, Any] | None:
    if type(expected) is not type(actual):
        return prefix or "$", expected, actual
    if isinstance(expected, dict):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                return path, expected.get(key, "<missing>"), actual.get(key, "<missing>")
            difference = _first_value_difference(expected[key], actual[key], path)
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{prefix}.length", len(expected), len(actual)
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = _first_value_difference(left, right, f"{prefix}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return prefix or "$", expected, actual
    return None


def diff_event_streams(
    expected: Iterable[SimulationEvent],
    actual: Iterable[SimulationEvent],
) -> TraceDifference | None:
    expected_values = tuple(expected)
    actual_values = tuple(actual)
    for index in range(max(len(expected_values), len(actual_values))):
        if index >= len(expected_values) or index >= len(actual_values):
            return TraceDifference(index, "event_count", str(len(expected_values)), str(len(actual_values)))
        difference = _first_value_difference(
            event_to_dict(expected_values[index]),
            event_to_dict(actual_values[index]),
        )
        if difference:
            path, left, right = difference
            return TraceDifference(index, path, repr(left), repr(right))
    return None


def stochastic_replay(
    spec: SimulationRunSpec,
    simulation_index: int,
    *,
    expected: SimulationPath | None = None,
) -> SimulationPath:
    """Regenerate a full event stream from its frozen spec and direct index."""

    actual = simulate_fight(spec, simulation_index, telemetry=TelemetryLevel.FULL)
    replay_trace(actual, scheduled_rounds=spec.bout.scheduled_rounds)
    if expected is not None:
        difference = diff_event_streams(expected.events, actual.events)
        if difference is not None:
            raise ValueError(
                f"stochastic replay diverged at event {difference.event_index} "
                f"field {difference.field_path}: {difference.expected} != {difference.actual}"
            )
        if expected.final_state_hash != actual.final_state_hash:
            raise ValueError("stochastic replay final state hash differs")
    return actual
