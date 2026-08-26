"""Pure fight-state reduction and event hash-chain verification."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from .domain import (
    EventType,
    FightState,
    JudgeRoundScore,
    Phase,
    SimulationEvent,
    StateDelta,
)
from .rng import sha256_hex


GENESIS_EVENT_HASH = "0" * 64


class ReplayError(ValueError):
    """Raised when an immutable trace cannot reproduce its declared state."""


def initial_state(matchup_id: str, scheduled_rounds: int) -> FightState:
    return FightState(matchup_id=matchup_id, scheduled_rounds=scheduled_rounds)


def state_hash(state: FightState) -> str:
    return sha256_hex(state.to_dict())


def _score_dict(score: JudgeRoundScore | None) -> dict[str, Any] | None:
    if score is None:
        return None
    return {
        "round_number": score.round_number,
        "red_points": list(score.red_points),
        "blue_points": list(score.blue_points),
        "red_effectiveness": score.red_effectiveness,
        "blue_effectiveness": score.blue_effectiveness,
    }


def delta_to_dict(delta: StateDelta) -> dict[str, Any]:
    return {
        "fight_time_us": delta.fight_time_us,
        "round_time_us": delta.round_time_us,
        "round_number": delta.round_number,
        "phase": delta.phase.value if delta.phase else None,
        "clear_top_position": delta.clear_top_position,
        "top_position": delta.top_position.value if delta.top_position else None,
        "red_stamina": delta.red_stamina,
        "blue_stamina": delta.blue_stamina,
        "red_hurt": delta.red_hurt,
        "blue_hurt": delta.blue_hurt,
        "red_damage": delta.red_damage,
        "blue_damage": delta.blue_damage,
        "red_stats_delta": delta.red_stats_delta.to_dict() if delta.red_stats_delta else None,
        "blue_stats_delta": delta.blue_stats_delta.to_dict() if delta.blue_stats_delta else None,
        "red_effectiveness_delta": delta.red_effectiveness_delta,
        "blue_effectiveness_delta": delta.blue_effectiveness_delta,
        "reset_round_effectiveness": delta.reset_round_effectiveness,
        "append_round_score": _score_dict(delta.append_round_score),
        "result": delta.result.to_dict() if delta.result else None,
    }


def apply_delta(state: FightState, delta: StateDelta) -> FightState:
    """Apply one complete immutable event delta.

    The reducer is the single authority for state mutation in traced and lean
    execution.  Creating a dataclass at every event is intentional: it keeps
    invalid intermediate states observable during validation and replay.
    """

    if state.result is not None:
        raise ReplayError("cannot apply events after termination")
    fight_time = state.fight_time_us if delta.fight_time_us is None else delta.fight_time_us
    round_time = state.round_time_us if delta.round_time_us is None else delta.round_time_us
    round_number = state.round_number if delta.round_number is None else delta.round_number
    if fight_time < state.fight_time_us:
        raise ReplayError("fight time cannot move backwards")
    if round_number < state.round_number:
        raise ReplayError("round number cannot move backwards")
    if round_number == state.round_number and round_time < state.round_time_us:
        raise ReplayError("round time cannot move backwards within a round")
    if round_number > state.round_number and round_time != 0:
        raise ReplayError("a new round must begin at zero round time")

    phase = state.phase if delta.phase is None else delta.phase
    top_position = state.top_position
    if delta.clear_top_position or phase.value != "ground":
        top_position = None
    if delta.top_position is not None:
        top_position = delta.top_position

    next_state = replace(
        state,
        fight_time_us=fight_time,
        round_time_us=round_time,
        round_number=round_number,
        phase=phase,
        top_position=top_position,
        red_stamina=state.red_stamina if delta.red_stamina is None else delta.red_stamina,
        blue_stamina=state.blue_stamina if delta.blue_stamina is None else delta.blue_stamina,
        red_hurt=state.red_hurt if delta.red_hurt is None else delta.red_hurt,
        blue_hurt=state.blue_hurt if delta.blue_hurt is None else delta.blue_hurt,
        red_damage=state.red_damage if delta.red_damage is None else delta.red_damage,
        blue_damage=state.blue_damage if delta.blue_damage is None else delta.blue_damage,
        red_stats=(
            state.red_stats
            if delta.red_stats_delta is None
            else state.red_stats.add(delta.red_stats_delta)
        ),
        blue_stats=(
            state.blue_stats
            if delta.blue_stats_delta is None
            else state.blue_stats.add(delta.blue_stats_delta)
        ),
        red_round_effectiveness=(
            0.0
            if delta.reset_round_effectiveness
            else state.red_round_effectiveness + delta.red_effectiveness_delta
        ),
        blue_round_effectiveness=(
            0.0
            if delta.reset_round_effectiveness
            else state.blue_round_effectiveness + delta.blue_effectiveness_delta
        ),
        round_scores=(
            state.round_scores
            if delta.append_round_score is None
            else (*state.round_scores, delta.append_round_score)
        ),
        result=state.result if delta.result is None else delta.result,
        event_count=state.event_count + 1,
    )
    if next_state.result is not None:
        if next_state.result.fight_time_us != next_state.fight_time_us:
            raise ReplayError("result fight clock must match terminal state")
        if next_state.result.round_time_us != next_state.round_time_us:
            raise ReplayError("result round clock must match terminal state")
        if next_state.result.round_number != next_state.round_number:
            raise ReplayError("result round must match terminal state")
    return next_state


def _event_hash_body(event: SimulationEvent) -> dict[str, Any]:
    return {
        "schema_version": event.schema_version,
        "sequence": event.sequence,
        "event_type": event.event_type.value,
        "fight_time_us": event.fight_time_us,
        "round_number": event.round_number,
        "actor": event.actor.value if event.actor else None,
        "target": event.target.value if event.target else None,
        "action": event.action,
        "phase_before": event.phase_before.value,
        "phase_after": event.phase_after.value,
        "delta": delta_to_dict(event.delta),
        "rng_draws": [draw.to_dict() for draw in event.rng_draws],
        "state_hash_before": event.state_hash_before,
        "state_hash_after": event.state_hash_after,
        "previous_event_hash": event.previous_event_hash,
        "payload": dict(event.payload),
    }


def event_to_dict(event: SimulationEvent) -> dict[str, Any]:
    return {**_event_hash_body(event), "event_hash": event.event_hash}


def make_event(
    state: FightState,
    event_type: EventType,
    delta: StateDelta,
    *,
    actor=None,
    target=None,
    action: str = "",
    rng_draws=(),
    previous_event_hash: str = GENESIS_EVENT_HASH,
    payload: dict[str, Any] | None = None,
) -> tuple[FightState, SimulationEvent]:
    before_hash = state_hash(state)
    next_state = apply_delta(state, delta)
    event = SimulationEvent(
        sequence=state.event_count,
        event_type=event_type,
        fight_time_us=next_state.fight_time_us,
        round_number=next_state.round_number,
        actor=actor,
        target=target,
        action=action,
        phase_before=state.phase,
        phase_after=next_state.phase,
        delta=delta,
        rng_draws=tuple(rng_draws),
        state_hash_before=before_hash,
        state_hash_after=state_hash(next_state),
        previous_event_hash=previous_event_hash,
        event_hash="",
        payload=tuple((str(key), str(value)) for key, value in sorted((payload or {}).items())),
    )
    event = replace(event, event_hash=sha256_hex(_event_hash_body(event)))
    return next_state, event


def reduce_event(
    state: FightState,
    event: SimulationEvent,
    *,
    verify_hashes: bool = True,
    expected_previous_event_hash: str | None = None,
) -> FightState:
    if event.sequence != state.event_count:
        raise ReplayError(
            f"event sequence {event.sequence} does not match expected {state.event_count}"
        )
    if event.phase_before is not state.phase:
        raise ReplayError("event phase_before does not match state")
    if expected_previous_event_hash is not None and event.previous_event_hash != expected_previous_event_hash:
        raise ReplayError("event hash chain is broken")
    if verify_hashes and event.state_hash_before != state_hash(state):
        raise ReplayError("state_hash_before mismatch")
    _validate_event_semantics(state, event)
    next_state = apply_delta(state, event.delta)
    if event.fight_time_us != next_state.fight_time_us or event.round_number != next_state.round_number:
        raise ReplayError("event clock metadata does not match reduced state")
    if event.phase_after is not next_state.phase:
        raise ReplayError("event phase_after does not match reduced state")
    if verify_hashes:
        if event.state_hash_after != state_hash(next_state):
            raise ReplayError("state_hash_after mismatch")
        if event.event_hash != sha256_hex(_event_hash_body(event)):
            raise ReplayError("event_hash mismatch")
    return next_state


_ACTION_PHASES = {
    "strike": {Phase.DISTANCE, Phase.CLINCH, Phase.GROUND},
    "clinch_entry": {Phase.DISTANCE},
    "clinch_exit": {Phase.CLINCH},
    "takedown": {Phase.DISTANCE, Phase.CLINCH},
    "submission": {Phase.GROUND},
    "escape": {Phase.GROUND},
    "scramble_ground": {Phase.SCRAMBLE},
    "scramble_distance": {Phase.SCRAMBLE},
    "no_contest": {Phase.DISTANCE, Phase.CLINCH, Phase.GROUND, Phase.SCRAMBLE},
    "other_finish": {Phase.DISTANCE, Phase.CLINCH, Phase.GROUND, Phase.SCRAMBLE},
    "knockdown_follow_up": {Phase.DISTANCE, Phase.CLINCH, Phase.GROUND},
}

_ACTION_RESULT_PHASES = {
    "clinch_entry": {Phase.DISTANCE, Phase.CLINCH},
    "clinch_exit": {Phase.CLINCH, Phase.DISTANCE},
    "takedown": {Phase.DISTANCE, Phase.CLINCH, Phase.GROUND},
    "escape": {Phase.GROUND, Phase.SCRAMBLE},
    "scramble_ground": {Phase.SCRAMBLE, Phase.GROUND},
    "scramble_distance": {Phase.SCRAMBLE, Phase.DISTANCE},
}


def _validate_event_semantics(state: FightState, event: SimulationEvent) -> None:
    """Reject hash-valid traces whose typed event mechanics are illegal."""

    if event.delta.result is not None and event.event_type is not EventType.TERMINATION:
        raise ReplayError("only a termination event may carry a result")
    if event.event_type is EventType.TERMINATION:
        if event.delta.result is None:
            raise ReplayError("termination event is missing its result")
        if event.action != event.delta.result.method.value:
            raise ReplayError("termination action disagrees with result method")
    elif event.delta.round_number is not None and event.event_type is not EventType.ROUND_START:
        raise ReplayError("only round_start may change the round number")

    fight_clock_changes = (
        event.delta.fight_time_us is not None
        and event.delta.fight_time_us != state.fight_time_us
    )
    round_clock_changes = (
        event.delta.round_time_us is not None
        and event.delta.round_time_us != state.round_time_us
    )
    if (fight_clock_changes or round_clock_changes) and event.event_type not in {
        EventType.TIME_ADVANCE,
        EventType.ROUND_START,
    }:
        raise ReplayError("only time_advance or round_start may change clocks")
    if event.event_type is EventType.TIME_ADVANCE and not fight_clock_changes:
        raise ReplayError("time_advance must advance the fight clock")

    if event.event_type in {
        EventType.ACTION_ATTEMPT,
        EventType.ACTION_RESOLUTION,
        EventType.ACTION_CONSEQUENCE,
    }:
        allowed = _ACTION_PHASES.get(event.action)
        if allowed is None or state.phase not in allowed:
            raise ReplayError(
                f"action {event.action!r} is illegal from phase {state.phase.value}"
            )
        if event.action == "no_contest":
            if event.actor is not None or event.target is not None:
                raise ReplayError("no-contest action must be actor-neutral")
        elif event.action != "knockdown_follow_up" and event.actor is None:
            raise ReplayError(f"action {event.action!r} requires an actor")
        if event.actor is not None and event.target is not event.actor.opponent:
            raise ReplayError("action target must be the actor's opponent")
        next_phase = state.phase if event.delta.phase is None else event.delta.phase
        allowed_results = _ACTION_RESULT_PHASES.get(event.action, {state.phase})
        if next_phase not in allowed_results:
            raise ReplayError(
                f"action {event.action!r} has illegal phase result {next_phase.value}"
            )
    elif event.delta.phase is not None:
        if event.event_type is not EventType.ROUND_START or event.delta.phase is not Phase.DISTANCE:
            raise ReplayError("non-action phase changes are limited to round_start distance")

    if event.event_type is EventType.ROUND_START:
        next_round = state.round_number if event.delta.round_number is None else event.delta.round_number
        next_time = state.round_time_us if event.delta.round_time_us is None else event.delta.round_time_us
        if next_time != 0 or event.phase_after is not Phase.DISTANCE:
            raise ReplayError("round_start must begin at zero in distance")
        if state.event_count > 0 and next_round not in {state.round_number, state.round_number + 1}:
            raise ReplayError("round_start round number is not consecutive")
    if event.event_type in {EventType.ROUND_SCORE, EventType.ROUND_BELL}:
        # BoutConfig permits shortened synthetic rounds for deterministic
        # contract tests.  The event stream does not duplicate that rules
        # field, so replay can enforce a positive, legal UFC upper bound while
        # the stochastic spec check supplies the exact configured duration.
        if not 0 < state.round_time_us <= 300_000_000:
            raise ReplayError("round scoring and bell require completed round exposure")


def reduce_events(
    state: FightState,
    events: Iterable[SimulationEvent],
    *,
    verify_hashes: bool = True,
) -> FightState:
    values = tuple(events)
    if not values:
        raise ReplayError("event stream is empty")
    if values[0].event_type is not EventType.FIGHT_START:
        raise ReplayError("event stream must begin with fight_start")
    if len(values) < 2 or values[1].event_type is not EventType.ROUND_START:
        raise ReplayError("fight_start must be followed by round_start")
    terminal_indices = [
        index
        for index, event in enumerate(values)
        if event.event_type is EventType.TERMINATION
    ]
    if terminal_indices != [len(values) - 1]:
        raise ReplayError("event stream requires exactly one final termination")
    for index, event in enumerate(values):
        previous = values[index - 1] if index else None
        if event.event_type is EventType.ACTION_RESOLUTION:
            if (
                previous is None
                or previous.event_type is not EventType.ACTION_ATTEMPT
                or previous.action != event.action
                or previous.actor is not event.actor
            ):
                raise ReplayError("action_resolution must match the preceding attempt")
        if event.event_type is EventType.ROUND_BELL and (
            previous is None or previous.event_type is not EventType.ROUND_SCORE
        ):
            raise ReplayError("round_bell must follow round_score")
        if event.event_type is EventType.ROUND_RECOVERY and (
            previous is None or previous.event_type is not EventType.ROUND_BELL
        ):
            raise ReplayError("round_recovery must follow round_bell")
        if event.event_type is EventType.ROUND_START and index > 1 and (
            previous is None or previous.event_type is not EventType.ROUND_RECOVERY
        ):
            raise ReplayError("later round_start must follow round_recovery")
    previous_hash = GENESIS_EVENT_HASH
    for event in values:
        state = reduce_event(
            state,
            event,
            verify_hashes=verify_hashes,
            expected_previous_event_hash=previous_hash,
        )
        previous_hash = event.event_hash
    return state
