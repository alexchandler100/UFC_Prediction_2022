"""Piecewise-constant, event-sourced UFC fight simulation engine."""

from __future__ import annotations

import math
from typing import Iterable

from .domain import (
    DecisionType,
    EventType,
    FightResult,
    FightState,
    FighterParameters,
    FighterStats,
    JudgeRoundScore,
    OutcomeMethod,
    Phase,
    Side,
    SimulationEvent,
    SimulationPath,
    SimulationRunSpec,
    StateDelta,
    TelemetryLevel,
)
from .reducer import GENESIS_EVENT_HASH, apply_delta, initial_state, make_event, state_hash
from .rng import NamedRandomStreams


MICROSECONDS = 1_000_000
STRIKES_PER_EXCHANGE = 3
_EMPTY_DELTA = StateDelta()


class SimulationInvariantError(RuntimeError):
    """The engine reached an invalid or deliberately bounded state."""

    def __init__(self, message: str, events: Iterable[SimulationEvent] = ()) -> None:
        super().__init__(message)
        self.events = tuple(events)


_Hazard = tuple[str, Side | None, float]


class _MutableStats:
    __slots__ = tuple(FighterStats.__dataclass_fields__)

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0)

    def add(self, delta: FighterStats) -> None:
        for name in self.__slots__:
            value = getattr(delta, name)
            if value:
                setattr(self, name, getattr(self, name) + value)

    def freeze(self) -> FighterStats:
        return FighterStats(**{name: getattr(self, name) for name in self.__slots__})


class _LeanState:
    """Mutable projection of FightState used only for unretained bulk paths."""

    def __init__(self, matchup_id: str, scheduled_rounds: int) -> None:
        self.matchup_id = matchup_id
        self.scheduled_rounds = scheduled_rounds
        self.round_number = 1
        self.fight_time_us = 0
        self.round_time_us = 0
        self.phase = Phase.DISTANCE
        self.top_position: Side | None = None
        self.red_stamina = 1.0
        self.blue_stamina = 1.0
        self.red_hurt = 0.0
        self.blue_hurt = 0.0
        self.red_damage = 0.0
        self.blue_damage = 0.0
        self.red_stats = _MutableStats()
        self.blue_stats = _MutableStats()
        self.red_round_effectiveness = 0.0
        self.blue_round_effectiveness = 0.0
        self.round_scores: list[JudgeRoundScore] = []
        self.result: FightResult | None = None
        self.event_count = 0

    def apply(self, delta: StateDelta) -> None:
        if delta is _EMPTY_DELTA:
            self.event_count += 1
            return
        if delta.fight_time_us is not None:
            self.fight_time_us = delta.fight_time_us
        if delta.round_time_us is not None:
            self.round_time_us = delta.round_time_us
        if delta.round_number is not None:
            self.round_number = delta.round_number
        if delta.phase is not None:
            self.phase = delta.phase
        if delta.clear_top_position or self.phase is not Phase.GROUND:
            self.top_position = None
        if delta.top_position is not None:
            self.top_position = delta.top_position
        for name in (
            "red_stamina",
            "blue_stamina",
            "red_hurt",
            "blue_hurt",
            "red_damage",
            "blue_damage",
        ):
            value = getattr(delta, name)
            if value is not None:
                setattr(self, name, value)
        if delta.red_stats_delta is not None:
            self.red_stats.add(delta.red_stats_delta)
        if delta.blue_stats_delta is not None:
            self.blue_stats.add(delta.blue_stats_delta)
        if delta.reset_round_effectiveness:
            self.red_round_effectiveness = 0.0
            self.blue_round_effectiveness = 0.0
        else:
            self.red_round_effectiveness += delta.red_effectiveness_delta
            self.blue_round_effectiveness += delta.blue_effectiveness_delta
        if delta.append_round_score is not None:
            self.round_scores.append(delta.append_round_score)
        if delta.result is not None:
            self.result = delta.result
        self.event_count += 1

    def freeze(self) -> FightState:
        return FightState(
            matchup_id=self.matchup_id,
            scheduled_rounds=self.scheduled_rounds,
            round_number=self.round_number,
            fight_time_us=self.fight_time_us,
            round_time_us=self.round_time_us,
            phase=self.phase,
            top_position=self.top_position,
            red_stamina=self.red_stamina,
            blue_stamina=self.blue_stamina,
            red_hurt=self.red_hurt,
            blue_hurt=self.blue_hurt,
            red_damage=self.red_damage,
            blue_damage=self.blue_damage,
            red_stats=self.red_stats.freeze(),
            blue_stats=self.blue_stats.freeze(),
            red_round_effectiveness=self.red_round_effectiveness,
            blue_round_effectiveness=self.blue_round_effectiveness,
            round_scores=tuple(self.round_scores),
            result=self.result,
            event_count=self.event_count,
        )


class _Runtime:
    def __init__(
        self,
        spec: SimulationRunSpec,
        simulation_index: int,
        telemetry: TelemetryLevel,
    ) -> None:
        self.spec = spec
        self.red_parameters = spec.red.parameters
        self.blue_parameters = spec.blue.parameters
        self.simulation_index = simulation_index
        self.telemetry = telemetry
        self.state: FightState | _LeanState = (
            initial_state(spec.bout.matchup_id, spec.bout.scheduled_rounds)
            if telemetry is TelemetryLevel.FULL
            else _LeanState(spec.bout.matchup_id, spec.bout.scheduled_rounds)
        )
        self.rng = NamedRandomStreams(
            spec,
            simulation_index,
            record=telemetry is TelemetryLevel.FULL,
        )
        self.events: list[SimulationEvent] = []
        self.previous_event_hash = GENESIS_EVENT_HASH
        self.phase_time_us = {phase: 0 for phase in Phase}

    def emit(
        self,
        event_type: EventType,
        delta: StateDelta = _EMPTY_DELTA,
        *,
        actor: Side | None = None,
        target: Side | None = None,
        action: str = "",
        payload: dict[str, object] | None = None,
    ) -> None:
        if delta.fight_time_us is not None:
            elapsed_us = int(delta.fight_time_us) - int(self.state.fight_time_us)
            if elapsed_us > 0:
                self.phase_time_us[self.state.phase] += elapsed_us
        draws = self.rng.drain()
        if self.telemetry is TelemetryLevel.FULL:
            self.state, event = make_event(
                self.state,
                event_type,
                delta,
                actor=actor,
                target=target,
                action=action,
                rng_draws=draws,
                previous_event_hash=self.previous_event_hash,
                payload=payload,
            )
            self.events.append(event)
            self.previous_event_hash = event.event_hash
        else:
            # NONE/COMPACT never retain individual events.  Applying the same
            # complete delta to a mutable projection removes allocation cost;
            # freeze() below validates the identical public FightState.
            assert isinstance(self.state, _LeanState)
            self.state.apply(delta)

    def frozen_state(self) -> FightState:
        return self.state if isinstance(self.state, FightState) else self.state.freeze()


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _runtime_parameters(runtime: _Runtime, side: Side) -> FighterParameters:
    return runtime.red_parameters if side is Side.RED else runtime.blue_parameters


def _stamina(state: FightState, side: Side) -> float:
    return state.red_stamina if side is Side.RED else state.blue_stamina


def _hurt(state: FightState, side: Side) -> float:
    return state.red_hurt if side is Side.RED else state.blue_hurt


def _damage(state: FightState, side: Side) -> float:
    return state.red_damage if side is Side.RED else state.blue_damage


def _effectiveness_delta(side: Side, amount: float) -> dict[str, float]:
    return (
        {"red_effectiveness_delta": amount}
        if side is Side.RED
        else {"blue_effectiveness_delta": amount}
    )


def _dynamic_delta(
    side: Side,
    *,
    stamina: float | None = None,
    hurt: float | None = None,
    damage: float | None = None,
) -> dict[str, float | None]:
    prefix = side.value
    values: dict[str, float | None] = {}
    if stamina is not None:
        values[f"{prefix}_stamina"] = stamina
    if hurt is not None:
        values[f"{prefix}_hurt"] = hurt
    if damage is not None:
        values[f"{prefix}_damage"] = damage
    return values


def _hazard_rate(spec: SimulationRunSpec, rate_per_minute: float) -> float:
    config = spec.simulator
    bounded = _clip(
        float(rate_per_minute),
        config.min_hazard_per_minute,
        config.max_hazard_per_minute,
    )
    return bounded / 60.0


def _hazards(runtime: _Runtime) -> tuple[_Hazard, ...]:
    state = runtime.state
    spec = runtime.spec
    hazards: list[_Hazard] = []

    def add(action: str, side: Side, rate: float, multiplier: float = 1.0) -> None:
        if side is Side.RED:
            pace = _clip((0.30 + 0.70 * state.red_stamina) * (1.0 - 0.45 * state.red_hurt), 0.15, 1.0)
        else:
            pace = _clip((0.30 + 0.70 * state.blue_stamina) * (1.0 - 0.45 * state.blue_hurt), 0.15, 1.0)
        value = _hazard_rate(spec, rate * multiplier * pace)
        if value > 0:
            hazards.append((action, side, value))

    for side in (Side.RED, Side.BLUE):
        params = _runtime_parameters(runtime, side)
        if state.phase is Phase.DISTANCE:
            add(
                "strike",
                side,
                params.strike_rate_distance / STRIKES_PER_EXCHANGE,
                spec.simulator.distance_strike_hazard_multiplier,
            )
            add(
                "clinch_entry",
                side,
                params.clinch_entry_rate,
                spec.simulator.clinch_entry_hazard_multiplier,
            )
            add(
                "takedown",
                side,
                params.takedown_attempt_rate,
                0.60 * spec.simulator.takedown_hazard_multiplier,
            )
        elif state.phase is Phase.CLINCH:
            add(
                "strike",
                side,
                params.strike_rate_clinch / STRIKES_PER_EXCHANGE,
                spec.simulator.clinch_strike_hazard_multiplier,
            )
            add(
                "takedown",
                side,
                params.takedown_attempt_rate,
                1.25 * spec.simulator.takedown_hazard_multiplier,
            )
            add(
                "clinch_exit",
                side,
                params.clinch_exit_rate,
                spec.simulator.clinch_exit_hazard_multiplier,
            )
        elif state.phase is Phase.GROUND:
            if state.top_position is side:
                add(
                    "strike",
                    side,
                    params.strike_rate_ground / STRIKES_PER_EXCHANGE,
                    spec.simulator.ground_strike_hazard_multiplier,
                )
                add(
                    "submission",
                    side,
                    params.submission_attempt_rate,
                    spec.simulator.submission_hazard_multiplier,
                )
            else:
                add(
                    "strike",
                    side,
                    params.strike_rate_ground / STRIKES_PER_EXCHANGE,
                    0.18 * spec.simulator.ground_strike_hazard_multiplier,
                )
                add(
                    "submission",
                    side,
                    params.submission_attempt_rate,
                    0.55 * spec.simulator.submission_hazard_multiplier,
                )
                add(
                    "escape",
                    side,
                    params.escape_rate,
                    spec.simulator.escape_hazard_multiplier,
                )
        else:
            add("scramble_ground", side, 2.0 + params.ground_control_rate)
            add("scramble_distance", side, 1.6 + params.escape_rate)
    # These rare outcomes are globally fitted, never fighter-specific.  The
    # total other-result hazard is split symmetrically between the two sides.
    no_contest = _hazard_rate(spec, spec.simulator.no_contest_rate_per_minute)
    if no_contest > 0:
        hazards.append(("no_contest", None, no_contest))
    other_side = _hazard_rate(
        spec, spec.simulator.other_finish_rate_per_minute / 2.0
    )
    if other_side > 0:
        hazards.extend(
            ("other_finish", side, other_side) for side in (Side.RED, Side.BLUE)
        )
    return tuple(hazards)


def _control_delta(state: FightState, elapsed_us: int) -> tuple[FighterStats | None, FighterStats | None]:
    if state.phase is not Phase.GROUND or state.top_position is None:
        return None, None
    stats = FighterStats(control_time_us=elapsed_us)
    return (stats, None) if state.top_position is Side.RED else (None, stats)


def _advance(runtime: _Runtime, target_round_time_us: int) -> None:
    state = runtime.state
    elapsed = target_round_time_us - state.round_time_us
    if elapsed <= 0:
        raise SimulationInvariantError("time advance must be positive", runtime.events)
    red_control, blue_control = _control_delta(state, elapsed)
    runtime.emit(
        EventType.TIME_ADVANCE,
        StateDelta(
            fight_time_us=state.fight_time_us + elapsed,
            round_time_us=target_round_time_us,
            red_stats_delta=red_control,
            blue_stats_delta=blue_control,
        ),
        payload={"elapsed_us": elapsed},
    )


def _tick(runtime: _Runtime, elapsed_seconds: float) -> None:
    state = runtime.state
    updates: dict[str, float] = {}
    intensity = {
        Phase.DISTANCE: 0.85,
        Phase.CLINCH: 1.05,
        Phase.GROUND: 1.00,
        Phase.SCRAMBLE: 1.20,
    }[state.phase]
    for side in (Side.RED, Side.BLUE):
        params = _runtime_parameters(runtime, side)
        minutes = elapsed_seconds / 60.0
        fatigue = minutes * intensity * (0.015 + 0.085 * params.pace_decay)
        updates.update(
            _dynamic_delta(
                side,
                stamina=_clip(_stamina(state, side) - fatigue, 0.0, 1.0),
                hurt=_clip(
                    _hurt(state, side)
                    * math.exp(-params.hurt_recovery_per_minute * minutes),
                    0.0,
                    1.0,
                ),
            )
        )
    runtime.emit(
        EventType.DYNAMICS_TICK,
        StateDelta(**updates),
        payload={"elapsed_seconds": elapsed_seconds},
    )


def _strike_probability(runtime: _Runtime, actor: Side) -> float:
    actor_params = _runtime_parameters(runtime, actor)
    target_params = _runtime_parameters(runtime, actor.opponent)
    state = runtime.state
    phase_modifier = {
        Phase.DISTANCE: 1.0,
        Phase.CLINCH: 1.08,
        Phase.GROUND: 1.14 if state.top_position is actor else 0.80,
        Phase.SCRAMBLE: 0.75,
    }[state.phase]
    defense_adjustment = 1.0 + 0.75 * (0.50 - target_params.strike_defense)
    condition = 0.75 + 0.25 * _stamina(state, actor)
    vulnerability = 1.0 + 0.25 * _hurt(state, actor.opponent)
    return _clip(
        actor_params.strike_accuracy * phase_modifier * defense_adjustment * condition * vulnerability,
        0.03,
        0.92,
    )


def _transition_probability(runtime: _Runtime, action: str, actor: Side) -> float:
    actor_params = _runtime_parameters(runtime, actor)
    target_params = _runtime_parameters(runtime, actor.opponent)
    state = runtime.state
    if action == "takedown":
        base = actor_params.takedown_accuracy * (1.0 + 0.75 * (0.50 - target_params.takedown_defense))
        return _clip(base * (0.75 + 0.25 * _stamina(state, actor)), 0.03, 0.92)
    if action == "submission":
        base = (
            actor_params.submission_finish_probability
            * runtime.spec.simulator.submission_finish_probability_multiplier
        )
        defense = 1.0 + 1.1 * (0.50 - target_params.submission_defense)
        position = 1.20 if state.top_position is actor else 0.72
        vulnerability = 1.0 + 0.65 * _hurt(state, actor.opponent) + 0.35 * (1.0 - _stamina(state, actor.opponent))
        return _clip(base * defense * position * vulnerability, 0.005, 0.75)
    if action == "escape":
        control = target_params.ground_control_rate
        return _clip(0.42 + 0.28 * actor_params.escape_rate - 0.16 * control, 0.08, 0.88)
    if action == "clinch_entry":
        return _clip(0.52 + 0.12 * actor_params.ground_control_rate - 0.10 * target_params.takedown_defense, 0.12, 0.85)
    if action == "clinch_exit":
        return _clip(0.58 + 0.16 * actor_params.clinch_exit_rate - 0.08 * target_params.ground_control_rate, 0.18, 0.92)
    return 1.0


def _strike_stats(
    runtime: _Runtime, actor: Side, attempts: int, landed: int
) -> FighterStats:
    if landed:
        parameters = _runtime_parameters(runtime, actor)
        head = runtime.rng.binomial(
            "strike.target.head", landed, parameters.head_target_share
        )
        remaining = landed - head
        non_head_share = parameters.body_target_share + parameters.leg_target_share
        conditional_body_share = (
            parameters.body_target_share / non_head_share
            if non_head_share > 0
            else 0.0
        )
        body = runtime.rng.binomial(
            "strike.target.body", remaining, conditional_body_share
        )
        leg = remaining - body
    else:
        head = body = leg = 0
    phase = runtime.state.phase
    return FighterStats(
        strike_attempts=attempts,
        strikes_landed=landed,
        significant_strike_attempts=attempts,
        significant_strikes_landed=landed,
        head_landed=head,
        body_landed=body,
        leg_landed=leg,
        distance_attempts=attempts if phase is Phase.DISTANCE else 0,
        distance_landed=landed if phase is Phase.DISTANCE else 0,
        clinch_attempts=attempts if phase is Phase.CLINCH else 0,
        clinch_landed=landed if phase is Phase.CLINCH else 0,
        ground_attempts=attempts if phase is Phase.GROUND else 0,
        ground_landed=landed if phase is Phase.GROUND else 0,
    )


def _resolution_stats(action: str, runtime: _Runtime, actor: Side, success: bool) -> FighterStats | None:
    if action == "takedown":
        return FighterStats(takedown_attempts=1, takedowns_landed=int(success))
    if action == "submission":
        return FighterStats(submission_attempts=1)
    return None


def _terminate(runtime: _Runtime, winner: Side | None, method: OutcomeMethod, reason: str, decision_type: DecisionType | None = None) -> None:
    state = runtime.state
    result = FightResult(
        winner=winner,
        method=method,
        round_number=state.round_number,
        fight_time_us=state.fight_time_us,
        round_time_us=state.round_time_us,
        reason=reason,
        decision_type=decision_type,
    )
    runtime.emit(
        EventType.TERMINATION,
        StateDelta(result=result),
        actor=winner,
        target=winner.opponent if winner else None,
        action=method.value,
        payload={"reason": reason},
    )


def _strike_consequence(runtime: _Runtime, actor: Side, landed: int) -> None:
    state = runtime.state
    target = actor.opponent
    actor_params = _runtime_parameters(runtime, actor)
    target_params = _runtime_parameters(runtime, target)
    severity = actor_params.strike_power * (0.70 + 0.60 * runtime.rng.uniform("strike.severity"))
    current_damage = _damage(state, target)
    current_hurt = _hurt(state, target)
    damage_increment = landed * (0.025 + 0.085 * severity)
    hurt_increment = landed * (0.035 + 0.22 * severity)
    knockdown_probability_per_landed = _clip(
        actor_params.knockdown_rate_per_landed
        * (0.75 + 0.90 * severity)
        * (1.25 - 0.70 * target_params.ko_resistance)
        * (1.0 + 0.80 * current_hurt + 0.18 * current_damage),
        0.0,
        0.75,
    )
    knockdown_probability_per_landed = _clip(
        knockdown_probability_per_landed
        * runtime.spec.simulator.knockdown_probability_multiplier,
        0.0,
        0.75,
    )
    knockdowns = runtime.rng.binomial("strike.knockdown", landed, knockdown_probability_per_landed)
    stats = FighterStats(knockdowns=knockdowns)
    effectiveness = landed * (1.0 + 1.75 * severity) + 4.0 * knockdowns
    target_hurt = _clip(current_hurt + hurt_increment + (0.40 if knockdowns else 0.0), 0.0, 1.0)
    target_damage = _clip(
        current_damage + damage_increment + 0.10 * knockdowns,
        0.0,
        1.0,
    )
    delta_values = {
        **_dynamic_delta(target, hurt=target_hurt, damage=target_damage),
        **_effectiveness_delta(actor, effectiveness),
        "red_stats_delta": stats if actor is Side.RED else None,
        "blue_stats_delta": stats if actor is Side.BLUE else None,
    }
    runtime.emit(
        EventType.ACTION_CONSEQUENCE,
        StateDelta(**delta_values),
        actor=actor,
        target=target,
        action="strike",
        payload={
            "severity": severity,
            "landed": landed,
            "knockdowns": knockdowns,
            "knockdown_probability_per_landed": knockdown_probability_per_landed,
        },
    )
    if knockdowns:
        finish_multiplier = runtime.spec.simulator.ko_tko_finish_probability_multiplier
        finish_probability_per_knockdown = _clip(
            actor_params.finish_after_knockdown
            * (1.30 - 0.65 * target_params.ko_resistance)
            * (1.0 + 0.45 * target_hurt + 0.12 * target_damage)
            * finish_multiplier,
            0.01 if finish_multiplier > 0.0 else 0.0,
            0.95,
        )
        finish_probability = 1.0 - (1.0 - finish_probability_per_knockdown) ** knockdowns
        finished = runtime.rng.uniform("strike.finish") < finish_probability
        # Keep the finish draw attached to a diagnostic consequence even when
        # it fails; otherwise the trace would hide a decisive stochastic draw.
        runtime.emit(
            EventType.ACTION_CONSEQUENCE,
            _EMPTY_DELTA,
            actor=actor,
            target=target,
            action="knockdown_follow_up",
            payload={"finished": finished, "finish_probability": finish_probability},
        )
        if finished:
            _terminate(runtime, actor, OutcomeMethod.KO_TKO, "finish after knockdown")


def _perform_action(runtime: _Runtime, hazard: _Hazard) -> None:
    action, actor, hazard_rate = hazard
    target = actor.opponent if actor is not None else None
    runtime.emit(
        EventType.ACTION_ATTEMPT,
        _EMPTY_DELTA,
        actor=actor,
        target=target,
        action=action,
        payload={"hazard_per_second": hazard_rate},
    )

    probability = 1.0
    if action in {"no_contest", "other_finish"}:
        success = True
        stats = None
        landed = 0
    elif action == "strike":
        probability = _strike_probability(runtime, actor)
        landed = runtime.rng.binomial(
            "resolution.strike.landed", STRIKES_PER_EXCHANGE, probability
        )
        success = landed > 0
        stats = _strike_stats(runtime, actor, STRIKES_PER_EXCHANGE, landed)
    elif action not in ("scramble_ground", "scramble_distance"):
        probability = _transition_probability(runtime, action, actor)
        success = runtime.rng.uniform(f"resolution.{action}") < probability
        stats = _resolution_stats(action, runtime, actor, success)
        landed = 0
    else:
        success = runtime.rng.uniform(f"resolution.{action}") < probability
        stats = None
        landed = 0
    runtime.emit(
        EventType.ACTION_RESOLUTION,
        StateDelta(
            red_stats_delta=stats if actor is Side.RED else None,
            blue_stats_delta=stats if actor is Side.BLUE else None,
        ),
        actor=actor,
        target=target,
        action=action,
        payload={"success": success, "probability": probability, "attempts": STRIKES_PER_EXCHANGE if action == "strike" else 1, "landed": landed},
    )

    if not success:
        runtime.emit(
            EventType.ACTION_CONSEQUENCE,
            _EMPTY_DELTA,
            actor=actor,
            target=target,
            action=action,
            payload={"state_change": False},
        )
        return
    if action == "no_contest":
        runtime.emit(
            EventType.ACTION_CONSEQUENCE,
            _EMPTY_DELTA,
            action=action,
            payload={"state_change": False, "global_rare_outcome": True},
        )
        _terminate(runtime, None, OutcomeMethod.NO_CONTEST, "global rare no-contest process")
        return
    if action == "other_finish":
        if actor is None:  # pragma: no cover - protected by hazard construction
            raise SimulationInvariantError("other finish requires an actor", runtime.events)
        runtime.emit(
            EventType.ACTION_CONSEQUENCE,
            StateDelta(**_effectiveness_delta(actor, 1.0)),
            actor=actor,
            target=target,
            action=action,
            payload={"state_change": True, "global_rare_outcome": True},
        )
        _terminate(runtime, actor, OutcomeMethod.OTHER, "global rare other-result process")
        return
    if action == "strike":
        _strike_consequence(runtime, actor, landed)
        return
    if action == "takedown":
        runtime.emit(
            EventType.ACTION_CONSEQUENCE,
            StateDelta(
                phase=Phase.GROUND,
                top_position=actor,
                **_effectiveness_delta(actor, 2.25),
            ),
            actor=actor,
            target=target,
            action=action,
            payload={"state_change": True},
        )
        return
    if action == "submission":
        runtime.emit(
            EventType.ACTION_CONSEQUENCE,
            StateDelta(**_effectiveness_delta(actor, 3.0)),
            actor=actor,
            target=target,
            action=action,
            payload={"state_change": True},
        )
        _terminate(runtime, actor, OutcomeMethod.SUBMISSION, "completed submission")
        return
    reversal = False
    if action == "escape":
        reversal_probability = _runtime_parameters(
            runtime, actor
        ).reversal_after_escape
        reversal = (
            reversal_probability >= 1.0
            or runtime.rng.uniform("escape.reversal") < reversal_probability
        )
    if action == "clinch_entry":
        phase, top = Phase.CLINCH, None
    elif action == "clinch_exit":
        phase, top = Phase.DISTANCE, None
    elif action == "escape":
        phase, top = Phase.SCRAMBLE, None
    elif action == "scramble_ground":
        phase, top = Phase.GROUND, actor
    else:
        phase, top = Phase.DISTANCE, None
    runtime.emit(
        EventType.ACTION_CONSEQUENCE,
        StateDelta(
            phase=phase,
            top_position=top,
            clear_top_position=top is None,
            red_stats_delta=(
                FighterStats(reversals=1)
                if actor is Side.RED and action == "escape" and reversal
                else None
            ),
            blue_stats_delta=(
                FighterStats(reversals=1)
                if actor is Side.BLUE and action == "escape" and reversal
                else None
            ),
        ),
        actor=actor,
        target=target,
        action=action,
        payload={
            "state_change": True,
            "phase": phase.value,
            "reversal": reversal,
        },
    )


def _round_score(runtime: _Runtime) -> JudgeRoundScore:
    state = runtime.state
    shared = runtime.rng.normal("judge.shared", 0.0, 1.0)
    rho = runtime.spec.simulator.judge_correlation
    red_points: list[int] = []
    blue_points: list[int] = []
    base_difference = state.red_round_effectiveness - state.blue_round_effectiveness
    for judge in range(3):
        independent = runtime.rng.normal(f"judge.{judge}", 0.0, 1.0)
        noise = runtime.spec.simulator.judge_noise_sd * (
            math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * independent
        )
        difference = base_difference + noise
        if abs(difference) < 0.15:
            red, blue = 10, 10
        elif difference > 0:
            red = 10
            blue = 8 if difference >= runtime.spec.simulator.ten_eight_threshold else 9
        else:
            blue = 10
            red = 8 if -difference >= runtime.spec.simulator.ten_eight_threshold else 9
        red_points.append(red)
        blue_points.append(blue)
    return JudgeRoundScore(
        round_number=state.round_number,
        red_points=tuple(red_points),  # type: ignore[arg-type]
        blue_points=tuple(blue_points),  # type: ignore[arg-type]
        red_effectiveness=state.red_round_effectiveness,
        blue_effectiveness=state.blue_round_effectiveness,
    )


def _decision(runtime: _Runtime) -> tuple[Side | None, OutcomeMethod, DecisionType, str]:
    red_votes = 0
    blue_votes = 0
    ties = 0
    for judge in range(3):
        red_total = sum(score.red_points[judge] for score in runtime.state.round_scores)
        blue_total = sum(score.blue_points[judge] for score in runtime.state.round_scores)
        if red_total > blue_total:
            red_votes += 1
        elif blue_total > red_total:
            blue_votes += 1
        else:
            ties += 1
    if red_votes >= 2:
        kind = DecisionType.UNANIMOUS if red_votes == 3 else (DecisionType.MAJORITY if ties else DecisionType.SPLIT)
        return Side.RED, OutcomeMethod.DECISION, kind, f"{red_votes}-{blue_votes}-{ties} judge votes"
    if blue_votes >= 2:
        kind = DecisionType.UNANIMOUS if blue_votes == 3 else (DecisionType.MAJORITY if ties else DecisionType.SPLIT)
        return Side.BLUE, OutcomeMethod.DECISION, kind, f"{blue_votes}-{red_votes}-{ties} judge votes"
    return None, OutcomeMethod.DRAW, DecisionType.DRAW, f"{red_votes}-{blue_votes}-{ties} judge votes"


def _close_round(runtime: _Runtime) -> None:
    state = runtime.state
    score = _round_score(runtime)
    runtime.emit(
        EventType.ROUND_SCORE,
        StateDelta(append_round_score=score),
        payload={"red_effectiveness": score.red_effectiveness, "blue_effectiveness": score.blue_effectiveness},
    )
    runtime.emit(EventType.ROUND_BELL, _EMPTY_DELTA, payload={"completed_round": state.round_number})
    if state.round_number == runtime.spec.bout.scheduled_rounds:
        winner, method, decision_type, reason = _decision(runtime)
        _terminate(runtime, winner, method, reason, decision_type)
        return

    updates: dict[str, float] = {}
    for side in (Side.RED, Side.BLUE):
        params = _runtime_parameters(runtime, side)
        updates.update(
            _dynamic_delta(
                side,
                stamina=_clip(_stamina(runtime.state, side) + params.stamina_recovery_between_rounds, 0.0, 1.0),
                hurt=_clip(_hurt(runtime.state, side) * 0.60, 0.0, 1.0),
            )
        )
    runtime.emit(
        EventType.ROUND_RECOVERY,
        StateDelta(**updates),
        payload={"rest_seconds": runtime.spec.bout.rest_seconds},
    )
    runtime.emit(
        EventType.ROUND_START,
        StateDelta(
            round_number=state.round_number + 1,
            round_time_us=0,
            phase=Phase.DISTANCE,
            clear_top_position=True,
            reset_round_effectiveness=True,
        ),
        payload={"round": state.round_number + 1},
    )


def _simulate_fight_once(
    spec: SimulationRunSpec,
    simulation_index: int,
    *,
    telemetry: TelemetryLevel | str = TelemetryLevel.NONE,
) -> SimulationPath:
    """Simulate one directly addressable path from a frozen run specification."""

    telemetry = TelemetryLevel(telemetry)
    runtime = _Runtime(spec, simulation_index, telemetry)
    runtime.emit(EventType.FIGHT_START, payload={"engine_version": spec.engine_version})
    runtime.emit(EventType.ROUND_START, payload={"round": 1})
    round_us = spec.bout.round_seconds * MICROSECONDS
    dynamics_us = spec.bout.dynamics_seconds * MICROSECONDS

    while runtime.state.result is None:
        if runtime.state.event_count >= spec.simulator.max_events:
            raise SimulationInvariantError(
                f"event cap {spec.simulator.max_events} reached without termination",
                runtime.events,
            )
        state = runtime.state
        boundary = min(((state.round_time_us // dynamics_us) + 1) * dynamics_us, round_us)
        hazards = _hazards(runtime)
        total_rate = math.fsum(item[2] for item in hazards)
        if total_rate > 0:
            wait_seconds = runtime.rng.exponential("clock", total_rate)
            wait_us = max(1, int(round(wait_seconds * MICROSECONDS)))
        else:
            wait_us = boundary - state.round_time_us

        # Ticks and bells preempt a sampled action at or beyond their boundary.
        if state.round_time_us + wait_us >= boundary:
            _advance(runtime, boundary)
            if boundary == round_us:
                _close_round(runtime)
            else:
                # Dynamics are integrated once for the complete fixed episode;
                # intervening actions do not make fatigue/recovery time vanish.
                _tick(runtime, float(spec.bout.dynamics_seconds))
            continue

        _advance(runtime, state.round_time_us + wait_us)
        if not hazards:
            raise SimulationInvariantError("positive action wait without hazards", runtime.events)
        chosen = runtime.rng.weighted_choice(
            "action.selection",
            ((hazard, hazard[2]) for hazard in hazards),
        )
        _perform_action(runtime, chosen)

    result = runtime.state.result
    if result is None:  # pragma: no cover - protected by loop condition
        raise SimulationInvariantError("simulation ended without a result", runtime.events)
    final_state = runtime.frozen_state()
    return SimulationPath(
        matchup_id=spec.bout.matchup_id,
        scheduled_rounds=spec.bout.scheduled_rounds,
        bootstrap_member=spec.bootstrap_member,
        simulation_index=simulation_index,
        result=result,
        red_stats=final_state.red_stats,
        blue_stats=final_state.blue_stats,
        final_state_hash=state_hash(final_state),
        phase_time_us=tuple(
            (phase.value, int(runtime.phase_time_us[phase])) for phase in Phase
        ),
        events=tuple(runtime.events),
    )


def simulate_fight(
    spec: SimulationRunSpec,
    simulation_index: int,
    *,
    telemetry: TelemetryLevel | str = TelemetryLevel.NONE,
) -> SimulationPath:
    """Simulate a path and deterministically capture any invariant failure."""

    level = TelemetryLevel(telemetry)
    try:
        return _simulate_fight_once(spec, simulation_index, telemetry=level)
    except SimulationInvariantError as original:
        if level is TelemetryLevel.FULL:
            raise
        try:
            _simulate_fight_once(spec, simulation_index, telemetry=TelemetryLevel.FULL)
        except SimulationInvariantError as traced:
            raise traced from original
        raise SimulationInvariantError(
            "invariant failure did not reproduce with full telemetry",
            original.events,
        ) from original


def simulate_indices(
    spec: SimulationRunSpec,
    simulation_indices: Iterable[int],
    *,
    telemetry: TelemetryLevel | str = TelemetryLevel.NONE,
) -> tuple[SimulationPath, ...]:
    """Simulate explicit indices in caller order without shared RNG state."""

    return tuple(
        simulate_fight(spec, int(index), telemetry=telemetry)
        for index in simulation_indices
    )
