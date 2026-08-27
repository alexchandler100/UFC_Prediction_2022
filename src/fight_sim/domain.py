"""Immutable, versioned contracts for the evidence-first fight simulator.

The simulator deliberately models only quantities that can be estimated from the
available aggregate and per-round data.  These contracts are independent of the
parameter fitter so historical snapshots and future fitting implementations can
be replayed against a frozen engine version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Iterable


SCHEMA_VERSION = "fight-sim.v1"
RNG_CONTRACT_VERSION = "fight-sim-rng.v1"
ENGINE_VERSION = "fight-sim-engine.v2"


class Side(str, Enum):
    RED = "red"
    BLUE = "blue"

    @property
    def opponent(self) -> "Side":
        return Side.BLUE if self is Side.RED else Side.RED


class Phase(str, Enum):
    DISTANCE = "distance"
    CLINCH = "clinch"
    GROUND = "ground"
    SCRAMBLE = "scramble"


class OutcomeMethod(str, Enum):
    KO_TKO = "ko_tko"
    SUBMISSION = "submission"
    DECISION = "decision"
    DRAW = "draw"
    OTHER = "other"
    NO_CONTEST = "no_contest"


class DecisionType(str, Enum):
    UNANIMOUS = "unanimous"
    SPLIT = "split"
    MAJORITY = "majority"
    DRAW = "draw"


class EventType(str, Enum):
    FIGHT_START = "fight_start"
    ROUND_START = "round_start"
    TIME_ADVANCE = "time_advance"
    DYNAMICS_TICK = "dynamics_tick"
    ACTION_ATTEMPT = "action_attempt"
    ACTION_RESOLUTION = "action_resolution"
    ACTION_CONSEQUENCE = "action_consequence"
    ROUND_SCORE = "round_score"
    ROUND_BELL = "round_bell"
    ROUND_RECOVERY = "round_recovery"
    TERMINATION = "termination"


class TelemetryLevel(str, Enum):
    NONE = "none"
    COMPACT = "compact"
    FULL = "full"


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _bounded(name: str, value: float, lower: float, upper: float) -> None:
    _finite(name, value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")


def _nonnegative(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class FighterParameters:
    """Pooled, observable fighter parameters for one frozen fit member.

    Rates are per active fight minute.  Probability fields are conditional on
    the corresponding attempt.  Defaults are intentionally population-like so
    sparse fighter snapshots shrink toward a usable, symmetric simulator.
    """

    strike_rate_distance: float = 7.0
    strike_rate_clinch: float = 4.0
    strike_rate_ground: float = 5.0
    distance_phase_share: float = 0.83
    clinch_phase_share: float = 0.085
    ground_phase_share: float = 0.085
    strike_accuracy: float = 0.45
    strike_defense: float = 0.55
    head_target_share: float = 0.62
    body_target_share: float = 0.24
    leg_target_share: float = 0.14
    strike_power: float = 0.50
    knockdown_rate_per_landed: float = 0.025
    finish_after_knockdown: float = 0.18
    clinch_entry_rate: float = 0.35
    clinch_exit_rate: float = 0.70
    takedown_attempt_rate: float = 0.65
    takedown_accuracy: float = 0.40
    takedown_defense: float = 0.60
    ground_control_rate: float = 0.55
    escape_rate: float = 0.45
    reversal_after_escape: float = 0.12
    submission_attempt_rate: float = 0.28
    submission_finish_probability: float = 0.12
    submission_defense: float = 0.58
    ko_resistance: float = 0.55
    hurt_recovery_per_minute: float = 0.35
    pace_decay: float = 0.12
    stamina_recovery_between_rounds: float = 0.15

    def __post_init__(self) -> None:
        for name in (
            "strike_rate_distance",
            "strike_rate_clinch",
            "strike_rate_ground",
            "clinch_entry_rate",
            "clinch_exit_rate",
            "takedown_attempt_rate",
            "ground_control_rate",
            "escape_rate",
            "submission_attempt_rate",
            "hurt_recovery_per_minute",
        ):
            _nonnegative(name, float(getattr(self, name)))
        for name in (
            "strike_accuracy",
            "strike_defense",
            "distance_phase_share",
            "clinch_phase_share",
            "ground_phase_share",
            "head_target_share",
            "body_target_share",
            "leg_target_share",
            "strike_power",
            "knockdown_rate_per_landed",
            "finish_after_knockdown",
            "takedown_accuracy",
            "takedown_defense",
            "reversal_after_escape",
            "submission_finish_probability",
            "submission_defense",
            "ko_resistance",
            "pace_decay",
            "stamina_recovery_between_rounds",
        ):
            _bounded(name, float(getattr(self, name)), 0.0, 1.0)
        if not math.isclose(
            self.head_target_share + self.body_target_share + self.leg_target_share,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("strike target shares must sum to one")
        if not math.isclose(
            self.distance_phase_share
            + self.clinch_phase_share
            + self.ground_phase_share,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("strike phase shares must sum to one")

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FighterParameters":
        return cls(**{name: float(value[name]) for name in cls.__dataclass_fields__ if name in value})


@dataclass(frozen=True, slots=True)
class FighterSnapshot:
    fighter_id: str
    fighter_name: str
    as_of_utc: str
    division: str
    parameters: FighterParameters = field(default_factory=FighterParameters)
    age_years: float | None = None
    experience_fights: int = 0
    layoff_days: int | None = None
    observed_fight_seconds: float = 0.0
    observed_rounds: int = 0
    data_quality: str = "unknown"
    source_hash: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.fighter_id.strip():
            raise ValueError("fighter_id is required")
        if not self.fighter_name.strip():
            raise ValueError("fighter_name is required")
        if not self.as_of_utc.strip():
            raise ValueError("as_of_utc is required")
        if self.age_years is not None:
            _bounded("age_years", float(self.age_years), 0.0, 100.0)
        if self.experience_fights < 0 or self.observed_rounds < 0:
            raise ValueError("experience counts must be nonnegative")
        if self.layoff_days is not None and self.layoff_days < 0:
            raise ValueError("layoff_days must be nonnegative")
        _nonnegative("observed_fight_seconds", float(self.observed_fight_seconds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fighter_id": self.fighter_id,
            "fighter_name": self.fighter_name,
            "as_of_utc": self.as_of_utc,
            "division": self.division,
            "age_years": self.age_years,
            "experience_fights": self.experience_fights,
            "layoff_days": self.layoff_days,
            "observed_fight_seconds": self.observed_fight_seconds,
            "observed_rounds": self.observed_rounds,
            "data_quality": self.data_quality,
            "source_hash": self.source_hash,
            "parameters": self.parameters.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FighterSnapshot":
        payload = dict(value)
        payload["parameters"] = FighterParameters.from_dict(payload.get("parameters", {}))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class BoutConfig:
    matchup_id: str
    red_fighter_id: str
    blue_fighter_id: str
    scheduled_rounds: int = 3
    round_seconds: int = 300
    rest_seconds: int = 60
    dynamics_seconds: int = 5
    division: str = ""
    title_bout: bool = False
    event_id: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.matchup_id.strip():
            raise ValueError("matchup_id is required")
        if not self.red_fighter_id or not self.blue_fighter_id:
            raise ValueError("both fighter ids are required")
        if self.red_fighter_id == self.blue_fighter_id:
            raise ValueError("fighters must be distinct")
        if self.scheduled_rounds < 1 or self.scheduled_rounds > 5:
            raise ValueError("scheduled_rounds must be between one and five")
        if self.round_seconds <= 0 or self.rest_seconds < 0 or self.dynamics_seconds <= 0:
            raise ValueError("round/dynamics seconds must be positive and rest nonnegative")
        if self.round_seconds % self.dynamics_seconds:
            raise ValueError("dynamics_seconds must divide round_seconds")

    @property
    def horizon_us(self) -> int:
        return self.scheduled_rounds * self.round_seconds * 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BoutConfig":
        return cls(**value)


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    judge_noise_sd: float = 1.25
    judge_correlation: float = 0.55
    ten_eight_threshold: float = 7.5
    max_events: int = 50_000
    min_hazard_per_minute: float = 0.0
    max_hazard_per_minute: float = 100.0
    no_contest_rate_per_minute: float = 0.0
    other_finish_rate_per_minute: float = 0.0
    distance_strike_hazard_multiplier: float = 1.0
    clinch_strike_hazard_multiplier: float = 1.0
    ground_strike_hazard_multiplier: float = 1.0
    takedown_hazard_multiplier: float = 1.0
    submission_hazard_multiplier: float = 1.0
    clinch_entry_hazard_multiplier: float = 1.0
    clinch_exit_hazard_multiplier: float = 1.0
    escape_hazard_multiplier: float = 1.0
    knockdown_probability_multiplier: float = 1.0
    official_knockdown_observation_probability: float = 1.0
    ko_tko_finish_probability_multiplier: float = 1.0
    submission_finish_probability_multiplier: float = 1.0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonnegative("judge_noise_sd", self.judge_noise_sd)
        _bounded("judge_correlation", self.judge_correlation, 0.0, 1.0)
        _bounded(
            "official_knockdown_observation_probability",
            self.official_knockdown_observation_probability,
            0.0,
            1.0,
        )
        _nonnegative("ten_eight_threshold", self.ten_eight_threshold)
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        _nonnegative("min_hazard_per_minute", self.min_hazard_per_minute)
        _nonnegative("no_contest_rate_per_minute", self.no_contest_rate_per_minute)
        _nonnegative("other_finish_rate_per_minute", self.other_finish_rate_per_minute)
        for name in (
            "distance_strike_hazard_multiplier",
            "clinch_strike_hazard_multiplier",
            "ground_strike_hazard_multiplier",
            "takedown_hazard_multiplier",
            "submission_hazard_multiplier",
            "clinch_entry_hazard_multiplier",
            "clinch_exit_hazard_multiplier",
            "escape_hazard_multiplier",
            "knockdown_probability_multiplier",
            "ko_tko_finish_probability_multiplier",
            "submission_finish_probability_multiplier",
        ):
            _nonnegative(name, float(getattr(self, name)))
        if self.max_hazard_per_minute <= self.min_hazard_per_minute:
            raise ValueError("max_hazard_per_minute must exceed minimum")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class SimulationRunSpec:
    bout: BoutConfig
    red: FighterSnapshot
    blue: FighterSnapshot
    root_seed: str | int
    parameter_artifact_id: str
    bootstrap_member: int = 0
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    engine_version: str = ENGINE_VERSION
    rng_contract: str = RNG_CONTRACT_VERSION
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.red.fighter_id != self.bout.red_fighter_id:
            raise ValueError("red snapshot does not match bout")
        if self.blue.fighter_id != self.bout.blue_fighter_id:
            raise ValueError("blue snapshot does not match bout")
        if self.bootstrap_member < 0:
            raise ValueError("bootstrap_member must be nonnegative")
        if not str(self.root_seed):
            raise ValueError("root_seed is required")
        if not self.parameter_artifact_id.strip():
            raise ValueError("parameter_artifact_id is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "rng_contract": self.rng_contract,
            "root_seed": str(self.root_seed),
            "parameter_artifact_id": self.parameter_artifact_id,
            "bootstrap_member": self.bootstrap_member,
            "bout": self.bout.to_dict(),
            "red": self.red.to_dict(),
            "blue": self.blue.to_dict(),
            "simulator": self.simulator.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SimulationRunSpec":
        return cls(
            bout=BoutConfig.from_dict(value["bout"]),
            red=FighterSnapshot.from_dict(value["red"]),
            blue=FighterSnapshot.from_dict(value["blue"]),
            root_seed=value["root_seed"],
            parameter_artifact_id=value["parameter_artifact_id"],
            bootstrap_member=int(value.get("bootstrap_member", 0)),
            simulator=SimulatorConfig(**value.get("simulator", {})),
            engine_version=value.get("engine_version", ENGINE_VERSION),
            rng_contract=value.get("rng_contract", RNG_CONTRACT_VERSION),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True)
class FighterStats:
    strike_attempts: int = 0
    strikes_landed: int = 0
    significant_strike_attempts: int = 0
    significant_strikes_landed: int = 0
    head_landed: int = 0
    body_landed: int = 0
    leg_landed: int = 0
    distance_attempts: int = 0
    distance_landed: int = 0
    clinch_attempts: int = 0
    clinch_landed: int = 0
    ground_attempts: int = 0
    ground_landed: int = 0
    knockdowns: int = 0
    takedown_attempts: int = 0
    takedowns_landed: int = 0
    submission_attempts: int = 0
    reversals: int = 0
    control_time_us: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.strikes_landed > self.strike_attempts:
            raise ValueError("strikes landed cannot exceed attempts")
        if self.significant_strikes_landed > self.significant_strike_attempts:
            raise ValueError("significant strikes landed cannot exceed attempts")
        if self.significant_strike_attempts > self.strike_attempts:
            raise ValueError("significant attempts cannot exceed all attempts")
        if self.takedowns_landed > self.takedown_attempts:
            raise ValueError("takedowns landed cannot exceed attempts")
        if self.head_landed + self.body_landed + self.leg_landed != self.significant_strikes_landed:
            raise ValueError("target strike partitions must equal significant strikes landed")
        position_attempts = (
            self.distance_attempts + self.clinch_attempts + self.ground_attempts
        )
        # A zero partition remains readable for legacy v1 traces written before
        # attempt-by-phase telemetry was added. Newly generated strike deltas
        # always populate the complete partition.
        if position_attempts and position_attempts != self.significant_strike_attempts:
            raise ValueError("position strike attempt partitions must equal significant strike attempts")
        if self.distance_landed + self.clinch_landed + self.ground_landed != self.significant_strikes_landed:
            raise ValueError("position strike partitions must equal significant strikes landed")
        if position_attempts and (
            self.distance_landed > self.distance_attempts
            or self.clinch_landed > self.clinch_attempts
            or self.ground_landed > self.ground_attempts
        ):
            raise ValueError("position strikes landed cannot exceed attempts")

    def add(self, other: "FighterStats") -> "FighterStats":
        return FighterStats(**{
            name: int(getattr(self, name)) + int(getattr(other, name))
            for name in self.__dataclass_fields__
        })

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class JudgeRoundScore:
    round_number: int
    red_points: tuple[int, int, int]
    blue_points: tuple[int, int, int]
    red_effectiveness: float
    blue_effectiveness: float

    def __post_init__(self) -> None:
        if self.round_number < 1:
            raise ValueError("round_number must be positive")
        if len(self.red_points) != 3 or len(self.blue_points) != 3:
            raise ValueError("exactly three judge views are required")
        for points in (*self.red_points, *self.blue_points):
            if points not in (8, 9, 10):
                raise ValueError("round points must be 8, 9, or 10")


@dataclass(frozen=True, slots=True)
class FightResult:
    winner: Side | None
    method: OutcomeMethod
    round_number: int
    fight_time_us: int
    round_time_us: int
    reason: str
    decision_type: DecisionType | None = None

    def __post_init__(self) -> None:
        if self.round_number < 1 or self.fight_time_us < 0 or self.round_time_us < 0:
            raise ValueError("result clocks and round must be nonnegative")
        if self.method in (OutcomeMethod.DRAW, OutcomeMethod.NO_CONTEST) and self.winner is not None:
            raise ValueError("draw/no-contest cannot have a winner")
        if self.method is OutcomeMethod.DECISION and self.winner is None:
            raise ValueError("a winning decision requires a winner")
        if self.method is OutcomeMethod.DECISION and self.decision_type is None:
            raise ValueError("decision results require decision_type")

    @property
    def outcome_key(self) -> str:
        if self.method in (OutcomeMethod.DRAW, OutcomeMethod.NO_CONTEST):
            return self.method.value
        if self.winner is None:
            return self.method.value
        return f"{self.winner.value}_{self.method.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner.value if self.winner else None,
            "method": self.method.value,
            "round_number": self.round_number,
            "fight_time_us": self.fight_time_us,
            "round_time_us": self.round_time_us,
            "reason": self.reason,
            "decision_type": self.decision_type.value if self.decision_type else None,
            "outcome_key": self.outcome_key,
        }


@dataclass(frozen=True, slots=True)
class FightState:
    matchup_id: str
    scheduled_rounds: int
    round_number: int = 1
    fight_time_us: int = 0
    round_time_us: int = 0
    phase: Phase = Phase.DISTANCE
    top_position: Side | None = None
    red_stamina: float = 1.0
    blue_stamina: float = 1.0
    red_hurt: float = 0.0
    blue_hurt: float = 0.0
    red_damage: float = 0.0
    blue_damage: float = 0.0
    red_stats: FighterStats = field(default_factory=FighterStats)
    blue_stats: FighterStats = field(default_factory=FighterStats)
    red_round_effectiveness: float = 0.0
    blue_round_effectiveness: float = 0.0
    round_scores: tuple[JudgeRoundScore, ...] = ()
    result: FightResult | None = None
    event_count: int = 0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.round_number < 1 or self.round_number > self.scheduled_rounds:
            raise ValueError("state round is outside scheduled rounds")
        if self.fight_time_us < 0 or self.round_time_us < 0:
            raise ValueError("state clocks must be nonnegative")
        for name in (
            "red_stamina",
            "blue_stamina",
            "red_hurt",
            "blue_hurt",
            "red_damage",
            "blue_damage",
        ):
            _bounded(name, float(getattr(self, name)), 0.0, 1.0)
        for name in ("red_round_effectiveness", "blue_round_effectiveness"):
            _nonnegative(name, float(getattr(self, name)))
        if self.phase is not Phase.GROUND and self.top_position is not None:
            raise ValueError("top_position is only valid on the ground")
        if self.event_count < 0:
            raise ValueError("event_count must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "matchup_id": self.matchup_id,
            "scheduled_rounds": self.scheduled_rounds,
            "round_number": self.round_number,
            "fight_time_us": self.fight_time_us,
            "round_time_us": self.round_time_us,
            "phase": self.phase.value,
            "top_position": self.top_position.value if self.top_position else None,
            "red_stamina": self.red_stamina,
            "blue_stamina": self.blue_stamina,
            "red_hurt": self.red_hurt,
            "blue_hurt": self.blue_hurt,
            "red_damage": self.red_damage,
            "blue_damage": self.blue_damage,
            "red_stats": self.red_stats.to_dict(),
            "blue_stats": self.blue_stats.to_dict(),
            "red_round_effectiveness": self.red_round_effectiveness,
            "blue_round_effectiveness": self.blue_round_effectiveness,
            "round_scores": [
                {
                    "round_number": score.round_number,
                    "red_points": list(score.red_points),
                    "blue_points": list(score.blue_points),
                    "red_effectiveness": score.red_effectiveness,
                    "blue_effectiveness": score.blue_effectiveness,
                }
                for score in self.round_scores
            ],
            "result": self.result.to_dict() if self.result else None,
            "event_count": self.event_count,
        }


@dataclass(frozen=True, slots=True)
class StateDelta:
    fight_time_us: int | None = None
    round_time_us: int | None = None
    round_number: int | None = None
    phase: Phase | None = None
    clear_top_position: bool = False
    top_position: Side | None = None
    red_stamina: float | None = None
    blue_stamina: float | None = None
    red_hurt: float | None = None
    blue_hurt: float | None = None
    red_damage: float | None = None
    blue_damage: float | None = None
    red_stats_delta: FighterStats | None = None
    blue_stats_delta: FighterStats | None = None
    red_effectiveness_delta: float = 0.0
    blue_effectiveness_delta: float = 0.0
    reset_round_effectiveness: bool = False
    append_round_score: JudgeRoundScore | None = None
    result: FightResult | None = None


@dataclass(frozen=True, slots=True)
class RngDraw:
    stream: str
    draw_index: int
    distribution: str
    parameters: tuple[tuple[str, str], ...]
    value: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "draw_index": self.draw_index,
            "distribution": self.distribution,
            "parameters": dict(self.parameters),
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    sequence: int
    event_type: EventType
    fight_time_us: int
    round_number: int
    actor: Side | None
    target: Side | None
    action: str
    phase_before: Phase
    phase_after: Phase
    delta: StateDelta
    rng_draws: tuple[RngDraw, ...]
    state_hash_before: str
    state_hash_after: str
    previous_event_hash: str
    event_hash: str
    payload: tuple[tuple[str, str], ...] = ()
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SimulationPath:
    matchup_id: str
    scheduled_rounds: int
    bootstrap_member: int
    simulation_index: int
    result: FightResult
    red_stats: FighterStats
    blue_stats: FighterStats
    final_state_hash: str
    phase_time_us: tuple[tuple[str, int], ...] = ()
    events: tuple[SimulationEvent, ...] = ()
    invariant_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        total = 0
        for phase, duration_us in self.phase_time_us:
            if phase in seen:
                raise ValueError("phase_time_us must contain each phase at most once")
            if phase not in {item.value for item in Phase}:
                raise ValueError(f"unknown phase in phase_time_us: {phase}")
            if int(duration_us) < 0:
                raise ValueError("phase durations must be nonnegative")
            seen.add(phase)
            total += int(duration_us)
        if self.phase_time_us and total != self.result.fight_time_us:
            raise ValueError("phase durations must sum to fight duration")

    @property
    def outcome_key(self) -> str:
        return self.result.outcome_key

    def phase_duration_us(self, phase: Phase | str) -> int:
        key = phase.value if isinstance(phase, Phase) else str(phase)
        return next((int(value) for name, value in self.phase_time_us if name == key), 0)


@dataclass(frozen=True, slots=True)
class OutcomeCount:
    outcome: str
    count: int


@dataclass(frozen=True, slots=True)
class BootstrapOutcomeCounts:
    bootstrap_member: int
    paths: int
    counts: tuple[OutcomeCount, ...]


@dataclass(frozen=True, slots=True)
class DurationBin:
    upper_seconds: int
    count: int


@dataclass(frozen=True, slots=True)
class MethodRoundCount:
    method: str
    round_number: int
    count: int


@dataclass(frozen=True, slots=True)
class TotalLineCount:
    half_rounds: float
    threshold_seconds: float
    over: int
    under: int
    push: int
    no_action: int


@dataclass(frozen=True, slots=True)
class StatisticSummary:
    statistic: str
    mean: float
    p05: float
    median: float
    p95: float


@dataclass(frozen=True, slots=True)
class StatisticValueCount:
    value: float
    count: int

    def __post_init__(self) -> None:
        _finite("statistic distribution value", float(self.value))
        if self.count < 0:
            raise ValueError("statistic distribution count must be nonnegative")


@dataclass(frozen=True, slots=True)
class StatisticDistribution:
    statistic: str
    total_paths: int
    counts: tuple[StatisticValueCount, ...]

    def __post_init__(self) -> None:
        if not self.statistic:
            raise ValueError("statistic distribution requires a name")
        if self.total_paths <= 0 or sum(item.count for item in self.counts) != self.total_paths:
            raise ValueError("statistic distribution counts must sum to total_paths")
        values = [item.value for item in self.counts]
        if values != sorted(values) or len(values) != len(set(values)):
            raise ValueError("statistic distribution values must be unique and sorted")


@dataclass(frozen=True, slots=True)
class BootstrapStatisticDistribution:
    bootstrap_member: int
    statistic: str
    paths: int
    counts: tuple[StatisticValueCount, ...]

    def __post_init__(self) -> None:
        if self.bootstrap_member < 0 or not self.statistic:
            raise ValueError("bootstrap statistic distribution identity is invalid")
        if self.paths <= 0 or sum(item.count for item in self.counts) != self.paths:
            raise ValueError("bootstrap statistic counts must sum to member paths")
        values = [item.value for item in self.counts]
        if values != sorted(values) or len(values) != len(set(values)):
            raise ValueError(
                "bootstrap statistic values must be unique and sorted"
            )


@dataclass(frozen=True, slots=True)
class StatisticUncertainty:
    statistic: str
    estimate_mean: float
    process_mcse_mean: float
    parameter_model_p025: float
    parameter_model_median: float
    parameter_model_p975: float
    conditional_means: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not self.statistic:
            raise ValueError("statistic uncertainty requires a name")
        for name in (
            "estimate_mean",
            "process_mcse_mean",
            "parameter_model_p025",
            "parameter_model_median",
            "parameter_model_p975",
        ):
            _finite(name, float(getattr(self, name)))
        if self.process_mcse_mean < 0:
            raise ValueError("statistic process MCSE cannot be negative")
        if not (
            self.parameter_model_p025
            <= self.parameter_model_median
            <= self.parameter_model_p975
        ):
            raise ValueError("statistic parameter/model quantiles must be ordered")
        members = [member for member, _ in self.conditional_means]
        if (
            not members
            or members != sorted(members)
            or len(members) != len(set(members))
            or any(member < 0 for member in members)
        ):
            raise ValueError(
                "statistic conditional means require unique sorted members"
            )
        for _, mean in self.conditional_means:
            _finite("conditional statistic mean", float(mean))


@dataclass(frozen=True, slots=True)
class MetricUncertainty:
    metric: str
    estimate: float
    process_mcse: float
    parameter_p025: float
    parameter_median: float
    parameter_p975: float
    conditional_probabilities: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class ProbabilityPoint:
    seconds: float
    probability: float


@dataclass(frozen=True, slots=True)
class AggregateForecast:
    matchup_id: str
    scheduled_rounds: int
    total_paths: int
    bootstrap_members: int
    outcome_counts: tuple[OutcomeCount, ...]
    bootstrap_outcome_counts: tuple[BootstrapOutcomeCounts, ...]
    duration_bins: tuple[DurationBin, ...]
    method_round_counts: tuple[MethodRoundCount, ...]
    decision_type_counts: tuple[OutcomeCount, ...]
    total_lines: tuple[TotalLineCount, ...]
    statistic_summaries: tuple[StatisticSummary, ...]
    uncertainty: tuple[MetricUncertainty, ...]
    survival: tuple[ProbabilityPoint, ...]
    statistic_distributions: tuple[StatisticDistribution, ...] = ()
    bootstrap_statistic_distributions: tuple[BootstrapStatisticDistribution, ...] = ()
    statistic_uncertainty: tuple[StatisticUncertainty, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.total_paths <= 0:
            raise ValueError("aggregate forecast requires paths")
        if sum(item.count for item in self.outcome_counts) != self.total_paths:
            raise ValueError("outcome counts must sum to total_paths")
        if sum(item.count for item in self.duration_bins) != self.total_paths:
            raise ValueError("duration bins must sum to total_paths")
        if any(item.total_paths != self.total_paths for item in self.statistic_distributions):
            raise ValueError("statistic distributions must use aggregate total_paths")
        if len({item.statistic for item in self.statistic_distributions}) != len(
            self.statistic_distributions
        ):
            raise ValueError("aggregate statistic distribution names must be unique")
        values = [point.probability for point in self.survival]
        if any(a < b for a, b in zip(values, values[1:])):
            raise ValueError("survival probabilities must be nonincreasing")

    @property
    def outcome_probabilities(self) -> dict[str, float]:
        return {item.outcome: item.count / self.total_paths for item in self.outcome_counts}

    def probability(self, outcome: str) -> float:
        return self.outcome_probabilities.get(outcome, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "matchup_id": self.matchup_id,
            "scheduled_rounds": self.scheduled_rounds,
            "total_paths": self.total_paths,
            "bootstrap_members": self.bootstrap_members,
            "outcome_counts": {item.outcome: item.count for item in self.outcome_counts},
            "outcome_probabilities": self.outcome_probabilities,
            "bootstrap_outcome_counts": [
                {
                    "bootstrap_member": item.bootstrap_member,
                    "paths": item.paths,
                    "counts": {count.outcome: count.count for count in item.counts},
                }
                for item in self.bootstrap_outcome_counts
            ],
            "duration_bins": [vars_without_slots(item) for item in self.duration_bins],
            "method_round_counts": [vars_without_slots(item) for item in self.method_round_counts],
            "decision_type_counts": {
                item.outcome: item.count for item in self.decision_type_counts
            },
            "total_lines": [vars_without_slots(item) for item in self.total_lines],
            "statistic_summaries": [vars_without_slots(item) for item in self.statistic_summaries],
            "statistic_distributions": [
                {
                    "statistic": item.statistic,
                    "total_paths": item.total_paths,
                    "counts": [vars_without_slots(count) for count in item.counts],
                }
                for item in self.statistic_distributions
            ],
            "bootstrap_statistic_distributions": [
                {
                    "bootstrap_member": item.bootstrap_member,
                    "statistic": item.statistic,
                    "paths": item.paths,
                    "counts": [vars_without_slots(count) for count in item.counts],
                }
                for item in self.bootstrap_statistic_distributions
            ],
            "statistic_uncertainty": [
                {
                    **{
                        key: value
                        for key, value in vars_without_slots(item).items()
                        if key != "conditional_means"
                    },
                    "conditional_means": dict(item.conditional_means),
                }
                for item in self.statistic_uncertainty
            ],
            "uncertainty": [
                {
                    **{key: value for key, value in vars_without_slots(item).items() if key != "conditional_probabilities"},
                    "conditional_probabilities": dict(item.conditional_probabilities),
                }
                for item in self.uncertainty
            ],
            "survival": [vars_without_slots(item) for item in self.survival],
        }


@dataclass(frozen=True, slots=True)
class TraceManifest:
    run_id: str
    matchup_id: str
    root_seed: str
    engine_version: str
    rng_contract: str
    parameter_artifact_id: str
    selected_simulation_indices: tuple[int, ...]
    selected_bootstrap_members: tuple[int, ...]
    trace_hashes: tuple[tuple[int, int, str], ...]
    selection_algorithm: str = "lowest_sha256_per_outcome_round_v1"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.selected_simulation_indices) != len(self.selected_bootstrap_members):
            raise ValueError("selected simulation indices and bootstrap members must align")
        if len(self.trace_hashes) != len(self.selected_simulation_indices):
            raise ValueError("trace hashes must align with selected paths")


def vars_without_slots(value: Any) -> dict[str, Any]:
    """Return dataclass slot values without relying on ``__dict__``."""

    return {name: getattr(value, name) for name in value.__dataclass_fields__}


def outcome_count_map(items: Iterable[OutcomeCount]) -> dict[str, int]:
    return {item.outcome: item.count for item in items}
