"""Pure data loading and trace projection for the local simulation GUI.

This module deliberately has no Qt or matplotlib dependency.  Keeping the
authoritative run parsing here makes the desktop views straightforward to test
in headless CI and useful to future non-GUI analysis tools.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


class RunBundleError(ValueError):
    """Raised when a directory is not a readable completed simulation run."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RunBundleError(f"required run file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunBundleError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise RunBundleError(f"expected a JSON object in {path}")
    return value


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def pretty_metric(name: str) -> str:
    return name.replace("_", " ").strip().title()


@dataclass(frozen=True, slots=True)
class DistributionSeries:
    key: str
    label: str
    values: tuple[float, ...]
    counts: tuple[int, ...]
    total: int
    observed: float | None = None
    unit: str = "count"
    validation: Mapping[str, Any] | None = None

    @property
    def probabilities(self) -> tuple[float, ...]:
        denominator = max(1, self.total)
        return tuple(count / denominator for count in self.counts)

    @property
    def mean(self) -> float:
        denominator = sum(self.counts)
        if not denominator:
            return 0.0
        return sum(value * count for value, count in zip(self.values, self.counts)) / denominator


@dataclass(frozen=True, slots=True)
class RunBundle:
    directory: Path
    aggregate: Mapping[str, Any]
    specs: tuple[Mapping[str, Any], ...]
    convergence: Mapping[str, Any]
    validation: Mapping[str, Any] | None
    trace_paths: tuple[Path, ...]
    red_name: str
    blue_name: str

    @property
    def matchup_id(self) -> str:
        return str(self.aggregate.get("matchup_id", self.directory.name))

    @property
    def total_paths(self) -> int:
        return int(self.aggregate.get("total_paths", 0))

    @property
    def distributions(self) -> tuple[DistributionSeries, ...]:
        validation_by_stat = {
            str(item.get("statistic")): item
            for item in (self.validation or {}).get("statistics", [])
            if isinstance(item, Mapping) and item.get("statistic")
        }
        result: list[DistributionSeries] = []
        raw = self.aggregate.get("statistic_distributions", [])
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                key = str(item.get("statistic", ""))
                points = item.get("counts", [])
                if not key or not isinstance(points, list):
                    continue
                parsed = sorted(
                    (
                        (_finite(point.get("value")), int(point.get("count", 0)))
                        for point in points
                        if isinstance(point, Mapping)
                    ),
                    key=lambda pair: pair[0],
                )
                validation = validation_by_stat.get(key)
                observed = _finite(validation.get("observed")) if validation else None
                result.append(
                    DistributionSeries(
                        key=key,
                        label=str((validation or {}).get("label") or pretty_metric(key)),
                        values=tuple(value for value, _ in parsed),
                        counts=tuple(count for _, count in parsed),
                        total=int(item.get("total_paths", self.total_paths)),
                        observed=observed,
                        unit=str(
                            (validation or {}).get(
                                "unit",
                                "seconds" if key.endswith("_seconds") else "count",
                            )
                        ),
                        validation=validation,
                    )
                )
        duration_validation = validation_by_stat.get("duration_seconds")
        duration_points = self.aggregate.get("duration_bins", [])
        if isinstance(duration_points, list) and duration_points:
            parsed_duration = [
                (_finite(item.get("upper_seconds")), int(item.get("count", 0)))
                for item in duration_points
                if isinstance(item, Mapping)
            ]
            result.append(
                DistributionSeries(
                    key="duration_seconds",
                    label="Fight Duration",
                    values=tuple(value for value, _ in parsed_duration),
                    counts=tuple(count for _, count in parsed_duration),
                    total=self.total_paths,
                    observed=(
                        _finite(duration_validation.get("observed"))
                        if duration_validation
                        else None
                    ),
                    unit="seconds",
                    validation=duration_validation,
                )
            )
        return tuple(result)

    def distribution(self, key: str) -> DistributionSeries:
        for series in self.distributions:
            if series.key == key:
                return series
        raise KeyError(key)


def load_run_bundle(path: str | Path) -> RunBundle:
    directory = Path(path).expanduser().resolve()
    if directory.is_file() and directory.name == "aggregate.json":
        directory = directory.parent
    if not directory.is_dir():
        raise RunBundleError(f"simulation run directory does not exist: {directory}")
    aggregate_wrapper = _load_json(directory / "aggregate.json")
    aggregate = aggregate_wrapper.get("aggregate", aggregate_wrapper)
    if not isinstance(aggregate, Mapping) or "total_paths" not in aggregate:
        raise RunBundleError(f"{directory / 'aggregate.json'} is not a simulation aggregate")
    spec_wrapper = _load_json(directory / "specs.json")
    specs_raw = spec_wrapper.get("specs", [])
    specs = tuple(item for item in specs_raw if isinstance(item, Mapping)) if isinstance(specs_raw, list) else ()
    first = specs[0] if specs else {}
    red = first.get("red", {}) if isinstance(first.get("red"), Mapping) else {}
    blue = first.get("blue", {}) if isinstance(first.get("blue"), Mapping) else {}
    convergence_path = directory / "convergence.json"
    convergence = _load_json(convergence_path) if convergence_path.is_file() else {}
    validation_path = directory / "validation.json"
    validation = _load_json(validation_path) if validation_path.is_file() else None
    trace_dir = directory / "traces"
    trace_paths = tuple(sorted(trace_dir.glob("*.json"))) if trace_dir.is_dir() else ()
    return RunBundle(
        directory=directory,
        aggregate=dict(aggregate),
        specs=specs,
        convergence=convergence,
        validation=validation,
        trace_paths=trace_paths,
        red_name=str(red.get("fighter_name") or red.get("fighter_id") or "Red fighter"),
        blue_name=str(blue.get("fighter_name") or blue.get("fighter_id") or "Blue fighter"),
    )


@dataclass(frozen=True, slots=True)
class TraceTimeline:
    simulation_index: int
    bootstrap_member: int
    result: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    seconds: tuple[float, ...]
    phases: tuple[str, ...]
    positions: tuple[str, ...]
    red_stamina: tuple[float, ...]
    blue_stamina: tuple[float, ...]
    red_hurt: tuple[float, ...]
    blue_hurt: tuple[float, ...]
    red_damage: tuple[float, ...]
    blue_damage: tuple[float, ...]
    red_stats: Mapping[str, tuple[float, ...]]
    blue_stats: Mapping[str, tuple[float, ...]]


_TRACE_STATS = (
    "significant_strike_attempts",
    "significant_strikes_landed",
    "distance_attempts",
    "distance_landed",
    "clinch_attempts",
    "clinch_landed",
    "ground_attempts",
    "ground_landed",
    "knockdowns",
    "takedown_attempts",
    "takedowns_landed",
    "submission_attempts",
    "reversals",
    "control_time_us",
)


def load_trace_timeline(path: str | Path) -> TraceTimeline:
    payload = _load_json(Path(path))
    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list):
        raise RunBundleError("trace events must be a list")
    events = tuple(item for item in raw_events if isinstance(item, Mapping))
    seconds: list[float] = []
    phases: list[str] = []
    positions: list[str] = []
    top_position: str | None = None
    dynamics = {
        "red_stamina": 1.0,
        "blue_stamina": 1.0,
        "red_hurt": 0.0,
        "blue_hurt": 0.0,
        "red_damage": 0.0,
        "blue_damage": 0.0,
    }
    dynamic_history = {key: [] for key in dynamics}
    cumulative = {
        "red": {key: 0.0 for key in _TRACE_STATS},
        "blue": {key: 0.0 for key in _TRACE_STATS},
    }
    statistic_history = {
        "red": {key: [] for key in _TRACE_STATS},
        "blue": {key: [] for key in _TRACE_STATS},
    }
    for event in events:
        delta = event.get("delta", {})
        if not isinstance(delta, Mapping):
            delta = {}
        seconds.append(_finite(event.get("fight_time_us")) / 1_000_000.0)
        phase = str(event.get("phase_after") or event.get("phase_before") or "distance")
        phases.append(phase)
        if bool(delta.get("clear_top_position")) or phase != "ground":
            top_position = None
        if delta.get("top_position") is not None:
            top_position = str(delta["top_position"])
        positions.append(
            f"{top_position}_top" if phase == "ground" and top_position else phase
        )
        for key in dynamics:
            if delta.get(key) is not None:
                dynamics[key] = _finite(delta[key], dynamics[key])
            dynamic_history[key].append(dynamics[key])
        for side in ("red", "blue"):
            stats_delta = delta.get(f"{side}_stats_delta")
            if isinstance(stats_delta, Mapping):
                for key in _TRACE_STATS:
                    cumulative[side][key] += _finite(stats_delta.get(key))
            for key in _TRACE_STATS:
                statistic_history[side][key].append(cumulative[side][key])
    result = payload.get("result", {})
    return TraceTimeline(
        simulation_index=int(payload.get("simulation_index", 0)),
        bootstrap_member=int(payload.get("bootstrap_member", 0)),
        result=dict(result) if isinstance(result, Mapping) else {},
        events=events,
        seconds=tuple(seconds),
        phases=tuple(phases),
        positions=tuple(positions),
        red_stamina=tuple(dynamic_history["red_stamina"]),
        blue_stamina=tuple(dynamic_history["blue_stamina"]),
        red_hurt=tuple(dynamic_history["red_hurt"]),
        blue_hurt=tuple(dynamic_history["blue_hurt"]),
        red_damage=tuple(dynamic_history["red_damage"]),
        blue_damage=tuple(dynamic_history["blue_damage"]),
        red_stats={key: tuple(values) for key, values in statistic_history["red"].items()},
        blue_stats={key: tuple(values) for key, values in statistic_history["blue"].items()},
    )
