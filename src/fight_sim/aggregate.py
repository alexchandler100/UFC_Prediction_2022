"""Deterministic nested Monte Carlo reduction and uncertainty decomposition."""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
import math
from typing import Callable, Iterable

import numpy as np

from .domain import (
    AggregateForecast,
    BootstrapOutcomeCounts,
    BootstrapStatisticDistribution,
    DurationBin,
    MethodRoundCount,
    MetricUncertainty,
    OutcomeCount,
    OutcomeMethod,
    ProbabilityPoint,
    SimulationPath,
    StatisticSummary,
    StatisticDistribution,
    StatisticUncertainty,
    StatisticValueCount,
    TotalLineCount,
)
from .markets import Settlement, settle_total, valid_total_round_lines


def _ordered_counts(counter: Counter[str]) -> tuple[OutcomeCount, ...]:
    return tuple(OutcomeCount(key, int(counter[key])) for key in sorted(counter))


def _quantiles(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(tuple(values), dtype=float)
    if not len(array):
        return 0.0, 0.0, 0.0
    result = np.quantile(array, [0.025, 0.5, 0.975])
    return tuple(float(value) for value in result)  # type: ignore[return-value]


def _stat_summary(name: str, values: Iterable[float]) -> StatisticSummary:
    array = np.asarray(tuple(values), dtype=float)
    quantiles = np.quantile(array, [0.05, 0.5, 0.95])
    return StatisticSummary(
        statistic=name,
        mean=float(math.fsum(float(value) for value in array) / len(array)),
        p05=float(quantiles[0]),
        median=float(quantiles[1]),
        p95=float(quantiles[2]),
    )


STATISTIC_EXTRACTORS: tuple[tuple[str, Callable[[SimulationPath], float]], ...] = (
    ("red_significant_strikes", lambda path: float(path.red_stats.significant_strikes_landed)),
    ("blue_significant_strikes", lambda path: float(path.blue_stats.significant_strikes_landed)),
    ("red_significant_strike_attempts", lambda path: float(path.red_stats.significant_strike_attempts)),
    ("blue_significant_strike_attempts", lambda path: float(path.blue_stats.significant_strike_attempts)),
    ("red_head_strikes_landed", lambda path: float(path.red_stats.head_landed)),
    ("blue_head_strikes_landed", lambda path: float(path.blue_stats.head_landed)),
    ("red_body_strikes_landed", lambda path: float(path.red_stats.body_landed)),
    ("blue_body_strikes_landed", lambda path: float(path.blue_stats.body_landed)),
    ("red_leg_strikes_landed", lambda path: float(path.red_stats.leg_landed)),
    ("blue_leg_strikes_landed", lambda path: float(path.blue_stats.leg_landed)),
    ("red_distance_strikes_landed", lambda path: float(path.red_stats.distance_landed)),
    ("blue_distance_strikes_landed", lambda path: float(path.blue_stats.distance_landed)),
    ("red_distance_strike_attempts", lambda path: float(path.red_stats.distance_attempts)),
    ("blue_distance_strike_attempts", lambda path: float(path.blue_stats.distance_attempts)),
    ("red_clinch_strikes_landed", lambda path: float(path.red_stats.clinch_landed)),
    ("blue_clinch_strikes_landed", lambda path: float(path.blue_stats.clinch_landed)),
    ("red_clinch_strike_attempts", lambda path: float(path.red_stats.clinch_attempts)),
    ("blue_clinch_strike_attempts", lambda path: float(path.blue_stats.clinch_attempts)),
    ("red_ground_strikes_landed", lambda path: float(path.red_stats.ground_landed)),
    ("blue_ground_strikes_landed", lambda path: float(path.blue_stats.ground_landed)),
    ("red_ground_strike_attempts", lambda path: float(path.red_stats.ground_attempts)),
    ("blue_ground_strike_attempts", lambda path: float(path.blue_stats.ground_attempts)),
    ("red_knockdowns", lambda path: float(path.red_stats.knockdowns)),
    ("blue_knockdowns", lambda path: float(path.blue_stats.knockdowns)),
    ("red_takedowns", lambda path: float(path.red_stats.takedowns_landed)),
    ("blue_takedowns", lambda path: float(path.blue_stats.takedowns_landed)),
    ("red_takedown_attempts", lambda path: float(path.red_stats.takedown_attempts)),
    ("blue_takedown_attempts", lambda path: float(path.blue_stats.takedown_attempts)),
    ("red_submission_attempts", lambda path: float(path.red_stats.submission_attempts)),
    ("blue_submission_attempts", lambda path: float(path.blue_stats.submission_attempts)),
    # UFCStats exposes control at whole-second resolution.  Rounding simulated
    # microseconds here keeps the exact published distribution bounded and
    # directly comparable to its observed target.
    ("red_control_seconds", lambda path: float(round(path.red_stats.control_time_us / 1_000_000.0))),
    ("blue_control_seconds", lambda path: float(round(path.blue_stats.control_time_us / 1_000_000.0))),
    ("distance_time_seconds", lambda path: float(round(path.phase_duration_us("distance") / 1_000_000.0))),
    ("clinch_time_seconds", lambda path: float(round(path.phase_duration_us("clinch") / 1_000_000.0))),
    ("ground_time_seconds", lambda path: float(round(path.phase_duration_us("ground") / 1_000_000.0))),
    ("scramble_time_seconds", lambda path: float(round(path.phase_duration_us("scramble") / 1_000_000.0))),
)


def _statistic_value_counts(values: Iterable[float]) -> tuple[StatisticValueCount, ...]:
    counts = Counter(float(value) for value in values)
    return tuple(
        StatisticValueCount(value=value, count=int(counts[value]))
        for value in sorted(counts)
    )


def _statistic_uncertainty(
    name: str,
    groups: dict[int, tuple[SimulationPath, ...]],
    extractor: Callable[[SimulationPath], float],
) -> StatisticUncertainty:
    conditional: list[tuple[int, float]] = []
    total_sum = 0.0
    total_paths = 0
    process_variance_numerator = 0.0
    for member in sorted(groups):
        member_values = np.asarray([extractor(path) for path in groups[member]], dtype=float)
        member_mean = float(np.mean(member_values))
        conditional.append((member, member_mean))
        total_sum += math.fsum(float(value) for value in member_values)
        total_paths += len(member_values)
        process_variance_numerator += len(member_values) * float(np.var(member_values, ddof=0))
    p025, median, p975 = _quantiles(value for _, value in conditional)
    return StatisticUncertainty(
        statistic=name,
        estimate_mean=total_sum / total_paths,
        process_mcse_mean=math.sqrt(process_variance_numerator) / total_paths,
        parameter_model_p025=p025,
        parameter_model_median=median,
        parameter_model_p975=p975,
        conditional_means=tuple(conditional),
    )


def _metric_uncertainty(
    name: str,
    groups: dict[int, tuple[SimulationPath, ...]],
    indicator: Callable[[SimulationPath], bool],
) -> MetricUncertainty:
    conditional: list[tuple[int, float]] = []
    successes = 0
    total = 0
    process_variance_numerator = 0.0
    for member in sorted(groups):
        paths = groups[member]
        count = sum(int(indicator(path)) for path in paths)
        probability = count / len(paths)
        conditional.append((member, probability))
        successes += count
        total += len(paths)
        process_variance_numerator += len(paths) * probability * (1.0 - probability)
    estimate = successes / total
    p025, median, p975 = _quantiles(value for _, value in conditional)
    return MetricUncertainty(
        metric=name,
        estimate=estimate,
        process_mcse=math.sqrt(process_variance_numerator) / total,
        parameter_p025=p025,
        parameter_median=median,
        parameter_p975=p975,
        conditional_probabilities=tuple(conditional),
    )


def _expanded_counter(counter: Counter[float]) -> np.ndarray:
    if not counter:
        return np.asarray([], dtype=float)
    support = np.asarray(sorted(counter), dtype=float)
    counts = np.asarray([counter[value] for value in support], dtype=np.int64)
    return np.repeat(support, counts)


def _summary_from_values(name: str, values: np.ndarray) -> StatisticSummary:
    if not len(values):
        raise ValueError(f"cannot summarize empty statistic: {name}")
    quantiles = np.quantile(values, [0.05, 0.5, 0.95])
    return StatisticSummary(
        statistic=name,
        mean=float(math.fsum(float(value) for value in values) / len(values)),
        p05=float(quantiles[0]),
        median=float(quantiles[1]),
        p95=float(quantiles[2]),
    )


def _summary_from_counter(name: str, counter: Counter[float]) -> StatisticSummary:
    return _summary_from_values(name, _expanded_counter(counter))


def _value_counts_from_counter(
    counter: Counter[float],
) -> tuple[StatisticValueCount, ...]:
    return tuple(
        StatisticValueCount(value=float(value), count=int(counter[value]))
        for value in sorted(counter)
    )


class ForecastAccumulator:
    """Mergeable exact sufficient statistics for an ``AggregateForecast``.

    The only path-sized buffer is a packed signed-64-bit duration array (eight
    bytes per path), needed to preserve exact duration quantiles.  Everything
    else is represented by integer counts, including member-level statistic
    distributions and convergence split counts.
    """

    def __init__(
        self,
        scheduled_rounds: int,
        *,
        duration_bin_seconds: int = 5,
        survival_step_seconds: int = 30,
    ) -> None:
        if scheduled_rounds < 1 or scheduled_rounds > 5:
            raise ValueError("scheduled_rounds must be between one and five")
        if duration_bin_seconds <= 0 or survival_step_seconds <= 0:
            raise ValueError("duration and survival steps must be positive")
        self.scheduled_rounds = int(scheduled_rounds)
        self.duration_bin_seconds = int(duration_bin_seconds)
        self.survival_step_seconds = int(survival_step_seconds)
        self.matchup_id: str | None = None
        self.total_paths = 0
        self.member_paths: Counter[int] = Counter()
        self.outcomes: Counter[str] = Counter()
        self.member_outcomes: dict[int, Counter[str]] = {}
        self.duration_bins: Counter[int] = Counter()
        self.duration_values_us = array("q")
        self.method_rounds: Counter[tuple[str, int]] = Counter()
        self.decision_types: Counter[str] = Counter()
        self.total_settlements: dict[float, Counter[str]] = {
            line: Counter() for line in valid_total_round_lines(scheduled_rounds)
        }
        self.statistics: dict[str, Counter[float]] = {
            name: Counter() for name, _ in STATISTIC_EXTRACTORS
        }
        self.member_statistics: dict[tuple[int, str], Counter[float]] = {}
        self.split_paths: Counter[tuple[int, int]] = Counter()
        self.split_red_wins: Counter[tuple[int, int]] = Counter()

    def add_path(self, path: SimulationPath) -> None:
        if path.scheduled_rounds != self.scheduled_rounds:
            raise ValueError("path scheduled rounds disagree with accumulator")
        if self.matchup_id is None:
            self.matchup_id = path.matchup_id
        elif path.matchup_id != self.matchup_id:
            raise ValueError("all paths must belong to one matchup")
        member = int(path.bootstrap_member)
        self.total_paths += 1
        self.member_paths[member] += 1
        self.outcomes[path.outcome_key] += 1
        self.member_outcomes.setdefault(member, Counter())[path.outcome_key] += 1

        fight_time_us = int(path.result.fight_time_us)
        self.duration_values_us.append(fight_time_us)
        seconds = fight_time_us / 1_000_000.0
        upper = int(
            math.ceil(seconds / self.duration_bin_seconds)
            * self.duration_bin_seconds
        )
        self.duration_bins[upper] += 1
        self.method_rounds[
            (path.result.method.value, path.result.round_number)
        ] += 1
        if path.result.decision_type is not None:
            self.decision_types[path.result.decision_type.value] += 1
        for line in self.total_settlements:
            self.total_settlements[line][settle_total(path.result, line).value] += 1

        for name, extractor in STATISTIC_EXTRACTORS:
            value = float(extractor(path))
            self.statistics[name][value] += 1
            self.member_statistics.setdefault((member, name), Counter())[value] += 1

        parity = int(path.simulation_index) % 2
        split_key = (member, parity)
        self.split_paths[split_key] += 1
        if path.result.winner is not None and path.result.winner.value == "red":
            self.split_red_wins[split_key] += 1

    @staticmethod
    def _merge_counter_maps(
        target: dict[object, Counter], source: dict[object, Counter]
    ) -> None:
        for key, counter in source.items():
            target.setdefault(key, Counter()).update(counter)

    def merge(self, other: "ForecastAccumulator") -> "ForecastAccumulator":
        if (
            self.scheduled_rounds != other.scheduled_rounds
            or self.duration_bin_seconds != other.duration_bin_seconds
            or self.survival_step_seconds != other.survival_step_seconds
        ):
            raise ValueError("cannot merge incompatible forecast accumulators")
        if other.matchup_id is not None:
            if self.matchup_id is None:
                self.matchup_id = other.matchup_id
            elif self.matchup_id != other.matchup_id:
                raise ValueError("cannot merge different matchups")
        self.total_paths += other.total_paths
        self.member_paths.update(other.member_paths)
        self.outcomes.update(other.outcomes)
        self._merge_counter_maps(self.member_outcomes, other.member_outcomes)
        self.duration_bins.update(other.duration_bins)
        self.duration_values_us.extend(other.duration_values_us)
        self.method_rounds.update(other.method_rounds)
        self.decision_types.update(other.decision_types)
        self._merge_counter_maps(self.total_settlements, other.total_settlements)
        self._merge_counter_maps(self.statistics, other.statistics)
        self._merge_counter_maps(self.member_statistics, other.member_statistics)
        self.split_paths.update(other.split_paths)
        self.split_red_wins.update(other.split_red_wins)
        return self

    def _statistic_uncertainty(self, name: str) -> StatisticUncertainty:
        conditional: list[tuple[int, float]] = []
        total_sum = 0.0
        process_variance_numerator = 0.0
        for member in sorted(self.member_paths):
            values = _expanded_counter(self.member_statistics[(member, name)])
            member_mean = float(np.mean(values))
            conditional.append((member, member_mean))
            total_sum += math.fsum(float(value) for value in values)
            process_variance_numerator += len(values) * float(
                np.var(values, ddof=0)
            )
        p025, median, p975 = _quantiles(value for _, value in conditional)
        return StatisticUncertainty(
            statistic=name,
            estimate_mean=total_sum / self.total_paths,
            process_mcse_mean=(
                math.sqrt(process_variance_numerator) / self.total_paths
            ),
            parameter_model_p025=p025,
            parameter_model_median=median,
            parameter_model_p975=p975,
            conditional_means=tuple(conditional),
        )

    @staticmethod
    def _metric_matches(metric: str, outcome: str) -> bool:
        if metric == "red_win":
            return outcome.startswith("red_")
        if metric == "blue_win":
            return outcome.startswith("blue_")
        if metric == "ko_tko":
            return outcome.endswith("_ko_tko") or outcome == "ko_tko"
        if metric == "submission":
            return outcome.endswith("_submission") or outcome == "submission"
        if metric == "decision":
            return outcome.endswith("_decision") or outcome == "decision"
        if metric == "goes_distance":
            return outcome == "draw" or outcome.endswith("_decision")
        raise ValueError(f"unsupported aggregate metric: {metric}")

    def _metric_uncertainty(self, metric: str) -> MetricUncertainty:
        conditional: list[tuple[int, float]] = []
        successes = 0
        process_variance_numerator = 0.0
        for member in sorted(self.member_paths):
            paths = self.member_paths[member]
            count = sum(
                value
                for outcome, value in self.member_outcomes[member].items()
                if self._metric_matches(metric, outcome)
            )
            probability = count / paths
            conditional.append((member, probability))
            successes += count
            process_variance_numerator += paths * probability * (1.0 - probability)
        p025, median, p975 = _quantiles(value for _, value in conditional)
        return MetricUncertainty(
            metric=metric,
            estimate=successes / self.total_paths,
            process_mcse=math.sqrt(process_variance_numerator) / self.total_paths,
            parameter_p025=p025,
            parameter_median=median,
            parameter_p975=p975,
            conditional_probabilities=tuple(conditional),
        )

    def forecast(self) -> AggregateForecast:
        if self.total_paths <= 0 or self.matchup_id is None:
            raise ValueError("cannot aggregate zero simulation paths")
        duration_us = np.frombuffer(self.duration_values_us, dtype=np.int64)
        duration_seconds = duration_us.astype(float) / 1_000_000.0
        summaries = tuple(
            _summary_from_counter(name, self.statistics[name])
            for name, _ in STATISTIC_EXTRACTORS
        ) + (_summary_from_values("duration_seconds", duration_seconds),)
        statistic_distributions = tuple(
            StatisticDistribution(
                statistic=name,
                total_paths=self.total_paths,
                counts=_value_counts_from_counter(self.statistics[name]),
            )
            for name, _ in STATISTIC_EXTRACTORS
        )
        bootstrap_statistic_distributions = tuple(
            BootstrapStatisticDistribution(
                bootstrap_member=member,
                statistic=name,
                paths=self.member_paths[member],
                counts=_value_counts_from_counter(
                    self.member_statistics[(member, name)]
                ),
            )
            for member in sorted(self.member_paths)
            for name, _ in STATISTIC_EXTRACTORS
        )
        metric_names = (
            "red_win",
            "blue_win",
            "ko_tko",
            "submission",
            "decision",
            "goes_distance",
        )
        horizon_seconds = self.scheduled_rounds * 300
        forecast = AggregateForecast(
            matchup_id=self.matchup_id,
            scheduled_rounds=self.scheduled_rounds,
            total_paths=self.total_paths,
            bootstrap_members=len(self.member_paths),
            outcome_counts=_ordered_counts(self.outcomes),
            bootstrap_outcome_counts=tuple(
                BootstrapOutcomeCounts(
                    bootstrap_member=member,
                    paths=self.member_paths[member],
                    counts=_ordered_counts(self.member_outcomes[member]),
                )
                for member in sorted(self.member_paths)
            ),
            duration_bins=tuple(
                DurationBin(upper_seconds=upper, count=self.duration_bins[upper])
                for upper in sorted(self.duration_bins)
            ),
            method_round_counts=tuple(
                MethodRoundCount(
                    method=method,
                    round_number=round_number,
                    count=count,
                )
                for (method, round_number), count in sorted(
                    self.method_rounds.items()
                )
            ),
            decision_type_counts=_ordered_counts(self.decision_types),
            total_lines=tuple(
                TotalLineCount(
                    half_rounds=line,
                    threshold_seconds=line * 300.0,
                    over=self.total_settlements[line][Settlement.OVER.value],
                    under=self.total_settlements[line][Settlement.UNDER.value],
                    push=self.total_settlements[line][Settlement.PUSH.value],
                    no_action=self.total_settlements[line][Settlement.NO_ACTION.value],
                )
                for line in valid_total_round_lines(self.scheduled_rounds)
            ),
            statistic_summaries=summaries,
            uncertainty=tuple(
                self._metric_uncertainty(name) for name in metric_names
            ),
            survival=tuple(
                ProbabilityPoint(
                    seconds=float(second),
                    probability=float(
                        np.count_nonzero(duration_us > second * 1_000_000)
                        / self.total_paths
                    ),
                )
                for second in range(
                    0, horizon_seconds + 1, self.survival_step_seconds
                )
            ),
            statistic_distributions=statistic_distributions,
            bootstrap_statistic_distributions=(
                bootstrap_statistic_distributions
            ),
            statistic_uncertainty=tuple(
                self._statistic_uncertainty(name)
                for name, _ in STATISTIC_EXTRACTORS
            ),
        )
        validate_aggregate_coherence(forecast)
        return forecast


def aggregate_paths(
    paths: Iterable[SimulationPath],
    scheduled_rounds: int,
    *,
    duration_bin_seconds: int = 5,
    survival_step_seconds: int = 30,
) -> AggregateForecast:
    """Reduce nested paths in a stable order; exact integer counts are authority."""

    values = tuple(sorted(paths, key=lambda item: (item.bootstrap_member, item.simulation_index)))
    if not values:
        raise ValueError("cannot aggregate zero simulation paths")
    keys = [(path.bootstrap_member, path.simulation_index) for path in values]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate bootstrap member/simulation index")
    accumulator = ForecastAccumulator(
        scheduled_rounds,
        duration_bin_seconds=duration_bin_seconds,
        survival_step_seconds=survival_step_seconds,
    )
    for path in values:
        accumulator.add_path(path)
    return accumulator.forecast()


def validate_aggregate_coherence(forecast: AggregateForecast) -> None:
    if sum(item.paths for item in forecast.bootstrap_outcome_counts) != forecast.total_paths:
        raise ValueError("bootstrap path totals do not match aggregate")
    for member in forecast.bootstrap_outcome_counts:
        if sum(item.count for item in member.counts) != member.paths:
            raise ValueError("bootstrap outcome counts do not match member paths")
    if sum(item.count for item in forecast.method_round_counts) != forecast.total_paths:
        raise ValueError("method-by-round counts do not match total paths")
    for line in forecast.total_lines:
        if line.over + line.under + line.push + line.no_action != forecast.total_paths:
            raise ValueError("total-line settlement counts do not match paths")
    for uncertainty in forecast.uncertainty:
        if not 0.0 <= uncertainty.estimate <= 1.0:
            raise ValueError("uncertainty estimate is outside probability bounds")
        if uncertainty.process_mcse < 0:
            raise ValueError("process MCSE cannot be negative")
    aggregate_statistics = {
        item.statistic: item for item in forecast.statistic_distributions
    }
    member_paths = {
        item.bootstrap_member: item.paths
        for item in forecast.bootstrap_outcome_counts
    }
    by_statistic: dict[str, list[BootstrapStatisticDistribution]] = defaultdict(list)
    for item in forecast.bootstrap_statistic_distributions:
        if item.bootstrap_member not in member_paths or item.paths != member_paths[item.bootstrap_member]:
            raise ValueError("bootstrap statistic distribution has the wrong member path count")
        by_statistic[item.statistic].append(item)
    if set(by_statistic) != set(aggregate_statistics):
        raise ValueError("bootstrap and aggregate statistic distributions disagree on names")
    for name, aggregate in aggregate_statistics.items():
        members = by_statistic[name]
        if (
            len(members) != forecast.bootstrap_members
            or {item.bootstrap_member for item in members} != set(member_paths)
        ):
            raise ValueError("each statistic requires one distribution per bootstrap member")
        combined: Counter[float] = Counter()
        for member in members:
            combined.update({item.value: item.count for item in member.counts})
        if combined != Counter({item.value: item.count for item in aggregate.counts}):
            raise ValueError("bootstrap statistic counts do not reproduce aggregate counts")
    uncertainty_names = [
        item.statistic for item in forecast.statistic_uncertainty
    ]
    if (
        set(uncertainty_names) != set(aggregate_statistics)
        or len(uncertainty_names) != len(set(uncertainty_names))
    ):
        raise ValueError("statistic uncertainty does not cover every exact distribution")
    for item in forecast.statistic_uncertainty:
        if {member for member, _ in item.conditional_means} != set(member_paths):
            raise ValueError(
                "statistic uncertainty does not cover every bootstrap member"
            )
