"""Settlement-aware market projections derived from coherent trajectories."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from .domain import AggregateForecast, OutcomeMethod, SimulationPath, TotalLineCount


class Settlement(str, Enum):
    OVER = "over"
    UNDER = "under"
    PUSH = "push"
    NO_ACTION = "no_action"


def valid_total_round_lines(scheduled_rounds: int) -> tuple[float, ...]:
    if scheduled_rounds < 1 or scheduled_rounds > 5:
        raise ValueError("scheduled_rounds must be between one and five")
    return tuple(index + 0.5 for index in range(scheduled_rounds))


def settle_total(result, line_rounds: float) -> Settlement:
    """Settle a full-fight total with explicit equality and no-contest rules."""

    if line_rounds <= 0:
        raise ValueError("total line must be positive")
    if result.method is OutcomeMethod.NO_CONTEST:
        return Settlement.NO_ACTION
    threshold_us = int(round(line_rounds * 300 * 1_000_000))
    if result.fight_time_us > threshold_us:
        return Settlement.OVER
    if result.fight_time_us < threshold_us:
        return Settlement.UNDER
    return Settlement.PUSH


def total_line_counts(
    paths: Iterable[SimulationPath],
    scheduled_rounds: int,
) -> tuple[TotalLineCount, ...]:
    values = tuple(paths)
    rows: list[TotalLineCount] = []
    for line in valid_total_round_lines(scheduled_rounds):
        counts = {settlement: 0 for settlement in Settlement}
        for path in values:
            counts[settle_total(path.result, line)] += 1
        rows.append(
            TotalLineCount(
                half_rounds=line,
                threshold_seconds=line * 300.0,
                over=counts[Settlement.OVER],
                under=counts[Settlement.UNDER],
                push=counts[Settlement.PUSH],
                no_action=counts[Settlement.NO_ACTION],
            )
        )
    return tuple(rows)


def coherent_market_probabilities(forecast: AggregateForecast) -> dict[str, object]:
    """Return labeled marginals without fitting independent market models."""

    outcomes = forecast.outcome_probabilities
    red_win = sum(value for key, value in outcomes.items() if key.startswith("red_"))
    blue_win = sum(value for key, value in outcomes.items() if key.startswith("blue_"))
    methods = {
        method.value: sum(
            value
            for key, value in outcomes.items()
            if key.endswith(f"_{method.value}") or key == method.value
        )
        for method in OutcomeMethod
    }
    method_round = {
        f"{item.method}_round_{item.round_number}": item.count / forecast.total_paths
        for item in forecast.method_round_counts
    }
    decisions = {
        item.outcome: item.count / forecast.total_paths
        for item in forecast.decision_type_counts
    }
    totals = {}
    for item in forecast.total_lines:
        actionable = item.over + item.under + item.push
        totals[format(item.half_rounds, ".1f")] = {
            "over": item.over / actionable if actionable else None,
            "under": item.under / actionable if actionable else None,
            "push": item.push / actionable if actionable else None,
            "no_action": item.no_action / forecast.total_paths,
            "counts": {
                "over": item.over,
                "under": item.under,
                "push": item.push,
                "no_action": item.no_action,
            },
        }
    return {
        "winner": {
            "red": red_win,
            "blue": blue_win,
            "draw": outcomes.get("draw", 0.0),
            "no_contest": outcomes.get("no_contest", 0.0),
        },
        "side_by_method": outcomes,
        "method": methods,
        "decision_type": decisions,
        "method_by_round": method_round,
        "total_rounds": totals,
        "goes_distance": methods[OutcomeMethod.DECISION.value] + methods[OutcomeMethod.DRAW.value],
    }
