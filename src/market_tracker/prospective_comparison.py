"""Frozen prospective comparison of model, market, and their equal blend."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

from ._common import canonical_hash
from .blend import forecast_metrics, symmetric_logit_blend
from .paper import PaperDecision, PaperSettlement


PROSPECTIVE_COMPARISON_POLICY_VERSION = "model-market-equal-logit-v1"
PROSPECTIVE_COMPARISON_FIRST_EVENT_DATE = "2026-09-01"
PROSPECTIVE_EQUAL_BLEND_GAMMA = 0.5
PROSPECTIVE_MINIMUM_SCORED_FIGHTS = 200
PROSPECTIVE_MINIMUM_SETTLED_EVENTS = 20
PROSPECTIVE_BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class _ScoredForecast:
    decision_id: str
    event_id: str
    target: int
    market_probability: float
    model_probability: float
    equal_blend_probability: float

    def to_mapping(self) -> dict[str, object]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _score(probability: float, target: int, metric: str) -> float:
    if metric == "log_loss":
        target_probability = probability if target == 1 else 1.0 - probability
        return -math.log(target_probability)
    if metric == "brier_score":
        return (probability - target) ** 2
    if metric == "accuracy":
        if probability == 0.5:
            return 0.5
        return float((probability > 0.5) == bool(target))
    raise ValueError(f"unsupported paired metric: {metric}")


def _paired_event_interval(
    rows: tuple[_ScoredForecast, ...],
    *,
    candidate_field: str,
    reference_field: str,
    metric: str,
) -> dict[str, object]:
    """Compare two forecasts while resampling whole UFC cards."""

    blocks: dict[str, tuple[float, int]] = {}
    for row in rows:
        candidate = float(getattr(row, candidate_field))
        reference = float(getattr(row, reference_field))
        difference = _score(candidate, row.target, metric) - _score(
            reference, row.target, metric
        )
        total, count = blocks.get(row.event_id, (0.0, 0))
        blocks[row.event_id] = (total + difference, count + 1)
    fight_count = sum(count for _, count in blocks.values())
    point = (
        sum(total for total, _ in blocks.values()) / fight_count
        if fight_count
        else None
    )
    favorable_direction = "positive" if metric == "accuracy" else "negative"
    result: dict[str, object] = {
        "definition": (
            f"candidate minus reference {metric}; {favorable_direction} favors candidate"
        ),
        "event_count": len(blocks),
        "fight_count": fight_count,
        "point_difference": point,
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(blocks) < 2 or fight_count == 0:
        return result
    ordered = [blocks[event_id] for event_id in sorted(blocks)]
    seed = canonical_hash(
        {
            "policy_version": PROSPECTIVE_COMPARISON_POLICY_VERSION,
            "candidate_field": candidate_field,
            "reference_field": reference_field,
            "metric": metric,
            "rows": [row.to_mapping() for row in rows],
        }
    )
    generator = random.Random(int(seed[:16], 16))
    samples: list[float] = []
    for _ in range(PROSPECTIVE_BOOTSTRAP_SAMPLES):
        selected = [generator.choice(ordered) for _ in ordered]
        selected_count = sum(count for _, count in selected)
        samples.append(sum(total for total, _ in selected) / selected_count)
    samples.sort()
    result.update(
        {
            "bootstrap_samples": len(samples),
            "ci_95_lower": _quantile(samples, 0.025),
            "ci_95_upper": _quantile(samples, 0.975),
        }
    )
    return result


def _comparison(
    rows: tuple[_ScoredForecast, ...],
    *,
    candidate_field: str,
    reference_field: str,
) -> dict[str, object]:
    return {
        metric: _paired_event_interval(
            rows,
            candidate_field=candidate_field,
            reference_field=reference_field,
            metric=metric,
        )
        for metric in ("log_loss", "brier_score", "accuracy")
    }


def _disagreement_summary(rows: tuple[_ScoredForecast, ...]) -> dict[str, int]:
    market_correct = model_correct = equal_correct = equal_ties = 0
    disagreements = 0
    for row in rows:
        market_side = row.market_probability > 0.5
        model_side = row.model_probability > 0.5
        if market_side == model_side:
            continue
        disagreements += 1
        market_correct += int(market_side == bool(row.target))
        model_correct += int(model_side == bool(row.target))
        if row.equal_blend_probability == 0.5:
            equal_ties += 1
        else:
            equal_correct += int(
                (row.equal_blend_probability > 0.5) == bool(row.target)
            )
    return {
        "fights_where_model_and_market_pick_different_winners": disagreements,
        "market_correct_on_those_fights": market_correct,
        "model_correct_on_those_fights": model_correct,
        "equal_blend_correct_on_those_fights": equal_correct,
        "equal_blend_ties_on_those_fights": equal_ties,
    }


def prospective_comparison_report(
    decisions: Iterable[PaperDecision],
    settlements: Iterable[PaperSettlement],
) -> dict[str, object]:
    """Score the predeclared equal blend only on future frozen decisions."""

    all_decisions = tuple(decisions)
    all_settlements = tuple(settlements)
    if len({item.decision_id for item in all_decisions}) != len(all_decisions):
        raise ValueError("paper decision IDs are not unique")
    if len({item.decision_id for item in all_settlements}) != len(all_settlements):
        raise ValueError("paper settlements contain duplicate decision IDs")
    decision_by_id = {item.decision_id: item for item in all_decisions}
    unknown_settlements = [
        item.decision_id
        for item in all_settlements
        if item.decision_id not in decision_by_id
    ]
    if unknown_settlements:
        raise ValueError("paper settlement references an unknown decision")

    eligible_decisions = tuple(
        sorted(
            (
                item
                for item in all_decisions
                if item.event_date >= PROSPECTIVE_COMPARISON_FIRST_EVENT_DATE
            ),
            key=lambda item: item.decision_id,
        )
    )
    eligible_ids = {item.decision_id for item in eligible_decisions}
    eligible_settlements = tuple(
        sorted(
            (
                item
                for item in all_settlements
                if item.decision_id in eligible_ids
            ),
            key=lambda item: item.decision_id,
        )
    )
    settlement_by_id = {item.decision_id: item for item in eligible_settlements}
    rows = tuple(
        _ScoredForecast(
            decision_id=decision.decision_id,
            event_id=decision.event_id,
            target=int(settlement_by_id[decision.decision_id].target),
            market_probability=float(decision.market_probability),
            model_probability=float(decision.model_probability),
            equal_blend_probability=symmetric_logit_blend(
                decision.market_probability,
                decision.model_probability,
                PROSPECTIVE_EQUAL_BLEND_GAMMA,
            ),
        )
        for decision in eligible_decisions
        if decision.decision_id in settlement_by_id
        and settlement_by_id[decision.decision_id].target is not None
    )

    scores: dict[str, object]
    comparisons: dict[str, object]
    if rows:
        targets = [row.target for row in rows]
        scores = {
            "market": forecast_metrics(
                [row.market_probability for row in rows], targets
            ).to_mapping(),
            "production_model": forecast_metrics(
                [row.model_probability for row in rows], targets
            ).to_mapping(),
            "fixed_equal_logit_blend": forecast_metrics(
                [row.equal_blend_probability for row in rows], targets
            ).to_mapping(),
        }
        comparisons = {
            "equal_blend_vs_market": _comparison(
                rows,
                candidate_field="equal_blend_probability",
                reference_field="market_probability",
            ),
            "equal_blend_vs_model": _comparison(
                rows,
                candidate_field="equal_blend_probability",
                reference_field="model_probability",
            ),
        }
    else:
        scores = {
            "market": None,
            "production_model": None,
            "fixed_equal_logit_blend": None,
        }
        comparisons = {
            "equal_blend_vs_market": None,
            "equal_blend_vs_model": None,
        }

    settled_events = len({row.event_id for row in rows})
    enough_data = (
        len(rows) >= PROSPECTIVE_MINIMUM_SCORED_FIGHTS
        and settled_events >= PROSPECTIVE_MINIMUM_SETTLED_EVENTS
    )
    equal_vs_market = comparisons["equal_blend_vs_market"]
    log_loss_supported = bool(
        equal_vs_market
        and equal_vs_market["log_loss"]["ci_95_upper"] is not None
        and float(equal_vs_market["log_loss"]["ci_95_upper"]) < 0.0
    )
    brier_not_worse = bool(
        equal_vs_market
        and equal_vs_market["brier_score"]["ci_95_upper"] is not None
        and float(equal_vs_market["brier_score"]["ci_95_upper"]) <= 0.0
    )
    if not enough_data:
        conclusion = "collecting_results"
    elif log_loss_supported and brier_not_worse:
        conclusion = "equal_blend_improves_probability_quality"
    else:
        conclusion = "equal_blend_not_proven_better"

    return {
        "policy_version": PROSPECTIVE_COMPARISON_POLICY_VERSION,
        "purpose": (
            "test whether the production model adds useful information to the "
            "market when both are frozen before the fight"
        ),
        "cohort": {
            "first_eligible_event_date": PROSPECTIVE_COMPARISON_FIRST_EVENT_DATE,
            "source": "immutable_t24_paper_decisions",
            "one_frozen_decision_per_matchup": True,
            "known_earlier_results_excluded": True,
        },
        "blend": {
            "method": "equal_weight_log_odds",
            "market_weight": 0.5,
            "model_weight": 0.5,
            "gamma": PROSPECTIVE_EQUAL_BLEND_GAMMA,
            "formula": (
                "logit(equal_blend) = 0.5*logit(market) + "
                "0.5*logit(model)"
            ),
            "retuning_allowed": False,
        },
        "paper_only": True,
        "execution_enabled": False,
        "status": conclusion,
        "eligible_frozen_decisions": len(eligible_decisions),
        "settled_decisions_including_voids": len(eligible_settlements),
        "void_or_unscored_decisions": sum(
            item.target is None for item in eligible_settlements
        ),
        "scored_fights": len(rows),
        "settled_events": settled_events,
        "model_ids": sorted({item.model_id for item in eligible_decisions}),
        "scores": scores,
        "score_notes": {
            "primary_measure": "log_loss",
            "accuracy_treats_an_exact_50_50_as_half_correct": True,
            "calibration_measure": "expected_calibration_error",
            "lower_is_better_for_log_loss_brier_and_calibration_error": True,
            "higher_is_better_for_accuracy_and_roc_auc": True,
        },
        "paired_event_intervals": comparisons,
        "winner_disagreements": _disagreement_summary(rows),
        "checkpoint": {
            "minimum_scored_fights": PROSPECTIVE_MINIMUM_SCORED_FIGHTS,
            "minimum_settled_events": PROSPECTIVE_MINIMUM_SETTLED_EVENTS,
            "sample_requirement_met": enough_data,
            "equal_blend_log_loss_better_than_market_requirement_met": (
                log_loss_supported
            ),
            "equal_blend_brier_not_worse_than_market_requirement_met": (
                brier_not_worse
            ),
            "review_required_before_any_production_change": True,
            "execution_enabled": False,
        },
        "decision_dataset_sha256": canonical_hash(
            [item.to_mapping() for item in eligible_decisions]
        ),
        "settlement_dataset_sha256": canonical_hash(
            [item.to_mapping() for item in eligible_settlements]
        ),
        "scored_input_sha256": canonical_hash(
            [row.to_mapping() for row in rows]
        ),
    }
