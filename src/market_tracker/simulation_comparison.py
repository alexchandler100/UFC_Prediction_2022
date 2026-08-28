"""Immutable prospective comparisons that include the candidate simulator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from pathlib import Path
import random
from typing import ClassVar, Iterable, Mapping

from ._common import (
    MarketDataError,
    StoreIntegrityError,
    canonical_hash,
    iso_date,
    nonempty_text,
    probability,
    require_before_event,
    stable_id,
    utc_datetime,
    utc_text,
    validated_sha256,
)
from .blend import forecast_metrics, symmetric_logit_blend
from .paper import PaperDecision, PaperSettlement, _PaperRecordStore


SIMULATION_COMPARISON_POLICY_VERSION = "model-market-simulation-fixed-blends-v1"
SIMULATION_COMPARISON_FIRST_EVENT_DATE = "2026-09-01"
SIMULATION_COMPARISON_MINIMUM_SCORED_FIGHTS = 200
SIMULATION_COMPARISON_MINIMUM_SETTLED_EVENTS = 20
SIMULATION_COMPARISON_BOOTSTRAP_SAMPLES = 10_000


def equal_logit_pool(probabilities: Iterable[object]) -> float:
    """Average any fixed set of probabilities in log-odds space."""

    values = tuple(probability(value, "pooled_probability") for value in probabilities)
    if not values:
        raise ValueError("at least one probability is required")
    first_directional = next((value for value in values if value != 0.5), 0.5)
    complement = first_directional > 0.5
    working = tuple(1.0 - value if complement else value for value in values)
    average_logit = math.fsum(
        math.log(value) - math.log1p(-value) for value in working
    ) / len(working)
    if average_logit >= 0.0:
        pooled = 1.0 / (1.0 + math.exp(-average_logit))
    else:
        exponential = math.exp(average_logit)
        pooled = exponential / (1.0 + exponential)
    pooled = min(max(pooled, 1e-15), 1.0 - 1e-15)
    return 1.0 - pooled if complement else pooled


def _publication_event_date(value: object) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            parsed = date.fromisoformat(text)
        except ValueError as error:
            raise MarketDataError("simulation publication event date is invalid") from error
    return parsed.isoformat()


def _validate_simulation_publication(publication: Mapping[str, object]) -> dict[str, object]:
    value = dict(publication)
    if (
        value.get("candidate_only") is not True
        or value.get("paper_only") is not True
        or value.get("execution_enabled") is not False
        or value.get("production_influence") != "none"
    ):
        raise StoreIntegrityError("simulation publication must remain candidate-only")
    supplied_hash = validated_sha256(
        value.get("publication_sha256"), "simulation publication_sha256"
    )
    unhashed = dict(value)
    unhashed.pop("publication_sha256", None)
    if supplied_hash != canonical_hash(unhashed):
        raise StoreIntegrityError("simulation publication hash is invalid")
    stable_id(value.get("event_id"), "simulation event_id")
    _publication_event_date(value.get("event_date"))
    utc_datetime(
        value.get("forecast_issued_at_utc"),
        "simulation forecast_issued_at_utc",
    )
    matchups = value.get("matchups")
    if not isinstance(matchups, list):
        raise MarketDataError("simulation publication matchups must be a list")
    seen: set[str] = set()
    for item in matchups:
        if not isinstance(item, dict):
            raise MarketDataError("simulation publication matchup must be an object")
        matchup_id = stable_id(item.get("matchup_id"), "simulation matchup_id")
        if matchup_id in seen:
            raise StoreIntegrityError("simulation publication matchup IDs are duplicated")
        seen.add(matchup_id)
    return value


def _simulation_fighter_probability(item: Mapping[str, object]) -> float:
    aggregate = item.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise MarketDataError("available simulation matchup has no aggregate")
    raw_probabilities = aggregate.get("outcome_probabilities")
    if isinstance(raw_probabilities, Mapping):
        try:
            values = {
                str(key): float(value) for key, value in raw_probabilities.items()
            }
        except (TypeError, ValueError) as error:
            raise MarketDataError("simulation outcome probability is not numeric") from error
    else:
        counts = aggregate.get("outcome_counts")
        if not isinstance(counts, Mapping):
            raise MarketDataError("simulation aggregate has no outcome distribution")
        try:
            numeric_counts = {
                str(key): float(value) for key, value in counts.items()
            }
        except (TypeError, ValueError) as error:
            raise MarketDataError("simulation outcome count is not numeric") from error
        if any(not math.isfinite(value) or value < 0.0 for value in numeric_counts.values()):
            raise MarketDataError("simulation outcome counts must be finite and nonnegative")
        total = math.fsum(numeric_counts.values())
        if not math.isfinite(total) or total <= 0.0:
            raise MarketDataError("simulation outcome count total is invalid")
        values = {key: value / total for key, value in numeric_counts.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise MarketDataError("simulation outcome probabilities must be finite and nonnegative")
    red = math.fsum(value for key, value in values.items() if key.startswith("red_"))
    blue = math.fsum(value for key, value in values.items() if key.startswith("blue_"))
    if red + blue <= 0.0:
        raise MarketDataError("simulation has no decisive winner paths")
    return probability(red / (red + blue), "simulation_probability")


@dataclass(frozen=True)
class SimulationComparisonDecision:
    schema_version: int
    comparison_id: str
    policy_version: str
    candidate_only: bool
    paper_only: bool
    execution_enabled: bool
    base_decision_id: str
    matchup_id: str
    event_id: str
    fighter_id: str
    opponent_id: str
    event_date: str
    timing_precision: str
    event_start_utc: str
    base_decision_issued_at_utc: str
    comparison_issued_at_utc: str
    simulation_forecast_issued_at_utc: str
    simulation_publication_sha256: str
    simulation_parameter_artifact_sha256: str
    mechanics_profile_id: str
    market_probability: float
    model_probability: float
    simulation_probability: float
    market_model_probability: float
    market_simulation_probability: float
    model_simulation_probability: float
    equal_three_probability: float

    FIELDNAMES: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "comparison_id",
        "policy_version",
        "candidate_only",
        "paper_only",
        "execution_enabled",
        "base_decision_id",
        "matchup_id",
        "event_id",
        "fighter_id",
        "opponent_id",
        "event_date",
        "timing_precision",
        "event_start_utc",
        "base_decision_issued_at_utc",
        "comparison_issued_at_utc",
        "simulation_forecast_issued_at_utc",
        "simulation_publication_sha256",
        "simulation_parameter_artifact_sha256",
        "mechanics_profile_id",
        "market_probability",
        "model_probability",
        "simulation_probability",
        "market_model_probability",
        "market_simulation_probability",
        "model_simulation_probability",
        "equal_three_probability",
    )

    @classmethod
    def create(
        cls,
        base: PaperDecision,
        publication: Mapping[str, object],
        *,
        comparison_issued_at_utc: datetime | str,
    ) -> "SimulationComparisonDecision":
        if base.event_date < SIMULATION_COMPARISON_FIRST_EVENT_DATE:
            raise MarketDataError("base decision predates the simulation comparison")
        value = _validate_simulation_publication(publication)
        if stable_id(value.get("event_id"), "simulation event_id") != base.event_id:
            raise StoreIntegrityError("simulation and market decisions identify different events")
        if _publication_event_date(value.get("event_date")) != base.event_date:
            raise StoreIntegrityError("simulation and market event dates disagree")
        simulation_issued = utc_text(
            value.get("forecast_issued_at_utc"),
            "simulation forecast_issued_at_utc",
        )
        comparison_issued = utc_text(
            comparison_issued_at_utc, "comparison_issued_at_utc"
        )
        if utc_datetime(simulation_issued, "simulation_forecast_issued_at_utc") > utc_datetime(
            base.decision_issued_at_utc, "base_decision_issued_at_utc"
        ):
            raise MarketDataError("simulation was not available at the T-24 decision")
        if utc_datetime(comparison_issued, "comparison_issued_at_utc") < utc_datetime(
            base.decision_issued_at_utc, "base_decision_issued_at_utc"
        ):
            raise MarketDataError("comparison cannot precede the base decision")
        require_before_event(
            comparison_issued,
            event_date=base.event_date,
            timing_precision=base.timing_precision,
            event_start_utc=base.event_start_utc,
            observed_field="comparison_issued_at_utc",
        )
        matchup = next(
            (
                item
                for item in value["matchups"]
                if isinstance(item, Mapping)
                and str(item.get("matchup_id")) == base.matchup_id
            ),
            None,
        )
        if matchup is None or matchup.get("status") != "available":
            raise MarketDataError("simulation is unavailable for this frozen matchup")
        red_id = stable_id(matchup.get("fighter_id"), "simulation fighter_id")
        blue_id = stable_id(matchup.get("opponent_id"), "simulation opponent_id")
        if {red_id, blue_id} != {base.fighter_id, base.opponent_id}:
            raise StoreIntegrityError("simulation and market fighters disagree")
        red_probability = _simulation_fighter_probability(matchup)
        simulation_probability = (
            red_probability if red_id == base.fighter_id else 1.0 - red_probability
        )
        market_probability = float(base.market_probability)
        model_probability = float(base.model_probability)
        body = {
            "schema_version": 1,
            "policy_version": SIMULATION_COMPARISON_POLICY_VERSION,
            "candidate_only": True,
            "paper_only": True,
            "execution_enabled": False,
            "base_decision_id": base.decision_id,
            "matchup_id": base.matchup_id,
            "event_id": base.event_id,
            "fighter_id": base.fighter_id,
            "opponent_id": base.opponent_id,
            "event_date": base.event_date,
            "timing_precision": base.timing_precision,
            "event_start_utc": str(base.event_start_utc or ""),
            "base_decision_issued_at_utc": base.decision_issued_at_utc,
            "comparison_issued_at_utc": comparison_issued,
            "simulation_forecast_issued_at_utc": simulation_issued,
            "simulation_publication_sha256": value["publication_sha256"],
            "simulation_parameter_artifact_sha256": validated_sha256(
                value.get("parameter_artifact_sha256"),
                "simulation parameter_artifact_sha256",
            ),
            "mechanics_profile_id": nonempty_text(
                value.get("mechanics_profile_id"), "mechanics_profile_id"
            ),
            "market_probability": market_probability,
            "model_probability": model_probability,
            "simulation_probability": simulation_probability,
            "market_model_probability": symmetric_logit_blend(
                market_probability, model_probability, 0.5
            ),
            "market_simulation_probability": symmetric_logit_blend(
                market_probability, simulation_probability, 0.5
            ),
            "model_simulation_probability": symmetric_logit_blend(
                model_probability, simulation_probability, 0.5
            ),
            "equal_three_probability": equal_logit_pool(
                (market_probability, model_probability, simulation_probability)
            ),
        }
        return cls(comparison_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(
        cls, record: Mapping[str, object]
    ) -> "SimulationComparisonDecision":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"simulation comparison schema mismatch; missing={missing}, extra={extra}"
            )
        try:
            decision = cls(**{field: record[field] for field in cls.FIELDNAMES})
        except TypeError as error:
            raise MarketDataError("invalid simulation comparison fields") from error
        decision.validate_integrity()
        return decision

    @property
    def natural_key(self) -> tuple[str]:
        return (self.base_decision_id,)

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}

    def validate_integrity(self) -> None:
        if self.schema_version != 1:
            raise MarketDataError("unsupported simulation comparison schema")
        if (
            self.policy_version != SIMULATION_COMPARISON_POLICY_VERSION
            or self.candidate_only is not True
            or self.paper_only is not True
            or self.execution_enabled is not False
        ):
            raise StoreIntegrityError("simulation comparison must remain fixed paper research")
        for field in (
            "base_decision_id",
            "matchup_id",
            "event_id",
            "fighter_id",
            "opponent_id",
        ):
            stable_id(getattr(self, field), field)
        if self.fighter_id == self.opponent_id:
            raise StoreIntegrityError("simulation comparison fighters must differ")
        iso_date(self.event_date)
        if self.event_date < SIMULATION_COMPARISON_FIRST_EVENT_DATE:
            raise StoreIntegrityError("simulation comparison predates its policy")
        require_before_event(
            self.comparison_issued_at_utc,
            event_date=self.event_date,
            timing_precision=self.timing_precision,
            event_start_utc=self.event_start_utc,
            observed_field="comparison_issued_at_utc",
        )
        base_time = utc_datetime(
            self.base_decision_issued_at_utc, "base_decision_issued_at_utc"
        )
        comparison_time = utc_datetime(
            self.comparison_issued_at_utc, "comparison_issued_at_utc"
        )
        simulation_time = utc_datetime(
            self.simulation_forecast_issued_at_utc,
            "simulation_forecast_issued_at_utc",
        )
        if simulation_time > base_time or base_time > comparison_time:
            raise StoreIntegrityError("simulation comparison timing order is invalid")
        validated_sha256(
            self.simulation_publication_sha256,
            "simulation_publication_sha256",
        )
        validated_sha256(
            self.simulation_parameter_artifact_sha256,
            "simulation_parameter_artifact_sha256",
        )
        nonempty_text(self.mechanics_profile_id, "mechanics_profile_id")
        values = (
            probability(self.market_probability, "market_probability"),
            probability(self.model_probability, "model_probability"),
            probability(self.simulation_probability, "simulation_probability"),
        )
        expected = (
            symmetric_logit_blend(values[0], values[1], 0.5),
            symmetric_logit_blend(values[0], values[2], 0.5),
            symmetric_logit_blend(values[1], values[2], 0.5),
            equal_logit_pool(values),
        )
        supplied = (
            self.market_model_probability,
            self.market_simulation_probability,
            self.model_simulation_probability,
            self.equal_three_probability,
        )
        if any(abs(float(left) - right) > 1e-12 for left, right in zip(supplied, expected)):
            raise StoreIntegrityError("simulation comparison probabilities are inconsistent")
        body = self.to_mapping()
        body.pop("comparison_id")
        if self.comparison_id != canonical_hash(body):
            raise StoreIntegrityError("simulation comparison ID is invalid")


class SimulationComparisonDecisionStore(_PaperRecordStore):
    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        super().__init__(
            csv_path,
            jsonl_path,
            record_type=SimulationComparisonDecision,
            id_field="comparison_id",
            time_field="comparison_issued_at_utc",
        )

    def read(self) -> tuple[SimulationComparisonDecision, ...]:
        return tuple(super().read())  # type: ignore[return-value]


@dataclass(frozen=True)
class SimulationComparisonBuild:
    decisions: tuple[SimulationComparisonDecision, ...]
    eligible_base_decisions: int
    already_frozen: int
    unavailable_or_mismatched: int
    publication_status: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "eligible_base_decisions": self.eligible_base_decisions,
            "already_frozen": self.already_frozen,
            "created": len(self.decisions),
            "unavailable_or_mismatched": self.unavailable_or_mismatched,
            "publication_status": self.publication_status,
        }


def build_simulation_comparison_decisions(
    base_decisions: Iterable[PaperDecision],
    existing: Iterable[SimulationComparisonDecision],
    publication: Mapping[str, object] | None,
    *,
    comparison_issued_at_utc: datetime | str,
) -> SimulationComparisonBuild:
    eligible = tuple(
        item
        for item in base_decisions
        if item.event_date >= SIMULATION_COMPARISON_FIRST_EVENT_DATE
    )
    existing_base_ids = {item.base_decision_id for item in existing}
    pending: list[SimulationComparisonDecision] = []
    unavailable = 0
    if publication is None:
        return SimulationComparisonBuild(
            decisions=(),
            eligible_base_decisions=len(eligible),
            already_frozen=sum(item.decision_id in existing_base_ids for item in eligible),
            unavailable_or_mismatched=sum(
                item.decision_id not in existing_base_ids for item in eligible
            ),
            publication_status="missing",
        )
    try:
        validated = _validate_simulation_publication(publication)
    except (MarketDataError, StoreIntegrityError):
        return SimulationComparisonBuild(
            decisions=(),
            eligible_base_decisions=len(eligible),
            already_frozen=sum(item.decision_id in existing_base_ids for item in eligible),
            unavailable_or_mismatched=sum(
                item.decision_id not in existing_base_ids for item in eligible
            ),
            publication_status="invalid",
        )
    for base in eligible:
        if base.decision_id in existing_base_ids:
            continue
        try:
            pending.append(
                SimulationComparisonDecision.create(
                    base,
                    validated,
                    comparison_issued_at_utc=comparison_issued_at_utc,
                )
            )
        except (MarketDataError, StoreIntegrityError):
            unavailable += 1
    return SimulationComparisonBuild(
        decisions=tuple(pending),
        eligible_base_decisions=len(eligible),
        already_frozen=sum(item.decision_id in existing_base_ids for item in eligible),
        unavailable_or_mismatched=unavailable,
        publication_status="validated",
    )


def _quantile(values: list[float], probability_value: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * probability_value
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _metric(probability_value: float, target: int, name: str) -> float:
    if name == "log_loss":
        correct = probability_value if target else 1.0 - probability_value
        return -math.log(correct)
    if name == "brier_score":
        return (probability_value - target) ** 2
    if name == "accuracy":
        if probability_value == 0.5:
            return 0.5
        return float((probability_value > 0.5) == bool(target))
    raise ValueError(f"unsupported metric: {name}")


def _paired_interval(
    rows: tuple[tuple[SimulationComparisonDecision, int], ...],
    *,
    candidate_field: str,
    reference_field: str,
    metric_name: str,
) -> dict[str, object]:
    blocks: dict[str, tuple[float, int]] = {}
    for record, target in rows:
        difference = _metric(float(getattr(record, candidate_field)), target, metric_name) - _metric(
            float(getattr(record, reference_field)), target, metric_name
        )
        total, count = blocks.get(record.event_id, (0.0, 0))
        blocks[record.event_id] = (total + difference, count + 1)
    count = sum(value[1] for value in blocks.values())
    result: dict[str, object] = {
        "definition": (
            f"candidate minus market {metric_name}; "
            f"{'positive' if metric_name == 'accuracy' else 'negative'} favors candidate"
        ),
        "event_count": len(blocks),
        "fight_count": count,
        "point_difference": (
            sum(value[0] for value in blocks.values()) / count if count else None
        ),
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(blocks) < 2 or not count:
        return result
    ordered = [blocks[key] for key in sorted(blocks)]
    seed = canonical_hash(
        {
            "policy": SIMULATION_COMPARISON_POLICY_VERSION,
            "candidate": candidate_field,
            "reference": reference_field,
            "metric": metric_name,
            "records": [
                {"comparison_id": record.comparison_id, "target": target}
                for record, target in rows
            ],
        }
    )
    generator = random.Random(int(seed[:16], 16))
    samples = []
    for _ in range(SIMULATION_COMPARISON_BOOTSTRAP_SAMPLES):
        selected = [generator.choice(ordered) for _ in ordered]
        selected_count = sum(value[1] for value in selected)
        samples.append(sum(value[0] for value in selected) / selected_count)
    samples.sort()
    result.update(
        {
            "bootstrap_samples": len(samples),
            "ci_95_lower": _quantile(samples, 0.025),
            "ci_95_upper": _quantile(samples, 0.975),
        }
    )
    return result


def simulation_comparison_report(
    records: Iterable[SimulationComparisonDecision],
    settlements: Iterable[PaperSettlement],
    base_decisions: Iterable[PaperDecision],
) -> dict[str, object]:
    comparisons = tuple(sorted(records, key=lambda item: item.comparison_id))
    all_settlements = tuple(settlements)
    all_base_decisions = tuple(base_decisions)
    if len({item.comparison_id for item in comparisons}) != len(comparisons):
        raise ValueError("simulation comparison IDs are not unique")
    if len({item.base_decision_id for item in comparisons}) != len(comparisons):
        raise ValueError("simulation comparisons repeat a base decision")
    if len({item.decision_id for item in all_settlements}) != len(all_settlements):
        raise ValueError("paper settlements contain duplicate decision IDs")
    if len({item.decision_id for item in all_base_decisions}) != len(all_base_decisions):
        raise ValueError("paper decision IDs are not unique")
    base_ids = {item.decision_id for item in all_base_decisions}
    if any(item.base_decision_id not in base_ids for item in comparisons):
        raise ValueError("simulation comparison references an unknown base decision")
    settlement_by_base = {item.decision_id: item for item in all_settlements}
    eligible_base = tuple(
        item
        for item in all_base_decisions
        if item.event_date >= SIMULATION_COMPARISON_FIRST_EVENT_DATE
    )
    rows = tuple(
        (record, int(settlement_by_base[record.base_decision_id].target))
        for record in comparisons
        if record.base_decision_id in settlement_by_base
        and settlement_by_base[record.base_decision_id].target is not None
    )
    fields = {
        "market": "market_probability",
        "production_model": "model_probability",
        "simulation": "simulation_probability",
        "market_model_half": "market_model_probability",
        "market_simulation_half": "market_simulation_probability",
        "model_simulation_half": "model_simulation_probability",
        "market_model_simulation_thirds": "equal_three_probability",
    }
    scores = {
        label: (
            forecast_metrics(
                [getattr(record, field) for record, _ in rows],
                [target for _, target in rows],
            ).to_mapping()
            if rows
            else None
        )
        for label, field in fields.items()
    }
    paired = {
        label: {
            metric_name: _paired_interval(
                rows,
                candidate_field=field,
                reference_field="market_probability",
                metric_name=metric_name,
            )
            for metric_name in ("log_loss", "brier_score", "accuracy")
        }
        for label, field in fields.items()
        if label != "market"
    } if rows else {}
    events = len({record.event_id for record, _ in rows})
    enough = (
        len(rows) >= SIMULATION_COMPARISON_MINIMUM_SCORED_FIGHTS
        and events >= SIMULATION_COMPARISON_MINIMUM_SETTLED_EVENTS
    )
    three_way = paired.get("market_model_simulation_thirds", {})
    log_loss_better = bool(
        three_way
        and three_way["log_loss"]["ci_95_upper"] is not None
        and float(three_way["log_loss"]["ci_95_upper"]) < 0.0
    )
    brier_not_worse = bool(
        three_way
        and three_way["brier_score"]["ci_95_upper"] is not None
        and float(three_way["brier_score"]["ci_95_upper"]) <= 0.0
    )
    if not enough:
        status = "collecting_results"
    elif log_loss_better and brier_not_worse:
        status = "equal_three_way_blend_improves_probability_quality"
    else:
        status = "equal_three_way_blend_not_proven_better"
    return {
        "policy_version": SIMULATION_COMPARISON_POLICY_VERSION,
        "cohort": {
            "first_eligible_event_date": SIMULATION_COMPARISON_FIRST_EVENT_DATE,
            "source": "immutable_t24_market_decision_plus_preexisting_simulation",
            "only_fights_with_all_three_probabilities": True,
            "withheld_simulations_are_not_imputed": True,
            "simulation_winner_probability_conditions_on_decisive_paths": True,
        },
        "paper_only": True,
        "execution_enabled": False,
        "status": status,
        "eligible_base_decisions": len(eligible_base),
        "frozen_simulation_comparisons": len(comparisons),
        "missing_simulation_comparisons": max(len(eligible_base) - len(comparisons), 0),
        "simulation_coverage": (
            len(comparisons) / len(eligible_base) if eligible_base else None
        ),
        "scored_fights": len(rows),
        "settled_events": events,
        "fixed_forecasts": {
            "individual": ["market", "production_model", "simulation"],
            "pairwise_half_logit": [
                "market_model_half",
                "market_simulation_half",
                "model_simulation_half",
            ],
            "equal_three_way_logit": "market_model_simulation_thirds",
            "weight_search_allowed": False,
        },
        "scores": scores,
        "paired_event_intervals_vs_market": paired,
        "checkpoint": {
            "minimum_scored_fights": SIMULATION_COMPARISON_MINIMUM_SCORED_FIGHTS,
            "minimum_settled_events": SIMULATION_COMPARISON_MINIMUM_SETTLED_EVENTS,
            "sample_requirement_met": enough,
            "three_way_log_loss_better_than_market_requirement_met": log_loss_better,
            "three_way_brier_not_worse_than_market_requirement_met": brier_not_worse,
            "review_required_before_any_production_change": True,
            "execution_enabled": False,
        },
        "comparison_dataset_sha256": canonical_hash(
            [item.to_mapping() for item in comparisons]
        ),
        "scored_input_sha256": canonical_hash(
            [
                {"comparison_id": record.comparison_id, "target": target}
                for record, target in rows
            ]
        ),
    }
