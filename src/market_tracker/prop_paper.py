"""Prospective paper decisions and settlement for full-fight round totals.

The ledger freezes three probabilities for every decision: the leave-one-book-
out market estimate, the independent duration model, and a market-residual
challenger.  The challenger may use the model only after earlier, settled
events demonstrate a statistically credible improvement over the market.
Nothing in this module can execute or size a real wager.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import math
import random
from statistics import median
from pathlib import Path
from typing import ClassVar, Iterable, Mapping

from ._common import (
    BETTING_STATUS,
    MarketDataError,
    SCHEMA_VERSION,
    StoreIntegrityError,
    binary_target,
    canonical_hash,
    implied_probability,
    moneyline,
    optional_stable_id,
    probability,
    require_before_event,
    utc_datetime,
    utc_text,
    validated_sha256,
)
from .blend import symmetric_logit_blend
from .paper import _PaperRecordStore, _profit_for_one_unit_risk
from .props import TotalRoundsForecastCapture, TotalRoundsQuoteSnapshot
from .quotes import AppendResult


TOTAL_DECISION_TARGET_LEAD_SECONDS = 24.0 * 60.0 * 60.0
TOTAL_DECISION_WINDOW_SECONDS = 4.0 * 60.0 * 60.0
TOTAL_MAX_SOURCE_QUOTE_AGE_SECONDS = 30.0 * 60.0
TOTAL_MIN_CONSENSUS_BOOKS = 2
TOTAL_MIN_EXPECTED_RETURN = 0.05
TOTAL_SHADOW_THRESHOLDS = (0.0, 0.025, 0.05, 0.075, 0.10)
RESIDUAL_WEIGHT_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
RESIDUAL_MIN_SCORED_LINES = 100
RESIDUAL_MIN_SETTLED_EVENTS = 10
RESIDUAL_MIN_LOG_LOSS_GAIN = 0.002
RESIDUAL_BOOTSTRAP_SAMPLES = 2_000
TOTAL_POLICY_VERSION = "prospective-total-round-residual-v1"


def _finite(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise MarketDataError(f"{field} must be numeric") from error
    if not math.isfinite(parsed):
        raise MarketDataError(f"{field} must be finite")
    return parsed


def _line(value: object) -> float:
    parsed = _finite(value, "line")
    if not 0.0 < parsed <= 25.0 or round(parsed, 3) != parsed:
        raise MarketDataError("line must be in (0, 25] with at most three decimals")
    return parsed


def _log_loss(target: int, predicted: float) -> float:
    bounded = min(max(float(predicted), 1e-15), 1.0 - 1e-15)
    return -math.log(bounded if target == 1 else 1.0 - bounded)


def _quantile(values: list[float], probability_value: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class ResidualWeightSelection:
    selected_weight: float
    scored_lines: int
    settled_events: int
    market_log_loss: float | None
    selected_log_loss: float | None
    selected_minus_market_log_loss: float | None
    ci_95_upper: float | None
    selection_status: str
    training_dataset_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class TotalRoundsPaperDecision:
    schema_version: int
    decision_id: str
    betting_status: str
    paper_only: bool
    policy_version: str
    capture_id: str
    matchup_id: str
    fight_id: str | None
    event_id: str
    fighter_id: str
    opponent_id: str
    fighter_name: str
    opponent_name: str
    event_date: str
    timing_precision: str
    event_start_utc: str | None
    decision_issued_at_utc: str
    market_as_of_utc: str
    reference_quote_id: str
    forecast_capture_id: str
    line: float
    target_book: str
    target_book_key: str
    target_source_quote_age_seconds: float
    maximum_source_quote_age_seconds: float
    consensus_id: str
    consensus_book_count: int
    consensus_book_keys: str
    market_over_probability: float
    model_over_probability: float
    residual_over_probability: float
    selected_residual_weight: float
    residual_selection_status: str
    residual_training_scored_lines: int
    residual_training_settled_events: int
    residual_training_dataset_sha256: str
    over_reference_moneyline: int
    under_reference_moneyline: int
    over_break_even_probability: float
    under_break_even_probability: float
    model_over_expected_return: float
    model_under_expected_return: float
    residual_over_expected_return: float
    residual_under_expected_return: float
    minimum_expected_return: float
    paper_action: str
    action_probability: float | None
    action_reference_moneyline: int | None
    hypothetical_risk_units: float

    FIELDNAMES: ClassVar[tuple[str, ...]] = tuple(__annotations__)

    @classmethod
    def create(
        cls,
        reference_quote: TotalRoundsQuoteSnapshot,
        consensus_quotes: Iterable[TotalRoundsQuoteSnapshot],
        forecast: TotalRoundsForecastCapture,
        residual_selection: ResidualWeightSelection,
        *,
        decision_issued_at_utc: datetime | str,
        minimum_expected_return: object = TOTAL_MIN_EXPECTED_RETURN,
        maximum_source_quote_age_seconds: object = TOTAL_MAX_SOURCE_QUOTE_AGE_SECONDS,
    ) -> "TotalRoundsPaperDecision":
        if not isinstance(reference_quote, TotalRoundsQuoteSnapshot):
            raise TypeError("reference_quote must be a TotalRoundsQuoteSnapshot")
        if not isinstance(forecast, TotalRoundsForecastCapture):
            raise TypeError("forecast must be a TotalRoundsForecastCapture")
        if not isinstance(residual_selection, ResidualWeightSelection):
            raise TypeError("residual_selection must be a ResidualWeightSelection")
        consensus = tuple(consensus_quotes)
        if len(consensus) < TOTAL_MIN_CONSENSUS_BOOKS:
            raise MarketDataError(
                f"total decisions require {TOTAL_MIN_CONSENSUS_BOOKS} non-target books"
            )
        book_keys = tuple(sorted(item.source_book_key.casefold() for item in consensus))
        if len(set(book_keys)) != len(book_keys):
            raise StoreIntegrityError("total consensus contains duplicate books")
        target_key = reference_quote.source_book_key.casefold()
        if target_key in book_keys:
            raise StoreIntegrityError("target total book influenced its own consensus")
        identity = (
            reference_quote.capture_id,
            reference_quote.matchup_id,
            reference_quote.event_id,
            reference_quote.fighter_id,
            reference_quote.opponent_id,
            reference_quote.event_date,
            reference_quote.timing_precision,
            reference_quote.event_start_utc,
            reference_quote.observed_at_utc,
            reference_quote.line,
        )
        forecast_identity = (
            forecast.capture_id,
            forecast.matchup_id,
            forecast.event_id,
            forecast.fighter_id,
            forecast.opponent_id,
            forecast.event_date,
            forecast.timing_precision,
            forecast.event_start_utc,
            forecast.forecast_issued_at_utc,
            forecast.line,
        )
        if identity[:8] != forecast_identity[:8] or identity[9] != forecast_identity[9]:
            raise StoreIntegrityError("total decision quote and forecast identities differ")
        for item in consensus:
            item_identity = (
                item.capture_id,
                item.matchup_id,
                item.event_id,
                item.fighter_id,
                item.opponent_id,
                item.event_date,
                item.timing_precision,
                item.event_start_utc,
                item.observed_at_utc,
                item.line,
            )
            if item_identity != identity:
                raise StoreIntegrityError("total consensus quotes are not synchronized")
        issued, event_day, precision, event_start = require_before_event(
            decision_issued_at_utc,
            event_date=reference_quote.event_date,
            timing_precision=reference_quote.timing_precision,
            event_start_utc=reference_quote.event_start_utc,
            observed_field="decision_issued_at_utc",
        )
        observed = utc_datetime(reference_quote.observed_at_utc, "observed_at_utc")
        if issued != observed:
            raise MarketDataError("total paper decisions must freeze at retrieval time")
        if issued < utc_datetime(forecast.forecast_issued_at_utc, "forecast_issued_at_utc"):
            raise MarketDataError("total decision predates its forecast")
        max_age = _finite(
            maximum_source_quote_age_seconds, "maximum_source_quote_age_seconds"
        )
        if max_age < 0.0:
            raise MarketDataError("maximum source quote age cannot be negative")
        ages = [reference_quote.source_quote_age_seconds, *(item.source_quote_age_seconds for item in consensus)]
        if any(age < -300.0 or age > max_age for age in ages):
            raise MarketDataError("total decision contains a stale source quote")
        threshold = _finite(minimum_expected_return, "minimum_expected_return")
        if threshold < 0.0:
            raise MarketDataError("minimum expected return cannot be negative")
        market_over = probability(
            median(item.no_vig_over_probability for item in consensus),
            "market_over_probability",
        )
        model_over = probability(forecast.over_probability, "model_over_probability")
        weight = _finite(residual_selection.selected_weight, "selected_residual_weight")
        if weight not in RESIDUAL_WEIGHT_GRID:
            raise MarketDataError("selected residual weight is outside the locked grid")
        residual_over = symmetric_logit_blend(market_over, model_over, weight)
        over_line = moneyline(reference_quote.over_moneyline, "over_reference_moneyline")
        under_line = moneyline(reference_quote.under_moneyline, "under_reference_moneyline")
        over_break_even = implied_probability(over_line)
        under_break_even = implied_probability(under_line)
        model_over_ev = model_over / over_break_even - 1.0
        model_under_ev = (1.0 - model_over) / under_break_even - 1.0
        residual_over_ev = residual_over / over_break_even - 1.0
        residual_under_ev = (1.0 - residual_over) / under_break_even - 1.0
        if residual_over_ev >= threshold and residual_over_ev > residual_under_ev:
            action = "over"
            action_probability = residual_over
            action_line = over_line
        elif residual_under_ev >= threshold and residual_under_ev > residual_over_ev:
            action = "under"
            action_probability = 1.0 - residual_over
            action_line = under_line
        else:
            action = "pass"
            action_probability = None
            action_line = None
        known_fight_ids = {
            value for value in (reference_quote.fight_id, forecast.fight_id) if value
        }
        if len(known_fight_ids) > 1:
            raise StoreIntegrityError("total quote and forecast fight IDs differ")
        consensus_body = {
            "capture_id": reference_quote.capture_id,
            "matchup_id": reference_quote.matchup_id,
            "line": reference_quote.line,
            "as_of_utc": reference_quote.observed_at_utc,
            "quote_ids": sorted(item.quote_id for item in consensus),
            "no_vig_over_probabilities": sorted(
                item.no_vig_over_probability for item in consensus
            ),
        }
        body = {
            "schema_version": SCHEMA_VERSION,
            "betting_status": BETTING_STATUS,
            "paper_only": True,
            "policy_version": TOTAL_POLICY_VERSION,
            "capture_id": reference_quote.capture_id,
            "matchup_id": reference_quote.matchup_id,
            "fight_id": next(iter(known_fight_ids), None),
            "event_id": reference_quote.event_id,
            "fighter_id": reference_quote.fighter_id,
            "opponent_id": reference_quote.opponent_id,
            "fighter_name": reference_quote.fighter_name,
            "opponent_name": reference_quote.opponent_name,
            "event_date": event_day,
            "timing_precision": precision,
            "event_start_utc": event_start,
            "decision_issued_at_utc": utc_text(issued, "decision_issued_at_utc"),
            "market_as_of_utc": reference_quote.observed_at_utc,
            "reference_quote_id": reference_quote.quote_id,
            "forecast_capture_id": forecast.forecast_capture_id,
            "line": reference_quote.line,
            "target_book": reference_quote.book,
            "target_book_key": target_key,
            "target_source_quote_age_seconds": reference_quote.source_quote_age_seconds,
            "maximum_source_quote_age_seconds": max_age,
            "consensus_id": canonical_hash(consensus_body),
            "consensus_book_count": len(consensus),
            "consensus_book_keys": ";".join(book_keys),
            "market_over_probability": market_over,
            "model_over_probability": model_over,
            "residual_over_probability": residual_over,
            "selected_residual_weight": weight,
            "residual_selection_status": residual_selection.selection_status,
            "residual_training_scored_lines": residual_selection.scored_lines,
            "residual_training_settled_events": residual_selection.settled_events,
            "residual_training_dataset_sha256": residual_selection.training_dataset_sha256,
            "over_reference_moneyline": over_line,
            "under_reference_moneyline": under_line,
            "over_break_even_probability": over_break_even,
            "under_break_even_probability": under_break_even,
            "model_over_expected_return": model_over_ev,
            "model_under_expected_return": model_under_ev,
            "residual_over_expected_return": residual_over_ev,
            "residual_under_expected_return": residual_under_ev,
            "minimum_expected_return": threshold,
            "paper_action": action,
            "action_probability": action_probability,
            "action_reference_moneyline": action_line,
            "hypothetical_risk_units": 0.0 if action == "pass" else 1.0,
        }
        return cls(decision_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "TotalRoundsPaperDecision":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"total decision schema mismatch; missing={missing}, extra={extra}"
            )
        decision = cls(**{field: record[field] for field in cls.FIELDNAMES})
        decision.validate_integrity()
        return decision

    @property
    def natural_key(self) -> tuple[str, float]:
        return self.matchup_id, float(self.line)

    def validate_integrity(self) -> None:
        if int(self.schema_version) != SCHEMA_VERSION:
            raise MarketDataError("unsupported total decision schema version")
        if self.betting_status != BETTING_STATUS or self.paper_only is not True:
            raise StoreIntegrityError("total decision must remain paper-only")
        if self.policy_version != TOTAL_POLICY_VERSION:
            raise MarketDataError("unsupported total decision policy")
        body = self.to_mapping()
        body.pop("decision_id")
        if self.decision_id != canonical_hash(body):
            raise StoreIntegrityError("total decision ID disagrees with its contents")
        require_before_event(
            self.decision_issued_at_utc,
            event_date=self.event_date,
            timing_precision=self.timing_precision,
            event_start_utc=self.event_start_utc,
            observed_field="decision_issued_at_utc",
        )
        if self.decision_issued_at_utc != self.market_as_of_utc:
            raise StoreIntegrityError("total decision market cutoff differs from issue time")
        line = _line(self.line)
        del line
        market = probability(self.market_over_probability, "market_over_probability")
        model = probability(self.model_over_probability, "model_over_probability")
        residual = probability(
            self.residual_over_probability, "residual_over_probability"
        )
        weight = _finite(self.selected_residual_weight, "selected_residual_weight")
        if weight not in RESIDUAL_WEIGHT_GRID:
            raise MarketDataError("total residual weight is outside the locked grid")
        if abs(residual - symmetric_logit_blend(market, model, weight)) > 1e-12:
            raise StoreIntegrityError("total residual probability is not reproducible")
        if int(self.consensus_book_count) < TOTAL_MIN_CONSENSUS_BOOKS:
            raise StoreIntegrityError("total decision lacks enough consensus books")
        books = tuple(part for part in str(self.consensus_book_keys).split(";") if part)
        if len(books) != int(self.consensus_book_count) or tuple(sorted(set(books))) != books:
            raise StoreIntegrityError("total decision consensus books are invalid")
        if self.target_book_key.casefold() in books:
            raise StoreIntegrityError("total decision target book is in its consensus")
        validated_sha256(self.consensus_id, "consensus_id")
        max_age = _finite(
            self.maximum_source_quote_age_seconds, "maximum_source_quote_age_seconds"
        )
        age = _finite(
            self.target_source_quote_age_seconds, "target_source_quote_age_seconds"
        )
        if max_age < 0.0 or not -300.0 <= age <= max_age:
            raise StoreIntegrityError("total decision reference quote is stale")
        if int(self.residual_training_scored_lines) < 0 or int(
            self.residual_training_settled_events
        ) < 0:
            raise MarketDataError("total residual training counts cannot be negative")
        validated_sha256(
            self.residual_training_dataset_sha256,
            "residual_training_dataset_sha256",
        )
        over_line = moneyline(self.over_reference_moneyline, "over_reference_moneyline")
        under_line = moneyline(
            self.under_reference_moneyline, "under_reference_moneyline"
        )
        expected = (
            implied_probability(over_line),
            implied_probability(under_line),
            model / implied_probability(over_line) - 1.0,
            (1.0 - model) / implied_probability(under_line) - 1.0,
            residual / implied_probability(over_line) - 1.0,
            (1.0 - residual) / implied_probability(under_line) - 1.0,
        )
        supplied = (
            self.over_break_even_probability,
            self.under_break_even_probability,
            self.model_over_expected_return,
            self.model_under_expected_return,
            self.residual_over_expected_return,
            self.residual_under_expected_return,
        )
        if any(abs(float(left) - right) > 1e-12 for left, right in zip(supplied, expected)):
            raise StoreIntegrityError("total decision price calculations are not reproducible")
        threshold = _finite(self.minimum_expected_return, "minimum_expected_return")
        if threshold < 0.0:
            raise MarketDataError("total decision threshold cannot be negative")
        best_side = "over" if expected[4] > expected[5] else "under"
        best_ev = max(expected[4], expected[5])
        expected_action = best_side if best_ev >= threshold and expected[4] != expected[5] else "pass"
        if self.paper_action != expected_action:
            raise StoreIntegrityError("total paper action is not reproducible")
        if expected_action == "pass":
            if self.action_probability is not None or self.action_reference_moneyline is not None or float(self.hypothetical_risk_units) != 0.0:
                raise StoreIntegrityError("total pass carries a paper stake")
        else:
            expected_probability = residual if expected_action == "over" else 1.0 - residual
            expected_line = over_line if expected_action == "over" else under_line
            if (
                self.action_probability is None
                or abs(float(self.action_probability) - expected_probability) > 1e-12
                or int(self.action_reference_moneyline) != expected_line
                or float(self.hypothetical_risk_units) != 1.0
            ):
                raise StoreIntegrityError("total selection fields are inconsistent")

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


@dataclass(frozen=True)
class TotalRoundsPaperSettlement:
    schema_version: int
    settlement_id: str
    betting_status: str
    paper_only: bool
    decision_id: str
    capture_id: str
    matchup_id: str
    fight_id: str | None
    event_id: str
    settled_at_utc: str
    result_source_sha256: str
    total_fight_seconds: float | None
    target: int | None
    settlement_status: str
    hypothetical_risk_units: float
    hypothetical_profit_units: float
    market_log_loss: float | None
    model_log_loss: float | None
    residual_log_loss: float | None
    market_brier_score: float | None
    model_brier_score: float | None
    residual_brier_score: float | None

    FIELDNAMES: ClassVar[tuple[str, ...]] = tuple(__annotations__)

    @classmethod
    def from_mapping(
        cls, record: Mapping[str, object]
    ) -> "TotalRoundsPaperSettlement":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"total settlement schema mismatch; missing={missing}, extra={extra}"
            )
        settlement = cls(**{field: record[field] for field in cls.FIELDNAMES})
        settlement.validate_integrity()
        return settlement

    @property
    def natural_key(self) -> tuple[str]:
        return (self.decision_id,)

    def validate_integrity(self) -> None:
        if int(self.schema_version) != SCHEMA_VERSION:
            raise MarketDataError("unsupported total settlement schema")
        if self.betting_status != BETTING_STATUS or self.paper_only is not True:
            raise StoreIntegrityError("total settlement must remain paper-only")
        body = self.to_mapping()
        body.pop("settlement_id")
        if self.settlement_id != canonical_hash(body):
            raise StoreIntegrityError("total settlement ID disagrees with contents")
        utc_datetime(self.settled_at_utc, "settled_at_utc")
        validated_sha256(self.result_source_sha256, "result_source_sha256")
        if self.total_fight_seconds is not None:
            duration = _finite(self.total_fight_seconds, "total_fight_seconds")
            if duration <= 0.0 or duration > 25.0 * 300.0:
                raise MarketDataError("total fight duration is outside the supported range")
        if self.target is not None:
            binary_target(self.target)
        allowed = {"paper_win", "paper_loss", "pass", "void", "pass_unscored"}
        if self.settlement_status not in allowed:
            raise MarketDataError("unsupported total settlement status")
        scores = (
            self.market_log_loss,
            self.model_log_loss,
            self.residual_log_loss,
            self.market_brier_score,
            self.model_brier_score,
            self.residual_brier_score,
        )
        if self.target is None:
            if any(value is not None for value in scores):
                raise StoreIntegrityError("unscored total settlement contains scores")
            if self.settlement_status not in {"void", "pass_unscored"}:
                raise StoreIntegrityError("unscored total settlement has a scored status")
        else:
            if any(value is None or not math.isfinite(float(value)) for value in scores):
                raise StoreIntegrityError("scored total settlement lacks finite scores")
            if self.settlement_status in {"void", "pass_unscored"}:
                raise StoreIntegrityError("scored total settlement has an unscored status")
        risk = _finite(self.hypothetical_risk_units, "hypothetical_risk_units")
        profit = _finite(self.hypothetical_profit_units, "hypothetical_profit_units")
        if self.settlement_status in {"paper_win", "paper_loss"}:
            if risk != 1.0:
                raise StoreIntegrityError("total selection must risk one paper unit")
            if self.settlement_status == "paper_loss" and profit != -1.0:
                raise StoreIntegrityError("total paper loss must lose one unit")
            if self.settlement_status == "paper_win" and profit <= 0.0:
                raise StoreIntegrityError("total paper win must have positive profit")
        elif risk != 0.0 or profit != 0.0:
            raise StoreIntegrityError("total pass/void cannot carry paper profit")

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


class TotalRoundsPaperDecisionStore(_PaperRecordStore):
    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        super().__init__(
            csv_path,
            jsonl_path,
            record_type=TotalRoundsPaperDecision,
            id_field="decision_id",
            time_field="decision_issued_at_utc",
        )

    def read(self) -> tuple[TotalRoundsPaperDecision, ...]:
        return tuple(super().read())  # type: ignore[return-value]

    def append(self, records: Iterable[TotalRoundsPaperDecision]) -> AppendResult:
        return super().append(records)


class TotalRoundsPaperSettlementStore(_PaperRecordStore):
    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        super().__init__(
            csv_path,
            jsonl_path,
            record_type=TotalRoundsPaperSettlement,
            id_field="settlement_id",
            time_field="settled_at_utc",
        )

    def read(self) -> tuple[TotalRoundsPaperSettlement, ...]:
        return tuple(super().read())  # type: ignore[return-value]

    def append(self, records: Iterable[TotalRoundsPaperSettlement]) -> AppendResult:
        return super().append(records)


def settle_total_round_decision(
    decision: TotalRoundsPaperDecision,
    *,
    total_fight_seconds: object | None,
    settled_at_utc: datetime | str,
    result_source_sha256: object,
    fight_id: object | None = None,
) -> TotalRoundsPaperSettlement:
    if not isinstance(decision, TotalRoundsPaperDecision):
        raise TypeError("decision must be a TotalRoundsPaperDecision")
    settled = utc_datetime(settled_at_utc, "settled_at_utc")
    if decision.timing_precision == "timestamp":
        if settled < utc_datetime(decision.event_start_utc, "event_start_utc"):
            raise MarketDataError("total settlement precedes the event")
    elif settled.date() <= date.fromisoformat(decision.event_date):
        raise MarketDataError("date-only total settlement must occur after the event date")
    resolved_fight_id = optional_stable_id(fight_id, "fight_id") or decision.fight_id
    if decision.fight_id and resolved_fight_id != decision.fight_id:
        raise StoreIntegrityError("total settlement fight ID conflicts with decision")
    duration = None
    target = None
    if total_fight_seconds is not None:
        duration = _finite(total_fight_seconds, "total_fight_seconds")
        if duration <= 0.0 or duration > 25.0 * 300.0:
            raise MarketDataError("total fight duration is outside the supported range")
        boundary = float(decision.line) * 300.0
        if abs(duration - boundary) > 1e-9:
            target = int(duration > boundary)
    if target is None:
        status = "void" if decision.paper_action != "pass" else "pass_unscored"
        risk = profit = 0.0
        market_loss = model_loss = residual_loss = None
        market_brier = model_brier = residual_brier = None
    else:
        probabilities = (
            float(decision.market_over_probability),
            float(decision.model_over_probability),
            float(decision.residual_over_probability),
        )
        market_loss, model_loss, residual_loss = (
            _log_loss(target, value) for value in probabilities
        )
        market_brier, model_brier, residual_brier = (
            (value - target) ** 2 for value in probabilities
        )
        if decision.paper_action == "pass":
            status = "pass"
            risk = profit = 0.0
        else:
            selected_target = 1 if decision.paper_action == "over" else 0
            won = selected_target == target
            status = "paper_win" if won else "paper_loss"
            risk = 1.0
            profit = (
                _profit_for_one_unit_risk(int(decision.action_reference_moneyline))
                if won
                else -1.0
            )
    body = {
        "schema_version": SCHEMA_VERSION,
        "betting_status": BETTING_STATUS,
        "paper_only": True,
        "decision_id": decision.decision_id,
        "capture_id": decision.capture_id,
        "matchup_id": decision.matchup_id,
        "fight_id": resolved_fight_id,
        "event_id": decision.event_id,
        "settled_at_utc": utc_text(settled, "settled_at_utc"),
        "result_source_sha256": validated_sha256(
            result_source_sha256, "result_source_sha256"
        ),
        "total_fight_seconds": duration,
        "target": target,
        "settlement_status": status,
        "hypothetical_risk_units": risk,
        "hypothetical_profit_units": profit,
        "market_log_loss": market_loss,
        "model_log_loss": model_loss,
        "residual_log_loss": residual_loss,
        "market_brier_score": market_brier,
        "model_brier_score": model_brier,
        "residual_brier_score": residual_brier,
    }
    return TotalRoundsPaperSettlement(settlement_id=canonical_hash(body), **body)


def _paired_records(
    decisions: Iterable[TotalRoundsPaperDecision],
    settlements: Iterable[TotalRoundsPaperSettlement],
) -> tuple[tuple[TotalRoundsPaperDecision, TotalRoundsPaperSettlement], ...]:
    decision_by_id: dict[str, TotalRoundsPaperDecision] = {}
    for decision in decisions:
        decision.validate_integrity()
        if decision.decision_id in decision_by_id:
            raise StoreIntegrityError("duplicate total decision")
        decision_by_id[decision.decision_id] = decision
    paired: list[tuple[TotalRoundsPaperDecision, TotalRoundsPaperSettlement]] = []
    seen: set[str] = set()
    for settlement in settlements:
        settlement.validate_integrity()
        if settlement.decision_id in seen:
            raise StoreIntegrityError("one total decision has multiple settlements")
        decision = decision_by_id.get(settlement.decision_id)
        if decision is None:
            raise StoreIntegrityError("total settlement has no matching decision")
        expected = settle_total_round_decision(
            decision,
            total_fight_seconds=settlement.total_fight_seconds,
            settled_at_utc=settlement.settled_at_utc,
            result_source_sha256=settlement.result_source_sha256,
            fight_id=settlement.fight_id,
        )
        if expected != settlement:
            raise StoreIntegrityError("total settlement is not reproducible")
        seen.add(settlement.decision_id)
        paired.append((decision, settlement))
    paired.sort(key=lambda pair: (pair[0].decision_issued_at_utc, pair[0].decision_id))
    return tuple(paired)


def select_residual_weight(
    decisions: Iterable[TotalRoundsPaperDecision],
    settlements: Iterable[TotalRoundsPaperSettlement],
) -> ResidualWeightSelection:
    paired = tuple(
        pair
        for pair in _paired_records(decisions, settlements)
        if pair[1].target is not None
    )
    event_count = len({decision.event_id for decision, _ in paired})
    fingerprint = canonical_hash(
        [
            {"decision_id": decision.decision_id, "settlement_id": settlement.settlement_id}
            for decision, settlement in paired
        ]
    )
    if not paired:
        return ResidualWeightSelection(
            0.0, 0, 0, None, None, None, None,
            "market_only_insufficient_history", fingerprint,
        )
    all_losses: dict[float, float] = {}
    for weight in RESIDUAL_WEIGHT_GRID:
        values: list[float] = []
        for decision, settlement in paired:
            target = int(settlement.target)
            candidate = symmetric_logit_blend(
                float(decision.market_over_probability),
                float(decision.model_over_probability),
                weight,
            )
            values.append(_log_loss(target, candidate))
        all_losses[weight] = sum(values) / len(values)
    market_loss = all_losses[0.0]
    if len(paired) < RESIDUAL_MIN_SCORED_LINES or event_count < max(
        2, RESIDUAL_MIN_SETTLED_EVENTS
    ):
        return ResidualWeightSelection(
            0.0, len(paired), event_count, market_loss, market_loss, 0.0, None,
            "market_only_insufficient_history", fingerprint,
        )

    # Choose the scalar correction on earlier cards, then require it to beat
    # market-only on later untouched cards. This avoids selecting and grading
    # the residual weight on the same fights.
    event_order = sorted(
        {
            (decision.event_date, decision.event_id)
            for decision, _ in paired
        }
    )
    development_count = min(
        len(event_order) - 1,
        max(1, int(len(event_order) * 0.70)),
    )
    development_events = {event_id for _, event_id in event_order[:development_count]}
    development = tuple(
        pair for pair in paired if pair[0].event_id in development_events
    )
    evaluation = tuple(
        pair for pair in paired if pair[0].event_id not in development_events
    )
    development_losses: dict[float, float] = {}
    for weight in RESIDUAL_WEIGHT_GRID:
        values = [
            _log_loss(
                int(settlement.target),
                symmetric_logit_blend(
                    float(decision.market_over_probability),
                    float(decision.model_over_probability),
                    weight,
                ),
            )
            for decision, settlement in development
        ]
        development_losses[weight] = sum(values) / len(values)
    best_weight = min(
        RESIDUAL_WEIGHT_GRID,
        key=lambda value: (development_losses[value], value),
    )
    evaluation_differences: list[tuple[str, float]] = []
    evaluation_market_losses: list[float] = []
    evaluation_candidate_losses: list[float] = []
    for decision, settlement in evaluation:
        target = int(settlement.target)
        evaluation_market_loss = _log_loss(
            target, float(decision.market_over_probability)
        )
        candidate = symmetric_logit_blend(
            float(decision.market_over_probability),
            float(decision.model_over_probability),
            best_weight,
        )
        candidate_loss = _log_loss(target, candidate)
        evaluation_market_losses.append(evaluation_market_loss)
        evaluation_candidate_losses.append(candidate_loss)
        evaluation_differences.append(
            (decision.event_id, candidate_loss - evaluation_market_loss)
        )
    market_loss = sum(evaluation_market_losses) / len(evaluation_market_losses)
    selected_loss = sum(evaluation_candidate_losses) / len(
        evaluation_candidate_losses
    )
    best_difference = selected_loss - market_loss
    grouped: dict[str, list[float]] = defaultdict(list)
    for event_id, difference in evaluation_differences:
        grouped[event_id].append(difference)
    blocks = [grouped[key] for key in sorted(grouped)]
    seed = canonical_hash({"total_residual_blocks": blocks, "weight": best_weight})
    generator = random.Random(int(seed[:16], 16))
    bootstrap: list[float] = []
    for _ in range(RESIDUAL_BOOTSTRAP_SAMPLES):
        selected = [generator.choice(blocks) for _ in blocks]
        values = [value for block in selected for value in block]
        bootstrap.append(sum(values) / len(values))
    upper = _quantile(bootstrap, 0.975)
    promoted = (
        best_weight > 0.0
        and best_difference <= -RESIDUAL_MIN_LOG_LOSS_GAIN
        and upper < 0.0
    )
    selected_weight = best_weight if promoted else 0.0
    return ResidualWeightSelection(
        selected_weight=selected_weight,
        scored_lines=len(paired),
        settled_events=event_count,
        market_log_loss=market_loss,
        selected_log_loss=selected_loss if promoted else market_loss,
        selected_minus_market_log_loss=best_difference if promoted else 0.0,
        ci_95_upper=upper,
        selection_status=(
            "residual_weight_promoted" if promoted else "market_only_no_credible_gain"
        ),
        training_dataset_sha256=fingerprint,
    )


@dataclass(frozen=True)
class TotalRoundsDecisionBuild:
    decisions: tuple[TotalRoundsPaperDecision, ...]
    eligible_horizon: bool
    lead_time_seconds: float | None
    markets_considered: int
    markets_already_frozen: int
    markets_without_fresh_quotes: int
    markets_without_forecast: int
    residual_selection: ResidualWeightSelection

    def to_mapping(self) -> dict[str, object]:
        return {
            "eligible_horizon": self.eligible_horizon,
            "lead_time_seconds": self.lead_time_seconds,
            "markets_considered": self.markets_considered,
            "markets_already_frozen": self.markets_already_frozen,
            "markets_without_fresh_quotes": self.markets_without_fresh_quotes,
            "markets_without_forecast": self.markets_without_forecast,
            "residual_selection": self.residual_selection.to_mapping(),
        }


def build_locked_total_round_decisions(
    quotes: Iterable[TotalRoundsQuoteSnapshot],
    forecasts: Iterable[TotalRoundsForecastCapture],
    existing_decisions: Iterable[TotalRoundsPaperDecision] = (),
    existing_settlements: Iterable[TotalRoundsPaperSettlement] = (),
) -> TotalRoundsDecisionBuild:
    quote_records = tuple(quotes)
    forecast_records = tuple(forecasts)
    existing = tuple(existing_decisions)
    residual_selection = select_residual_weight(existing, existing_settlements)
    if not quote_records:
        return TotalRoundsDecisionBuild(
            (), False, None, 0, 0, 0, 0, residual_selection
        )
    capture_ids = {item.capture_id for item in quote_records}
    if len(capture_ids) != 1:
        raise StoreIntegrityError("total decision input must contain one capture")
    timing = {
        (item.timing_precision, item.event_start_utc, item.observed_at_utc)
        for item in quote_records
    }
    if len(timing) != 1:
        raise StoreIntegrityError("total decision quotes have conflicting timing")
    precision, event_start, observed = next(iter(timing))
    lead = None
    if precision == "timestamp" and event_start:
        lead = (
            utc_datetime(event_start, "event_start_utc")
            - utc_datetime(observed, "observed_at_utc")
        ).total_seconds()
    grouped: dict[tuple[str, float], list[TotalRoundsQuoteSnapshot]] = defaultdict(list)
    for quote in quote_records:
        grouped[(quote.matchup_id, quote.line)].append(quote)
    eligible_horizon = lead is not None and abs(
        lead - TOTAL_DECISION_TARGET_LEAD_SECONDS
    ) <= TOTAL_DECISION_WINDOW_SECONDS
    if not eligible_horizon:
        return TotalRoundsDecisionBuild(
            (), False, lead, len(grouped), 0, 0, 0, residual_selection
        )
    capture_id = next(iter(capture_ids))
    forecast_index: dict[tuple[str, float], TotalRoundsForecastCapture] = {}
    for forecast in forecast_records:
        if forecast.capture_id != capture_id:
            continue
        key = forecast.matchup_id, forecast.line
        if key in forecast_index:
            raise StoreIntegrityError("duplicate total forecast for capture/line")
        forecast_index[key] = forecast
    frozen = {item.natural_key for item in existing}
    decisions: list[TotalRoundsPaperDecision] = []
    already = no_fresh = no_forecast = 0
    for key in sorted(grouped):
        if key in frozen:
            already += 1
            continue
        forecast = forecast_index.get(key)
        if forecast is None:
            no_forecast += 1
            continue
        fresh = tuple(
            item
            for item in grouped[key]
            if -300.0
            <= item.source_quote_age_seconds
            <= TOTAL_MAX_SOURCE_QUOTE_AGE_SECONDS
        )
        if len({item.source_book_key.casefold() for item in fresh}) < TOTAL_MIN_CONSENSUS_BOOKS + 1:
            no_fresh += 1
            continue
        candidates: list[TotalRoundsPaperDecision] = []
        for target in sorted(fresh, key=lambda item: (item.book.casefold(), item.quote_id)):
            other = tuple(
                item
                for item in fresh
                if item.source_book_key.casefold() != target.source_book_key.casefold()
            )
            if len(other) < TOTAL_MIN_CONSENSUS_BOOKS:
                continue
            candidates.append(
                TotalRoundsPaperDecision.create(
                    target,
                    other,
                    forecast,
                    residual_selection,
                    decision_issued_at_utc=target.observed_at_utc,
                )
            )
        if not candidates:
            no_fresh += 1
            continue
        decisions.append(
            min(
                candidates,
                key=lambda item: (
                    -max(
                        item.residual_over_expected_return,
                        item.residual_under_expected_return,
                    ),
                    item.target_book.casefold(),
                    item.decision_id,
                ),
            )
        )
    return TotalRoundsDecisionBuild(
        tuple(decisions),
        True,
        lead,
        len(grouped),
        already,
        no_fresh,
        no_forecast,
        residual_selection,
    )


def _forecast_summary(
    pairs: tuple[tuple[TotalRoundsPaperDecision, TotalRoundsPaperSettlement], ...]
) -> dict[str, object]:
    scored = tuple(pair for pair in pairs if pair[1].target is not None)
    result: dict[str, object] = {"count": len(scored)}
    for label, field in (
        ("market", "market_over_probability"),
        ("model", "model_over_probability"),
        ("residual", "residual_over_probability"),
    ):
        probabilities = [float(getattr(decision, field)) for decision, _ in scored]
        targets = [int(settlement.target) for _, settlement in scored]
        if probabilities:
            losses = [_log_loss(target, value) for target, value in zip(targets, probabilities)]
            briers = [(value - target) ** 2 for target, value in zip(targets, probabilities)]
            ece = 0.0
            for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
                indexes = [
                    index
                    for index, value in enumerate(probabilities)
                    if lower <= value < lower + 0.2 or (lower == 0.8 and value == 1.0)
                ]
                if indexes:
                    ece += len(indexes) / len(probabilities) * abs(
                        sum(probabilities[index] for index in indexes) / len(indexes)
                        - sum(targets[index] for index in indexes) / len(indexes)
                    )
            result[label] = {
                "log_loss": sum(losses) / len(losses),
                "brier_score": sum(briers) / len(briers),
                "ece_5_bin": ece,
            }
        else:
            result[label] = {
                "log_loss": None,
                "brier_score": None,
                "ece_5_bin": None,
            }
    market_loss = result["market"]["log_loss"]
    result["model_minus_market_log_loss"] = (
        None if market_loss is None else result["model"]["log_loss"] - market_loss
    )
    result["residual_minus_market_log_loss"] = (
        None if market_loss is None else result["residual"]["log_loss"] - market_loss
    )
    return result


def _strategy_summary(
    pairs: tuple[tuple[TotalRoundsPaperDecision, TotalRoundsPaperSettlement], ...],
    probability_field: str,
    threshold: float,
) -> dict[str, object]:
    selections = wins = losses = 0
    profit = risk = equity = peak = max_drawdown = 0.0
    for decision, settlement in pairs:
        if settlement.target is None:
            continue
        predicted_over = float(getattr(decision, probability_field))
        over_ev = predicted_over / float(decision.over_break_even_probability) - 1.0
        under_ev = (1.0 - predicted_over) / float(decision.under_break_even_probability) - 1.0
        best_ev = max(over_ev, under_ev)
        if best_ev < threshold or over_ev == under_ev:
            continue
        selected_target = 1 if over_ev > under_ev else 0
        line = (
            int(decision.over_reference_moneyline)
            if selected_target == 1
            else int(decision.under_reference_moneyline)
        )
        won = selected_target == int(settlement.target)
        outcome_profit = _profit_for_one_unit_risk(line) if won else -1.0
        selections += 1
        wins += int(won)
        losses += int(not won)
        risk += 1.0
        profit += outcome_profit
        equity += outcome_profit
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "threshold": threshold,
        "selections": selections,
        "wins": wins,
        "losses": losses,
        "hypothetical_risk_units": risk,
        "hypothetical_profit_units": profit,
        "hypothetical_roi": profit / risk if risk else None,
        "hypothetical_max_drawdown_units": max_drawdown,
    }


def summarize_total_round_performance(
    decisions: Iterable[TotalRoundsPaperDecision],
    settlements: Iterable[TotalRoundsPaperSettlement],
) -> dict[str, object]:
    decision_records = tuple(decisions)
    settlement_records = tuple(settlements)
    pairs = _paired_records(decision_records, settlement_records)
    strategies = {
        label: {
            f"{threshold:.3f}": _strategy_summary(pairs, field, threshold)
            for threshold in TOTAL_SHADOW_THRESHOLDS
        }
        for label, field in (
            ("independent_model", "model_over_probability"),
            ("market_residual", "residual_over_probability"),
        )
    }
    official = strategies["market_residual"][f"{TOTAL_MIN_EXPECTED_RETURN:.3f}"]
    return {
        "policy_version": TOTAL_POLICY_VERSION,
        "betting_status": BETTING_STATUS,
        "paper_only": True,
        "execution_enabled": False,
        "residual_selection_protocol": (
            "select weight on earliest 70% of settled events; require at least "
            "0.002 lower log loss and an event-block 95% upper confidence bound "
            "below zero on the latest 30%"
        ),
        "decisions": len(decision_records),
        "settlements": len(settlement_records),
        "unsettled_decisions": len(decision_records) - len(settlement_records),
        "scored_forecasts": sum(settlement.target is not None for settlement in settlement_records),
        "voids": sum(settlement.target is None for settlement in settlement_records),
        "official_strategy": {
            "probability_source": "market_residual",
            **official,
        },
        "forecast_comparators": _forecast_summary(pairs),
        "shadow_threshold_strategies": strategies,
        "next_residual_weight_selection": select_residual_weight(
            decision_records, settlement_records
        ).to_mapping(),
    }


__all__ = (
    "RESIDUAL_MIN_LOG_LOSS_GAIN",
    "RESIDUAL_MIN_SCORED_LINES",
    "RESIDUAL_MIN_SETTLED_EVENTS",
    "RESIDUAL_WEIGHT_GRID",
    "TOTAL_DECISION_TARGET_LEAD_SECONDS",
    "TOTAL_DECISION_WINDOW_SECONDS",
    "TOTAL_MAX_SOURCE_QUOTE_AGE_SECONDS",
    "TOTAL_MIN_CONSENSUS_BOOKS",
    "TOTAL_MIN_EXPECTED_RETURN",
    "TOTAL_POLICY_VERSION",
    "TOTAL_SHADOW_THRESHOLDS",
    "ResidualWeightSelection",
    "TotalRoundsDecisionBuild",
    "TotalRoundsPaperDecision",
    "TotalRoundsPaperDecisionStore",
    "TotalRoundsPaperSettlement",
    "TotalRoundsPaperSettlementStore",
    "build_locked_total_round_decisions",
    "select_residual_weight",
    "settle_total_round_decision",
    "summarize_total_round_performance",
)
