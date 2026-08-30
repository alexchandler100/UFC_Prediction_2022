"""Frozen prospective paper test for the historical market-first candidate.

The policy has no execution interface.  It records one immutable hypothetical
decision per matchup near T-24, then scores it after UFCStats publishes the
result.  Every target sportsbook is excluded from the fair-market consensus
used to evaluate that sportsbook's own price.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path
import random
from typing import Any, ClassVar, Iterable, Mapping

from ._common import (
    BETTING_STATUS,
    MarketDataError,
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
from .forecasts import ForecastCapture
from .paper import (
    PaperSettlement,
    PaperSettlementStore,
    _PaperRecordStore,
    _profit_for_one_unit_risk,
)
from .quotes import MarketConsensus, QuoteSnapshot, consensus_as_of
from .source_metadata import QuoteSourceMetadata
from .blend import forecast_metrics


POLICY_SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "content"
    / "data"
    / "model_research"
    / "market_first_t24_policy.json"
)


def _logit(value: float) -> float:
    bounded = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(bounded / (1.0 - bounded))


def _sigmoid(value: float) -> float:
    bounded = min(max(float(value), -40.0), 40.0)
    return 1.0 / (1.0 + math.exp(-bounded))


@dataclass(frozen=True)
class FrozenMarketFirstPolicy:
    """Validated, immutable coefficients and prospective collection rules."""

    policy_version: str
    artifact_sha256: str
    feature_names: tuple[str, ...]
    feature_scales: tuple[float, ...]
    feature_coefficients: tuple[float, ...]
    minimum_expected_return: float
    minimum_other_books: int
    prospective_first_capture_utc: str
    target_lead_seconds: float
    window_seconds: float
    maximum_source_quote_age_seconds: float
    decision_maximum_latency_seconds: float
    artifact: Mapping[str, object]

    @classmethod
    def load(cls, path: str | Path) -> "FrozenMarketFirstPolicy":
        artifact_path = Path(path)
        value = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise MarketDataError("market-first policy artifact is not an object")
        required = {
            "schema_version",
            "policy_version",
            "paper_only",
            "candidate_only",
            "execution_enabled",
            "features",
            "minimum_expected_return",
            "minimum_other_books",
            "prospective_capture",
        }
        missing = sorted(required - set(value))
        if missing:
            raise MarketDataError(f"market-first policy is missing fields: {missing}")
        if (
            value["schema_version"] != POLICY_SCHEMA_VERSION
            or value["paper_only"] is not True
            or value["candidate_only"] is not True
            or value["execution_enabled"] is not False
        ):
            raise StoreIntegrityError(
                "market-first policy must remain a version-one paper candidate"
            )
        raw_features = value["features"]
        if not isinstance(raw_features, list):
            raise MarketDataError("market-first policy features must be a list")
        names: list[str] = []
        scales: list[float] = []
        coefficients: list[float] = []
        for feature in raw_features:
            if not isinstance(feature, dict):
                raise MarketDataError("market-first feature is not an object")
            names.append(str(feature.get("name", "")).strip())
            scales.append(float(feature.get("scale")))
            coefficients.append(float(feature.get("coefficient")))
        expected_names = (
            "model_disagreement",
            "book_disagreement_market_strength",
        )
        if tuple(names) != expected_names:
            raise MarketDataError(
                "market-first policy has an unsupported feature contract"
            )
        capture = value["prospective_capture"]
        if not isinstance(capture, dict):
            raise MarketDataError("prospective capture policy is not an object")
        numeric = (
            *scales,
            *coefficients,
            float(value["minimum_expected_return"]),
            float(capture["target_lead_seconds"]),
            float(capture["window_seconds"]),
            float(capture["maximum_source_quote_age_seconds"]),
            float(capture["decision_maximum_latency_seconds"]),
        )
        if any(not math.isfinite(item) for item in numeric):
            raise MarketDataError("market-first policy contains non-finite values")
        if any(item <= 0.0 for item in scales):
            raise MarketDataError("market-first feature scales must be positive")
        minimum_ev = float(value["minimum_expected_return"])
        minimum_books = int(value["minimum_other_books"])
        if minimum_ev < 0.0 or minimum_books < 2:
            raise MarketDataError("market-first thresholds are invalid")
        timing = (
            float(capture["target_lead_seconds"]),
            float(capture["window_seconds"]),
            float(capture["maximum_source_quote_age_seconds"]),
            float(capture["decision_maximum_latency_seconds"]),
        )
        if any(item <= 0.0 for item in timing):
            raise MarketDataError("market-first timing values must be positive")
        prospective_first_capture = utc_text(
            capture["prospective_first_capture_utc"],
            "prospective_first_capture_utc",
        )
        policy_version = str(value["policy_version"]).strip()
        if not policy_version:
            raise MarketDataError("market-first policy_version is blank")
        return cls(
            policy_version=policy_version,
            artifact_sha256=canonical_hash(value),
            feature_names=tuple(names),
            feature_scales=tuple(scales),
            feature_coefficients=tuple(coefficients),
            minimum_expected_return=minimum_ev,
            minimum_other_books=minimum_books,
            prospective_first_capture_utc=prospective_first_capture,
            target_lead_seconds=timing[0],
            window_seconds=timing[1],
            maximum_source_quote_age_seconds=timing[2],
            decision_maximum_latency_seconds=timing[3],
            artifact=value,
        )

    def probability(
        self,
        *,
        market_probability: float,
        model_probability: float,
        book_probability_range: float,
    ) -> float:
        market = probability(market_probability, "market_probability")
        model = probability(model_probability, "model_probability")
        spread = float(book_probability_range)
        if not math.isfinite(spread) or not 0.0 <= spread <= 1.0:
            raise MarketDataError("book_probability_range is outside [0, 1]")
        feature_values = (
            _logit(model) - _logit(market),
            _logit(market) * spread,
        )
        adjustment = sum(
            coefficient * feature / scale
            for feature, scale, coefficient in zip(
                feature_values,
                self.feature_scales,
                self.feature_coefficients,
                strict=True,
            )
        )
        return _sigmoid(_logit(market) + adjustment)


@dataclass(frozen=True)
class MarketFirstPaperDecision:
    """One immutable prospective probability and hypothetical action."""

    schema_version: int
    decision_id: str
    betting_status: str
    paper_only: bool
    candidate_only: bool
    execution_enabled: bool
    policy_version: str
    policy_artifact_sha256: str
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
    reference_book: str
    market_consensus_id: str
    forecast_capture_id: str
    model_id: str
    other_book_count: int
    market_probability: float
    model_probability: float
    book_probability_range: float
    candidate_probability: float
    model_disagreement_scale: float
    model_disagreement_coefficient: float
    book_disagreement_scale: float
    book_disagreement_coefficient: float
    minimum_expected_return: float
    fighter_reference_moneyline: int
    opponent_reference_moneyline: int
    fighter_expected_return: float
    opponent_expected_return: float
    paper_action: str
    action_probability: float | None
    action_reference_moneyline: int | None
    hypothetical_risk_units: float

    FIELDNAMES: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "decision_id",
        "betting_status",
        "paper_only",
        "candidate_only",
        "execution_enabled",
        "policy_version",
        "policy_artifact_sha256",
        "capture_id",
        "matchup_id",
        "fight_id",
        "event_id",
        "fighter_id",
        "opponent_id",
        "fighter_name",
        "opponent_name",
        "event_date",
        "timing_precision",
        "event_start_utc",
        "decision_issued_at_utc",
        "market_as_of_utc",
        "reference_quote_id",
        "reference_book",
        "market_consensus_id",
        "forecast_capture_id",
        "model_id",
        "other_book_count",
        "market_probability",
        "model_probability",
        "book_probability_range",
        "candidate_probability",
        "model_disagreement_scale",
        "model_disagreement_coefficient",
        "book_disagreement_scale",
        "book_disagreement_coefficient",
        "minimum_expected_return",
        "fighter_reference_moneyline",
        "opponent_reference_moneyline",
        "fighter_expected_return",
        "opponent_expected_return",
        "paper_action",
        "action_probability",
        "action_reference_moneyline",
        "hypothetical_risk_units",
    )

    @classmethod
    def create(
        cls,
        market: MarketConsensus,
        reference_quote: QuoteSnapshot,
        forecast: ForecastCapture,
        *,
        policy: FrozenMarketFirstPolicy,
        book_probability_range: float,
        decision_issued_at_utc: datetime | str,
    ) -> "MarketFirstPaperDecision":
        if not isinstance(market, MarketConsensus):
            raise TypeError("market must be a MarketConsensus")
        if not isinstance(reference_quote, QuoteSnapshot):
            raise TypeError("reference_quote must be a QuoteSnapshot")
        if not isinstance(forecast, ForecastCapture):
            raise TypeError("forecast must be a ForecastCapture")
        identity = (
            market.capture_id,
            market.matchup_id,
            market.event_id,
            market.fighter_id,
            market.opponent_id,
            market.event_date,
        )
        if identity != (
            reference_quote.capture_id,
            reference_quote.matchup_id,
            reference_quote.event_id,
            reference_quote.fighter_id,
            reference_quote.opponent_id,
            reference_quote.event_date,
        ) or identity != (
            forecast.capture_id,
            forecast.matchup_id,
            forecast.event_id,
            forecast.fighter_id,
            forecast.opponent_id,
            forecast.event_date,
        ):
            raise StoreIntegrityError(
                "market-first inputs identify different matchups"
            )
        target_key = reference_quote.book.casefold()
        if (
            reference_quote.quote_id in market.quote_ids
            or target_key in market.included_book_keys
            or market.excluded_book_keys != (target_key,)
        ):
            raise StoreIntegrityError(
                "reference sportsbook influenced its own fair-market estimate"
            )
        if market.book_count < policy.minimum_other_books:
            raise MarketDataError("market-first decision has too few other books")
        if reference_quote.observed_at_utc != market.latest_observed_at_utc:
            raise StoreIntegrityError(
                "reference price and other books are not from one retrieval"
            )
        issued, _, _, _ = require_before_event(
            decision_issued_at_utc,
            event_date=market.event_date,
            timing_precision=market.timing_precision,
            event_start_utc=market.event_start_utc,
            observed_field="decision_issued_at_utc",
        )
        as_of = utc_datetime(market.as_of_utc, "market.as_of_utc")
        latency = (issued - as_of).total_seconds()
        if latency < 0.0 or latency > policy.decision_maximum_latency_seconds:
            raise MarketDataError("market-first decision used an expired retrieval")
        if issued < utc_datetime(
            forecast.forecast_issued_at_utc, "forecast.forecast_issued_at_utc"
        ):
            raise MarketDataError("market-first decision predates its model forecast")
        candidate = policy.probability(
            market_probability=market.no_vig_fighter_probability,
            model_probability=forecast.model_probability,
            book_probability_range=book_probability_range,
        )
        fighter_line = moneyline(
            reference_quote.fighter_moneyline, "fighter_reference_moneyline"
        )
        opponent_line = moneyline(
            reference_quote.opponent_moneyline, "opponent_reference_moneyline"
        )
        fighter_ev = candidate * (1.0 + _profit_for_one_unit_risk(fighter_line)) - 1.0
        opponent_ev = (
            (1.0 - candidate)
            * (1.0 + _profit_for_one_unit_risk(opponent_line))
            - 1.0
        )
        if (
            fighter_ev >= policy.minimum_expected_return
            and fighter_ev > opponent_ev
        ):
            action = "fighter"
            action_probability = candidate
            action_line = fighter_line
        elif (
            opponent_ev >= policy.minimum_expected_return
            and opponent_ev > fighter_ev
        ):
            action = "opponent"
            action_probability = 1.0 - candidate
            action_line = opponent_line
        else:
            action = "pass"
            action_probability = None
            action_line = None
        fight_ids = {
            value
            for value in (
                market.fight_id,
                reference_quote.fight_id,
                forecast.fight_id,
            )
            if value is not None
        }
        if len(fight_ids) > 1:
            raise StoreIntegrityError("market-first inputs have conflicting fight IDs")
        body: dict[str, object] = {
            "schema_version": POLICY_SCHEMA_VERSION,
            "betting_status": BETTING_STATUS,
            "paper_only": True,
            "candidate_only": True,
            "execution_enabled": False,
            "policy_version": policy.policy_version,
            "policy_artifact_sha256": policy.artifact_sha256,
            "capture_id": market.capture_id,
            "matchup_id": market.matchup_id,
            "fight_id": next(iter(fight_ids), None),
            "event_id": market.event_id,
            "fighter_id": market.fighter_id,
            "opponent_id": market.opponent_id,
            "fighter_name": reference_quote.fighter_name,
            "opponent_name": reference_quote.opponent_name,
            "event_date": market.event_date,
            "timing_precision": market.timing_precision,
            "event_start_utc": market.event_start_utc,
            "decision_issued_at_utc": utc_text(issued, "decision_issued_at_utc"),
            "market_as_of_utc": market.as_of_utc,
            "reference_quote_id": reference_quote.quote_id,
            "reference_book": reference_quote.book,
            "market_consensus_id": market.consensus_id,
            "forecast_capture_id": forecast.forecast_capture_id,
            "model_id": forecast.model_id,
            "other_book_count": market.book_count,
            "market_probability": market.no_vig_fighter_probability,
            "model_probability": forecast.model_probability,
            "book_probability_range": float(book_probability_range),
            "candidate_probability": candidate,
            "model_disagreement_scale": policy.feature_scales[0],
            "model_disagreement_coefficient": policy.feature_coefficients[0],
            "book_disagreement_scale": policy.feature_scales[1],
            "book_disagreement_coefficient": policy.feature_coefficients[1],
            "minimum_expected_return": policy.minimum_expected_return,
            "fighter_reference_moneyline": fighter_line,
            "opponent_reference_moneyline": opponent_line,
            "fighter_expected_return": fighter_ev,
            "opponent_expected_return": opponent_ev,
            "paper_action": action,
            "action_probability": action_probability,
            "action_reference_moneyline": action_line,
            "hypothetical_risk_units": 0.0 if action == "pass" else 1.0,
        }
        return cls(decision_id=canonical_hash(body), **body)  # type: ignore[arg-type]

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "MarketFirstPaperDecision":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                "market-first decision schema mismatch; "
                f"missing={missing}, extra={extra}"
            )
        try:
            result = cls(**{field: record[field] for field in cls.FIELDNAMES})
        except TypeError as error:
            raise MarketDataError("invalid market-first decision fields") from error
        result.validate_integrity()
        return result

    @property
    def natural_key(self) -> tuple[str, str]:
        return (self.matchup_id, self.policy_version)

    def validate_integrity(self) -> None:
        if (
            self.schema_version != POLICY_SCHEMA_VERSION
            or self.betting_status != BETTING_STATUS
            or self.paper_only is not True
            or self.candidate_only is not True
            or self.execution_enabled is not False
        ):
            raise StoreIntegrityError("market-first decision is not paper-only")
        validated_sha256(
            self.policy_artifact_sha256, "policy_artifact_sha256"
        )
        frozen_policy = FrozenMarketFirstPolicy.load(DEFAULT_POLICY_PATH)
        frozen_values = (
            self.policy_version,
            self.policy_artifact_sha256,
            float(self.model_disagreement_scale),
            float(self.model_disagreement_coefficient),
            float(self.book_disagreement_scale),
            float(self.book_disagreement_coefficient),
            float(self.minimum_expected_return),
        )
        expected_frozen_values = (
            frozen_policy.policy_version,
            frozen_policy.artifact_sha256,
            frozen_policy.feature_scales[0],
            frozen_policy.feature_coefficients[0],
            frozen_policy.feature_scales[1],
            frozen_policy.feature_coefficients[1],
            frozen_policy.minimum_expected_return,
        )
        if frozen_values != expected_frozen_values:
            raise StoreIntegrityError(
                "market-first decision does not use the frozen policy artifact"
            )
        issued, _, _, _ = require_before_event(
            self.decision_issued_at_utc,
            event_date=self.event_date,
            timing_precision=self.timing_precision,
            event_start_utc=self.event_start_utc,
            observed_field="decision_issued_at_utc",
        )
        if issued < utc_datetime(
            frozen_policy.prospective_first_capture_utc,
            "prospective_first_capture_utc",
        ):
            raise StoreIntegrityError(
                "market-first decision predates prospective deployment"
            )
        market = probability(self.market_probability, "market_probability")
        model = probability(self.model_probability, "model_probability")
        candidate = probability(self.candidate_probability, "candidate_probability")
        numeric = (
            self.book_probability_range,
            self.model_disagreement_scale,
            self.model_disagreement_coefficient,
            self.book_disagreement_scale,
            self.book_disagreement_coefficient,
            self.minimum_expected_return,
            self.fighter_expected_return,
            self.opponent_expected_return,
            self.hypothetical_risk_units,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise MarketDataError("market-first decision has non-finite values")
        if (
            int(self.other_book_count) < frozen_policy.minimum_other_books
            or not 0.0 <= float(self.book_probability_range) <= 1.0
            or float(self.model_disagreement_scale) <= 0.0
            or float(self.book_disagreement_scale) <= 0.0
            or float(self.minimum_expected_return) < 0.0
        ):
            raise MarketDataError("market-first decision has invalid thresholds")
        feature_values = (
            _logit(model) - _logit(market),
            _logit(market) * float(self.book_probability_range),
        )
        expected_candidate = _sigmoid(
            _logit(market)
            + float(self.model_disagreement_coefficient)
            * feature_values[0]
            / float(self.model_disagreement_scale)
            + float(self.book_disagreement_coefficient)
            * feature_values[1]
            / float(self.book_disagreement_scale)
        )
        fighter_line = moneyline(
            self.fighter_reference_moneyline, "fighter_reference_moneyline"
        )
        opponent_line = moneyline(
            self.opponent_reference_moneyline, "opponent_reference_moneyline"
        )
        fighter_ev = candidate * (1.0 + _profit_for_one_unit_risk(fighter_line)) - 1.0
        opponent_ev = (
            (1.0 - candidate)
            * (1.0 + _profit_for_one_unit_risk(opponent_line))
            - 1.0
        )
        if any(
            abs(float(supplied) - expected) > 1e-12
            for supplied, expected in (
                (candidate, expected_candidate),
                (self.fighter_expected_return, fighter_ev),
                (self.opponent_expected_return, opponent_ev),
            )
        ):
            raise StoreIntegrityError(
                "market-first decision derived values are inconsistent"
            )
        if (
            fighter_ev >= float(self.minimum_expected_return)
            and fighter_ev > opponent_ev
        ):
            expected_action = "fighter"
            expected_probability = candidate
            expected_line = fighter_line
        elif (
            opponent_ev >= float(self.minimum_expected_return)
            and opponent_ev > fighter_ev
        ):
            expected_action = "opponent"
            expected_probability = 1.0 - candidate
            expected_line = opponent_line
        else:
            expected_action = "pass"
            expected_probability = None
            expected_line = None
        expected_risk = 0.0 if expected_action == "pass" else 1.0
        if (
            self.paper_action != expected_action
            or self.action_probability != expected_probability
            or self.action_reference_moneyline != expected_line
            or float(self.hypothetical_risk_units) != expected_risk
        ):
            raise StoreIntegrityError(
                "market-first action does not follow its frozen threshold"
            )
        body = self.to_mapping()
        body.pop("decision_id")
        if self.decision_id != canonical_hash(body):
            raise StoreIntegrityError(
                "market-first decision_id does not match its contents"
            )

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


class MarketFirstPaperDecisionStore(_PaperRecordStore):
    """Append-only store for the separate market-first paper experiment."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        super().__init__(
            csv_path,
            jsonl_path,
            record_type=MarketFirstPaperDecision,  # type: ignore[arg-type]
            id_field="decision_id",
            time_field="decision_issued_at_utc",
        )

    def read(self) -> tuple[MarketFirstPaperDecision, ...]:
        return tuple(super().read())  # type: ignore[return-value]


@dataclass(frozen=True)
class MarketFirstDecisionBuild:
    decisions: tuple[MarketFirstPaperDecision, ...]
    eligible_horizon: bool
    lead_time_seconds: float | None
    matchups_considered: int
    matchups_already_frozen: int
    matchups_without_fresh_quotes: int
    matchups_without_forecast: int


def _lead_time(quotes: tuple[QuoteSnapshot, ...]) -> float | None:
    contracts = {
        (item.timing_precision, item.event_start_utc, item.observed_at_utc)
        for item in quotes
    }
    if len(contracts) != 1:
        raise StoreIntegrityError("one market-first capture has multiple clocks")
    precision, event_start, observed = next(iter(contracts))
    if precision != "timestamp" or event_start is None:
        return None
    return (
        utc_datetime(event_start, "event_start_utc")
        - utc_datetime(observed, "observed_at_utc")
    ).total_seconds()


def build_market_first_decisions(
    quotes: Iterable[QuoteSnapshot],
    forecasts: Iterable[ForecastCapture],
    source_metadata: Iterable[QuoteSourceMetadata],
    *,
    policy: FrozenMarketFirstPolicy,
    existing_decisions: Iterable[MarketFirstPaperDecision] = (),
) -> MarketFirstDecisionBuild:
    """Build decisions from one capture, choosing the best offered price."""

    quote_records = tuple(quotes)
    if not quote_records:
        raise ValueError("at least one quote is required")
    capture_ids = {item.capture_id for item in quote_records}
    if len(capture_ids) != 1:
        raise StoreIntegrityError("market-first builder accepts one capture")
    capture_id = next(iter(capture_ids))
    lead = _lead_time(quote_records)
    grouped: dict[str, list[QuoteSnapshot]] = defaultdict(list)
    for quote in quote_records:
        grouped[quote.matchup_id].append(quote)
    capture_observed_at = min(item.observed_at_utc for item in quote_records)
    deployed = utc_datetime(
        capture_observed_at, "capture_observed_at"
    ) >= utc_datetime(
        policy.prospective_first_capture_utc, "prospective_first_capture_utc"
    )
    eligible = (
        deployed
        and lead is not None
        and abs(lead - policy.target_lead_seconds) <= policy.window_seconds
    )
    if not eligible:
        return MarketFirstDecisionBuild(
            (), False, lead, len(grouped), 0, 0, 0
        )
    metadata_records = tuple(source_metadata)
    metadata_by_quote = {item.quote_id: item for item in metadata_records}
    if len(metadata_by_quote) != len(metadata_records):
        raise StoreIntegrityError("duplicate market-first source metadata")
    forecast_by_matchup = {
        item.matchup_id: item
        for item in forecasts
        if item.capture_id == capture_id
    }
    if len(forecast_by_matchup) != sum(
        item.capture_id == capture_id for item in forecasts
    ):
        raise StoreIntegrityError("duplicate market-first matchup forecast")
    frozen = {
        (item.matchup_id, item.policy_version) for item in existing_decisions
    }
    decisions: list[MarketFirstPaperDecision] = []
    already = no_fresh = no_forecast = 0
    for matchup_id in sorted(grouped):
        if (matchup_id, policy.policy_version) in frozen:
            already += 1
            continue
        forecast = forecast_by_matchup.get(matchup_id)
        if forecast is None:
            no_forecast += 1
            continue
        fresh: list[QuoteSnapshot] = []
        for quote in grouped[matchup_id]:
            metadata = metadata_by_quote.get(quote.quote_id)
            if metadata is None:
                continue
            if (
                metadata.capture_id != quote.capture_id
                or metadata.matchup_id != quote.matchup_id
                or metadata.source != quote.source
                or metadata.book != quote.book
                or metadata.observed_at_utc != quote.observed_at_utc
            ):
                raise StoreIntegrityError(
                    "market-first source metadata disagrees with its quote"
                )
            age = float(metadata.source_quote_age_seconds)
            if -300.0 <= age <= policy.maximum_source_quote_age_seconds:
                fresh.append(quote)
        if len(fresh) < policy.minimum_other_books + 1:
            no_fresh += 1
            continue
        observed_times = {item.observed_at_utc for item in fresh}
        if len(observed_times) != 1:
            raise StoreIntegrityError("fresh market-first quotes span retrievals")
        observed_at = next(iter(observed_times))
        candidates: list[MarketFirstPaperDecision] = []
        for target in sorted(fresh, key=lambda item: (item.book.casefold(), item.quote_id)):
            other = [
                item
                for item in fresh
                if item.book.casefold() != target.book.casefold()
            ]
            probabilities = [item.no_vig_fighter_probability for item in other]
            if len(probabilities) < policy.minimum_other_books:
                continue
            try:
                market = consensus_as_of(
                    fresh,
                    capture_id=capture_id,
                    matchup_id=matchup_id,
                    as_of_utc=observed_at,
                    min_books=policy.minimum_other_books,
                    exclude_books=(target.book,),
                )
                candidates.append(
                    MarketFirstPaperDecision.create(
                        market,
                        target,
                        forecast,
                        policy=policy,
                        book_probability_range=max(probabilities) - min(probabilities),
                        decision_issued_at_utc=observed_at,
                    )
                )
            except MarketDataError:
                continue
        if not candidates:
            no_fresh += 1
            continue
        candidates.sort(
            key=lambda item: (
                -max(item.fighter_expected_return, item.opponent_expected_return),
                item.reference_book.casefold(),
                item.decision_id,
            )
        )
        decisions.append(candidates[0])
        frozen.add((matchup_id, policy.policy_version))
    return MarketFirstDecisionBuild(
        tuple(decisions),
        True,
        lead,
        len(grouped),
        already,
        no_fresh,
        no_forecast,
    )


def settle_market_first_decision(
    decision: MarketFirstPaperDecision,
    *,
    target: object | None,
    settled_at_utc: datetime | str,
    result_source_sha256: object,
    fight_id: object | None = None,
) -> PaperSettlement:
    """Settle one immutable candidate decision after the event."""

    if not isinstance(decision, MarketFirstPaperDecision):
        raise TypeError("decision must be MarketFirstPaperDecision")
    decision.validate_integrity()
    settled = utc_datetime(settled_at_utc, "settled_at_utc")
    if decision.timing_precision == "timestamp":
        if settled < utc_datetime(decision.event_start_utc, "event_start_utc"):
            raise MarketDataError("settlement precedes the event")
    elif settled.date() <= date.fromisoformat(decision.event_date):
        raise MarketDataError("date-only settlement must occur after event date")
    outcome = None if target is None else binary_target(target)
    resolved_fight_id = optional_stable_id(fight_id, "fight_id")
    if (
        decision.fight_id is not None
        and resolved_fight_id is not None
        and decision.fight_id != resolved_fight_id
    ):
        raise StoreIntegrityError("settlement fight ID conflicts with decision")
    resolved_fight_id = resolved_fight_id or decision.fight_id
    if outcome is None:
        status = "void" if decision.paper_action != "pass" else "pass_unscored"
        risk = profit = 0.0
        log_loss = brier = accuracy = None
    else:
        target_probability = (
            decision.candidate_probability
            if outcome == 1
            else 1.0 - decision.candidate_probability
        )
        log_loss = -math.log(target_probability)
        brier = (decision.candidate_probability - outcome) ** 2
        accuracy = (
            0.5
            if decision.candidate_probability == 0.5
            else float(
                (decision.candidate_probability > 0.5) == bool(outcome)
            )
        )
        if decision.paper_action == "pass":
            status = "pass"
            risk = profit = 0.0
        else:
            selected_target = 1 if decision.paper_action == "fighter" else 0
            won = selected_target == outcome
            status = "paper_win" if won else "paper_loss"
            risk = 1.0
            profit = (
                _profit_for_one_unit_risk(decision.action_reference_moneyline)
                if won
                else -1.0
            )
    body = {
        "schema_version": 1,
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
        "target": outcome,
        "settlement_status": status,
        "hypothetical_risk_units": risk,
        "hypothetical_profit_units": profit,
        "forecast_log_loss": log_loss,
        "forecast_brier_score": brier,
        "forecast_accuracy_credit": accuracy,
    }
    return PaperSettlement(settlement_id=canonical_hash(body), **body)


def _quantile(values: list[float], probability_value: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability_value * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _event_interval(
    values: Mapping[str, tuple[float, float]],
    *,
    ratio: bool,
    bootstrap_samples: int,
    seed_label: str,
) -> dict[str, object]:
    ordered = [values[key] for key in sorted(values)]
    denominator = sum(item[1] for item in ordered)
    point = sum(item[0] for item in ordered) / denominator if denominator else None
    result: dict[str, object] = {
        "event_count": len(ordered),
        "point": point,
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(ordered) < 2 or denominator == 0.0:
        return result
    generator = random.Random(
        int(canonical_hash({"label": seed_label, "values": ordered})[:16], 16)
    )
    samples: list[float] = []
    for _ in range(bootstrap_samples):
        chosen = [generator.choice(ordered) for _ in ordered]
        if ratio:
            sample_denominator = sum(item[1] for item in chosen)
            if sample_denominator:
                samples.append(sum(item[0] for item in chosen) / sample_denominator)
        else:
            samples.append(sum(item[0] for item in chosen) / sum(item[1] for item in chosen))
    result.update(
        {
            "bootstrap_samples": len(samples),
            "ci_95_lower": _quantile(samples, 0.025),
            "ci_95_upper": _quantile(samples, 0.975),
        }
    )
    return result


def summarize_market_first_paper(
    decisions: Iterable[MarketFirstPaperDecision],
    settlements: Iterable[PaperSettlement],
    quotes: Iterable[QuoteSnapshot],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, object]:
    """Return a plain, bounded report for the frozen future experiment."""

    decision_records = tuple(decisions)
    settlement_records = tuple(settlements)
    quote_records = tuple(quotes)
    by_id: dict[str, MarketFirstPaperDecision] = {}
    for decision in decision_records:
        decision.validate_integrity()
        if decision.decision_id in by_id:
            raise StoreIntegrityError("duplicate market-first decision")
        by_id[decision.decision_id] = decision
    settled_by_id: dict[str, PaperSettlement] = {}
    for settlement in settlement_records:
        settlement.validate_integrity()
        decision = by_id.get(settlement.decision_id)
        if decision is None:
            raise StoreIntegrityError("market-first settlement has no decision")
        if settlement.decision_id in settled_by_id:
            raise StoreIntegrityError("market-first decision settled twice")
        expected = settle_market_first_decision(
            decision,
            target=settlement.target,
            settled_at_utc=settlement.settled_at_utc,
            result_source_sha256=settlement.result_source_sha256,
            fight_id=settlement.fight_id,
        )
        if expected != settlement:
            raise StoreIntegrityError("market-first settlement was rewritten")
        settled_by_id[settlement.decision_id] = settlement
    scored = [item for item in settlement_records if item.target is not None]
    targets = [int(item.target) for item in scored]
    candidate_probabilities = [
        by_id[item.decision_id].candidate_probability for item in scored
    ]
    market_probabilities = [
        by_id[item.decision_id].market_probability for item in scored
    ]
    model_probabilities = [
        by_id[item.decision_id].model_probability for item in scored
    ]
    selections = [
        item
        for item in settlement_records
        if item.settlement_status in {"paper_win", "paper_loss", "void"}
    ]
    wins = sum(item.settlement_status == "paper_win" for item in selections)
    losses = sum(item.settlement_status == "paper_loss" for item in selections)
    voids = sum(item.settlement_status == "void" for item in selections)
    risk = sum(item.hypothetical_risk_units for item in selections)
    profit = sum(item.hypothetical_profit_units for item in selections)
    cumulative = peak = drawdown = 0.0
    for item in sorted(
        settlement_records, key=lambda value: (value.settled_at_utc, value.settlement_id)
    ):
        cumulative += item.hypothetical_profit_units
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    event_returns: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    for item in selections:
        current_profit, current_risk = event_returns[item.event_id]
        event_returns[item.event_id] = (
            current_profit + item.hypothetical_profit_units,
            current_risk + item.hypothetical_risk_units,
        )
    log_loss_events: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    for settlement in scored:
        decision = by_id[settlement.decision_id]
        outcome = int(settlement.target)
        candidate_target = (
            decision.candidate_probability
            if outcome == 1
            else 1.0 - decision.candidate_probability
        )
        market_target = (
            decision.market_probability
            if outcome == 1
            else 1.0 - decision.market_probability
        )
        difference = -math.log(candidate_target) + math.log(market_target)
        total, count = log_loss_events[decision.event_id]
        log_loss_events[decision.event_id] = (total + difference, count + 1.0)

    quote_by_id = {item.quote_id: item for item in quote_records}
    clv_events: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    clv_values: list[float] = []
    for decision in decision_records:
        if decision.paper_action == "pass":
            continue
        reference = quote_by_id.get(decision.reference_quote_id)
        if reference is None:
            continue
        later = [
            item
            for item in quote_records
            if item.matchup_id == decision.matchup_id
            and item.book.casefold() == decision.reference_book.casefold()
            and item.observed_at_utc > decision.market_as_of_utc
        ]
        if not later:
            continue
        latest = max(later, key=lambda item: (item.observed_at_utc, item.quote_id))
        locked_break_even = implied_probability(decision.action_reference_moneyline)
        latest_implied = (
            latest.fighter_implied_probability
            if decision.paper_action == "fighter"
            else latest.opponent_implied_probability
        )
        advantage = latest_implied - locked_break_even
        clv_values.append(advantage)
        total, count = clv_events[decision.event_id]
        clv_events[decision.event_id] = (total + advantage, count + 1.0)

    candidate_metrics = (
        forecast_metrics(candidate_probabilities, targets).to_mapping()
        if scored
        else None
    )
    market_metrics = (
        forecast_metrics(market_probabilities, targets).to_mapping()
        if scored
        else None
    )
    model_metrics = (
        forecast_metrics(model_probabilities, targets).to_mapping()
        if scored
        else None
    )
    return_interval = _event_interval(
        event_returns,
        ratio=True,
        bootstrap_samples=bootstrap_samples,
        seed_label="market-first-return",
    )
    log_loss_interval = _event_interval(
        log_loss_events,
        ratio=False,
        bootstrap_samples=bootstrap_samples,
        seed_label="market-first-minus-market-log-loss",
    )
    clv_interval = _event_interval(
        clv_events,
        ratio=False,
        bootstrap_samples=bootstrap_samples,
        seed_label="market-first-clv",
    )
    settled_events = {by_id[item.decision_id].event_id for item in scored}
    return {
        "schema_version": 1,
        "betting_status": BETTING_STATUS,
        "paper_only": True,
        "candidate_only": True,
        "execution_enabled": False,
        "policy_version": (
            decision_records[0].policy_version if decision_records else None
        ),
        "policy_artifact_sha256": (
            decision_records[0].policy_artifact_sha256 if decision_records else None
        ),
        "decisions_total": len(decision_records),
        "settlements_total": len(settlement_records),
        "unsettled_decisions": len(decision_records) - len(settlement_records),
        "settled_events": len(settled_events),
        "results": {
            "scored_fights": len(scored),
            "recommended_bets": wins + losses,
            "wins": wins,
            "losses": losses,
            "voids": voids,
            "risk_units": risk,
            "profit_units": profit,
            "roi": profit / risk if risk else None,
            "maximum_drawdown_units": drawdown,
        },
        "probability_quality_on_all_scored_fights": {
            "market_first_candidate": candidate_metrics,
            "leave_one_out_market": market_metrics,
            "current_model": model_metrics,
            "candidate_minus_market_log_loss": log_loss_interval,
        },
        "paper_return_interval": return_interval,
        "latest_available_same_book_price_movement": {
            "definition": (
                "later implied probability minus locked break-even probability; "
                "positive means the recorded paper price was better"
            ),
            "count": len(clv_values),
            "mean_probability_advantage": (
                sum(clv_values) / len(clv_values) if clv_values else None
            ),
            "positive_rate": (
                sum(value > 0.0 for value in clv_values) / len(clv_values)
                if clv_values
                else None
            ),
            "interval": clv_interval,
        },
        "evidence_gate": {
            "status": "collecting_new_fights",
            "minimum_recommended_bets": 100,
            "minimum_settled_events": 40,
            "count_requirements_met": (
                wins + losses >= 100 and len(settled_events) >= 40
            ),
            "positive_return_interval_met": (
                return_interval["ci_95_lower"] is not None
                and float(return_interval["ci_95_lower"]) > 0.0
            ),
            "candidate_beats_market_interval_met": (
                log_loss_interval["ci_95_upper"] is not None
                and float(log_loss_interval["ci_95_upper"]) < 0.0
            ),
            "positive_price_movement_interval_met": (
                clv_interval["ci_95_lower"] is not None
                and float(clv_interval["ci_95_lower"]) > 0.0
            ),
            "automatic_promotion": False,
            "execution_enabled": False,
        },
        "decision_dataset_sha256": canonical_hash(
            [item.to_mapping() for item in decision_records]
        ),
        "settlement_dataset_sha256": canonical_hash(
            [item.to_mapping() for item in settlement_records]
        ),
    }


__all__ = (
    "FrozenMarketFirstPolicy",
    "MarketFirstDecisionBuild",
    "MarketFirstPaperDecision",
    "MarketFirstPaperDecisionStore",
    "PaperSettlementStore",
    "build_market_first_decisions",
    "settle_market_first_decision",
    "summarize_market_first_paper",
)
