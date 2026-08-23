"""Paper-only decision records, settlement, and monitoring metrics.

This module intentionally exposes no bankroll, order, account, or execution
API. Every record carries a machine-readable disabled status and uses a fixed
one-unit hypothetical risk solely for reproducible offline evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import csv
import io
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import ClassVar, Iterable, Mapping

from ._common import (
    BETTING_STATUS,
    MarketDataError,
    SCHEMA_VERSION,
    StoreIntegrityError,
    binary_target,
    canonical_hash,
    canonical_json,
    implied_probability,
    moneyline,
    probability,
    optional_stable_id,
    require_before_event,
    utc_datetime,
    utc_text,
    validated_sha256,
)
from ._storage import atomic_write_text, exclusive_store_lock
from .blend import ForecastMetrics, forecast_metrics, symmetric_logit_blend
from .forecasts import ForecastCapture
from .quotes import AppendResult, MarketConsensus, QuoteSnapshot


BAYESIAN_FILTER_POLICY_VERSION = "bayesian-filtered-existing-moneyline-v1"
BAYESIAN_FILTER_MINIMUM_MEAN_EV = 0.05
BAYESIAN_FILTER_MINIMUM_PROBABILITY_POSITIVE_EV = 0.80
_STANDARD_NORMAL = NormalDist()


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True)
class PaperDecision:
    schema_version: int
    decision_id: str
    betting_status: str
    paper_only: bool
    capture_id: str
    matchup_id: str
    fight_id: str | None
    event_id: str
    fighter_id: str
    opponent_id: str
    event_date: str
    timing_precision: str
    event_start_utc: str | None
    decision_issued_at_utc: str
    market_as_of_utc: str
    quote_age_seconds_at_decision: float
    maximum_quote_age_seconds: float
    market_consensus_id: str
    reference_quote_id: str
    forecast_capture_id: str
    model_id: str
    selected_gamma: float
    market_probability: float
    model_probability: float
    blend_probability: float
    minimum_expected_return: float
    fighter_reference_moneyline: int
    opponent_reference_moneyline: int
    fighter_break_even_probability: float
    opponent_break_even_probability: float
    fighter_edge: float
    opponent_edge: float
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
        "capture_id",
        "matchup_id",
        "fight_id",
        "event_id",
        "fighter_id",
        "opponent_id",
        "event_date",
        "timing_precision",
        "event_start_utc",
        "decision_issued_at_utc",
        "market_as_of_utc",
        "quote_age_seconds_at_decision",
        "maximum_quote_age_seconds",
        "market_consensus_id",
        "reference_quote_id",
        "forecast_capture_id",
        "model_id",
        "selected_gamma",
        "market_probability",
        "model_probability",
        "blend_probability",
        "minimum_expected_return",
        "fighter_reference_moneyline",
        "opponent_reference_moneyline",
        "fighter_break_even_probability",
        "opponent_break_even_probability",
        "fighter_edge",
        "opponent_edge",
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
        selected_gamma: object,
        decision_issued_at_utc: datetime | str,
        minimum_expected_return: object = 0.05,
        maximum_quote_age_seconds: object = 300.0,
        fight_id: object | None = None,
    ) -> "PaperDecision":
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
            market.timing_precision,
            market.event_start_utc,
        )
        quote_identity = (
            reference_quote.capture_id,
            reference_quote.matchup_id,
            reference_quote.event_id,
            reference_quote.fighter_id,
            reference_quote.opponent_id,
            reference_quote.event_date,
            reference_quote.timing_precision,
            reference_quote.event_start_utc,
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
        )
        if identity != quote_identity or identity != forecast_identity:
            raise StoreIntegrityError(
                "paper decision inputs identify different matchups"
            )
        reference_book_key = reference_quote.book.casefold()
        if reference_quote.quote_id in market.quote_ids:
            raise StoreIntegrityError(
                "target reference quote must be excluded from market consensus"
            )
        if market.book_count < 3:
            raise StoreIntegrityError(
                "paper decisions require at least three non-target consensus books"
            )
        if reference_book_key in market.included_book_keys:
            raise StoreIntegrityError(
                "target reference book must not influence its own consensus"
            )
        if market.excluded_book_keys != (reference_book_key,):
            raise StoreIntegrityError(
                "paper consensus must exclude exactly the target reference book"
            )
        if utc_datetime(
            reference_quote.observed_at_utc, "reference_quote.observed_at_utc"
        ) > utc_datetime(market.as_of_utc, "market.as_of_utc"):
            raise StoreIntegrityError(
                "reference quote was not available at the consensus as-of cutoff"
            )
        if reference_quote.observed_at_utc != market.latest_observed_at_utc:
            raise StoreIntegrityError(
                "target quote and consensus books must come from the same retrieval time"
            )
        issued, _, _, _ = require_before_event(
            decision_issued_at_utc,
            event_date=market.event_date,
            timing_precision=market.timing_precision,
            event_start_utc=market.event_start_utc,
            observed_field="decision_issued_at_utc",
        )
        market_cutoff = utc_datetime(market.as_of_utc, "market.as_of_utc")
        if issued < market_cutoff:
            raise MarketDataError("decision cannot precede its market consensus")
        quote_age_seconds = (issued - market_cutoff).total_seconds()
        try:
            maximum_quote_age = float(maximum_quote_age_seconds)
        except (TypeError, ValueError) as error:
            raise MarketDataError("maximum_quote_age_seconds must be numeric") from error
        if not math.isfinite(maximum_quote_age) or maximum_quote_age < 0.0:
            raise MarketDataError(
                "maximum_quote_age_seconds must be finite and nonnegative"
            )
        if quote_age_seconds > maximum_quote_age:
            raise MarketDataError(
                "paper decision exceeded the maximum allowed quote age"
            )
        if issued < utc_datetime(
            forecast.forecast_issued_at_utc, "forecast.forecast_issued_at_utc"
        ):
            raise MarketDataError("decision cannot precede its model forecast")
        try:
            expected_return_threshold = float(minimum_expected_return)
        except (TypeError, ValueError) as error:
            raise MarketDataError("minimum_expected_return must be numeric") from error
        if not math.isfinite(expected_return_threshold) or expected_return_threshold < 0.0:
            raise MarketDataError(
                "minimum_expected_return must be finite and nonnegative"
            )
        blended = symmetric_logit_blend(
            market.no_vig_fighter_probability,
            forecast.model_probability,
            selected_gamma,
        )
        gamma = float(selected_gamma)
        fighter_break_even = reference_quote.fighter_implied_probability
        opponent_break_even = reference_quote.opponent_implied_probability
        fighter_edge = blended - fighter_break_even
        opponent_edge = (1.0 - blended) - opponent_break_even
        fighter_decimal_return = 1.0 + _profit_for_one_unit_risk(
            reference_quote.fighter_moneyline
        )
        opponent_decimal_return = 1.0 + _profit_for_one_unit_risk(
            reference_quote.opponent_moneyline
        )
        fighter_expected_return = blended * fighter_decimal_return - 1.0
        opponent_expected_return = (
            (1.0 - blended) * opponent_decimal_return - 1.0
        )
        if (
            fighter_expected_return >= expected_return_threshold
            and fighter_expected_return > opponent_expected_return
        ):
            action = "fighter"
            action_probability = blended
            action_line = reference_quote.fighter_moneyline
        elif (
            opponent_expected_return >= expected_return_threshold
            and opponent_expected_return > fighter_expected_return
        ):
            action = "opponent"
            action_probability = 1.0 - blended
            action_line = reference_quote.opponent_moneyline
        else:
            action = "pass"
            action_probability = None
            action_line = None
        known_fight_ids = {
            value
            for value in (
                market.fight_id,
                reference_quote.fight_id,
                forecast.fight_id,
                optional_stable_id(fight_id, "fight_id"),
            )
            if value is not None
        }
        if len(known_fight_ids) > 1:
            raise StoreIntegrityError("matchup resolves to conflicting fight IDs")
        body = {
            "schema_version": SCHEMA_VERSION,
            "betting_status": BETTING_STATUS,
            "paper_only": True,
            "capture_id": market.capture_id,
            "matchup_id": market.matchup_id,
            "fight_id": next(iter(known_fight_ids), None),
            "event_id": market.event_id,
            "fighter_id": market.fighter_id,
            "opponent_id": market.opponent_id,
            "event_date": market.event_date,
            "timing_precision": market.timing_precision,
            "event_start_utc": market.event_start_utc,
            "decision_issued_at_utc": utc_text(issued, "decision_issued_at_utc"),
            "market_as_of_utc": market.as_of_utc,
            "quote_age_seconds_at_decision": quote_age_seconds,
            "maximum_quote_age_seconds": maximum_quote_age,
            "market_consensus_id": market.consensus_id,
            "reference_quote_id": reference_quote.quote_id,
            "forecast_capture_id": forecast.forecast_capture_id,
            "model_id": forecast.model_id,
            "selected_gamma": gamma,
            "market_probability": market.no_vig_fighter_probability,
            "model_probability": forecast.model_probability,
            "blend_probability": blended,
            "minimum_expected_return": expected_return_threshold,
            "fighter_reference_moneyline": reference_quote.fighter_moneyline,
            "opponent_reference_moneyline": reference_quote.opponent_moneyline,
            "fighter_break_even_probability": fighter_break_even,
            "opponent_break_even_probability": opponent_break_even,
            "fighter_edge": fighter_edge,
            "opponent_edge": opponent_edge,
            "fighter_expected_return": fighter_expected_return,
            "opponent_expected_return": opponent_expected_return,
            "paper_action": action,
            "action_probability": action_probability,
            "action_reference_moneyline": action_line,
            "hypothetical_risk_units": 0.0 if action == "pass" else 1.0,
        }
        return cls(decision_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "PaperDecision":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"paper decision schema mismatch; missing={missing}, extra={extra}"
            )
        try:
            decision = cls(**{field: record[field] for field in cls.FIELDNAMES})
        except TypeError as error:
            raise MarketDataError("invalid paper decision fields") from error
        decision.validate_integrity()
        return decision

    @property
    def natural_key(self) -> tuple[str, str, str, str]:
        return (
            self.capture_id,
            self.matchup_id,
            self.reference_quote_id,
            self.forecast_capture_id,
        )

    def validate_integrity(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MarketDataError("unsupported paper decision schema version")
        if self.betting_status != BETTING_STATUS or self.paper_only is not True:
            raise StoreIntegrityError("paper decision must keep execution disabled")
        body = self.to_mapping()
        body.pop("decision_id")
        if self.decision_id != canonical_hash(body):
            raise StoreIntegrityError("decision_id does not match canonical contents")
        require_before_event(
            self.decision_issued_at_utc,
            event_date=self.event_date,
            timing_precision=self.timing_precision,
            event_start_utc=self.event_start_utc,
            observed_field="decision_issued_at_utc",
        )
        for field in (
            "market_probability",
            "model_probability",
            "blend_probability",
            "fighter_break_even_probability",
            "opponent_break_even_probability",
        ):
            probability(getattr(self, field), field)
        numeric = (
            self.selected_gamma,
            self.minimum_expected_return,
            self.fighter_edge,
            self.opponent_edge,
            self.fighter_expected_return,
            self.opponent_expected_return,
            self.hypothetical_risk_units,
            self.quote_age_seconds_at_decision,
            self.maximum_quote_age_seconds,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise MarketDataError("paper decision contains a non-finite number")
        if not 0.0 <= float(self.selected_gamma) <= 1.0:
            raise MarketDataError("paper selected_gamma is outside [0, 1]")
        if float(self.minimum_expected_return) < 0.0:
            raise MarketDataError("paper minimum_expected_return is negative")
        issued = utc_datetime(
            self.decision_issued_at_utc, "decision_issued_at_utc"
        )
        market_cutoff = utc_datetime(self.market_as_of_utc, "market_as_of_utc")
        expected_quote_age = (issued - market_cutoff).total_seconds()
        if abs(float(self.quote_age_seconds_at_decision) - expected_quote_age) > 1e-6:
            raise StoreIntegrityError("paper quote age disagrees with its timestamps")
        if float(self.quote_age_seconds_at_decision) < 0.0:
            raise MarketDataError("paper quote age cannot be negative")
        if float(self.maximum_quote_age_seconds) < 0.0:
            raise MarketDataError("paper maximum quote age cannot be negative")
        if float(self.quote_age_seconds_at_decision) > float(
            self.maximum_quote_age_seconds
        ):
            raise StoreIntegrityError("paper decision exceeded its maximum quote age")
        fighter_line = moneyline(
            self.fighter_reference_moneyline, "fighter_reference_moneyline"
        )
        opponent_line = moneyline(
            self.opponent_reference_moneyline, "opponent_reference_moneyline"
        )
        expected_blend = symmetric_logit_blend(
            self.market_probability,
            self.model_probability,
            self.selected_gamma,
        )
        expected_values = (
            expected_blend,
            implied_probability(fighter_line),
            implied_probability(opponent_line),
            expected_blend - implied_probability(fighter_line),
            (1.0 - expected_blend) - implied_probability(opponent_line),
            expected_blend * (1.0 + _profit_for_one_unit_risk(fighter_line)) - 1.0,
            (1.0 - expected_blend)
            * (1.0 + _profit_for_one_unit_risk(opponent_line))
            - 1.0,
        )
        supplied_values = (
            self.blend_probability,
            self.fighter_break_even_probability,
            self.opponent_break_even_probability,
            self.fighter_edge,
            self.opponent_edge,
            self.fighter_expected_return,
            self.opponent_expected_return,
        )
        if any(
            abs(float(supplied) - expected) > 1e-12
            for supplied, expected in zip(supplied_values, expected_values)
        ):
            raise StoreIntegrityError("paper decision derived values are inconsistent")
        if (
            self.fighter_expected_return >= self.minimum_expected_return
            and self.fighter_expected_return > self.opponent_expected_return
        ):
            expected_action = "fighter"
            expected_probability = self.blend_probability
            expected_line = fighter_line
        elif (
            self.opponent_expected_return >= self.minimum_expected_return
            and self.opponent_expected_return > self.fighter_expected_return
        ):
            expected_action = "opponent"
            expected_probability = 1.0 - self.blend_probability
            expected_line = opponent_line
        else:
            expected_action = "pass"
            expected_probability = expected_line = None
        if (
            self.paper_action != expected_action
            or self.action_probability != expected_probability
            or self.action_reference_moneyline != expected_line
        ):
            raise StoreIntegrityError("paper_action does not follow the locked EV policy")
        if self.paper_action not in {"fighter", "opponent", "pass"}:
            raise MarketDataError("unsupported paper_action")
        if self.paper_action == "pass":
            if (
                self.action_probability is not None
                or self.action_reference_moneyline is not None
                or float(self.hypothetical_risk_units) != 0.0
            ):
                raise StoreIntegrityError("pass decision carries hypothetical risk")
        elif (
            self.action_probability is None
            or self.action_reference_moneyline is None
            or float(self.hypothetical_risk_units) != 1.0
        ):
            raise StoreIntegrityError("paper selection must use exactly one unit")

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


@dataclass(frozen=True)
class BayesianFilteredDecision:
    """Immutable T-24 veto layered on one existing moneyline decision.

    The filter never invents a different side.  A base paper selection survives
    only when the Bayesian posterior also has at least the configured mean EV
    and posterior probability of positive EV at the same frozen target price.
    """

    schema_version: int
    filtered_decision_id: str
    betting_status: str
    paper_only: bool
    execution_enabled: bool
    policy_version: str
    base_decision_id: str
    capture_id: str
    matchup_id: str
    fight_id: str | None
    event_id: str
    fighter_id: str
    opponent_id: str
    event_date: str
    decision_issued_at_utc: str
    reference_quote_id: str
    source_vegas_sha256: str
    bayesian_artifact_sha256: str
    bayesian_model_id: str
    bayesian_status: str
    credible_level: float
    fighter_posterior_mean: float
    fighter_posterior_median: float
    fighter_probability_lower: float
    fighter_probability_upper: float
    fighter_calibrated_logit_location: float
    calibrated_logit_scale: float
    minimum_mean_expected_return: float
    minimum_probability_positive_expected_return: float
    base_paper_action: str
    base_expected_return: float
    candidate_moneyline: int | None
    candidate_posterior_mean_probability: float | None
    candidate_posterior_mean_expected_return: float | None
    candidate_expected_return_lower: float | None
    candidate_expected_return_upper: float | None
    candidate_probability_positive_expected_return: float | None
    filtered_paper_action: str
    filter_status: str
    hypothetical_risk_units: float

    FIELDNAMES: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "filtered_decision_id",
        "betting_status",
        "paper_only",
        "execution_enabled",
        "policy_version",
        "base_decision_id",
        "capture_id",
        "matchup_id",
        "fight_id",
        "event_id",
        "fighter_id",
        "opponent_id",
        "event_date",
        "decision_issued_at_utc",
        "reference_quote_id",
        "source_vegas_sha256",
        "bayesian_artifact_sha256",
        "bayesian_model_id",
        "bayesian_status",
        "credible_level",
        "fighter_posterior_mean",
        "fighter_posterior_median",
        "fighter_probability_lower",
        "fighter_probability_upper",
        "fighter_calibrated_logit_location",
        "calibrated_logit_scale",
        "minimum_mean_expected_return",
        "minimum_probability_positive_expected_return",
        "base_paper_action",
        "base_expected_return",
        "candidate_moneyline",
        "candidate_posterior_mean_probability",
        "candidate_posterior_mean_expected_return",
        "candidate_expected_return_lower",
        "candidate_expected_return_upper",
        "candidate_probability_positive_expected_return",
        "filtered_paper_action",
        "filter_status",
        "hypothetical_risk_units",
    )

    @staticmethod
    def _candidate_values(
        base: PaperDecision,
        *,
        fighter_mean: float,
        fighter_lower: float,
        fighter_upper: float,
        fighter_logit_location: float,
        logit_scale: float,
    ) -> tuple[int, float, float, float, float, float]:
        fighter_selected = base.paper_action == "fighter"
        line = (
            base.fighter_reference_moneyline
            if fighter_selected
            else base.opponent_reference_moneyline
        )
        mean_probability = fighter_mean if fighter_selected else 1.0 - fighter_mean
        lower_probability = fighter_lower if fighter_selected else 1.0 - fighter_upper
        upper_probability = fighter_upper if fighter_selected else 1.0 - fighter_lower
        location = fighter_logit_location if fighter_selected else -fighter_logit_location
        decimal_return = 1.0 + _profit_for_one_unit_risk(line)
        mean_ev = decimal_return * mean_probability - 1.0
        lower_ev = decimal_return * lower_probability - 1.0
        upper_ev = decimal_return * upper_probability - 1.0
        break_even = implied_probability(line)
        if logit_scale == 0.0:
            probability_positive = float(mean_probability > break_even)
        else:
            threshold_logit = math.log(break_even / (1.0 - break_even))
            probability_positive = 1.0 - _STANDARD_NORMAL.cdf(
                (threshold_logit - location) / logit_scale
            )
        return (
            line,
            mean_probability,
            mean_ev,
            lower_ev,
            upper_ev,
            probability_positive,
        )

    @classmethod
    def create(
        cls,
        base: PaperDecision,
        *,
        source_vegas_sha256: object,
        bayesian_artifact_sha256: object,
        bayesian_model_id: object,
        bayesian_status: object,
        credible_level: object,
        fighter_posterior_mean: object,
        fighter_posterior_median: object,
        fighter_probability_lower: object,
        fighter_probability_upper: object,
        fighter_calibrated_logit_location: object,
        calibrated_logit_scale: object,
        minimum_mean_expected_return: object = BAYESIAN_FILTER_MINIMUM_MEAN_EV,
        minimum_probability_positive_expected_return: object = (
            BAYESIAN_FILTER_MINIMUM_PROBABILITY_POSITIVE_EV
        ),
    ) -> "BayesianFilteredDecision":
        if not isinstance(base, PaperDecision):
            raise TypeError("base must be a PaperDecision")
        base.validate_integrity()
        try:
            credible = float(credible_level)
            fighter_mean = float(fighter_posterior_mean)
            fighter_median = float(fighter_posterior_median)
            fighter_lower = float(fighter_probability_lower)
            fighter_upper = float(fighter_probability_upper)
            fighter_location = float(fighter_calibrated_logit_location)
            logit_scale = float(calibrated_logit_scale)
            minimum_mean_ev = float(minimum_mean_expected_return)
            minimum_probability = float(
                minimum_probability_positive_expected_return
            )
        except (TypeError, ValueError) as error:
            raise MarketDataError("Bayesian filter values must be numeric") from error
        numeric = (
            credible,
            fighter_mean,
            fighter_median,
            fighter_lower,
            fighter_upper,
            fighter_location,
            logit_scale,
            minimum_mean_ev,
            minimum_probability,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise MarketDataError("Bayesian filter values must be finite")
        if not 0.0 < fighter_lower <= fighter_median <= fighter_upper < 1.0:
            raise MarketDataError("Bayesian fighter probability interval is invalid")
        if not 0.0 < fighter_mean < 1.0:
            raise MarketDataError("Bayesian fighter posterior mean is invalid")
        if not 0.0 < credible < 1.0 or logit_scale < 0.0:
            raise MarketDataError("Bayesian credible level or logit scale is invalid")
        if minimum_mean_ev < 0.0 or not 0.0 < minimum_probability <= 1.0:
            raise MarketDataError("Bayesian filter thresholds are invalid")
        status = " ".join(str(bayesian_status or "").split())
        if not status:
            raise MarketDataError("bayesian_status must be nonempty")
        model_id = " ".join(str(bayesian_model_id or "").split())
        if not model_id:
            raise MarketDataError("bayesian_model_id must be nonempty")

        base_expected_return = max(
            base.fighter_expected_return, base.opponent_expected_return
        )
        candidate: tuple[int, float, float, float, float, float] | None = None
        if base.paper_action in {"fighter", "opponent"}:
            candidate = cls._candidate_values(
                base,
                fighter_mean=fighter_mean,
                fighter_lower=fighter_lower,
                fighter_upper=fighter_upper,
                fighter_logit_location=fighter_location,
                logit_scale=logit_scale,
            )
        if base.paper_action == "pass":
            filtered_action = "pass"
            filter_status = "base_policy_pass"
        elif status != "paper_only_challenger":
            filtered_action = "pass"
            filter_status = "bayesian_status_veto"
        elif candidate is not None and candidate[2] < minimum_mean_ev:
            filtered_action = "pass"
            filter_status = "bayesian_mean_ev_veto"
        elif candidate is not None and candidate[5] < minimum_probability:
            filtered_action = "pass"
            filter_status = "bayesian_probability_veto"
        else:
            filtered_action = base.paper_action
            filter_status = "qualified"
        body: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "betting_status": BETTING_STATUS,
            "paper_only": True,
            "execution_enabled": False,
            "policy_version": BAYESIAN_FILTER_POLICY_VERSION,
            "base_decision_id": base.decision_id,
            "capture_id": base.capture_id,
            "matchup_id": base.matchup_id,
            "fight_id": base.fight_id,
            "event_id": base.event_id,
            "fighter_id": base.fighter_id,
            "opponent_id": base.opponent_id,
            "event_date": base.event_date,
            "decision_issued_at_utc": base.decision_issued_at_utc,
            "reference_quote_id": base.reference_quote_id,
            "source_vegas_sha256": validated_sha256(
                source_vegas_sha256, "source_vegas_sha256"
            ),
            "bayesian_artifact_sha256": validated_sha256(
                bayesian_artifact_sha256, "bayesian_artifact_sha256"
            ),
            "bayesian_model_id": model_id,
            "bayesian_status": status,
            "credible_level": credible,
            "fighter_posterior_mean": fighter_mean,
            "fighter_posterior_median": fighter_median,
            "fighter_probability_lower": fighter_lower,
            "fighter_probability_upper": fighter_upper,
            "fighter_calibrated_logit_location": fighter_location,
            "calibrated_logit_scale": logit_scale,
            "minimum_mean_expected_return": minimum_mean_ev,
            "minimum_probability_positive_expected_return": minimum_probability,
            "base_paper_action": base.paper_action,
            "base_expected_return": base_expected_return,
            "candidate_moneyline": candidate[0] if candidate else None,
            "candidate_posterior_mean_probability": (
                candidate[1] if candidate else None
            ),
            "candidate_posterior_mean_expected_return": (
                candidate[2] if candidate else None
            ),
            "candidate_expected_return_lower": candidate[3] if candidate else None,
            "candidate_expected_return_upper": candidate[4] if candidate else None,
            "candidate_probability_positive_expected_return": (
                candidate[5] if candidate else None
            ),
            "filtered_paper_action": filtered_action,
            "filter_status": filter_status,
            "hypothetical_risk_units": 1.0 if filter_status == "qualified" else 0.0,
        }
        return cls(filtered_decision_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "BayesianFilteredDecision":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                "Bayesian filtered decision schema mismatch; "
                f"missing={missing}, extra={extra}"
            )
        try:
            decision = cls(**{field: record[field] for field in cls.FIELDNAMES})
        except TypeError as error:
            raise MarketDataError("invalid Bayesian filtered decision fields") from error
        decision.validate_integrity()
        return decision

    @property
    def natural_key(self) -> tuple[str]:
        return (self.base_decision_id,)

    def validate_integrity(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MarketDataError("unsupported Bayesian filtered decision schema")
        if (
            self.betting_status != BETTING_STATUS
            or self.paper_only is not True
            or self.execution_enabled is not False
        ):
            raise StoreIntegrityError("Bayesian filter must remain paper-only")
        if self.policy_version != BAYESIAN_FILTER_POLICY_VERSION:
            raise MarketDataError("unsupported Bayesian filtered decision policy")
        validated_sha256(self.source_vegas_sha256, "source_vegas_sha256")
        validated_sha256(
            self.bayesian_artifact_sha256, "bayesian_artifact_sha256"
        )
        if not str(self.bayesian_model_id).strip() or not str(
            self.bayesian_status
        ).strip():
            raise MarketDataError("Bayesian filter model ID/status is blank")
        numeric = (
            self.credible_level,
            self.fighter_posterior_mean,
            self.fighter_posterior_median,
            self.fighter_probability_lower,
            self.fighter_probability_upper,
            self.fighter_calibrated_logit_location,
            self.calibrated_logit_scale,
            self.minimum_mean_expected_return,
            self.minimum_probability_positive_expected_return,
            self.base_expected_return,
            self.hypothetical_risk_units,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise MarketDataError("Bayesian filtered decision has non-finite values")
        if not (
            0.0 < float(self.fighter_probability_lower)
            <= float(self.fighter_posterior_median)
            <= float(self.fighter_probability_upper)
            < 1.0
            and 0.0 < float(self.fighter_posterior_mean) < 1.0
            and 0.0 < float(self.credible_level) < 1.0
            and float(self.calibrated_logit_scale) >= 0.0
            and float(self.minimum_mean_expected_return) >= 0.0
            and 0.0
            < float(self.minimum_probability_positive_expected_return)
            <= 1.0
        ):
            raise MarketDataError("Bayesian filtered probability contract is invalid")
        median_from_logit = _sigmoid(
            float(self.fighter_calibrated_logit_location)
        )
        tail = (1.0 - float(self.credible_level)) / 2.0
        z_value = _STANDARD_NORMAL.inv_cdf(1.0 - tail)
        lower_from_logit = _sigmoid(
            float(self.fighter_calibrated_logit_location)
            - z_value * float(self.calibrated_logit_scale)
        )
        upper_from_logit = _sigmoid(
            float(self.fighter_calibrated_logit_location)
            + z_value * float(self.calibrated_logit_scale)
        )
        if any(
            # vegas_odds.json is a compact pandas publication and rounds
            # floating values to ten decimal places before the T-24 capture.
            abs(float(supplied) - expected) > 1e-8
            for supplied, expected in (
                (self.fighter_posterior_median, median_from_logit),
                (self.fighter_probability_lower, lower_from_logit),
                (self.fighter_probability_upper, upper_from_logit),
            )
        ):
            raise StoreIntegrityError(
                "Bayesian posterior interval disagrees with its logit distribution"
            )
        body = self.to_mapping()
        body.pop("filtered_decision_id")
        if self.filtered_decision_id != canonical_hash(body):
            raise StoreIntegrityError(
                "filtered_decision_id does not match canonical contents"
            )
        if self.filtered_paper_action not in {"fighter", "opponent", "pass"}:
            raise MarketDataError("unsupported Bayesian filtered paper action")
        if self.base_paper_action not in {"fighter", "opponent", "pass"}:
            raise MarketDataError("unsupported base paper action")
        if self.filter_status not in {
            "base_policy_pass",
            "bayesian_status_veto",
            "bayesian_mean_ev_veto",
            "bayesian_probability_veto",
            "qualified",
        }:
            raise MarketDataError("unsupported Bayesian filter status")
        if self.filter_status == "qualified":
            if (
                self.filtered_paper_action != self.base_paper_action
                or self.base_paper_action == "pass"
                or float(self.hypothetical_risk_units) != 1.0
            ):
                raise StoreIntegrityError("qualified Bayesian filter changed its base action")
        elif (
            self.filtered_paper_action != "pass"
            or float(self.hypothetical_risk_units) != 0.0
        ):
            raise StoreIntegrityError("Bayesian filter veto carries hypothetical risk")
        candidate_fields = (
            self.candidate_moneyline,
            self.candidate_posterior_mean_probability,
            self.candidate_posterior_mean_expected_return,
            self.candidate_expected_return_lower,
            self.candidate_expected_return_upper,
            self.candidate_probability_positive_expected_return,
        )
        if self.base_paper_action == "pass":
            if any(value is not None for value in candidate_fields) or self.filter_status != "base_policy_pass":
                raise StoreIntegrityError(
                    "base-policy pass contains a Bayesian price candidate"
                )
            return
        if any(value is None for value in candidate_fields):
            raise StoreIntegrityError(
                "base-policy selection lacks Bayesian candidate values"
            )
        line = moneyline(self.candidate_moneyline, "candidate_moneyline")
        fighter_selected = self.base_paper_action == "fighter"
        mean_probability = (
            float(self.fighter_posterior_mean)
            if fighter_selected
            else 1.0 - float(self.fighter_posterior_mean)
        )
        lower_probability = (
            float(self.fighter_probability_lower)
            if fighter_selected
            else 1.0 - float(self.fighter_probability_upper)
        )
        upper_probability = (
            float(self.fighter_probability_upper)
            if fighter_selected
            else 1.0 - float(self.fighter_probability_lower)
        )
        location = (
            float(self.fighter_calibrated_logit_location)
            if fighter_selected
            else -float(self.fighter_calibrated_logit_location)
        )
        decimal_return = 1.0 + _profit_for_one_unit_risk(line)
        break_even = implied_probability(line)
        if float(self.calibrated_logit_scale) == 0.0:
            probability_positive = float(mean_probability > break_even)
        else:
            threshold_logit = math.log(break_even / (1.0 - break_even))
            probability_positive = 1.0 - _STANDARD_NORMAL.cdf(
                (threshold_logit - location)
                / float(self.calibrated_logit_scale)
            )
        expected_values = (
            mean_probability,
            decimal_return * mean_probability - 1.0,
            decimal_return * lower_probability - 1.0,
            decimal_return * upper_probability - 1.0,
            probability_positive,
        )
        supplied_values = candidate_fields[1:]
        if any(
            abs(float(supplied) - expected) > 1e-12
            for supplied, expected in zip(supplied_values, expected_values)
        ):
            raise StoreIntegrityError(
                "Bayesian candidate values are not reproducible"
            )
        if self.bayesian_status != "paper_only_challenger":
            expected_status = "bayesian_status_veto"
        elif expected_values[1] < float(self.minimum_mean_expected_return):
            expected_status = "bayesian_mean_ev_veto"
        elif expected_values[4] < float(
            self.minimum_probability_positive_expected_return
        ):
            expected_status = "bayesian_probability_veto"
        else:
            expected_status = "qualified"
        if self.filter_status != expected_status:
            raise StoreIntegrityError(
                "Bayesian filter status does not follow its frozen thresholds"
            )

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


@dataclass(frozen=True)
class PaperSettlement:
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
    target: int | None
    settlement_status: str
    hypothetical_risk_units: float
    hypothetical_profit_units: float
    forecast_log_loss: float | None
    forecast_brier_score: float | None
    forecast_accuracy_credit: float | None

    FIELDNAMES: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "settlement_id",
        "betting_status",
        "paper_only",
        "decision_id",
        "capture_id",
        "matchup_id",
        "fight_id",
        "event_id",
        "settled_at_utc",
        "result_source_sha256",
        "target",
        "settlement_status",
        "hypothetical_risk_units",
        "hypothetical_profit_units",
        "forecast_log_loss",
        "forecast_brier_score",
        "forecast_accuracy_credit",
    )

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "PaperSettlement":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"paper settlement schema mismatch; missing={missing}, extra={extra}"
            )
        try:
            settlement = cls(**{field: record[field] for field in cls.FIELDNAMES})
        except TypeError as error:
            raise MarketDataError("invalid paper settlement fields") from error
        settlement.validate_integrity()
        return settlement

    @property
    def natural_key(self) -> tuple[str]:
        return (self.decision_id,)

    def validate_integrity(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise MarketDataError("unsupported paper settlement schema version")
        if self.betting_status != BETTING_STATUS or self.paper_only is not True:
            raise StoreIntegrityError("paper settlement must keep execution disabled")
        body = self.to_mapping()
        body.pop("settlement_id")
        if self.settlement_id != canonical_hash(body):
            raise StoreIntegrityError("settlement_id does not match canonical contents")
        validated_sha256(self.result_source_sha256, "result_source_sha256")
        utc_datetime(self.settled_at_utc, "settled_at_utc")
        if self.target is not None:
            binary_target(self.target)
        numeric = (
            self.hypothetical_risk_units,
            self.hypothetical_profit_units,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise MarketDataError("paper settlement contains a non-finite number")
        allowed_statuses = {
            "paper_win",
            "paper_loss",
            "pass",
            "void",
            "pass_unscored",
        }
        if self.settlement_status not in allowed_statuses:
            raise MarketDataError("unsupported paper settlement_status")
        unscored = self.target is None
        if unscored != (self.forecast_log_loss is None):
            raise StoreIntegrityError("settlement forecast score presence is inconsistent")
        if unscored != (self.forecast_brier_score is None):
            raise StoreIntegrityError("settlement Brier score presence is inconsistent")
        if unscored != (self.forecast_accuracy_credit is None):
            raise StoreIntegrityError("settlement accuracy presence is inconsistent")
        if unscored and self.settlement_status not in {"void", "pass_unscored"}:
            raise StoreIntegrityError("unscored settlement has a scored status")
        if not unscored and self.settlement_status in {"void", "pass_unscored"}:
            raise StoreIntegrityError("scored settlement has an unscored status")
        if self.settlement_status in {"paper_win", "paper_loss"}:
            if float(self.hypothetical_risk_units) != 1.0:
                raise StoreIntegrityError("paper selection must risk exactly one unit")
        elif float(self.hypothetical_risk_units) != 0.0:
            raise StoreIntegrityError("pass/void settlement cannot carry risk")
        if unscored:
            if float(self.hypothetical_profit_units) != 0.0:
                raise StoreIntegrityError("unscored settlement cannot carry profit")
        else:
            scores = (
                float(self.forecast_log_loss),
                float(self.forecast_brier_score),
                float(self.forecast_accuracy_credit),
            )
            if any(not math.isfinite(value) for value in scores):
                raise MarketDataError("paper settlement has a non-finite score")
            if scores[0] < 0.0 or not 0.0 <= scores[1] <= 1.0:
                raise MarketDataError("paper settlement forecast score is out of range")
            if scores[2] not in {0.0, 0.5, 1.0}:
                raise MarketDataError("paper settlement accuracy credit is invalid")
            if self.settlement_status == "paper_loss" and float(
                self.hypothetical_profit_units
            ) != -1.0:
                raise StoreIntegrityError("paper loss must lose one hypothetical unit")
            if self.settlement_status == "paper_win" and float(
                self.hypothetical_profit_units
            ) <= 0.0:
                raise StoreIntegrityError("paper win must have positive hypothetical profit")
            if self.settlement_status == "pass" and float(
                self.hypothetical_profit_units
            ) != 0.0:
                raise StoreIntegrityError("paper pass cannot carry profit")

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


def _profit_for_one_unit_risk(moneyline: int) -> float:
    return moneyline / 100.0 if moneyline > 0 else 100.0 / abs(moneyline)


def settle_paper_decision(
    decision: PaperDecision,
    *,
    target: object | None,
    settled_at_utc: datetime | str,
    result_source_sha256: object,
    fight_id: object | None = None,
) -> PaperSettlement:
    """Score a paper record; ``target=None`` represents D/NC/cancel/unknown."""

    if not isinstance(decision, PaperDecision):
        raise TypeError("decision must be a PaperDecision")
    if decision.betting_status != BETTING_STATUS or not decision.paper_only:
        raise StoreIntegrityError("refusing to settle a record not marked paper-only")
    settled = utc_datetime(settled_at_utc, "settled_at_utc")
    if decision.timing_precision == "timestamp":
        event_start = utc_datetime(decision.event_start_utc, "event_start_utc")
        if settled < event_start:
            raise MarketDataError("settled_at_utc cannot precede event_start_utc")
    elif settled.date() <= date.fromisoformat(decision.event_date):
        raise MarketDataError(
            "date-only events can be settled only on a later UTC date"
        )
    outcome = None if target is None else binary_target(target)
    resolved_fight_id = optional_stable_id(fight_id, "fight_id")
    if (
        decision.fight_id is not None
        and resolved_fight_id is not None
        and decision.fight_id != resolved_fight_id
    ):
        raise StoreIntegrityError("settlement fight_id conflicts with the decision")
    resolved_fight_id = resolved_fight_id or decision.fight_id
    if outcome is None:
        status = "void" if decision.paper_action != "pass" else "pass_unscored"
        risk = profit = 0.0
        log_loss = brier = accuracy_credit = None
    else:
        probability_of_target = (
            decision.blend_probability
            if outcome == 1
            else 1.0 - decision.blend_probability
        )
        log_loss = -math.log(probability_of_target)
        brier = (decision.blend_probability - outcome) ** 2
        accuracy_credit = (
            0.5
            if decision.blend_probability == 0.5
            else float(
                (decision.blend_probability > 0.5) == bool(outcome)
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
        "target": outcome,
        "settlement_status": status,
        "hypothetical_risk_units": risk,
        "hypothetical_profit_units": profit,
        "forecast_log_loss": log_loss,
        "forecast_brier_score": brier,
        "forecast_accuracy_credit": accuracy_credit,
    }
    return PaperSettlement(settlement_id=canonical_hash(body), **body)


@dataclass(frozen=True)
class PaperMetrics:
    betting_status: str
    paper_only: bool
    decisions: int
    scored_forecasts: int
    paper_selections: int
    passes: int
    voids: int
    wins: int
    losses: int
    selection_coverage: float | None
    forecast_metrics: ForecastMetrics | None
    hypothetical_risk_units: float
    hypothetical_profit_units: float
    hypothetical_roi: float | None
    hypothetical_max_drawdown_units: float

    def to_mapping(self) -> dict[str, object]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result["forecast_metrics"] = (
            self.forecast_metrics.to_mapping() if self.forecast_metrics else None
        )
        return result


def summarize_paper_settlements(
    decisions: Iterable[PaperDecision],
    settlements: Iterable[PaperSettlement],
) -> PaperMetrics:
    decision_records = tuple(decisions)
    settlement_records = tuple(settlements)
    decision_by_id: dict[str, PaperDecision] = {}
    for decision in decision_records:
        if not isinstance(decision, PaperDecision):
            raise TypeError("decisions must contain PaperDecision instances")
        decision.validate_integrity()
        if decision.decision_id in decision_by_id:
            raise StoreIntegrityError("duplicate decision_id in paper summary")
        if decision.betting_status != BETTING_STATUS or not decision.paper_only:
            raise StoreIntegrityError("paper summary contains a non-paper decision")
        decision_by_id[decision.decision_id] = decision
    settlement_by_decision: dict[str, PaperSettlement] = {}
    for settlement in settlement_records:
        if not isinstance(settlement, PaperSettlement):
            raise TypeError("settlements must contain PaperSettlement instances")
        if settlement.decision_id not in decision_by_id:
            raise StoreIntegrityError("paper settlement has no matching decision")
        if settlement.decision_id in settlement_by_decision:
            raise StoreIntegrityError("one paper decision has multiple settlements")
        if settlement.betting_status != BETTING_STATUS or not settlement.paper_only:
            raise StoreIntegrityError("paper summary contains a non-paper settlement")
        settlement.validate_integrity()
        decision = decision_by_id[settlement.decision_id]
        expected = settle_paper_decision(
            decision,
            target=settlement.target,
            settled_at_utc=settlement.settled_at_utc,
            result_source_sha256=settlement.result_source_sha256,
            fight_id=settlement.fight_id,
        )
        if expected != settlement:
            raise StoreIntegrityError(
                "paper settlement contents disagree with its matching decision"
            )
        settlement_by_decision[settlement.decision_id] = settlement

    scored = [item for item in settlement_records if item.target is not None]
    if scored:
        scored_probabilities = [
            decision_by_id[item.decision_id].blend_probability for item in scored
        ]
        scored_targets = [item.target for item in scored]
        metrics = forecast_metrics(scored_probabilities, scored_targets)
    else:
        metrics = None
    wins = sum(item.settlement_status == "paper_win" for item in settlement_records)
    losses = sum(item.settlement_status == "paper_loss" for item in settlement_records)
    voids = sum(item.settlement_status == "void" for item in settlement_records)
    passes = sum(
        item.settlement_status in {"pass", "pass_unscored"}
        for item in settlement_records
    )
    selections = wins + losses + voids
    risk = sum(item.hypothetical_risk_units for item in settlement_records)
    profit = sum(item.hypothetical_profit_units for item in settlement_records)
    cumulative = peak = max_drawdown = 0.0
    for settlement in sorted(
        settlement_records, key=lambda item: (item.settled_at_utc, item.settlement_id)
    ):
        cumulative += settlement.hypothetical_profit_units
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return PaperMetrics(
        betting_status=BETTING_STATUS,
        paper_only=True,
        decisions=len(decision_records),
        scored_forecasts=len(scored),
        paper_selections=selections,
        passes=passes,
        voids=voids,
        wins=wins,
        losses=losses,
        selection_coverage=((wins + losses) / len(scored) if scored else None),
        forecast_metrics=metrics,
        hypothetical_risk_units=risk,
        hypothetical_profit_units=profit,
        hypothetical_roi=(profit / risk if risk else None),
        hypothetical_max_drawdown_units=max_drawdown,
    )


class _PaperRecordStore:
    """Atomic JSONL authority plus an exactly checked CSV audit mirror."""

    def __init__(
        self,
        csv_path: str | Path,
        jsonl_path: str | Path,
        *,
        record_type: (
            type[PaperDecision]
            | type[PaperSettlement]
            | type[BayesianFilteredDecision]
        ),
        id_field: str,
        time_field: str,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        if self.csv_path.resolve() == self.jsonl_path.resolve():
            raise ValueError("csv_path and jsonl_path must be different")
        self.record_type = record_type
        self.id_field = id_field
        self.time_field = time_field
        self.fieldnames = record_type.FIELDNAMES

    def _validate_records(
        self, records: Iterable[PaperDecision | PaperSettlement]
    ) -> dict[str, PaperDecision | PaperSettlement | BayesianFilteredDecision]:
        indexed: dict[
            str, PaperDecision | PaperSettlement | BayesianFilteredDecision
        ] = {}
        natural: dict[tuple, str] = {}
        for record in records:
            if not isinstance(record, self.record_type):
                raise TypeError(
                    f"store accepts {self.record_type.__name__} instances only"
                )
            record.validate_integrity()
            record_id = getattr(record, self.id_field)
            existing = indexed.get(record_id)
            if existing is not None and existing != record:
                raise StoreIntegrityError(f"{self.id_field} was rewritten")
            prior_id = natural.get(record.natural_key)
            if prior_id is not None and prior_id != record_id:
                raise StoreIntegrityError("an immutable paper ledger key was rewritten")
            indexed[record_id] = record
            natural[record.natural_key] = record_id
        return indexed

    def _read_jsonl(
        self,
    ) -> list[PaperDecision | PaperSettlement | BayesianFilteredDecision]:
        if not self.jsonl_path.exists():
            return []
        records: list[PaperDecision | PaperSettlement | BayesianFilteredDecision] = []
        with self.jsonl_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise StoreIntegrityError(
                        f"blank paper JSONL record at line {line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StoreIntegrityError(
                        f"invalid paper JSONL at line {line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise StoreIntegrityError(
                        f"paper JSONL line {line_number} is not an object"
                    )
                records.append(self.record_type.from_mapping(value))
        self._validate_records(records)
        return records

    def _read_csv_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != self.fieldnames:
                raise StoreIntegrityError("paper CSV columns do not match the schema")
            return list(reader)

    def _render_jsonl(
        self,
        records: Iterable[PaperDecision | PaperSettlement | BayesianFilteredDecision],
    ) -> str:
        return "".join(f"{canonical_json(record.to_mapping())}\n" for record in records)

    def _render_csv(
        self,
        records: Iterable[PaperDecision | PaperSettlement | BayesianFilteredDecision],
    ) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=self.fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_mapping())
        return output.getvalue()

    def read(self) -> tuple[PaperDecision | PaperSettlement, ...]:
        records = self._read_jsonl()
        csv_rows = self._read_csv_rows()
        if not records:
            if csv_rows:
                raise StoreIntegrityError(
                    "paper CSV has records but authoritative JSONL is missing"
                )
            return ()
        if not self.csv_path.exists():
            return tuple(records)
        expected_reader = csv.DictReader(io.StringIO(self._render_csv(records)))
        expected_rows = list(expected_reader)
        if len(csv_rows) > len(expected_rows) or csv_rows != expected_rows[: len(csv_rows)]:
            raise StoreIntegrityError("paper CSV and JSONL mirrors diverged")
        return tuple(records)

    def append(
        self,
        pending_records: Iterable[
            PaperDecision | PaperSettlement | BayesianFilteredDecision
        ],
    ) -> AppendResult:
        pending = tuple(pending_records)
        lock_path = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock_path):
            existing = list(self.read())
            indexed = self._validate_records(existing)
            natural = {
                record.natural_key: getattr(record, self.id_field)
                for record in existing
            }
            additions: list[
                PaperDecision | PaperSettlement | BayesianFilteredDecision
            ] = []
            duplicates: list[str] = []
            for record in pending:
                if not isinstance(record, self.record_type):
                    raise TypeError(
                        f"append accepts {self.record_type.__name__} instances only"
                    )
                record.validate_integrity()
                record_id = getattr(record, self.id_field)
                if record_id in indexed:
                    if indexed[record_id] != record:
                        raise StoreIntegrityError(f"{self.id_field} was rewritten")
                    duplicates.append(record_id)
                    continue
                prior_id = natural.get(record.natural_key)
                if prior_id is not None and prior_id != record_id:
                    raise StoreIntegrityError("an immutable paper ledger key was rewritten")
                indexed[record_id] = record
                natural[record.natural_key] = record_id
                additions.append(record)
            additions.sort(
                key=lambda record: (
                    getattr(record, self.time_field),
                    getattr(record, self.id_field),
                )
            )
            combined = [*existing, *additions]
            atomic_write_text(self.jsonl_path, self._render_jsonl(combined))
            atomic_write_text(self.csv_path, self._render_csv(combined))
            return AppendResult(
                added_ids=tuple(getattr(record, self.id_field) for record in additions),
                duplicate_ids=tuple(duplicates),
                total_records=len(combined),
                dataset_sha256=canonical_hash(
                    [record.to_mapping() for record in combined]
                ),
            )


class PaperDecisionStore(_PaperRecordStore):
    """Append-only atomic store for prospective paper decisions."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        super().__init__(
            csv_path,
            jsonl_path,
            record_type=PaperDecision,
            id_field="decision_id",
            time_field="decision_issued_at_utc",
        )

    def read(self) -> tuple[PaperDecision, ...]:
        return tuple(super().read())  # type: ignore[return-value]


class PaperSettlementStore(_PaperRecordStore):
    """Append-only atomic store allowing exactly one settlement per decision."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        super().__init__(
            csv_path,
            jsonl_path,
            record_type=PaperSettlement,
            id_field="settlement_id",
            time_field="settled_at_utc",
        )

    def read(self) -> tuple[PaperSettlement, ...]:
        return tuple(super().read())  # type: ignore[return-value]


class BayesianFilteredDecisionStore(_PaperRecordStore):
    """Append-only T-24 ledger for the Bayesian veto policy."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        super().__init__(
            csv_path,
            jsonl_path,
            record_type=BayesianFilteredDecision,
            id_field="filtered_decision_id",
            time_field="decision_issued_at_utc",
        )

    def read(self) -> tuple[BayesianFilteredDecision, ...]:
        return tuple(super().read())  # type: ignore[return-value]
