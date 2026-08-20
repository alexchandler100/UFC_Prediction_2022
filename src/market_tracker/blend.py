"""Leakage-safe evaluation of a symmetric market/stats probability blend.

The blend has one fitted parameter and no intercept. Selection uses only
settled cards on strictly earlier calendar dates; all cards on the same date
are held out as one batch. This is intentionally conservative when UFCStats
provides an event date but no exact start timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from ._common import (
    MarketDataError,
    SCHEMA_VERSION,
    StoreIntegrityError,
    binary_target,
    canonical_hash,
    optional_stable_id,
    probability,
    require_before_event,
    stable_id,
    utc_datetime,
)
from .forecasts import ForecastCapture
from .quotes import MarketConsensus


DEFAULT_GAMMA_GRID = tuple(index / 20.0 for index in range(21))
_EPSILON = 1e-15


def _logit(value: float) -> float:
    return math.log(value) - math.log1p(-value)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def symmetric_logit_blend(
    market_probability: object,
    model_probability: object,
    gamma: object,
) -> float:
    """Interpolate market and model log odds without breaking side symmetry."""

    market = probability(market_probability, "market_probability")
    model = probability(model_probability, "model_probability")
    try:
        weight = float(gamma)
    except (TypeError, ValueError) as error:
        raise MarketDataError("gamma must be numeric") from error
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise MarketDataError("gamma must be finite and between zero and one")
    # Always perform the transcendental calculation on one deterministic side
    # of the matchup, then complement for the other side. Besides preserving
    # the no-intercept equation, this makes ordinary swapped floating-point
    # calls exact complements rather than merely numerically close.
    complement_result = market > 0.5 or (market == 0.5 and model > 0.5)
    calculation_market = 1.0 - market if complement_result else market
    calculation_model = 1.0 - model if complement_result else model
    blended_logit = _logit(calculation_market) + weight * (
        _logit(calculation_model) - _logit(calculation_market)
    )
    result = min(max(_sigmoid(blended_logit), _EPSILON), 1.0 - _EPSILON)
    return 1.0 - result if complement_result else result


@dataclass(frozen=True)
class BlendObservation:
    """A settled outcome joined to separately frozen market/model captures."""

    schema_version: int
    observation_id: str
    capture_id: str
    matchup_id: str
    fight_id: str | None
    event_id: str
    fighter_id: str
    opponent_id: str
    event_date: str
    timing_precision: str
    event_start_utc: str | None
    blend_issued_at_utc: str
    market_as_of_utc: str
    market_observed_at_utc: str
    forecast_issued_at_utc: str
    market_probability: float
    model_probability: float
    target: int
    market_consensus_id: str
    forecast_capture_id: str
    model_id: str
    model_version: str
    model_trained_through: str
    model_training_cutoff_precision: str
    probability_provenance: str
    source_commit_sha: str

    @classmethod
    def from_captures(
        cls,
        market: MarketConsensus,
        forecast: ForecastCapture,
        *,
        target: object,
        fight_id: object | None = None,
    ) -> "BlendObservation":
        if not isinstance(market, MarketConsensus):
            raise TypeError("market must be a MarketConsensus")
        if not isinstance(forecast, ForecastCapture):
            raise TypeError("forecast must be a ForecastCapture")
        market_identity = (
            market.capture_id,
            market.matchup_id,
            market.event_id,
            market.fighter_id,
            market.opponent_id,
            market.event_date,
            market.timing_precision,
            market.event_start_utc,
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
        if market_identity != forecast_identity:
            raise StoreIntegrityError(
                "market consensus and forecast capture identify different matchups"
            )
        known_fight_ids = {
            value
            for value in (
                market.fight_id,
                forecast.fight_id,
                optional_stable_id(fight_id, "fight_id"),
            )
            if value is not None
        }
        if len(known_fight_ids) > 1:
            raise StoreIntegrityError("matchup resolves to conflicting fight IDs")
        body = {
            "schema_version": SCHEMA_VERSION,
            "capture_id": market.capture_id,
            "matchup_id": market.matchup_id,
            "fight_id": next(iter(known_fight_ids), None),
            "event_id": market.event_id,
            "fighter_id": market.fighter_id,
            "opponent_id": market.opponent_id,
            "event_date": market.event_date,
            "timing_precision": market.timing_precision,
            "event_start_utc": market.event_start_utc,
            "blend_issued_at_utc": max(
                market.as_of_utc, forecast.forecast_issued_at_utc
            ),
            "market_as_of_utc": market.as_of_utc,
            "market_observed_at_utc": market.latest_observed_at_utc,
            "forecast_issued_at_utc": forecast.forecast_issued_at_utc,
            "market_probability": market.no_vig_fighter_probability,
            "model_probability": forecast.model_probability,
            "target": binary_target(target),
            "market_consensus_id": market.consensus_id,
            "forecast_capture_id": forecast.forecast_capture_id,
            "model_id": forecast.model_id,
            "model_version": forecast.model_version,
            "model_trained_through": forecast.model_trained_through,
            "model_training_cutoff_precision": (
                forecast.model_training_cutoff_precision
            ),
            "probability_provenance": forecast.probability_provenance,
            "source_commit_sha": forecast.source_commit_sha,
        }
        return cls(observation_id=canonical_hash(body), **body)

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def select_latest_observations_by_horizon(
    observations: Iterable[BlendObservation],
    event_cutoffs_utc: Mapping[object, object],
) -> tuple[BlendObservation, ...]:
    """Select one capture per matchup using predeclared event cutoffs.

    This is the required outcome-independent bridge between an append-only
    ledger (which may contain retries) and walk-forward scoring. A cutoff must
    be supplied for every event, and a matchup with no capture at or before its
    cutoff fails closed instead of silently selecting a later snapshot.
    """

    records = tuple(observations)
    if not records:
        raise ValueError("at least one BlendObservation is required")
    if any(not isinstance(record, BlendObservation) for record in records):
        raise TypeError("observations must contain BlendObservation instances")
    cutoffs: dict[str, object] = {}
    for raw_event_id, raw_cutoff in event_cutoffs_utc.items():
        event_id = stable_id(raw_event_id, "event cutoff ID")
        if event_id in cutoffs:
            raise MarketDataError(f"duplicate cutoff for event_id {event_id}")
        cutoffs[event_id] = raw_cutoff

    grouped: dict[str, list[BlendObservation]] = {}
    for record in records:
        grouped.setdefault(record.matchup_id, []).append(record)

    selected: list[BlendObservation] = []
    for matchup_id, candidates in sorted(grouped.items()):
        identities = {
            (
                item.event_id,
                item.fighter_id,
                item.opponent_id,
                item.event_date,
                item.timing_precision,
                item.event_start_utc,
                item.fight_id,
                item.target,
            )
            for item in candidates
        }
        if len(identities) != 1:
            raise StoreIntegrityError(
                "captures for one matchup disagree on identity, timing, or outcome"
            )
        event_id = candidates[0].event_id
        if event_id not in cutoffs:
            raise MarketDataError(f"missing predeclared cutoff for event_id {event_id}")
        cutoff, _, _, _ = require_before_event(
            cutoffs[event_id],
            event_date=candidates[0].event_date,
            timing_precision=candidates[0].timing_precision,
            event_start_utc=candidates[0].event_start_utc,
            observed_field="event_cutoff_utc",
        )
        # Both inputs must have existed by the predeclared decision horizon.
        # A quote observed before the cutoff cannot make a model forecast issued
        # afterwards available retroactively.
        eligible = [
            item
            for item in candidates
            if utc_datetime(item.blend_issued_at_utc, "blend_issued_at_utc")
            <= cutoff
        ]
        if not eligible:
            raise MarketDataError(
                f"no capture for matchup_id {matchup_id} was available by its cutoff"
            )
        selected.append(
            max(
                eligible,
                key=lambda item: (
                    item.blend_issued_at_utc,
                    item.market_as_of_utc,
                    item.capture_id,
                    item.observation_id,
                ),
            )
        )
    return tuple(
        sorted(selected, key=lambda item: (item.event_date, item.event_id, item.matchup_id))
    )


@dataclass(frozen=True)
class ForecastMetrics:
    count: int
    positive_rate: float
    accuracy: float
    log_loss: float
    brier_score: float
    roc_auc: float | None
    expected_calibration_error: float

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _roc_auc(probabilities: Sequence[float], targets: Sequence[int]) -> float | None:
    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ordered = sorted(zip(probabilities, targets), key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(
            target for _, target in ordered[index:end]
        )
        index = end
    return (
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def forecast_metrics(
    probabilities: Iterable[object],
    targets: Iterable[object],
    *,
    calibration_bins: int = 10,
) -> ForecastMetrics:
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least two")
    parsed_probabilities = [probability(value, "probability") for value in probabilities]
    parsed_targets = [binary_target(value) for value in targets]
    if len(parsed_probabilities) != len(parsed_targets):
        raise ValueError("probabilities and targets must have the same length")
    if not parsed_targets:
        raise ValueError("at least one forecast is required")
    count = len(parsed_targets)
    log_loss = -sum(
        target * math.log(prediction)
        + (1 - target) * math.log1p(-prediction)
        for prediction, target in zip(parsed_probabilities, parsed_targets)
    ) / count
    brier = sum(
        (prediction - target) ** 2
        for prediction, target in zip(parsed_probabilities, parsed_targets)
    ) / count
    correct = sum(
        0.5 if prediction == 0.5 else float((prediction > 0.5) == bool(target))
        for prediction, target in zip(parsed_probabilities, parsed_targets)
    )
    bin_counts = [0] * calibration_bins
    bin_probability_sums = [0.0] * calibration_bins
    bin_target_sums = [0.0] * calibration_bins
    for prediction, target in zip(parsed_probabilities, parsed_targets):
        bin_index = min(int(prediction * calibration_bins), calibration_bins - 1)
        bin_counts[bin_index] += 1
        bin_probability_sums[bin_index] += prediction
        bin_target_sums[bin_index] += target
    calibration_error = sum(
        (bin_count / count)
        * abs(
            bin_probability_sums[index] / bin_count
            - bin_target_sums[index] / bin_count
        )
        for index, bin_count in enumerate(bin_counts)
        if bin_count
    )
    return ForecastMetrics(
        count=count,
        positive_rate=sum(parsed_targets) / count,
        accuracy=correct / count,
        log_loss=log_loss,
        brier_score=brier,
        roc_auc=_roc_auc(parsed_probabilities, parsed_targets),
        expected_calibration_error=calibration_error,
    )


@dataclass(frozen=True)
class BlendPrediction:
    prediction_id: str
    observation_id: str
    capture_id: str
    matchup_id: str
    fight_id: str | None
    event_id: str
    event_date: str
    status: str
    prior_card_count: int
    prior_fight_count: int
    selection_training_through_event_date: str | None
    selected_gamma: float | None
    selection_prior_card_log_loss: float | None
    market_probability: float
    model_probability: float
    blend_probability: float | None
    target: int
    market_consensus_id: str
    forecast_capture_id: str

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class BlendEvaluation:
    schema_version: int
    evaluation_id: str
    input_sha256: str
    gamma_grid: tuple[float, ...]
    min_prior_cards: int
    min_prior_fights: int
    lookback_cards: int | None
    predictions: tuple[BlendPrediction, ...]
    evaluated_fights: int
    skipped_fights: int
    market_metrics: ForecastMetrics | None
    model_metrics: ForecastMetrics | None
    blend_metrics: ForecastMetrics | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "input_sha256": self.input_sha256,
            "gamma_grid": list(self.gamma_grid),
            "min_prior_cards": self.min_prior_cards,
            "min_prior_fights": self.min_prior_fights,
            "lookback_cards": self.lookback_cards,
            "predictions": [item.to_mapping() for item in self.predictions],
            "evaluated_fights": self.evaluated_fights,
            "skipped_fights": self.skipped_fights,
            "market_metrics": (
                self.market_metrics.to_mapping() if self.market_metrics else None
            ),
            "model_metrics": (
                self.model_metrics.to_mapping() if self.model_metrics else None
            ),
            "blend_metrics": (
                self.blend_metrics.to_mapping() if self.blend_metrics else None
            ),
        }


def _mean_card_log_loss(
    cards: Sequence[Sequence[BlendObservation]], gamma: float
) -> float:
    card_losses: list[float] = []
    for card in cards:
        loss = 0.0
        for observation in card:
            blended = symmetric_logit_blend(
                observation.market_probability,
                observation.model_probability,
                gamma,
            )
            loss -= observation.target * math.log(blended) + (
                1 - observation.target
            ) * math.log1p(-blended)
        card_losses.append(loss / len(card))
    return sum(card_losses) / len(card_losses)


class PriorCardBlendEvaluator:
    """Walk forward by event date, fitting only a symmetric blend weight."""

    def __init__(
        self,
        *,
        gamma_grid: Iterable[object] = DEFAULT_GAMMA_GRID,
        min_prior_cards: int = 12,
        min_prior_fights: int = 100,
        lookback_cards: int | None = 52,
    ) -> None:
        parsed_grid: list[float] = []
        for candidate in gamma_grid:
            try:
                parsed = float(candidate)
            except (TypeError, ValueError) as error:
                raise MarketDataError("gamma_grid values must be numeric") from error
            if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
                raise MarketDataError("gamma_grid values must be between zero and one")
            parsed_grid.append(parsed)
        self.gamma_grid = tuple(sorted(set(parsed_grid)))
        if not self.gamma_grid:
            raise ValueError("gamma_grid must not be empty")
        if min_prior_cards < 1 or min_prior_fights < 1:
            raise ValueError("minimum prior history must be positive")
        if lookback_cards is not None and lookback_cards < min_prior_cards:
            raise ValueError("lookback_cards cannot be smaller than min_prior_cards")
        self.min_prior_cards = min_prior_cards
        self.min_prior_fights = min_prior_fights
        self.lookback_cards = lookback_cards

    @staticmethod
    def _validate(
        observations: Iterable[BlendObservation],
    ) -> tuple[BlendObservation, ...]:
        records = tuple(observations)
        if not records:
            raise ValueError("at least one BlendObservation is required")
        if any(not isinstance(record, BlendObservation) for record in records):
            raise TypeError("evaluate accepts BlendObservation instances only")
        by_matchup: dict[str, BlendObservation] = {}
        event_dates: dict[str, str] = {}
        for record in records:
            if record.matchup_id in by_matchup:
                raise StoreIntegrityError(
                    f"duplicate matchup_id in evaluation: {record.matchup_id}"
                )
            by_matchup[record.matchup_id] = record
            prior_date = event_dates.setdefault(record.event_id, record.event_date)
            if prior_date != record.event_date:
                raise StoreIntegrityError("one event_id has multiple event_date values")
        return tuple(
            sorted(
                records,
                key=lambda item: (item.event_date, item.event_id, item.matchup_id),
            )
        )

    def evaluate(self, observations: Iterable[BlendObservation]) -> BlendEvaluation:
        records = self._validate(observations)
        input_sha256 = canonical_hash([record.to_mapping() for record in records])
        batches: dict[str, dict[str, list[BlendObservation]]] = {}
        for record in records:
            batches.setdefault(record.event_date, {}).setdefault(
                record.event_id, []
            ).append(record)

        completed_cards: list[tuple[str, str, tuple[BlendObservation, ...]]] = []
        predictions: list[BlendPrediction] = []
        for event_day in sorted(batches):
            available = completed_cards[
                -self.lookback_cards if self.lookback_cards is not None else 0 :
            ]
            prior_cards = [card for _, _, card in available]
            prior_fight_count = sum(len(card) for card in prior_cards)
            enough_history = (
                len(prior_cards) >= self.min_prior_cards
                and prior_fight_count >= self.min_prior_fights
            )
            selected_gamma: float | None = None
            selection_loss: float | None = None
            if enough_history:
                selection_loss, selected_gamma = min(
                    (
                        _mean_card_log_loss(prior_cards, gamma),
                        gamma,
                    )
                    for gamma in self.gamma_grid
                )
            training_through = available[-1][0] if available else None
            for event_id in sorted(batches[event_day]):
                card = sorted(
                    batches[event_day][event_id], key=lambda item: item.matchup_id
                )
                for observation in card:
                    blended = (
                        symmetric_logit_blend(
                            observation.market_probability,
                            observation.model_probability,
                            selected_gamma,
                        )
                        if selected_gamma is not None
                        else None
                    )
                    status = (
                        "evaluated" if enough_history else "insufficient_prior_history"
                    )
                    body = {
                        "schema_version": SCHEMA_VERSION,
                        "observation_id": observation.observation_id,
                        "status": status,
                        "prior_card_count": len(prior_cards),
                        "prior_fight_count": prior_fight_count,
                        "selection_training_through_event_date": training_through,
                        "selected_gamma": selected_gamma,
                        "selection_prior_card_log_loss": selection_loss,
                        "blend_probability": blended,
                    }
                    predictions.append(
                        BlendPrediction(
                            prediction_id=canonical_hash(body),
                            observation_id=observation.observation_id,
                            capture_id=observation.capture_id,
                            matchup_id=observation.matchup_id,
                            fight_id=observation.fight_id,
                            event_id=observation.event_id,
                            event_date=observation.event_date,
                            status=status,
                            prior_card_count=len(prior_cards),
                            prior_fight_count=prior_fight_count,
                            selection_training_through_event_date=training_through,
                            selected_gamma=selected_gamma,
                            selection_prior_card_log_loss=selection_loss,
                            market_probability=observation.market_probability,
                            model_probability=observation.model_probability,
                            blend_probability=blended,
                            target=observation.target,
                            market_consensus_id=observation.market_consensus_id,
                            forecast_capture_id=observation.forecast_capture_id,
                        )
                    )
            # Cards on the current date become available only after every
            # current-date prediction is frozen.
            for event_id in sorted(batches[event_day]):
                card_records = tuple(
                    sorted(
                        batches[event_day][event_id],
                        key=lambda item: item.matchup_id,
                    )
                )
                completed_cards.append((event_day, event_id, card_records))

        evaluated = [item for item in predictions if item.status == "evaluated"]
        if evaluated:
            targets = [item.target for item in evaluated]
            market_metrics = forecast_metrics(
                [item.market_probability for item in evaluated], targets
            )
            model_metrics = forecast_metrics(
                [item.model_probability for item in evaluated], targets
            )
            blend_metrics = forecast_metrics(
                [item.blend_probability for item in evaluated], targets
            )
        else:
            market_metrics = model_metrics = blend_metrics = None
        result_body = {
            "schema_version": SCHEMA_VERSION,
            "input_sha256": input_sha256,
            "gamma_grid": self.gamma_grid,
            "min_prior_cards": self.min_prior_cards,
            "min_prior_fights": self.min_prior_fights,
            "lookback_cards": self.lookback_cards,
            "prediction_ids": tuple(item.prediction_id for item in predictions),
        }
        return BlendEvaluation(
            schema_version=SCHEMA_VERSION,
            evaluation_id=canonical_hash(result_body),
            input_sha256=input_sha256,
            gamma_grid=self.gamma_grid,
            min_prior_cards=self.min_prior_cards,
            min_prior_fights=self.min_prior_fights,
            lookback_cards=self.lookback_cards,
            predictions=tuple(predictions),
            evaluated_fights=len(evaluated),
            skipped_fights=len(predictions) - len(evaluated),
            market_metrics=market_metrics,
            model_metrics=model_metrics,
            blend_metrics=blend_metrics,
        )
