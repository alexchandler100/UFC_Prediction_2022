"""Locked prospective paper policy built only from one fresh capture."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from ._common import MarketDataError, StoreIntegrityError, utc_datetime
from .forecasts import ForecastCapture
from .paper import PaperDecision
from .quotes import QuoteSnapshot, consensus_as_of
from .source_metadata import QuoteSourceMetadata


DECISION_TARGET_LEAD_SECONDS = 24.0 * 60.0 * 60.0
DECISION_WINDOW_SECONDS = 4.0 * 60.0 * 60.0
MAX_SOURCE_QUOTE_AGE_SECONDS = 30.0 * 60.0
MAX_DECISION_LATENCY_SECONDS = 5.0 * 60.0
MIN_CONSENSUS_BOOKS = 3
MIN_EXPECTED_RETURN = 0.05
LOCKED_GAMMA = 0.0


@dataclass(frozen=True)
class PaperDecisionBuild:
    decisions: tuple[PaperDecision, ...]
    eligible_horizon: bool
    lead_time_seconds: float | None
    matchups_considered: int
    matchups_already_frozen: int
    matchups_without_fresh_quotes: int
    matchups_without_forecast: int

    def to_mapping(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "decisions"
        }


def _lead_time(quotes: tuple[QuoteSnapshot, ...]) -> float | None:
    contracts = {
        (item.timing_precision, item.event_start_utc, item.observed_at_utc)
        for item in quotes
    }
    if len(contracts) != 1:
        raise StoreIntegrityError("one decision capture has multiple timing contracts")
    precision, event_start, observed = next(iter(contracts))
    if precision != "timestamp" or event_start is None:
        return None
    return (
        utc_datetime(event_start, "event_start_utc")
        - utc_datetime(observed, "observed_at_utc")
    ).total_seconds()


def build_locked_paper_decisions(
    quotes: Iterable[QuoteSnapshot],
    forecasts: Iterable[ForecastCapture],
    source_metadata: Iterable[QuoteSourceMetadata],
    existing_decisions: Iterable[PaperDecision] = (),
) -> PaperDecisionBuild:
    """Freeze one target-book decision per matchup near the locked T-24 horizon."""

    quote_records = tuple(quotes)
    forecast_records = tuple(forecasts)
    metadata_records = tuple(source_metadata)
    existing = tuple(existing_decisions)
    if not quote_records:
        raise ValueError("at least one quote is required")
    capture_ids = {item.capture_id for item in quote_records}
    if len(capture_ids) != 1:
        raise StoreIntegrityError("paper decision input must contain one capture")
    capture_id = next(iter(capture_ids))
    lead = _lead_time(quote_records)
    eligible_horizon = lead is not None and abs(
        lead - DECISION_TARGET_LEAD_SECONDS
    ) <= DECISION_WINDOW_SECONDS
    grouped_quotes: dict[str, list[QuoteSnapshot]] = defaultdict(list)
    for quote in quote_records:
        grouped_quotes[quote.matchup_id].append(quote)
    if not eligible_horizon:
        return PaperDecisionBuild(
            decisions=(),
            eligible_horizon=False,
            lead_time_seconds=lead,
            matchups_considered=len(grouped_quotes),
            matchups_already_frozen=0,
            matchups_without_fresh_quotes=0,
            matchups_without_forecast=0,
        )

    metadata_by_quote = {item.quote_id: item for item in metadata_records}
    if len(metadata_by_quote) != len(metadata_records):
        raise StoreIntegrityError("duplicate quote source metadata")
    forecast_by_matchup: dict[str, ForecastCapture] = {}
    for forecast in forecast_records:
        if forecast.capture_id != capture_id:
            continue
        if forecast.matchup_id in forecast_by_matchup:
            raise StoreIntegrityError("one capture has duplicate matchup forecasts")
        forecast_by_matchup[forecast.matchup_id] = forecast
    frozen_matchups = {item.matchup_id for item in existing}
    decisions: list[PaperDecision] = []
    already_frozen = no_fresh = no_forecast = 0

    for matchup_id in sorted(grouped_quotes):
        if matchup_id in frozen_matchups:
            already_frozen += 1
            continue
        forecast = forecast_by_matchup.get(matchup_id)
        if forecast is None:
            no_forecast += 1
            continue
        fresh_quotes: list[QuoteSnapshot] = []
        for quote in grouped_quotes[matchup_id]:
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
                    "quote source metadata disagrees with its quote"
                )
            age = float(metadata.source_quote_age_seconds)
            if -300.0 <= age <= MAX_SOURCE_QUOTE_AGE_SECONDS:
                fresh_quotes.append(quote)
        if len(fresh_quotes) < MIN_CONSENSUS_BOOKS + 1:
            no_fresh += 1
            continue
        observed_times = {item.observed_at_utc for item in fresh_quotes}
        if len(observed_times) != 1:
            raise StoreIntegrityError("fresh paper quotes span retrieval times")
        observed_at = next(iter(observed_times))
        candidates: list[PaperDecision] = []
        for target in sorted(
            fresh_quotes, key=lambda item: (item.book.casefold(), item.quote_id)
        ):
            try:
                market = consensus_as_of(
                    fresh_quotes,
                    capture_id=capture_id,
                    matchup_id=matchup_id,
                    as_of_utc=observed_at,
                    min_books=MIN_CONSENSUS_BOOKS,
                    exclude_books=(target.book,),
                )
                candidate = PaperDecision.create(
                    market,
                    target,
                    forecast,
                    selected_gamma=LOCKED_GAMMA,
                    decision_issued_at_utc=observed_at,
                    minimum_expected_return=MIN_EXPECTED_RETURN,
                    maximum_quote_age_seconds=MAX_DECISION_LATENCY_SECONDS,
                )
            except MarketDataError:
                continue
            candidates.append(candidate)
        if not candidates:
            no_fresh += 1
            continue

        def rank(item: PaperDecision) -> tuple[float, str, str]:
            best_expected_return = max(
                item.fighter_expected_return, item.opponent_expected_return
            )
            reference = next(
                quote for quote in fresh_quotes if quote.quote_id == item.reference_quote_id
            )
            return (-best_expected_return, reference.book.casefold(), item.decision_id)

        decisions.append(min(candidates, key=rank))

    return PaperDecisionBuild(
        decisions=tuple(decisions),
        eligible_horizon=True,
        lead_time_seconds=lead,
        matchups_considered=len(grouped_quotes),
        matchups_already_frozen=already_frozen,
        matchups_without_fresh_quotes=no_fresh,
        matchups_without_forecast=no_forecast,
    )
