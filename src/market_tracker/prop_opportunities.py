"""Candidate-only total-round expected-value views for the website."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Iterable

from ._common import StoreIntegrityError
from .props import TotalRoundsForecastCapture, TotalRoundsQuoteSnapshot
from .prop_paper import TotalRoundsPaperDecision
from .prospective import MAX_SOURCE_QUOTE_AGE_SECONDS


PROP_OPPORTUNITY_POLICY_VERSION = "candidate-total-round-model-ev-v1"
MIN_PROP_EXPECTED_RETURN = 0.05
MIN_PROP_CONSENSUS_BOOKS = 2


def _expected_return(probability: float, implied_probability: float) -> float:
    return probability / implied_probability - 1.0


def build_prop_market_view(
    quotes: Iterable[TotalRoundsQuoteSnapshot],
    forecasts: Iterable[TotalRoundsForecastCapture],
    *,
    capture_id: str,
    method_price_status: str = "unavailable_from_configured_provider",
    decisions: Iterable[TotalRoundsPaperDecision] = (),
) -> dict[str, object]:
    current_quotes = tuple(item for item in quotes if item.capture_id == capture_id)
    current_forecasts = tuple(
        item for item in forecasts if item.capture_id == capture_id
    )
    decision_index: dict[tuple[str, float], TotalRoundsPaperDecision] = {}
    for decision in decisions:
        key = decision.matchup_id, float(decision.line)
        if key in decision_index:
            raise StoreIntegrityError("duplicate locked total decision for matchup/line")
        decision_index[key] = decision
    forecast_index: dict[tuple[str, float], TotalRoundsForecastCapture] = {}
    for forecast in current_forecasts:
        key = forecast.matchup_id, forecast.line
        if key in forecast_index:
            raise StoreIntegrityError("duplicate total-round forecast for capture/line")
        forecast_index[key] = forecast

    grouped: dict[tuple[str, float], list[TotalRoundsQuoteSnapshot]] = defaultdict(list)
    for quote in current_quotes:
        grouped[(quote.matchup_id, quote.line)].append(quote)

    markets: list[dict[str, object]] = []
    positive: list[dict[str, object]] = []
    for matchup_id, line in sorted(grouped, key=lambda value: (value[0], value[1])):
        line_quotes = tuple(
            sorted(
                grouped[(matchup_id, line)],
                key=lambda item: (item.book.casefold(), item.quote_id),
            )
        )
        first = line_quotes[0]
        forecast = forecast_index.get((matchup_id, line))
        locked = decision_index.get((matchup_id, line))
        eligible = tuple(
            item
            for item in line_quotes
            if -300.0
            <= item.source_quote_age_seconds
            <= MAX_SOURCE_QUOTE_AGE_SECONDS
        )
        candidates: list[dict[str, object]] = []
        if forecast is not None:
            for quote in eligible:
                for side, probability, moneyline, break_even in (
                    (
                        "over",
                        forecast.over_probability,
                        quote.over_moneyline,
                        quote.over_implied_probability,
                    ),
                    (
                        "under",
                        1.0 - forecast.over_probability,
                        quote.under_moneyline,
                        quote.under_implied_probability,
                    ),
                ):
                    other_books = tuple(
                        item
                        for item in eligible
                        if item.source_book_key.casefold()
                        != quote.source_book_key.casefold()
                    )
                    consensus = (
                        median(item.no_vig_over_probability for item in other_books)
                        if len(other_books) >= MIN_PROP_CONSENSUS_BOOKS
                        else None
                    )
                    candidate = {
                        "matchup_id": matchup_id,
                        "fighter_name": first.fighter_name,
                        "opponent_name": first.opponent_name,
                        "market": "total_rounds",
                        "period": "full_fight",
                        "side": side,
                        "selection": f"{side.title()} {line:g} rounds",
                        "line": line,
                        "target_book": quote.book,
                        "offered_moneyline": moneyline,
                        "break_even_probability": break_even,
                        "model_probability": probability,
                        "probability_edge": probability - break_even,
                        "estimated_expected_return": _expected_return(
                            probability, break_even
                        ),
                        "positive_expected_value": _expected_return(
                            probability, break_even
                        ) > 0.0,
                        "paper_threshold_met": _expected_return(
                            probability, break_even
                        ) >= MIN_PROP_EXPECTED_RETURN,
                        "minimum_paper_expected_return": MIN_PROP_EXPECTED_RETURN,
                        "scheduled_rounds": forecast.scheduled_rounds,
                        "schedule_basis": forecast.schedule_basis,
                        "model_id": forecast.model_id,
                        "model_version": forecast.model_version,
                        "forecast_issued_at_utc": forecast.forecast_issued_at_utc,
                        "observed_at_utc": quote.observed_at_utc,
                        "source_quote_age_seconds": quote.source_quote_age_seconds,
                        "target_book_excluded_from_market_context": True,
                        "other_book_consensus_over_probability": consensus,
                        "other_book_consensus_count": len(other_books),
                    }
                    candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                -float(item["estimated_expected_return"]),
                str(item["target_book"]).casefold(),
                str(item["side"]),
            )
        )
        if candidates and candidates[0]["positive_expected_value"]:
            # Publish only the best available book/side for one matchup/line;
            # inferior prices for the same market remain visible in book_quotes.
            positive.append(candidates[0])
        markets.append(
            {
                "matchup_id": matchup_id,
                "fighter_name": first.fighter_name,
                "opponent_name": first.opponent_name,
                "line": line,
                "quote_count": len(line_quotes),
                "eligible_quote_count": len(eligible),
                "forecast_available": forecast is not None,
                "forecast_unavailable_reason": (
                    None
                    if forecast is not None
                    else "No frozen candidate duration forecast matched this line."
                ),
                "best_candidate": candidates[0] if candidates else None,
                "locked_t24_decision": (
                    {
                        "decision_id": locked.decision_id,
                        "captured_at_utc": locked.decision_issued_at_utc,
                        "selection": (
                            f"{locked.paper_action.title()} {line:g} rounds"
                            if locked.paper_action != "pass"
                            else None
                        ),
                        "paper_action": locked.paper_action,
                        "target_book": locked.target_book,
                        "offered_moneyline": locked.action_reference_moneyline,
                        "market_over_probability": locked.market_over_probability,
                        "model_over_probability": locked.model_over_probability,
                        "residual_over_probability": locked.residual_over_probability,
                        "selected_residual_weight": locked.selected_residual_weight,
                        "residual_selection_status": locked.residual_selection_status,
                        "estimated_expected_return": (
                            max(
                                locked.residual_over_expected_return,
                                locked.residual_under_expected_return,
                            )
                        ),
                        "minimum_expected_return": locked.minimum_expected_return,
                        "consensus_book_count": locked.consensus_book_count,
                    }
                    if locked is not None
                    else None
                ),
                "book_quotes": [
                    {
                        "book": quote.book,
                        "over_moneyline": quote.over_moneyline,
                        "under_moneyline": quote.under_moneyline,
                        "no_vig_over_probability": quote.no_vig_over_probability,
                        "source_quote_age_seconds": quote.source_quote_age_seconds,
                        "eligible": quote in eligible,
                    }
                    for quote in line_quotes
                ],
            }
        )

    positive.sort(
        key=lambda item: (
            -float(item["estimated_expected_return"]),
            str(item["fighter_name"]).casefold(),
            float(item["line"]),
            str(item["target_book"]).casefold(),
        )
    )
    return {
        "policy_version": PROP_OPPORTUNITY_POLICY_VERSION,
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "probability_source": "candidate_discrete_time_competing_risks_model",
        "expected_return_formula": "model_probability / break_even_probability - 1",
        "minimum_paper_expected_return": MIN_PROP_EXPECTED_RETURN,
        "total_rounds": {
            "price_status": "available" if current_quotes else "awaiting_capture",
            "quote_count": len(current_quotes),
            "forecast_count": len(current_forecasts),
            "market_count": len(markets),
            "positive_candidate_count": len(positive),
            "locked_decision_count": len(decision_index),
            "positive_candidates": positive[:50],
            "markets": markets,
        },
        "method_of_victory": {
            "model_probability_status": "available_in_candidate_outcome_forecast",
            "price_status": method_price_status,
            "expected_value_status": (
                "unavailable_without_book_price"
                if method_price_status != "available"
                else "available"
            ),
            "positive_candidates": [],
        },
    }
