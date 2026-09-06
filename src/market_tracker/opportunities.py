"""Bounded website publication for transparent paper-market observations.

The publication produced here is deliberately read-only and paper-only.  It
uses the exact leave-one-book-out probability and expected-return calculation
used by the prospective decision ledger, while keeping the current observation
separate from any decision that was actually frozen in the T-24 window.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from ._common import (
    BETTING_STATUS,
    MarketDataError,
    StoreIntegrityError,
    canonical_hash,
)
from .forecasts import ForecastCapture
from .paper import BayesianFilteredDecision, PaperDecision
from .prospective import (
    DECISION_TARGET_LEAD_SECONDS,
    DECISION_WINDOW_SECONDS,
    LOCKED_GAMMA,
    MAX_DECISION_LATENCY_SECONDS,
    MAX_SOURCE_QUOTE_AGE_SECONDS,
    MIN_CONSENSUS_BOOKS,
    MIN_EXPECTED_RETURN,
    _lead_time,
)
from .quotes import QuoteSnapshot, consensus_as_of
from .source_metadata import QuoteSourceMetadata
from .props import TotalRoundsForecastCapture, TotalRoundsQuoteSnapshot
from .prop_opportunities import build_prop_market_view
from .prop_paper import TotalRoundsPaperDecision


# Current-card book prices and calibration evidence can exceed the old 256 KiB
# cap. Share one finite UTF-8 byte limit across writers and offline validators.
CURRENT_OPPORTUNITIES_SIZE_LIMIT = 1024 * 1024
OPPORTUNITY_POLICY_VERSION = "leave-one-book-out-paper-v1"
PUBLICATION_FLOAT_REL_TOLERANCE = 1e-12
PUBLICATION_FLOAT_ABS_TOLERANCE = 1e-12


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _first_publication_difference(
    published: object,
    rebuilt: object,
    *,
    path: str = "$",
) -> str | None:
    """Return the first material difference in two publication bodies.

    Opportunity publications can be produced on Windows and later validated on
    a Linux runner. Derived IEEE-754 values may differ by a final bit across
    those environments, so floats use a deliberately tiny tolerance while all
    identifiers, prices, policy fields, collection shapes, and text remain
    exact. The sole identifier exception is the unlocked ``current_signal``
    decision hash: it fingerprints those same derived floats, so an immaterial
    final-bit difference necessarily produces an unrelated SHA-256 value. Its
    displayed fields are still compared below and both IDs must be valid hashes.
    """

    if path.endswith(".current_signal.decision_id"):
        return None if _is_sha256(published) and _is_sha256(rebuilt) else path
    if isinstance(published, dict) and isinstance(rebuilt, dict):
        if published.keys() != rebuilt.keys():
            missing = sorted(str(key) for key in rebuilt.keys() - published.keys())
            extra = sorted(str(key) for key in published.keys() - rebuilt.keys())
            return f"{path} keys (missing={missing}, extra={extra})"
        for key in published:
            difference = _first_publication_difference(
                published[key],
                rebuilt[key],
                path=f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(published, list) and isinstance(rebuilt, list):
        if len(published) != len(rebuilt):
            return f"{path} length"
        for index, (published_item, rebuilt_item) in enumerate(
            zip(published, rebuilt, strict=True)
        ):
            difference = _first_publication_difference(
                published_item,
                rebuilt_item,
                path=f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(published, float) and isinstance(rebuilt, float):
        if math.isclose(
            published,
            rebuilt,
            rel_tol=PUBLICATION_FLOAT_REL_TOLERANCE,
            abs_tol=PUBLICATION_FLOAT_ABS_TOLERANCE,
        ):
            return None
    return None if published == rebuilt else path


def _american_odds(probability: float) -> int:
    if probability == 0.5:
        return 100
    if probability > 0.5:
        return round(-100.0 * probability / (1.0 - probability))
    return round(100.0 * (1.0 - probability) / probability)


def _timing_status(lead_time_seconds: float | None) -> str:
    if lead_time_seconds is None:
        return "exact_event_time_unavailable"
    lower = DECISION_TARGET_LEAD_SECONDS - DECISION_WINDOW_SECONDS
    upper = DECISION_TARGET_LEAD_SECONDS + DECISION_WINDOW_SECONDS
    if lead_time_seconds > upper:
        return "before_t24_decision_window"
    if lead_time_seconds < lower:
        return "after_t24_decision_window"
    return "inside_t24_decision_window"


def _decision_view(
    decision: PaperDecision,
    reference: QuoteSnapshot,
    *,
    fighter_name: str,
    opponent_name: str,
) -> dict[str, object]:
    if decision.fighter_expected_return >= decision.opponent_expected_return:
        best_side = "fighter"
        best_name = fighter_name
        offered_line = decision.fighter_reference_moneyline
        break_even = decision.fighter_break_even_probability
        market_probability = decision.market_probability
        expected_return = decision.fighter_expected_return
        edge = decision.fighter_edge
    else:
        best_side = "opponent"
        best_name = opponent_name
        offered_line = decision.opponent_reference_moneyline
        break_even = decision.opponent_break_even_probability
        market_probability = 1.0 - decision.market_probability
        expected_return = decision.opponent_expected_return
        edge = decision.opponent_edge

    action_name = None
    if decision.paper_action == "fighter":
        action_name = fighter_name
    elif decision.paper_action == "opponent":
        action_name = opponent_name

    if decision.paper_action == "pass":
        reason = (
            f"Pass: best estimated EV is {expected_return:.1%}, below the "
            f"{decision.minimum_expected_return:.1%} paper threshold."
        )
    else:
        reason = (
            f"Paper signal: estimated EV is {expected_return:.1%}, meeting the "
            f"{decision.minimum_expected_return:.1%} threshold."
        )

    return {
        "decision_id": decision.decision_id,
        "paper_action": decision.paper_action,
        "action_name": action_name,
        "best_candidate_side": best_side,
        "best_candidate_name": best_name,
        "target_book": reference.book,
        "target_book_excluded_from_consensus": True,
        "offered_moneyline": offered_line,
        "break_even_probability": break_even,
        "market_probability": market_probability,
        "market_fair_moneyline": _american_odds(market_probability),
        "probability_edge": edge,
        "estimated_expected_return": expected_return,
        "minimum_expected_return": decision.minimum_expected_return,
        "consensus_book_count": 0,  # replaced by the caller
        "consensus_books": [],
        "model_probability_for_fighter": decision.model_probability,
        "model_weight": decision.selected_gamma,
        "probability_source": "leave_one_book_out_no_vig_market_consensus",
        "observed_at_utc": decision.market_as_of_utc,
        "reason": reason,
    }


def _candidate_for_matchup(
    quotes: tuple[QuoteSnapshot, ...],
    forecast: ForecastCapture,
) -> tuple[PaperDecision, QuoteSnapshot, tuple[str, ...]] | None:
    candidates: list[tuple[PaperDecision, QuoteSnapshot, tuple[str, ...]]] = []
    for target in sorted(quotes, key=lambda item: (item.book.casefold(), item.quote_id)):
        try:
            market = consensus_as_of(
                quotes,
                capture_id=target.capture_id,
                matchup_id=target.matchup_id,
                as_of_utc=target.observed_at_utc,
                min_books=MIN_CONSENSUS_BOOKS,
                exclude_books=(target.book,),
            )
            decision = PaperDecision.create(
                market,
                target,
                forecast,
                selected_gamma=LOCKED_GAMMA,
                decision_issued_at_utc=target.observed_at_utc,
                minimum_expected_return=MIN_EXPECTED_RETURN,
                maximum_quote_age_seconds=MAX_DECISION_LATENCY_SECONDS,
            )
        except MarketDataError:
            continue
        candidates.append((decision, target, market.included_book_keys))
    if not candidates:
        return None

    def rank(
        item: tuple[PaperDecision, QuoteSnapshot, tuple[str, ...]],
    ) -> tuple[float, str, str]:
        decision, reference, _ = item
        best_return = max(
            decision.fighter_expected_return,
            decision.opponent_expected_return,
        )
        return (-best_return, reference.book.casefold(), decision.decision_id)

    return min(candidates, key=rank)


def build_current_opportunities(
    quotes: Iterable[QuoteSnapshot],
    forecasts: Iterable[ForecastCapture],
    source_metadata: Iterable[QuoteSourceMetadata],
    decisions: Iterable[PaperDecision] = (),
    *,
    capture_id: str,
    total_round_quotes: Iterable[TotalRoundsQuoteSnapshot] | None = None,
    total_round_forecasts: Iterable[TotalRoundsForecastCapture] | None = None,
    total_round_decisions: Iterable[TotalRoundsPaperDecision] | None = None,
    bayesian_filtered_decisions: Iterable[BayesianFilteredDecision] | None = None,
    method_price_status: str = "unavailable_from_configured_provider",
) -> dict[str, object]:
    """Build a deterministic, bounded view for one immutable capture."""

    all_quotes = tuple(quotes)
    current_quotes = tuple(item for item in all_quotes if item.capture_id == capture_id)
    if not current_quotes:
        raise ValueError("opportunity publication capture has no quotes")
    capture_contracts = {
        (
            item.event_id,
            item.event_date,
            item.timing_precision,
            item.event_start_utc,
            item.observed_at_utc,
            item.source,
        )
        for item in current_quotes
    }
    if len(capture_contracts) != 1:
        raise StoreIntegrityError("opportunity capture has conflicting event contracts")
    event_id, event_date, precision, event_start, observed_at, source = next(
        iter(capture_contracts)
    )
    lead_time_seconds = _lead_time(current_quotes)

    metadata_records = tuple(source_metadata)
    metadata_by_quote = {item.quote_id: item for item in metadata_records}
    if len(metadata_by_quote) != len(metadata_records):
        raise StoreIntegrityError("duplicate quote source metadata")
    forecast_by_matchup: dict[str, ForecastCapture] = {}
    for forecast in forecasts:
        if forecast.capture_id != capture_id:
            continue
        if forecast.matchup_id in forecast_by_matchup:
            raise StoreIntegrityError("duplicate forecast in opportunity capture")
        forecast_by_matchup[forecast.matchup_id] = forecast
    decisions_by_matchup: dict[str, PaperDecision] = {}
    for decision in decisions:
        if decision.event_id != event_id:
            continue
        if decision.matchup_id in decisions_by_matchup:
            raise StoreIntegrityError("more than one locked decision exists for a matchup")
        decisions_by_matchup[decision.matchup_id] = decision
    filtered_by_matchup: dict[str, BayesianFilteredDecision] = {}
    for filtered in bayesian_filtered_decisions or ():
        if filtered.event_id != event_id:
            continue
        if filtered.matchup_id in filtered_by_matchup:
            raise StoreIntegrityError(
                "more than one Bayesian filtered decision exists for a matchup"
            )
        filtered_by_matchup[filtered.matchup_id] = filtered
    quote_by_id = {item.quote_id: item for item in all_quotes}
    display_books = {item.book.casefold(): item.book for item in current_quotes}

    grouped: dict[str, list[QuoteSnapshot]] = defaultdict(list)
    for quote in current_quotes:
        grouped[quote.matchup_id].append(quote)

    matchup_views: list[dict[str, object]] = []
    for matchup_id in sorted(grouped):
        matchup_quotes = tuple(
            sorted(grouped[matchup_id], key=lambda item: (item.book.casefold(), item.quote_id))
        )
        first = matchup_quotes[0]
        fresh_quotes: list[QuoteSnapshot] = []
        book_quotes: list[dict[str, object]] = []
        for quote in matchup_quotes:
            metadata = metadata_by_quote.get(quote.quote_id)
            eligible = False
            updated_at = None
            age_seconds = None
            if metadata is not None:
                identity = (
                    metadata.capture_id,
                    metadata.matchup_id,
                    metadata.event_id,
                    metadata.source,
                    metadata.book,
                    metadata.observed_at_utc,
                )
                expected = (
                    quote.capture_id,
                    quote.matchup_id,
                    quote.event_id,
                    quote.source,
                    quote.book,
                    quote.observed_at_utc,
                )
                if identity != expected:
                    raise StoreIntegrityError("quote metadata disagrees with opportunity quote")
                age_seconds = float(metadata.source_quote_age_seconds)
                updated_at = metadata.source_quote_updated_at_utc
                eligible = -300.0 <= age_seconds <= MAX_SOURCE_QUOTE_AGE_SECONDS
            if eligible:
                fresh_quotes.append(quote)
            book_quotes.append(
                {
                    "book": quote.book,
                    "fighter_moneyline": quote.fighter_moneyline,
                    "opponent_moneyline": quote.opponent_moneyline,
                    "source_quote_updated_at_utc": updated_at,
                    "source_quote_age_seconds": age_seconds,
                    "eligible_for_consensus": eligible,
                }
            )

        full_market = None
        if len(fresh_quotes) >= MIN_CONSENSUS_BOOKS:
            full_consensus = consensus_as_of(
                fresh_quotes,
                capture_id=capture_id,
                matchup_id=matchup_id,
                as_of_utc=observed_at,
                min_books=MIN_CONSENSUS_BOOKS,
            )
            full_market = {
                "fighter_probability": full_consensus.no_vig_fighter_probability,
                "opponent_probability": full_consensus.no_vig_opponent_probability,
                "fighter_fair_moneyline": _american_odds(
                    full_consensus.no_vig_fighter_probability
                ),
                "opponent_fair_moneyline": _american_odds(
                    full_consensus.no_vig_opponent_probability
                ),
                "book_count": full_consensus.book_count,
                "books": [
                    display_books.get(key, key) for key in full_consensus.included_book_keys
                ],
            }

        forecast = forecast_by_matchup.get(matchup_id)
        current_signal = None
        unavailable_reason = None
        if forecast is None:
            unavailable_reason = "No independent model forecast was paired with this capture."
        elif len(fresh_quotes) < MIN_CONSENSUS_BOOKS + 1:
            unavailable_reason = (
                f"Need at least {MIN_CONSENSUS_BOOKS + 1} fresh books to price one "
                f"target while excluding it; found {len(fresh_quotes)}."
            )
        else:
            selected = _candidate_for_matchup(tuple(fresh_quotes), forecast)
            if selected is None:
                unavailable_reason = "No valid leave-one-book-out paper evaluation was available."
            else:
                decision, reference, consensus_book_keys = selected
                current_signal = _decision_view(
                    decision,
                    reference,
                    fighter_name=first.fighter_name,
                    opponent_name=first.opponent_name,
                )
                current_signal["consensus_book_count"] = len(consensus_book_keys)
                current_signal["consensus_books"] = [
                    display_books.get(key, key) for key in consensus_book_keys
                ]

        locked_view = None
        locked = decisions_by_matchup.get(matchup_id)
        if locked is not None:
            reference = quote_by_id.get(locked.reference_quote_id)
            if reference is None:
                raise StoreIntegrityError("locked decision references an unknown quote")
            locked_view = _decision_view(
                locked,
                reference,
                fighter_name=first.fighter_name,
                opponent_name=first.opponent_name,
            )
            locked_quotes = tuple(
                quote
                for quote in all_quotes
                if quote.capture_id == locked.capture_id
                and quote.matchup_id == locked.matchup_id
                and quote.quote_id in metadata_by_quote
                and -300.0
                <= float(metadata_by_quote[quote.quote_id].source_quote_age_seconds)
                <= MAX_SOURCE_QUOTE_AGE_SECONDS
            )
            locked_consensus = consensus_as_of(
                locked_quotes,
                capture_id=locked.capture_id,
                matchup_id=locked.matchup_id,
                as_of_utc=locked.market_as_of_utc,
                min_books=MIN_CONSENSUS_BOOKS,
                exclude_books=(reference.book,),
            )
            locked_display_books = {
                quote.book.casefold(): quote.book
                for quote in all_quotes
                if quote.capture_id == locked.capture_id
            }
            locked_view["consensus_book_count"] = locked_consensus.book_count
            locked_view["consensus_books"] = [
                locked_display_books.get(key, key)
                for key in locked_consensus.included_book_keys
            ]

        matchup_view: dict[str, object] = {
                "matchup_id": matchup_id,
                "fighter_id": first.fighter_id,
                "opponent_id": first.opponent_id,
                "fighter_name": first.fighter_name,
                "opponent_name": first.opponent_name,
                "model_probability_for_fighter": (
                    forecast.model_probability if forecast is not None else None
                ),
                "model_id": forecast.model_id if forecast is not None else None,
                "full_market_consensus": full_market,
                "current_signal": current_signal,
                "current_signal_unavailable_reason": unavailable_reason,
                "locked_t24_decision": locked_view,
                "book_quotes": book_quotes,
            }
        if bayesian_filtered_decisions is not None:
            filtered = filtered_by_matchup.get(matchup_id)
            matchup_view["locked_t24_bayesian_filter"] = (
                {
                        "filtered_decision_id": filtered.filtered_decision_id,
                        "base_decision_id": filtered.base_decision_id,
                        "policy_version": filtered.policy_version,
                        "bayesian_model_id": filtered.bayesian_model_id,
                        "base_paper_action": filtered.base_paper_action,
                        "filtered_paper_action": filtered.filtered_paper_action,
                        "filter_status": filtered.filter_status,
                        "candidate_moneyline": filtered.candidate_moneyline,
                        "candidate_posterior_mean_probability": (
                            filtered.candidate_posterior_mean_probability
                        ),
                        "candidate_posterior_mean_expected_return": (
                            filtered.candidate_posterior_mean_expected_return
                        ),
                        "candidate_expected_return_lower": (
                            filtered.candidate_expected_return_lower
                        ),
                        "candidate_expected_return_upper": (
                            filtered.candidate_expected_return_upper
                        ),
                        "candidate_probability_positive_expected_return": (
                            filtered.candidate_probability_positive_expected_return
                        ),
                        "minimum_mean_expected_return": (
                            filtered.minimum_mean_expected_return
                        ),
                        "minimum_probability_positive_expected_return": (
                            filtered.minimum_probability_positive_expected_return
                        ),
                        "paper_only": filtered.paper_only,
                        "execution_enabled": filtered.execution_enabled,
                }
                if filtered is not None
                else None
            )
        matchup_views.append(matchup_view)

    prop_contract = (
        total_round_quotes is not None
        or total_round_forecasts is not None
        or total_round_decisions is not None
    )
    if prop_contract and (total_round_quotes is None or total_round_forecasts is None):
        raise ValueError("prop opportunity publication requires quotes and forecasts")
    body: dict[str, object] = {
        "schema_version": (
            4
            if bayesian_filtered_decisions is not None
            else 3
            if total_round_decisions is not None
            else 2
            if prop_contract
            else 1
        ),
        "policy_version": OPPORTUNITY_POLICY_VERSION,
        "betting_status": BETTING_STATUS,
        "paper_only": True,
        "execution_enabled": False,
        "capture_id": capture_id,
        "event_id": event_id,
        "event_date": event_date,
        "timing_precision": precision,
        "event_start_utc": event_start,
        "observed_at_utc": observed_at,
        "source": source,
        "capture_lead_time_seconds": lead_time_seconds,
        "timing_status": _timing_status(lead_time_seconds),
        "decision_target_lead_seconds": DECISION_TARGET_LEAD_SECONDS,
        "decision_window_seconds": DECISION_WINDOW_SECONDS,
        "maximum_source_quote_age_seconds": MAX_SOURCE_QUOTE_AGE_SECONDS,
        "minimum_consensus_books_excluding_target": MIN_CONSENSUS_BOOKS,
        "minimum_expected_return": MIN_EXPECTED_RETURN,
        "model_weight": LOCKED_GAMMA,
        "matchup_count": len(matchup_views),
        "matchups": matchup_views,
    }
    if prop_contract:
        body["prop_markets"] = build_prop_market_view(
            total_round_quotes or (),
            total_round_forecasts or (),
            capture_id=capture_id,
            method_price_status=method_price_status,
            decisions=total_round_decisions or (),
        )
    body["publication_sha256"] = canonical_hash(body)
    return body


def validate_current_opportunities(
    publication: object,
    quotes: Iterable[QuoteSnapshot],
    forecasts: Iterable[ForecastCapture],
    source_metadata: Iterable[QuoteSourceMetadata],
    decisions: Iterable[PaperDecision],
    *,
    capture_id: str,
    total_round_quotes: Iterable[TotalRoundsQuoteSnapshot] = (),
    total_round_forecasts: Iterable[TotalRoundsForecastCapture] = (),
    total_round_decisions: Iterable[TotalRoundsPaperDecision] = (),
    bayesian_filtered_decisions: Iterable[BayesianFilteredDecision] = (),
    method_price_status: str = "unavailable_from_configured_provider",
) -> dict[str, object]:
    """Rebuild the publication from ledgers and require an exact match."""

    if not isinstance(publication, dict):
        raise ValueError("current opportunity publication must be an object")
    schema_version = publication.get("schema_version")
    if schema_version == 1:
        rebuilt = build_current_opportunities(
            quotes,
            forecasts,
            source_metadata,
            decisions,
            capture_id=capture_id,
        )
    elif schema_version == 2:
        rebuilt = build_current_opportunities(
            quotes,
            forecasts,
            source_metadata,
            decisions,
            capture_id=capture_id,
            total_round_quotes=total_round_quotes,
            total_round_forecasts=total_round_forecasts,
            method_price_status=method_price_status,
        )
    elif schema_version == 3:
        rebuilt = build_current_opportunities(
            quotes,
            forecasts,
            source_metadata,
            decisions,
            capture_id=capture_id,
            total_round_quotes=total_round_quotes,
            total_round_forecasts=total_round_forecasts,
            total_round_decisions=total_round_decisions,
            method_price_status=method_price_status,
        )
    elif schema_version == 4:
        rebuilt = build_current_opportunities(
            quotes,
            forecasts,
            source_metadata,
            decisions,
            capture_id=capture_id,
            total_round_quotes=total_round_quotes,
            total_round_forecasts=total_round_forecasts,
            total_round_decisions=total_round_decisions,
            bayesian_filtered_decisions=bayesian_filtered_decisions,
            method_price_status=method_price_status,
        )
    else:
        raise ValueError("unsupported current opportunity schema version")
    published_body = dict(publication)
    published_fingerprint = published_body.pop("publication_sha256", None)
    if published_fingerprint != canonical_hash(published_body):
        raise StoreIntegrityError(
            "current opportunity publication fingerprint does not match its contents"
        )
    rebuilt_body = dict(rebuilt)
    rebuilt_body.pop("publication_sha256", None)
    difference = _first_publication_difference(published_body, rebuilt_body)
    if difference is not None:
        raise StoreIntegrityError(
            "current opportunity publication cannot be reproduced from market ledgers "
            f"at {difference}"
        )
    return rebuilt
