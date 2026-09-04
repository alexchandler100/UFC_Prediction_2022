"""Publish all announced UFC forecasts and a threshold-only paper bet board.

The board is a current research view, not an execution system.  It applies the
same leave-one-book-out moneyline policy as the prospective market tracker and
contains only prices meeting that policy's expected-return threshold.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import pandas as pd

from bayesian_total_calibration import (
    BayesianTotalCalibrator,
    KELLY_POLICY_VERSION as TOTAL_BAYESIAN_KELLY_POLICY_VERSION,
    unavailable_total_assessment,
    validate_total_bayesian_kelly_assessment,
)
from fight_stat_helpers import same_name
from market_tracker import (
    LOCKED_GAMMA,
    MIN_EXPECTED_RETURN,
    EarlyMarketObservation,
    matchup_id_for,
)
from market_tracker._common import canonical_hash
from market_tracker.bayesian_kelly import (
    BayesianKellyCalibrator,
    POLICY_VERSION as BAYESIAN_KELLY_POLICY_VERSION,
    validate_bayesian_kelly_assessment,
)
from market_tracker.paper import symmetric_logit_blend
from market_tracker.prospective import MIN_CONSENSUS_BOOKS


ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = ROOT / "content/data/external"
MARKET_ROOT = ROOT / "content/data/market"
UPCOMING_FORECAST_PATH = EXTERNAL_ROOT / "all_upcoming_forecasts.json"
UPCOMING_BOARD_PATH = MARKET_ROOT / "upcoming_bet_board.json"
FORECAST_SCHEMA_VERSION = 1
BOARD_SCHEMA_VERSION = 2
FORECAST_VERSION = "all-announced-ufc-forecasts-v1"
BOARD_POLICY_VERSION = "calibrated-upcoming-paper-allocation-v2"
LEGACY_BOARD_POLICY_VERSION = "all-upcoming-qualified-bets-v1"
MAX_SOURCE_QUOTE_AGE_SECONDS = 30.0 * 60.0
MAX_BOARD_BETS = 100
MAX_BOARD_OFFERS = 2000
ALLOCATION_POLICY = {
    "maximum_fight_fraction": 0.01,
    "maximum_card_fraction": 0.05,
    "maximum_outstanding_fraction": 0.10,
    "one_selection_per_fight": True,
    "assumes_no_existing_open_bets": True,
    "ranking": "calibrated_mean_expected_return_descending",
}


def _offer_rank(item: Mapping[str, object]) -> tuple:
    return (-float(item["estimated_expected_return"]), str(item.get("event_date")),
            str(item.get("event_id")), str(item.get("matchup_id")),
            str(item.get("selection")).casefold(), str(item.get("target_book")).casefold(),
            str(item.get("side")))


def allocate_paper_offers(offers: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Allocate a snapshot with no pre-existing open bets; freeze IDs last."""
    result = []
    fights: set[tuple] = set()
    cards: dict[str, float] = {}
    outstanding = 0.0
    for source in sorted(offers, key=_offer_rank):
        offer = dict(source)
        offer.pop("bet_id", None)
        event = str(offer.get("event_id"))
        fighter_ids = sorted(str(offer.get(key) or "") for key in ("fighter_id", "opponent_id"))
        fight = (event, *fighter_ids) if all(fighter_ids) else (event, str(offer.get("matchup_id")))
        allocation = 0.0
        if fight not in fights:
            allocation = max(0.0, min(
                float(offer["bayesian_kelly"]["recommended_fraction"]),
                ALLOCATION_POLICY["maximum_fight_fraction"],
                ALLOCATION_POLICY["maximum_card_fraction"] - cards.get(event, 0.0),
                ALLOCATION_POLICY["maximum_outstanding_fraction"] - outstanding,
            ))
        if allocation > 1e-12:
            fights.add(fight)
            cards[event] = cards.get(event, 0.0) + allocation
            outstanding += allocation
        else:
            allocation = 0.0
        offer["allocated_fraction"] = allocation
        offer["bet_id"] = canonical_hash(offer)
        result.append(offer)
    return result


def _calibrated_offer(bet: Mapping[str, object], observed_at: datetime) -> dict[str, object] | None:
    if bet.get("category") == "Total rounds" and bet.get("betting_performance_validated") is not True:
        return None
    assessment = bet.get("bayesian_kelly")
    if not isinstance(assessment, dict) or assessment.get("status") != "available":
        return None
    mean = _float(assessment.get("posterior_mean_probability"))
    lower = _float(assessment.get("posterior_lower_probability"))
    stake = _float(assessment.get("recommended_fraction"))
    if mean is None or lower is None or stake is None or stake <= 0:
        return None
    updated_text = bet.get("source_quote_updated_at_utc")
    start_text = bet.get("event_start_utc")
    if not updated_text or not start_text:
        return None
    updated = _utc(updated_text, "source quote update")
    start = _utc(start_text, "event start")
    if start <= observed_at or not 0 <= (observed_at - updated).total_seconds() <= MAX_SOURCE_QUOTE_AGE_SECONDS:
        return None
    decimal = _decimal_odds(int(bet["offered_moneyline"]))
    expected, floor_expected = mean * decimal - 1.0, lower * decimal - 1.0
    if expected < MIN_EXPECTED_RETURN or floor_expected <= 0:
        return None
    result = dict(bet)
    result.pop("bet_id", None)
    result["raw_estimated_win_probability"] = bet["estimated_win_probability"]
    result["raw_estimated_expected_return"] = float(bet["estimated_win_probability"]) * decimal - 1.0
    result["estimated_win_probability"] = mean
    result["estimated_expected_return"] = expected
    result["robust_lower_expected_return"] = floor_expected
    result["threshold_met"] = True
    result["qualification_probability_source"] = "calibrated_posterior_mean"
    return result


def _text(value: object) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return " ".join(str(value).split())


def _float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: object) -> int | None:
    number = _float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _utc(value: object, field: str) -> datetime:
    parsed = pd.to_datetime(_text(value), errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"{field} must be a valid UTC timestamp")
    return parsed.to_pydatetime().astimezone(timezone.utc)


def _utc_text(value: object, field: str) -> str:
    return _utc(value, field).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _event_day(value: object) -> str:
    parsed = pd.to_datetime(_text(value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError("upcoming event date is invalid")
    return parsed.date().isoformat()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(text)
            output.flush()
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def build_upcoming_forecast_publication(
    frame: pd.DataFrame,
    *,
    generated_at_utc: object,
) -> dict[str, object]:
    """Create a small row-oriented forecast file for every announced card."""

    required = {
        "date",
        "event id",
        "event url",
        "event title",
        "bout order",
        "fighter id",
        "opponent id",
        "fighter name",
        "opponent name",
        "division",
        "model id",
        "model version",
        "model trained through",
        "model probability",
        "model status",
        "forecast issued at",
        "forecast source commit",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"all-upcoming forecasts are missing columns: {missing}")
    if frame.empty:
        raise ValueError("all-upcoming forecast publication cannot be empty")

    matchups: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in frame.to_dict("records"):
        event_id = _text(row.get("event id"))
        event_url = _text(row.get("event url"))
        fighter_id = _text(row.get("fighter id"))
        opponent_id = _text(row.get("opponent id"))
        status = _text(row.get("model status"))
        if not event_id or not event_url or not event_url.rstrip("/").endswith(event_id):
            raise ValueError("all-upcoming forecast has invalid UFCStats event identity")
        matchup_id = (
            matchup_id_for(event_id, fighter_id, opponent_id)
            if fighter_id and opponent_id and fighter_id != opponent_id
            else ""
        )
        if not matchup_id and not status.startswith("abstain"):
            raise ValueError("resolved all-upcoming forecast lacks stable fighter IDs")
        uniqueness = matchup_id or canonical_hash(
            {
                "event_id": event_id,
                "fighter": _text(row.get("fighter name")).casefold(),
                "opponent": _text(row.get("opponent name")).casefold(),
            }
        )
        if uniqueness in seen:
            raise ValueError("all-upcoming publication repeats a matchup")
        seen.add(uniqueness)
        probability = _float(row.get("model probability"))
        if not status.startswith("abstain") and (
            probability is None or not 0.0 < probability < 1.0
        ):
            raise ValueError("resolved all-upcoming probability must be bounded")
        bout_order = _integer(row.get("bout order"))
        if bout_order is None or bout_order < 0:
            raise ValueError("all-upcoming bout order must be nonnegative")
        matchups.append(
            {
                "event_id": event_id,
                "event_url": event_url,
                "event_title": _text(row.get("event title")),
                "event_date": _event_day(row.get("date")),
                "bout_order": bout_order,
                "matchup_id": matchup_id or None,
                "fighter_id": fighter_id or None,
                "opponent_id": opponent_id or None,
                "fighter_name": _text(row.get("fighter name")),
                "opponent_name": _text(row.get("opponent name")),
                "division": _text(row.get("division")),
                "model_id": _text(row.get("model id")),
                "model_version": _text(row.get("model version")),
                "model_trained_through": _text(row.get("model trained through")),
                "model_probability_for_fighter": probability,
                "model_status": status,
                "forecast_issued_at_utc": _utc_text(
                    row.get("forecast issued at"), "forecast issued at"
                ),
                "forecast_source_commit": _text(
                    row.get("forecast source commit")
                ),
            }
        )
    matchups.sort(
        key=lambda item: (
            item["event_date"],
            int(item["bout_order"]),
            str(item["matchup_id"] or ""),
        )
    )
    events: list[dict[str, object]] = []
    for event_id in dict.fromkeys(str(item["event_id"]) for item in matchups):
        rows = [item for item in matchups if item["event_id"] == event_id]
        orders = [int(item["bout_order"]) for item in rows]
        if sorted(orders) != list(range(len(rows))):
            raise ValueError("all-upcoming event bout order must be contiguous")
        events.append(
            {
                "event_id": event_id,
                "event_url": rows[0]["event_url"],
                "event_title": rows[0]["event_title"],
                "event_date": rows[0]["event_date"],
                "matchup_count": len(rows),
            }
        )
    events.sort(key=lambda item: (item["event_date"], item["event_id"]))
    body: dict[str, object] = {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "publication_version": FORECAST_VERSION,
        "generated_at_utc": _utc_text(generated_at_utc, "generated_at_utc"),
        "paper_only": True,
        "execution_enabled": False,
        "event_count": len(events),
        "matchup_count": len(matchups),
        "events": events,
        "matchups": matchups,
    }
    body["publication_sha256"] = canonical_hash(body)
    return validate_upcoming_forecast_publication(body)


def validate_upcoming_forecast_publication(
    publication: object,
) -> dict[str, object]:
    if not isinstance(publication, dict):
        raise ValueError("all-upcoming forecast publication must be an object")
    supplied_hash = publication.get("publication_sha256")
    unhashed = dict(publication)
    unhashed.pop("publication_sha256", None)
    if supplied_hash != canonical_hash(unhashed):
        raise ValueError("all-upcoming forecast publication hash is invalid")
    if (
        publication.get("schema_version") != FORECAST_SCHEMA_VERSION
        or publication.get("publication_version") != FORECAST_VERSION
        or publication.get("paper_only") is not True
        or publication.get("execution_enabled") is not False
    ):
        raise ValueError("all-upcoming forecast publication policy is invalid")
    events = publication.get("events")
    matchups = publication.get("matchups")
    if not isinstance(events, list) or not isinstance(matchups, list):
        raise ValueError("all-upcoming forecast events and matchups must be lists")
    if publication.get("event_count") != len(events) or publication.get(
        "matchup_count"
    ) != len(matchups):
        raise ValueError("all-upcoming forecast counts are inconsistent")
    if not events or not matchups or not all(isinstance(item, dict) for item in [*events, *matchups]):
        raise ValueError("all-upcoming forecast requires object events and matchups")
    if len({str(item.get("event_id")) for item in events}) != len(events):
        raise ValueError("all-upcoming forecast repeats an event")
    event_by_id = {str(item.get("event_id") or ""): item for item in events}
    if "" in event_by_id:
        raise ValueError("all-upcoming forecast has a blank event ID")
    matchup_keys = [
        str(item.get("matchup_id") or "")
        or f"{item.get('event_id')}|{item.get('fighter_name')}|{item.get('opponent_name')}"
        for item in matchups
    ]
    if len(set(matchup_keys)) != len(matchup_keys):
        raise ValueError("all-upcoming forecast matchup identities are invalid")
    for event_id, event in event_by_id.items():
        rows = [item for item in matchups if str(item.get("event_id")) == event_id]
        if not rows or event.get("matchup_count") != len(rows):
            raise ValueError("all-upcoming forecast event matchup count is invalid")
        if any(
            str(row.get("event_url")) != str(event.get("event_url"))
            or str(row.get("event_title")) != str(event.get("event_title"))
            or str(row.get("event_date")) != str(event.get("event_date"))
            for row in rows
        ):
            raise ValueError("all-upcoming forecast event metadata is inconsistent")
        orders = [_integer(row.get("bout_order")) for row in rows]
        if any(order is None for order in orders) or sorted(orders) != list(range(len(rows))):
            raise ValueError("all-upcoming forecast bout order is invalid")
    if any(str(item.get("event_id")) not in event_by_id for item in matchups):
        raise ValueError("all-upcoming forecast matchup references an unknown event")
    if events != sorted(events, key=lambda item: (str(item.get("event_date")), str(item.get("event_id")))):
        raise ValueError("all-upcoming forecast events are not date ordered")
    _utc(publication.get("generated_at_utc"), "generated_at_utc")
    return publication


def write_upcoming_forecast_publication(
    publication: Mapping[str, object],
    path: Path = UPCOMING_FORECAST_PATH,
) -> None:
    _atomic_json(path, validate_upcoming_forecast_publication(dict(publication)))


def _implied_probability(moneyline: int) -> float:
    return (
        100.0 / (moneyline + 100.0)
        if moneyline > 0
        else -moneyline / (-moneyline + 100.0)
    )


def _decimal_odds(moneyline: int) -> float:
    return 1.0 + (moneyline / 100.0 if moneyline > 0 else 100.0 / -moneyline)


def _book_quotes(
    observations: Iterable[EarlyMarketObservation],
    *,
    reversed_orientation: bool,
    observed_at: datetime,
) -> list[dict[str, object]]:
    by_book: dict[str, dict[str, object]] = {}
    for item in observations:
        if item.market != "h2h":
            continue
        updated = _utc(item.source_quote_updated_at_utc, "source quote update")
        age = (observed_at - updated).total_seconds()
        if not -300.0 <= age <= MAX_SOURCE_QUOTE_AGE_SECONDS:
            continue
        fighter_line = (
            item.outcome_b_moneyline
            if reversed_orientation
            else item.outcome_a_moneyline
        )
        opponent_line = (
            item.outcome_a_moneyline
            if reversed_orientation
            else item.outcome_b_moneyline
        )
        first = _implied_probability(fighter_line)
        second = _implied_probability(opponent_line)
        quote = {
            "book": item.book,
            "book_key": item.source_book_key.casefold(),
            "fighter_moneyline": fighter_line,
            "opponent_moneyline": opponent_line,
            "no_vig_fighter_probability": first / (first + second),
            "source_quote_updated_at_utc": item.source_quote_updated_at_utc,
            "source_quote_age_seconds": age,
        }
        current = by_book.get(str(quote["book_key"]))
        if current is None or str(quote["source_quote_updated_at_utc"]) > str(
            current["source_quote_updated_at_utc"]
        ):
            by_book[str(quote["book_key"])] = quote
    return sorted(by_book.values(), key=lambda item: str(item["book_key"]))


def _stored_book_quotes(
    observations: Iterable[EarlyMarketObservation],
    *,
    reversed_orientation: bool,
) -> list[dict[str, object]]:
    """Return the latest captured price from each book, even when it is not fresh."""

    by_book: dict[str, dict[str, object]] = {}
    for item in observations:
        if item.market != "h2h":
            continue
        fighter_line = (
            item.outcome_b_moneyline
            if reversed_orientation
            else item.outcome_a_moneyline
        )
        opponent_line = (
            item.outcome_a_moneyline
            if reversed_orientation
            else item.outcome_b_moneyline
        )
        first = _implied_probability(fighter_line)
        second = _implied_probability(opponent_line)
        quote = {
            "book": item.book,
            "book_key": item.source_book_key.casefold(),
            "fighter_moneyline": fighter_line,
            "opponent_moneyline": opponent_line,
            "no_vig_fighter_probability": first / (first + second),
            "source_quote_updated_at_utc": item.source_quote_updated_at_utc,
            "first_observed_at_utc": item.first_observed_at_utc,
        }
        key = str(quote["book_key"])
        current = by_book.get(key)
        quote_order = (
            str(quote["source_quote_updated_at_utc"]),
            str(quote["first_observed_at_utc"]),
        )
        current_order = (
            str(current["source_quote_updated_at_utc"]),
            str(current["first_observed_at_utc"]),
        ) if current is not None else None
        if current_order is None or quote_order > current_order:
            by_book[key] = quote
    return sorted(
        by_book.values(),
        key=lambda item: (str(item["book"]).casefold(), str(item["book_key"])),
    )


def _moneyline_offers(
    matchup: Mapping[str, object],
    observations: Iterable[EarlyMarketObservation],
    *,
    reversed_orientation: bool,
    observed_at: datetime,
    bayesian_kelly: BayesianKellyCalibrator,
) -> list[dict[str, object]]:
    observation_records = tuple(observations)
    quotes = _book_quotes(
        observation_records,
        reversed_orientation=reversed_orientation,
        observed_at=observed_at,
    )
    if len(quotes) < MIN_CONSENSUS_BOOKS + 1:
        return []
    model_probability = _float(matchup.get("model_probability_for_fighter"))
    if model_probability is None or not 0.0 < model_probability < 1.0:
        return []
    candidates: list[dict[str, object]] = []
    for target in quotes:
        consensus = [
            item
            for item in quotes
            if item["book_key"] != target["book_key"]
        ]
        if len(consensus) < MIN_CONSENSUS_BOOKS:
            continue
        market_probability = sum(
            float(item["no_vig_fighter_probability"]) for item in consensus
        ) / len(consensus)
        probability = symmetric_logit_blend(
            market_probability, model_probability, LOCKED_GAMMA
        )
        for side, name, moneyline, side_probability in (
            (
                "fighter",
                matchup.get("fighter_name"),
                target["fighter_moneyline"],
                probability,
            ),
            (
                "opponent",
                matchup.get("opponent_name"),
                target["opponent_moneyline"],
                1.0 - probability,
            ),
        ):
            expected_return = side_probability * _decimal_odds(int(moneyline)) - 1.0
            candidates.append(
                {
                    "side": side,
                    "selection": _text(name),
                    "target_book": target["book"],
                    "offered_moneyline": int(moneyline),
                    "estimated_expected_return": expected_return,
                    "estimated_win_probability": side_probability,
                    "market_probability_for_fighter": market_probability,
                    "consensus_book_count": len(consensus),
                    "consensus_books": [item["book"] for item in consensus],
                    "source_quote_updated_at_utc": target[
                        "source_quote_updated_at_utc"
                    ],
                    "source_quote_age_seconds": target[
                        "source_quote_age_seconds"
                    ],
                    "event_start_utc": observation_records[0].source_commence_time_utc,
                    "bayesian_kelly": bayesian_kelly.assessment(
                        side_probability,
                        int(moneyline),
                    ),
                }
            )
    return candidates


def _names_match(
    matchup: Mapping[str, object], observations: list[EarlyMarketObservation]
) -> bool | None:
    first = observations[0]
    direct = same_name(
        _text(matchup.get("fighter_name")), first.source_fighter_name
    ) and same_name(
        _text(matchup.get("opponent_name")), first.source_opponent_name
    )
    reverse = same_name(
        _text(matchup.get("fighter_name")), first.source_opponent_name
    ) and same_name(
        _text(matchup.get("opponent_name")), first.source_fighter_name
    )
    if direct == reverse:
        return None
    return reverse


def _base_bet(
    matchup: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    observed_at: str,
    source: str,
) -> dict[str, object]:
    body = {
        "category": "Moneyline",
        "event_id": matchup.get("event_id"),
        "event_url": matchup.get("event_url"),
        "event_title": matchup.get("event_title"),
        "event_date": matchup.get("event_date"),
        "event_start_utc": candidate.get("event_start_utc"),
        "bout_order": matchup.get("bout_order"),
        "matchup_id": matchup.get("matchup_id"),
        "fighter_id": matchup.get("fighter_id"),
        "opponent_id": matchup.get("opponent_id"),
        "fighter_name": matchup.get("fighter_name"),
        "opponent_name": matchup.get("opponent_name"),
        "selection": candidate.get("selection"),
        "side": candidate.get("side"),
        "target_book": candidate.get("target_book"),
        "offered_moneyline": candidate.get("offered_moneyline"),
        "estimated_win_probability": candidate.get("estimated_win_probability"),
        "estimated_expected_return": candidate.get("estimated_expected_return"),
        "minimum_expected_return": MIN_EXPECTED_RETURN,
        "threshold_met": True,
        "candidate_only": False,
        "probability_source": (
            "leave_one_book_out_no_vig_market_consensus"
            if LOCKED_GAMMA == 0.0
            else "locked_market_model_log_odds_blend"
        ),
        "model_weight": LOCKED_GAMMA,
        "consensus_book_count": candidate.get("consensus_book_count"),
        "consensus_books": candidate.get("consensus_books"),
        "observed_at_utc": observed_at,
        "source": source,
        "source_quote_updated_at_utc": candidate.get(
            "source_quote_updated_at_utc"
        ),
        "source_quote_age_seconds": candidate.get("source_quote_age_seconds"),
        "paper_only": True,
        "execution_enabled": False,
        "bayesian_kelly": candidate.get("bayesian_kelly"),
    }
    body["bet_id"] = canonical_hash(body)
    return body


def _current_opportunity_bets(
    publication: Mapping[str, object] | None,
    forecasts: Mapping[str, Mapping[str, object]],
    bayesian_kelly: BayesianKellyCalibrator,
    total_bayesian_kelly: BayesianTotalCalibrator | None,
) -> list[dict[str, object]]:
    if not publication:
        return []
    event_id = _text(publication.get("event_id"))
    observed = _text(publication.get("observed_at_utc"))
    source = _text(publication.get("source"))
    rows: list[dict[str, object]] = []
    for matchup in publication.get("matchups", []):
        if not isinstance(matchup, dict):
            continue
        signal = matchup.get("current_signal")
        matchup_id = _text(matchup.get("matchup_id"))
        forecast = forecasts.get(matchup_id)
        if not isinstance(signal, dict) or not forecast:
            continue
        action = _text(signal.get("paper_action"))
        expected = _float(signal.get("estimated_expected_return"))
        if action not in {"fighter", "opponent"} or expected is None:
            continue
        target_quote = next(
            (
                quote for quote in matchup.get("book_quotes", [])
                if isinstance(quote, dict)
                and _text(quote.get("book")) == _text(signal.get("target_book"))
            ),
            {},
        )
        candidate = {
            "selection": signal.get("action_name") or signal.get("best_candidate_name"),
            "side": action,
            "target_book": signal.get("target_book"),
            "offered_moneyline": signal.get("offered_moneyline"),
            "estimated_win_probability": signal.get("market_probability"),
            "estimated_expected_return": expected,
            "consensus_book_count": signal.get("consensus_book_count"),
            "consensus_books": signal.get("consensus_books"),
            "source_quote_updated_at_utc": target_quote.get(
                "source_quote_updated_at_utc"
            ),
            "source_quote_age_seconds": target_quote.get(
                "source_quote_age_seconds"
            ),
            "event_start_utc": matchup.get("event_start_utc") or publication.get("event_start_utc"),
            "bayesian_kelly": bayesian_kelly.assessment(
                signal.get("market_probability"),
                signal.get("offered_moneyline"),
            ),
        }
        rows.append(
            _base_bet(forecast, candidate, observed_at=observed, source=source)
        )

    total_view = publication.get("prop_markets", {}).get("total_rounds", {})
    totals = total_view.get("candidate_offers", total_view.get("positive_candidates", []))
    for candidate in totals:
        if not isinstance(candidate, dict):
            continue
        expected = _float(candidate.get("estimated_expected_return"))
        matchup_id = _text(candidate.get("matchup_id"))
        forecast = forecasts.get(matchup_id)
        if expected is None or not forecast:
            continue
        side = _text(candidate.get("side")).casefold()
        line = _float(candidate.get("line"))
        side_probability = _float(candidate.get("model_probability"))
        total_assessment = unavailable_total_assessment(
            "The historical Bayesian total calibration is not available."
        )
        if (
            total_bayesian_kelly is not None
            and total_bayesian_kelly.artifact.get("schedule_contract_version") == "verified-pre-fight-schedule-v1"
            and candidate.get("schedule_contract_version") == "verified-pre-fight-schedule-v1"
            and candidate.get("model_version") == "candidate-discrete-time-competing-risks-v2-verified-schedules"
            and side in {"over", "under"}
            and line is not None
            and side_probability is not None
            and 0.0 < side_probability < 1.0
        ):
            over_probability = (
                side_probability if side == "over" else 1.0 - side_probability
            )
            total_assessment = total_bayesian_kelly.assessment(
                over_probability,
                side,
                line,
                candidate.get("offered_moneyline"),
            )
        body = {
            **{key: forecast.get(key) for key in (
                "event_id", "event_url", "event_title", "event_date",
                "event_start_utc",
                "bout_order", "matchup_id", "fighter_id", "opponent_id",
                "fighter_name", "opponent_name",
            )},
            "category": "Total rounds",
            "betting_performance_validated": candidate.get("betting_performance_validated") is True,
            "event_start_utc": candidate.get("event_start_utc") or publication.get("event_start_utc"),
            "selection": candidate.get("selection"),
            "side": candidate.get("side"),
            "target_book": candidate.get("target_book"),
            "offered_moneyline": candidate.get("offered_moneyline"),
            "estimated_win_probability": candidate.get("model_probability"),
            "estimated_expected_return": expected,
            "minimum_expected_return": MIN_EXPECTED_RETURN,
            "threshold_met": True,
            "candidate_only": True,
            "probability_source": candidate.get("probability_source")
            or "candidate_duration_model",
            "line": candidate.get("line"),
            "scheduled_rounds": candidate.get("scheduled_rounds"),
            "schedule_basis": candidate.get("schedule_basis"),
            "model_id": candidate.get("model_id"),
            "model_version": candidate.get("model_version"),
            "schedule_contract_version": candidate.get("schedule_contract_version"),
            "model_trained_through": candidate.get("model_trained_through"),
            "forecast_issued_at_utc": candidate.get("forecast_issued_at_utc"),
            "break_even_probability": candidate.get("break_even_probability"),
            "model_weight": candidate.get("selected_residual_weight"),
            "consensus_book_count": candidate.get(
                "other_book_consensus_count"
            ),
            "consensus_books": [],
            "observed_at_utc": observed,
            "source": source,
            "source_quote_updated_at_utc": candidate.get("source_quote_updated_at_utc"),
            "source_quote_age_seconds": candidate.get("source_quote_age_seconds"),
            "paper_only": True,
            "execution_enabled": False,
            "bayesian_kelly": total_assessment,
        }
        body["bet_id"] = canonical_hash(body)
        rows.append(body)
    return rows


def build_upcoming_bet_board(
    forecast_publication: Mapping[str, object],
    observations: Iterable[EarlyMarketObservation],
    *,
    observed_at_utc: object,
    source: str,
    current_opportunities: Mapping[str, object] | None = None,
    bayesian_kelly_calibrator: BayesianKellyCalibrator | None = None,
    total_bayesian_kelly_calibrator: BayesianTotalCalibrator | None = None,
    enable_total_bayesian_calibration: bool = True,
) -> dict[str, object]:
    """Rank every qualified price across all officially announced UFC cards."""

    forecasts = validate_upcoming_forecast_publication(
        dict(forecast_publication)
    )
    bayesian_kelly = (
        bayesian_kelly_calibrator or BayesianKellyCalibrator.load()
    )
    total_bayesian_kelly = total_bayesian_kelly_calibrator
    if total_bayesian_kelly is None and enable_total_bayesian_calibration:
        try:
            total_bayesian_kelly = BayesianTotalCalibrator.load()
        except (OSError, ValueError, json.JSONDecodeError):
            total_bayesian_kelly = None
    observed_at = _utc(observed_at_utc, "observed_at_utc")
    observed_text = _utc_text(observed_at, "observed_at_utc")
    official = [
        item
        for item in forecasts["matchups"]
        if str(item.get("event_date")) >= observed_at.date().isoformat()
        and item.get("matchup_id")
        and not str(item.get("model_status") or "").startswith("abstain")
    ]
    grouped: dict[str, list[EarlyMarketObservation]] = {}
    for item in observations:
        if item.market == "h2h":
            grouped.setdefault(item.source_event_id, []).append(item)
    bets_by_key: dict[tuple, dict[str, object]] = {}
    market_matchups: list[dict[str, object]] = []
    matched = evaluable = 0
    for matchup in official:
        event_day = pd.Timestamp(str(matchup["event_date"]))
        source_groups = []
        for rows in grouped.values():
            source_day = pd.Timestamp(rows[0].source_commence_time_utc).tz_localize(
                None
            ).normalize()
            if abs((source_day - event_day).days) > 1:
                continue
            reversed_orientation = _names_match(matchup, rows)
            if reversed_orientation is not None:
                source_groups.append((rows, reversed_orientation))
        if len(source_groups) != 1:
            continue
        matched += 1
        rows, reversed_orientation = source_groups[0]
        stored_quotes = _stored_book_quotes(
            rows,
            reversed_orientation=reversed_orientation,
        )
        consensus_probability = (
            sum(float(item["no_vig_fighter_probability"]) for item in stored_quotes)
            / len(stored_quotes)
            if len(stored_quotes) >= 2
            else None
        )
        market_matchups.append(
            {
                "event_id": matchup["event_id"],
                "matchup_id": matchup["matchup_id"],
                "book_count": len(stored_quotes),
                "consensus_fighter_probability": consensus_probability,
                "latest_source_quote_updated_at_utc": max(
                    str(item["source_quote_updated_at_utc"])
                    for item in stored_quotes
                ),
                "book_quotes": stored_quotes,
            }
        )
        candidates = _moneyline_offers(
            matchup,
            rows,
            reversed_orientation=reversed_orientation,
            observed_at=observed_at,
            bayesian_kelly=bayesian_kelly,
        )
        if len(_book_quotes(rows, reversed_orientation=reversed_orientation, observed_at=observed_at)) >= MIN_CONSENSUS_BOOKS + 1:
            evaluable += 1
        for candidate in candidates:
            bet = _base_bet(
                matchup,
                candidate,
                observed_at=observed_text,
                source=source,
            )
            bets_by_key[(str(bet["event_id"]), str(bet["matchup_id"]), "Moneyline", str(bet["side"]), str(bet["target_book"]))] = bet

    forecast_by_matchup = {
        str(item.get("matchup_id")): item
        for item in forecasts["matchups"]
        if item.get("matchup_id")
    }
    for bet in _current_opportunity_bets(
        current_opportunities,
        forecast_by_matchup,
        bayesian_kelly,
        total_bayesian_kelly,
    ):
        category = str(bet["category"])
        key = (
            str(bet["event_id"]),
            str(bet["matchup_id"]),
            category,
            str(bet["selection"]) if category != "Moneyline" else str(bet["side"]),
            str(bet["target_book"]),
        )
        # Complete fresh source observations take precedence over legacy views.
        bets_by_key.setdefault(key, bet)
    qualified_offers = [offer for bet in bets_by_key.values()
                        if (offer := _calibrated_offer(bet, observed_at)) is not None]
    offers = allocate_paper_offers(sorted(qualified_offers, key=_offer_rank)[:MAX_BOARD_OFFERS])
    bets = [offer for offer in offers if offer["allocated_fraction"] > 0]
    body: dict[str, object] = {
        "schema_version": BOARD_SCHEMA_VERSION,
        "policy_version": BOARD_POLICY_VERSION,
        "observed_at_utc": observed_text,
        "source": _text(source),
        "paper_only": True,
        "execution_enabled": False,
        "minimum_expected_return": MIN_EXPECTED_RETURN,
        "qualification_policy": "calibrated_mean_5pct_positive_probability_floor_and_stake",
        "total_betting_policy": "research_only_until_standalone_duration_policy_passes_price_matched_prospective_evidence",
        "allocation_policy": dict(ALLOCATION_POLICY),
        "allocated_fraction": sum(bet["allocated_fraction"] for bet in bets),
        "minimum_consensus_books_excluding_target": MIN_CONSENSUS_BOOKS,
        "model_weight": LOCKED_GAMMA,
        "bayesian_kelly": {
            "policy_version": BAYESIAN_KELLY_POLICY_VERSION,
            "calibration_artifact_sha256": bayesian_kelly.artifact[
                "artifact_sha256"
            ],
            "moneyline_only": False,
            "total_rounds": (
                {
                    "status": "available",
                    "policy_version": TOTAL_BAYESIAN_KELLY_POLICY_VERSION,
                    "calibration_artifact_sha256": total_bayesian_kelly.artifact[
                        "artifact_sha256"
                    ],
                }
                if total_bayesian_kelly is not None
                else {
                    "status": "unavailable",
                    "policy_version": TOTAL_BAYESIAN_KELLY_POLICY_VERSION,
                    "reason": "Historical total calibration is not available.",
                }
            ),
        },
        "forecast_publication_sha256": forecasts["publication_sha256"],
        "announced_event_count": forecasts["event_count"],
        "announced_matchup_count": len(official),
        "market_matched_matchup_count": matched,
        "market_evaluable_matchup_count": evaluable,
        "market_matchups": sorted(
            market_matchups,
            key=lambda item: (str(item["event_id"]), str(item["matchup_id"])),
        ),
        "qualified_bet_count": len(bets),
        "eligible_offer_count": len(offers),
        "offers": offers,
        "bets": bets,
    }
    body["publication_sha256"] = canonical_hash(body)
    return validate_upcoming_bet_board(body)


def validate_upcoming_bet_board(publication: object) -> dict[str, object]:
    if not isinstance(publication, dict):
        raise ValueError("upcoming bet board must be an object")
    supplied_hash = publication.get("publication_sha256")
    unhashed = dict(publication)
    unhashed.pop("publication_sha256", None)
    if supplied_hash != canonical_hash(unhashed):
        raise ValueError("upcoming bet board hash is invalid")
    if (
        (publication.get("schema_version"), publication.get("policy_version")) not in {
            (1, LEGACY_BOARD_POLICY_VERSION), (BOARD_SCHEMA_VERSION, BOARD_POLICY_VERSION)
        }
        or publication.get("paper_only") is not True
        or publication.get("execution_enabled") is not False
        or _float(publication.get("minimum_expected_return")) != MIN_EXPECTED_RETURN
    ):
        raise ValueError("upcoming bet board policy is invalid")
    bets = publication.get("bets")
    if not isinstance(bets, list) or publication.get("qualified_bet_count") != len(bets):
        raise ValueError("upcoming bet board count is inconsistent")
    if len(bets) > (MAX_BOARD_OFFERS if publication.get("schema_version") == BOARD_SCHEMA_VERSION else MAX_BOARD_BETS):
        raise ValueError("upcoming bet board exceeds its size bound")
    current_schema = publication.get("schema_version") == BOARD_SCHEMA_VERSION
    offers = publication.get("offers") if current_schema else bets
    if current_schema:
        if not isinstance(offers, list) or len(offers) > MAX_BOARD_OFFERS or publication.get("eligible_offer_count") != len(offers):
            raise ValueError("upcoming bet board offer count is inconsistent")
        if publication.get("allocation_policy") != ALLOCATION_POLICY:
            raise ValueError("upcoming bet board allocation policy is invalid")
    market_matchups = publication.get("market_matchups")
    if market_matchups is not None:
        if not isinstance(market_matchups, list) or publication.get(
            "market_matched_matchup_count"
        ) != len(market_matchups):
            raise ValueError("upcoming bet board market-matchup count is inconsistent")
        identities: list[tuple[str, str]] = []
        for matchup in market_matchups:
            if not isinstance(matchup, dict):
                raise ValueError("upcoming bet board contains a non-object market matchup")
            identity = (
                _text(matchup.get("event_id")),
                _text(matchup.get("matchup_id")),
            )
            book_count = _integer(matchup.get("book_count"))
            if not all(identity) or book_count is None or book_count < 1:
                raise ValueError("upcoming bet board contains invalid market availability")
            quotes = matchup.get("book_quotes")
            # Historical publications contain only the availability count.
            # Validate the richer contract whenever stored prices are present.
            if quotes is not None:
                if not isinstance(quotes, list) or len(quotes) != book_count:
                    raise ValueError("upcoming bet board stored-price count is inconsistent")
                book_keys: list[str] = []
                for quote in quotes:
                    if not isinstance(quote, dict):
                        raise ValueError("upcoming bet board contains a non-object stored price")
                    book_key = _text(quote.get("book_key")).casefold()
                    fighter_line = _integer(quote.get("fighter_moneyline"))
                    opponent_line = _integer(quote.get("opponent_moneyline"))
                    probability = _float(quote.get("no_vig_fighter_probability"))
                    if (
                        not _text(quote.get("book"))
                        or not book_key
                        or fighter_line is None
                        or opponent_line is None
                        or fighter_line == 0
                        or opponent_line == 0
                        or abs(fighter_line) < 100
                        or abs(opponent_line) < 100
                        or probability is None
                        or not 0.0 < probability < 1.0
                    ):
                        raise ValueError("upcoming bet board contains an invalid stored price")
                    _utc(quote.get("source_quote_updated_at_utc"), "stored quote update")
                    _utc(quote.get("first_observed_at_utc"), "stored quote observation")
                    book_keys.append(book_key)
                if len(set(book_keys)) != len(book_keys):
                    raise ValueError("upcoming bet board repeats a stored book price")
                consensus = _float(matchup.get("consensus_fighter_probability"))
                if len(quotes) >= 2 and (consensus is None or not 0.0 < consensus < 1.0):
                    raise ValueError("upcoming bet board stored consensus is invalid")
                _utc(
                    matchup.get("latest_source_quote_updated_at_utc"),
                    "latest stored quote update",
                )
            identities.append(identity)
        if len(set(identities)) != len(identities):
            raise ValueError("upcoming bet board repeats market availability")
        if market_matchups != sorted(
            market_matchups,
            key=lambda item: (str(item.get("event_id")), str(item.get("matchup_id"))),
        ):
            raise ValueError("upcoming bet board market availability is not sorted")
    values: list[float] = []
    ids: list[str] = []
    for bet in [*offers, *bets] if current_schema else bets:
        if not isinstance(bet, dict):
            raise ValueError("upcoming bet board contains a non-object bet")
        expected = _float(bet.get("estimated_expected_return"))
        if (
            expected is None
            or expected < MIN_EXPECTED_RETURN
            or bet.get("threshold_met") is not True
            or type(bet.get("candidate_only")) is not bool
            or bet.get("paper_only") is not True
            or bet.get("execution_enabled") is not False
        ):
            raise ValueError("upcoming bet board contains a below-policy bet")
        probability = _float(bet.get("estimated_win_probability"))
        moneyline = _integer(bet.get("offered_moneyline"))
        if probability is None or not 0.0 < probability < 1.0:
            raise ValueError("upcoming bet board contains an invalid probability")
        if moneyline is None or moneyline == 0 or abs(moneyline) < 100:
            raise ValueError("upcoming bet board contains an invalid moneyline")
        if _float(bet.get("minimum_expected_return")) != MIN_EXPECTED_RETURN:
            raise ValueError("upcoming bet board bet threshold is inconsistent")
        if current_schema:
            if (not _text(bet.get("event_id")) or not _text(bet.get("fighter_id"))
                    or not _text(bet.get("opponent_id"))
                    or bet.get("fighter_id") == bet.get("opponent_id")):
                raise ValueError("upcoming bet board lacks a unique physical fight identity")
            assessment = bet.get("bayesian_kelly") or {}
            mean = _float(assessment.get("posterior_mean_probability"))
            lower = _float(assessment.get("posterior_lower_probability"))
            stake = _float(assessment.get("recommended_fraction"))
            raw_probability = _float(bet.get("raw_estimated_win_probability"))
            raw_expected = _float(bet.get("raw_estimated_expected_return"))
            floor_expected = _float(bet.get("robust_lower_expected_return"))
            allocated = _float(bet.get("allocated_fraction"))
            decimal = _decimal_odds(moneyline)
            if (assessment.get("status") != "available" or mean is None or lower is None
                    or stake is None or stake <= 0 or raw_probability is None or not 0 < raw_probability < 1 or raw_expected is None
                    or floor_expected is None or floor_expected <= 0
                    or allocated is None or not 0 <= allocated <= 0.01
                    or abs(probability - mean) > 1e-12
                    or abs(expected - (probability * decimal - 1)) > 1e-12
                    or abs(raw_expected - (raw_probability * decimal - 1)) > 1e-12
                    or _float(assessment.get("nominal_probability")) is None
                    or abs(float(assessment["nominal_probability"]) - raw_probability) > 1e-12
                    or abs(floor_expected - (lower * decimal - 1)) > 1e-12):
                raise ValueError("upcoming bet board calibrated probability or allocation is invalid")
            observed = _utc(publication.get("observed_at_utc"), "board observation")
            updated = _utc(bet.get("source_quote_updated_at_utc"), "source update")
            start = _utc(bet.get("event_start_utc"), "event start")
            if start <= observed or not 0 <= (observed - updated).total_seconds() <= MAX_SOURCE_QUOTE_AGE_SECONDS:
                raise ValueError("upcoming bet board offer is expired or started")
            if bet.get("category") == "Total rounds" and (
                    bet.get("schedule_contract_version") != "verified-pre-fight-schedule-v1"
                    or assessment.get("schedule_contract_version") != "verified-pre-fight-schedule-v1"
                    or bet.get("betting_performance_validated") is not True
                    or bet.get("model_version") != "candidate-discrete-time-competing-risks-v2-verified-schedules"):
                raise ValueError("upcoming bet board total lacks verified pre-fight schedule")
        if current_schema or publication.get("bayesian_kelly") is not None:
            if bet.get("category") == "Total rounds":
                validate_total_bayesian_kelly_assessment(
                    bet.get("bayesian_kelly")
                )
            else:
                validate_bayesian_kelly_assessment(bet.get("bayesian_kelly"))
        supplied_bet_id = _text(bet.get("bet_id"))
        unhashed_bet = dict(bet)
        unhashed_bet.pop("bet_id", None)
        if supplied_bet_id != canonical_hash(unhashed_bet):
            raise ValueError("upcoming bet board bet ID is invalid")
        if not current_schema or bet in offers:
            # Each funded bet is also an offer and is validated above; collect
            # the offer list once for ordering and uniqueness below.
            if not current_schema or supplied_bet_id not in ids:
                values.append(expected)
                ids.append(supplied_bet_id)
    if values != sorted(values, reverse=True):
        raise ValueError("upcoming bet board is not sorted by expected return")
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("upcoming bet board bet IDs are invalid")
    if current_schema:
        if offers != allocate_paper_offers(offers):
            raise ValueError("upcoming bet board does not match capped allocation")
        funded = [offer for offer in offers if offer["allocated_fraction"] > 0]
        if bets != funded or len(ids) != len(offers):
            raise ValueError("upcoming bet board funded bets differ from eligible offers")
        if abs(float(publication.get("allocated_fraction", -1)) - sum(bet["allocated_fraction"] for bet in bets)) > 1e-12:
            raise ValueError("upcoming bet board total allocation is inconsistent")
    _utc(publication.get("observed_at_utc"), "observed_at_utc")
    return publication


def write_upcoming_bet_board(
    publication: Mapping[str, object],
    path: Path = UPCOMING_BOARD_PATH,
) -> None:
    _atomic_json(path, validate_upcoming_bet_board(dict(publication)))


def bootstrap_current_publications() -> tuple[dict[str, object], dict[str, object]]:
    """Seed the new files from the existing current-card publications."""

    card = json.loads((EXTERNAL_ROOT / "card_info.json").read_text(encoding="utf-8"))
    vegas = pd.read_json(EXTERNAL_ROOT / "vegas_odds.json")
    opportunities = json.loads(
        (MARKET_ROOT / "current_opportunities.json").read_text(encoding="utf-8")
    )
    vegas["event title"] = card["title"]
    vegas["bout order"] = list(range(len(vegas)))
    generated = _text(vegas["forecast issued at"].iloc[0])
    forecasts = build_upcoming_forecast_publication(
        vegas, generated_at_utc=generated
    )
    board = build_upcoming_bet_board(
        forecasts,
        (),
        observed_at_utc=opportunities["observed_at_utc"],
        source=str(opportunities.get("source") or "current-opportunities"),
        current_opportunities=opportunities,
    )
    write_upcoming_forecast_publication(forecasts)
    write_upcoming_bet_board(board)
    return forecasts, board


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-current", action="store_true")
    arguments = parser.parse_args()
    if not arguments.bootstrap_current:
        parser.error("choose --bootstrap-current")
    forecasts, board = bootstrap_current_publications()
    print(
        f"Published {forecasts['event_count']} event(s), "
        f"{forecasts['matchup_count']} matchups, and "
        f"{board['qualified_bet_count']} qualified bet(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
