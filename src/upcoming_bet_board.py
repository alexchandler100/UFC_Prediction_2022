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

from fight_stat_helpers import same_name
from market_tracker import (
    LOCKED_GAMMA,
    MIN_EXPECTED_RETURN,
    EarlyMarketObservation,
    matchup_id_for,
)
from market_tracker._common import canonical_hash
from market_tracker.paper import symmetric_logit_blend
from market_tracker.prospective import MIN_CONSENSUS_BOOKS


ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = ROOT / "content/data/external"
MARKET_ROOT = ROOT / "content/data/market"
UPCOMING_FORECAST_PATH = EXTERNAL_ROOT / "all_upcoming_forecasts.json"
UPCOMING_BOARD_PATH = MARKET_ROOT / "upcoming_bet_board.json"
FORECAST_SCHEMA_VERSION = 1
BOARD_SCHEMA_VERSION = 1
FORECAST_VERSION = "all-announced-ufc-forecasts-v1"
BOARD_POLICY_VERSION = "all-upcoming-qualified-bets-v1"
MAX_SOURCE_QUOTE_AGE_SECONDS = 30.0 * 60.0
MAX_BOARD_BETS = 100


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


def _qualified_moneyline(
    matchup: Mapping[str, object],
    observations: Iterable[EarlyMarketObservation],
    *,
    reversed_orientation: bool,
    observed_at: datetime,
) -> dict[str, object] | None:
    observation_records = tuple(observations)
    quotes = _book_quotes(
        observation_records,
        reversed_orientation=reversed_orientation,
        observed_at=observed_at,
    )
    if len(quotes) < MIN_CONSENSUS_BOOKS + 1:
        return None
    model_probability = _float(matchup.get("model_probability_for_fighter"))
    if model_probability is None or not 0.0 < model_probability < 1.0:
        return None
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
                }
            )
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda item: (
            -float(item["estimated_expected_return"]),
            str(item["target_book"]).casefold(),
            str(item["side"]),
        ),
    )
    if float(selected["estimated_expected_return"]) < MIN_EXPECTED_RETURN:
        return None
    return selected


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
    }
    body["bet_id"] = canonical_hash(body)
    return body


def _current_opportunity_bets(
    publication: Mapping[str, object] | None,
    forecasts: Mapping[str, Mapping[str, object]],
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
        if action not in {"fighter", "opponent"} or expected is None or expected < MIN_EXPECTED_RETURN:
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
            "event_start_utc": matchup.get("event_start_utc"),
        }
        rows.append(
            _base_bet(forecast, candidate, observed_at=observed, source=source)
        )

    totals = (
        publication.get("prop_markets", {})
        .get("total_rounds", {})
        .get("positive_candidates", [])
    )
    for candidate in totals:
        if not isinstance(candidate, dict) or candidate.get("paper_threshold_met") is not True:
            continue
        expected = _float(candidate.get("estimated_expected_return"))
        matchup_id = _text(candidate.get("matchup_id"))
        forecast = forecasts.get(matchup_id)
        if expected is None or expected < MIN_EXPECTED_RETURN or not forecast:
            continue
        body = {
            **{key: forecast.get(key) for key in (
                "event_id", "event_url", "event_title", "event_date",
                "event_start_utc",
                "bout_order", "matchup_id", "fighter_id", "opponent_id",
                "fighter_name", "opponent_name",
            )},
            "category": "Total rounds",
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
            "model_weight": candidate.get("selected_residual_weight"),
            "consensus_book_count": candidate.get(
                "other_book_consensus_count"
            ),
            "consensus_books": [],
            "observed_at_utc": observed,
            "source": source,
            "source_quote_updated_at_utc": None,
            "source_quote_age_seconds": candidate.get("source_quote_age_seconds"),
            "paper_only": True,
            "execution_enabled": False,
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
) -> dict[str, object]:
    """Rank every qualified price across all officially announced UFC cards."""

    forecasts = validate_upcoming_forecast_publication(
        dict(forecast_publication)
    )
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
    bets_by_key: dict[tuple[str, str, str, str], dict[str, object]] = {}
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
        market_matchups.append(
            {
                "event_id": matchup["event_id"],
                "matchup_id": matchup["matchup_id"],
                "book_count": len(
                    {
                        (item.source_book_key or item.book).casefold()
                        for item in rows
                    }
                ),
            }
        )
        candidate = _qualified_moneyline(
            matchup,
            rows,
            reversed_orientation=reversed_orientation,
            observed_at=observed_at,
        )
        if len(_book_quotes(rows, reversed_orientation=reversed_orientation, observed_at=observed_at)) >= MIN_CONSENSUS_BOOKS + 1:
            evaluable += 1
        if candidate:
            bet = _base_bet(
                matchup,
                candidate,
                observed_at=observed_text,
                source=source,
            )
            bets_by_key[(str(bet["event_id"]), str(bet["matchup_id"]), "Moneyline", "")] = bet

    forecast_by_matchup = {
        str(item.get("matchup_id")): item
        for item in forecasts["matchups"]
        if item.get("matchup_id")
    }
    for bet in _current_opportunity_bets(current_opportunities, forecast_by_matchup):
        category = str(bet["category"])
        key = (
            str(bet["event_id"]),
            str(bet["matchup_id"]),
            category,
            str(bet["selection"]) if category != "Moneyline" else "",
        )
        bets_by_key[key] = bet
    bets = sorted(
        bets_by_key.values(),
        key=lambda item: (
            -float(item["estimated_expected_return"]),
            str(item["event_date"]),
            str(item["selection"]).casefold(),
            str(item["bet_id"]),
        ),
    )[:MAX_BOARD_BETS]
    body: dict[str, object] = {
        "schema_version": BOARD_SCHEMA_VERSION,
        "policy_version": BOARD_POLICY_VERSION,
        "observed_at_utc": observed_text,
        "source": _text(source),
        "paper_only": True,
        "execution_enabled": False,
        "minimum_expected_return": MIN_EXPECTED_RETURN,
        "minimum_consensus_books_excluding_target": MIN_CONSENSUS_BOOKS,
        "model_weight": LOCKED_GAMMA,
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
        publication.get("schema_version") != BOARD_SCHEMA_VERSION
        or publication.get("policy_version") != BOARD_POLICY_VERSION
        or publication.get("paper_only") is not True
        or publication.get("execution_enabled") is not False
        or _float(publication.get("minimum_expected_return")) != MIN_EXPECTED_RETURN
    ):
        raise ValueError("upcoming bet board policy is invalid")
    bets = publication.get("bets")
    if not isinstance(bets, list) or publication.get("qualified_bet_count") != len(bets):
        raise ValueError("upcoming bet board count is inconsistent")
    if len(bets) > MAX_BOARD_BETS:
        raise ValueError("upcoming bet board exceeds its size bound")
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
    for bet in bets:
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
        supplied_bet_id = _text(bet.get("bet_id"))
        unhashed_bet = dict(bet)
        unhashed_bet.pop("bet_id", None)
        if supplied_bet_id != canonical_hash(unhashed_bet):
            raise ValueError("upcoming bet board bet ID is invalid")
        values.append(expected)
        ids.append(supplied_bet_id)
    if values != sorted(values, reverse=True):
        raise ValueError("upcoming bet board is not sorted by expected return")
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("upcoming bet board bet IDs are invalid")
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
