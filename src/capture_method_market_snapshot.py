"""Capture bounded UFC method-of-victory prices from BestFightOdds.

The command is designed to run immediately after the existing moneyline/totals
capture, whose report supplies the verified current UFC card and start time.
It stores at most four snapshots per book/fight/card: first available, about
72 hours, 24 hours, and 6 hours before the card.  All output is paper-only
market research; there is no wager or recommendation code here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import pandas as pd
import requests

from audit_historical_odds_sources import BESTFIGHTODDS_ROOT
from backfill_bestfightodds_history import _robots_allows_public_paths
from bestfightodds_props import (
    METHODS,
    MethodPropSelection,
    available_method_markets,
    complete_method_markets,
    parse_bestfightodds_method_props,
)
from capture_market_snapshot import (
    BAYESIAN_MODEL_PATH,
    CARD_PATH,
    MODEL_PATH,
    REPORT_PATH as MONEYLINE_REPORT_PATH,
    VEGAS_PATH,
    CaptureError,
    PublishedMatchup,
    _as_utc,
    _json_object,
    _publication_payloads,
    _published_matchups,
    _text,
)
from fight_stat_helpers import same_name
from fight_predictor.outcome_publication import validate_outcome_forecast_publication
from market_tracker import (
    METHOD_MARKET_CONTRACT,
    MethodForecastCapture,
    MethodForecastStore,
    MethodMarketSnapshot,
    MethodMarketStore,
)
from market_tracker._common import StoreIntegrityError, canonical_hash
from market_tracker._storage import atomic_write_text


ROOT = Path(__file__).resolve().parent
MARKET_ROOT = ROOT / "content" / "data" / "market"
METHOD_CSV_PATH = MARKET_ROOT / "method_market_snapshots.csv"
METHOD_JSONL_PATH = MARKET_ROOT / "method_market_snapshots.jsonl"
METHOD_FORECAST_CSV_PATH = MARKET_ROOT / "method_forecast_captures.csv"
METHOD_FORECAST_JSONL_PATH = MARKET_ROOT / "method_forecast_captures.jsonl"
REPORT_PATH = MARKET_ROOT / "method_capture_report.json"
CURRENT_METHOD_PATH = MARKET_ROOT / "current_method_markets.json"
OUTCOME_FORECAST_PATH = ROOT / "content" / "data" / "external" / "outcome_forecasts.json"
SOURCE = "bestfightodds.com"
SOURCE_POLICY_ENV = "BESTFIGHTODDS_SOURCE_POLICY_ACKNOWLEDGED"
REPORT_SIZE_LIMIT = 32 * 1024
PUBLICATION_SIZE_LIMIT = 512 * 1024
SOURCE_TIMEOUT_SECONDS = 45.0


class MethodCaptureSkipped(RuntimeError):
    """Expected no-op because no capture horizon is currently due."""


def _capture_id(observed: datetime, payload_sha: str) -> str:
    stamp = observed.strftime("%Y%m%dT%H%M%S%fZ")
    return f"method_capture_{stamp}_{payload_sha[:12]}"


def _fetch(url: str) -> requests.Response:
    session = requests.Session()
    session.headers.update({"User-Agent": "UFC-Research-Method-Capture/1.0"})
    last: requests.RequestException | None = None
    for attempt in range(2):
        try:
            response = session.get(url, timeout=SOURCE_TIMEOUT_SECONDS)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 0:
                    time.sleep(3.0)
                    continue
            response.raise_for_status()
            if len(response.content) > 8 * 1024 * 1024:
                raise CaptureError("BestFightOdds response exceeded the 8 MiB safety cap")
            return response
        except requests.RequestException as error:
            last = error
            if attempt == 0:
                time.sleep(2.0)
    assert last is not None
    raise CaptureError(f"BestFightOdds method retrieval failed: {type(last).__name__}")


def _load_publication(observed: datetime) -> tuple[
    str, str, tuple[PublishedMatchup, ...], str, float
]:
    payloads = _publication_payloads()
    card = _json_object(payloads[CARD_PATH], CARD_PATH)
    artifact = _json_object(payloads[MODEL_PATH], MODEL_PATH)
    bayesian = _json_object(payloads[BAYESIAN_MODEL_PATH], BAYESIAN_MODEL_PATH)
    try:
        vegas = pd.read_json(io.BytesIO(payloads[VEGAS_PATH]))
    except (TypeError, ValueError) as error:
        raise CaptureError("vegas_odds.json cannot be loaded as a table") from error
    event_day, event_id, _, _, _, published = _published_matchups(
        vegas, card, artifact, bayesian, observed
    )
    if not MONEYLINE_REPORT_PATH.is_file():
        raise CaptureError(
            "method capture requires a successful current moneyline capture report"
        )
    report = _json_object(MONEYLINE_REPORT_PATH.read_bytes(), MONEYLINE_REPORT_PATH)
    if _text(report.get("event_id")) != event_id:
        raise CaptureError("moneyline capture report is for a different UFC card")
    if report.get("timing_precision") != "timestamp":
        raise CaptureError("moneyline capture report lacks a verified card start time")
    event_start = _as_utc(report.get("event_start_utc"), "event_start_utc")
    if observed >= event_start:
        raise MethodCaptureSkipped("the published UFC card has already commenced")
    lead_seconds = (event_start - observed).total_seconds()
    return event_day, event_id, published, event_start.isoformat().replace("+00:00", "Z"), lead_seconds


def _timed_horizon(lead_seconds: float) -> str | None:
    lead_hours = lead_seconds / 3600.0
    if 66.0 <= lead_hours <= 72.5:
        return "t72"
    if 18.0 <= lead_hours <= 24.5:
        return "t24"
    if 0.5 <= lead_hours <= 6.5:
        return "t6"
    return None


def _capture_is_due(
    event_records: Sequence[MethodMarketSnapshot], timed_horizon: str | None
) -> bool:
    if not event_records:
        return True
    if timed_horizon is None:
        return False
    return timed_horizon not in {record.horizon for record in event_records}


def _source_matches(
    selections: Sequence[MethodPropSelection],
    published: Sequence[PublishedMatchup],
) -> dict[int, tuple[PublishedMatchup, bool]]:
    pairs: dict[int, tuple[str, str]] = {}
    for selection in selections:
        pair = selection.fighter_1_name, selection.fighter_2_name
        previous = pairs.setdefault(selection.source_matchup_id, pair)
        if previous != pair:
            raise CaptureError("one BestFightOdds matchup has conflicting fighter names")
    output: dict[int, tuple[PublishedMatchup, bool]] = {}
    for source_matchup_id, (source_1, source_2) in pairs.items():
        candidates: list[tuple[PublishedMatchup, bool]] = []
        for matchup in published:
            if matchup.matchup_id is None:
                continue
            direct = same_name(source_1, matchup.fighter_name) and same_name(
                source_2, matchup.opponent_name
            )
            reverse = same_name(source_1, matchup.opponent_name) and same_name(
                source_2, matchup.fighter_name
            )
            if direct:
                candidates.append((matchup, False))
            if reverse:
                candidates.append((matchup, True))
        if not candidates:
            continue
        if len(candidates) != 1:
            raise CaptureError(
                f"BestFightOdds matchup {source_1!r} vs {source_2!r} is ambiguous"
            )
        matchup, reversed_source = candidates[0]
        output[source_matchup_id] = matchup, reversed_source
    return output


def _book_names(
    selections: Sequence[MethodPropSelection],
) -> dict[tuple[int, int], str]:
    output: dict[tuple[int, int], str] = {}
    for selection in selections:
        for price in selection.book_prices:
            key = selection.source_matchup_id, price.book_id
            previous = output.setdefault(key, price.book_name)
            if previous.casefold() != price.book_name.casefold():
                raise CaptureError("one BestFightOdds book ID has conflicting names")
    return output


def _build_snapshots(
    *,
    selections: Sequence[MethodPropSelection],
    published: Sequence[PublishedMatchup],
    event_day: str,
    event_id: str,
    event_start_utc: str,
    observed: datetime,
    payload_sha: str,
    existing: Sequence[MethodMarketSnapshot],
    timed_horizon: str | None,
) -> tuple[list[MethodMarketSnapshot], dict[str, int]]:
    matches = _source_matches(selections, published)
    available = available_method_markets(selections)
    complete = complete_method_markets(selections)
    source_ids_by_published: dict[str, list[int]] = {}
    for source_matchup_id, (matchup, _reversed_source) in matches.items():
        if matchup.matchup_id is not None:
            source_ids_by_published.setdefault(matchup.matchup_id, []).append(
                source_matchup_id
            )
    names = _book_names(selections)
    merged: dict[tuple[str, int], dict[str, object]] = {}
    ambiguous_boards: set[tuple[str, int]] = set()
    for (source_matchup_id, book_id), prices in sorted(available.items()):
        mapped = matches.get(source_matchup_id)
        if mapped is None:
            continue
        matchup, reversed_source = mapped
        if (
            matchup.matchup_id is None
            or matchup.fighter_id is None
            or matchup.opponent_id is None
        ):
            continue
        board_key = matchup.matchup_id, book_id
        board = merged.setdefault(
            board_key,
            {
                "matchup": matchup,
                "book_name": names[(source_matchup_id, book_id)],
                "source_ids": set(),
                "prices": {},
            },
        )
        if str(board["book_name"]).casefold() != names[
            (source_matchup_id, book_id)
        ].casefold():
            ambiguous_boards.add(board_key)
            continue
        source_ids = board["source_ids"]
        merged_prices = board["prices"]
        assert isinstance(source_ids, set) and isinstance(merged_prices, dict)
        source_ids.add(source_matchup_id)
        for (source_side, method), price in prices.items():
            published_side = (
                "opponent"
                if (source_side == 1 and reversed_source)
                or (source_side == 2 and not reversed_source)
                else "fighter"
            )
            price_key = published_side, method
            prior = merged_prices.get(price_key)
            if prior is not None and prior != price:
                # Two source IDs for the same current pair should be
                # complementary. A conflicting displayed price makes the
                # whole book board ambiguous, so omit it rather than guess.
                ambiguous_boards.add(board_key)
                continue
            merged_prices[price_key] = price

    existing_keys = {item.natural_key for item in existing}
    capture_id = _capture_id(observed, payload_sha)
    output: list[MethodMarketSnapshot] = []
    matched_fights_with_prices: set[str] = set()
    merged_complete_count = 0
    for (matchup_id, book_id), board in sorted(merged.items()):
        if (matchup_id, book_id) in ambiguous_boards:
            continue
        matchup = board["matchup"]
        prices = board["prices"]
        source_ids = board["source_ids"]
        assert isinstance(matchup, PublishedMatchup)
        assert isinstance(prices, dict) and isinstance(source_ids, set)
        matched_fights_with_prices.add(matchup_id)
        fighter_prices = {
            method: prices[("fighter", method)]
            for method in METHODS
            if ("fighter", method) in prices
        }
        opponent_prices = {
            method: prices[("opponent", method)]
            for method in METHODS
            if ("opponent", method) in prices
        }
        if len(fighter_prices) + len(opponent_prices) == 6:
            merged_complete_count += 1
        for horizon in ("opening", timed_horizon):
            if horizon is None:
                continue
            natural_key = (
                matchup_id,
                horizon,
                SOURCE,
                f"book_{book_id}",
            )
            if natural_key in existing_keys:
                continue
            snapshot = MethodMarketSnapshot.create(
                capture_id=capture_id,
                matchup_id=matchup_id,
                fight_id=matchup.fight_id,
                event_id=event_id,
                fighter_id=matchup.fighter_id,
                opponent_id=matchup.opponent_id,
                fighter_name=matchup.fighter_name,
                opponent_name=matchup.opponent_name,
                event_date=event_day,
                timing_precision="timestamp",
                event_start_utc=event_start_utc,
                observed_at_utc=observed,
                source=SOURCE,
                source_event_id="matchups_" + "_".join(
                    str(value) for value in sorted(source_ids)
                ),
                source_book_key=f"book_{book_id}",
                book=str(board["book_name"]),
                horizon=horizon,
                fighter_prices=fighter_prices,
                opponent_prices=opponent_prices,
                source_payload_sha256=payload_sha,
            )
            output.append(snapshot)
            existing_keys.add(snapshot.natural_key)
    return output, {
        "source_method_selections": len(selections),
        "source_nonempty_book_markets": len(available),
        "source_complete_six_way_markets": len(complete),
        "source_matchups_mapped_to_card": len(matches),
        "source_duplicate_matchups_merged": sum(
            max(0, len(source_ids) - 1)
            for source_ids in source_ids_by_published.values()
        ),
        "ambiguous_book_markets_discarded": len(ambiguous_boards),
        "merged_complete_six_way_markets": merged_complete_count,
        "matched_fights_with_prices": len(matched_fights_with_prices),
    }


def _decimal_odds(moneyline: int) -> float:
    return 1.0 + (moneyline / 100.0 if moneyline > 0 else 100.0 / abs(moneyline))


def _outcome_forecasts(event_id: str) -> dict[str, object] | None:
    if not OUTCOME_FORECAST_PATH.is_file():
        return None
    publication = validate_outcome_forecast_publication(_json_object(
        OUTCOME_FORECAST_PATH.read_bytes(), OUTCOME_FORECAST_PATH
    ))
    if _text(publication.get("event_id")) != event_id:
        return None
    return publication


def _build_method_forecast_captures(
    snapshots: Sequence[MethodMarketSnapshot],
    *,
    outcome_forecasts: Mapping[str, object] | None,
    existing: Sequence[MethodForecastCapture],
) -> tuple[list[MethodForecastCapture], int]:
    """Freeze the exact candidate probabilities paired with new price rows."""

    if outcome_forecasts is None:
        return [], len({(row.matchup_id, row.horizon) for row in snapshots})
    raw_matchups = outcome_forecasts.get("matchups")
    if not isinstance(raw_matchups, list):
        raise CaptureError("outcome forecast matchups must be a list")
    by_pair: dict[frozenset[str], Mapping[str, object]] = {}
    for raw in raw_matchups:
        if not isinstance(raw, Mapping):
            continue
        fighter_id = _text(raw.get("fighter_id"))
        opponent_id = _text(raw.get("opponent_id"))
        if not fighter_id or not opponent_id or fighter_id == opponent_id:
            continue
        pair = frozenset((fighter_id, opponent_id))
        if pair in by_pair:
            raise CaptureError("outcome forecasts repeat one fighter pair")
        by_pair[pair] = raw
    unique_prices: dict[tuple[str, str], MethodMarketSnapshot] = {}
    for snapshot in snapshots:
        unique_prices.setdefault((snapshot.matchup_id, snapshot.horizon), snapshot)
    existing_keys = {record.natural_key for record in existing}
    output: list[MethodForecastCapture] = []
    missing = 0
    for natural_key, snapshot in sorted(unique_prices.items()):
        if natural_key in existing_keys:
            continue
        forecast = by_pair.get(frozenset((snapshot.fighter_id, snapshot.opponent_id)))
        if forecast is None:
            missing += 1
            continue
        terminal = forecast.get("terminal_probabilities")
        if not isinstance(terminal, Mapping):
            raise CaptureError("one outcome forecast lacks terminal probabilities")
        output.append(
            MethodForecastCapture.create(
                capture_id=snapshot.capture_id,
                matchup_id=snapshot.matchup_id,
                event_id=snapshot.event_id,
                fighter_id=forecast.get("fighter_id"),
                opponent_id=forecast.get("opponent_id"),
                fighter_name=forecast.get("fighter_name"),
                opponent_name=forecast.get("opponent_name"),
                event_date=snapshot.event_date,
                event_start_utc=snapshot.event_start_utc,
                observed_at_utc=snapshot.observed_at_utc,
                horizon=snapshot.horizon,
                forecast_issued_at_utc=outcome_forecasts.get("forecast_issued_at_utc"),
                model_id=outcome_forecasts.get("model_id"),
                model_version=outcome_forecasts.get("model_version"),
                model_trained_through=outcome_forecasts.get("model_trained_through"),
                source_commit_sha=outcome_forecasts.get("source_commit_sha"),
                training_input_sha256=outcome_forecasts.get("training_input_sha256"),
                source_publication_sha256=outcome_forecasts.get("publication_sha256"),
                scheduled_rounds=forecast.get("scheduled_rounds"),
                terminal_probabilities=terminal,
            )
        )
    return output, missing


def _build_current_method_publication(
    records: Sequence[MethodMarketSnapshot],
    *,
    event_id: str,
    event_date: str,
    event_start_utc: str,
    outcome_forecasts: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the bounded website view from immutable method-price rows."""

    event_records = [record for record in records if record.event_id == event_id]
    contracts = {
        (record.event_date, record.event_start_utc, record.contract_version)
        for record in event_records
    }
    if len(contracts) > 1:
        raise StoreIntegrityError("current method records have conflicting event contracts")

    forecast_rows = (
        outcome_forecasts.get("matchups", []) if outcome_forecasts is not None else []
    )
    if not isinstance(forecast_rows, list):
        raise CaptureError("outcome forecast matchups must be a list")
    forecasts: dict[frozenset[str], Mapping[str, object]] = {}
    for raw in forecast_rows:
        if not isinstance(raw, Mapping):
            continue
        fighter_id = _text(raw.get("fighter_id"))
        opponent_id = _text(raw.get("opponent_id"))
        if not fighter_id or not opponent_id or fighter_id == opponent_id:
            continue
        key = frozenset((fighter_id, opponent_id))
        if key in forecasts:
            raise CaptureError("outcome forecasts repeat one fighter pair")
        forecasts[key] = raw

    # A book can have opening/T-72/T-24/T-6 observations. The website shows the
    # newest immutable observation while the ledger preserves every horizon.
    latest: dict[tuple[str, str], MethodMarketSnapshot] = {}
    for record in event_records:
        key = record.matchup_id, record.source_book_key
        previous = latest.get(key)
        if previous is None or (
            record.observed_at_utc,
            record.horizon,
            record.quote_id,
        ) > (
            previous.observed_at_utc,
            previous.horizon,
            previous.quote_id,
        ):
            latest[key] = record

    grouped: dict[str, list[MethodMarketSnapshot]] = {}
    for record in latest.values():
        grouped.setdefault(record.matchup_id, []).append(record)

    method_markets: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for matchup_id, quotes in grouped.items():
        first = quotes[0]
        forecast = forecasts.get(frozenset((first.fighter_id, first.opponent_id)))
        if forecast is not None:
            display_fighter_id = _text(forecast.get("fighter_id"))
            display_opponent_id = _text(forecast.get("opponent_id"))
            display_fighter_name = _text(forecast.get("fighter_name"))
            display_opponent_name = _text(forecast.get("opponent_name"))
            bout_order = forecast.get("bout_order")
            terminal = forecast.get("terminal_probabilities")
            terminal_probabilities = terminal if isinstance(terminal, Mapping) else {}
        else:
            display_fighter_id = first.fighter_id
            display_opponent_id = first.opponent_id
            display_fighter_name = first.fighter_name
            display_opponent_name = first.opponent_name
            bout_order = None
            terminal_probabilities = {}

        if {display_fighter_id, display_opponent_id} != {
            first.fighter_id,
            first.opponent_id,
        }:
            raise CaptureError("method price and outcome forecast fighter IDs disagree")
        source_side_for_display = {
            "fighter": "fighter"
            if display_fighter_id == first.fighter_id
            else "opponent",
            "opponent": "opponent"
            if display_opponent_id == first.opponent_id
            else "fighter",
        }
        book_views: list[dict[str, object]] = []
        for quote in sorted(quotes, key=lambda item: (item.book.casefold(), item.quote_id)):
            selections: list[dict[str, object]] = []
            for display_side, selected_name in (
                ("fighter", display_fighter_name),
                ("opponent", display_opponent_name),
            ):
                source_side = source_side_for_display[display_side]
                for method in METHODS:
                    raw_moneyline = getattr(
                        quote, f"{source_side}_{method}_moneyline"
                    )
                    if raw_moneyline is None:
                        continue
                    moneyline = int(raw_moneyline)
                    implied = float(
                        getattr(quote, f"{source_side}_{method}_implied_probability")
                    )
                    raw_no_vig = getattr(
                        quote, f"{source_side}_{method}_no_vig_probability"
                    )
                    no_vig = float(raw_no_vig) if raw_no_vig is not None else None
                    model_value = terminal_probabilities.get(
                        f"{display_side}_{method}"
                    )
                    model_probability = (
                        float(model_value)
                        if isinstance(model_value, (int, float))
                        and math.isfinite(float(model_value))
                        and 0.0 <= float(model_value) <= 1.0
                        else None
                    )
                    estimated_return = (
                        model_probability * _decimal_odds(moneyline) - 1.0
                        if model_probability is not None
                        else None
                    )
                    view = {
                        "side": display_side,
                        "selected_fighter_id": (
                            display_fighter_id
                            if display_side == "fighter"
                            else display_opponent_id
                        ),
                        "selected_fighter_name": selected_name,
                        "method": method,
                        "selection": f"{selected_name} by {method.replace('_', '/').upper()}",
                        "moneyline": moneyline,
                        "break_even_probability": implied,
                        "book_no_vig_probability": no_vig,
                        "candidate_model_probability": model_probability,
                        "candidate_model_estimated_return": estimated_return,
                    }
                    selections.append(view)
                    if estimated_return is not None and estimated_return > 0.0:
                        comparison_rows.append(
                            {
                                "matchup_id": matchup_id,
                                "fighter_name": display_fighter_name,
                                "opponent_name": display_opponent_name,
                                "bout_order": bout_order,
                                "book": quote.book,
                                "horizon": quote.horizon,
                                **view,
                            }
                        )
            book_views.append(
                {
                    "book": quote.book,
                    "source_book_key": quote.source_book_key,
                    "horizon": quote.horizon,
                    "observed_at_utc": quote.observed_at_utc,
                    "selection_count": quote.selection_count,
                    "is_complete_six_way": quote.is_complete_six_way,
                    "six_way_overround": quote.six_way_overround,
                    "selections": selections,
                }
            )
        method_markets.append(
            {
                "matchup_id": matchup_id,
                "fighter_id": display_fighter_id,
                "opponent_id": display_opponent_id,
                "fighter_name": display_fighter_name,
                "opponent_name": display_opponent_name,
                "bout_order": bout_order,
                "book_count": len(book_views),
                "book_quotes": book_views,
            }
        )

    def order_value(value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 1_000_000

    method_markets.sort(
        key=lambda row: (
            order_value(row.get("bout_order")),
            str(row.get("fighter_name", "")).casefold(),
        )
    )
    comparison_rows.sort(
        key=lambda row: (
            order_value(row.get("bout_order")),
            -float(row["candidate_model_estimated_return"]),
            str(row.get("book", "")).casefold(),
        )
    )
    observed_times = [record.observed_at_utc for record in latest.values()]
    body: dict[str, object] = {
        "schema_version": 1,
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "source": SOURCE,
        "market": "fighter_method_of_victory",
        "contract_version": METHOD_MARKET_CONTRACT,
        "event_id": event_id,
        "event_date": event_date,
        "event_start_utc": event_start_utc,
        "latest_observed_at_utc": max(observed_times) if observed_times else None,
        "price_status": "available" if method_markets else "no_method_prices",
        "model_probability_status": (
            "available_in_candidate_outcome_forecast"
            if outcome_forecasts is not None
            else "current_outcome_forecast_unavailable"
        ),
        "expected_value_status": (
            "candidate_comparison_only" if outcome_forecasts is not None else "unavailable"
        ),
        "method_market_count": len(method_markets),
        "book_market_count": sum(row["book_count"] for row in method_markets),
        "positive_candidate_comparison_count": len(comparison_rows),
        "positive_candidate_comparisons": comparison_rows[:100],
        "method_markets": method_markets,
        "settlement_status": "not_enabled_until_book_contracts_are_verified",
    }
    body["publication_sha256"] = canonical_hash(body)
    rendered = json.dumps(body, sort_keys=True, allow_nan=False, separators=(",", ":"))
    if len(rendered.encode("utf-8")) > PUBLICATION_SIZE_LIMIT:
        raise CaptureError("current method publication exceeded its 512 KiB size cap")
    return body


def _write_current_method_publication(publication: Mapping[str, object]) -> None:
    atomic_write_text(
        CURRENT_METHOD_PATH,
        json.dumps(publication, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def capture_method_snapshot() -> dict[str, object]:
    if os.environ.get(SOURCE_POLICY_ENV, "").strip().casefold() not in {
        "1",
        "true",
        "yes",
    }:
        raise CaptureError(
            f"set {SOURCE_POLICY_ENV}=1 after acknowledging that public paths are "
            "allowed by robots.txt but bulk/scheduled reuse is not explicitly covered "
            "by the source's short terms"
        )
    started = datetime.now(timezone.utc)
    event_day, event_id, published, event_start, lead_seconds = _load_publication(started)
    store = MethodMarketStore(METHOD_CSV_PATH, METHOD_JSONL_PATH)
    existing = store.read()
    timed_horizon = _timed_horizon(lead_seconds)
    event_existing = [item for item in existing if item.event_id == event_id]
    if not _capture_is_due(event_existing, timed_horizon):
        raise MethodCaptureSkipped(
            "opening method prices and the currently due horizon are already stored"
        )

    robots = _fetch(f"{BESTFIGHTODDS_ROOT}/robots.txt")
    if not _robots_allows_public_paths(robots.text):
        raise CaptureError("BestFightOdds robots policy no longer allows this collector")
    response = _fetch(f"{BESTFIGHTODDS_ROOT}/")
    observed = datetime.now(timezone.utc)
    if observed >= _as_utc(event_start, "event_start_utc"):
        raise CaptureError("the UFC card commenced during method-price retrieval")
    payload_sha = sha256(response.content).hexdigest()
    selections = parse_bestfightodds_method_props(response.text)
    snapshots, counters = _build_snapshots(
        selections=selections,
        published=published,
        event_day=event_day,
        event_id=event_id,
        event_start_utc=event_start,
        observed=observed,
        payload_sha=payload_sha,
        existing=existing,
        timed_horizon=timed_horizon,
    )
    result = store.append(snapshots)
    final = store.read()
    outcome_publication = _outcome_forecasts(event_id)
    forecast_store = MethodForecastStore(
        METHOD_FORECAST_CSV_PATH, METHOD_FORECAST_JSONL_PATH
    )
    existing_forecasts = forecast_store.read()
    forecast_rows, unmatched_forecasts = _build_method_forecast_captures(
        snapshots,
        outcome_forecasts=outcome_publication,
        existing=existing_forecasts,
    )
    if forecast_rows:
        forecast_result = forecast_store.append(forecast_rows)
        final_forecasts = forecast_store.read()
        forecast_added = len(forecast_result.added_ids)
        forecast_duplicates = len(forecast_result.duplicate_ids)
    else:
        final_forecasts = existing_forecasts
        forecast_added = 0
        forecast_duplicates = 0
    method_publication = _build_current_method_publication(
        final,
        event_id=event_id,
        event_date=event_day,
        event_start_utc=event_start,
        outcome_forecasts=outcome_publication,
    )
    _write_current_method_publication(method_publication)
    report: dict[str, object] = {
        "schema_version": 1,
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "source": SOURCE,
        "event_id": event_id,
        "event_date": event_day,
        "event_start_utc": event_start,
        "observed_at_utc": observed.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "lead_time_seconds": float("%.6f" % lead_seconds),
        "timed_horizon": timed_horizon,
        "capture_id": _capture_id(observed, payload_sha),
        "source_payload_sha256": payload_sha,
        **counters,
        "records_built": len(snapshots),
        "records_added": len(result.added_ids),
        "records_duplicate": len(result.duplicate_ids),
        "records_total": len(final),
        "dataset_sha256": MethodMarketStore.dataset_sha256(final),
        "method_forecasts_built": len(forecast_rows),
        "method_forecasts_added": forecast_added,
        "method_forecasts_duplicate": forecast_duplicates,
        "method_forecasts_unmatched": unmatched_forecasts,
        "method_forecasts_total": len(final_forecasts),
        "method_forecast_dataset_sha256": MethodForecastStore.dataset_sha256(
            final_forecasts
        ),
        "current_publication_sha256": method_publication["publication_sha256"],
        "stored_horizons": sorted(
            {item.horizon for item in final if item.event_id == event_id}
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if len(rendered.encode("utf-8")) > REPORT_SIZE_LIMIT:
        raise CaptureError("method capture report exceeded its size limit")
    atomic_write_text(REPORT_PATH, rendered)
    return report


def validate_generated_capture() -> dict[str, object]:
    if not REPORT_PATH.is_file():
        raise CaptureError("method capture report is missing")
    report = _json_object(REPORT_PATH.read_bytes(), REPORT_PATH)
    records = MethodMarketStore(METHOD_CSV_PATH, METHOD_JSONL_PATH).read()
    if int(report.get("records_total", -1)) != len(records):
        raise CaptureError("method report record count differs from the ledger")
    if report.get("dataset_sha256") != MethodMarketStore.dataset_sha256(records):
        raise CaptureError("method report hash differs from the ledger")
    if "method_forecasts_total" in report:
        forecasts = MethodForecastStore(
            METHOD_FORECAST_CSV_PATH, METHOD_FORECAST_JSONL_PATH
        ).read()
        if int(report.get("method_forecasts_total", -1)) != len(forecasts):
            raise CaptureError("method report forecast count differs from the ledger")
        if report.get(
            "method_forecast_dataset_sha256"
        ) != MethodForecastStore.dataset_sha256(forecasts):
            raise CaptureError("method report forecast hash differs from the ledger")
    if report.get("paper_only") is not True or report.get("execution_enabled") is not False:
        raise CaptureError("method capture must remain paper-only with execution disabled")
    if not CURRENT_METHOD_PATH.is_file():
        raise CaptureError("current method publication is missing")
    publication = _json_object(
        CURRENT_METHOD_PATH.read_bytes(), CURRENT_METHOD_PATH
    )
    supplied_hash = publication.get("publication_sha256")
    unhashed = dict(publication)
    unhashed.pop("publication_sha256", None)
    if supplied_hash != canonical_hash(unhashed):
        raise CaptureError("current method publication hash is invalid")
    rebuilt = _build_current_method_publication(
        records,
        event_id=_text(report.get("event_id")),
        event_date=_text(report.get("event_date")),
        event_start_utc=_text(report.get("event_start_utc")),
        outcome_forecasts=_outcome_forecasts(_text(report.get("event_id"))),
    )
    if publication != rebuilt:
        raise CaptureError("current method publication cannot be reproduced")
    if report.get("current_publication_sha256") != supplied_hash:
        raise CaptureError("method report and current publication hashes disagree")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_generated_capture() if args.validate_only else capture_method_snapshot()
        print(
            f"Method prices: {report['records_added'] if not args.validate_only else 0} added; "
            f"{report['records_total']} total book/fight/horizon boards."
        )
        return 0
    except MethodCaptureSkipped as skipped:
        print(f"Method price capture skipped: {skipped}")
        return 0
    except (CaptureError, StoreIntegrityError, ValueError, OSError) as error:
        print(f"method price capture: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
