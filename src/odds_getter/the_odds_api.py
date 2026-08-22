"""Small, strict client for The Odds API's MMA moneyline and totals endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd
import requests


API_URL = (
    "https://api.the-odds-api.com/v4/sports/"
    "mma_mixed_martial_arts/odds"
)


class OddsApiError(RuntimeError):
    """Raised when The Odds API cannot provide a trustworthy MMA payload."""


@dataclass(frozen=True)
class OddsApiResponse:
    frame: pd.DataFrame
    payload: list[dict[str, object]]
    requests_remaining: int | None
    requests_used: int | None
    request_cost: int | None
    total_rounds_frame: pd.DataFrame | None = None

    def quota_mapping(self) -> dict[str, int | None]:
        return {
            "requests_remaining": self.requests_remaining,
            "requests_used": self.requests_used,
            "request_cost": self.request_cost,
        }


def _text(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _header_integer(headers: Mapping[str, object], name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        # ``requests`` uses a case-insensitive mapping, while test doubles may
        # not. Keep the parser deterministic for either representation.
        raw = next(
            (value for key, value in headers.items() if str(key).casefold() == name),
            None,
        )
    if raw is None or not _text(raw):
        return None
    try:
        parsed = int(_text(raw))
    except ValueError as error:
        raise OddsApiError(f"The Odds API returned an invalid {name} header") from error
    if parsed < 0:
        raise OddsApiError(f"The Odds API returned a negative {name} header")
    return parsed


def _iso_timestamp(value: object, name: str) -> str:
    text = _text(value)
    try:
        parsed = pd.to_datetime(text, errors="raise", utc=True)
    except (TypeError, ValueError) as error:
        raise OddsApiError(f"The Odds API returned an invalid {name}") from error
    if pd.isna(parsed):
        raise OddsApiError(f"The Odds API returned an invalid {name}")
    return parsed.isoformat().replace("+00:00", "Z")


def _american_price(value: object, *, book: str, participant: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise OddsApiError(
            f"The Odds API returned invalid {book!r} odds for {participant!r}"
        ) from error
    if not numeric.is_integer() or not 100 <= abs(numeric) <= 100_000:
        raise OddsApiError(
            f"The Odds API returned invalid {book!r} odds for {participant!r}"
        )
    return int(numeric)


def _total_point(value: object, *, book: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise OddsApiError(
            f"The Odds API returned an invalid totals point for {book!r}"
        ) from error
    if not 0.0 < numeric <= 25.0 or round(numeric, 3) != numeric:
        raise OddsApiError(
            f"The Odds API returned an invalid totals point for {book!r}"
        )
    return numeric


def _normalize_payload(payload: object) -> pd.DataFrame:
    if not isinstance(payload, list):
        raise OddsApiError("The Odds API response must be a JSON list")
    rows: list[dict[str, object]] = []
    seen_event_ids: set[str] = set()
    for event in payload:
        if not isinstance(event, dict):
            raise OddsApiError("The Odds API response contains a non-object event")
        event_id = _text(event.get("id"))
        sport_key = _text(event.get("sport_key"))
        fighter_name = _text(event.get("home_team"))
        opponent_name = _text(event.get("away_team"))
        commence_time = _iso_timestamp(event.get("commence_time"), "commence_time")
        if not event_id or not fighter_name or not opponent_name:
            raise OddsApiError("The Odds API event is missing ID, participants, or start time")
        if sport_key != "mma_mixed_martial_arts":
            raise OddsApiError("The Odds API response contains a non-MMA event")
        if event_id in seen_event_ids:
            raise OddsApiError(f"The Odds API repeated event ID {event_id}")
        seen_event_ids.add(event_id)
        if fighter_name.casefold() == opponent_name.casefold():
            raise OddsApiError("The Odds API event contains the same participant twice")

        row: dict[str, object] = {
            "source event id": event_id,
            "source commence time": commence_time,
            "fighter name": fighter_name,
            "opponent name": opponent_name,
        }
        bookmakers = event.get("bookmakers", [])
        if not isinstance(bookmakers, list):
            raise OddsApiError("The Odds API event bookmakers field is not a list")
        seen_books: set[str] = set()
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                raise OddsApiError("The Odds API contains a non-object bookmaker")
            book_key = _text(bookmaker.get("key"))
            book_title = _text(bookmaker.get("title")) or book_key
            normalized_book = book_title.casefold()
            if not book_key or not book_title:
                raise OddsApiError("The Odds API bookmaker is missing its key or title")
            if normalized_book in seen_books:
                raise OddsApiError(
                    f"The Odds API repeated bookmaker {book_title!r} for one event"
                )
            seen_books.add(normalized_book)
            markets = bookmaker.get("markets", [])
            if not isinstance(markets, list):
                raise OddsApiError("The Odds API bookmaker markets field is not a list")
            head_to_head = [
                market
                for market in markets
                if isinstance(market, dict) and _text(market.get("key")) == "h2h"
            ]
            if len(head_to_head) > 1:
                raise OddsApiError(
                    f"The Odds API repeated h2h market for bookmaker {book_title!r}"
                )
            if not head_to_head:
                continue
            outcomes = head_to_head[0].get("outcomes", [])
            if not isinstance(outcomes, list):
                raise OddsApiError("The Odds API h2h outcomes field is not a list")
            prices: dict[str, object] = {}
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    raise OddsApiError("The Odds API contains a non-object outcome")
                name = _text(outcome.get("name"))
                if name in prices:
                    raise OddsApiError(
                        f"The Odds API repeated outcome {name!r} for {book_title!r}"
                    )
                prices[name] = _american_price(
                    outcome.get("price"), book=book_title, participant=name
                )
            if fighter_name not in prices or opponent_name not in prices:
                # An incomplete book is omitted; other complete books remain
                # independently useful and downstream coverage is reported.
                continue
            market_update = head_to_head[0].get("last_update")
            book_update = market_update or bookmaker.get("last_update")
            if not _text(book_update):
                raise OddsApiError(
                    f"The Odds API omitted last_update for bookmaker {book_title!r}"
                )
            book_update = _iso_timestamp(book_update, "bookmaker last_update")
            row[f"fighter {book_title}"] = prices[fighter_name]
            row[f"opponent {book_title}"] = prices[opponent_name]
            row[f"source {book_title} key"] = book_key
            row[f"source {book_title} last update"] = book_update
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise OddsApiError("The Odds API returned no upcoming MMA events")
    return frame


def _normalize_total_rounds_payload(payload: object) -> pd.DataFrame:
    """Flatten complete full-fight Over/Under round pairs without imputing gaps."""

    if not isinstance(payload, list):
        raise OddsApiError("The Odds API response must be a JSON list")
    rows: list[dict[str, object]] = []
    seen_event_ids: set[str] = set()
    for event in payload:
        if not isinstance(event, dict):
            raise OddsApiError("The Odds API response contains a non-object event")
        event_id = _text(event.get("id"))
        sport_key = _text(event.get("sport_key"))
        fighter_name = _text(event.get("home_team"))
        opponent_name = _text(event.get("away_team"))
        commence_time = _iso_timestamp(event.get("commence_time"), "commence_time")
        if not event_id or not fighter_name or not opponent_name:
            raise OddsApiError("The Odds API event is missing ID, participants, or start time")
        if sport_key != "mma_mixed_martial_arts":
            raise OddsApiError("The Odds API response contains a non-MMA event")
        if event_id in seen_event_ids:
            raise OddsApiError(f"The Odds API repeated event ID {event_id}")
        seen_event_ids.add(event_id)
        if fighter_name.casefold() == opponent_name.casefold():
            raise OddsApiError("The Odds API event contains the same participant twice")
        bookmakers = event.get("bookmakers", [])
        if not isinstance(bookmakers, list):
            raise OddsApiError("The Odds API event bookmakers field is not a list")
        seen_books: set[str] = set()
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                raise OddsApiError("The Odds API contains a non-object bookmaker")
            book_key = _text(bookmaker.get("key"))
            book_title = _text(bookmaker.get("title")) or book_key
            normalized_book = book_key.casefold()
            if not book_key or not book_title:
                raise OddsApiError("The Odds API bookmaker is missing its key or title")
            if normalized_book in seen_books:
                raise OddsApiError(
                    f"The Odds API repeated bookmaker {book_title!r} for one event"
                )
            seen_books.add(normalized_book)
            markets = bookmaker.get("markets", [])
            if not isinstance(markets, list):
                raise OddsApiError("The Odds API bookmaker markets field is not a list")
            totals = [
                market
                for market in markets
                if isinstance(market, dict) and _text(market.get("key")) == "totals"
            ]
            if len(totals) > 1:
                raise OddsApiError(
                    f"The Odds API repeated totals market for bookmaker {book_title!r}"
                )
            if not totals:
                continue
            outcomes = totals[0].get("outcomes", [])
            if not isinstance(outcomes, list):
                raise OddsApiError("The Odds API totals outcomes field is not a list")
            by_point: dict[float, dict[str, int]] = {}
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    raise OddsApiError("The Odds API contains a non-object outcome")
                side = _text(outcome.get("name")).casefold()
                if side not in {"over", "under"}:
                    raise OddsApiError(
                        f"The Odds API returned unsupported totals outcome {side!r}"
                    )
                point = _total_point(outcome.get("point"), book=book_title)
                point_prices = by_point.setdefault(point, {})
                if side in point_prices:
                    raise OddsApiError(
                        f"The Odds API repeated {side} {point:g} for {book_title!r}"
                    )
                point_prices[side] = _american_price(
                    outcome.get("price"),
                    book=book_title,
                    participant=f"{side} {point:g}",
                )
            market_update = totals[0].get("last_update") or bookmaker.get("last_update")
            if by_point and not _text(market_update):
                raise OddsApiError(
                    f"The Odds API omitted last_update for bookmaker {book_title!r}"
                )
            update = (
                _iso_timestamp(market_update, "bookmaker last_update")
                if by_point
                else ""
            )
            for point, prices in sorted(by_point.items()):
                if set(prices) != {"over", "under"}:
                    continue
                rows.append(
                    {
                        "source event id": event_id,
                        "source commence time": commence_time,
                        "fighter name": fighter_name,
                        "opponent name": opponent_name,
                        "book": book_title,
                        "source book key": book_key,
                        "source last update": update,
                        "market": "total_rounds",
                        "period": "full_fight",
                        "line": point,
                        "over moneyline": prices["over"],
                        "under moneyline": prices["under"],
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "source event id",
            "source commence time",
            "fighter name",
            "opponent name",
            "book",
            "source book key",
            "source last update",
            "market",
            "period",
            "line",
            "over moneyline",
            "under moneyline",
        ],
    )


class TheOddsApiClient:
    """Fetch current UFC/MMA head-to-head prices without browser automation."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def fetch(
        self,
        api_key: object,
        *,
        regions: str = "us,us2",
        timeout_seconds: float = 30.0,
        include_total_rounds: bool = False,
    ) -> OddsApiResponse:
        key = _text(api_key)
        if not key:
            raise OddsApiError(
                "THE_ODDS_API_KEY is missing; add the free API key as a GitHub "
                "Actions repository secret"
            )
        normalized_regions = ",".join(
            part.strip().casefold() for part in str(regions).split(",") if part.strip()
        )
        if not normalized_regions:
            raise OddsApiError("ODDS_API_REGIONS must name at least one region")
        try:
            response = self.session.get(
                API_URL,
                params={
                    "apiKey": key,
                    "regions": normalized_regions,
                    "markets": "h2h,totals" if include_total_rounds else "h2h",
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
                timeout=timeout_seconds,
            )
        except requests.RequestException as error:
            # Never stringify the exception: requests may include the query URL
            # and therefore the secret API key in its message.
            raise OddsApiError(
                f"The Odds API request failed ({type(error).__name__})"
            ) from None
        if response.status_code >= 400:
            if response.status_code in {401, 403}:
                reason = "authentication was rejected; verify THE_ODDS_API_KEY"
            elif response.status_code == 429:
                reason = "monthly request quota or rate limit was exceeded"
            else:
                reason = f"HTTP {response.status_code}"
            raise OddsApiError(f"The Odds API request failed: {reason}")
        try:
            payload = response.json()
        except (ValueError, requests.JSONDecodeError) as error:
            raise OddsApiError("The Odds API returned invalid JSON") from error
        frame = _normalize_payload(payload)
        return OddsApiResponse(
            frame=frame,
            payload=payload,
            requests_remaining=_header_integer(
                response.headers, "x-requests-remaining"
            ),
            requests_used=_header_integer(response.headers, "x-requests-used"),
            request_cost=_header_integer(response.headers, "x-requests-last"),
            total_rounds_frame=(
                _normalize_total_rounds_payload(payload)
                if include_total_rounds
                else None
            ),
        )
