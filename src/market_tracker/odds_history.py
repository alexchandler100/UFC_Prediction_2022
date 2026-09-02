"""Compact, deterministic moneyline history for the public website."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable, Mapping

from ._common import MarketDataError, SCHEMA_VERSION, canonical_hash, stable_id
from .quotes import QuoteSnapshot, consensus_as_of


ODDS_HISTORY_CONTRACT = "moneyline-odds-history-v1"
CONSENSUS_MIN_BOOKS = 2


def _event_date(card: Mapping[str, object]) -> str:
    value = str(card.get("date") or "").strip()
    for pattern in ("%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            pass
    raise MarketDataError("card date is invalid")


def build_odds_history(
    snapshots: Iterable[QuoteSnapshot],
    card: Mapping[str, object],
) -> dict[str, object]:
    """Publish only the current card's history, leaving the full ledger private."""

    event_id = stable_id(card.get("event_id"), "event_id")
    event_date = _event_date(card)
    selected = sorted(
        (item for item in snapshots if item.event_id == event_id),
        key=lambda item: (
            item.matchup_id,
            item.observed_at_utc,
            item.book.casefold(),
            item.quote_id,
        ),
    )
    by_matchup: dict[str, list[QuoteSnapshot]] = defaultdict(list)
    for item in selected:
        if item.event_date != event_date:
            raise MarketDataError("current-card quote date does not match card date")
        by_matchup[item.matchup_id].append(item)

    matchups: list[dict[str, object]] = []
    capture_ids: set[str] = set()
    for matchup_id, quotes in sorted(by_matchup.items()):
        identity = {
            (item.fighter_id, item.opponent_id, item.event_date) for item in quotes
        }
        if len(identity) != 1:
            raise MarketDataError("odds-history matchup identity is inconsistent")
        fighter_id, opponent_id, _ = next(iter(identity))
        latest = max(quotes, key=lambda item: (item.observed_at_utc, item.quote_id))
        fighter_name = latest.fighter_name or fighter_id
        opponent_name = latest.opponent_name or opponent_id
        by_capture: dict[str, list[QuoteSnapshot]] = defaultdict(list)
        by_book: dict[str, list[QuoteSnapshot]] = defaultdict(list)
        book_labels: dict[str, str] = {}
        for quote in quotes:
            capture_ids.add(quote.capture_id)
            by_capture[quote.capture_id].append(quote)
            book_key = quote.book.casefold()
            by_book[book_key].append(quote)
            book_labels[book_key] = quote.book

        series: list[dict[str, object]] = []
        consensus_points: list[dict[str, object]] = []
        for capture_id, capture_quotes in sorted(
            by_capture.items(),
            key=lambda item: (item[1][0].observed_at_utc, item[0]),
        ):
            try:
                consensus = consensus_as_of(
                    capture_quotes,
                    capture_id=capture_id,
                    matchup_id=matchup_id,
                    as_of_utc=capture_quotes[0].observed_at_utc,
                    min_books=CONSENSUS_MIN_BOOKS,
                )
            except MarketDataError:
                continue
            consensus_points.append(
                {
                    "observed_at_utc": consensus.as_of_utc,
                    "fighter_probability": consensus.no_vig_fighter_probability,
                    "opponent_probability": consensus.no_vig_opponent_probability,
                    "fighter_moneyline": None,
                    "opponent_moneyline": None,
                    "book_count": consensus.book_count,
                }
            )
        if consensus_points:
            series.append(
                {
                    "key": "consensus",
                    "label": "Consensus",
                    "kind": "consensus",
                    "points": consensus_points,
                }
            )

        for book_key, book_quotes in sorted(
            by_book.items(), key=lambda item: book_labels[item[0]].casefold()
        ):
            points = [
                {
                    "observed_at_utc": quote.observed_at_utc,
                    "fighter_probability": quote.no_vig_fighter_probability,
                    "opponent_probability": 1.0
                    - quote.no_vig_fighter_probability,
                    "fighter_moneyline": quote.fighter_moneyline,
                    "opponent_moneyline": quote.opponent_moneyline,
                    "book_count": 1,
                }
                for quote in sorted(
                    book_quotes,
                    key=lambda item: (item.observed_at_utc, item.quote_id),
                )
            ]
            series.append(
                {
                    "key": f"book:{book_key}",
                    "label": book_labels[book_key],
                    "kind": "book",
                    "points": points,
                }
            )

        matchups.append(
            {
                "matchup_id": matchup_id,
                "fighter_id": fighter_id,
                "opponent_id": opponent_id,
                "fighter_name": fighter_name,
                "opponent_name": opponent_name,
                "capture_count": len(by_capture),
                "series": series,
            }
        )

    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "contract": ODDS_HISTORY_CONTRACT,
        "event_id": event_id,
        "event_date": event_date,
        "event_title": str(card.get("title") or "").strip(),
        "matchup_count": len(matchups),
        "capture_count": len(capture_ids),
        "quote_count": len(selected),
        "latest_observed_at_utc": max(
            (item.observed_at_utc for item in selected), default=None
        ),
        "matchups": matchups,
    }
    return {**body, "publication_sha256": canonical_hash(body)}


def validate_odds_history(
    publication: Mapping[str, object],
    snapshots: Iterable[QuoteSnapshot],
    card: Mapping[str, object],
) -> None:
    rebuilt = build_odds_history(snapshots, card)
    if dict(publication) != rebuilt:
        raise MarketDataError(
            "published odds history does not reproduce from the quote ledger"
        )
