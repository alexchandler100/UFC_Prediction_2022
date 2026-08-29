"""Strict parsing helpers for BestFightOdds method-of-victory markets.

BestFightOdds renders a compact label-only table for small screens and a second
desktop table containing the actual price cells.  This parser deliberately
walks every table and then de-duplicates by the site's stable fight/prop keys.
Only the six unambiguous fighter-by-method selections are admitted here;
round-specific and decision-subtype markets remain available for later schema
versions without being accidentally mixed into the primary method market.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Iterable

from bs4 import BeautifulSoup


METHOD_MARKET = "fighter_method_of_victory"
METHODS = ("ko_tko", "submission", "decision")
NON_SPORTSBOOK_BOOK_IDS = frozenset({28, 29})

_METHOD_SUFFIXES = (
    (re.compile(r"\s+wins\s+by\s+(?:tko/ko|ko/tko(?:/dq)?)\s*$", re.I), "ko_tko"),
    (re.compile(r"\s+wins\s+by\s+submission\s*$", re.I), "submission"),
    (re.compile(r"\s+wins\s+by\s+decision\s*$", re.I), "decision"),
)


class PropParseError(ValueError):
    """Raised when one stable source key is presented with conflicting data."""


def parse_american_moneyline(value: object) -> int | None:
    """Parse a displayed American price while discarding arrows and blanks."""

    text = "".join(str(value or "").split())
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("▲", "")
        .replace("▼", "")
    )
    if not text or text.casefold() in {"n/a", "na", "-"}:
        return None
    if text.casefold() in {"ev", "even", "pk", "pick"}:
        return 100
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or not 100 <= abs(numeric) <= 100_000
    ):
        return None
    return int(numeric)


def method_from_label(label: object) -> str | None:
    """Return the primary method represented by a complete prop-row label."""

    text = " ".join(str(label or "").split())
    for pattern, method in _METHOD_SUFFIXES:
        if pattern.search(text):
            return method
    return None


@dataclass(frozen=True)
class PropBookPrice:
    book_id: int
    book_name: str
    american_odds: int


@dataclass(frozen=True)
class MethodPropSelection:
    source_matchup_id: int
    source_fighter_side: int
    source_prop_type_id: int
    source_outcome_number: int
    fighter_1_name: str
    fighter_2_name: str
    market: str
    method: str
    raw_label: str
    mean_history_available: bool
    book_prices: tuple[PropBookPrice, ...]

    @property
    def source_selection_id(self) -> str:
        return (
            f"{self.source_matchup_id}:{self.source_fighter_side}:"
            f"{self.source_prop_type_id}:{self.source_outcome_number}"
        )


def _book_headers(table: object) -> dict[int, str]:
    output: dict[int, str] = {}
    for header in table.find_all("th", attrs={"data-b": True}):
        raw_id = str(header.get("data-b", ""))
        if not raw_id.isdigit():
            continue
        label = header.find("a") or header.find("span")
        name = (
            label.get_text(" ", strip=True)
            if label is not None
            else header.get_text(" ", strip=True)
        )
        output[int(raw_id)] = " ".join(name.split()) or f"Book {raw_id}"
    return output


def _json_integer_list(cell: object, expected: int) -> tuple[int, ...] | None:
    try:
        value = json.loads(cell.get("data-li", ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, list)
        or len(value) != expected
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return None
    return tuple(value)


def _merge_name(
    target: dict[int, dict[int, str]], matchup_id: int, side: int, name: str
) -> None:
    if not name:
        return
    names = target.setdefault(matchup_id, {})
    previous = names.get(side)
    if previous is not None and previous.casefold() != name.casefold():
        raise PropParseError(
            f"source matchup {matchup_id} side {side} has conflicting fighter names"
        )
    names[side] = name


def parse_bestfightodds_method_props(html: str) -> tuple[MethodPropSelection, ...]:
    """Parse complete displayed prices and stable history keys from a page."""

    soup = BeautifulSoup(html, "html.parser")
    matchup_names: dict[int, dict[int, str]] = {}
    # Values are accumulated across duplicated responsive tables.
    selections: dict[tuple[int, int, int, int, str], dict[str, object]] = {}

    for table in soup.find_all("table"):
        headers = _book_headers(table)
        if not headers:
            # Label-only mobile tables contain no useful source price keys.
            continue
        for row in table.find_all("tr"):
            header = row.find("th", attrs={"scope": "row"}, recursive=False)
            if header is None:
                continue
            fighter_link = next(
                (
                    anchor
                    for anchor in header.find_all("a", href=True)
                    if "/fighters/" in str(anchor.get("href"))
                ),
                None,
            )
            if fighter_link is not None:
                fighter_name = " ".join(
                    fighter_link.get_text(" ", strip=True).split()
                )
                for cell in row.find_all("td", attrs={"data-li": True}):
                    classes = set(cell.get("class", []))
                    identifiers = None
                    if "but-sg" in classes:
                        parsed = _json_integer_list(cell, 3)
                        if parsed is not None:
                            _, side, matchup_id = parsed
                            identifiers = side, matchup_id
                    elif "but-si" in classes:
                        parsed = _json_integer_list(cell, 2)
                        if parsed is not None:
                            side, matchup_id = parsed
                            identifiers = side, matchup_id
                    if identifiers is not None and identifiers[0] in {1, 2}:
                        _merge_name(
                            matchup_names,
                            identifiers[1],
                            identifiers[0],
                            fighter_name,
                        )
                continue

            raw_label = " ".join(header.get_text(" ", strip=True).split())
            method = method_from_label(raw_label)
            if method is None:
                continue
            for cell in row.find_all("td", attrs={"data-li": True}):
                classes = set(cell.get("class", []))
                book_id: int | None
                if "but-sgp" in classes:
                    parsed = _json_integer_list(cell, 5)
                    if parsed is None:
                        continue
                    book_id, side, matchup_id, prop_type, outcome_number = parsed
                elif "but-sip" in classes:
                    parsed = _json_integer_list(cell, 4)
                    if parsed is None:
                        continue
                    side, matchup_id, prop_type, outcome_number = parsed
                    book_id = None
                else:
                    continue
                if side not in {1, 2} or outcome_number != 1:
                    # The paired "Any other result" row is not a method pick.
                    continue
                key = (matchup_id, side, prop_type, outcome_number, method)
                item = selections.setdefault(
                    key,
                    {
                        "raw_label": raw_label,
                        "mean": False,
                        "prices": {},
                    },
                )
                prior_label = str(item["raw_label"])
                if prior_label.casefold() != raw_label.casefold():
                    raise PropParseError(
                        f"source method key {key} has conflicting labels"
                    )
                if book_id is None:
                    item["mean"] = True
                    continue
                if book_id in NON_SPORTSBOOK_BOOK_IDS:
                    continue
                price = parse_american_moneyline(cell.get_text(" ", strip=True))
                if price is None:
                    continue
                book_name = headers.get(book_id, f"Book {book_id}")
                prices = item["prices"]
                assert isinstance(prices, dict)
                candidate = PropBookPrice(book_id, book_name, price)
                previous = prices.get(book_id)
                if previous is not None and previous != candidate:
                    raise PropParseError(
                        f"source method key {key} has conflicting {book_name} prices"
                    )
                prices[book_id] = candidate

    output: list[MethodPropSelection] = []
    for key, item in sorted(selections.items()):
        matchup_id, side, prop_type, outcome_number, method = key
        names = matchup_names.get(matchup_id, {})
        if set(names) != {1, 2}:
            # A prop without the linked two-fighter moneyline cannot be joined
            # safely to UFCStats and is deliberately excluded.
            continue
        prices = item["prices"]
        assert isinstance(prices, dict)
        output.append(
            MethodPropSelection(
                source_matchup_id=matchup_id,
                source_fighter_side=side,
                source_prop_type_id=prop_type,
                source_outcome_number=outcome_number,
                fighter_1_name=names[1],
                fighter_2_name=names[2],
                market=METHOD_MARKET,
                method=method,
                raw_label=str(item["raw_label"]),
                mean_history_available=bool(item["mean"]),
                book_prices=tuple(prices[book_id] for book_id in sorted(prices)),
            )
        )
    return tuple(output)


def available_method_markets(
    selections: Iterable[MethodPropSelection],
) -> dict[tuple[int, int], dict[tuple[int, str], int]]:
    """Return every nonempty displayed method board by matchup and book."""

    grouped: dict[tuple[int, int], dict[tuple[int, str], int]] = {}
    for selection in selections:
        for price in selection.book_prices:
            values = grouped.setdefault(
                (selection.source_matchup_id, price.book_id), {}
            )
            key = selection.source_fighter_side, selection.method
            previous = values.get(key)
            if previous is not None and previous != price.american_odds:
                raise PropParseError(
                    f"method market has conflicting displayed price for {key}"
                )
            values[key] = price.american_odds
    return grouped


def complete_method_markets(
    selections: Iterable[MethodPropSelection],
) -> dict[tuple[int, int], dict[tuple[int, str], int]]:
    """Return complete six-way displayed method markets by matchup and book."""

    grouped = available_method_markets(selections)
    expected = {(side, method) for side in (1, 2) for method in METHODS}
    return {key: value for key, value in grouped.items() if set(value) == expected}
