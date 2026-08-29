"""Bounded, read-only audit of free historical MMA odds sources.

This command answers a narrow question before any large backfill is attempted:
does a public archive expose bookmaker-specific prices with absolute timestamps
that precede the event?  It stores only a compact audit report, never raw HTML
or a reusable copy of another site's odds archive.

BestFightOdds is sampled because its public event page links each displayed
price to the same chart endpoint used by a browser.  FightOdds.io is limited to
a policy/access check unless it publishes a machine-readable robots policy.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import requests


ALGORITHM_VERSION = 1
USER_AGENT = (
    "UFC-Prediction-Research-Audit/1.0 "
    "(+local noncommercial historical-odds feasibility check)"
)
BESTFIGHTODDS_ROOT = "https://www.bestfightodds.com"
FIGHTODDS_ROOT = "https://fightodds.io"
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_HISTORY_EVENTS = 10
DEFAULT_HISTORY_BOOKS = 3
DEFAULT_MEAN_HISTORY_EVENTS = 15
DEFAULT_LEGACY_BOOK_MAX_ID = 29
DEFAULT_MINIMUM_YEAR = 2012
DEFAULT_MAX_RUNTIME_MINUTES = 20.0
DEFAULT_REQUEST_DELAY_SECONDS = 0.35
DEFAULT_OUTPUT = Path(
    "src/content/data/model_research/historical_odds_feasibility.json"
)
DEFAULT_MARKDOWN = Path(
    "src/content/data/model_research/HISTORICAL_ODDS_FEASIBILITY.md"
)

_MONEYLINE = re.compile(r"^[+\-−–]?\d{3,6}$")
_EVENT_TOKEN = re.compile(r"(?:^|-)ufc(?:-|$)", re.IGNORECASE)


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _parse_iso_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _rot47(value: str) -> str:
    return "".join(
        chr(33 + ((ord(character) - 33 + 47) % 94))
        if 33 <= ord(character) <= 126
        else character
        for character in value
    )


def decode_bestfightodds_chart(payload: str) -> list[dict[str, Any]]:
    """Decode the chart response using the reversible transform in site JS."""

    compact = "".join(str(payload).split())
    if not compact:
        raise ValueError("empty chart payload")
    try:
        if compact.startswith("["):
            value = json.loads(compact)
        else:
            decoded = base64.b64decode(compact, validate=True).decode("utf-8")
            value = json.loads(_rot47(decoded))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("chart payload is not valid encoded JSON") from error
    if not isinstance(value, list):
        raise ValueError("chart payload must contain a list of series")
    for series in value:
        if not isinstance(series, dict) or not isinstance(series.get("data"), list):
            raise ValueError("chart series is missing a data list")
        for point in series["data"]:
            if not isinstance(point, dict):
                raise ValueError("chart point is not an object")
            timestamp = point.get("x")
            price = point.get("y")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(float(timestamp))
                or isinstance(price, bool)
                or not isinstance(price, (int, float))
                or not math.isfinite(float(price))
                or float(price) <= 1.0
            ):
                raise ValueError("chart point has an invalid timestamp or price")
    return value


@dataclass(frozen=True)
class SitemapEvent:
    url: str
    event_date: date

    def to_mapping(self) -> dict[str, str]:
        return {"url": self.url, "event_date": self.event_date.isoformat()}


def parse_bestfightodds_sitemap(xml_text: str) -> tuple[SitemapEvent, ...]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ValueError("event sitemap is not valid XML") from error
    output: list[SitemapEvent] = []
    for node in root:
        location = node.findtext("{*}loc", "").strip()
        modified = _parse_iso_date(node.findtext("{*}lastmod", ""))
        parsed = urlparse(location)
        slug = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        if (
            parsed.scheme != "https"
            or parsed.netloc.casefold() != "www.bestfightodds.com"
            or "/events/" not in parsed.path
            or not _EVENT_TOKEN.search(slug)
            or modified is None
        ):
            continue
        output.append(SitemapEvent(url=location, event_date=modified))
    return tuple(sorted(output, key=lambda item: (item.event_date, item.url)))


def _evenly_spaced(items: Sequence[SitemapEvent], count: int) -> list[SitemapEvent]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indexes = {
        round(position * (len(items) - 1) / (count - 1)) for position in range(count)
    }
    if len(indexes) != count:
        for index in range(len(items)):
            indexes.add(index)
            if len(indexes) == count:
                break
    return [items[index] for index in sorted(indexes)[:count]]


def select_stratified_events(
    events: Iterable[SitemapEvent],
    *,
    sample_size: int,
    minimum_year: int,
    maximum_date: date,
) -> tuple[SitemapEvent, ...]:
    """Select a deterministic sample spread across every available year."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    eligible = [
        item
        for item in events
        if minimum_year <= item.event_date.year and item.event_date <= maximum_date
    ]
    if not eligible:
        raise ValueError("no eligible UFC events were found in the sitemap")
    if len(eligible) <= sample_size:
        return tuple(eligible)

    by_year: dict[int, list[SitemapEvent]] = {}
    for item in eligible:
        by_year.setdefault(item.event_date.year, []).append(item)
    years = sorted(by_year)
    quotas = dict.fromkeys(years, sample_size // len(years))
    for year in years[: sample_size % len(years)]:
        quotas[year] += 1

    selected: list[SitemapEvent] = []
    unused: list[SitemapEvent] = []
    for year in years:
        rows = sorted(by_year[year], key=lambda item: (item.event_date, item.url))
        chosen = _evenly_spaced(rows, min(quotas[year], len(rows)))
        selected.extend(chosen)
        chosen_urls = {item.url for item in chosen}
        unused.extend(item for item in rows if item.url not in chosen_urls)
    if len(selected) < sample_size:
        selected.extend(_evenly_spaced(unused, sample_size - len(selected)))
    return tuple(
        sorted(selected[:sample_size], key=lambda item: (item.event_date, item.url))
    )


def _sports_event_json(soup: BeautifulSoup) -> Mapping[str, Any]:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            value = json.loads(script.get_text())
        except (TypeError, json.JSONDecodeError):
            continue
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("@type") == "SportsEvent":
                return candidate
    return {}


def _clean_moneyline(value: str) -> str | None:
    text = "".join(str(value).split()).replace("−", "-").replace("–", "-")
    return text if _MONEYLINE.fullmatch(text) else None


def parse_bestfightodds_event_page(
    html: str,
    *,
    url: str,
    sitemap_date: date,
) -> dict[str, Any]:
    """Extract the minimum structure needed to judge backfill feasibility."""

    soup = BeautifulSoup(html, "html.parser")
    structured = _sports_event_json(soup)
    structured_date = _parse_iso_date(structured.get("startDate"))
    organizer = structured.get("organizer")
    organizer_name = (
        str(organizer.get("name", "")).strip()
        if isinstance(organizer, Mapping)
        else ""
    )
    title = str(structured.get("name", "")).strip()
    if not title:
        heading = soup.find("h1")
        title = heading.get_text(" ", strip=True) if heading else ""

    odds_table = None
    book_headers: dict[int, str] = {}
    for table in soup.find_all("table"):
        headers = table.find_all("th", attrs={"data-b": True})
        if len(headers) > len(book_headers):
            odds_table = table
            book_headers = {
                int(header["data-b"]): header.get_text(" ", strip=True).split("\n")[0]
                for header in headers
                if str(header.get("data-b", "")).isdigit()
            }

    matchups: dict[str, dict[str, Any]] = {}
    if odds_table is not None:
        for row in odds_table.find_all("tr", recursive=True):
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
            fighter_name = (
                fighter_link.get_text(" ", strip=True) if fighter_link is not None else ""
            )
            for cell in row.find_all("td", attrs={"data-li": True}, recursive=False):
                classes = set(cell.get("class", []))
                try:
                    identifiers = json.loads(cell["data-li"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    "but-sg" in classes
                    and isinstance(identifiers, list)
                    and len(identifiers) == 3
                ):
                    book_id, side, matchup_id = identifiers
                elif (
                    "but-si" in classes
                    and isinstance(identifiers, list)
                    and len(identifiers) == 2
                ):
                    side, matchup_id = identifiers
                    book_id = None
                else:
                    continue
                values = (side, matchup_id) if book_id is None else (book_id, side, matchup_id)
                if not all(isinstance(value, int) for value in values):
                    continue
                matchup = matchups.setdefault(
                    str(matchup_id), {"fighters": {}, "books": {}}
                )
                if fighter_name:
                    matchup["fighters"][str(side)] = fighter_name
                if book_id is None:
                    continue
                displayed = _clean_moneyline(cell.get_text(" ", strip=True).rstrip("▲▼"))
                book = matchup["books"].setdefault(
                    str(book_id),
                    {"book_id": book_id, "book": book_headers.get(book_id, "")},
                )
                book[f"side_{side}_present"] = displayed is not None

    normalized_matchups: list[dict[str, Any]] = []
    for matchup_id, matchup in sorted(matchups.items(), key=lambda item: int(item[0])):
        paired_books = [
            value
            for value in matchup["books"].values()
            if value.get("side_1_present") and value.get("side_2_present")
        ]
        normalized_matchups.append(
            {
                "matchup_id": matchup_id,
                "fighter_1": matchup["fighters"].get("1", ""),
                "fighter_2": matchup["fighters"].get("2", ""),
                "paired_books": sorted(
                    paired_books, key=lambda item: (item["book"], item["book_id"])
                ),
            }
        )

    page_text = soup.get_text(" ", strip=True).casefold()
    return {
        "url": url,
        "event_date": (structured_date or sitemap_date).isoformat(),
        "sitemap_date": sitemap_date.isoformat(),
        "title": title,
        "organizer": organizer_name,
        "is_ufc": organizer_name.casefold() == "ufc" or "ufc" in title.casefold(),
        "html_sha256": _sha256_text(html),
        "html_bytes": len(html.encode("utf-8")),
        "matchup_count": len(normalized_matchups),
        "book_count": len(book_headers),
        "books": [book_headers[key] for key in sorted(book_headers)],
        "matchups": normalized_matchups,
        "has_line_movement_control": "line movement" in page_text,
        "has_absolute_quote_time_in_html": bool(
            soup.find("time", attrs={"datetime": True})
        ),
    }


def summarize_chart(
    series: Sequence[Mapping[str, Any]], *, event_date: date
) -> dict[str, Any]:
    cutoff_ms = int(
        datetime.combine(event_date, datetime.min.time(), tzinfo=timezone.utc).timestamp()
        * 1000
    )
    points: list[tuple[int, float]] = []
    names: list[str] = []
    for item in series:
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
        for point in item.get("data", []):
            points.append((int(point["x"]), float(point["y"])))
    points.sort()
    pre_event = [point for point in points if point[0] < cutoff_ms]

    def timestamp_text(milliseconds: int) -> str:
        return datetime.fromtimestamp(
            milliseconds / 1000.0, tz=timezone.utc
        ).isoformat(timespec="seconds").replace("+00:00", "Z")

    return {
        "series_names": sorted(set(names)),
        "point_count": len(points),
        "strict_pre_event_point_count": len(pre_event),
        "first_point_utc": timestamp_text(points[0][0]) if points else None,
        "last_point_utc": timestamp_text(points[-1][0]) if points else None,
        "last_strict_pre_event_point_utc": (
            timestamp_text(pre_event[-1][0]) if pre_event else None
        ),
        "finite_decimal_prices": all(
            math.isfinite(price) and price > 1.0 for _, price in points
        ),
    }


class BoundedFetcher:
    def __init__(
        self,
        *,
        maximum_runtime_minutes: float,
        delay_seconds: float,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not 0 < maximum_runtime_minutes <= 60:
            raise ValueError("maximum runtime must be within (0, 60] minutes")
        if not 0 <= delay_seconds <= 10:
            raise ValueError("request delay must be within [0, 10] seconds")
        self.deadline = time.monotonic() + maximum_runtime_minutes * 60.0
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.request_count = 0
        self.response_bytes = 0
        self._last_request_at: float | None = None

    def get(self, url: str, *, referer: str | None = None) -> requests.Response:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("historical-odds audit reached its runtime limit")
        if self._last_request_at is not None:
            remaining = self.delay_seconds - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        headers = {"Referer": referer} if referer else None
        response = self.session.get(url, headers=headers, timeout=self.timeout_seconds)
        self._last_request_at = time.monotonic()
        self.request_count += 1
        self.response_bytes += len(response.content)
        response.raise_for_status()
        return response


def _robots_allows_all(text: str) -> bool:
    lines = [line.split("#", 1)[0].strip().casefold() for line in text.splitlines()]
    applies = False
    allowed_root = False
    disallowed_root = False
    for line in lines:
        if line.startswith("user-agent:"):
            applies = line.split(":", 1)[1].strip() == "*"
        elif applies and line.startswith("allow:"):
            allowed_root |= line.split(":", 1)[1].strip() == "/"
        elif applies and line.startswith("disallow:"):
            disallowed_root |= line.split(":", 1)[1].strip() == "/"
    return allowed_root and not disallowed_root


def _select_history_pages(
    pages: Sequence[Mapping[str, Any]], count: int
) -> list[Mapping[str, Any]]:
    eligible = [
        page
        for page in pages
        if page.get("is_ufc")
        and any(
            len(matchup.get("paired_books", [])) >= 3
            for matchup in page.get("matchups", [])
        )
    ]
    if count >= len(eligible):
        return eligible
    positions = _evenly_spaced(
        [
            SitemapEvent(
                url=str(page["url"]), event_date=date.fromisoformat(str(page["event_date"]))
            )
            for page in eligible
        ],
        count,
    )
    selected_urls = {item.url for item in positions}
    return [page for page in eligible if page["url"] in selected_urls]


def _select_pages_across_years(
    pages: Sequence[Mapping[str, Any]], count: int
) -> list[Mapping[str, Any]]:
    eligible = [page for page in pages if page.get("is_ufc") and page.get("matchups")]
    if count >= len(eligible):
        return eligible
    by_year: dict[str, list[Mapping[str, Any]]] = {}
    for page in eligible:
        by_year.setdefault(str(page["event_date"])[:4], []).append(page)
    selected: list[Mapping[str, Any]] = []
    unused: list[Mapping[str, Any]] = []
    for year in sorted(by_year):
        rows = sorted(by_year[year], key=lambda item: (item["event_date"], item["url"]))
        selected.append(rows[len(rows) // 2])
        unused.extend(item for item in rows if item is not selected[-1])
    if len(selected) > count:
        selected_events = _evenly_spaced(
            [
                SitemapEvent(
                    url=str(page["url"]),
                    event_date=date.fromisoformat(str(page["event_date"])),
                )
                for page in selected
            ],
            count,
        )
        urls = {item.url for item in selected_events}
        return [page for page in selected if page["url"] in urls]
    if len(selected) < count:
        needed = count - len(selected)
        extra_events = _evenly_spaced(
            [
                SitemapEvent(
                    url=str(page["url"]),
                    event_date=date.fromisoformat(str(page["event_date"])),
                )
                for page in unused
            ],
            needed,
        )
        extra_urls = {item.url for item in extra_events}
        selected.extend(page for page in unused if page["url"] in extra_urls)
    return sorted(selected[:count], key=lambda item: (item["event_date"], item["url"]))


def audit_bestfightodds(
    fetcher: BoundedFetcher,
    *,
    sample_size: int,
    history_events: int,
    history_books: int,
    mean_history_events: int,
    legacy_book_max_id: int,
    minimum_year: int,
    maximum_date: date,
) -> dict[str, Any]:
    robots_response = fetcher.get(f"{BESTFIGHTODDS_ROOT}/robots.txt")
    robots_text = robots_response.text
    if not _robots_allows_all(robots_text):
        raise RuntimeError("BestFightOdds robots policy does not allow the audit")
    terms_response = fetcher.get(f"{BESTFIGHTODDS_ROOT}/terms")
    terms_text = BeautifulSoup(terms_response.text, "html.parser").get_text(
        " ", strip=True
    )
    sitemap_response = fetcher.get(f"{BESTFIGHTODDS_ROOT}/sitemap-events.xml")
    sitemap_events = parse_bestfightodds_sitemap(sitemap_response.text)
    selected = select_stratified_events(
        sitemap_events,
        sample_size=sample_size,
        minimum_year=minimum_year,
        maximum_date=maximum_date,
    )

    pages: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for position, event in enumerate(selected, start=1):
        try:
            response = fetcher.get(event.url)
            parsed = parse_bestfightodds_event_page(
                response.text, url=event.url, sitemap_date=event.event_date
            )
            pages.append(parsed)
        except (requests.RequestException, ValueError) as error:
            failures.append(
                {"url": event.url, "error_type": type(error).__name__}
            )
        if position % 10 == 0:
            print(f"Audited {position}/{len(selected)} BestFightOdds event pages.", flush=True)

    history_pages = _select_history_pages(pages, history_events)
    history_checks: list[dict[str, Any]] = []
    for page in history_pages:
        matchups = sorted(
            page["matchups"],
            key=lambda item: (-len(item["paired_books"]), int(item["matchup_id"])),
        )
        if not matchups:
            continue
        matchup = matchups[0]
        books = matchup["paired_books"][:history_books]
        book_checks: list[dict[str, Any]] = []
        for book in books:
            sides: dict[str, Any] = {}
            for side in (1, 2):
                chart_url = (
                    f"{BESTFIGHTODDS_ROOT}/api/ggd?b={book['book_id']}"
                    f"&m={matchup['matchup_id']}&p={side}"
                )
                try:
                    response = fetcher.get(chart_url, referer=str(page["url"]))
                    chart = decode_bestfightodds_chart(response.text)
                    sides[str(side)] = summarize_chart(
                        chart, event_date=date.fromisoformat(page["event_date"])
                    )
                except (requests.RequestException, ValueError) as error:
                    sides[str(side)] = {
                        "error_type": type(error).__name__,
                        "point_count": 0,
                        "strict_pre_event_point_count": 0,
                    }
            book_checks.append(
                {
                    "book_id": book["book_id"],
                    "book": book["book"],
                    "sides": sides,
                    "paired_strict_pre_event": all(
                        sides[str(side)].get("strict_pre_event_point_count", 0) > 0
                        for side in (1, 2)
                    ),
                }
            )
        history_checks.append(
            {
                "url": page["url"],
                "event_date": page["event_date"],
                "title": page["title"],
                "matchup_id": matchup["matchup_id"],
                "fighter_1": matchup["fighter_1"],
                "fighter_2": matchup["fighter_2"],
                "books": book_checks,
                "paired_pre_event_book_count": sum(
                    bool(item["paired_strict_pre_event"]) for item in book_checks
                ),
            }
        )
        print(
            f"Checked timestamped history for {len(history_checks)}/{len(history_pages)} sampled events.",
            flush=True,
        )

    mean_history_checks: list[dict[str, Any]] = []
    for page in _select_pages_across_years(pages, mean_history_events):
        matchup = max(
            page["matchups"],
            key=lambda item: (len(item["paired_books"]), -int(item["matchup_id"])),
        )
        sides: dict[str, Any] = {}
        for side in (1, 2):
            chart_url = (
                f"{BESTFIGHTODDS_ROOT}/api/ggd?m={matchup['matchup_id']}&p={side}"
            )
            try:
                response = fetcher.get(chart_url, referer=str(page["url"]))
                chart = decode_bestfightodds_chart(response.text)
                sides[str(side)] = summarize_chart(
                    chart, event_date=date.fromisoformat(page["event_date"])
                )
            except (requests.RequestException, ValueError) as error:
                sides[str(side)] = {
                    "error_type": type(error).__name__,
                    "point_count": 0,
                    "strict_pre_event_point_count": 0,
                }
        mean_history_checks.append(
            {
                "url": page["url"],
                "event_date": page["event_date"],
                "title": page["title"],
                "matchup_id": matchup["matchup_id"],
                "fighter_1": matchup["fighter_1"],
                "fighter_2": matchup["fighter_2"],
                "sides": sides,
                "paired_strict_pre_event": all(
                    sides[str(side)].get("strict_pre_event_point_count", 0) > 0
                    for side in (1, 2)
                ),
            }
        )

    legacy_book_probe: dict[str, Any] | None = None
    if mean_history_checks:
        oldest_check = mean_history_checks[0]
        discovered: list[dict[str, Any]] = []
        for book_id in range(1, legacy_book_max_id + 1):
            chart_url = (
                f"{BESTFIGHTODDS_ROOT}/api/ggd?b={book_id}"
                f"&m={oldest_check['matchup_id']}&p=1"
            )
            try:
                response = fetcher.get(chart_url, referer=str(oldest_check["url"]))
                chart = decode_bestfightodds_chart(response.text)
                side_1 = summarize_chart(
                    chart, event_date=date.fromisoformat(oldest_check["event_date"])
                )
            except (requests.RequestException, ValueError):
                continue
            if side_1["point_count"] <= 0:
                continue
            side_2_url = (
                f"{BESTFIGHTODDS_ROOT}/api/ggd?b={book_id}"
                f"&m={oldest_check['matchup_id']}&p=2"
            )
            try:
                response = fetcher.get(side_2_url, referer=str(oldest_check["url"]))
                side_2 = summarize_chart(
                    decode_bestfightodds_chart(response.text),
                    event_date=date.fromisoformat(oldest_check["event_date"]),
                )
            except (requests.RequestException, ValueError):
                side_2 = {"point_count": 0, "strict_pre_event_point_count": 0}
            discovered.append(
                {
                    "book_id": book_id,
                    "series_names": side_1["series_names"],
                    "side_1": side_1,
                    "side_2": side_2,
                    "paired_strict_pre_event": all(
                        side.get("strict_pre_event_point_count", 0) > 0
                        for side in (side_1, side_2)
                    ),
                }
            )
        legacy_book_probe = {
            "event_date": oldest_check["event_date"],
            "url": oldest_check["url"],
            "matchup_id": oldest_check["matchup_id"],
            "fighter_1": oldest_check["fighter_1"],
            "fighter_2": oldest_check["fighter_2"],
            "book_ids_scanned": [1, legacy_book_max_id],
            "books_with_history": discovered,
            "paired_pre_event_book_count": sum(
                bool(item["paired_strict_pre_event"]) for item in discovered
            ),
        }

    page_years: dict[str, int] = {}
    for page in pages:
        year = str(page["event_date"])[:4]
        page_years[year] = page_years.get(year, 0) + 1
    three_book_pages = sum(
        any(len(matchup["paired_books"]) >= 3 for matchup in page["matchups"])
        for page in pages
    )
    three_book_pages_by_year: dict[str, dict[str, int]] = {}
    for page in pages:
        year = str(page["event_date"])[:4]
        row = three_book_pages_by_year.setdefault(year, {"pages": 0, "three_book_pages": 0})
        row["pages"] += 1
        row["three_book_pages"] += int(
            any(len(matchup["paired_books"]) >= 3 for matchup in page["matchups"])
        )
    three_book_history = sum(
        check["paired_pre_event_book_count"] >= 3 for check in history_checks
    )
    chart_points = sum(
        side.get("point_count", 0)
        for check in history_checks
        for book in check["books"]
        for side in book["sides"].values()
    )
    pre_event_chart_points = sum(
        side.get("strict_pre_event_point_count", 0)
        for check in history_checks
        for book in check["books"]
        for side in book["sides"].values()
    )
    mean_years = sorted(
        {
            check["event_date"][:4]
            for check in mean_history_checks
            if check["paired_strict_pre_event"]
        }
    )
    three_book_years = sorted(
        {
            check["event_date"][:4]
            for check in history_checks
            if check["paired_pre_event_book_count"] >= 3
        }
    )

    terms_lower = terms_text.casefold()
    automation_terms = [
        token
        for token in ("scrap", "robot", "automated", "data mining", "bulk")
        if token in terms_lower
    ]
    page_summaries = []
    for page in pages:
        summary = {key: value for key, value in page.items() if key != "matchups"}
        summary["max_paired_books_for_one_matchup"] = max(
            (len(matchup["paired_books"]) for matchup in page["matchups"]),
            default=0,
        )
        page_summaries.append(summary)
    return {
        "source": "bestfightodds",
        "root_url": BESTFIGHTODDS_ROOT,
        "robots_allows_public_paths": True,
        "robots_sha256": _sha256_text(robots_text),
        "terms_url": f"{BESTFIGHTODDS_ROOT}/terms",
        "terms_sha256": _sha256_text(terms_text),
        "terms_automation_words_found": automation_terms,
        "bulk_reuse_permission": "not_explicitly_granted_or_prohibited_in_published_terms",
        "sitemap_ufc_event_count": len(sitemap_events),
        "sample_requested": sample_size,
        "sample_succeeded": len(pages),
        "sample_failures": failures,
        "sample_by_year": dict(sorted(page_years.items())),
        "pages_with_at_least_one_three_book_matchup": three_book_pages,
        "pages_with_at_least_one_three_book_matchup_rate": (
            three_book_pages / len(pages) if pages else 0.0
        ),
        "three_book_page_coverage_by_year": dict(sorted(three_book_pages_by_year.items())),
        "pages_with_line_movement_control": sum(
            bool(page["has_line_movement_control"]) for page in pages
        ),
        "pages_with_absolute_quote_time_in_html": sum(
            bool(page["has_absolute_quote_time_in_html"]) for page in pages
        ),
        "history_events_requested": history_events,
        "history_events_checked": len(history_checks),
        "history_events_with_three_paired_pre_event_books": three_book_history,
        "history_events_with_three_paired_pre_event_books_rate": (
            three_book_history / len(history_checks) if history_checks else 0.0
        ),
        "decoded_chart_points": chart_points,
        "decoded_strict_pre_event_chart_points": pre_event_chart_points,
        "mean_history_events_checked": len(mean_history_checks),
        "mean_history_events_with_paired_pre_event_data": sum(
            bool(check["paired_strict_pre_event"]) for check in mean_history_checks
        ),
        "mean_history_years_with_paired_pre_event_data": mean_years,
        "three_book_history_years_verified": three_book_years,
        "earliest_verified_mean_history_year": mean_years[0] if mean_years else None,
        "earliest_verified_three_book_history_year": (
            three_book_years[0] if three_book_years else None
        ),
        "technical_finding": (
            "absolute_timestamps_available; older mean history and newer multi-book history have different evidentiary strength"
            if mean_years and three_book_years
            else "no_usable_timestamped_chart_history_found"
        ),
        "event_pages": page_summaries,
        "history_checks": history_checks,
        "mean_history_checks": mean_history_checks,
        "legacy_book_probe": legacy_book_probe,
    }


def audit_fightodds_policy(fetcher: BoundedFetcher) -> dict[str, Any]:
    """Do not crawl when robots.txt resolves to the generic web application."""

    try:
        response = fetcher.get(f"{FIGHTODDS_ROOT}/robots.txt")
    except requests.RequestException as error:
        return {
            "source": "fightodds_io",
            "root_url": FIGHTODDS_ROOT,
            "status": "not_sampled",
            "reason": "robots_request_failed",
            "error_type": type(error).__name__,
        }
    content_type = response.headers.get("content-type", "")
    body = response.text
    looks_like_robots = "user-agent:" in body.casefold() and "text/plain" in content_type
    return {
        "source": "fightodds_io",
        "root_url": FIGHTODDS_ROOT,
        "robots_url_status": response.status_code,
        "robots_content_type": content_type,
        "robots_sha256": _sha256_text(body),
        "machine_readable_robots_policy": looks_like_robots,
        "status": "policy_check_only",
        "reason": (
            "machine_readable_robots_policy_available"
            if looks_like_robots
            else "robots_path_returns_generic_html_application; automated sampling skipped"
        ),
    }


def _build_decision(bestfightodds: Mapping[str, Any]) -> dict[str, Any]:
    checked = int(bestfightodds.get("history_events_checked", 0))
    usable = int(
        bestfightodds.get("history_events_with_three_paired_pre_event_books", 0)
    )
    technically_feasible = checked >= 5 and usable / checked >= 0.6
    return {
        "historical_timestamped_odds_technically_feasible": technically_feasible,
        "coverage_is_partial": True,
        "earliest_verified_mean_history_year": bestfightodds.get(
            "earliest_verified_mean_history_year"
        ),
        "earliest_verified_three_book_history_year": bestfightodds.get(
            "earliest_verified_three_book_history_year"
        ),
        "large_backfill_authorized_by_this_audit": False,
        "why_not_authorized": (
            "technical_sample_failed"
            if not technically_feasible
            else "published terms do not explicitly address bulk automated reuse"
        ),
        "recommended_next_action": (
            "request source permission, then run a low-rate resumable backfill outside Git"
            if technically_feasible
            else "continue prospective collection; do not backfill this source"
        ),
        "accepted_historical_record_rule": (
            "book, matchup, both sides, absolute quote timestamp strictly before event, "
            "and at least three distinct books at the chosen cutoff"
        ),
        "development_horizons": ["opening", "T-72", "T-24", "T-6", "closing"],
        "production_rule": (
            "historical data may develop a frozen method but cannot replace future-only confirmation"
        ),
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    bfo = report["sources"]["bestfightodds"]
    fightodds = report["sources"]["fightodds_io"]
    decision = report["decision"]
    years = bfo.get("sample_by_year", {})
    year_span = (
        f"{min(years)}-{max(years)}" if isinstance(years, dict) and years else "none"
    )
    feasible = bool(decision["historical_timestamped_odds_technically_feasible"])
    return f"""# Free historical odds feasibility audit

Generated {report['generated_at_utc']}. This is a bounded, read-only research
audit. It did not change production predictions and did not store raw pages.

## Bottom line

BestFightOdds is **{'technically usable' if feasible else 'not technically usable'}** for a historical research
backfill. Its visible event tables do not contain quote times, but the public
line-movement chart used by the site returns bookmaker-specific prices with
absolute timestamps. The sample recovered {bfo['decoded_strict_pre_event_chart_points']:,}
strictly pre-event price points.

Coverage is not uniform. Mean market history was verified back to
{bfo['earliest_verified_mean_history_year']}; a strict three-book history was
verified only from {bfo['earliest_verified_three_book_history_year']} in this
sample. Older mean or single-book prices can benchmark the model, but they must
not be presented as equally strong as a multi-book consensus.

A large backfill is **not started automatically**. The public robots policy
allows the sampled paths, while the short published terms neither grant nor
prohibit bulk automated reuse. Ask the source for permission or clarification
before copying the archive at scale. If permission is received, store raw
history outside Git and commit only compact derived consensus data and audits.

FightOdds.io was not crawled: `{fightodds['reason']}`.

## What was checked

- {bfo['sample_succeeded']} of {bfo['sample_requested']} BestFightOdds event pages succeeded, spread across {year_span}.
- {bfo['pages_with_at_least_one_three_book_matchup']} pages exposed at least one matchup with paired prices from three books.
- Detailed line history was tested on {bfo['history_events_checked']} events.
- {bfo['history_events_with_three_paired_pre_event_books']} detailed events had timestamped, pre-event prices for both fighters from all three sampled books.
- Mean history was checked across {bfo['mean_history_events_checked']} events; {bfo['mean_history_events_with_paired_pre_event_data']} had pre-event data for both fighters.
- {bfo['decoded_chart_points']:,} chart points decoded; {bfo['decoded_strict_pre_event_chart_points']:,} occurred strictly before the event calendar date.
- The audit made {report['network']['request_count']} requests and downloaded {report['network']['response_bytes'] / 1_048_576:.2f} MiB.

## Data rule for any future backfill

Accept a price only when it has a bookmaker, matchup, both fighter sides, and
an absolute timestamp before the event. Build separate opening, T-72, T-24,
T-6, and closing datasets; never mix those horizons. Require at least three
books for a market consensus. With date-only historical event times, the
strict safe cutoff is before 00:00 UTC on the event date until an authoritative
event start time is added.

Historical results may be used to develop and freeze a model/market blend.
They do not replace the already scheduled future-only confirmation over at
least 200 fights and 20 events.

## Next action

{decision['recommended_next_action'].capitalize()}. The backfill should be
resumable, rate-limited, capped by requests and disk space, and retain source
timestamps. If permission is not available, continue the repository's T-24
prospective capture instead of using weakly dated closing lines.
"""


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    fetcher = BoundedFetcher(
        maximum_runtime_minutes=args.max_runtime_minutes,
        delay_seconds=args.request_delay_seconds,
    )
    bestfightodds = audit_bestfightodds(
        fetcher,
        sample_size=args.sample_size,
        history_events=args.history_events,
        history_books=args.history_books,
        mean_history_events=args.mean_history_events,
        legacy_book_max_id=args.legacy_book_max_id,
        minimum_year=args.minimum_year,
        maximum_date=date.fromisoformat(args.maximum_date),
    )
    fightodds = audit_fightodds_policy(fetcher)
    report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "generated_at_utc": _utc_now_text(),
        "scope": "no-cost historical odds feasibility; research only",
        "sources": {
            "bestfightodds": bestfightodds,
            "fightodds_io": fightodds,
        },
        "decision": _build_decision(bestfightodds),
        "network": {
            "user_agent": USER_AGENT,
            "request_count": fetcher.request_count,
            "response_bytes": fetcher.response_bytes,
            "request_delay_seconds": args.request_delay_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "raw_pages_stored": False,
        },
    }
    report["report_sha256"] = _sha256_text(_canonical_json(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--history-events", type=int, default=DEFAULT_HISTORY_EVENTS
    )
    parser.add_argument("--history-books", type=int, default=DEFAULT_HISTORY_BOOKS)
    parser.add_argument(
        "--mean-history-events", type=int, default=DEFAULT_MEAN_HISTORY_EVENTS
    )
    parser.add_argument(
        "--legacy-book-max-id", type=int, default=DEFAULT_LEGACY_BOOK_MAX_ID
    )
    parser.add_argument("--minimum-year", type=int, default=DEFAULT_MINIMUM_YEAR)
    parser.add_argument("--maximum-date", default=date.today().isoformat())
    parser.add_argument(
        "--max-runtime-minutes", type=float, default=DEFAULT_MAX_RUNTIME_MINUTES
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.history_events <= 0 or args.history_events > args.sample_size:
        raise SystemExit("--history-events must be within (0, sample-size]")
    if args.history_books < 3 or args.history_books > 10:
        raise SystemExit("--history-books must be within [3, 10]")
    if args.mean_history_events <= 0 or args.mean_history_events > args.sample_size:
        raise SystemExit("--mean-history-events must be within (0, sample-size]")
    if args.legacy_book_max_id <= 0 or args.legacy_book_max_id > 100:
        raise SystemExit("--legacy-book-max-id must be within (0, 100]")
    report = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(_markdown_report(report), encoding="utf-8")
    decision = report["decision"]
    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(
        "Timestamped historical odds technically feasible: "
        f"{decision['historical_timestamped_odds_technically_feasible']}"
    )
    print(f"Large backfill started: {decision['large_backfill_authorized_by_this_audit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
