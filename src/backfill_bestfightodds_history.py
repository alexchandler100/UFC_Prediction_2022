"""Resumable, bounded BestFightOdds history collector for local research.

The collector follows the public event pages and the same read-only chart
endpoint used by the site's line-movement UI.  It stores decoded timestamped
prices in a compact SQLite database outside Git.  It is deliberately separate
from production predictions and never stores raw HTML.

The source publishes an allow-all robots policy, while its short terms page
does not explicitly discuss automated bulk research reuse.  Network runs
therefore require an acknowledgement flag and use conservative request,
runtime, disk, and free-space caps.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import requests

from audit_historical_odds_sources import (
    ALGORITHM_VERSION as AUDIT_ALGORITHM_VERSION,
    BESTFIGHTODDS_ROOT,
    USER_AGENT,
    decode_bestfightodds_chart,
    parse_bestfightodds_event_page,
    parse_bestfightodds_sitemap,
)
from external_mma.identity import normalize_name
from fight_stat_helpers import maybe_replace_alias_by_default_name


SCHEMA_VERSION = 1
COLLECTOR_VERSION = 1
DEFAULT_FROM_YEAR = 2021
DEFAULT_TO_YEAR = date.today().year
DEFAULT_MODE = "both"
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_MAX_RUNTIME_HOURS = 6.0
DEFAULT_MAX_REQUESTS = 25_000
DEFAULT_MAX_SOURCE_ATTEMPTS = 2
DEFAULT_MAX_DATABASE_MIB = 1024.0
DEFAULT_MINIMUM_FREE_GIB = 5.0
DEFAULT_MINIMUM_CONSENSUS_BOOKS = 3
NON_SPORTSBOOK_BOOK_IDS = frozenset({28, 29})
DEFAULT_RAW_FIGHTS = Path(
    "src/content/data/processed/ufc_fights_reported_doubled.csv"
)

HORIZONS = (
    ("opening", None),
    ("safe_t72", 72),
    ("safe_t24", 24),
    ("safe_t6", 6),
    ("strict_latest_before_event_date", 0),
)


class BackfillError(RuntimeError):
    """Invalid or unsafe backfill state."""


class BudgetReached(RuntimeError):
    """A configured request, runtime, or disk budget stopped the session."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime | None = None) -> str:
    return (value or _utc_now()).astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def default_database_path() -> Path:
    configured = os.environ.get("UFC_HISTORICAL_ODDS_DIR", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".ufc-data-lab"
    return root / "historical-odds" / "bestfightodds" / "history.sqlite3"


def _stable_id(value: object) -> str:
    return str(value or "").strip().rstrip("/").rsplit("/", 1)[-1]


def _event_source_id(url: str) -> str:
    token = urlparse(url).path.rstrip("/").rsplit("-", 1)[-1]
    return token if token.isdigit() else ""


def _name_key(value: object) -> str:
    canonical = maybe_replace_alias_by_default_name(str(value or "").strip())
    return " ".join(sorted(normalize_name(canonical).split()))


def _pair_key(left: object, right: object) -> tuple[str, str] | None:
    first = _name_key(left)
    second = _name_key(right)
    if not first or not second or first == second:
        return None
    return tuple(sorted((first, second)))


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _ordered_name_distances(
    source_1: str,
    source_2: str,
    reference: "UFCFightReference",
) -> tuple[tuple[int, int], bool]:
    source_1_key = _name_key(source_1)
    source_2_key = _name_key(source_2)
    reference_1_key = _name_key(reference.fighter_1_name)
    reference_2_key = _name_key(reference.fighter_2_name)
    direct = (
        _edit_distance(source_1_key, reference_1_key),
        _edit_distance(source_2_key, reference_2_key),
    )
    swapped = (
        _edit_distance(source_1_key, reference_2_key),
        _edit_distance(source_2_key, reference_1_key),
    )
    return (direct, True) if sum(direct) <= sum(swapped) else (swapped, False)


@dataclass(frozen=True)
class UFCFightReference:
    event_date: date
    event_id: str
    fight_id: str
    fighter_1_id: str
    fighter_2_id: str
    fighter_1_name: str
    fighter_2_name: str


class UFCFightIndex:
    def __init__(self, rows: Iterable[UFCFightReference]) -> None:
        self.by_pair: dict[tuple[str, str], list[UFCFightReference]] = defaultdict(list)
        self.by_date: dict[date, list[UFCFightReference]] = defaultdict(list)
        for row in rows:
            key = _pair_key(row.fighter_1_name, row.fighter_2_name)
            if key is not None:
                self.by_pair[key].append(row)
                self.by_date[row.event_date].append(row)
        for values in self.by_pair.values():
            values.sort(key=lambda item: (item.event_date, item.fight_id))

    @classmethod
    def from_csv(cls, path: str | Path) -> "UFCFightIndex":
        source = Path(path)
        if not source.is_file():
            raise BackfillError(f"UFC fight file does not exist: {source}")
        references: list[UFCFightReference] = []
        seen: set[str] = set()
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "date",
                "event_url",
                "fight_url",
                "fighter",
                "opponent",
                "fighter_url",
                "opponent_url",
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise BackfillError(f"UFC fight file is missing columns: {sorted(missing)}")
            for row in reader:
                fight_id = _stable_id(row["fight_url"])
                if not fight_id or fight_id in seen:
                    continue
                seen.add(fight_id)
                try:
                    event_date = date.fromisoformat(row["date"][:10])
                except (TypeError, ValueError):
                    continue
                references.append(
                    UFCFightReference(
                        event_date=event_date,
                        event_id=_stable_id(row["event_url"]),
                        fight_id=fight_id,
                        fighter_1_id=_stable_id(row["fighter_url"]),
                        fighter_2_id=_stable_id(row["opponent_url"]),
                        fighter_1_name=str(row["fighter"]).strip(),
                        fighter_2_name=str(row["opponent"]).strip(),
                    )
                )
        return cls(references)

    def match(
        self, *, event_date: date, fighter_1: str, fighter_2: str
    ) -> tuple[UFCFightReference | None, str, int | None]:
        key = _pair_key(fighter_1, fighter_2)
        if key is None:
            return None, "invalid_fighter_pair", None
        candidates = self.by_pair.get(key, [])
        exact = [item for item in candidates if item.event_date == event_date]
        if len(exact) == 1:
            return exact[0], "exact_date_and_fighters", 0
        if len(exact) > 1:
            return None, "ambiguous_exact_date_and_fighters", None
        adjacent = [
            item for item in candidates if abs((item.event_date - event_date).days) <= 1
        ]
        if len(adjacent) == 1:
            offset = (adjacent[0].event_date - event_date).days
            return adjacent[0], "unique_fighters_within_one_day", offset
        if len(adjacent) > 1:
            return None, "ambiguous_fighters_within_one_day", None
        near_name_candidates: list[UFCFightReference] = []
        for day_offset in (-1, 0, 1):
            candidate_date = event_date + timedelta(days=day_offset)
            for candidate in self.by_date.get(candidate_date, []):
                distances, _ = _ordered_name_distances(
                    fighter_1, fighter_2, candidate
                )
                if min(distances) == 0 and max(distances) <= 1:
                    near_name_candidates.append(candidate)
        unique_near = {item.fight_id: item for item in near_name_candidates}
        if len(unique_near) == 1:
            reference = next(iter(unique_near.values()))
            offset = (reference.event_date - event_date).days
            return reference, "unique_near_name_within_one_day", offset
        if len(unique_near) > 1:
            return None, "ambiguous_near_name_within_one_day", None
        return None, "unmatched", None


def _source_first_is_reference_first(
    source_1: str, source_2: str, reference: UFCFightReference
) -> bool:
    _, direct = _ordered_name_distances(source_1, source_2, reference)
    return direct


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_external_database_path(path: Path, repository_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if _is_inside(resolved, repository_root):
        raise BackfillError(
            "historical odds database must be outside the repository; "
            "use ~/.ufc-data-lab or set UFC_HISTORICAL_ODDS_DIR"
        )
    return resolved


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_url TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL,
    source_event_date TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    organizer TEXT NOT NULL DEFAULT '',
    html_sha256 TEXT NOT NULL DEFAULT '',
    page_status TEXT NOT NULL DEFAULT 'pending',
    page_attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS matchups (
    matchup_id INTEGER PRIMARY KEY,
    event_url TEXT NOT NULL REFERENCES events(event_url),
    fighter_1_name TEXT NOT NULL,
    fighter_2_name TEXT NOT NULL,
    mapping_status TEXT NOT NULL,
    ufc_event_date TEXT,
    ufc_event_id TEXT,
    ufc_fight_id TEXT,
    ufc_fighter_1_id TEXT,
    ufc_fighter_2_id TEXT,
    source_to_ufc_date_offset_days INTEGER,
    updated_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS matchups_event_url_idx ON matchups(event_url);
CREATE INDEX IF NOT EXISTS matchups_ufc_fight_id_idx ON matchups(ufc_fight_id);
CREATE TABLE IF NOT EXISTS matchup_books (
    matchup_id INTEGER NOT NULL REFERENCES matchups(matchup_id),
    book_key TEXT NOT NULL,
    book_id INTEGER,
    book_name TEXT NOT NULL,
    PRIMARY KEY (matchup_id, book_key)
);
CREATE TABLE IF NOT EXISTS downloads (
    matchup_id INTEGER NOT NULL REFERENCES matchups(matchup_id),
    book_key TEXT NOT NULL,
    side INTEGER NOT NULL CHECK (side IN (1, 2)),
    endpoint TEXT NOT NULL,
    status TEXT NOT NULL,
    point_count INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (matchup_id, book_key, side)
);
CREATE TABLE IF NOT EXISTS quotes (
    matchup_id INTEGER NOT NULL REFERENCES matchups(matchup_id),
    book_key TEXT NOT NULL,
    side INTEGER NOT NULL CHECK (side IN (1, 2)),
    observed_at_ms INTEGER NOT NULL,
    decimal_odds REAL NOT NULL CHECK (decimal_odds > 1.0),
    series_name TEXT NOT NULL,
    PRIMARY KEY (matchup_id, book_key, side, observed_at_ms)
);
CREATE INDEX IF NOT EXISTS quotes_lookup_idx
    ON quotes(matchup_id, book_key, side, observed_at_ms);
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    response_bytes INTEGER NOT NULL DEFAULT 0,
    events_processed INTEGER NOT NULL DEFAULT 0,
    message TEXT
);
"""


def open_database(path: Path, *, mode: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.executescript(SCHEMA_SQL)
    expected = {
        "schema_version": str(SCHEMA_VERSION),
        "collector_version": str(COLLECTOR_VERSION),
        "audit_parser_version": str(AUDIT_ALGORITHM_VERSION),
        "download_mode": mode,
    }
    existing = {
        row["key"]: row["value"]
        for row in connection.execute("SELECT key, value FROM metadata")
    }
    for key, value in expected.items():
        if key in existing and existing[key] != value:
            connection.close()
            raise BackfillError(
                f"database {key} is {existing[key]!r}, expected {value!r}; "
                "use a new database path"
            )
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", (key, value)
        )
    if "created_at_utc" not in existing:
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES ('created_at_utc', ?)",
            (_utc_text(),),
        )
    connection.commit()
    return connection


def open_database_readonly(path: Path, *, mode: str) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    existing = {
        row["key"]: row["value"]
        for row in connection.execute("SELECT key, value FROM metadata")
    }
    expected = {
        "schema_version": str(SCHEMA_VERSION),
        "collector_version": str(COLLECTOR_VERSION),
        "audit_parser_version": str(AUDIT_ALGORITHM_VERSION),
        "download_mode": mode,
    }
    for key, value in expected.items():
        if existing.get(key) != value:
            connection.close()
            raise BackfillError(
                f"database {key} is {existing.get(key)!r}, expected {value!r}"
            )
    return connection


def _plain_error(error: BaseException) -> str:
    return type(error).__name__


class BackfillFetcher:
    def __init__(
        self,
        *,
        max_runtime_hours: float,
        max_requests: int,
        delay_seconds: float,
        timeout_seconds: float,
    ) -> None:
        if not 0 < max_runtime_hours <= 24:
            raise BackfillError("max runtime must be within (0, 24] hours")
        if not 0 < max_requests <= 100_000:
            raise BackfillError("max requests must be within (0, 100000]")
        if not 0.35 <= delay_seconds <= 10:
            raise BackfillError("request delay must be within [0.35, 10] seconds")
        self.deadline = time.monotonic() + max_runtime_hours * 3600.0
        self.max_requests = max_requests
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.requests = 0
        self.response_bytes = 0
        self._last_request_at: float | None = None
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _check_budget(self) -> None:
        if self.requests >= self.max_requests:
            raise BudgetReached("request cap reached")
        if time.monotonic() >= self.deadline:
            raise BudgetReached("runtime cap reached")

    def get(self, url: str, *, referer: str | None = None) -> requests.Response:
        self._check_budget()
        if self._last_request_at is not None:
            wait = self.delay_seconds - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
        headers = {"Referer": referer} if referer else None
        last_error: requests.RequestException | None = None
        for attempt in range(2):
            self._check_budget()
            try:
                response = self.session.get(
                    url, headers=headers, timeout=self.timeout_seconds
                )
                self.requests += 1
                self.response_bytes += len(response.content)
                self._last_request_at = time.monotonic()
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 0:
                        time.sleep(min(10.0, 2.0 + attempt * 2.0))
                        continue
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                self.requests += 1
                self._last_request_at = time.monotonic()
                last_error = error
                if attempt == 0:
                    time.sleep(2.0)
        assert last_error is not None
        raise last_error


def _robots_allows_public_paths(text: str) -> bool:
    user_agent_all = False
    root_allowed = False
    root_disallowed = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip().casefold()
        if line.startswith("user-agent:"):
            user_agent_all = line.split(":", 1)[1].strip() == "*"
        elif user_agent_all and line.startswith("allow:"):
            root_allowed |= line.split(":", 1)[1].strip() == "/"
        elif user_agent_all and line.startswith("disallow:"):
            root_disallowed |= line.split(":", 1)[1].strip() == "/"
    return root_allowed and not root_disallowed


def _database_size_mib(path: Path) -> float:
    paths = [path, Path(f"{path}-wal"), Path(f"{path}-shm")]
    return sum(item.stat().st_size for item in paths if item.exists()) / 1_048_576.0


def check_storage_budget(
    path: Path, *, max_database_mib: float, minimum_free_gib: float
) -> None:
    size = _database_size_mib(path)
    if size > max_database_mib:
        raise BudgetReached(
            f"database cap reached ({size:.1f} MiB > {max_database_mib:.1f} MiB)"
        )
    free_gib = shutil.disk_usage(path.parent).free / (1024.0**3)
    if free_gib < minimum_free_gib:
        raise BudgetReached(
            f"free-space floor reached ({free_gib:.1f} GiB < {minimum_free_gib:.1f} GiB)"
        )


def _upsert_event_page(
    connection: sqlite3.Connection,
    *,
    page: Mapping[str, Any],
    fight_index: UFCFightIndex,
) -> None:
    now = _utc_text()
    event_url = str(page["url"])
    connection.execute(
        """
        INSERT INTO events(
            event_url, source_event_id, source_event_date, title, organizer,
            html_sha256, page_status, page_attempts, last_error, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, 'parsed', 1, NULL, ?)
        ON CONFLICT(event_url) DO UPDATE SET
            source_event_date=excluded.source_event_date,
            title=excluded.title,
            organizer=excluded.organizer,
            html_sha256=excluded.html_sha256,
            page_status='parsed',
            page_attempts=events.page_attempts + 1,
            last_error=NULL,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            event_url,
            _event_source_id(event_url),
            page["event_date"],
            page["title"],
            page["organizer"],
            page["html_sha256"],
            now,
        ),
    )
    source_date = date.fromisoformat(str(page["event_date"]))
    for matchup in page["matchups"]:
        matchup_id = int(matchup["matchup_id"])
        reference, status, offset = fight_index.match(
            event_date=source_date,
            fighter_1=str(matchup["fighter_1"]),
            fighter_2=str(matchup["fighter_2"]),
        )
        values: tuple[Any, ...]
        if reference is None:
            values = (None, None, None, None, None)
        else:
            source_first_is_reference_first = _source_first_is_reference_first(
                matchup["fighter_1"], matchup["fighter_2"], reference
            )
            source_fighter_1_id = (
                reference.fighter_1_id
                if source_first_is_reference_first
                else reference.fighter_2_id
            )
            source_fighter_2_id = (
                reference.fighter_2_id
                if source_first_is_reference_first
                else reference.fighter_1_id
            )
            values = (
                reference.event_date.isoformat(),
                reference.event_id,
                reference.fight_id,
                source_fighter_1_id,
                source_fighter_2_id,
            )
        connection.execute(
            """
            INSERT INTO matchups(
                matchup_id, event_url, fighter_1_name, fighter_2_name,
                mapping_status, ufc_event_date, ufc_event_id, ufc_fight_id,
                ufc_fighter_1_id, ufc_fighter_2_id,
                source_to_ufc_date_offset_days, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(matchup_id) DO UPDATE SET
                event_url=excluded.event_url,
                fighter_1_name=excluded.fighter_1_name,
                fighter_2_name=excluded.fighter_2_name,
                mapping_status=excluded.mapping_status,
                ufc_event_date=excluded.ufc_event_date,
                ufc_event_id=excluded.ufc_event_id,
                ufc_fight_id=excluded.ufc_fight_id,
                ufc_fighter_1_id=excluded.ufc_fighter_1_id,
                ufc_fighter_2_id=excluded.ufc_fighter_2_id,
                source_to_ufc_date_offset_days=excluded.source_to_ufc_date_offset_days,
                updated_at_utc=excluded.updated_at_utc
            """,
            (
                matchup_id,
                event_url,
                matchup["fighter_1"],
                matchup["fighter_2"],
                status,
                *values,
                offset,
                now,
            ),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO matchup_books(
                matchup_id, book_key, book_id, book_name
            ) VALUES (?, 'mean', NULL, 'BestFightOdds mean')
            """,
            (matchup_id,),
        )
        for book in matchup["paired_books"]:
            book_id = int(book["book_id"])
            if book_id in NON_SPORTSBOOK_BOOK_IDS:
                continue
            connection.execute(
                """
                INSERT INTO matchup_books(matchup_id, book_key, book_id, book_name)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(matchup_id, book_key) DO UPDATE SET
                    book_name=excluded.book_name
                """,
                (matchup_id, f"book:{book_id}", book_id, book["book"]),
            )
    connection.commit()


def _record_event_failure(
    connection: sqlite3.Connection, *, event_url: str, event_date: str, error: Exception
) -> None:
    connection.execute(
        """
        INSERT INTO events(
            event_url, source_event_id, source_event_date, page_status,
            page_attempts, last_error, updated_at_utc
        ) VALUES (?, ?, ?, 'failed', 1, ?, ?)
        ON CONFLICT(event_url) DO UPDATE SET
            page_status='failed',
            page_attempts=events.page_attempts + 1,
            last_error=excluded.last_error,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            event_url,
            _event_source_id(event_url),
            event_date,
            _plain_error(error),
            _utc_text(),
        ),
    )
    connection.commit()


def refresh_fight_mappings(
    connection: sqlite3.Connection, *, fight_index: UFCFightIndex
) -> int:
    rows = connection.execute(
        """
        SELECT m.matchup_id, m.fighter_1_name, m.fighter_2_name,
               e.source_event_date
        FROM matchups AS m
        JOIN events AS e ON e.event_url=m.event_url
        ORDER BY m.matchup_id
        """
    ).fetchall()
    changed = 0
    for row in rows:
        reference, status, offset = fight_index.match(
            event_date=date.fromisoformat(row["source_event_date"]),
            fighter_1=row["fighter_1_name"],
            fighter_2=row["fighter_2_name"],
        )
        if reference is None:
            values: tuple[Any, ...] = (None, None, None, None, None)
        else:
            source_first_is_reference_first = _source_first_is_reference_first(
                row["fighter_1_name"], row["fighter_2_name"], reference
            )
            source_fighter_1_id = (
                reference.fighter_1_id
                if source_first_is_reference_first
                else reference.fighter_2_id
            )
            source_fighter_2_id = (
                reference.fighter_2_id
                if source_first_is_reference_first
                else reference.fighter_1_id
            )
            values = (
                reference.event_date.isoformat(),
                reference.event_id,
                reference.fight_id,
                source_fighter_1_id,
                source_fighter_2_id,
            )
        previous = connection.execute(
            """
            SELECT mapping_status, ufc_event_date, ufc_event_id, ufc_fight_id,
                   ufc_fighter_1_id, ufc_fighter_2_id,
                   source_to_ufc_date_offset_days
            FROM matchups WHERE matchup_id=?
            """,
            (row["matchup_id"],),
        ).fetchone()
        candidate = (status, *values, offset)
        if tuple(previous) != candidate:
            changed += 1
        connection.execute(
            """
            UPDATE matchups SET mapping_status=?, ufc_event_date=?,
                ufc_event_id=?, ufc_fight_id=?, ufc_fighter_1_id=?,
                ufc_fighter_2_id=?, source_to_ufc_date_offset_days=?,
                updated_at_utc=?
            WHERE matchup_id=?
            """,
            (*candidate, _utc_text(), row["matchup_id"]),
        )
    connection.commit()
    return changed


@dataclass(frozen=True)
class DownloadSpec:
    matchup_id: int
    event_url: str
    book_key: str
    book_id: int | None
    book_name: str
    side: int

    @property
    def endpoint(self) -> str:
        book = f"b={self.book_id}&" if self.book_id is not None else ""
        return (
            f"{BESTFIGHTODDS_ROOT}/api/ggd?{book}"
            f"m={self.matchup_id}&p={self.side}"
        )


def pending_downloads_for_event(
    connection: sqlite3.Connection,
    *,
    event_url: str,
    mode: str,
    maximum_attempts: int = DEFAULT_MAX_SOURCE_ATTEMPTS,
) -> list[DownloadSpec]:
    if maximum_attempts < 1:
        raise ValueError("maximum attempts must be at least one")
    include_mean = mode in {"mean", "both"}
    include_books = mode in {"books", "both"}
    rows = connection.execute(
        """
        SELECT m.matchup_id, m.event_url, m.mapping_status,
               b.book_key, b.book_id, b.book_name,
               d1.status AS side_1_status, d2.status AS side_2_status,
               d1.attempts AS side_1_attempts, d2.attempts AS side_2_attempts
        FROM matchups AS m
        JOIN matchup_books AS b ON b.matchup_id = m.matchup_id
        LEFT JOIN downloads AS d1
          ON d1.matchup_id = m.matchup_id AND d1.book_key = b.book_key AND d1.side = 1
        LEFT JOIN downloads AS d2
          ON d2.matchup_id = m.matchup_id AND d2.book_key = b.book_key AND d2.side = 2
        WHERE m.event_url = ?
        ORDER BY m.matchup_id, b.book_key
        """,
        (event_url,),
    ).fetchall()
    output: list[DownloadSpec] = []
    for row in rows:
        if not str(row["mapping_status"]).startswith(("exact_", "unique_")):
            continue
        is_mean = row["book_key"] == "mean"
        if (is_mean and not include_mean) or (not is_mean and not include_books):
            continue
        for side in (1, 2):
            status = row[f"side_{side}_status"]
            if status in {"complete", "empty"}:
                continue
            attempts = int(row[f"side_{side}_attempts"] or 0)
            if status == "failed" and attempts >= maximum_attempts:
                continue
            output.append(
                DownloadSpec(
                    matchup_id=int(row["matchup_id"]),
                    event_url=str(row["event_url"]),
                    book_key=str(row["book_key"]),
                    book_id=(int(row["book_id"]) if row["book_id"] is not None else None),
                    book_name=str(row["book_name"]),
                    side=side,
                )
            )
    return output


def _flatten_chart_points(series: Sequence[Mapping[str, Any]]) -> list[tuple[int, float, str]]:
    output: dict[int, tuple[float, str]] = {}
    for item in series:
        name = str(item.get("name", "")).strip()
        for point in item.get("data", []):
            observed = int(point["x"])
            decimal_odds = float(point["y"])
            if observed in output and output[observed][0] != decimal_odds:
                raise ValueError("chart contains conflicting prices at one timestamp")
            output[observed] = (decimal_odds, name)
    return [
        (observed, price, name)
        for observed, (price, name) in sorted(output.items())
    ]


def store_download(
    connection: sqlite3.Connection,
    *,
    spec: DownloadSpec,
    series: Sequence[Mapping[str, Any]],
) -> int:
    points = _flatten_chart_points(series)
    status = "complete" if points else "empty"
    now = _utc_text()
    with connection:
        connection.execute(
            "DELETE FROM quotes WHERE matchup_id=? AND book_key=? AND side=?",
            (spec.matchup_id, spec.book_key, spec.side),
        )
        connection.executemany(
            """
            INSERT INTO quotes(
                matchup_id, book_key, side, observed_at_ms,
                decimal_odds, series_name
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    spec.matchup_id,
                    spec.book_key,
                    spec.side,
                    observed,
                    price,
                    name,
                )
                for observed, price, name in points
            ],
        )
        connection.execute(
            """
            INSERT INTO downloads(
                matchup_id, book_key, side, endpoint, status, point_count,
                attempts, last_error, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, 1, NULL, ?)
            ON CONFLICT(matchup_id, book_key, side) DO UPDATE SET
                endpoint=excluded.endpoint,
                status=excluded.status,
                point_count=excluded.point_count,
                attempts=downloads.attempts + 1,
                last_error=NULL,
                updated_at_utc=excluded.updated_at_utc
            """,
            (
                spec.matchup_id,
                spec.book_key,
                spec.side,
                spec.endpoint,
                status,
                len(points),
                now,
            ),
        )
    return len(points)


def store_download_failure(
    connection: sqlite3.Connection, *, spec: DownloadSpec, error: Exception
) -> None:
    connection.execute(
        """
        INSERT INTO downloads(
            matchup_id, book_key, side, endpoint, status, point_count,
            attempts, last_error, updated_at_utc
        ) VALUES (?, ?, ?, ?, 'failed', 0, 1, ?, ?)
        ON CONFLICT(matchup_id, book_key, side) DO UPDATE SET
            endpoint=excluded.endpoint,
            status='failed',
            attempts=downloads.attempts + 1,
            last_error=excluded.last_error,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            spec.matchup_id,
            spec.book_key,
            spec.side,
            spec.endpoint,
            _plain_error(error),
            _utc_text(),
        ),
    )
    connection.commit()


def _epoch_ms_text(value: int) -> str:
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def _latest_common_at_or_before(
    side_1: Mapping[int, float], side_2: Mapping[int, float], cutoff_ms: int
) -> tuple[int, float, float] | None:
    common = sorted(set(side_1).intersection(side_2))
    eligible = [timestamp for timestamp in common if timestamp <= cutoff_ms]
    if not eligible:
        return None
    timestamp = eligible[-1]
    return timestamp, side_1[timestamp], side_2[timestamp]


def _first_common_before(
    side_1: Mapping[int, float], side_2: Mapping[int, float], cutoff_ms: int
) -> tuple[int, float, float] | None:
    common = sorted(
        timestamp
        for timestamp in set(side_1).intersection(side_2)
        if timestamp < cutoff_ms
    )
    if not common:
        return None
    timestamp = common[0]
    return timestamp, side_1[timestamp], side_2[timestamp]


def _no_vig_probability(decimal_1: float, decimal_2: float) -> float:
    implied_1 = 1.0 / decimal_1
    implied_2 = 1.0 / decimal_2
    return implied_1 / (implied_1 + implied_2)


def derive_horizon_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT m.matchup_id, m.fighter_1_name, m.fighter_2_name,
               m.mapping_status, m.ufc_event_date, m.ufc_event_id,
               m.ufc_fight_id, m.ufc_fighter_1_id, m.ufc_fighter_2_id,
               m.source_to_ufc_date_offset_days,
               e.source_event_date, e.event_url, e.title,
               b.book_key, b.book_id, b.book_name
        FROM matchups AS m
        JOIN events AS e ON e.event_url = m.event_url
        JOIN matchup_books AS b ON b.matchup_id = m.matchup_id
        WHERE m.ufc_fight_id IS NOT NULL
        ORDER BY m.ufc_fight_id, m.matchup_id, b.book_key
        """
    ).fetchall()
    output: list[dict[str, Any]] = []
    for row in rows:
        quotes = connection.execute(
            """
            SELECT side, observed_at_ms, decimal_odds
            FROM quotes
            WHERE matchup_id=? AND book_key=?
            ORDER BY observed_at_ms
            """,
            (row["matchup_id"], row["book_key"]),
        ).fetchall()
        side_1 = {
            int(item["observed_at_ms"]): float(item["decimal_odds"])
            for item in quotes
            if item["side"] == 1
        }
        side_2 = {
            int(item["observed_at_ms"]): float(item["decimal_odds"])
            for item in quotes
            if item["side"] == 2
        }
        event_day = date.fromisoformat(str(row["source_event_date"]))
        event_day_utc = datetime.combine(
            event_day, datetime.min.time(), tzinfo=timezone.utc
        )
        strict_cutoff_ms = int(event_day_utc.timestamp() * 1000) - 1
        for horizon, hours in HORIZONS:
            if horizon == "opening":
                selected = _first_common_before(side_1, side_2, strict_cutoff_ms + 1)
                cutoff_text = event_day_utc.isoformat().replace("+00:00", "Z")
            else:
                cutoff = event_day_utc - timedelta(hours=int(hours or 0))
                cutoff_ms = int(cutoff.timestamp() * 1000)
                if horizon == "strict_latest_before_event_date":
                    cutoff_ms -= 1
                selected = _latest_common_at_or_before(side_1, side_2, cutoff_ms)
                cutoff_text = cutoff.isoformat().replace("+00:00", "Z")
            if selected is None:
                continue
            observed, decimal_1, decimal_2 = selected
            output.append(
                {
                    "ufc_event_date": row["ufc_event_date"],
                    "ufc_event_id": row["ufc_event_id"],
                    "ufc_fight_id": row["ufc_fight_id"],
                    "ufc_fighter_1_id": row["ufc_fighter_1_id"],
                    "ufc_fighter_2_id": row["ufc_fighter_2_id"],
                    "fighter_1_name": row["fighter_1_name"],
                    "fighter_2_name": row["fighter_2_name"],
                    "source_event_date": row["source_event_date"],
                    "source_event_url": row["event_url"],
                    "source_event_title": row["title"],
                    "source_matchup_id": row["matchup_id"],
                    "mapping_status": row["mapping_status"],
                    "source_to_ufc_date_offset_days": row[
                        "source_to_ufc_date_offset_days"
                    ],
                    "book_key": row["book_key"],
                    "book_id": row["book_id"],
                    "book_name": row["book_name"],
                    "book_kind": "mean" if row["book_key"] == "mean" else "book",
                    "horizon": horizon,
                    "cutoff_utc": cutoff_text,
                    "cutoff_basis": "source_event_calendar_date_at_00_utc",
                    "actual_event_start_time_known": False,
                    "observed_at_utc": _epoch_ms_text(observed),
                    "fighter_1_decimal_odds": decimal_1,
                    "fighter_2_decimal_odds": decimal_2,
                    "fighter_1_no_vig_probability": _no_vig_probability(
                        decimal_1, decimal_2
                    ),
                }
            )
    return output


def derive_consensus_rows(
    horizon_rows: Sequence[Mapping[str, Any]], *, minimum_books: int
) -> list[dict[str, Any]]:
    source_matchups_by_fight: dict[str, set[int]] = defaultdict(set)
    for row in horizon_rows:
        source_matchups_by_fight[str(row["ufc_fight_id"])].add(
            int(row["source_matchup_id"])
        )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in horizon_rows:
        fight_id = str(row["ufc_fight_id"])
        if row["book_kind"] != "book" or len(source_matchups_by_fight[fight_id]) != 1:
            continue
        grouped[(fight_id, str(row["horizon"]))].append(row)
    output: list[dict[str, Any]] = []
    for (_, horizon), values in sorted(grouped.items()):
        by_book: dict[str, Mapping[str, Any]] = {}
        for value in values:
            key = str(value["book_name"]).casefold()
            current = by_book.get(key)
            if current is None or str(value["observed_at_utc"]) > str(
                current["observed_at_utc"]
            ):
                by_book[key] = value
        books = list(by_book.values())
        if len(books) < minimum_books:
            continue
        base = books[0]
        probabilities = [
            float(item["fighter_1_no_vig_probability"]) for item in books
        ]
        output.append(
            {
                "ufc_event_date": base["ufc_event_date"],
                "ufc_event_id": base["ufc_event_id"],
                "ufc_fight_id": base["ufc_fight_id"],
                "ufc_fighter_1_id": base["ufc_fighter_1_id"],
                "ufc_fighter_2_id": base["ufc_fighter_2_id"],
                "fighter_1_name": base["fighter_1_name"],
                "fighter_2_name": base["fighter_2_name"],
                "horizon": horizon,
                "cutoff_utc": base["cutoff_utc"],
                "cutoff_basis": base["cutoff_basis"],
                "actual_event_start_time_known": False,
                "book_count": len(books),
                "books": "|".join(sorted(str(item["book_name"]) for item in books)),
                "fighter_1_market_probability": sum(probabilities)
                / len(probabilities),
                "minimum_book_probability": min(probabilities),
                "maximum_book_probability": max(probabilities),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def database_summary(
    connection: sqlite3.Connection,
    *,
    database_path: Path,
    session_requests: int = 0,
    session_response_bytes: int = 0,
    maximum_source_attempts: int = DEFAULT_MAX_SOURCE_ATTEMPTS,
) -> dict[str, Any]:
    scalar = lambda sql: int(connection.execute(sql).fetchone()[0])
    by_status = {
        row["status"]: int(row["count"])
        for row in connection.execute(
            "SELECT status, COUNT(*) AS count FROM downloads GROUP BY status"
        )
    }
    mappings = {
        row["mapping_status"]: int(row["count"])
        for row in connection.execute(
            "SELECT mapping_status, COUNT(*) AS count FROM matchups GROUP BY mapping_status"
        )
    }
    return {
        "generated_at_utc": _utc_text(),
        "database": str(database_path),
        "database_mib": _database_size_mib(database_path),
        "events": scalar("SELECT COUNT(*) FROM events"),
        "parsed_events": scalar(
            "SELECT COUNT(*) FROM events WHERE page_status='parsed'"
        ),
        "failed_events": scalar(
            "SELECT COUNT(*) FROM events WHERE page_status='failed'"
        ),
        "matchups": scalar("SELECT COUNT(*) FROM matchups"),
        "mapped_matchups": scalar(
            "SELECT COUNT(*) FROM matchups WHERE ufc_fight_id IS NOT NULL"
        ),
        "mapping_statuses": mappings,
        "downloads_by_status": by_status,
        "source_failures_at_retry_cap": {
            "event_pages": int(
                connection.execute(
                    "SELECT COUNT(*) FROM events "
                    "WHERE page_status='failed' AND page_attempts>=?",
                    (maximum_source_attempts,),
                ).fetchone()[0]
            ),
            "chart_series": int(
                connection.execute(
                    "SELECT COUNT(*) FROM downloads "
                    "WHERE status='failed' AND attempts>=?",
                    (maximum_source_attempts,),
                ).fetchone()[0]
            ),
            "retry_cap": maximum_source_attempts,
        },
        "quote_points": scalar("SELECT COUNT(*) FROM quotes"),
        "session_requests": session_requests,
        "session_response_bytes": session_response_bytes,
    }


def export_database(
    connection: sqlite3.Connection,
    *,
    database_path: Path,
    minimum_consensus_books: int,
    session_requests: int = 0,
    session_response_bytes: int = 0,
    maximum_source_attempts: int = DEFAULT_MAX_SOURCE_ATTEMPTS,
) -> dict[str, Any]:
    export_dir = database_path.parent / "exports"
    horizons = derive_horizon_rows(connection)
    consensus = derive_consensus_rows(
        horizons, minimum_books=minimum_consensus_books
    )
    unmatched = [
        dict(row)
        for row in connection.execute(
            """
            SELECT e.source_event_date, e.event_url, e.title,
                   m.matchup_id, m.fighter_1_name, m.fighter_2_name,
                   m.mapping_status
            FROM matchups AS m
            JOIN events AS e ON e.event_url=m.event_url
            WHERE m.ufc_fight_id IS NULL
            ORDER BY e.source_event_date, m.matchup_id
            """
        )
    ]
    duplicate_mappings = [
        dict(row)
        for row in connection.execute(
            """
            SELECT ufc_fight_id, COUNT(*) AS source_matchup_count,
                   GROUP_CONCAT(matchup_id) AS source_matchup_ids
            FROM matchups
            WHERE ufc_fight_id IS NOT NULL
            GROUP BY ufc_fight_id
            HAVING COUNT(*) > 1
            ORDER BY ufc_fight_id
            """
        )
    ]
    _write_csv(export_dir / "horizon_quotes.csv", horizons)
    _write_csv(export_dir / "market_consensus.csv", consensus)
    _write_csv(export_dir / "unmatched_matchups.csv", unmatched)
    _write_csv(export_dir / "duplicate_source_matchups.csv", duplicate_mappings)
    summary = database_summary(
        connection,
        database_path=database_path,
        session_requests=session_requests,
        session_response_bytes=session_response_bytes,
        maximum_source_attempts=maximum_source_attempts,
    )
    summary["exports"] = {
        "horizon_quote_rows": len(horizons),
        "consensus_rows": len(consensus),
        "unmatched_rows": len(unmatched),
        "duplicate_source_mapping_rows": len(duplicate_mappings),
        "directory": str(export_dir),
    }
    (export_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _print_summary(summary: Mapping[str, Any]) -> None:
    print(f"Database: {summary['database']}")
    print(
        f"Events: {summary['parsed_events']} parsed, {summary['failed_events']} failed; "
        f"matchups: {summary['mapped_matchups']}/{summary['matchups']} matched to UFCStats"
    )
    print(
        f"Quotes: {summary['quote_points']:,}; database: {summary['database_mib']:.2f} MiB; "
        f"session requests: {summary['session_requests']:,}"
    )
    failures = summary.get("source_failures_at_retry_cap", {})
    if failures and (failures.get("event_pages") or failures.get("chart_series")):
        print(
            "Unavailable after retry cap: "
            f"{failures.get('event_pages', 0):,} event pages, "
            f"{failures.get('chart_series', 0):,} chart series "
            f"({failures.get('retry_cap')} attempts each)."
        )
    if "exports" in summary:
        exports = summary["exports"]
        print(
            f"Exports: {exports['horizon_quote_rows']:,} horizon rows, "
            f"{exports['consensus_rows']:,} multi-book consensus rows"
        )
        print(f"Export directory: {exports['directory']}")


def _event_already_parsed(connection: sqlite3.Connection, event_url: str) -> bool:
    row = connection.execute(
        "SELECT page_status FROM events WHERE event_url=?", (event_url,)
    ).fetchone()
    return row is not None and row["page_status"] == "parsed"


def _event_page_needs_work(
    connection: sqlite3.Connection,
    *,
    event_url: str,
    maximum_attempts: int,
) -> bool:
    row = connection.execute(
        "SELECT page_status, page_attempts FROM events WHERE event_url=?",
        (event_url,),
    ).fetchone()
    if row is None:
        return True
    if row["page_status"] == "parsed":
        return False
    return not (
        row["page_status"] == "failed"
        and int(row["page_attempts"] or 0) >= maximum_attempts
    )


def _event_has_pending_downloads(
    connection: sqlite3.Connection,
    *,
    event_url: str,
    mode: str,
    maximum_attempts: int = DEFAULT_MAX_SOURCE_ATTEMPTS,
) -> bool:
    return bool(
        pending_downloads_for_event(
            connection,
            event_url=event_url,
            mode=mode,
            maximum_attempts=maximum_attempts,
        )
    )


def run_backfill(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    repository_root = Path(__file__).resolve().parents[1]
    database_path = validate_external_database_path(args.database, repository_root)
    connection = open_database(database_path, mode=args.mode)
    fight_index = UFCFightIndex.from_csv(args.raw_fights)
    remapped = refresh_fight_mappings(connection, fight_index=fight_index)
    if remapped:
        print(f"Refreshed {remapped} existing UFCStats matchup mappings.", flush=True)
    fetcher = BackfillFetcher(
        max_runtime_hours=args.max_runtime_hours,
        max_requests=args.max_requests,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    config = {
        "from_year": args.from_year,
        "to_year": args.to_year,
        "mode": args.mode,
        "max_events": args.max_events,
        "max_requests": args.max_requests,
        "max_runtime_hours": args.max_runtime_hours,
        "delay_seconds": args.delay_seconds,
        "maximum_database_mib": args.max_database_mib,
        "minimum_free_gib": args.minimum_free_gib,
        "maximum_source_attempts": args.max_source_attempts,
    }
    cursor = connection.execute(
        "INSERT INTO runs(started_at_utc, status, config_json) VALUES (?, 'running', ?)",
        (_utc_text(), _canonical_json(config)),
    )
    run_id = int(cursor.lastrowid)
    connection.commit()
    status = "complete"
    message = "selected events completed"
    events_processed = 0
    try:
        check_storage_budget(
            database_path,
            max_database_mib=args.max_database_mib,
            minimum_free_gib=args.minimum_free_gib,
        )
        robots = fetcher.get(f"{BESTFIGHTODDS_ROOT}/robots.txt").text
        if not _robots_allows_public_paths(robots):
            raise BackfillError("source robots policy no longer allows public paths")
        terms = fetcher.get(f"{BESTFIGHTODDS_ROOT}/terms").text
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('source_policy_acknowledged_at_utc', ?)",
            (_utc_text(),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('source_terms_observed_bytes', ?)",
            (str(len(terms.encode("utf-8"))),),
        )
        sitemap_response = fetcher.get(f"{BESTFIGHTODDS_ROOT}/sitemap-events.xml")
        selected = [
            item
            for item in parse_bestfightodds_sitemap(sitemap_response.text)
            if args.from_year <= item.event_date.year <= args.to_year
            and item.event_date <= date.today()
        ]
        selected.sort(
            key=lambda item: (item.event_date, item.url),
            reverse=args.order == "newest",
        )
        work = [
            item
            for item in selected
            if _event_page_needs_work(
                connection,
                event_url=item.url,
                maximum_attempts=args.max_source_attempts,
            )
            or _event_has_pending_downloads(
                connection,
                event_url=item.url,
                mode=args.mode,
                maximum_attempts=args.max_source_attempts,
            )
        ]
        if args.max_events:
            work = work[: args.max_events]
        print(
            f"Eligible events: {len(selected)}; events needing work this session: {len(work)}",
            flush=True,
        )
        for position, event in enumerate(work, start=1):
            check_storage_budget(
                database_path,
                max_database_mib=args.max_database_mib,
                minimum_free_gib=args.minimum_free_gib,
            )
            if _event_page_needs_work(
                connection,
                event_url=event.url,
                maximum_attempts=args.max_source_attempts,
            ):
                try:
                    response = fetcher.get(event.url)
                    page = parse_bestfightodds_event_page(
                        response.text, url=event.url, sitemap_date=event.event_date
                    )
                    _upsert_event_page(
                        connection, page=page, fight_index=fight_index
                    )
                except BudgetReached:
                    raise
                except (requests.RequestException, ValueError) as error:
                    _record_event_failure(
                        connection,
                        event_url=event.url,
                        event_date=event.event_date.isoformat(),
                        error=error,
                    )
                    print(
                        f"Event {position}/{len(work)} page failed: {_plain_error(error)}",
                        flush=True,
                    )
                    continue
            specs = pending_downloads_for_event(
                connection,
                event_url=event.url,
                mode=args.mode,
                maximum_attempts=args.max_source_attempts,
            )
            for spec in specs:
                try:
                    response = fetcher.get(spec.endpoint, referer=spec.event_url)
                    store_download(
                        connection,
                        spec=spec,
                        series=decode_bestfightodds_chart(response.text),
                    )
                except BudgetReached:
                    raise
                except (requests.RequestException, ValueError) as error:
                    store_download_failure(connection, spec=spec, error=error)
            events_processed += 1
            print(
                f"Processed event {position}/{len(work)}: {event.event_date} "
                f"({len(specs)} series); session requests {fetcher.requests}",
                flush=True,
            )
        remaining = sum(
            _event_has_pending_downloads(
                connection,
                event_url=item.url,
                mode=args.mode,
                maximum_attempts=args.max_source_attempts,
            )
            or _event_page_needs_work(
                connection,
                event_url=item.url,
                maximum_attempts=args.max_source_attempts,
            )
            for item in selected
        )
        if remaining:
            status = "paused"
            message = f"{remaining} eligible events still need work"
    except BudgetReached as error:
        status = "paused"
        message = str(error)
        print(f"Paused cleanly: {message}", flush=True)
    except Exception as error:
        status = "failed"
        message = _plain_error(error)
        raise
    finally:
        connection.execute(
            """
            UPDATE runs SET finished_at_utc=?, status=?, requests=?,
                response_bytes=?, events_processed=?, message=?
            WHERE run_id=?
            """,
            (
                _utc_text(),
                status,
                fetcher.requests,
                fetcher.response_bytes,
                events_processed,
                message,
                run_id,
            ),
        )
        connection.commit()
    summary = export_database(
        connection,
        database_path=database_path,
        minimum_consensus_books=args.minimum_consensus_books,
        session_requests=fetcher.requests,
        session_response_bytes=fetcher.response_bytes,
        maximum_source_attempts=args.max_source_attempts,
    )
    connection.close()
    return status, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=default_database_path())
    parser.add_argument("--raw-fights", type=Path, default=DEFAULT_RAW_FIGHTS)
    parser.add_argument("--from-year", type=int, default=DEFAULT_FROM_YEAR)
    parser.add_argument("--to-year", type=int, default=DEFAULT_TO_YEAR)
    parser.add_argument("--mode", choices=("mean", "books", "both"), default=DEFAULT_MODE)
    parser.add_argument("--order", choices=("oldest", "newest"), default="oldest")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument(
        "--max-source-attempts",
        type=int,
        default=DEFAULT_MAX_SOURCE_ATTEMPTS,
        help=(
            "stop retrying an unavailable event page or chart series after this "
            "many failed attempts (default: 2)"
        ),
    )
    parser.add_argument(
        "--max-runtime-hours", type=float, default=DEFAULT_MAX_RUNTIME_HOURS
    )
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--max-database-mib", type=float, default=DEFAULT_MAX_DATABASE_MIB
    )
    parser.add_argument(
        "--minimum-free-gib", type=float, default=DEFAULT_MINIMUM_FREE_GIB
    )
    parser.add_argument(
        "--minimum-consensus-books",
        type=int,
        default=DEFAULT_MINIMUM_CONSENSUS_BOOKS,
    )
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--acknowledge-source-policy", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.from_year < 2007 or args.to_year < args.from_year:
        raise BackfillError("year range must be valid and start in 2007 or later")
    if args.max_events < 0:
        raise BackfillError("max events cannot be negative")
    if not 1 <= args.max_source_attempts <= 10:
        raise BackfillError("maximum source attempts must be within [1, 10]")
    if not 3 <= args.minimum_consensus_books <= 20:
        raise BackfillError("minimum consensus books must be within [3, 20]")
    if args.max_database_mib <= 0 or args.minimum_free_gib < 0:
        raise BackfillError("storage limits must be nonnegative")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        repository_root = Path(__file__).resolve().parents[1]
        database_path = validate_external_database_path(args.database, repository_root)
        if args.status_only or args.export_only:
            if not database_path.is_file():
                raise BackfillError(f"database does not exist: {database_path}")
            connection = open_database_readonly(database_path, mode=args.mode)
            if args.export_only:
                summary = export_database(
                    connection,
                    database_path=database_path,
                    minimum_consensus_books=args.minimum_consensus_books,
                )
            else:
                summary = database_summary(
                    connection, database_path=database_path
                )
            connection.close()
            _print_summary(summary)
            return 0
        if not args.acknowledge_source_policy:
            raise BackfillError(
                "network collection requires --acknowledge-source-policy; "
                "the source allows public paths in robots.txt but its terms do not "
                "explicitly address automated bulk research reuse"
            )
        status, summary = run_backfill(args)
        _print_summary(summary)
        print(f"Session status: {status}")
        if status == "paused":
            print("Run the identical command again to continue from the checkpoint.")
        return 0 if status in {"complete", "paused"} else 1
    except (BackfillError, OSError, sqlite3.Error) as error:
        print(f"historical odds backfill: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
