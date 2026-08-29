"""Resumable BestFightOdds method-of-victory history collector.

This is intentionally separate from the winner-price database.  It reuses the
completed event and UFCStats identity mappings from that database, fetches only
the six primary fighter-by-method histories, and stores compact decoded points
outside Git.  A default run refuses to start while the winner backfill reports
an active session so two bulk crawls do not run against the source together.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping, Sequence

from audit_historical_odds_sources import (
    BESTFIGHTODDS_ROOT,
    decode_bestfightodds_chart,
)
from backfill_bestfightodds_history import (
    BackfillError,
    BackfillFetcher,
    BudgetReached,
    _database_size_mib,
    _flatten_chart_points,
    _plain_error,
    _robots_allows_public_paths,
    _utc_text,
    check_storage_budget,
    default_database_path as default_winner_database_path,
    validate_external_database_path,
)
from bestfightodds_props import (
    METHODS,
    MethodPropSelection,
    PropParseError,
    parse_bestfightodds_method_props,
)


SCHEMA_VERSION = 1
COLLECTOR_VERSION = 1
DEFAULT_FROM_YEAR = 2021
DEFAULT_TO_YEAR = date.today().year
DEFAULT_MODE = "mean"
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_MAX_RUNTIME_HOURS = 6.0
DEFAULT_MAX_REQUESTS = 25_000
DEFAULT_MAX_DATABASE_MIB = 1024.0
DEFAULT_MINIMUM_FREE_GIB = 5.0
DEFAULT_MAX_QUOTE_SKEW_SECONDS = 600

HORIZONS = (
    ("opening", None),
    ("safe_t72", 72),
    ("safe_t24", 24),
    ("safe_t6", 6),
    ("strict_latest_before_event_date", 0),
)


def default_database_path() -> Path:
    return default_winner_database_path().with_name("method_history.sqlite3")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_url TEXT PRIMARY KEY,
    source_event_date TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    html_sha256 TEXT NOT NULL DEFAULT '',
    page_status TEXT NOT NULL DEFAULT 'pending',
    page_attempts INTEGER NOT NULL DEFAULT 0,
    selection_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS selections (
    selection_id TEXT PRIMARY KEY,
    event_url TEXT NOT NULL REFERENCES events(event_url),
    source_matchup_id INTEGER NOT NULL,
    source_fighter_side INTEGER NOT NULL CHECK(source_fighter_side IN (1, 2)),
    source_prop_type_id INTEGER NOT NULL,
    source_outcome_number INTEGER NOT NULL,
    market TEXT NOT NULL,
    method TEXT NOT NULL,
    raw_label TEXT NOT NULL,
    fighter_1_name TEXT NOT NULL,
    fighter_2_name TEXT NOT NULL,
    ufc_event_date TEXT NOT NULL,
    ufc_event_id TEXT NOT NULL,
    ufc_fight_id TEXT NOT NULL,
    ufc_fighter_1_id TEXT NOT NULL,
    ufc_fighter_2_id TEXT NOT NULL,
    selected_fighter_id TEXT NOT NULL,
    mean_history_available INTEGER NOT NULL CHECK(mean_history_available IN (0, 1)),
    updated_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS selections_event_idx ON selections(event_url);
CREATE INDEX IF NOT EXISTS selections_fight_idx ON selections(ufc_fight_id);
CREATE TABLE IF NOT EXISTS selection_books (
    selection_id TEXT NOT NULL REFERENCES selections(selection_id),
    book_key TEXT NOT NULL,
    book_id INTEGER,
    book_name TEXT NOT NULL,
    PRIMARY KEY(selection_id, book_key)
);
CREATE TABLE IF NOT EXISTS downloads (
    selection_id TEXT NOT NULL REFERENCES selections(selection_id),
    book_key TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    status TEXT NOT NULL,
    point_count INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY(selection_id, book_key)
);
CREATE TABLE IF NOT EXISTS quotes (
    selection_id TEXT NOT NULL REFERENCES selections(selection_id),
    book_key TEXT NOT NULL,
    observed_at_ms INTEGER NOT NULL,
    decimal_odds REAL NOT NULL CHECK(decimal_odds > 1.0),
    series_name TEXT NOT NULL,
    PRIMARY KEY(selection_id, book_key, observed_at_ms)
);
CREATE INDEX IF NOT EXISTS method_quotes_lookup_idx
    ON quotes(selection_id, book_key, observed_at_ms);
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


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _open_database(path: Path, *, mode: str) -> sqlite3.Connection:
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
                f"method database {key} is {existing[key]!r}, expected {value!r}; "
                "use a new database path"
            )
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
            (key, value),
        )
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('created_at_utc', ?)",
        (_utc_text(),),
    )
    connection.commit()
    return connection


def _open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _winner_run_active(connection: sqlite3.Connection) -> bool:
    return bool(
        connection.execute(
            "SELECT 1 FROM runs WHERE status='running' AND finished_at_utc IS NULL LIMIT 1"
        ).fetchone()
    )


def _eligible_winner_events(
    connection: sqlite3.Connection,
    *,
    from_year: int,
    to_year: int,
    order: str,
) -> list[sqlite3.Row]:
    direction = "ASC" if order == "oldest" else "DESC"
    return connection.execute(
        f"""
        SELECT event_url, source_event_date, title
        FROM events
        WHERE page_status='parsed'
          AND CAST(substr(source_event_date, 1, 4) AS INTEGER) BETWEEN ? AND ?
        ORDER BY source_event_date {direction}, event_url {direction}
        """,
        (from_year, to_year),
    ).fetchall()


def _winner_matchups_for_event(
    connection: sqlite3.Connection, event_url: str
) -> dict[int, sqlite3.Row]:
    return {
        int(row["matchup_id"]): row
        for row in connection.execute(
            """
            SELECT matchup_id, fighter_1_name, fighter_2_name,
                   mapping_status, ufc_event_date, ufc_event_id, ufc_fight_id,
                   ufc_fighter_1_id, ufc_fighter_2_id
            FROM matchups WHERE event_url=?
            """,
            (event_url,),
        )
        if row["ufc_fight_id"] is not None
        and str(row["mapping_status"]).startswith(("exact_", "unique_"))
    }


def _upsert_event(
    connection: sqlite3.Connection,
    *,
    event: Mapping[str, object],
    html: str,
    selections: Sequence[MethodPropSelection],
    winner_matchups: Mapping[int, sqlite3.Row],
) -> int:
    now = _utc_text()
    event_url = str(event["event_url"])
    accepted = [
        selection
        for selection in selections
        if selection.source_matchup_id in winner_matchups
    ]
    with connection:
        connection.execute(
            """
            INSERT INTO events(
                event_url, source_event_date, title, html_sha256,
                page_status, page_attempts, selection_count, last_error,
                updated_at_utc
            ) VALUES (?, ?, ?, ?, 'parsed', 1, ?, NULL, ?)
            ON CONFLICT(event_url) DO UPDATE SET
                source_event_date=excluded.source_event_date,
                title=excluded.title,
                html_sha256=excluded.html_sha256,
                page_status='parsed',
                page_attempts=events.page_attempts + 1,
                selection_count=excluded.selection_count,
                last_error=NULL,
                updated_at_utc=excluded.updated_at_utc
            """,
            (
                event_url,
                event["source_event_date"],
                event["title"],
                hashlib.sha256(html.encode("utf-8")).hexdigest(),
                len(accepted),
                now,
            ),
        )
        for selection in accepted:
            matchup = winner_matchups[selection.source_matchup_id]
            selected_id = matchup[f"ufc_fighter_{selection.source_fighter_side}_id"]
            connection.execute(
                """
                INSERT INTO selections(
                    selection_id, event_url, source_matchup_id,
                    source_fighter_side, source_prop_type_id,
                    source_outcome_number, market, method, raw_label,
                    fighter_1_name, fighter_2_name, ufc_event_date,
                    ufc_event_id, ufc_fight_id, ufc_fighter_1_id,
                    ufc_fighter_2_id, selected_fighter_id,
                    mean_history_available, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(selection_id) DO UPDATE SET
                    raw_label=excluded.raw_label,
                    fighter_1_name=excluded.fighter_1_name,
                    fighter_2_name=excluded.fighter_2_name,
                    ufc_event_date=excluded.ufc_event_date,
                    ufc_event_id=excluded.ufc_event_id,
                    ufc_fight_id=excluded.ufc_fight_id,
                    ufc_fighter_1_id=excluded.ufc_fighter_1_id,
                    ufc_fighter_2_id=excluded.ufc_fighter_2_id,
                    selected_fighter_id=excluded.selected_fighter_id,
                    mean_history_available=excluded.mean_history_available,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (
                    selection.source_selection_id,
                    event_url,
                    selection.source_matchup_id,
                    selection.source_fighter_side,
                    selection.source_prop_type_id,
                    selection.source_outcome_number,
                    selection.market,
                    selection.method,
                    selection.raw_label,
                    selection.fighter_1_name,
                    selection.fighter_2_name,
                    matchup["ufc_event_date"],
                    matchup["ufc_event_id"],
                    matchup["ufc_fight_id"],
                    matchup["ufc_fighter_1_id"],
                    matchup["ufc_fighter_2_id"],
                    selected_id,
                    int(selection.mean_history_available),
                    now,
                ),
            )
            if selection.mean_history_available:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO selection_books(
                        selection_id, book_key, book_id, book_name
                    ) VALUES (?, 'mean', NULL, 'BestFightOdds mean')
                    """,
                    (selection.source_selection_id,),
                )
            for price in selection.book_prices:
                connection.execute(
                    """
                    INSERT INTO selection_books(
                        selection_id, book_key, book_id, book_name
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(selection_id, book_key) DO UPDATE SET
                        book_name=excluded.book_name
                    """,
                    (
                        selection.source_selection_id,
                        f"book:{price.book_id}",
                        price.book_id,
                        price.book_name,
                    ),
                )
    return len(accepted)


def _record_event_failure(
    connection: sqlite3.Connection,
    *,
    event: Mapping[str, object],
    error: Exception,
) -> None:
    connection.execute(
        """
        INSERT INTO events(
            event_url, source_event_date, title, page_status,
            page_attempts, last_error, updated_at_utc
        ) VALUES (?, ?, ?, 'failed', 1, ?, ?)
        ON CONFLICT(event_url) DO UPDATE SET
            page_status='failed', page_attempts=events.page_attempts + 1,
            last_error=excluded.last_error, updated_at_utc=excluded.updated_at_utc
        """,
        (
            event["event_url"],
            event["source_event_date"],
            event["title"],
            _plain_error(error),
            _utc_text(),
        ),
    )
    connection.commit()


@dataclass(frozen=True)
class DownloadSpec:
    selection_id: str
    event_url: str
    matchup_id: int
    fighter_side: int
    prop_type_id: int
    outcome_number: int
    book_key: str
    book_id: int | None
    book_name: str

    @property
    def endpoint(self) -> str:
        book = f"b={self.book_id}&" if self.book_id is not None else ""
        return (
            f"{BESTFIGHTODDS_ROOT}/api/ggd?{book}m={self.matchup_id}"
            f"&p={self.fighter_side}&pt={self.prop_type_id}"
            f"&tn={self.outcome_number}"
        )


def _pending_downloads(
    connection: sqlite3.Connection, *, event_url: str, mode: str
) -> list[DownloadSpec]:
    rows = connection.execute(
        """
        SELECT s.selection_id, s.event_url, s.source_matchup_id,
               s.source_fighter_side, s.source_prop_type_id,
               s.source_outcome_number, b.book_key, b.book_id, b.book_name,
               d.status
        FROM selections AS s
        JOIN selection_books AS b ON b.selection_id=s.selection_id
        LEFT JOIN downloads AS d
          ON d.selection_id=s.selection_id AND d.book_key=b.book_key
        WHERE s.event_url=?
        ORDER BY s.source_matchup_id, s.source_fighter_side,
                 s.method, b.book_key
        """,
        (event_url,),
    ).fetchall()
    output: list[DownloadSpec] = []
    for row in rows:
        is_mean = row["book_key"] == "mean"
        if (is_mean and mode not in {"mean", "both"}) or (
            not is_mean and mode not in {"books", "both"}
        ):
            continue
        if row["status"] in {"complete", "empty"}:
            continue
        output.append(
            DownloadSpec(
                selection_id=row["selection_id"],
                event_url=row["event_url"],
                matchup_id=int(row["source_matchup_id"]),
                fighter_side=int(row["source_fighter_side"]),
                prop_type_id=int(row["source_prop_type_id"]),
                outcome_number=int(row["source_outcome_number"]),
                book_key=row["book_key"],
                book_id=(int(row["book_id"]) if row["book_id"] is not None else None),
                book_name=row["book_name"],
            )
        )
    return output


def _store_download(
    connection: sqlite3.Connection,
    *,
    spec: DownloadSpec,
    series: Sequence[Mapping[str, Any]],
) -> int:
    points = _flatten_chart_points(series)
    status = "complete" if points else "empty"
    with connection:
        connection.execute(
            "DELETE FROM quotes WHERE selection_id=? AND book_key=?",
            (spec.selection_id, spec.book_key),
        )
        connection.executemany(
            """
            INSERT INTO quotes(
                selection_id, book_key, observed_at_ms, decimal_odds, series_name
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (spec.selection_id, spec.book_key, observed, price, name)
                for observed, price, name in points
            ],
        )
        connection.execute(
            """
            INSERT INTO downloads(
                selection_id, book_key, endpoint, status, point_count,
                attempts, last_error, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, 1, NULL, ?)
            ON CONFLICT(selection_id, book_key) DO UPDATE SET
                endpoint=excluded.endpoint, status=excluded.status,
                point_count=excluded.point_count,
                attempts=downloads.attempts + 1, last_error=NULL,
                updated_at_utc=excluded.updated_at_utc
            """,
            (
                spec.selection_id,
                spec.book_key,
                spec.endpoint,
                status,
                len(points),
                _utc_text(),
            ),
        )
    return len(points)


def _store_download_failure(
    connection: sqlite3.Connection, *, spec: DownloadSpec, error: Exception
) -> None:
    connection.execute(
        """
        INSERT INTO downloads(
            selection_id, book_key, endpoint, status, point_count,
            attempts, last_error, updated_at_utc
        ) VALUES (?, ?, ?, 'failed', 0, 1, ?, ?)
        ON CONFLICT(selection_id, book_key) DO UPDATE SET
            endpoint=excluded.endpoint, status='failed',
            attempts=downloads.attempts + 1, last_error=excluded.last_error,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            spec.selection_id,
            spec.book_key,
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


def _horizon_rows(connection: sqlite3.Connection) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    selections = connection.execute(
        """
        SELECT s.*, b.book_key, b.book_name,
               q.observed_at_ms, q.decimal_odds
        FROM selections AS s
        JOIN selection_books AS b ON b.selection_id=s.selection_id
        JOIN quotes AS q
          ON q.selection_id=s.selection_id AND q.book_key=b.book_key
        ORDER BY s.ufc_event_date, s.ufc_fight_id, b.book_key,
                 s.source_fighter_side, s.method, q.observed_at_ms
        """
    ).fetchall()
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in selections:
        grouped[(row["selection_id"], row["book_key"])].append(row)
    for (_, _), rows in grouped.items():
        base = rows[0]
        event_day = date.fromisoformat(base["ufc_event_date"])
        event_cutoff = datetime.combine(
            event_day, datetime.min.time(), tzinfo=timezone.utc
        )
        event_cutoff_ms = int(event_cutoff.timestamp() * 1000)
        pre_event = [row for row in rows if row["observed_at_ms"] < event_cutoff_ms]
        if not pre_event:
            continue
        for horizon, hours in HORIZONS:
            if hours is None:
                picked = pre_event[0]
                cutoff = event_cutoff
                basis = "first_strictly_before_source_event_calendar_date"
            else:
                cutoff = event_cutoff - timedelta(hours=hours)
                eligible = [
                    row
                    for row in pre_event
                    if row["observed_at_ms"] <= int(cutoff.timestamp() * 1000)
                ]
                if not eligible:
                    continue
                picked = eligible[-1]
                basis = f"source_event_calendar_date_00_utc_minus_{hours}_hours"
            decimal_odds = float(picked["decimal_odds"])
            values.append(
                {
                    "ufc_event_date": base["ufc_event_date"],
                    "ufc_event_id": base["ufc_event_id"],
                    "ufc_fight_id": base["ufc_fight_id"],
                    "source_matchup_id": base["source_matchup_id"],
                    "fighter_1_name": base["fighter_1_name"],
                    "fighter_2_name": base["fighter_2_name"],
                    "ufc_fighter_1_id": base["ufc_fighter_1_id"],
                    "ufc_fighter_2_id": base["ufc_fighter_2_id"],
                    "selected_fighter_id": base["selected_fighter_id"],
                    "method": base["method"],
                    "book_key": base["book_key"],
                    "book_name": base["book_name"],
                    "horizon": horizon,
                    "cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
                    "cutoff_basis": basis,
                    "actual_event_start_time_known": False,
                    "observed_at_utc": _epoch_ms_text(picked["observed_at_ms"]),
                    "decimal_odds": decimal_odds,
                    "implied_probability": 1.0 / decimal_odds,
                }
            )
    return values


def _coherent_rows(
    horizon_rows: Sequence[Mapping[str, object]], *, max_skew_seconds: int
) -> list[dict[str, object]]:
    source_matchups: dict[str, set[int]] = defaultdict(set)
    for row in horizon_rows:
        source_matchups[str(row["ufc_fight_id"])].add(
            int(row["source_matchup_id"])
        )
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in horizon_rows:
        fight_id = str(row["ufc_fight_id"])
        if len(source_matchups[fight_id]) != 1:
            continue
        grouped[(fight_id, str(row["book_key"]), str(row["horizon"]))].append(row)
    expected = {
        (fighter_slot, method)
        for fighter_slot in ("fighter_1", "fighter_2")
        for method in METHODS
    }
    output: list[dict[str, object]] = []
    for (_, _, _), rows in sorted(grouped.items()):
        base = rows[0]
        side_by_id = {
            str(base["ufc_fighter_1_id"]): "fighter_1",
            str(base["ufc_fighter_2_id"]): "fighter_2",
        }
        by_selection: dict[tuple[str, str], Mapping[str, object]] = {}
        for row in rows:
            slot = side_by_id.get(str(row["selected_fighter_id"]))
            if slot is None:
                continue
            key = slot, str(row["method"])
            if key in by_selection:
                by_selection = {}
                break
            by_selection[key] = row
        if set(by_selection) != expected:
            continue
        timestamps = [
            datetime.fromisoformat(str(row["observed_at_utc"]).replace("Z", "+00:00"))
            for row in by_selection.values()
        ]
        skew = (max(timestamps) - min(timestamps)).total_seconds()
        if skew > max_skew_seconds:
            continue
        overround = sum(
            float(row["implied_probability"]) for row in by_selection.values()
        )
        if not 0.8 <= overround <= 2.0:
            continue
        market_id = hashlib.sha256(
            _canonical_json(
                {
                    "fight": base["ufc_fight_id"],
                    "book": base["book_key"],
                    "horizon": base["horizon"],
                    "quotes": {
                        f"{slot}_{method}": row["observed_at_utc"]
                        for (slot, method), row in sorted(by_selection.items())
                    },
                }
            ).encode("utf-8")
        ).hexdigest()
        for (slot, method), row in sorted(by_selection.items()):
            output.append(
                {
                    "method_market_id": market_id,
                    "ufc_event_date": base["ufc_event_date"],
                    "ufc_event_id": base["ufc_event_id"],
                    "ufc_fight_id": base["ufc_fight_id"],
                    "fighter_1_name": base["fighter_1_name"],
                    "fighter_2_name": base["fighter_2_name"],
                    "book_key": base["book_key"],
                    "book_name": base["book_name"],
                    "horizon": base["horizon"],
                    "selected_fighter": slot,
                    "selected_fighter_id": row["selected_fighter_id"],
                    "method": method,
                    "observed_at_utc": row["observed_at_utc"],
                    "decimal_odds": row["decimal_odds"],
                    "implied_probability": row["implied_probability"],
                    "six_way_overround": overround,
                    "no_vig_probability": float(row["implied_probability"])
                    / overround,
                    "quote_time_skew_seconds": skew,
                    "actual_event_start_time_known": False,
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def database_summary(
    connection: sqlite3.Connection,
    *,
    database_path: Path,
    session_requests: int = 0,
    session_response_bytes: int = 0,
) -> dict[str, object]:
    scalar = lambda sql: int(connection.execute(sql).fetchone()[0])
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
        "selections": scalar("SELECT COUNT(*) FROM selections"),
        "fights": scalar("SELECT COUNT(DISTINCT ufc_fight_id) FROM selections"),
        "quote_points": scalar("SELECT COUNT(*) FROM quotes"),
        "pending_downloads": scalar(
            """
            SELECT COUNT(*) FROM selection_books AS b
            LEFT JOIN downloads AS d
              ON d.selection_id=b.selection_id AND d.book_key=b.book_key
            WHERE d.status IS NULL OR d.status NOT IN ('complete', 'empty')
            """
        ),
        "session_requests": session_requests,
        "session_response_bytes": session_response_bytes,
    }


def export_database(
    connection: sqlite3.Connection,
    *,
    database_path: Path,
    max_quote_skew_seconds: int,
    session_requests: int = 0,
    session_response_bytes: int = 0,
) -> dict[str, object]:
    export_dir = database_path.parent / "method_exports"
    horizons = _horizon_rows(connection)
    coherent = _coherent_rows(horizons, max_skew_seconds=max_quote_skew_seconds)
    _write_csv(export_dir / "horizon_method_prices.csv", horizons)
    _write_csv(export_dir / "coherent_method_probabilities.csv", coherent)
    summary = database_summary(
        connection,
        database_path=database_path,
        session_requests=session_requests,
        session_response_bytes=session_response_bytes,
    )
    summary["exports"] = {
        "horizon_rows": len(horizons),
        "coherent_probability_rows": len(coherent),
        "coherent_markets": len(
            {row["method_market_id"] for row in coherent}
        ),
        "directory": str(export_dir),
    }
    (export_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _print_summary(summary: Mapping[str, object]) -> None:
    print(f"Method database: {summary['database']}")
    print(
        f"Events: {summary['parsed_events']} parsed, {summary['failed_events']} failed; "
        f"method selections: {summary['selections']} across {summary['fights']} fights"
    )
    print(
        f"Quotes: {int(summary['quote_points']):,}; "
        f"pending downloads: {int(summary['pending_downloads']):,}; "
        f"database: {float(summary['database_mib']):.2f} MiB; "
        f"session requests: {int(summary['session_requests']):,}"
    )
    if "exports" in summary:
        exports = summary["exports"]
        assert isinstance(exports, Mapping)
        print(
            f"Exports: {int(exports['horizon_rows']):,} price rows, "
            f"{int(exports['coherent_markets']):,} complete six-way markets"
        )
        print(f"Export directory: {exports['directory']}")


def run_backfill(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    repository_root = Path(__file__).resolve().parents[1]
    database_path = validate_external_database_path(args.database, repository_root)
    winner_path = validate_external_database_path(
        args.winner_database, repository_root
    )
    if database_path == winner_path:
        raise BackfillError("method and winner databases must use different paths")
    if not winner_path.is_file():
        raise BackfillError(f"winner history database does not exist: {winner_path}")
    winner = _open_readonly(winner_path)
    if _winner_run_active(winner) and not args.allow_concurrent_source_access:
        winner.close()
        raise BackfillError(
            "the winner-price backfill is still marked running; let it finish or pause "
            "before starting method history"
        )
    connection = _open_database(database_path, mode=args.mode)
    fetcher = BackfillFetcher(
        max_runtime_hours=args.max_runtime_hours,
        max_requests=args.max_requests,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    config = {
        "winner_database": str(winner_path),
        "from_year": args.from_year,
        "to_year": args.to_year,
        "mode": args.mode,
        "order": args.order,
        "max_events": args.max_events,
        "max_runtime_hours": args.max_runtime_hours,
        "max_requests": args.max_requests,
    }
    cursor = connection.execute(
        "INSERT INTO runs(started_at_utc, status, config_json) VALUES (?, 'running', ?)",
        (_utc_text(), _canonical_json(config)),
    )
    run_id = int(cursor.lastrowid)
    connection.commit()
    events_processed = 0
    status = "complete"
    message = "eligible event list exhausted"
    try:
        robots = fetcher.get(f"{BESTFIGHTODDS_ROOT}/robots.txt").text
        if not _robots_allows_public_paths(robots):
            raise BackfillError("BestFightOdds robots policy no longer allows this collector")
        events = _eligible_winner_events(
            winner,
            from_year=args.from_year,
            to_year=args.to_year,
            order=args.order,
        )
        for event in events:
            if args.max_events and events_processed >= args.max_events:
                status, message = "paused", "event cap reached"
                break
            check_storage_budget(
                database_path,
                max_database_mib=args.max_database_mib,
                minimum_free_gib=args.minimum_free_gib,
            )
            parsed = connection.execute(
                "SELECT page_status FROM events WHERE event_url=?",
                (event["event_url"],),
            ).fetchone()
            if parsed is None or parsed["page_status"] != "parsed":
                try:
                    response = fetcher.get(event["event_url"])
                    selections = parse_bestfightodds_method_props(response.text)
                    mappings = _winner_matchups_for_event(winner, event["event_url"])
                    count = _upsert_event(
                        connection,
                        event=event,
                        html=response.text,
                        selections=selections,
                        winner_matchups=mappings,
                    )
                    print(
                        f"Parsed {event['source_event_date']} {event['title']}: "
                        f"{count} matched method selections.",
                        flush=True,
                    )
                except (OSError, ValueError, PropParseError) as error:
                    _record_event_failure(connection, event=event, error=error)
                    print(
                        f"Method page failed for {event['source_event_date']} "
                        f"{event['title']}: {_plain_error(error)}",
                        flush=True,
                    )
                    events_processed += 1
                    continue
            for spec in _pending_downloads(
                connection, event_url=event["event_url"], mode=args.mode
            ):
                check_storage_budget(
                    database_path,
                    max_database_mib=args.max_database_mib,
                    minimum_free_gib=args.minimum_free_gib,
                )
                try:
                    response = fetcher.get(spec.endpoint, referer=spec.event_url)
                    _store_download(
                        connection,
                        spec=spec,
                        series=decode_bestfightodds_chart(response.text),
                    )
                except BudgetReached:
                    raise
                except (OSError, ValueError) as error:
                    _store_download_failure(connection, spec=spec, error=error)
            events_processed += 1
            print(
                f"Completed method event {events_processed}: "
                f"{event['source_event_date']} {event['title']}",
                flush=True,
            )
    except BudgetReached as error:
        status, message = "paused", str(error)
    except KeyboardInterrupt:
        status, message = "paused", "interrupted cleanly"
    except Exception as error:
        status, message = "failed", f"{type(error).__name__}: {error}"
    finally:
        connection.execute(
            """
            UPDATE runs SET finished_at_utc=?, status=?, requests=?,
                response_bytes=?, events_processed=?, message=? WHERE run_id=?
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
        winner.close()
    summary = export_database(
        connection,
        database_path=database_path,
        max_quote_skew_seconds=args.max_quote_skew_seconds,
        session_requests=fetcher.requests,
        session_response_bytes=fetcher.response_bytes,
    )
    connection.close()
    return status, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=default_database_path())
    parser.add_argument(
        "--winner-database", type=Path, default=default_winner_database_path()
    )
    parser.add_argument("--from-year", type=int, default=DEFAULT_FROM_YEAR)
    parser.add_argument("--to-year", type=int, default=DEFAULT_TO_YEAR)
    parser.add_argument("--mode", choices=("mean", "books", "both"), default=DEFAULT_MODE)
    parser.add_argument("--order", choices=("oldest", "newest"), default="newest")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
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
        "--max-quote-skew-seconds",
        type=int,
        default=DEFAULT_MAX_QUOTE_SKEW_SECONDS,
    )
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--acknowledge-source-policy", action="store_true")
    parser.add_argument("--allow-concurrent-source-access", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.from_year < 2007 or args.to_year < args.from_year:
        raise BackfillError("year range must be valid and start in 2007 or later")
    if args.max_events < 0:
        raise BackfillError("max events cannot be negative")
    if not 0 < args.max_quote_skew_seconds <= 86_400:
        raise BackfillError("max quote skew must be within (0, 86400] seconds")
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
                raise BackfillError(f"method database does not exist: {database_path}")
            connection = _open_database(database_path, mode=args.mode)
            summary = (
                export_database(
                    connection,
                    database_path=database_path,
                    max_quote_skew_seconds=args.max_quote_skew_seconds,
                )
                if args.export_only
                else database_summary(connection, database_path=database_path)
            )
            connection.close()
            _print_summary(summary)
            return 0
        if not args.acknowledge_source_policy:
            raise BackfillError(
                "network collection requires --acknowledge-source-policy; the source "
                "allows public paths in robots.txt but its terms do not explicitly "
                "address automated bulk research reuse"
            )
        status, summary = run_backfill(args)
        _print_summary(summary)
        print(f"Session status: {status}")
        if status == "paused":
            print("Run the identical command again to continue from the checkpoint.")
        return 0 if status in {"complete", "paused"} else 1
    except (BackfillError, OSError, sqlite3.Error) as error:
        print(f"method odds backfill: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
