"""Conservatively reconstruct point-in-time UFC market data from Git history.

This command is deliberately a legacy-data backfill, not a betting system.  It
reads blobs from ``main`` without checking them out, maps names only when a
single UFCStats fight is defensible, rejects date-only same-day observations,
and writes immutable records through :mod:`market_tracker`'s public stores.

The exploratory report is development-only.  Git commit time is merely an
upper bound on when a file was observed; it is not a sportsbook quote time or
evidence that any listed price could have been executed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


if __package__:
    from .fight_stat_helpers import (
        maybe_replace_alias_by_default_name,
        regularize_name,
        same_name,
    )
    from .market_tracker import (
        BETTING_STATUS,
        ForecastCapture,
        ForecastCaptureStore,
        MarketDataError,
        PaperDecision,
        PaperDecisionStore,
        PaperSettlement,
        PaperSettlementStore,
        PriorCardBlendEvaluator,
        QuoteSnapshot,
        QuoteSnapshotStore,
        BlendObservation,
        consensus_as_of,
        forecast_metrics,
        matchup_id_for,
        settle_paper_decision,
        summarize_paper_settlements,
    )
else:
    from fight_stat_helpers import (  # type: ignore[no-redef]
        maybe_replace_alias_by_default_name,
        regularize_name,
        same_name,
    )
    from market_tracker import (  # type: ignore[no-redef]
        BETTING_STATUS,
        ForecastCapture,
        ForecastCaptureStore,
        MarketDataError,
        PaperDecision,
        PaperDecisionStore,
        PaperSettlement,
        PaperSettlementStore,
        PriorCardBlendEvaluator,
        QuoteSnapshot,
        QuoteSnapshotStore,
        BlendObservation,
        consensus_as_of,
        forecast_metrics,
        matchup_id_for,
        settle_paper_decision,
        summarize_paper_settlements,
    )


ALGORITHM_VERSION = 1
SOURCE_REF = "main"
SOURCE_NAME = "legacy_git_fightodds"
CORE_BOOKS = ("DraftKings", "BetMGM", "Caesars", "BetRivers", "FanDuel")
VEGAS_PATHS = (
    "src/content/data/external/vegas_odds.json",
    "src/models/buildingMLModel/data/external/vegas_odds.json",
)
HISTORY_PATHS = (
    "src/content/data/external/prediction_history.json",
    "src/models/buildingMLModel/data/external/prediction_history.json",
)
RAW_FIGHTS_PATH = "src/content/data/processed/ufc_fights_reported_doubled.csv"
MIN_OVERROUND = 0.90
MAX_OVERROUND = 1.30
MAX_UNDATED_LEAD_DAYS = 30
MIN_CONSENSUS_BOOKS = 3
PAPER_MINIMUM_EXPECTED_RETURN = 0.05
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_REPLICATES = 10_000

NON_PROMOTABLE_FLAGS = (
    "legacy_commit_timestamp_not_source_quote_timestamp",
    "legacy_rounded_american_odds_probability",
    "legacy_model_training_contract_unknown",
    "training_cutoff_inferred_from_current_raw",
    "current_reconciled_raw_not_as_of_snapshot",
    "completed_matched_selection_survivorship",
    "unverified_execution_and_closing_price",
    "model_and_extraction_not_prospectively_locked",
    "missing_2024_market_history",
    "development_only",
)

EXECUTION_BLOCKERS = (
    "commit_timestamp_not_source_quote_timestamp",
    "listed_price_execution_not_verified",
    "book_access_and_jurisdiction_unknown",
    "legacy_model_probability_is_rounded_reconstruction",
    "historical_sample_not_prospectively_locked",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_hash(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_aware_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp is not timezone-aware: {value!r}")
    return parsed.astimezone(timezone.utc)


def _parse_date(value: object) -> date | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        # Pandas' JSON default for datetime columns is Unix milliseconds.
        if abs(number) >= 100_000_000_000:
            return datetime.fromtimestamp(number / 1000.0, tz=timezone.utc).date()
        if abs(number) >= 1_000_000_000:
            return datetime.fromtimestamp(number, tz=timezone.utc).date()
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "nat", "none", "null"}:
        return None
    if text.lstrip("+-").isdigit():
        return _parse_date(int(text))
    for pattern in (
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            pass
    return None


def _stable_url_id(value: object) -> str | None:
    text = str(value or "").strip().rstrip("/")
    token = text.rsplit("/", 1)[-1]
    if not token or token in {".", ".."} or any(char.isspace() for char in token):
        return None
    return token


@lru_cache(maxsize=None)
def _name_key(value: str) -> str:
    canonical = maybe_replace_alias_by_default_name(value)
    return " ".join(sorted(regularize_name(canonical).split()))


@lru_cache(maxsize=None)
def _same_name(left: str, right: str) -> bool:
    return bool(same_name(left, right))


def _pair_key(fighter: object, opponent: object) -> tuple[str, str] | None:
    fighter_text = " ".join(str(fighter or "").split())
    opponent_text = " ".join(str(opponent or "").split())
    if not fighter_text or not opponent_text:
        return None
    return tuple(sorted((_name_key(fighter_text), _name_key(opponent_text))))


def _parse_moneyline(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    text = (
        str(value)
        .strip()
        .replace("−", "-")
        .replace("–", "-")
        .replace("âˆ’", "-")
        .replace("â€“", "-")
    )
    if not text:
        return None
    if text.upper() in {"EV", "EVEN", "PK", "PICK"}:
        return 100
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    parsed = int(number)
    if abs(parsed) < 100 or abs(parsed) > 100_000:
        return None
    return parsed


def _implied_probability(line: int) -> float:
    return 100.0 / (line + 100.0) if line > 0 else abs(line) / (abs(line) + 100.0)


def _valid_quote_pairs(row: Mapping[str, object]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for column in sorted(row):
        if not column.startswith("fighter "):
            continue
        book = column[len("fighter ") :].strip()
        if not book or book.casefold() == "name":
            continue
        opponent_column = f"opponent {book}"
        if opponent_column not in row:
            continue
        fighter_line = _parse_moneyline(row.get(column))
        opponent_line = _parse_moneyline(row.get(opponent_column))
        if fighter_line is None or opponent_line is None:
            continue
        overround = _implied_probability(fighter_line) + _implied_probability(
            opponent_line
        )
        if MIN_OVERROUND <= overround <= MAX_OVERROUND:
            result[book] = (fighter_line, opponent_line)
    return result


def _valid_legacy_prediction(row: Mapping[str, object]) -> int | None:
    fighter_line = _parse_moneyline(row.get("predicted fighter odds"))
    opponent_line = _parse_moneyline(row.get("predicted opponent odds"))
    if fighter_line is None or opponent_line is None:
        return None
    total = _implied_probability(fighter_line) + _implied_probability(opponent_line)
    # Historical values are rounded fair odds.  Requiring a near-complementary
    # pair prevents a stray sportsbook column from becoming a model forecast.
    if abs(total - 1.0) > 0.015:
        return None
    return fighter_line


def _row_index_key(value: object) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _rows_from_json(payload: bytes) -> list[dict[str, object]]:
    decoded = json.loads(payload.decode("utf-8-sig"))
    if isinstance(decoded, list):
        return [dict(item) for item in decoded if isinstance(item, Mapping)]
    if not isinstance(decoded, Mapping):
        raise ValueError("historical JSON root is neither an object nor an array")
    if not decoded:
        return []
    if all(isinstance(value, Mapping) for value in decoded.values()):
        indices = sorted(
            {str(index) for column in decoded.values() for index in column},
            key=_row_index_key,
        )
        return [
            {str(column): values.get(index) for column, values in decoded.items()}
            for index in indices
        ]
    if all(isinstance(value, list) for value in decoded.values()):
        length = max((len(value) for value in decoded.values()), default=0)
        return [
            {
                str(column): values[index] if index < len(values) else None
                for column, values in decoded.items()
            }
            for index in range(length)
        ]
    raise ValueError("historical JSON is not a supported table orientation")


@dataclass(frozen=True)
class GitRevision:
    commit_sha: str
    author_at_utc: datetime
    committer_at_utc: datetime
    observed_at_utc: datetime
    path: str
    blob_oid: str
    payload: bytes


def _git_command(repo_root: Path, *arguments: str) -> list[str]:
    return [
        "git",
        "-c",
        f"safe.directory={repo_root.as_posix()}",
        "--no-pager",
        *arguments,
    ]


def _git_log(repo_root: Path, current_path: str) -> list[tuple[str, datetime, datetime]]:
    completed = subprocess.run(
        _git_command(
            repo_root,
            "log",
            SOURCE_REF,
            "--follow",
            "--format=%H%x1f%aI%x1f%cI",
            "--",
            current_path,
        ),
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    revisions: list[tuple[str, datetime, datetime]] = []
    seen: set[str] = set()
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pieces = line.split("\x1f")
        if len(pieces) != 3:
            raise RuntimeError(f"unexpected git log record: {line!r}")
        commit, author_text, committer_text = pieces
        if commit in seen:
            continue
        seen.add(commit)
        revisions.append(
            (commit, _parse_aware_datetime(author_text), _parse_aware_datetime(committer_text))
        )
    return revisions


class _GitBlobReader:
    """Efficient read-only ``git cat-file --batch`` client."""

    def __init__(self, repo_root: Path):
        self.process = subprocess.Popen(
            _git_command(repo_root, "cat-file", "--batch"),
            cwd=repo_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read(self, expression: str) -> tuple[str, bytes] | None:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("git cat-file pipes are unavailable")
        self.process.stdin.write(expression.encode("utf-8") + b"\n")
        self.process.stdin.flush()
        header = self.process.stdout.readline()
        if not header:
            raise RuntimeError("git cat-file terminated unexpectedly")
        header_text = header.decode("utf-8", errors="replace").rstrip("\n")
        if header_text.endswith(" missing"):
            return None
        pieces = header_text.split()
        if len(pieces) != 3 or pieces[1] != "blob":
            raise RuntimeError(f"unexpected git cat-file response: {header_text!r}")
        oid, _, size_text = pieces
        payload = self.process.stdout.read(int(size_text))
        terminator = self.process.stdout.read(1)
        if len(payload) != int(size_text) or terminator != b"\n":
            raise RuntimeError("git cat-file returned a truncated blob")
        return oid, payload

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=10)
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
        if return_code:
            raise RuntimeError(
                f"git cat-file failed ({return_code}): "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )

    def __enter__(self) -> "_GitBlobReader":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _load_lineage(repo_root: Path, paths: Sequence[str]) -> list[GitRevision]:
    log_records = _git_log(repo_root, paths[0])
    revisions: list[GitRevision] = []
    with _GitBlobReader(repo_root) as reader:
        for commit, author, committer in log_records:
            matches: list[tuple[str, str, bytes]] = []
            for path in paths:
                blob = reader.read(f"{commit}:{path}")
                if blob is not None:
                    matches.append((path, blob[0], blob[1]))
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected exactly one historical path at {commit}, found "
                    f"{[path for path, _, _ in matches]}"
                )
            path, blob_oid, payload = matches[0]
            revisions.append(
                GitRevision(
                    commit_sha=commit,
                    author_at_utc=author,
                    committer_at_utc=committer,
                    observed_at_utc=max(author, committer),
                    path=path,
                    blob_oid=blob_oid,
                    payload=payload,
                )
            )
    return sorted(revisions, key=lambda item: (item.observed_at_utc, item.commit_sha))


@dataclass(frozen=True)
class RawFight:
    fight_id: str
    event_id: str
    event_date: date
    fighter_id: str
    opponent_id: str
    fighter_name: str
    opponent_name: str
    target: int | None
    raw_result: str


def _load_raw_fights(repo_root: Path) -> tuple[list[RawFight], str, dict[str, int]]:
    path = repo_root / RAW_FIGHTS_PATH
    payload = path.read_bytes()
    rows = list(csv.DictReader(payload.decode("utf-8-sig").splitlines()))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("fight_url") or "").strip(), []).append(row)
    fights: list[RawFight] = []
    rejected = 0
    for fight_url, sides in grouped.items():
        if len(sides) != 2:
            rejected += 1
            continue
        fight_id = _stable_url_id(fight_url)
        side_data: list[tuple[str, str, str, str, date, str]] = []
        for row in sides:
            fighter_id = _stable_url_id(row.get("fighter_url"))
            opponent_id = _stable_url_id(row.get("opponent_url"))
            event_id = _stable_url_id(row.get("event_url"))
            event_day = _parse_date(row.get("date"))
            fighter_name = " ".join(str(row.get("fighter") or "").split())
            opponent_name = " ".join(str(row.get("opponent") or "").split())
            if not all(
                (fighter_id, opponent_id, event_id, event_day, fighter_name, opponent_name)
            ):
                side_data = []
                break
            side_data.append(
                (
                    str(fighter_id),
                    str(opponent_id),
                    str(event_id),
                    fighter_name,
                    event_day,
                    str(row.get("result") or "").strip().upper(),
                )
            )
        if fight_id is None or len(side_data) != 2:
            rejected += 1
            continue
        left, right = side_data
        if (
            left[0] != right[1]
            or left[1] != right[0]
            or left[2] != right[2]
            or left[4] != right[4]
        ):
            rejected += 1
            continue
        canonical_id, other_id = sorted((left[0], left[1]))
        canonical_side = left if left[0] == canonical_id else right
        other_side = right if canonical_side is left else left
        result = canonical_side[5]
        target = 1 if result == "W" else 0 if result == "L" else None
        fights.append(
            RawFight(
                fight_id=fight_id,
                event_id=canonical_side[2],
                event_date=canonical_side[4],
                fighter_id=canonical_id,
                opponent_id=other_id,
                fighter_name=canonical_side[3],
                opponent_name=other_side[3],
                target=target,
                raw_result=result,
            )
        )
    fights.sort(key=lambda item: (item.event_date, item.event_id, item.fight_id))
    return fights, sha256(payload).hexdigest(), {
        "source_rows": len(rows),
        "physical_fight_groups": len(grouped),
        "accepted_physical_fights": len(fights),
        "rejected_physical_fights": rejected,
    }


class _FightMatcher:
    def __init__(self, fights: Sequence[RawFight]):
        self.fights = tuple(fights)
        self.by_pair: dict[tuple[str, str], list[RawFight]] = {}
        for fight in fights:
            key = _pair_key(fight.fighter_name, fight.opponent_name)
            if key is not None:
                self.by_pair.setdefault(key, []).append(fight)

    @lru_cache(maxsize=None)
    def match(
        self,
        fighter_name: str,
        opponent_name: str,
        event_day: date | None,
        observed_day: date | None,
        undated_window: bool,
    ) -> tuple[str, RawFight | None, bool | None]:
        key = _pair_key(fighter_name, opponent_name)
        candidates = [] if key is None else list(self.by_pair.get(key, ()))
        if event_day is not None:
            candidates = [fight for fight in candidates if fight.event_date == event_day]
        elif undated_window:
            if observed_day is None:
                return "no_match", None, None
            candidates = [
                fight
                for fight in candidates
                if 1 <= (fight.event_date - observed_day).days <= MAX_UNDATED_LEAD_DAYS
            ]
        oriented: list[tuple[RawFight, bool]] = []
        for fight in candidates:
            direct = _same_name(fighter_name, fight.fighter_name) and _same_name(
                opponent_name, fight.opponent_name
            )
            reverse = _same_name(fighter_name, fight.opponent_name) and _same_name(
                opponent_name, fight.fighter_name
            )
            if direct != reverse:
                oriented.append((fight, direct))
        if not oriented:
            return "no_match", None, None
        if len(oriented) != 1:
            return "ambiguous", None, None
        return "matched", oriented[0][0], oriented[0][1]


@dataclass(frozen=True)
class MappedSnapshot:
    revision: GitRevision
    row_index: int
    fight: RawFight
    source_fighter_id: str
    source_opponent_id: str
    source_fighter_name: str
    source_opponent_name: str
    quotes: Mapping[str, tuple[int, int]]
    legacy_prediction: int | None
    explicit_date: bool

    @property
    def core_quotes(self) -> dict[str, tuple[int, int]]:
        return {book: self.quotes[book] for book in CORE_BOOKS if book in self.quotes}


def _snapshot_signature(snapshot: MappedSnapshot) -> str:
    return _canonical_hash(
        {
            "fight_id": snapshot.fight.fight_id,
            "source_fighter_id": snapshot.source_fighter_id,
            "source_opponent_id": snapshot.source_opponent_id,
            "quotes": {
                book: list(lines) for book, lines in sorted(snapshot.quotes.items())
            },
            "legacy_prediction": snapshot.legacy_prediction,
        }
    )


def _map_vegas_lineage(
    revisions: Sequence[GitRevision], matcher: _FightMatcher
) -> tuple[list[MappedSnapshot], dict[str, int]]:
    counts = {
        "source_rows": 0,
        "dated_matched": 0,
        "dated_no_match": 0,
        "dated_ambiguous": 0,
        "undated_matched_within_30_days": 0,
        "undated_no_match_within_30_days": 0,
        "undated_ambiguous_within_30_days": 0,
        "mapped_rows_without_valid_paired_quote": 0,
        "strict_prior_day_rows_before_duplicate_check": 0,
        "same_day_rows_rejected": 0,
        "after_event_rows_rejected": 0,
        "duplicate_commit_fight_rows_rejected": 0,
        "conflicting_duplicate_groups_rejected": 0,
    }
    strict: list[MappedSnapshot] = []
    for revision in revisions:
        rows = _rows_from_json(revision.payload)
        counts["source_rows"] += len(rows)
        for row_index, row in enumerate(rows):
            fighter_name = " ".join(str(row.get("fighter name") or "").split())
            opponent_name = " ".join(str(row.get("opponent name") or "").split())
            event_day = _parse_date(row.get("date"))
            status, fight, direct = matcher.match(
                fighter_name,
                opponent_name,
                event_day,
                revision.observed_at_utc.date(),
                event_day is None,
            )
            prefix = "dated" if event_day is not None else "undated"
            suffix = "" if event_day is not None else "_within_30_days"
            if status == "matched":
                counts[f"{prefix}_matched{suffix}"] += 1
            else:
                counts[f"{prefix}_{status}{suffix}"] += 1
                continue
            assert fight is not None and direct is not None
            quotes = _valid_quote_pairs(row)
            if not quotes:
                counts["mapped_rows_without_valid_paired_quote"] += 1
                continue
            source_fighter_id = fight.fighter_id if direct else fight.opponent_id
            source_opponent_id = fight.opponent_id if direct else fight.fighter_id
            snapshot = MappedSnapshot(
                revision=revision,
                row_index=row_index,
                fight=fight,
                source_fighter_id=source_fighter_id,
                source_opponent_id=source_opponent_id,
                source_fighter_name=fighter_name,
                source_opponent_name=opponent_name,
                quotes=quotes,
                legacy_prediction=_valid_legacy_prediction(row),
                explicit_date=event_day is not None,
            )
            delta = (fight.event_date - revision.observed_at_utc.date()).days
            if delta > 0:
                counts["strict_prior_day_rows_before_duplicate_check"] += 1
                strict.append(snapshot)
            elif delta == 0:
                counts["same_day_rows_rejected"] += 1
            else:
                counts["after_event_rows_rejected"] += 1

    accepted: list[MappedSnapshot] = []
    grouped: dict[tuple[str, str], list[MappedSnapshot]] = {}
    for snapshot in strict:
        grouped.setdefault(
            (snapshot.revision.commit_sha, snapshot.fight.fight_id), []
        ).append(snapshot)
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: item.row_index)
        if len(group) == 1:
            accepted.append(group[0])
            continue
        signatures = {_snapshot_signature(item) for item in group}
        if len(signatures) == 1:
            accepted.append(group[0])
            counts["duplicate_commit_fight_rows_rejected"] += len(group) - 1
        else:
            counts["duplicate_commit_fight_rows_rejected"] += len(group)
            counts["conflicting_duplicate_groups_rejected"] += 1
    return sorted(
        accepted,
        key=lambda item: (
            item.revision.observed_at_utc,
            item.revision.commit_sha,
            item.fight.event_id,
            item.fight.fight_id,
        ),
    ), counts


def _canonical_lines(snapshot: MappedSnapshot, lines: tuple[int, int]) -> tuple[int, int]:
    return lines if snapshot.source_fighter_id < snapshot.source_opponent_id else lines[::-1]


def _capture_id(snapshot: MappedSnapshot) -> str:
    revision = snapshot.revision
    return "capture_" + _canonical_hash(
        {
            "capture_schema": 1,
            "source": SOURCE_NAME,
            "source_ref": SOURCE_REF,
            "commit_sha": revision.commit_sha,
            "event_id": snapshot.fight.event_id,
            "observed_at_utc": _utc_text(revision.observed_at_utc),
            "source_path": revision.path,
            "source_payload_sha256": sha256(revision.payload).hexdigest(),
        }
    )


def _training_cutoff(observed_day: date, fights: Sequence[RawFight]) -> date:
    eligible = [fight.event_date for fight in fights if fight.event_date < observed_day]
    if not eligible:
        raise RuntimeError(
            f"cannot infer a pre-forecast training cutoff before {observed_day}"
        )
    return max(eligible)


def _build_market_records(
    snapshots: Sequence[MappedSnapshot], fights: Sequence[RawFight]
) -> tuple[
    tuple[QuoteSnapshot, ...],
    tuple[ForecastCapture, ...],
    dict[tuple[str, str], tuple[QuoteSnapshot, ...]],
    dict[tuple[str, str], ForecastCapture],
]:
    first_seen: dict[tuple[str, str, int, int], datetime] = {}
    for snapshot in snapshots:
        for book, lines in snapshot.core_quotes.items():
            canonical = _canonical_lines(snapshot, lines)
            key = (snapshot.fight.fight_id, book.casefold(), *canonical)
            first_seen[key] = min(
                first_seen.get(key, snapshot.revision.observed_at_utc),
                snapshot.revision.observed_at_utc,
            )

    quotes: list[QuoteSnapshot] = []
    forecasts: list[ForecastCapture] = []
    for snapshot in snapshots:
        capture_id = _capture_id(snapshot)
        payload_digest = sha256(snapshot.revision.payload).hexdigest()
        for book, lines in snapshot.core_quotes.items():
            canonical = _canonical_lines(snapshot, lines)
            key = (snapshot.fight.fight_id, book.casefold(), *canonical)
            quotes.append(
                QuoteSnapshot.create(
                    capture_id=capture_id,
                    event_id=snapshot.fight.event_id,
                    fighter_id=snapshot.source_fighter_id,
                    opponent_id=snapshot.source_opponent_id,
                    fighter_name=snapshot.source_fighter_name,
                    opponent_name=snapshot.source_opponent_name,
                    event_date=snapshot.fight.event_date,
                    timing_precision="date",
                    event_start_utc=None,
                    observed_at_utc=snapshot.revision.observed_at_utc,
                    quote_first_seen_at_utc=first_seen[key],
                    source=SOURCE_NAME,
                    book=book,
                    fighter_moneyline=lines[0],
                    opponent_moneyline=lines[1],
                    fight_id=snapshot.fight.fight_id,
                    source_payload_sha256=payload_digest,
                )
            )
        if snapshot.legacy_prediction is not None:
            # Feed the legacy factory an already-canonical side.  Besides
            # making provenance explicit, this avoids deriving canonical
            # probability as ``1 - implied(source_odds)``: that expression is
            # not always bit-identical to ``implied(canonical_odds)`` and a
            # content-addressed JSON round trip must be exact.
            canonical_prediction = (
                snapshot.legacy_prediction
                if snapshot.source_fighter_id < snapshot.source_opponent_id
                else 100
                if abs(snapshot.legacy_prediction) == 100
                else -snapshot.legacy_prediction
            )
            forecasts.append(
                ForecastCapture.from_legacy_american_odds(
                    predicted_american_odds=canonical_prediction,
                    capture_id=capture_id,
                    event_id=snapshot.fight.event_id,
                    fighter_id=snapshot.fight.fighter_id,
                    opponent_id=snapshot.fight.opponent_id,
                    fighter_name=snapshot.fight.fighter_name,
                    opponent_name=snapshot.fight.opponent_name,
                    event_date=snapshot.fight.event_date,
                    timing_precision="date",
                    event_start_utc=None,
                    forecast_issued_at_utc=snapshot.revision.observed_at_utc,
                    model_id="legacy_ufc_prediction",
                    model_version="historical_unknown_contract_v1",
                    model_trained_through=_training_cutoff(
                        snapshot.revision.observed_at_utc.date(), fights
                    ),
                    model_training_cutoff_precision="date",
                    source_commit_sha=snapshot.revision.commit_sha,
                    fight_id=snapshot.fight.fight_id,
                )
            )

    quotes.sort(key=lambda item: (item.observed_at_utc, item.quote_id))
    forecasts.sort(
        key=lambda item: (item.forecast_issued_at_utc, item.forecast_capture_id)
    )
    quotes_by_key: dict[tuple[str, str], list[QuoteSnapshot]] = {}
    for quote in quotes:
        quotes_by_key.setdefault((quote.capture_id, quote.matchup_id), []).append(quote)
    forecast_by_key: dict[tuple[str, str], ForecastCapture] = {}
    for forecast in forecasts:
        key = (forecast.capture_id, forecast.matchup_id)
        prior = forecast_by_key.get(key)
        if prior is not None and prior != forecast:
            raise RuntimeError(f"multiple legacy forecasts for {key}")
        forecast_by_key[key] = forecast
    capture_times: dict[str, set[str]] = {}
    for quote in quotes:
        capture_times.setdefault(quote.capture_id, set()).add(quote.observed_at_utc)
    for forecast in forecasts:
        capture_times.setdefault(forecast.capture_id, set()).add(
            forecast.forecast_issued_at_utc
        )
    if any(len(times) != 1 for times in capture_times.values()):
        raise RuntimeError(
            "legacy capture contains mixed quote/forecast observation timestamps"
        )
    return (
        tuple(quotes),
        tuple(forecasts),
        {key: tuple(value) for key, value in quotes_by_key.items()},
        forecast_by_key,
    )


def _canonical_quote_tuple(
    fight_id: str,
    fighter_id: str,
    opponent_id: str,
    book: str,
    lines: tuple[int, int],
) -> tuple[str, str, int, int]:
    canonical = lines if fighter_id < opponent_id else lines[::-1]
    return (fight_id, book.casefold(), canonical[0], canonical[1])


def _history_audit(
    revisions: Sequence[GitRevision],
    matcher: _FightMatcher,
    strict_snapshots: Sequence[MappedSnapshot],
) -> dict[str, object]:
    entities: dict[
        tuple[str, tuple[str, str]],
        dict[str, object],
    ] = {}
    source_rows = 0
    for revision in revisions:
        for row in _rows_from_json(revision.payload):
            source_rows += 1
            fighter_name = " ".join(str(row.get("fighter name") or "").split())
            opponent_name = " ".join(str(row.get("opponent name") or "").split())
            pair = _pair_key(fighter_name, opponent_name)
            if pair is None:
                continue
            event_day = _parse_date(row.get("date"))
            entity_key = (event_day.isoformat() if event_day else "", pair)
            entity = entities.setdefault(
                entity_key,
                {
                    "fighter_name": fighter_name,
                    "opponent_name": opponent_name,
                    "event_day": event_day,
                    "quotes": set(),
                },
            )
            quote_set = entity["quotes"]
            assert isinstance(quote_set, set)
            for book, lines in _valid_quote_pairs(row).items():
                quote_set.add((book, lines[0], lines[1]))

    mapped_entities = 0
    no_match = 0
    ambiguous = 0
    valid_quote_records = 0
    mapped_tuples: set[tuple[str, str, int, int]] = set()
    mapped_tuples_ex_ref: set[tuple[str, str, int, int]] = set()
    dated_history_fights: set[str] = set()
    undated_history_fights: set[str] = set()
    for entity in entities.values():
        fighter_name = str(entity["fighter_name"])
        opponent_name = str(entity["opponent_name"])
        event_day = entity["event_day"]
        assert event_day is None or isinstance(event_day, date)
        status, fight, direct = matcher.match(
            fighter_name,
            opponent_name,
            event_day,
            None,
            False,
        )
        quote_set = entity["quotes"]
        assert isinstance(quote_set, set)
        valid_quote_records += len(quote_set)
        if status == "ambiguous":
            ambiguous += 1
            continue
        if status != "matched" or fight is None or direct is None:
            no_match += 1
            continue
        mapped_entities += 1
        if quote_set:
            (
                dated_history_fights
                if event_day is not None
                else undated_history_fights
            ).add(fight.fight_id)
        source_fighter_id = fight.fighter_id if direct else fight.opponent_id
        source_opponent_id = fight.opponent_id if direct else fight.fighter_id
        for book, fighter_line, opponent_line in quote_set:
            item = _canonical_quote_tuple(
                fight.fight_id,
                source_fighter_id,
                source_opponent_id,
                book,
                (fighter_line, opponent_line),
            )
            mapped_tuples.add(item)
            if book.casefold() != "ref":
                mapped_tuples_ex_ref.add(item)

    vegas_tuples: set[tuple[str, str, int, int]] = set()
    for snapshot in strict_snapshots:
        for book, lines in snapshot.quotes.items():
            vegas_tuples.add(
                _canonical_quote_tuple(
                    snapshot.fight.fight_id,
                    snapshot.source_fighter_id,
                    snapshot.source_opponent_id,
                    book,
                    lines,
                )
            )
    intersection = mapped_tuples & vegas_tuples
    intersection_ex_ref = {item for item in intersection if item[1] != "ref"}
    vegas_fights = {snapshot.fight.fight_id for snapshot in strict_snapshots}
    return {
        "lineage_source_rows_including_cumulative_repetitions": source_rows,
        "unique_matchup_date_entities": len(entities),
        "stable_matched_entities": mapped_entities,
        "no_match_entities": no_match,
        "ambiguous_entities": ambiguous,
        "unique_valid_entity_book_price_records": valid_quote_records,
        "stable_mapped_price_tuples_including_ref": len(mapped_tuples),
        "stable_mapped_price_tuples_excluding_ref": len(mapped_tuples_ex_ref),
        "intersection_with_strict_vegas_including_ref": len(intersection),
        "intersection_with_strict_vegas_excluding_ref": len(intersection_ex_ref),
        "intersection_fights": len({item[0] for item in intersection}),
        "history_only_price_tuples": len(mapped_tuples - vegas_tuples),
        "fights_with_any_history_only_price_tuple": len(
            {item[0] for item in mapped_tuples - vegas_tuples}
        ),
        "history_only_stable_fights": len(
            {item[0] for item in mapped_tuples}
            - {item[0] for item in vegas_tuples}
        ),
        "fights_with_any_undated_history_price": len(undated_history_fights),
        "fights_with_only_undated_history_price": len(
            undated_history_fights - dated_history_fights
        ),
        "fights_recoverable_only_from_untimed_history": len(
            undated_history_fights - vegas_fights
        ),
        "timing_eligibility": "excluded",
        "timing_exclusion_reason": (
            "prediction_history is cumulative and generally committed after "
            "settlement; its commit timestamp is not a defensible quote timestamp"
        ),
    }


def _latest_eligible_snapshots(
    snapshots: Sequence[MappedSnapshot], minimum_books: int = MIN_CONSENSUS_BOOKS
) -> tuple[MappedSnapshot, ...]:
    eligible: dict[str, list[MappedSnapshot]] = {}
    for snapshot in snapshots:
        if len(snapshot.core_quotes) >= minimum_books:
            eligible.setdefault(snapshot.fight.fight_id, []).append(snapshot)
    selected = [
        max(
            group,
            key=lambda item: (
                item.revision.observed_at_utc,
                item.revision.commit_sha,
                _snapshot_signature(item),
            ),
        )
        for group in eligible.values()
    ]
    return tuple(
        sorted(selected, key=lambda item: (item.fight.event_date, item.fight.fight_id))
    )


def _lead_bucket(days: int) -> str:
    if days == 1:
        return "D-1"
    if 2 <= days <= 3:
        return "D-2_to_D-3"
    if 4 <= days <= 7:
        return "D-4_to_D-7"
    if 8 <= days <= 14:
        return "D-8_to_D-14"
    return "D-15_to_D-30_or_more"


def _metric_mapping(probabilities: Sequence[float], targets: Sequence[int]) -> dict[str, object] | None:
    if not probabilities:
        return None
    return forecast_metrics(probabilities, targets).to_mapping()


def _build_observations(
    latest: Sequence[MappedSnapshot],
    quotes_by_key: Mapping[tuple[str, str], tuple[QuoteSnapshot, ...]],
    forecast_by_key: Mapping[tuple[str, str], ForecastCapture],
) -> tuple[
    tuple[BlendObservation, ...],
    dict[str, object],
    dict[str, tuple[MappedSnapshot, ForecastCapture]],
]:
    observations: list[BlendObservation] = []
    observation_context: dict[str, tuple[MappedSnapshot, ForecastCapture]] = {}
    missing_forecast = 0
    draws_or_no_contests = 0
    consensus_probabilities: list[float] = []
    consensus_targets: list[int] = []
    for snapshot in latest:
        if snapshot.fight.target is None:
            draws_or_no_contests += 1
            continue
        capture_id = _capture_id(snapshot)
        matchup_id = matchup_id_for(
            snapshot.fight.event_id,
            snapshot.source_fighter_id,
            snapshot.source_opponent_id,
        )
        key = (capture_id, matchup_id)
        quotes = quotes_by_key.get(key, ())
        market = consensus_as_of(
            quotes,
            capture_id=capture_id,
            matchup_id=matchup_id,
            as_of_utc=snapshot.revision.observed_at_utc,
            min_books=MIN_CONSENSUS_BOOKS,
        )
        consensus_probabilities.append(market.no_vig_fighter_probability)
        consensus_targets.append(snapshot.fight.target)
        forecast = forecast_by_key.get(key)
        if forecast is None:
            missing_forecast += 1
            continue
        observation = BlendObservation.from_captures(
            market,
            forecast,
            target=snapshot.fight.target,
            fight_id=snapshot.fight.fight_id,
        )
        observations.append(observation)
        observation_context[observation.observation_id] = (snapshot, forecast)
    observations.sort(
        key=lambda item: (item.event_date, item.event_id, item.matchup_id)
    )
    return (
        tuple(observations),
        {
            "latest_eligible_snapshots": len(latest),
            "w_l_outcomes": len(consensus_targets),
            "draw_nc_outcomes": draws_or_no_contests,
            "missing_legacy_forecast": missing_forecast,
            "blend_observations": len(observations),
            "market_metrics_all_w_l": _metric_mapping(
                consensus_probabilities, consensus_targets
            ),
        },
        observation_context,
    )


def _loss(probability: float, target: int) -> float:
    return -(target * math.log(probability) + (1 - target) * math.log1p(-probability))


def _quantile(values: Sequence[float], probability_value: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _event_block_uncertainty(predictions: Sequence[object]) -> dict[str, object] | None:
    evaluated = [item for item in predictions if item.status == "evaluated"]
    if not evaluated:
        return None
    blocks: dict[str, list[tuple[float, float]]] = {}
    for item in evaluated:
        assert item.blend_probability is not None
        log_delta = _loss(item.blend_probability, item.target) - _loss(
            item.market_probability, item.target
        )
        brier_delta = (item.blend_probability - item.target) ** 2 - (
            item.market_probability - item.target
        ) ** 2
        blocks.setdefault(item.event_id, []).append((log_delta, brier_delta))
    block_ids = sorted(blocks)
    point_pairs = [pair for block in blocks.values() for pair in block]
    point_log = sum(pair[0] for pair in point_pairs) / len(point_pairs)
    point_brier = sum(pair[1] for pair in point_pairs) / len(point_pairs)
    rng = random.Random(BOOTSTRAP_SEED)
    log_samples: list[float] = []
    brier_samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected_ids = [rng.choice(block_ids) for _ in block_ids]
        pairs = [pair for event_id in selected_ids for pair in blocks[event_id]]
        log_samples.append(sum(pair[0] for pair in pairs) / len(pairs))
        brier_samples.append(sum(pair[1] for pair in pairs) / len(pairs))
    return {
        "method": "nonparametric_event_block_bootstrap",
        "unit": "event_id_card",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": 0.95,
        "event_blocks": len(block_ids),
        "evaluated_fights": len(evaluated),
        "blend_minus_market_log_loss": {
            "point_estimate": point_log,
            "percentile_interval": [
                _quantile(log_samples, 0.025),
                _quantile(log_samples, 0.975),
            ],
        },
        "blend_minus_market_brier_score": {
            "point_estimate": point_brier,
            "percentile_interval": [
                _quantile(brier_samples, 0.025),
                _quantile(brier_samples, 0.975),
            ],
        },
    }


def _year_metrics(predictions: Sequence[object]) -> dict[str, object]:
    by_year: dict[str, list[object]] = {}
    for item in predictions:
        if item.status == "evaluated":
            by_year.setdefault(item.event_date[:4], []).append(item)
    result: dict[str, object] = {}
    for year, items in sorted(by_year.items()):
        targets = [item.target for item in items]
        result[year] = {
            "fights": len(items),
            "market": _metric_mapping([item.market_probability for item in items], targets),
            "model": _metric_mapping([item.model_probability for item in items], targets),
            "blend": _metric_mapping([item.blend_probability for item in items], targets),
        }
    return result


def _gamma_distribution(predictions: Sequence[object]) -> dict[str, object]:
    evaluated = [item for item in predictions if item.status == "evaluated"]
    fight_counts: dict[str, int] = {}
    card_gamma: dict[tuple[str, str], float] = {}
    per_year_fights: dict[str, dict[str, int]] = {}
    for item in evaluated:
        assert item.selected_gamma is not None
        gamma_key = f"{item.selected_gamma:.2f}"
        fight_counts[gamma_key] = fight_counts.get(gamma_key, 0) + 1
        year = item.event_date[:4]
        year_counts = per_year_fights.setdefault(year, {})
        year_counts[gamma_key] = year_counts.get(gamma_key, 0) + 1
        card_key = (item.event_date, item.event_id)
        prior = card_gamma.setdefault(card_key, item.selected_gamma)
        if prior != item.selected_gamma:
            raise RuntimeError("one event card was assigned multiple blend gammas")
    card_counts: dict[str, int] = {}
    per_year_cards: dict[str, dict[str, int]] = {}
    for (event_day, _), gamma in sorted(card_gamma.items()):
        gamma_key = f"{gamma:.2f}"
        card_counts[gamma_key] = card_counts.get(gamma_key, 0) + 1
        year_counts = per_year_cards.setdefault(event_day[:4], {})
        year_counts[gamma_key] = year_counts.get(gamma_key, 0) + 1
    return {
        "evaluated_fights_by_gamma": dict(sorted(fight_counts.items())),
        "evaluated_cards_by_gamma": dict(sorted(card_counts.items())),
        "per_year": {
            year: {
                "fights_by_gamma": dict(sorted(per_year_fights.get(year, {}).items())),
                "cards_by_gamma": dict(sorted(per_year_cards.get(year, {}).items())),
            }
            for year in sorted(set(per_year_fights) | set(per_year_cards))
        },
    }


def _poisson_binomial_cdf(probabilities: Sequence[float], maximum_wins: int) -> float:
    distribution = [1.0] + [0.0] * len(probabilities)
    for probability_value in probabilities:
        for wins in range(len(probabilities), 0, -1):
            distribution[wins] = (
                distribution[wins] * (1.0 - probability_value)
                + distribution[wins - 1] * probability_value
            )
        distribution[0] *= 1.0 - probability_value
    return sum(distribution[: maximum_wins + 1])


def _paper_selection_diagnostics(
    decisions: Sequence[PaperDecision],
    settlements: Sequence[PaperSettlement],
    quotes_by_key: Mapping[tuple[str, str], tuple[QuoteSnapshot, ...]],
) -> dict[str, object]:
    settlement_by_decision = {item.decision_id: item for item in settlements}
    quote_by_id = {
        quote.quote_id: quote for group in quotes_by_key.values() for quote in group
    }
    rows: list[dict[str, object]] = []
    orientation_errors = 0
    expected_return_errors = 0
    for decision in decisions:
        if decision.paper_action == "pass":
            continue
        settlement = settlement_by_decision[decision.decision_id]
        quote = quote_by_id[decision.reference_quote_id]
        selected_target = 1 if decision.paper_action == "fighter" else 0
        won = settlement.target == selected_target
        recorded_win = settlement.settlement_status == "paper_win"
        if won != recorded_win:
            orientation_errors += 1
        if decision.paper_action == "fighter":
            line = decision.fighter_reference_moneyline
            selected_probability = decision.blend_probability
            model_probability = decision.model_probability
            market_probability = decision.market_probability
            expected_return = decision.fighter_expected_return
            break_even = decision.fighter_break_even_probability
            other_break_evens = [
                item.fighter_implied_probability
                for item in quotes_by_key[(decision.capture_id, decision.matchup_id)]
                if item.book.casefold() != quote.book.casefold()
            ]
        else:
            line = decision.opponent_reference_moneyline
            selected_probability = 1.0 - decision.blend_probability
            model_probability = 1.0 - decision.model_probability
            market_probability = 1.0 - decision.market_probability
            expected_return = decision.opponent_expected_return
            break_even = decision.opponent_break_even_probability
            other_break_evens = [
                item.opponent_implied_probability
                for item in quotes_by_key[(decision.capture_id, decision.matchup_id)]
                if item.book.casefold() != quote.book.casefold()
            ]
        decimal_profit = line / 100.0 if line > 0 else 100.0 / abs(line)
        recomputed_expected_return = selected_probability * (1.0 + decimal_profit) - 1.0
        if abs(recomputed_expected_return - expected_return) > 1e-12:
            expected_return_errors += 1
        rows.append(
            {
                "year": decision.event_date[:4],
                "book": quote.book,
                "line": line,
                "break_even_probability": break_even,
                "other_book_median_break_even_probability": statistics.median(
                    other_break_evens
                ),
                "market_probability": market_probability,
                "model_probability": model_probability,
                "blend_probability": selected_probability,
                "expected_return": expected_return,
                "won": won,
            }
        )
    if not rows:
        return {
            "selections": 0,
            "orientation_and_settlement_consistent": orientation_errors == 0,
            "expected_return_recomputation_consistent": expected_return_errors == 0,
        }

    def summarize(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
        probabilities = [float(item["blend_probability"]) for item in items]
        wins = sum(bool(item["won"]) for item in items)
        lines = [int(item["line"]) for item in items]
        expected_returns = [float(item["expected_return"]) for item in items]
        break_evens = [float(item["break_even_probability"]) for item in items]
        median_gaps = [
            float(item["break_even_probability"])
            - float(item["other_book_median_break_even_probability"])
            for item in items
        ]
        return {
            "selections": len(items),
            "wins": wins,
            "losses": len(items) - wins,
            "actual_win_rate": wins / len(items),
            "expected_wins_from_blend": sum(probabilities),
            "mean_offered_american_odds": sum(lines) / len(lines),
            "median_offered_american_odds": statistics.median(lines),
            "mean_selected_side_break_even_probability": sum(break_evens) / len(items),
            "mean_selected_side_market_probability": sum(
                float(item["market_probability"]) for item in items
            )
            / len(items),
            "mean_selected_side_model_probability": sum(
                float(item["model_probability"]) for item in items
            )
            / len(items),
            "mean_selected_side_blend_probability": sum(probabilities) / len(items),
            "mean_expected_return_at_quoted_price": sum(expected_returns) / len(items),
            "median_expected_return_at_quoted_price": statistics.median(
                expected_returns
            ),
            "mean_break_even_gap_vs_other_book_median": sum(median_gaps) / len(items),
            "selected_price_better_than_other_book_median": sum(
                gap < 0.0 for gap in median_gaps
            ),
            "selected_price_at_least_2pp_better_than_other_book_median": sum(
                gap <= -0.02 for gap in median_gaps
            ),
            "selected_price_at_least_5pp_better_than_other_book_median": sum(
                gap <= -0.05 for gap in median_gaps
            ),
            "poisson_binomial_probability_of_at_most_actual_wins": (
                _poisson_binomial_cdf(probabilities, wins)
            ),
        }

    book_counts: dict[str, int] = {}
    for row in rows:
        book = str(row["book"])
        book_counts[book] = book_counts.get(book, 0) + 1
    per_year = {
        year: summarize([item for item in rows if item["year"] == year])
        for year in sorted({str(item["year"]) for item in rows})
    }
    return {
        **summarize(rows),
        "selected_reference_books": dict(sorted(book_counts.items())),
        "per_year": per_year,
        "orientation_and_settlement_consistent": orientation_errors == 0,
        "expected_return_recomputation_consistent": expected_return_errors == 0,
        "diagnosis": (
            "no orientation, settlement, or EV arithmetic defect detected; every "
            "selection used the best-priced side relative to the other-book median, "
            "but none was at least five probability points better. The loss is not "
            "dominated by one gross line outlier. It is consistent with a small, "
            "underdog-heavy, price-shopped sample whose legacy model overestimated "
            "the selected sides; unverified historical quote freshness remains an "
            "additional non-execution risk."
        ),
    }


def _build_paper_records(
    evaluation: object,
    observation_context: Mapping[str, tuple[MappedSnapshot, ForecastCapture]],
    quotes_by_key: Mapping[tuple[str, str], tuple[QuoteSnapshot, ...]],
    result_source_sha256: str,
    *,
    fixed_gamma: float | None = None,
) -> tuple[tuple[PaperDecision, ...], tuple[PaperSettlement, ...], dict[str, object]]:
    decisions: list[PaperDecision] = []
    settlements: list[PaperSettlement] = []
    insufficient_four_book_price = 0
    candidate_target_prices = 0
    for prediction in evaluation.predictions:
        if prediction.status != "evaluated":
            continue
        snapshot, forecast = observation_context[prediction.observation_id]
        key = (prediction.capture_id, prediction.matchup_id)
        quotes = tuple(sorted(quotes_by_key[key], key=lambda item: item.book.casefold()))
        candidates: list[tuple[PaperDecision, str, float]] = []
        for target_quote in quotes:
            try:
                market = consensus_as_of(
                    quotes,
                    capture_id=prediction.capture_id,
                    matchup_id=prediction.matchup_id,
                    as_of_utc=snapshot.revision.observed_at_utc,
                    min_books=MIN_CONSENSUS_BOOKS,
                    exclude_books=(target_quote.book,),
                )
            except MarketDataError:
                continue
            candidate_target_prices += 1
            if (
                target_quote.observed_at_utc != market.as_of_utc
                or market.latest_observed_at_utc != market.as_of_utc
            ):
                raise RuntimeError(
                    "paper target and leave-one-out consensus are not from the "
                    "same fresh capture timestamp"
                )
            decision = PaperDecision.create(
                market,
                target_quote,
                forecast,
                selected_gamma=(
                    prediction.selected_gamma
                    if fixed_gamma is None
                    else fixed_gamma
                ),
                decision_issued_at_utc=snapshot.revision.observed_at_utc,
                minimum_expected_return=PAPER_MINIMUM_EXPECTED_RETURN,
                fight_id=snapshot.fight.fight_id,
            )
            score = (
                decision.fighter_expected_return
                if decision.paper_action == "fighter"
                else decision.opponent_expected_return
                if decision.paper_action == "opponent"
                else max(
                    decision.fighter_expected_return,
                    decision.opponent_expected_return,
                )
            )
            candidates.append((decision, target_quote.book.casefold(), score))
        if not candidates:
            insufficient_four_book_price += 1
            continue
        candidates.sort(
            key=lambda item: (
                0 if item[0].paper_action != "pass" else 1,
                -item[2],
                item[1],
                item[0].decision_id,
            )
        )
        selected = candidates[0][0]
        decisions.append(selected)
        settlements.append(
            settle_paper_decision(
                selected,
                target=snapshot.fight.target,
                settled_at_utc=datetime.combine(
                    snapshot.fight.event_date + timedelta(days=1),
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                ),
                result_source_sha256=result_source_sha256,
                fight_id=snapshot.fight.fight_id,
            )
        )
    decisions.sort(key=lambda item: (item.decision_issued_at_utc, item.decision_id))
    settlements.sort(key=lambda item: (item.settled_at_utc, item.settlement_id))
    paper_metrics = summarize_paper_settlements(decisions, settlements)
    return_uncertainty = _paper_return_uncertainty(decisions, settlements)
    selection_diagnostics = _paper_selection_diagnostics(
        decisions, settlements, quotes_by_key
    )
    if fixed_gamma == 0.0:
        selection_diagnostics = {
            **selection_diagnostics,
            "model_probability_used_in_decision": False,
            "diagnosis": (
                "fixed gamma zero removes the legacy model from every probability. "
                "Selections therefore arise only from same-capture cross-book price "
                "dispersion after excluding the target book. Historical quote "
                "freshness and executable availability remain unverified."
            ),
        }
    return (
        tuple(decisions),
        tuple(settlements),
        {
            "evaluated_fights_considered": evaluation.evaluated_fights,
            "candidate_target_book_prices_with_three_other_books": candidate_target_prices,
            "fights_without_a_four_book_price": insufficient_four_book_price,
            "one_deterministic_decision_per_eligible_fight": len(decisions),
            "paper_selections": paper_metrics.paper_selections,
            "passes": paper_metrics.passes,
            "voids": paper_metrics.voids,
            "settled_selection_outcomes": {
                "wins": paper_metrics.wins,
                "losses": paper_metrics.losses,
            },
            "selection_coverage": paper_metrics.selection_coverage,
            "selection_diagnostics": selection_diagnostics,
            "hypothetical_quoted_price_return_metrics": {
                "label": "hypothetical_flat_1u_at_listed_quote_not_realized_or_executable",
                "selections": paper_metrics.paper_selections,
                "risk_units": paper_metrics.hypothetical_risk_units,
                "profit_units": paper_metrics.hypothetical_profit_units,
                "profit_per_bet": paper_metrics.hypothetical_roi,
                "hypothetical_roi": paper_metrics.hypothetical_roi,
                "max_drawdown_units": paper_metrics.hypothetical_max_drawdown_units,
                "event_block_uncertainty": return_uncertainty,
            },
            "forecast_metrics_for_all_decisions": (
                paper_metrics.forecast_metrics.to_mapping()
                if paper_metrics.forecast_metrics
                else None
            ),
            "execution_return_metrics": {
                "reportable": False,
                "roi": None,
                "profit_units": None,
                "max_drawdown_units": None,
                "reasons": list(EXECUTION_BLOCKERS),
            },
        },
    )


def _paper_return_uncertainty(
    decisions: Sequence[PaperDecision],
    settlements: Sequence[PaperSettlement],
) -> dict[str, object] | None:
    """Event-block interval for the explicitly hypothetical flat-unit ledger."""

    decision_by_id = {item.decision_id: item for item in decisions}
    selected = [
        item
        for item in settlements
        if item.hypothetical_risk_units > 0.0
        and item.decision_id in decision_by_id
    ]
    if not selected:
        return None
    blocks: dict[str, list[PaperSettlement]] = {}
    for settlement in selected:
        event_id = decision_by_id[settlement.decision_id].event_id
        blocks.setdefault(event_id, []).append(settlement)
    block_ids = sorted(blocks)
    rng = random.Random(BOOTSTRAP_SEED + 1)
    profit_samples: list[float] = []
    per_bet_samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_ids = [rng.choice(block_ids) for _ in block_ids]
        sampled = [item for event_id in sampled_ids for item in blocks[event_id]]
        profit = sum(item.hypothetical_profit_units for item in sampled)
        risk = sum(item.hypothetical_risk_units for item in sampled)
        profit_samples.append(profit)
        per_bet_samples.append(profit / risk)
    return {
        "method": "nonparametric_event_block_bootstrap",
        "unit": "selection_bearing_event_id_card",
        "seed": BOOTSTRAP_SEED + 1,
        "replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": 0.95,
        "event_blocks": len(block_ids),
        "selections": len(selected),
        "hypothetical_profit_units_percentile_interval": [
            _quantile(profit_samples, 0.025),
            _quantile(profit_samples, 0.975),
        ],
        "hypothetical_profit_per_bet_percentile_interval": [
            _quantile(per_bet_samples, 0.025),
            _quantile(per_bet_samples, 0.975),
        ],
        "warning": (
            "quoted-price counterfactual only; interval excludes execution, limits, "
            "availability, and selection-process uncertainty"
        ),
    }


def _lineage_audit(revisions: Sequence[GitRevision]) -> dict[str, object]:
    return {
        "commits": len(revisions),
        "unique_blob_oids": len({item.blob_oid for item in revisions}),
        "first_observed_at_utc": _utc_text(min(item.observed_at_utc for item in revisions)),
        "last_observed_at_utc": _utc_text(max(item.observed_at_utc for item in revisions)),
        "author_committer_timestamp_mismatches": sum(
            item.author_at_utc != item.committer_at_utc for item in revisions
        ),
        "paths_observed": sorted({item.path for item in revisions}),
    }


def _snapshot_audit(snapshots: Sequence[MappedSnapshot]) -> dict[str, object]:
    all_quote_observations = sum(len(item.quotes) for item in snapshots)
    ex_ref_observations = sum(
        book.casefold() != "ref" for item in snapshots for book in item.quotes
    )
    all_tuples = {
        _canonical_quote_tuple(
            item.fight.fight_id,
            item.source_fighter_id,
            item.source_opponent_id,
            book,
            lines,
        )
        for item in snapshots
        for book, lines in item.quotes.items()
    }
    capture_signatures: dict[tuple[str, str], str] = {}
    grouped: dict[tuple[str, str], list[MappedSnapshot]] = {}
    for item in snapshots:
        grouped.setdefault((item.revision.commit_sha, item.fight.event_id), []).append(item)
    for key, group in grouped.items():
        capture_signatures[key] = _canonical_hash(
            sorted(_snapshot_signature(item) for item in group)
        )
    return {
        "strict_fight_snapshot_rows": len(snapshots),
        "unique_commit_event_card_snapshots": len(grouped),
        "distinct_card_price_signatures": len(set(capture_signatures.values())),
        "events": len({item.fight.event_id for item in snapshots}),
        "quote_observations_including_ref": all_quote_observations,
        "quote_observations_excluding_ref": ex_ref_observations,
        "canonical_price_tuples_including_ref": len(all_tuples),
        "canonical_price_tuples_excluding_ref": len(
            {item for item in all_tuples if item[1] != "ref"}
        ),
        "unique_fights_with_any_strict_quote": len(
            {item.fight.fight_id for item in snapshots}
        ),
    }


def _latest_audit(latest: Sequence[MappedSnapshot]) -> dict[str, object]:
    outcomes = [item for item in latest if item.fight.target is not None]
    voids = [item for item in latest if item.fight.target is None]
    years: dict[str, int] = {}
    cards_by_year: dict[str, set[str]] = {}
    leads: dict[str, int] = {}
    for item in outcomes:
        year = str(item.fight.event_date.year)
        years[year] = years.get(year, 0) + 1
        cards_by_year.setdefault(year, set()).add(item.fight.event_id)
        lead = (item.fight.event_date - item.revision.observed_at_utc.date()).days
        bucket = _lead_bucket(lead)
        leads[bucket] = leads.get(bucket, 0) + 1
    return {
        "minimum_core_books": MIN_CONSENSUS_BOOKS,
        "latest_eligible_outcomes": len(latest),
        "w_l_outcomes": len(outcomes),
        "draw_nc_outcomes": len(voids),
        "w_l_by_year": dict(sorted(years.items())),
        "cards_by_year": {
            year: len(events) for year, events in sorted(cards_by_year.items())
        },
        "commit_to_event_lead_buckets": dict(sorted(leads.items())),
        "lead_bucket_warning": (
            "commit-to-event upper bounds are not quote ages or T-minus execution times"
        ),
        "first_event_date": min(item.fight.event_date for item in latest).isoformat(),
        "last_event_date": max(item.fight.event_date for item in latest).isoformat(),
    }


def _preflight_exact(store: object, records: Sequence[object], id_field: str) -> None:
    existing = store.read()
    generated = {getattr(item, id_field): item for item in records}
    for item in existing:
        identifier = getattr(item, id_field)
        if identifier not in generated:
            raise RuntimeError(
                f"output store has an extraneous {id_field} not produced by this backfill: "
                f"{identifier}"
            )
        if generated[identifier] != item:
            raise RuntimeError(f"output store conflicts at {id_field}={identifier}")


def _append_and_verify(store: object, records: Sequence[object], id_field: str) -> str:
    result = store.append(records)
    loaded = store.read()
    expected = {getattr(item, id_field): item for item in records}
    actual = {getattr(item, id_field): item for item in loaded}
    if expected != actual:
        raise RuntimeError(f"{id_field} store failed exact post-write verification")
    return result.dataset_sha256


def _dataset_hash(records: Sequence[object]) -> str:
    return _canonical_hash([item.to_mapping() for item in records])


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run_backfill(repo_root: Path, output_dir: Path, *, dry_run: bool) -> dict[str, object]:
    repo_root = repo_root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else repo_root / output_dir
    output_dir = output_dir.resolve()
    if not (repo_root / ".git").exists():
        raise ValueError(f"--repo-root is not a Git working tree: {repo_root}")
    if not (repo_root / RAW_FIGHTS_PATH).is_file():
        raise ValueError(f"raw fight source is missing: {RAW_FIGHTS_PATH}")

    fights, raw_sha256, raw_audit = _load_raw_fights(repo_root)
    matcher = _FightMatcher(fights)
    vegas_revisions = _load_lineage(repo_root, VEGAS_PATHS)
    history_revisions = _load_lineage(repo_root, HISTORY_PATHS)
    snapshots, mapping_counts = _map_vegas_lineage(vegas_revisions, matcher)
    quotes, forecasts, quotes_by_key, forecast_by_key = _build_market_records(
        snapshots, fights
    )
    latest = _latest_eligible_snapshots(snapshots)
    observations, sample_audit, observation_context = _build_observations(
        latest, quotes_by_key, forecast_by_key
    )
    if not observations:
        raise RuntimeError("no conservative blend observations were recovered")
    evaluation = PriorCardBlendEvaluator().evaluate(observations)
    decisions, settlements, paper_report = _build_paper_records(
        evaluation, observation_context, quotes_by_key, raw_sha256
    )
    (
        market_only_decisions,
        market_only_settlements,
        market_only_paper_report,
    ) = _build_paper_records(
        evaluation,
        observation_context,
        quotes_by_key,
        raw_sha256,
        fixed_gamma=0.0,
    )
    blended_eligibility = {
        (item.capture_id, item.matchup_id) for item in decisions
    }
    market_only_eligibility = {
        (item.capture_id, item.matchup_id) for item in market_only_decisions
    }
    if blended_eligibility != market_only_eligibility:
        raise RuntimeError(
            "market-only comparator did not reuse the blended paper fight eligibility"
        )
    if any(item.selected_gamma != 0.0 for item in market_only_decisions):
        raise RuntimeError("market-only comparator emitted a nonzero gamma")
    if (
        paper_report["candidate_target_book_prices_with_three_other_books"]
        != market_only_paper_report[
            "candidate_target_book_prices_with_three_other_books"
        ]
    ):
        raise RuntimeError(
            "market-only comparator did not reuse the identical target-price universe"
        )
    blended_reference_by_fight = {
        (item.capture_id, item.matchup_id): item.reference_quote_id
        for item in decisions
    }
    market_only_matching_reference_prices = sum(
        blended_reference_by_fight[(item.capture_id, item.matchup_id)]
        == item.reference_quote_id
        for item in market_only_decisions
    )
    market_only_comparator_report: dict[str, object] = {
        "persisted": False,
        "report_only": True,
        "gamma_policy": {
            "policy": "fixed_market_only",
            "gamma": 0.0,
            "selected_before_all_target_fights": True,
            "legacy_model_probability_used": False,
        },
        "comparison_contract": {
            "same_evaluated_fights": True,
            "same_four_book_price_eligibility": True,
            "same_candidate_target_prices": True,
            "same_leave_target_book_out_rule": True,
            "same_minimum_expected_return": PAPER_MINIMUM_EXPECTED_RETURN,
            "same_price_selection_rule": True,
            "same_event_block_uncertainty_method": True,
            "eligible_fights": len(market_only_decisions),
            "selected_reference_price_matches_blended": (
                market_only_matching_reference_prices
            ),
        },
        "report_only_dataset_sha256": {
            "decisions": _dataset_hash(market_only_decisions),
            "settlements": _dataset_hash(market_only_settlements),
        },
        **market_only_paper_report,
    }

    quote_hash = _dataset_hash(quotes)
    forecast_hash = _dataset_hash(forecasts)
    decision_hash = _dataset_hash(decisions)
    settlement_hash = _dataset_hash(settlements)
    exploratory_report: dict[str, object] = {
        "report_schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "betting_status": BETTING_STATUS,
        "paper_only": True,
        "promotable": False,
        "non_promotable_flags": list(NON_PROMOTABLE_FLAGS),
        "sample": sample_audit,
        "walk_forward_blend": {
            "evaluation_id": evaluation.evaluation_id,
            "input_sha256": evaluation.input_sha256,
            "gamma_grid": list(evaluation.gamma_grid),
            "selection_rule": "prior_completed_cards_only",
            "min_prior_cards": evaluation.min_prior_cards,
            "min_prior_fights": evaluation.min_prior_fights,
            "lookback_cards": evaluation.lookback_cards,
            "evaluated_fights": evaluation.evaluated_fights,
            "skipped_fights": evaluation.skipped_fights,
            "market_metrics": (
                evaluation.market_metrics.to_mapping()
                if evaluation.market_metrics
                else None
            ),
            "model_metrics": (
                evaluation.model_metrics.to_mapping()
                if evaluation.model_metrics
                else None
            ),
            "blend_metrics": (
                evaluation.blend_metrics.to_mapping()
                if evaluation.blend_metrics
                else None
            ),
            "per_year_metrics": _year_metrics(evaluation.predictions),
            "selected_gamma_distribution": _gamma_distribution(
                evaluation.predictions
            ),
            "event_block_uncertainty": _event_block_uncertainty(
                evaluation.predictions
            ),
        },
        "paper_decision_exploration": {
            "core_books": list(CORE_BOOKS),
            "reference_book_excluded_from_its_consensus": True,
            "minimum_other_books": MIN_CONSENSUS_BOOKS,
            "minimum_expected_return": PAPER_MINIMUM_EXPECTED_RETURN,
            "selection_rule": (
                "one price per fight: qualifying action first, then highest expected "
                "return, then lexical book key"
            ),
            **paper_report,
        },
        "market_only_paper_comparator": market_only_comparator_report,
        "record_dataset_sha256": {
            "quotes": quote_hash,
            "forecasts": forecast_hash,
            "paper_decisions": decision_hash,
            "paper_settlements": settlement_hash,
        },
    }
    report_sha256 = sha256(
        (json.dumps(exploratory_report, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    audit: dict[str, object] = {
        "audit_schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "source_ref": SOURCE_REF,
        "betting_status": BETTING_STATUS,
        "dry_run": dry_run,
        "core_books": list(CORE_BOOKS),
        "ref_excluded": True,
        "source_paths": {
            "vegas_odds": list(VEGAS_PATHS),
            "prediction_history": list(HISTORY_PATHS),
            "raw_fights": RAW_FIGHTS_PATH,
        },
        "conservative_rules": {
            "observed_bound": "max(author_timestamp,committer_timestamp)_utc",
            "dated_mapping": "exact_event_date_only",
            "undated_mapping": "unique_name_pair_and_1_to_30_utc_calendar_days",
            "name_matching": "explicit_aliases_plus_order_insensitive_same_name_no_fuzzy_distance",
            "date_only_timing": "commit_utc_date_strictly_before_event_date",
            "moneyline": "paired_integer_abs_ge_100_and_overround_0.90_to_1.30",
            "duplicate_rule": "one_identical_row_per_commit_fight_conflicts_rejected",
            "snapshot_selection": "latest_eligible_before_event_without_result",
        },
        "lineage": {
            "vegas_odds": _lineage_audit(vegas_revisions),
            "prediction_history": _lineage_audit(history_revisions),
        },
        "raw_fights": {**raw_audit, "source_sha256": raw_sha256},
        "vegas_mapping": mapping_counts,
        "strict_snapshot_dataset": _snapshot_audit(snapshots),
        "latest_core_five_sample": _latest_audit(latest),
        "prediction_history_feasibility": _history_audit(
            history_revisions, matcher, snapshots
        ),
        "generated_records": {
            "quotes": len(quotes),
            "forecasts": len(forecasts),
            "paper_decisions": len(decisions),
            "paper_settlements": len(settlements),
            "dataset_sha256": {
                "quotes": quote_hash,
                "forecasts": forecast_hash,
                "paper_decisions": decision_hash,
                "paper_settlements": settlement_hash,
            },
        },
        "report_only_market_only_comparator": market_only_comparator_report,
        "exploratory_report_sha256": report_sha256,
        "non_promotable_flags": list(NON_PROMOTABLE_FLAGS),
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        quote_store = QuoteSnapshotStore(
            output_dir / "market_quotes.csv", output_dir / "market_quotes.jsonl"
        )
        forecast_store = ForecastCaptureStore(
            output_dir / "legacy_forecasts.csv",
            output_dir / "legacy_forecasts.jsonl",
        )
        decision_store = PaperDecisionStore(
            output_dir / "paper_decisions.csv",
            output_dir / "paper_decisions.jsonl",
        )
        settlement_store = PaperSettlementStore(
            output_dir / "paper_settlements.csv",
            output_dir / "paper_settlements.jsonl",
        )
        stores = (
            (quote_store, quotes, "quote_id"),
            (forecast_store, forecasts, "forecast_capture_id"),
            (decision_store, decisions, "decision_id"),
            (settlement_store, settlements, "settlement_id"),
        )
        for store, records, id_field in stores:
            _preflight_exact(store, records, id_field)
        persisted_hashes = {
            id_field: _append_and_verify(store, records, id_field)
            for store, records, id_field in stores
        }
        expected_hashes = {
            "quote_id": quote_hash,
            "forecast_capture_id": forecast_hash,
            "decision_id": decision_hash,
            "settlement_id": settlement_hash,
        }
        if persisted_hashes != expected_hashes:
            raise RuntimeError(
                f"persisted dataset hashes disagree: {persisted_hashes} != {expected_hashes}"
            )
        _atomic_write_json(
            output_dir / "exploratory_blend_paper_report.json", exploratory_report
        )
        _atomic_write_json(output_dir / "backfill_audit.json", audit)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository working-tree root (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/content/data/market_history_backfill"),
        help="output directory, relative to repo root unless absolute",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform the complete reconstruction and audit without writing files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    audit = run_backfill(
        arguments.repo_root,
        arguments.output_dir,
        dry_run=arguments.dry_run,
    )
    summary = {
        "dry_run": arguments.dry_run,
        "strict_fight_snapshot_rows": audit["strict_snapshot_dataset"][
            "strict_fight_snapshot_rows"
        ],
        "latest_core_five_w_l": audit["latest_core_five_sample"]["w_l_outcomes"],
        "generated_records": audit["generated_records"],
        "report_only_market_only_comparator": {
            "gamma_policy": audit["report_only_market_only_comparator"][
                "gamma_policy"
            ],
            "eligible_fights": audit["report_only_market_only_comparator"][
                "one_deterministic_decision_per_eligible_fight"
            ],
            "paper_selections": audit["report_only_market_only_comparator"][
                "paper_selections"
            ],
            "passes": audit["report_only_market_only_comparator"]["passes"],
            "settled_selection_outcomes": audit[
                "report_only_market_only_comparator"
            ]["settled_selection_outcomes"],
            "hypothetical_quoted_price_return_metrics": audit[
                "report_only_market_only_comparator"
            ]["hypothetical_quoted_price_return_metrics"],
        },
        "output_dir": str(arguments.output_dir),
        "promotable": False,
        "betting_status": BETTING_STATUS,
    }
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
