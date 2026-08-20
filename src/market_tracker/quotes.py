"""Immutable quote captures, deterministic consensus, and atomic storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import csv
import io
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from ._common import (
    MarketDataError,
    SCHEMA_VERSION,
    StoreIntegrityError,
    canonical_hash,
    canonical_json,
    canonical_pair,
    implied_probability,
    matchup_id_for,
    moneyline,
    nonempty_text,
    optional_stable_id,
    payload_hash,
    probability,
    require_before_event,
    stable_id,
    utc_datetime,
    utc_text,
    validated_sha256,
)
from ._storage import atomic_write_text, exclusive_store_lock


MIN_OVERROUND = 0.90
MAX_OVERROUND = 1.30


def _audit_name(value: object) -> str:
    """Normalize an optional display name; it is never used as identity."""

    return " ".join(str(value or "").split())


@dataclass(frozen=True)
class QuoteSnapshot:
    """One freshly retrieved two-sided book quote.

    ``matchup_id`` is deterministically derived from event and fighter IDs.
    ``fight_id`` remains nullable because UFCStats does not assign it to an
    upcoming bout. Display names are audit-only. An unchanged line captured
    again uses a new ``capture_id`` and fresh ``observed_at_utc`` and therefore
    has a new quote ID; its
    separate ``quote_first_seen_at_utc`` may point to the earlier observation.
    """

    schema_version: int
    quote_id: str
    capture_id: str
    matchup_id: str
    fight_id: str | None
    event_id: str
    fighter_id: str
    opponent_id: str
    fighter_name: str
    opponent_name: str
    event_date: str
    timing_precision: str
    event_start_utc: str | None
    observed_at_utc: str
    quote_first_seen_at_utc: str
    source: str
    book: str
    fighter_moneyline: int
    opponent_moneyline: int
    fighter_implied_probability: float
    opponent_implied_probability: float
    overround: float
    no_vig_fighter_probability: float
    source_payload_sha256: str

    FIELDNAMES = (
        "schema_version",
        "quote_id",
        "capture_id",
        "matchup_id",
        "fight_id",
        "event_id",
        "fighter_id",
        "opponent_id",
        "fighter_name",
        "opponent_name",
        "event_date",
        "timing_precision",
        "event_start_utc",
        "observed_at_utc",
        "quote_first_seen_at_utc",
        "source",
        "book",
        "fighter_moneyline",
        "opponent_moneyline",
        "fighter_implied_probability",
        "opponent_implied_probability",
        "overround",
        "no_vig_fighter_probability",
        "source_payload_sha256",
    )

    @classmethod
    def create(
        cls,
        *,
        capture_id: object,
        event_id: object,
        fighter_id: object,
        opponent_id: object,
        event_date: date | str,
        timing_precision: str,
        event_start_utc: datetime | str | None,
        observed_at_utc: datetime | str,
        source: object,
        book: object,
        fighter_moneyline: object,
        opponent_moneyline: object,
        matchup_id: object | None = None,
        fight_id: object | None = None,
        fighter_name: object = "",
        opponent_name: object = "",
        quote_first_seen_at_utc: datetime | str | None = None,
        source_payload: bytes | str | Mapping | list | None = None,
        source_payload_sha256: str | None = None,
    ) -> "QuoteSnapshot":
        if (source_payload is None) == (source_payload_sha256 is None):
            raise MarketDataError(
                "provide exactly one of source_payload or source_payload_sha256"
            )
        fighter_line = moneyline(fighter_moneyline, "fighter_moneyline")
        opponent_line = moneyline(opponent_moneyline, "opponent_moneyline")
        fighter, opponent, fighter_side, opponent_side, _ = canonical_pair(
            fighter_id,
            opponent_id,
            (fighter_line, _audit_name(fighter_name)),
            (opponent_line, _audit_name(opponent_name)),
        )
        fighter_line, fighter_display = fighter_side
        opponent_line, opponent_display = opponent_side
        event = stable_id(event_id, "event_id")
        derived_matchup_id = matchup_id_for(event, fighter, opponent)
        if matchup_id is not None and str(matchup_id).strip():
            supplied_matchup_id = stable_id(matchup_id, "matchup_id")
            if supplied_matchup_id != derived_matchup_id:
                raise MarketDataError(
                    "matchup_id does not match event_id and canonical fighter IDs"
                )
        observed, event_day, precision, event_start = require_before_event(
            observed_at_utc,
            event_date=event_date,
            timing_precision=timing_precision,
            event_start_utc=event_start_utc,
            observed_field="observed_at_utc",
        )
        first_seen_value = (
            observed if quote_first_seen_at_utc is None else quote_first_seen_at_utc
        )
        first_seen, _, _, _ = require_before_event(
            first_seen_value,
            event_date=event_day,
            timing_precision=precision,
            event_start_utc=event_start,
            observed_field="quote_first_seen_at_utc",
        )
        if first_seen > observed:
            raise MarketDataError(
                "quote_first_seen_at_utc must not be later than observed_at_utc"
            )
        fighter_implied = implied_probability(fighter_line)
        opponent_implied = implied_probability(opponent_line)
        overround = fighter_implied + opponent_implied
        if not MIN_OVERROUND <= overround <= MAX_OVERROUND:
            raise MarketDataError(
                f"quote overround {overround:.6f} is outside "
                f"[{MIN_OVERROUND}, {MAX_OVERROUND}]"
            )
        digest = (
            payload_hash(source_payload)
            if source_payload is not None
            else validated_sha256(source_payload_sha256, "source_payload_sha256")
        )
        body = {
            "schema_version": SCHEMA_VERSION,
            "capture_id": stable_id(capture_id, "capture_id"),
            "matchup_id": derived_matchup_id,
            "fight_id": optional_stable_id(fight_id, "fight_id"),
            "event_id": event,
            "fighter_id": fighter,
            "opponent_id": opponent,
            "fighter_name": fighter_display,
            "opponent_name": opponent_display,
            "event_date": event_day,
            "timing_precision": precision,
            "event_start_utc": event_start,
            "observed_at_utc": utc_text(observed, "observed_at_utc"),
            "quote_first_seen_at_utc": utc_text(
                first_seen, "quote_first_seen_at_utc"
            ),
            "source": nonempty_text(source, "source"),
            "book": nonempty_text(book, "book"),
            "fighter_moneyline": fighter_line,
            "opponent_moneyline": opponent_line,
            "fighter_implied_probability": fighter_implied,
            "opponent_implied_probability": opponent_implied,
            "overround": overround,
            "no_vig_fighter_probability": fighter_implied / overround,
            "source_payload_sha256": digest,
        }
        return cls(quote_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "QuoteSnapshot":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        if missing:
            raise MarketDataError(f"quote snapshot is missing fields: {missing}")
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if extra:
            raise MarketDataError(f"quote snapshot has unexpected fields: {extra}")
        try:
            schema_version = int(record["schema_version"])
        except (TypeError, ValueError) as error:
            raise MarketDataError("invalid quote snapshot schema version") from error
        if schema_version != SCHEMA_VERSION:
            raise MarketDataError("unsupported quote snapshot schema version")
        rebuilt = cls.create(
            capture_id=record["capture_id"],
            matchup_id=record["matchup_id"],
            fight_id=record["fight_id"],
            event_id=record["event_id"],
            fighter_id=record["fighter_id"],
            opponent_id=record["opponent_id"],
            fighter_name=record["fighter_name"],
            opponent_name=record["opponent_name"],
            event_date=record["event_date"],
            timing_precision=str(record["timing_precision"]),
            event_start_utc=record["event_start_utc"],
            observed_at_utc=record["observed_at_utc"],
            quote_first_seen_at_utc=record["quote_first_seen_at_utc"],
            source=record["source"],
            book=record["book"],
            fighter_moneyline=record["fighter_moneyline"],
            opponent_moneyline=record["opponent_moneyline"],
            source_payload_sha256=str(record["source_payload_sha256"]),
        )
        if str(record["quote_id"]) != rebuilt.quote_id:
            raise MarketDataError("quote_id does not match canonical snapshot contents")
        try:
            supplied = (
                float(record["fighter_implied_probability"]),
                float(record["opponent_implied_probability"]),
                float(record["overround"]),
                float(record["no_vig_fighter_probability"]),
            )
        except (TypeError, ValueError) as error:
            raise MarketDataError("stored quote probabilities must be numeric") from error
        expected = (
            rebuilt.fighter_implied_probability,
            rebuilt.opponent_implied_probability,
            rebuilt.overround,
            rebuilt.no_vig_fighter_probability,
        )
        if any(not math.isfinite(value) for value in supplied) or any(
            abs(left - right) > 1e-12 for left, right in zip(supplied, expected)
        ):
            raise MarketDataError("stored quote probabilities disagree with its moneylines")
        return rebuilt

    @property
    def natural_key(self) -> tuple[str, str, str]:
        return (
            self.matchup_id,
            self.capture_id,
            self.book.casefold(),
        )

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


@dataclass(frozen=True)
class AppendResult:
    added_ids: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    total_records: int
    dataset_sha256: str


@dataclass(frozen=True)
class MarketConsensus:
    schema_version: int
    consensus_id: str
    capture_id: str
    matchup_id: str
    fight_id: str | None
    event_id: str
    fighter_id: str
    opponent_id: str
    event_date: str
    timing_precision: str
    event_start_utc: str | None
    as_of_utc: str
    latest_observed_at_utc: str
    no_vig_fighter_probability: float
    book_count: int
    included_book_keys: tuple[str, ...]
    excluded_book_keys: tuple[str, ...]
    quote_ids: tuple[str, ...]
    quote_dataset_sha256: str

    @property
    def no_vig_opponent_probability(self) -> float:
        return 1.0 - self.no_vig_fighter_probability

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


class QuoteSnapshotStore:
    """Append-only logical store with atomic JSONL and CSV mirrors.

    Existing quote IDs and capture natural keys are immutable. A crash between
    mirror replacements leaves one strict superset; the next append recovers
    it and rewrites both mirrors. Divergent mirrors fail closed.
    """

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        if self.csv_path.resolve() == self.jsonl_path.resolve():
            raise ValueError("csv_path and jsonl_path must be different")

    @staticmethod
    def _records_by_id(records: Iterable[QuoteSnapshot]) -> dict[str, QuoteSnapshot]:
        indexed: dict[str, QuoteSnapshot] = {}
        natural: dict[tuple[str, str, str], str] = {}
        capture_identity: dict[
            str, tuple[str, str, str, str | None, str]
        ] = {}
        capture_source_payload: dict[tuple[str, str], str] = {}
        for record in records:
            existing = indexed.get(record.quote_id)
            if existing is not None and existing != record:
                raise StoreIntegrityError(
                    f"duplicate quote_id has different data: {record.quote_id}"
                )
            prior_id = natural.get(record.natural_key)
            if prior_id is not None and prior_id != record.quote_id:
                raise StoreIntegrityError(
                    "an existing matchup/capture/book key was rewritten"
                )
            indexed[record.quote_id] = record
            natural[record.natural_key] = record.quote_id
            identity = (
                record.event_id,
                record.event_date,
                record.timing_precision,
                record.event_start_utc,
                record.observed_at_utc,
            )
            prior_identity = capture_identity.setdefault(record.capture_id, identity)
            if prior_identity != identity:
                raise StoreIntegrityError(
                    "one capture_id must identify one event and retrieval timestamp"
                )
            source_key = (record.capture_id, record.source.casefold())
            prior_payload = capture_source_payload.setdefault(
                source_key, record.source_payload_sha256
            )
            if prior_payload != record.source_payload_sha256:
                raise StoreIntegrityError(
                    "one capture/source must identify one retrieved payload"
                )
        return indexed

    def _read_jsonl(self) -> list[QuoteSnapshot]:
        if not self.jsonl_path.exists():
            return []
        records: list[QuoteSnapshot] = []
        with self.jsonl_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise StoreIntegrityError(f"blank JSONL record at line {line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StoreIntegrityError(
                        f"invalid JSONL record at line {line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise StoreIntegrityError(f"JSONL line {line_number} is not an object")
                records.append(QuoteSnapshot.from_mapping(value))
        self._records_by_id(records)
        return records

    def _read_csv(self) -> list[QuoteSnapshot]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != QuoteSnapshot.FIELDNAMES:
                raise StoreIntegrityError("quote CSV columns or order do not match the schema")
            records = [QuoteSnapshot.from_mapping(row) for row in reader]
        self._records_by_id(records)
        return records

    def read(self) -> tuple[QuoteSnapshot, ...]:
        jsonl_records = self._read_jsonl()
        csv_records = self._read_csv()
        if not jsonl_records:
            return tuple(csv_records)
        if not csv_records:
            return tuple(jsonl_records)
        jsonl_index = self._records_by_id(jsonl_records)
        csv_index = self._records_by_id(csv_records)
        common = set(jsonl_index) & set(csv_index)
        if any(jsonl_index[key] != csv_index[key] for key in common):
            raise StoreIntegrityError("CSV and JSONL contain different data for a quote_id")
        jsonl_ids = [record.quote_id for record in jsonl_records]
        csv_ids = [record.quote_id for record in csv_records]
        if set(csv_index) == set(jsonl_index):
            if csv_ids != jsonl_ids:
                raise StoreIntegrityError("CSV and JSONL quote order diverged")
            return tuple(jsonl_records)
        if set(csv_index) < set(jsonl_index) and csv_ids == jsonl_ids[: len(csv_ids)]:
            return tuple(jsonl_records)
        if set(jsonl_index) < set(csv_index) and jsonl_ids == csv_ids[: len(jsonl_ids)]:
            return tuple(csv_records)
        raise StoreIntegrityError("CSV and JSONL quote mirrors diverged")

    @staticmethod
    def dataset_sha256(records: Iterable[QuoteSnapshot]) -> str:
        return canonical_hash([record.to_mapping() for record in records])

    @staticmethod
    def _render_jsonl(records: Iterable[QuoteSnapshot]) -> str:
        return "".join(f"{canonical_json(record.to_mapping())}\n" for record in records)

    @staticmethod
    def _render_csv(records: Iterable[QuoteSnapshot]) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output, fieldnames=QuoteSnapshot.FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_mapping())
        return output.getvalue()

    def append(self, snapshots: Iterable[QuoteSnapshot]) -> AppendResult:
        pending = tuple(snapshots)
        lock_path = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock_path):
            existing = list(self.read())
            existing_index = self._records_by_id(existing)
            natural_index = {record.natural_key: record.quote_id for record in existing}
            additions: list[QuoteSnapshot] = []
            duplicates: list[str] = []
            for snapshot in pending:
                if not isinstance(snapshot, QuoteSnapshot):
                    raise TypeError("append accepts QuoteSnapshot instances only")
                if snapshot.quote_id in existing_index:
                    if existing_index[snapshot.quote_id] != snapshot:
                        raise StoreIntegrityError("an existing quote_id was rewritten")
                    duplicates.append(snapshot.quote_id)
                    continue
                prior_id = natural_index.get(snapshot.natural_key)
                if prior_id is not None and prior_id != snapshot.quote_id:
                    raise StoreIntegrityError(
                        "conflicting quote for the same matchup/capture/book"
                    )
                existing_index[snapshot.quote_id] = snapshot
                natural_index[snapshot.natural_key] = snapshot.quote_id
                additions.append(snapshot)
            additions.sort(key=lambda item: (item.observed_at_utc, item.quote_id))
            combined = [*existing, *additions]
            self._records_by_id(combined)
            # Each replacement is atomic. JSONL goes first and is the recovery
            # authority if interrupted before CSV replacement.
            atomic_write_text(self.jsonl_path, self._render_jsonl(combined))
            atomic_write_text(self.csv_path, self._render_csv(combined))
            return AppendResult(
                added_ids=tuple(item.quote_id for item in additions),
                duplicate_ids=tuple(duplicates),
                total_records=len(combined),
                dataset_sha256=self.dataset_sha256(combined),
            )


def consensus_as_of(
    snapshots: Iterable[QuoteSnapshot],
    *,
    capture_id: object,
    matchup_id: object,
    as_of_utc: datetime | str,
    min_books: int = 2,
    exclude_books: Iterable[object] = (),
) -> MarketConsensus:
    """Build a consensus from exactly one retrieval capture.

    ``exclude_books`` supports leave-one-book-out evaluation. Mixing latest
    quotes across capture runs is deliberately impossible.
    """

    if min_books < 1:
        raise ValueError("min_books must be at least one")
    requested_capture = stable_id(capture_id, "capture_id")
    requested_matchup = stable_id(matchup_id, "matchup_id")
    excluded_book_keys = tuple(
        sorted({nonempty_text(book, "exclude_book").casefold() for book in exclude_books})
    )
    candidates = [
        item
        for item in snapshots
        if item.matchup_id == requested_matchup
        and item.capture_id == requested_capture
    ]
    if not candidates:
        raise MarketDataError(
            "no quotes exist for the requested matchup_id and capture_id"
        )
    identity = {
        (
            item.event_id,
            item.fighter_id,
            item.opponent_id,
            item.event_date,
            item.timing_precision,
            item.event_start_utc,
        )
        for item in candidates
    }
    if len(identity) != 1:
        raise StoreIntegrityError("quotes for a matchup disagree on event or fighter identity")
    observed_times = {item.observed_at_utc for item in candidates}
    if len(observed_times) != 1:
        raise StoreIntegrityError(
            "one capture_id contains multiple quote retrieval timestamps"
        )
    source_payloads: dict[str, set[str]] = {}
    for item in candidates:
        source_payloads.setdefault(item.source.casefold(), set()).add(
            item.source_payload_sha256
        )
    if any(len(payloads) != 1 for payloads in source_payloads.values()):
        raise StoreIntegrityError(
            "one capture/source contains multiple retrieved payloads"
        )
    event_id, fighter, opponent, event_day, precision, event_start = next(iter(identity))
    as_of, _, _, _ = require_before_event(
        as_of_utc,
        event_date=event_day,
        timing_precision=precision,
        event_start_utc=event_start,
        observed_field="as_of_utc",
    )
    capture_observed = utc_datetime(next(iter(observed_times)), "observed_at_utc")
    if as_of != capture_observed:
        raise StoreIntegrityError(
            "market consensus as_of_utc must equal the capture retrieval timestamp"
        )
    eligible = [
        item
        for item in candidates
        if utc_datetime(item.observed_at_utc, "observed_at_utc") <= as_of
        and item.book.casefold() not in excluded_book_keys
    ]
    latest_by_book: dict[str, QuoteSnapshot] = {}
    for item in eligible:
        key = item.book.casefold()
        current = latest_by_book.get(key)
        if current is not None and current.quote_id != item.quote_id:
            raise StoreIntegrityError(
                "one capture contains multiple quotes for the same matchup/book"
            )
        latest_by_book[key] = item
    selected = sorted(
        latest_by_book.values(), key=lambda item: (item.book.casefold(), item.quote_id)
    )
    if len(selected) < min_books:
        raise MarketDataError(
            f"only {len(selected)} distinct books were available by the requested cutoff"
        )
    market_probability = sum(
        item.no_vig_fighter_probability for item in selected
    ) / len(selected)
    probability(market_probability, "no_vig_fighter_probability")
    nonnull_fight_ids = {item.fight_id for item in candidates if item.fight_id is not None}
    if len(nonnull_fight_ids) > 1:
        raise StoreIntegrityError("one matchup_id resolves to multiple fight IDs")
    quote_ids = tuple(item.quote_id for item in selected)
    body = {
        "schema_version": SCHEMA_VERSION,
        "capture_id": requested_capture,
        "matchup_id": requested_matchup,
        "fight_id": next(iter(nonnull_fight_ids), None),
        "event_id": event_id,
        "fighter_id": fighter,
        "opponent_id": opponent,
        "event_date": event_day,
        "timing_precision": precision,
        "event_start_utc": event_start,
        "as_of_utc": utc_text(as_of, "as_of_utc"),
        "latest_observed_at_utc": max(item.observed_at_utc for item in selected),
        "no_vig_fighter_probability": market_probability,
        "book_count": len(selected),
        "included_book_keys": tuple(item.book.casefold() for item in selected),
        "excluded_book_keys": excluded_book_keys,
        "quote_ids": quote_ids,
        "quote_dataset_sha256": canonical_hash(
            [item.to_mapping() for item in selected]
        ),
    }
    return MarketConsensus(consensus_id=canonical_hash(body), **body)
