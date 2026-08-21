"""Immutable source timing metadata paired one-to-one with quote snapshots."""

from __future__ import annotations

from dataclasses import dataclass
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
    nonempty_text,
    stable_id,
    utc_datetime,
    utc_text,
)
from ._storage import atomic_write_text, exclusive_store_lock
from .quotes import AppendResult, QuoteSnapshot


@dataclass(frozen=True)
class QuoteSourceMetadata:
    schema_version: int
    metadata_id: str
    quote_id: str
    capture_id: str
    matchup_id: str
    event_id: str
    source: str
    book: str
    source_book_key: str
    source_event_id: str
    source_quote_updated_at_utc: str
    source_commence_time_utc: str
    observed_at_utc: str
    source_quote_age_seconds: float

    FIELDNAMES = (
        "schema_version",
        "metadata_id",
        "quote_id",
        "capture_id",
        "matchup_id",
        "event_id",
        "source",
        "book",
        "source_book_key",
        "source_event_id",
        "source_quote_updated_at_utc",
        "source_commence_time_utc",
        "observed_at_utc",
        "source_quote_age_seconds",
    )

    @classmethod
    def create(
        cls,
        quote: QuoteSnapshot,
        *,
        source_book_key: object,
        source_event_id: object,
        source_quote_updated_at_utc: object,
        source_commence_time_utc: object,
    ) -> "QuoteSourceMetadata":
        if not isinstance(quote, QuoteSnapshot):
            raise TypeError("quote must be a QuoteSnapshot")
        observed = utc_datetime(quote.observed_at_utc, "observed_at_utc")
        updated = utc_datetime(
            source_quote_updated_at_utc, "source_quote_updated_at_utc"
        )
        commence = utc_datetime(
            source_commence_time_utc, "source_commence_time_utc"
        )
        age = (observed - updated).total_seconds()
        if age < -300.0:
            raise MarketDataError(
                "source quote update is implausibly later than its retrieval"
            )
        if not observed < commence:
            raise MarketDataError("source quote was not captured before commence time")
        body = {
            "schema_version": SCHEMA_VERSION,
            "quote_id": quote.quote_id,
            "capture_id": quote.capture_id,
            "matchup_id": quote.matchup_id,
            "event_id": quote.event_id,
            "source": quote.source,
            "book": quote.book,
            "source_book_key": stable_id(source_book_key, "source_book_key"),
            "source_event_id": stable_id(source_event_id, "source_event_id"),
            "source_quote_updated_at_utc": utc_text(
                updated, "source_quote_updated_at_utc"
            ),
            "source_commence_time_utc": utc_text(
                commence, "source_commence_time_utc"
            ),
            "observed_at_utc": quote.observed_at_utc,
            "source_quote_age_seconds": age,
        }
        return cls(metadata_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "QuoteSourceMetadata":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"quote source metadata schema mismatch; missing={missing}, extra={extra}"
            )
        try:
            item = cls(**{field: record[field] for field in cls.FIELDNAMES})
        except TypeError as error:
            raise MarketDataError("invalid quote source metadata fields") from error
        item.validate_integrity()
        return item

    @property
    def natural_key(self) -> tuple[str]:
        return (self.quote_id,)

    def validate_integrity(self) -> None:
        if int(self.schema_version) != SCHEMA_VERSION:
            raise MarketDataError("unsupported quote source metadata schema version")
        body = self.to_mapping()
        body.pop("metadata_id")
        if self.metadata_id != canonical_hash(body):
            raise StoreIntegrityError("metadata_id does not match canonical contents")
        for value, field in (
            (self.quote_id, "quote_id"),
            (self.capture_id, "capture_id"),
            (self.matchup_id, "matchup_id"),
            (self.event_id, "event_id"),
            (self.source_book_key, "source_book_key"),
            (self.source_event_id, "source_event_id"),
        ):
            stable_id(value, field)
        nonempty_text(self.source, "source")
        nonempty_text(self.book, "book")
        observed = utc_datetime(self.observed_at_utc, "observed_at_utc")
        updated = utc_datetime(
            self.source_quote_updated_at_utc, "source_quote_updated_at_utc"
        )
        commence = utc_datetime(
            self.source_commence_time_utc, "source_commence_time_utc"
        )
        try:
            age = float(self.source_quote_age_seconds)
        except (TypeError, ValueError) as error:
            raise MarketDataError("source_quote_age_seconds must be numeric") from error
        if not math.isfinite(age) or age < -300.0:
            raise MarketDataError("source_quote_age_seconds is invalid")
        if abs(age - (observed - updated).total_seconds()) > 1e-6:
            raise StoreIntegrityError("source quote age disagrees with its timestamps")
        if not observed < commence:
            raise StoreIntegrityError("source quote was not captured before commence time")

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


class QuoteSourceMetadataStore:
    """Atomic JSONL authority with an exactly mirrored CSV audit file."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        if self.csv_path.resolve() == self.jsonl_path.resolve():
            raise ValueError("csv_path and jsonl_path must be different")

    @staticmethod
    def _validate(
        records: Iterable[QuoteSourceMetadata],
    ) -> dict[str, QuoteSourceMetadata]:
        indexed: dict[str, QuoteSourceMetadata] = {}
        by_quote: dict[str, str] = {}
        for record in records:
            if not isinstance(record, QuoteSourceMetadata):
                raise TypeError("metadata store accepts QuoteSourceMetadata only")
            record.validate_integrity()
            existing = indexed.get(record.metadata_id)
            if existing is not None and existing != record:
                raise StoreIntegrityError("metadata_id was rewritten")
            prior = by_quote.get(record.quote_id)
            if prior is not None and prior != record.metadata_id:
                raise StoreIntegrityError("one quote has conflicting source metadata")
            indexed[record.metadata_id] = record
            by_quote[record.quote_id] = record.metadata_id
        return indexed

    def _read_jsonl(self) -> list[QuoteSourceMetadata]:
        if not self.jsonl_path.exists():
            return []
        records: list[QuoteSourceMetadata] = []
        with self.jsonl_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise StoreIntegrityError(
                        f"blank source metadata JSONL record at line {line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StoreIntegrityError(
                        f"invalid source metadata JSONL at line {line_number}"
                    ) from error
                if not isinstance(value, dict):
                    raise StoreIntegrityError("source metadata JSONL record is not an object")
                records.append(QuoteSourceMetadata.from_mapping(value))
        self._validate(records)
        return records

    def _read_csv_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != QuoteSourceMetadata.FIELDNAMES:
                raise StoreIntegrityError("source metadata CSV columns do not match")
            return list(reader)

    @staticmethod
    def _render_jsonl(records: Iterable[QuoteSourceMetadata]) -> str:
        return "".join(f"{canonical_json(item.to_mapping())}\n" for item in records)

    @staticmethod
    def _render_csv(records: Iterable[QuoteSourceMetadata]) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output, fieldnames=QuoteSourceMetadata.FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_mapping())
        return output.getvalue()

    def read(self) -> tuple[QuoteSourceMetadata, ...]:
        records = self._read_jsonl()
        csv_rows = self._read_csv_rows()
        if not records:
            if csv_rows:
                raise StoreIntegrityError(
                    "source metadata CSV has records but JSONL is missing"
                )
            return ()
        expected = list(
            csv.DictReader(io.StringIO(self._render_csv(records)))
        )
        if len(csv_rows) > len(expected) or csv_rows != expected[: len(csv_rows)]:
            raise StoreIntegrityError("source metadata CSV and JSONL mirrors diverged")
        return tuple(records)

    def append(self, pending: Iterable[QuoteSourceMetadata]) -> AppendResult:
        additions_input = tuple(pending)
        lock_path = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock_path):
            existing = list(self.read())
            indexed = self._validate(existing)
            by_quote = {item.quote_id: item.metadata_id for item in existing}
            additions: list[QuoteSourceMetadata] = []
            duplicates: list[str] = []
            for item in additions_input:
                if not isinstance(item, QuoteSourceMetadata):
                    raise TypeError("append accepts QuoteSourceMetadata instances only")
                item.validate_integrity()
                if item.metadata_id in indexed:
                    if indexed[item.metadata_id] != item:
                        raise StoreIntegrityError("metadata_id was rewritten")
                    duplicates.append(item.metadata_id)
                    continue
                if item.quote_id in by_quote:
                    raise StoreIntegrityError("one quote has conflicting source metadata")
                indexed[item.metadata_id] = item
                by_quote[item.quote_id] = item.metadata_id
                additions.append(item)
            additions.sort(key=lambda item: (item.observed_at_utc, item.metadata_id))
            combined = [*existing, *additions]
            self._validate(combined)
            atomic_write_text(self.jsonl_path, self._render_jsonl(combined))
            atomic_write_text(self.csv_path, self._render_csv(combined))
            return AppendResult(
                added_ids=tuple(item.metadata_id for item in additions),
                duplicate_ids=tuple(duplicates),
                total_records=len(combined),
                dataset_sha256=canonical_hash(
                    [item.to_mapping() for item in combined]
                ),
            )
