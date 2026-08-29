"""Immutable fighter-method sportsbook boards, including partial boards."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from ._common import (
    SCHEMA_VERSION,
    MarketDataError,
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
    require_before_event,
    stable_id,
    utc_text,
    validated_sha256,
)
from ._storage import atomic_write_text, exclusive_store_lock
from .quotes import AppendResult


METHODS = ("ko_tko", "submission", "decision")
METHOD_MARKET_CONTRACT = "fighter-method-book-board-v1"


def _audit_name(value: object) -> str:
    return " ".join(str(value or "").split())


@dataclass(frozen=True)
class MethodMarketSnapshot:
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
    source: str
    source_event_id: str
    source_book_key: str
    book: str
    horizon: str
    market: str
    contract_version: str
    selection_count: int
    is_complete_six_way: bool
    fighter_ko_tko_moneyline: int | None
    fighter_submission_moneyline: int | None
    fighter_decision_moneyline: int | None
    opponent_ko_tko_moneyline: int | None
    opponent_submission_moneyline: int | None
    opponent_decision_moneyline: int | None
    fighter_ko_tko_implied_probability: float | None
    fighter_submission_implied_probability: float | None
    fighter_decision_implied_probability: float | None
    opponent_ko_tko_implied_probability: float | None
    opponent_submission_implied_probability: float | None
    opponent_decision_implied_probability: float | None
    six_way_overround: float | None
    fighter_ko_tko_no_vig_probability: float | None
    fighter_submission_no_vig_probability: float | None
    fighter_decision_no_vig_probability: float | None
    opponent_ko_tko_no_vig_probability: float | None
    opponent_submission_no_vig_probability: float | None
    opponent_decision_no_vig_probability: float | None
    source_payload_sha256: str

    FIELDNAMES = tuple(__annotations__)

    @classmethod
    def create(
        cls,
        *,
        capture_id: object,
        event_id: object,
        fighter_id: object,
        opponent_id: object,
        fighter_name: object,
        opponent_name: object,
        event_date: object,
        timing_precision: object,
        event_start_utc: object,
        observed_at_utc: object,
        source: object,
        source_event_id: object,
        source_book_key: object,
        book: object,
        horizon: object,
        fighter_prices: Mapping[str, object],
        opponent_prices: Mapping[str, object],
        matchup_id: object | None = None,
        fight_id: object | None = None,
        source_payload: bytes | str | Mapping | list | None = None,
        source_payload_sha256: str | None = None,
    ) -> "MethodMarketSnapshot":
        if (source_payload is None) == (source_payload_sha256 is None):
            raise MarketDataError(
                "provide exactly one of source_payload or source_payload_sha256"
            )
        if not set(fighter_prices).issubset(METHODS) or not set(
            opponent_prices
        ).issubset(METHODS):
            raise MarketDataError("method prices contain an unsupported method")
        if not fighter_prices and not opponent_prices:
            raise MarketDataError("a method board requires at least one price")
        fighter, opponent, fighter_display, opponent_display, reversed_pair = canonical_pair(
            fighter_id,
            opponent_id,
            _audit_name(fighter_name),
            _audit_name(opponent_name),
        )
        first_prices = opponent_prices if reversed_pair else fighter_prices
        second_prices = fighter_prices if reversed_pair else opponent_prices
        event = stable_id(event_id, "event_id")
        derived_matchup = matchup_id_for(event, fighter, opponent)
        if matchup_id is not None and str(matchup_id).strip():
            if stable_id(matchup_id, "matchup_id") != derived_matchup:
                raise MarketDataError("matchup_id disagrees with event and fighter IDs")
        observed, event_day, precision, event_start = require_before_event(
            observed_at_utc,
            event_date=event_date,
            timing_precision=timing_precision,
            event_start_utc=event_start_utc,
            observed_field="observed_at_utc",
        )
        normalized_horizon = str(horizon or "").strip().casefold()
        if normalized_horizon not in {"opening", "t72", "t24", "t6"}:
            raise MarketDataError("unsupported method-price capture horizon")
        lines: dict[str, int | None] = {}
        for side, prices in (("fighter", first_prices), ("opponent", second_prices)):
            for method in METHODS:
                raw = prices.get(method)
                lines[f"{side}_{method}"] = (
                    moneyline(raw, f"{side}_{method}_moneyline")
                    if raw is not None and str(raw).strip()
                    else None
                )
        selection_count = sum(value is not None for value in lines.values())
        complete = selection_count == 6
        implied = {
            key: implied_probability(value) if value is not None else None
            for key, value in lines.items()
        }
        overround = (
            sum(value for value in implied.values() if value is not None)
            if complete
            else None
        )
        if overround is not None and not 0.8 <= overround <= 2.0:
            raise MarketDataError(
                f"six-way method overround {overround:.6f} is outside [0.8, 2.0]"
            )
        no_vig = {
            key: value / overround
            if value is not None and overround is not None
            else None
            for key, value in implied.items()
        }
        digest = (
            payload_hash(source_payload)
            if source_payload is not None
            else validated_sha256(source_payload_sha256, "source_payload_sha256")
        )
        body = {
            "schema_version": SCHEMA_VERSION,
            "capture_id": stable_id(capture_id, "capture_id"),
            "matchup_id": derived_matchup,
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
            "source": nonempty_text(source, "source"),
            "source_event_id": stable_id(source_event_id, "source_event_id"),
            "source_book_key": stable_id(source_book_key, "source_book_key"),
            "book": nonempty_text(book, "book"),
            "horizon": normalized_horizon,
            "market": "fighter_method_of_victory",
            "contract_version": METHOD_MARKET_CONTRACT,
            "selection_count": selection_count,
            "is_complete_six_way": complete,
            **{f"{key}_moneyline": value for key, value in lines.items()},
            **{
                f"{key}_implied_probability": value
                for key, value in implied.items()
            },
            "six_way_overround": overround,
            **{
                f"{key}_no_vig_probability": value
                for key, value in no_vig.items()
            },
            "source_payload_sha256": digest,
        }
        return cls(quote_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "MethodMarketSnapshot":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"method quote schema mismatch; missing={missing}, extra={extra}"
            )
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
            timing_precision=record["timing_precision"],
            event_start_utc=record["event_start_utc"],
            observed_at_utc=record["observed_at_utc"],
            source=record["source"],
            source_event_id=record["source_event_id"],
            source_book_key=record["source_book_key"],
            book=record["book"],
            horizon=record["horizon"],
            fighter_prices={
                method: record[f"fighter_{method}_moneyline"]
                for method in METHODS
                if record[f"fighter_{method}_moneyline"] not in (None, "")
            },
            opponent_prices={
                method: record[f"opponent_{method}_moneyline"]
                for method in METHODS
                if record[f"opponent_{method}_moneyline"] not in (None, "")
            },
            source_payload_sha256=str(record["source_payload_sha256"]),
        )
        if str(record["quote_id"]) != rebuilt.quote_id:
            raise MarketDataError("method quote_id does not match canonical contents")
        if int(record["schema_version"]) != SCHEMA_VERSION:
            raise MarketDataError("unsupported method quote schema version")
        if int(record["selection_count"]) != rebuilt.selection_count:
            raise MarketDataError("stored method selection_count disagrees with prices")
        supplied_complete = str(record["is_complete_six_way"]).strip().casefold()
        if supplied_complete not in {"true", "false"} or (
            supplied_complete == "true"
        ) != rebuilt.is_complete_six_way:
            raise MarketDataError("stored six-way completeness disagrees with prices")
        numeric_names = [
            field
            for field in cls.FIELDNAMES
            if field.endswith("_probability") or field == "six_way_overround"
        ]
        for field in numeric_names:
            rebuilt_value = getattr(rebuilt, field)
            if rebuilt_value is None:
                if record[field] not in (None, ""):
                    raise MarketDataError(f"stored {field} must be unavailable")
                continue
            try:
                supplied = float(record[field])
            except (TypeError, ValueError) as error:
                raise MarketDataError(f"stored {field} must be numeric") from error
            if not math.isfinite(supplied) or abs(supplied - rebuilt_value) > 1e-12:
                raise MarketDataError(f"stored {field} disagrees with method prices")
        if record["market"] != rebuilt.market or record["contract_version"] != rebuilt.contract_version:
            raise MarketDataError("stored method market contract is unsupported")
        return rebuilt

    @property
    def natural_key(self) -> tuple[str, str, str, str]:
        return (
            self.matchup_id,
            self.horizon,
            self.source.casefold(),
            self.source_book_key.casefold(),
        )

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


class MethodMarketStore:
    """Atomic append-only CSV/JSONL mirrors for method book boards."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        if self.csv_path.resolve() == self.jsonl_path.resolve():
            raise ValueError("method CSV and JSONL paths must differ")

    @staticmethod
    def _index(
        records: Iterable[MethodMarketSnapshot],
    ) -> dict[str, MethodMarketSnapshot]:
        by_id: dict[str, MethodMarketSnapshot] = {}
        natural: dict[tuple[str, str, str, str], str] = {}
        for record in records:
            previous = by_id.get(record.quote_id)
            if previous is not None and previous != record:
                raise StoreIntegrityError("duplicate method quote_id has different data")
            prior_id = natural.get(record.natural_key)
            if prior_id is not None and prior_id != record.quote_id:
                raise StoreIntegrityError(
                    "an existing matchup/horizon/source/book method quote was rewritten"
                )
            by_id[record.quote_id] = record
            natural[record.natural_key] = record.quote_id
        return by_id

    def _read_jsonl(self) -> list[MethodMarketSnapshot]:
        if not self.jsonl_path.exists():
            return []
        records: list[MethodMarketSnapshot] = []
        with self.jsonl_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise StoreIntegrityError(
                        f"blank method JSONL record at line {line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StoreIntegrityError(
                        f"invalid method JSONL record at line {line_number}"
                    ) from error
                if not isinstance(value, dict):
                    raise StoreIntegrityError(
                        f"method JSONL line {line_number} is not an object"
                    )
                records.append(MethodMarketSnapshot.from_mapping(value))
        self._index(records)
        return records

    def _read_csv(self) -> list[MethodMarketSnapshot]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != MethodMarketSnapshot.FIELDNAMES:
                raise StoreIntegrityError("method CSV columns do not match the schema")
            records = [MethodMarketSnapshot.from_mapping(row) for row in reader]
        self._index(records)
        return records

    def read(self) -> tuple[MethodMarketSnapshot, ...]:
        jsonl = self._read_jsonl()
        csv_records = self._read_csv()
        if not jsonl:
            return tuple(csv_records)
        if not csv_records:
            return tuple(jsonl)
        json_ids = [record.quote_id for record in jsonl]
        csv_ids = [record.quote_id for record in csv_records]
        json_index = self._index(jsonl)
        csv_index = self._index(csv_records)
        common = set(json_index) & set(csv_index)
        if any(json_index[key] != csv_index[key] for key in common):
            raise StoreIntegrityError("method CSV and JSONL disagree")
        if json_ids == csv_ids:
            return tuple(jsonl)
        if csv_ids == json_ids[: len(csv_ids)]:
            return tuple(jsonl)
        if json_ids == csv_ids[: len(json_ids)]:
            return tuple(csv_records)
        raise StoreIntegrityError("method CSV and JSONL mirrors diverged")

    @staticmethod
    def dataset_sha256(records: Iterable[MethodMarketSnapshot]) -> str:
        return canonical_hash([record.to_mapping() for record in records])

    @staticmethod
    def _render_jsonl(records: Iterable[MethodMarketSnapshot]) -> str:
        return "".join(
            f"{canonical_json(record.to_mapping())}\n" for record in records
        )

    @staticmethod
    def _render_csv(records: Iterable[MethodMarketSnapshot]) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=MethodMarketSnapshot.FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_mapping())
        return output.getvalue()

    def append(self, snapshots: Iterable[MethodMarketSnapshot]) -> AppendResult:
        pending = tuple(snapshots)
        lock_path = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock_path):
            existing = list(self.read())
            by_id = self._index(existing)
            natural = {record.natural_key: record.quote_id for record in existing}
            additions: list[MethodMarketSnapshot] = []
            duplicates: list[str] = []
            for snapshot in pending:
                if not isinstance(snapshot, MethodMarketSnapshot):
                    raise TypeError("append accepts MethodMarketSnapshot instances")
                if snapshot.quote_id in by_id:
                    if by_id[snapshot.quote_id] != snapshot:
                        raise StoreIntegrityError("an existing method quote_id was rewritten")
                    duplicates.append(snapshot.quote_id)
                    continue
                if snapshot.natural_key in natural:
                    raise StoreIntegrityError(
                        "conflicting method quote for matchup/horizon/source/book"
                    )
                by_id[snapshot.quote_id] = snapshot
                natural[snapshot.natural_key] = snapshot.quote_id
                additions.append(snapshot)
            combined = existing + additions
            self._index(combined)
            atomic_write_text(self.jsonl_path, self._render_jsonl(combined))
            atomic_write_text(self.csv_path, self._render_csv(combined))
        return AppendResult(
            added_ids=tuple(record.quote_id for record in additions),
            duplicate_ids=tuple(duplicates),
            total_records=len(combined),
            dataset_sha256=self.dataset_sha256(combined),
        )
