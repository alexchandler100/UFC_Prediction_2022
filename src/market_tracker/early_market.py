"""Research-only early MMA prices and later official-UFC identity links.

The Odds API's MMA endpoint does not identify a promotion.  These ledgers
therefore preserve source observations without guessing that they are UFC.
An independent append-only link is created only after a source event matches
an officially published UFCStats matchup.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
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
    implied_probability,
    matchup_id_for,
    moneyline,
    nonempty_text,
    stable_id,
    utc_datetime,
    utc_text,
    validated_sha256,
)
from ._storage import atomic_write_text, exclusive_store_lock
from .quotes import AppendResult, MAX_OVERROUND, MIN_OVERROUND


EARLY_MARKET_CONTRACT = "distinct_source_price_states_v1"
EARLY_MARKETS = frozenset({"h2h", "total_rounds"})
EARLY_PERIOD = "full_fight"


def _boolean(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise MarketDataError(f"{field} must be a boolean")


def _line_text(value: object) -> str:
    if value is None or not str(value).strip():
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise MarketDataError("line must be numeric") from error
    if not math.isfinite(number) or not 0.0 < number <= 25.0:
        raise MarketDataError("line is outside the supported range")
    if round(number, 3) != number:
        raise MarketDataError("line may have at most three decimal places")
    return f"{number:.3f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class EarlyMarketObservation:
    """The first local sighting of one distinct source price state."""

    schema_version: int
    observation_id: str
    first_capture_id: str
    first_observed_at_utc: str
    source: str
    source_payload_sha256: str
    source_event_id: str
    source_commence_time_utc: str
    source_fighter_name: str
    source_opponent_name: str
    book: str
    source_book_key: str
    source_quote_updated_at_utc: str
    market: str
    period: str
    line: str
    outcome_a: str
    outcome_b: str
    outcome_a_moneyline: int
    outcome_b_moneyline: int
    paper_only: bool
    execution_enabled: bool

    FIELDNAMES = (
        "schema_version",
        "observation_id",
        "first_capture_id",
        "first_observed_at_utc",
        "source",
        "source_payload_sha256",
        "source_event_id",
        "source_commence_time_utc",
        "source_fighter_name",
        "source_opponent_name",
        "book",
        "source_book_key",
        "source_quote_updated_at_utc",
        "market",
        "period",
        "line",
        "outcome_a",
        "outcome_b",
        "outcome_a_moneyline",
        "outcome_b_moneyline",
        "paper_only",
        "execution_enabled",
    )

    @classmethod
    def create(
        cls,
        *,
        first_capture_id: object,
        first_observed_at_utc: object,
        source: object,
        source_payload_sha256: object,
        source_event_id: object,
        source_commence_time_utc: object,
        source_fighter_name: object,
        source_opponent_name: object,
        book: object,
        source_book_key: object,
        source_quote_updated_at_utc: object,
        market: object,
        period: object = EARLY_PERIOD,
        line: object = None,
        outcome_a: object,
        outcome_b: object,
        outcome_a_moneyline: object,
        outcome_b_moneyline: object,
    ) -> "EarlyMarketObservation":
        observed = utc_datetime(first_observed_at_utc, "first_observed_at_utc")
        commence = utc_datetime(
            source_commence_time_utc, "source_commence_time_utc"
        )
        updated = utc_datetime(
            source_quote_updated_at_utc, "source_quote_updated_at_utc"
        )
        if not observed < commence:
            raise MarketDataError("early price was not observed before commence time")
        if (observed - updated).total_seconds() < -300.0:
            raise MarketDataError(
                "source quote update is implausibly later than its retrieval"
            )
        source_fighter = nonempty_text(source_fighter_name, "source_fighter_name")
        source_opponent = nonempty_text(
            source_opponent_name, "source_opponent_name"
        )
        if source_fighter.casefold() == source_opponent.casefold():
            raise MarketDataError("an early market matchup must contain two fighters")
        market_key = nonempty_text(market, "market").casefold()
        if market_key not in EARLY_MARKETS:
            raise MarketDataError(f"unsupported early market {market_key!r}")
        period_key = nonempty_text(period, "period").casefold()
        if period_key != EARLY_PERIOD:
            raise MarketDataError("only full-fight early markets are supported")
        normalized_line = _line_text(line)
        first_outcome = nonempty_text(outcome_a, "outcome_a")
        second_outcome = nonempty_text(outcome_b, "outcome_b")
        if market_key == "h2h":
            if normalized_line:
                raise MarketDataError("h2h observations cannot have a line")
            if (first_outcome, second_outcome) != (
                source_fighter,
                source_opponent,
            ):
                raise MarketDataError(
                    "h2h outcomes must preserve the source fighter orientation"
                )
        elif (
            not normalized_line
            or first_outcome.casefold() != "over"
            or second_outcome.casefold() != "under"
        ):
            raise MarketDataError(
                "total-round observations require a line and Over/Under outcomes"
            )
        first_line = moneyline(outcome_a_moneyline, "outcome_a_moneyline")
        second_line = moneyline(outcome_b_moneyline, "outcome_b_moneyline")
        overround = implied_probability(first_line) + implied_probability(second_line)
        if not MIN_OVERROUND <= overround <= MAX_OVERROUND:
            raise MarketDataError(
                f"early price overround {overround:.6f} is outside "
                f"[{MIN_OVERROUND}, {MAX_OVERROUND}]"
            )
        state = {
            "schema_version": SCHEMA_VERSION,
            "source": nonempty_text(source, "source"),
            "source_event_id": stable_id(source_event_id, "source_event_id"),
            "source_commence_time_utc": utc_text(
                commence, "source_commence_time_utc"
            ),
            "source_fighter_name": source_fighter,
            "source_opponent_name": source_opponent,
            "book": nonempty_text(book, "book"),
            "source_book_key": stable_id(source_book_key, "source_book_key"),
            "source_quote_updated_at_utc": utc_text(
                updated, "source_quote_updated_at_utc"
            ),
            "market": market_key,
            "period": period_key,
            "line": normalized_line,
            "outcome_a": first_outcome,
            "outcome_b": second_outcome,
            "outcome_a_moneyline": first_line,
            "outcome_b_moneyline": second_line,
        }
        return cls(
            observation_id=canonical_hash(state),
            first_capture_id=stable_id(first_capture_id, "first_capture_id"),
            first_observed_at_utc=utc_text(observed, "first_observed_at_utc"),
            source_payload_sha256=validated_sha256(
                source_payload_sha256, "source_payload_sha256"
            ),
            paper_only=True,
            execution_enabled=False,
            **state,
        )

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "EarlyMarketObservation":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"early observation schema mismatch; missing={missing}, extra={extra}"
            )
        item = cls.create(
            first_capture_id=record["first_capture_id"],
            first_observed_at_utc=record["first_observed_at_utc"],
            source=record["source"],
            source_payload_sha256=record["source_payload_sha256"],
            source_event_id=record["source_event_id"],
            source_commence_time_utc=record["source_commence_time_utc"],
            source_fighter_name=record["source_fighter_name"],
            source_opponent_name=record["source_opponent_name"],
            book=record["book"],
            source_book_key=record["source_book_key"],
            source_quote_updated_at_utc=record["source_quote_updated_at_utc"],
            market=record["market"],
            period=record["period"],
            line=record["line"],
            outcome_a=record["outcome_a"],
            outcome_b=record["outcome_b"],
            outcome_a_moneyline=record["outcome_a_moneyline"],
            outcome_b_moneyline=record["outcome_b_moneyline"],
        )
        if str(record["observation_id"]) != item.observation_id:
            raise StoreIntegrityError(
                "observation_id does not match the source price state"
            )
        if not _boolean(record["paper_only"], "paper_only") or _boolean(
            record["execution_enabled"], "execution_enabled"
        ):
            raise MarketDataError("early observations must remain research-only")
        return item

    @property
    def natural_key(self) -> tuple[str]:
        return (self.observation_id,)

    def validate_integrity(self) -> None:
        if EarlyMarketObservation.from_mapping(self.to_mapping()) != self:
            raise StoreIntegrityError("early observation does not round-trip")

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


@dataclass(frozen=True)
class EarlyMarketLink:
    """One append-only association between a source event and official UFC IDs."""

    schema_version: int
    link_id: str
    first_linked_at_utc: str
    first_capture_id: str
    source: str
    source_event_id: str
    source_commence_time_utc: str
    source_fighter_name: str
    source_opponent_name: str
    ufc_event_id: str
    matchup_id: str
    source_fighter_ufcstats_id: str
    source_opponent_ufcstats_id: str
    source_is_reversed: bool
    paper_only: bool
    execution_enabled: bool

    FIELDNAMES = (
        "schema_version",
        "link_id",
        "first_linked_at_utc",
        "first_capture_id",
        "source",
        "source_event_id",
        "source_commence_time_utc",
        "source_fighter_name",
        "source_opponent_name",
        "ufc_event_id",
        "matchup_id",
        "source_fighter_ufcstats_id",
        "source_opponent_ufcstats_id",
        "source_is_reversed",
        "paper_only",
        "execution_enabled",
    )

    @classmethod
    def create(
        cls,
        *,
        first_linked_at_utc: object,
        first_capture_id: object,
        source: object,
        source_event_id: object,
        source_commence_time_utc: object,
        source_fighter_name: object,
        source_opponent_name: object,
        ufc_event_id: object,
        matchup_id: object,
        source_fighter_ufcstats_id: object,
        source_opponent_ufcstats_id: object,
        source_is_reversed: object,
    ) -> "EarlyMarketLink":
        linked = utc_datetime(first_linked_at_utc, "first_linked_at_utc")
        commence = utc_datetime(
            source_commence_time_utc, "source_commence_time_utc"
        )
        if not linked < commence:
            raise MarketDataError("an early market link must be created before commence")
        event_id = stable_id(ufc_event_id, "ufc_event_id")
        fighter_id = stable_id(
            source_fighter_ufcstats_id, "source_fighter_ufcstats_id"
        )
        opponent_id = stable_id(
            source_opponent_ufcstats_id, "source_opponent_ufcstats_id"
        )
        derived_matchup = matchup_id_for(event_id, fighter_id, opponent_id)
        if stable_id(matchup_id, "matchup_id") != derived_matchup:
            raise MarketDataError("early link matchup_id disagrees with official IDs")
        reversed_value = _boolean(source_is_reversed, "source_is_reversed")
        source_fighter_name_value = nonempty_text(
            source_fighter_name, "source_fighter_name"
        )
        source_opponent_name_value = nonempty_text(
            source_opponent_name, "source_opponent_name"
        )
        if source_fighter_name_value.casefold() == source_opponent_name_value.casefold():
            raise MarketDataError("an early market link must contain two fighters")
        identity = {
            "schema_version": SCHEMA_VERSION,
            "source": nonempty_text(source, "source"),
            "source_event_id": stable_id(source_event_id, "source_event_id"),
            "ufc_event_id": event_id,
            "matchup_id": derived_matchup,
            "source_fighter_ufcstats_id": fighter_id,
            "source_opponent_ufcstats_id": opponent_id,
            "source_is_reversed": reversed_value,
        }
        return cls(
            link_id=canonical_hash(identity),
            first_linked_at_utc=utc_text(linked, "first_linked_at_utc"),
            first_capture_id=stable_id(first_capture_id, "first_capture_id"),
            source_commence_time_utc=utc_text(
                commence, "source_commence_time_utc"
            ),
            source_fighter_name=source_fighter_name_value,
            source_opponent_name=source_opponent_name_value,
            paper_only=True,
            execution_enabled=False,
            **identity,
        )

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "EarlyMarketLink":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"early link schema mismatch; missing={missing}, extra={extra}"
            )
        item = cls.create(
            first_linked_at_utc=record["first_linked_at_utc"],
            first_capture_id=record["first_capture_id"],
            source=record["source"],
            source_event_id=record["source_event_id"],
            source_commence_time_utc=record["source_commence_time_utc"],
            source_fighter_name=record["source_fighter_name"],
            source_opponent_name=record["source_opponent_name"],
            ufc_event_id=record["ufc_event_id"],
            matchup_id=record["matchup_id"],
            source_fighter_ufcstats_id=record["source_fighter_ufcstats_id"],
            source_opponent_ufcstats_id=record["source_opponent_ufcstats_id"],
            source_is_reversed=record["source_is_reversed"],
        )
        if str(record["link_id"]) != item.link_id:
            raise StoreIntegrityError("link_id does not match the official association")
        if not _boolean(record["paper_only"], "paper_only") or _boolean(
            record["execution_enabled"], "execution_enabled"
        ):
            raise MarketDataError("early links must remain research-only")
        return item

    @property
    def natural_key(self) -> tuple[str]:
        return (self.link_id,)

    def validate_integrity(self) -> None:
        if EarlyMarketLink.from_mapping(self.to_mapping()) != self:
            raise StoreIntegrityError("early link does not round-trip")

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


class _EarlyLedgerStore:
    """Crash-recoverable JSONL/CSV mirrors for one immutable record type."""

    def __init__(
        self,
        csv_path: str | Path,
        jsonl_path: str | Path,
        *,
        record_type: type[EarlyMarketObservation] | type[EarlyMarketLink],
        id_field: str,
        time_field: str,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        if self.csv_path.resolve() == self.jsonl_path.resolve():
            raise ValueError("csv_path and jsonl_path must be different")
        self.record_type = record_type
        self.id_field = id_field
        self.time_field = time_field

    def _index(self, records: Iterable[object]) -> dict[str, object]:
        indexed: dict[str, object] = {}
        natural: dict[tuple[str], str] = {}
        for record in records:
            if not isinstance(record, self.record_type):
                raise TypeError(f"store accepts {self.record_type.__name__} only")
            record.validate_integrity()
            record_id = getattr(record, self.id_field)
            prior = indexed.get(record_id)
            if prior is not None and prior != record:
                raise StoreIntegrityError(f"{self.id_field} was rewritten")
            prior_id = natural.get(record.natural_key)
            if prior_id is not None and prior_id != record_id:
                raise StoreIntegrityError("an early ledger natural key was rewritten")
            indexed[record_id] = record
            natural[record.natural_key] = record_id
        return indexed

    def _read_jsonl(self) -> list[object]:
        if not self.jsonl_path.exists():
            return []
        records: list[object] = []
        with self.jsonl_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise StoreIntegrityError(
                        f"blank early-market JSONL line {line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StoreIntegrityError(
                        f"invalid early-market JSONL line {line_number}"
                    ) from error
                if not isinstance(value, dict):
                    raise StoreIntegrityError("early-market JSONL record is not an object")
                records.append(self.record_type.from_mapping(value))
        self._index(records)
        return records

    def _read_csv(self) -> list[object]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != self.record_type.FIELDNAMES:
                raise StoreIntegrityError("early-market CSV columns do not match")
            records = [self.record_type.from_mapping(row) for row in reader]
        self._index(records)
        return records

    def read(self) -> tuple[object, ...]:
        jsonl_records = self._read_jsonl()
        csv_records = self._read_csv()
        if not jsonl_records:
            return tuple(csv_records)
        if not csv_records:
            return tuple(jsonl_records)
        jsonl_index = self._index(jsonl_records)
        csv_index = self._index(csv_records)
        common = set(jsonl_index) & set(csv_index)
        if any(jsonl_index[key] != csv_index[key] for key in common):
            raise StoreIntegrityError("early-market mirrors disagree on record data")
        jsonl_ids = [getattr(item, self.id_field) for item in jsonl_records]
        csv_ids = [getattr(item, self.id_field) for item in csv_records]
        if set(csv_index) == set(jsonl_index) and csv_ids == jsonl_ids:
            return tuple(jsonl_records)
        if set(csv_index) < set(jsonl_index) and csv_ids == jsonl_ids[: len(csv_ids)]:
            return tuple(jsonl_records)
        if set(jsonl_index) < set(csv_index) and jsonl_ids == csv_ids[: len(jsonl_ids)]:
            return tuple(csv_records)
        raise StoreIntegrityError("early-market CSV and JSONL mirrors diverged")

    def _render_jsonl(self, records: Iterable[object]) -> str:
        return "".join(
            f"{canonical_json(record.to_mapping())}\n" for record in records
        )

    def _render_csv(self, records: Iterable[object]) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=self.record_type.FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_mapping())
        return output.getvalue()

    def append(self, pending: Iterable[object]) -> AppendResult:
        candidates = tuple(pending)
        lock_path = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock_path):
            existing = list(self.read())
            indexed = self._index(existing)
            additions: list[object] = []
            duplicates: list[str] = []
            for record in candidates:
                if not isinstance(record, self.record_type):
                    raise TypeError(f"append accepts {self.record_type.__name__} only")
                record.validate_integrity()
                record_id = getattr(record, self.id_field)
                prior = indexed.get(record_id)
                if prior is not None:
                    if prior != record:
                        raise StoreIntegrityError(f"{self.id_field} was rewritten")
                    duplicates.append(record_id)
                    continue
                indexed[record_id] = record
                additions.append(record)
            additions.sort(
                key=lambda item: (getattr(item, self.time_field), getattr(item, self.id_field))
            )
            combined = [*existing, *additions]
            self._index(combined)
            atomic_write_text(self.jsonl_path, self._render_jsonl(combined))
            atomic_write_text(self.csv_path, self._render_csv(combined))
            return AppendResult(
                added_ids=tuple(getattr(item, self.id_field) for item in additions),
                duplicate_ids=tuple(duplicates),
                total_records=len(combined),
                dataset_sha256=canonical_hash(
                    [item.to_mapping() for item in combined]
                ),
            )


class EarlyMarketObservationStore(_EarlyLedgerStore):
    def __init__(self, csv_path: str | Path, jsonl_path: str | Path) -> None:
        super().__init__(
            csv_path,
            jsonl_path,
            record_type=EarlyMarketObservation,
            id_field="observation_id",
            time_field="first_observed_at_utc",
        )


class EarlyMarketLinkStore(_EarlyLedgerStore):
    def __init__(self, csv_path: str | Path, jsonl_path: str | Path) -> None:
        super().__init__(
            csv_path,
            jsonl_path,
            record_type=EarlyMarketLink,
            id_field="link_id",
            time_field="first_linked_at_utc",
        )
