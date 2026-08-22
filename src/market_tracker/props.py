"""Immutable full-fight total-round quote snapshots and atomic storage."""

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
    canonical_pair,
    implied_probability,
    iso_date,
    moneyline,
    nonempty_text,
    optional_stable_id,
    payload_hash,
    probability,
    require_before_event,
    stable_id,
    utc_datetime,
    utc_text,
    validated_git_commit,
    validated_sha256,
)
from ._storage import atomic_write_text, exclusive_store_lock
from .quotes import AppendResult, MAX_OVERROUND, MIN_OVERROUND


def _audit_name(value: object) -> str:
    return " ".join(str(value or "").split())


def _round_line(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise MarketDataError("total-round line must be numeric") from error
    if not math.isfinite(parsed) or not 0.0 < parsed <= 25.0:
        raise MarketDataError("total-round line is outside (0, 25]")
    if round(parsed, 3) != parsed:
        raise MarketDataError("total-round line supports at most three decimals")
    return parsed


@dataclass(frozen=True)
class TotalRoundsQuoteSnapshot:
    """A complete Over/Under pair for one full-fight round line at one book."""

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
    source_event_id: str
    source_book_key: str
    source_quote_updated_at_utc: str
    source_commence_time_utc: str
    source_quote_age_seconds: float
    book: str
    market: str
    period: str
    line: float
    over_moneyline: int
    under_moneyline: int
    over_implied_probability: float
    under_implied_probability: float
    overround: float
    no_vig_over_probability: float
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
        event_date: object,
        timing_precision: object,
        event_start_utc: object,
        observed_at_utc: object,
        source: object,
        source_event_id: object,
        source_book_key: object,
        source_quote_updated_at_utc: object,
        source_commence_time_utc: object,
        book: object,
        line: object,
        over_moneyline: object,
        under_moneyline: object,
        matchup_id: object | None = None,
        fight_id: object | None = None,
        fighter_name: object = "",
        opponent_name: object = "",
        quote_first_seen_at_utc: object | None = None,
        source_payload: bytes | str | Mapping | list | None = None,
        source_payload_sha256: str | None = None,
    ) -> "TotalRoundsQuoteSnapshot":
        if (source_payload is None) == (source_payload_sha256 is None):
            raise MarketDataError(
                "provide exactly one of source_payload or source_payload_sha256"
            )
        over_line = moneyline(over_moneyline, "over_moneyline")
        under_line = moneyline(under_moneyline, "under_moneyline")
        fighter, opponent, fighter_display, opponent_display, derived_matchup = canonical_pair(
            fighter_id,
            opponent_id,
            _audit_name(fighter_name),
            _audit_name(opponent_name),
        )
        event = stable_id(event_id, "event_id")
        # canonical_pair's matchup token is participant-only; market matchups
        # also include the event to prevent cross-event identity collisions.
        from ._common import matchup_id_for

        derived_matchup = matchup_id_for(event, fighter, opponent)
        if matchup_id is not None and str(matchup_id).strip():
            supplied = stable_id(matchup_id, "matchup_id")
            if supplied != derived_matchup:
                raise MarketDataError("matchup_id disagrees with event and fighter IDs")
        observed, event_day, precision, event_start = require_before_event(
            observed_at_utc,
            event_date=event_date,
            timing_precision=timing_precision,
            event_start_utc=event_start_utc,
            observed_field="observed_at_utc",
        )
        first_seen, _, _, _ = require_before_event(
            observed if quote_first_seen_at_utc is None else quote_first_seen_at_utc,
            event_date=event_day,
            timing_precision=precision,
            event_start_utc=event_start,
            observed_field="quote_first_seen_at_utc",
        )
        if first_seen > observed:
            raise MarketDataError("quote_first_seen_at_utc cannot follow observed_at_utc")
        updated = utc_datetime(
            source_quote_updated_at_utc, "source_quote_updated_at_utc"
        )
        commence = utc_datetime(source_commence_time_utc, "source_commence_time_utc")
        age = (observed - updated).total_seconds()
        if age < -300.0:
            raise MarketDataError("source quote update is implausibly later than retrieval")
        if not observed < commence:
            raise MarketDataError("total-round quote was not captured before commence time")
        over_implied = implied_probability(over_line)
        under_implied = implied_probability(under_line)
        overround = over_implied + under_implied
        if not MIN_OVERROUND <= overround <= MAX_OVERROUND:
            raise MarketDataError(
                f"total-round overround {overround:.6f} is outside "
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
            "quote_first_seen_at_utc": utc_text(first_seen, "quote_first_seen_at_utc"),
            "source": nonempty_text(source, "source"),
            "source_event_id": stable_id(source_event_id, "source_event_id"),
            "source_book_key": stable_id(source_book_key, "source_book_key"),
            "source_quote_updated_at_utc": utc_text(
                updated, "source_quote_updated_at_utc"
            ),
            "source_commence_time_utc": utc_text(
                commence, "source_commence_time_utc"
            ),
            "source_quote_age_seconds": age,
            "book": nonempty_text(book, "book"),
            "market": "total_rounds",
            "period": "full_fight",
            "line": _round_line(line),
            "over_moneyline": over_line,
            "under_moneyline": under_line,
            "over_implied_probability": over_implied,
            "under_implied_probability": under_implied,
            "overround": overround,
            "no_vig_over_probability": over_implied / overround,
            "source_payload_sha256": digest,
        }
        return cls(quote_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "TotalRoundsQuoteSnapshot":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"total-round quote schema mismatch; missing={missing}, extra={extra}"
            )
        try:
            schema_version = int(record["schema_version"])
        except (TypeError, ValueError) as error:
            raise MarketDataError("invalid total-round schema version") from error
        if schema_version != SCHEMA_VERSION:
            raise MarketDataError("unsupported total-round schema version")
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
            quote_first_seen_at_utc=record["quote_first_seen_at_utc"],
            source=record["source"],
            source_event_id=record["source_event_id"],
            source_book_key=record["source_book_key"],
            source_quote_updated_at_utc=record["source_quote_updated_at_utc"],
            source_commence_time_utc=record["source_commence_time_utc"],
            book=record["book"],
            line=record["line"],
            over_moneyline=record["over_moneyline"],
            under_moneyline=record["under_moneyline"],
            source_payload_sha256=str(record["source_payload_sha256"]),
        )
        if str(record["quote_id"]) != rebuilt.quote_id:
            raise StoreIntegrityError("total-round quote_id disagrees with contents")
        numeric_fields = (
            "source_quote_age_seconds",
            "over_implied_probability",
            "under_implied_probability",
            "overround",
            "no_vig_over_probability",
        )
        for field in numeric_fields:
            try:
                supplied = float(record[field])
            except (TypeError, ValueError) as error:
                raise MarketDataError(f"{field} must be numeric") from error
            if not math.isfinite(supplied) or abs(supplied - getattr(rebuilt, field)) > 1e-12:
                raise StoreIntegrityError(f"stored {field} disagrees with quote inputs")
        return rebuilt

    @property
    def natural_key(self) -> tuple[str, str, str, float]:
        return (
            self.matchup_id,
            self.capture_id,
            self.source_book_key.casefold(),
            self.line,
        )

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


class TotalRoundsQuoteStore:
    """Append-only JSONL authority with an atomically mirrored CSV."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        if self.csv_path.resolve() == self.jsonl_path.resolve():
            raise ValueError("csv_path and jsonl_path must be different")

    @staticmethod
    def _index(
        records: Iterable[TotalRoundsQuoteSnapshot],
    ) -> dict[str, TotalRoundsQuoteSnapshot]:
        indexed: dict[str, TotalRoundsQuoteSnapshot] = {}
        natural: dict[tuple[str, str, str, float], str] = {}
        capture_contracts: dict[str, tuple[str, str, str, str | None, str]] = {}
        payloads: dict[tuple[str, str], str] = {}
        for record in records:
            if not isinstance(record, TotalRoundsQuoteSnapshot):
                raise TypeError("total-round store accepts TotalRoundsQuoteSnapshot only")
            previous = indexed.get(record.quote_id)
            if previous is not None and previous != record:
                raise StoreIntegrityError("total-round quote_id was rewritten")
            prior_natural = natural.get(record.natural_key)
            if prior_natural is not None and prior_natural != record.quote_id:
                raise StoreIntegrityError("total-round capture/book/line was rewritten")
            contract = (
                record.event_id,
                record.event_date,
                record.timing_precision,
                record.event_start_utc,
                record.observed_at_utc,
            )
            prior_contract = capture_contracts.setdefault(record.capture_id, contract)
            if prior_contract != contract:
                raise StoreIntegrityError("one prop capture spans multiple event timings")
            payload_key = (record.capture_id, record.source.casefold())
            prior_payload = payloads.setdefault(payload_key, record.source_payload_sha256)
            if prior_payload != record.source_payload_sha256:
                raise StoreIntegrityError("one prop capture/source has multiple payloads")
            indexed[record.quote_id] = record
            natural[record.natural_key] = record.quote_id
        return indexed

    def _read_jsonl(self) -> list[TotalRoundsQuoteSnapshot]:
        if not self.jsonl_path.exists():
            return []
        records: list[TotalRoundsQuoteSnapshot] = []
        with self.jsonl_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise StoreIntegrityError(
                        f"blank total-round JSONL row at line {line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StoreIntegrityError(
                        f"invalid total-round JSONL at line {line_number}"
                    ) from error
                if not isinstance(value, dict):
                    raise StoreIntegrityError("total-round JSONL row is not an object")
                records.append(TotalRoundsQuoteSnapshot.from_mapping(value))
        self._index(records)
        return records

    def _read_csv(self) -> list[TotalRoundsQuoteSnapshot]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != TotalRoundsQuoteSnapshot.FIELDNAMES:
                raise StoreIntegrityError("total-round CSV columns do not match schema")
            records = [TotalRoundsQuoteSnapshot.from_mapping(row) for row in reader]
        self._index(records)
        return records

    def read(self) -> tuple[TotalRoundsQuoteSnapshot, ...]:
        jsonl = self._read_jsonl()
        csv_rows = self._read_csv()
        if not jsonl:
            return tuple(csv_rows)
        if not csv_rows:
            return tuple(jsonl)
        json_ids = [item.quote_id for item in jsonl]
        csv_ids = [item.quote_id for item in csv_rows]
        json_index = self._index(jsonl)
        csv_index = self._index(csv_rows)
        common = set(json_index) & set(csv_index)
        if any(json_index[key] != csv_index[key] for key in common):
            raise StoreIntegrityError("total-round mirrors disagree on quote contents")
        if set(json_index) == set(csv_index) and json_ids == csv_ids:
            return tuple(jsonl)
        if set(csv_index) < set(json_index) and csv_ids == json_ids[: len(csv_ids)]:
            return tuple(jsonl)
        if set(json_index) < set(csv_index) and json_ids == csv_ids[: len(json_ids)]:
            return tuple(csv_rows)
        raise StoreIntegrityError("total-round CSV and JSONL mirrors diverged")

    @staticmethod
    def _render_jsonl(records: Iterable[TotalRoundsQuoteSnapshot]) -> str:
        return "".join(f"{canonical_json(item.to_mapping())}\n" for item in records)

    @staticmethod
    def _render_csv(records: Iterable[TotalRoundsQuoteSnapshot]) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=TotalRoundsQuoteSnapshot.FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_mapping())
        return output.getvalue()

    @staticmethod
    def dataset_sha256(records: Iterable[TotalRoundsQuoteSnapshot]) -> str:
        return canonical_hash([item.to_mapping() for item in records])

    def append(self, snapshots: Iterable[TotalRoundsQuoteSnapshot]) -> AppendResult:
        pending = tuple(snapshots)
        lock_path = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock_path):
            existing = list(self.read())
            index = self._index(existing)
            natural = {item.natural_key: item.quote_id for item in existing}
            additions: list[TotalRoundsQuoteSnapshot] = []
            duplicates: list[str] = []
            for snapshot in pending:
                if not isinstance(snapshot, TotalRoundsQuoteSnapshot):
                    raise TypeError("append accepts TotalRoundsQuoteSnapshot instances only")
                previous = index.get(snapshot.quote_id)
                if previous is not None:
                    if previous != snapshot:
                        raise StoreIntegrityError("an existing total-round quote was rewritten")
                    duplicates.append(snapshot.quote_id)
                    continue
                prior_id = natural.get(snapshot.natural_key)
                if prior_id is not None and prior_id != snapshot.quote_id:
                    raise StoreIntegrityError("conflicting total-round capture/book/line")
                index[snapshot.quote_id] = snapshot
                natural[snapshot.natural_key] = snapshot.quote_id
                additions.append(snapshot)
            additions.sort(key=lambda item: (item.observed_at_utc, item.quote_id))
            combined = [*existing, *additions]
            self._index(combined)
            atomic_write_text(self.jsonl_path, self._render_jsonl(combined))
            atomic_write_text(self.csv_path, self._render_csv(combined))
            return AppendResult(
                added_ids=tuple(item.quote_id for item in additions),
                duplicate_ids=tuple(duplicates),
                total_records=len(combined),
                dataset_sha256=self.dataset_sha256(combined),
            )


@dataclass(frozen=True)
class TotalRoundsForecastCapture:
    """One frozen candidate probability paired to a quoted round line."""

    schema_version: int
    forecast_capture_id: str
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
    forecast_issued_at_utc: str
    scheduled_rounds: int
    schedule_basis: str
    line: float
    over_probability: float
    model_id: str
    model_version: str
    model_trained_through: str
    source_commit_sha: str
    source_publication_sha256: str

    FIELDNAMES = tuple(__annotations__)

    @classmethod
    def create(
        cls,
        *,
        capture_id: object,
        event_id: object,
        fighter_id: object,
        opponent_id: object,
        event_date: object,
        timing_precision: object,
        event_start_utc: object,
        forecast_issued_at_utc: object,
        scheduled_rounds: object,
        schedule_basis: object,
        line: object,
        over_probability: object,
        model_id: object,
        model_version: object,
        model_trained_through: object,
        source_commit_sha: object,
        source_publication_sha256: object,
        matchup_id: object | None = None,
        fight_id: object | None = None,
        fighter_name: object = "",
        opponent_name: object = "",
    ) -> "TotalRoundsForecastCapture":
        fighter, opponent, fighter_display, opponent_display, _ = canonical_pair(
            fighter_id,
            opponent_id,
            _audit_name(fighter_name),
            _audit_name(opponent_name),
        )
        event = stable_id(event_id, "event_id")
        from ._common import matchup_id_for

        derived_matchup = matchup_id_for(event, fighter, opponent)
        if matchup_id is not None and str(matchup_id).strip():
            if stable_id(matchup_id, "matchup_id") != derived_matchup:
                raise MarketDataError("total forecast matchup_id disagrees with identities")
        issued, event_day, precision, event_start = require_before_event(
            forecast_issued_at_utc,
            event_date=event_date,
            timing_precision=timing_precision,
            event_start_utc=event_start_utc,
            observed_field="forecast_issued_at_utc",
        )
        try:
            rounds = int(scheduled_rounds)
        except (TypeError, ValueError) as error:
            raise MarketDataError("scheduled_rounds must be an integer") from error
        if rounds not in {3, 5}:
            raise MarketDataError("scheduled_rounds must be three or five")
        parsed_line = _round_line(line)
        if parsed_line * 300 >= rounds * 300:
            raise MarketDataError("total-round line must precede the scheduled horizon")
        trained = iso_date(model_trained_through, "model_trained_through")
        if not trained < issued.date().isoformat():
            raise MarketDataError("outcome model training cutoff must precede issuance")
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
            "forecast_issued_at_utc": utc_text(issued, "forecast_issued_at_utc"),
            "scheduled_rounds": rounds,
            "schedule_basis": nonempty_text(schedule_basis, "schedule_basis"),
            "line": parsed_line,
            "over_probability": probability(over_probability, "over_probability"),
            "model_id": stable_id(model_id, "model_id"),
            "model_version": nonempty_text(model_version, "model_version"),
            "model_trained_through": trained,
            "source_commit_sha": validated_git_commit(source_commit_sha),
            "source_publication_sha256": validated_sha256(
                source_publication_sha256, "source_publication_sha256"
            ),
        }
        return cls(forecast_capture_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(
        cls, record: Mapping[str, object]
    ) -> "TotalRoundsForecastCapture":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"total forecast schema mismatch; missing={missing}, extra={extra}"
            )
        rebuilt = cls.create(
            **{
                key: record[key]
                for key in cls.FIELDNAMES
                if key not in {"schema_version", "forecast_capture_id"}
            }
        )
        if int(record["schema_version"]) != SCHEMA_VERSION:
            raise MarketDataError("unsupported total forecast schema version")
        if str(record["forecast_capture_id"]) != rebuilt.forecast_capture_id:
            raise StoreIntegrityError("total forecast ID disagrees with contents")
        return rebuilt

    @property
    def natural_key(self) -> tuple[str, str, float]:
        return self.matchup_id, self.capture_id, self.line

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


class TotalRoundsForecastStore:
    """Append-only total-round forecast ledger with mirrored CSV and JSONL."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        if self.csv_path.resolve() == self.jsonl_path.resolve():
            raise ValueError("csv_path and jsonl_path must be different")

    @staticmethod
    def _index(
        records: Iterable[TotalRoundsForecastCapture],
    ) -> dict[str, TotalRoundsForecastCapture]:
        indexed: dict[str, TotalRoundsForecastCapture] = {}
        natural: dict[tuple[str, str, float], str] = {}
        for record in records:
            if not isinstance(record, TotalRoundsForecastCapture):
                raise TypeError("total forecast store accepts forecast captures only")
            previous = indexed.get(record.forecast_capture_id)
            if previous is not None and previous != record:
                raise StoreIntegrityError("total forecast ID was rewritten")
            prior = natural.get(record.natural_key)
            if prior is not None and prior != record.forecast_capture_id:
                raise StoreIntegrityError("total forecast capture/line was rewritten")
            indexed[record.forecast_capture_id] = record
            natural[record.natural_key] = record.forecast_capture_id
        return indexed

    def _read_jsonl(self) -> list[TotalRoundsForecastCapture]:
        if not self.jsonl_path.exists():
            return []
        records: list[TotalRoundsForecastCapture] = []
        with self.jsonl_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise StoreIntegrityError(
                        f"blank total forecast JSONL row at line {line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StoreIntegrityError(
                        f"invalid total forecast JSONL at line {line_number}"
                    ) from error
                if not isinstance(value, dict):
                    raise StoreIntegrityError("total forecast JSONL row is not an object")
                records.append(TotalRoundsForecastCapture.from_mapping(value))
        self._index(records)
        return records

    def _read_csv(self) -> list[TotalRoundsForecastCapture]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != TotalRoundsForecastCapture.FIELDNAMES:
                raise StoreIntegrityError("total forecast CSV columns do not match schema")
            records = [TotalRoundsForecastCapture.from_mapping(row) for row in reader]
        self._index(records)
        return records

    def read(self) -> tuple[TotalRoundsForecastCapture, ...]:
        jsonl = self._read_jsonl()
        csv_rows = self._read_csv()
        if not jsonl:
            return tuple(csv_rows)
        if not csv_rows:
            return tuple(jsonl)
        if jsonl == csv_rows:
            return tuple(jsonl)
        raise StoreIntegrityError("total forecast CSV and JSONL mirrors diverged")

    @staticmethod
    def dataset_sha256(records: Iterable[TotalRoundsForecastCapture]) -> str:
        return canonical_hash([item.to_mapping() for item in records])

    @staticmethod
    def _render_jsonl(records: Iterable[TotalRoundsForecastCapture]) -> str:
        return "".join(f"{canonical_json(item.to_mapping())}\n" for item in records)

    @staticmethod
    def _render_csv(records: Iterable[TotalRoundsForecastCapture]) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=TotalRoundsForecastCapture.FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_mapping())
        return output.getvalue()

    def append(
        self, snapshots: Iterable[TotalRoundsForecastCapture]
    ) -> AppendResult:
        pending = tuple(snapshots)
        lock_path = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock_path):
            existing = list(self.read())
            indexed = self._index(existing)
            natural = {item.natural_key: item.forecast_capture_id for item in existing}
            additions: list[TotalRoundsForecastCapture] = []
            duplicates: list[str] = []
            for snapshot in pending:
                previous = indexed.get(snapshot.forecast_capture_id)
                if previous is not None:
                    if previous != snapshot:
                        raise StoreIntegrityError("an existing total forecast was rewritten")
                    duplicates.append(snapshot.forecast_capture_id)
                    continue
                prior = natural.get(snapshot.natural_key)
                if prior is not None and prior != snapshot.forecast_capture_id:
                    raise StoreIntegrityError("conflicting total forecast capture/line")
                indexed[snapshot.forecast_capture_id] = snapshot
                natural[snapshot.natural_key] = snapshot.forecast_capture_id
                additions.append(snapshot)
            additions.sort(
                key=lambda item: (
                    item.forecast_issued_at_utc,
                    item.forecast_capture_id,
                )
            )
            combined = [*existing, *additions]
            self._index(combined)
            atomic_write_text(self.jsonl_path, self._render_jsonl(combined))
            atomic_write_text(self.csv_path, self._render_csv(combined))
            return AppendResult(
                added_ids=tuple(item.forecast_capture_id for item in additions),
                duplicate_ids=tuple(duplicates),
                total_records=len(combined),
                dataset_sha256=self.dataset_sha256(combined),
            )
