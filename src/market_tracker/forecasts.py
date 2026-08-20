"""Immutable stats-model forecast captures with explicit provenance."""

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
    iso_date,
    matchup_id_for,
    moneyline,
    nonempty_text,
    optional_stable_id,
    probability,
    require_before_event,
    stable_id,
    utc_datetime,
    utc_text,
    validated_git_commit,
)
from ._storage import atomic_write_text, exclusive_store_lock
from .quotes import AppendResult


NATIVE_PROBABILITY = "native_probability"
LEGACY_RECONSTRUCTED = "legacy_reconstructed_american_odds"


def _audit_name(value: object) -> str:
    return " ".join(str(value or "").split())


def _training_cutoff(
    value: date | datetime | str,
    precision: object,
    issued_at: datetime,
) -> tuple[str, str]:
    parsed_precision = str(precision or "").strip().lower()
    if parsed_precision == "date":
        cutoff = iso_date(value, "model_trained_through")
        if not date.fromisoformat(cutoff) < issued_at.date():
            raise MarketDataError(
                "date-only model_trained_through must precede the forecast UTC date"
            )
        return cutoff, parsed_precision
    if parsed_precision == "timestamp":
        cutoff_time = utc_datetime(value, "model_trained_through")
        if not cutoff_time < issued_at:
            raise MarketDataError(
                "timestamp model_trained_through must precede forecast_issued_at_utc"
            )
        return utc_text(cutoff_time, "model_trained_through"), parsed_precision
    raise MarketDataError(
        "model_training_cutoff_precision must be 'date' or 'timestamp'"
    )


@dataclass(frozen=True)
class ForecastCapture:
    """A model probability frozen independently at one retrieval/issue time."""

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
    model_probability: float
    model_id: str
    model_version: str
    model_trained_through: str
    model_training_cutoff_precision: str
    source_commit_sha: str
    probability_provenance: str
    legacy_predicted_american_odds: int | None

    FIELDNAMES = (
        "schema_version",
        "forecast_capture_id",
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
        "forecast_issued_at_utc",
        "model_probability",
        "model_id",
        "model_version",
        "model_trained_through",
        "model_training_cutoff_precision",
        "source_commit_sha",
        "probability_provenance",
        "legacy_predicted_american_odds",
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
        forecast_issued_at_utc: datetime | str,
        model_probability: object,
        model_id: object,
        model_version: object,
        model_trained_through: date | datetime | str,
        model_training_cutoff_precision: str,
        source_commit_sha: object,
        matchup_id: object | None = None,
        fight_id: object | None = None,
        fighter_name: object = "",
        opponent_name: object = "",
    ) -> "ForecastCapture":
        """Create a capture from a native, unrounded model probability."""

        return cls._build(
            event_id=event_id,
            capture_id=capture_id,
            fighter_id=fighter_id,
            opponent_id=opponent_id,
            event_date=event_date,
            timing_precision=timing_precision,
            event_start_utc=event_start_utc,
            forecast_issued_at_utc=forecast_issued_at_utc,
            model_probability=model_probability,
            model_id=model_id,
            model_version=model_version,
            model_trained_through=model_trained_through,
            model_training_cutoff_precision=model_training_cutoff_precision,
            source_commit_sha=source_commit_sha,
            matchup_id=matchup_id,
            fight_id=fight_id,
            fighter_name=fighter_name,
            opponent_name=opponent_name,
            provenance=NATIVE_PROBABILITY,
            legacy_odds=None,
        )

    @classmethod
    def from_legacy_american_odds(
        cls,
        *,
        predicted_american_odds: object,
        capture_id: object,
        event_id: object,
        fighter_id: object,
        opponent_id: object,
        event_date: date | str,
        timing_precision: str,
        event_start_utc: datetime | str | None,
        forecast_issued_at_utc: datetime | str,
        model_id: object,
        model_version: object,
        model_trained_through: date | datetime | str,
        model_training_cutoff_precision: str,
        source_commit_sha: object,
        matchup_id: object | None = None,
        fight_id: object | None = None,
        fighter_name: object = "",
        opponent_name: object = "",
    ) -> "ForecastCapture":
        """Reconstruct a lossy probability from explicitly tagged legacy odds."""

        legacy_odds = moneyline(
            predicted_american_odds, "predicted_american_odds"
        )
        return cls._build(
            event_id=event_id,
            capture_id=capture_id,
            fighter_id=fighter_id,
            opponent_id=opponent_id,
            event_date=event_date,
            timing_precision=timing_precision,
            event_start_utc=event_start_utc,
            forecast_issued_at_utc=forecast_issued_at_utc,
            model_probability=implied_probability(legacy_odds),
            model_id=model_id,
            model_version=model_version,
            model_trained_through=model_trained_through,
            model_training_cutoff_precision=model_training_cutoff_precision,
            source_commit_sha=source_commit_sha,
            matchup_id=matchup_id,
            fight_id=fight_id,
            fighter_name=fighter_name,
            opponent_name=opponent_name,
            provenance=LEGACY_RECONSTRUCTED,
            legacy_odds=legacy_odds,
        )

    @classmethod
    def _build(
        cls,
        *,
        capture_id: object,
        event_id: object,
        fighter_id: object,
        opponent_id: object,
        event_date: date | str,
        timing_precision: str,
        event_start_utc: datetime | str | None,
        forecast_issued_at_utc: datetime | str,
        model_probability: object,
        model_id: object,
        model_version: object,
        model_trained_through: date | datetime | str,
        model_training_cutoff_precision: str,
        source_commit_sha: object,
        matchup_id: object | None,
        fight_id: object | None,
        fighter_name: object,
        opponent_name: object,
        provenance: str,
        legacy_odds: int | None,
    ) -> "ForecastCapture":
        model_p = probability(model_probability, "model_probability")
        fighter, opponent, fighter_side, _, swapped = canonical_pair(
            fighter_id,
            opponent_id,
            (model_p, _audit_name(fighter_name)),
            (1.0 - model_p, _audit_name(opponent_name)),
        )
        model_p, fighter_display = fighter_side
        opponent_display = (
            _audit_name(fighter_name) if swapped else _audit_name(opponent_name)
        )
        canonical_legacy_odds = legacy_odds
        if swapped and legacy_odds is not None:
            canonical_legacy_odds = 100 if abs(legacy_odds) == 100 else -legacy_odds
        # Recompute legacy probability from the canonical moneyline instead of
        # complementing the source-side float.  Mathematically those operations
        # are identical, but for some lines (for example +884 / -884) they
        # differ by a few ulps.  Content-addressed records must rebuild to the
        # exact same bytes after canonical orientation and serialization.
        if canonical_legacy_odds is not None:
            model_p = implied_probability(canonical_legacy_odds)
        event = stable_id(event_id, "event_id")
        derived_matchup_id = matchup_id_for(event, fighter, opponent)
        if matchup_id is not None and str(matchup_id).strip():
            if stable_id(matchup_id, "matchup_id") != derived_matchup_id:
                raise MarketDataError(
                    "matchup_id does not match event_id and canonical fighter IDs"
                )
        issued, event_day, precision, event_start = require_before_event(
            forecast_issued_at_utc,
            event_date=event_date,
            timing_precision=timing_precision,
            event_start_utc=event_start_utc,
            observed_field="forecast_issued_at_utc",
        )
        trained_through, cutoff_precision = _training_cutoff(
            model_trained_through,
            model_training_cutoff_precision,
            issued,
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
            "forecast_issued_at_utc": utc_text(issued, "forecast_issued_at_utc"),
            "model_probability": model_p,
            "model_id": stable_id(model_id, "model_id"),
            "model_version": nonempty_text(model_version, "model_version"),
            "model_trained_through": trained_through,
            "model_training_cutoff_precision": cutoff_precision,
            "source_commit_sha": validated_git_commit(source_commit_sha),
            "probability_provenance": provenance,
            "legacy_predicted_american_odds": canonical_legacy_odds,
        }
        return cls(forecast_capture_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "ForecastCapture":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        if missing:
            raise MarketDataError(f"forecast capture is missing fields: {missing}")
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if extra:
            raise MarketDataError(f"forecast capture has unexpected fields: {extra}")
        try:
            schema_version = int(record["schema_version"])
        except (TypeError, ValueError) as error:
            raise MarketDataError("invalid forecast capture schema version") from error
        if schema_version != SCHEMA_VERSION:
            raise MarketDataError("unsupported forecast capture schema version")
        common = {
            "capture_id": record["capture_id"],
            "event_id": record["event_id"],
            "fighter_id": record["fighter_id"],
            "opponent_id": record["opponent_id"],
            "event_date": record["event_date"],
            "timing_precision": str(record["timing_precision"]),
            "event_start_utc": record["event_start_utc"],
            "forecast_issued_at_utc": record["forecast_issued_at_utc"],
            "model_id": record["model_id"],
            "model_version": record["model_version"],
            "model_trained_through": record["model_trained_through"],
            "model_training_cutoff_precision": record[
                "model_training_cutoff_precision"
            ],
            "source_commit_sha": record["source_commit_sha"],
            "matchup_id": record["matchup_id"],
            "fight_id": record["fight_id"],
            "fighter_name": record["fighter_name"],
            "opponent_name": record["opponent_name"],
        }
        provenance = str(record["probability_provenance"])
        if provenance == NATIVE_PROBABILITY:
            legacy_value = record["legacy_predicted_american_odds"]
            if legacy_value is not None and str(legacy_value).strip():
                raise MarketDataError("native probability cannot contain legacy odds")
            rebuilt = cls.create(
                model_probability=record["model_probability"], **common
            )
        elif provenance == LEGACY_RECONSTRUCTED:
            legacy_value = record["legacy_predicted_american_odds"]
            if legacy_value is None or not str(legacy_value).strip():
                raise MarketDataError("legacy reconstruction is missing its source odds")
            rebuilt = cls.from_legacy_american_odds(
                predicted_american_odds=legacy_value, **common
            )
            try:
                supplied_probability = float(record["model_probability"])
            except (TypeError, ValueError) as error:
                raise MarketDataError("model_probability must be numeric") from error
            if not math.isfinite(supplied_probability) or abs(
                supplied_probability - rebuilt.model_probability
            ) > 1e-12:
                raise MarketDataError(
                    "legacy model_probability disagrees with reconstructed odds"
                )
        else:
            raise MarketDataError("unsupported probability_provenance")
        if str(record["forecast_capture_id"]) != rebuilt.forecast_capture_id:
            raise MarketDataError(
                "forecast_capture_id does not match canonical capture contents"
            )
        return rebuilt

    @property
    def natural_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.matchup_id,
            self.capture_id,
            self.model_id,
            self.model_version,
            self.source_commit_sha,
        )

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


class ForecastCaptureStore:
    """Append-only, hash-validated atomic CSV/JSONL forecast mirror."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        if self.csv_path.resolve() == self.jsonl_path.resolve():
            raise ValueError("csv_path and jsonl_path must be different")

    @staticmethod
    def _records_by_id(
        records: Iterable[ForecastCapture],
    ) -> dict[str, ForecastCapture]:
        indexed: dict[str, ForecastCapture] = {}
        natural: dict[tuple[str, str, str, str, str], str] = {}
        capture_contract: dict[
            str, tuple[str, str, str, str, str, str]
        ] = {}
        for record in records:
            existing = indexed.get(record.forecast_capture_id)
            if existing is not None and existing != record:
                raise StoreIntegrityError("duplicate forecast_capture_id has different data")
            prior_id = natural.get(record.natural_key)
            if prior_id is not None and prior_id != record.forecast_capture_id:
                raise StoreIntegrityError("an immutable forecast capture was rewritten")
            indexed[record.forecast_capture_id] = record
            natural[record.natural_key] = record.forecast_capture_id
            contract = (
                record.event_id,
                record.forecast_issued_at_utc,
                record.model_id,
                record.model_version,
                record.model_trained_through,
                record.source_commit_sha,
            )
            prior_contract = capture_contract.setdefault(record.capture_id, contract)
            if prior_contract != contract:
                raise StoreIntegrityError(
                    "one capture_id must use one frozen event/model forecast issuance"
                )
        return indexed

    def _read_jsonl(self) -> list[ForecastCapture]:
        if not self.jsonl_path.exists():
            return []
        records: list[ForecastCapture] = []
        with self.jsonl_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise StoreIntegrityError(
                        f"blank forecast JSONL record at line {line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StoreIntegrityError(
                        f"invalid forecast JSONL at line {line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise StoreIntegrityError(
                        f"forecast JSONL line {line_number} is not an object"
                    )
                records.append(ForecastCapture.from_mapping(value))
        self._records_by_id(records)
        return records

    def _read_csv(self) -> list[ForecastCapture]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != ForecastCapture.FIELDNAMES:
                raise StoreIntegrityError("forecast CSV columns do not match the schema")
            records = [ForecastCapture.from_mapping(row) for row in reader]
        self._records_by_id(records)
        return records

    def read(self) -> tuple[ForecastCapture, ...]:
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
            raise StoreIntegrityError(
                "CSV and JSONL contain different data for a forecast_capture_id"
            )
        jsonl_ids = [record.forecast_capture_id for record in jsonl_records]
        csv_ids = [record.forecast_capture_id for record in csv_records]
        if set(csv_index) == set(jsonl_index):
            if csv_ids != jsonl_ids:
                raise StoreIntegrityError("CSV and JSONL forecast order diverged")
            return tuple(jsonl_records)
        if set(csv_index) < set(jsonl_index) and csv_ids == jsonl_ids[: len(csv_ids)]:
            return tuple(jsonl_records)
        if set(jsonl_index) < set(csv_index) and jsonl_ids == csv_ids[: len(jsonl_ids)]:
            return tuple(csv_records)
        raise StoreIntegrityError("CSV and JSONL forecast mirrors diverged")

    @staticmethod
    def _render_jsonl(records: Iterable[ForecastCapture]) -> str:
        return "".join(f"{canonical_json(record.to_mapping())}\n" for record in records)

    @staticmethod
    def _render_csv(records: Iterable[ForecastCapture]) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output, fieldnames=ForecastCapture.FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_mapping())
        return output.getvalue()

    def append(self, captures: Iterable[ForecastCapture]) -> AppendResult:
        pending = tuple(captures)
        lock_path = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock_path):
            existing = list(self.read())
            indexed = self._records_by_id(existing)
            natural = {record.natural_key: record.forecast_capture_id for record in existing}
            additions: list[ForecastCapture] = []
            duplicates: list[str] = []
            for capture in pending:
                if not isinstance(capture, ForecastCapture):
                    raise TypeError("append accepts ForecastCapture instances only")
                if capture.forecast_capture_id in indexed:
                    if indexed[capture.forecast_capture_id] != capture:
                        raise StoreIntegrityError("a forecast_capture_id was rewritten")
                    duplicates.append(capture.forecast_capture_id)
                    continue
                prior_id = natural.get(capture.natural_key)
                if prior_id is not None and prior_id != capture.forecast_capture_id:
                    raise StoreIntegrityError("an immutable forecast capture was rewritten")
                indexed[capture.forecast_capture_id] = capture
                natural[capture.natural_key] = capture.forecast_capture_id
                additions.append(capture)
            additions.sort(
                key=lambda item: (item.forecast_issued_at_utc, item.forecast_capture_id)
            )
            combined = [*existing, *additions]
            self._records_by_id(combined)
            atomic_write_text(self.jsonl_path, self._render_jsonl(combined))
            atomic_write_text(self.csv_path, self._render_csv(combined))
            return AppendResult(
                added_ids=tuple(item.forecast_capture_id for item in additions),
                duplicate_ids=tuple(duplicates),
                total_records=len(combined),
                dataset_sha256=canonical_hash(
                    [item.to_mapping() for item in combined]
                ),
            )
