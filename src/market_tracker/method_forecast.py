"""Immutable candidate outcome forecasts paired with method-price captures."""

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
    iso_date,
    matchup_id_for,
    nonempty_text,
    require_before_event,
    stable_id,
    utc_text,
    validated_git_commit,
    validated_sha256,
)
from ._storage import atomic_write_text, exclusive_store_lock
from .quotes import AppendResult


METHOD_FORECAST_CONTRACT = "fighter-method-forecast-capture-v1"
METHOD_FORECAST_OUTCOMES = (
    "fighter_ko_tko",
    "fighter_submission",
    "fighter_decision",
    "fighter_other",
    "opponent_ko_tko",
    "opponent_submission",
    "opponent_decision",
    "opponent_other",
)


def _probability(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise MarketDataError(f"{field} must be numeric") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise MarketDataError(f"{field} must be a finite probability")
    return parsed


def _boolean(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise MarketDataError(f"{field} must be a boolean")


@dataclass(frozen=True)
class MethodForecastCapture:
    schema_version: int
    forecast_id: str
    contract_version: str
    capture_id: str
    matchup_id: str
    event_id: str
    fighter_id: str
    opponent_id: str
    fighter_name: str
    opponent_name: str
    event_date: str
    event_start_utc: str
    observed_at_utc: str
    horizon: str
    forecast_issued_at_utc: str
    model_id: str
    model_version: str
    model_trained_through: str
    source_commit_sha: str
    training_input_sha256: str
    source_publication_sha256: str
    scheduled_rounds: int
    fighter_ko_tko_probability: float
    fighter_submission_probability: float
    fighter_decision_probability: float
    fighter_other_probability: float
    opponent_ko_tko_probability: float
    opponent_submission_probability: float
    opponent_decision_probability: float
    opponent_other_probability: float
    candidate_only: bool
    paper_only: bool
    execution_enabled: bool

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
        event_start_utc: object,
        observed_at_utc: object,
        horizon: object,
        forecast_issued_at_utc: object,
        model_id: object,
        model_version: object,
        model_trained_through: object,
        source_commit_sha: object,
        training_input_sha256: object,
        source_publication_sha256: object,
        scheduled_rounds: object,
        terminal_probabilities: Mapping[str, object],
        matchup_id: object | None = None,
    ) -> "MethodForecastCapture":
        fighter, opponent, fighter_display, opponent_display, reversed_pair = canonical_pair(
            fighter_id,
            opponent_id,
            " ".join(str(fighter_name or "").split()),
            " ".join(str(opponent_name or "").split()),
        )
        event = stable_id(event_id, "event_id")
        derived_matchup = matchup_id_for(event, fighter, opponent)
        if matchup_id is not None and str(matchup_id).strip():
            if stable_id(matchup_id, "matchup_id") != derived_matchup:
                raise MarketDataError("matchup_id disagrees with event and fighter IDs")
        observed, event_day, _, event_start = require_before_event(
            observed_at_utc,
            event_date=event_date,
            timing_precision="timestamp",
            event_start_utc=event_start_utc,
            observed_field="observed_at_utc",
        )
        issued, _, _, _ = require_before_event(
            forecast_issued_at_utc,
            event_date=event_day,
            timing_precision="timestamp",
            event_start_utc=event_start,
            observed_field="forecast_issued_at_utc",
        )
        if issued > observed:
            raise MarketDataError("forecast was issued after the method-price capture")
        normalized_horizon = str(horizon or "").strip().casefold()
        if normalized_horizon not in {"opening", "t72", "t24", "t6"}:
            raise MarketDataError("unsupported method forecast horizon")
        if set(terminal_probabilities) != set(METHOD_FORECAST_OUTCOMES):
            raise MarketDataError("terminal probabilities must contain all eight outcomes")
        probabilities = {
            outcome: _probability(terminal_probabilities[outcome], outcome)
            for outcome in METHOD_FORECAST_OUTCOMES
        }
        if abs(sum(probabilities.values()) - 1.0) > 1e-9:
            raise MarketDataError("terminal probabilities must sum to one")
        if reversed_pair:
            probabilities = {
                f"{side}_{method}": probabilities[f"{other}_{method}"]
                for side, other in (("fighter", "opponent"), ("opponent", "fighter"))
                for method in ("ko_tko", "submission", "decision", "other")
            }
        try:
            rounds = int(scheduled_rounds)
        except (TypeError, ValueError) as error:
            raise MarketDataError("scheduled_rounds must be an integer") from error
        if rounds not in {3, 5}:
            raise MarketDataError("scheduled_rounds must be three or five")
        body = {
            "schema_version": SCHEMA_VERSION,
            "contract_version": METHOD_FORECAST_CONTRACT,
            "capture_id": stable_id(capture_id, "capture_id"),
            "matchup_id": derived_matchup,
            "event_id": event,
            "fighter_id": fighter,
            "opponent_id": opponent,
            "fighter_name": fighter_display,
            "opponent_name": opponent_display,
            "event_date": event_day,
            "event_start_utc": event_start,
            "observed_at_utc": utc_text(observed, "observed_at_utc"),
            "horizon": normalized_horizon,
            "forecast_issued_at_utc": utc_text(issued, "forecast_issued_at_utc"),
            "model_id": stable_id(model_id, "model_id"),
            "model_version": nonempty_text(model_version, "model_version"),
            "model_trained_through": iso_date(model_trained_through, "model_trained_through"),
            "source_commit_sha": validated_git_commit(source_commit_sha),
            "training_input_sha256": validated_sha256(training_input_sha256, "training_input_sha256"),
            "source_publication_sha256": validated_sha256(source_publication_sha256, "source_publication_sha256"),
            "scheduled_rounds": rounds,
            **{f"{outcome}_probability": probabilities[outcome] for outcome in METHOD_FORECAST_OUTCOMES},
            "candidate_only": True,
            "paper_only": True,
            "execution_enabled": False,
        }
        return cls(forecast_id=canonical_hash(body), **body)

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "MethodForecastCapture":
        missing = sorted(set(cls.FIELDNAMES) - set(record))
        extra = sorted(str(key) for key in set(record) - set(cls.FIELDNAMES))
        if missing or extra:
            raise MarketDataError(
                f"method forecast schema mismatch; missing={missing}, extra={extra}"
            )
        terminal = {
            outcome: record[f"{outcome}_probability"]
            for outcome in METHOD_FORECAST_OUTCOMES
        }
        rebuilt = cls.create(
            capture_id=record["capture_id"],
            matchup_id=record["matchup_id"],
            event_id=record["event_id"],
            fighter_id=record["fighter_id"],
            opponent_id=record["opponent_id"],
            fighter_name=record["fighter_name"],
            opponent_name=record["opponent_name"],
            event_date=record["event_date"],
            event_start_utc=record["event_start_utc"],
            observed_at_utc=record["observed_at_utc"],
            horizon=record["horizon"],
            forecast_issued_at_utc=record["forecast_issued_at_utc"],
            model_id=record["model_id"],
            model_version=record["model_version"],
            model_trained_through=record["model_trained_through"],
            source_commit_sha=record["source_commit_sha"],
            training_input_sha256=record["training_input_sha256"],
            source_publication_sha256=record["source_publication_sha256"],
            scheduled_rounds=record["scheduled_rounds"],
            terminal_probabilities=terminal,
        )
        if int(record["schema_version"]) != SCHEMA_VERSION:
            raise MarketDataError("unsupported method forecast schema version")
        if str(record["forecast_id"]) != rebuilt.forecast_id:
            raise MarketDataError("method forecast ID does not match canonical contents")
        if str(record["contract_version"]) != METHOD_FORECAST_CONTRACT:
            raise MarketDataError("unsupported method forecast contract")
        if not _boolean(record["candidate_only"], "candidate_only"):
            raise MarketDataError("method forecast must remain candidate-only")
        if not _boolean(record["paper_only"], "paper_only") or _boolean(
            record["execution_enabled"], "execution_enabled"
        ):
            raise MarketDataError("method forecast must remain paper-only")
        return rebuilt

    @property
    def natural_key(self) -> tuple[str, str]:
        return self.matchup_id, self.horizon

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.FIELDNAMES}


class MethodForecastStore:
    """Atomic append-only CSV/JSONL mirrors for method forecasts."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        if self.csv_path.resolve() == self.jsonl_path.resolve():
            raise ValueError("method forecast CSV and JSONL paths must differ")

    @staticmethod
    def _index(records: Iterable[MethodForecastCapture]) -> dict[str, MethodForecastCapture]:
        indexed: dict[str, MethodForecastCapture] = {}
        natural: dict[tuple[str, str], str] = {}
        for record in records:
            if not isinstance(record, MethodForecastCapture):
                raise TypeError("store accepts MethodForecastCapture instances")
            previous = indexed.get(record.forecast_id)
            if previous is not None and previous != record:
                raise StoreIntegrityError("method forecast ID was rewritten")
            prior_id = natural.get(record.natural_key)
            if prior_id is not None and prior_id != record.forecast_id:
                raise StoreIntegrityError("method forecast natural key was rewritten")
            indexed[record.forecast_id] = record
            natural[record.natural_key] = record.forecast_id
        return indexed

    def _read_jsonl(self) -> list[MethodForecastCapture]:
        if not self.jsonl_path.exists():
            return []
        rows = []
        with self.jsonl_path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    raise StoreIntegrityError(f"blank method forecast JSONL line {number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise StoreIntegrityError("method forecast JSONL row is not an object")
                rows.append(MethodForecastCapture.from_mapping(value))
        self._index(rows)
        return rows

    def _read_csv(self) -> list[MethodForecastCapture]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != MethodForecastCapture.FIELDNAMES:
                raise StoreIntegrityError("method forecast CSV columns do not match")
            rows = [MethodForecastCapture.from_mapping(row) for row in reader]
        self._index(rows)
        return rows

    def read(self) -> tuple[MethodForecastCapture, ...]:
        jsonl, csv_rows = self._read_jsonl(), self._read_csv()
        if not jsonl:
            return tuple(csv_rows)
        if not csv_rows:
            return tuple(jsonl)
        json_index, csv_index = self._index(jsonl), self._index(csv_rows)
        common = set(json_index) & set(csv_index)
        if any(json_index[key] != csv_index[key] for key in common):
            raise StoreIntegrityError("method forecast CSV and JSONL mirrors disagree")
        json_ids = [row.forecast_id for row in jsonl]
        csv_ids = [row.forecast_id for row in csv_rows]
        if json_ids == csv_ids:
            return tuple(jsonl)
        if csv_ids == json_ids[: len(csv_ids)]:
            return tuple(jsonl)
        if json_ids == csv_ids[: len(json_ids)]:
            return tuple(csv_rows)
        raise StoreIntegrityError("method forecast CSV and JSONL mirrors diverged")

    @staticmethod
    def dataset_sha256(records: Iterable[MethodForecastCapture]) -> str:
        return canonical_hash([record.to_mapping() for record in records])

    @staticmethod
    def _jsonl_text(records: Iterable[MethodForecastCapture]) -> str:
        return "".join(f"{canonical_json(record.to_mapping())}\n" for record in records)

    @staticmethod
    def _csv_text(records: Iterable[MethodForecastCapture]) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=MethodForecastCapture.FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(record.to_mapping() for record in records)
        return output.getvalue()

    def append(self, records: Iterable[MethodForecastCapture]) -> AppendResult:
        pending = tuple(records)
        lock = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock):
            existing = list(self.read())
            indexed = self._index(existing)
            additions = []
            duplicates = []
            for record in pending:
                if not isinstance(record, MethodForecastCapture):
                    raise TypeError("append accepts MethodForecastCapture instances")
                previous = indexed.get(record.forecast_id)
                if previous is not None:
                    if previous != record:
                        raise StoreIntegrityError("method forecast ID was rewritten")
                    duplicates.append(record.forecast_id)
                    continue
                additions.append(record)
                indexed[record.forecast_id] = record
            combined = [*existing, *additions]
            self._index(combined)
            atomic_write_text(self.jsonl_path, self._jsonl_text(combined))
            atomic_write_text(self.csv_path, self._csv_text(combined))
        return AppendResult(
            added_ids=tuple(record.forecast_id for record in additions),
            duplicate_ids=tuple(duplicates),
            total_records=len(combined),
            dataset_sha256=self.dataset_sha256(combined),
        )
