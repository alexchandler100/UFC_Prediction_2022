"""Strict, prospective ledger for free public expert fight picks.

This module deliberately does not scrape or execute wagers.  It records only
picks whose public source and publication time can be preserved before a fight.
Advertised win rates or ROI are not accepted as evidence.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlparse

from ._common import (
    MarketDataError,
    SCHEMA_VERSION,
    StoreIntegrityError,
    canonical_hash,
    canonical_json,
    iso_date,
    moneyline,
    nonempty_text,
    require_before_event,
    stable_id,
    utc_datetime,
    utc_text,
    validated_sha256,
)
from ._storage import atomic_write_text, exclusive_store_lock
from .quotes import AppendResult


EXPERT_SIGNAL_CONTRACT = "prospective_free_public_pick_v1"


def _source_url(value: object) -> str:
    text = nonempty_text(value, "source_url")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MarketDataError("source_url must be a public HTTP(S) URL")
    if parsed.username or parsed.password:
        raise MarketDataError("source_url cannot contain credentials")
    return text


def _optional_moneyline(value: object) -> int | None:
    if value is None or not str(value).strip():
        return None
    return moneyline(value, "posted_moneyline")


@dataclass(frozen=True)
class ExpertSourcePolicy:
    analyst_id: str
    display_name: str
    enabled: bool
    free_public: bool
    timestamp_verifiable: bool
    allowed_hosts: tuple[str, ...]
    notes: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ExpertSourcePolicy":
        analyst_id = stable_id(value.get("analyst_id"), "analyst_id")
        hosts = tuple(
            sorted(
                {
                    str(host).strip().casefold().removeprefix("www.")
                    for host in value.get("allowed_hosts", ())
                    if str(host).strip()
                }
            )
        )
        if not hosts:
            raise MarketDataError(f"{analyst_id}: allowed_hosts cannot be empty")
        flags = (value.get("enabled", False), value.get("free_public", False), value.get("timestamp_verifiable", False))
        if any(not isinstance(flag, bool) for flag in flags):
            raise MarketDataError(f"{analyst_id}: registry flags must be JSON booleans")
        return cls(
            analyst_id=analyst_id,
            display_name=nonempty_text(value.get("display_name"), "display_name"),
            enabled=bool(value.get("enabled", False)),
            free_public=bool(value.get("free_public", False)),
            timestamp_verifiable=bool(value.get("timestamp_verifiable", False)),
            allowed_hosts=hosts,
            notes=" ".join(str(value.get("notes", "")).split()),
        )

    def validate_url(self, source_url: str) -> None:
        host = (urlparse(source_url).hostname or "").casefold().removeprefix("www.")
        if not any(host == allowed or host.endswith(f".{allowed}") for allowed in self.allowed_hosts):
            raise MarketDataError(
                f"{self.analyst_id}: source URL host {host!r} is not allowed"
            )
        if not (self.enabled and self.free_public and self.timestamp_verifiable):
            raise MarketDataError(
                f"{self.analyst_id}: source is not enabled as free, public, and timestamp-verifiable"
            )


def load_expert_source_registry(path: str | Path) -> dict[str, ExpertSourcePolicy]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MarketDataError(f"cannot read expert source registry {source}: {error}") from error
    rows = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise MarketDataError("expert source registry must contain a sources list")
    policies = [ExpertSourcePolicy.from_mapping(row) for row in rows]
    result = {policy.analyst_id: policy for policy in policies}
    if len(result) != len(policies):
        raise MarketDataError("expert source registry contains duplicate analyst IDs")
    return result


@dataclass(frozen=True)
class ExpertPick:
    schema_version: int
    pick_id: str
    contract: str
    observed_at_utc: str
    issued_at_utc: str
    event_date: str
    timing_precision: str
    event_start_utc: str
    analyst_id: str
    source_url: str
    source_record_id: str
    source_text_sha256: str
    event_id: str
    matchup_id: str
    selected_fighter_id: str
    opponent_id: str
    selected_fighter_name: str
    opponent_name: str
    market: str
    posted_moneyline: str
    paper_only: bool
    execution_enabled: bool

    FIELDNAMES = (
        "schema_version", "pick_id", "contract", "observed_at_utc",
        "issued_at_utc", "event_date", "timing_precision", "event_start_utc",
        "analyst_id", "source_url", "source_record_id", "source_text_sha256",
        "event_id", "matchup_id", "selected_fighter_id", "opponent_id",
        "selected_fighter_name", "opponent_name", "market", "posted_moneyline",
        "paper_only", "execution_enabled",
    )

    @classmethod
    def create(
        cls,
        *,
        observed_at_utc: object,
        issued_at_utc: object,
        event_date: object,
        timing_precision: object,
        event_start_utc: object = None,
        analyst_id: object,
        source_url: object,
        source_record_id: object,
        source_text_sha256: object,
        event_id: object,
        matchup_id: object,
        selected_fighter_id: object,
        opponent_id: object,
        selected_fighter_name: object,
        opponent_name: object,
        market: object = "moneyline",
        posted_moneyline: object = None,
    ) -> "ExpertPick":
        observed, event_day, precision, event_start = require_before_event(
            observed_at_utc,
            event_date=event_date,
            timing_precision=timing_precision,
            event_start_utc=event_start_utc,
            observed_field="observed_at_utc",
        )
        issued, _, _, _ = require_before_event(
            issued_at_utc,
            event_date=event_day,
            timing_precision=precision,
            event_start_utc=event_start,
            observed_field="issued_at_utc",
        )
        if issued > observed:
            raise MarketDataError("issued_at_utc cannot be later than observed_at_utc")
        selected_id = stable_id(selected_fighter_id, "selected_fighter_id")
        other_id = stable_id(opponent_id, "opponent_id")
        if selected_id == other_id:
            raise MarketDataError("selected fighter and opponent must differ")
        market_key = nonempty_text(market, "market").casefold()
        if market_key != "moneyline":
            raise MarketDataError("v1 expert signals support only straight moneyline picks")
        price = _optional_moneyline(posted_moneyline)
        identity = {
            "contract": EXPERT_SIGNAL_CONTRACT,
            "analyst_id": stable_id(analyst_id, "analyst_id"),
            "source_record_id": stable_id(source_record_id, "source_record_id"),
            "event_id": stable_id(event_id, "event_id"),
            "matchup_id": stable_id(matchup_id, "matchup_id"),
            "selected_fighter_id": selected_id,
            "opponent_id": other_id,
            "market": market_key,
        }
        return cls(
            schema_version=SCHEMA_VERSION,
            pick_id=canonical_hash(identity),
            contract=EXPERT_SIGNAL_CONTRACT,
            observed_at_utc=utc_text(observed, "observed_at_utc"),
            issued_at_utc=utc_text(issued, "issued_at_utc"),
            event_date=iso_date(event_day),
            timing_precision=precision,
            event_start_utc=event_start or "",
            analyst_id=identity["analyst_id"],
            source_url=_source_url(source_url),
            source_record_id=identity["source_record_id"],
            source_text_sha256=validated_sha256(source_text_sha256, "source_text_sha256"),
            event_id=identity["event_id"],
            matchup_id=identity["matchup_id"],
            selected_fighter_id=selected_id,
            opponent_id=other_id,
            selected_fighter_name=nonempty_text(selected_fighter_name, "selected_fighter_name"),
            opponent_name=nonempty_text(opponent_name, "opponent_name"),
            market=market_key,
            posted_moneyline="" if price is None else str(price),
            paper_only=True,
            execution_enabled=False,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ExpertPick":
        rebuilt = cls.create(**{key: value.get(key) for key in cls.FIELDNAMES if key not in {
            "schema_version", "pick_id", "contract", "paper_only", "execution_enabled"
        }})
        if int(value.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
            raise StoreIntegrityError("unsupported expert pick schema version")
        if str(value.get("contract", EXPERT_SIGNAL_CONTRACT)) != EXPERT_SIGNAL_CONTRACT:
            raise StoreIntegrityError("expert pick contract was rewritten")
        paper = str(value.get("paper_only", "true")).strip().casefold()
        execution = str(value.get("execution_enabled", "false")).strip().casefold()
        if paper not in {"true", "1"} or execution not in {"false", "0"}:
            raise StoreIntegrityError("expert pick paper/execution status was rewritten")
        supplied = str(value.get("pick_id", rebuilt.pick_id)).strip()
        if supplied and supplied != rebuilt.pick_id:
            raise StoreIntegrityError("expert pick ID does not match its logical identity")
        return rebuilt

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.FIELDNAMES}


class ExpertPickStore:
    """Crash-safe CSV/JSONL mirrors with immutable pick identities."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        if self.csv_path.resolve() == self.jsonl_path.resolve():
            raise ValueError("csv_path and jsonl_path must differ")

    def _read_csv(self) -> tuple[ExpertPick, ...]:
        if not self.csv_path.exists():
            return ()
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            return tuple(ExpertPick.from_mapping(row) for row in csv.DictReader(source))

    def _read_jsonl(self) -> tuple[ExpertPick, ...]:
        if not self.jsonl_path.exists():
            return ()
        with self.jsonl_path.open("r", encoding="utf-8") as source:
            return tuple(
                ExpertPick.from_mapping(json.loads(line))
                for line in source
                if line.strip()
            )

    def read(self) -> tuple[ExpertPick, ...]:
        csv_rows, jsonl_rows = self._read_csv(), self._read_jsonl()
        if csv_rows and jsonl_rows and csv_rows != jsonl_rows:
            raise StoreIntegrityError("expert-pick CSV and JSONL mirrors disagree")
        return jsonl_rows or csv_rows

    @staticmethod
    def _csv_text(rows: Iterable[ExpertPick]) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=ExpertPick.FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
        return output.getvalue()

    @staticmethod
    def _jsonl_text(rows: Iterable[ExpertPick]) -> str:
        return "".join(f"{canonical_json(row.to_dict())}\n" for row in rows)

    def append(self, records: Iterable[ExpertPick]) -> AppendResult:
        incoming = tuple(records)
        lock = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock):
            existing = self.read()
            by_id = {row.pick_id: row for row in existing}
            added: list[str] = []
            duplicates: list[str] = []
            for row in incoming:
                if not isinstance(row, ExpertPick):
                    raise TypeError("append accepts ExpertPick instances")
                previous = by_id.get(row.pick_id)
                if previous is not None:
                    if previous != row:
                        raise StoreIntegrityError(f"expert pick {row.pick_id} was rewritten")
                    duplicates.append(row.pick_id)
                    continue
                by_id[row.pick_id] = row
                added.append(row.pick_id)
            combined = (*existing, *(by_id[key] for key in added))
            atomic_write_text(self.csv_path, self._csv_text(combined))
            atomic_write_text(self.jsonl_path, self._jsonl_text(combined))
        return AppendResult(
            added_ids=tuple(added),
            duplicate_ids=tuple(duplicates),
            total_records=len(combined),
            dataset_sha256=canonical_hash([row.to_dict() for row in combined]),
        )


def validate_expert_pick(policy: ExpertSourcePolicy, pick: ExpertPick) -> None:
    if policy.analyst_id != pick.analyst_id:
        raise MarketDataError("expert pick analyst does not match source policy")
    policy.validate_url(pick.source_url)
