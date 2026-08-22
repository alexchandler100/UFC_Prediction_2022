"""Canonical, source-attributed schema for non-UFC fight results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
import json
import math
import re
from typing import Mapping

import pandas as pd


SCHEMA_VERSION = 1
RESULTS = {"W", "L", "D", "NC"}


class ExternalDataError(ValueError):
    """Raised when an external observation violates the data contract."""


def clean_text(value: object) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split())


def stable_token(value: object) -> str:
    """Return a stable source identifier without treating a display name as one."""
    text = clean_text(value).rstrip("/")
    if not text:
        return ""
    return text.rsplit("/", 1)[-1]


def parse_optional_int(value: object) -> int | None:
    text = clean_text(value)
    if not text or text.casefold() in {"n/a", "na", "unknown", "none", "-"}:
        return None
    number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)) or float(number) % 1:
        raise ExternalDataError(f"expected an integer, got {value!r}")
    return int(number)


def parse_optional_seconds(value: object) -> int | None:
    text = clean_text(value)
    if not text or text.casefold() in {"n/a", "na", "unknown", "none", "-"}:
        return None
    if re.fullmatch(r"\d{1,3}:\d{2}", text):
        minutes, seconds = (int(part) for part in text.split(":"))
        if seconds >= 60:
            raise ExternalDataError(f"invalid clock {value!r}")
        return minutes * 60 + seconds
    parsed = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.isna(parsed) or not math.isfinite(float(parsed)) or float(parsed) < 0:
        raise ExternalDataError(f"invalid duration {value!r}")
    return int(round(float(parsed)))


def normalize_result(value: object) -> str:
    text = clean_text(value).casefold()
    aliases = {
        "w": "W", "win": "W", "winner": "W",
        "l": "L", "loss": "L", "lose": "L",
        "d": "D", "draw": "D",
        "nc": "NC", "no contest": "NC", "no-contest": "NC",
    }
    result = aliases.get(text, text.upper())
    if result not in RESULTS:
        raise ExternalDataError(f"unsupported result {value!r}")
    return result


def invert_result(result: str) -> str:
    return {"W": "L", "L": "W", "D": "D", "NC": "NC"}[result]


def normalize_method(method: object, details: object = "") -> str:
    main = clean_text(method).casefold()
    detail = clean_text(details).casefold()
    joined = f"{main} {detail}".strip()
    if "decision" in main:
        if "split" in joined:
            return "S-DEC"
        if "majority" in joined:
            return "M-DEC"
        return "U-DEC"
    if "submission" in main or main == "sub":
        return "SUB"
    if "technical submission" in joined:
        return "SUB"
    if "knockout" in main or "tko" in main or main == "ko":
        return "KO/TKO"
    if "disqualification" in main or main == "dq":
        return "DQ"
    if "overturned" in joined or main in {"nc", "no contest"}:
        return "CNC"
    return clean_text(method).upper() or "OTHER"


@dataclass(frozen=True)
class ExternalBoutObservation:
    """One deterministic perspective on one completed professional MMA bout."""

    schema_version: int
    observation_id: str
    source: str
    snapshot_sha256: str
    source_bout_id: str
    source_bout_order: int | None
    source_event_id: str
    source_url: str
    event_date: str
    event_name: str
    promotion: str
    discipline: str
    professional: bool
    fighter_source_id: str
    fighter_name: str
    opponent_source_id: str
    opponent_name: str
    result: str
    method: str
    division: str
    finish_round: int | None
    finish_clock_seconds: int | None
    scheduled_rounds: int | None

    @classmethod
    def create(
        cls,
        *,
        source: object,
        snapshot_sha256: object,
        source_bout_id: object,
        source_bout_order: object = None,
        source_event_id: object,
        source_url: object,
        event_date: object,
        event_name: object,
        promotion: object,
        fighter_source_id: object,
        fighter_name: object,
        opponent_source_id: object,
        opponent_name: object,
        result: object,
        method: object,
        division: object = "Unknown",
        finish_round: object = None,
        finish_clock_seconds: object = None,
        scheduled_rounds: object = None,
        discipline: object = "mma",
        professional: object = True,
    ) -> "ExternalBoutObservation":
        source_text = clean_text(source)
        bout_id = clean_text(source_bout_id)
        event_id = clean_text(source_event_id)
        first_id = clean_text(fighter_source_id)
        second_id = clean_text(opponent_source_id)
        first_name = clean_text(fighter_name)
        second_name = clean_text(opponent_name)
        normalized_result = normalize_result(result)
        if second_id < first_id:
            first_id, second_id = second_id, first_id
            first_name, second_name = second_name, first_name
            normalized_result = invert_result(normalized_result)
        observation_key = json.dumps(
            [source_text, bout_id], ensure_ascii=False, separators=(",", ":")
        )
        observation = cls(
            schema_version=SCHEMA_VERSION,
            observation_id=sha256(observation_key.encode("utf-8")).hexdigest(),
            source=source_text,
            snapshot_sha256=clean_text(snapshot_sha256).lower(),
            source_bout_id=bout_id,
            source_bout_order=parse_optional_int(source_bout_order),
            source_event_id=event_id,
            source_url=clean_text(source_url),
            event_date=pd.to_datetime(event_date, errors="raise").date().isoformat(),
            event_name=clean_text(event_name),
            promotion=clean_text(promotion),
            discipline=clean_text(discipline).casefold(),
            professional=bool(professional),
            fighter_source_id=first_id,
            fighter_name=first_name,
            opponent_source_id=second_id,
            opponent_name=second_name,
            result=normalized_result,
            method=normalize_method(method),
            division=clean_text(division) or "Unknown",
            finish_round=parse_optional_int(finish_round),
            finish_clock_seconds=parse_optional_seconds(finish_clock_seconds),
            scheduled_rounds=parse_optional_int(scheduled_rounds),
        )
        observation.validate()
        return observation

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ExternalBoutObservation":
        allowed = set(cls.__dataclass_fields__)
        try:
            observation = cls(**{key: row.get(key) for key in allowed})
        except TypeError as error:
            raise ExternalDataError(str(error)) from error
        observation.validate()
        return observation

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ExternalDataError(f"unsupported schema_version {self.schema_version!r}")
        required = {
            "observation_id": self.observation_id,
            "source": self.source,
            "snapshot_sha256": self.snapshot_sha256,
            "source_bout_id": self.source_bout_id,
            "source_event_id": self.source_event_id,
            "source_url": self.source_url,
            "event_name": self.event_name,
            "promotion": self.promotion,
            "fighter_source_id": self.fighter_source_id,
            "fighter_name": self.fighter_name,
            "opponent_source_id": self.opponent_source_id,
            "opponent_name": self.opponent_name,
            "method": self.method,
        }
        blanks = sorted(key for key, value in required.items() if not clean_text(value))
        if blanks:
            raise ExternalDataError(f"blank required fields: {blanks}")
        for field_name, value in (
            ("observation_id", self.observation_id),
            ("snapshot_sha256", self.snapshot_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ExternalDataError(f"{field_name} must be a lowercase SHA-256")
        parsed_date = date.fromisoformat(self.event_date)
        if parsed_date < date(1990, 1, 1) or parsed_date > date.today():
            raise ExternalDataError(f"implausible completed event date {self.event_date}")
        if self.discipline != "mma" or not self.professional:
            raise ExternalDataError("only completed professional MMA belongs in this ledger")
        if self.fighter_source_id == self.opponent_source_id:
            raise ExternalDataError("source fighter IDs collapse to the same person")
        if self.fighter_source_id > self.opponent_source_id:
            raise ExternalDataError("participant perspective is not canonical")
        if self.result not in RESULTS:
            raise ExternalDataError(f"unsupported result {self.result!r}")
        for field_name, value, lower, upper in (
            ("source_bout_order", self.source_bout_order, 0, 1000),
            ("finish_round", self.finish_round, 1, 9),
            ("finish_clock_seconds", self.finish_clock_seconds, 0, 3600),
            ("scheduled_rounds", self.scheduled_rounds, 1, 9),
        ):
            if value is not None and (not isinstance(value, int) or not lower <= value <= upper):
                raise ExternalDataError(f"{field_name} is outside [{lower}, {upper}]")
        if (
            self.finish_round is not None
            and self.scheduled_rounds is not None
            and self.finish_round > self.scheduled_rounds
        ):
            raise ExternalDataError("finish_round exceeds scheduled_rounds")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
