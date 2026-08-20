"""Shared validation and canonical serialization helpers.

The market tracker treats identifiers, timestamps, and hashes as data-contract
fields rather than presentation strings.  Keeping these rules in one module
prevents the quote store, evaluator, and paper ledger from drifting apart.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
import re
from typing import Any, Mapping


SCHEMA_VERSION = 1
BETTING_STATUS = "disabled_paper_only_no_execution"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class MarketDataError(ValueError):
    """Raised when a record violates the market-data contract."""


class StoreIntegrityError(RuntimeError):
    """Raised when append-only mirrors disagree or a key is rewritten."""


def stable_id(value: object, field: str) -> str:
    """Return a nonempty stable identifier, accepting a UFCStats URL.

    Display names are intentionally not accepted as a fallback.  URL inputs
    are reduced to their final path token so callers can pass either the raw
    UFCStats identifier or its canonical URL.
    """

    text = str(value or "").strip().rstrip("/")
    token = text.rsplit("/", 1)[-1]
    if not token or token in {".", ".."} or any(character.isspace() for character in token):
        raise MarketDataError(f"{field} must be a nonempty stable ID or URL")
    if len(token) > 256:
        raise MarketDataError(f"{field} is implausibly long")
    return token


def optional_stable_id(value: object, field: str) -> str | None:
    """Validate an optional stable ID, normalizing CSV blanks to ``None``."""

    if value is None or not str(value).strip():
        return None
    return stable_id(value, field)


def nonempty_text(value: object, field: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise MarketDataError(f"{field} must be nonempty")
    return text


def utc_datetime(value: datetime | str, field: str) -> datetime:
    """Parse an explicitly timezone-aware timestamp and normalize it to UTC."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise MarketDataError(f"{field} must be a timezone-aware timestamp")
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as error:
            raise MarketDataError(f"{field} is not a valid ISO-8601 timestamp") from error
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise MarketDataError(f"{field} must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MarketDataError(f"{field} must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def utc_text(value: datetime | str, field: str) -> str:
    parsed = utc_datetime(value, field)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def iso_date(value: date | str, field: str = "event_date") -> str:
    if isinstance(value, datetime):
        raise MarketDataError(f"{field} must be a calendar date, not a timestamp")
    if isinstance(value, date):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        try:
            parsed = date.fromisoformat(text)
        except ValueError as error:
            raise MarketDataError(f"{field} must use YYYY-MM-DD") from error
        if text != parsed.isoformat():
            raise MarketDataError(f"{field} must use canonical YYYY-MM-DD")
    else:
        raise MarketDataError(f"{field} must be a date or YYYY-MM-DD string")
    return parsed.isoformat()


def event_timing(
    event_date: date | str,
    timing_precision: object,
    event_start_utc: datetime | str | None,
) -> tuple[str, str, str | None]:
    """Validate honest date-only or timestamp event timing metadata."""

    event_day = iso_date(event_date)
    precision = str(timing_precision or "").strip().lower()
    if precision not in {"date", "timestamp"}:
        raise MarketDataError("timing_precision must be 'date' or 'timestamp'")
    if precision == "date":
        if event_start_utc is not None and str(event_start_utc).strip():
            raise MarketDataError(
                "event_start_utc must be blank when timing_precision is date"
            )
        return event_day, precision, None
    if event_start_utc is None or not str(event_start_utc).strip():
        raise MarketDataError(
            "event_start_utc is required when timing_precision is timestamp"
        )
    return event_day, precision, utc_text(event_start_utc, "event_start_utc")


def require_before_event(
    observed_at: datetime | str,
    *,
    event_date: date | str,
    timing_precision: object,
    event_start_utc: datetime | str | None,
    observed_field: str,
) -> tuple[datetime, str, str, str | None]:
    """Enforce the strongest pre-event assertion supported by source timing.

    Date-only events are accepted only when the UTC observation date is
    strictly before the event date.  This intentionally rejects same-day
    captures because no event start time is available to prove they are prior.
    """

    observed = utc_datetime(observed_at, observed_field)
    event_day, precision, event_start = event_timing(
        event_date, timing_precision, event_start_utc
    )
    if precision == "timestamp":
        if not observed < utc_datetime(event_start, "event_start_utc"):
            raise MarketDataError(
                f"{observed_field} must be strictly earlier than event_start_utc"
            )
    elif not observed.date() < date.fromisoformat(event_day):
        raise MarketDataError(
            f"{observed_field} UTC date must be strictly before a date-only event"
        )
    return observed, event_day, precision, event_start


def require_pre_event(
    observed_at: datetime | str,
    event_start: datetime | str,
    *,
    observed_field: str = "observed_at_utc",
) -> tuple[datetime, datetime]:
    observed = utc_datetime(observed_at, observed_field)
    event = utc_datetime(event_start, "event_start_utc")
    if not observed < event:
        raise MarketDataError(
            f"{observed_field} must be strictly earlier than event_start_utc"
        )
    return observed, event


def probability(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise MarketDataError(f"{field} must be numeric") from error
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise MarketDataError(f"{field} must be finite and strictly between zero and one")
    return parsed


def binary_target(value: object, field: str = "target") -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise MarketDataError(f"{field} must be 0 or 1") from error
    if not math.isfinite(parsed) or parsed not in (0.0, 1.0):
        raise MarketDataError(f"{field} must be 0 or 1")
    return int(parsed)


def moneyline(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise MarketDataError(f"{field} must be a nonzero integer moneyline")
    text = str(value).strip().replace("−", "-").replace("–", "-")
    if text.upper() in {"EV", "EVEN", "PK", "PICK"}:
        return 100
    try:
        number = float(text)
    except (TypeError, ValueError) as error:
        raise MarketDataError(f"{field} must be a nonzero integer moneyline") from error
    if not math.isfinite(number) or not number.is_integer() or int(number) == 0:
        raise MarketDataError(f"{field} must be a nonzero integer moneyline")
    parsed = int(number)
    if abs(parsed) < 100 or abs(parsed) > 100_000:
        raise MarketDataError(f"{field} is outside the supported moneyline range")
    return parsed


def implied_probability(odds: int) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def canonical_json(value: Mapping[str, Any] | list[Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_hash(value: Mapping[str, Any] | list[Any]) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_hash(payload: bytes | str | Mapping[str, Any] | list[Any]) -> str:
    if isinstance(payload, bytes):
        encoded = payload
    elif isinstance(payload, str):
        encoded = payload.encode("utf-8")
    elif isinstance(payload, (Mapping, list)):
        encoded = canonical_json(payload).encode("utf-8")
    else:
        raise MarketDataError("source payload must be bytes, text, a mapping, or a list")
    return sha256(encoded).hexdigest()


def validated_sha256(value: object, field: str) -> str:
    text = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise MarketDataError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def validated_git_commit(value: object, field: str = "source_commit_sha") -> str:
    text = str(value or "").strip().lower()
    if not _GIT_COMMIT_RE.fullmatch(text):
        raise MarketDataError(f"{field} must be a full 40- or 64-character commit SHA")
    return text


def canonical_pair(
    fighter_id: object,
    opponent_id: object,
    fighter_value: Any,
    opponent_value: Any,
) -> tuple[str, str, Any, Any, bool]:
    """Return lexicographically canonical IDs and aligned side values."""

    fighter = stable_id(fighter_id, "fighter_id")
    opponent = stable_id(opponent_id, "opponent_id")
    if fighter == opponent:
        raise MarketDataError("fighter_id and opponent_id must be different")
    if fighter < opponent:
        return fighter, opponent, fighter_value, opponent_value, False
    return opponent, fighter, opponent_value, fighter_value, True


def matchup_id_for(event_id: object, fighter_id: object, opponent_id: object) -> str:
    """Derive a stable pre-bout identity without pretending a fight ID exists."""

    event = stable_id(event_id, "event_id")
    fighter, opponent, _, _, _ = canonical_pair(
        fighter_id, opponent_id, None, None
    )
    digest = canonical_hash(
        {
            "identity_schema": 1,
            "event_id": event,
            "fighter_id": fighter,
            "opponent_id": opponent,
        }
    )
    return f"matchup_{digest}"
