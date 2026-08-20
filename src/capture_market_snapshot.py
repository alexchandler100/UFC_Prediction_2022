"""Capture one frozen-model/FightOdds market observation without republishing.

This command is intentionally separate from the authoritative UFCStats/model
update.  It treats ``card_info.json``, ``vegas_odds.json`` and
``winner_model.json`` as immutable inputs, maps source display names only when
they identify one published stable-ID matchup, and appends validated quote and
forecast captures to the market-tracker ledgers.  It never creates a wager or
modifies a website publication file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

import pandas as pd

from fight_stat_helpers import same_name
from market_tracker import (
    BETTING_STATUS,
    ForecastCapture,
    ForecastCaptureStore,
    MarketDataError,
    QuoteSnapshot,
    QuoteSnapshotStore,
    matchup_id_for,
)
from odds_getter import OddsGetter


SOURCE = "fightodds.io"
ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = ROOT / "content" / "data" / "external"
MARKET_ROOT = ROOT / "content" / "data" / "market"
CARD_PATH = EXTERNAL_ROOT / "card_info.json"
VEGAS_PATH = EXTERNAL_ROOT / "vegas_odds.json"
MODEL_PATH = EXTERNAL_ROOT / "winner_model.json"
QUOTE_CSV_PATH = MARKET_ROOT / "quote_snapshots.csv"
QUOTE_JSONL_PATH = MARKET_ROOT / "quote_snapshots.jsonl"
FORECAST_CSV_PATH = MARKET_ROOT / "forecast_captures.csv"
FORECAST_JSONL_PATH = MARKET_ROOT / "forecast_captures.jsonl"
REPORT_PATH = MARKET_ROOT / "capture_report.json"
REPORT_SIZE_LIMIT = 64 * 1024
SOURCE_RETRY_DELAYS_SECONDS = (15.0, 60.0)


class CaptureError(RuntimeError):
    """Raised when the frozen publication and fresh source cannot be joined."""


@dataclass(frozen=True)
class PublishedMatchup:
    fighter_name: str
    opponent_name: str
    fighter_id: str | None
    opponent_id: str | None
    matchup_id: str | None
    fight_id: str | None
    model_probability: float
    model_status: str
    forecast_issued_at_utc: str
    forecast_source_commit: str


@dataclass(frozen=True)
class SourceMatch:
    source_row: dict[str, object]
    published: PublishedMatchup
    source_is_reversed: bool


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    # Pandas/numpy scalar missing checks may return ``numpy.bool_`` rather
    # than the built-in bool.  Refuse to stringify either kind as ``nan`` or
    # ``<NA>``; non-scalar containers are never valid text fields here.
    if not hasattr(missing, "__len__") and bool(missing):
        return ""
    return " ".join(str(value).split())


def _stable_token(value: object, field: str) -> str:
    text = _text(value).rstrip("/")
    token = text.rsplit("/", 1)[-1]
    if not token or any(character.isspace() for character in token):
        raise CaptureError(f"{field} must contain a stable ID or URL")
    return token


def _optional_token(value: object, field: str) -> str | None:
    return _stable_token(value, field) if _text(value) else None


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise CaptureError(f"required frozen publication file is unreadable: {path}") from error


def _json_object(payload: bytes, path: Path) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"{path.name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CaptureError(f"{path.name} must contain a JSON object")
    return value


def _publication_payloads() -> dict[Path, bytes]:
    return {
        path: _read_bytes(path)
        for path in (CARD_PATH, VEGAS_PATH, MODEL_PATH)
    }


def _publication_hashes(payloads: dict[Path, bytes]) -> dict[str, str]:
    return {
        path.name: sha256(payload).hexdigest()
        for path, payload in payloads.items()
    }


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CaptureError(f"published vegas_odds is missing columns: {missing}")


def _one_nonempty_value(frame: pd.DataFrame, column: str) -> str:
    values = {_text(value) for value in frame[column] if _text(value)}
    if len(values) != 1:
        raise CaptureError(f"published {column!r} must contain one nonempty value")
    return next(iter(values))


def _event_date(value: object) -> str:
    parsed = pd.to_datetime(_text(value), errors="coerce")
    if pd.isna(parsed):
        raise CaptureError("card_info date is invalid")
    return parsed.date().isoformat()


def _as_utc(value: object, field: str) -> datetime:
    parsed = pd.to_datetime(_text(value), errors="coerce")
    if (
        pd.isna(parsed)
        or getattr(parsed, "tzinfo", None) is None
        or parsed.utcoffset() is None
    ):
        raise CaptureError(f"{field} must be a timezone-aware timestamp")
    return parsed.to_pydatetime().astimezone(timezone.utc)


def _published_matchups(
    vegas: pd.DataFrame,
    card: dict[str, object],
    artifact: dict[str, object],
    observed_at: datetime,
) -> tuple[str, str, str, str, str, tuple[PublishedMatchup, ...]]:
    required = {
        "fighter name",
        "opponent name",
        "date",
        "event id",
        "event url",
        "fighter id",
        "opponent id",
        "model id",
        "model version",
        "model trained through",
        "model probability",
        "model status",
        "forecast issued at",
        "forecast source commit",
        "betting status",
    }
    _require_columns(vegas, required)
    if vegas.empty:
        raise CaptureError("published vegas_odds contains no upcoming matchups")

    event_day = _event_date(card.get("date"))
    # UFCStats currently publishes only a calendar date for upcoming events.
    # A same-UTC-day observation cannot be proven pre-event and is rejected.
    if observed_at.date() >= date.fromisoformat(event_day):
        raise CaptureError(
            "market capture requires an observation UTC date strictly before "
            "the date-only UFC event"
        )

    event_url = _text(card.get("event_url"))
    event_id = _stable_token(card.get("event_id"), "card_info event_id")
    if not event_url or _stable_token(event_url, "card_info event_url") != event_id:
        raise CaptureError("card_info event URL and event ID disagree")
    title = _text(card.get("title"))
    if not title:
        raise CaptureError("card_info title is blank")

    model_id = _stable_token(artifact.get("model_id"), "winner model_id")
    model_version = _text(artifact.get("model_version"))
    trained_through = _text(artifact.get("data_through"))
    if not model_version or not trained_through:
        raise CaptureError("winner model metadata is incomplete")

    vegas_dates = pd.to_datetime(vegas["date"], errors="coerce")
    if vegas_dates.isna().any() or not vegas_dates.dt.date.map(
        lambda value: value.isoformat() == event_day
    ).all():
        raise CaptureError("published matchup dates do not match card_info")
    if not vegas["event id"].map(_text).eq(event_id).all():
        raise CaptureError("published matchup event IDs do not match card_info")
    if not vegas["event url"].map(lambda value: _text(value).rstrip("/")).eq(
        event_url.rstrip("/")
    ).all():
        raise CaptureError("published matchup event URLs do not match card_info")
    if not vegas["model id"].map(_text).eq(model_id).all():
        raise CaptureError("published matchup model IDs do not match winner_model")
    if not vegas["model version"].map(_text).eq(model_version).all():
        raise CaptureError("published matchup model versions do not match winner_model")
    if not vegas["model trained through"].map(_text).eq(trained_through).all():
        raise CaptureError("published matchup training cutoffs do not match winner_model")
    if not vegas["betting status"].map(_text).str.casefold().str.startswith(
        "disabled"
    ).all():
        raise CaptureError("all published betting statuses must remain disabled")

    # These values should be shared by the whole frozen forecast issuance.
    forecast_issued = _one_nonempty_value(vegas, "forecast issued at")
    forecast_commit = _one_nonempty_value(vegas, "forecast source commit")
    if _as_utc(forecast_issued, "forecast issued at") > observed_at:
        raise CaptureError("published model forecast was issued after this retrieval")

    matchups: list[PublishedMatchup] = []
    seen_matchups: set[str] = set()
    for row in vegas.to_dict("records"):
        fighter_name = _text(row.get("fighter name"))
        opponent_name = _text(row.get("opponent name"))
        if not fighter_name or not opponent_name:
            raise CaptureError("published matchup has a blank audit name")
        status = _text(row.get("model status"))
        fighter_id = _optional_token(row.get("fighter id"), "fighter id")
        opponent_id = _optional_token(row.get("opponent id"), "opponent id")
        matchup_id: str | None = None
        if fighter_id and opponent_id and fighter_id != opponent_id:
            matchup_id = matchup_id_for(event_id, fighter_id, opponent_id)
            if matchup_id in seen_matchups:
                raise CaptureError(
                    "published vegas_odds contains a duplicate stable-ID matchup"
                )
            seen_matchups.add(matchup_id)
        elif status.casefold() != "abstain_unresolved_identity":
            raise CaptureError(
                "a non-identity-abstention matchup lacks two distinct stable fighter IDs"
            )
        if status.casefold().startswith("abstain"):
            # It remains part of card-mismatch detection, but cannot become a
            # forecast capture because no valid model probability was issued.
            probability = math.nan
        else:
            try:
                probability = float(row.get("model probability"))
            except (TypeError, ValueError) as error:
                raise CaptureError("resolved matchup model probability is not numeric") from error
            if not math.isfinite(probability) or not 0.0 < probability < 1.0:
                raise CaptureError(
                    "resolved matchup model probability must be strictly between zero and one"
                )
        matchups.append(
            PublishedMatchup(
                fighter_name=fighter_name,
                opponent_name=opponent_name,
                fighter_id=fighter_id,
                opponent_id=opponent_id,
                matchup_id=matchup_id,
                fight_id=_optional_token(row.get("fight id"), "fight id"),
                model_probability=probability,
                model_status=status,
                forecast_issued_at_utc=forecast_issued,
                forecast_source_commit=forecast_commit,
            )
        )
    return (
        event_day,
        event_id,
        event_url,
        title,
        model_version,
        tuple(matchups),
    )


def _map_source_rows(
    odds: pd.DataFrame,
    published: tuple[PublishedMatchup, ...],
) -> tuple[tuple[SourceMatch, ...], int]:
    if odds.empty:
        raise CaptureError("FightOdds returned no matchup rows")
    _require_columns(odds, {"fighter name", "opponent name"})
    matches: list[SourceMatch] = []
    used_matchups: set[str] = set()
    unmatched_rows = 0
    for source_row in odds.to_dict("records"):
        source_fighter = _text(source_row.get("fighter name"))
        source_opponent = _text(source_row.get("opponent name"))
        candidates: list[tuple[PublishedMatchup, bool]] = []
        for matchup in published:
            # Display names may help join an external row, but no quote is
            # admitted unless both UFCStats fighter IDs yielded a stable
            # matchup identity in the frozen publication.
            if matchup.matchup_id is None:
                continue
            direct = same_name(source_fighter, matchup.fighter_name) and same_name(
                source_opponent, matchup.opponent_name
            )
            reverse = same_name(source_fighter, matchup.opponent_name) and same_name(
                source_opponent, matchup.fighter_name
            )
            if direct:
                candidates.append((matchup, False))
            if reverse:
                candidates.append((matchup, True))
        # FightOdds can expose more than one event.  Unrelated rows are safe to
        # skip; ambiguous rows are not.  A strong frozen-card coverage check
        # below still rejects a previous/wrong page.
        if not candidates:
            unmatched_rows += 1
            continue
        if len(candidates) > 1:
            raise CaptureError(
                "FightOdds contains an ambiguous matchup: "
                f"{source_fighter!r} vs {source_opponent!r} matched "
                f"{len(candidates)} published rows"
            )
        matchup, reversed_source = candidates[0]
        if matchup.matchup_id in used_matchups:
            raise CaptureError("FightOdds contains duplicate orientations of one matchup")
        used_matchups.add(matchup.matchup_id)
        matches.append(SourceMatch(source_row, matchup, reversed_source))

    minimum_identifying_rows = (
        1 if len(published) == 1 else max(2, math.ceil(len(published) * 0.5))
    )
    if len(matches) < minimum_identifying_rows:
        raise CaptureError(
            "too few uniquely matching FightOdds rows to identify the published card: "
            f"{len(matches)}/{len(published)} (required {minimum_identifying_rows})"
        )
    return tuple(matches), unmatched_rows


def _source_payload_sha256(odds: pd.DataFrame) -> str:
    """Fingerprint the complete parsed retrieval, not one selected book row."""

    try:
        payload = json.loads(
            odds.to_json(orient="split", date_format="iso", date_unit="us")
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise CaptureError("FightOdds table cannot be canonically fingerprinted") from error
    return _canonical_hash({"source": SOURCE, "parsed_table": payload})


def _retrieve_fresh_odds() -> pd.DataFrame:
    """Retry only the isolated browser retrieval, using a fresh driver each time."""

    failures: list[str] = []
    for attempt in range(len(SOURCE_RETRY_DELAYS_SECONDS) + 1):
        try:
            odds = OddsGetter().make_odds_df()
            if not isinstance(odds, pd.DataFrame) or odds.empty:
                raise CaptureError("FightOdds retrieval returned no table rows")
            return odds
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
            if attempt >= len(SOURCE_RETRY_DELAYS_SECONDS):
                break
            delay = SOURCE_RETRY_DELAYS_SECONDS[attempt]
            print(
                f"FightOdds retrieval attempt {attempt + 1} failed; "
                f"retrying with a fresh browser in {delay:g}s ({failures[-1]})"
            )
            time.sleep(delay)
    raise CaptureError(
        "FightOdds retrieval failed after three fresh-browser attempts: "
        + " | ".join(failures)
    )


def _book_columns(odds: pd.DataFrame) -> tuple[tuple[str, str, str], ...]:
    fighter: dict[str, tuple[str, str]] = {}
    opponent: dict[str, tuple[str, str]] = {}
    for column in odds.columns:
        text = str(column)
        if text.startswith("fighter ") and text != "fighter name":
            book = text[len("fighter ") :].strip()
            key = book.casefold()
            if not book or key in fighter:
                raise CaptureError("FightOdds contains duplicate fighter book columns")
            fighter[key] = (book, text)
        elif text.startswith("opponent ") and text != "opponent name":
            book = text[len("opponent ") :].strip()
            key = book.casefold()
            if not book or key in opponent:
                raise CaptureError("FightOdds contains duplicate opponent book columns")
            opponent[key] = (book, text)
    if set(fighter) != set(opponent) or not fighter:
        raise CaptureError("FightOdds fighter/opponent book columns are incomplete")
    return tuple(
        (fighter[key][0], fighter[key][1], opponent[key][1])
        for key in sorted(fighter)
    )


def _prior_first_seen(
    existing: tuple[QuoteSnapshot, ...],
    candidate: QuoteSnapshot,
) -> str | None:
    matching = [
        item.quote_first_seen_at_utc
        for item in existing
        if item.matchup_id == candidate.matchup_id
        and item.source.casefold() == candidate.source.casefold()
        and item.book.casefold() == candidate.book.casefold()
        and item.fighter_moneyline == candidate.fighter_moneyline
        and item.opponent_moneyline == candidate.opponent_moneyline
    ]
    return min(matching) if matching else None


def _capture_id(observed_at: datetime) -> str:
    supplied = _text(os.environ.get("MARKET_CAPTURE_ID"))
    if supplied:
        return _stable_token(supplied, "MARKET_CAPTURE_ID")
    run_id = _text(os.environ.get("GITHUB_RUN_ID")) or "local"
    attempt = _text(os.environ.get("GITHUB_RUN_ATTEMPT")) or "1"
    timestamp = observed_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"capture_{timestamp}_{run_id}_{attempt}_{uuid.uuid4().hex[:12]}"


def _build_captures(
    source_matches: tuple[SourceMatch, ...],
    book_columns: tuple[tuple[str, str, str], ...],
    existing_quotes: tuple[QuoteSnapshot, ...],
    *,
    capture_id: str,
    event_id: str,
    event_day: str,
    observed_at: datetime,
    artifact: dict[str, object],
    source_payload_sha256: str,
) -> tuple[tuple[QuoteSnapshot, ...], tuple[ForecastCapture, ...], dict[str, int]]:
    quotes: list[QuoteSnapshot] = []
    forecasts: list[ForecastCapture] = []
    counters = {
        "incomplete_quote_pairs": 0,
        "invalid_quote_pairs": 0,
        "matchups_without_valid_quotes": 0,
        "matchups_without_model_forecast": 0,
        "paper_evaluable_matchups": 0,
    }
    parser = OddsGetter()
    for source_match in source_matches:
        source_row = source_match.source_row
        matchup = source_match.published
        if (
            matchup.matchup_id is None
            or matchup.fighter_id is None
            or matchup.opponent_id is None
        ):
            raise CaptureError("an unstable-identity matchup reached capture construction")
        matchup_quotes: list[QuoteSnapshot] = []
        for book, fighter_column, opponent_column in book_columns:
            if source_match.source_is_reversed:
                raw_fighter = source_row.get(opponent_column)
                raw_opponent = source_row.get(fighter_column)
            else:
                raw_fighter = source_row.get(fighter_column)
                raw_opponent = source_row.get(opponent_column)
            fighter_line = parser.parse_american_odds(raw_fighter)
            opponent_line = parser.parse_american_odds(raw_opponent)
            if fighter_line is None or opponent_line is None:
                counters["incomplete_quote_pairs"] += 1
                continue
            try:
                provisional = QuoteSnapshot.create(
                    capture_id=capture_id,
                    event_id=event_id,
                    fighter_id=matchup.fighter_id,
                    opponent_id=matchup.opponent_id,
                    fighter_name=matchup.fighter_name,
                    opponent_name=matchup.opponent_name,
                    matchup_id=matchup.matchup_id,
                    fight_id=matchup.fight_id,
                    event_date=event_day,
                    timing_precision="date",
                    event_start_utc=None,
                    observed_at_utc=observed_at,
                    source=SOURCE,
                    book=book,
                    fighter_moneyline=fighter_line,
                    opponent_moneyline=opponent_line,
                    source_payload_sha256=source_payload_sha256,
                )
                first_seen = _prior_first_seen(existing_quotes, provisional)
                snapshot = (
                    provisional
                    if first_seen is None
                    else QuoteSnapshot.create(
                        capture_id=capture_id,
                        event_id=event_id,
                        fighter_id=matchup.fighter_id,
                        opponent_id=matchup.opponent_id,
                        fighter_name=matchup.fighter_name,
                        opponent_name=matchup.opponent_name,
                        matchup_id=matchup.matchup_id,
                        fight_id=matchup.fight_id,
                        event_date=event_day,
                        timing_precision="date",
                        event_start_utc=None,
                        observed_at_utc=observed_at,
                        quote_first_seen_at_utc=first_seen,
                        source=SOURCE,
                        book=book,
                        fighter_moneyline=fighter_line,
                        opponent_moneyline=opponent_line,
                        source_payload_sha256=source_payload_sha256,
                    )
                )
            except MarketDataError:
                counters["invalid_quote_pairs"] += 1
                continue
            matchup_quotes.append(snapshot)

        if not matchup_quotes:
            counters["matchups_without_valid_quotes"] += 1
            continue
        # Market coverage is useful even when the model abstains; keeping it
        # prevents forecast availability from selecting the quote sample.
        quotes.extend(matchup_quotes)
        if matchup.model_status.casefold().startswith("abstain") or not math.isfinite(
            matchup.model_probability
        ):
            counters["matchups_without_model_forecast"] += 1
            continue
        if len(matchup_quotes) >= 4:
            counters["paper_evaluable_matchups"] += 1
        forecasts.append(
            ForecastCapture.create(
                capture_id=capture_id,
                event_id=event_id,
                fighter_id=matchup.fighter_id,
                opponent_id=matchup.opponent_id,
                fighter_name=matchup.fighter_name,
                opponent_name=matchup.opponent_name,
                matchup_id=matchup.matchup_id,
                fight_id=matchup.fight_id,
                event_date=event_day,
                timing_precision="date",
                event_start_utc=None,
                forecast_issued_at_utc=matchup.forecast_issued_at_utc,
                model_probability=matchup.model_probability,
                model_id=artifact["model_id"],
                model_version=artifact["model_version"],
                model_trained_through=artifact["data_through"],
                model_training_cutoff_precision="date",
                source_commit_sha=matchup.forecast_source_commit,
            )
        )
    if not quotes or not forecasts:
        raise CaptureError(
            "FightOdds contained no valid paired stable-identity quote/forecast capture"
        )
    quote_matchups = {item.matchup_id for item in quotes}
    forecast_matchups = {item.matchup_id for item in forecasts}
    if not forecast_matchups <= quote_matchups:
        raise CaptureError("a forecast capture has no quote in the same retrieval")
    return tuple(quotes), tuple(forecasts), counters


def _atomic_write_report(report: dict[str, object]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if len(encoded.encode("utf-8")) > REPORT_SIZE_LIMIT:
        raise CaptureError("bounded capture report exceeded its size limit")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{REPORT_PATH.name}.", suffix=".tmp", dir=REPORT_PATH.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, REPORT_PATH)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _append_summary(lines: list[str]) -> None:
    summary_path = _text(os.environ.get("GITHUB_STEP_SUMMARY"))
    if not summary_path:
        return
    try:
        with open(summary_path, "a", encoding="utf-8", newline="") as summary:
            summary.write("\n".join(lines) + "\n")
    except OSError as error:
        print(f"Could not write optional Actions summary: {error}")


def _dataset_hash(records: object) -> str:
    return _canonical_hash([record.to_mapping() for record in records])


def validate_generated_capture() -> dict[str, object]:
    required_paths = (
        QUOTE_CSV_PATH,
        QUOTE_JSONL_PATH,
        FORECAST_CSV_PATH,
        FORECAST_JSONL_PATH,
        REPORT_PATH,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise CaptureError(f"market capture outputs are missing: {missing}")
    if REPORT_PATH.stat().st_size > REPORT_SIZE_LIMIT:
        raise CaptureError("capture_report.json is not bounded")
    report = _json_object(_read_bytes(REPORT_PATH), REPORT_PATH)
    supplied_hash = _text(report.get("report_sha256"))
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if supplied_hash != _canonical_hash(unhashed):
        raise CaptureError("capture report hash does not match its contents")
    if report.get("betting_status") != BETTING_STATUS:
        raise CaptureError("capture report does not preserve paper-only betting status")
    if report.get("paper_decisions_created") != 0:
        raise CaptureError("market capture must not create paper or live decisions")
    capture_started = _as_utc(
        report.get("capture_started_at_utc"), "capture_started_at_utc"
    )
    captured_at = _as_utc(report.get("captured_at_utc"), "captured_at_utc")
    if capture_started > captured_at:
        raise CaptureError("capture completion precedes its start")

    quote_store = QuoteSnapshotStore(QUOTE_CSV_PATH, QUOTE_JSONL_PATH)
    forecast_store = ForecastCaptureStore(FORECAST_CSV_PATH, FORECAST_JSONL_PATH)
    quotes = quote_store.read()
    forecasts = forecast_store.read()
    if _dataset_hash(quotes) != report.get("quote_dataset_sha256"):
        raise CaptureError("quote ledger fingerprint differs from capture report")
    if _dataset_hash(forecasts) != report.get("forecast_dataset_sha256"):
        raise CaptureError("forecast ledger fingerprint differs from capture report")
    capture_id = _stable_token(report.get("capture_id"), "capture report capture_id")
    capture_quotes = tuple(item for item in quotes if item.capture_id == capture_id)
    capture_forecasts = tuple(item for item in forecasts if item.capture_id == capture_id)
    if len(capture_quotes) != int(report.get("quote_records_in_capture", -1)):
        raise CaptureError("capture report quote count differs from the quote ledger")
    if len(capture_forecasts) != int(report.get("forecast_records_in_capture", -1)):
        raise CaptureError("capture report forecast count differs from the forecast ledger")
    quote_matchups = {item.matchup_id for item in capture_quotes}
    forecast_matchups = {item.matchup_id for item in capture_forecasts}
    if not forecast_matchups <= quote_matchups:
        raise CaptureError("a stored forecast has no quote in the same capture")
    if not capture_quotes or not capture_forecasts:
        raise CaptureError("capture report identifies an empty market capture")
    if len(quote_matchups) != int(report.get("quote_matchup_count", -1)):
        raise CaptureError("capture report quote matchup count differs from the ledger")
    if len(forecast_matchups) != int(
        report.get("paired_forecast_matchup_count", -1)
    ):
        raise CaptureError("capture report forecast matchup count differs from the ledger")
    if any(item.event_id != report.get("event_id") for item in capture_quotes):
        raise CaptureError("capture quotes disagree with the report event ID")
    if any(item.event_date != report.get("event_date") for item in capture_quotes):
        raise CaptureError("capture quotes disagree with the report event date")
    if any(
        item.timing_precision != report.get("timing_precision")
        or item.event_start_utc is not None
        for item in capture_quotes
    ):
        raise CaptureError("capture quote timing precision disagrees with the report")
    if any(item.event_date != report.get("event_date") for item in capture_quotes):
        raise CaptureError("capture quotes disagree with the report event date")
    if any(item.event_id != report.get("event_id") for item in capture_forecasts):
        raise CaptureError("capture forecasts disagree with the report event ID")
    source_payloads = {item.source_payload_sha256 for item in capture_quotes}
    if source_payloads != {report.get("source_payload_sha256")}:
        raise CaptureError("capture quotes disagree with the report source payload")
    if any(item.source != report.get("source") for item in capture_quotes):
        raise CaptureError("capture quotes disagree with the report source")
    observed_times = {item.observed_at_utc for item in capture_quotes}
    if observed_times != {report.get("captured_at_utc")}:
        raise CaptureError("capture quotes disagree with the report retrieval time")
    forecast_contracts = {
        (
            item.event_id,
            item.event_date,
            item.timing_precision,
            item.event_start_utc,
            item.model_id,
            item.model_version,
            item.model_trained_through,
            item.forecast_issued_at_utc,
            item.source_commit_sha,
        )
        for item in capture_forecasts
    }
    expected_forecast_contract = {
        (
            report.get("event_id"),
            report.get("event_date"),
            report.get("timing_precision"),
            None,
            report.get("model_id"),
            report.get("model_version"),
            report.get("model_trained_through"),
            report.get("forecast_issued_at_utc"),
            report.get("forecast_source_commit_sha"),
        )
    }
    if forecast_contracts != expected_forecast_contract:
        raise CaptureError("capture forecasts disagree with the report model/timing lineage")
    captured_at = _as_utc(report.get("captured_at_utc"), "captured_at_utc")
    if any(
        _as_utc(item.forecast_issued_at_utc, "forecast_issued_at_utc") > captured_at
        for item in capture_forecasts
    ):
        raise CaptureError("a captured forecast was issued after the quote retrieval")
    return report


def capture_market_snapshot() -> dict[str, object]:
    capture_started_at = datetime.now(timezone.utc)
    capture_id = _capture_id(capture_started_at)
    payloads = _publication_payloads()
    input_hashes = _publication_hashes(payloads)
    card = _json_object(payloads[CARD_PATH], CARD_PATH)
    artifact = _json_object(payloads[MODEL_PATH], MODEL_PATH)
    try:
        vegas = pd.read_json(io.BytesIO(payloads[VEGAS_PATH]))
    except (TypeError, ValueError) as error:
        raise CaptureError("vegas_odds.json cannot be loaded as a table") from error

    # This is the only network operation. OddsGetter returns an in-memory table
    # and has no publication-write capability.
    fresh_odds = _retrieve_fresh_odds()
    # The quote became observable only after Selenium returned and parsed the
    # retrieved page.  Never backdate it to the start of a potentially slow
    # network call; the validation below rechecks the strict date-only cutoff
    # using this completion timestamp.
    observed_at = datetime.now(timezone.utc)
    (
        event_day,
        event_id,
        event_url,
        title,
        model_version,
        published,
    ) = _published_matchups(vegas, card, artifact, observed_at)
    source_matches, unmatched_source_rows = _map_source_rows(fresh_odds, published)
    books = _book_columns(fresh_odds)
    source_payload_sha256 = _source_payload_sha256(fresh_odds)

    quote_store = QuoteSnapshotStore(QUOTE_CSV_PATH, QUOTE_JSONL_PATH)
    forecast_store = ForecastCaptureStore(FORECAST_CSV_PATH, FORECAST_JSONL_PATH)
    existing_quotes = quote_store.read()
    # Fail closed on either mirror before constructing any new records.
    forecast_store.read()
    quotes, forecasts, counters = _build_captures(
        source_matches,
        books,
        existing_quotes,
        capture_id=capture_id,
        event_id=event_id,
        event_day=event_day,
        observed_at=observed_at,
        artifact=artifact,
        source_payload_sha256=source_payload_sha256,
    )

    if _publication_payloads() != payloads:
        raise CaptureError("a frozen publication input changed before ledger append")

    # Append forecast metadata first. A filesystem interruption can therefore
    # never leave apparently evaluable quotes whose probability provenance is
    # absent. Each individual mirror replacement is atomic inside the stores.
    forecast_result = forecast_store.append(forecasts)
    quote_result = quote_store.append(quotes)
    final_quotes = quote_store.read()
    final_forecasts = forecast_store.read()
    quote_matchups = len({item.matchup_id for item in quotes})
    paired_forecast_matchups = len({item.matchup_id for item in forecasts})
    published_without_stable_ids = sum(
        matchup.matchup_id is None for matchup in published
    )
    report_body: dict[str, object] = {
        "schema_version": 1,
        "capture_id": capture_id,
        "capture_started_at_utc": capture_started_at.isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z"),
        "captured_at_utc": observed_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "source": SOURCE,
        "source_payload_sha256": source_payload_sha256,
        "event_id": event_id,
        "event_url": event_url,
        "event_title": title,
        "event_date": event_day,
        "timing_precision": "date",
        "model_id": _text(artifact.get("model_id")),
        "model_version": model_version,
        "model_trained_through": _text(artifact.get("data_through")),
        "forecast_issued_at_utc": forecasts[0].forecast_issued_at_utc,
        "forecast_source_commit_sha": forecasts[0].source_commit_sha,
        "publication_commit_sha": _text(os.environ.get("GITHUB_SHA")) or "local",
        "input_sha256": input_hashes,
        "published_matchup_count": len(published),
        "source_matchup_count": len(fresh_odds),
        "source_unmatched_matchup_count": unmatched_source_rows,
        "uniquely_mapped_matchup_count": len(source_matches),
        "published_unmapped_matchup_count": len(published) - len(source_matches),
        "published_matchups_without_stable_ids": published_without_stable_ids,
        "quote_matchup_count": quote_matchups,
        "paired_forecast_matchup_count": paired_forecast_matchups,
        "quote_records_in_capture": len(quotes),
        "forecast_records_in_capture": len(forecasts),
        "quote_records_added": len(quote_result.added_ids),
        "quote_records_duplicate": len(quote_result.duplicate_ids),
        "forecast_records_added": len(forecast_result.added_ids),
        "forecast_records_duplicate": len(forecast_result.duplicate_ids),
        "quote_records_total": quote_result.total_records,
        "forecast_records_total": forecast_result.total_records,
        "quote_dataset_sha256": _dataset_hash(final_quotes),
        "forecast_dataset_sha256": _dataset_hash(final_forecasts),
        **counters,
        "paper_decisions_created": 0,
        "betting_status": BETTING_STATUS,
        "publication_files_unchanged": True,
    }

    current_payloads = _publication_payloads()
    if current_payloads != payloads:
        raise CaptureError("a frozen publication input changed during market capture")
    report_body["report_sha256"] = _canonical_hash(report_body)
    _atomic_write_report(report_body)
    validated = validate_generated_capture()
    _append_summary(
        [
            "## UFC market capture",
            "",
            f"- Event: {title} ({event_day})",
            f"- Capture: `{capture_id}`",
            (
                f"- Quote matchups: {quote_matchups}/{len(published)} captured "
                f"from {len(source_matches)} uniquely mapped source rows"
            ),
            f"- Paired model forecasts: {paired_forecast_matchups}",
            f"- Unrelated source rows skipped: {unmatched_source_rows}",
            f"- Two-sided book quotes: {len(quotes)}",
            (
                "- Paper-evaluable matchups (4+ books): "
                f"{counters['paper_evaluable_matchups']}"
            ),
            f"- Quote ledger: {quote_result.total_records} records",
            f"- Forecast ledger: {forecast_result.total_records} records",
            f"- Betting: `{BETTING_STATUS}`",
            "",
        ]
    )
    return validated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the current bounded report and both ledger mirrors",
    )
    args = parser.parse_args()
    try:
        report = (
            validate_generated_capture()
            if args.validate_only
            else capture_market_snapshot()
        )
    except Exception as error:
        _append_summary(
            [
                "## UFC market capture",
                "",
                f"> Capture failed closed: {type(error).__name__}: {error}",
                "",
                f"- Betting: `{BETTING_STATUS}`",
                "",
            ]
        )
        raise
    print(
        "Validated market capture "
        f"{report['capture_id']}: {report['quote_records_in_capture']} quotes / "
        f"{report['forecast_records_in_capture']} forecasts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
