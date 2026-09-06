"""Capture one frozen-model sportsbook observation and refresh its web view.

This command is intentionally separate from the authoritative UFCStats/model
update.  It treats ``card_info.json``, ``vegas_odds.json``,
``winner_model.json``, and ``bayesian_winner_challenger.json`` as immutable
inputs, maps source display names only when
they identify one published stable-ID matchup, and appends validated quote and
forecast captures to the market-tracker ledgers.  It never creates a wager;
the only website file it writes is a bounded paper-only view reproduced from
those immutable ledgers.
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
from fight_predictor.outcome_publication import (
    outcome_forecasts_usable,
    validate_outcome_forecast_publication,
)
from market_tracker import (
    BETTING_STATUS,
    BAYESIAN_FILTER_MINIMUM_MEAN_EV,
    BAYESIAN_FILTER_MINIMUM_PROBABILITY_POSITIVE_EV,
    BAYESIAN_FILTER_POLICY_VERSION,
    BayesianFilteredDecision,
    BayesianFilteredDecisionStore,
    EARLY_MARKET_CONTRACT,
    EarlyMarketLink,
    EarlyMarketLinkStore,
    EarlyMarketObservation,
    EarlyMarketObservationStore,
    ForecastCapture,
    ForecastCaptureStore,
    MarketDataError,
    PaperDecisionStore,
    QuoteSnapshot,
    QuoteSnapshotStore,
    QuoteSourceMetadata,
    QuoteSourceMetadataStore,
    TotalRoundsQuoteSnapshot,
    TotalRoundsQuoteStore,
    TotalRoundsForecastCapture,
    TotalRoundsForecastStore,
    TotalRoundsPaperDecisionStore,
    TotalRoundsPaperSettlementStore,
    build_current_opportunities,
    build_locked_paper_decisions,
    build_locked_total_round_decisions,
    matchup_id_for,
    validate_current_opportunities,
)
from market_tracker.opportunities import CURRENT_OPPORTUNITIES_SIZE_LIMIT
from odds_getter import OddsApiError, OddsApiResponse, OddsGetter, TheOddsApiClient
from upcoming_bet_board import (
    build_upcoming_bet_board,
    validate_upcoming_bet_board,
    validate_upcoming_forecast_publication,
    write_upcoming_bet_board,
)
from market_tracker.bankroll import archive_upcoming_bet_board


ODDS_API_SOURCE = "the-odds-api.com"
FIGHTODDS_SOURCE = "fightodds.io"
ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = ROOT / "content" / "data" / "external"
MARKET_ROOT = ROOT / "content" / "data" / "market"
CARD_PATH = EXTERNAL_ROOT / "card_info.json"
VEGAS_PATH = EXTERNAL_ROOT / "vegas_odds.json"
MODEL_PATH = EXTERNAL_ROOT / "winner_model.json"
BAYESIAN_MODEL_PATH = EXTERNAL_ROOT / "bayesian_winner_challenger.json"
OUTCOME_FORECAST_PATH = EXTERNAL_ROOT / "outcome_forecasts.json"
ALL_UPCOMING_FORECAST_PATH = EXTERNAL_ROOT / "all_upcoming_forecasts.json"
QUOTE_CSV_PATH = MARKET_ROOT / "quote_snapshots.csv"
QUOTE_JSONL_PATH = MARKET_ROOT / "quote_snapshots.jsonl"
FORECAST_CSV_PATH = MARKET_ROOT / "forecast_captures.csv"
FORECAST_JSONL_PATH = MARKET_ROOT / "forecast_captures.jsonl"
SOURCE_METADATA_CSV_PATH = MARKET_ROOT / "quote_source_metadata.csv"
SOURCE_METADATA_JSONL_PATH = MARKET_ROOT / "quote_source_metadata.jsonl"
TOTAL_ROUNDS_CSV_PATH = MARKET_ROOT / "total_round_quote_snapshots.csv"
TOTAL_ROUNDS_JSONL_PATH = MARKET_ROOT / "total_round_quote_snapshots.jsonl"
TOTAL_ROUNDS_FORECAST_CSV_PATH = MARKET_ROOT / "total_round_forecast_captures.csv"
TOTAL_ROUNDS_FORECAST_JSONL_PATH = MARKET_ROOT / "total_round_forecast_captures.jsonl"
TOTAL_ROUNDS_DECISION_CSV_PATH = MARKET_ROOT / "total_round_paper_decisions.csv"
TOTAL_ROUNDS_DECISION_JSONL_PATH = MARKET_ROOT / "total_round_paper_decisions.jsonl"
TOTAL_ROUNDS_SETTLEMENT_CSV_PATH = MARKET_ROOT / "total_round_paper_settlements.csv"
TOTAL_ROUNDS_SETTLEMENT_JSONL_PATH = MARKET_ROOT / "total_round_paper_settlements.jsonl"
EARLY_MARKET_CSV_PATH = MARKET_ROOT / "early_market_observations.csv"
EARLY_MARKET_JSONL_PATH = MARKET_ROOT / "early_market_observations.jsonl"
EARLY_LINK_CSV_PATH = MARKET_ROOT / "early_market_ufc_links.csv"
EARLY_LINK_JSONL_PATH = MARKET_ROOT / "early_market_ufc_links.jsonl"
DECISION_CSV_PATH = MARKET_ROOT / "paper_decisions.csv"
DECISION_JSONL_PATH = MARKET_ROOT / "paper_decisions.jsonl"
BAYESIAN_FILTER_DECISION_CSV_PATH = (
    MARKET_ROOT / "bayesian_filtered_paper_decisions.csv"
)
BAYESIAN_FILTER_DECISION_JSONL_PATH = (
    MARKET_ROOT / "bayesian_filtered_paper_decisions.jsonl"
)
REPORT_PATH = MARKET_ROOT / "capture_report.json"
CURRENT_OPPORTUNITIES_PATH = MARKET_ROOT / "current_opportunities.json"
UPCOMING_BET_BOARD_PATH = MARKET_ROOT / "upcoming_bet_board.json"
PUBLISHED_BET_ARCHIVE_PATH = MARKET_ROOT / "published_bet_snapshots.json"
REPORT_SIZE_LIMIT = 64 * 1024
SOURCE_RETRY_DELAYS_SECONDS = (15.0, 60.0)
API_RETRY_DELAYS_SECONDS = (5.0, 30.0)
MAX_EARLY_PRICE_STATES_PER_CAPTURE = 10_000


class CaptureError(RuntimeError):
    """Raised when the frozen publication and fresh source cannot be joined."""


class CaptureSkipped(CaptureError):
    """Raised for an expected no-op after the published card has commenced."""


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
    bayesian_model_id: str
    bayesian_status: str
    bayesian_credible_level: float
    bayesian_posterior_mean: float
    bayesian_posterior_median: float
    bayesian_probability_lower: float
    bayesian_probability_upper: float
    bayesian_calibrated_logit_location: float
    bayesian_calibrated_logit_scale: float


@dataclass(frozen=True)
class SourceMatch:
    source_row: dict[str, object]
    published: PublishedMatchup
    source_is_reversed: bool


@dataclass(frozen=True)
class SourceBookColumns:
    book: str
    fighter_column: str
    opponent_column: str
    source_key_column: str | None
    source_update_column: str | None


@dataclass(frozen=True)
class RetrievedOdds:
    source: str
    frame: pd.DataFrame
    source_payload_sha256: str
    request_metadata: dict[str, object]
    total_rounds_frame: pd.DataFrame | None = None


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
    payloads = {
        path: _read_bytes(path)
        for path in (CARD_PATH, VEGAS_PATH, MODEL_PATH, BAYESIAN_MODEL_PATH)
    }
    if OUTCOME_FORECAST_PATH.is_file():
        payloads[OUTCOME_FORECAST_PATH] = _read_bytes(OUTCOME_FORECAST_PATH)
    if ALL_UPCOMING_FORECAST_PATH.is_file():
        payloads[ALL_UPCOMING_FORECAST_PATH] = _read_bytes(
            ALL_UPCOMING_FORECAST_PATH
        )
    return payloads


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


def _skip_if_prior_capture_card_started(
    card: dict[str, object], observed_at: datetime
) -> None:
    """Avoid spending credits or failing after a previously timed card starts."""

    if not REPORT_PATH.exists():
        return
    try:
        report = _json_object(REPORT_PATH.read_bytes(), REPORT_PATH)
    except CaptureError:
        # The strict validator will diagnose a corrupt report. Do not let an
        # unreadable optional shortcut suppress a fresh capture attempt.
        return
    card_event_id = _text(card.get("event_id"))
    card_event_url = _text(card.get("event_url"))
    same_event = bool(card_event_id) and card_event_id == _text(
        report.get("event_id")
    )
    if not same_event and card_event_url:
        same_event = card_event_url == _text(report.get("event_url"))
    if not same_event or report.get("timing_precision") != "timestamp":
        return
    start_text = _text(report.get("event_start_utc"))
    if start_text and observed_at >= _as_utc(start_text, "event_start_utc"):
        raise CaptureSkipped(
            "published card has commenced; retaining the last validated pre-event snapshot"
        )


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
    bayesian_artifact: dict[str, object],
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
        "bayesian model id",
        "bayesian status",
        "bayesian credible level",
        "bayesian posterior mean",
        "bayesian posterior median",
        "bayesian probability lower",
        "bayesian probability upper",
        "bayesian calibrated logit location",
        "bayesian calibrated logit scale",
    }
    _require_columns(vegas, required)
    if vegas.empty:
        raise CaptureError("published vegas_odds contains no upcoming matchups")

    event_day = _event_date(card.get("date"))
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
    bayesian_model_id = _stable_token(
        bayesian_artifact.get("model_id"), "Bayesian model_id"
    )
    if (
        _text(bayesian_artifact.get("base_model_id")) != model_id
        or bayesian_artifact.get("paper_only") is not True
        or bayesian_artifact.get("execution_enabled") is not False
    ):
        raise CaptureError(
            "Bayesian challenger is not bound paper-only to the winner model"
        )

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
        bayesian_status = _text(row.get("bayesian status"))
        row_bayesian_model_id = _text(row.get("bayesian model id"))
        if matchup_id is not None and row_bayesian_model_id != bayesian_model_id:
            raise CaptureError(
                "resolved matchup Bayesian model ID does not match its artifact"
            )
        bayesian_values: list[float] = []
        if matchup_id is not None:
            try:
                bayesian_values = [
                    float(row.get("bayesian credible level")),
                    float(row.get("bayesian posterior mean")),
                    float(row.get("bayesian posterior median")),
                    float(row.get("bayesian probability lower")),
                    float(row.get("bayesian probability upper")),
                    float(row.get("bayesian calibrated logit location")),
                    float(row.get("bayesian calibrated logit scale")),
                ]
            except (TypeError, ValueError) as error:
                raise CaptureError(
                    "resolved matchup Bayesian posterior is not numeric"
                ) from error
            if not all(math.isfinite(value) for value in bayesian_values):
                raise CaptureError(
                    "resolved matchup Bayesian posterior must be finite"
                )
            credible, mean, median, lower, upper, _, scale = bayesian_values
            if not (
                0.0 < credible < 1.0
                and 0.0 < lower <= median <= upper < 1.0
                and 0.0 < mean < 1.0
                and scale >= 0.0
            ):
                raise CaptureError(
                    "resolved matchup Bayesian posterior contract is invalid"
                )
            if not bayesian_status:
                raise CaptureError("resolved matchup Bayesian status is blank")
        else:
            bayesian_values = [math.nan] * 7
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
                bayesian_model_id=row_bayesian_model_id,
                bayesian_status=bayesian_status,
                bayesian_credible_level=bayesian_values[0],
                bayesian_posterior_mean=bayesian_values[1],
                bayesian_posterior_median=bayesian_values[2],
                bayesian_probability_lower=bayesian_values[3],
                bayesian_probability_upper=bayesian_values[4],
                bayesian_calibrated_logit_location=bayesian_values[5],
                bayesian_calibrated_logit_scale=bayesian_values[6],
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
    *,
    event_day: str | None = None,
) -> tuple[tuple[SourceMatch, ...], int]:
    if odds.empty:
        raise CaptureError("market source returned no matchup rows")
    _require_columns(odds, {"fighter name", "opponent name"})
    matches: list[SourceMatch] = []
    used_matchups: set[str] = set()
    unmatched_rows = 0
    for source_row in odds.to_dict("records"):
        source_fighter = _text(source_row.get("fighter name"))
        source_opponent = _text(source_row.get("opponent name"))
        source_start = _text(source_row.get("source commence time"))
        if event_day and source_start:
            start_day = pd.to_datetime(source_start, errors="coerce", utc=True)
            expected_day = pd.to_datetime(event_day, errors="coerce", utc=True)
            if (
                pd.isna(start_day)
                or pd.isna(expected_day)
                or abs((start_day.normalize() - expected_day.normalize()).days) > 1
            ):
                unmatched_rows += 1
                continue
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
        # A source can expose more than one event. Unrelated rows are safe to
        # skip; ambiguous rows are not.  A strong frozen-card coverage check
        # below still rejects a previous/wrong page.
        if not candidates:
            unmatched_rows += 1
            continue
        if len(candidates) > 1:
            raise CaptureError(
                "market source contains an ambiguous matchup: "
                f"{source_fighter!r} vs {source_opponent!r} matched "
                f"{len(candidates)} published rows"
            )
        matchup, reversed_source = candidates[0]
        if matchup.matchup_id in used_matchups:
            raise CaptureError("market source contains duplicate orientations of one matchup")
        used_matchups.add(matchup.matchup_id)
        matches.append(SourceMatch(source_row, matchup, reversed_source))

    minimum_identifying_rows = (
        1 if len(published) == 1 else max(2, math.ceil(len(published) * 0.5))
    )
    if len(matches) < minimum_identifying_rows:
        raise CaptureError(
            "too few uniquely matching market rows to identify the published card: "
            f"{len(matches)}/{len(published)} (required {minimum_identifying_rows})"
        )
    return tuple(matches), unmatched_rows


def _map_total_round_rows(
    odds: pd.DataFrame,
    published: tuple[PublishedMatchup, ...],
    *,
    event_day: str,
) -> tuple[tuple[SourceMatch, ...], int]:
    """Map a flat book/line prop table onto stable published matchup IDs."""

    if odds.empty:
        return (), 0
    _require_columns(
        odds,
        {
            "fighter name",
            "opponent name",
            "source commence time",
            "source event id",
            "book",
            "source book key",
            "source last update",
            "market",
            "period",
            "line",
            "over moneyline",
            "under moneyline",
        },
    )
    matches: list[SourceMatch] = []
    unmatched = 0
    for source_row in odds.to_dict("records"):
        if _text(source_row.get("market")) != "total_rounds" or _text(
            source_row.get("period")
        ) != "full_fight":
            raise CaptureError("prop source contains an unsupported market or period")
        start = pd.to_datetime(
            _text(source_row.get("source commence time")), errors="coerce", utc=True
        )
        expected = pd.to_datetime(event_day, errors="coerce", utc=True)
        if (
            pd.isna(start)
            or pd.isna(expected)
            or abs((start.normalize() - expected.normalize()).days) > 1
        ):
            unmatched += 1
            continue
        source_fighter = _text(source_row.get("fighter name"))
        source_opponent = _text(source_row.get("opponent name"))
        candidates: list[tuple[PublishedMatchup, bool]] = []
        for matchup in published:
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
        if not candidates:
            unmatched += 1
            continue
        if len(candidates) > 1:
            raise CaptureError(
                "totals source contains an ambiguous matchup: "
                f"{source_fighter!r} vs {source_opponent!r}"
            )
        matchup, reversed_source = candidates[0]
        matches.append(SourceMatch(source_row, matchup, reversed_source))
    return tuple(matches), unmatched


def _prior_total_first_seen(
    existing: tuple[TotalRoundsQuoteSnapshot, ...],
    candidate: TotalRoundsQuoteSnapshot,
) -> str | None:
    matching = [
        item.quote_first_seen_at_utc
        for item in existing
        if item.matchup_id == candidate.matchup_id
        and item.source.casefold() == candidate.source.casefold()
        and item.source_book_key.casefold() == candidate.source_book_key.casefold()
        and item.line == candidate.line
        and item.over_moneyline == candidate.over_moneyline
        and item.under_moneyline == candidate.under_moneyline
    ]
    return min(matching) if matching else None


def _build_total_round_captures(
    source_matches: tuple[SourceMatch, ...],
    existing: tuple[TotalRoundsQuoteSnapshot, ...],
    *,
    capture_id: str,
    event_id: str,
    event_day: str,
    observed_at: datetime,
    source: str,
    source_payload_sha256: str,
    timing_precision: str,
    event_start_utc: str | None,
) -> tuple[tuple[TotalRoundsQuoteSnapshot, ...], dict[str, int]]:
    snapshots: list[TotalRoundsQuoteSnapshot] = []
    invalid = 0
    for source_match in source_matches:
        row = source_match.source_row
        matchup = source_match.published
        if (
            matchup.matchup_id is None
            or matchup.fighter_id is None
            or matchup.opponent_id is None
        ):
            raise CaptureError("an unstable identity reached total-round capture")
        try:
            provisional = TotalRoundsQuoteSnapshot.create(
                capture_id=capture_id,
                event_id=event_id,
                fighter_id=matchup.fighter_id,
                opponent_id=matchup.opponent_id,
                fighter_name=matchup.fighter_name,
                opponent_name=matchup.opponent_name,
                matchup_id=matchup.matchup_id,
                fight_id=matchup.fight_id,
                event_date=event_day,
                timing_precision=timing_precision,
                event_start_utc=event_start_utc,
                observed_at_utc=observed_at,
                source=source,
                source_event_id=row.get("source event id"),
                source_book_key=row.get("source book key"),
                source_quote_updated_at_utc=row.get("source last update"),
                source_commence_time_utc=row.get("source commence time"),
                book=row.get("book"),
                line=row.get("line"),
                over_moneyline=row.get("over moneyline"),
                under_moneyline=row.get("under moneyline"),
                source_payload_sha256=source_payload_sha256,
            )
            first_seen = _prior_total_first_seen(existing, provisional)
            snapshot = (
                provisional
                if first_seen is None
                else TotalRoundsQuoteSnapshot.create(
                    capture_id=capture_id,
                    event_id=event_id,
                    fighter_id=matchup.fighter_id,
                    opponent_id=matchup.opponent_id,
                    fighter_name=matchup.fighter_name,
                    opponent_name=matchup.opponent_name,
                    matchup_id=matchup.matchup_id,
                    fight_id=matchup.fight_id,
                    event_date=event_day,
                    timing_precision=timing_precision,
                    event_start_utc=event_start_utc,
                    observed_at_utc=observed_at,
                    quote_first_seen_at_utc=first_seen,
                    source=source,
                    source_event_id=row.get("source event id"),
                    source_book_key=row.get("source book key"),
                    source_quote_updated_at_utc=row.get("source last update"),
                    source_commence_time_utc=row.get("source commence time"),
                    book=row.get("book"),
                    line=row.get("line"),
                    over_moneyline=row.get("over moneyline"),
                    under_moneyline=row.get("under moneyline"),
                    source_payload_sha256=source_payload_sha256,
                )
            )
        except MarketDataError:
            invalid += 1
            continue
        snapshots.append(snapshot)
    return tuple(snapshots), {
        "total_round_source_rows": len(source_matches),
        "total_round_invalid_quote_pairs": invalid,
        "total_round_matchups": len({item.matchup_id for item in snapshots}),
    }


def _build_total_round_forecasts(
    quotes: tuple[TotalRoundsQuoteSnapshot, ...],
    publication: dict[str, object] | None,
) -> tuple[tuple[TotalRoundsForecastCapture, ...], dict[str, int]]:
    counters = {
        "total_round_forecast_lines": 0,
        "total_round_quotes_without_forecast": 0,
    }
    if not quotes or publication is None:
        counters["total_round_quotes_without_forecast"] = len(quotes)
        return (), counters
    validated = validate_outcome_forecast_publication(publication)
    if not outcome_forecasts_usable(validated):
        counters["total_round_quotes_without_forecast"] = len(quotes)
        return (), counters
    if (
        validated.get("event_id") != quotes[0].event_id
        or _event_date(validated.get("event_date")) != quotes[0].event_date
    ):
        # A stale optional candidate artifact must never be joined across cards,
        # but it also must not discard a healthy moneyline/totals observation.
        counters["total_round_quotes_without_forecast"] = len(quotes)
        return (), counters
    by_matchup = {
        item.get("matchup_id"): item
        for item in validated["matchups"]
        if isinstance(item, dict) and item.get("matchup_id")
    }
    unique_lines: dict[tuple[str, float], TotalRoundsQuoteSnapshot] = {}
    for quote in quotes:
        unique_lines.setdefault((quote.matchup_id, quote.line), quote)
    forecasts: list[TotalRoundsForecastCapture] = []
    for (matchup_id, line), quote in sorted(unique_lines.items()):
        item = by_matchup.get(matchup_id)
        probability = (
            item.get("total_round_over_probabilities", {}).get(f"{line:.1f}")
            if isinstance(item, dict)
            else None
        )
        if probability is None:
            counters["total_round_quotes_without_forecast"] += sum(
                candidate.matchup_id == matchup_id and candidate.line == line
                for candidate in quotes
            )
            continue
        forecasts.append(
            TotalRoundsForecastCapture.create(
                capture_id=quote.capture_id,
                event_id=quote.event_id,
                fighter_id=quote.fighter_id,
                opponent_id=quote.opponent_id,
                fighter_name=quote.fighter_name,
                opponent_name=quote.opponent_name,
                matchup_id=quote.matchup_id,
                fight_id=quote.fight_id,
                event_date=quote.event_date,
                timing_precision=quote.timing_precision,
                event_start_utc=quote.event_start_utc,
                forecast_issued_at_utc=validated["forecast_issued_at_utc"],
                scheduled_rounds=item["scheduled_rounds"],
                schedule_basis=item["schedule_basis"],
                line=line,
                over_probability=probability,
                model_id=validated["model_id"],
                model_version=validated["model_version"],
                model_trained_through=validated["model_trained_through"],
                source_commit_sha=validated["source_commit_sha"],
                source_publication_sha256=validated["publication_sha256"],
            )
        )
    counters["total_round_forecast_lines"] = len(forecasts)
    return tuple(forecasts), counters


def _source_payload_sha256(
    odds: pd.DataFrame, *, source: str = FIGHTODDS_SOURCE
) -> str:
    """Fingerprint the complete parsed retrieval, not one selected book row."""

    try:
        payload = json.loads(
            odds.to_json(orient="split", date_format="iso", date_unit="us")
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise CaptureError("market table cannot be canonically fingerprinted") from error
    return _canonical_hash({"source": source, "parsed_table": payload})


def _api_payload_sha256(payload: object) -> str:
    return _canonical_hash({"source": ODDS_API_SOURCE, "raw_response": payload})


def _capture_timing(
    source_matches: tuple[SourceMatch, ...],
    *,
    event_day: str,
    observed_at: datetime,
) -> tuple[str, str | None, float | None]:
    """Use a conservative card start when the source supplies timestamps."""

    starts = [
        _text(item.source_row.get("source commence time"))
        for item in source_matches
    ]
    if starts and all(starts):
        parsed = [
            _as_utc(value, "source commence time")
            for value in starts
        ]
        card_start = min(parsed)
        event_date = date.fromisoformat(event_day)
        if abs((card_start.date() - event_date).days) > 1:
            raise CaptureError(
                "source commence times are inconsistent with the published card date"
            )
        if observed_at >= card_start:
            raise CaptureError("market capture occurred at or after card commencement")
        return (
            "timestamp",
            card_start.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            (card_start - observed_at).total_seconds(),
        )
    if observed_at.date() >= date.fromisoformat(event_day):
        raise CaptureError(
            "market capture requires an observation UTC date strictly before "
            "a date-only UFC event"
        )
    return "date", None, None


def _retrieve_fresh_odds() -> pd.DataFrame:
    """Retry the optional FightOdds browser fallback with a fresh driver."""

    failures: list[str] = []
    for attempt in range(len(SOURCE_RETRY_DELAYS_SECONDS) + 1):
        try:
            odds = OddsGetter().make_fightodds_df()
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


def _retrieve_market_odds() -> RetrievedOdds:
    """Fetch the configured source without exposing credentials in diagnostics."""

    configured = _text(os.environ.get("MARKET_ODDS_SOURCE", "the-odds-api")).casefold()
    if configured in {"the-odds-api", "the-odds-api.com", "odds-api"}:
        api_key = os.environ.get("THE_ODDS_API_KEY", "")
        regions = ",".join(
            part.strip().casefold()
            for part in _text(
                os.environ.get("ODDS_API_REGIONS", "us,us2")
            ).split(",")
            if part.strip()
        )
        failures: list[str] = []
        for attempt in range(len(API_RETRY_DELAYS_SECONDS) + 1):
            try:
                client = TheOddsApiClient()
                totals_status = "captured"
                try:
                    response: OddsApiResponse = client.fetch(
                        api_key, regions=regions, include_total_rounds=True
                    )
                except OddsApiError as totals_error:
                    if "THE_ODDS_API_KEY" in str(totals_error) or "quota" in str(
                        totals_error
                    ):
                        raise
                    # Totals are optional enrichment. A provider coverage or
                    # schema problem must not discard the established h2h
                    # capture, so retry once with the legacy market contract.
                    print(
                        "The Odds API totals market is unavailable; "
                        f"falling back to h2h only ({totals_error})"
                    )
                    response = client.fetch(
                        api_key, regions=regions, include_total_rounds=False
                    )
                    totals_status = "unavailable_fallback_h2h"
                requested_markets = (
                    "h2h,totals"
                    if response.total_rounds_frame is not None
                    else "h2h"
                )
                return RetrievedOdds(
                    source=ODDS_API_SOURCE,
                    frame=response.frame,
                    source_payload_sha256=_api_payload_sha256(response.payload),
                    request_metadata={
                        "sport": "mma_mixed_martial_arts",
                        "market": requested_markets,
                        "totals_status": totals_status,
                        "regions": regions,
                        "odds_format": "american",
                        **response.quota_mapping(),
                    },
                    total_rounds_frame=response.total_rounds_frame,
                )
            except OddsApiError as error:
                failures.append(str(error))
                # Credential and quota errors cannot be healed by retrying.
                if "THE_ODDS_API_KEY" in str(error) or "quota" in str(error):
                    break
                if attempt >= len(API_RETRY_DELAYS_SECONDS):
                    break
                delay = API_RETRY_DELAYS_SECONDS[attempt]
                print(
                    f"The Odds API attempt {attempt + 1} failed; retrying in "
                    f"{delay:g}s ({error})"
                )
                time.sleep(delay)
        raise CaptureError(
            "The Odds API retrieval failed: " + " | ".join(failures)
        )
    if configured in {"fightodds", "fightodds.io"}:
        frame = _retrieve_fresh_odds()
        return RetrievedOdds(
            source=FIGHTODDS_SOURCE,
            frame=frame,
            source_payload_sha256=_source_payload_sha256(
                frame, source=FIGHTODDS_SOURCE
            ),
            request_metadata={
                "fallback": True,
                "transport": "selenium_headless_browser",
            },
        )
    raise CaptureError(
        "MARKET_ODDS_SOURCE must be 'the-odds-api' or the optional 'fightodds' fallback"
    )


def _book_columns(odds: pd.DataFrame) -> tuple[SourceBookColumns, ...]:
    fighter: dict[str, tuple[str, str]] = {}
    opponent: dict[str, tuple[str, str]] = {}
    for column in odds.columns:
        text = str(column)
        if text.startswith("fighter ") and text != "fighter name":
            book = text[len("fighter ") :].strip()
            key = book.casefold()
            if not book or key in fighter:
                raise CaptureError("market source contains duplicate fighter book columns")
            fighter[key] = (book, text)
        elif text.startswith("opponent ") and text != "opponent name":
            book = text[len("opponent ") :].strip()
            key = book.casefold()
            if not book or key in opponent:
                raise CaptureError("market source contains duplicate opponent book columns")
            opponent[key] = (book, text)
    if set(fighter) != set(opponent) or not fighter:
        raise CaptureError("market source fighter/opponent book columns are incomplete")
    columns = {str(column).casefold(): str(column) for column in odds.columns}
    return tuple(
        SourceBookColumns(
            book=fighter[key][0],
            fighter_column=fighter[key][1],
            opponent_column=opponent[key][1],
            source_key_column=columns.get(
                f"source {fighter[key][0]} key".casefold()
            ),
            source_update_column=columns.get(
                f"source {fighter[key][0]} last update".casefold()
            ),
        )
        for key in sorted(fighter)
    )


def _build_early_market_observations(
    odds: pd.DataFrame,
    total_rounds: pd.DataFrame | None,
    book_columns: tuple[SourceBookColumns, ...],
    existing: tuple[EarlyMarketObservation, ...],
    *,
    capture_id: str,
    observed_at: datetime,
    source: str,
    source_payload_sha256: str,
    published_event_day: str,
) -> tuple[tuple[EarlyMarketObservation, ...], dict[str, int]]:
    """Preserve every distinct pre-fight price state from the existing call.

    The provider labels these only as MMA.  Promotion is deliberately left
    unknown until a separate official-UFC link can be established.
    """

    counters = {
        "early_source_matchups_seen": 0,
        "early_source_matchups_beyond_published_card": 0,
        "early_h2h_price_states_seen": 0,
        "early_total_round_price_states_seen": 0,
        "early_incomplete_price_pairs_skipped": 0,
        "early_invalid_price_pairs_skipped": 0,
        "early_commenced_source_rows_skipped": 0,
    }
    if source != ODDS_API_SOURCE:
        return (), counters

    existing_by_id = {item.observation_id: item for item in existing}
    candidates: dict[str, EarlyMarketObservation] = {}
    future_event_ids: set[str] = set()
    beyond_card_event_ids: set[str] = set()
    expected_day = pd.to_datetime(published_event_day, errors="coerce", utc=True)

    def add(candidate: EarlyMarketObservation) -> None:
        prior = candidates.get(candidate.observation_id)
        if prior is not None and prior != candidate:
            raise CaptureError("one source price state produced conflicting records")
        candidates[candidate.observation_id] = existing_by_id.get(
            candidate.observation_id, candidate
        )
        if len(candidates) > MAX_EARLY_PRICE_STATES_PER_CAPTURE:
            raise CaptureError(
                "The Odds API response exceeds the bounded early-price ledger limit"
            )

    for row in odds.to_dict("records"):
        source_event_id = _text(row.get("source event id"))
        commence_text = _text(row.get("source commence time"))
        commence = pd.to_datetime(commence_text, errors="coerce", utc=True)
        if not source_event_id or pd.isna(commence):
            counters["early_invalid_price_pairs_skipped"] += len(book_columns)
            continue
        if commence.to_pydatetime() <= observed_at:
            counters["early_commenced_source_rows_skipped"] += 1
            continue
        future_event_ids.add(source_event_id)
        if not pd.isna(expected_day) and commence.normalize() > (
            expected_day.normalize() + pd.Timedelta(days=1)
        ):
            beyond_card_event_ids.add(source_event_id)
        fighter_name = _text(row.get("fighter name"))
        opponent_name = _text(row.get("opponent name"))
        for columns in book_columns:
            fighter_price = _text(row.get(columns.fighter_column))
            opponent_price = _text(row.get(columns.opponent_column))
            if not fighter_price and not opponent_price:
                continue
            source_book_key = (
                _text(row.get(columns.source_key_column))
                if columns.source_key_column
                else ""
            )
            source_update = (
                _text(row.get(columns.source_update_column))
                if columns.source_update_column
                else ""
            )
            if (
                not fighter_price
                or not opponent_price
                or not source_book_key
                or not source_update
            ):
                counters["early_incomplete_price_pairs_skipped"] += 1
                continue
            try:
                observation = EarlyMarketObservation.create(
                    first_capture_id=capture_id,
                    first_observed_at_utc=observed_at,
                    source=source,
                    source_payload_sha256=source_payload_sha256,
                    source_event_id=source_event_id,
                    source_commence_time_utc=commence_text,
                    source_fighter_name=fighter_name,
                    source_opponent_name=opponent_name,
                    book=columns.book,
                    source_book_key=source_book_key,
                    source_quote_updated_at_utc=source_update,
                    market="h2h",
                    outcome_a=fighter_name,
                    outcome_b=opponent_name,
                    outcome_a_moneyline=fighter_price,
                    outcome_b_moneyline=opponent_price,
                )
            except MarketDataError:
                counters["early_invalid_price_pairs_skipped"] += 1
                continue
            add(observation)
            counters["early_h2h_price_states_seen"] += 1

    if total_rounds is not None:
        for row in total_rounds.to_dict("records"):
            source_event_id = _text(row.get("source event id"))
            commence_text = _text(row.get("source commence time"))
            commence = pd.to_datetime(commence_text, errors="coerce", utc=True)
            if not source_event_id or pd.isna(commence):
                counters["early_invalid_price_pairs_skipped"] += 1
                continue
            if commence.to_pydatetime() <= observed_at:
                counters["early_commenced_source_rows_skipped"] += 1
                continue
            future_event_ids.add(source_event_id)
            if not pd.isna(expected_day) and commence.normalize() > (
                expected_day.normalize() + pd.Timedelta(days=1)
            ):
                beyond_card_event_ids.add(source_event_id)
            required = (
                "fighter name",
                "opponent name",
                "book",
                "source book key",
                "source last update",
                "line",
                "over moneyline",
                "under moneyline",
            )
            if any(not _text(row.get(field)) for field in required):
                counters["early_incomplete_price_pairs_skipped"] += 1
                continue
            try:
                observation = EarlyMarketObservation.create(
                    first_capture_id=capture_id,
                    first_observed_at_utc=observed_at,
                    source=source,
                    source_payload_sha256=source_payload_sha256,
                    source_event_id=source_event_id,
                    source_commence_time_utc=commence_text,
                    source_fighter_name=row.get("fighter name"),
                    source_opponent_name=row.get("opponent name"),
                    book=row.get("book"),
                    source_book_key=row.get("source book key"),
                    source_quote_updated_at_utc=row.get("source last update"),
                    market="total_rounds",
                    line=row.get("line"),
                    outcome_a="Over",
                    outcome_b="Under",
                    outcome_a_moneyline=row.get("over moneyline"),
                    outcome_b_moneyline=row.get("under moneyline"),
                )
            except MarketDataError:
                counters["early_invalid_price_pairs_skipped"] += 1
                continue
            add(observation)
            counters["early_total_round_price_states_seen"] += 1

    counters["early_source_matchups_seen"] = len(future_event_ids)
    counters["early_source_matchups_beyond_published_card"] = len(
        beyond_card_event_ids
    )
    return tuple(candidates.values()), counters


def _build_early_market_links(
    source_matches: tuple[SourceMatch, ...],
    existing: tuple[EarlyMarketLink, ...],
    *,
    capture_id: str,
    observed_at: datetime,
    source: str,
    event_id: str,
) -> tuple[EarlyMarketLink, ...]:
    if source != ODDS_API_SOURCE:
        return ()
    existing_by_id = {item.link_id: item for item in existing}
    candidates: dict[str, EarlyMarketLink] = {}
    for source_match in source_matches:
        row = source_match.source_row
        published = source_match.published
        if (
            not _text(row.get("source event id"))
            or not _text(row.get("source commence time"))
            or published.matchup_id is None
            or published.fighter_id is None
            or published.opponent_id is None
        ):
            continue
        source_fighter_id = (
            published.opponent_id
            if source_match.source_is_reversed
            else published.fighter_id
        )
        source_opponent_id = (
            published.fighter_id
            if source_match.source_is_reversed
            else published.opponent_id
        )
        try:
            link = EarlyMarketLink.create(
                first_linked_at_utc=observed_at,
                first_capture_id=capture_id,
                source=source,
                source_event_id=row.get("source event id"),
                source_commence_time_utc=row.get("source commence time"),
                source_fighter_name=row.get("fighter name"),
                source_opponent_name=row.get("opponent name"),
                ufc_event_id=event_id,
                matchup_id=published.matchup_id,
                source_fighter_ufcstats_id=source_fighter_id,
                source_opponent_ufcstats_id=source_opponent_id,
                source_is_reversed=source_match.source_is_reversed,
            )
        except MarketDataError as error:
            raise CaptureError(f"official early-price link is invalid: {error}") from error
        candidates[link.link_id] = existing_by_id.get(link.link_id, link)
    return tuple(candidates.values())


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
    book_columns: tuple[SourceBookColumns, ...],
    existing_quotes: tuple[QuoteSnapshot, ...],
    *,
    capture_id: str,
    event_id: str,
    event_day: str,
    observed_at: datetime,
    artifact: dict[str, object],
    source_payload_sha256: str,
    source: str,
    timing_precision: str,
    event_start_utc: str | None,
) -> tuple[
    tuple[QuoteSnapshot, ...],
    tuple[ForecastCapture, ...],
    tuple[QuoteSourceMetadata, ...],
    dict[str, int],
]:
    quotes: list[QuoteSnapshot] = []
    forecasts: list[ForecastCapture] = []
    source_metadata: list[QuoteSourceMetadata] = []
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
        for columns in book_columns:
            book = columns.book
            fighter_column = columns.fighter_column
            opponent_column = columns.opponent_column
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
                    timing_precision=timing_precision,
                    event_start_utc=event_start_utc,
                    observed_at_utc=observed_at,
                    source=source,
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
                        timing_precision=timing_precision,
                        event_start_utc=event_start_utc,
                        observed_at_utc=observed_at,
                        quote_first_seen_at_utc=first_seen,
                        source=source,
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
            if source == ODDS_API_SOURCE:
                source_book_key = (
                    source_row.get(columns.source_key_column)
                    if columns.source_key_column
                    else None
                )
                source_updated = (
                    source_row.get(columns.source_update_column)
                    if columns.source_update_column
                    else None
                )
                source_event_id = source_row.get("source event id")
                source_commence = source_row.get("source commence time")
                if not all(
                    _text(value)
                    for value in (
                        source_book_key,
                        source_updated,
                        source_event_id,
                        source_commence,
                    )
                ):
                    raise CaptureError(
                        "The Odds API quote is missing stable source timing metadata"
                    )
                source_metadata.append(
                    QuoteSourceMetadata.create(
                        snapshot,
                        source_book_key=source_book_key,
                        source_event_id=source_event_id,
                        source_quote_updated_at_utc=source_updated,
                        source_commence_time_utc=source_commence,
                    )
                )

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
                timing_precision=timing_precision,
                event_start_utc=event_start_utc,
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
            "market source contained no valid paired stable-identity quote/forecast capture"
        )
    quote_matchups = {item.matchup_id for item in quotes}
    forecast_matchups = {item.matchup_id for item in forecasts}
    if not forecast_matchups <= quote_matchups:
        raise CaptureError("a forecast capture has no quote in the same retrieval")
    if source == ODDS_API_SOURCE and len(source_metadata) != len(quotes):
        raise CaptureError("every API quote requires source timing metadata")
    return tuple(quotes), tuple(forecasts), tuple(source_metadata), counters


def _build_bayesian_filtered_decisions(
    base_decisions: tuple,
    published: tuple[PublishedMatchup, ...],
    *,
    source_vegas_sha256: str,
    bayesian_artifact_sha256: str,
) -> tuple[BayesianFilteredDecision, ...]:
    """Apply the Bayesian veto only to newly frozen T-24 base decisions."""

    published_by_matchup = {
        item.matchup_id: item for item in published if item.matchup_id is not None
    }
    filtered: list[BayesianFilteredDecision] = []
    for base in base_decisions:
        matchup = published_by_matchup.get(base.matchup_id)
        if matchup is None:
            raise CaptureError(
                "a new base paper decision has no frozen Bayesian matchup"
            )
        filtered.append(
            BayesianFilteredDecision.create(
                base,
                source_vegas_sha256=source_vegas_sha256,
                bayesian_artifact_sha256=bayesian_artifact_sha256,
                bayesian_model_id=matchup.bayesian_model_id,
                bayesian_status=matchup.bayesian_status,
                credible_level=matchup.bayesian_credible_level,
                fighter_posterior_mean=matchup.bayesian_posterior_mean,
                fighter_posterior_median=matchup.bayesian_posterior_median,
                fighter_probability_lower=matchup.bayesian_probability_lower,
                fighter_probability_upper=matchup.bayesian_probability_upper,
                fighter_calibrated_logit_location=(
                    matchup.bayesian_calibrated_logit_location
                ),
                calibrated_logit_scale=matchup.bayesian_calibrated_logit_scale,
                minimum_mean_expected_return=BAYESIAN_FILTER_MINIMUM_MEAN_EV,
                minimum_probability_positive_expected_return=(
                    BAYESIAN_FILTER_MINIMUM_PROBABILITY_POSITIVE_EV
                ),
            )
        )
    return tuple(filtered)


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


def _atomic_write_current_opportunities(publication: dict[str, object]) -> None:
    CURRENT_OPPORTUNITIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        publication, separators=(",", ":"), sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    encoded_size = len(encoded.encode("utf-8"))
    if encoded_size > CURRENT_OPPORTUNITIES_SIZE_LIMIT:
        raise CaptureError(
            f"current opportunity publication exceeded its size limit: "
            f"{encoded_size:,} bytes > {CURRENT_OPPORTUNITIES_SIZE_LIMIT:,} bytes"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{CURRENT_OPPORTUNITIES_PATH.name}.",
        suffix=".tmp",
        dir=CURRENT_OPPORTUNITIES_PATH.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, CURRENT_OPPORTUNITIES_PATH)
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


def _validate_source_request(report: dict[str, object]) -> None:
    metadata = report.get("source_request")
    if not isinstance(metadata, dict):
        raise CaptureError("capture report source_request must be an object")
    for key, value in metadata.items():
        normalized_key = _text(key).casefold().replace("-", "_")
        if normalized_key in {"apikey", "api_key", "authorization"}:
            raise CaptureError("capture report must not contain a source credential")
        if isinstance(value, (dict, list, tuple, set)):
            raise CaptureError("capture report source_request values must be scalar")

    if report.get("source") != ODDS_API_SOURCE:
        return
    expected = {
        "sport": "mma_mixed_martial_arts",
        "odds_format": "american",
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise CaptureError(
                f"capture report has unexpected The Odds API {key} metadata"
            )
    if metadata.get("market") not in {"h2h", "h2h,totals"}:
        raise CaptureError(
            "capture report has unexpected The Odds API market metadata"
        )
    if "total_round_dataset_sha256" in report and metadata.get("market") != "h2h,totals":
        raise CaptureError(
            "a total-round capture must identify the h2h,totals API request"
        )
    regions = _text(metadata.get("regions"))
    if not regions or regions != ",".join(
        part.strip().casefold() for part in regions.split(",") if part.strip()
    ):
        raise CaptureError("capture report has invalid The Odds API regions metadata")
    for key in ("requests_remaining", "requests_used", "request_cost"):
        value = metadata.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise CaptureError(f"capture report {key} must be a nonnegative integer")


def validate_generated_capture() -> dict[str, object]:
    core_paths = (
        QUOTE_CSV_PATH,
        QUOTE_JSONL_PATH,
        FORECAST_CSV_PATH,
        FORECAST_JSONL_PATH,
        REPORT_PATH,
    )
    missing = [str(path) for path in core_paths if not path.is_file()]
    if missing:
        raise CaptureError(f"market capture outputs are missing: {missing}")
    if REPORT_PATH.stat().st_size > REPORT_SIZE_LIMIT:
        raise CaptureError("capture_report.json is not bounded")
    report = _json_object(_read_bytes(REPORT_PATH), REPORT_PATH)
    enhanced_fields = {
        "source_metadata_dataset_sha256",
        "paper_decision_dataset_sha256",
    }
    present_enhanced_fields = enhanced_fields & set(report)
    if present_enhanced_fields and present_enhanced_fields != enhanced_fields:
        raise CaptureError("capture report has a partial enhanced market contract")
    enhanced_contract = enhanced_fields <= set(report)
    opportunity_contract = "opportunity_publication_sha256" in report
    upcoming_board_fields = {
        "upcoming_bet_board_sha256",
        "upcoming_bet_board_qualified_bets",
        "upcoming_bet_board_announced_events",
        "upcoming_bet_board_execution_enabled",
    }
    present_upcoming_board_fields = upcoming_board_fields & set(report)
    if (
        present_upcoming_board_fields
        and present_upcoming_board_fields != upcoming_board_fields
    ):
        raise CaptureError("capture report has a partial upcoming-bet-board contract")
    upcoming_board_contract = upcoming_board_fields <= set(report)
    if upcoming_board_contract and report.get(
        "upcoming_bet_board_execution_enabled"
    ) is not False:
        raise CaptureError("upcoming bet board must keep execution disabled")
    early_market_fields = {
        "early_market_contract",
        "early_market_source_scope",
        "early_source_matchups_seen",
        "early_source_matchups_beyond_published_card",
        "early_h2h_price_states_seen",
        "early_total_round_price_states_seen",
        "early_incomplete_price_pairs_skipped",
        "early_invalid_price_pairs_skipped",
        "early_commenced_source_rows_skipped",
        "early_price_states_in_response",
        "early_price_states_added",
        "early_price_states_duplicate",
        "early_price_states_total",
        "early_ufc_links_in_response",
        "early_ufc_links_added",
        "early_ufc_links_duplicate",
        "early_ufc_links_total",
        "early_market_dataset_sha256",
        "early_market_link_dataset_sha256",
        "early_market_paper_only",
        "early_market_execution_enabled",
    }
    present_early_market_fields = early_market_fields & set(report)
    if (
        present_early_market_fields
        and present_early_market_fields != early_market_fields
    ):
        raise CaptureError("capture report has a partial early-market contract")
    early_market_contract = early_market_fields <= set(report)
    if early_market_contract and (
        report.get("early_market_contract") != EARLY_MARKET_CONTRACT
        or report.get("early_market_source_scope")
        != "all_mma_promotion_unknown_until_official_ufc_link"
        or report.get("early_market_paper_only") is not True
        or report.get("early_market_execution_enabled") is not False
    ):
        raise CaptureError("capture report early-market policy is invalid")
    bayesian_filter_fields = {
        "bayesian_model_id",
        "bayesian_filtered_decision_dataset_sha256",
        "bayesian_filtered_policy",
        "bayesian_filtered_decisions_created",
        "bayesian_filtered_decisions_added",
        "bayesian_filtered_decisions_total",
    }
    present_bayesian_filter_fields = bayesian_filter_fields & set(report)
    if (
        present_bayesian_filter_fields
        and present_bayesian_filter_fields != bayesian_filter_fields
    ):
        raise CaptureError("capture report has a partial Bayesian filter contract")
    bayesian_filter_contract = bayesian_filter_fields <= set(report)
    if bayesian_filter_contract and not enhanced_contract:
        raise CaptureError("Bayesian filtered decisions require the base decision contract")
    if bayesian_filter_contract and report.get("bayesian_filtered_policy") != {
        "policy_version": BAYESIAN_FILTER_POLICY_VERSION,
        "minimum_mean_expected_return": BAYESIAN_FILTER_MINIMUM_MEAN_EV,
        "minimum_probability_positive_expected_return": (
            BAYESIAN_FILTER_MINIMUM_PROBABILITY_POSITIVE_EV
        ),
        "paper_only": True,
        "execution_enabled": False,
    }:
        raise CaptureError("capture report Bayesian filter policy is invalid")
    total_round_fields = {
        "total_round_source_rows",
        "total_round_unmatched_source_rows",
        "total_round_invalid_quote_pairs",
        "total_round_matchups",
        "total_round_records_in_capture",
        "total_round_records_added",
        "total_round_records_duplicate",
        "total_round_records_total",
        "total_round_dataset_sha256",
    }
    total_round_forecast_fields = {
        "total_round_forecast_lines",
        "total_round_quotes_without_forecast",
        "total_round_forecast_records_in_capture",
        "total_round_forecast_records_added",
        "total_round_forecast_records_duplicate",
        "total_round_forecast_records_total",
        "total_round_forecast_dataset_sha256",
    }
    total_round_decision_fields = {
        "total_round_paper_decision_policy",
        "total_round_paper_decisions_created",
        "total_round_paper_decisions_added",
        "total_round_paper_decisions_total",
        "total_round_paper_decision_dataset_sha256",
    }
    present_total_round_fields = total_round_fields & set(report)
    if present_total_round_fields and present_total_round_fields != total_round_fields:
        raise CaptureError("capture report has a partial total-round contract")
    total_round_contract = total_round_fields <= set(report)
    present_total_round_forecast_fields = total_round_forecast_fields & set(report)
    if (
        present_total_round_forecast_fields
        and present_total_round_forecast_fields != total_round_forecast_fields
    ):
        raise CaptureError("capture report has a partial total-round forecast contract")
    total_round_forecast_contract = total_round_forecast_fields <= set(report)
    if total_round_forecast_contract and not total_round_contract:
        raise CaptureError("total-round forecasts require the quote contract")
    present_total_round_decision_fields = total_round_decision_fields & set(report)
    if (
        present_total_round_decision_fields
        and present_total_round_decision_fields != total_round_decision_fields
    ):
        raise CaptureError("capture report has a partial total-round decision contract")
    total_round_decision_contract = total_round_decision_fields <= set(report)
    if total_round_decision_contract and not total_round_forecast_contract:
        raise CaptureError("total-round decisions require the forecast contract")
    if enhanced_contract:
        extended_paths = (
            SOURCE_METADATA_CSV_PATH,
            SOURCE_METADATA_JSONL_PATH,
            DECISION_CSV_PATH,
            DECISION_JSONL_PATH,
        )
        missing = [str(path) for path in extended_paths if not path.is_file()]
        if missing:
            raise CaptureError(f"enhanced market capture outputs are missing: {missing}")
    if early_market_contract:
        early_paths = (
            EARLY_MARKET_CSV_PATH,
            EARLY_MARKET_JSONL_PATH,
            EARLY_LINK_CSV_PATH,
            EARLY_LINK_JSONL_PATH,
        )
        missing = [str(path) for path in early_paths if not path.is_file()]
        if missing:
            raise CaptureError(f"early-market capture outputs are missing: {missing}")
    if opportunity_contract:
        if not CURRENT_OPPORTUNITIES_PATH.is_file():
            raise CaptureError("current opportunity publication is missing")
        if (
            CURRENT_OPPORTUNITIES_PATH.stat().st_size
            > CURRENT_OPPORTUNITIES_SIZE_LIMIT
        ):
            raise CaptureError("current opportunity publication is not bounded")
    if bayesian_filter_contract:
        filtered_paths = (
            BAYESIAN_FILTER_DECISION_CSV_PATH,
            BAYESIAN_FILTER_DECISION_JSONL_PATH,
        )
        missing = [str(path) for path in filtered_paths if not path.is_file()]
        if missing:
            raise CaptureError(
                f"Bayesian filtered decision outputs are missing: {missing}"
            )
    if total_round_contract:
        total_paths = (
            TOTAL_ROUNDS_CSV_PATH,
            TOTAL_ROUNDS_JSONL_PATH,
        )
        missing = [str(path) for path in total_paths if not path.is_file()]
        if missing:
            raise CaptureError(f"total-round capture outputs are missing: {missing}")
    if total_round_forecast_contract:
        total_forecast_paths = (
            TOTAL_ROUNDS_FORECAST_CSV_PATH,
            TOTAL_ROUNDS_FORECAST_JSONL_PATH,
        )
        missing = [str(path) for path in total_forecast_paths if not path.is_file()]
        if missing:
            raise CaptureError(f"total-round forecast outputs are missing: {missing}")
    if total_round_decision_contract:
        total_decision_paths = (
            TOTAL_ROUNDS_DECISION_CSV_PATH,
            TOTAL_ROUNDS_DECISION_JSONL_PATH,
        )
        missing = [str(path) for path in total_decision_paths if not path.is_file()]
        if missing:
            raise CaptureError(f"total-round decision outputs are missing: {missing}")
    supplied_hash = _text(report.get("report_sha256"))
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if supplied_hash != _canonical_hash(unhashed):
        raise CaptureError("capture report hash does not match its contents")
    if report.get("betting_status") != BETTING_STATUS:
        raise CaptureError("capture report does not preserve paper-only betting status")
    try:
        paper_decisions_created = int(report.get("paper_decisions_created", 0))
    except (TypeError, ValueError) as error:
        raise CaptureError("paper_decisions_created must be an integer") from error
    if paper_decisions_created < 0:
        raise CaptureError("paper_decisions_created cannot be negative")
    _validate_source_request(report)
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
    source_metadata = (
        QuoteSourceMetadataStore(
            SOURCE_METADATA_CSV_PATH, SOURCE_METADATA_JSONL_PATH
        ).read()
        if enhanced_contract
        else ()
    )
    decisions = (
        PaperDecisionStore(DECISION_CSV_PATH, DECISION_JSONL_PATH).read()
        if enhanced_contract
        else ()
    )
    early_market = (
        EarlyMarketObservationStore(
            EARLY_MARKET_CSV_PATH, EARLY_MARKET_JSONL_PATH
        ).read()
        if early_market_contract
        else ()
    )
    early_links = (
        EarlyMarketLinkStore(EARLY_LINK_CSV_PATH, EARLY_LINK_JSONL_PATH).read()
        if early_market_contract
        else ()
    )


    bayesian_filtered_decisions = (
        BayesianFilteredDecisionStore(
            BAYESIAN_FILTER_DECISION_CSV_PATH,
            BAYESIAN_FILTER_DECISION_JSONL_PATH,
        ).read()
        if bayesian_filter_contract
        else ()
    )
    total_rounds = (
        TotalRoundsQuoteStore(
            TOTAL_ROUNDS_CSV_PATH, TOTAL_ROUNDS_JSONL_PATH
        ).read()
        if total_round_forecast_contract
        else ()
    )
    total_round_forecasts = (
        TotalRoundsForecastStore(
            TOTAL_ROUNDS_FORECAST_CSV_PATH,
            TOTAL_ROUNDS_FORECAST_JSONL_PATH,
        ).read()
        if total_round_contract
        else ()
    )
    total_round_decisions = (
        TotalRoundsPaperDecisionStore(
            TOTAL_ROUNDS_DECISION_CSV_PATH,
            TOTAL_ROUNDS_DECISION_JSONL_PATH,
        ).read()
        if total_round_decision_contract
        else ()
    )
    if _dataset_hash(quotes) != report.get("quote_dataset_sha256"):
        raise CaptureError("quote ledger fingerprint differs from capture report")
    if _dataset_hash(forecasts) != report.get("forecast_dataset_sha256"):
        raise CaptureError("forecast ledger fingerprint differs from capture report")
    if enhanced_contract and _dataset_hash(source_metadata) != report.get("source_metadata_dataset_sha256"):
        raise CaptureError("source metadata fingerprint differs from capture report")
    if enhanced_contract and _dataset_hash(decisions) != report.get("paper_decision_dataset_sha256"):
        raise CaptureError("paper decision fingerprint differs from capture report")
    if early_market_contract:
        if _dataset_hash(early_market) != report.get("early_market_dataset_sha256"):
            raise CaptureError("early-market ledger fingerprint differs from report")
        if _dataset_hash(early_links) != report.get(
            "early_market_link_dataset_sha256"
        ):
            raise CaptureError("early-market link fingerprint differs from report")
        integer_fields = early_market_fields - {
            "early_market_contract",
            "early_market_source_scope",
            "early_market_dataset_sha256",
            "early_market_link_dataset_sha256",
            "early_market_paper_only",
            "early_market_execution_enabled",
        }
        if any(
            type(report.get(field)) is not int or int(report[field]) < 0
            for field in integer_fields
        ):
            raise CaptureError("early-market report counts must be nonnegative integers")
        if int(report["early_price_states_total"]) != len(early_market):
            raise CaptureError("early-market report total differs from ledger")
        if int(report["early_ufc_links_total"]) != len(early_links):
            raise CaptureError("early-market link total differs from ledger")
        if int(report["early_price_states_in_response"]) != (
            int(report["early_price_states_added"])
            + int(report["early_price_states_duplicate"])
        ):
            raise CaptureError("early-market response accounting is inconsistent")
        if int(report["early_ufc_links_in_response"]) != (
            int(report["early_ufc_links_added"])
            + int(report["early_ufc_links_duplicate"])
        ):
            raise CaptureError("early-market link accounting is inconsistent")
        if any(
            not item.paper_only or item.execution_enabled for item in early_market
        ) or any(not item.paper_only or item.execution_enabled for item in early_links):
            raise CaptureError("an early-market record is not research-only")
    if bayesian_filter_contract and _dataset_hash(
        bayesian_filtered_decisions
    ) != report.get("bayesian_filtered_decision_dataset_sha256"):
        raise CaptureError(
            "Bayesian filtered decision fingerprint differs from capture report"
        )
    if total_round_contract and _dataset_hash(total_rounds) != report.get(
        "total_round_dataset_sha256"
    ):
        raise CaptureError("total-round ledger fingerprint differs from capture report")
    if total_round_forecast_contract and _dataset_hash(total_round_forecasts) != report.get(
        "total_round_forecast_dataset_sha256"
    ):
        raise CaptureError("total-round forecast fingerprint differs from capture report")
    if total_round_decision_contract and _dataset_hash(total_round_decisions) != report.get(
        "total_round_paper_decision_dataset_sha256"
    ):
        raise CaptureError("total-round decision fingerprint differs from capture report")
    capture_id = _stable_token(report.get("capture_id"), "capture report capture_id")
    capture_quotes = tuple(item for item in quotes if item.capture_id == capture_id)
    capture_forecasts = tuple(item for item in forecasts if item.capture_id == capture_id)
    capture_metadata = tuple(
        item for item in source_metadata if item.capture_id == capture_id
    )
    capture_decisions = tuple(
        item for item in decisions if item.capture_id == capture_id
    )
    capture_bayesian_filtered_decisions = tuple(
        item
        for item in bayesian_filtered_decisions
        if item.capture_id == capture_id
    )
    capture_total_rounds = tuple(
        item for item in total_rounds if item.capture_id == capture_id
    )
    capture_total_round_forecasts = tuple(
        item for item in total_round_forecasts if item.capture_id == capture_id
    )
    capture_total_round_decisions = tuple(
        item for item in total_round_decisions if item.capture_id == capture_id
    )
    if len(capture_quotes) != int(report.get("quote_records_in_capture", -1)):
        raise CaptureError("capture report quote count differs from the quote ledger")
    if len(capture_forecasts) != int(report.get("forecast_records_in_capture", -1)):
        raise CaptureError("capture report forecast count differs from the forecast ledger")
    if enhanced_contract and len(capture_metadata) != int(
        report.get("source_metadata_records_in_capture", -1)
    ):
        raise CaptureError("capture report metadata count differs from the metadata ledger")
    if len(capture_decisions) != paper_decisions_created:
        raise CaptureError("capture report paper count differs from the decision ledger")
    if bayesian_filter_contract:
        if len(bayesian_filtered_decisions) != int(
            report.get("bayesian_filtered_decisions_total", -1)
        ):
            raise CaptureError(
                "capture report Bayesian filtered total differs from the ledger"
            )
        if len(capture_bayesian_filtered_decisions) != int(
            report.get("bayesian_filtered_decisions_created", -1)
        ):
            raise CaptureError(
                "capture report Bayesian filtered count differs from the ledger"
            )
        base_ids = {item.decision_id for item in capture_decisions}
        if {
            item.base_decision_id for item in capture_bayesian_filtered_decisions
        } != base_ids:
            raise CaptureError(
                "capture Bayesian filter does not exactly cover new base decisions"
            )
        input_hashes = report.get("input_sha256")
        if not isinstance(input_hashes, dict):
            raise CaptureError("capture report input_sha256 must be an object")
        if any(
            item.bayesian_model_id != report.get("bayesian_model_id")
            or item.source_vegas_sha256 != input_hashes.get(VEGAS_PATH.name)
            or item.bayesian_artifact_sha256
            != input_hashes.get(BAYESIAN_MODEL_PATH.name)
            for item in capture_bayesian_filtered_decisions
        ):
            raise CaptureError(
                "capture Bayesian filter disagrees with frozen publication lineage"
            )
    if total_round_contract:
        if len(total_rounds) != int(report.get("total_round_records_total", -1)):
            raise CaptureError("capture report total-round total differs from ledger")
        if len(capture_total_rounds) != int(
            report.get("total_round_records_in_capture", -1)
        ):
            raise CaptureError("capture report total-round count differs from ledger")
        if len({item.matchup_id for item in capture_total_rounds}) != int(
            report.get("total_round_matchups", -1)
        ):
            raise CaptureError("capture report total-round matchup count differs")
        if any(
            item.event_id != report.get("event_id")
            or item.event_date != report.get("event_date")
            or item.timing_precision != report.get("timing_precision")
            or item.event_start_utc != report.get("event_start_utc")
            or item.observed_at_utc != report.get("captured_at_utc")
            or item.source_payload_sha256 != report.get("source_payload_sha256")
            or item.source != report.get("source")
            for item in capture_total_rounds
        ):
            raise CaptureError("total-round quotes disagree with capture lineage")
    if total_round_forecast_contract:
        if len(total_round_forecasts) != int(
            report.get("total_round_forecast_records_total", -1)
        ):
            raise CaptureError("capture report total forecast total differs from ledger")
        if len(capture_total_round_forecasts) != int(
            report.get("total_round_forecast_records_in_capture", -1)
        ):
            raise CaptureError("capture report total forecast count differs from ledger")
        quote_lines = {
            (item.matchup_id, item.line) for item in capture_total_rounds
        }
        if not {
            (item.matchup_id, item.line)
            for item in capture_total_round_forecasts
        } <= quote_lines:
            raise CaptureError("a total forecast has no quote in the same capture")
    if total_round_decision_contract:
        if len(total_round_decisions) != int(
            report.get("total_round_paper_decisions_total", -1)
        ):
            raise CaptureError("capture report total decision total differs from ledger")
        if len(capture_total_round_decisions) != int(
            report.get("total_round_paper_decisions_created", -1)
        ):
            raise CaptureError("capture report total decision count differs from ledger")
        forecast_lines = {
            (item.matchup_id, item.line) for item in capture_total_round_forecasts
        }
        if not {
            (item.matchup_id, item.line) for item in capture_total_round_decisions
        } <= forecast_lines:
            raise CaptureError("a total decision has no forecast in the same capture")
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
        or item.event_start_utc != report.get("event_start_utc")
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
    if enhanced_contract and report.get("source") == ODDS_API_SOURCE:
        quote_ids = {item.quote_id for item in capture_quotes}
        if {item.quote_id for item in capture_metadata} != quote_ids:
            raise CaptureError("API quote metadata does not exactly cover the capture")
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
            report.get("event_start_utc"),
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
    if opportunity_contract:
        opportunities = _json_object(
            _read_bytes(CURRENT_OPPORTUNITIES_PATH), CURRENT_OPPORTUNITIES_PATH
        )
        validated_opportunities = validate_current_opportunities(
            opportunities,
            quotes,
            forecasts,
            source_metadata,
            decisions,
            capture_id=capture_id,
            total_round_quotes=total_rounds,
            total_round_forecasts=total_round_forecasts,
            total_round_decisions=total_round_decisions,
            bayesian_filtered_decisions=bayesian_filtered_decisions,
            method_price_status=(
                opportunities.get("prop_markets", {})
                .get("method_of_victory", {})
                .get("price_status", "unavailable_from_configured_provider")
            ),
        )
        if (
            validated_opportunities.get("publication_sha256")
            != report.get("opportunity_publication_sha256")
        ):
            raise CaptureError(
                "opportunity publication fingerprint differs from capture report"
            )
    if upcoming_board_contract:
        board = validate_upcoming_bet_board(
            _json_object(
                _read_bytes(UPCOMING_BET_BOARD_PATH),
                UPCOMING_BET_BOARD_PATH,
            )
        )
        if board.get("publication_sha256") != report.get(
            "upcoming_bet_board_sha256"
        ):
            raise CaptureError("upcoming bet board fingerprint differs from report")
        if board.get("qualified_bet_count") != report.get(
            "upcoming_bet_board_qualified_bets"
        ) or board.get("announced_event_count") != report.get(
            "upcoming_bet_board_announced_events"
        ):
            raise CaptureError("upcoming bet board counts differ from report")
    return report


def capture_market_snapshot() -> dict[str, object]:
    capture_started_at = datetime.now(timezone.utc)
    capture_id = _capture_id(capture_started_at)
    payloads = _publication_payloads()
    input_hashes = _publication_hashes(payloads)
    card = _json_object(payloads[CARD_PATH], CARD_PATH)
    artifact = _json_object(payloads[MODEL_PATH], MODEL_PATH)
    bayesian_artifact = _json_object(
        payloads[BAYESIAN_MODEL_PATH], BAYESIAN_MODEL_PATH
    )
    outcome_publication = (
        _json_object(payloads[OUTCOME_FORECAST_PATH], OUTCOME_FORECAST_PATH)
        if OUTCOME_FORECAST_PATH in payloads
        else None
    )
    if outcome_publication is not None:
        validate_outcome_forecast_publication(outcome_publication)
    all_upcoming_forecasts = (
        validate_upcoming_forecast_publication(
            _json_object(
                payloads[ALL_UPCOMING_FORECAST_PATH],
                ALL_UPCOMING_FORECAST_PATH,
            )
        )
        if ALL_UPCOMING_FORECAST_PATH in payloads
        else None
    )
    _skip_if_prior_capture_card_started(card, capture_started_at)
    try:
        vegas = pd.read_json(io.BytesIO(payloads[VEGAS_PATH]))
    except (TypeError, ValueError) as error:
        raise CaptureError("vegas_odds.json cannot be loaded as a table") from error

    # This is the only network operation. The source adapter returns an
    # in-memory table and has no publication-write capability.
    retrieved_odds = _retrieve_market_odds()
    fresh_odds = retrieved_odds.frame
    # The quote became observable only after the source response returned and
    # was parsed. Never backdate it to the start of a potentially slow
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
    ) = _published_matchups(
        vegas, card, artifact, bayesian_artifact, observed_at
    )
    source_matches, unmatched_source_rows = _map_source_rows(
        fresh_odds, published, event_day=event_day
    )
    books = _book_columns(fresh_odds)
    source_payload_sha256 = retrieved_odds.source_payload_sha256
    timing_precision, event_start_utc, lead_time_seconds = _capture_timing(
        source_matches, event_day=event_day, observed_at=observed_at
    )

    quote_store = QuoteSnapshotStore(QUOTE_CSV_PATH, QUOTE_JSONL_PATH)
    forecast_store = ForecastCaptureStore(FORECAST_CSV_PATH, FORECAST_JSONL_PATH)
    metadata_store = QuoteSourceMetadataStore(
        SOURCE_METADATA_CSV_PATH, SOURCE_METADATA_JSONL_PATH
    )
    early_market_store = EarlyMarketObservationStore(
        EARLY_MARKET_CSV_PATH, EARLY_MARKET_JSONL_PATH
    )
    early_link_store = EarlyMarketLinkStore(
        EARLY_LINK_CSV_PATH, EARLY_LINK_JSONL_PATH
    )
    decision_store = PaperDecisionStore(DECISION_CSV_PATH, DECISION_JSONL_PATH)
    bayesian_filter_store = BayesianFilteredDecisionStore(
        BAYESIAN_FILTER_DECISION_CSV_PATH,
        BAYESIAN_FILTER_DECISION_JSONL_PATH,
    )
    existing_quotes = quote_store.read()
    # Fail closed on either mirror before constructing any new records.
    forecast_store.read()
    existing_metadata = metadata_store.read()
    existing_early_market = early_market_store.read()
    existing_early_links = early_link_store.read()
    existing_decisions = decision_store.read()
    bayesian_filter_store.read()
    quotes, forecasts, source_metadata, counters = _build_captures(
        source_matches,
        books,
        existing_quotes,
        capture_id=capture_id,
        event_id=event_id,
        event_day=event_day,
        observed_at=observed_at,
        artifact=artifact,
        source_payload_sha256=source_payload_sha256,
        source=retrieved_odds.source,
        timing_precision=timing_precision,
        event_start_utc=event_start_utc,
    )
    early_market_observations, early_market_counters = (
        _build_early_market_observations(
            fresh_odds,
            retrieved_odds.total_rounds_frame,
            books,
            existing_early_market,
            capture_id=capture_id,
            observed_at=observed_at,
            source=retrieved_odds.source,
            source_payload_sha256=source_payload_sha256,
            published_event_day=event_day,
        )
    )
    early_market_links = _build_early_market_links(
        source_matches,
        existing_early_links,
        capture_id=capture_id,
        observed_at=observed_at,
        source=retrieved_odds.source,
        event_id=event_id,
    )

    total_round_store: TotalRoundsQuoteStore | None = None
    total_round_forecast_store: TotalRoundsForecastStore | None = None
    total_round_decision_store: TotalRoundsPaperDecisionStore | None = None
    total_round_quotes: tuple[TotalRoundsQuoteSnapshot, ...] = ()
    total_round_forecasts: tuple[TotalRoundsForecastCapture, ...] = ()
    total_round_decision_build = None
    total_round_counters = {
        "total_round_source_rows": 0,
        "total_round_unmatched_source_rows": 0,
        "total_round_invalid_quote_pairs": 0,
        "total_round_matchups": 0,
        "total_round_forecast_lines": 0,
        "total_round_quotes_without_forecast": 0,
    }
    if retrieved_odds.total_rounds_frame is not None:
        total_round_store = TotalRoundsQuoteStore(
            TOTAL_ROUNDS_CSV_PATH, TOTAL_ROUNDS_JSONL_PATH
        )
        total_round_forecast_store = TotalRoundsForecastStore(
            TOTAL_ROUNDS_FORECAST_CSV_PATH,
            TOTAL_ROUNDS_FORECAST_JSONL_PATH,
        )
        total_round_decision_store = TotalRoundsPaperDecisionStore(
            TOTAL_ROUNDS_DECISION_CSV_PATH,
            TOTAL_ROUNDS_DECISION_JSONL_PATH,
        )
        total_round_settlement_store = TotalRoundsPaperSettlementStore(
            TOTAL_ROUNDS_SETTLEMENT_CSV_PATH,
            TOTAL_ROUNDS_SETTLEMENT_JSONL_PATH,
        )
        total_round_forecast_store.read()
        existing_total_round_decisions = total_round_decision_store.read()
        existing_total_round_settlements = total_round_settlement_store.read()
        existing_total_rounds = total_round_store.read()
        total_source_matches, total_unmatched = _map_total_round_rows(
            retrieved_odds.total_rounds_frame,
            published,
            event_day=event_day,
        )
        total_round_quotes, built_total_counters = _build_total_round_captures(
            total_source_matches,
            existing_total_rounds,
            capture_id=capture_id,
            event_id=event_id,
            event_day=event_day,
            observed_at=observed_at,
            source=retrieved_odds.source,
            source_payload_sha256=source_payload_sha256,
            timing_precision=timing_precision,
            event_start_utc=event_start_utc,
        )
        total_round_counters.update(built_total_counters)
        total_round_counters["total_round_unmatched_source_rows"] = total_unmatched
        total_round_forecasts, forecast_counters = _build_total_round_forecasts(
            total_round_quotes, outcome_publication
        )
        total_round_counters.update(forecast_counters)
        total_round_decision_build = build_locked_total_round_decisions(
            total_round_quotes,
            total_round_forecasts,
            existing_total_round_decisions,
            existing_total_round_settlements,
        )

    paper_build = build_locked_paper_decisions(
        quotes, forecasts, source_metadata, existing_decisions
    )
    bayesian_filtered_decisions = _build_bayesian_filtered_decisions(
        paper_build.decisions,
        published,
        source_vegas_sha256=input_hashes[VEGAS_PATH.name],
        bayesian_artifact_sha256=input_hashes[BAYESIAN_MODEL_PATH.name],
    )

    if _publication_payloads() != payloads:
        raise CaptureError("a frozen publication input changed before ledger append")

    # Append forecast metadata first. A filesystem interruption can therefore
    # never leave apparently evaluable quotes whose probability provenance is
    # absent. Each individual mirror replacement is atomic inside the stores.
    forecast_result = forecast_store.append(forecasts)
    quote_result = quote_store.append(quotes)
    metadata_result = metadata_store.append(source_metadata)
    early_market_result = early_market_store.append(early_market_observations)
    early_link_result = early_link_store.append(early_market_links)
    decision_result = decision_store.append(paper_build.decisions)
    bayesian_filter_result = bayesian_filter_store.append(
        bayesian_filtered_decisions
    )
    total_round_forecast_result = (
        total_round_forecast_store.append(total_round_forecasts)
        if total_round_forecast_store is not None
        else None
    )
    total_round_result = (
        total_round_store.append(total_round_quotes)
        if total_round_store is not None
        else None
    )
    total_round_decision_result = (
        total_round_decision_store.append(total_round_decision_build.decisions)
        if total_round_decision_store is not None
        and total_round_decision_build is not None
        else None
    )
    final_quotes = quote_store.read()
    final_forecasts = forecast_store.read()
    final_metadata = metadata_store.read()
    final_early_market = early_market_store.read()
    final_early_links = early_link_store.read()
    final_decisions = decision_store.read()
    final_bayesian_filtered_decisions = bayesian_filter_store.read()
    final_total_rounds = (
        total_round_store.read() if total_round_store is not None else ()
    )
    final_total_round_forecasts = (
        total_round_forecast_store.read()
        if total_round_forecast_store is not None
        else ()
    )
    final_total_round_decisions = (
        total_round_decision_store.read()
        if total_round_decision_store is not None
        else ()
    )
    current_opportunities = build_current_opportunities(
        final_quotes,
        final_forecasts,
        final_metadata,
        final_decisions,
        capture_id=capture_id,
        total_round_quotes=final_total_rounds,
        total_round_forecasts=final_total_round_forecasts,
        total_round_decisions=final_total_round_decisions,
        bayesian_filtered_decisions=final_bayesian_filtered_decisions,
        method_price_status=(
            str(outcome_publication.get("method_price_status"))
            if outcome_publication is not None
            else "unavailable_from_configured_provider"
        ),
    )
    _atomic_write_current_opportunities(current_opportunities)
    upcoming_bet_board = None
    if all_upcoming_forecasts is not None:
        upcoming_bet_board = build_upcoming_bet_board(
            all_upcoming_forecasts,
            early_market_observations,
            observed_at_utc=observed_at,
            source=retrieved_odds.source,
            current_opportunities=current_opportunities,
        )
        write_upcoming_bet_board(upcoming_bet_board, UPCOMING_BET_BOARD_PATH)
        archive_upcoming_bet_board(
            upcoming_bet_board, PUBLISHED_BET_ARCHIVE_PATH
        )
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
        "source": retrieved_odds.source,
        "source_payload_sha256": source_payload_sha256,
        "source_request": retrieved_odds.request_metadata,
        "event_id": event_id,
        "event_url": event_url,
        "event_title": title,
        "event_date": event_day,
        "timing_precision": timing_precision,
        "event_start_utc": event_start_utc,
        "capture_lead_time_seconds": lead_time_seconds,
        "model_id": _text(artifact.get("model_id")),
        "model_version": model_version,
        "model_trained_through": _text(artifact.get("data_through")),
        "bayesian_model_id": _text(bayesian_artifact.get("model_id")),
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
        "source_metadata_records_in_capture": len(source_metadata),
        "source_metadata_records_added": len(metadata_result.added_ids),
        "source_metadata_records_duplicate": len(metadata_result.duplicate_ids),
        "source_metadata_records_total": metadata_result.total_records,
        "early_market_contract": EARLY_MARKET_CONTRACT,
        "early_market_source_scope": "all_mma_promotion_unknown_until_official_ufc_link",
        **early_market_counters,
        "early_price_states_in_response": len(early_market_observations),
        "early_price_states_added": len(early_market_result.added_ids),
        "early_price_states_duplicate": len(early_market_result.duplicate_ids),
        "early_price_states_total": early_market_result.total_records,
        "early_ufc_links_in_response": len(early_market_links),
        "early_ufc_links_added": len(early_link_result.added_ids),
        "early_ufc_links_duplicate": len(early_link_result.duplicate_ids),
        "early_ufc_links_total": early_link_result.total_records,
        "early_market_dataset_sha256": _dataset_hash(final_early_market),
        "early_market_link_dataset_sha256": _dataset_hash(final_early_links),
        "early_market_paper_only": True,
        "early_market_execution_enabled": False,
        "quote_dataset_sha256": _dataset_hash(final_quotes),
        "forecast_dataset_sha256": _dataset_hash(final_forecasts),
        "source_metadata_dataset_sha256": _dataset_hash(final_metadata),
        "paper_decision_dataset_sha256": _dataset_hash(final_decisions),
        "bayesian_filtered_decision_dataset_sha256": _dataset_hash(
            final_bayesian_filtered_decisions
        ),
        "opportunity_publication_sha256": current_opportunities[
            "publication_sha256"
        ],
        "paper_decision_policy": paper_build.to_mapping(),
        **counters,
        "paper_decisions_created": len(paper_build.decisions),
        "paper_decisions_added": len(decision_result.added_ids),
        "paper_decisions_total": decision_result.total_records,
        "bayesian_filtered_policy": {
            "policy_version": BAYESIAN_FILTER_POLICY_VERSION,
            "minimum_mean_expected_return": BAYESIAN_FILTER_MINIMUM_MEAN_EV,
            "minimum_probability_positive_expected_return": (
                BAYESIAN_FILTER_MINIMUM_PROBABILITY_POSITIVE_EV
            ),
            "paper_only": True,
            "execution_enabled": False,
        },
        "bayesian_filtered_decisions_created": len(
            bayesian_filtered_decisions
        ),
        "bayesian_filtered_decisions_added": len(
            bayesian_filter_result.added_ids
        ),
        "bayesian_filtered_decisions_total": (
            bayesian_filter_result.total_records
        ),
        "betting_status": BETTING_STATUS,
        "publication_files_unchanged": True,
    }
    if upcoming_bet_board is not None:
        report_body.update(
            {
                "upcoming_bet_board_sha256": upcoming_bet_board[
                    "publication_sha256"
                ],
                "upcoming_bet_board_qualified_bets": upcoming_bet_board[
                    "qualified_bet_count"
                ],
                "upcoming_bet_board_announced_events": upcoming_bet_board[
                    "announced_event_count"
                ],
                "upcoming_bet_board_execution_enabled": False,
            }
        )
    if total_round_result is not None:
        if (
            total_round_forecast_result is None
            or total_round_decision_result is None
            or total_round_decision_build is None
        ):
            raise CaptureError(
                "total-round quotes require forecast and decision ledger results"
            )
        report_body.update(
            {
                **total_round_counters,
                "total_round_records_in_capture": len(total_round_quotes),
                "total_round_records_added": len(total_round_result.added_ids),
                "total_round_records_duplicate": len(
                    total_round_result.duplicate_ids
                ),
                "total_round_records_total": total_round_result.total_records,
                "total_round_dataset_sha256": _dataset_hash(final_total_rounds),
                "total_round_forecast_records_in_capture": len(
                    total_round_forecasts
                ),
                "total_round_forecast_records_added": len(
                    total_round_forecast_result.added_ids
                ),
                "total_round_forecast_records_duplicate": len(
                    total_round_forecast_result.duplicate_ids
                ),
                "total_round_forecast_records_total": (
                    total_round_forecast_result.total_records
                ),
                "total_round_forecast_dataset_sha256": _dataset_hash(
                    final_total_round_forecasts
                ),
                "total_round_paper_decision_policy": (
                    total_round_decision_build.to_mapping()
                ),
                "total_round_paper_decisions_created": len(
                    total_round_decision_build.decisions
                ),
                "total_round_paper_decisions_added": len(
                    total_round_decision_result.added_ids
                ),
                "total_round_paper_decisions_total": (
                    total_round_decision_result.total_records
                ),
                "total_round_paper_decision_dataset_sha256": _dataset_hash(
                    final_total_round_decisions
                ),
            }
        )

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
            f"- Source: `{retrieved_odds.source}`",
            (
                f"- Quote matchups: {quote_matchups}/{len(published)} captured "
                f"from {len(source_matches)} uniquely mapped source rows"
            ),
            f"- Paired model forecasts: {paired_forecast_matchups}",
            f"- Unrelated source rows skipped: {unmatched_source_rows}",
            f"- Two-sided book quotes: {len(quotes)}",
            f"- Full-fight total-round quotes: {len(total_round_quotes)}",
            (
                "- T-24 total-round decisions: "
                f"{len(total_round_decision_build.decisions) if total_round_decision_build else 0} "
                f"(`{BETTING_STATUS}`)"
            ),
            (
                "- Paper-evaluable matchups (4+ books): "
                f"{counters['paper_evaluable_matchups']}"
            ),
            f"- Quote ledger: {quote_result.total_records} records",
            f"- Forecast ledger: {forecast_result.total_records} records",
            f"- Source metadata ledger: {metadata_result.total_records} records",
            (
                "- Distinct early MMA price states: "
                f"{early_market_result.total_records} total "
                f"({len(early_market_result.added_ids)} new; "
                f"{early_market_counters['early_source_matchups_beyond_published_card']} "
                "farther-out matchups seen)"
            ),
            f"- Official UFC early-price links: {early_link_result.total_records}",
            (
                f"- T-24 paper decisions: {len(paper_build.decisions)} "
                f"(`{BETTING_STATUS}`)"
            ),
            (
                "- Bayesian-filtered T-24 decisions: "
                f"{len(bayesian_filtered_decisions)} (`{BETTING_STATUS}`)"
            ),
            "- Website opportunity view: refreshed from this capture",
            (
                "- API credits remaining: "
                f"{retrieved_odds.request_metadata.get('requests_remaining')}"
                if retrieved_odds.source == ODDS_API_SOURCE
                else "- API credits remaining: not applicable"
            ),
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
    except CaptureSkipped as skipped:
        _append_summary(
            [
                "## UFC market capture",
                "",
                f"> Expected no-op: {skipped}",
                "",
                f"- Betting: `{BETTING_STATUS}`",
                "",
            ]
        )
        print(f"Market capture skipped: {skipped}")
        return 0
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
