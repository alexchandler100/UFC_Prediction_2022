"""Published paper-bet archive and browser-safe bankroll research data.

This module never places a wager.  It preserves the qualified rows that were
actually published and produces a compact, hash-validated data file used by
the website's hypothetical bankroll calculator.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping

from ._common import canonical_hash, implied_probability, utc_datetime
from .bayesian_kelly import (
    BayesianKellyCalibrator,
    unavailable_assessment,
    validate_bayesian_kelly_assessment,
)


ARCHIVE_SCHEMA_VERSION = 1
PERFORMANCE_SCHEMA_VERSION = 1
ARCHIVE_POLICY_VERSION = "published-qualified-paper-bets-v1"
PERFORMANCE_POLICY_VERSION = "published-paper-bankroll-replay-v1"
RESEARCH_STAKING_STRATEGIES = (
    "robust_bayesian_kelly",
    "half_kelly_model_blend",
    "half_kelly_sim_blend",
    "half_kelly_model_sim_blend",
)
SUPPORT_SOURCES = ("model", "simulation")


def profit_multiple(moneyline: object) -> float:
    """Profit returned for one unit risked at American odds."""

    line = int(moneyline)
    if line == 0 or abs(line) < 100:
        raise ValueError("American odds must be at least 100 in magnitude")
    return line / 100.0 if line > 0 else 100.0 / abs(line)


def kelly_fraction(probability: object, moneyline: object) -> float:
    """Full Kelly fraction, clamped to the valid no-borrowing range [0, 1]."""

    chance = float(probability)
    if not math.isfinite(chance) or not 0.0 < chance < 1.0:
        raise ValueError("Kelly probability must be strictly between zero and one")
    profit = profit_multiple(moneyline)
    fraction = (profit * chance - (1.0 - chance)) / profit
    return min(1.0, max(0.0, fraction))


def bet_support_key(
    *, event_id: object, fighter_id: object, opponent_id: object,
    category: object, side: object, selection: object,
) -> tuple[str, ...]:
    """Stable key for a saved prediction that supports one published bet."""

    fighter = str(fighter_id or "")
    opponent = str(opponent_id or "")
    event = str(event_id or "")
    if not event or not fighter or not opponent or fighter == opponent:
        raise ValueError("bet support identity is incomplete")
    pair = tuple(sorted((fighter, opponent)))
    market = str(category or "")
    bet_side = str(side or "").casefold()
    if market == "Moneyline":
        if bet_side not in {"fighter", "opponent"}:
            raise ValueError("moneyline support side is invalid")
        selected = fighter if bet_side == "fighter" else opponent
        return (event, *pair, "moneyline", selected)
    if market == "Total rounds":
        match = re.search(
            r"(Over|Under)\s+([0-9]+(?:\.[0-9]+)?)",
            str(selection or ""),
            re.I,
        )
        if match is None:
            raise ValueError("total-round support selection is invalid")
        return (
            event,
            *pair,
            "total_rounds",
            f"{float(match.group(2)):g}",
            match.group(1).casefold(),
        )
    raise ValueError("unsupported bet category")


def _support_fields(source: str, value: Mapping[str, object] | None) -> dict[str, object]:
    probability_key = f"{source}_support_probability"
    source_key = f"{source}_support_source"
    issued_key = f"{source}_support_issued_at_utc"
    if value is None:
        return {probability_key: None, source_key: None, issued_key: None}
    chance = float(value.get("probability"))
    if not math.isfinite(chance) or not 0.0 < chance < 1.0:
        raise ValueError(f"{source} support probability is invalid")
    issued = str(value.get("issued_at_utc") or "")
    utc_datetime(issued, f"{source}_support_issued_at_utc")
    label = str(value.get("source") or "").strip()
    if not label:
        raise ValueError(f"{source} support source is missing")
    return {probability_key: chance, source_key: label, issued_key: issued}


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False,
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    ) as output:
        output.write(encoded)
        temporary = Path(output.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_bet(bet: object, *, board_hash: str) -> dict[str, object]:
    if not isinstance(bet, dict):
        raise ValueError("published bet must be an object")
    supplied = str(bet.get("bet_id") or "")
    body = dict(bet)
    body.pop("bet_id", None)
    if not supplied or supplied != canonical_hash(body):
        raise ValueError("published bet ID does not match its contents")
    probability = float(bet.get("estimated_win_probability"))
    expected_return = float(bet.get("estimated_expected_return"))
    threshold = float(bet.get("minimum_expected_return"))
    line = int(bet.get("offered_moneyline"))
    if (
        bet.get("threshold_met") is not True
        or bet.get("paper_only") is not True
        or bet.get("execution_enabled") is not False
        or not 0.0 < probability < 1.0
        or expected_return < threshold
    ):
        raise ValueError("archive contains a non-qualified paper bet")
    profit_multiple(line)
    utc_datetime(bet.get("observed_at_utc"), "observed_at_utc")
    record = dict(bet)
    record["board_publication_sha256"] = board_hash
    record["snapshot_id"] = canonical_hash(record)
    return record


def validate_published_bet_archive(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("published bet archive must be an object")
    supplied = value.get("archive_sha256")
    body = dict(value)
    body.pop("archive_sha256", None)
    if supplied != canonical_hash(body):
        raise ValueError("published bet archive hash is invalid")
    if (
        value.get("schema_version") != ARCHIVE_SCHEMA_VERSION
        or value.get("policy_version") != ARCHIVE_POLICY_VERSION
        or value.get("paper_only") is not True
        or value.get("execution_enabled") is not False
    ):
        raise ValueError("published bet archive contract is invalid")
    snapshots = value.get("snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("published bet archive snapshots must be a list")
    if value.get("snapshot_count") != len(snapshots):
        raise ValueError("published bet archive count is inconsistent")
    ids: list[str] = []
    ordering: list[tuple[str, str]] = []
    for record in snapshots:
        if not isinstance(record, dict):
            raise ValueError("published bet snapshot must be an object")
        supplied_id = str(record.get("snapshot_id") or "")
        unhashed = dict(record)
        unhashed.pop("snapshot_id", None)
        if supplied_id != canonical_hash(unhashed):
            raise ValueError("published bet snapshot hash is invalid")
        _validated_bet(
            {key: item for key, item in record.items()
             if key not in {"snapshot_id", "board_publication_sha256"}},
            board_hash=str(record.get("board_publication_sha256") or ""),
        )
        ids.append(supplied_id)
        ordering.append((str(record["observed_at_utc"]), supplied_id))
    if len(ids) != len(set(ids)) or ordering != sorted(ordering):
        raise ValueError("published bet archive contains duplicates or is unsorted")
    return value


def archive_upcoming_bet_board(
    board: Mapping[str, object], path: Path
) -> dict[str, object]:
    """Append every qualified row in one published board, idempotently."""

    board_hash = str(board.get("publication_sha256") or "")
    if not board_hash:
        raise ValueError("upcoming board lacks its publication hash")
    existing: list[dict[str, object]] = []
    if path.exists():
        prior = validate_published_bet_archive(
            json.loads(path.read_text(encoding="utf-8"))
        )
        existing = [dict(item) for item in prior["snapshots"]]
    indexed = {str(item["snapshot_id"]): item for item in existing}
    for bet in board.get("bets", []):
        record = _validated_bet(bet, board_hash=board_hash)
        indexed.setdefault(str(record["snapshot_id"]), record)
    snapshots = sorted(
        indexed.values(), key=lambda item: (str(item["observed_at_utc"]), str(item["snapshot_id"]))
    )
    body: dict[str, object] = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "policy_version": ARCHIVE_POLICY_VERSION,
        "paper_only": True,
        "execution_enabled": False,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
    }
    body["archive_sha256"] = canonical_hash(body)
    validate_published_bet_archive(body)
    _atomic_json(path, body)
    return body


def empty_published_bet_archive(path: Path) -> dict[str, object]:
    if path.exists():
        return validate_published_bet_archive(json.loads(path.read_text(encoding="utf-8")))
    body: dict[str, object] = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "policy_version": ARCHIVE_POLICY_VERSION,
        "paper_only": True,
        "execution_enabled": False,
        "snapshot_count": 0,
        "snapshots": [],
    }
    body["archive_sha256"] = canonical_hash(body)
    _atomic_json(path, body)
    return body


def _selection_role(probability: float) -> str:
    if abs(probability - 0.5) <= 1e-12:
        return "pickem"
    return "favorite" if probability > 0.5 else "underdog"


def _lead_hours(event_start: object, observed_at: object) -> float | None:
    if not event_start:
        return None
    return (
        utc_datetime(event_start, "event_start_utc")
        - utc_datetime(observed_at, "observed_at_utc")
    ).total_seconds() / 3600.0


def _status(settlement: object | None) -> tuple[str, float | None, str | None]:
    if settlement is None:
        return "pending", None, None
    raw = str(getattr(settlement, "settlement_status", ""))
    mapping = {
        "paper_win": "won", "paper_loss": "lost", "void": "void",
        "pass": "not_a_bet", "pass_unscored": "not_a_bet",
    }
    return (
        mapping.get(raw, raw or "pending"),
        float(getattr(settlement, "hypothetical_profit_units", 0.0)),
        str(getattr(settlement, "settled_at_utc", "")) or None,
    )


def _official_moneyline_records(
    decisions: Iterable[object], settlements: Iterable[object],
    quotes: Iterable[object], forecasts: Iterable[object],
) -> list[dict[str, object]]:
    settlement_by_decision = {item.decision_id: item for item in settlements}
    quote_by_id = {item.quote_id: item for item in quotes}
    forecast_by_id = {item.forecast_capture_id: item for item in forecasts}
    rows: list[dict[str, object]] = []
    for decision in decisions:
        if decision.paper_action == "pass":
            continue
        forecast = forecast_by_id.get(decision.forecast_capture_id)
        quote = quote_by_id.get(decision.reference_quote_id)
        if forecast is None or quote is None:
            raise ValueError("official moneyline bet lacks its forecast or quote")
        fighter_selected = decision.paper_action == "fighter"
        probability = float(decision.action_probability)
        model_probability = float(forecast.model_probability)
        if not fighter_selected:
            model_probability = 1.0 - model_probability
        line = int(decision.action_reference_moneyline)
        status, unit_profit, settled_at = _status(
            settlement_by_decision.get(decision.decision_id)
        )
        rows.append({
            "record_id": decision.decision_id,
            "record_type": "official_locked_t24",
            "official": True,
            "market_key": f"moneyline:{decision.matchup_id}",
            "category": "Moneyline",
            "event_id": decision.event_id,
            "event_title": "",
            "event_date": decision.event_date,
            "event_start_utc": decision.event_start_utc,
            "bout_order": None,
            "matchup_id": decision.matchup_id,
            "fighter_id": decision.fighter_id,
            "opponent_id": decision.opponent_id,
            "fighter_name": forecast.fighter_name,
            "opponent_name": forecast.opponent_name,
            "selection": forecast.fighter_name if fighter_selected else forecast.opponent_name,
            "side": decision.paper_action,
            "selection_role": _selection_role(probability),
            "target_book": quote.book,
            "offered_moneyline": line,
            "estimated_win_probability": probability,
            "estimated_expected_return": (
                float(decision.fighter_expected_return) if fighter_selected
                else float(decision.opponent_expected_return)
            ),
            "minimum_expected_return": float(decision.minimum_expected_return),
            "published_at_utc": decision.market_as_of_utc,
            "hours_before_start": _lead_hours(decision.event_start_utc, decision.market_as_of_utc),
            "kelly_fraction": kelly_fraction(probability, line),
            "status": status,
            "unit_profit": unit_profit,
            "settled_at_utc": settled_at,
            "probability_source": (
                "leave_one_book_out_no_vig_market_consensus"
                if float(decision.selected_gamma) == 0.0
                else "locked_market_model_log_odds_blend"
            ),
            "model_support_probability": model_probability,
            "model_support_source": "production_winner_model",
            "model_support_issued_at_utc": forecast.forecast_issued_at_utc,
            **_support_fields("simulation", None),
        })
    return rows


def _official_total_records(
    decisions: Iterable[object], settlements: Iterable[object],
) -> list[dict[str, object]]:
    settlement_by_decision = {item.decision_id: item for item in settlements}
    rows: list[dict[str, object]] = []
    for decision in decisions:
        if decision.paper_action == "pass":
            continue
        probability = float(decision.action_probability)
        line = int(decision.action_reference_moneyline)
        status, unit_profit, settled_at = _status(
            settlement_by_decision.get(decision.decision_id)
        )
        selection = f"{decision.paper_action.title()} {float(decision.line):g} rounds"
        rows.append({
            "record_id": decision.decision_id,
            "record_type": "official_locked_t24",
            "official": True,
            "market_key": f"total:{decision.matchup_id}:{float(decision.line):g}",
            "category": "Total rounds",
            "event_id": decision.event_id,
            "event_title": "",
            "event_date": decision.event_date,
            "event_start_utc": decision.event_start_utc,
            "bout_order": None,
            "matchup_id": decision.matchup_id,
            "fighter_id": decision.fighter_id,
            "opponent_id": decision.opponent_id,
            "fighter_name": decision.fighter_name,
            "opponent_name": decision.opponent_name,
            "selection": selection,
            "side": decision.paper_action,
            "selection_role": "total",
            "target_book": decision.target_book,
            "offered_moneyline": line,
            "estimated_win_probability": probability,
            "estimated_expected_return": (
                float(decision.residual_over_expected_return)
                if decision.paper_action == "over"
                else float(decision.residual_under_expected_return)
            ),
            "minimum_expected_return": float(decision.minimum_expected_return),
            "published_at_utc": decision.market_as_of_utc,
            "hours_before_start": _lead_hours(decision.event_start_utc, decision.market_as_of_utc),
            "kelly_fraction": kelly_fraction(probability, line),
            "status": status,
            "unit_profit": unit_profit,
            "settled_at_utc": settled_at,
            "probability_source": "locked_residual_duration_model",
            **_support_fields("model", None),
            **_support_fields("simulation", None),
        })
    return rows


def _archive_records(
    archive: Mapping[str, object],
    outcomes: Mapping[tuple[str, str, str], int | None],
    durations: Mapping[tuple[str, str, str], float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in archive.get("snapshots", []):
        fighter_id = str(item.get("fighter_id") or "")
        opponent_id = str(item.get("opponent_id") or "")
        key = (str(item.get("event_id") or ""), *sorted((fighter_id, opponent_id)))
        category = str(item.get("category") or "")
        side = str(item.get("side") or "")
        status = "pending"
        unit_profit: float | None = None
        target = outcomes.get(key)
        if key in outcomes:
            if target is None:
                status, unit_profit = "void", 0.0
            elif category == "Moneyline":
                canonical_fighter = sorted((fighter_id, opponent_id))[0]
                selected_id = fighter_id if side == "fighter" else opponent_id
                selected_won = (target == 1 and selected_id == canonical_fighter) or (
                    target == 0 and selected_id != canonical_fighter
                )
                status = "won" if selected_won else "lost"
                unit_profit = profit_multiple(item["offered_moneyline"]) if selected_won else -1.0
            elif category == "Total rounds" and key in durations:
                import re
                match = re.search(r"(Over|Under)\s+([0-9]+(?:\.[0-9]+)?)", str(item.get("selection")), re.I)
                if match:
                    threshold = float(match.group(2)) * 300.0
                    if abs(durations[key] - threshold) <= 1e-9:
                        status, unit_profit = "void", 0.0
                    else:
                        won = durations[key] > threshold if match.group(1).casefold() == "over" else durations[key] < threshold
                        status = "won" if won else "lost"
                        unit_profit = profit_multiple(item["offered_moneyline"]) if won else -1.0
        probability = float(item["estimated_win_probability"])
        line = int(item["offered_moneyline"])
        market_key = (
            f"moneyline:{item['matchup_id']}" if category == "Moneyline"
            else f"total:{item['matchup_id']}:{item.get('selection', '')}"
        )
        rows.append({
            "record_id": item["snapshot_id"],
            "record_type": "published_snapshot",
            "official": False,
            "market_key": market_key,
            "category": category,
            "event_id": item.get("event_id"),
            "event_title": item.get("event_title") or "",
            "event_date": item.get("event_date"),
            "event_start_utc": item.get("event_start_utc"),
            "bout_order": item.get("bout_order"),
            "matchup_id": item.get("matchup_id"),
            "fighter_id": fighter_id,
            "opponent_id": opponent_id,
            "fighter_name": item.get("fighter_name"),
            "opponent_name": item.get("opponent_name"),
            "selection": item.get("selection"),
            "side": side,
            "selection_role": _selection_role(probability) if category == "Moneyline" else "total",
            "target_book": item.get("target_book"),
            "offered_moneyline": line,
            "estimated_win_probability": probability,
            "estimated_expected_return": float(item["estimated_expected_return"]),
            "minimum_expected_return": float(item["minimum_expected_return"]),
            "published_at_utc": item["observed_at_utc"],
            "hours_before_start": _lead_hours(item.get("event_start_utc"), item["observed_at_utc"]),
            "kelly_fraction": kelly_fraction(probability, line),
            "status": status,
            "unit_profit": unit_profit,
            "settled_at_utc": None,
            "probability_source": item.get("probability_source"),
            "bayesian_kelly": item.get("bayesian_kelly"),
            **_support_fields("model", None),
            **_support_fields("simulation", None),
        })
    return rows


def _attach_research_support(
    records: list[dict[str, object]],
    *,
    model_support: Mapping[tuple[str, ...], Mapping[str, object]],
    simulation_support: Mapping[tuple[str, ...], Mapping[str, object]],
    prior_support: Mapping[str, Mapping[str, object]],
) -> None:
    support_maps = {"model": model_support, "simulation": simulation_support}
    for record in records:
        key = bet_support_key(
            event_id=record["event_id"],
            fighter_id=record["fighter_id"],
            opponent_id=record["opponent_id"],
            category=record["category"],
            side=record["side"],
            selection=record["selection"],
        )
        prior = prior_support.get(str(record["record_id"]), {})
        published = utc_datetime(record["published_at_utc"], "published_at_utc")
        for source, support_map in support_maps.items():
            probability_key = f"{source}_support_probability"
            existing = None
            if record.get(probability_key) is not None:
                existing = {
                    "probability": record[probability_key],
                    "source": record.get(f"{source}_support_source"),
                    "issued_at_utc": record.get(f"{source}_support_issued_at_utc"),
                }
            candidate = support_map.get(key) or existing
            if candidate is None and prior.get(probability_key) is not None:
                candidate = {
                    "probability": prior[probability_key],
                    "source": prior.get(f"{source}_support_source"),
                    "issued_at_utc": prior.get(f"{source}_support_issued_at_utc"),
                }
            fields = _support_fields(source, candidate)
            issued = fields[f"{source}_support_issued_at_utc"]
            if issued is not None and utc_datetime(
                issued, f"{source}_support_issued_at_utc"
            ) > published:
                fields = _support_fields(source, None)
            record.update(fields)


def _attach_bayesian_kelly(
    records: list[dict[str, object]],
    calibrator: BayesianKellyCalibrator,
    prior_support: Mapping[str, Mapping[str, object]],
) -> None:
    trained_through = utc_datetime(
        f"{calibrator.artifact['training_last_event_date']}T00:00:00Z",
        "Bayesian Kelly training cutoff",
    )
    for record in records:
        existing = record.get("bayesian_kelly")
        if existing is None:
            prior = prior_support.get(str(record.get("record_id") or ""), {})
            existing = prior.get("bayesian_kelly")
        if existing is not None:
            record["bayesian_kelly"] = validate_bayesian_kelly_assessment(existing)
            continue
        if record.get("category") != "Moneyline":
            record["bayesian_kelly"] = unavailable_assessment(
                "Fight totals do not yet have a validated posterior calibration."
            )
            continue
        if record.get("probability_source") != "leave_one_book_out_no_vig_market_consensus":
            record["bayesian_kelly"] = unavailable_assessment(
                "This probability did not come from the calibrated moneyline-consensus source."
            )
            continue
        published = utc_datetime(record["published_at_utc"], "published_at_utc")
        if trained_through >= published:
            record["bayesian_kelly"] = unavailable_assessment(
                "The calibration data was not strictly earlier than this published bet."
            )
            continue
        record["bayesian_kelly"] = calibrator.assessment(
            record["estimated_win_probability"],
            record["offered_moneyline"],
            assessment_timing=(
                "retrospective_policy_using_only_prepublication_calibration_data"
            ),
        )


def build_bet_performance_publication(
    *, decisions: Iterable[object], settlements: Iterable[object],
    quotes: Iterable[object], forecasts: Iterable[object],
    total_decisions: Iterable[object] = (), total_settlements: Iterable[object] = (),
    archive: Mapping[str, object],
    outcomes: Mapping[tuple[str, str, str], int | None] | None = None,
    durations: Mapping[tuple[str, str, str], float] | None = None,
    model_support: Mapping[tuple[str, ...], Mapping[str, object]] | None = None,
    simulation_support: Mapping[tuple[str, ...], Mapping[str, object]] | None = None,
    prior_support: Mapping[str, Mapping[str, object]] | None = None,
    bayesian_kelly_calibrator: BayesianKellyCalibrator | None = None,
) -> dict[str, object]:
    decisions = tuple(decisions)
    settlements = tuple(settlements)
    quotes = tuple(quotes)
    forecasts = tuple(forecasts)
    total_decisions = tuple(total_decisions)
    total_settlements = tuple(total_settlements)
    outcomes = outcomes or {}
    durations = durations or {}
    records = [
        *_official_moneyline_records(decisions, settlements, quotes, forecasts),
        *_official_total_records(total_decisions, total_settlements),
        *_archive_records(archive, outcomes, durations),
    ]
    _attach_research_support(
        records,
        model_support=model_support or {},
        simulation_support=simulation_support or {},
        prior_support=prior_support or {},
    )
    calibrator = bayesian_kelly_calibrator or BayesianKellyCalibrator.load()
    _attach_bayesian_kelly(records, calibrator, prior_support or {})
    records.sort(key=lambda item: (str(item["published_at_utc"]), str(item["record_id"])))
    official = [item for item in records if item["official"]]
    body: dict[str, object] = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "policy_version": PERFORMANCE_POLICY_VERSION,
        "paper_only": True,
        "execution_enabled": False,
        "as_of_utc": max(
            (str(item.get("settled_at_utc") or item["published_at_utc"]) for item in records),
            default="1970-01-01T00:00:00Z",
        ),
        "archive_started_at_utc": (
            min((str(item["published_at_utc"]) for item in records if not item["official"]), default=None)
        ),
        "record_count": len(records),
        "official_bet_count": len(official),
        "official_settled_count": sum(item["status"] in {"won", "lost", "void"} for item in official),
        "official_wins": sum(item["status"] == "won" for item in official),
        "official_losses": sum(item["status"] == "lost" for item in official),
        "staking_strategies": [
            "full_kelly", "half_kelly", "third_kelly", "flat_one_percent",
            *RESEARCH_STAKING_STRATEGIES,
        ],
        "research_support": {
            "model_supported_records": sum(
                item["model_support_probability"] is not None for item in records
            ),
            "simulation_supported_records": sum(
                item["simulation_support_probability"] is not None for item in records
            ),
            "both_supported_records": sum(
                item["model_support_probability"] is not None
                and item["simulation_support_probability"] is not None
                for item in records
            ),
            "bayesian_kelly_supported_records": sum(
                item["bayesian_kelly"]["status"] == "available"
                for item in records
            ),
            "bayesian_kelly_calibration_artifact_sha256": calibrator.artifact[
                "artifact_sha256"
            ],
            "rule": (
                "Robust Bayesian Kelly sizes moneylines from the lower 10th-percentile "
                "calibrated chance and caps one bet at 5% of bankroll. Other research "
                "strategies combine probabilities in log-odds space, then apply half "
                "Kelly. Unsupported bets are excluded."
            ),
        },
        "timing_strategies": ["official_t24", "first_qualifying", "nearest_t48", "nearest_t24", "favorite_early_underdog_late", "latest_qualifying"],
        "records": records,
        "source_hashes": {
            "published_archive": archive.get("archive_sha256"),
            "official_decisions": canonical_hash([item.to_mapping() for item in decisions]),
            "official_settlements": canonical_hash([item.to_mapping() for item in settlements]),
            "official_total_decisions": canonical_hash([item.to_mapping() for item in total_decisions]),
            "official_total_settlements": canonical_hash([item.to_mapping() for item in total_settlements]),
        },
        "limitations": (
            "Only timestamped paper bets are included. The exact official history starts "
            "with the locked ledger; alternative timing replays start when the publication archive begins."
        ),
    }
    body["publication_sha256"] = canonical_hash(body)
    return validate_bet_performance_publication(body)


def validate_bet_performance_publication(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("bet performance publication must be an object")
    supplied = value.get("publication_sha256")
    body = dict(value)
    body.pop("publication_sha256", None)
    if supplied != canonical_hash(body):
        raise ValueError("bet performance publication hash is invalid")
    if (
        value.get("schema_version") != PERFORMANCE_SCHEMA_VERSION
        or value.get("policy_version") != PERFORMANCE_POLICY_VERSION
        or value.get("paper_only") is not True
        or value.get("execution_enabled") is not False
    ):
        raise ValueError("bet performance publication contract is invalid")
    records = value.get("records")
    if not isinstance(records, list) or value.get("record_count") != len(records):
        raise ValueError("bet performance record count is inconsistent")
    research_support = value.get("research_support")
    has_bayesian_contract = (
        isinstance(research_support, dict)
        and bool(research_support.get("bayesian_kelly_calibration_artifact_sha256"))
    )
    identifiers: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not str(record.get("record_id") or ""):
            raise ValueError("bet performance contains an invalid record")
        if record["record_id"] in identifiers:
            raise ValueError("bet performance contains duplicate record IDs")
        identifiers.add(record["record_id"])
        if record.get("status") not in {"won", "lost", "void", "pending"}:
            raise ValueError("bet performance contains an invalid result")
        fraction = float(record.get("kelly_fraction"))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("bet performance contains an invalid Kelly fraction")
        expected = kelly_fraction(record["estimated_win_probability"], record["offered_moneyline"])
        if abs(fraction - expected) > 1e-12:
            raise ValueError("bet performance Kelly fraction is inconsistent")
        for source in SUPPORT_SOURCES:
            chance = record.get(f"{source}_support_probability")
            label = record.get(f"{source}_support_source")
            issued = record.get(f"{source}_support_issued_at_utc")
            if chance is None and label is None and issued is None:
                continue
            if chance is None or label is None or issued is None:
                raise ValueError(f"bet performance {source} support is incomplete")
            numeric = float(chance)
            if not math.isfinite(numeric) or not 0.0 < numeric < 1.0:
                raise ValueError(f"bet performance {source} support is invalid")
            if utc_datetime(issued, f"{source}_support_issued_at_utc") > utc_datetime(
                record["published_at_utc"], "published_at_utc"
            ):
                raise ValueError(f"bet performance {source} support is from the future")
        if record["category"] != "Moneyline" and record.get("model_support_probability") is not None:
            raise ValueError("winner-model support cannot size a total-round bet")
        if has_bayesian_contract:
            validate_bayesian_kelly_assessment(record.get("bayesian_kelly"))
    return value


def write_bet_performance_publication(value: Mapping[str, object], path: Path) -> None:
    _atomic_json(path, validate_bet_performance_publication(dict(value)))
