"""Bounded candidate outcome forecasts for upcoming UFC matchups.

This publication is deliberately separate from the production winner model.
It freezes coherent winner, method, and duration probabilities so a later odds
capture can evaluate the exact forecast that existed before the event.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
import os
import re
from typing import Mapping

import pandas as pd

from bayesian_total_calibration import (
    BayesianTotalCalibrator,
    CALIBRATION_POLICY_VERSION,
)
from fight_semantics import SCHEDULE_CONTRACT_VERSION, upcoming_schedule
from market_tracker import matchup_id_for

from .outcome_model import DiscreteTimeOutcomeModel


OUTCOME_FORECAST_SCHEMA_VERSION = 1
OUTCOME_MODEL_VERSION = "candidate-discrete-time-competing-risks-v2-verified-schedules"


def outcome_forecasts_usable(publication: Mapping[str, object]) -> bool:
    """Legacy artifacts remain readable history, not eligible current forecasts."""
    return (
        publication.get("model_version") == OUTCOME_MODEL_VERSION
        and publication.get("schedule_contract_version") == SCHEDULE_CONTRACT_VERSION
        and int(publication.get("forecast_matchup_count", 0)) > 0
    )


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
    return " ".join(str(value or "").split())


def scheduled_rounds_for_upcoming(
    bout_index: int, division: object
) -> tuple[int, str]:
    """Resolve scheduled rounds from durable upcoming-card information.

    UFCStats lists the main event first and identifies title bouts in the bout
    label. The basis is published with every forecast so this assumption can be
    audited and replaced if UFCStats later exposes an explicit format field.
    """

    return upcoming_schedule(bout_index, division)


def build_outcome_forecast_publication(
    model: DiscreteTimeOutcomeModel | None,
    feature_builder: object,
    upcoming: pd.DataFrame,
    card: Mapping[str, object],
    *,
    selected_c: float,
    training_input_sha256: str,
    model_trained_through: str,
    forecast_issued_at_utc: str,
    source_commit_sha: str,
    unavailable_reason: str = "Insufficient independently verified scheduled fight lengths.",
) -> dict[str, object]:
    """Build one content-addressed, candidate-only upcoming-card forecast."""

    event_id = _text(card.get("event_id"))
    event_url = _text(card.get("event_url"))
    event_date = _text(card.get("date"))
    event_title = _text(card.get("title"))
    if not all((event_id, event_url, event_date, event_title)):
        raise ValueError("outcome forecast card metadata is incomplete")
    if len(training_input_sha256) != 64:
        raise ValueError("outcome training input requires a SHA-256 fingerprint")

    if model is not None and getattr(model, "schedule_contract_version", None) != SCHEDULE_CONTRACT_VERSION:
        raise ValueError("Outcome model lacks independently verified schedules")
    model_contract = {
        "model_version": OUTCOME_MODEL_VERSION,
        "schedule_contract_version": SCHEDULE_CONTRACT_VERSION,
        "interval_seconds": model.interval_seconds if model is not None else 30,
        "selected_c": float(selected_c),
        "training_input_sha256": training_input_sha256,
        "feature_columns": list(model.feature_columns) if model is not None else [],
    }
    model_id = _canonical_hash(model_contract)[:24]
    total_calibrator = (
        BayesianTotalCalibrator(model.total_calibration_artifact)
        if model is not None and model.total_calibration_artifact is not None
        else None
    )
    matchups: list[dict[str, object]] = []
    for bout_index, row in upcoming.reset_index(drop=True).iterrows():
        fighter_id = _text(row.get("fighter id"))
        opponent_id = _text(row.get("opponent id"))
        fighter_name = _text(row.get("fighter name"))
        opponent_name = _text(row.get("opponent name"))
        division = _text(row.get("division"))
        status = _text(row.get("model status"))
        rounds, schedule_basis = scheduled_rounds_for_upcoming(
            int(bout_index), division
        )
        item: dict[str, object] = {
            "bout_order": int(bout_index),
            "fighter_id": fighter_id or None,
            "opponent_id": opponent_id or None,
            "fighter_name": fighter_name,
            "opponent_name": opponent_name,
            "division": division,
            "scheduled_rounds": rounds,
            "schedule_basis": schedule_basis,
            "forecast_status": status,
        }
        if (
            model is None
            or not fighter_id
            or not opponent_id
            or fighter_id == opponent_id
            or status.casefold().startswith("abstain")
        ):
            item["matchup_id"] = None
            item["forecast_status"] = (
                "unavailable_verified_schedule_history" if model is None
                else "abstain_unresolved_identity"
            )
            if model is None:
                item["reason"] = unavailable_reason
            matchups.append(item)
            continue
        features = feature_builder.matchup_features(
            fighter_id, opponent_id, event_date, division
        )
        prediction = model.predict(features.iloc[0], scheduled_rounds=rounds)
        total_probabilities: dict[str, float] = {}
        total_posteriors: dict[str, object] = {}
        for half_round in range(1, rounds * 2, 2):
            line = half_round / 2.0
            probability = prediction.probability_over_seconds(int(line * 300))
            if probability is not None:
                total_probabilities[f"{line:.1f}"] = float(probability)
                if total_calibrator is not None:
                    total_posteriors[f"{line:.1f}"] = total_calibrator.summary(
                        probability, line
                    )
        item.update(
            {
                "matchup_id": matchup_id_for(event_id, fighter_id, opponent_id),
                "forecast_status": "candidate_model",
                "fighter_win_probability": float(
                    prediction.fighter_win_probability
                ),
                "terminal_probabilities": {
                    key: float(value)
                    for key, value in prediction.terminal_probabilities.items()
                },
                "method_probabilities": {
                    key: float(value)
                    for key, value in prediction.method_probabilities.items()
                },
                "total_round_over_probabilities": total_probabilities,
                "total_round_over_probability_posteriors": total_posteriors,
            }
        )
        matchups.append(item)

    body: dict[str, object] = {
        "schema_version": OUTCOME_FORECAST_SCHEMA_VERSION,
        "model_version": OUTCOME_MODEL_VERSION,
        "schedule_contract_version": SCHEDULE_CONTRACT_VERSION,
        "availability_status": "available" if model is not None else "unavailable",
        "unavailable_reason": unavailable_reason if model is None else None,
        "model_id": model_id,
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "event_id": event_id,
        "event_url": event_url,
        "event_date": event_date,
        "event_title": event_title,
        "forecast_issued_at_utc": forecast_issued_at_utc,
        "source_commit_sha": source_commit_sha,
        "model_trained_through": model_trained_through,
        "training_input_sha256": training_input_sha256,
        "selected_c": float(selected_c),
        "interval_seconds": model.interval_seconds if model is not None else 30,
        "training_fights": model.training_fights if model is not None else 0,
        "training_risk_rows": model.training_risk_rows if model is not None else 0,
        "omitted_training_unknown_schedule": model.omitted_unknown_schedule if model is not None else 0,
        "schedule_contract": (
            "five rounds for UFCStats title labels and the first-listed main "
            "event; otherwise three rounds"
        ),
        "method_price_status": "unavailable_from_configured_provider",
        "bayesian_total_calibration": (
            {
                "status": "available",
                "policy_version": CALIBRATION_POLICY_VERSION,
                "artifact_sha256": total_calibrator.artifact["artifact_sha256"],
                "trained_through": total_calibrator.artifact[
                    "training_last_event_date"
                ],
            }
            if total_calibrator is not None
            else {
                "status": "unavailable",
                "policy_version": CALIBRATION_POLICY_VERSION,
                "reason": "No historical total calibration was supplied.",
            }
        ),
        "matchup_count": len(matchups),
        "forecast_matchup_count": sum(
            item.get("matchup_id") is not None for item in matchups
        ),
        "matchups": matchups,
    }
    body["publication_sha256"] = _canonical_hash(body)
    return validate_outcome_forecast_publication(body)


def validate_outcome_forecast_publication(
    publication: object,
) -> dict[str, object]:
    if not isinstance(publication, dict):
        raise ValueError("outcome forecast publication must be an object")
    if publication.get("schema_version") != OUTCOME_FORECAST_SCHEMA_VERSION:
        raise ValueError("unsupported outcome forecast schema version")
    if (
        publication.get("candidate_only") is not True
        or publication.get("paper_only") is not True
        or publication.get("execution_enabled") is not False
    ):
        raise ValueError("outcome forecasts must remain candidate paper research")
    event_id = _text(publication.get("event_id"))
    if not event_id:
        raise ValueError("outcome forecast event ID is blank")
    issued = pd.to_datetime(
        publication.get("forecast_issued_at_utc"), errors="coerce", utc=True
    )
    event_date = pd.to_datetime(publication.get("event_date"), errors="coerce")
    trained_through = pd.to_datetime(
        publication.get("model_trained_through"), errors="coerce"
    )
    if pd.isna(issued) or pd.isna(event_date) or pd.isna(trained_through):
        raise ValueError("outcome forecast timing metadata is invalid")
    if not issued.date() < event_date.date():
        raise ValueError("outcome forecasts must be issued before the event date")
    if not trained_through.date() < issued.date():
        raise ValueError("outcome training cutoff must precede forecast issuance")
    if not re.fullmatch(
        r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})",
        _text(publication.get("source_commit_sha")),
    ):
        raise ValueError("outcome forecast source revision is invalid")
    supplied_hash = publication.get("publication_sha256")
    unhashed = dict(publication)
    unhashed.pop("publication_sha256", None)
    if supplied_hash != _canonical_hash(unhashed):
        raise ValueError("outcome forecast publication hash is invalid")
    calibration = publication.get("bayesian_total_calibration")
    if not isinstance(calibration, dict) or calibration.get(
        "policy_version"
    ) != CALIBRATION_POLICY_VERSION:
        raise ValueError("outcome forecast Bayesian total calibration is invalid")
    if calibration.get("status") == "available":
        if len(_text(calibration.get("artifact_sha256"))) != 64:
            raise ValueError("outcome forecast total calibration identity is invalid")
    elif calibration.get("status") != "unavailable" or not _text(
        calibration.get("reason")
    ):
        raise ValueError("outcome forecast total calibration status is invalid")
    matchups = publication.get("matchups")
    if not isinstance(matchups, list) or len(matchups) != publication.get(
        "matchup_count"
    ):
        raise ValueError("outcome forecast matchup count is invalid")
    seen: set[str] = set()
    resolved = 0
    for item in matchups:
        if not isinstance(item, dict):
            raise ValueError("outcome forecast matchup must be an object")
        matchup_id = item.get("matchup_id")
        if matchup_id is None:
            continue
        resolved += 1
        if not isinstance(matchup_id, str) or matchup_id in seen:
            raise ValueError("outcome forecast matchup IDs must be unique")
        seen.add(matchup_id)
        fighter_id = _text(item.get("fighter_id"))
        opponent_id = _text(item.get("opponent_id"))
        if matchup_id != matchup_id_for(event_id, fighter_id, opponent_id):
            raise ValueError("outcome forecast matchup ID disagrees with fighter IDs")
        rounds = item.get("scheduled_rounds")
        if rounds not in (3, 5):
            raise ValueError("outcome forecast schedule must be three or five rounds")
        for field in ("terminal_probabilities", "method_probabilities"):
            probabilities = item.get(field)
            if not isinstance(probabilities, dict) or not probabilities:
                raise ValueError(f"{field} is missing")
            values = [float(value) for value in probabilities.values()]
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"{field} contains an invalid probability")
            if abs(sum(values) - 1.0) > 1e-9:
                raise ValueError(f"{field} probabilities do not sum to one")
        terminal = item["terminal_probabilities"]
        methods = item["method_probabilities"]
        fighter_win = float(item.get("fighter_win_probability"))
        expected_fighter_win = sum(
            float(value)
            for key, value in terminal.items()
            if str(key).startswith("fighter_")
        )
        if abs(fighter_win - expected_fighter_win) > 1e-9:
            raise ValueError("fighter win probability disagrees with terminal outcomes")
        for method in ("ko_tko", "submission", "decision", "other"):
            expected_method = sum(
                float(value)
                for key, value in terminal.items()
                if str(key).endswith(f"_{method}")
            )
            if abs(float(methods.get(method, -1.0)) - expected_method) > 1e-9:
                raise ValueError("method probability disagrees with terminal outcomes")
        totals = item.get("total_round_over_probabilities")
        if not isinstance(totals, dict) or not totals:
            raise ValueError("total-round probabilities are missing")
        ordered = [float(value) for _, value in sorted(totals.items(), key=lambda pair: float(pair[0]))]
        if any(not 0.0 <= value <= 1.0 for value in ordered):
            raise ValueError("total-round probability is outside [0, 1]")
        if any(left < right for left, right in zip(ordered, ordered[1:])):
            raise ValueError("over probabilities must not increase with the line")
        expected_lines = {f"{value / 2:.1f}" for value in range(1, rounds * 2, 2)}
        if set(totals) != expected_lines:
            raise ValueError("total-round forecast lines disagree with scheduled rounds")
        posteriors = item.get("total_round_over_probability_posteriors")
        if not isinstance(posteriors, dict):
            raise ValueError("total-round probability posteriors are missing")
        if calibration.get("status") == "available" and set(posteriors) != expected_lines:
            raise ValueError("total-round posterior lines disagree with point forecasts")
        for line, summary in posteriors.items():
            if line not in expected_lines or not isinstance(summary, dict):
                raise ValueError("total-round posterior line is invalid")
            if summary.get("status") == "unavailable":
                if not _text(summary.get("reason")):
                    raise ValueError("unavailable total-round posterior needs a reason")
                continue
            if summary.get("status") != "available":
                raise ValueError("total-round posterior status is invalid")
            nominal = float(summary.get("nominal_over_probability"))
            lower = float(summary.get("posterior_lower_over_probability"))
            mean = float(summary.get("posterior_mean_over_probability"))
            upper = float(summary.get("posterior_upper_over_probability"))
            if (
                abs(nominal - float(totals[line])) > 1e-12
                or not 0.0 < lower <= mean <= upper < 1.0
            ):
                raise ValueError("total-round posterior probabilities are inconsistent")
            if summary.get("calibration_artifact_sha256") != calibration.get(
                "artifact_sha256"
            ):
                raise ValueError("total-round posterior calibration identity changed")
    if resolved != publication.get("forecast_matchup_count"):
        raise ValueError("outcome forecast resolved count is invalid")
    return publication


def write_outcome_forecast_publication(
    path: str | Path, publication: Mapping[str, object]
) -> None:
    validated = validate_outcome_forecast_publication(dict(publication))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        validated,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
