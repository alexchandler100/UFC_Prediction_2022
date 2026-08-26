"""Append-only, candidate-only publication schema for simulation forecasts."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping

import pandas as pd

from market_tracker import matchup_id_for

from .domain import ENGINE_VERSION, SCHEMA_VERSION
from .parameters import (
    PARAMETER_MODEL_VERSION,
    ParameterEnsembleArtifact,
    canonical_sha256,
)


SHADOW_FORECAST_SCHEMA_VERSION = 1
SHADOW_MODEL_VERSION = "candidate-fight-sim-v2"
SHADOW_AGGREGATE_DETAIL_LEVEL = "compact_shadow_v1"
SHADOW_LOCAL_ONLY_FIELDS = ("bootstrap_statistic_distributions",)
MAX_SHADOW_PUBLICATION_BYTES = 16 * 1024 * 1024


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).split())


def _mapping(value: object) -> dict[str, object]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("aggregate forecast must be a mapping or expose to_dict")
    return dict(value)


def _artifact_metadata(
    artifact: ParameterEnsembleArtifact | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(artifact, ParameterEnsembleArtifact):
        artifact.validate()
        return {
            "artifact_sha256": artifact.artifact_sha256,
            "as_of_utc": artifact.as_of_utc,
            "trained_through": artifact.trained_through,
            "input_sha256": artifact.input_sha256,
            "bootstrap_members": len(artifact.members),
            "parameter_model_version": artifact.model_version,
        }
    value = dict(artifact)
    required = {
        "artifact_sha256",
        "as_of_utc",
        "trained_through",
        "input_sha256",
        "bootstrap_members",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"parameter artifact metadata is missing fields: {missing}")
    return value


def _normalize_matchups(matchups: pd.DataFrame | Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    if isinstance(matchups, pd.DataFrame):
        rows = matchups.to_dict("records")
    else:
        rows = [dict(row) for row in matchups]
    aliases = {
        "fighter id": "red_fighter_id",
        "opponent id": "blue_fighter_id",
        "fighter name": "red_fighter_name",
        "opponent name": "blue_fighter_name",
        "fighter_id": "red_fighter_id",
        "opponent_id": "blue_fighter_id",
        "fighter_name": "red_fighter_name",
        "opponent_name": "blue_fighter_name",
    }
    output: list[dict[str, object]] = []
    for index, original in enumerate(rows):
        row = dict(original)
        for source, target in aliases.items():
            if target not in row and source in row:
                row[target] = row[source]
        row.setdefault("bout_order", index)
        output.append(row)
    return output


def _validate_aggregate(
    value: Mapping[str, object],
    matchup_id: str,
    *,
    require_compact: bool = True,
) -> dict[str, object]:
    aggregate = dict(value)
    if _text(aggregate.get("matchup_id")) != matchup_id:
        raise ValueError("aggregate matchup ID disagrees with publication matchup")
    total_paths = int(aggregate.get("total_paths") or 0)
    members = int(aggregate.get("bootstrap_members") or 0)
    if total_paths <= 0 or members <= 0:
        raise ValueError("aggregate forecast requires paths and bootstrap members")
    counts = aggregate.get("outcome_counts")
    if isinstance(counts, list):
        counts = {
            str(item["outcome"]): int(item["count"])
            for item in counts
        }
    if not isinstance(counts, Mapping) or not counts:
        raise ValueError("aggregate outcome counts are missing")
    parsed_counts = {str(key): int(count) for key, count in counts.items()}
    if any(count < 0 for count in parsed_counts.values()) or sum(parsed_counts.values()) != total_paths:
        raise ValueError("aggregate outcome counts do not sum to total paths")
    probabilities = aggregate.get("outcome_probabilities")
    if probabilities is None:
        probabilities = {
            key: count / total_paths for key, count in parsed_counts.items()
        }
    parsed_probabilities = {
        str(key): float(item) for key, item in dict(probabilities).items()
    }
    if set(parsed_probabilities) != set(parsed_counts):
        raise ValueError("aggregate outcome probability support disagrees with counts")
    for key, probability in parsed_probabilities.items():
        if not math.isfinite(probability) or probability < 0:
            raise ValueError("aggregate contains invalid outcome probability")
        if abs(probability - parsed_counts[key] / total_paths) > 1e-12:
            raise ValueError("aggregate probabilities are not derived from exact counts")
    if abs(sum(parsed_probabilities.values()) - 1.0) > 1e-12:
        raise ValueError("aggregate probabilities do not sum to one")
    rounds = int(aggregate.get("scheduled_rounds") or 0)
    if rounds not in (3, 5):
        raise ValueError("shadow forecasts support three or five scheduled rounds")
    for line in list(aggregate.get("total_lines") or []):
        settlement = dict(line)
        settlement_total = sum(
            int(settlement.get(name) or 0)
            for name in ("over", "under", "push", "no_action")
        )
        if settlement_total != total_paths:
            raise ValueError("total-line settlement counts do not sum to paths")
    survival = [dict(point) for point in list(aggregate.get("survival") or [])]
    ordered = [float(point["probability"]) for point in survival]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in ordered):
        raise ValueError("survival curve contains invalid probability")
    if any(left < right for left, right in zip(ordered, ordered[1:])):
        raise ValueError("survival probabilities must not increase")
    if require_compact:
        if aggregate.get("detail_level") != SHADOW_AGGREGATE_DETAIL_LEVEL:
            raise ValueError("shadow aggregate does not use the compact publication schema")
        local_hash = _text(aggregate.get("local_aggregate_sha256"))
        if not re.fullmatch(r"[0-9a-f]{64}", local_hash):
            raise ValueError("shadow aggregate local authority hash is invalid")
        omitted = aggregate.get("omitted_local_authority_fields")
        if omitted != list(SHADOW_LOCAL_ONLY_FIELDS):
            raise ValueError("shadow aggregate local-only field manifest is invalid")
        present = sorted(field for field in SHADOW_LOCAL_ONLY_FIELDS if field in aggregate)
        if present:
            raise ValueError(
                f"shadow aggregate contains local-only authority fields: {present}"
            )
    aggregate["outcome_counts"] = parsed_counts
    aggregate["outcome_probabilities"] = parsed_probabilities
    return aggregate


def compact_shadow_aggregate(value: object) -> dict[str, object]:
    """Create the bounded paper-publication view of a local aggregate.

    Exact per-bootstrap statistic histograms remain in the ignored local run
    artifact.  Their canonical aggregate hash is carried into the immutable
    shadow file so the compact publication can be tied back to that authority.
    Overall exact histograms, per-bootstrap outcome counts, and both process
    and parameter/model uncertainty summaries remain in the shadow.
    """

    aggregate = _mapping(value)
    matchup_id = _text(aggregate.get("matchup_id"))
    if not matchup_id:
        raise ValueError("aggregate forecast matchup ID is blank")
    if aggregate.get("detail_level") == SHADOW_AGGREGATE_DETAIL_LEVEL:
        return _validate_aggregate(aggregate, matchup_id, require_compact=True)
    aggregate = _validate_aggregate(aggregate, matchup_id, require_compact=False)
    local_hash = canonical_sha256(aggregate)
    for field in SHADOW_LOCAL_ONLY_FIELDS:
        aggregate.pop(field, None)
    aggregate["detail_level"] = SHADOW_AGGREGATE_DETAIL_LEVEL
    aggregate["local_aggregate_sha256"] = local_hash
    aggregate["omitted_local_authority_fields"] = list(SHADOW_LOCAL_ONLY_FIELDS)
    return _validate_aggregate(aggregate, matchup_id, require_compact=True)


def build_shadow_forecast_publication(
    forecasts: Mapping[str, object],
    matchups: pd.DataFrame | Iterable[Mapping[str, object]],
    card: Mapping[str, object],
    parameter_artifact: ParameterEnsembleArtifact | Mapping[str, object],
    *,
    forecast_issued_at_utc: str,
    source_commit_sha: str,
    engine_version: str = ENGINE_VERSION,
) -> dict[str, object]:
    """Build a separate, paper-only simulation publication for one card."""

    event_id = _text(card.get("event_id"))
    event_url = _text(card.get("event_url"))
    event_date = _text(card.get("date") or card.get("event_date"))
    event_title = _text(card.get("title") or card.get("event_title"))
    if not all((event_id, event_date, event_title)):
        raise ValueError("shadow forecast card metadata is incomplete")
    artifact = _artifact_metadata(parameter_artifact)
    normalized_forecasts = {
        str(key): compact_shadow_aggregate(value) for key, value in forecasts.items()
    }
    items: list[dict[str, object]] = []
    used: set[str] = set()
    for row in _normalize_matchups(matchups):
        red_id = _text(row.get("red_fighter_id"))
        blue_id = _text(row.get("blue_fighter_id"))
        supplied_matchup = _text(row.get("matchup_id"))
        if not red_id or not blue_id or red_id == blue_id:
            items.append(
                {
                    "bout_order": int(row.get("bout_order") or 0),
                    "matchup_id": None,
                    "red_fighter_id": red_id or None,
                    "blue_fighter_id": blue_id or None,
                    "red_fighter_name": _text(row.get("red_fighter_name")),
                    "blue_fighter_name": _text(row.get("blue_fighter_name")),
                    "division": _text(row.get("division")),
                    "forecast_status": "abstain_unresolved_identity",
                }
            )
            continue
        matchup_id = supplied_matchup or matchup_id_for(event_id, red_id, blue_id)
        if matchup_id != matchup_id_for(event_id, red_id, blue_id):
            raise ValueError("shadow forecast matchup ID disagrees with stable IDs")
        if matchup_id in used:
            raise ValueError("shadow forecast matchup IDs must be unique")
        used.add(matchup_id)
        if matchup_id not in normalized_forecasts:
            raise ValueError(f"missing aggregate simulation forecast for {matchup_id}")
        aggregate = _validate_aggregate(normalized_forecasts[matchup_id], matchup_id)
        items.append(
            {
                "bout_order": int(row.get("bout_order") or 0),
                "matchup_id": matchup_id,
                "red_fighter_id": red_id,
                "blue_fighter_id": blue_id,
                "red_fighter_name": _text(row.get("red_fighter_name")),
                "blue_fighter_name": _text(row.get("blue_fighter_name")),
                "division": _text(row.get("division")),
                "forecast_status": "candidate_simulation",
                "aggregate": aggregate,
            }
        )
    unexpected = set(normalized_forecasts) - used
    if unexpected:
        raise ValueError(f"forecasts do not belong to supplied card matchups: {sorted(unexpected)}")
    body: dict[str, object] = {
        "schema_version": SHADOW_FORECAST_SCHEMA_VERSION,
        "simulator_schema_version": SCHEMA_VERSION,
        "model_version": SHADOW_MODEL_VERSION,
        "engine_version": engine_version,
        "parameter_model_version": artifact.get(
            "parameter_model_version", PARAMETER_MODEL_VERSION
        ),
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "event_id": event_id,
        "event_url": event_url,
        "event_date": event_date,
        "event_title": event_title,
        "forecast_issued_at_utc": forecast_issued_at_utc,
        "source_commit_sha": source_commit_sha,
        "parameter_artifact_sha256": artifact["artifact_sha256"],
        "parameter_input_sha256": artifact["input_sha256"],
        "parameter_as_of_utc": artifact["as_of_utc"],
        "model_trained_through": artifact["trained_through"],
        "bootstrap_members": int(artifact["bootstrap_members"]),
        "method_price_status": "unavailable_from_configured_provider",
        "production_influence": "none",
        "matchup_count": len(items),
        "forecast_matchup_count": len(used),
        "matchups": items,
    }
    body["publication_sha256"] = canonical_sha256(body)
    return validate_shadow_forecast_publication(body)


def validate_shadow_forecast_publication(publication: object) -> dict[str, object]:
    if not isinstance(publication, dict):
        raise ValueError("simulation shadow forecast must be an object")
    value = dict(publication)
    if value.get("schema_version") != SHADOW_FORECAST_SCHEMA_VERSION:
        raise ValueError("unsupported simulation shadow schema")
    if (
        value.get("candidate_only") is not True
        or value.get("paper_only") is not True
        or value.get("execution_enabled") is not False
    ):
        raise ValueError("simulation shadows must remain candidate paper research")
    if value.get("production_influence") != "none":
        raise ValueError("simulation shadows cannot influence production")
    event_id = _text(value.get("event_id"))
    if not event_id:
        raise ValueError("simulation shadow event ID is blank")
    issued = pd.to_datetime(value.get("forecast_issued_at_utc"), errors="coerce", utc=True)
    event_date = pd.to_datetime(value.get("event_date"), errors="coerce", utc=True)
    parameter_as_of = pd.to_datetime(value.get("parameter_as_of_utc"), errors="coerce", utc=True)
    trained_through = pd.to_datetime(value.get("model_trained_through"), errors="coerce", utc=True)
    if any(pd.isna(item) for item in (issued, event_date, parameter_as_of, trained_through)):
        raise ValueError("simulation shadow timing metadata is invalid")
    if not issued < event_date:
        raise ValueError("simulation shadow must be issued before the event")
    if parameter_as_of > issued:
        raise ValueError("simulation parameters cannot be fit after forecast issuance")
    if trained_through >= issued:
        raise ValueError("simulation training data must precede forecast issuance")
    if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", _text(value.get("source_commit_sha"))):
        raise ValueError("simulation shadow source revision is invalid")
    for field in ("parameter_artifact_sha256", "parameter_input_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", _text(value.get(field))):
            raise ValueError(f"{field} is not a SHA-256")
    supplied_hash = value.get("publication_sha256")
    unhashed = dict(value)
    unhashed.pop("publication_sha256", None)
    if supplied_hash != canonical_sha256(unhashed):
        raise ValueError("simulation shadow publication hash is invalid")
    matchups = value.get("matchups")
    if not isinstance(matchups, list) or len(matchups) != value.get("matchup_count"):
        raise ValueError("simulation shadow matchup count is invalid")
    seen: set[str] = set()
    resolved = 0
    for item in matchups:
        if not isinstance(item, dict):
            raise ValueError("simulation shadow matchup must be an object")
        matchup_id = item.get("matchup_id")
        if matchup_id is None:
            if item.get("forecast_status") != "abstain_unresolved_identity":
                raise ValueError("unresolved shadow matchup has invalid status")
            continue
        resolved += 1
        if not isinstance(matchup_id, str) or matchup_id in seen:
            raise ValueError("simulation shadow matchup IDs must be unique")
        seen.add(matchup_id)
        red_id = _text(item.get("red_fighter_id"))
        blue_id = _text(item.get("blue_fighter_id"))
        if matchup_id != matchup_id_for(event_id, red_id, blue_id):
            raise ValueError("simulation shadow matchup ID disagrees with fighter IDs")
        if item.get("forecast_status") != "candidate_simulation":
            raise ValueError("resolved shadow matchup has invalid status")
        _validate_aggregate(dict(item.get("aggregate") or {}), matchup_id)
    if resolved != value.get("forecast_matchup_count"):
        raise ValueError("simulation shadow forecast count is invalid")
    return value


def write_shadow_forecast_publication(
    path: str | Path, publication: Mapping[str, object]
) -> None:
    """Atomically write one validated publication JSON file."""

    validated = validate_shadow_forecast_publication(dict(publication))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            validated,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    encoded_bytes = encoded.encode("utf-8")
    if len(encoded_bytes) > MAX_SHADOW_PUBLICATION_BYTES:
        raise ValueError(
            "withholding simulation shadow publication larger than "
            f"{MAX_SHADOW_PUBLICATION_BYTES:,} bytes"
        )
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


def append_shadow_forecast_publication(
    directory: str | Path, publication: Mapping[str, object]
) -> Path:
    """Persist a content-addressed pre-event publication without overwriting.

    Retrying the identical publication is idempotent.  Every distinct issuance
    receives a distinct immutable file, which prevents a later card refresh
    from rewriting the probability record that existed before an event.
    """

    validated = validate_shadow_forecast_publication(dict(publication))
    event_id = _text(validated["event_id"])
    event_date = pd.to_datetime(validated["event_date"], utc=True).date().isoformat()
    digest = str(validated["publication_sha256"])
    destination = Path(directory) / f"{event_date}_{event_id}_{digest}.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        validate_shadow_forecast_publication(existing)
        if existing != validated:
            raise ValueError("content-addressed shadow path contains different data")
        return destination
    write_shadow_forecast_publication(destination, validated)
    return destination
