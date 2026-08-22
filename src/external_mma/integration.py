"""Convert canonical external results into state-only doubled replay rows."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fight_predictor.point_in_time import COUNT_STATS

from .identity import IDENTITY_COLUMNS, normalize_name
from .schema import ExternalBoutObservation, invert_result


def load_identity_map(path: str | Path) -> dict[tuple[str, str], str]:
    source_path = Path(path)
    if not source_path.exists() or source_path.stat().st_size == 0:
        return {}
    frame = pd.read_csv(source_path, dtype=object, keep_default_na=False)
    missing = set(IDENTITY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"identity map is missing columns: {sorted(missing)}")
    approved = frame[frame["status"].astype(str).str.casefold().eq("approved")]
    mapping: dict[tuple[str, str], str] = {}
    for row in approved.to_dict("records"):
        key = (str(row["source"]).strip(), str(row["source_fighter_id"]).strip())
        canonical = str(row["canonical_fighter_id"]).strip()
        if not all((*key, canonical)):
            raise ValueError("approved identity rows require source IDs and canonical ID")
        previous = mapping.setdefault(key, canonical)
        if previous != canonical:
            raise ValueError(f"conflicting approved identity mapping for {key}")
    return mapping


def load_approved_auxiliary(
    auxiliary_path: str | Path,
    policy_path: str | Path,
) -> pd.DataFrame | None:
    """Load auxiliary replay only after an explicit hash-pinned approval."""
    policy_file = Path(policy_path)
    if not policy_file.exists():
        return None
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("external MMA model policy has an unsupported schema")
    enabled = policy.get("enable_auxiliary_replay", False)
    if not isinstance(enabled, bool):
        raise ValueError("enable_auxiliary_replay must be a JSON boolean")
    if not enabled:
        return None
    path = Path(auxiliary_path)
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("external MMA replay is enabled but its CSV is missing")
    expected_hash = str(policy.get("approved_auxiliary_sha256", "")).strip().lower()
    actual_hash = sha256(path.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError(
            "external MMA auxiliary CSV differs from the explicitly approved hash"
        )
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError("external MMA replay is enabled but contains no bouts")
    if "emit_training_target" not in frame:
        raise ValueError("external MMA replay lacks its non-label safety flag")
    flags = frame["emit_training_target"].astype(str).str.casefold()
    if not flags.isin({"false", "0"}).all():
        raise ValueError("external MMA replay may not emit training targets")
    return frame


def _external_fighter_id(source: str, source_fighter_id: str) -> str:
    digest = sha256(f"{source}\0{source_fighter_id}".encode("utf-8")).hexdigest()[:24]
    return f"external_{digest}"


def _canonical_id(
    observation: ExternalBoutObservation,
    source_fighter_id: str,
    identity_map: dict[tuple[str, str], str],
) -> str:
    return identity_map.get(
        (observation.source, source_fighter_id),
        _external_fighter_id(observation.source, source_fighter_id),
    )


def is_ufc_promotion(value: object) -> bool:
    normalized = normalize_name(value)
    return normalized == "ufc" or "ultimate fighting championship" in normalized


def build_auxiliary_doubled(
    observations: list[ExternalBoutObservation],
    identity_map: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Build non-label replay rows, excluding UFC duplicates by construction."""
    identities = identity_map or {}
    physical: list[dict[str, object]] = []
    seen_fights: set[str] = set()
    for observation in observations:
        if is_ufc_promotion(observation.promotion):
            continue
        fighter_id = _canonical_id(observation, observation.fighter_source_id, identities)
        opponent_id = _canonical_id(observation, observation.opponent_source_id, identities)
        if fighter_id == opponent_id:
            raise ValueError(
                f"identity mapping collapsed bout {observation.observation_id}"
            )
        fight_id = f"external_{observation.observation_id[:32]}"
        if fight_id in seen_fights:
            raise ValueError(f"duplicate auxiliary fight ID {fight_id}")
        seen_fights.add(fight_id)
        event_digest = sha256(
            f"{observation.source}\0{observation.source_event_id}".encode("utf-8")
        ).hexdigest()[:32]
        physical.append(
            {
                "observation": observation,
                "date": observation.event_date,
                "fight_id": fight_id,
                "event_id": f"external_{event_digest}",
                "fighter_id": fighter_id,
                "opponent_id": opponent_id,
            }
        )
    physical.sort(
        key=lambda row: (
            row["date"], row["event_id"],
            row["observation"].source_bout_order is None,
            row["observation"].source_bout_order or 0,
            row["observation"].source_bout_id,
            row["fight_id"],
        )
    )
    for (_date, _event), indices in pd.Series(
        range(len(physical)),
        index=pd.MultiIndex.from_tuples(
            [(row["date"], row["event_id"]) for row in physical]
        ) if physical else pd.MultiIndex.from_arrays([[], []]),
    ).groupby(level=[0, 1]):
        ordered_indices = list(indices.to_numpy(dtype=int))
        for bout_order, index in enumerate(ordered_indices):
            physical[index]["bout_order"] = bout_order
            physical[index]["source_card_index"] = len(ordered_indices) - 1 - bout_order

    output: list[dict[str, object]] = []
    for item in physical:
        observation = item["observation"]
        first = {
            "date": item["date"],
            "fight_url": f"https://external-mma.invalid/fight-details/{item['fight_id']}",
            "event_url": f"https://external-mma.invalid/event-details/{item['event_id']}",
            "fighter_url": f"https://external-mma.invalid/fighter-details/{item['fighter_id']}",
            "opponent_url": f"https://external-mma.invalid/fighter-details/{item['opponent_id']}",
            "fighter": observation.fighter_name,
            "opponent": observation.opponent_name,
            "result": observation.result,
            "method": observation.method,
            "division": observation.division,
            "round": observation.finish_round,
            "time": "",
            "time_format": "",
            "total_fight_time": np.nan,
            "source_card_index": item["source_card_index"],
            "bout_order": item["bout_order"],
            "emit_training_target": False,
            "history_source": observation.source,
            **dict.fromkeys(COUNT_STATS, np.nan),
        }
        second = {
            **first,
            "fighter_url": first["opponent_url"],
            "opponent_url": first["fighter_url"],
            "fighter": first["opponent"],
            "opponent": first["fighter"],
            "result": invert_result(observation.result),
        }
        output.extend((first, second))
    columns = [
        "date", "fight_url", "event_url", "fighter_url", "opponent_url",
        "fighter", "opponent", "result", "method", "division", "round",
        "time", "time_format", "total_fight_time", "source_card_index",
        "bout_order", "emit_training_target", "history_source", *COUNT_STATS,
    ]
    return pd.DataFrame(output, columns=columns)
