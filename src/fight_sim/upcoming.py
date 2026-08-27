"""Candidate-only upcoming-card simulations and website publication.

This is deliberately separate from the production model forecast and from the
review-gated long-horizon shadow workflow.  It supports local research and a
compact, read-only website view without changing any betting decision.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping

import pandas as pd

from fight_semantics import stable_ufcstats_id
from market_tracker import matchup_id_for

from .domain import ENGINE_VERSION, RNG_CONTRACT_VERSION, SimulatorConfig
from .monte_carlo import run_adaptive_nested
from .parameters import (
    CausalParameterFitter,
    ParameterFitConfig,
    cache_materialized_parameter_artifact,
    canonical_json,
    canonical_sha256,
    load_parameter_artifact_cached,
    save_parameter_artifact,
)
from .publication import compact_shadow_aggregate
from .research import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_FIGHTER_PROFILES,
    DEFAULT_RAW_FIGHTS,
    DEFAULT_ROUND_STATS,
    atomic_write_json,
    build_specs,
    load_research_inputs,
)


UPCOMING_WEBSITE_SCHEMA_VERSION = 1
UPCOMING_WEBSITE_MODEL_VERSION = "candidate-fight-sim-card-v1"
UPCOMING_RUN_SCHEMA_VERSION = 1
UPCOMING_MATCHUP_RESULT_SCHEMA_VERSION = 1
UPCOMING_ADAPTIVE_WRAPPER_SCHEMA_VERSION = 1
DEFAULT_PARAMETER_CACHE = DEFAULT_ARTIFACT_ROOT / "parameter-materialized-cache"
AVAILABLE = "available"
WITHHELD_HISTORY = "withheld_insufficient_history"
WITHHELD_NONCONVERGED = "withheld_nonconverged"
UPCOMING_WEBSITE_OMITTED_AGGREGATE_FIELDS = (
    "bootstrap_outcome_counts",
    "statistic_distributions",
    "statistic_uncertainty",
    "uncertainty[].conditional_probabilities",
)


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).split())


def prior_ufc_exposure(raw: pd.DataFrame, cutoff: object) -> dict[str, int]:
    """Count distinct UFCStats bouts strictly before ``cutoff`` per fighter."""

    required = {"date", "fight_url", "fighter_url"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"raw fights are missing exposure fields: {missing}")
    frame = raw.loc[:, ["date", "fight_url", "fighter_url"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True)
    cutoff_utc = pd.Timestamp(cutoff)
    if cutoff_utc.tzinfo is None:
        cutoff_utc = cutoff_utc.tz_localize("UTC")
    else:
        cutoff_utc = cutoff_utc.tz_convert("UTC")
    frame = frame.loc[frame["date"].lt(cutoff_utc)]
    frame["fighter_id"] = frame["fighter_url"].map(stable_ufcstats_id)
    frame["fight_id"] = frame["fight_url"].map(stable_ufcstats_id)
    frame = frame.loc[frame["fighter_id"].ne("") & frame["fight_id"].ne("")]
    return {
        str(fighter_id): int(count)
        for fighter_id, count in frame.groupby("fighter_id", sort=True)[
            "fight_id"
        ].nunique().items()
    }


def _normalize_card_matchups(
    publication: Mapping[str, object], event_id: str
) -> list[dict[str, object]]:
    if _text(publication.get("event_id")) != event_id:
        raise ValueError("card and outcome forecast event IDs disagree")
    raw_matchups = publication.get("matchups")
    if not isinstance(raw_matchups, list) or not raw_matchups:
        raise ValueError("outcome forecast contains no card matchups")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_matchups):
        if not isinstance(value, Mapping):
            raise ValueError("outcome forecast matchup must be an object")
        row = dict(value)
        fighter_id = stable_ufcstats_id(row.get("fighter_id"))
        opponent_id = stable_ufcstats_id(row.get("opponent_id"))
        if not fighter_id or not opponent_id or fighter_id == opponent_id:
            raise ValueError("upcoming matchup requires two distinct stable fighter IDs")
        expected = matchup_id_for(event_id, fighter_id, opponent_id)
        matchup_id = _text(row.get("matchup_id")) or expected
        if matchup_id != expected:
            raise ValueError("upcoming matchup ID disagrees with stable fighter IDs")
        if matchup_id in seen:
            raise ValueError("upcoming matchup IDs must be unique")
        seen.add(matchup_id)
        scheduled_rounds = int(row.get("scheduled_rounds") or 0)
        if scheduled_rounds not in (3, 5):
            raise ValueError("upcoming matchup must schedule three or five rounds")
        normalized.append(
            {
                "bout_order": int(row.get("bout_order", index)),
                "matchup_id": matchup_id,
                "fighter_id": fighter_id,
                "fighter_name": _text(row.get("fighter_name")) or fighter_id,
                "opponent_id": opponent_id,
                "opponent_name": _text(row.get("opponent_name")) or opponent_id,
                "division": _text(row.get("division")) or "Unknown",
                "scheduled_rounds": scheduled_rounds,
            }
        )
    return sorted(normalized, key=lambda item: (int(item["bout_order"]), str(item["matchup_id"])))


def validate_website_simulation_publication(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("website simulation publication must be an object")
    publication = dict(value)
    if publication.get("schema_version") != UPCOMING_WEBSITE_SCHEMA_VERSION:
        raise ValueError("unsupported website simulation schema")
    if (
        publication.get("candidate_only") is not True
        or publication.get("paper_only") is not True
        or publication.get("execution_enabled") is not False
        or publication.get("production_influence") != "none"
    ):
        raise ValueError("website simulations must remain candidate-only research")
    matchups = publication.get("matchups")
    if not isinstance(matchups, list) or not matchups:
        raise ValueError("website simulation publication has no matchups")
    if int(publication.get("matchup_count") or 0) != len(matchups):
        raise ValueError("website simulation matchup count is invalid")
    available = 0
    excluded = 0
    seen: set[str] = set()
    for item in matchups:
        if not isinstance(item, dict):
            raise ValueError("website simulation matchup must be an object")
        matchup_id = _text(item.get("matchup_id"))
        if not matchup_id or matchup_id in seen:
            raise ValueError("website simulation matchup IDs must be unique")
        seen.add(matchup_id)
        status = item.get("status")
        if status == AVAILABLE:
            compact = compact_website_aggregate(item.get("aggregate"))
            if compact != item.get("aggregate"):
                raise ValueError("website aggregate is not in compact authority-linked form")
            available += 1
        elif status in {WITHHELD_HISTORY, WITHHELD_NONCONVERGED}:
            if "aggregate" in item:
                raise ValueError("withheld website matchup cannot contain an aggregate")
            if not _text(item.get("unavailable_reason")):
                raise ValueError("withheld website matchup requires a reason")
            excluded += 1
        else:
            raise ValueError("website simulation matchup has invalid status")
    if available != int(publication.get("available_matchups") or 0):
        raise ValueError("website available matchup count is invalid")
    if excluded != int(publication.get("excluded_matchups") or 0):
        raise ValueError("website excluded matchup count is invalid")
    supplied = _text(publication.get("publication_sha256"))
    unhashed = dict(publication)
    unhashed.pop("publication_sha256", None)
    if supplied != canonical_sha256(unhashed):
        raise ValueError("website simulation publication hash is invalid")
    return publication


def compact_website_aggregate(value: object) -> dict[str, object]:
    """Strip local research detail that the read-only website never consumes."""

    # Hash the JSON representation that is actually durable. Numeric mapping
    # keys otherwise sort differently before and after a JSON round trip.
    normalized = json.loads(canonical_json(value))
    aggregate = compact_shadow_aggregate(normalized)
    for field in UPCOMING_WEBSITE_OMITTED_AGGREGATE_FIELDS:
        if "[]" not in field:
            aggregate.pop(field, None)
    uncertainty = []
    for original in list(aggregate.get("uncertainty") or []):
        item = dict(original)
        item.pop("conditional_probabilities", None)
        uncertainty.append(item)
    aggregate["uncertainty"] = uncertainty
    aggregate["website_omitted_fields"] = list(
        UPCOMING_WEBSITE_OMITTED_AGGREGATE_FIELDS
    )
    return aggregate


def _write_authority(path: Path, aggregate: Mapping[str, object]) -> Path:
    payload = json.loads(canonical_json(dict(aggregate)))
    digest = canonical_sha256(payload)
    destination = path / f"{payload['matchup_id']}-{digest}.json.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = json.loads(gzip.decompress(destination.read_bytes()))
        if canonical_sha256(existing) != digest:
            raise ValueError("local upcoming authority contains different data")
        return destination
    encoded = gzip.compress(
        (canonical_json(payload) + "\n").encode("utf-8"),
        compresslevel=9,
        mtime=0,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
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
    return destination


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_gzip_json(path: str | Path, value: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = gzip.compress(
        (canonical_json(dict(value)) + "\n").encode("utf-8"),
        compresslevel=6,
        mtime=0,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
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
    return destination


def _read_gzip_json(path: str | Path) -> dict[str, object]:
    value = json.loads(gzip.decompress(Path(path).read_bytes()))
    if not isinstance(value, dict):
        raise ValueError(f"compressed JSON must contain an object: {path}")
    return value


def _self_hashed(value: Mapping[str, object], field: str) -> dict[str, object]:
    payload = dict(value)
    payload[field] = canonical_sha256(payload)
    return payload


def _validate_self_hash(
    value: object, *, field: str, label: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    payload = dict(value)
    supplied = payload.pop(field, None)
    if supplied != canonical_sha256(payload):
        raise ValueError(f"{label} hash is invalid")
    payload[field] = supplied
    return payload


def _load_run_manifest(path: Path) -> dict[str, object]:
    manifest = _validate_self_hash(
        json.loads(path.read_text(encoding="utf-8")),
        field="manifest_sha256",
        label="upcoming run manifest",
    )
    if manifest.get("schema_version") != UPCOMING_RUN_SCHEMA_VERSION:
        raise ValueError("unsupported upcoming run manifest schema")
    contract = manifest.get("run_contract")
    if not isinstance(contract, dict):
        raise ValueError("upcoming run manifest is missing its contract")
    if manifest.get("run_contract_sha256") != canonical_sha256(contract):
        raise ValueError("upcoming run contract hash is invalid")
    return manifest


def _matchup_result_path(destination: Path, matchup_id: object) -> Path:
    return destination / "matchup-results" / f"{matchup_id}.json"


def _write_matchup_result(
    destination: Path,
    *,
    run_contract_sha256: str,
    item: Mapping[str, object],
) -> Path:
    payload = _self_hashed(
        {
            "schema_version": UPCOMING_MATCHUP_RESULT_SCHEMA_VERSION,
            "run_contract_sha256": run_contract_sha256,
            "matchup_id": str(item["matchup_id"]),
            "item": dict(item),
        },
        "result_sha256",
    )
    return atomic_write_json(
        _matchup_result_path(destination, item["matchup_id"]), payload
    )


def _load_matchup_result(
    path: Path,
    *,
    run_contract_sha256: str,
    matchup_id: str,
) -> dict[str, object]:
    result = _validate_self_hash(
        json.loads(path.read_text(encoding="utf-8")),
        field="result_sha256",
        label="upcoming matchup result",
    )
    if result.get("schema_version") != UPCOMING_MATCHUP_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported upcoming matchup result schema")
    if result.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("upcoming matchup result belongs to another run")
    if result.get("matchup_id") != matchup_id or not isinstance(result.get("item"), dict):
        raise ValueError("upcoming matchup result identity is invalid")
    item = dict(result["item"])
    if item.get("matchup_id") != matchup_id:
        raise ValueError("upcoming matchup result item identity is invalid")
    status = item.get("status")
    if status == AVAILABLE:
        compact = compact_website_aggregate(item.get("aggregate"))
        if compact != item.get("aggregate"):
            raise ValueError("completed upcoming matchup aggregate is invalid")
    elif status not in {WITHHELD_HISTORY, WITHHELD_NONCONVERGED}:
        raise ValueError("completed upcoming matchup status is invalid")
    return item


def _reconcile_completed_authority(
    destination: Path,
    *,
    run_contract_sha256: str,
    item: Mapping[str, object],
) -> dict[str, object]:
    """Normalize legacy pre-JSON hashes without rerunning completed paths."""

    completed = dict(item)
    if completed.get("status") == WITHHELD_HISTORY:
        return completed
    matchup_id = str(completed["matchup_id"])
    diagnostics_path = destination / "convergence-diagnostics" / f"{matchup_id}.json"
    if not diagnostics_path.is_file():
        raise ValueError("completed simulated matchup is missing convergence diagnostics")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if not isinstance(diagnostics, dict) or diagnostics.get("matchup_id") != matchup_id:
        raise ValueError("completed matchup convergence diagnostics are invalid")
    old_authority = Path(str(diagnostics.get("aggregate_authority") or ""))
    if not old_authority.is_file():
        raise ValueError("completed matchup aggregate authority is missing")
    full = json.loads(gzip.decompress(old_authority.read_bytes()))
    if not isinstance(full, dict) or full.get("matchup_id") != matchup_id:
        raise ValueError("completed matchup aggregate authority identity is invalid")
    authority_root = destination / "aggregate-authority"
    normalized_authority = _write_authority(authority_root, full)
    diagnostics["aggregate_authority"] = str(normalized_authority)
    atomic_write_json(diagnostics_path, diagnostics)
    if completed.get("status") == AVAILABLE:
        completed["aggregate"] = compact_website_aggregate(full)
    _write_matchup_result(
        destination,
        run_contract_sha256=run_contract_sha256,
        item=completed,
    )
    if old_authority.resolve() != normalized_authority.resolve():
        if old_authority.resolve().parent != authority_root.resolve():
            raise ValueError("refusing to remove authority outside the upcoming run")
        old_authority.unlink()
    return completed


def _adaptive_checkpoint_path(destination: Path, matchup_id: object) -> Path:
    return destination / "adaptive-checkpoints" / f"{matchup_id}.json.gz"


def _write_adaptive_checkpoint(
    path: Path,
    *,
    run_contract_sha256: str,
    matchup_id: str,
    adaptive_checkpoint: Mapping[str, object],
) -> Path:
    wrapper = _self_hashed(
        {
            "schema_version": UPCOMING_ADAPTIVE_WRAPPER_SCHEMA_VERSION,
            "run_contract_sha256": run_contract_sha256,
            "matchup_id": matchup_id,
            "adaptive_checkpoint": dict(adaptive_checkpoint),
        },
        "wrapper_sha256",
    )
    return _atomic_write_gzip_json(path, wrapper)


def _load_adaptive_checkpoint(
    path: Path,
    *,
    run_contract_sha256: str,
    matchup_id: str,
) -> dict[str, object]:
    wrapper = _validate_self_hash(
        _read_gzip_json(path),
        field="wrapper_sha256",
        label="upcoming adaptive checkpoint",
    )
    if wrapper.get("schema_version") != UPCOMING_ADAPTIVE_WRAPPER_SCHEMA_VERSION:
        raise ValueError("unsupported upcoming adaptive checkpoint schema")
    if wrapper.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("upcoming adaptive checkpoint belongs to another run")
    if wrapper.get("matchup_id") != matchup_id:
        raise ValueError("upcoming adaptive checkpoint identity is invalid")
    adaptive = wrapper.get("adaptive_checkpoint")
    if not isinstance(adaptive, dict):
        raise ValueError("upcoming adaptive checkpoint payload is invalid")
    return dict(adaptive)


def execute_upcoming_card(
    *,
    card_path: str | Path,
    outcome_path: str | Path,
    output_dir: str | Path,
    website_output: str | Path,
    raw_path: str | Path = DEFAULT_RAW_FIGHTS,
    profiles_path: str | Path = DEFAULT_FIGHTER_PROFILES,
    round_path: str | Path = DEFAULT_ROUND_STATS,
    minimum_prior_ufc_fights: int = 3,
    bootstrap_members: int = 200,
    initial_paths_per_member: int = 512,
    max_paths_per_member: int = 2048,
    random_seed: int = 81173,
    workers: int = 1,
    chunk_size: int = 64,
    simulator_config: SimulatorConfig | None = None,
    parameter_artifact_path: str | Path | None = None,
    parameter_cache_dir: str | Path = DEFAULT_PARAMETER_CACHE,
    issued_at_utc: object | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Fit once and publish completed/withheld states for one upcoming card."""

    if minimum_prior_ufc_fights < 0:
        raise ValueError("minimum prior UFC fights must be nonnegative")
    if bootstrap_members <= 0:
        raise ValueError("bootstrap members must be positive")
    if initial_paths_per_member <= 0 or max_paths_per_member < initial_paths_per_member:
        raise ValueError("upcoming path bounds are invalid")
    destination = Path(output_dir)
    manifest_path = destination / "run-manifest.json"
    destination_has_files = destination.exists() and any(destination.iterdir())
    existing_manifest: dict[str, object] | None = None
    if destination_has_files and not resume:
        raise ValueError(
            f"upcoming output directory is not empty: {destination}; choose a new "
            "directory or pass --resume"
        )
    if resume:
        if not manifest_path.is_file():
            raise ValueError("upcoming-card --resume requires a valid run-manifest.json")
        existing_manifest = _load_run_manifest(manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    card = json.loads(Path(card_path).read_text(encoding="utf-8"))
    outcomes = json.loads(Path(outcome_path).read_text(encoding="utf-8"))
    if not isinstance(card, dict) or not isinstance(outcomes, dict):
        raise ValueError("card and outcome inputs must be JSON objects")
    event_id = stable_ufcstats_id(card.get("event_id") or card.get("event_url"))
    if not event_id:
        raise ValueError("upcoming card event ID is blank")
    event_date = pd.to_datetime(card.get("date"), errors="raise", utc=True)
    if existing_manifest is not None:
        existing_contract = dict(existing_manifest["run_contract"])
        manifest_issued = existing_contract.get("forecast_issued_at_utc")
        if issued_at_utc is not None:
            requested = pd.Timestamp(issued_at_utc)
            requested = (
                requested.tz_localize("UTC")
                if requested.tzinfo is None
                else requested.tz_convert("UTC")
            )
            if requested.isoformat() != manifest_issued:
                raise ValueError("resume issue time differs from the existing run")
        issued = pd.Timestamp(manifest_issued)
    else:
        issued = pd.Timestamp(issued_at_utc or datetime.now(timezone.utc))
    issued = issued.tz_localize("UTC") if issued.tzinfo is None else issued.tz_convert("UTC")
    if not issued < event_date:
        raise ValueError("upcoming simulation must be issued before the event date")
    matchups = _normalize_card_matchups(outcomes, event_id)
    raw, profiles, rounds = load_research_inputs(raw_path, profiles_path, round_path)
    exposure = prior_ufc_exposure(raw, issued)
    base = simulator_config or SimulatorConfig()
    mechanics_profile_id = f"mechanics-{canonical_sha256(base.to_dict())[:12]}"
    fitter = CausalParameterFitter(raw, profiles, rounds)
    if parameter_artifact_path is None:
        if progress:
            progress(
                f"Fitting {bootstrap_members} pre-event bootstrap members for {len(matchups)} card matchups."
            )
        artifact = fitter.fit(
            issued,
            config=ParameterFitConfig(
                bootstrap_members=bootstrap_members,
                random_seed=random_seed,
            ),
            created_at_utc=issued,
        )
        cache_materialized_parameter_artifact(artifact, parameter_cache_dir)
    else:
        artifact, parameter_cache_hit, _ = load_parameter_artifact_cached(
            parameter_artifact_path, parameter_cache_dir
        )
        artifact.validate()
        if len(artifact.members) != bootstrap_members:
            raise ValueError("reused parameter artifact bootstrap member count disagrees")
        artifact_cutoff = pd.to_datetime(artifact.as_of_utc, errors="raise", utc=True)
        if not artifact_cutoff < event_date or artifact_cutoff > issued:
            raise ValueError("reused parameter artifact cutoff is invalid for this card")
        raw_dates = pd.to_datetime(raw["date"], errors="raise", utc=True)
        if raw_dates.ge(artifact_cutoff).where(raw_dates.lt(issued), False).any():
            raise ValueError(
                "completed fights were added after the reused parameter artifact cutoff"
            )
        if progress:
            progress(
                f"Reusing validated {bootstrap_members}-member pre-event artifact "
                f"({'materialized cache' if parameter_cache_hit else 'newly materialized'}) for "
                f"{len(matchups)} card matchups."
            )
    run_contract: dict[str, object] = {
        "schema_version": UPCOMING_RUN_SCHEMA_VERSION,
        "event_id": event_id,
        "event_date": event_date.isoformat(),
        "forecast_issued_at_utc": issued.isoformat(),
        "card_sha256": _file_sha256(card_path),
        "outcomes_sha256": _file_sha256(outcome_path),
        "raw_sha256": _file_sha256(raw_path),
        "profiles_sha256": _file_sha256(profiles_path),
        "round_stats_sha256": (
            _file_sha256(round_path) if Path(round_path).is_file() else None
        ),
        "matchups": matchups,
        "minimum_prior_ufc_fights": minimum_prior_ufc_fights,
        "bootstrap_members": bootstrap_members,
        "initial_paths_per_member": initial_paths_per_member,
        "max_paths_per_member": max_paths_per_member,
        "random_seed": random_seed,
        "mechanics_profile_id": mechanics_profile_id,
        "simulator_config": base.to_dict(),
        "parameter_artifact_sha256": artifact.artifact_sha256,
        "parameter_input_sha256": artifact.input_sha256,
        "engine_version": ENGINE_VERSION,
        "rng_contract": RNG_CONTRACT_VERSION,
    }
    run_contract_sha256 = canonical_sha256(run_contract)
    if existing_manifest is not None:
        if (
            existing_manifest.get("run_contract_sha256") != run_contract_sha256
            or existing_manifest.get("run_contract") != run_contract
        ):
            raise ValueError(
                "resume contract differs from the existing upcoming-card run"
            )
    else:
        manifest = _self_hashed(
            {
                "schema_version": UPCOMING_RUN_SCHEMA_VERSION,
                "run_contract_sha256": run_contract_sha256,
                "run_contract": run_contract,
            },
            "manifest_sha256",
        )
        atomic_write_json(manifest_path, manifest)
    parameter_model_path = destination / "parameter_model.json.gz"
    if not parameter_model_path.is_file():
        save_parameter_artifact(parameter_model_path, artifact)
    output_matchups: list[dict[str, object]] = []
    authority_root = destination / "aggregate-authority"
    for position, matchup in enumerate(matchups, start=1):
        matchup_id = str(matchup["matchup_id"])
        completed_path = _matchup_result_path(destination, matchup_id)
        if completed_path.is_file():
            completed = _load_matchup_result(
                completed_path,
                run_contract_sha256=run_contract_sha256,
                matchup_id=matchup_id,
            )
            completed = _reconcile_completed_authority(
                destination,
                run_contract_sha256=run_contract_sha256,
                item=completed,
            )
            output_matchups.append(completed)
            if progress:
                progress(
                    f"Resumed {position}/{len(matchups)}: "
                    f"{completed['fighter_name']} vs {completed['opponent_name']} "
                    f"({completed['status']})."
                )
            continue
        item = dict(matchup)
        fighter_history = int(exposure.get(str(matchup["fighter_id"]), 0))
        opponent_history = int(exposure.get(str(matchup["opponent_id"]), 0))
        item["fighter_prior_ufc_fights"] = fighter_history
        item["opponent_prior_ufc_fights"] = opponent_history
        if min(fighter_history, opponent_history) < minimum_prior_ufc_fights:
            item["status"] = WITHHELD_HISTORY
            item["unavailable_reason"] = (
                "Simulation withheld: both fighters need at least "
                f"{minimum_prior_ufc_fights} prior UFCStats bouts; observed "
                f"{fighter_history} and {opponent_history}."
            )
            output_matchups.append(item)
            _write_matchup_result(
                destination,
                run_contract_sha256=run_contract_sha256,
                item=item,
            )
            if progress:
                progress(
                    f"Withheld {position}/{len(matchups)}: {item['fighter_name']} vs "
                    f"{item['opponent_name']} ({fighter_history}/{opponent_history} prior bouts)."
                )
            continue
        root_seed = (
            f"upcoming:{event_id}:{item['matchup_id']}:{artifact.artifact_sha256}:"
            f"{mechanics_profile_id}"
        )
        specs = build_specs(
            fitter,
            artifact,
            red_fighter_id=str(item["fighter_id"]),
            blue_fighter_id=str(item["opponent_id"]),
            division=str(item["division"]),
            scheduled_rounds=int(item["scheduled_rounds"]),
            event_id=event_id,
            matchup_id=str(item["matchup_id"]),
            root_seed=root_seed,
            simulator_base=base,
            _artifact_validated=True,
        )
        adaptive_path = _adaptive_checkpoint_path(destination, matchup_id)
        adaptive_checkpoint = (
            _load_adaptive_checkpoint(
                adaptive_path,
                run_contract_sha256=run_contract_sha256,
                matchup_id=matchup_id,
            )
            if adaptive_path.is_file()
            else None
        )
        if adaptive_checkpoint is not None and progress:
            progress(
                f"Continuing {position}/{len(matchups)}: {item['fighter_name']} vs "
                f"{item['opponent_name']} from "
                f"{adaptive_checkpoint.get('paths_per_member', '?')} paths/member."
            )
        result = run_adaptive_nested(
            specs,
            initial_paths_per_member=initial_paths_per_member,
            max_paths_per_member=max_paths_per_member,
            workers=workers,
            chunk_size=chunk_size,
            max_traces=0,
            retain_paths=False,
            resume_checkpoint=adaptive_checkpoint,
            checkpoint_callback=lambda checkpoint, path=adaptive_path, identity=matchup_id: (
                _write_adaptive_checkpoint(
                    path,
                    run_contract_sha256=run_contract_sha256,
                    matchup_id=identity,
                    adaptive_checkpoint=checkpoint,
                )
            ),
        )
        full = result.forecast.to_dict()
        authority_path = _write_authority(authority_root, full)
        atomic_write_json(
            destination / "convergence-diagnostics" / f"{item['matchup_id']}.json",
            {
                "matchup_id": item["matchup_id"],
                "fighter_name": item["fighter_name"],
                "opponent_name": item["opponent_name"],
                "bootstrap_members": bootstrap_members,
                "paths_per_bootstrap_member": (
                    int(result.forecast.total_paths) // bootstrap_members
                ),
                "total_paths": int(result.forecast.total_paths),
                "converged": result.converged,
                "invariant_failure_count": len(result.invariant_failures),
                "aggregate_authority": str(authority_path),
                "history": [
                    {**asdict(diagnostic), "converged": diagnostic.converged}
                    for diagnostic in result.convergence
                ],
            },
        )
        if not result.converged:
            item["status"] = WITHHELD_NONCONVERGED
            item["unavailable_reason"] = (
                "Simulation withheld because the maximum run did not meet the "
                "predeclared Monte Carlo convergence checks."
            )
            item["convergence_batches"] = len(result.convergence)
        else:
            item["status"] = AVAILABLE
            item["aggregate"] = compact_website_aggregate(full)
            item["convergence_batches"] = len(result.convergence)
            item["paths_per_bootstrap_member"] = (
                int(result.forecast.total_paths) // bootstrap_members
            )
        output_matchups.append(item)
        _write_matchup_result(
            destination,
            run_contract_sha256=run_contract_sha256,
            item=item,
        )
        if adaptive_path.is_file():
            adaptive_path.unlink()
        if progress:
            progress(
                f"Completed {position}/{len(matchups)}: {item['fighter_name']} vs "
                f"{item['opponent_name']} ({item['status']})."
            )
    available = sum(item["status"] == AVAILABLE for item in output_matchups)
    publication: dict[str, object] = {
        "schema_version": UPCOMING_WEBSITE_SCHEMA_VERSION,
        "model_version": UPCOMING_WEBSITE_MODEL_VERSION,
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "production_influence": "none",
        "event_id": event_id,
        "event_url": _text(card.get("event_url")),
        "event_date": event_date.isoformat(),
        "event_title": _text(card.get("title")),
        "forecast_issued_at_utc": issued.isoformat(),
        "minimum_prior_ufc_fights": minimum_prior_ufc_fights,
        "mechanics_profile_id": mechanics_profile_id,
        "simulator_config": base.to_dict(),
        "parameter_artifact_sha256": artifact.artifact_sha256,
        "parameter_input_sha256": artifact.input_sha256,
        "bootstrap_members": bootstrap_members,
        "matchup_count": len(output_matchups),
        "available_matchups": available,
        "excluded_matchups": len(output_matchups) - available,
        "matchups": output_matchups,
        "coverage_warnings": [
            "candidate_only_simulation_does_not_authorize_a_wager",
            "method_and_round_ev_unavailable_without_real_synchronized_prop_prices",
            "official_ufcstats_control_is_broader_than_simulated_ground_top_control",
            "fighter_uncertainty_is_large_near_the_minimum_history_threshold",
        ],
    }
    publication["publication_sha256"] = canonical_sha256(publication)
    validated = validate_website_simulation_publication(publication)
    website_path = atomic_write_json(website_output, validated)
    atomic_write_json(destination / "website-publication.json", validated)
    return website_path, validated
