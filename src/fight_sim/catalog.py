"""Incremental, paper-only simulations for every announced UFC matchup.

The scheduled data updater discovers announced fights.  This module compares
that publication with immutable per-matchup simulation records, runs only the
new eligible matchups, and rebuilds the compact website view.  A completed
fight simulation is never partially published.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable, Mapping

import pandas as pd

from fight_semantics import stable_ufcstats_id, upcoming_schedule
from upcoming_bet_board import validate_upcoming_forecast_publication

from .domain import ENGINE_VERSION, RNG_CONTRACT_VERSION, SimulatorConfig
from .monte_carlo import run_nested
from .parameters import CausalParameterFitter, ParameterFitConfig, canonical_sha256
from .research import (
    DEFAULT_FIGHTER_PROFILES,
    DEFAULT_RAW_FIGHTS,
    DEFAULT_ROUND_STATS,
    atomic_write_json,
    build_specs,
    load_research_inputs,
)
from .upcoming import compact_website_aggregate, prior_ufc_exposure


AUTOMATIC_RECORD_SCHEMA_VERSION = 1
AUTOMATIC_RECORD_VERSION = "automatic-upcoming-simulation-v1"
AUTOMATIC_WEBSITE_SCHEMA_VERSION = 2
AUTOMATIC_WEBSITE_VERSION = "candidate-fight-sim-catalog-v1"
DEFAULT_BOOTSTRAP_MEMBERS = 64
DEFAULT_PATHS_PER_MEMBER = 64
DEFAULT_MINIMUM_PRIOR_UFC_FIGHTS = 3
DEFAULT_CATALOG_PATH = Path("src/content/data/external/all_upcoming_forecasts.json")
DEFAULT_RECORD_DIRECTORY = Path("src/content/data/simulation/upcoming_matchups")
DEFAULT_WEBSITE_OUTPUT = Path("src/content/data/external/simulation_forecasts.json")

AVAILABLE = "available"
WITHHELD_HISTORY = "withheld_insufficient_history"
WITHHELD_IDENTITY = "withheld_unresolved_identity"
PENDING_NEW = "pending_new"
PENDING_RETRY = "pending_retry"


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).split())


def _utc(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


def _record_without_hash(record: Mapping[str, object]) -> dict[str, object]:
    result = dict(record)
    result.pop("record_sha256", None)
    return result


def _simulation_input(
    matchup: Mapping[str, object],
    *,
    simulator_config: SimulatorConfig,
    mechanics_profile_id: str,
    bootstrap_members: int,
    paths_per_member: int,
) -> dict[str, object] | None:
    event_id = stable_ufcstats_id(matchup.get("event_id") or matchup.get("event_url"))
    matchup_id = _text(matchup.get("matchup_id"))
    fighter_id = stable_ufcstats_id(matchup.get("fighter_id"))
    opponent_id = stable_ufcstats_id(matchup.get("opponent_id"))
    if not event_id or not matchup_id or not fighter_id or not opponent_id:
        return None
    if fighter_id == opponent_id:
        return None
    bout_order = int(matchup.get("bout_order") or 0)
    scheduled_rounds, schedule_basis = upcoming_schedule(
        bout_order, matchup.get("division")
    )
    base = {
        "event_id": event_id,
        "matchup_id": matchup_id,
        "fighter_id": fighter_id,
        "opponent_id": opponent_id,
        "division": _text(matchup.get("division")) or "Unknown",
        "scheduled_rounds": scheduled_rounds,
        "schedule_basis": schedule_basis,
        "bootstrap_members": int(bootstrap_members),
        "paths_per_bootstrap_member": int(paths_per_member),
        "engine_version": ENGINE_VERSION,
        "rng_contract": RNG_CONTRACT_VERSION,
        "mechanics_profile_id": mechanics_profile_id,
        "simulator_config": simulator_config.to_dict(),
    }
    base["simulation_input_sha256"] = canonical_sha256(base)
    return base


def validate_automatic_record(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("automatic simulation record must be an object")
    record = dict(value)
    supplied = _text(record.get("record_sha256"))
    if supplied != canonical_sha256(_record_without_hash(record)):
        raise ValueError("automatic simulation record hash is invalid")
    if (
        record.get("schema_version") != AUTOMATIC_RECORD_SCHEMA_VERSION
        or record.get("record_version") != AUTOMATIC_RECORD_VERSION
        or record.get("candidate_only") is not True
        or record.get("paper_only") is not True
        or record.get("execution_enabled") is not False
        or record.get("production_influence") != "none"
        or record.get("status") != AVAILABLE
    ):
        raise ValueError("automatic simulation record policy is invalid")
    required_text = (
        "simulation_input_sha256",
        "matchup_id",
        "event_id",
        "fighter_id",
        "opponent_id",
        "forecast_issued_at_utc",
        "parameter_artifact_sha256",
        "parameter_input_sha256",
        "mechanics_profile_id",
        "source_upcoming_publication_sha256",
    )
    if any(not _text(record.get(field)) for field in required_text):
        raise ValueError("automatic simulation record identity is incomplete")
    _utc(record["forecast_issued_at_utc"])
    if int(record.get("bootstrap_members") or 0) <= 0:
        raise ValueError("automatic simulation record bootstrap count is invalid")
    if int(record.get("paths_per_bootstrap_member") or 0) <= 0:
        raise ValueError("automatic simulation record path count is invalid")
    aggregate = compact_website_aggregate(record.get("aggregate"))
    if aggregate != record.get("aggregate"):
        raise ValueError("automatic simulation aggregate is not compact")
    if _text(aggregate.get("matchup_id")) != _text(record.get("matchup_id")):
        raise ValueError("automatic simulation aggregate belongs to another matchup")
    expected_paths = int(record["bootstrap_members"]) * int(
        record["paths_per_bootstrap_member"]
    )
    if int(aggregate.get("total_paths") or 0) != expected_paths:
        raise ValueError("automatic simulation aggregate path count is inconsistent")
    return record


def load_automatic_records(directory: str | Path) -> dict[str, dict[str, object]]:
    root = Path(directory)
    records: dict[str, dict[str, object]] = {}
    if not root.exists():
        return records
    for path in sorted(root.glob("*.json")):
        record = validate_automatic_record(
            json.loads(path.read_text(encoding="utf-8"))
        )
        identity = str(record["simulation_input_sha256"])
        if identity in records:
            raise ValueError(f"duplicate automatic simulation input: {identity}")
        records[identity] = record
    return records


def _write_automatic_record(
    directory: str | Path, record: Mapping[str, object]
) -> Path:
    validated = validate_automatic_record(dict(record))
    destination = Path(directory) / f"{validated['simulation_input_sha256']}.json"
    if destination.exists():
        existing = validate_automatic_record(
            json.loads(destination.read_text(encoding="utf-8"))
        )
        if existing != validated:
            raise ValueError("immutable automatic simulation record already differs")
        return destination
    return atomic_write_json(destination, validated)


def _catalog_rows(
    catalog: Mapping[str, object],
    *,
    simulator_config: SimulatorConfig,
    mechanics_profile_id: str,
    bootstrap_members: int,
    paths_per_member: int,
) -> list[dict[str, object]]:
    validated = validate_upcoming_forecast_publication(dict(catalog))
    rows: list[dict[str, object]] = []
    event_counts = {
        str(event["event_id"]): int(event["matchup_count"])
        for event in validated["events"]
    }
    for matchup in validated["matchups"]:
        row = dict(matchup)
        row["event_matchup_count"] = event_counts[str(row["event_id"])]
        simulation_input = _simulation_input(
            row,
            simulator_config=simulator_config,
            mechanics_profile_id=mechanics_profile_id,
            bootstrap_members=bootstrap_members,
            paths_per_member=paths_per_member,
        )
        row["simulation_input"] = simulation_input
        if simulation_input is not None:
            row["scheduled_rounds"] = simulation_input["scheduled_rounds"]
            row["schedule_basis"] = simulation_input["schedule_basis"]
            row["simulation_input_sha256"] = simulation_input[
                "simulation_input_sha256"
            ]
        rows.append(row)
    return rows


def _website_matchups(
    rows: list[dict[str, object]],
    records: Mapping[str, Mapping[str, object]],
    exposure: Mapping[str, int],
    *,
    minimum_prior_ufc_fights: int,
    failures: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    failures = failures or {}
    output: list[dict[str, object]] = []
    for row in rows:
        public = {
            field: row.get(field)
            for field in (
                "event_id",
                "event_url",
                "event_title",
                "event_date",
                "event_matchup_count",
                "bout_order",
                "matchup_id",
                "fighter_id",
                "fighter_name",
                "opponent_id",
                "opponent_name",
                "division",
                "scheduled_rounds",
                "schedule_basis",
                "simulation_input_sha256",
            )
            if row.get(field) is not None
        }
        simulation_input = row.get("simulation_input")
        if not isinstance(simulation_input, Mapping):
            public["status"] = WITHHELD_IDENTITY
            public["unavailable_reason"] = (
                "Simulation withheld because both stable UFCStats fighter identities "
                "are not available yet."
            )
            output.append(public)
            continue
        fighter_history = int(exposure.get(str(row.get("fighter_id")), 0))
        opponent_history = int(exposure.get(str(row.get("opponent_id")), 0))
        public["fighter_prior_ufc_fights"] = fighter_history
        public["opponent_prior_ufc_fights"] = opponent_history
        if min(fighter_history, opponent_history) < minimum_prior_ufc_fights:
            public["status"] = WITHHELD_HISTORY
            public["unavailable_reason"] = (
                "Simulation withheld: both fighters need at least "
                f"{minimum_prior_ufc_fights} prior UFCStats fights; observed "
                f"{fighter_history} and {opponent_history}."
            )
            output.append(public)
            continue
        identity = str(simulation_input["simulation_input_sha256"])
        record = records.get(identity)
        if record is not None:
            public.update(
                {
                    "status": AVAILABLE,
                    "forecast_issued_at_utc": record["forecast_issued_at_utc"],
                    "parameter_artifact_sha256": record[
                        "parameter_artifact_sha256"
                    ],
                    "parameter_input_sha256": record["parameter_input_sha256"],
                    "mechanics_profile_id": record["mechanics_profile_id"],
                    "bootstrap_members": record["bootstrap_members"],
                    "paths_per_bootstrap_member": record[
                        "paths_per_bootstrap_member"
                    ],
                    "precision_tier": record["precision_tier"],
                    "aggregate": record["aggregate"],
                }
            )
        else:
            public["status"] = PENDING_RETRY if identity in failures else PENDING_NEW
            public["unavailable_reason"] = failures.get(
                identity,
                "This newly announced matchup is queued for its first automatic simulation.",
            )
        output.append(public)
    return output


def validate_automatic_website_publication(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("automatic website simulation publication must be an object")
    publication = dict(value)
    supplied = _text(publication.get("publication_sha256"))
    unhashed = dict(publication)
    unhashed.pop("publication_sha256", None)
    if supplied != canonical_sha256(unhashed):
        raise ValueError("automatic website simulation publication hash is invalid")
    if (
        publication.get("schema_version") != AUTOMATIC_WEBSITE_SCHEMA_VERSION
        or publication.get("publication_version") != AUTOMATIC_WEBSITE_VERSION
        or publication.get("candidate_only") is not True
        or publication.get("paper_only") is not True
        or publication.get("execution_enabled") is not False
        or publication.get("production_influence") != "none"
    ):
        raise ValueError("automatic website simulation publication policy is invalid")
    events = publication.get("events")
    matchups = publication.get("matchups")
    if not isinstance(events, list) or not isinstance(matchups, list):
        raise ValueError("automatic website simulation events and matchups must be lists")
    if int(publication.get("event_count") or 0) != len(events):
        raise ValueError("automatic website simulation event count is inconsistent")
    if int(publication.get("matchup_count") or 0) != len(matchups):
        raise ValueError("automatic website simulation matchup count is inconsistent")
    statuses = [str(item.get("status")) for item in matchups if isinstance(item, dict)]
    if len(statuses) != len(matchups):
        raise ValueError("automatic website simulation matchup must be an object")
    allowed = {AVAILABLE, WITHHELD_HISTORY, WITHHELD_IDENTITY, PENDING_NEW, PENDING_RETRY}
    if any(status not in allowed for status in statuses):
        raise ValueError("automatic website simulation matchup status is invalid")
    expected = {
        "available_matchups": statuses.count(AVAILABLE),
        "excluded_matchups": statuses.count(WITHHELD_HISTORY)
        + statuses.count(WITHHELD_IDENTITY),
        "pending_matchups": statuses.count(PENDING_NEW) + statuses.count(PENDING_RETRY),
    }
    if any(int(publication.get(key) or 0) != count for key, count in expected.items()):
        raise ValueError("automatic website simulation status counts are inconsistent")
    event_ids = {str(event.get("event_id")) for event in events if isinstance(event, dict)}
    if len(event_ids) != len(events) or "" in event_ids:
        raise ValueError("automatic website simulation event identities are invalid")
    for event in events:
        event_rows = [m for m in matchups if str(m.get("event_id")) == str(event["event_id"])]
        if int(event.get("matchup_count") or 0) != len(event_rows):
            raise ValueError("automatic website simulation event matchup count is invalid")
    if any(str(matchup.get("event_id")) not in event_ids for matchup in matchups):
        raise ValueError("automatic website simulation matchup references an unknown event")
    for matchup in matchups:
        if matchup["status"] == AVAILABLE:
            compact = compact_website_aggregate(matchup.get("aggregate"))
            if compact != matchup.get("aggregate"):
                raise ValueError("automatic website simulation aggregate is not compact")
        elif not _text(matchup.get("unavailable_reason")):
            raise ValueError("unavailable automatic simulation requires a reason")
    _utc(publication["generated_at_utc"])
    return publication


def build_automatic_website_publication(
    catalog: Mapping[str, object],
    rows: list[dict[str, object]],
    records: Mapping[str, Mapping[str, object]],
    exposure: Mapping[str, int],
    *,
    generated_at_utc: object,
    minimum_prior_ufc_fights: int,
    bootstrap_members: int,
    paths_per_member: int,
    mechanics_profile_id: str,
    failures: Mapping[str, str] | None = None,
) -> dict[str, object]:
    matchups = _website_matchups(
        rows,
        records,
        exposure,
        minimum_prior_ufc_fights=minimum_prior_ufc_fights,
        failures=failures,
    )
    events: list[dict[str, object]] = []
    for source in catalog["events"]:
        event_rows = [
            matchup
            for matchup in matchups
            if str(matchup.get("event_id")) == str(source["event_id"])
        ]
        statuses = [str(matchup["status"]) for matchup in event_rows]
        events.append(
            {
                **dict(source),
                "available_matchups": statuses.count(AVAILABLE),
                "excluded_matchups": statuses.count(WITHHELD_HISTORY)
                + statuses.count(WITHHELD_IDENTITY),
                "pending_matchups": statuses.count(PENDING_NEW)
                + statuses.count(PENDING_RETRY),
            }
        )
    statuses = [str(matchup["status"]) for matchup in matchups]
    body: dict[str, object] = {
        "schema_version": AUTOMATIC_WEBSITE_SCHEMA_VERSION,
        "publication_version": AUTOMATIC_WEBSITE_VERSION,
        "model_version": "candidate-fight-sim-automatic-preview-v1",
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "production_influence": "none",
        "generated_at_utc": _utc(generated_at_utc).isoformat(),
        "source_upcoming_publication_sha256": catalog["publication_sha256"],
        "minimum_prior_ufc_fights": int(minimum_prior_ufc_fights),
        "mechanics_profile_id": mechanics_profile_id,
        "precision_tier": "automatic_preview",
        "bootstrap_members": int(bootstrap_members),
        "paths_per_bootstrap_member": int(paths_per_member),
        "paths_per_matchup": int(bootstrap_members) * int(paths_per_member),
        "event_count": len(events),
        "matchup_count": len(matchups),
        "available_matchups": statuses.count(AVAILABLE),
        "excluded_matchups": statuses.count(WITHHELD_HISTORY)
        + statuses.count(WITHHELD_IDENTITY),
        "pending_matchups": statuses.count(PENDING_NEW) + statuses.count(PENDING_RETRY),
        "events": events,
        "matchups": matchups,
        "coverage_warnings": [
            "candidate_only_simulation_does_not_authorize_a_wager",
            "automatic_preview_uses_4096_paths_per_matchup",
            "method_and_round_ev_require_real_synchronized_prop_prices",
            "official_ufcstats_control_is_broader_than_simulated_ground_top_control",
            "fighter_uncertainty_is_large_near_the_minimum_history_threshold",
        ],
    }
    body["publication_sha256"] = canonical_sha256(body)
    return validate_automatic_website_publication(body)


def execute_upcoming_catalog(
    *,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    record_directory: str | Path = DEFAULT_RECORD_DIRECTORY,
    website_output: str | Path = DEFAULT_WEBSITE_OUTPUT,
    raw_path: str | Path = DEFAULT_RAW_FIGHTS,
    profiles_path: str | Path = DEFAULT_FIGHTER_PROFILES,
    round_path: str | Path = DEFAULT_ROUND_STATS,
    minimum_prior_ufc_fights: int = DEFAULT_MINIMUM_PRIOR_UFC_FIGHTS,
    bootstrap_members: int = DEFAULT_BOOTSTRAP_MEMBERS,
    paths_per_member: int = DEFAULT_PATHS_PER_MEMBER,
    random_seed: int = 81173,
    workers: int = 2,
    chunk_size: int = 64,
    max_new_matchups: int = 100,
    max_runtime_seconds: float = 9000.0,
    simulator_config: SimulatorConfig | None = None,
    mechanics_profile_id: str | None = None,
    publish_only: bool = False,
    issued_at_utc: object | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    if minimum_prior_ufc_fights < 0:
        raise ValueError("minimum prior UFC fights must be nonnegative")
    if bootstrap_members <= 0 or paths_per_member <= 0:
        raise ValueError("automatic simulation precision must be positive")
    if max_new_matchups <= 0 or max_runtime_seconds <= 0:
        raise ValueError("automatic simulation work limits must be positive")
    catalog = validate_upcoming_forecast_publication(
        json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    )
    issued = _utc(issued_at_utc or datetime.now(timezone.utc))
    raw, profiles, rounds = load_research_inputs(raw_path, profiles_path, round_path)
    exposure = prior_ufc_exposure(raw, issued)
    base = simulator_config or SimulatorConfig()
    selected_mechanics_profile_id = mechanics_profile_id or (
        f"mechanics-{canonical_sha256(base.to_dict())[:12]}"
    )
    rows = _catalog_rows(
        catalog,
        simulator_config=base,
        mechanics_profile_id=selected_mechanics_profile_id,
        bootstrap_members=bootstrap_members,
        paths_per_member=paths_per_member,
    )
    records = load_automatic_records(record_directory)
    failures: dict[str, str] = {}
    eligible: list[dict[str, object]] = []
    for row in rows:
        simulation_input = row.get("simulation_input")
        if not isinstance(simulation_input, Mapping):
            continue
        identity = str(simulation_input["simulation_input_sha256"])
        if identity in records:
            continue
        fighter_history = int(exposure.get(str(row.get("fighter_id")), 0))
        opponent_history = int(exposure.get(str(row.get("opponent_id")), 0))
        if min(fighter_history, opponent_history) < minimum_prior_ufc_fights:
            continue
        event_date = _utc(row["event_date"])
        if issued >= event_date:
            failures[identity] = (
                "Simulation not started because the date-only event record can no "
                "longer prove that this forecast would precede the event."
            )
            continue
        eligible.append(row)

    if progress:
        progress(
            f"Found {len(eligible)} eligible matchup(s) without a matching "
            f"automatic simulation record; {len(records)} record(s) are reusable."
        )
    started = time.monotonic()
    completed = 0
    artifact = None
    fitter = None
    if eligible and not publish_only:
        fitter = CausalParameterFitter(raw, profiles, rounds)
        if progress:
            progress(
                f"Fitting {bootstrap_members} causal parameter replicas once for "
                "this batch."
            )
        artifact = fitter.fit(
            issued,
            config=ParameterFitConfig(
                bootstrap_members=bootstrap_members,
                random_seed=random_seed,
            ),
            created_at_utc=issued,
        )

    for position, row in enumerate(eligible, start=1):
        identity = str(row["simulation_input_sha256"])
        if publish_only:
            break
        if completed >= max_new_matchups or time.monotonic() - started >= max_runtime_seconds:
            break
        assert artifact is not None and fitter is not None
        if progress:
            progress(
                f"Simulating new matchup {position}/{len(eligible)}: "
                f"{row['fighter_name']} vs {row['opponent_name']}."
            )
        try:
            specs = build_specs(
                fitter,
                artifact,
                red_fighter_id=str(row["fighter_id"]),
                blue_fighter_id=str(row["opponent_id"]),
                division=str(row.get("division") or "Unknown"),
                scheduled_rounds=int(row["scheduled_rounds"]),
                event_id=str(row["event_id"]),
                matchup_id=str(row["matchup_id"]),
                root_seed=(
                    f"automatic-upcoming:{row['simulation_input_sha256']}:"
                    f"{artifact.artifact_sha256}"
                ),
                simulator_base=base,
                _artifact_validated=True,
            )
            result = run_nested(
                specs,
                paths_per_member=paths_per_member,
                workers=workers,
                chunk_size=chunk_size,
                max_traces=0,
                retain_paths=False,
            )
            full = result.forecast.to_dict()
            record: dict[str, object] = {
                "schema_version": AUTOMATIC_RECORD_SCHEMA_VERSION,
                "record_version": AUTOMATIC_RECORD_VERSION,
                "candidate_only": True,
                "paper_only": True,
                "execution_enabled": False,
                "production_influence": "none",
                "status": AVAILABLE,
                "precision_tier": "automatic_preview",
                **dict(row["simulation_input"]),
                "fighter_name": row["fighter_name"],
                "opponent_name": row["opponent_name"],
                "event_title_at_discovery": row["event_title"],
                "event_date_at_discovery": row["event_date"],
                "source_upcoming_publication_sha256": catalog[
                    "publication_sha256"
                ],
                "forecast_issued_at_utc": issued.isoformat(),
                "parameter_artifact_sha256": artifact.artifact_sha256,
                "parameter_input_sha256": artifact.input_sha256,
                "mechanics_profile_id": selected_mechanics_profile_id,
                "aggregate": compact_website_aggregate(full),
            }
            record["record_sha256"] = canonical_sha256(record)
            validated_record = validate_automatic_record(record)
            _write_automatic_record(record_directory, validated_record)
            records[identity] = validated_record
            completed += 1
        except Exception as error:  # one bad matchup must not erase completed fights
            failures[identity] = (
                "Automatic simulation failed and will be retried on the next update: "
                f"{type(error).__name__}: {error}"
            )
            if progress:
                progress(f"WARNING: {row['fighter_name']} vs {row['opponent_name']}: {error}")

    remaining = sum(
        1
        for row in eligible
        if str(row["simulation_input_sha256"]) not in records
    )
    if progress:
        progress(
            f"Completed {completed} new simulation(s); {remaining} eligible "
            "matchup(s) remain queued."
        )
    publication = build_automatic_website_publication(
        catalog,
        rows,
        records,
        exposure,
        generated_at_utc=issued,
        minimum_prior_ufc_fights=minimum_prior_ufc_fights,
        bootstrap_members=bootstrap_members,
        paths_per_member=paths_per_member,
        mechanics_profile_id=selected_mechanics_profile_id,
        failures=failures,
    )
    destination = atomic_write_json(website_output, publication)
    return destination, publication
