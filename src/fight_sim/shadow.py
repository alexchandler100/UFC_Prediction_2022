"""Opt-in bridge from frozen research artifacts to upcoming shadow forecasts.

Nothing in this module changes production probabilities, odds, or decisions.
The weekly updater calls it only when a separately reviewed research-status
file explicitly enables paper-only shadows for an exact parameter/backtest
pair.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import pandas as pd

from fight_semantics import upcoming_schedule
from market_tracker import matchup_id_for

from .domain import BoutConfig, SimulationRunSpec
from .evaluation import BacktestReport, load_backtest_report
from .monte_carlo import MonteCarloResult, run_adaptive_nested
from .parameters import (
    CausalParameterFitter,
    ParameterEnsembleArtifact,
    canonical_json,
    canonical_sha256,
    load_parameter_artifact,
    simulator_config_for_member,
)
from .publication import (
    append_shadow_forecast_publication,
    build_shadow_forecast_publication,
    compact_shadow_aggregate,
)


RESEARCH_STATUS_SCHEMA_VERSION = 1
SHADOW_REQUIRED_BOOTSTRAP_MEMBERS = 200
SHADOW_REQUIRED_HISTORICAL_FIGHTS = 200
SHADOW_REQUIRED_HISTORICAL_PATHS = 4096
SHADOW_REQUIRED_SEED_REPEATS = 2


def _write_local_aggregate_authority(
    directory: str | Path,
    aggregate: Mapping[str, object],
) -> Path:
    """Persist the exact aggregate under ignored local artifacts.

    Standard-runner files remain ephemeral and are never uploaded.  A local
    execution retains the exact member-level count authority; either way the
    public hash remains a deterministic replay commitment.
    """

    value = dict(aggregate)
    digest = canonical_sha256(value)
    matchup_id = _text(value.get("matchup_id"))
    if not matchup_id:
        raise ValueError("local aggregate authority has no matchup ID")
    destination = Path(directory) / f"{matchup_id}-{digest}.json.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(
        (canonical_json(value) + "\n").encode("utf-8"),
        compresslevel=9,
        mtime=0,
    )
    if destination.exists():
        existing = json.loads(gzip.decompress(destination.read_bytes()))
        if canonical_sha256(existing) != digest:
            raise ValueError("local aggregate authority path contains different data")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
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


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).split())


def validate_research_status(
    value: object,
    *,
    artifact: ParameterEnsembleArtifact | None = None,
    backtest: BacktestReport | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("simulation research status must be a JSON object")
    status = dict(value)
    if status.get("schema_version") != RESEARCH_STATUS_SCHEMA_VERSION:
        raise ValueError("unsupported simulation research-status schema")
    if (
        status.get("candidate_only") is not True
        or status.get("paper_only") is not True
        or status.get("production_enabled") is not False
        or status.get("execution_enabled") is not False
    ):
        raise ValueError("research status must preserve the candidate-only boundary")
    if status.get("integrity_gate_passed") is not True:
        raise ValueError("simulation integrity gate has not passed")
    if status.get("causal_backtest_gate_passed") is not True:
        raise ValueError("simulation causal backtest gate has not passed")
    if not isinstance(status.get("shadow_enabled"), bool):
        raise ValueError("shadow_enabled must be explicitly true or false")
    if artifact is not None and status.get("parameter_artifact_sha256") != artifact.artifact_sha256:
        raise ValueError("research status does not name the frozen parameter artifact")
    if backtest is not None and status.get("backtest_report_sha256") != backtest.report_sha256:
        raise ValueError("research status does not name the validated backtest report")
    if (
        status.get("shadow_enabled") is True
        and artifact is not None
        and backtest is not None
    ):
        validate_shadow_evidence(artifact, backtest)
    return status


def validate_shadow_evidence(
    artifact: ParameterEnsembleArtifact,
    backtest: BacktestReport,
) -> None:
    """Verify measured evidence behind an enabled paper-shadow declaration.

    The reviewed status file remains a human approval, but its two gate booleans
    cannot substitute for the minimum causal ensemble/backtest evidence.
    Production promotion has a separate, much stricter prospective contract.
    """

    artifact.validate()
    backtest.validate()
    errors: list[str] = []
    if len(artifact.members) != SHADOW_REQUIRED_BOOTSTRAP_MEMBERS:
        errors.append(
            f"frozen ensemble requires {SHADOW_REQUIRED_BOOTSTRAP_MEMBERS} members"
        )
    if artifact.observed_round_sides <= 0:
        errors.append("frozen ensemble has no reconciled per-round evidence")
    if artifact.round_reconciliation_counts.get("matched", 0) != artifact.observed_round_sides:
        errors.append("frozen ensemble is not based exclusively on matched round rows")
    if backtest.primary_metric != "joint_side_by_method_log_loss":
        errors.append("backtest primary metric is not joint side-by-method log loss")
    if len(backtest.folds) < 3:
        errors.append("backtest requires at least three chronological folds")
    scored_fights = int(backtest.aggregate.get("n_fights") or 0)
    if scored_fights < SHADOW_REQUIRED_HISTORICAL_FIGHTS:
        errors.append(
            f"backtest requires at least {SHADOW_REQUIRED_HISTORICAL_FIGHTS} scored fights"
        )
    noise = backtest.simulation_noise
    if int(noise.get("seed_repeats") or 0) < SHADOW_REQUIRED_SEED_REPEATS:
        errors.append(
            f"backtest requires at least {SHADOW_REQUIRED_SEED_REPEATS} independent seeds"
        )
    if int(noise.get("paths_per_matchup") or 0) < SHADOW_REQUIRED_HISTORICAL_PATHS:
        errors.append(
            f"backtest requires at least {SHADOW_REQUIRED_HISTORICAL_PATHS} paths per matchup"
        )
    repeat_hashes = list(noise.get("repeat_forecast_sha256") or [])
    if len(repeat_hashes) < SHADOW_REQUIRED_SEED_REPEATS or any(
        not isinstance(value, str) or len(value) != 64 for value in repeat_hashes
    ):
        errors.append("backtest independent-seed forecast hashes are incomplete")

    required_comparisons = {
        "population_joint": 0.99,
        "division_joint": 0.99,
        "production_winner": 0.90,
        "competing_risk_joint": 0.90,
        # Market history can be sparse, but both configured real-price views
        # must have at least one timestamp-aligned settled observation.
        "timestamped_market": 0.0,
        "timestamped_market_totals": 0.0,
    }
    for name, minimum_coverage in required_comparisons.items():
        comparison = backtest.comparisons.get(name)
        if not isinstance(comparison, Mapping):
            errors.append(f"backtest is missing required comparator {name}")
            continue
        covered = int(comparison.get("n_covered") or 0)
        coverage = float(comparison.get("coverage") or 0.0)
        if covered <= 0 or coverage < minimum_coverage:
            errors.append(
                f"backtest comparator {name} has insufficient aligned coverage"
            )
    if errors:
        raise ValueError("shadow evidence gate failed: " + "; ".join(errors))


def load_enabled_shadow_artifacts(
    simulation_directory: str | Path,
) -> tuple[ParameterEnsembleArtifact, BacktestReport] | None:
    """Return the reviewed frozen pair, or ``None`` when shadows are disabled."""

    root = Path(simulation_directory)
    status_path = root / "research_status.json"
    if not status_path.exists():
        return None
    status_value = json.loads(status_path.read_text(encoding="utf-8"))
    # Validate the non-production boundary before trusting any path field.
    preliminary = validate_research_status(status_value)
    if preliminary["shadow_enabled"] is not True:
        return None
    artifact_path = root / "parameter_model.json.gz"
    backtest_path = root / "backtest_report.json"
    artifact = load_parameter_artifact(artifact_path)
    backtest = load_backtest_report(backtest_path)
    validate_research_status(preliminary, artifact=artifact, backtest=backtest)
    return artifact, backtest


def _normalize_upcoming(upcoming: pd.DataFrame, event_id: str) -> pd.DataFrame:
    aliases = {
        "fighter id": "red_fighter_id",
        "opponent id": "blue_fighter_id",
        "fighter name": "red_fighter_name",
        "opponent name": "blue_fighter_name",
    }
    frame = upcoming.copy().reset_index(drop=True)
    for source, target in aliases.items():
        if target not in frame and source in frame:
            frame[target] = frame[source]
    required = {"red_fighter_id", "blue_fighter_id", "division"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"upcoming card is missing simulation identity fields: {missing}")
    frame["bout_order"] = range(len(frame))
    frame["matchup_id"] = [
        matchup_id_for(event_id, _text(row.red_fighter_id), _text(row.blue_fighter_id))
        if _text(row.red_fighter_id) and _text(row.blue_fighter_id)
        else ""
        for row in frame.itertuples(index=False)
    ]
    return frame


def build_upcoming_run_specs(
    artifact: ParameterEnsembleArtifact,
    fitter: CausalParameterFitter,
    row: Mapping[str, object],
    *,
    event_id: str,
    root_seed: str,
) -> tuple[SimulationRunSpec, ...]:
    red_id = _text(row.get("red_fighter_id"))
    blue_id = _text(row.get("blue_fighter_id"))
    matchup_id = _text(row.get("matchup_id")) or matchup_id_for(event_id, red_id, blue_id)
    rounds, _basis = upcoming_schedule(int(row.get("bout_order") or 0), row.get("division"))
    bout = BoutConfig(
        matchup_id=matchup_id,
        red_fighter_id=red_id,
        blue_fighter_id=blue_id,
        scheduled_rounds=rounds,
        division=_text(row.get("division")),
        title_bout="title" in _text(row.get("division")).casefold(),
        event_id=event_id,
    )
    return tuple(
        SimulationRunSpec(
            bout=bout,
            red=fitter.snapshot_for(
                artifact,
                red_id,
                division=bout.division,
                member_index=member.member_index,
            ),
            blue=fitter.snapshot_for(
                artifact,
                blue_id,
                division=bout.division,
                member_index=member.member_index,
            ),
            root_seed=root_seed,
            parameter_artifact_id=artifact.artifact_sha256,
            bootstrap_member=member.member_index,
            simulator=simulator_config_for_member(member),
        )
        for member in artifact.members
    )


def generate_upcoming_shadow_publication(
    *,
    simulation_directory: str | Path,
    raw_fights: pd.DataFrame,
    fighter_profiles: pd.DataFrame,
    round_stats: pd.DataFrame | None,
    upcoming: pd.DataFrame,
    card: Mapping[str, object],
    source_commit_sha: str,
    forecast_issued_at_utc: str,
    workers: int = 1,
    initial_paths_per_member: int = 512,
    max_paths_per_member: int = 2048,
    local_authority_directory: str | Path | None = None,
) -> Path | None:
    """Generate and append reviewed paper-only card shadows.

    Any failed or non-converged matchup aborts before publication, so no
    partial card can be mistaken for a completed candidate forecast.
    """

    loaded = load_enabled_shadow_artifacts(simulation_directory)
    if loaded is None:
        return None
    artifact, _backtest = loaded
    event_id = _text(card.get("event_id"))
    normalized = _normalize_upcoming(upcoming, event_id)
    fitter = CausalParameterFitter(raw_fights, fighter_profiles, round_stats)
    authority_root = (
        Path(local_authority_directory)
        if local_authority_directory is not None
        else Path.cwd() / "artifacts/simulations/shadow-authority" / event_id
    )
    forecasts: dict[str, object] = {}
    for row in normalized.to_dict("records"):
        if not row["matchup_id"]:
            continue
        root_seed = f"shadow:{event_id}:{row['matchup_id']}:{artifact.artifact_sha256}"
        specs = build_upcoming_run_specs(
            artifact, fitter, row, event_id=event_id, root_seed=root_seed
        )
        result: MonteCarloResult = run_adaptive_nested(
            specs,
            initial_paths_per_member=initial_paths_per_member,
            max_paths_per_member=max_paths_per_member,
            workers=workers,
            max_traces=0,
        )
        if not result.converged:
            raise RuntimeError(
                f"withholding non-converged simulation shadow for {row['matchup_id']}"
            )
        # Persist exact member-level histograms locally, then release them
        # after preserving their content hash in the compact publication view.
        full_aggregate = result.forecast.to_dict()
        compact = compact_shadow_aggregate(full_aggregate)
        authority_path = _write_local_aggregate_authority(
            authority_root, full_aggregate
        )
        if compact["local_aggregate_sha256"] not in authority_path.name:
            raise RuntimeError("local aggregate authority hash disagrees with shadow")
        forecasts[str(row["matchup_id"])] = compact
    publication = build_shadow_forecast_publication(
        forecasts,
        normalized,
        card,
        artifact,
        forecast_issued_at_utc=forecast_issued_at_utc,
        source_commit_sha=source_commit_sha,
    )
    return append_shadow_forecast_publication(
        Path(simulation_directory) / "shadow_forecasts", publication
    )


def maybe_generate_weekly_shadows(
    *,
    simulation_directory: str | Path,
    raw_fights: pd.DataFrame,
    fighter_profiles: pd.DataFrame,
    round_stats: pd.DataFrame | None,
    upcoming: pd.DataFrame,
    card: Mapping[str, object],
    source_commit_sha: str,
    forecast_issued_at_utc: str,
) -> Path | None:
    """Updater adapter with bounded local/GitHub-hosted parallelism."""

    workers = max(1, min(int(os.environ.get("FIGHT_SIM_WORKERS", "2")), 8))
    return generate_upcoming_shadow_publication(
        simulation_directory=simulation_directory,
        raw_fights=raw_fights,
        fighter_profiles=fighter_profiles,
        round_stats=round_stats,
        upcoming=upcoming,
        card=card,
        source_commit_sha=source_commit_sha,
        forecast_issued_at_utc=forecast_issued_at_utc,
        workers=workers,
    )
