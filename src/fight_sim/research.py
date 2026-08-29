"""Local-only orchestration for the evidence-first simulation CLI.

This module intentionally writes only to caller-selected files.  It has no
production updater, website, odds, or wagering integration.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from fight_semantics import historical_schedule, method_bucket, stable_ufcstats_id
from market_tracker import matchup_id_for

from .analysis import write_analysis_report
from .domain import (
    ENGINE_VERSION,
    RNG_CONTRACT_VERSION,
    BoutConfig,
    SimulationRunSpec,
    SimulatorConfig,
)
from .evaluation import (
    BacktestConfig,
    BacktestReport,
    add_simulation_win_probability_column,
    evaluate_simulation_ledger,
    evaluate_chronological_winner_stack,
    posterior_predictive_rows,
    repeated_seed_summary,
    run_chronological_backtest,
    write_backtest_report,
)
from .monte_carlo import (
    MonteCarloResult,
    NestedSimulationBatchError,
    run_adaptive_nested,
    run_nested,
)
from .parameters import (
    CausalParameterFitter,
    PARAMETER_MODEL_VERSION,
    SNAPSHOT_PARAMETER_MODES,
    TAKEDOWN_CONTROL_PARAMETER_MODEL_VERSION,
    ParameterEnsembleArtifact,
    ParameterFitConfig,
    canonical_sha256,
    load_parameter_artifact,
    save_parameter_artifact,
    simulator_config_for_member,
)
from .replay import diff_event_streams, replay_trace, stochastic_replay
from .telemetry import trace_from_dict, trace_to_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA = REPO_ROOT / "src/content/data/processed"
DEFAULT_RAW_FIGHTS = PROCESSED_DATA / "ufc_fights_reported_doubled.csv"
DEFAULT_FIGHTER_PROFILES = PROCESSED_DATA / "fighter_stats.csv"
DEFAULT_ROUND_STATS = PROCESSED_DATA / "ufc_fight_round_stats_doubled.csv"
DEFAULT_POINT_IN_TIME = PROCESSED_DATA / "ufc_fights_point_in_time.csv"
DEFAULT_MARKET_DIRECTORY = REPO_ROOT / "src/content/data/market"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts/simulations"
DEFAULT_PARAMETER_ARTIFACT = DEFAULT_ARTIFACT_ROOT / "parameters.json.gz"
DEFAULT_POSTERIOR_FIT_CACHE = DEFAULT_ARTIFACT_ROOT / "causal-fit-cache"
BASELINE_WARNINGS_ATTR = "fight_sim_baseline_warnings"
MONEYLINE_MINIMUM_BOOKS = 3
TOTAL_MINIMUM_BOOKS = 2

ALL_TERMINAL_OUTCOMES = (
    "red_ko_tko",
    "red_submission",
    "red_decision",
    "red_other",
    "blue_ko_tko",
    "blue_submission",
    "blue_decision",
    "blue_other",
    "draw",
    "no_contest",
)


class NonConvergedSimulationError(RuntimeError):
    """Raised after diagnostics are saved but forecast files are withheld."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__(
            "simulation did not pass convergence gates; aggregate/traces/report "
            f"were withheld (diagnostics: {output_dir})"
        )
        self.output_dir = output_dir


def _baseline_warnings(frame: pd.DataFrame) -> list[str]:
    return [str(item) for item in frame.attrs.get(BASELINE_WARNINGS_ATTR, ())]


def _add_baseline_warning(frame: pd.DataFrame, warning: str) -> None:
    warnings = _baseline_warnings(frame)
    if warning not in warnings:
        warnings.append(warning)
    frame.attrs[BASELINE_WARNINGS_ATTR] = tuple(warnings)


def _copy_baseline_warnings(source: pd.DataFrame, target: pd.DataFrame) -> None:
    target.attrs[BASELINE_WARNINGS_ATTR] = tuple(_baseline_warnings(source))


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[attr-defined]
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_ledger_value(value: object) -> object:
    """Convert expected missing ledger values to JSON null without hiding report NaNs."""

    if isinstance(value, np.generic):
        return _json_ledger_value(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ledger_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ledger_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "to_dict"):
        return _json_ledger_value(value.to_dict())  # type: ignore[attr-defined]
    return value


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(text)
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


def atomic_write_json(path: str | Path, value: object) -> Path:
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ) + "\n"
    return atomic_write_text(path, encoded)


def load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _file_sha256(path: str | Path, *, required: bool) -> str | None:
    source = Path(path)
    if not source.is_file():
        if required:
            raise FileNotFoundError(source)
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json_gzip(path: str | Path, value: Mapping[str, object]) -> Path:
    return _atomic_write_jsonl_gzip(path, (value,))


def _load_json_gzip(path: str | Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        lines = [line for line in stream if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"checkpoint must contain exactly one JSON object: {path}")
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint must contain a JSON object: {path}")
    return value


def _read_csv(path: str | Path, *, required: bool = True) -> pd.DataFrame | None:
    source = Path(path)
    if not source.is_file():
        if required:
            raise FileNotFoundError(source)
        return None
    return pd.read_csv(source, low_memory=False)


def load_research_inputs(
    raw_path: str | Path = DEFAULT_RAW_FIGHTS,
    profiles_path: str | Path = DEFAULT_FIGHTER_PROFILES,
    round_path: str | Path = DEFAULT_ROUND_STATS,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    raw = _read_csv(raw_path)
    assert raw is not None
    profiles = _read_csv(profiles_path, required=False)
    rounds = _read_csv(round_path, required=False)
    return raw, profiles, rounds


def execute_backfill(
    *,
    max_fights: int,
    checkpoint_every: int,
    summary_output: str | Path,
    refresh_existing: bool = False,
    max_runtime_seconds: float = 3000.0,
) -> dict[str, object]:
    # Lazy import avoids initializing legacy odds helpers for every CLI command.
    from data_handler.data_handler import DataHandler

    handler = DataHandler()
    summary = handler.backfill_ufc_fight_round_stats_doubled(
        max_fights=max_fights,
        checkpoint_every=checkpoint_every,
        refresh_existing=refresh_existing,
        max_runtime_seconds=max_runtime_seconds,
    )
    payload = asdict(summary)
    atomic_write_json(summary_output, payload)
    return payload


def execute_fit(
    *,
    raw_path: str | Path = DEFAULT_RAW_FIGHTS,
    profiles_path: str | Path = DEFAULT_FIGHTER_PROFILES,
    round_path: str | Path = DEFAULT_ROUND_STATS,
    output: str | Path = DEFAULT_PARAMETER_ARTIFACT,
    as_of: object | None = None,
    bootstrap_members: int = 200,
    random_seed: int = 1729,
    created_at_utc: object | None = None,
    use_takedown_control_association: bool = False,
) -> ParameterEnsembleArtifact:
    raw, profiles, rounds = load_research_inputs(raw_path, profiles_path, round_path)
    cutoff = as_of or datetime.now(timezone.utc).isoformat()
    artifact = CausalParameterFitter(
        raw,
        profiles,
        rounds,
        use_takedown_control_association=use_takedown_control_association,
    ).fit(
        cutoff,
        config=ParameterFitConfig(
            bootstrap_members=bootstrap_members,
            random_seed=random_seed,
        ),
        created_at_utc=created_at_utc,
    )
    save_parameter_artifact(output, artifact)
    return artifact


def _identity_token(value: object) -> str:
    return stable_ufcstats_id(value)


def build_specs(
    fitter: CausalParameterFitter,
    artifact: ParameterEnsembleArtifact,
    *,
    red_fighter_id: str,
    blue_fighter_id: str,
    division: str,
    scheduled_rounds: int,
    event_id: str,
    root_seed: str | int,
    matchup_id: str | None = None,
    simulator_base: SimulatorConfig | None = None,
    snapshot_parameter_mode: str = "full",
    _artifact_validated: bool = False,
) -> tuple[SimulationRunSpec, ...]:
    if not _artifact_validated:
        artifact.validate()
    red_id = _identity_token(red_fighter_id)
    blue_id = _identity_token(blue_fighter_id)
    if not red_id or not blue_id or red_id == blue_id:
        raise ValueError("run requires two distinct stable fighter IDs")
    resolved_matchup_id = matchup_id or matchup_id_for(event_id, red_id, blue_id)
    bout = BoutConfig(
        matchup_id=resolved_matchup_id,
        red_fighter_id=red_id,
        blue_fighter_id=blue_id,
        scheduled_rounds=scheduled_rounds,
        division=division,
        title_bout=scheduled_rounds == 5,
        event_id=event_id,
    )
    specs = []
    for member in artifact.members:
        # Both sides are always drawn from the exact same bootstrap member.
        red = fitter.snapshot_for(
            artifact,
            red_id,
            division=division,
            member_index=member.member_index,
            parameter_mode=snapshot_parameter_mode,
            _artifact_validated=True,
        )
        blue = fitter.snapshot_for(
            artifact,
            blue_id,
            division=division,
            member_index=member.member_index,
            parameter_mode=snapshot_parameter_mode,
            _artifact_validated=True,
        )
        specs.append(
            SimulationRunSpec(
                bout=bout,
                red=red,
                blue=blue,
                root_seed=root_seed,
                parameter_artifact_id=artifact.artifact_sha256,
                bootstrap_member=member.member_index,
                simulator=simulator_config_for_member(member, base=simulator_base),
            )
        )
    return tuple(specs)


def _default_run_dir(specs: Sequence[SimulationRunSpec]) -> Path:
    first = specs[0]
    token = canonical_sha256(
        {
            "matchup_id": first.bout.matchup_id,
            "parameter_artifact_id": first.parameter_artifact_id,
            "root_seed": str(first.root_seed),
        }
    )[:12]
    return DEFAULT_ARTIFACT_ROOT / "runs" / f"{first.bout.matchup_id}-{token}"


def _convergence_payload(
    result: MonteCarloResult, specs: Sequence[SimulationRunSpec]
) -> dict[str, object]:
    batches = [
        {**asdict(item), "converged": item.converged}
        for item in result.convergence
    ]
    return {
        "schema_version": 1,
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "matchup_id": specs[0].bout.matchup_id,
        "parameter_artifact_id": specs[0].parameter_artifact_id,
        "bootstrap_members": len(specs),
        "converged": result.converged,
        "convergence": batches,
        "coverage_warnings": (
            [] if result.converged else ["simulation_convergence_gates_not_met"]
        ),
    }


def execute_run(
    *,
    parameter_path: str | Path = DEFAULT_PARAMETER_ARTIFACT,
    raw_path: str | Path = DEFAULT_RAW_FIGHTS,
    profiles_path: str | Path = DEFAULT_FIGHTER_PROFILES,
    round_path: str | Path = DEFAULT_ROUND_STATS,
    red_fighter_id: str,
    blue_fighter_id: str,
    division: str,
    scheduled_rounds: int = 3,
    event_id: str = "local-research",
    matchup_id: str | None = None,
    root_seed: str | int = "20220813",
    output_dir: str | Path | None = None,
    bootstrap_member_limit: int | None = None,
    initial_paths_per_member: int = 512,
    max_paths_per_member: int = 2048,
    workers: int = 1,
    chunk_size: int = 64,
    max_traces: int = 32,
    winner_mcse_target: float = 0.002,
    parameter_quantile_tolerance: float = 0.01,
    allow_nonconverged_research: bool = False,
    simulator_config: SimulatorConfig | None = None,
) -> tuple[Path, MonteCarloResult]:
    artifact = load_parameter_artifact(parameter_path)
    raw, profiles, rounds = load_research_inputs(raw_path, profiles_path, round_path)
    fitter = CausalParameterFitter(raw, profiles, rounds)
    specs = build_specs(
        fitter,
        artifact,
        red_fighter_id=red_fighter_id,
        blue_fighter_id=blue_fighter_id,
        division=division,
        scheduled_rounds=scheduled_rounds,
        event_id=event_id,
        root_seed=root_seed,
        matchup_id=matchup_id,
        simulator_base=simulator_config,
    )
    if bootstrap_member_limit is not None:
        if bootstrap_member_limit <= 0:
            raise ValueError("bootstrap_member_limit must be positive")
        if bootstrap_member_limit < len(specs):
            # Cover the complete ordered ensemble instead of taking only its
            # first members. The selection is deterministic and swap-neutral.
            positions = tuple(
                index * len(specs) // bootstrap_member_limit
                for index in range(bootstrap_member_limit)
            )
            specs = tuple(specs[position] for position in positions)
    destination = Path(output_dir) if output_dir is not None else _default_run_dir(specs)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            f"run output directory is not empty: {destination}; choose a new directory"
        )
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        destination / "specs.json",
        {
            "schema_version": 1,
            "same_member_pairing": True,
            "specs": [spec.to_dict() for spec in specs],
        },
    )
    try:
        result = run_adaptive_nested(
            specs,
            initial_paths_per_member=initial_paths_per_member,
            max_paths_per_member=max_paths_per_member,
            workers=workers,
            chunk_size=chunk_size,
            max_traces=max_traces,
            winner_mcse_target=winner_mcse_target,
            parameter_quantile_tolerance=parameter_quantile_tolerance,
        )
    except NestedSimulationBatchError as error:
        # The run is deliberately unpublished, but the invariant trace must
        # survive the process so it can be replayed and diagnosed locally.
        failure_path = atomic_write_json(
            destination / "invariant-failure.json", error.to_dict()
        )
        error.failure_path = failure_path
        raise
    convergence = _convergence_payload(result, specs)
    atomic_write_json(destination / "convergence.json", convergence)
    if not result.converged and not allow_nonconverged_research:
        raise NonConvergedSimulationError(destination)

    aggregate = result.forecast.to_dict()
    run_status = {
        "schema_version": 1,
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "converged": result.converged,
        "nonconverged_research_override": not result.converged,
        "aggregate": aggregate,
    }
    atomic_write_json(destination / "aggregate.json", run_status)
    if result.trace_manifest is not None:
        atomic_write_json(destination / "trace-manifest.json", asdict(result.trace_manifest))
    trace_dir = destination / "traces"
    serialized_traces: list[dict[str, object]] = []
    for trace in result.traces:
        value = trace_to_dict(trace)
        serialized_traces.append(value)
        atomic_write_json(
            trace_dir
            / f"member-{trace.bootstrap_member:03d}-index-{trace.simulation_index:08d}.json",
            value,
        )
    write_analysis_report(
        destination / "analysis.html",
        aggregate,
        run_spec={"specs": [spec.to_dict() for spec in specs]},
        traces=serialized_traces,
        evaluation=convergence,
        title=f"Fight simulation: {specs[0].red.fighter_name} vs {specs[0].blue.fighter_name}",
    )
    return destination, result


def _scheduled_rounds(row: Mapping[str, object]) -> int | None:
    rounds, _basis = historical_schedule(
        time_format=row.get("time_format"),
        method=row.get("method"),
        total_fight_seconds=row.get("total_fight_time"),
        finish_round=row.get("round"),
    )
    return rounds


def physical_backtest_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse mirrored source rows into deterministic red/blue truth rows."""

    frame = raw.copy()
    frame["fight_id"] = frame["fight_url"].map(_identity_token)
    frame["event_id"] = frame["event_url"].map(_identity_token)
    frame["fighter_id"] = frame["fighter_url"].map(_identity_token)
    frame["opponent_id"] = frame["opponent_url"].map(_identity_token)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True)
    # Build UFC experience from strictly earlier event dates. Grouping first by
    # fighter/date prevents a same-day tournament or ambiguous bout order from
    # leaking another bout on the current card into this pre-fight quantity.
    date_experience = (
        frame.groupby(["fighter_id", "date"], as_index=False, sort=True)
        .size()
        .sort_values(["fighter_id", "date"], kind="stable")
    )
    date_experience["prior_ufc_fights"] = (
        date_experience.groupby("fighter_id", sort=False)["size"].cumsum()
        - date_experience["size"]
    )
    frame = frame.merge(
        date_experience[["fighter_id", "date", "prior_ufc_fights"]],
        on=["fighter_id", "date"],
        how="left",
        validate="many_to_one",
    )
    counts = frame.groupby("fight_id", sort=False).size()
    if not counts.eq(2).all():
        raise ValueError("backtest raw data requires two mirrored sides per fight")
    rows: list[dict[str, object]] = []
    for fight_id, sides in frame.groupby("fight_id", sort=False):
        sides = sides.sort_values("fighter_id", kind="stable")
        red = sides.iloc[0]
        blue = sides.iloc[1]
        rounds = _scheduled_rounds(red)
        if rounds not in (3, 5):
            continue
        results = set(sides["result"].fillna("").astype(str).str.upper())
        method = method_bucket(red.get("method"), result=red.get("result"))
        if "NC" in results or method == "no_contest":
            outcome = "no_contest"
        elif "D" in results:
            outcome = "draw"
        else:
            winner = "red" if str(red.get("result")).upper() == "W" else "blue"
            outcome = f"{winner}_{method}"
        red_experience = int(red["prior_ufc_fights"])
        blue_experience = int(blue["prior_ufc_fights"])
        minimum_experience = min(red_experience, blue_experience)
        if minimum_experience == 0:
            experience_band = "debutant_in_matchup"
        elif minimum_experience <= 2:
            experience_band = "both_1_to_2_prior"
        elif minimum_experience <= 5:
            experience_band = "both_3_to_5_prior"
        elif minimum_experience <= 10:
            experience_band = "both_6_to_10_prior"
        else:
            experience_band = "both_11_plus_prior"
        rows.append(
            {
                "date": red["date"],
                "event_id": str(red["event_id"]),
                "fight_id": str(fight_id),
                "red_fighter_id": str(red["fighter_id"]),
                "blue_fighter_id": str(blue["fighter_id"]),
                "red_fighter_name": str(red.get("fighter") or red["fighter_id"]),
                "blue_fighter_name": str(blue.get("fighter") or blue["fighter_id"]),
                "division": str(red.get("division") or "Unknown"),
                "scheduled_rounds": int(rounds),
                "red_prior_ufc_fights": red_experience,
                "blue_prior_ufc_fights": blue_experience,
                "experience_band": experience_band,
                "actual_outcome": outcome,
                "actual_duration_seconds": (
                    None
                    if pd.isna(red.get("total_fight_time"))
                    else float(red["total_fight_time"])
                ),
                "era": f"{(int(red['date'].year) // 5) * 5}s",
                "sex": (
                    "women"
                    if "women" in str(red.get("division") or "").casefold()
                    else "men"
                ),
                **{
                    f"actual_red_{target}": (
                        None if pd.isna(red.get(source)) else float(red[source])
                    )
                    for target, source in (
                        ("significant_strikes", "sig_strikes_landed"),
                        ("significant_strike_attempts", "sig_strikes_attempts"),
                        ("head_strikes_landed", "head_strikes_landed"),
                        ("body_strikes_landed", "body_strikes_landed"),
                        ("leg_strikes_landed", "leg_strikes_landed"),
                        ("distance_strikes_landed", "distance_strikes_landed"),
                        ("distance_strike_attempts", "distance_strikes_attempts"),
                        ("clinch_strikes_landed", "clinch_strikes_landed"),
                        ("clinch_strike_attempts", "clinch_strikes_attempts"),
                        ("ground_strikes_landed", "ground_strikes_landed"),
                        ("ground_strike_attempts", "ground_strikes_attempts"),
                        ("knockdowns", "knockdowns"),
                        ("takedowns", "takedowns_landed"),
                        ("takedown_attempts", "takedowns_attempts"),
                        ("submission_attempts", "sub_attempts"),
                        ("control_seconds", "control"),
                    )
                },
                **{
                    f"actual_blue_{target}": (
                        None if pd.isna(blue.get(source)) else float(blue[source])
                    )
                    for target, source in (
                        ("significant_strikes", "sig_strikes_landed"),
                        ("significant_strike_attempts", "sig_strikes_attempts"),
                        ("head_strikes_landed", "head_strikes_landed"),
                        ("body_strikes_landed", "body_strikes_landed"),
                        ("leg_strikes_landed", "leg_strikes_landed"),
                        ("distance_strikes_landed", "distance_strikes_landed"),
                        ("distance_strike_attempts", "distance_strikes_attempts"),
                        ("clinch_strikes_landed", "clinch_strikes_landed"),
                        ("clinch_strike_attempts", "clinch_strikes_attempts"),
                        ("ground_strikes_landed", "ground_strikes_landed"),
                        ("ground_strike_attempts", "ground_strikes_attempts"),
                        ("knockdowns", "knockdowns"),
                        ("takedowns", "takedowns_landed"),
                        ("takedown_attempts", "takedowns_attempts"),
                        ("submission_attempts", "sub_attempts"),
                        ("control_seconds", "control"),
                    )
                },
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["date", "event_id", "fight_id"], kind="stable"
    ).reset_index(drop=True)
    pairs = {
        "total_significant_strikes": ("significant_strikes", "sum"),
        "significant_strike_differential": ("significant_strikes", "difference"),
        "total_significant_strike_attempts": ("significant_strike_attempts", "sum"),
        "total_ground_strikes_landed": ("ground_strikes_landed", "sum"),
        "ground_strike_differential": ("ground_strikes_landed", "difference"),
        "total_knockdowns": ("knockdowns", "sum"),
        "knockdown_differential": ("knockdowns", "difference"),
        "total_takedowns": ("takedowns", "sum"),
        "takedown_differential": ("takedowns", "difference"),
        "total_submission_attempts": ("submission_attempts", "sum"),
        "total_control_seconds": ("control_seconds", "sum"),
        "control_differential_seconds": ("control_seconds", "difference"),
    }
    for output, (source, operation) in pairs.items():
        red = pd.to_numeric(result[f"actual_red_{source}"], errors="coerce")
        blue = pd.to_numeric(result[f"actual_blue_{source}"], errors="coerce")
        result[f"actual_{output}"] = red + blue if operation == "sum" else red - blue
    return result


def _bounded_backtest_input(
    physical: pd.DataFrame,
    *,
    first_year: int,
    last_year: int,
    max_fights: int,
) -> pd.DataFrame:
    bounded = physical.copy()
    bounded["_backtest_selected"] = False
    candidates = physical.loc[
        physical["date"].dt.year.between(first_year, last_year)
    ]
    by_year = {
        int(year): group
        for year, group in candidates.groupby(candidates["date"].dt.year, sort=True)
    }
    selected_indices: list[object] = []
    depth = 0
    while len(selected_indices) < max_fights:
        added = False
        for year in sorted(by_year):
            if depth < len(by_year[year]):
                selected_indices.append(by_year[year].index[depth])
                added = True
                if len(selected_indices) >= max_fights:
                    break
        if not added:
            break
        depth += 1
    if selected_indices:
        bounded.loc[selected_indices, "_backtest_selected"] = True
    return bounded.sort_values(
        ["date", "event_id", "fight_id"], kind="stable"
    ).reset_index(drop=True)


def _forecast_with_full_support(value: Mapping[str, object]) -> dict[str, object]:
    forecast = dict(value)
    probabilities = {
        str(key): float(item)
        for key, item in dict(forecast.get("outcome_probabilities") or {}).items()
    }
    for outcome in ALL_TERMINAL_OUTCOMES:
        probabilities.setdefault(outcome, 0.0)
    forecast["outcome_probabilities"] = probabilities
    return forecast


def _aligned_red_probability(
    row: Mapping[str, object], red_fighter_id: str, probability_field: str
) -> float:
    probability = float(row[probability_field])
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(f"{probability_field} must be strictly between zero and one")
    fighter_id = _identity_token(row.get("fighter_id"))
    opponent_id = _identity_token(row.get("opponent_id"))
    if fighter_id == red_fighter_id:
        return probability
    if opponent_id == red_fighter_id:
        return 1.0 - probability
    raise ValueError("baseline fighter identities do not contain the red fighter")


def _assign_if_covered(
    frame: pd.DataFrame,
    column: str,
    values: Sequence[float],
    *,
    missing_warning: str,
) -> bool:
    numeric = pd.to_numeric(pd.Series(values, index=frame.index), errors="coerce")
    covered = numeric.map(lambda value: math.isfinite(float(value)))
    if not covered.any():
        _add_baseline_warning(frame, missing_warning)
        return False
    frame[column] = numeric.astype(float)
    return True


def _first_existing(root: Path, names: Sequence[str]) -> Path | None:
    return next((root / name for name in names if (root / name).is_file()), None)


def _validated_csv_records(
    path: Path,
    record_type: object,
    frame: pd.DataFrame,
    *,
    label: str,
) -> list[object]:
    """Read one exact append-only market schema without pandas NA coercion."""

    expected = tuple(getattr(record_type, "FIELDNAMES"))
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != expected:
                _add_baseline_warning(frame, f"{label}_unsupported_schema")
                return []
            records: list[object] = []
            invalid = 0
            for row in reader:
                try:
                    records.append(record_type.from_mapping(row))  # type: ignore[attr-defined]
                except Exception:
                    invalid += 1
            if invalid:
                _add_baseline_warning(frame, f"{label}_invalid_rows:{invalid}")
            return records
    except (OSError, UnicodeError, csv.Error):
        _add_baseline_warning(frame, f"{label}_unreadable")
        return []


def _market_record_matches(
    source: Mapping[str, object], physical: Mapping[str, object]
) -> bool:
    source_fight = _identity_token(source.get("fight_id"))
    physical_fight = _identity_token(physical.get("fight_id"))
    source_ids = {
        _identity_token(source.get("fighter_id")),
        _identity_token(source.get("opponent_id")),
    }
    physical_ids = {
        _identity_token(physical.get("red_fighter_id")),
        _identity_token(physical.get("blue_fighter_id")),
    }
    try:
        physical_date = pd.Timestamp(physical["date"]).date().isoformat()
    except (KeyError, TypeError, ValueError):
        return False
    return (
        _identity_token(source.get("matchup_id"))
        == _identity_token(physical.get("matchup_id"))
        and _identity_token(source.get("event_id"))
        == _identity_token(physical.get("event_id"))
        and source_ids == physical_ids
        and str(source.get("event_date") or "") == physical_date
        and (not source_fight or source_fight == physical_fight)
    )


def attach_timestamped_market_baselines(
    physical: pd.DataFrame, market_directory: str | Path
) -> pd.DataFrame:
    """Attach schema-validated, aligned captures that prove they are pre-event."""

    from market_tracker import (
        ForecastCapture,
        QuoteSnapshot,
        TotalRoundsQuoteSnapshot,
        consensus_as_of,
    )

    result = physical.copy()
    _copy_baseline_warnings(physical, result)
    result["matchup_id"] = [
        matchup_id_for(row.event_id, row.red_fighter_id, row.blue_fighter_id)
        for row in result.itertuples()
    ]
    root = Path(market_directory)
    forecast_path = _first_existing(
        root, ("forecast_captures.csv", "legacy_forecasts.csv")
    )
    if forecast_path is None:
        _add_baseline_warning(result, "forecast_captures_missing")
    else:
        captures = _validated_csv_records(
            forecast_path, ForecastCapture, result, label="forecast_captures"
        )
        by_matchup: dict[str, list[object]] = {}
        for capture in captures:
            by_matchup.setdefault(str(capture.matchup_id), []).append(capture)
        probabilities: list[float] = []
        for physical_row in result.to_dict("records"):
            candidates = [
                item
                for item in by_matchup.get(str(physical_row["matchup_id"]), ())
                if _market_record_matches(item.to_mapping(), physical_row)
            ]
            if not candidates:
                probabilities.append(math.nan)
                continue
            latest = max(
                candidates,
                key=lambda item: (
                    item.forecast_issued_at_utc,
                    item.forecast_capture_id,
                ),
            )
            probabilities.append(
                _aligned_red_probability(
                    latest.to_mapping(),
                    str(physical_row["red_fighter_id"]),
                    "model_probability",
                )
            )
        _assign_if_covered(
            result,
            "production_red_win_probability",
            probabilities,
            missing_warning="forecast_captures_no_aligned_pre_event_coverage",
        )

    quote_path = _first_existing(root, ("quote_snapshots.csv", "market_quotes.csv"))
    if quote_path is None:
        _add_baseline_warning(result, "moneyline_quotes_missing")
    else:
        quotes = _validated_csv_records(
            quote_path, QuoteSnapshot, result, label="moneyline_quotes"
        )
        groups: dict[tuple[str, str], list[object]] = {}
        for quote in quotes:
            groups.setdefault((quote.matchup_id, quote.capture_id), []).append(quote)
        by_matchup: dict[str, list[list[object]]] = {}
        for (matchup, _capture), group in groups.items():
            by_matchup.setdefault(matchup, []).append(group)
        probabilities = []
        for physical_row in result.to_dict("records"):
            markets = []
            for group in by_matchup.get(str(physical_row["matchup_id"]), ()):
                if not group or not all(
                    _market_record_matches(item.to_mapping(), physical_row)
                    for item in group
                ):
                    continue
                if len({item.book.casefold() for item in group}) < MONEYLINE_MINIMUM_BOOKS:
                    continue
                observed = {item.observed_at_utc for item in group}
                if len(observed) != 1:
                    continue
                try:
                    market = consensus_as_of(
                        group,
                        capture_id=group[0].capture_id,
                        matchup_id=group[0].matchup_id,
                        as_of_utc=next(iter(observed)),
                        min_books=MONEYLINE_MINIMUM_BOOKS,
                    )
                except Exception as error:
                    _add_baseline_warning(
                        result,
                        f"moneyline_consensus_failed:{type(error).__name__}",
                    )
                    continue
                markets.append(market)
            if not markets:
                probabilities.append(math.nan)
                continue
            latest = max(
                markets, key=lambda item: (item.as_of_utc, item.capture_id)
            )
            probabilities.append(
                _aligned_red_probability(
                    latest.to_mapping(),
                    str(physical_row["red_fighter_id"]),
                    "no_vig_fighter_probability",
                )
            )
        _assign_if_covered(
            result,
            "market_red_win_probability",
            probabilities,
            missing_warning="moneyline_quotes_no_aligned_pre_event_coverage",
        )

    total_path = _first_existing(root, ("total_round_quote_snapshots.csv",))
    if total_path is None:
        _add_baseline_warning(result, "total_round_quotes_missing")
    else:
        total_quotes = _validated_csv_records(
            total_path,
            TotalRoundsQuoteSnapshot,
            result,
            label="total_round_quotes",
        )
        total_groups: dict[tuple[str, str, float], list[object]] = {}
        for quote in total_quotes:
            total_groups.setdefault(
                (quote.matchup_id, quote.capture_id, float(quote.line)), []
            ).append(quote)
        total_by_matchup: dict[str, list[list[object]]] = {}
        for (matchup, _capture, _line), group in total_groups.items():
            total_by_matchup.setdefault(matchup, []).append(group)
        lines: list[float] = []
        over_probabilities: list[float] = []
        for physical_row in result.to_dict("records"):
            candidates: list[tuple[str, int, float, str, float]] = []
            scheduled = int(physical_row["scheduled_rounds"])
            for group in total_by_matchup.get(str(physical_row["matchup_id"]), ()):
                if not group or not all(
                    _market_record_matches(item.to_mapping(), physical_row)
                    for item in group
                ):
                    continue
                line = float(group[0].line)
                doubled = round(line * 2.0)
                if (
                    abs(line * 2.0 - doubled) > 1e-9
                    or doubled % 2 != 1
                    or not 0.0 < line < scheduled
                ):
                    continue
                books = {item.source_book_key.casefold() for item in group}
                observed = {item.observed_at_utc for item in group}
                if len(books) < TOTAL_MINIMUM_BOOKS or len(observed) != 1:
                    continue
                probability = float(
                    median(item.no_vig_over_probability for item in group)
                )
                candidates.append(
                    (
                        next(iter(observed)),
                        len(books),
                        line,
                        str(group[0].capture_id),
                        probability,
                    )
                )
            if not candidates:
                lines.append(math.nan)
                over_probabilities.append(math.nan)
                continue
            latest = max(
                candidates,
                key=lambda item: (item[0], item[1], -item[2], item[3]),
            )
            lines.append(latest[2])
            over_probabilities.append(latest[4])
        if _assign_if_covered(
            result,
            "market_total_over_probability",
            over_probabilities,
            missing_warning="total_round_quotes_no_aligned_pre_event_coverage",
        ):
            result["market_total_line_rounds"] = pd.Series(
                lines, index=result.index, dtype=float
            )
    return result


def attach_chronological_model_baselines(
    physical: pd.DataFrame,
    *,
    raw: pd.DataFrame,
    profiles: pd.DataFrame,
    point_in_time_path: str | Path,
    years: Sequence[int],
) -> pd.DataFrame:
    """Reconstruct incumbent and competing-risk predictions inside each fold."""

    result = physical.copy()
    _copy_baseline_warnings(physical, result)
    try:
        point = pd.read_csv(point_in_time_path, low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError):
        _add_baseline_warning(result, "point_in_time_unreadable")
        return result
    required = {
        "date",
        "event_id",
        "fight_id",
        "fighter_id",
        "opponent_id",
        "target",
        "bout_order",
        "label_method",
    }
    if not required <= set(point.columns):
        _add_baseline_warning(result, "point_in_time_unsupported_schema")
        return result
    point["date"] = pd.to_datetime(point["date"], errors="coerce", format="mixed")
    invalid_dates = int(point["date"].isna().sum())
    if invalid_dates:
        _add_baseline_warning(
            result, f"point_in_time_invalid_dates:{invalid_dates}"
        )
        point = point.loc[point["date"].notna()].copy()
    duplicate_ids = point["fight_id"].astype(str).duplicated(keep=False)
    if duplicate_ids.any():
        _add_baseline_warning(
            result,
            f"point_in_time_duplicate_fight_ids:{int(duplicate_ids.sum())}",
        )
        point = point.loc[~duplicate_ids].copy()
    if point.empty:
        _add_baseline_warning(result, "point_in_time_no_usable_rows")
        return result

    physical_index = {
        str(row["fight_id"]): (index, row)
        for index, row in result.to_dict("index").items()
    }

    def aligned_point_row(source: Mapping[str, object]) -> tuple[object, dict[str, object]] | None:
        matched = physical_index.get(str(source.get("fight_id")))
        if matched is None:
            return None
        index, physical_row = matched
        source_ids = {
            _identity_token(source.get("fighter_id")),
            _identity_token(source.get("opponent_id")),
        }
        physical_ids = {
            _identity_token(physical_row.get("red_fighter_id")),
            _identity_token(physical_row.get("blue_fighter_id")),
        }
        try:
            same_date = pd.Timestamp(source["date"]).date() == pd.Timestamp(
                physical_row["date"]
            ).date()
        except (KeyError, TypeError, ValueError):
            return None
        if (
            source_ids != physical_ids
            or _identity_token(source.get("event_id"))
            != _identity_token(physical_row.get("event_id"))
            or not same_date
        ):
            return None
        return index, physical_row

    incumbent_values = pd.Series(math.nan, index=result.index, dtype=float)
    if "production_red_win_probability" in result:
        incumbent_values = pd.to_numeric(
            result["production_red_win_probability"], errors="coerce"
        ).astype(float)
    outcome_forecasts: dict[str, dict[str, float]] = {}
    outcome_values = pd.Series(math.nan, index=result.index, dtype=float)

    from fight_predictor.outcome_model import DiscreteTimeOutcomeModel, evaluate_outcome_model
    from fight_predictor.point_in_time import PointInTimeDatasetBuilder, TemporalFightPredictor

    try:
        builder = PointInTimeDatasetBuilder(raw, profiles)
        incumbent = TemporalFightPredictor(point, builder)
    except Exception as error:
        incumbent = None
        _add_baseline_warning(
            result, f"incumbent_initialization_failed:{type(error).__name__}"
        )
    if incumbent is not None:
        for year in years:
            try:
                predictions = incumbent.walk_forward_predictions((int(year),))
            except Exception as error:
                _add_baseline_warning(
                    result,
                    f"incumbent_fold_{int(year)}_failed:{type(error).__name__}",
                )
                continue
            needed = {
                "date",
                "event_id",
                "fight_id",
                "fighter_id",
                "opponent_id",
                "model_probability",
                "training_through",
            }
            if not needed <= set(predictions.columns):
                _add_baseline_warning(
                    result, f"incumbent_fold_{int(year)}_unsupported_output"
                )
                continue
            if predictions["fight_id"].astype(str).duplicated().any():
                _add_baseline_warning(
                    result, f"incumbent_fold_{int(year)}_duplicate_fight_ids"
                )
                continue
            cutoff = pd.Timestamp(f"{int(year)}-01-01")
            for source in predictions.to_dict("records"):
                aligned = aligned_point_row(source)
                if aligned is None:
                    continue
                index, physical_row = aligned
                trained = pd.to_datetime(
                    source.get("training_through"), errors="coerce"
                )
                if pd.isna(trained) or not trained < cutoff:
                    _add_baseline_warning(
                        result, f"incumbent_fold_{int(year)}_noncausal_output"
                    )
                    continue
                try:
                    probability = _aligned_red_probability(
                        source,
                        str(physical_row["red_fighter_id"]),
                        "model_probability",
                    )
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(float(incumbent_values.loc[index])):
                    incumbent_values.loc[index] = probability
    if incumbent_values.map(lambda value: math.isfinite(float(value))).any():
        result["production_red_win_probability"] = incumbent_values
    else:
        if "production_red_win_probability" in result:
            result.pop("production_red_win_probability")
        _add_baseline_warning(result, "incumbent_no_causal_aligned_coverage")

    feature_columns = tuple(column for column in point if column.endswith("_diff"))
    if not feature_columns:
        _add_baseline_warning(result, "outcome_model_no_feature_columns")
        return result
    for year in years:
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = point.loc[point["date"] < cutoff].copy()
        selected_ids = set(
            result.loc[
                result["date"].dt.year.eq(year)
                & result["_backtest_selected"].astype(bool),
                "fight_id",
            ].astype(str)
        )
        test = point.loc[point["fight_id"].astype(str).isin(selected_ids)]
        if len(train) < 1000 or test.empty:
            if selected_ids:
                _add_baseline_warning(
                    result, f"outcome_fold_{int(year)}_insufficient_training_or_test"
                )
            continue
        try:
            model, _report = evaluate_outcome_model(train, feature_columns)
        except Exception as error:
            _add_baseline_warning(
                result,
                f"outcome_fold_{int(year)}_failed:{type(error).__name__}",
            )
            continue
        for _, source in test.iterrows():
            fight_id = str(source["fight_id"])
            aligned = aligned_point_row(source.to_dict())
            if aligned is None:
                _add_baseline_warning(
                    result, f"outcome_fold_{int(year)}_unaligned_test_row"
                )
                continue
            index, physical_row = aligned
            rounds = int(physical_row["scheduled_rounds"])
            try:
                prediction = model.predict(source, rounds)
            except Exception as error:
                _add_baseline_warning(
                    result,
                    f"outcome_fold_{int(year)}_prediction_failed:{type(error).__name__}",
                )
                continue
            fighter_is_red = _identity_token(source["fighter_id"]) == _identity_token(
                physical_row["red_fighter_id"]
            )
            mapped: dict[str, float] = {}
            for key, probability in prediction.terminal_probabilities.items():
                try:
                    side, method = key.split("_", 1)
                except ValueError:
                    continue
                red_side = (side == "fighter") == fighter_is_red
                mapped[f"{'red' if red_side else 'blue'}_{method}"] = float(probability)
            for outcome in ALL_TERMINAL_OUTCOMES:
                mapped.setdefault(outcome, 0.0)
            if (
                any(not math.isfinite(value) or value < 0.0 for value in mapped.values())
                or abs(sum(mapped.values()) - 1.0) > 1e-8
            ):
                _add_baseline_warning(
                    result, f"outcome_fold_{int(year)}_invalid_probability_output"
                )
                continue
            outcome_forecasts[fight_id] = mapped
            red_probability = sum(
                probability for key, probability in mapped.items() if key.startswith("red_")
            )
            outcome_values.loc[index] = red_probability
    if outcome_values.map(lambda value: math.isfinite(float(value))).any():
        result["outcome_model_red_win_probability"] = outcome_values
        result["outcome_model_forecast"] = [
            outcome_forecasts.get(str(fight_id)) for fight_id in result["fight_id"]
        ]
    else:
        _add_baseline_warning(result, "outcome_model_no_causal_aligned_coverage")
    return result


def _atomic_write_ledger(path: str | Path, ledger: pd.DataFrame) -> None:
    lines = []
    for row in ledger.to_dict("records"):
        lines.append(
            json.dumps(
                _json_ledger_value(row),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
        )
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _atomic_write_jsonl_gzip(
    path: str | Path, rows: Iterable[Mapping[str, object]]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                for row in rows:
                    line = json.dumps(
                        _json_ledger_value(dict(row)),
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                        default=_json_default,
                        separators=(",", ":"),
                    )
                    compressed.write(line.encode("utf-8") + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


_BASELINE_OUTCOME_CATEGORIES = (
    "ko_tko",
    "submission",
    "decision",
    "other",
    "draw",
    "no_contest",
)


def _baseline_category(outcome: object) -> str | None:
    value = str(outcome or "").strip().lower()
    if value in {"draw", "no_contest"}:
        return value
    for side in ("red_", "blue_"):
        if value.startswith(side):
            method = value[len(side) :]
            return method if method in _BASELINE_OUTCOME_CATEGORIES[:4] else None
    return None


def _joint_forecast_from_category_mass(
    mass: Mapping[str, float]
) -> dict[str, object]:
    outcomes: dict[str, float] = {}
    for method in _BASELINE_OUTCOME_CATEGORIES[:4]:
        probability = float(mass[method]) / 2.0
        outcomes[f"red_{method}"] = probability
        outcomes[f"blue_{method}"] = probability
    outcomes["draw"] = float(mass["draw"])
    outcomes["no_contest"] = float(mass["no_contest"])
    if abs(sum(outcomes.values()) - 1.0) > 1e-12:
        raise RuntimeError("causal baseline probabilities do not sum to one")
    return {"outcome_probabilities": outcomes}


def causal_joint_baseline_forecasts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    smoothing: float = 1.0,
) -> pd.DataFrame:
    """Build causal, side-symmetric population and pooled-division priors."""

    if not math.isfinite(smoothing) or smoothing <= 0.0:
        raise ValueError("baseline smoothing must be finite and positive")
    if "actual_outcome" not in train or "fight_id" not in test:
        raise ValueError("causal baselines require training outcomes and test fight IDs")

    def counts(frame: pd.DataFrame) -> dict[str, float]:
        values = {category: 0.0 for category in _BASELINE_OUTCOME_CATEGORIES}
        for outcome in frame["actual_outcome"]:
            category = _baseline_category(outcome)
            if category is not None:
                values[category] += 1.0
        return values

    population_counts = counts(train)
    population_denominator = sum(population_counts.values()) + smoothing * len(
        _BASELINE_OUTCOME_CATEGORIES
    )
    population_mass = {
        category: (population_counts[category] + smoothing)
        / population_denominator
        for category in _BASELINE_OUTCOME_CATEGORIES
    }
    population = _joint_forecast_from_category_mass(population_mass)
    division_prior_weight = smoothing * len(_BASELINE_OUTCOME_CATEGORIES)
    rows: list[dict[str, object]] = []
    for source in test.to_dict("records"):
        if "division" in train and source.get("division") is not None:
            division_train = train.loc[
                train["division"].astype(str).eq(str(source["division"]))
            ]
        else:
            division_train = train.iloc[0:0]
        division_counts = counts(division_train)
        division_denominator = sum(division_counts.values()) + division_prior_weight
        division_mass = {
            category: (
                division_counts[category]
                + division_prior_weight * population_mass[category]
            )
            / division_denominator
            for category in _BASELINE_OUTCOME_CATEGORIES
        }
        rows.append(
            {
                "fight_id": source["fight_id"],
                "population_forecast": population,
                "division_forecast": _joint_forecast_from_category_mass(
                    division_mass
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=("fight_id", "population_forecast", "division_forecast"),
    )


def _repeat_forecast_sha256(ledger: pd.DataFrame) -> str:
    records = [
        {"fight_id": str(row["fight_id"]), "forecast": row["forecast"]}
        for row in ledger.sort_values("fight_id", kind="stable").to_dict("records")
    ]
    return canonical_sha256(records)


def _compact_evaluation_forecast(forecast: object) -> dict[str, object]:
    """Keep the exact aggregate evaluation ledger without member histograms.

    Historical scoring consumes aggregate statistic distributions, conditional
    member means/intervals, and exact per-member outcome counts.  Retaining the
    much larger per-member statistic support for every fight and independent
    seed adds no evaluation information and defeats the compact-ledger
    contract, so it is committed only by its deterministic local hash.
    """

    value = forecast.to_dict() if hasattr(forecast, "to_dict") else dict(forecast)
    authority_hash = canonical_sha256(value)
    value.pop("bootstrap_statistic_distributions", None)
    value["evaluation_detail_level"] = "compact_backtest_v1"
    value["local_aggregate_sha256"] = authority_hash
    value["omitted_local_authority_fields"] = [
        "bootstrap_statistic_distributions"
    ]
    return value


def _simulation_noise_summary(
    ledgers: Sequence[pd.DataFrame],
    *,
    paths_per_matchup: int,
    random_seed: int,
    stack_comparisons: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    summary = repeated_seed_summary(ledgers)
    summary.update(
        {
            "schema_version": 1,
            "authoritative_repeat": 1,
            "paths_per_matchup": int(paths_per_matchup),
            "root_seed_contract": (
                "repeat 1: backtest:{random_seed}:{fight_id}; repeat N: "
                "backtest:{random_seed}:repeat:{N}:{fight_id}"
            ),
            "random_seed": int(random_seed),
            "repeat_forecast_sha256": [
                _repeat_forecast_sha256(ledger) for ledger in ledgers
            ],
        }
    )
    evaluated_stacks = [
        value for value in stack_comparisons if value.get("status") == "evaluated"
    ]
    if evaluated_stacks:
        stack_losses = [
            float(dict(value["stack"])["log_loss"]) for value in evaluated_stacks
        ]
        summary["winner_stack"] = {
            "evaluated_repeats": len(evaluated_stacks),
            "winner_log_loss_by_repeat": stack_losses,
            "winner_log_loss_range": float(max(stack_losses) - min(stack_losses)),
            "fold_coefficients_sha256_by_repeat": [
                canonical_sha256(value.get("folds") or [])
                for value in evaluated_stacks
            ],
        }
    return summary


def _borderline_joint_comparisons(report: BacktestReport) -> tuple[str, ...]:
    borderline: list[str] = []
    for name in (
        "competing_risk_joint",
        "population_joint",
        "division_joint",
    ):
        comparison = report.comparisons.get(name)
        if not isinstance(comparison, Mapping):
            continue
        interval = comparison.get("paired_event_card_interval")
        if not isinstance(interval, Mapping):
            continue
        try:
            lower = float(interval["interval_p025"])
            upper = float(interval["interval_p975"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(lower) and math.isfinite(upper) and lower <= 0.0 <= upper:
            borderline.append(name)
    return tuple(borderline)


def _is_default_repository_backtest(
    *,
    raw_path: str | Path,
    profiles_path: str | Path,
    round_path: str | Path,
    bootstrap_members: int,
    paths_per_matchup: int,
    first_test_year: int,
    last_test_year: int,
    max_fights: int,
    min_training_fights: int,
    random_seed: int,
    point_in_time_path: str | Path | None,
    market_directory: str | Path | None,
    include_baselines: bool,
) -> bool:
    """Limit automatic precision escalation to the reviewed default study."""

    return (
        Path(raw_path).resolve() == DEFAULT_RAW_FIGHTS.resolve()
        and Path(profiles_path).resolve() == DEFAULT_FIGHTER_PROFILES.resolve()
        and Path(round_path).resolve() == DEFAULT_ROUND_STATS.resolve()
        and bootstrap_members == 64
        and paths_per_matchup == 4096
        and first_test_year == 2017
        and last_test_year == 2026
        and max_fights == 200
        and min_training_fights == 500
        and random_seed == 2903
        and point_in_time_path is None
        and market_directory is None
        and include_baselines
    )


def _with_backtest_research_metadata(
    report: BacktestReport,
    *,
    simulation_noise: Mapping[str, object],
    baseline_warnings: Sequence[str],
    precision: Mapping[str, object],
    seed_repeats: int,
) -> BacktestReport:
    combined_warnings = tuple(
        dict.fromkeys((*report.coverage_warnings, *map(str, baseline_warnings)))
    )
    config = dict(report.config)
    config.update(
        {
            "seed_repeats": int(seed_repeats),
            "precision": dict(precision),
        }
    )
    unhashed = replace(
        report,
        config=config,
        coverage_warnings=combined_warnings,
        simulation_noise=dict(simulation_noise),
        report_sha256="",
    )
    return replace(
        unhashed,
        report_sha256=canonical_sha256(unhashed.unhashed_dict()),
    ).validate()


def execute_backtest(
    *,
    raw_path: str | Path = DEFAULT_RAW_FIGHTS,
    profiles_path: str | Path = DEFAULT_FIGHTER_PROFILES,
    round_path: str | Path = DEFAULT_ROUND_STATS,
    output: str | Path = DEFAULT_ARTIFACT_ROOT / "backtest-summary.json",
    ledger_output: str | Path | None = None,
    bootstrap_members: int = 64,
    paths_per_matchup: int = 4096,
    first_test_year: int = 2017,
    last_test_year: int = 2026,
    max_fights: int = 200,
    min_training_fights: int = 500,
    random_seed: int = 2903,
    workers: int = 1,
    chunk_size: int = 64,
    point_in_time_path: str | Path | None = None,
    market_directory: str | Path | None = None,
    include_baselines: bool = True,
    seed_repeats: int = 2,
    skip_borderline_rerun: bool = False,
    stack_min_training_fights: int = 100,
    stack_l2_penalty: float = 0.01,
) -> tuple[pd.DataFrame, BacktestReport]:
    if paths_per_matchup % bootstrap_members:
        raise ValueError("paths_per_matchup must be divisible by bootstrap_members")
    if not 2 <= seed_repeats <= 4:
        raise ValueError("seed_repeats must be between 2 and 4")
    raw, profiles, rounds = load_research_inputs(raw_path, profiles_path, round_path)
    physical = _bounded_backtest_input(
        physical_backtest_frame(raw),
        first_year=first_test_year,
        last_year=last_test_year,
        max_fights=max_fights,
    )
    if include_baselines:
        effective_market = (
            Path(market_directory)
            if market_directory is not None
            else (
                DEFAULT_MARKET_DIRECTORY
                if Path(raw_path).resolve() == DEFAULT_RAW_FIGHTS.resolve()
                else None
            )
        )
        if effective_market is not None and effective_market.is_dir():
            physical = attach_timestamped_market_baselines(
                physical, effective_market
            )
        elif effective_market is not None:
            _add_baseline_warning(physical, "market_directory_missing")
        effective_point = (
            Path(point_in_time_path)
            if point_in_time_path is not None
            else (
                DEFAULT_POINT_IN_TIME
                if Path(raw_path).resolve() == DEFAULT_RAW_FIGHTS.resolve()
                else None
            )
        )
        if effective_point is not None and effective_point.is_file() and profiles is not None:
            physical = attach_chronological_model_baselines(
                physical,
                raw=raw,
                profiles=profiles,
                point_in_time_path=effective_point,
                years=tuple(range(first_test_year, last_test_year + 1)),
            )
        elif effective_point is not None:
            _add_baseline_warning(physical, "point_in_time_missing_or_profiles_unavailable")
    baseline_warnings = tuple(_baseline_warnings(physical))
    artifact_cache: dict[
        str, tuple[CausalParameterFitter, ParameterEnsembleArtifact]
    ] = {}
    backtest_config = BacktestConfig(
        first_test_year=first_test_year,
        last_test_year=last_test_year,
        min_training_fights=min_training_fights,
        card_bootstrap_replicates=2000,
        random_seed=random_seed,
        stack_min_training_fights=stack_min_training_fights,
        stack_l2_penalty=stack_l2_penalty,
    )

    def fitted_fold(
        cutoff: pd.Timestamp,
    ) -> tuple[CausalParameterFitter, ParameterEnsembleArtifact]:
        key = cutoff.isoformat()
        cached = artifact_cache.get(key)
        if cached is not None:
            return cached
        fitter = CausalParameterFitter(raw, profiles, rounds)
        artifact = fitter.fit(
            cutoff,
            config=ParameterFitConfig.historical(
                bootstrap_members=bootstrap_members,
                random_seed=random_seed + int(cutoff.year),
            ),
            created_at_utc=cutoff,
        )
        artifact_cache[key] = (fitter, artifact)
        return fitter, artifact

    def run_precision(
        total_paths_per_matchup: int,
    ) -> tuple[pd.DataFrame, BacktestReport, dict[str, object]]:
        if total_paths_per_matchup % bootstrap_members:
            raise ValueError(
                "precision path count must be divisible by bootstrap_members"
            )
        inner_paths = total_paths_per_matchup // bootstrap_members
        repeat_forecasts: list[dict[str, dict[str, object]]] = [
            {} for _ in range(seed_repeats)
        ]

        def predict_fold(
            train: pd.DataFrame, test: pd.DataFrame, cutoff: pd.Timestamp
        ) -> pd.DataFrame:
            fitter, artifact = fitted_fold(cutoff)
            predictions: list[dict[str, object]] = []
            rows = test.sort_values(
                ["date", "event_id", "fight_id"], kind="stable"
            ).to_dict("records")
            for row in rows:
                fight_id = str(row["fight_id"])
                matchup = matchup_id_for(
                    row["event_id"],
                    row["red_fighter_id"],
                    row["blue_fighter_id"],
                )
                authoritative: dict[str, object] | None = None
                for repeat_index in range(seed_repeats):
                    repeat_number = repeat_index + 1
                    root_seed = (
                        f"backtest:{random_seed}:{fight_id}"
                        if repeat_number == 1
                        else (
                            f"backtest:{random_seed}:repeat:{repeat_number}:"
                            f"{fight_id}"
                        )
                    )
                    specs = build_specs(
                        fitter,
                        artifact,
                        red_fighter_id=str(row["red_fighter_id"]),
                        blue_fighter_id=str(row["blue_fighter_id"]),
                        division=str(row["division"]),
                        scheduled_rounds=int(row["scheduled_rounds"]),
                        event_id=str(row["event_id"]),
                        root_seed=root_seed,
                        matchup_id=matchup,
                    )
                    simulation = run_nested(
                        specs,
                        inner_paths,
                        workers=workers,
                        chunk_size=chunk_size,
                        max_traces=0,
                        retain_paths=False,
                    )
                    forecast = _forecast_with_full_support(
                        _compact_evaluation_forecast(simulation.forecast)
                    )
                    repeat_forecasts[repeat_index][fight_id] = forecast
                    if repeat_number == 1:
                        authoritative = forecast
                if authoritative is None:  # pragma: no cover - validated repeats
                    raise RuntimeError("authoritative seed repeat was not simulated")
                predictions.append(
                    {"fight_id": row["fight_id"], "forecast": authoritative}
                )
            forecast_rows = pd.DataFrame(predictions)
            causal_baselines = causal_joint_baseline_forecasts(train, test)
            return forecast_rows.merge(
                causal_baselines,
                on="fight_id",
                how="left",
                validate="one_to_one",
            )

        authoritative_ledger, precision_report = run_chronological_backtest(
            physical,
            predict_fold,
            config=backtest_config,
            test_filter_column="_backtest_selected",
        )
        repeat_ledgers: list[pd.DataFrame] = []
        repeat_stack_comparisons: list[dict[str, object]] = []
        for repeat_index, forecasts in enumerate(repeat_forecasts, start=1):
            missing = set(authoritative_ledger["fight_id"].astype(str)) - set(
                forecasts
            )
            if missing:
                raise RuntimeError(
                    f"seed repeat {repeat_index} omitted {len(missing)} forecasts"
                )
            repeated = authoritative_ledger.copy()
            repeated["forecast"] = [
                forecasts[str(fight_id)] for fight_id in repeated["fight_id"]
            ]
            repeated = add_simulation_win_probability_column(repeated)
            if "production_red_win_probability" in repeated:
                repeated, stack_comparison = evaluate_chronological_winner_stack(
                    repeated,
                    min_training_fights=backtest_config.stack_min_training_fights,
                    l2_penalty=backtest_config.stack_l2_penalty,
                    card_bootstrap_replicates=backtest_config.card_bootstrap_replicates,
                    random_seed=backtest_config.random_seed,
                )
                repeat_stack_comparisons.append(stack_comparison)
            repeat_ledgers.append(repeated)
        noise = _simulation_noise_summary(
            repeat_ledgers,
            paths_per_matchup=total_paths_per_matchup,
            random_seed=random_seed,
            stack_comparisons=repeat_stack_comparisons,
        )
        return authoritative_ledger, precision_report, noise

    ledger, report, simulation_noise = run_precision(paths_per_matchup)
    borderline = _borderline_joint_comparisons(report)
    default_repository_run = _is_default_repository_backtest(
        raw_path=raw_path,
        profiles_path=profiles_path,
        round_path=round_path,
        bootstrap_members=bootstrap_members,
        paths_per_matchup=paths_per_matchup,
        first_test_year=first_test_year,
        last_test_year=last_test_year,
        max_fights=max_fights,
        min_training_fights=min_training_fights,
        random_seed=random_seed,
        point_in_time_path=point_in_time_path,
        market_directory=market_directory,
        include_baselines=include_baselines,
    )
    should_rerun = (
        default_repository_run
        and not skip_borderline_rerun
        and paths_per_matchup < 16384
        and bool(borderline)
    )
    precision: dict[str, object] = {
        "initial_paths_per_matchup": int(paths_per_matchup),
        "final_paths_per_matchup": int(paths_per_matchup),
        "automatic_borderline_rerun_eligible": default_repository_run,
        "automatic_borderline_rerun_skipped": bool(skip_borderline_rerun),
        "borderline_comparisons": list(borderline),
        "borderline_rerun_triggered": False,
    }
    if should_rerun:
        initial = _with_backtest_research_metadata(
            report,
            simulation_noise=simulation_noise,
            baseline_warnings=baseline_warnings,
            precision=precision,
            seed_repeats=seed_repeats,
        )
        ledger, report, simulation_noise = run_precision(16384)
        precision.update(
            {
                "final_paths_per_matchup": 16384,
                "borderline_rerun_triggered": True,
                "initial_report_sha256": initial.report_sha256,
                "initial_simulation_noise_sha256": canonical_sha256(
                    initial.simulation_noise
                ),
            }
        )
    report = _with_backtest_research_metadata(
        report,
        simulation_noise=simulation_noise,
        baseline_warnings=baseline_warnings,
        precision=precision,
        seed_repeats=seed_repeats,
    )
    write_backtest_report(output, report)
    if ledger_output is not None:
        _atomic_write_ledger(ledger_output, ledger)
    return ledger, report


def load_specs(path: str | Path) -> tuple[SimulationRunSpec, ...]:
    value = load_json(path)
    raw_specs = value.get("specs")
    if raw_specs is None:
        raw_specs = [value]
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("spec file must contain one or more specs")
    return tuple(SimulationRunSpec.from_dict(dict(item)) for item in raw_specs)


def select_spec(
    specs: Sequence[SimulationRunSpec], bootstrap_member: int | None
) -> SimulationRunSpec:
    if bootstrap_member is None:
        if len(specs) != 1:
            raise ValueError("spec file has multiple members; select --bootstrap-member")
        return specs[0]
    matches = [spec for spec in specs if spec.bootstrap_member == bootstrap_member]
    if len(matches) != 1:
        raise ValueError(f"bootstrap member {bootstrap_member} is absent or duplicated")
    return matches[0]


def execute_replay(
    *,
    trace_path: str | Path | None = None,
    spec_path: str | Path | None = None,
    simulation_index: int | None = None,
    bootstrap_member: int | None = None,
    expected_trace_path: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, object]:
    if (trace_path is None) == (spec_path is None):
        raise ValueError("replay requires exactly one of trace_path or spec_path")
    if trace_path is not None:
        path = trace_from_dict(load_json(trace_path), verify=True)
        digest = replay_trace(path)
        result: dict[str, object] = {
            "mode": "reducer",
            "verified": True,
            "matchup_id": path.matchup_id,
            "bootstrap_member": path.bootstrap_member,
            "simulation_index": path.simulation_index,
            "final_state_hash": digest,
        }
    else:
        if simulation_index is None:
            raise ValueError("stochastic replay requires simulation_index")
        spec = select_spec(load_specs(spec_path), bootstrap_member)  # type: ignore[arg-type]
        expected = (
            trace_from_dict(load_json(expected_trace_path), verify=True)
            if expected_trace_path is not None
            else None
        )
        regenerated = stochastic_replay(
            spec, simulation_index, expected=expected
        )
        result = {
            "mode": "stochastic",
            "verified": True,
            "matchup_id": regenerated.matchup_id,
            "bootstrap_member": regenerated.bootstrap_member,
            "simulation_index": regenerated.simulation_index,
            "final_state_hash": regenerated.final_state_hash,
            "trace": trace_to_dict(regenerated),
        }
    if output is not None:
        atomic_write_json(output, result)
    return result


def execute_reduce(
    trace_path: str | Path, *, output: str | Path | None = None
) -> dict[str, object]:
    path = trace_from_dict(load_json(trace_path), verify=True)
    digest = replay_trace(path)
    result = {
        "verified": True,
        "matchup_id": path.matchup_id,
        "scheduled_rounds": path.scheduled_rounds,
        "bootstrap_member": path.bootstrap_member,
        "simulation_index": path.simulation_index,
        "event_count": len(path.events),
        "final_state_hash": digest,
        "result": path.result.to_dict(),
        "red_stats": path.red_stats.to_dict(),
        "blue_stats": path.blue_stats.to_dict(),
    }
    if output is not None:
        atomic_write_json(output, result)
    return result


def execute_diff(
    expected_path: str | Path,
    actual_path: str | Path,
    *,
    output: str | Path | None = None,
) -> tuple[dict[str, object], bool]:
    expected = trace_from_dict(load_json(expected_path), verify=False)
    actual = trace_from_dict(load_json(actual_path), verify=False)
    difference = diff_event_streams(expected.events, actual.events)
    if difference is None and expected.final_state_hash == actual.final_state_hash:
        result: dict[str, object] = {"identical": True, "difference": None}
        differs = False
    else:
        if difference is None:
            detail = {
                "event_index": None,
                "field_path": "final_state_hash",
                "expected": expected.final_state_hash,
                "actual": actual.final_state_hash,
            }
        else:
            detail = asdict(difference)
        result = {"identical": False, "difference": detail}
        differs = True
    if output is not None:
        atomic_write_json(output, result)
    return result, differs


def _recent_complete_event_selection(
    physical: pd.DataFrame,
    *,
    last_events: int,
    min_prior_ufc_fights: int,
    skip_latest_events: int = 0,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, int]]:
    if last_events <= 0:
        raise ValueError("last_events must be positive")
    if min_prior_ufc_fights < 0:
        raise ValueError("min_prior_ufc_fights must be nonnegative")
    if skip_latest_events < 0:
        raise ValueError("skip_latest_events must be nonnegative")
    all_events = (
        physical[["date", "event_id"]]
        .drop_duplicates()
        .sort_values(["date", "event_id"], kind="stable")
    )
    eligible_window = (
        all_events.iloc[:-skip_latest_events]
        if skip_latest_events
        else all_events
    )
    events = eligible_window.tail(last_events)
    if len(events) != last_events:
        raise ValueError("not enough complete events for the requested selection window")
    selected = physical.merge(
        events.assign(_selected_event=True),
        on=["date", "event_id"],
        how="inner",
        validate="many_to_one",
    )
    exposure = (
        selected["red_prior_ufc_fights"].ge(min_prior_ufc_fights)
        & selected["blue_prior_ufc_fights"].ge(min_prior_ufc_fights)
    )
    eligible = selected.loc[exposure].copy()
    event_rows: list[dict[str, object]] = []
    for source in events.to_dict("records"):
        event_id = str(source["event_id"])
        date = pd.Timestamp(source["date"])
        all_card = selected.loc[
            selected["event_id"].astype(str).eq(event_id)
            & selected["date"].eq(date)
        ]
        card = eligible.loc[
            eligible["event_id"].astype(str).eq(event_id)
            & eligible["date"].eq(date)
        ]
        event_rows.append(
            {
                "event_id": event_id,
                "date": date.date().isoformat(),
                "card_fights": int(len(all_card)),
                "eligible_fights": int(len(card)),
                "excluded_low_exposure": int(len(all_card) - len(card)),
            }
        )
    return (
        eligible.sort_values(["date", "event_id", "fight_id"], kind="stable").reset_index(drop=True),
        event_rows,
        {
            "selected_card_fights": int(len(selected)),
            "eligible_fights": int(len(eligible)),
            "excluded_low_exposure": int((~exposure).sum()),
        },
    )


def _frozen_cohort_selection(
    physical: pd.DataFrame,
    *,
    manifest_path: str | Path,
    cohort_name: str,
    min_prior_ufc_fights: int,
    source_sha256: Mapping[str, str | None] | None = None,
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    dict[str, int],
    dict[str, object],
]:
    """Select and verify a named immutable research cohort.

    The event identities, exposure rule, expected fight count, and checksum of
    the sorted eligible fight IDs are all sealed in the tracked manifest.  A
    refreshed dataset therefore fails loudly instead of moving the window.
    """

    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported simulation cohort manifest schema")
    contract = dict(manifest.get("selection_contract") or {})
    declared_minimum = int(contract.get("min_prior_ufc_fights", -1))
    if declared_minimum != min_prior_ufc_fights:
        raise ValueError(
            "cohort manifest exposure rule differs from min_prior_ufc_fights"
        )
    declared_sources = dict(contract.get("source_sha256") or {})
    if source_sha256 is not None and declared_sources != dict(source_sha256):
        raise ValueError(
            "cohort manifest source fingerprints differ from the research inputs"
        )
    cohorts = dict(manifest.get("cohorts") or {})
    if cohort_name not in cohorts:
        raise ValueError(f"unknown frozen simulation cohort: {cohort_name}")
    cohort = dict(cohorts[cohort_name])
    event_values = list(cohort.get("events") or [])
    if not event_values:
        raise ValueError("frozen simulation cohort contains no events")
    events = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(str(item["date"]), utc=True),
                "event_id": str(item["event_id"]),
            }
            for item in event_values
        ]
    ).sort_values(["date", "event_id"], kind="stable")
    if events.duplicated(["date", "event_id"]).any():
        raise ValueError("frozen simulation cohort contains duplicate events")
    available = physical[["date", "event_id"]].drop_duplicates()
    checked = events.merge(
        available.assign(_available=True),
        on=["date", "event_id"],
        how="left",
        validate="one_to_one",
    )
    if not checked["_available"].fillna(False).all():
        missing = checked.loc[checked["_available"].isna(), "event_id"].tolist()
        raise ValueError(f"frozen simulation cohort events are missing: {missing}")
    selected = physical.merge(
        events.assign(_selected_event=True),
        on=["date", "event_id"],
        how="inner",
        validate="many_to_one",
    )
    exposure = (
        selected["red_prior_ufc_fights"].ge(min_prior_ufc_fights)
        & selected["blue_prior_ufc_fights"].ge(min_prior_ufc_fights)
    )
    eligible = selected.loc[exposure].copy()
    fight_ids = sorted(eligible["fight_id"].astype(str).tolist())
    expected_count = int(cohort["eligible_fights"])
    expected_hash = str(cohort["fight_ids_sha256"])
    actual_hash = canonical_sha256(fight_ids)
    if len(fight_ids) != expected_count or actual_hash != expected_hash:
        raise ValueError(
            "frozen simulation cohort fight identities changed: "
            f"expected {expected_count}/{expected_hash}, got "
            f"{len(fight_ids)}/{actual_hash}"
        )
    event_rows: list[dict[str, object]] = []
    for source in events.to_dict("records"):
        event_id = str(source["event_id"])
        date = pd.Timestamp(source["date"])
        all_card = selected.loc[
            selected["event_id"].astype(str).eq(event_id)
            & selected["date"].eq(date)
        ]
        card = eligible.loc[
            eligible["event_id"].astype(str).eq(event_id)
            & eligible["date"].eq(date)
        ]
        event_rows.append(
            {
                "event_id": event_id,
                "date": date.date().isoformat(),
                "card_fights": int(len(all_card)),
                "eligible_fights": int(len(card)),
                "excluded_low_exposure": int(len(all_card) - len(card)),
            }
        )
    metadata = {
        "cohort_name": cohort_name,
        "cohort_manifest_sha256": canonical_sha256(manifest),
        "fight_ids_sha256": actual_hash,
    }
    return (
        eligible.sort_values(
            ["date", "event_id", "fight_id"], kind="stable"
        ).reset_index(drop=True),
        event_rows,
        {
            "selected_card_fights": int(len(selected)),
            "eligible_fights": int(len(eligible)),
            "excluded_low_exposure": int((~exposure).sum()),
        },
        metadata,
    )


def _joint_log_loss(frame: pd.DataFrame, column: str) -> float | None:
    losses: list[float] = []
    for row in frame.to_dict("records"):
        forecast = row.get(column)
        if forecast is None:
            continue
        value = dict(forecast) if isinstance(forecast, Mapping) else {}
        probabilities = dict(value.get("outcome_probabilities") or value)
        probability = float(probabilities.get(str(row["actual_outcome"]), 0.0))
        losses.append(-math.log(max(probability, 1e-12)))
    return float(np.mean(losses)) if losses else None


_POSTERIOR_FIDELITIES = {"screen", "confirm", "final", "custom"}


def _posterior_checkpoint_path(
    destination: Path, fight_id: object, repeat_index: int
) -> Path:
    identity = canonical_sha256(
        {"fight_id": str(fight_id), "repeat_index": int(repeat_index)}
    )[:24]
    return destination / "checkpoints" / f"repeat-{repeat_index + 1:02d}" / f"{identity}.json.gz"


def _load_posterior_checkpoint(
    path: Path,
    *,
    run_contract_sha256: str,
    fight_id: object,
    repeat_index: int,
) -> dict[str, object]:
    value = _load_json_gzip(path)
    if value.get("schema_version") != 1:
        raise ValueError(f"unsupported posterior checkpoint schema: {path}")
    if value.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError(f"posterior checkpoint belongs to another run: {path}")
    if value.get("fight_id") != str(fight_id) or value.get("repeat_index") != repeat_index:
        raise ValueError(f"posterior checkpoint identity is invalid: {path}")
    record = value.get("record")
    if not isinstance(record, dict):
        raise ValueError(f"posterior checkpoint is missing its forecast record: {path}")
    return dict(record)


def _posterior_single_seed_summary(ledger: pd.DataFrame) -> dict[str, object]:
    metrics = evaluate_simulation_ledger(ledger)
    return {
        "seed_repeats": 1,
        "joint_log_loss_mean": float(
            metrics["primary_joint_side_method_log_loss"]
        ),
        "joint_log_loss_sd": None,
        "winner_log_loss_mean": float(dict(metrics["winner"])["log_loss"]),
        "winner_log_loss_sd": None,
        "screening_only": True,
    }


def _fit_cache_contract(
    *,
    cutoff: pd.Timestamp,
    config: ParameterFitConfig,
    source_sha256: Mapping[str, str | None],
    parameter_model: str = PARAMETER_MODEL_VERSION,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "parameter_model": parameter_model,
        "strictly_before_utc": cutoff.isoformat(),
        "config": asdict(config),
        "source_sha256": dict(source_sha256),
    }


def _load_or_fit_posterior_artifact(
    fitter: CausalParameterFitter,
    *,
    cutoff: pd.Timestamp,
    config: ParameterFitConfig,
    source_sha256: Mapping[str, str | None],
    fit_cache_dir: str | Path | None,
) -> tuple[ParameterEnsembleArtifact, bool]:
    if fit_cache_dir is None:
        return (
            fitter.fit(cutoff, config=config, created_at_utc=cutoff),
            False,
        )
    contract = _fit_cache_contract(
        cutoff=cutoff,
        config=config,
        source_sha256=source_sha256,
        parameter_model=fitter.parameter_model_version,
    )
    key = canonical_sha256(contract)
    cache_path = Path(fit_cache_dir) / f"fit-{key}.json.gz"
    if cache_path.is_file():
        artifact = load_parameter_artifact(cache_path)
        if (
            artifact.as_of_utc != cutoff.isoformat()
            or artifact.config != config
            or artifact.model_version != fitter.parameter_model_version
        ):
            raise ValueError(f"cached causal fit contract is invalid: {cache_path}")
        return artifact, True
    artifact = fitter.fit(cutoff, config=config, created_at_utc=cutoff)
    save_parameter_artifact(cache_path, artifact, materialized=True)
    return artifact, False


def execute_posterior_backtest(
    *,
    output_dir: str | Path,
    last_events: int = 20,
    skip_latest_events: int = 0,
    min_prior_ufc_fights: int = 3,
    bootstrap_members: int = 64,
    paths_per_matchup: int = 4096,
    seed_repeats: int = 2,
    min_training_fights: int = 500,
    random_seed: int = 2903,
    workers: int = 1,
    chunk_size: int = 64,
    resume: bool = False,
    fit_cache_dir: str | Path | None = DEFAULT_POSTERIOR_FIT_CACHE,
    fidelity: str = "final",
    raw_path: str | Path = DEFAULT_RAW_FIGHTS,
    profiles_path: str | Path = DEFAULT_FIGHTER_PROFILES,
    round_path: str | Path = DEFAULT_ROUND_STATS,
    progress: Callable[[str], None] | None = None,
    simulator_config: SimulatorConfig | None = None,
    use_takedown_control_association: bool = False,
    max_runtime_seconds: float | None = None,
    cohort_manifest_path: str | Path | None = None,
    cohort_name: str | None = None,
    snapshot_parameter_mode: str = "full",
) -> tuple[Path, dict[str, object]]:
    """Run a causal event-cutoff posterior-predictive population study."""

    if paths_per_matchup <= 0 or paths_per_matchup % bootstrap_members:
        raise ValueError("paths_per_matchup must be positive and divisible by bootstrap_members")
    if not 1 <= seed_repeats <= 4:
        raise ValueError("seed_repeats must be between 1 and 4")
    if fidelity not in _POSTERIOR_FIDELITIES:
        raise ValueError(f"unsupported posterior fidelity: {fidelity}")
    if max_runtime_seconds is not None and not 0 < max_runtime_seconds <= 3300:
        raise ValueError("max_runtime_seconds must be in (0, 3300]")
    if (cohort_manifest_path is None) != (cohort_name is None):
        raise ValueError("cohort_manifest_path and cohort_name must be provided together")
    if snapshot_parameter_mode not in SNAPSHOT_PARAMETER_MODES:
        raise ValueError(
            "snapshot_parameter_mode must be full, context_only, "
            "opponent_adjusted_v1, opponent_adjusted_v2, or reliability_weighted"
        )
    fingerprint_started = time.perf_counter()
    source_sha256 = {
        "raw": _file_sha256(raw_path, required=True),
        "profiles": _file_sha256(profiles_path, required=False),
        "round_stats": _file_sha256(round_path, required=False),
    }
    cohort_manifest_sha256 = (
        None
        if cohort_manifest_path is None
        else canonical_sha256(load_json(cohort_manifest_path))
    )
    fingerprint_seconds = time.perf_counter() - fingerprint_started
    simulator = simulator_config or SimulatorConfig()
    parameter_model = (
        TAKEDOWN_CONTROL_PARAMETER_MODEL_VERSION
        if use_takedown_control_association
        else PARAMETER_MODEL_VERSION
    )
    run_contract: dict[str, object] = {
        "schema_version": 1,
        "engine_version": ENGINE_VERSION,
        "rng_contract": RNG_CONTRACT_VERSION,
        "parameter_model": parameter_model,
        "source_sha256": source_sha256,
        "selection": {
            "last_events": last_events,
            "skip_latest_events": skip_latest_events,
            "min_prior_ufc_fights": min_prior_ufc_fights,
            "min_training_fights": min_training_fights,
            "use_takedown_control_association": bool(
                use_takedown_control_association
            ),
            "max_runtime_seconds": max_runtime_seconds,
            "cohort_manifest_sha256": cohort_manifest_sha256,
            "cohort_name": cohort_name,
            "snapshot_parameter_mode": snapshot_parameter_mode,
        },
        "simulation": {
            "bootstrap_members": bootstrap_members,
            "paths_per_matchup": paths_per_matchup,
            "seed_repeats": seed_repeats,
            "random_seed": random_seed,
            "simulator_config": simulator.to_dict(),
        },
        "fidelity": fidelity,
    }
    run_contract_sha256 = canonical_sha256(run_contract)
    destination = Path(output_dir)
    manifest_path = destination / "run-manifest.json"
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise ValueError(
            f"population output directory is not empty: {destination}; choose a new directory"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if resume and manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("run_contract_sha256") != run_contract_sha256:
            raise ValueError(
                "resume contract differs from the existing population run; "
                "use a new output directory"
            )
    elif resume and any(destination.iterdir()):
        raise ValueError("resume requires an existing valid run-manifest.json")
    else:
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "run_contract": run_contract,
                "run_contract_sha256": run_contract_sha256,
            },
        )
    started = time.perf_counter()
    deadline = (
        None
        if max_runtime_seconds is None
        else started + float(max_runtime_seconds)
    )
    raw, profiles, rounds = load_research_inputs(raw_path, profiles_path, round_path)
    physical = physical_backtest_frame(raw)
    cohort_metadata: dict[str, object] = {}
    if cohort_manifest_path is not None and cohort_name is not None:
        (
            selected,
            event_manifest,
            selection_counts,
            cohort_metadata,
        ) = _frozen_cohort_selection(
            physical,
            manifest_path=cohort_manifest_path,
            cohort_name=cohort_name,
            min_prior_ufc_fights=min_prior_ufc_fights,
            source_sha256=source_sha256,
        )
    else:
        selected, event_manifest, selection_counts = _recent_complete_event_selection(
            physical,
            last_events=last_events,
            min_prior_ufc_fights=min_prior_ufc_fights,
            skip_latest_events=skip_latest_events,
        )
    if selected.empty:
        raise ValueError("recent-event exposure filter selected no fights")
    if progress:
        progress(
            f"Selected {len(selected)} eligible fights across {len(event_manifest)} events "
            f"({selection_counts['excluded_low_exposure']} low-exposure fights excluded)."
        )
    repeat_records: list[list[dict[str, object]]] = [
        [] for _ in range(seed_repeats)
    ]
    completed = 0
    total_work = len(selected) * seed_repeats
    resumed_pairs = 0
    simulated_pairs = 0
    fit_cache_hits = 0
    fit_cache_misses = 0
    fit_seconds = 0.0
    simulation_seconds = 0.0
    checkpoint_seconds = 0.0
    fitter = CausalParameterFitter(
        raw,
        profiles,
        rounds,
        use_takedown_control_association=use_takedown_control_association,
    )
    grouped_events = list(selected.groupby(["date", "event_id"], sort=True))
    stopped_by_time_limit = False
    for event_position, ((date, event_id), event_test) in enumerate(
        grouped_events, start=1
    ):
        if deadline is not None and time.perf_counter() >= deadline:
            stopped_by_time_limit = True
            break
        cutoff = pd.Timestamp(date)
        train = physical.loc[physical["date"].lt(cutoff)].copy()
        if len(train) < min_training_fights:
            raise ValueError(
                f"event {event_id} has only {len(train)} prior fights; "
                f"minimum is {min_training_fights}"
            )
        event_seed = random_seed + int(
            canonical_sha256({"event_id": str(event_id), "date": cutoff.isoformat()})[:8],
            16,
        )
        ordered_rows = event_test.sort_values("fight_id", kind="stable").to_dict(
            "records"
        )
        checkpoint_records: dict[tuple[str, int], dict[str, object]] = {}
        checkpoint_started = time.perf_counter()
        for row in ordered_rows:
            for repeat_index in range(seed_repeats):
                checkpoint_path = _posterior_checkpoint_path(
                    destination, row["fight_id"], repeat_index
                )
                if checkpoint_path.is_file():
                    checkpoint_records[(str(row["fight_id"]), repeat_index)] = (
                        _load_posterior_checkpoint(
                            checkpoint_path,
                            run_contract_sha256=run_contract_sha256,
                            fight_id=row["fight_id"],
                            repeat_index=repeat_index,
                        )
                    )
        checkpoint_seconds += time.perf_counter() - checkpoint_started
        missing_pairs = len(ordered_rows) * seed_repeats - len(checkpoint_records)
        if missing_pairs == 0:
            for row in ordered_rows:
                for repeat_index in range(seed_repeats):
                    repeat_records[repeat_index].append(
                        checkpoint_records[(str(row["fight_id"]), repeat_index)]
                    )
                    completed += 1
                    resumed_pairs += 1
            if progress:
                progress(
                    f"Event {event_position}/{len(grouped_events)}: restored all "
                    f"{len(ordered_rows) * seed_repeats} fight/seed pairs from checkpoints."
                )
            continue
        if progress:
            progress(
                f"Event {event_position}/{len(grouped_events)}: "
                f"loading/fitting {bootstrap_members} causal members for {cutoff.date()} "
                f"({len(event_test)} eligible fights)."
            )
        fit_config = ParameterFitConfig.historical(
            bootstrap_members=bootstrap_members,
            random_seed=event_seed,
        )
        fit_started = time.perf_counter()
        artifact, cache_hit = _load_or_fit_posterior_artifact(
            fitter,
            cutoff=cutoff,
            config=fit_config,
            source_sha256=source_sha256,
            fit_cache_dir=fit_cache_dir,
        )
        fit_seconds += time.perf_counter() - fit_started
        fit_cache_hits += int(cache_hit)
        fit_cache_misses += int(not cache_hit)
        artifact.validate()
        baselines = causal_joint_baseline_forecasts(train, event_test).set_index("fight_id")
        for row in ordered_rows:
            # Stop only between complete fights, keeping all requested seed
            # repeats aligned and every written checkpoint independently valid.
            if deadline is not None and time.perf_counter() >= deadline:
                stopped_by_time_limit = True
                break
            missing_repeats = [
                repeat_index
                for repeat_index in range(seed_repeats)
                if (str(row["fight_id"]), repeat_index) not in checkpoint_records
            ]
            if not missing_repeats:
                for repeat_index in range(seed_repeats):
                    repeat_records[repeat_index].append(
                        checkpoint_records[(str(row["fight_id"]), repeat_index)]
                    )
                    completed += 1
                    resumed_pairs += 1
                continue
            first_root_seed = f"posterior:{random_seed}:{row['fight_id']}"
            first_specs = build_specs(
                fitter,
                artifact,
                red_fighter_id=str(row["red_fighter_id"]),
                blue_fighter_id=str(row["blue_fighter_id"]),
                division=str(row["division"]),
                scheduled_rounds=int(row["scheduled_rounds"]),
                event_id=str(row["event_id"]),
                matchup_id=matchup_id_for(
                    row["event_id"], row["red_fighter_id"], row["blue_fighter_id"]
                ),
                root_seed=first_root_seed,
                simulator_base=simulator,
                snapshot_parameter_mode=snapshot_parameter_mode,
                _artifact_validated=True,
            )
            for repeat_index in range(seed_repeats):
                repeat_number = repeat_index + 1
                checkpoint_key = (str(row["fight_id"]), repeat_index)
                if checkpoint_key in checkpoint_records:
                    repeat_records[repeat_index].append(
                        checkpoint_records[checkpoint_key]
                    )
                    completed += 1
                    resumed_pairs += 1
                    continue
                specs = (
                    first_specs
                    if repeat_number == 1
                    else tuple(
                        replace(
                            spec,
                            root_seed=(
                                f"posterior:{random_seed}:repeat:"
                                f"{repeat_number}:{row['fight_id']}"
                            ),
                        )
                        for spec in first_specs
                    )
                )
                simulation_started = time.perf_counter()
                simulation = run_nested(
                    specs,
                    paths_per_matchup // bootstrap_members,
                    workers=workers,
                    chunk_size=chunk_size,
                    max_traces=0,
                    retain_paths=False,
                )
                simulation_seconds += time.perf_counter() - simulation_started
                record = dict(row)
                record["forecast"] = _forecast_with_full_support(
                    _compact_evaluation_forecast(simulation.forecast)
                )
                record["causal_cutoff_utc"] = cutoff.isoformat()
                if str(row["fight_id"]) in baselines.index:
                    baseline = baselines.loc[str(row["fight_id"])]
                    record["population_forecast"] = baseline["population_forecast"]
                    record["division_forecast"] = baseline["division_forecast"]
                checkpoint_started = time.perf_counter()
                _atomic_write_json_gzip(
                    _posterior_checkpoint_path(
                        destination, row["fight_id"], repeat_index
                    ),
                    {
                        "schema_version": 1,
                        "run_contract_sha256": run_contract_sha256,
                        "fight_id": str(row["fight_id"]),
                        "repeat_index": repeat_index,
                        "record": record,
                    },
                )
                checkpoint_seconds += time.perf_counter() - checkpoint_started
                repeat_records[repeat_index].append(record)
                completed += 1
                simulated_pairs += 1
                if progress:
                    progress(
                        f"Completed {completed}/{total_work}: {row['red_fighter_name']} vs "
                        f"{row['blue_fighter_name']} (seed {repeat_number}/{seed_repeats})."
                    )
        if stopped_by_time_limit:
            break
    ledgers = [pd.DataFrame(records) for records in repeat_records]
    authoritative = ledgers[0]
    if authoritative.empty:
        raise RuntimeError(
            "posterior backtest reached its runtime budget before completing a fight"
        )
    aggregate = evaluate_simulation_ledger(authoritative)
    high_information = authoritative.loc[
        authoritative["red_prior_ufc_fights"].ge(5)
        & authoritative["blue_prior_ufc_fights"].ge(5)
    ]
    event_summaries = []
    for (date, event_id), group in authoritative.groupby(["date", "event_id"], sort=True):
        metrics = evaluate_simulation_ledger(group)
        event_summaries.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "event_id": str(event_id),
                "fights": int(len(group)),
                "joint_log_loss": metrics["primary_joint_side_method_log_loss"],
                "winner_log_loss": metrics["winner"]["log_loss"],
                "duration_crps_seconds": metrics["duration_crps_seconds"],
            }
        )
    diagnostics = posterior_predictive_rows(authoritative)
    metadata = authoritative.set_index("fight_id")[
        ["red_fighter_name", "blue_fighter_name", "date", "actual_outcome"]
    ].to_dict("index")
    diagnostic_rows = [
        {**row, **metadata.get(str(row["fight_id"]), {})}
        for row in diagnostics
    ]
    elapsed = time.perf_counter() - started
    comparisons = {
        "population_joint_log_loss": _joint_log_loss(authoritative, "population_forecast"),
        "division_joint_log_loss": _joint_log_loss(authoritative, "division_forecast"),
        "simulation_joint_log_loss": aggregate["primary_joint_side_method_log_loss"],
    }
    simulation_noise = (
        repeated_seed_summary(ledgers)
        if seed_repeats >= 2
        else _posterior_single_seed_summary(authoritative)
    )
    coverage_warnings = [
        "nominal_pit_pvalues_do_not_account_for_event_card_clustering_or_multiple_comparisons",
        "low_exposure_fights_are_excluded_from_primary_mechanics_calibration",
        "control_definition_differs_from_broader_ufcstats_control",
    ]
    if seed_repeats == 1:
        coverage_warnings.append(
            "single_seed_low_fidelity_screen_cannot_estimate_end_to_end_simulation_noise"
        )
    if stopped_by_time_limit:
        coverage_warnings.append(
            "time_bounded_partial_screen_completed_only_checkpointed_fights"
        )
    report_body: dict[str, object] = {
        "schema_version": 1,
        "evaluation_version": "fight-sim-posterior-population-v1",
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "primary_metric": "posterior_predictive_calibration",
        "fidelity": fidelity,
        "selection_eligible": (
            fidelity == "final"
            and bootstrap_members >= 64
            and paths_per_matchup >= 4096
            and seed_repeats >= 2
        ),
        "run_contract_sha256": run_contract_sha256,
        "config": {
            "last_events": last_events,
            "skip_latest_events": skip_latest_events,
            "min_prior_ufc_fights": min_prior_ufc_fights,
            "high_information_min_prior_ufc_fights": 5,
            "bootstrap_members": bootstrap_members,
            "paths_per_matchup": paths_per_matchup,
            "seed_repeats": seed_repeats,
            "min_training_fights": min_training_fights,
            "random_seed": random_seed,
            "use_takedown_control_association": bool(
                use_takedown_control_association
            ),
            "parameter_model": parameter_model,
            "snapshot_parameter_mode": snapshot_parameter_mode,
            "max_runtime_seconds": max_runtime_seconds,
            **cohort_metadata,
            "workers": workers,
            "chunk_size": chunk_size,
            "simulator_config": simulator.to_dict(),
        },
        "selection": {
            **selection_counts,
            "events": event_manifest,
            "completed_fights": int(len(authoritative)),
            "completed_event_cards": int(authoritative["event_id"].nunique()),
            "first_event_date": min(item["date"] for item in event_manifest),
            "last_event_date": max(item["date"] for item in event_manifest),
        },
        "aggregate": aggregate,
        "high_information_5_plus": (
            evaluate_simulation_ledger(high_information)
            if not high_information.empty
            else None
        ),
        "event_summaries": event_summaries,
        "comparisons": comparisons,
        "simulation_noise": simulation_noise,
        "runtime": {
            "elapsed_seconds": elapsed,
            "input_fingerprint_seconds": fingerprint_seconds,
            "causal_fit_load_seconds": fit_seconds,
            "simulation_seconds": simulation_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "planned_fight_seed_pairs": total_work,
            "simulated_fight_seed_pairs": int(sum(len(frame) for frame in ledgers)),
            "completed_fight_seed_pairs": int(sum(len(frame) for frame in ledgers)),
            "computed_fight_seed_pairs_this_invocation": simulated_pairs,
            "resumed_fight_seed_pairs": resumed_pairs,
            "total_paths": int(sum(len(frame) for frame in ledgers) * paths_per_matchup),
            "paths_computed_this_invocation": int(
                simulated_pairs * paths_per_matchup
            ),
            "fit_cache_hits": fit_cache_hits,
            "fit_cache_misses": fit_cache_misses,
            "stopped_by_time_limit": stopped_by_time_limit,
        },
        "coverage_warnings": coverage_warnings,
    }
    report_body["report_sha256"] = canonical_sha256(report_body)
    report_path = atomic_write_json(destination / "population-summary.json", report_body)
    _atomic_write_jsonl_gzip(
        destination / "fight-diagnostics.jsonl.gz", diagnostic_rows
    )
    _atomic_write_jsonl_gzip(
        destination / "forecast-ledger.jsonl.gz", authoritative.to_dict("records")
    )
    from .population_report import write_population_report

    write_population_report(destination / "population-report.html", report_body)
    if progress:
        status = "checkpoint written" if stopped_by_time_limit else "complete"
        progress(
            f"Population study {status} in {elapsed / 60.0:.1f} minutes: "
            f"{report_path}"
        )
    return destination, report_body


def execute_analyze(
    input_path: str | Path,
    *,
    output: str | Path | None = None,
    title: str | None = None,
) -> Path:
    source = Path(input_path)
    run_dir = source if source.is_dir() else source.parent
    aggregate_path = source / "aggregate.json" if source.is_dir() else source
    aggregate_payload = load_json(aggregate_path)
    is_backtest_report = (
        isinstance(aggregate_payload.get("folds"), list)
        and isinstance(aggregate_payload.get("comparisons"), dict)
        and "primary_metric" in aggregate_payload
    )
    aggregate = (
        {
            "matchup_id": "chronological-backtest",
            "outcome_probabilities": {},
            "statistic_summaries": [],
            "total_lines": [],
            "survival": [],
            "uncertainty": [],
        }
        if is_backtest_report
        else aggregate_payload.get("aggregate", aggregate_payload)
    )
    specs_path = run_dir / "specs.json"
    convergence_path = run_dir / "convergence.json"
    traces_dir = run_dir / "traces"
    specs = (
        None
        if is_backtest_report
        else (load_json(specs_path) if specs_path.is_file() else None)
    )
    evaluation = (
        aggregate_payload
        if is_backtest_report
        else (load_json(convergence_path) if convergence_path.is_file() else None)
    )
    traces = [
        load_json(path) for path in sorted(traces_dir.glob("*.json"))
    ] if traces_dir.is_dir() else []
    destination = Path(output) if output is not None else run_dir / "analysis.html"
    return write_analysis_report(
        destination,
        aggregate,
        run_spec=specs,
        traces=traces,
        evaluation=evaluation,
        title=(title or "Fight simulation chronological backtest")
        if is_backtest_report
        else title,
    )
