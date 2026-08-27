"""Development-only global mechanics calibration from posterior checks."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Mapping

import pandas as pd

from .domain import SimulatorConfig
from .evaluation import (
    _joint_event_card_paired_interval,
    evaluate_simulation_ledger,
    event_card_paired_interval,
)
from .parameters import canonical_sha256
from .research import atomic_write_json


TUNING_SCHEMA_VERSION = 1

SELECTION_STATISTICS = (
    "distance_strike_attempts",
    "clinch_strike_attempts",
    "ground_strike_attempts",
    "takedown_attempts",
    "submission_attempts",
    "knockdowns",
)


def load_population_ledger(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.is_dir():
        source = source / "forecast-ledger.jsonl.gz"
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[dict[str, object]] = []
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"population ledger line {line_number} is not an object")
            records.append(value)
    if not records:
        raise ValueError("population ledger is empty")
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True)
    return frame.sort_values(["date", "event_id", "fight_id"], kind="stable").reset_index(drop=True)


def _expected_event_fight_counts(path: str | Path) -> dict[str, int]:
    source = Path(path)
    report_path = source / "population-summary.json" if source.is_dir() else source.parent / "population-summary.json"
    if not report_path.is_file():
        raise ValueError(
            "balanced outcome comparison requires each population-summary.json"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = dict(report.get("selection") or {})
    events = list(selection.get("events") or [])
    counts = {
        str(item["event_id"]): int(item["eligible_fights"])
        for item in events
    }
    if not counts:
        raise ValueError("population summary contains no selected event manifest")
    return counts


def _forecast_probabilities(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("forecast must be an object")
    probabilities = value.get("outcome_probabilities", value)
    if not isinstance(probabilities, Mapping):
        raise ValueError("forecast outcome probabilities must be an object")
    return {str(key): float(item) for key, item in probabilities.items()}


def _projected_method_counts(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    methods = ("decision", "ko_tko", "submission", "other", "draw", "no_contest")
    observed = {method: 0.0 for method in methods}
    predicted = {method: 0.0 for method in methods}
    for row in frame.to_dict("records"):
        actual = str(row["actual_outcome"])
        actual_method = next(
            (method for method in methods if actual == method or actual.endswith(f"_{method}")),
            "other",
        )
        observed[actual_method] += 1.0
        for outcome, probability in _forecast_probabilities(row["forecast"]).items():
            method = next(
                (name for name in methods if outcome == name or outcome.endswith(f"_{name}")),
                "other",
            )
            predicted[method] += probability
    return {
        method: {
            "observed": observed[method],
            "predicted": predicted[method],
            "bias": predicted[method] - observed[method],
        }
        for method in methods
    }


def _check(metrics: Mapping[str, object], statistic: str) -> dict[str, object]:
    checks = dict(metrics.get("posterior_predictive_checks") or {})
    value = checks.get(statistic)
    if not isinstance(value, Mapping):
        raise ValueError(f"posterior metrics are missing {statistic}")
    return dict(value)


def _count_check(metrics: Mapping[str, object], statistic: str) -> dict[str, object]:
    checks = dict(metrics.get("count_distribution_predictive_checks") or {})
    value = checks.get(statistic)
    if not isinstance(value, Mapping):
        raise ValueError(f"count metrics are missing {statistic}")
    return dict(value)


def _bias_reduction(baseline: float, candidate: float) -> float:
    if abs(baseline) <= 1e-12:
        return 0.0 if abs(candidate) <= 1e-12 else float("-inf")
    return 1.0 - abs(candidate) / abs(baseline)


def compare_outcome_mechanics(
    baseline_population_run: str | Path,
    candidate_population_run: str | Path,
    *,
    output: str | Path,
    minimum_balanced_events: int = 5,
) -> dict[str, object]:
    """Evaluate two outcome engines only on identical complete event cards."""

    if minimum_balanced_events <= 0:
        raise ValueError("minimum_balanced_events must be positive")
    baseline_all = load_population_ledger(baseline_population_run)
    candidate_all = load_population_ledger(candidate_population_run)
    baseline_expected = _expected_event_fight_counts(baseline_population_run)
    candidate_expected = _expected_event_fight_counts(candidate_population_run)
    baseline_ids = {
        str(event_id): frozenset(group["fight_id"].astype(str))
        for event_id, group in baseline_all.groupby("event_id", sort=True)
    }
    candidate_ids = {
        str(event_id): frozenset(group["fight_id"].astype(str))
        for event_id, group in candidate_all.groupby("event_id", sort=True)
    }
    balanced_events = sorted(
        event_id
        for event_id in set(baseline_ids) & set(candidate_ids)
        if baseline_ids[event_id] == candidate_ids[event_id]
        and baseline_ids[event_id]
        and len(baseline_ids[event_id]) == baseline_expected.get(event_id)
        and len(candidate_ids[event_id]) == candidate_expected.get(event_id)
    )
    if len(balanced_events) < minimum_balanced_events:
        raise ValueError(
            "outcome mechanics comparison has too few identical complete event cards"
        )
    baseline = baseline_all.loc[
        baseline_all["event_id"].astype(str).isin(balanced_events)
    ].copy()
    candidate = candidate_all.loc[
        candidate_all["event_id"].astype(str).isin(balanced_events)
    ].copy()
    ordered = ["date", "event_id", "fight_id", "actual_outcome"]
    left_identity = baseline[ordered].sort_values(ordered[:3], kind="stable").reset_index(drop=True)
    right_identity = candidate[ordered].sort_values(ordered[:3], kind="stable").reset_index(drop=True)
    if not left_identity.equals(right_identity):
        raise ValueError("balanced outcome mechanics rows disagree on identity or truth")
    baseline_paths = baseline.sort_values(ordered[:3], kind="stable")["forecast"].map(
        lambda value: int(dict(value).get("total_paths", 0))
    ).tolist()
    candidate_paths = candidate.sort_values(ordered[:3], kind="stable")["forecast"].map(
        lambda value: int(dict(value).get("total_paths", 0))
    ).tolist()
    if baseline_paths != candidate_paths or any(value <= 0 for value in baseline_paths):
        raise ValueError("balanced outcome mechanics rows require equal positive path counts")

    baseline_metrics = evaluate_simulation_ledger(baseline)
    candidate_metrics = evaluate_simulation_ledger(candidate)
    baseline_methods = _projected_method_counts(baseline)
    candidate_methods = _projected_method_counts(candidate)
    paired = left_identity.copy()
    paired["baseline_forecast"] = baseline.sort_values(
        ordered[:3], kind="stable"
    )["forecast"].tolist()
    paired["candidate_forecast"] = candidate.sort_values(
        ordered[:3], kind="stable"
    )["forecast"].tolist()
    paired["baseline_red_win_probability"] = paired["baseline_forecast"].map(
        lambda value: sum(
            probability
            for outcome, probability in _forecast_probabilities(value).items()
            if outcome.startswith("red_")
        )
    )
    paired["candidate_red_win_probability"] = paired["candidate_forecast"].map(
        lambda value: sum(
            probability
            for outcome, probability in _forecast_probabilities(value).items()
            if outcome.startswith("red_")
        )
    )
    winner_interval = event_card_paired_interval(
        paired,
        "candidate_red_win_probability",
        "baseline_red_win_probability",
        random_seed=2903,
    )
    joint_interval = _joint_event_card_paired_interval(
        paired,
        "candidate_forecast",
        "baseline_forecast",
        replicates=2000,
        random_seed=2903,
    )

    baseline_kd = _count_check(baseline_metrics, "total_knockdowns")
    candidate_kd = _count_check(candidate_metrics, "total_knockdowns")
    baseline_duration = _check(baseline_metrics, "duration_seconds")
    candidate_duration = _check(candidate_metrics, "duration_seconds")
    kd_reduction = _bias_reduction(
        float(baseline_kd["predictive_minus_observed_mean"]),
        float(candidate_kd["predictive_minus_observed_mean"]),
    )
    ko_reduction = _bias_reduction(
        float(baseline_methods["ko_tko"]["bias"]),
        float(candidate_methods["ko_tko"]["bias"]),
    )
    decision_reduction = _bias_reduction(
        float(baseline_methods["decision"]["bias"]),
        float(candidate_methods["decision"]["bias"]),
    )
    duration_reduction = _bias_reduction(
        float(baseline_duration["predictive_minus_observed_mean"]),
        float(candidate_duration["predictive_minus_observed_mean"]),
    )
    action_statistics = (
        "total_significant_strike_attempts",
        "total_ground_strikes_landed",
        "total_takedowns",
        "total_submission_attempts",
        "total_control_seconds",
    )
    action_crps_ratios = {
        statistic: float(_count_check(candidate_metrics, statistic)["count_distribution_crps"])
        / max(
            float(_count_check(baseline_metrics, statistic)["count_distribution_crps"]),
            1e-12,
        )
        for statistic in action_statistics
    }
    baseline_joint = float(baseline_metrics["primary_joint_side_method_log_loss"])
    candidate_joint = float(candidate_metrics["primary_joint_side_method_log_loss"])
    baseline_winner = float(dict(baseline_metrics["winner"])["log_loss"])
    candidate_winner = float(dict(candidate_metrics["winner"])["log_loss"])
    baseline_method = float(baseline_metrics["method_log_loss"])
    candidate_method = float(candidate_metrics["method_log_loss"])
    gates = {
        "knockdown_absolute_bias_reduced_at_least_40_percent": kd_reduction >= 0.40,
        "ko_tko_absolute_bias_reduced_at_least_40_percent": ko_reduction >= 0.40,
        "decision_absolute_bias_reduced_at_least_30_percent": decision_reduction >= 0.30,
        "duration_absolute_bias_reduced_at_least_30_percent": duration_reduction >= 0.30,
        "method_log_loss_improves": candidate_method < baseline_method,
        "joint_log_loss_not_worse_by_more_than_0_01": candidate_joint <= baseline_joint + 0.01,
        "winner_log_loss_not_worse_by_more_than_0_01": candidate_winner <= baseline_winner + 0.01,
        "no_action_crps_worse_by_more_than_5_percent": max(action_crps_ratios.values()) <= 1.05,
    }
    retained = all(gates.values())
    body: dict[str, object] = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "comparison_type": "balanced_complete_card_outcome_mechanics_development",
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "balanced_events": balanced_events,
        "balanced_event_count": len(balanced_events),
        "balanced_fight_count": int(len(baseline)),
        "paths_per_fight": sorted(set(baseline_paths)),
        "dropped_unbalanced_events": sorted(
            (set(baseline_ids) | set(candidate_ids)) - set(balanced_events)
        ),
        "baseline": {
            "metrics": baseline_metrics,
            "projected_method_counts": baseline_methods,
        },
        "candidate": {
            "metrics": candidate_metrics,
            "projected_method_counts": candidate_methods,
        },
        "candidate_minus_baseline": {
            "joint_log_loss": candidate_joint - baseline_joint,
            "winner_log_loss": candidate_winner - baseline_winner,
            "method_log_loss": candidate_method - baseline_method,
            "duration_crps_seconds": float(candidate_metrics["duration_crps_seconds"])
            - float(baseline_metrics["duration_crps_seconds"]),
        },
        "absolute_bias_reduction_fraction": {
            "total_knockdowns": kd_reduction,
            "ko_tko_count": ko_reduction,
            "decision_count": decision_reduction,
            "duration_seconds": duration_reduction,
        },
        "action_crps_candidate_over_baseline": action_crps_ratios,
        "paired_event_card_intervals": {
            "winner": winner_interval,
            "joint_side_method": joint_interval,
        },
        "gates": gates,
        "development_status": "retained_for_confirmation" if retained else "rejected",
    }
    body["comparison_sha256"] = canonical_sha256(body)
    atomic_write_json(output, body)
    return body


def _summary_mean(forecast: object, statistic: str) -> float | None:
    if not isinstance(forecast, Mapping):
        return None
    for item in list(forecast.get("statistic_summaries") or []):
        value = dict(item)
        if value.get("statistic") == statistic:
            result = float(value["mean"])
            return result if math.isfinite(result) else None
    return None


def _pooled_moment(frame: pd.DataFrame, statistic: str) -> dict[str, float | int]:
    observed: list[float] = []
    predictive: list[float] = []
    for row in frame.to_dict("records"):
        for side in ("red", "blue"):
            actual = row.get(f"actual_{side}_{statistic}")
            predicted = _summary_mean(row.get("forecast"), f"{side}_{statistic}")
            try:
                actual_value = float(actual)
            except (TypeError, ValueError):
                continue
            if predicted is None or not math.isfinite(actual_value):
                continue
            observed.append(actual_value)
            predictive.append(predicted)
    if not observed or sum(predictive) <= 0:
        raise ValueError(f"development ledger has no usable {statistic} moment")
    observed_mean = float(sum(observed) / len(observed))
    predictive_mean = float(sum(predictive) / len(predictive))
    return {
        "n_fighter_sides": len(observed),
        "observed_mean": observed_mean,
        "predictive_mean": predictive_mean,
        "observed_to_predictive_ratio": observed_mean / predictive_mean,
    }


def _regularized_ratio(
    ratio: float,
    *,
    n_fights: int,
    prior_strength: float,
    lower: float = 0.25,
    upper: float = 8.0,
) -> tuple[float, float]:
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("calibration ratio must be finite and positive")
    weight = n_fights / (n_fights + prior_strength)
    multiplier = math.exp(weight * math.log(ratio))
    return min(max(multiplier, lower), upper), weight


def derive_mechanics_profile(
    ledger_path: str | Path,
    *,
    output: str | Path,
    holdout_latest_events: int = 5,
    prior_strength_events: float = 20.0,
) -> dict[str, object]:
    """Derive one conservative global profile without looking at holdout outcomes."""

    if holdout_latest_events <= 0:
        raise ValueError("at least one latest event must remain held out")
    if prior_strength_events < 0:
        raise ValueError("prior strength must be nonnegative")
    ledger = load_population_ledger(ledger_path)
    events = (
        ledger[["date", "event_id"]]
        .drop_duplicates()
        .sort_values(["date", "event_id"], kind="stable")
        .to_dict("records")
    )
    if len(events) <= holdout_latest_events:
        raise ValueError("population ledger has too few events for the requested holdout")
    development_events = events[:-holdout_latest_events]
    holdout_events = events[-holdout_latest_events:]
    development_ids = {str(item["event_id"]) for item in development_events}
    development = ledger.loc[ledger["event_id"].astype(str).isin(development_ids)].copy()
    moments = {
        statistic: _pooled_moment(development, statistic)
        for statistic in (
            "distance_strike_attempts",
            "clinch_strike_attempts",
            "ground_strike_attempts",
            "takedown_attempts",
            "submission_attempts",
            "significant_strikes",
            "knockdowns",
        )
    }
    n_events = len(development_events)
    multiplier_fields = {
        "distance_strike_hazard_multiplier": "distance_strike_attempts",
        "clinch_strike_hazard_multiplier": "clinch_strike_attempts",
        "ground_strike_hazard_multiplier": "ground_strike_attempts",
        "takedown_hazard_multiplier": "takedown_attempts",
        "submission_hazard_multiplier": "submission_attempts",
    }
    config_values = SimulatorConfig().to_dict()
    calibration_targets: dict[str, object] = {}
    for field, statistic in multiplier_fields.items():
        ratio = float(moments[statistic]["observed_to_predictive_ratio"])
        multiplier, weight = _regularized_ratio(
            ratio,
            n_fights=n_events,
            prior_strength=prior_strength_events,
        )
        config_values[field] = multiplier
        calibration_targets[field] = {
            "target_statistic": statistic,
            "raw_ratio": ratio,
            "log_shrinkage_weight": weight,
            "selected_multiplier": multiplier,
        }
    # Knockdowns are consequences of landed significant strikes.  Calibrate
    # the conditional conversion rate so increasing strike volume is not
    # counted twice as extra knockdown probability.
    observed_kd_rate = float(moments["knockdowns"]["observed_mean"]) / max(
        float(moments["significant_strikes"]["observed_mean"]), 1e-12
    )
    predictive_kd_rate = float(moments["knockdowns"]["predictive_mean"]) / max(
        float(moments["significant_strikes"]["predictive_mean"]), 1e-12
    )
    knockdown_ratio = observed_kd_rate / predictive_kd_rate
    knockdown_multiplier, knockdown_weight = _regularized_ratio(
        knockdown_ratio,
        n_fights=n_events,
        prior_strength=prior_strength_events,
    )
    config_values["knockdown_probability_multiplier"] = knockdown_multiplier
    calibration_targets["knockdown_probability_multiplier"] = {
        "target_statistic": "knockdowns_per_significant_strike_landed",
        "raw_ratio": knockdown_ratio,
        "log_shrinkage_weight": knockdown_weight,
        "selected_multiplier": knockdown_multiplier,
    }
    config = SimulatorConfig(**config_values)
    report_path = Path(ledger_path)
    if report_path.is_dir():
        report_path = report_path / "population-summary.json"
    source_report_hash = None
    if report_path.name != "population-summary.json":
        sibling = report_path.parent / "population-summary.json"
        report_path = sibling if sibling.is_file() else report_path
    if report_path.is_file() and report_path.name == "population-summary.json":
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source_report_hash = report.get("report_sha256")
    body: dict[str, object] = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "profile_type": "development_only_global_moment_calibration",
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "source_population_report_sha256": source_report_hash,
        "development_event_count": len(development_events),
        "development_fight_count": int(len(development)),
        "development_events": [
            {"date": pd.Timestamp(item["date"]).date().isoformat(), "event_id": str(item["event_id"])}
            for item in development_events
        ],
        "held_out_event_count": len(holdout_events),
        "held_out_events": [
            {"date": pd.Timestamp(item["date"]).date().isoformat(), "event_id": str(item["event_id"])}
            for item in holdout_events
        ],
        "selection_rule": (
            "global observable action-volume moments, log-shrunk toward neutral; "
            "no winner, method, duration, or held-out outcomes used"
        ),
        "prior_strength_events": prior_strength_events,
        "moments": moments,
        "calibration_targets": calibration_targets,
        "development_baseline_metrics": evaluate_simulation_ledger(development),
        "simulator_config": config.to_dict(),
        "deferred_targets": {
            "control_seconds": "definition mismatch: UFCStats control is broader than ground top control",
            "phase_transitions": "observed bout totals do not identify phase entry/exit hazards",
            "winner_and_finish": "reserved for held-out validation, never directly tuned",
        },
    }
    body["mechanics_profile_id"] = (
        f"mechanics-{canonical_sha256(config.to_dict())[:12]}"
    )
    body["profile_sha256"] = canonical_sha256(body)
    atomic_write_json(output, body)
    return body


def _action_moment_error(metrics: Mapping[str, object]) -> tuple[float, dict[str, float]]:
    checks = dict(metrics.get("count_distribution_predictive_checks") or {})
    errors: dict[str, float] = {}
    for statistic in SELECTION_STATISTICS:
        sides = [dict(checks.get(f"{side}_{statistic}") or {}) for side in ("red", "blue")]
        if any(not side for side in sides):
            raise ValueError(f"candidate metrics are missing {statistic} checks")
        observed = sum(float(side["observed_mean"]) for side in sides) / 2.0
        predictive = sum(float(side["predictive_mean"]) for side in sides) / 2.0
        # A small fixed pseudocount keeps rare knockdown/submission ratios
        # finite without allowing them to dominate the six equally weighted
        # observable action-volume families.
        errors[statistic] = abs(math.log((predictive + 0.05) / (observed + 0.05)))
    return float(sum(errors.values()) / len(errors)), errors


def select_mechanics_profile(
    baseline_ledger_path: str | Path,
    candidates: Mapping[str, str | Path],
    *,
    output: str | Path,
    selection_events: int = 5,
    skip_latest_events: int = 5,
) -> dict[str, object]:
    """Select a predeclared profile on an intermediate chronological window."""

    if not candidates:
        raise ValueError("mechanics selection requires at least one candidate")
    if selection_events <= 0 or skip_latest_events < 0:
        raise ValueError("mechanics selection event window is invalid")
    ledger = load_population_ledger(baseline_ledger_path)
    events = (
        ledger[["date", "event_id"]]
        .drop_duplicates()
        .sort_values(["date", "event_id"], kind="stable")
    )
    end = len(events) - skip_latest_events if skip_latest_events else len(events)
    start = end - selection_events
    if start < 0:
        raise ValueError("baseline ledger has too few events for mechanics selection")
    selected_events = events.iloc[start:end]
    selected_ids = set(selected_events["event_id"].astype(str))
    baseline_frame = ledger.loc[ledger["event_id"].astype(str).isin(selected_ids)].copy()
    baseline = evaluate_simulation_ledger(baseline_frame)
    baseline_action_error, baseline_stat_errors = _action_moment_error(baseline)
    baseline_joint = float(baseline["primary_joint_side_method_log_loss"])
    baseline_method = float(baseline["method_log_loss"])
    baseline_winner = float(dict(baseline["winner"])["log_loss"])
    baseline_duration = float(baseline["duration_crps_seconds"])
    rows: list[dict[str, object]] = []
    for label, path_value in sorted(candidates.items()):
        path = Path(path_value)
        if path.is_dir():
            path = path / "population-summary.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = dict(report.get("aggregate") or {})
        selection = dict(report.get("selection") or {})
        report_events = {
            str(item["event_id"]) for item in list(selection.get("events") or [])
        }
        if report_events != selected_ids:
            raise ValueError(f"candidate {label} was not evaluated on the selection events")
        action_error, stat_errors = _action_moment_error(metrics)
        joint = float(metrics["primary_joint_side_method_log_loss"])
        method = float(metrics["method_log_loss"])
        winner = float(dict(metrics["winner"])["log_loss"])
        duration = float(metrics["duration_crps_seconds"])
        gates = {
            "joint_log_loss_not_worse_by_more_than_0.02": joint <= baseline_joint + 0.02,
            "method_log_loss_not_worse_by_more_than_0.02": method <= baseline_method + 0.02,
            "winner_log_loss_not_worse_by_more_than_0.02": winner <= baseline_winner + 0.02,
            "duration_crps_not_worse_by_more_than_5_percent": duration <= baseline_duration * 1.05,
            "observable_action_error_improves": action_error < baseline_action_error,
        }
        rows.append(
            {
                "label": label,
                "report_path": str(path),
                "report_sha256": report.get("report_sha256"),
                "simulator_config": dict(report.get("config") or {}).get("simulator_config"),
                "joint_log_loss": joint,
                "method_log_loss": method,
                "winner_log_loss": winner,
                "duration_crps_seconds": duration,
                "observable_action_error": action_error,
                "observable_action_errors": stat_errors,
                "gates": gates,
                "eligible": all(gates.values()),
            }
        )
    eligible = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: (float(row["observable_action_error"]), str(row["label"])),
    )
    selected = eligible[0] if eligible else None
    selected_config = (
        dict(selected["simulator_config"])
        if selected is not None
        else SimulatorConfig().to_dict()
    )
    body: dict[str, object] = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "selection_type": "intermediate_chronological_mechanics_selection",
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "selection_rule": (
            "pass all outcome/duration preservation gates, improve observable "
            "action-volume error, then minimize mean absolute log moment error"
        ),
        "selection_events": [
            {"date": pd.Timestamp(row.date).date().isoformat(), "event_id": str(row.event_id)}
            for row in selected_events.itertuples(index=False)
        ],
        "reserved_latest_event_count": skip_latest_events,
        "baseline": {
            "joint_log_loss": baseline_joint,
            "method_log_loss": baseline_method,
            "winner_log_loss": baseline_winner,
            "duration_crps_seconds": baseline_duration,
            "observable_action_error": baseline_action_error,
            "observable_action_errors": baseline_stat_errors,
        },
        "candidates": rows,
        "selection_status": "selected" if selected is not None else "neutral_fallback",
        "selected_label": selected["label"] if selected is not None else "neutral",
        "simulator_config": selected_config,
    }
    body["mechanics_profile_id"] = (
        f"mechanics-{canonical_sha256(selected_config)[:12]}"
    )
    body["selection_sha256"] = canonical_sha256(body)
    atomic_write_json(output, body)
    return body


def _population_report(path_value: str | Path) -> tuple[Path, dict[str, object]]:
    path = Path(path_value)
    if path.is_dir():
        path = path / "population-summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"population report is not an object: {path}")
    return path, value


def _finish_selection_metrics(path_value: str | Path) -> dict[str, object]:
    frame = load_population_ledger(path_value)
    metrics = evaluate_simulation_ledger(frame)
    action_error, action_errors = _action_moment_error(metrics)
    duration_check = dict(
        dict(metrics.get("posterior_predictive_checks") or {}).get(
            "duration_seconds"
        )
        or {}
    )
    knockdown_check = dict(
        dict(metrics.get("count_distribution_predictive_checks") or {}).get(
            "total_knockdowns"
        )
        or {}
    )
    path, report = _population_report(path_value)
    config = dict(dict(report.get("config") or {}).get("simulator_config") or {})
    return {
        "report_path": str(path),
        "report_sha256": report.get("report_sha256"),
        "event_ids": sorted(frame["event_id"].astype(str).unique().tolist()),
        "fight_ids": sorted(frame["fight_id"].astype(str).unique().tolist()),
        "joint_log_loss": float(metrics["primary_joint_side_method_log_loss"]),
        "method_log_loss": float(metrics["method_log_loss"]),
        "winner_log_loss": float(dict(metrics["winner"])["log_loss"]),
        "duration_crps_seconds": float(metrics["duration_crps_seconds"]),
        "duration_mean_bias_seconds": float(
            duration_check["predictive_minus_observed_mean"]
        ),
        "knockdown_crps": float(knockdown_check["count_distribution_crps"]),
        "knockdown_observed_mean": float(knockdown_check["observed_mean"]),
        "knockdown_predictive_mean": float(knockdown_check["predictive_mean"]),
        "knockdown_mean_bias": float(
            knockdown_check["predictive_minus_observed_mean"]
        ),
        "observable_action_error": action_error,
        "observable_action_errors": action_errors,
        "simulator_config": config,
    }


def select_finish_profile(
    baseline_population_run: str | Path,
    candidates: Mapping[str, str | Path],
    *,
    output: str | Path,
    objective: str = "duration",
) -> dict[str, object]:
    """Select a finish-conversion candidate on one chronological dev window."""

    if not candidates:
        raise ValueError("finish selection requires at least one candidate")
    if objective not in {"duration", "joint"}:
        raise ValueError("finish selection objective must be 'duration' or 'joint'")
    baseline = _finish_selection_metrics(baseline_population_run)
    rows: list[dict[str, object]] = []
    for label, path in sorted(candidates.items()):
        row = _finish_selection_metrics(path)
        if row["event_ids"] != baseline["event_ids"] or row["fight_ids"] != baseline["fight_ids"]:
            raise ValueError(f"finish candidate {label} does not match the baseline cohort")
        preservation_gates = {
            "joint_log_loss_not_worse_by_more_than_0.02": (
                float(row["joint_log_loss"]) <= float(baseline["joint_log_loss"]) + 0.02
            ),
            "method_log_loss_not_worse_by_more_than_0.02": (
                float(row["method_log_loss"]) <= float(baseline["method_log_loss"]) + 0.02
            ),
            "winner_log_loss_not_worse_by_more_than_0.02": (
                float(row["winner_log_loss"]) <= float(baseline["winner_log_loss"]) + 0.02
            ),
            "observable_action_error_not_worse_by_more_than_0.02": (
                float(row["observable_action_error"])
                <= float(baseline["observable_action_error"]) + 0.02
            ),
        }
        if objective == "duration":
            objective_gates = {
                "duration_crps_improves": (
                    float(row["duration_crps_seconds"])
                    < float(baseline["duration_crps_seconds"])
                ),
            }
        else:
            objective_gates = {
                "joint_log_loss_improves": (
                    float(row["joint_log_loss"])
                    < float(baseline["joint_log_loss"])
                ),
                "duration_crps_not_worse_by_more_than_5_seconds": (
                    float(row["duration_crps_seconds"])
                    <= float(baseline["duration_crps_seconds"]) + 5.0
                ),
                "absolute_duration_bias_not_worse_by_more_than_15_seconds": (
                    abs(float(row["duration_mean_bias_seconds"]))
                    <= abs(float(baseline["duration_mean_bias_seconds"])) + 15.0
                ),
            }
        gates = {**preservation_gates, **objective_gates}
        row.update({"label": label, "gates": gates, "eligible": all(gates.values())})
        rows.append(row)
    ranking_fields = (
        ("duration_crps_seconds", "joint_log_loss")
        if objective == "duration"
        else ("joint_log_loss", "duration_crps_seconds")
    )
    eligible = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: (
            float(row[ranking_fields[0]]),
            float(row[ranking_fields[1]]),
            str(row["label"]),
        ),
    )
    selected = eligible[0] if eligible else None
    selected_config = (
        dict(selected["simulator_config"])
        if selected is not None
        else dict(baseline["simulator_config"])
    )
    body: dict[str, object] = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "selection_type": "intermediate_chronological_finish_conversion_selection",
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "objective": objective,
        "selection_rule": (
            "pass joint/method/winner/action preservation gates and improve duration "
            "CRPS, then minimize duration CRPS with joint log loss as tie-breaker"
            if objective == "duration"
            else "improve joint side/method log loss while passing method, winner, "
            "action, duration-CRPS, and duration-bias preservation gates; then "
            "minimize joint log loss with duration CRPS as tie-breaker"
        ),
        "baseline": baseline,
        "candidates": rows,
        "selection_status": "selected" if selected is not None else "baseline_fallback",
        "selected_label": selected["label"] if selected is not None else "baseline",
        "simulator_config": selected_config,
    }
    body["mechanics_profile_id"] = f"mechanics-{canonical_sha256(selected_config)[:12]}"
    body["selection_sha256"] = canonical_sha256(body)
    atomic_write_json(output, body)
    return body


def validate_knockdown_observation_profile(
    baseline_holdout_run: str | Path,
    candidate_holdout_run: str | Path,
    *,
    output: str | Path,
) -> dict[str, object]:
    """Validate official-knockdown thinning without changing latent outcomes."""

    baseline = _finish_selection_metrics(baseline_holdout_run)
    candidate = _finish_selection_metrics(candidate_holdout_run)
    if candidate["event_ids"] != baseline["event_ids"] or candidate["fight_ids"] != baseline["fight_ids"]:
        raise ValueError("knockdown-observation candidate does not match the baseline cohort")

    baseline_config = dict(baseline["simulator_config"])
    candidate_config = dict(candidate["simulator_config"])
    observation_field = "official_knockdown_observation_probability"
    baseline_without_observation = dict(baseline_config)
    candidate_without_observation = dict(candidate_config)
    baseline_without_observation.pop(observation_field, None)
    candidate_without_observation.pop(observation_field, None)

    invariant_metric_names = (
        "joint_log_loss",
        "method_log_loss",
        "winner_log_loss",
        "duration_crps_seconds",
        "duration_mean_bias_seconds",
    )
    invariant_metrics_equal = all(
        math.isclose(
            float(candidate[name]),
            float(baseline[name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name in invariant_metric_names
    )
    baseline_action_errors = dict(baseline["observable_action_errors"])
    candidate_action_errors = dict(candidate["observable_action_errors"])
    non_knockdown_action_errors_equal = all(
        name in candidate_action_errors
        and math.isclose(
            float(candidate_action_errors[name]),
            float(value),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name, value in baseline_action_errors.items()
        if name != "knockdowns"
    )
    gates = {
        "only_observation_probability_changes": (
            baseline_without_observation == candidate_without_observation
            and float(candidate_config.get(observation_field, 1.0)) < 1.0
        ),
        "outcome_and_duration_metrics_are_identical": invariant_metrics_equal,
        "non_knockdown_action_errors_are_identical": non_knockdown_action_errors_equal,
        "knockdown_crps_improves": (
            float(candidate["knockdown_crps"])
            < float(baseline["knockdown_crps"])
        ),
        "absolute_knockdown_mean_bias_improves": (
            abs(float(candidate["knockdown_mean_bias"]))
            < abs(float(baseline["knockdown_mean_bias"]))
        ),
        "observable_action_error_improves": (
            float(candidate["observable_action_error"])
            < float(baseline["observable_action_error"])
        ),
    }
    retained = all(gates.values())
    config = candidate_config if retained else baseline_config
    body: dict[str, object] = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "validation_type": "locked_knockdown_observation_holdout",
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "validation_status": "retained_for_prospective_shadow" if retained else "rejected_baseline_fallback",
        "gates": gates,
        "baseline": baseline,
        "candidate": candidate,
        "simulator_config": config,
    }
    body["mechanics_profile_id"] = f"mechanics-{canonical_sha256(config)[:12]}"
    body["validation_sha256"] = canonical_sha256(body)
    atomic_write_json(output, body)
    return body


def validate_finish_profile(
    baseline_holdout_run: str | Path,
    candidate_holdout_run: str | Path,
    *,
    output: str | Path,
) -> dict[str, object]:
    """Retain or reject finish conversion on its one untouched holdout."""

    baseline = _finish_selection_metrics(baseline_holdout_run)
    candidate = _finish_selection_metrics(candidate_holdout_run)
    if candidate["event_ids"] != baseline["event_ids"] or candidate["fight_ids"] != baseline["fight_ids"]:
        raise ValueError("finish holdout candidate does not match the baseline cohort")
    gates = {
        "joint_log_loss_not_worse_by_more_than_0.02": (
            float(candidate["joint_log_loss"]) <= float(baseline["joint_log_loss"]) + 0.02
        ),
        "method_log_loss_not_worse_by_more_than_0.02": (
            float(candidate["method_log_loss"]) <= float(baseline["method_log_loss"]) + 0.02
        ),
        "winner_log_loss_not_worse_by_more_than_0.02": (
            float(candidate["winner_log_loss"]) <= float(baseline["winner_log_loss"]) + 0.02
        ),
        "observable_action_error_not_worse_by_more_than_0.02": (
            float(candidate["observable_action_error"])
            <= float(baseline["observable_action_error"]) + 0.02
        ),
        "duration_crps_improves": (
            float(candidate["duration_crps_seconds"])
            < float(baseline["duration_crps_seconds"])
        ),
        "absolute_duration_bias_improves": (
            abs(float(candidate["duration_mean_bias_seconds"]))
            < abs(float(baseline["duration_mean_bias_seconds"]))
        ),
    }
    retained = all(gates.values())
    config = dict(
        candidate["simulator_config"] if retained else baseline["simulator_config"]
    )
    body: dict[str, object] = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "validation_type": "final_untouched_finish_conversion_holdout",
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "validation_status": "retained" if retained else "rejected_baseline_fallback",
        "gates": gates,
        "baseline": baseline,
        "candidate": candidate,
        "simulator_config": config,
    }
    body["mechanics_profile_id"] = f"mechanics-{canonical_sha256(config)[:12]}"
    body["validation_sha256"] = canonical_sha256(body)
    atomic_write_json(output, body)
    return body


def validate_mechanics_holdout(
    baseline_ledger_path: str | Path,
    tuned_population_run: str | Path,
    *,
    output: str | Path,
    holdout_latest_events: int = 5,
) -> dict[str, object]:
    """Make the final retain/reject decision on untouched latest cards."""

    if holdout_latest_events <= 0:
        raise ValueError("holdout event count must be positive")
    baseline_ledger = load_population_ledger(baseline_ledger_path)
    events = (
        baseline_ledger[["date", "event_id"]]
        .drop_duplicates()
        .sort_values(["date", "event_id"], kind="stable")
        .tail(holdout_latest_events)
    )
    if len(events) != holdout_latest_events:
        raise ValueError("baseline ledger has too few holdout events")
    event_ids = set(events["event_id"].astype(str))
    baseline_frame = baseline_ledger.loc[
        baseline_ledger["event_id"].astype(str).isin(event_ids)
    ].copy()
    tuned_frame = load_population_ledger(tuned_population_run)
    if set(tuned_frame["event_id"].astype(str)) != event_ids:
        raise ValueError("tuned run was not evaluated on the final holdout events")
    if set(tuned_frame["fight_id"].astype(str)) != set(
        baseline_frame["fight_id"].astype(str)
    ):
        raise ValueError("baseline and tuned holdout fight IDs disagree")
    baseline = evaluate_simulation_ledger(baseline_frame)
    tuned = evaluate_simulation_ledger(tuned_frame)
    baseline_action, baseline_errors = _action_moment_error(baseline)
    tuned_action, tuned_errors = _action_moment_error(tuned)
    baseline_joint = float(baseline["primary_joint_side_method_log_loss"])
    tuned_joint = float(tuned["primary_joint_side_method_log_loss"])
    baseline_method = float(baseline["method_log_loss"])
    tuned_method = float(tuned["method_log_loss"])
    baseline_winner = float(dict(baseline["winner"])["log_loss"])
    tuned_winner = float(dict(tuned["winner"])["log_loss"])
    baseline_duration = float(baseline["duration_crps_seconds"])
    tuned_duration = float(tuned["duration_crps_seconds"])
    gates = {
        "joint_log_loss_not_worse_by_more_than_0.02": tuned_joint <= baseline_joint + 0.02,
        "method_log_loss_not_worse_by_more_than_0.02": tuned_method <= baseline_method + 0.02,
        "winner_log_loss_not_worse_by_more_than_0.02": tuned_winner <= baseline_winner + 0.02,
        "duration_crps_not_worse_by_more_than_5_percent": tuned_duration <= baseline_duration * 1.05,
        "observable_action_error_improves": tuned_action < baseline_action,
    }
    retained = all(gates.values())
    tuned_report_path = Path(tuned_population_run)
    if tuned_report_path.is_dir():
        tuned_report_path = tuned_report_path / "population-summary.json"
    tuned_report = json.loads(tuned_report_path.read_text(encoding="utf-8"))
    candidate_config = dict(tuned_report.get("config") or {}).get("simulator_config")
    if not isinstance(candidate_config, dict):
        candidate_config = SimulatorConfig().to_dict()
    selected_config = candidate_config if retained else SimulatorConfig().to_dict()
    body: dict[str, object] = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "validation_type": "final_untouched_chronological_holdout",
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "holdout_events": [
            {"date": pd.Timestamp(row.date).date().isoformat(), "event_id": str(row.event_id)}
            for row in events.itertuples(index=False)
        ],
        "holdout_fights": int(len(baseline_frame)),
        "baseline": {
            "joint_log_loss": baseline_joint,
            "method_log_loss": baseline_method,
            "winner_log_loss": baseline_winner,
            "duration_crps_seconds": baseline_duration,
            "observable_action_error": baseline_action,
            "observable_action_errors": baseline_errors,
            "full_metrics": baseline,
        },
        "tuned": {
            "joint_log_loss": tuned_joint,
            "method_log_loss": tuned_method,
            "winner_log_loss": tuned_winner,
            "duration_crps_seconds": tuned_duration,
            "observable_action_error": tuned_action,
            "observable_action_errors": tuned_errors,
            "full_metrics": tuned,
        },
        "tuned_minus_baseline": {
            "joint_log_loss": tuned_joint - baseline_joint,
            "method_log_loss": tuned_method - baseline_method,
            "winner_log_loss": tuned_winner - baseline_winner,
            "duration_crps_seconds": tuned_duration - baseline_duration,
            "observable_action_error": tuned_action - baseline_action,
        },
        "gates": gates,
        "validation_status": "retained" if retained else "rejected_neutral_fallback",
        "candidate_simulator_config": candidate_config,
        "simulator_config": selected_config,
    }
    body["mechanics_profile_id"] = (
        f"mechanics-{canonical_sha256(selected_config)[:12]}"
    )
    body["validation_sha256"] = canonical_sha256(body)
    atomic_write_json(output, body)
    return body
