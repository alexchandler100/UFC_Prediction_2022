"""Evaluate weight-group-specific winner models without changing production.

The grouping follows the user's earlier research: men below welterweight,
welterweight through middleweight, light heavyweight and heavyweight, and all
women. Model design and pooling weights are selected on 2019--2022, then the
frozen choices are scored on 2023--2026.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from evaluate_model_families import (
    _align_reference,
    _atomic_write,
    _lineage,
    _metric_mapping,
)
from evaluate_style_matchup_challenger import event_block_difference_interval
from fight_predictor import PointInTimeDatasetBuilder, TemporalFightPredictor
from fight_predictor.point_in_time import REGULARIZATION_C_GRID
from market_tracker._common import canonical_hash


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "content/data"
DEFAULT_POINT_IN_TIME = DATA / "processed/ufc_fights_point_in_time.csv"
DEFAULT_RAW_FIGHTS = DATA / "processed/ufc_fights_reported_doubled.csv"
DEFAULT_FIGHTERS = DATA / "processed/fighter_stats.csv"
DEFAULT_MODEL_ARTIFACT = DATA / "external/winner_model.json"
DEFAULT_REPORT = DATA / "model_research/division_group_model_comparison.json"
DEFAULT_DETAIL = DATA / "model_research/division_group_model_comparison.csv"
DEFAULT_DEVELOPMENT_DETAIL = (
    DATA / "model_research/division_group_model_development.csv"
)
DEFAULT_DEVELOPMENT_YEARS = (2019, 2020, 2021, 2022)
DEFAULT_EVALUATION_YEARS = (2023, 2024, 2025, 2026)
EXPERIMENT_VERSION = "division-group-logistic-pooling-v1"
REPORT_SCHEMA_VERSION = 1
MAX_RUNTIME_MINUTES = 60.0
MINIMUM_GROUP_TRAINING_FIGHTS = 250
POOLING_WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

LOWER_MENS_DIVISIONS = frozenset(
    {"Flyweight", "Bantamweight", "Featherweight", "Lightweight"}
)
MIDDLE_MENS_DIVISIONS = frozenset({"Welterweight", "Middleweight"})
UPPER_MENS_DIVISIONS = frozenset(
    {"Light Heavyweight", "Heavyweight", "Super Heavyweight"}
)
GROUP_ORDER = (
    "mens_below_welterweight",
    "mens_welter_to_middle",
    "mens_light_heavy_plus",
    "womens_all",
    "unclassified",
)
CANDIDATE_ORDER = (
    "global_only",
    "shared_pooling_weight",
    "group_specific_pooling_weights",
    "fully_separate_groups",
)


def division_group(division: object) -> str:
    """Map one normalized UFCStats division into the predeclared groups."""

    value = " ".join(str(division or "").split())
    if value.startswith("Women's "):
        return "womens_all"
    if value in LOWER_MENS_DIVISIONS:
        return "mens_below_welterweight"
    if value in MIDDLE_MENS_DIVISIONS:
        return "mens_welter_to_middle"
    if value in UPPER_MENS_DIVISIONS:
        return "mens_light_heavy_plus"
    return "unclassified"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(bounded / (1.0 - bounded))


def pooled_probability(
    global_probability: np.ndarray,
    group_probability: np.ndarray,
    group_weight: float | np.ndarray,
) -> np.ndarray:
    """Blend in log-odds space while preserving exact side-swap symmetry."""

    weight = np.asarray(group_weight, dtype=float)
    if (weight < 0.0).any() or (weight > 1.0).any():
        raise ValueError("division pooling weights must be between zero and one")
    return _sigmoid(
        (1.0 - weight) * _logit(global_probability)
        + weight * _logit(group_probability)
    )


def _rolling_group_probabilities(
    frame: pd.DataFrame,
    features: Sequence[str],
    c_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    targets = []
    for train_indices, test_indices in TemporalFightPredictor._rolling_splits(
        frame["date"]
    ):
        training = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        if training["target"].nunique() < 2 or test.empty:
            continue
        pipeline = TemporalFightPredictor._fit_pipeline(
            training[list(features)], training["target"], c_value
        )
        probabilities.append(
            TemporalFightPredictor._pipeline_probability(
                pipeline, test[list(features)]
            )
        )
        targets.append(test["target"].to_numpy(dtype=int))
    if not probabilities:
        raise ValueError("division group has no valid rolling validation splits")
    return np.concatenate(targets), np.concatenate(probabilities)


def _fit_group_fold(
    training: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
) -> tuple[np.ndarray, dict[str, object]]:
    if len(training) < MINIMUM_GROUP_TRAINING_FIGHTS:
        raise ValueError(
            f"division group has only {len(training)} prior fights; "
            f"needs {MINIMUM_GROUP_TRAINING_FIGHTS}"
        )
    if training["target"].nunique() < 2:
        raise ValueError("division group training has only one outcome class")
    scores = {}
    rolling_by_c = {}
    for c_value in REGULARIZATION_C_GRID:
        targets, probability = _rolling_group_probabilities(
            training, features, float(c_value)
        )
        rolling_by_c[float(c_value)] = (targets, probability)
        scores[str(c_value)] = float(
            log_loss(targets, probability, labels=[0, 1])
        )
    selected_c = min(
        (float(value) for value in REGULARIZATION_C_GRID),
        key=lambda value: scores[str(value)],
    )
    calibration_target, calibration_probability = rolling_by_c[selected_c]
    slope = TemporalFightPredictor._fit_symmetric_calibration_slope(
        calibration_target, calibration_probability
    )
    pipeline = TemporalFightPredictor._fit_pipeline(
        training[list(features)], training["target"], selected_c
    )
    raw = TemporalFightPredictor._pipeline_probability(
        pipeline, test[list(features)]
    )
    probability = TemporalFightPredictor._calibrate(raw, slope)
    return probability, {
        "training_fights": len(training),
        "test_fights": len(test),
        "training_start": training["date"].min().strftime("%Y-%m-%d"),
        "training_through": training["date"].max().strftime("%Y-%m-%d"),
        "selected_c": selected_c,
        "calibration_slope": slope,
        "rolling_log_loss_by_c": scores,
    }


def _training_and_test(
    point: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_start = pd.Timestamp(year=year, month=1, day=1)
    training = point.loc[
        (point["date"] >= test_start - pd.DateOffset(years=10))
        & (point["date"] < test_start)
    ].copy()
    test = point.loc[point["date"].dt.year.eq(year)].copy()
    if len(training) < 500 or test.empty:
        raise ValueError(f"year {year} lacks training or test fights")
    return training, test


def _group_predictions(
    point: pd.DataFrame,
    baseline: pd.DataFrame,
    features: Sequence[str],
    years: Sequence[int],
    *,
    started: float,
    maximum_minutes: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    detail_rows = point.loc[point["date"].dt.year.isin(years)].copy()
    detail = _align_reference(_lineage(detail_rows), baseline)
    detail = detail.merge(
        detail_rows[["fight_id", "division"]],
        on="fight_id",
        how="left",
        validate="one_to_one",
    )
    detail["division_group"] = detail["division"].map(division_group)
    probability_by_fight: dict[str, float] = {}
    status_by_fight: dict[str, str] = {}
    diagnostics: dict[str, object] = {}
    for year in years:
        if time.monotonic() - started > maximum_minutes * 60.0:
            raise TimeoutError("division group experiment reached its runtime limit")
        training, test = _training_and_test(point, int(year))
        training = training.assign(
            division_group=training["division"].map(division_group)
        )
        test = test.assign(division_group=test["division"].map(division_group))
        year_diagnostics = {}
        for group in GROUP_ORDER:
            group_test = test.loc[test["division_group"].eq(group)].copy()
            if group_test.empty:
                continue
            if group == "unclassified":
                group_reference = detail.loc[
                    detail["fight_id"].astype(str).isin(
                        group_test["fight_id"].astype(str)
                    )
                ].set_index("fight_id")["current_logistic_probability"]
                for fight_id in group_test["fight_id"].astype(str):
                    probability_by_fight[fight_id] = float(
                        group_reference.loc[fight_id]
                    )
                    status_by_fight[fight_id] = "global_fallback_unclassified"
                year_diagnostics[group] = {
                    "training_fights": 0,
                    "test_fights": len(group_test),
                    "status": "global_fallback_unclassified",
                }
                continue
            group_training = training.loc[
                training["division_group"].eq(group)
            ].copy()
            probability, fold = _fit_group_fold(
                group_training, group_test, features
            )
            probability_by_fight.update(
                zip(group_test["fight_id"].astype(str), probability.astype(float))
            )
            status_by_fight.update(
                (fight_id, "separate_group_model")
                for fight_id in group_test["fight_id"].astype(str)
            )
            year_diagnostics[group] = {
                **fold,
                "status": "separate_group_model",
            }
        diagnostics[str(year)] = year_diagnostics
    detail["group_model_probability"] = detail["fight_id"].astype(str).map(
        probability_by_fight
    )
    detail["group_model_status"] = detail["fight_id"].astype(str).map(
        status_by_fight
    )
    if detail[["group_model_probability", "group_model_status"]].isna().any().any():
        raise ValueError("division group model missed an evaluation fight")
    return detail, diagnostics


def _candidate_probabilities(
    frame: pd.DataFrame,
    shared_weight: float,
    group_weights: Mapping[str, float],
) -> dict[str, np.ndarray]:
    global_probability = frame["current_logistic_probability"].to_numpy(
        dtype=float
    )
    group_probability = frame["group_model_probability"].to_numpy(dtype=float)
    weights = frame["division_group"].map(group_weights).to_numpy(dtype=float)
    return {
        "global_only": global_probability,
        "shared_pooling_weight": pooled_probability(
            global_probability, group_probability, shared_weight
        ),
        "group_specific_pooling_weights": pooled_probability(
            global_probability, group_probability, weights
        ),
        "fully_separate_groups": group_probability,
    }


def _loss(frame: pd.DataFrame, probability: np.ndarray) -> float:
    return float(log_loss(frame["target"], probability, labels=[0, 1]))


def select_pooling_design(
    development: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Choose weights only from the earlier development fights."""

    global_probability = development["current_logistic_probability"].to_numpy(
        dtype=float
    )
    group_probability = development["group_model_probability"].to_numpy(
        dtype=float
    )
    shared_scores = {
        str(weight): _loss(
            development,
            pooled_probability(global_probability, group_probability, weight),
        )
        for weight in POOLING_WEIGHT_GRID
    }
    shared_weight = min(
        POOLING_WEIGHT_GRID, key=lambda value: shared_scores[str(value)]
    )
    group_weights = {}
    group_weight_scores = {}
    for group in GROUP_ORDER:
        rows = development.loc[development["division_group"].eq(group)]
        if group == "unclassified" or rows.empty:
            group_weights[group] = 0.0
            group_weight_scores[group] = (
                {
                    "0.0": _loss(
                        rows,
                        rows["current_logistic_probability"].to_numpy(
                            dtype=float
                        ),
                    )
                }
                if not rows.empty
                else {}
            )
            continue
        scores = {
            str(weight): _loss(
                rows,
                pooled_probability(
                    rows["current_logistic_probability"].to_numpy(dtype=float),
                    rows["group_model_probability"].to_numpy(dtype=float),
                    weight,
                ),
            )
            for weight in POOLING_WEIGHT_GRID
        }
        group_weights[group] = min(
            POOLING_WEIGHT_GRID, key=lambda value: scores[str(value)]
        )
        group_weight_scores[group] = scores
    candidates = _candidate_probabilities(
        development, float(shared_weight), group_weights
    )
    candidate_scores = {
        name: _loss(development, probability)
        for name, probability in candidates.items()
    }
    selected = min(
        CANDIDATE_ORDER,
        key=lambda name: (candidate_scores[name], CANDIDATE_ORDER.index(name)),
    )
    return {
        "selected_design": selected,
        "selected_shared_group_weight": float(shared_weight),
        "selected_group_weights": group_weights,
        "candidate_log_loss": candidate_scores,
        "shared_weight_log_loss": shared_scores,
        "group_weight_log_loss": group_weight_scores,
        "weight_grid": list(POOLING_WEIGHT_GRID),
    }, candidates


def _slice_metrics(
    frame: pd.DataFrame,
    selected_column: str,
    group_column: str,
) -> dict[str, object]:
    result = {}
    for name, rows in frame.groupby(group_column, sort=True):
        current = _metric_mapping(rows, "current_logistic_probability")
        selected = _metric_mapping(rows, selected_column)
        result[str(name)] = {
            "fights": len(rows),
            "current_logistic": current,
            "selected_division_design": selected,
            "selected_minus_current_log_loss": selected["log_loss"]
            - current["log_loss"],
        }
    return result


def build_comparison(
    *,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTERS,
    model_artifact_path: Path = DEFAULT_MODEL_ARTIFACT,
    development_years: Sequence[int] = DEFAULT_DEVELOPMENT_YEARS,
    evaluation_years: Sequence[int] = DEFAULT_EVALUATION_YEARS,
    max_runtime_minutes: float = 55.0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if not 0.0 < max_runtime_minutes <= MAX_RUNTIME_MINUTES:
        raise ValueError("maximum runtime must be positive and at most 60 minutes")
    started = time.monotonic()
    point = pd.read_csv(point_in_time_path, low_memory=False)
    point["date"] = pd.to_datetime(point["date"], errors="raise")
    raw = pd.read_csv(raw_fights_path, low_memory=False)
    fighters = pd.read_csv(fighter_stats_path, low_memory=False)
    artifact = json.loads(model_artifact_path.read_text(encoding="utf-8"))
    features = tuple(str(value) for value in artifact["feature_columns"])
    builder = PointInTimeDatasetBuilder(raw, fighters)
    if features != tuple(builder.feature_columns):
        raise ValueError("winner artifact and division experiment features disagree")
    development_years = tuple(sorted(set(int(value) for value in development_years)))
    evaluation_years = tuple(sorted(set(int(value) for value in evaluation_years)))
    if set(development_years) & set(evaluation_years):
        raise ValueError("development and evaluation years overlap")
    predictor = TemporalFightPredictor(point, builder)

    development_reference = predictor.walk_forward_predictions(development_years)
    development, development_folds = _group_predictions(
        point,
        development_reference,
        features,
        development_years,
        started=started,
        maximum_minutes=max_runtime_minutes,
    )
    selection, development_candidates = select_pooling_design(development)
    for name, probability_values in development_candidates.items():
        development[f"{name}_probability"] = probability_values

    evaluation_reference = predictor.walk_forward_predictions(evaluation_years)
    detail, evaluation_folds = _group_predictions(
        point,
        evaluation_reference,
        features,
        evaluation_years,
        started=started,
        maximum_minutes=max_runtime_minutes,
    )
    evaluation_candidates = _candidate_probabilities(
        detail,
        float(selection["selected_shared_group_weight"]),
        selection["selected_group_weights"],
    )
    for name, probability_values in evaluation_candidates.items():
        detail[f"{name}_probability"] = probability_values
    selected_design = str(selection["selected_design"])
    selected_column = f"{selected_design}_probability"
    probability_columns = {
        "current_logistic": "current_logistic_probability",
        **{
            name: f"{name}_probability"
            for name in CANDIDATE_ORDER
            if name != "global_only"
        },
    }
    evaluation_metrics = {
        name: _metric_mapping(detail, column)
        for name, column in probability_columns.items()
    }
    development_metrics = {
        name: _metric_mapping(
            development,
            "current_logistic_probability"
            if name == "current_logistic"
            else f"{name}_probability",
        )
        for name in ("current_logistic", *CANDIDATE_ORDER[1:])
    }
    intervals = {
        name: event_block_difference_interval(
            detail, column, "current_logistic_probability"
        )
        for name, column in probability_columns.items()
        if name != "current_logistic"
    }
    if selected_design == "global_only":
        selected_interval = {
            "definition": "shared model selected; difference is exactly zero",
            "point_difference": 0.0,
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.0,
            "event_count": int(detail["event_id"].nunique()),
            "fight_count": len(detail),
        }
        selected_difference = 0.0
        conclusion = (
            "Earlier fights selected the shared model, so separate division-group "
            "coefficients did not earn activation."
        )
    else:
        selected_interval = intervals[selected_design]
        selected_difference = float(selected_interval["point_difference"])
        if (
            selected_difference < 0.0
            and float(selected_interval["ci_95_upper"]) < 0.0
        ):
            conclusion = (
                "The development-selected division design improved later "
                "probability quality, but it still requires prospective "
                "confirmation."
            )
        elif selected_difference < 0.0:
            conclusion = (
                "The selected division design had a better later point estimate, "
                "but the uncertainty range includes no improvement."
            )
        else:
            conclusion = (
                "The selected division design was worse on the later fights and "
                "should not replace the shared production model."
            )

    logical_detail = detail.copy()
    logical_detail["date"] = logical_detail["date"].dt.strftime("%Y-%m-%d")
    logical_development = development.copy()
    logical_development["date"] = logical_development["date"].dt.strftime(
        "%Y-%m-%d"
    )
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "paper_only": True,
        "production_action": "none",
        "plain_language_method": (
            "Fit the same 82-input zero-intercept logistic model separately for "
            "three men's weight ranges and all women's divisions. Compare fully "
            "separate fits with fixed partial pooling back toward the shared model."
        ),
        "division_groups": {
            "mens_below_welterweight": sorted(LOWER_MENS_DIVISIONS),
            "mens_welter_to_middle": sorted(MIDDLE_MENS_DIVISIONS),
            "mens_light_heavy_plus": sorted(UPPER_MENS_DIVISIONS),
            "womens_all": "every division whose label starts with Women's",
            "unclassified": "catch weight and open weight; shared-model fallback",
        },
        "development": {
            "years": list(development_years),
            "fights": len(development),
            "events": int(development["event_id"].nunique()),
            "selection": selection,
            "metrics": development_metrics,
            "by_group": _slice_metrics(
                development,
                f"{selection['selected_design']}_probability",
                "division_group",
            ),
            "by_division": _slice_metrics(
                development,
                f"{selection['selected_design']}_probability",
                "division",
            ),
            "group_counts": development["division_group"].value_counts().to_dict(),
            "folds": development_folds,
        },
        "evaluation": {
            "years": list(evaluation_years),
            "fights": len(detail),
            "events": int(detail["event_id"].nunique()),
            "selected_design": selected_design,
            "selected_design_interval": selected_interval,
            "metrics": evaluation_metrics,
            "paired_log_loss_intervals": intervals,
            "by_group": _slice_metrics(
                detail, selected_column, "division_group"
            ),
            "by_division": _slice_metrics(detail, selected_column, "division"),
            "folds": evaluation_folds,
        },
        "plain_language_conclusion": conclusion,
        "important_limits": [
            "2023-2026 has been reused by earlier research and is not a pristine final test",
            "smaller groups estimate the same 82 coefficients from fewer fights",
            "catch-weight and open-weight fights retain the shared model",
            "predictability by division is descriptive and can change by era",
            "production predictions and betting behavior are unchanged",
        ],
        "source_sha256": {
            "evaluator": sha256(Path(__file__).read_bytes()).hexdigest(),
            "point_in_time": sha256(point_in_time_path.read_bytes()).hexdigest(),
            "raw_fights": sha256(raw_fights_path.read_bytes()).hexdigest(),
            "fighter_stats": sha256(fighter_stats_path.read_bytes()).hexdigest(),
            "model_artifact": sha256(model_artifact_path.read_bytes()).hexdigest(),
            "evaluation_detail_logical": canonical_hash(
                logical_detail.to_dict("records")
            ),
            "development_detail_logical": canonical_hash(
                logical_development.to_dict("records")
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    return report, detail, development


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-in-time", type=Path, default=DEFAULT_POINT_IN_TIME)
    parser.add_argument("--raw-fights", type=Path, default=DEFAULT_RAW_FIGHTS)
    parser.add_argument("--fighter-stats", type=Path, default=DEFAULT_FIGHTERS)
    parser.add_argument("--model-artifact", type=Path, default=DEFAULT_MODEL_ARTIFACT)
    parser.add_argument(
        "--development-years",
        nargs="+",
        type=int,
        default=DEFAULT_DEVELOPMENT_YEARS,
    )
    parser.add_argument(
        "--evaluation-years",
        nargs="+",
        type=int,
        default=DEFAULT_EVALUATION_YEARS,
    )
    parser.add_argument("--max-runtime-minutes", type=float, default=55.0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument(
        "--development-detail", type=Path, default=DEFAULT_DEVELOPMENT_DETAIL
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report, detail, development = build_comparison(
        point_in_time_path=arguments.point_in_time,
        raw_fights_path=arguments.raw_fights,
        fighter_stats_path=arguments.fighter_stats,
        model_artifact_path=arguments.model_artifact,
        development_years=arguments.development_years,
        evaluation_years=arguments.evaluation_years,
        max_runtime_minutes=arguments.max_runtime_minutes,
    )
    if not arguments.dry_run:
        _atomic_write(
            arguments.report,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        _atomic_write(
            arguments.detail,
            detail.to_csv(index=False, lineterminator="\n", float_format="%.15g"),
        )
        _atomic_write(
            arguments.development_detail,
            development.to_csv(
                index=False, lineterminator="\n", float_format="%.15g"
            ),
        )
    print(
        "Selected division design on 2019-2022: "
        f"{report['development']['selection']['selected_design']}"
    )
    for name, metrics in report["evaluation"]["metrics"].items():
        print(
            f"{name}: log loss={metrics['log_loss']:.5f}, "
            f"accuracy={metrics['accuracy']:.2%}, Brier={metrics['brier']:.5f}"
        )
    print(report["plain_language_conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
