"""Chronologically evaluate smaller and combined UFC winner feature sets.

The current 82-variable logistic model is the fixed reference.  Candidate
methods select variables using only fights before each evaluated year:

* robust scaling of all 82 variables;
* correlation pruning followed by L2 logistic regression;
* elastic-net selection of a smaller baseline subset;
* that selected subset plus every combination of up to three derived families;
* elastic net over the complete baseline-and-derived pool; and
* an uncentered SVD reduction of the complete pool.

All transformations are odd linear maps and every logistic model has zero
intercept, preserving p(A, B) = 1 - p(B, A).  This is development evidence and
cannot update the production model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import RobustScaler, StandardScaler

from evaluate_current_model_vs_market import (
    DEFAULT_DETAIL as DEFAULT_MARKET_DETAIL,
    DEFAULT_REPORT as DEFAULT_MARKET_REPORT,
    _atomic_write_text,
    _file_sha256,
    _metric_mapping,
)
from evaluate_style_matchup_challenger import (
    _align_predictions,
    _metric,
    _validate_candidate_baseline_features,
    event_block_difference_interval,
)
from evaluate_winner_feature_challengers import (
    DEFAULT_FIGHTER_STATS,
    DEFAULT_POINT_IN_TIME,
    DEFAULT_RAW_FIGHTS,
)
from fight_predictor import (
    PointInTimeDatasetBuilder,
    StanceMatchupDatasetBuilder,
    StyleMatchupDatasetBuilder,
    TemporalFightPredictor,
)
from fight_predictor.point_in_time import REGULARIZATION_C_GRID


REPORT_SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "chronological-feature-selection-v1"
ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "content/data/model_research/feature_selection.json"
DEFAULT_DETAIL = ROOT / "content/data/model_research/feature_selection.csv"
DEFAULT_MAX_RUNTIME_MINUTES = 55.0
MAX_ALLOWED_RUNTIME_MINUTES = 60.0
OUTER_YEARS = (2022, 2023, 2024, 2025, 2026)
PRODUCTION_YEARS = (2023, 2024, 2025, 2026)
INNER_SPLITS = 4
CORRELATION_THRESHOLD = 0.90

REFERENCE = "current_82"
CANDIDATES = (
    "robust_82",
    "correlation_pruned",
    "selected_baseline",
    "selected_plus_derived",
    "elastic_full_pool",
    "svd_full_pool",
)
ALL_MODELS = (REFERENCE, *CANDIDATES)


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    c_value: float
    l1_ratio: float | None = None
    components: int | None = None

    @property
    def key(self) -> str:
        values = [self.kind, f"c={self.c_value:g}"]
        if self.l1_ratio is not None:
            values.append(f"l1={self.l1_ratio:g}")
        if self.components is not None:
            values.append(f"k={self.components}")
        return ":".join(values)


@dataclass
class FittedPipeline:
    imputer: SimpleImputer
    scaler: StandardScaler | RobustScaler
    reducer: TruncatedSVD | None
    model: LogisticRegression


def _rolling_splits(dates: pd.Series, n_splits: int = INNER_SPLITS):
    parsed = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = np.array(sorted(parsed.unique()))
    if len(unique_dates) < n_splits + 2:
        raise ValueError("not enough event dates for chronological validation")
    initial = max(1, len(unique_dates) // 2)
    chunks = [
        chunk
        for chunk in np.array_split(unique_dates[initial:], n_splits)
        if len(chunk)
    ]
    for chunk in chunks:
        train_mask = parsed < chunk[0]
        test_mask = parsed.isin(chunk)
        if train_mask.any() and test_mask.any():
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def _fit_pipeline(
    frame: pd.DataFrame,
    features: Sequence[str],
    config: ModelConfig,
) -> FittedPipeline:
    y = frame["target"]
    if y.nunique() < 2:
        raise ValueError("logistic training data requires both winners and losers")
    imputer = SimpleImputer(
        strategy="constant", fill_value=0.0, keep_empty_features=True
    )
    values = imputer.fit_transform(frame[list(features)])
    if config.kind == "robust":
        scaler: StandardScaler | RobustScaler = RobustScaler(
            with_centering=False, quantile_range=(10.0, 90.0)
        )
    else:
        scaler = StandardScaler(with_mean=False)
    values = scaler.fit_transform(values)

    reducer: TruncatedSVD | None = None
    if config.kind == "svd":
        if config.components is None:
            raise ValueError("SVD configuration requires a component count")
        reducer = TruncatedSVD(
            n_components=min(config.components, values.shape[1] - 1),
            algorithm="randomized",
            n_iter=7,
            random_state=48,
        )
        values = reducer.fit_transform(values)

    if config.kind == "elastic":
        model = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=config.l1_ratio,
            C=config.c_value,
            fit_intercept=False,
            max_iter=8_000,
            tol=1e-4,
            random_state=48,
        )
    else:
        model = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=config.c_value,
            fit_intercept=False,
            max_iter=30_000,
            random_state=48,
        )
    model.fit(values, y)
    return FittedPipeline(imputer, scaler, reducer, model)


def _pipeline_probability(
    pipeline: FittedPipeline,
    frame: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    values = pipeline.imputer.transform(frame[list(features)])
    values = pipeline.scaler.transform(values)
    if pipeline.reducer is not None:
        values = pipeline.reducer.transform(values)
    return pipeline.model.predict_proba(values)[:, 1]


def _oof_probability(
    frame: pd.DataFrame,
    features: Sequence[str],
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    for train_index, test_index in _rolling_splits(frame["date"]):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        pipeline = _fit_pipeline(train, features, config)
        probabilities.append(_pipeline_probability(pipeline, test, features))
        truth.append(test["target"].to_numpy(dtype=int))
    return np.concatenate(truth), np.concatenate(probabilities)


def _tune(
    frame: pd.DataFrame,
    features: Sequence[str],
    configs: Sequence[ModelConfig],
) -> tuple[ModelConfig, dict[str, float], np.ndarray, np.ndarray]:
    results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for config in configs:
        y_true, probability = _oof_probability(frame, features, config)
        score = float(log_loss(y_true, probability, labels=[0, 1]))
        results[config.key] = (score, y_true, probability)
    selected = min(configs, key=lambda item: results[item.key][0])
    scores = {key: value[0] for key, value in results.items()}
    _, y_true, probability = results[selected.key]
    return selected, scores, y_true, probability


def _calibration_slope(y_true: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(
        solver="lbfgs", C=1_000_000.0, fit_intercept=False, max_iter=10_000
    ).fit(logits, y_true)
    return float(np.clip(calibrator.coef_[0, 0], 0.25, 2.0))


def _calibrate(probability: np.ndarray, slope: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-np.clip(slope * logits, -709, 709)))


def _fit_predict_selected(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    configs: Sequence[ModelConfig],
) -> tuple[np.ndarray, dict[str, object], FittedPipeline]:
    selected, scores, y_oof, p_oof = _tune(train, features, configs)
    slope = _calibration_slope(y_oof, p_oof)
    pipeline = _fit_pipeline(train, features, selected)
    probability = _calibrate(
        _pipeline_probability(pipeline, test, features), slope
    )
    return probability, {
        "features": list(features),
        "feature_count": len(features),
        "selected_config": selected.key,
        "calibration_slope": slope,
        "inner_log_loss": scores,
    }, pipeline


def _l2_configs(kind: str = "standard") -> tuple[ModelConfig, ...]:
    return tuple(ModelConfig(kind, float(value)) for value in REGULARIZATION_C_GRID)


ELASTIC_CONFIGS = tuple(
    ModelConfig("elastic", c_value, l1_ratio=l1_ratio)
    for c_value in (0.003, 0.01, 0.03, 0.1, 0.3)
    for l1_ratio in (0.5, 0.8, 1.0)
)
SVD_CONFIGS = tuple(
    ModelConfig("svd", c_value, components=components)
    for components in (16, 32, 48, 64)
    for c_value in (0.003, 0.01, 0.03)
)


def _correlation_components(
    frame: pd.DataFrame,
    features: Sequence[str],
    threshold: float = CORRELATION_THRESHOLD,
) -> list[list[str]]:
    values = frame[list(features)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    correlation = values.corr().abs()
    parent = {feature: feature for feature in features}

    def find(feature: str) -> str:
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]
        return feature

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(features):
        for right in features[index + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and float(value) >= threshold:
                union(left, right)
    groups: dict[str, list[str]] = {}
    for feature in features:
        groups.setdefault(find(feature), []).append(feature)
    return [sorted(group) for group in groups.values()]


def _select_correlation_representatives(
    train: pd.DataFrame,
    features: Sequence[str],
) -> tuple[list[str], dict[str, object]]:
    groups = _correlation_components(train, features)
    selected: list[str] = []
    removed: dict[str, list[str]] = {}
    fixed = ModelConfig("standard", 0.03)
    for group in groups:
        if len(group) == 1:
            selected.extend(group)
            continue
        scores: dict[str, float] = {}
        for feature in group:
            y_true, probability = _oof_probability(train, [feature], fixed)
            scores[feature] = float(log_loss(y_true, probability, labels=[0, 1]))
        winner = min(group, key=lambda item: (scores[item], item))
        selected.append(winner)
        removed[winner] = [item for item in group if item != winner]
    ordered = [feature for feature in features if feature in set(selected)]
    return ordered, {
        "threshold": CORRELATION_THRESHOLD,
        "component_count": len(groups),
        "selected_count": len(ordered),
        "removed_by_representative": removed,
    }


def _elastic_selected_features(
    train: pd.DataFrame,
    features: Sequence[str],
) -> tuple[list[str], dict[str, object]]:
    config, scores, _y, _p = _tune(train, features, ELASTIC_CONFIGS)
    pipeline = _fit_pipeline(train, features, config)
    coefficients = np.asarray(pipeline.model.coef_[0], dtype=float)
    nonzero = np.flatnonzero(np.abs(coefficients) > 1e-8)
    if len(nonzero) < 5:
        nonzero = np.argsort(np.abs(coefficients))[-5:]
    selected_set = {features[int(index)] for index in nonzero}
    selected = [feature for feature in features if feature in selected_set]
    return selected, {
        "selection_config": config.key,
        "selection_inner_log_loss": scores,
        "selected_count": len(selected),
        "selected_features": selected,
    }


def _derived_combinations(
    families: dict[str, tuple[str, ...]],
) -> list[tuple[str, ...]]:
    names = sorted(families)
    return [
        combination
        for size in range(4)
        for combination in itertools.combinations(names, size)
    ]


def _select_derived_combination(
    train: pd.DataFrame,
    base_features: Sequence[str],
    families: dict[str, tuple[str, ...]],
    fixed_c: float,
) -> tuple[list[str], dict[str, object]]:
    scores: dict[tuple[str, ...], float] = {}
    fixed = ModelConfig("standard", fixed_c)
    for combination in _derived_combinations(families):
        additions = [
            feature
            for family in combination
            for feature in families[family]
        ]
        features = [*base_features, *additions]
        y_true, probability = _oof_probability(train, features, fixed)
        scores[combination] = float(log_loss(y_true, probability, labels=[0, 1]))
    selected = min(scores, key=lambda item: (scores[item], len(item), item))
    selected_features = [
        *base_features,
        *(feature for family in selected for feature in families[family]),
    ]
    ranked = sorted(scores, key=lambda item: (scores[item], len(item), item))
    return selected_features, {
        "tested_combinations": len(scores),
        "selected_families": list(selected),
        "selected_family_count": len(selected),
        "fixed_selection_c": fixed_c,
        "top_combinations": [
            {
                "families": list(combination),
                "inner_log_loss": scores[combination],
            }
            for combination in ranked[:10]
        ],
        "no_derived_inner_log_loss": scores[()],
        "selected_inner_log_loss": scores[selected],
    }


def _derived_families(
    style_columns: Sequence[str],
    stance_columns: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    style = set(style_columns)
    families = {
        "target_shares": tuple(sorted(
            item for item in style
            if any(f"_{part}_" in item for part in ("head", "body", "leg"))
            and "attempt_share" in item
        )),
        "position_shares": tuple(sorted(
            item for item in style
            if any(f"_{part}_" in item for part in ("distance", "clinch", "ground"))
            and "attempt_share" in item
        )),
        "career_matchups": tuple(sorted(
            item for item in style
            if item.startswith("career_")
            and item.endswith("_matchup")
            and "_style_matchup" not in item
        )),
        "recent_matchups": tuple(sorted(
            item for item in style if item.startswith("recent_3y_")
        )),
        "category_matchups": tuple(sorted(
            item for item in style if "_style_matchup" in item
        )),
        "stance_profile": tuple(sorted(
            item for item in stance_columns if item.startswith("stance_")
        )),
        "stance_open_matchups": tuple(sorted(
            item for item in stance_columns if item.startswith("open_stance_")
        )),
    }
    if any(not values for values in families.values()):
        missing = [name for name, values in families.items() if not values]
        raise RuntimeError(f"derived family construction produced empty groups: {missing}")
    flattened = [item for values in families.values() for item in values]
    expected = set(style_columns) | set(stance_columns)
    if len(flattened) != len(set(flattened)) or set(flattened) != expected:
        raise RuntimeError("derived feature families are not an exact partition")
    return families


def _build_pool(
    production: pd.DataFrame,
    raw: pd.DataFrame,
    fighters: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, tuple[str, ...]]]:
    baseline_builder = PointInTimeDatasetBuilder(raw, fighters)
    rebuilt = baseline_builder.build()
    style_builder = StyleMatchupDatasetBuilder(raw, fighters)
    style = style_builder.build()
    stance_builder = StanceMatchupDatasetBuilder(raw, fighters)
    stance = stance_builder.build()
    baseline_columns = list(baseline_builder.feature_columns)
    _validate_candidate_baseline_features(production, rebuilt, baseline_columns)
    _validate_candidate_baseline_features(production, style, baseline_columns)
    _validate_candidate_baseline_features(production, stance, baseline_columns)

    expected_ids = production["fight_id"].astype(str).tolist()
    for name, frame in (("style", style), ("stance", stance)):
        if frame["fight_id"].astype(str).tolist() != expected_ids:
            raise RuntimeError(f"{name} changed the fight set or order")
    style_extra = list(style_builder.feature_columns[len(baseline_columns):])
    stance_extra = list(stance_builder.feature_columns[len(baseline_columns):])
    pool = production.copy()
    pool[style_extra] = style[style_extra].to_numpy()
    pool[stance_extra] = stance[stance_extra].to_numpy()
    families = _derived_families(style_extra, stance_extra)
    return pool, baseline_columns, families


def _candidate_frame(
    test: pd.DataFrame,
    probability: np.ndarray,
) -> pd.DataFrame:
    columns = (
        "date", "event_id", "fight_id", "fighter_id", "opponent_id",
        "fighter", "opponent", "target",
    )
    result = test[list(columns)].copy()
    result["model_probability"] = probability
    return result


def _check_runtime(started: float, maximum_seconds: float) -> None:
    if time.monotonic() - started > maximum_seconds:
        raise RuntimeError("feature-selection experiment exceeded its runtime limit")


def _evaluate_candidates(
    pool: pd.DataFrame,
    baseline_columns: list[str],
    families: dict[str, tuple[str, ...]],
    reference: pd.DataFrame,
    years: tuple[int, ...],
    maximum_seconds: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    started = time.monotonic()
    frame = pool.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    derived_columns = [item for values in families.values() for item in values]
    full_pool = [*baseline_columns, *derived_columns]
    prediction_parts: dict[str, list[pd.DataFrame]] = {
        name: [] for name in CANDIDATES
    }
    fold_reports: dict[str, object] = {}

    for year in years:
        _check_runtime(started, maximum_seconds)
        fold_started = time.monotonic()
        test_start = pd.Timestamp(year=year, month=1, day=1)
        train_start = test_start - pd.DateOffset(years=10)
        train = frame[
            (frame["date"] >= train_start) & (frame["date"] < test_start)
        ].reset_index(drop=True)
        test = frame[frame["date"].dt.year.eq(year)].reset_index(drop=True)
        if len(train) < 500 or test.empty:
            continue
        print(f"Selecting features from {len(train)} earlier fights; testing {year}")
        year_report: dict[str, object] = {
            "train_fights": len(train),
            "test_fights": len(test),
        }

        probability, report, _ = _fit_predict_selected(
            train, test, baseline_columns, _l2_configs("robust")
        )
        prediction_parts["robust_82"].append(_candidate_frame(test, probability))
        year_report["robust_82"] = report

        pruned, pruning = _select_correlation_representatives(
            train, baseline_columns
        )
        probability, report, _ = _fit_predict_selected(
            train, test, pruned, _l2_configs()
        )
        report["selection"] = pruning
        prediction_parts["correlation_pruned"].append(
            _candidate_frame(test, probability)
        )
        year_report["correlation_pruned"] = report

        selected_base, selection = _elastic_selected_features(
            train, baseline_columns
        )
        probability, selected_report, _ = _fit_predict_selected(
            train, test, selected_base, _l2_configs()
        )
        selected_report["selection"] = selection
        prediction_parts["selected_baseline"].append(
            _candidate_frame(test, probability)
        )
        year_report["selected_baseline"] = selected_report

        selected_config = str(selected_report["selected_config"])
        fixed_c = float(selected_config.split("c=", 1)[1])
        combined, combination = _select_derived_combination(
            train, selected_base, families, fixed_c
        )
        probability, combined_report, _ = _fit_predict_selected(
            train, test, combined, _l2_configs()
        )
        combined_report["base_selection"] = selection
        combined_report["derived_selection"] = combination
        prediction_parts["selected_plus_derived"].append(
            _candidate_frame(test, probability)
        )
        year_report["selected_plus_derived"] = combined_report

        probability, report, pipeline = _fit_predict_selected(
            train, test, full_pool, ELASTIC_CONFIGS
        )
        coefficients = np.asarray(pipeline.model.coef_[0], dtype=float)
        report["nonzero_features"] = [
            feature
            for feature, coefficient in zip(full_pool, coefficients)
            if abs(float(coefficient)) > 1e-8
        ]
        report["nonzero_feature_count"] = len(report["nonzero_features"])
        prediction_parts["elastic_full_pool"].append(
            _candidate_frame(test, probability)
        )
        year_report["elastic_full_pool"] = report

        valid_svd = tuple(
            config for config in SVD_CONFIGS
            if config.components is not None and config.components < len(full_pool)
        )
        probability, report, pipeline = _fit_predict_selected(
            train, test, full_pool, valid_svd
        )
        if pipeline.reducer is not None:
            report["fitted_explained_variance_ratio"] = float(
                pipeline.reducer.explained_variance_ratio_.sum()
            )
        prediction_parts["svd_full_pool"].append(
            _candidate_frame(test, probability)
        )
        year_report["svd_full_pool"] = report

        year_report["elapsed_seconds"] = time.monotonic() - fold_started
        fold_reports[str(year)] = year_report
        print(
            f"Completed {year} selection in "
            f"{year_report['elapsed_seconds']:.1f} seconds"
        )

    predictions = {
        REFERENCE: reference.copy(),
        **{
            name: pd.concat(parts, ignore_index=True)
            for name, parts in prediction_parts.items()
            if parts
        },
    }
    if set(predictions) != set(ALL_MODELS):
        raise RuntimeError("one or more candidate methods produced no predictions")
    fold_reports["total_elapsed_seconds"] = time.monotonic() - started
    return predictions, fold_reports


def _align_all(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    aligned = predictions[REFERENCE].rename(
        columns={"model_probability": f"{REFERENCE}_probability"}
    )
    for name in CANDIDATES:
        aligned = _align_predictions(
            aligned,
            predictions[name],
            probability_name=f"{name}_probability",
        )
    return aligned


def _metrics(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        name: _metric(frame, f"{name}_probability") for name in ALL_MODELS
    }


def _comparisons(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        f"{name}_minus_{REFERENCE}": event_block_difference_interval(
            frame, f"{name}_probability", f"{REFERENCE}_probability"
        )
        for name in CANDIDATES
    }


def _source_identity_hash(frame: pd.DataFrame) -> str:
    values = "\n".join(frame["fight_id"].astype(str)).encode("utf-8")
    return sha256(values).hexdigest()


def build_evaluation(
    *,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTER_STATS,
    market_report_path: Path = DEFAULT_MARKET_REPORT,
    market_detail_path: Path = DEFAULT_MARKET_DETAIL,
    years: tuple[int, ...] = OUTER_YEARS,
    max_runtime_minutes: float = DEFAULT_MAX_RUNTIME_MINUTES,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not 0 < max_runtime_minutes <= MAX_ALLOWED_RUNTIME_MINUTES:
        raise ValueError("max runtime must be between 0 and 60 minutes")
    experiment_started = time.monotonic()
    maximum_seconds = max_runtime_minutes * 60.0
    production = pd.read_csv(point_in_time_path, low_memory=False)
    raw = pd.read_csv(raw_fights_path, low_memory=False)
    fighters = pd.read_csv(fighter_stats_path, low_memory=False)
    stored_market = pd.read_csv(market_detail_path, low_memory=False)
    market_report = json.loads(market_report_path.read_text(encoding="utf-8"))

    pool, baseline_columns, families = _build_pool(production, raw, fighters)
    reference_builder = PointInTimeDatasetBuilder(raw, fighters)
    reference = TemporalFightPredictor(
        production, reference_builder
    ).walk_forward_predictions(years)
    preparation_seconds = time.monotonic() - experiment_started
    remaining_seconds = maximum_seconds - preparation_seconds
    if remaining_seconds <= 0:
        raise RuntimeError(
            "Experiment exceeded its runtime limit while preparing features and "
            "reference predictions"
        )
    predictions, fold_reports = _evaluate_candidates(
        pool,
        baseline_columns,
        families,
        reference,
        years,
        remaining_seconds,
    )
    fold_reports["preparation_elapsed_seconds"] = preparation_seconds
    fold_reports["whole_experiment_elapsed_seconds"] = (
        time.monotonic() - experiment_started
    )
    aligned = _align_all(predictions)
    production_horizon = aligned[
        aligned["evaluation_year"].isin(PRODUCTION_YEARS)
    ].copy()
    production_metrics = _metrics(production_horizon)
    extended_metrics = _metrics(aligned)

    available_fight_ids = set(predictions[REFERENCE]["fight_id"].astype(str))
    market_source = stored_market[
        stored_market["fight_id"].astype(str).isin(available_fight_ids)
    ].copy()
    if market_source.empty:
        raise ValueError("No stored market fights overlap the requested evaluation years")
    market_paired = market_source.rename(
        columns={"model_probability": "stored_reference_probability"}
    )
    for name in ALL_MODELS:
        market_paired = _align_predictions(
            market_paired,
            predictions[name],
            probability_name=f"{name}_probability",
        )
    market_metrics = {
        "market": _metric_mapping(market_paired, "market_probability"),
        **{
            name: _metric_mapping(market_paired, f"{name}_probability")
            for name in ALL_MODELS
        },
    }
    ranked = sorted(
        ALL_MODELS,
        key=lambda name: float(production_metrics[name]["log_loss"]),
    )
    best = ranked[0]
    best_improves = (
        best != REFERENCE
        and production_metrics[best]["log_loss"]
        < production_metrics[REFERENCE]["log_loss"]
        and production_metrics[best]["brier"]
        <= production_metrics[REFERENCE]["brier"]
    )

    report: dict[str, object] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "promotable": False,
        "production_action": "none",
        "plain_language_method": (
            "For each test year, choose variables and model settings using only "
            "earlier fights, then compare predictions on that untouched year."
        ),
        "feature_pool": {
            "baseline_count": len(baseline_columns),
            "derived_count": sum(len(values) for values in families.values()),
            "total_count": len(baseline_columns)
            + sum(len(values) for values in families.values()),
            "derived_families": {
                name: list(values) for name, values in families.items()
            },
            "cardio_excluded": (
                "round history begins in late 2024 and the prior experiment "
                "showed unsafe early-year behavior plus worse 2026 probabilities"
            ),
        },
        "methods": {
            "current_82": "current StandardScaler plus L2 logistic reference",
            "robust_82": "same 82 variables with outlier-resistant scaling",
            "correlation_pruned": (
                "one chronologically best representative per >=0.90 correlation group"
            ),
            "selected_baseline": (
                "elastic net chooses baseline variables, then L2 refits them"
            ),
            "selected_plus_derived": (
                "selected baseline plus exhaustive zero/one/two/three derived-family search"
            ),
            "elastic_full_pool": (
                "elastic-net logistic selection across all baseline and derived variables"
            ),
            "svd_full_pool": (
                "uncentered SVD reduction followed by L2 logistic regression"
            ),
        },
        "symmetry_contract": {
            "centered_scaling_used": False,
            "intercepts_used": False,
            "all_transforms_preserve_negation_on_fighter_swap": True,
        },
        "sample": {
            "years": list(years),
            "extended_fights": len(aligned),
            "production_years": list(PRODUCTION_YEARS),
            "production_fights": len(production_horizon),
            "market_paired_fights": len(market_paired),
            "fight_identity_sha256": _source_identity_hash(production),
        },
        "selection_by_year": fold_reports,
        "production_walk_forward": {
            "metrics": production_metrics,
            "paired_log_loss_intervals": _comparisons(production_horizon),
        },
        "extended_walk_forward": {
            "metrics": extended_metrics,
            "paired_log_loss_intervals": _comparisons(aligned),
        },
        "market_paired": {
            "metrics": market_metrics,
            "candidate_minus_reference_intervals": _comparisons(market_paired),
        },
        "decision": {
            "ranked_by_production_log_loss": ranked,
            "best_method": best,
            "best_improves_reference_log_loss_and_brier": best_improves,
            "recommendation": (
                f"retain {best} as a research candidate"
                if best_improves
                else "retain the current 82-variable model as the reference"
            ),
            "production_action": "none; selection used the evaluation period",
        },
        "non_promotable_flags": [
            "model_methods_were_compared_on_the_current_evaluation_period",
            "profile_stance_is_not_historically_timestamped",
            "market_history_missing_2024",
            "prospective_confirmation_required",
        ],
        "source_sha256": {
            "point_in_time": _file_sha256(point_in_time_path),
            "raw_fights": _file_sha256(raw_fights_path),
            "fighter_stats": _file_sha256(fighter_stats_path),
            "market_report": _file_sha256(market_report_path),
            "market_detail": _file_sha256(market_detail_path),
        },
        "inherited_market_contract": market_report["market_contract"],
    }
    return report, aligned


def _parse_years(value: str) -> tuple[int, ...]:
    years = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not years or any(year < 1994 or year > 2100 for year in years):
        raise argparse.ArgumentTypeError("years must be comma-separated calendar years")
    return years


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--years", type=_parse_years, default=OUTER_YEARS)
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=DEFAULT_MAX_RUNTIME_MINUTES,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report, detail = build_evaluation(
        years=arguments.years,
        max_runtime_minutes=arguments.max_runtime_minutes,
    )
    if not arguments.dry_run:
        _atomic_write_text(
            arguments.detail,
            detail.to_csv(index=False, lineterminator="\n", float_format="%.15g"),
        )
        _atomic_write_text(
            arguments.report,
            json.dumps(
                report,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
        )
    metrics = report["production_walk_forward"]["metrics"]
    print(
        "Feature-selection log loss: "
        + ", ".join(
            f"{name}={metrics[name]['log_loss']:.6f}" for name in ALL_MODELS
        )
    )
    print(f"Decision: {report['decision']['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
