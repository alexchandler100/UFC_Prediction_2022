"""Compare distinct UFC winner-model families on identical future fights.

Every candidate uses the frozen 82-column point-in-time feature matrix.  For
each evaluated year, all fitting, tuning, and calibration use earlier fights;
the whole evaluated year is then untouched test data.  Nonlinear classifiers
are explicitly symmetrized so swapping the fighters complements the forecast.
Nothing in this script changes the production model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from evaluate_style_matchup_challenger import event_block_difference_interval
from fight_predictor import PointInTimeDatasetBuilder, TemporalFightPredictor
from fight_predictor.hierarchical_bayes import (
    HierarchicalBayesConfig,
    hierarchical_bayes_predict,
)
from fight_predictor.point_in_time import _metrics
from market_tracker._common import canonical_hash


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "content/data"
DEFAULT_POINT_IN_TIME = DATA / "processed/ufc_fights_point_in_time.csv"
DEFAULT_RAW_FIGHTS = DATA / "processed/ufc_fights_reported_doubled.csv"
DEFAULT_FIGHTERS = DATA / "processed/fighter_stats.csv"
DEFAULT_MODEL_ARTIFACT = DATA / "external/winner_model.json"
DEFAULT_REPORT = DATA / "model_research/model_family_comparison.json"
DEFAULT_DETAIL = DATA / "model_research/model_family_comparison.csv"
DEFAULT_YEARS = (2023, 2024, 2025, 2026)
DEFAULT_FAMILIES = (
    "gaussian_naive_bayes",
    "random_forest",
    "hist_gradient_boosting",
    "neural_net",
    "hierarchical_bayes",
    "xgboost",
)
EXPERIMENT_VERSION = "chronological-winner-model-families-v1"
REPORT_SCHEMA_VERSION = 1
MAX_RUNTIME_MINUTES = 60.0


@dataclass(frozen=True)
class CandidateConfig:
    key: str
    parameters: dict[str, object]


@dataclass
class FittedClassifier:
    imputer: SimpleImputer
    scaler: StandardScaler | None
    classifier: object


CONFIGS: dict[str, tuple[CandidateConfig, ...]] = {
    "gaussian_naive_bayes": tuple(
        CandidateConfig(
            key=f"var_smoothing={value:g}", parameters={"var_smoothing": value}
        )
        for value in (1e-8, 1e-6, 1e-4, 1e-2)
    ),
    "random_forest": (
        CandidateConfig("depth=6,leaf=10", {"max_depth": 6, "min_samples_leaf": 10}),
        CandidateConfig("depth=10,leaf=10", {"max_depth": 10, "min_samples_leaf": 10}),
        CandidateConfig("depth=none,leaf=10", {"max_depth": None, "min_samples_leaf": 10}),
        CandidateConfig("depth=none,leaf=25", {"max_depth": None, "min_samples_leaf": 25}),
    ),
    "hist_gradient_boosting": (
        CandidateConfig("leaves=15,rate=.03", {"max_leaf_nodes": 15, "learning_rate": 0.03}),
        CandidateConfig("leaves=31,rate=.03", {"max_leaf_nodes": 31, "learning_rate": 0.03}),
        CandidateConfig("leaves=15,rate=.07", {"max_leaf_nodes": 15, "learning_rate": 0.07}),
        CandidateConfig("leaves=31,rate=.07", {"max_leaf_nodes": 31, "learning_rate": 0.07}),
    ),
    "neural_net": (
        CandidateConfig("hidden=32,alpha=.001", {"hidden_layer_sizes": (32,), "alpha": 0.001}),
        CandidateConfig("hidden=32,alpha=.01", {"hidden_layer_sizes": (32,), "alpha": 0.01}),
        CandidateConfig("hidden=64x32,alpha=.001", {"hidden_layer_sizes": (64, 32), "alpha": 0.001}),
        CandidateConfig("hidden=64x32,alpha=.01", {"hidden_layer_sizes": (64, 32), "alpha": 0.01}),
    ),
    "xgboost": (
        CandidateConfig("depth=2,rate=.03", {"max_depth": 2, "learning_rate": 0.03}),
        CandidateConfig("depth=3,rate=.03", {"max_depth": 3, "learning_rate": 0.03}),
        CandidateConfig("depth=2,rate=.07", {"max_depth": 2, "learning_rate": 0.07}),
        CandidateConfig("depth=3,rate=.07", {"max_depth": 3, "learning_rate": 0.07}),
    ),
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(bounded / (1.0 - bounded))


def _fit_calibration_slope(target: np.ndarray, probability: np.ndarray) -> float:
    logits = _logit(probability)

    def objective(slope: float) -> float:
        return float(log_loss(target, _sigmoid(float(slope) * logits), labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(0.05, 3.0), method="bounded")
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError("probability calibration did not converge")
    return float(result.x)


def _calibrate(probability: np.ndarray, slope: float) -> np.ndarray:
    return _sigmoid(float(slope) * _logit(probability))


def _build_classifier(
    family: str,
    config: CandidateConfig,
    *,
    workers: int,
    seed: int,
) -> object:
    parameters = dict(config.parameters)
    if family == "gaussian_naive_bayes":
        return GaussianNB(**parameters)
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=180,
            criterion="log_loss",
            max_features="sqrt",
            class_weight=None,
            n_jobs=workers,
            random_state=seed,
            **parameters,
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=240,
            l2_regularization=1.0,
            min_samples_leaf=20,
            random_state=seed,
            **parameters,
        )
    if family == "neural_net":
        return MLPClassifier(
            activation="relu",
            solver="adam",
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
            **parameters,
        )
    if family == "xgboost":
        if importlib.util.find_spec("xgboost") is None:
            raise ModuleNotFoundError("xgboost is not installed")
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(
            n_estimators=300,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_lambda=3.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=workers,
            random_state=seed,
            **parameters,
        )
    raise ValueError(f"unknown classifier family: {family}")


def _needs_scaling(family: str) -> bool:
    return family in {"gaussian_naive_bayes", "neural_net"}


def _fit_classifier(
    family: str,
    config: CandidateConfig,
    training: pd.DataFrame,
    features: Sequence[str],
    *,
    workers: int,
    seed: int,
) -> FittedClassifier:
    imputer = SimpleImputer(
        strategy="constant", fill_value=0.0, keep_empty_features=True
    )
    values = imputer.fit_transform(training[list(features)])
    scaler: StandardScaler | None = None
    if _needs_scaling(family):
        scaler = StandardScaler(with_mean=False)
        values = scaler.fit_transform(values)
    classifier = _build_classifier(family, config, workers=workers, seed=seed)
    classifier.fit(values, training["target"].to_numpy(dtype=int))
    return FittedClassifier(imputer, scaler, classifier)


def _class_one_probability(classifier: object, values: np.ndarray) -> np.ndarray:
    probability = np.asarray(classifier.predict_proba(values), dtype=float)
    classes = np.asarray(classifier.classes_, dtype=int)
    positions = np.flatnonzero(classes == 1)
    if len(positions) != 1:
        raise ValueError("classifier does not expose exactly one positive class")
    return probability[:, int(positions[0])]


def _symmetrized_probability(
    fitted: FittedClassifier,
    frame: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    values = fitted.imputer.transform(frame[list(features)])
    if fitted.scaler is not None:
        values = fitted.scaler.transform(values)
    direct = _class_one_probability(fitted.classifier, values)
    swapped = _class_one_probability(fitted.classifier, -values)
    probability = 0.5 * (direct + 1.0 - swapped)
    if not np.isfinite(probability).all():
        raise RuntimeError("symmetrized model produced non-finite probabilities")
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def _chronological_inner_split(
    training: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    years = sorted(training["date"].dt.year.unique())
    if len(years) < 2:
        raise ValueError("model-family tuning needs at least two training years")
    validation_year = int(years[-1])
    fit = training.loc[training["date"].dt.year < validation_year].copy()
    validation = training.loc[training["date"].dt.year == validation_year].copy()
    if len(fit) < 500 or len(validation) < 100:
        raise ValueError(
            "model-family tuning needs at least 500 earlier and 100 validation fights"
        )
    return fit, validation, {
        "fit_through": fit["date"].max().strftime("%Y-%m-%d"),
        "validation_year": validation_year,
        "fit_fights": len(fit),
        "validation_fights": len(validation),
    }


def _evaluate_classifier_fold(
    family: str,
    training: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    *,
    workers: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    inner_fit, validation, inner_contract = _chronological_inner_split(training)
    scores: dict[str, float] = {}
    for index, config in enumerate(CONFIGS[family]):
        fitted = _fit_classifier(
            family,
            config,
            inner_fit,
            features,
            workers=workers,
            seed=seed + index,
        )
        probability = _symmetrized_probability(fitted, validation, features)
        scores[config.key] = float(
            log_loss(validation["target"], probability, labels=[0, 1])
        )
    selected = min(CONFIGS[family], key=lambda item: scores[item.key])
    calibration_fit = _fit_classifier(
        family,
        selected,
        inner_fit,
        features,
        workers=workers,
        seed=seed + 100,
    )
    validation_probability = _symmetrized_probability(
        calibration_fit, validation, features
    )
    slope = _fit_calibration_slope(
        validation["target"].to_numpy(dtype=int), validation_probability
    )
    final = _fit_classifier(
        family,
        selected,
        training,
        features,
        workers=workers,
        seed=seed + 200,
    )
    raw_test_probability = _symmetrized_probability(final, test, features)
    return _calibrate(raw_test_probability, slope), {
        **inner_contract,
        "candidate_validation_log_loss": scores,
        "selected_config": selected.key,
        "selected_parameters": selected.parameters,
        "calibration_slope": slope,
        "symmetry": "average direct prediction with complement of swapped prediction",
    }


def _evaluate_bayesian_fold(
    training: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    *,
    config: HierarchicalBayesConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    inner_fit, validation, inner_contract = _chronological_inner_split(training)
    inner = hierarchical_bayes_predict(
        inner_fit, validation, features, config=config
    )
    slope = _fit_calibration_slope(
        validation["target"].to_numpy(dtype=int), inner.probability
    )
    final = hierarchical_bayes_predict(training, test, features, config=config)
    return (
        _calibrate(final.probability, slope),
        _calibrate(final.lower_probability, slope),
        _calibrate(final.upper_probability, slope),
        {
            **inner_contract,
            "selected_config": "predeclared_hierarchical_probit",
            "calibration_slope": slope,
            "inner_sampler_diagnostics": inner.diagnostics,
            "final_sampler_diagnostics": final.diagnostics,
            "symmetry": "probit of feature and fighter-ability differences",
        },
    )


def _lineage(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "event_id",
        "fight_id",
        "fighter_id",
        "opponent_id",
        "fighter",
        "opponent",
        "target",
    ]
    return frame[columns].copy()


def _align_reference(
    detail: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    if reference["fight_id"].astype(str).duplicated().any():
        raise ValueError("production reference contains duplicate fight IDs")
    right = reference[["fight_id", "fighter_id", "opponent_id", "model_probability"]].copy()
    merged = detail.merge(
        right,
        on="fight_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    if merged["model_probability"].isna().any():
        raise ValueError("production reference is missing evaluated fights")
    if not (
        merged["fighter_id"].astype(str).eq(merged["fighter_id_reference"].astype(str))
        & merged["opponent_id"].astype(str).eq(
            merged["opponent_id_reference"].astype(str)
        )
    ).all():
        raise ValueError("production reference fighter orientation differs")
    return merged.drop(columns=["fighter_id_reference", "opponent_id_reference"]).rename(
        columns={"model_probability": "current_logistic_probability"}
    )


def _metric_mapping(frame: pd.DataFrame, column: str) -> dict[str, object]:
    return _metrics(
        frame["target"].to_numpy(dtype=int),
        frame[column].to_numpy(dtype=float),
    )


def build_comparison(
    *,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTERS,
    model_artifact_path: Path = DEFAULT_MODEL_ARTIFACT,
    years: Sequence[int] = DEFAULT_YEARS,
    families: Sequence[str] = DEFAULT_FAMILIES,
    workers: int = 4,
    max_runtime_minutes: float = 55.0,
    bayes_burn_in: int = 120,
    bayes_draws: int = 120,
    bayes_chains: int = 2,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not 0.0 < max_runtime_minutes <= MAX_RUNTIME_MINUTES:
        raise ValueError("maximum runtime must be positive and no more than 60 minutes")
    unknown = set(families) - set(DEFAULT_FAMILIES)
    if unknown:
        raise ValueError(f"unknown model families: {sorted(unknown)}")
    if workers < 1:
        raise ValueError("workers must be positive")
    started = time.monotonic()
    point = pd.read_csv(point_in_time_path, low_memory=False)
    point["date"] = pd.to_datetime(point["date"], errors="raise")
    raw = pd.read_csv(raw_fights_path, low_memory=False)
    fighters = pd.read_csv(fighter_stats_path, low_memory=False)
    artifact = json.loads(model_artifact_path.read_text(encoding="utf-8"))
    features = tuple(str(value) for value in artifact["feature_columns"])
    missing_features = set(features) - set(point.columns)
    if missing_features:
        raise ValueError(f"point-in-time data is missing features: {sorted(missing_features)}")
    builder = PointInTimeDatasetBuilder(raw, fighters)
    if features != tuple(builder.feature_columns):
        raise ValueError("production artifact, builder, and point-in-time features disagree")
    selected_years = tuple(sorted(set(int(year) for year in years)))
    reference = TemporalFightPredictor(point, builder).walk_forward_predictions(
        selected_years
    )

    test_rows = point.loc[point["date"].dt.year.isin(selected_years)].copy()
    detail = _align_reference(_lineage(test_rows), reference)
    family_folds: dict[str, dict[str, object]] = {}
    skipped: dict[str, str] = {}
    bayes_config = HierarchicalBayesConfig(
        burn_in=bayes_burn_in,
        posterior_draws=bayes_draws,
        chains=bayes_chains,
    )

    for family_index, family in enumerate(families):
        if family == "xgboost" and importlib.util.find_spec("xgboost") is None:
            skipped[family] = (
                "xgboost is not installed; the no-cost sklearn histogram-gradient-"
                "boosting comparison still runs"
            )
            continue
        fold_reports: dict[str, object] = {}
        probability_by_fight: dict[str, float] = {}
        lower_by_fight: dict[str, float] = {}
        upper_by_fight: dict[str, float] = {}
        for year_index, year in enumerate(selected_years):
            if time.monotonic() - started > max_runtime_minutes * 60.0:
                raise TimeoutError("model-family comparison reached its runtime limit")
            test_start = pd.Timestamp(year=year, month=1, day=1)
            train_start = test_start - pd.DateOffset(years=10)
            training = point.loc[
                (point["date"] >= train_start) & (point["date"] < test_start)
            ].copy()
            test = point.loc[point["date"].dt.year.eq(year)].copy()
            if len(training) < 500 or test.empty:
                continue
            if family == "hierarchical_bayes":
                probability, lower, upper, fold = _evaluate_bayesian_fold(
                    training, test, features, config=bayes_config
                )
                lower_by_fight.update(
                    zip(test["fight_id"].astype(str), lower.astype(float))
                )
                upper_by_fight.update(
                    zip(test["fight_id"].astype(str), upper.astype(float))
                )
            else:
                probability, fold = _evaluate_classifier_fold(
                    family,
                    training,
                    test,
                    features,
                    workers=workers,
                    seed=48 + family_index * 10_000 + year_index * 1_000,
                )
            probability_by_fight.update(
                zip(test["fight_id"].astype(str), probability.astype(float))
            )
            fold_reports[str(year)] = {
                "training_start": training["date"].min().strftime("%Y-%m-%d"),
                "training_through": training["date"].max().strftime("%Y-%m-%d"),
                "training_fights": len(training),
                "test_fights": len(test),
                **fold,
            }
        column = f"{family}_probability"
        detail[column] = detail["fight_id"].astype(str).map(probability_by_fight)
        if detail[column].isna().any():
            raise ValueError(f"{family} did not predict every reference fight")
        if family == "hierarchical_bayes":
            detail["hierarchical_bayes_lower_probability"] = (
                detail["fight_id"].astype(str).map(lower_by_fight)
            )
            detail["hierarchical_bayes_upper_probability"] = (
                detail["fight_id"].astype(str).map(upper_by_fight)
            )
        family_folds[family] = fold_reports

    probability_columns = {
        "current_logistic": "current_logistic_probability",
        **{
            family: f"{family}_probability"
            for family in families
            if family not in skipped
        },
    }
    metrics = {
        name: _metric_mapping(detail, column)
        for name, column in probability_columns.items()
    }
    per_year = {
        str(year): {
            name: _metric_mapping(rows, column)
            for name, column in probability_columns.items()
        }
        for year, rows in detail.groupby(
            pd.to_datetime(detail["date"], errors="raise").dt.year, sort=True
        )
    }
    intervals = {
        f"{name}_minus_current_logistic": event_block_difference_interval(
            detail, column, "current_logistic_probability"
        )
        for name, column in probability_columns.items()
        if name != "current_logistic"
    }
    fixed_blend_metrics: dict[str, object] = {}
    fixed_blend_intervals: dict[str, object] = {}
    logistic_logit = _logit(detail["current_logistic_probability"].to_numpy(dtype=float))
    for name, column in probability_columns.items():
        if name == "current_logistic":
            continue
        blend_column = f"current_logistic_{name}_equal_blend_probability"
        detail[blend_column] = _sigmoid(
            0.5 * (logistic_logit + _logit(detail[column].to_numpy(dtype=float)))
        )
        fixed_blend_metrics[name] = _metric_mapping(detail, blend_column)
        fixed_blend_intervals[name] = event_block_difference_interval(
            detail, blend_column, "current_logistic_probability"
        )
    ranked = sorted(metrics, key=lambda name: float(metrics[name]["log_loss"]))
    best = ranked[0]
    if best == "current_logistic":
        conclusion = (
            "None of the tested model families beat the current logistic model on "
            "the combined untouched yearly tests."
        )
    else:
        interval = intervals[f"{best}_minus_current_logistic"]
        if interval["ci_95_upper"] < 0.0:
            conclusion = (
                f"{best} beat the current logistic model and its whole-event 95% "
                "range stayed below zero. Freeze it for an independent future test."
            )
        else:
            conclusion = (
                f"{best} had the best point estimate, but its uncertainty range "
                "still includes no improvement over logistic regression."
            )
    elapsed = time.monotonic() - started
    logical_detail = detail.copy()
    logical_detail["date"] = pd.to_datetime(
        logical_detail["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "production_action": "none",
        "plain_language_method": (
            "Every model sees the same pre-fight 82 variables and the same future "
            "test fights. For each test year, settings and probability calibration "
            "use only earlier years. Trees, Naive Bayes, and neural nets are forced "
            "to give complementary probabilities when fighter sides are swapped."
        ),
        "years": list(selected_years),
        "sample": {
            "fights": len(detail),
            "events": int(detail["event_id"].nunique()),
            "first_date": detail["date"].min().strftime("%Y-%m-%d"),
            "last_date": detail["date"].max().strftime("%Y-%m-%d"),
            "features": len(features),
        },
        "model_meanings": {
            "current_logistic": "the existing regularized production algorithm",
            "gaussian_naive_bayes": "independent bell-shaped evidence per variable",
            "random_forest": "an average of many nonlinear decision trees",
            "hist_gradient_boosting": "sequential boosted trees, similar in purpose to XGBoost",
            "neural_net": "a small feed-forward network with early stopping",
            "hierarchical_bayes": (
                "a fully Bayesian probit model with coefficient priors, partially "
                "pooled fighter abilities, and a sampled population ability variance"
            ),
            "xgboost": "external gradient-boosted trees, run only when installed",
        },
        "fully_bayesian_contract": {
            "sampler": "two-chain Albert-Chib Gibbs sampler",
            "not_the_existing_laplace_model": True,
            "posterior_components_sampled": [
                "all feature coefficients",
                "every observed fighter ability",
                "population fighter-ability variance",
                "latent fight performance",
            ],
            "config": {
                "burn_in_per_chain": bayes_config.burn_in,
                "posterior_draws_per_chain": bayes_config.posterior_draws,
                "chains": bayes_config.chains,
                "coefficient_prior_scale": bayes_config.coefficient_prior_scale,
                "ability_variance_shape": bayes_config.ability_variance_shape,
                "ability_variance_scale": bayes_config.ability_variance_scale,
            },
        },
        "skipped": skipped,
        "metrics": metrics,
        "per_year": per_year,
        "paired_log_loss_intervals": intervals,
        "fixed_equal_logit_blends_with_current_logistic": {
            "purpose": (
                "check whether a model adds useful information even when it is "
                "worse by itself; weights are fixed at 50/50 and not tuned"
            ),
            "metrics": fixed_blend_metrics,
            "paired_log_loss_intervals": fixed_blend_intervals,
        },
        "ranked_by_log_loss": ranked,
        "plain_language_conclusion": conclusion,
        "folds": family_folds,
        "important_limits": [
            "this development period has already influenced research choices and cannot be final promotion evidence",
            "a single fixed feature set isolates model-family differences but does not find each family's ideal variables",
            "the Bayesian sampler needs stronger convergence checks and prior sensitivity tests before promotion",
            "sportsbook market probabilities remain a stronger practical reference where timestamp-aligned history exists",
        ],
        "source_sha256": {
            "point_in_time": sha256(point_in_time_path.read_bytes()).hexdigest(),
            "raw_fights": sha256(raw_fights_path.read_bytes()).hexdigest(),
            "fighter_stats": sha256(fighter_stats_path.read_bytes()).hexdigest(),
            "model_artifact": sha256(model_artifact_path.read_bytes()).hexdigest(),
            "detail_logical": canonical_hash(logical_detail.to_dict("records")),
        },
        "elapsed_seconds": elapsed,
    }
    return report, detail


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-in-time", type=Path, default=DEFAULT_POINT_IN_TIME)
    parser.add_argument("--raw-fights", type=Path, default=DEFAULT_RAW_FIGHTS)
    parser.add_argument("--fighter-stats", type=Path, default=DEFAULT_FIGHTERS)
    parser.add_argument("--model-artifact", type=Path, default=DEFAULT_MODEL_ARTIFACT)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument(
        "--families", nargs="+", choices=DEFAULT_FAMILIES, default=DEFAULT_FAMILIES
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-runtime-minutes", type=float, default=55.0)
    parser.add_argument("--bayes-burn-in", type=int, default=120)
    parser.add_argument("--bayes-draws", type=int, default=120)
    parser.add_argument("--bayes-chains", type=int, default=2)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report, detail = build_comparison(
        point_in_time_path=arguments.point_in_time,
        raw_fights_path=arguments.raw_fights,
        fighter_stats_path=arguments.fighter_stats,
        model_artifact_path=arguments.model_artifact,
        years=arguments.years,
        families=arguments.families,
        workers=arguments.workers,
        max_runtime_minutes=arguments.max_runtime_minutes,
        bayes_burn_in=arguments.bayes_burn_in,
        bayes_draws=arguments.bayes_draws,
        bayes_chains=arguments.bayes_chains,
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
    print(
        f"Compared {len(report['metrics'])} models on "
        f"{report['sample']['fights']} identical future fights."
    )
    for name in report["ranked_by_log_loss"]:
        metric = report["metrics"][name]
        print(
            f"{name}: log loss={metric['log_loss']:.5f}, "
            f"accuracy={metric['accuracy']:.3%}, Brier={metric['brier']:.5f}"
        )
    for name, reason in report["skipped"].items():
        print(f"{name}: skipped ({reason})")
    print(report["plain_language_conclusion"])
    if not arguments.dry_run:
        print(f"Report: {arguments.report}")
        print(f"Fight detail: {arguments.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
