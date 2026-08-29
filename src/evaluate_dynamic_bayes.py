"""Select and evaluate improved fully Bayesian UFC winner models.

Model structure and any logistic/Bayesian blend weight are selected on
2019--2022.  The selected design is then refit using earlier fights and scored
on 2023--2026.  This is research-only and never changes production forecasts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from evaluate_model_families import (
    _align_reference,
    _atomic_write,
    _calibrate,
    _chronological_inner_split,
    _fit_calibration_slope,
    _lineage,
    _logit,
    _metric_mapping,
    _sigmoid,
)
from evaluate_style_matchup_challenger import event_block_difference_interval
from fight_predictor import PointInTimeDatasetBuilder, TemporalFightPredictor
from fight_predictor.dynamic_bayes import (
    DynamicBayesConfig,
    dynamic_bayes_predict,
    without_elo_features,
)
from fight_predictor.hierarchical_bayes import (
    CoefficientBayesConfig,
    HierarchicalBayesConfig,
    HierarchicalBayesPrediction,
    coefficient_bayes_predict,
    hierarchical_bayes_predict,
)
from market_tracker._common import canonical_hash


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "content/data"
DEFAULT_POINT_IN_TIME = DATA / "processed/ufc_fights_point_in_time.csv"
DEFAULT_RAW_FIGHTS = DATA / "processed/ufc_fights_reported_doubled.csv"
DEFAULT_FIGHTERS = DATA / "processed/fighter_stats.csv"
DEFAULT_MODEL_ARTIFACT = DATA / "external/winner_model.json"
DEFAULT_REPORT = DATA / "model_research/dynamic_bayes_comparison.json"
DEFAULT_DETAIL = DATA / "model_research/dynamic_bayes_comparison.csv"
DEFAULT_DEVELOPMENT_DETAIL = (
    DATA / "model_research/dynamic_bayes_development.csv"
)
DEFAULT_DEVELOPMENT_YEARS = (2019, 2020, 2021, 2022)
DEFAULT_EVALUATION_YEARS = (2023, 2024, 2025, 2026)
EXPERIMENT_VERSION = "dynamic-fully-bayesian-winner-v1"
REPORT_SCHEMA_VERSION = 1
MAX_RUNTIME_MINUTES = 60.0


@dataclass(frozen=True)
class BayesianVariant:
    key: str
    dynamic: bool
    include_elo: bool
    grouped_priors: bool = False
    coefficient_only: bool = False


VARIANTS = (
    BayesianVariant("static_all_features", dynamic=False, include_elo=True),
    BayesianVariant("static_without_elo", dynamic=False, include_elo=False),
    BayesianVariant(
        "static_all_features_grouped_priors",
        dynamic=False,
        include_elo=True,
        grouped_priors=True,
    ),
    BayesianVariant(
        "static_without_elo_grouped_priors",
        dynamic=False,
        include_elo=False,
        grouped_priors=True,
    ),
    BayesianVariant("dynamic_all_features", dynamic=True, include_elo=True),
    BayesianVariant("dynamic_without_elo", dynamic=True, include_elo=False),
    BayesianVariant(
        "dynamic_without_elo_grouped_priors",
        dynamic=True,
        include_elo=False,
        grouped_priors=True,
    ),
    BayesianVariant(
        "coefficient_only_all_features",
        dynamic=False,
        include_elo=True,
        coefficient_only=True,
    ),
    BayesianVariant(
        "coefficient_only_all_features_grouped_priors",
        dynamic=False,
        include_elo=True,
        grouped_priors=True,
        coefficient_only=True,
    ),
)


def _features_for(
    variant: BayesianVariant, features: Sequence[str]
) -> tuple[str, ...]:
    selected = tuple(features)
    return selected if variant.include_elo else without_elo_features(selected)


def _predict(
    variant: BayesianVariant,
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    features: Sequence[str],
    *,
    burn_in: int,
    draws: int,
    chains: int,
    seed: int,
) -> HierarchicalBayesPrediction:
    selected_features = _features_for(variant, features)
    if variant.coefficient_only:
        config = CoefficientBayesConfig(
            burn_in=burn_in,
            posterior_draws=draws,
            chains=chains,
            grouped_coefficient_priors=variant.grouped_priors,
            seed=seed,
        )
        return coefficient_bayes_predict(
            training, prediction, selected_features, config=config
        )
    if variant.dynamic:
        config = DynamicBayesConfig(
            burn_in=burn_in,
            posterior_draws=draws,
            chains=chains,
            grouped_coefficient_priors=variant.grouped_priors,
            seed=seed,
        )
        return dynamic_bayes_predict(
            training, prediction, selected_features, config=config
        )
    config = HierarchicalBayesConfig(
        burn_in=burn_in,
        posterior_draws=draws,
        chains=chains,
        grouped_coefficient_priors=variant.grouped_priors,
        seed=seed,
    )
    return hierarchical_bayes_predict(
        training, prediction, selected_features, config=config
    )


def _training_and_test(
    point: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_start = pd.Timestamp(year=int(year), month=1, day=1)
    train_start = test_start - pd.DateOffset(years=10)
    training = point.loc[
        (point["date"] >= train_start) & (point["date"] < test_start)
    ].copy()
    test = point.loc[point["date"].dt.year.eq(int(year))].copy()
    if len(training) < 500 or test.empty:
        raise ValueError(f"year {year} does not have enough training and test fights")
    return training, test


def _fit_blend_weight(
    target: np.ndarray,
    logistic_probability: np.ndarray,
    bayes_probability: np.ndarray,
) -> float:
    logistic_logit = _logit(logistic_probability)
    bayes_logit = _logit(bayes_probability)

    def objective(weight: float) -> float:
        probability = _sigmoid(
            (1.0 - float(weight)) * logistic_logit
            + float(weight) * bayes_logit
        )
        bounded = np.clip(probability, 1e-12, 1.0 - 1e-12)
        return float(
            -np.mean(target * np.log(bounded) + (1 - target) * np.log1p(-bounded))
        )

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError("Bayesian/logistic blend selection did not converge")
    value = float(result.x)
    if value < 1e-5:
        return 0.0
    if value > 1.0 - 1e-5:
        return 1.0
    return value


def _blend(
    logistic_probability: np.ndarray,
    bayes_probability: np.ndarray,
    bayes_weight: float,
) -> np.ndarray:
    return _sigmoid(
        (1.0 - bayes_weight) * _logit(logistic_probability)
        + bayes_weight * _logit(bayes_probability)
    )


def _runtime_guard(started: float, maximum_minutes: float) -> None:
    if time.monotonic() - started > maximum_minutes * 60.0:
        raise TimeoutError("dynamic Bayesian experiment reached its runtime limit")


def build_comparison(
    *,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTERS,
    model_artifact_path: Path = DEFAULT_MODEL_ARTIFACT,
    development_years: Sequence[int] = DEFAULT_DEVELOPMENT_YEARS,
    evaluation_years: Sequence[int] = DEFAULT_EVALUATION_YEARS,
    selection_burn_in: int = 80,
    selection_draws: int = 80,
    final_burn_in: int = 300,
    final_draws: int = 300,
    chains: int = 2,
    max_runtime_minutes: float = 55.0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if not 0.0 < max_runtime_minutes <= MAX_RUNTIME_MINUTES:
        raise ValueError("maximum runtime must be positive and no more than 60 minutes")
    if min(selection_burn_in, selection_draws, final_burn_in, final_draws, chains) < 1:
        raise ValueError("sampler settings must be positive")
    started = time.monotonic()
    point = pd.read_csv(point_in_time_path, low_memory=False)
    point["date"] = pd.to_datetime(point["date"], errors="raise")
    raw = pd.read_csv(raw_fights_path, low_memory=False)
    fighters = pd.read_csv(fighter_stats_path, low_memory=False)
    artifact = json.loads(model_artifact_path.read_text(encoding="utf-8"))
    features = tuple(str(value) for value in artifact["feature_columns"])
    builder = PointInTimeDatasetBuilder(raw, fighters)
    if features != tuple(builder.feature_columns):
        raise ValueError(
            "production artifact, builder, and point-in-time features disagree"
        )

    development_years = tuple(sorted(set(int(value) for value in development_years)))
    evaluation_years = tuple(sorted(set(int(value) for value in evaluation_years)))
    if set(development_years) & set(evaluation_years):
        raise ValueError("development and evaluation years must not overlap")

    # The current logistic forecasts are generated with the repository's exact
    # chronological procedure. They are needed to select a blend weight without
    # consulting the later evaluation fights.
    predictor = TemporalFightPredictor(point, builder)
    development_logistic = predictor.walk_forward_predictions(development_years)
    development_rows = point.loc[
        point["date"].dt.year.isin(development_years)
    ].copy()
    development = _align_reference(
        _lineage(development_rows), development_logistic
    )

    development_folds: dict[str, dict[str, object]] = {}
    for variant_index, variant in enumerate(VARIANTS):
        probability_by_fight: dict[str, float] = {}
        diagnostics: dict[str, object] = {}
        for year_index, year in enumerate(development_years):
            _runtime_guard(started, max_runtime_minutes)
            training, test = _training_and_test(point, year)
            result = _predict(
                variant,
                training,
                test,
                features,
                burn_in=selection_burn_in,
                draws=selection_draws,
                chains=chains,
                seed=20_190_000 + variant_index * 10_000 + year_index * 1_000,
            )
            probability_by_fight.update(
                zip(test["fight_id"].astype(str), result.probability.astype(float))
            )
            diagnostics[str(year)] = {
                "training_fights": len(training),
                "test_fights": len(test),
                "training_through": training["date"].max().strftime("%Y-%m-%d"),
                "sampler": result.diagnostics,
            }
        raw_column = f"{variant.key}_raw_probability"
        calibrated_column = f"{variant.key}_probability"
        development[raw_column] = development["fight_id"].astype(str).map(
            probability_by_fight
        )
        if development[raw_column].isna().any():
            raise ValueError(f"{variant.key} missed a development fight")
        slope = _fit_calibration_slope(
            development["target"].to_numpy(dtype=int),
            development[raw_column].to_numpy(dtype=float),
        )
        development[calibrated_column] = _calibrate(
            development[raw_column].to_numpy(dtype=float), slope
        )
        development_folds[variant.key] = {
            "variant": asdict(variant),
            "feature_count": len(_features_for(variant, features)),
            "development_calibration_slope": slope,
            "raw_metrics": _metric_mapping(development, raw_column),
            "calibrated_metrics": _metric_mapping(development, calibrated_column),
            "years": diagnostics,
        }

    selected_variant = min(
        VARIANTS,
        key=lambda value: float(
            development_folds[value.key]["calibrated_metrics"]["log_loss"]
        ),
    )
    selected_development_column = f"{selected_variant.key}_probability"
    blend_weight = _fit_blend_weight(
        development["target"].to_numpy(dtype=int),
        development["current_logistic_probability"].to_numpy(dtype=float),
        development[selected_development_column].to_numpy(dtype=float),
    )
    development["selected_logistic_bayes_blend_probability"] = _blend(
        development["current_logistic_probability"].to_numpy(dtype=float),
        development[selected_development_column].to_numpy(dtype=float),
        blend_weight,
    )

    evaluation_reference = predictor.walk_forward_predictions(evaluation_years)
    evaluation_rows = point.loc[
        point["date"].dt.year.isin(evaluation_years)
    ].copy()
    detail = _align_reference(_lineage(evaluation_rows), evaluation_reference)
    probability_by_fight: dict[str, float] = {}
    lower_by_fight: dict[str, float] = {}
    upper_by_fight: dict[str, float] = {}
    evaluation_folds: dict[str, object] = {}
    selected_index = next(
        index for index, value in enumerate(VARIANTS) if value == selected_variant
    )
    for year_index, year in enumerate(evaluation_years):
        _runtime_guard(started, max_runtime_minutes)
        training, test = _training_and_test(point, year)
        inner_fit, validation, inner_contract = _chronological_inner_split(training)
        inner = _predict(
            selected_variant,
            inner_fit,
            validation,
            features,
            burn_in=final_burn_in,
            draws=final_draws,
            chains=chains,
            seed=20_230_000 + selected_index * 10_000 + year_index * 1_000,
        )
        slope = _fit_calibration_slope(
            validation["target"].to_numpy(dtype=int), inner.probability
        )
        final = _predict(
            selected_variant,
            training,
            test,
            features,
            burn_in=final_burn_in,
            draws=final_draws,
            chains=chains,
            seed=20_240_000 + selected_index * 10_000 + year_index * 1_000,
        )
        probability_by_fight.update(
            zip(
                test["fight_id"].astype(str),
                _calibrate(final.probability, slope).astype(float),
            )
        )
        lower_by_fight.update(
            zip(
                test["fight_id"].astype(str),
                _calibrate(final.lower_probability, slope).astype(float),
            )
        )
        upper_by_fight.update(
            zip(
                test["fight_id"].astype(str),
                _calibrate(final.upper_probability, slope).astype(float),
            )
        )
        evaluation_folds[str(year)] = {
            **inner_contract,
            "training_fights": len(training),
            "test_fights": len(test),
            "calibration_slope": slope,
            "inner_sampler": inner.diagnostics,
            "final_sampler": final.diagnostics,
        }

    detail["selected_bayes_probability"] = detail["fight_id"].astype(str).map(
        probability_by_fight
    )
    detail["selected_bayes_lower_probability"] = detail["fight_id"].astype(str).map(
        lower_by_fight
    )
    detail["selected_bayes_upper_probability"] = detail["fight_id"].astype(str).map(
        upper_by_fight
    )
    if detail["selected_bayes_probability"].isna().any():
        raise ValueError("selected Bayesian model missed an evaluation fight")
    detail["selected_logistic_bayes_blend_probability"] = _blend(
        detail["current_logistic_probability"].to_numpy(dtype=float),
        detail["selected_bayes_probability"].to_numpy(dtype=float),
        blend_weight,
    )

    metrics = {
        "current_logistic": _metric_mapping(detail, "current_logistic_probability"),
        "selected_bayes": _metric_mapping(detail, "selected_bayes_probability"),
        "development_selected_blend": _metric_mapping(
            detail, "selected_logistic_bayes_blend_probability"
        ),
    }
    intervals = {
        "selected_bayes_minus_current_logistic": event_block_difference_interval(
            detail, "selected_bayes_probability", "current_logistic_probability"
        ),
        "development_selected_blend_minus_current_logistic": (
            event_block_difference_interval(
                detail,
                "selected_logistic_bayes_blend_probability",
                "current_logistic_probability",
            )
        ),
    }
    development_metrics = {
        "current_logistic": _metric_mapping(
            development, "current_logistic_probability"
        ),
        **{
            variant.key: development_folds[variant.key]["calibrated_metrics"]
            for variant in VARIANTS
        },
        "selected_blend": _metric_mapping(
            development, "selected_logistic_bayes_blend_probability"
        ),
    }
    bayes_difference = (
        float(metrics["selected_bayes"]["log_loss"])
        - float(metrics["current_logistic"]["log_loss"])
    )
    blend_difference = (
        float(metrics["development_selected_blend"]["log_loss"])
        - float(metrics["current_logistic"]["log_loss"])
    )
    if blend_weight == 0.0:
        conclusion = (
            "The earlier development fights assigned no useful weight to the "
            "Bayesian model, so the preselected blend is exactly the logistic model."
        )
    elif blend_difference < 0.0:
        conclusion = (
            "The preselected logistic/Bayesian blend improved probability quality "
            "on the later evaluation fights; it still needs prospective confirmation."
        )
    else:
        conclusion = (
            "The improved Bayesian design did not add useful probability information "
            "to the logistic model on the later evaluation fights."
        )

    logical_detail = detail.copy()
    logical_detail["date"] = pd.to_datetime(logical_detail["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    logical_development = development.copy()
    logical_development["date"] = pd.to_datetime(
        logical_development["date"]
    ).dt.strftime("%Y-%m-%d")
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "production_action": "none",
        "plain_language_method": (
            "Nine Bayesian designs were compared on 2019-2022 only. The best "
            "design and its logistic/Bayesian mixing weight were frozen before "
            "scoring 2023-2026. Every yearly forecast used only earlier fights."
        ),
        "development": {
            "years": list(development_years),
            "fights": len(development),
            "events": int(development["event_id"].nunique()),
            "variants": development_folds,
            "metrics": development_metrics,
            "selected_variant": selected_variant.key,
            "selected_bayes_weight_in_log_odds_blend": blend_weight,
            "selection_sampler": {
                "burn_in_per_chain": selection_burn_in,
                "draws_per_chain": selection_draws,
                "chains": chains,
            },
        },
        "evaluation": {
            "years": list(evaluation_years),
            "fights": len(detail),
            "events": int(detail["event_id"].nunique()),
            "selected_variant": asdict(selected_variant),
            "feature_count": len(_features_for(selected_variant, features)),
            "metrics": metrics,
            "per_year": {
                str(year): {
                    "current_logistic": _metric_mapping(
                        rows, "current_logistic_probability"
                    ),
                    "selected_bayes": _metric_mapping(
                        rows, "selected_bayes_probability"
                    ),
                    "development_selected_blend": _metric_mapping(
                        rows, "selected_logistic_bayes_blend_probability"
                    ),
                }
                for year, rows in detail.groupby(
                    pd.to_datetime(detail["date"]).dt.year, sort=True
                )
            },
            "paired_log_loss_intervals": intervals,
            "bayes_minus_logistic_log_loss": bayes_difference,
            "blend_minus_logistic_log_loss": blend_difference,
            "folds": evaluation_folds,
        },
        "plain_language_conclusion": conclusion,
        "important_limits": [
            (
                "2019-2022 chose the design and blend weight, so its scores are "
                "not an independent result"
            ),
            (
                "2023-2026 was examined in earlier model research and is reused "
                "development evidence, not final promotion proof"
            ),
            (
                "posterior intervals omit uncertainty in the separately estimated "
                "calibration slope"
            ),
            "production forecasts and betting behavior are unchanged",
        ],
        "source_sha256": {
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
        "--development-years", nargs="+", type=int, default=DEFAULT_DEVELOPMENT_YEARS
    )
    parser.add_argument(
        "--evaluation-years", nargs="+", type=int, default=DEFAULT_EVALUATION_YEARS
    )
    parser.add_argument("--selection-burn-in", type=int, default=80)
    parser.add_argument("--selection-draws", type=int, default=80)
    parser.add_argument("--final-burn-in", type=int, default=300)
    parser.add_argument("--final-draws", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
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
        selection_burn_in=arguments.selection_burn_in,
        selection_draws=arguments.selection_draws,
        final_burn_in=arguments.final_burn_in,
        final_draws=arguments.final_draws,
        chains=arguments.chains,
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
        f"Selected {report['development']['selected_variant']} on "
        f"{report['development']['fights']} development fights."
    )
    print(
        "Development-selected Bayesian blend weight: "
        f"{report['development']['selected_bayes_weight_in_log_odds_blend']:.3f}"
    )
    for name, metric in report["evaluation"]["metrics"].items():
        print(
            f"{name}: log loss={metric['log_loss']:.5f}, "
            f"accuracy={metric['accuracy']:.3%}, Brier={metric['brier']:.5f}"
        )
    print(report["plain_language_conclusion"])
    if not arguments.dry_run:
        print(f"Report: {arguments.report}")
        print(f"Evaluation fights: {arguments.detail}")
        print(f"Development fights: {arguments.development_detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
