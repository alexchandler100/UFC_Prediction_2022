"""Evaluate a Bayesian logistic UFC winner model with learned shrinkage.

Hyperprior design and a possible production-logistic blend weight are selected
using 2019--2022 only.  The frozen design is then refit chronologically and
scored on 2023--2026.  The experiment is research-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd

from evaluate_dynamic_bayes import (
    _blend,
    _fit_blend_weight,
    _runtime_guard,
    _training_and_test,
)
from evaluate_model_families import (
    _align_reference,
    _atomic_write,
    _calibrate,
    _chronological_inner_split,
    _fit_calibration_slope,
    _lineage,
    _metric_mapping,
)
from evaluate_style_matchup_challenger import event_block_difference_interval
from fight_predictor import PointInTimeDatasetBuilder, TemporalFightPredictor
from fight_predictor.bayesian_logistic import (
    BayesianLogisticConfig,
    bayesian_logistic_predict,
)
from market_tracker._common import canonical_hash


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "content/data"
DEFAULT_POINT_IN_TIME = DATA / "processed/ufc_fights_point_in_time.csv"
DEFAULT_RAW_FIGHTS = DATA / "processed/ufc_fights_reported_doubled.csv"
DEFAULT_FIGHTERS = DATA / "processed/fighter_stats.csv"
DEFAULT_MODEL_ARTIFACT = DATA / "external/winner_model.json"
DEFAULT_PREVIOUS_DETAIL = DATA / "model_research/dynamic_bayes_comparison.csv"
DEFAULT_REPORT = DATA / "model_research/bayesian_logistic_comparison.json"
DEFAULT_DETAIL = DATA / "model_research/bayesian_logistic_comparison.csv"
DEFAULT_DEVELOPMENT_DETAIL = DATA / "model_research/bayesian_logistic_development.csv"
DEFAULT_DEVELOPMENT_YEARS = (2019, 2020, 2021, 2022)
DEFAULT_EVALUATION_YEARS = (2023, 2024, 2025, 2026)
EXPERIMENT_VERSION = "fully-bayesian-logistic-group-shrinkage-v1"
REPORT_SCHEMA_VERSION = 1
MAX_RUNTIME_MINUTES = 60.0


@dataclass(frozen=True)
class ShrinkageVariant:
    key: str
    grouped: bool
    variance_prior_shape: float
    variance_prior_scale: float


VARIANTS = (
    ShrinkageVariant("global_moderate", False, 3.0, 0.08),
    ShrinkageVariant("grouped_tight", True, 3.0, 0.02),
    ShrinkageVariant("grouped_moderate", True, 3.0, 0.08),
    ShrinkageVariant("grouped_weak", True, 3.0, 0.32),
    ShrinkageVariant("grouped_heavy_tailed", True, 1.5, 0.02),
)


def _predict(
    variant: ShrinkageVariant,
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    features: Sequence[str],
    *,
    burn_in: int,
    draws: int,
    chains: int,
    seed: int,
):
    return bayesian_logistic_predict(
        training,
        prediction,
        features,
        config=BayesianLogisticConfig(
            burn_in=burn_in,
            posterior_draws=draws,
            chains=chains,
            grouped_shrinkage=variant.grouped,
            variance_prior_shape=variant.variance_prior_shape,
            variance_prior_scale=variant.variance_prior_scale,
            seed=seed,
        ),
    )


def _add_previous_bayesian(
    detail: pd.DataFrame, previous_detail_path: Path
) -> pd.DataFrame:
    previous = pd.read_csv(previous_detail_path, low_memory=False)
    required = {
        "fight_id",
        "fighter_id",
        "opponent_id",
        "selected_bayes_probability",
    }
    missing = required - set(previous.columns)
    if missing:
        raise ValueError(f"previous Bayesian detail is missing: {sorted(missing)}")
    if previous["fight_id"].astype(str).duplicated().any():
        raise ValueError("previous Bayesian detail has duplicate fight IDs")
    columns = [
        "fight_id",
        "fighter_id",
        "opponent_id",
        "selected_bayes_probability",
    ]
    merged = detail.merge(
        previous[columns],
        on="fight_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_previous"),
    )
    if merged["selected_bayes_probability"].isna().any():
        raise ValueError("previous Bayesian detail is missing an evaluation fight")
    orientation_matches = (
        merged["fighter_id"].astype(str).eq(
            merged["fighter_id_previous"].astype(str)
        )
        & merged["opponent_id"].astype(str).eq(
            merged["opponent_id_previous"].astype(str)
        )
    )
    if not orientation_matches.all():
        raise ValueError("previous Bayesian detail uses a different orientation")
    return merged.drop(
        columns=["fighter_id_previous", "opponent_id_previous"]
    ).rename(
        columns={
            "selected_bayes_probability": "previous_bayesian_probit_probability"
        }
    )


def build_comparison(
    *,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTERS,
    model_artifact_path: Path = DEFAULT_MODEL_ARTIFACT,
    previous_detail_path: Path = DEFAULT_PREVIOUS_DETAIL,
    development_years: Sequence[int] = DEFAULT_DEVELOPMENT_YEARS,
    evaluation_years: Sequence[int] = DEFAULT_EVALUATION_YEARS,
    selection_burn_in: int = 600,
    selection_draws: int = 600,
    final_burn_in: int = 1_000,
    final_draws: int = 1_000,
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

    predictor = TemporalFightPredictor(point, builder)
    development_reference = predictor.walk_forward_predictions(development_years)
    development_rows = point.loc[
        point["date"].dt.year.isin(development_years)
    ].copy()
    development = _align_reference(
        _lineage(development_rows), development_reference
    )
    variant_reports: dict[str, dict[str, object]] = {}
    for variant_index, variant in enumerate(VARIANTS):
        probability_by_fight: dict[str, float] = {}
        yearly_diagnostics: dict[str, object] = {}
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
                seed=30_190_000 + variant_index * 10_000 + year_index * 1_000,
            )
            probability_by_fight.update(
                zip(test["fight_id"].astype(str), result.probability.astype(float))
            )
            yearly_diagnostics[str(year)] = {
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
        variant_reports[variant.key] = {
            "variant": asdict(variant),
            "development_calibration_slope": slope,
            "raw_metrics": _metric_mapping(development, raw_column),
            "calibrated_metrics": _metric_mapping(development, calibrated_column),
            "years": yearly_diagnostics,
        }

    selected_variant = min(
        VARIANTS,
        key=lambda value: float(
            variant_reports[value.key]["calibrated_metrics"]["log_loss"]
        ),
    )
    selected_development_column = f"{selected_variant.key}_probability"
    blend_weight = _fit_blend_weight(
        development["target"].to_numpy(dtype=int),
        development["current_logistic_probability"].to_numpy(dtype=float),
        development[selected_development_column].to_numpy(dtype=float),
    )
    development["selected_logistic_bayesian_blend_probability"] = _blend(
        development["current_logistic_probability"].to_numpy(dtype=float),
        development[selected_development_column].to_numpy(dtype=float),
        blend_weight,
    )

    evaluation_reference = predictor.walk_forward_predictions(evaluation_years)
    evaluation_rows = point.loc[
        point["date"].dt.year.isin(evaluation_years)
    ].copy()
    detail = _align_reference(_lineage(evaluation_rows), evaluation_reference)
    detail = _add_previous_bayesian(detail, previous_detail_path)
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
            seed=30_230_000 + selected_index * 10_000 + year_index * 1_000,
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
            seed=30_240_000 + selected_index * 10_000 + year_index * 1_000,
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

    detail["bayesian_logistic_probability"] = detail["fight_id"].astype(str).map(
        probability_by_fight
    )
    detail["bayesian_logistic_lower_probability"] = detail["fight_id"].astype(
        str
    ).map(lower_by_fight)
    detail["bayesian_logistic_upper_probability"] = detail["fight_id"].astype(
        str
    ).map(upper_by_fight)
    if detail["bayesian_logistic_probability"].isna().any():
        raise ValueError("Bayesian logistic model missed an evaluation fight")
    detail["selected_logistic_bayesian_blend_probability"] = _blend(
        detail["current_logistic_probability"].to_numpy(dtype=float),
        detail["bayesian_logistic_probability"].to_numpy(dtype=float),
        blend_weight,
    )

    probability_columns = {
        "current_logistic": "current_logistic_probability",
        "previous_bayesian_probit": "previous_bayesian_probit_probability",
        "bayesian_logistic": "bayesian_logistic_probability",
        "development_selected_blend": (
            "selected_logistic_bayesian_blend_probability"
        ),
    }
    metrics = {
        name: _metric_mapping(detail, column)
        for name, column in probability_columns.items()
    }
    intervals = {
        "bayesian_logistic_minus_current_logistic": (
            event_block_difference_interval(
                detail,
                "bayesian_logistic_probability",
                "current_logistic_probability",
            )
        ),
        "blend_minus_current_logistic": event_block_difference_interval(
            detail,
            "selected_logistic_bayesian_blend_probability",
            "current_logistic_probability",
        ),
        "bayesian_logistic_minus_previous_bayesian_probit": (
            event_block_difference_interval(
                detail,
                "bayesian_logistic_probability",
                "previous_bayesian_probit_probability",
            )
        ),
    }
    development_metrics = {
        "current_logistic": _metric_mapping(
            development, "current_logistic_probability"
        ),
        **{
            variant.key: variant_reports[variant.key]["calibrated_metrics"]
            for variant in VARIANTS
        },
        "selected_blend": _metric_mapping(
            development, "selected_logistic_bayesian_blend_probability"
        ),
    }
    new_difference = (
        float(metrics["bayesian_logistic"]["log_loss"])
        - float(metrics["current_logistic"]["log_loss"])
    )
    blend_difference = (
        float(metrics["development_selected_blend"]["log_loss"])
        - float(metrics["current_logistic"]["log_loss"])
    )
    previous_difference = (
        float(metrics["bayesian_logistic"]["log_loss"])
        - float(metrics["previous_bayesian_probit"]["log_loss"])
    )
    new_interval = intervals["bayesian_logistic_minus_current_logistic"]
    blend_interval = intervals["blend_minus_current_logistic"]
    if new_difference < 0.0 and float(new_interval["ci_95_upper"]) < 0.0:
        conclusion = (
            "The Bayesian logistic model improved probability quality on the "
            "later evaluation fights; prospective confirmation is still required."
        )
    elif blend_difference < 0.0 and float(blend_interval["ci_95_upper"]) < 0.0:
        conclusion = (
            "The Bayesian logistic model did not win alone, but its preselected "
            "blend improved the later evaluation fights."
        )
    elif blend_difference < 0.0:
        conclusion = (
            "The preselected blend had a tiny better point estimate, but its "
            "uncertainty range includes no improvement. It should not be promoted."
        )
    else:
        conclusion = (
            "Learned Bayesian group shrinkage did not beat or improve the current "
            "logistic model on the later evaluation fights."
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
            "Five Bayesian logistic prior designs were compared on 2019-2022. "
            "The selected design learns separate shrinkage amounts for six feature "
            "groups. Its design and blend weight were frozen before 2023-2026."
        ),
        "fully_bayesian_contract": {
            "likelihood": "logistic winner likelihood",
            "sampler": (
                "Laplace-preconditioned Hamiltonian Monte Carlo with exact "
                "accept/reject correction plus conditional variance draws"
            ),
            "sampled": [
                "all 82 matchup coefficients",
                "coefficient variance for every enabled feature group",
            ],
            "not_laplace_or_point_estimate": True,
        },
        "development": {
            "years": list(development_years),
            "fights": len(development),
            "events": int(development["event_id"].nunique()),
            "variants": variant_reports,
            "metrics": development_metrics,
            "selected_variant": selected_variant.key,
            "selected_bayesian_weight_in_log_odds_blend": blend_weight,
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
            "metrics": metrics,
            "per_year": {
                str(year): {
                    name: _metric_mapping(rows, column)
                    for name, column in probability_columns.items()
                }
                for year, rows in detail.groupby(
                    pd.to_datetime(detail["date"]).dt.year, sort=True
                )
            },
            "paired_log_loss_intervals": intervals,
            "bayesian_logistic_minus_current_logistic_log_loss": new_difference,
            "blend_minus_current_logistic_log_loss": blend_difference,
            "bayesian_logistic_minus_previous_probit_log_loss": previous_difference,
            "folds": evaluation_folds,
        },
        "plain_language_conclusion": conclusion,
        "important_limits": [
            (
                "2019-2022 chose the hyperprior and blend weight, so those scores "
                "are not independent evidence"
            ),
            (
                "2023-2026 has influenced earlier research and is reused development "
                "evidence rather than final promotion proof"
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
            "previous_bayesian_detail": sha256(
                previous_detail_path.read_bytes()
            ).hexdigest(),
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
        "--previous-detail", type=Path, default=DEFAULT_PREVIOUS_DETAIL
    )
    parser.add_argument(
        "--development-years", nargs="+", type=int, default=DEFAULT_DEVELOPMENT_YEARS
    )
    parser.add_argument(
        "--evaluation-years", nargs="+", type=int, default=DEFAULT_EVALUATION_YEARS
    )
    parser.add_argument("--selection-burn-in", type=int, default=600)
    parser.add_argument("--selection-draws", type=int, default=600)
    parser.add_argument("--final-burn-in", type=int, default=1_000)
    parser.add_argument("--final-draws", type=int, default=1_000)
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
        previous_detail_path=arguments.previous_detail,
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
        f"{report['development']['selected_bayesian_weight_in_log_odds_blend']:.3f}"
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
