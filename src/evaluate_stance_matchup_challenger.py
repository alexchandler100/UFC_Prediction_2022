"""Evaluate a bounded stance group alone and with external MMA history.

This is a paper-only 2x2 development experiment.  Every variant is evaluated
with the same nested expanding-year winner-model procedure, and this command
cannot update the production artifact or enable external-history ingestion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

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
    DEFAULT_AUXILIARY,
    DEFAULT_EXTERNAL_REPORT,
    DEFAULT_FIGHTER_STATS,
    DEFAULT_POINT_IN_TIME,
    DEFAULT_RAW_FIGHTS,
    load_frozen_research_auxiliary,
)
from fight_predictor import (
    PointInTimeDatasetBuilder,
    StanceMatchupDatasetBuilder,
    TemporalFightPredictor,
)
from fight_predictor.stance_matchup import OPEN_STANCE_INTERACTION_KEYS


REPORT_SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "stance-matchup-factorial-v1"
ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "content/data/external_mma/stance_matchup_factorial.json"
DEFAULT_DETAIL = ROOT / "content/data/external_mma/stance_matchup_factorial.csv"

VARIANTS = ("baseline", "external_history", "stance", "external_stance")
CHALLENGERS = VARIANTS[1:]


def _build_variant_points(
    production: pd.DataFrame,
    raw: pd.DataFrame,
    fighters: pd.DataFrame,
    auxiliary: pd.DataFrame,
) -> tuple[dict[str, PointInTimeDatasetBuilder], dict[str, pd.DataFrame]]:
    builders: dict[str, PointInTimeDatasetBuilder] = {
        "baseline": PointInTimeDatasetBuilder(raw, fighters),
        "external_history": PointInTimeDatasetBuilder(
            raw, fighters, auxiliary_fights=auxiliary
        ),
        "stance": StanceMatchupDatasetBuilder(raw, fighters),
        "external_stance": StanceMatchupDatasetBuilder(
            raw, fighters, auxiliary_fights=auxiliary
        ),
    }
    points = {name: builder.build() for name, builder in builders.items()}

    _validate_candidate_baseline_features(
        production, points["baseline"], builders["baseline"].feature_columns
    )
    _validate_candidate_baseline_features(
        production, points["stance"], builders["baseline"].feature_columns
    )
    _validate_candidate_baseline_features(
        points["external_history"],
        points["external_stance"],
        builders["external_history"].feature_columns,
    )
    expected_ids = production["fight_id"].astype(str).tolist()
    for name, point in points.items():
        if point["fight_id"].astype(str).tolist() != expected_ids:
            raise RuntimeError(f"{name} changed the UFC training-label set or order")
        if point["target"].tolist() != production["target"].tolist():
            raise RuntimeError(f"{name} changed a UFC training label")
    return builders, points


def _walk_forward_variants(
    production: pd.DataFrame,
    builders: dict[str, PointInTimeDatasetBuilder],
    points: dict[str, pd.DataFrame],
    years: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    for name in VARIANTS:
        print(f"Evaluating {name} over calendar folds {years}")
        training = production if name == "baseline" else points[name]
        predictions[name] = TemporalFightPredictor(
            training, builders[name]
        ).walk_forward_predictions(years)
    return predictions


def _align_variants(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    aligned = predictions["baseline"].rename(
        columns={"model_probability": "baseline_probability"}
    )
    for name in CHALLENGERS:
        aligned = _align_predictions(
            aligned,
            predictions[name],
            probability_name=f"{name}_probability",
        )
    return aligned


def _metrics_by_variant(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        name: _metric(frame, f"{name}_probability") for name in VARIANTS
    }


def _comparisons(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    pairs = {
        "external_history_minus_baseline": ("external_history", "baseline"),
        "stance_minus_baseline": ("stance", "baseline"),
        "external_stance_minus_baseline": ("external_stance", "baseline"),
        "external_stance_minus_external_history": (
            "external_stance",
            "external_history",
        ),
        "external_stance_minus_stance": ("external_stance", "stance"),
    }
    return {
        label: event_block_difference_interval(
            frame,
            f"{candidate}_probability",
            f"{reference}_probability",
        )
        for label, (candidate, reference) in pairs.items()
    }


def _factorial_summary(metrics: dict[str, dict[str, object]]) -> dict[str, float]:
    losses = {name: float(values["log_loss"]) for name, values in metrics.items()}
    return {
        "external_effect_without_stance": (
            losses["external_history"] - losses["baseline"]
        ),
        "stance_effect_without_external": losses["stance"] - losses["baseline"],
        "external_effect_with_stance": (
            losses["external_stance"] - losses["stance"]
        ),
        "stance_effect_with_external": (
            losses["external_stance"] - losses["external_history"]
        ),
        "interaction": (
            losses["external_stance"]
            - losses["external_history"]
            - losses["stance"]
            + losses["baseline"]
        ),
    }


def _stance_coverage(
    frame: pd.DataFrame,
    builder: StanceMatchupDatasetBuilder,
) -> dict[str, object]:
    fighter = frame["fighter_id"].astype(str).map(builder.stance_for)
    opponent = frame["opponent_id"].astype(str).map(builder.stance_for)
    fighter_known = fighter.ne("unknown")
    opponent_known = opponent.ne("unknown")
    both_known = fighter_known & opponent_known
    open_stance = pd.Series(
        [left != right and {left, right} == {"orthodox", "southpaw"}
         for left, right in zip(fighter, opponent)],
        index=frame.index,
    )
    return {
        "fights": len(frame),
        "both_stances_known": int(both_known.sum()),
        "both_stances_known_fraction": float(both_known.mean()),
        "orthodox_southpaw_fights": int(open_stance.sum()),
        "orthodox_southpaw_fraction": float(open_stance.mean()),
    }


def build_evaluation(
    *,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTER_STATS,
    auxiliary_path: Path = DEFAULT_AUXILIARY,
    external_report_path: Path = DEFAULT_EXTERNAL_REPORT,
    market_report_path: Path = DEFAULT_MARKET_REPORT,
    market_detail_path: Path = DEFAULT_MARKET_DETAIL,
) -> tuple[dict[str, object], pd.DataFrame]:
    for path in (
        point_in_time_path,
        raw_fights_path,
        fighter_stats_path,
        market_report_path,
        market_detail_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    production = pd.read_csv(point_in_time_path, low_memory=False)
    raw = pd.read_csv(raw_fights_path, low_memory=False)
    fighters = pd.read_csv(fighter_stats_path, low_memory=False)
    auxiliary, external_report = load_frozen_research_auxiliary(
        auxiliary_path, external_report_path
    )
    market_report = json.loads(market_report_path.read_text(encoding="utf-8"))
    stored_market = pd.read_csv(market_detail_path, low_memory=False)

    builders, points = _build_variant_points(production, raw, fighters, auxiliary)
    first_year = int(stored_market["evaluation_year"].min())
    final_year = int(pd.to_datetime(production["date"], errors="raise").dt.year.max())
    years = tuple(range(first_year, final_year + 1))
    predictions = _walk_forward_variants(production, builders, points, years)
    all_predictions = _align_variants(predictions)

    production_years = tuple(
        int(year) for year in sorted(all_predictions["evaluation_year"].unique())[-4:]
    )
    production_horizon = all_predictions[
        all_predictions["evaluation_year"].isin(production_years)
    ].copy()

    market_paired = stored_market.rename(
        columns={"model_probability": "stored_baseline_probability"}
    )
    market_paired = _align_predictions(
        market_paired,
        predictions["baseline"],
        probability_name="baseline_probability",
    )
    baseline_probability_drift = np.abs(
        market_paired["baseline_probability"].to_numpy(dtype=float)
        - market_paired["stored_baseline_probability"].to_numpy(dtype=float)
    )
    for name in CHALLENGERS:
        market_paired = _align_predictions(
            market_paired,
            predictions[name],
            probability_name=f"{name}_probability",
        )

    production_metrics = _metrics_by_variant(production_horizon)
    extended_metrics = _metrics_by_variant(all_predictions)
    market_metrics = {
        "market": _metric_mapping(market_paired, "market_probability"),
        **{
            name: _metric_mapping(market_paired, f"{name}_probability")
            for name in VARIANTS
        },
    }
    ranked = sorted(
        VARIANTS,
        key=lambda name: float(production_metrics[name]["log_loss"]),
    )
    stance_helps_baseline = (
        production_metrics["stance"]["log_loss"]
        < production_metrics["baseline"]["log_loss"]
    )
    stance_helps_external = (
        production_metrics["external_stance"]["log_loss"]
        < production_metrics["external_history"]["log_loss"]
    )
    retain_stance = bool(stance_helps_baseline and stance_helps_external)

    report: dict[str, object] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "promotable": False,
        "primary_metric": "production_walk_forward_log_loss",
        "non_promotable_flags": [
            "stance_group_is_development_evidence",
            "current_profile_stance_is_not_historically_timestamped",
            "current_data_period_informed_candidate_selection",
            "external_source_is_a_stale_bootstrap_through_2021_08_11",
            "market_history_missing_2024",
            "legacy_market_timestamps_are_not_verified_close_times",
        ],
        "feature_contract": {
            "baseline_feature_count": len(builders["baseline"].feature_columns),
            "stance_feature_count": len(builders["stance"].feature_columns),
            "additional_feature_count": (
                len(builders["stance"].feature_columns)
                - len(builders["baseline"].feature_columns)
            ),
            "additional_features": list(
                builders["stance"].feature_columns[
                    len(builders["baseline"].feature_columns):
                ]
            ),
            "standard_categories": ["orthodox", "southpaw", "switch"],
            "nonstandard_categories_treated_as_unknown": True,
            "open_stance_definition": (
                "exactly one orthodox and one southpaw profile; switch and "
                "unknown never inferred open"
            ),
            "open_stance_interaction_inputs": list(OPEN_STANCE_INTERACTION_KEYS),
            "every_feature_antisymmetric": True,
            "external_history_adds_feature_columns": False,
            "training_and_calibration": (
                "identical nested expanding-year L2 logistic procedure"
            ),
            "ufc_training_labels_identical": True,
        },
        "coverage": {
            "all_labeled_fights": _stance_coverage(
                production, builders["stance"]
            ),
            "production_horizon": _stance_coverage(
                production_horizon, builders["stance"]
            ),
        },
        "external_history_contract": {
            "auxiliary_sha256": external_report["auxiliary_sha256"],
            "physical_bouts": int(auxiliary["fight_url"].nunique()),
            "emit_training_target": False,
            "detailed_statistics_fabricated": False,
            "production_policy_enabled_during_test": False,
        },
        "sample": {
            "extended_years": list(years),
            "extended_fights": len(all_predictions),
            "extended_events": int(all_predictions["event_id"].nunique()),
            "production_years": list(production_years),
            "production_fights": len(production_horizon),
            "production_events": int(production_horizon["event_id"].nunique()),
            "market_paired_fights": len(market_paired),
            "market_paired_events": int(market_paired["event_id"].nunique()),
        },
        "stored_baseline_drift_audit": {
            "changed_fights": int(
                np.count_nonzero(baseline_probability_drift > 1e-12)
            ),
            "maximum_absolute_probability_difference": float(
                baseline_probability_drift.max()
            ),
            "mean_absolute_probability_difference": float(
                baseline_probability_drift.mean()
            ),
            "current_recomputed_baseline_used_for_comparisons": True,
        },
        "production_walk_forward": {
            "metrics": production_metrics,
            "paired_log_loss_intervals": _comparisons(production_horizon),
            "factorial_log_loss_effects": _factorial_summary(production_metrics),
        },
        "extended_walk_forward": {
            "metrics": extended_metrics,
            "paired_log_loss_intervals": _comparisons(all_predictions),
            "factorial_log_loss_effects": _factorial_summary(extended_metrics),
        },
        "market_paired": {
            "metrics": market_metrics,
            "challenger_log_loss_intervals": _comparisons(market_paired),
            "external_stance_minus_market_interval": (
                event_block_difference_interval(
                    market_paired,
                    "external_stance_probability",
                    "market_probability",
                )
            ),
            "factorial_log_loss_effects": _factorial_summary(
                {name: market_metrics[name] for name in VARIANTS}
            ),
        },
        "decision": {
            "ranked_by_production_walk_forward_log_loss": ranked,
            "best_variant": ranked[0],
            "stance_improved_baseline_point_log_loss": stance_helps_baseline,
            "stance_improved_external_point_log_loss": stance_helps_external,
            "stance_group_retained_for_leading_challenger": retain_stance,
            "recommendation": (
                "retain the stance group for prospective challenger testing"
                if retain_stance
                else "reject the stance group from the leading winner challenger"
            ),
            "production_action": "none",
        },
        "source_sha256": {
            "point_in_time": _file_sha256(point_in_time_path),
            "raw_fights": _file_sha256(raw_fights_path),
            "fighter_stats": _file_sha256(fighter_stats_path),
            "auxiliary": _file_sha256(auxiliary_path),
            "external_report": _file_sha256(external_report_path),
            "market_report": _file_sha256(market_report_path),
            "market_detail": _file_sha256(market_detail_path),
        },
        "inherited_market_contract": market_report["market_contract"],
    }
    return report, all_predictions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report, detail = build_evaluation()
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
        "Stance factorial log loss: "
        + ", ".join(
            f"{name}={metrics[name]['log_loss']:.6f}" for name in VARIANTS
        )
    )
    print(f"Decision: {report['decision']['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
