"""Run a causal 2x2 winner-feature experiment for external history and style.

The four variants use an identical nested expanding-year logistic procedure:

* frozen production baseline;
* baseline plus the hash-pinned external MMA state replay;
* baseline plus the frozen style-matchup feature group;
* external MMA state replay plus the style-matchup feature group.

This is a development comparison.  It cannot update the production model or
enable the external-history policy.  Its purpose is to determine whether the
two previously retained feature candidates remain useful when combined.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
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
from fight_predictor import (
    PointInTimeDatasetBuilder,
    StyleMatchupDatasetBuilder,
    TemporalFightPredictor,
)
from fight_predictor.point_in_time import COUNT_STATS
from fight_predictor.style_matchup import STYLE_COUNT_STATS


REPORT_SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "winner-feature-factorial-v1"
ROOT = Path(__file__).resolve().parent
DEFAULT_POINT_IN_TIME = ROOT / "content/data/processed/ufc_fights_point_in_time.csv"
DEFAULT_RAW_FIGHTS = ROOT / "content/data/processed/ufc_fights_reported_doubled.csv"
DEFAULT_FIGHTER_STATS = ROOT / "content/data/processed/fighter_stats.csv"
DEFAULT_AUXILIARY = (
    ROOT / "content/data/processed/external_mma_auxiliary_doubled.csv"
)
DEFAULT_EXTERNAL_REPORT = ROOT / "content/data/external_mma/evaluation_report.json"
DEFAULT_REPORT = (
    ROOT / "content/data/external_mma/winner_feature_factorial.json"
)
DEFAULT_DETAIL = (
    ROOT / "content/data/external_mma/winner_feature_factorial.csv"
)

VARIANTS = ("baseline", "external_history", "style", "external_style")
CHALLENGERS = VARIANTS[1:]


def load_frozen_research_auxiliary(
    auxiliary_path: Path,
    external_report_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the evaluated auxiliary file without bypassing production policy.

    Research may compare the disabled candidate, but only the separate
    production loader may honor ``model_policy.json``.  The evaluation report
    pins the exact bytes used here.
    """

    if not auxiliary_path.is_file():
        raise FileNotFoundError(auxiliary_path)
    if not external_report_path.is_file():
        raise FileNotFoundError(external_report_path)
    report = json.loads(external_report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("external MMA evaluation report has an unsupported schema")
    expected = str(report.get("auxiliary_sha256", "")).strip().lower()
    actual = sha256(auxiliary_path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError("external MMA auxiliary does not match its evaluation hash")

    auxiliary = pd.read_csv(auxiliary_path, low_memory=False)
    required = {
        "fight_url",
        "fighter_url",
        "opponent_url",
        "emit_training_target",
        *COUNT_STATS,
    }
    missing = required - set(auxiliary.columns)
    if missing:
        raise ValueError(f"external MMA auxiliary is missing columns: {sorted(missing)}")
    if auxiliary.empty:
        raise ValueError("external MMA auxiliary is empty")
    flags = auxiliary["emit_training_target"].astype(str).str.casefold()
    if not flags.isin({"false", "0"}).all():
        raise ValueError("external MMA auxiliary contains a training-label row")
    perspectives = auxiliary.groupby("fight_url", sort=False).size()
    if not perspectives.eq(2).all():
        raise ValueError("external MMA auxiliary is not exactly doubled by bout")
    if auxiliary[list(COUNT_STATS)].notna().any().any():
        raise ValueError("external MMA metadata may not fabricate detailed statistics")
    if auxiliary[["fighter_url", "opponent_url"]].isna().any().any():
        raise ValueError("external MMA auxiliary contains an unidentified participant")
    return auxiliary, report


def _build_variant_points(
    production: pd.DataFrame,
    raw: pd.DataFrame,
    fighters: pd.DataFrame,
    auxiliary: pd.DataFrame,
) -> tuple[
    dict[str, PointInTimeDatasetBuilder],
    dict[str, pd.DataFrame],
]:
    style_auxiliary = auxiliary.copy()
    for column in STYLE_COUNT_STATS:
        if column not in style_auxiliary:
            style_auxiliary[column] = np.nan

    builders: dict[str, PointInTimeDatasetBuilder] = {
        "baseline": PointInTimeDatasetBuilder(raw, fighters),
        "external_history": PointInTimeDatasetBuilder(
            raw, fighters, auxiliary_fights=auxiliary
        ),
        "style": StyleMatchupDatasetBuilder(raw, fighters),
        "external_style": StyleMatchupDatasetBuilder(
            raw, fighters, auxiliary_fights=style_auxiliary
        ),
    }
    points = {name: builder.build() for name, builder in builders.items()}

    _validate_candidate_baseline_features(
        production,
        points["baseline"],
        builders["baseline"].feature_columns,
    )
    _validate_candidate_baseline_features(
        production,
        points["style"],
        builders["baseline"].feature_columns,
    )
    _validate_candidate_baseline_features(
        points["external_history"],
        points["external_style"],
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


def _align_variants(
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
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
        name: _metric(frame, f"{name}_probability")
        for name in VARIANTS
    }


def _comparisons(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    pairs = {
        "external_history_minus_baseline": ("external_history", "baseline"),
        "style_minus_baseline": ("style", "baseline"),
        "external_style_minus_baseline": ("external_style", "baseline"),
        "external_style_minus_external_history": (
            "external_style",
            "external_history",
        ),
        "external_style_minus_style": ("external_style", "style"),
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
        "external_effect_without_style": (
            losses["external_history"] - losses["baseline"]
        ),
        "style_effect_without_external": losses["style"] - losses["baseline"],
        "external_effect_with_style": (
            losses["external_style"] - losses["style"]
        ),
        "style_effect_with_external": (
            losses["external_style"] - losses["external_history"]
        ),
        "interaction": (
            losses["external_style"]
            - losses["external_history"]
            - losses["style"]
            + losses["baseline"]
        ),
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

    builders, points = _build_variant_points(
        production, raw, fighters, auxiliary
    )
    first_year = int(stored_market["evaluation_year"].min())
    final_year = int(pd.to_datetime(production["date"], errors="raise").dt.year.max())
    years = tuple(range(first_year, final_year + 1))
    predictions = _walk_forward_variants(production, builders, points, years)
    all_predictions = _align_variants(predictions)

    production_years = tuple(
        int(year)
        for year in sorted(all_predictions["evaluation_year"].unique())[-4:]
    )
    production_horizon = all_predictions[
        all_predictions["evaluation_year"].isin(production_years)
    ].copy()
    # The historical replay intentionally records the exact baseline produced
    # when that audit ran.  Later source/profile corrections can legitimately
    # move today's reconstructed probabilities.  Use its immutable market
    # observations and identities, but compare every feature variant against
    # the baseline recomputed from the same current input bytes.
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
    report: dict[str, object] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "promotable": False,
        "non_promotable_flags": [
            "factorial_feature_test_is_development_evidence",
            "current_data_period_informed_candidate_selection",
            "external_source_is_a_stale_bootstrap_through_2021_08_11",
            "market_history_missing_2024",
            "legacy_market_timestamps_are_not_verified_close_times",
        ],
        "feature_contract": {
            "baseline_feature_count": len(builders["baseline"].feature_columns),
            "style_feature_count": len(builders["style"].feature_columns),
            "external_history_adds_feature_columns": False,
            "style_additional_feature_count": (
                len(builders["style"].feature_columns)
                - len(builders["baseline"].feature_columns)
            ),
            "training_and_calibration": (
                "identical nested expanding-year L2 logistic procedure"
            ),
            "ufc_training_labels_identical": True,
        },
        "external_history_contract": {
            "auxiliary_sha256": external_report["auxiliary_sha256"],
            "physical_bouts": int(auxiliary["fight_url"].nunique()),
            "perspective_rows": len(auxiliary),
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
            "definition": (
                "absolute difference between the original market-replay baseline "
                "and the current-algorithm baseline rebuilt from current source bytes"
            ),
            "changed_fights": int(np.count_nonzero(baseline_probability_drift > 1e-12)),
            "maximum_absolute_probability_difference": float(
                baseline_probability_drift.max()
            ),
            "mean_absolute_probability_difference": float(
                baseline_probability_drift.mean()
            ),
            "current_recomputed_baseline_used_for_factorial_comparisons": True,
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
            "external_style_minus_market_interval": (
                event_block_difference_interval(
                    market_paired,
                    "external_style_probability",
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
            "external_history_improved_production_point_log_loss": (
                production_metrics["external_history"]["log_loss"]
                < production_metrics["baseline"]["log_loss"]
            ),
            "style_improved_production_point_log_loss": (
                production_metrics["style"]["log_loss"]
                < production_metrics["baseline"]["log_loss"]
            ),
            "combined_improved_on_both_single_candidates": (
                production_metrics["external_style"]["log_loss"]
                < min(
                    production_metrics["external_history"]["log_loss"],
                    production_metrics["style"]["log_loss"],
                )
            ),
            "recommendation": (
                "retain external+style as a prospective shadow challenger"
                if ranked[0] == "external_style"
                else f"retain {ranked[0]} as the leading research challenger"
            ),
            "production_action": (
                "none; separately review dependent outcome-model effects and "
                "prospective evidence before changing the weekly feature contract"
            ),
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
        detail_text = detail.to_csv(
            index=False,
            lineterminator="\n",
            float_format="%.15g",
        )
        _atomic_write_text(arguments.detail, detail_text)
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
        "Winner feature factorial log loss: "
        + ", ".join(
            f"{name}={metrics[name]['log_loss']:.6f}" for name in VARIANTS
        )
    )
    print(f"Decision: {report['decision']['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
