"""Evaluate bounded round-cardio features alone and with external history.

The experiment is paper-only.  Its evidence-active primary horizon is 2026:
the round backfill begins in late 2024, making 2025 the first year that can
provide feature-bearing training examples to a strictly chronological fold.
Standard four-year, extended, coverage-sliced, and market comparisons are also
reported.  This command cannot modify production artifacts or policy.
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
    TemporalFightPredictor,
)
from fight_predictor.round_cardio import (
    CARDIO_LATE_ROUNDS,
    CARDIO_METRICS,
    CARDIO_PRIOR_FIGHTS,
    RoundCardioDatasetBuilder,
)


REPORT_SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "round-cardio-factorial-v1"
PRIMARY_EVIDENCE_YEAR = 2026
ROOT = Path(__file__).resolve().parent
DEFAULT_ROUND_STATS = (
    ROOT / "content/data/processed/ufc_fight_round_stats_doubled.csv"
)
DEFAULT_REPORT = ROOT / "content/data/external_mma/round_cardio_factorial.json"
DEFAULT_DETAIL = ROOT / "content/data/external_mma/round_cardio_factorial.csv"

VARIANTS = ("baseline", "external_history", "cardio", "external_cardio")
CHALLENGERS = VARIANTS[1:]


def _build_variant_points(
    production: pd.DataFrame,
    raw: pd.DataFrame,
    fighters: pd.DataFrame,
    round_stats: pd.DataFrame,
    auxiliary: pd.DataFrame,
) -> tuple[dict[str, PointInTimeDatasetBuilder], dict[str, pd.DataFrame]]:
    builders: dict[str, PointInTimeDatasetBuilder] = {
        "baseline": PointInTimeDatasetBuilder(raw, fighters),
        "external_history": PointInTimeDatasetBuilder(
            raw, fighters, auxiliary_fights=auxiliary
        ),
        "cardio": RoundCardioDatasetBuilder(raw, fighters, round_stats),
        "external_cardio": RoundCardioDatasetBuilder(
            raw,
            fighters,
            round_stats,
            auxiliary_fights=auxiliary,
        ),
    }
    points = {name: builder.build() for name, builder in builders.items()}
    _validate_candidate_baseline_features(
        production, points["baseline"], builders["baseline"].feature_columns
    )
    _validate_candidate_baseline_features(
        production, points["cardio"], builders["baseline"].feature_columns
    )
    _validate_candidate_baseline_features(
        points["external_history"],
        points["external_cardio"],
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
        "cardio_minus_baseline": ("cardio", "baseline"),
        "external_cardio_minus_baseline": ("external_cardio", "baseline"),
        "external_cardio_minus_external_history": (
            "external_cardio",
            "external_history",
        ),
        "external_cardio_minus_cardio": ("external_cardio", "cardio"),
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
        "external_effect_without_cardio": (
            losses["external_history"] - losses["baseline"]
        ),
        "cardio_effect_without_external": losses["cardio"] - losses["baseline"],
        "external_effect_with_cardio": (
            losses["external_cardio"] - losses["cardio"]
        ),
        "cardio_effect_with_external": (
            losses["external_cardio"] - losses["external_history"]
        ),
        "interaction": (
            losses["external_cardio"]
            - losses["external_history"]
            - losses["cardio"]
            + losses["baseline"]
        ),
    }


def _causal_counts(
    frame: pd.DataFrame,
    builder: RoundCardioDatasetBuilder,
    round_number: int = 2,
) -> pd.DataFrame:
    rows = []
    date_column = "date" if "date" in frame.columns else "event_date"
    if date_column not in frame.columns:
        raise ValueError("cardio coverage requires date or event_date")
    for item in frame[
        ["fighter_id", "opponent_id", date_column]
    ].itertuples(index=False, name=None):
        fighter_id, opponent_id, fight_date = item
        rows.append(
            (
                builder.cardio_sample_count(
                    str(fighter_id), fight_date, round_number
                ),
                builder.cardio_sample_count(
                    str(opponent_id), fight_date, round_number
                ),
            )
        )
    return pd.DataFrame(rows, columns=("fighter_samples", "opponent_samples"), index=frame.index)


def _coverage(
    frame: pd.DataFrame,
    builder: RoundCardioDatasetBuilder,
) -> dict[str, object]:
    r2 = _causal_counts(frame, builder, 2)
    r3 = _causal_counts(frame, builder, 3)

    def summary(counts: pd.DataFrame) -> dict[str, object]:
        any_side = counts.max(axis=1).gt(0)
        both_sides = counts.min(axis=1).gt(0)
        both_two = counts.min(axis=1).ge(2)
        return {
            "any_side_has_prior_sample": int(any_side.sum()),
            "any_side_fraction": float(any_side.mean()),
            "both_sides_have_prior_sample": int(both_sides.sum()),
            "both_sides_fraction": float(both_sides.mean()),
            "both_sides_have_two_or_more": int(both_two.sum()),
            "side_sample_fraction": float(
                (counts["fighter_samples"].gt(0).sum()
                 + counts["opponent_samples"].gt(0).sum())
                / (2 * len(counts))
            ),
        }

    return {
        "fights": len(frame),
        "round_2_vs_1": summary(r2),
        "round_3_vs_1": summary(r3),
    }


def _evaluation_block(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "metrics": _metrics_by_variant(frame),
        "paired_log_loss_intervals": _comparisons(frame),
        "factorial_log_loss_effects": _factorial_summary(
            _metrics_by_variant(frame)
        ),
    }


def build_evaluation(
    *,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTER_STATS,
    round_stats_path: Path = DEFAULT_ROUND_STATS,
    auxiliary_path: Path = DEFAULT_AUXILIARY,
    external_report_path: Path = DEFAULT_EXTERNAL_REPORT,
    market_report_path: Path = DEFAULT_MARKET_REPORT,
    market_detail_path: Path = DEFAULT_MARKET_DETAIL,
) -> tuple[dict[str, object], pd.DataFrame]:
    for path in (
        point_in_time_path,
        raw_fights_path,
        fighter_stats_path,
        round_stats_path,
        market_report_path,
        market_detail_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    production = pd.read_csv(point_in_time_path, low_memory=False)
    raw = pd.read_csv(raw_fights_path, low_memory=False)
    fighters = pd.read_csv(fighter_stats_path, low_memory=False)
    round_stats = pd.read_csv(round_stats_path, low_memory=False)
    auxiliary, external_report = load_frozen_research_auxiliary(
        auxiliary_path, external_report_path
    )
    market_report = json.loads(market_report_path.read_text(encoding="utf-8"))
    stored_market = pd.read_csv(market_detail_path, low_memory=False)

    builders, points = _build_variant_points(
        production, raw, fighters, round_stats, auxiliary
    )
    cardio_builder = builders["cardio"]
    if not isinstance(cardio_builder, RoundCardioDatasetBuilder):
        raise TypeError("cardio builder contract was not preserved")
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
    evidence_active = all_predictions[
        all_predictions["evaluation_year"].eq(PRIMARY_EVIDENCE_YEAR)
    ].copy()
    if evidence_active.empty:
        raise RuntimeError(f"primary evidence year {PRIMARY_EVIDENCE_YEAR} is absent")
    evidence_counts = _causal_counts(evidence_active, cardio_builder, 2)
    any_prior = evidence_active[evidence_counts.max(axis=1).gt(0)].copy()
    both_prior = evidence_active[evidence_counts.min(axis=1).gt(0)].copy()

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
    active_metrics = _metrics_by_variant(evidence_active)
    extended_metrics = _metrics_by_variant(all_predictions)
    market_metrics = {
        "market": _metric_mapping(market_paired, "market_probability"),
        **{
            name: _metric_mapping(market_paired, f"{name}_probability")
            for name in VARIANTS
        },
    }
    ranked = sorted(
        VARIANTS, key=lambda name: float(active_metrics[name]["log_loss"])
    )
    cardio_helps_baseline = (
        active_metrics["cardio"]["log_loss"]
        < active_metrics["baseline"]["log_loss"]
        and active_metrics["cardio"]["brier"]
        <= active_metrics["baseline"]["brier"]
    )
    cardio_helps_external = (
        active_metrics["external_cardio"]["log_loss"]
        < active_metrics["external_history"]["log_loss"]
        and active_metrics["external_cardio"]["brier"]
        <= active_metrics["external_history"]["brier"]
    )
    retain_cardio = bool(cardio_helps_baseline and cardio_helps_external)

    report: dict[str, object] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "promotable": False,
        "primary_metric": "evidence_active_2026_walk_forward_log_loss",
        "primary_horizon_predeclared_from_data_availability_not_outcomes": True,
        "non_promotable_flags": [
            "round_cardio_group_is_development_evidence",
            "round_backfill_is_incomplete_before_2024_09_28",
            "complete_late_round_conditioning_has_survivorship_selection",
            "current_data_period_informed_candidate_selection",
            "external_source_is_a_stale_bootstrap_through_2021_08_11",
            "market_history_missing_2024",
            "legacy_market_timestamps_are_not_verified_close_times",
        ],
        "feature_contract": {
            "baseline_feature_count": len(builders["baseline"].feature_columns),
            "cardio_feature_count": len(builders["cardio"].feature_columns),
            "additional_feature_count": (
                len(builders["cardio"].feature_columns)
                - len(builders["baseline"].feature_columns)
            ),
            "additional_features": list(
                builders["cardio"].feature_columns[
                    len(builders["baseline"].feature_columns):
                ]
            ),
            "late_rounds": list(CARDIO_LATE_ROUNDS),
            "metrics_per_round": list(CARDIO_METRICS),
            "shrinkage_prior_fights_at_zero_change": CARDIO_PRIOR_FIGHTS,
            "complete_five_minute_round_pairs_only": True,
            "partial_rounds_imputed": False,
            "unmatched_or_discrepant_round_rows_used": False,
            "every_feature_antisymmetric": True,
            "training_and_calibration": (
                "identical nested expanding-year L2 logistic procedure"
            ),
            "ufc_training_labels_identical": True,
        },
        "round_source": {
            "rows": len(round_stats),
            "physical_fights": int(round_stats["fight_id"].nunique()),
            "first_date": str(pd.to_datetime(round_stats["date"]).min().date()),
            "last_date": str(pd.to_datetime(round_stats["date"]).max().date()),
        },
        "coverage": {
            "production_horizon": _coverage(production_horizon, cardio_builder),
            "evidence_active_horizon": _coverage(evidence_active, cardio_builder),
            "market_paired": _coverage(market_paired, cardio_builder),
        },
        "sample": {
            "extended_years": list(years),
            "extended_fights": len(all_predictions),
            "production_years": list(production_years),
            "production_fights": len(production_horizon),
            "evidence_active_year": PRIMARY_EVIDENCE_YEAR,
            "evidence_active_fights": len(evidence_active),
            "evidence_active_any_prior_round_2_sample": len(any_prior),
            "evidence_active_both_prior_round_2_samples": len(both_prior),
            "market_paired_fights": len(market_paired),
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
        "evidence_active_walk_forward": {
            "metrics": active_metrics,
            "paired_log_loss_intervals": _comparisons(evidence_active),
            "factorial_log_loss_effects": _factorial_summary(active_metrics),
        },
        "evidence_active_coverage_slices": {
            "any_side_has_prior_round_2_sample": _evaluation_block(any_prior),
            "both_sides_have_prior_round_2_sample": _evaluation_block(both_prior),
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
            "external_cardio_minus_market_interval": (
                event_block_difference_interval(
                    market_paired,
                    "external_cardio_probability",
                    "market_probability",
                )
            ),
            "factorial_log_loss_effects": _factorial_summary(
                {name: market_metrics[name] for name in VARIANTS}
            ),
        },
        "decision": {
            "ranked_by_primary_log_loss": ranked,
            "best_variant": ranked[0],
            "cardio_improved_baseline_log_loss_and_brier": cardio_helps_baseline,
            "cardio_improved_external_log_loss_and_brier": cardio_helps_external,
            "cardio_group_retained_for_leading_challenger": retain_cardio,
            "recommendation": (
                "retain round cardio for prospective challenger testing"
                if retain_cardio
                else "reject round cardio from the leading winner challenger"
            ),
            "production_action": "none",
        },
        "source_sha256": {
            "point_in_time": _file_sha256(point_in_time_path),
            "raw_fights": _file_sha256(raw_fights_path),
            "fighter_stats": _file_sha256(fighter_stats_path),
            "round_stats": _file_sha256(round_stats_path),
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
    metrics = report["evidence_active_walk_forward"]["metrics"]
    print(
        "Round-cardio 2026 log loss: "
        + ", ".join(
            f"{name}={metrics[name]['log_loss']:.6f}" for name in VARIANTS
        )
    )
    print(f"Decision: {report['decision']['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
