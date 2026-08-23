"""Evaluate a frozen style-matchup feature group against model and market.

The challenger uses the same nested expanding-year logistic-regression process
as production.  It is development-only and cannot update the production model
artifact.  Historical market comparisons inherit the limitations recorded in
the baseline current-model replay.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import pandas as pd

from evaluate_current_model_vs_market import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DEFAULT_DETAIL as DEFAULT_BASELINE_DETAIL,
    DEFAULT_REPORT as DEFAULT_BASELINE_REPORT,
    _atomic_write_text,
    _file_sha256,
    _fold_contract,
    _metric_mapping,
    _quantile,
    evaluate_prior_card_blend,
)
from external_mma import load_approved_auxiliary
from fight_predictor import (
    PointInTimeDatasetBuilder,
    StyleMatchupDatasetBuilder,
    TemporalFightPredictor,
)
from fight_predictor.point_in_time import _metrics
from fight_predictor.style_matchup import STYLE_COUNT_STATS


REPORT_SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "style-matchup-challenger-v1"
ROOT = Path(__file__).resolve().parent
DEFAULT_POINT_IN_TIME = ROOT / "content/data/processed/ufc_fights_point_in_time.csv"
DEFAULT_RAW_FIGHTS = ROOT / "content/data/processed/ufc_fights_reported_doubled.csv"
DEFAULT_FIGHTER_STATS = ROOT / "content/data/processed/fighter_stats.csv"
DEFAULT_AUXILIARY = (
    ROOT / "content/data/processed/external_mma_auxiliary_doubled.csv"
)
DEFAULT_AUXILIARY_POLICY = ROOT / "content/data/external_mma/model_policy.json"
DEFAULT_REPORT = (
    ROOT / "content/data/market_history_backfill/style_matchup_challenger.json"
)
DEFAULT_DETAIL = (
    ROOT / "content/data/market_history_backfill/style_matchup_challenger.csv"
)

NON_PROMOTABLE_FLAGS = (
    "single_predeclared_feature_group_development_test",
    "retrospective_current_algorithm_not_historical_frozen_forecast",
    "legacy_commit_timestamp_not_source_quote_timestamp",
    "current_reconciled_raw_not_as_of_snapshot",
    "profile_and_source_corrections_may_postdate_fight",
    "feature_contract_was_developed_using_some_evaluation_years",
    "missing_2024_market_history",
    "unverified_execution_and_closing_price",
    "development_only",
)


def event_block_difference_interval(
    frame: pd.DataFrame,
    candidate_column: str,
    reference_column: str,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    """Bootstrap paired candidate-minus-reference log loss by event."""

    if frame.empty:
        raise ValueError("cannot bootstrap an empty comparison")
    required = {"event_id", "target", candidate_column, reference_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"comparison is missing columns: {sorted(missing)}")
    blocks: dict[str, list[float]] = {}
    for row in frame.to_dict("records"):
        target = int(row["target"])
        candidate = float(np.clip(row[candidate_column], 1e-15, 1.0 - 1e-15))
        reference = float(np.clip(row[reference_column], 1e-15, 1.0 - 1e-15))
        candidate_loss = -(
            target * math.log(candidate) + (1 - target) * math.log1p(-candidate)
        )
        reference_loss = -(
            target * math.log(reference) + (1 - target) * math.log1p(-reference)
        )
        blocks.setdefault(str(row["event_id"]), []).append(
            candidate_loss - reference_loss
        )
    block_ids = sorted(blocks)
    all_deltas = [delta for values in blocks.values() for delta in values]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [rng.choice(block_ids) for _ in block_ids]
        deltas = [delta for block_id in selected for delta in blocks[block_id]]
        samples.append(sum(deltas) / len(deltas))
    return {
        "definition": (
            f"{candidate_column} minus {reference_column} paired log loss; "
            "negative favors candidate"
        ),
        "method": "nonparametric whole-event block bootstrap",
        "seed": seed,
        "bootstrap_samples": replicates,
        "event_count": len(block_ids),
        "fight_count": len(frame),
        "point_difference": sum(all_deltas) / len(all_deltas),
        "ci_95_lower": _quantile(samples, 0.025),
        "ci_95_upper": _quantile(samples, 0.975),
    }


def _align_predictions(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    probability_name: str,
) -> pd.DataFrame:
    if reference["fight_id"].duplicated().any() or candidate["fight_id"].duplicated().any():
        raise ValueError("prediction comparison contains duplicate fight IDs")
    candidate_by_fight = {
        str(row["fight_id"]): row for row in candidate.to_dict("records")
    }
    probabilities: list[float] = []
    for row in reference.to_dict("records"):
        other = candidate_by_fight.get(str(row["fight_id"]))
        if other is None:
            raise ValueError(f"candidate is missing fight {row['fight_id']}")
        same = (
            str(row["fighter_id"]) == str(other["fighter_id"])
            and str(row["opponent_id"]) == str(other["opponent_id"])
        )
        reversed_sides = (
            str(row["fighter_id"]) == str(other["opponent_id"])
            and str(row["opponent_id"]) == str(other["fighter_id"])
        )
        if not same and not reversed_sides:
            raise ValueError(f"fighter identity mismatch for fight {row['fight_id']}")
        probability = float(other["model_probability"])
        target = int(other["target"])
        if reversed_sides:
            probability = 1.0 - probability
            target = 1 - target
        if target != int(row["target"]):
            raise ValueError(f"target mismatch for fight {row['fight_id']}")
        if str(row["event_id"]) != str(other["event_id"]):
            raise ValueError(f"event mismatch for fight {row['fight_id']}")
        probabilities.append(probability)
    result = reference.copy()
    result[probability_name] = probabilities
    return result


def _validate_baseline_reproduction(
    stored: pd.DataFrame,
    reproduced: pd.DataFrame,
) -> None:
    aligned = _align_predictions(
        stored,
        reproduced,
        probability_name="reproduced_baseline_probability",
    )
    if not np.allclose(
        aligned["model_probability"].to_numpy(dtype=float),
        aligned["reproduced_baseline_probability"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        difference = np.max(
            np.abs(
                aligned["model_probability"].to_numpy(dtype=float)
                - aligned["reproduced_baseline_probability"].to_numpy(dtype=float)
            )
        )
        raise RuntimeError(
            f"stored baseline replay is stale; maximum probability drift is {difference}"
        )


def _validate_candidate_baseline_features(
    production: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_columns: Sequence[str],
) -> None:
    if production["fight_id"].tolist() != candidate["fight_id"].tolist():
        raise RuntimeError("candidate changed the UFC training-label order or set")
    production_values = production[list(baseline_columns)].to_numpy(dtype=float)
    candidate_values = candidate[list(baseline_columns)].to_numpy(dtype=float)
    if not np.allclose(
        production_values,
        candidate_values,
        equal_nan=True,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("candidate does not reproduce the frozen baseline features")


def _metric(frame: pd.DataFrame, column: str) -> dict[str, object]:
    return _metrics(
        frame["target"].to_numpy(dtype=int),
        frame[column].to_numpy(dtype=float),
    )


def _per_year(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for year, rows in frame.groupby(frame["event_date"].astype(str).str[:4], sort=True):
        evaluated = rows[rows["candidate_blend_status"] == "evaluated"]
        result[str(year)] = {
            "fights": len(rows),
            "market": _metric_mapping(rows, "market_probability"),
            "baseline": _metric_mapping(rows, "baseline_probability"),
            "style_challenger": _metric_mapping(rows, "candidate_probability"),
            "baseline_blend": (
                _metric_mapping(evaluated, "baseline_blend_probability")
                if not evaluated.empty
                else None
            ),
            "style_blend": (
                _metric_mapping(evaluated, "candidate_blend_probability")
                if not evaluated.empty
                else None
            ),
        }
    return result


def build_evaluation(
    *,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTER_STATS,
    auxiliary_path: Path = DEFAULT_AUXILIARY,
    auxiliary_policy_path: Path = DEFAULT_AUXILIARY_POLICY,
    baseline_report_path: Path = DEFAULT_BASELINE_REPORT,
    baseline_detail_path: Path = DEFAULT_BASELINE_DETAIL,
) -> tuple[dict[str, object], pd.DataFrame]:
    for path in (
        point_in_time_path,
        raw_fights_path,
        fighter_stats_path,
        auxiliary_policy_path,
        baseline_report_path,
        baseline_detail_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    production = pd.read_csv(point_in_time_path, low_memory=False)
    raw = pd.read_csv(raw_fights_path, low_memory=False)
    fighters = pd.read_csv(fighter_stats_path, low_memory=False)
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    stored_paired = pd.read_csv(baseline_detail_path, low_memory=False)
    auxiliary = load_approved_auxiliary(auxiliary_path, auxiliary_policy_path)
    if auxiliary is not None and not auxiliary.empty:
        auxiliary = auxiliary.copy()
        for column in StyleMatchupDatasetBuilder.STATE_COUNT_STATS:
            if column not in auxiliary:
                auxiliary[column] = np.nan

    baseline_builder = PointInTimeDatasetBuilder(
        raw, fighters, auxiliary_fights=auxiliary
    )
    candidate_builder = StyleMatchupDatasetBuilder(
        raw, fighters, auxiliary_fights=auxiliary
    )
    candidate_point = candidate_builder.build()
    _validate_candidate_baseline_features(
        production,
        candidate_point,
        baseline_builder.feature_columns,
    )

    market_years = tuple(
        int(year)
        for year in sorted(
            stored_paired["event_date"].astype(str).str[:4].astype(int).unique()
        )
    )
    baseline_predictions = TemporalFightPredictor(
        production, baseline_builder
    ).walk_forward_predictions(market_years)
    candidate_predictor = TemporalFightPredictor(candidate_point, candidate_builder)
    candidate_predictions = candidate_predictor.walk_forward_predictions(market_years)
    _validate_baseline_reproduction(stored_paired, baseline_predictions)

    all_predictions = baseline_predictions.rename(
        columns={"model_probability": "baseline_probability"}
    )
    all_predictions = _align_predictions(
        all_predictions,
        candidate_predictions,
        probability_name="candidate_probability",
    )
    paired = stored_paired.drop(
        columns=[
            column
            for column in stored_paired.columns
            if column in {
                "market_log_loss",
                "model_log_loss",
                "model_minus_market_log_loss",
                "blend_log_loss",
                "blend_minus_market_log_loss",
            }
        ],
        errors="ignore",
    ).rename(
        columns={
            "model_probability": "baseline_probability",
            "blend_status": "baseline_blend_status",
            "prior_card_count": "baseline_prior_card_count",
            "prior_fight_count": "baseline_prior_fight_count",
            "blend_training_through_event_date": (
                "baseline_blend_training_through_event_date"
            ),
            "selected_gamma": "baseline_selected_gamma",
            "selection_prior_card_log_loss": (
                "baseline_selection_prior_card_log_loss"
            ),
            "blend_probability": "baseline_blend_probability",
        }
    )
    paired = _align_predictions(
        paired,
        candidate_predictions,
        probability_name="candidate_probability",
    )

    blend_input = paired.copy()
    blend_input["model_probability"] = blend_input["candidate_probability"]
    candidate_blend = evaluate_prior_card_blend(blend_input)
    rename = {
        "blend_status": "candidate_blend_status",
        "prior_card_count": "candidate_prior_card_count",
        "prior_fight_count": "candidate_prior_fight_count",
        "blend_training_through_event_date": (
            "candidate_blend_training_through_event_date"
        ),
        "selected_gamma": "candidate_selected_gamma",
        "selection_prior_card_log_loss": "candidate_selection_prior_card_log_loss",
        "blend_probability": "candidate_blend_probability",
    }
    for source, destination in rename.items():
        paired[destination] = candidate_blend[source]

    evaluated = paired[paired["candidate_blend_status"] == "evaluated"].copy()
    if not (
        paired["candidate_blend_status"] == paired["baseline_blend_status"]
    ).all():
        raise RuntimeError("baseline and candidate blend eligibility differs")
    candidate_better_all = (
        _metric(all_predictions, "candidate_probability")["log_loss"]
        < _metric(all_predictions, "baseline_probability")["log_loss"]
    )
    candidate_better_market_sample = (
        _metric_mapping(paired, "candidate_probability")["log_loss"]
        < _metric_mapping(paired, "baseline_probability")["log_loss"]
    )
    candidate_blend_interval = event_block_difference_interval(
        evaluated,
        "candidate_blend_probability",
        "market_probability",
    )
    candidate_vs_baseline_blend_interval = event_block_difference_interval(
        evaluated,
        "candidate_blend_probability",
        "baseline_blend_probability",
    )
    detail_text = paired.to_csv(
        index=False, lineterminator="\n", float_format="%.15g"
    )
    additional_features = list(
        candidate_builder.feature_columns[len(baseline_builder.feature_columns):]
    )
    report: dict[str, object] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "paper_only": True,
        "promotable": False,
        "non_promotable_flags": list(NON_PROMOTABLE_FLAGS),
        "feature_contract": {
            "baseline_feature_count": len(baseline_builder.feature_columns),
            "challenger_feature_count": len(candidate_builder.feature_columns),
            "additional_feature_count": len(additional_features),
            "additional_features": additional_features,
            "baseline_features_reproduced": True,
            "all_features_antisymmetric_by_construction": True,
            "training_and_calibration": (
                "same nested expanding-year logistic-regression procedure as baseline"
            ),
        },
        "data_quality": {
            "raw_rows": len(raw),
            "raw_physical_fights": int(raw["fight_url"].nunique()),
            "granular_style_missing_cells": int(
                raw[list(STYLE_COUNT_STATS)]
                .isna()
                .sum()
                .sum()
            ),
            "significant_strike_partitions_validated": True,
            "landed_not_above_attempted_validated": True,
            "future_append_invariance_unit_tested": True,
        },
        "sample": {
            "all_walk_forward_fights": len(all_predictions),
            "all_walk_forward_events": int(all_predictions["event_id"].nunique()),
            "paired_market_fights": len(paired),
            "paired_market_events": int(paired["event_id"].nunique()),
            "market_years": list(market_years),
        },
        "all_walk_forward_fights": {
            "baseline": _metric(all_predictions, "baseline_probability"),
            "style_challenger": _metric(all_predictions, "candidate_probability"),
            "challenger_minus_baseline_log_loss_interval": (
                event_block_difference_interval(
                    all_predictions,
                    "candidate_probability",
                    "baseline_probability",
                )
            ),
        },
        "paired_market_fights": {
            "market": _metric_mapping(paired, "market_probability"),
            "baseline": _metric_mapping(paired, "baseline_probability"),
            "style_challenger": _metric_mapping(paired, "candidate_probability"),
            "challenger_minus_baseline_log_loss_interval": (
                event_block_difference_interval(
                    paired,
                    "candidate_probability",
                    "baseline_probability",
                )
            ),
            "challenger_minus_market_log_loss_interval": (
                event_block_difference_interval(
                    paired,
                    "candidate_probability",
                    "market_probability",
                )
            ),
        },
        "prior_card_selected_style_blend": {
            "selection_rule": "completed cards on strictly earlier event dates only",
            "evaluated_fights": len(evaluated),
            "skipped_warmup_fights": len(paired) - len(evaluated),
            "market": _metric_mapping(evaluated, "market_probability"),
            "style_challenger": _metric_mapping(
                evaluated, "candidate_probability"
            ),
            "baseline_blend": _metric_mapping(
                evaluated, "baseline_blend_probability"
            ),
            "style_blend": _metric_mapping(
                evaluated, "candidate_blend_probability"
            ),
            "blend_minus_market_log_loss_interval": candidate_blend_interval,
            "style_blend_minus_baseline_blend_log_loss_interval": (
                candidate_vs_baseline_blend_interval
            ),
            "baseline_selected_gamma_by_fight": {
                f"{gamma:.2f}": int(count)
                for gamma, count in evaluated["baseline_selected_gamma"]
                .value_counts()
                .sort_index()
                .items()
            },
            "selected_gamma_by_fight": {
                f"{gamma:.2f}": int(count)
                for gamma, count in evaluated["candidate_selected_gamma"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "per_year": _per_year(paired),
        "decision": {
            "challenger_improved_all_walk_forward_point_log_loss": (
                candidate_better_all
            ),
            "challenger_improved_market_sample_point_log_loss": (
                candidate_better_market_sample
            ),
            "blend_beat_market_with_95_percent_interval": (
                candidate_blend_interval["ci_95_upper"] < 0.0
            ),
            "style_blend_improved_baseline_blend_point_log_loss": (
                candidate_vs_baseline_blend_interval["point_difference"] < 0.0
            ),
            "recommendation": (
                "retain_as_challenger_pending independent future cards"
                if candidate_better_all and candidate_better_market_sample
                else "reject this feature group and retain the production baseline"
            ),
        },
        "folds": {
            "baseline": _fold_contract(baseline_predictions),
            "style_challenger": _fold_contract(candidate_predictions),
        },
        "source_sha256": {
            "point_in_time": _file_sha256(point_in_time_path),
            "raw_fights": _file_sha256(raw_fights_path),
            "fighter_stats": _file_sha256(fighter_stats_path),
            "baseline_report": _file_sha256(baseline_report_path),
            "baseline_detail": _file_sha256(baseline_detail_path),
            "detail_csv": sha256(detail_text.encode("utf-8")).hexdigest(),
        },
        "inherited_baseline_market_contract": baseline_report["market_contract"],
    }
    return report, paired


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
            index=False, lineterminator="\n", float_format="%.15g"
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
    all_fights = report["all_walk_forward_fights"]
    market = report["paired_market_fights"]
    blend = report["prior_card_selected_style_blend"]
    print(
        "Style challenger: "
        f"all baseline={all_fights['baseline']['log_loss']:.5f}, "
        f"style={all_fights['style_challenger']['log_loss']:.5f}; "
        f"market sample baseline={market['baseline']['log_loss']:.5f}, "
        f"style={market['style_challenger']['log_loss']:.5f}, "
        f"market={market['market']['log_loss']:.5f}; "
        f"style blend={blend['style_blend']['log_loss']:.5f}"
    )
    print(f"Decision: {report['decision']['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
