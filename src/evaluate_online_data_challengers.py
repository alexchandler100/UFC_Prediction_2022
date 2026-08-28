"""Audit and test free online data without changing production predictions.

The comparison covers historical UFC rankings, a large public MMA result
archive, and odds rows with verifiably pre-event capture times.  Each calendar
year is predicted by models trained only on earlier fights.  Ranking feature
groups are selected using only the training period for that year.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd

from data_handler.data_handler import atomic_write_text
from evaluate_current_model_vs_market import (
    evaluate_prior_card_blend,
    symmetric_logit_blend,
)
from evaluate_feature_selection import (
    _calibrate,
    _calibration_slope,
    _candidate_frame,
    _fit_pipeline,
    _l2_configs,
    _pipeline_probability,
    _tune,
)
from evaluate_style_matchup_challenger import (
    _align_predictions,
    _metric,
    event_block_difference_interval,
)
from fight_predictor import PointInTimeDatasetBuilder, TemporalFightPredictor
from online_data_research import (
    RANKING_FAMILIES,
    RANKING_FEATURES,
    add_ranking_features,
    file_sha256,
    load_mma_archive_observations,
    prepare_mma_auxiliary,
    prepare_pre_event_odds,
    prepare_rankings,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src/content/data"
AUDIT_DATA = ROOT / "artifacts/free_data_audit"
DEFAULT_POINT_IN_TIME = DATA / "processed/ufc_fights_point_in_time.csv"
DEFAULT_RAW_FIGHTS = DATA / "processed/ufc_fights_reported_doubled.csv"
DEFAULT_FIGHTERS = DATA / "processed/fighter_stats.csv"
DEFAULT_RANKINGS = AUDIT_DATA / "rankings/UFC_rankings_history.csv"
DEFAULT_ODDS = AUDIT_DATA / "odds/UFC_betting_odds.csv"
DEFAULT_MMA_DATABASE = AUDIT_DATA / "mmastats/dataset_global_v3.duckdb"
DEFAULT_REPORT = DATA / "model_research/online_data_challengers.json"
DEFAULT_DETAIL = DATA / "model_research/online_data_challengers.csv"
DEFAULT_YEARS = (2022, 2023, 2024, 2025, 2026)
MAX_RUNTIME_MINUTES = 60.0


def _ranking_family_combinations() -> list[tuple[str, ...]]:
    names = sorted(RANKING_FAMILIES)
    return [
        combination
        for size in range(len(names) + 1)
        for combination in itertools.combinations(names, size)
    ]


def _ranking_features(combination: Sequence[str], baseline: Sequence[str]) -> list[str]:
    return [
        *baseline,
        *(
            feature
            for family in combination
            for feature in RANKING_FAMILIES[family]
        ),
    ]


def _ranking_walk_forward(
    pool: pd.DataFrame,
    baseline_features: Sequence[str],
    years: tuple[int, ...],
    started: float,
    maximum_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frame = pool.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    selected_parts: list[pd.DataFrame] = []
    all_parts: list[pd.DataFrame] = []
    folds: dict[str, object] = {}
    combinations = _ranking_family_combinations()
    for year in years:
        if time.monotonic() - started > maximum_seconds:
            raise RuntimeError("online-data experiment exceeded its 60-minute limit")
        test_start = pd.Timestamp(year=year, month=1, day=1)
        train = frame.loc[
            (frame["date"] >= test_start - pd.DateOffset(years=10))
            & (frame["date"] < test_start)
        ].reset_index(drop=True)
        test = frame.loc[frame["date"].dt.year.eq(year)].reset_index(drop=True)
        if len(train) < 500 or test.empty:
            continue
        fold_started = time.monotonic()
        choices: list[dict[str, object]] = []
        tuned: dict[tuple[str, ...], tuple[object, np.ndarray, np.ndarray]] = {}
        for combination in combinations:
            features = _ranking_features(combination, baseline_features)
            config, scores, y_oof, p_oof = _tune(train, features, _l2_configs())
            best_score = float(scores[config.key])
            tuned[combination] = (config, y_oof, p_oof)
            choices.append({
                "families": list(combination),
                "feature_count": len(features),
                "selected_config": config.key,
                "training_validation_log_loss": best_score,
            })
        selected = min(
            combinations,
            key=lambda item: (
                next(row["training_validation_log_loss"] for row in choices if row["families"] == list(item)),
                len(item),
                item,
            ),
        )
        selected_features = _ranking_features(selected, baseline_features)
        config, y_oof, p_oof = tuned[selected]
        slope = _calibration_slope(y_oof, p_oof)
        pipeline = _fit_pipeline(train, selected_features, config)
        selected_probability = _calibrate(
            _pipeline_probability(pipeline, test, selected_features), slope
        )
        selected_parts.append(_candidate_frame(test, selected_probability))

        all_features = [*baseline_features, *RANKING_FEATURES]
        all_config, all_scores, all_y, all_p = _tune(
            train, all_features, _l2_configs()
        )
        all_slope = _calibration_slope(all_y, all_p)
        all_pipeline = _fit_pipeline(train, all_features, all_config)
        all_probability = _calibrate(
            _pipeline_probability(all_pipeline, test, all_features), all_slope
        )
        all_parts.append(_candidate_frame(test, all_probability))
        folds[str(year)] = {
            "train_fights": int(len(train)),
            "test_fights": int(len(test)),
            "selected_ranking_families": list(selected),
            "selected_ranking_feature_count": len(selected_features) - len(baseline_features),
            "selected_config": config.key,
            "calibration_slope": slope,
            "all_rankings_config": all_config.key,
            "all_rankings_calibration_slope": all_slope,
            "family_combination_training_results": sorted(
                choices,
                key=lambda row: (
                    row["training_validation_log_loss"],
                    len(row["families"]),
                    row["families"],
                ),
            ),
            "elapsed_seconds": time.monotonic() - fold_started,
        }
        print(
            f"Rankings: trained on {len(train)} earlier fights, tested {len(test)} "
            f"fights in {year}, selected {list(selected) or ['no ranking features']}"
        )
    if not selected_parts or not all_parts:
        raise ValueError("no eligible years for ranking evaluation")
    return (
        pd.concat(selected_parts, ignore_index=True),
        pd.concat(all_parts, ignore_index=True),
        folds,
    )


def _fixed_feature_walk_forward(
    pool: pd.DataFrame,
    features: Sequence[str],
    years: tuple[int, ...],
    started: float,
    maximum_seconds: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate one predeclared feature set with earlier-only model tuning."""

    frame = pool.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    parts: list[pd.DataFrame] = []
    folds: dict[str, object] = {}
    for year in years:
        if time.monotonic() - started > maximum_seconds:
            raise RuntimeError("online-data experiment exceeded its 60-minute limit")
        test_start = pd.Timestamp(year=year, month=1, day=1)
        train = frame.loc[
            (frame["date"] >= test_start - pd.DateOffset(years=10))
            & (frame["date"] < test_start)
        ].reset_index(drop=True)
        test = frame.loc[frame["date"].dt.year.eq(year)].reset_index(drop=True)
        if len(train) < 500 or test.empty:
            continue
        config, scores, y_oof, p_oof = _tune(train, features, _l2_configs())
        slope = _calibration_slope(y_oof, p_oof)
        pipeline = _fit_pipeline(train, features, config)
        probability = _calibrate(
            _pipeline_probability(pipeline, test, features), slope
        )
        parts.append(_candidate_frame(test, probability))
        folds[str(year)] = {
            "train_fights": int(len(train)),
            "test_fights": int(len(test)),
            "selected_config": config.key,
            "calibration_slope": slope,
            "training_validation_log_loss": float(scores[config.key]),
        }
    if not parts:
        raise ValueError("no eligible years for fixed-feature evaluation")
    return pd.concat(parts, ignore_index=True), folds


def _metrics_and_interval(
    aligned: pd.DataFrame,
    candidate_column: str,
    reference_column: str = "current_model_probability",
) -> dict[str, object]:
    return {
        "current_model": _metric(aligned, reference_column),
        "candidate": _metric(aligned, candidate_column),
        "candidate_minus_current_model": event_block_difference_interval(
            aligned, candidate_column, reference_column
        ),
    }


def _plain_evidence(interval: dict[str, object]) -> str:
    point = float(interval["point_difference"])
    lower = float(interval["ci_95_lower"])
    upper = float(interval["ci_95_upper"])
    if upper < 0.0:
        return "improved probability quality with a 95% interval entirely below zero"
    if lower > 0.0:
        return "made probability quality worse with a 95% interval entirely above zero"
    if point < 0.0:
        return "had a small average improvement, but the data also support no improvement"
    return "had a small average decline, but the data also support no difference"


def _per_year(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> dict[str, object]:
    return {
        str(int(year)): {
            name: _metric(rows, column) for name, column in columns
        }
        for year, rows in frame.groupby(pd.to_datetime(frame["date"]).dt.year, sort=True)
    }


def _market_report(
    market: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    if market.empty:
        return {
            "evaluable": False,
            "reason": "No odds fights had enough safely pre-event books and model predictions.",
        }, market
    reference_by_fight = reference.set_index("fight_id")
    paired = market.loc[market["fight_id"].isin(reference_by_fight.index)].copy()
    paired["model_probability"] = paired["fight_id"].map(
        reference_by_fight["model_probability"]
    )
    paired["fixed_equal_logit_blend_probability"] = [
        symmetric_logit_blend(market_probability, model_probability, 0.5)
        for market_probability, model_probability in paired[
            ["market_probability", "model_probability"]
        ].itertuples(index=False, name=None)
    ]
    rolling_input = paired.rename(columns={"date": "event_date"})
    rolling = evaluate_prior_card_blend(rolling_input)
    evaluated = rolling.loc[rolling["blend_status"].eq("evaluated")].copy()
    report: dict[str, object] = {
        "evaluable": True,
        "paired_fights": int(len(paired)),
        "paired_events": int(paired["event_id"].nunique()),
        "years": sorted(pd.to_datetime(paired["date"]).dt.year.unique().astype(int).tolist()),
        "market": _metric(paired, "market_probability"),
        "current_model": _metric(paired, "model_probability"),
        "fixed_equal_logit_blend": _metric(
            paired, "fixed_equal_logit_blend_probability"
        ),
        "fixed_blend_minus_market": event_block_difference_interval(
            paired,
            "fixed_equal_logit_blend_probability",
            "market_probability",
        ),
        "fixed_blend_minus_current_model": event_block_difference_interval(
            paired,
            "fixed_equal_logit_blend_probability",
            "model_probability",
        ),
        "rolling_blend_evaluated_fights": int(len(evaluated)),
        "rolling_blend": (
            _metric(evaluated, "blend_probability") if not evaluated.empty else None
        ),
        "interpretation_limit": (
            "This free archive only supports a narrow 2025 pre-event sample, so it "
            "cannot establish a historical production improvement by itself."
        ),
    }
    rolling = rolling.rename(columns={"event_date": "date"})
    return report, rolling


def build_evaluation(
    *,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTERS,
    rankings_path: Path = DEFAULT_RANKINGS,
    odds_path: Path = DEFAULT_ODDS,
    mma_database_path: Path = DEFAULT_MMA_DATABASE,
    years: tuple[int, ...] = DEFAULT_YEARS,
    max_runtime_minutes: float = 55.0,
    audit_only: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not 0 < max_runtime_minutes <= MAX_RUNTIME_MINUTES:
        raise ValueError("max runtime must be greater than zero and at most 60 minutes")
    paths = (
        point_in_time_path, raw_fights_path, fighter_stats_path,
        rankings_path, odds_path, mma_database_path,
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"research input files are missing: {missing}")
    started = time.monotonic()
    maximum_seconds = max_runtime_minutes * 60.0
    point = pd.read_csv(point_in_time_path, low_memory=False)
    raw = pd.read_csv(raw_fights_path, low_memory=False)
    fighters = pd.read_csv(fighter_stats_path, low_memory=False)
    rankings_source = pd.read_csv(rankings_path, low_memory=False)
    odds_source = pd.read_csv(odds_path, low_memory=False)

    rankings, rankings_audit = prepare_rankings(rankings_source, fighters, raw)
    ranked_pool, ranking_join_audit = add_ranking_features(point, rankings)
    market, odds_audit = prepare_pre_event_odds(odds_source, point)
    observations, mma_source_audit = load_mma_archive_observations(mma_database_path)
    auxiliary, identity_map, mma_join_audit = prepare_mma_auxiliary(observations, raw)
    source_audits = {
        "rankings": {**rankings_audit, **ranking_join_audit},
        "odds": odds_audit,
        "expanded_mma_history": {**mma_source_audit, **mma_join_audit},
    }
    if time.monotonic() - started > maximum_seconds:
        raise RuntimeError("online-data source validation exceeded its runtime limit")
    common = {
        "report_schema_version": 1,
        "experiment_version": "free-online-data-challengers-v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "promotable": False,
        "production_action": "none",
        "plain_language_method": (
            "For each tested calendar year, train on earlier fights only. Ranking "
            "feature groups are chosen without seeing that test year. External fight "
            "history never supplies UFC labels or detailed statistics. Odds count only "
            "when their recorded collection day is before the event day."
        ),
        "source_audits": source_audits,
        "source_sha256": {
            "point_in_time": file_sha256(point_in_time_path),
            "raw_fights": file_sha256(raw_fights_path),
            "fighter_stats": file_sha256(fighter_stats_path),
            "rankings": file_sha256(rankings_path),
            "odds": file_sha256(odds_path),
            "expanded_mma_history": file_sha256(mma_database_path),
        },
        "identity_map_rows": int(len(identity_map)),
        "elapsed_seconds": time.monotonic() - started,
    }
    if audit_only:
        return {**common, "audit_only": True}, pd.DataFrame()

    baseline_builder = PointInTimeDatasetBuilder(raw, fighters)
    reference = TemporalFightPredictor(point, baseline_builder).walk_forward_predictions(years)
    selected_rankings, all_rankings, ranking_folds = _ranking_walk_forward(
        ranked_pool,
        baseline_builder.feature_columns,
        years,
        started,
        maximum_seconds,
    )
    aligned = reference.rename(
        columns={"model_probability": "current_model_probability"}
    )
    aligned = _align_predictions(
        aligned,
        selected_rankings,
        probability_name="selected_rankings_probability",
    )
    aligned = _align_predictions(
        aligned,
        all_rankings,
        probability_name="all_rankings_probability",
    )

    if time.monotonic() - started > maximum_seconds:
        raise RuntimeError("online-data experiment exceeded its 60-minute limit")
    enriched_builder = PointInTimeDatasetBuilder(raw, fighters, auxiliary_fights=auxiliary)
    enriched_point = enriched_builder.build()
    if enriched_point["fight_id"].astype(str).tolist() != point["fight_id"].astype(str).tolist():
        raise RuntimeError("expanded history changed the UFC fight labels or order")
    expanded = TemporalFightPredictor(
        enriched_point, enriched_builder
    ).walk_forward_predictions(years)
    aligned = _align_predictions(
        aligned,
        expanded,
        probability_name="expanded_history_probability",
    )
    baseline_values = point[list(baseline_builder.feature_columns)].to_numpy(dtype=float)
    enriched_values = enriched_point[list(baseline_builder.feature_columns)].to_numpy(dtype=float)
    expanded_changed = ~np.isclose(
        baseline_values, enriched_values, rtol=0.0, atol=1e-12, equal_nan=True
    ).all(axis=1)
    changed_fight_ids = set(point.loc[expanded_changed, "fight_id"].astype(str))
    aligned["expanded_history_features_changed"] = aligned["fight_id"].astype(str).isin(
        changed_fight_ids
    )
    formerly_one_sided = point["has_history_diff"].abs().eq(1.0)
    newly_two_sided = formerly_one_sided & enriched_point["has_history_diff"].eq(0.0)

    # Ranking values depend only on date and stable fighter IDs. The enriched
    # frame has the exact same UFC rows, so reuse the already validated join.
    expanded_ranked_pool = enriched_point.copy()
    ranking_join_columns = [*RANKING_FEATURES, "ranking_snapshot_date"]
    expanded_ranked_pool[ranking_join_columns] = ranked_pool[
        ranking_join_columns
    ].to_numpy()
    expanded_ranking_audit = ranking_join_audit
    combined, combined_folds = _fixed_feature_walk_forward(
        expanded_ranked_pool,
        [*enriched_builder.feature_columns, *RANKING_FEATURES],
        years,
        started,
        maximum_seconds,
    )
    aligned = _align_predictions(
        aligned,
        combined,
        probability_name="expanded_history_all_rankings_probability",
    )
    market_results, market_detail = _market_report(market, reference)
    if not market_detail.empty:
        market_columns = [
            "fight_id", "market_probability", "book_count",
            "fixed_equal_logit_blend_probability", "blend_probability",
            "blend_status", "selected_gamma",
        ]
        available = [item for item in market_columns if item in market_detail]
        aligned = aligned.merge(
            market_detail[available], on="fight_id", how="left", validate="one_to_one"
        )

    ranking_results = {
        "selected_groups": _metrics_and_interval(
            aligned, "selected_rankings_probability"
        ),
        "all_ranking_features": _metrics_and_interval(
            aligned, "all_rankings_probability"
        ),
        "per_year": _per_year(
            aligned,
            (
                ("current_model", "current_model_probability"),
                ("selected_groups", "selected_rankings_probability"),
                ("all_features", "all_rankings_probability"),
            ),
        ),
        "selection_by_test_year": ranking_folds,
    }
    expanded_results = {
        **_metrics_and_interval(aligned, "expanded_history_probability"),
        "ufc_fights_with_any_feature_changed": int(expanded_changed.sum()),
        "one_sided_histories_made_two_sided": int(newly_two_sided.sum()),
        "per_year": _per_year(
            aligned,
            (
                ("current_model", "current_model_probability"),
                ("expanded_history", "expanded_history_probability"),
            ),
        ),
    }
    changed_rows = aligned.loc[aligned["expanded_history_features_changed"]].copy()
    if not changed_rows.empty:
        expanded_results["fights_whose_features_changed"] = _metrics_and_interval(
            changed_rows, "expanded_history_probability"
        )
    combined_results = {
        **_metrics_and_interval(
            aligned, "expanded_history_all_rankings_probability"
        ),
        "per_year": _per_year(
            aligned,
            (
                ("current_model", "current_model_probability"),
                (
                    "expanded_history_all_rankings",
                    "expanded_history_all_rankings_probability",
                ),
            ),
        ),
        "training_by_test_year": combined_folds,
        "ranking_join": expanded_ranking_audit,
    }
    selected_ranking_interval = ranking_results["selected_groups"][
        "candidate_minus_current_model"
    ]
    all_ranking_interval = ranking_results["all_ranking_features"][
        "candidate_minus_current_model"
    ]
    expanded_interval = expanded_results["candidate_minus_current_model"]
    combined_interval = combined_results["candidate_minus_current_model"]
    report = {
        **common,
        "audit_only": False,
        "sample": {
            "test_years": list(years),
            "tested_fights": int(len(aligned)),
            "tested_events": int(aligned["event_id"].nunique()),
        },
        "ranking_features": {
            "families": {key: list(value) for key, value in RANKING_FAMILIES.items()},
            "results": ranking_results,
        },
        "expanded_history": {"results": expanded_results},
        "expanded_history_plus_all_rankings": {"results": combined_results},
        "market": {"results": market_results},
        "decision": {
            "training_selected_ranking_groups": _plain_evidence(
                selected_ranking_interval
            ),
            "predeclared_all_ranking_features": _plain_evidence(
                all_ranking_interval
            ),
            "expanded_history": _plain_evidence(expanded_interval),
            "expanded_history_plus_all_rankings": _plain_evidence(
                combined_interval
            ),
            "market_note": (
                "The market and fixed model-market blend look promising on only five "
                "events; this is much too small for a production decision."
            ),
            "production_action": "none; these sources remain research-only",
        },
        "non_promotable_reasons": [
            "the same evaluation years were used to compare source ideas",
            "ranking identity uses unique normalized names rather than provider fighter IDs",
            "the expanded archive internally links bout names to its master IDs",
            "the free odds archive has only a narrow truly pre-event collection window",
            "a later untouched period is required before any production change",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    return report, aligned


def _parse_years(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one year is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-in-time", type=Path, default=DEFAULT_POINT_IN_TIME)
    parser.add_argument("--raw-fights", type=Path, default=DEFAULT_RAW_FIGHTS)
    parser.add_argument("--fighter-stats", type=Path, default=DEFAULT_FIGHTERS)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--odds", type=Path, default=DEFAULT_ODDS)
    parser.add_argument("--mma-database", type=Path, default=DEFAULT_MMA_DATABASE)
    parser.add_argument("--years", type=_parse_years, default=DEFAULT_YEARS)
    parser.add_argument("--max-runtime-minutes", type=float, default=55.0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report, detail = build_evaluation(
        point_in_time_path=arguments.point_in_time,
        raw_fights_path=arguments.raw_fights,
        fighter_stats_path=arguments.fighter_stats,
        rankings_path=arguments.rankings,
        odds_path=arguments.odds,
        mma_database_path=arguments.mma_database,
        years=arguments.years,
        max_runtime_minutes=arguments.max_runtime_minutes,
        audit_only=arguments.audit_only,
    )
    if not arguments.dry_run:
        atomic_write_text(
            arguments.report,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        if not detail.empty:
            atomic_write_text(
                arguments.detail,
                detail.to_csv(index=False, lineterminator="\n", float_format="%.15g"),
            )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
