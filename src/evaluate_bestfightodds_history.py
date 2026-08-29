"""Evaluate the current winner model against timestamped BestFightOdds history.

The database is opened read-only and one SQLite read transaction supplies the
entire evaluation snapshot, so this command can run while the resumable
collector is active.  Nothing here changes production predictions or betting
behavior.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

import pandas as pd

from backfill_bestfightodds_history import (
    database_summary,
    default_database_path,
    derive_consensus_rows,
    derive_horizon_rows,
    open_database_readonly,
)
from evaluate_current_model_vs_market import evaluate_prior_card_blend
from evaluate_style_matchup_challenger import event_block_difference_interval
from fight_predictor import PointInTimeDatasetBuilder, TemporalFightPredictor
from market_tracker import forecast_metrics, symmetric_logit_blend
from market_tracker._common import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src/content/data"
DEFAULT_POINT_IN_TIME = DATA / "processed/ufc_fights_point_in_time.csv"
DEFAULT_RAW_FIGHTS = DATA / "processed/ufc_fights_reported_doubled.csv"
DEFAULT_FIGHTERS = DATA / "processed/fighter_stats.csv"
DEFAULT_MODEL_ARTIFACT = DATA / "external/winner_model.json"
DEFAULT_DATABASE = default_database_path()
DEFAULT_ANALYSIS_DIRECTORY = DEFAULT_DATABASE.parent / "analysis"
DEFAULT_REPORT = DEFAULT_ANALYSIS_DIRECTORY / "current_model_market_evaluation.json"
DEFAULT_DETAIL = DEFAULT_ANALYSIS_DIRECTORY / "current_model_market_evaluation.csv"
HORIZON_ORDER = (
    "opening",
    "safe_t72",
    "safe_t24",
    "safe_t6",
    "strict_latest_before_event_date",
)
REPORT_SCHEMA_VERSION = 1
MAX_RUNTIME_MINUTES = 60.0


def _metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, object]:
    return forecast_metrics(
        frame[probability_column].astype(float).tolist(),
        frame["target"].astype(int).tolist(),
    ).to_mapping()


def _validate_consensus(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    required = {
        "ufc_event_date",
        "ufc_event_id",
        "ufc_fight_id",
        "ufc_fighter_1_id",
        "ufc_fighter_2_id",
        "fighter_1_name",
        "fighter_2_name",
        "horizon",
        "cutoff_basis",
        "actual_event_start_time_known",
        "book_count",
        "fighter_1_market_probability",
        "minimum_book_probability",
        "maximum_book_probability",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"market consensus is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("market database has no three-book consensus rows")
    if frame.duplicated(["ufc_fight_id", "horizon"]).any():
        raise ValueError("market consensus contains duplicate fight/horizon rows")
    unknown_horizons = set(frame["horizon"].astype(str)) - set(HORIZON_ORDER)
    if unknown_horizons:
        raise ValueError(f"market consensus has unknown horizons: {sorted(unknown_horizons)}")
    if not frame["cutoff_basis"].eq("source_event_calendar_date_at_00_utc").all():
        raise ValueError("market consensus has an unsupported cutoff basis")
    known_start = frame["actual_event_start_time_known"].map(
        lambda value: value is True or str(value).strip().casefold() == "true"
    )
    if known_start.any():
        raise ValueError("calendar-date horizons unexpectedly claim exact event times")
    numeric_columns = (
        "book_count",
        "fighter_1_market_probability",
        "minimum_book_probability",
        "maximum_book_probability",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["book_count"] < 3).any():
        raise ValueError("market consensus contains fewer than three books")
    probabilities = frame[
        [
            "fighter_1_market_probability",
            "minimum_book_probability",
            "maximum_book_probability",
        ]
    ]
    if not probabilities.map(lambda value: math.isfinite(float(value))).all().all():
        raise ValueError("market consensus contains a non-finite probability")
    if ((probabilities <= 0.0) | (probabilities >= 1.0)).any().any():
        raise ValueError("market consensus probabilities must be strictly within (0, 1)")
    tolerance = 1e-12
    if (
        (
            frame["minimum_book_probability"]
            - frame["fighter_1_market_probability"]
            > tolerance
        )
        | (
            frame["fighter_1_market_probability"]
            - frame["maximum_book_probability"]
            > tolerance
        )
    ).any():
        raise ValueError("market consensus lies outside its constituent book range")
    return frame


def pair_consensus_with_predictions(
    consensus_rows: Sequence[Mapping[str, object]],
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Align every market row to the model's fighter orientation by stable IDs."""

    market = _validate_consensus(consensus_rows)
    if predictions.empty or "fight_id" not in predictions:
        raise ValueError("walk-forward model predictions are empty")
    if predictions["fight_id"].astype(str).duplicated().any():
        raise ValueError("walk-forward model predictions contain duplicate fight IDs")
    by_fight = {
        str(row["fight_id"]): row for row in predictions.to_dict("records")
    }
    paired: list[dict[str, object]] = []
    missing_predictions = 0
    for row in market.to_dict("records"):
        fight_id = str(row["ufc_fight_id"])
        prediction = by_fight.get(fight_id)
        if prediction is None:
            missing_predictions += 1
            continue
        event_date = pd.Timestamp(prediction["date"]).strftime("%Y-%m-%d")
        if str(row["ufc_event_id"]) != str(prediction["event_id"]):
            raise ValueError(f"event ID mismatch for fight {fight_id}")
        if str(row["ufc_event_date"]) != event_date:
            raise ValueError(f"event date mismatch for fight {fight_id}")
        market_first = str(row["ufc_fighter_1_id"])
        market_second = str(row["ufc_fighter_2_id"])
        model_first = str(prediction["fighter_id"])
        model_second = str(prediction["opponent_id"])
        if (market_first, market_second) == (model_first, model_second):
            market_probability = float(row["fighter_1_market_probability"])
            minimum = float(row["minimum_book_probability"])
            maximum = float(row["maximum_book_probability"])
        elif (market_first, market_second) == (model_second, model_first):
            market_probability = 1.0 - float(row["fighter_1_market_probability"])
            minimum = 1.0 - float(row["maximum_book_probability"])
            maximum = 1.0 - float(row["minimum_book_probability"])
        else:
            raise ValueError(f"fighter identity mismatch for fight {fight_id}")
        training_through = str(prediction.get("training_through", ""))
        if training_through and training_through >= event_date:
            raise ValueError(f"model training reaches the event date for fight {fight_id}")
        paired.append(
            {
                "event_date": event_date,
                "event_id": str(prediction["event_id"]),
                "fight_id": fight_id,
                "fighter_id": model_first,
                "opponent_id": model_second,
                "fighter_name": str(prediction["fighter"]),
                "opponent_name": str(prediction["opponent"]),
                "target": int(prediction["target"]),
                "horizon": str(row["horizon"]),
                "book_count": int(row["book_count"]),
                "market_probability": market_probability,
                "minimum_book_probability": minimum,
                "maximum_book_probability": maximum,
                "book_probability_range": maximum - minimum,
                "model_probability": float(prediction["model_probability"]),
                "model_training_through": training_through,
                "model_selected_c": float(prediction.get("selected_c", math.nan)),
                "model_calibration_slope": float(
                    prediction.get("calibration_slope", math.nan)
                ),
            }
        )
    if not paired:
        raise ValueError("no consensus rows matched walk-forward model predictions")
    result = pd.DataFrame(paired).sort_values(
        ["event_date", "event_id", "fight_id", "horizon"], kind="stable"
    ).reset_index(drop=True)
    result["fixed_equal_logit_blend_probability"] = [
        symmetric_logit_blend(market_probability, model_probability, 0.5)
        for market_probability, model_probability in result[
            ["market_probability", "model_probability"]
        ].itertuples(index=False, name=None)
    ]
    return result, {
        "consensus_rows": int(len(market)),
        "paired_rows": int(len(result)),
        "missing_model_prediction_rows": missing_predictions,
        "paired_fights": int(result["fight_id"].nunique()),
        "paired_events": int(result["event_id"].nunique()),
    }


def _horizon_report(rows: pd.DataFrame) -> dict[str, object]:
    rolling = evaluate_prior_card_blend(rows)
    evaluated = rolling.loc[rolling["blend_status"].eq("evaluated")]
    evaluated_cards = evaluated.drop_duplicates(["event_date", "event_id"])
    fight_weights = evaluated["selected_gamma"].value_counts().sort_index()
    card_weights = evaluated_cards["selected_gamma"].value_counts().sort_index()
    return {
        "fights": int(len(rows)),
        "events": int(rows["event_id"].nunique()),
        "years": sorted(rows["event_date"].str[:4].astype(int).unique().tolist()),
        "average_books": float(rows["book_count"].mean()),
        "average_probability_range_across_books": float(
            rows["book_probability_range"].mean()
        ),
        "market": _metrics(rows, "market_probability"),
        "current_model": _metrics(rows, "model_probability"),
        "fixed_equal_logit_blend": _metrics(
            rows, "fixed_equal_logit_blend_probability"
        ),
        "model_minus_market_log_loss": event_block_difference_interval(
            rows, "model_probability", "market_probability"
        ),
        "fixed_blend_minus_market_log_loss": event_block_difference_interval(
            rows, "fixed_equal_logit_blend_probability", "market_probability"
        ),
        "fixed_blend_minus_model_log_loss": event_block_difference_interval(
            rows, "fixed_equal_logit_blend_probability", "model_probability"
        ),
        "earlier_cards_selected_blend": {
            "evaluated_fights": int(len(evaluated)),
            "model_weight_fight_counts": {
                f"{float(weight):.2f}": int(count)
                for weight, count in fight_weights.items()
            },
            "model_weight_card_counts": {
                f"{float(weight):.2f}": int(count)
                for weight, count in card_weights.items()
            },
            "weight_note": (
                "0.00 means market only; 1.00 means current model only. Each "
                "card's weight is selected using completed earlier cards only."
            ),
            "metrics": (
                _metrics(evaluated, "blend_probability")
                if not evaluated.empty
                else None
            ),
        },
    }


def evaluate_paired_snapshot(paired: pd.DataFrame) -> dict[str, object]:
    available_horizons = [
        horizon for horizon in HORIZON_ORDER if horizon in set(paired["horizon"])
    ]
    by_horizon = {
        horizon: _horizon_report(
            paired.loc[paired["horizon"].eq(horizon)].copy()
        )
        for horizon in available_horizons
    }
    fight_sets = [
        set(paired.loc[paired["horizon"].eq(horizon), "fight_id"])
        for horizon in available_horizons
    ]
    common_fights = set.intersection(*fight_sets) if fight_sets else set()
    common_results: dict[str, object] = {}
    if common_fights:
        for horizon in available_horizons:
            rows = paired.loc[
                paired["horizon"].eq(horizon)
                & paired["fight_id"].isin(common_fights)
            ]
            common_results[horizon] = {
                "market": _metrics(rows, "market_probability"),
                "current_model": _metrics(rows, "model_probability"),
                "fixed_equal_logit_blend": _metrics(
                    rows, "fixed_equal_logit_blend_probability"
                ),
            }
    movement: dict[str, object] = {}
    if "opening" in available_horizons:
        opening = paired.loc[paired["horizon"].eq("opening"), [
            "fight_id", "event_id", "target", "market_probability"
        ]].rename(columns={"market_probability": "opening_probability"})
        for horizon in available_horizons:
            if horizon == "opening":
                continue
            later = paired.loc[paired["horizon"].eq(horizon), [
                "fight_id", "event_id", "target", "market_probability"
            ]].rename(columns={"market_probability": "later_probability"})
            aligned = opening.merge(
                later,
                on=["fight_id", "event_id", "target"],
                how="inner",
                validate="one_to_one",
            )
            if not aligned.empty:
                movement[f"{horizon}_minus_opening"] = event_block_difference_interval(
                    aligned, "later_probability", "opening_probability"
                )
    return {
        "horizons": by_horizon,
        "same_fights_at_every_available_horizon": {
            "horizons": available_horizons,
            "fights": len(common_fights),
            "results": common_results,
        },
        "market_movement_from_opening": movement,
    }


def _read_database_snapshot(
    database_path: Path, *, mode: str, minimum_consensus_books: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    connection = open_database_readonly(database_path, mode=mode)
    try:
        connection.execute("BEGIN")
        horizons = derive_horizon_rows(connection)
        consensus = derive_consensus_rows(
            horizons, minimum_books=minimum_consensus_books
        )
        summary = database_summary(connection, database_path=database_path)
        connection.execute("ROLLBACK")
    finally:
        connection.close()
    return consensus, summary


def build_evaluation(
    *,
    database_path: Path = DEFAULT_DATABASE,
    mode: str = "both",
    minimum_consensus_books: int = 3,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTERS,
    model_artifact_path: Path = DEFAULT_MODEL_ARTIFACT,
    max_runtime_minutes: float = 55.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not 0 < max_runtime_minutes <= MAX_RUNTIME_MINUTES:
        raise ValueError("max runtime must be greater than zero and at most 60 minutes")
    for path in (
        database_path,
        point_in_time_path,
        raw_fights_path,
        fighter_stats_path,
        model_artifact_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.monotonic()
    consensus, database = _read_database_snapshot(
        database_path,
        mode=mode,
        minimum_consensus_books=minimum_consensus_books,
    )
    point = pd.read_csv(point_in_time_path, low_memory=False)
    raw = pd.read_csv(raw_fights_path, low_memory=False)
    fighters = pd.read_csv(fighter_stats_path, low_memory=False)
    artifact = json.loads(model_artifact_path.read_text(encoding="utf-8"))
    builder = PointInTimeDatasetBuilder(raw, fighters)
    if list(builder.feature_columns) != list(artifact["feature_columns"]):
        raise ValueError("current model artifact and feature builder do not agree")
    years = tuple(sorted({int(str(row["ufc_event_date"])[:4]) for row in consensus}))
    if time.monotonic() - started > max_runtime_minutes * 60.0:
        raise TimeoutError("evaluation exceeded its runtime limit before model fitting")
    predictions = TemporalFightPredictor(point, builder).walk_forward_predictions(years)
    paired, coverage = pair_consensus_with_predictions(consensus, predictions)
    evaluation = evaluate_paired_snapshot(paired)
    elapsed = time.monotonic() - started
    if elapsed > max_runtime_minutes * 60.0:
        raise TimeoutError("evaluation exceeded its runtime limit")
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_version": "bestfightodds-current-model-horizons-v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "production_action": "none",
        "plain_language_method": (
            "For each calendar year, the current model is trained and calibrated only "
            "on earlier fights. It is compared with three-or-more-book no-vig market "
            "prices captured before that fight. Fixed 50/50 blends average log odds."
        ),
        "interpretation_limits": [
            "this is retrospective development evidence, not a frozen historical production forecast",
            "UFCStats corrections made after a fight may exist in the current source files",
            "historical event start times are unavailable, so cutoffs use midnight UTC on the event date",
            "the same historical sample must not be used both to invent and to confirm a new rule",
        ],
        "database_snapshot": database,
        "minimum_consensus_books": minimum_consensus_books,
        "model_evaluation_years": list(years),
        "coverage": coverage,
        "consensus_snapshot_sha256": canonical_hash(consensus),
        "evaluation": evaluation,
        "elapsed_seconds": elapsed,
    }
    return report, paired


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--mode", choices=("mean", "books", "both"), default="both")
    parser.add_argument("--minimum-consensus-books", type=int, default=3)
    parser.add_argument("--point-in-time", type=Path, default=DEFAULT_POINT_IN_TIME)
    parser.add_argument("--raw-fights", type=Path, default=DEFAULT_RAW_FIGHTS)
    parser.add_argument("--fighter-stats", type=Path, default=DEFAULT_FIGHTERS)
    parser.add_argument("--model-artifact", type=Path, default=DEFAULT_MODEL_ARTIFACT)
    parser.add_argument("--max-runtime-minutes", type=float, default=55.0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report, detail = build_evaluation(
        database_path=arguments.database,
        mode=arguments.mode,
        minimum_consensus_books=arguments.minimum_consensus_books,
        point_in_time_path=arguments.point_in_time,
        raw_fights_path=arguments.raw_fights,
        fighter_stats_path=arguments.fighter_stats,
        model_artifact_path=arguments.model_artifact,
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
    best = report["evaluation"]["horizons"]
    print(
        f"Evaluated {report['coverage']['paired_fights']} fights across "
        f"{report['coverage']['paired_events']} events and {len(best)} horizons."
    )
    for horizon, result in best.items():
        print(
            f"{horizon}: fights={result['fights']}, "
            f"market={result['market']['log_loss']:.5f}, "
            f"model={result['current_model']['log_loss']:.5f}, "
            f"50/50={result['fixed_equal_logit_blend']['log_loss']:.5f}"
        )
    if not arguments.dry_run:
        print(f"Report: {arguments.report}")
        print(f"Fight detail: {arguments.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
