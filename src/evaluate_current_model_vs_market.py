"""Replay the current winner-model algorithm against historical market consensus.

This is a retrospective, development-only comparison. Model probabilities are
strict whole-year out-of-fold forecasts: every test year is trained, tuned, and
calibrated only on earlier fights. The historical odds remain subject to the
legacy Git-timestamp limitations documented by ``backfill_market_history.py``.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Iterable, Sequence

import pandas as pd

from fight_predictor import PointInTimeDatasetBuilder, TemporalFightPredictor
from market_tracker import (
    DEFAULT_GAMMA_GRID,
    QuoteSnapshot,
    QuoteSnapshotStore,
    consensus_as_of,
    forecast_metrics,
    symmetric_logit_blend,
)


REPORT_SCHEMA_VERSION = 1
ALGORITHM_VERSION = 1
MINIMUM_CONSENSUS_BOOKS = 3
MINIMUM_PRIOR_CARDS = 12
MINIMUM_PRIOR_FIGHTS = 100
LOOKBACK_CARDS = 52
BOOTSTRAP_SEED = 20260822
BOOTSTRAP_REPLICATES = 10_000

ROOT = Path(__file__).resolve().parent
DEFAULT_POINT_IN_TIME = ROOT / "content/data/processed/ufc_fights_point_in_time.csv"
DEFAULT_RAW_FIGHTS = ROOT / "content/data/processed/ufc_fights_reported_doubled.csv"
DEFAULT_FIGHTER_STATS = ROOT / "content/data/processed/fighter_stats.csv"
DEFAULT_MODEL_ARTIFACT = ROOT / "content/data/external/winner_model.json"
DEFAULT_QUOTE_CSV = ROOT / "content/data/market_history_backfill/market_quotes.csv"
DEFAULT_QUOTE_JSONL = ROOT / "content/data/market_history_backfill/market_quotes.jsonl"
DEFAULT_REPORT = (
    ROOT / "content/data/market_history_backfill/current_model_market_replay.json"
)
DEFAULT_DETAIL = (
    ROOT / "content/data/market_history_backfill/current_model_market_replay.csv"
)

NON_PROMOTABLE_FLAGS = (
    "retrospective_current_algorithm_not_historical_frozen_forecast",
    "legacy_commit_timestamp_not_source_quote_timestamp",
    "current_reconciled_raw_not_as_of_snapshot",
    "profile_and_source_corrections_may_postdate_fight",
    "feature_contract_was_developed_using_some_evaluation_years",
    "missing_2024_market_history",
    "unverified_execution_and_closing_price",
    "development_only",
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


def _loss(probability: float, target: int) -> float:
    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return -(target * math.log(clipped) + (1 - target) * math.log1p(-clipped))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _metric_mapping(frame: pd.DataFrame, probability_column: str) -> dict[str, object]:
    return forecast_metrics(
        frame[probability_column].astype(float).tolist(),
        frame["target"].astype(int).tolist(),
    ).to_mapping()


def latest_eligible_consensus(
    quotes: Iterable[QuoteSnapshot],
    *,
    minimum_books: int = MINIMUM_CONSENSUS_BOOKS,
) -> tuple[object, ...]:
    """Select the latest pre-event capture with enough books for each fight."""

    grouped: dict[tuple[str, str, str], list[QuoteSnapshot]] = {}
    for quote in quotes:
        if quote.fight_id is None:
            continue
        grouped.setdefault(
            (quote.fight_id, quote.capture_id, quote.matchup_id), []
        ).append(quote)

    candidates: dict[str, list[object]] = {}
    for (fight_id, capture_id, matchup_id), capture_quotes in grouped.items():
        distinct_books = {item.book.casefold() for item in capture_quotes}
        if len(distinct_books) < minimum_books:
            continue
        observed_times = {item.observed_at_utc for item in capture_quotes}
        if len(observed_times) != 1:
            raise RuntimeError("one historical capture has multiple observation times")
        market = consensus_as_of(
            capture_quotes,
            capture_id=capture_id,
            matchup_id=matchup_id,
            as_of_utc=next(iter(observed_times)),
            min_books=minimum_books,
        )
        candidates.setdefault(fight_id, []).append(market)

    selected = [
        max(
            markets,
            key=lambda item: (item.as_of_utc, item.capture_id, item.matchup_id),
        )
        for markets in candidates.values()
    ]
    return tuple(
        sorted(selected, key=lambda item: (item.event_date, item.event_id, item.fight_id))
    )


def _mean_card_log_loss(cards: Sequence[pd.DataFrame], gamma: float) -> float:
    card_losses: list[float] = []
    for card in cards:
        probabilities = [
            symmetric_logit_blend(market, model, gamma)
            for market, model in card[
                ["market_probability", "model_probability"]
            ].itertuples(index=False, name=None)
        ]
        losses = [
            _loss(probability, int(target))
            for probability, target in zip(probabilities, card["target"])
        ]
        card_losses.append(sum(losses) / len(losses))
    return sum(card_losses) / len(card_losses)


def evaluate_prior_card_blend(
    paired: pd.DataFrame,
    *,
    gamma_grid: Sequence[float] = DEFAULT_GAMMA_GRID,
    minimum_prior_cards: int = MINIMUM_PRIOR_CARDS,
    minimum_prior_fights: int = MINIMUM_PRIOR_FIGHTS,
    lookback_cards: int | None = LOOKBACK_CARDS,
) -> pd.DataFrame:
    """Select a blend weight using completed cards on earlier dates only."""

    required = {
        "event_date",
        "event_id",
        "fight_id",
        "market_probability",
        "model_probability",
        "target",
    }
    missing = required - set(paired.columns)
    if missing:
        raise ValueError(f"paired replay is missing columns: {sorted(missing)}")
    if paired.empty:
        raise ValueError("paired replay must not be empty")
    if paired["fight_id"].duplicated().any():
        raise ValueError("paired replay contains duplicate fight IDs")

    result = paired.copy().sort_values(
        ["event_date", "event_id", "fight_id"], kind="stable"
    )
    result["blend_status"] = "insufficient_prior_history"
    result["prior_card_count"] = 0
    result["prior_fight_count"] = 0
    result["blend_training_through_event_date"] = ""
    result["selected_gamma"] = math.nan
    result["selection_prior_card_log_loss"] = math.nan
    result["blend_probability"] = math.nan

    cards_by_date: dict[str, list[tuple[str, pd.Index]]] = {}
    for (event_date, event_id), indices in result.groupby(
        ["event_date", "event_id"], sort=True
    ).groups.items():
        cards_by_date.setdefault(str(event_date), []).append((str(event_id), indices))

    completed: list[tuple[str, str, pd.DataFrame]] = []
    parsed_grid = tuple(sorted(set(float(value) for value in gamma_grid)))
    if not parsed_grid or any(not 0.0 <= value <= 1.0 for value in parsed_grid):
        raise ValueError("gamma_grid must contain values between zero and one")

    for event_date in sorted(cards_by_date):
        available = completed[-lookback_cards:] if lookback_cards is not None else completed
        prior_cards = [card for _, _, card in available]
        prior_fights = sum(len(card) for card in prior_cards)
        enough = (
            len(prior_cards) >= minimum_prior_cards
            and prior_fights >= minimum_prior_fights
        )
        selected_gamma = math.nan
        selected_loss = math.nan
        if enough:
            selected_loss, selected_gamma = min(
                (_mean_card_log_loss(prior_cards, gamma), gamma)
                for gamma in parsed_grid
            )
        training_through = available[-1][0] if available else ""
        for _event_id, indices in sorted(cards_by_date[event_date]):
            result.loc[indices, "prior_card_count"] = len(prior_cards)
            result.loc[indices, "prior_fight_count"] = prior_fights
            result.loc[indices, "blend_training_through_event_date"] = training_through
            if enough:
                result.loc[indices, "blend_status"] = "evaluated"
                result.loc[indices, "selected_gamma"] = selected_gamma
                result.loc[indices, "selection_prior_card_log_loss"] = selected_loss
                result.loc[indices, "blend_probability"] = [
                    symmetric_logit_blend(market, model, selected_gamma)
                    for market, model in result.loc[
                        indices, ["market_probability", "model_probability"]
                    ].itertuples(index=False, name=None)
                ]
        # Events on the same UTC date become available together only after all
        # of that date's predictions have been assigned.
        for event_id, indices in sorted(cards_by_date[event_date]):
            completed.append((event_date, event_id, result.loc[indices].copy()))
    return result.reset_index(drop=True)


def event_block_log_loss_interval(
    frame: pd.DataFrame,
    candidate_column: str,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    """Bootstrap candidate-minus-market log loss by whole event cards."""

    if frame.empty:
        raise ValueError("cannot bootstrap an empty replay")
    blocks: dict[str, list[float]] = {}
    for row in frame.to_dict("records"):
        delta = _loss(float(row[candidate_column]), int(row["target"])) - _loss(
            float(row["market_probability"]), int(row["target"])
        )
        blocks.setdefault(str(row["event_id"]), []).append(delta)
    block_ids = sorted(blocks)
    all_deltas = [value for values in blocks.values() for value in values]
    point = sum(all_deltas) / len(all_deltas)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [rng.choice(block_ids) for _ in block_ids]
        deltas = [value for block_id in selected for value in blocks[block_id]]
        samples.append(sum(deltas) / len(deltas))
    return {
        "definition": "candidate minus market paired log loss; negative favors candidate",
        "method": "nonparametric whole-event block bootstrap",
        "seed": seed,
        "bootstrap_samples": replicates,
        "event_count": len(block_ids),
        "fight_count": len(frame),
        "point_difference": point,
        "ci_95_lower": _quantile(samples, 0.025),
        "ci_95_upper": _quantile(samples, 0.975),
    }


def _per_year_metrics(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for year, rows in frame.groupby(frame["event_date"].str[:4], sort=True):
        result[str(year)] = {
            "fights": len(rows),
            "market": _metric_mapping(rows, "market_probability"),
            "model": _metric_mapping(rows, "model_probability"),
        }
        evaluated = rows[rows["blend_status"] == "evaluated"]
        result[str(year)]["blend"] = (
            _metric_mapping(evaluated, "blend_probability")
            if not evaluated.empty
            else None
        )
    return result


def _gamma_distribution(frame: pd.DataFrame) -> dict[str, object]:
    evaluated = frame[frame["blend_status"] == "evaluated"]
    fight_counts = (
        evaluated["selected_gamma"].map(lambda value: f"{value:.2f}").value_counts()
    )
    cards = evaluated.drop_duplicates(["event_date", "event_id"])
    card_counts = cards["selected_gamma"].map(lambda value: f"{value:.2f}").value_counts()
    return {
        "evaluated_fights_by_gamma": {
            key: int(value) for key, value in sorted(fight_counts.items())
        },
        "evaluated_cards_by_gamma": {
            key: int(value) for key, value in sorted(card_counts.items())
        },
    }


def _fold_contract(predictions: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for year, rows in predictions.groupby("evaluation_year", sort=True):
        contract_columns = [
            "training_start",
            "training_through",
            "selected_c",
            "calibration_slope",
        ]
        unique = rows[contract_columns].drop_duplicates()
        if len(unique) != 1:
            raise RuntimeError(f"year {year} has multiple model fold contracts")
        contract = unique.iloc[0]
        result[str(int(year))] = {
            "training_start": str(contract["training_start"]),
            "training_through": str(contract["training_through"]),
            "selected_c": float(contract["selected_c"]),
            "calibration_slope": float(contract["calibration_slope"]),
            "predicted_fights": int(len(rows)),
        }
    return result


def build_replay(
    *,
    point_in_time_path: Path = DEFAULT_POINT_IN_TIME,
    raw_fights_path: Path = DEFAULT_RAW_FIGHTS,
    fighter_stats_path: Path = DEFAULT_FIGHTER_STATS,
    model_artifact_path: Path = DEFAULT_MODEL_ARTIFACT,
    quote_csv_path: Path = DEFAULT_QUOTE_CSV,
    quote_jsonl_path: Path = DEFAULT_QUOTE_JSONL,
) -> tuple[dict[str, object], pd.DataFrame]:
    for path in (
        point_in_time_path,
        raw_fights_path,
        fighter_stats_path,
        model_artifact_path,
        quote_csv_path,
        quote_jsonl_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    point_in_time = pd.read_csv(point_in_time_path, low_memory=False)
    raw_fights = pd.read_csv(raw_fights_path, low_memory=False)
    fighter_stats = pd.read_csv(fighter_stats_path, low_memory=False)
    artifact = json.loads(model_artifact_path.read_text(encoding="utf-8"))
    builder = PointInTimeDatasetBuilder(raw_fights, fighter_stats)
    if list(builder.feature_columns) != list(artifact["feature_columns"]):
        raise RuntimeError("current artifact and point-in-time builder feature contracts differ")
    predictor = TemporalFightPredictor(point_in_time, builder)

    quote_store = QuoteSnapshotStore(quote_csv_path, quote_jsonl_path)
    quotes = quote_store.read()
    markets = latest_eligible_consensus(quotes)
    market_years = tuple(sorted({int(item.event_date[:4]) for item in markets}))
    predictions = predictor.walk_forward_predictions(market_years)
    if predictions["fight_id"].duplicated().any():
        raise RuntimeError("walk-forward predictions contain duplicate fight IDs")
    prediction_by_fight = {
        str(row["fight_id"]): row for row in predictions.to_dict("records")
    }

    paired_rows: list[dict[str, object]] = []
    missing_prediction = 0
    for market in markets:
        prediction = prediction_by_fight.get(str(market.fight_id))
        if prediction is None:
            missing_prediction += 1
            continue
        if str(prediction["event_id"]) != market.event_id:
            raise RuntimeError(f"event ID mismatch for fight {market.fight_id}")
        prediction_date = pd.Timestamp(prediction["date"]).strftime("%Y-%m-%d")
        if prediction_date != market.event_date:
            raise RuntimeError(f"event date mismatch for fight {market.fight_id}")
        if (
            market.fighter_id == str(prediction["fighter_id"])
            and market.opponent_id == str(prediction["opponent_id"])
        ):
            model_probability = float(prediction["model_probability"])
            target = int(prediction["target"])
            fighter_name = str(prediction["fighter"])
            opponent_name = str(prediction["opponent"])
        elif (
            market.fighter_id == str(prediction["opponent_id"])
            and market.opponent_id == str(prediction["fighter_id"])
        ):
            model_probability = 1.0 - float(prediction["model_probability"])
            target = 1 - int(prediction["target"])
            fighter_name = str(prediction["opponent"])
            opponent_name = str(prediction["fighter"])
        else:
            raise RuntimeError(f"fighter identity mismatch for fight {market.fight_id}")
        paired_rows.append(
            {
                "event_date": market.event_date,
                "event_id": market.event_id,
                "fight_id": market.fight_id,
                "matchup_id": market.matchup_id,
                "capture_id": market.capture_id,
                "market_consensus_id": market.consensus_id,
                "market_as_of_utc": market.as_of_utc,
                "market_book_count": int(market.book_count),
                "fighter_id": market.fighter_id,
                "opponent_id": market.opponent_id,
                "fighter_name": fighter_name,
                "opponent_name": opponent_name,
                "target": target,
                "market_probability": float(market.no_vig_fighter_probability),
                "model_probability": model_probability,
                "evaluation_year": int(prediction["evaluation_year"]),
                "model_training_start": str(prediction["training_start"]),
                "model_training_through": str(prediction["training_through"]),
                "model_selected_c": float(prediction["selected_c"]),
                "model_calibration_slope": float(prediction["calibration_slope"]),
            }
        )
    if not paired_rows:
        raise RuntimeError("no market snapshots matched out-of-fold model predictions")
    paired = pd.DataFrame(paired_rows)
    if any(
        training_through >= event_date
        for training_through, event_date in paired[
            ["model_training_through", "event_date"]
        ].itertuples(index=False, name=None)
    ):
        raise RuntimeError("an out-of-fold model training cutoff reaches its test event")
    paired = evaluate_prior_card_blend(paired)
    paired["market_log_loss"] = [
        _loss(probability, int(target))
        for probability, target in paired[
            ["market_probability", "target"]
        ].itertuples(index=False, name=None)
    ]
    paired["model_log_loss"] = [
        _loss(probability, int(target))
        for probability, target in paired[
            ["model_probability", "target"]
        ].itertuples(index=False, name=None)
    ]
    paired["model_minus_market_log_loss"] = (
        paired["model_log_loss"] - paired["market_log_loss"]
    )
    paired["blend_log_loss"] = [
        (
            _loss(float(probability), int(target))
            if status == "evaluated"
            else math.nan
        )
        for probability, target, status in paired[
            ["blend_probability", "target", "blend_status"]
        ].itertuples(index=False, name=None)
    ]
    paired["blend_minus_market_log_loss"] = (
        paired["blend_log_loss"] - paired["market_log_loss"]
    )

    evaluated = paired[paired["blend_status"] == "evaluated"].copy()
    detail_csv = paired.to_csv(index=False, lineterminator="\n", float_format="%.15g")
    report: dict[str, object] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "paper_only": True,
        "promotable": False,
        "non_promotable_flags": list(NON_PROMOTABLE_FLAGS),
        "model_contract": {
            "model_version": artifact["model_version"],
            "production_artifact_id": artifact["model_id"],
            "feature_count": len(artifact["feature_columns"]),
            "regularization_grid": artifact["regularization_c_grid"],
            "probability_provenance": (
                "retrospective whole-year out-of-fold current algorithm; not the "
                "fully fitted production artifact applied backwards"
            ),
            "folds": _fold_contract(predictions),
        },
        "market_contract": {
            "selection": "latest eligible pre-event capture per stable fight ID",
            "minimum_distinct_books": MINIMUM_CONSENSUS_BOOKS,
            "probability": "mean of per-book two-sided no-vig probabilities",
        },
        "sample": {
            "historical_quotes": len(quotes),
            "latest_eligible_market_fights_including_non_w_l": len(markets),
            "paired_w_l_fights": len(paired),
            "paired_events": int(paired["event_id"].nunique()),
            "market_fights_without_binary_out_of_fold_prediction": missing_prediction,
            "years": sorted(paired["evaluation_year"].unique().astype(int).tolist()),
        },
        "paired_all_fights": {
            "market": _metric_mapping(paired, "market_probability"),
            "current_model_algorithm": _metric_mapping(paired, "model_probability"),
            "model_minus_market_log_loss_interval": event_block_log_loss_interval(
                paired, "model_probability"
            ),
        },
        "prior_card_selected_blend": {
            "selection_rule": "completed cards on strictly earlier event dates only",
            "gamma_grid": list(DEFAULT_GAMMA_GRID),
            "minimum_prior_cards": MINIMUM_PRIOR_CARDS,
            "minimum_prior_fights": MINIMUM_PRIOR_FIGHTS,
            "lookback_cards": LOOKBACK_CARDS,
            "evaluated_fights": len(evaluated),
            "skipped_warmup_fights": len(paired) - len(evaluated),
            "market": _metric_mapping(evaluated, "market_probability"),
            "current_model_algorithm": _metric_mapping(
                evaluated, "model_probability"
            ),
            "blend": _metric_mapping(evaluated, "blend_probability"),
            "blend_minus_market_log_loss_interval": event_block_log_loss_interval(
                evaluated, "blend_probability"
            ),
            "selected_gamma_distribution": _gamma_distribution(evaluated),
        },
        "per_year": _per_year_metrics(paired),
        "source_dataset_sha256": {
            "point_in_time": _file_sha256(point_in_time_path),
            "model_artifact": _file_sha256(model_artifact_path),
            "market_quote_csv": _file_sha256(quote_csv_path),
            "market_quote_jsonl": _file_sha256(quote_jsonl_path),
            "detail_csv": sha256(detail_csv.encode("utf-8")).hexdigest(),
        },
    }
    report["input_contract_sha256"] = _canonical_hash(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "model_contract": report["model_contract"],
            "market_contract": report["market_contract"],
            "source_dataset_sha256": report["source_dataset_sha256"],
        }
    )
    return report, paired


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report, detail = build_replay()
    if not arguments.dry_run:
        detail_text = detail.to_csv(
            index=False, lineterminator="\n", float_format="%.15g"
        )
        report_text = json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        _atomic_write_text(arguments.detail, detail_text)
        _atomic_write_text(arguments.report, report_text)
    all_fights = report["paired_all_fights"]
    blend = report["prior_card_selected_blend"]
    print(
        "Current algorithm versus historical market: "
        f"{report['sample']['paired_w_l_fights']} paired fights, "
        f"market={all_fights['market']['log_loss']:.5f}, "
        f"model={all_fights['current_model_algorithm']['log_loss']:.5f}; "
        f"evaluated blend={blend['blend']['log_loss']:.5f} versus "
        f"market={blend['market']['log_loss']:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
