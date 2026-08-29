"""Test whether a few pre-fight signals improve a sportsbook consensus.

This is a paper-only retrospective experiment.  The earliest 60% of event
dates fit each candidate, the next 20% choose one candidate, and the latest
20% are left untouched until the final score.  No result changes production
predictions or betting behavior.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from evaluate_bestfightodds_history import (
    DEFAULT_DATABASE,
    DEFAULT_RAW_FIGHTS,
    HORIZON_ORDER,
    build_evaluation as build_market_snapshot,
)
from evaluate_style_matchup_challenger import event_block_difference_interval
from market_tracker import forecast_metrics
from market_tracker._common import canonical_hash


REPORT_SCHEMA_VERSION = 1
EXPERIMENT_VERSION = "market-first-four-signal-factorial-v1"
DEFAULT_ANALYSIS_DIRECTORY = DEFAULT_DATABASE.parent / "analysis"
DEFAULT_REPORT = DEFAULT_ANALYSIS_DIRECTORY / "market_first_challenger.json"
DEFAULT_DETAIL = DEFAULT_ANALYSIS_DIRECTORY / "market_first_challenger.csv"
FEATURE_COLUMNS = (
    "model_disagreement",
    "market_movement_from_opening",
    "book_disagreement_market_strength",
    "low_history_model_disagreement",
)
L2_GRID = (0.001, 0.01, 0.1, 1.0)


def _url_token(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    token = str(value or "").strip().rstrip("/").rsplit("/", 1)[-1]
    return token


def attach_strict_prior_ufc_counts(
    paired: pd.DataFrame,
    raw_fights: pd.DataFrame,
) -> pd.DataFrame:
    """Attach UFC fight counts using only event dates before each bout.

    Results from an earlier bout on the same card are deliberately unavailable:
    all bouts with the same event date see the history that existed before that
    date.
    """

    required_raw = {"date", "fight_url", "fighter_url", "opponent_url"}
    missing = required_raw - set(raw_fights.columns)
    if missing:
        raise ValueError(f"raw fights are missing columns: {sorted(missing)}")
    required_paired = {"fight_id", "fighter_id", "opponent_id"}
    missing = required_paired - set(paired.columns)
    if missing:
        raise ValueError(f"paired rows are missing columns: {sorted(missing)}")

    raw = raw_fights[list(required_raw)].copy()
    raw["event_date"] = pd.to_datetime(raw["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    raw["fight_id"] = raw["fight_url"].map(_url_token)
    raw["fighter_id"] = raw["fighter_url"].map(_url_token)
    raw["opponent_id"] = raw["opponent_url"].map(_url_token)
    if raw[["fight_id", "fighter_id", "opponent_id"]].eq("").any().any():
        raise ValueError("raw fights contain an empty stable identity")

    bouts: list[dict[str, object]] = []
    for fight_id, rows in raw.groupby("fight_id", sort=False):
        dates = sorted(set(rows["event_date"].astype(str)))
        if len(dates) != 1:
            raise ValueError(f"fight {fight_id} appears on more than one event date")
        fighter_ids = set(rows["fighter_id"].astype(str)) | set(
            rows["opponent_id"].astype(str)
        )
        if len(fighter_ids) != 2:
            raise ValueError(f"fight {fight_id} does not contain exactly two fighters")
        bouts.append(
            {
                "event_date": dates[0],
                "fight_id": str(fight_id),
                "fighter_ids": tuple(sorted(fighter_ids)),
            }
        )

    history: dict[str, int] = {}
    prior: dict[tuple[str, str], int] = {}
    bout_frame = pd.DataFrame(bouts).sort_values(
        ["event_date", "fight_id"], kind="stable"
    )
    for _event_date, same_day in bout_frame.groupby("event_date", sort=True):
        for bout in same_day.to_dict("records"):
            for fighter_id in bout["fighter_ids"]:
                prior[(str(bout["fight_id"]), str(fighter_id))] = history.get(
                    str(fighter_id), 0
                )
        for bout in same_day.to_dict("records"):
            for fighter_id in bout["fighter_ids"]:
                fighter_id = str(fighter_id)
                history[fighter_id] = history.get(fighter_id, 0) + 1

    result = paired.copy()
    result["fighter_prior_ufc_fights"] = [
        prior.get((str(fight_id), str(fighter_id)), math.nan)
        for fight_id, fighter_id in result[
            ["fight_id", "fighter_id"]
        ].itertuples(index=False, name=None)
    ]
    result["opponent_prior_ufc_fights"] = [
        prior.get((str(fight_id), str(fighter_id)), math.nan)
        for fight_id, fighter_id in result[
            ["fight_id", "opponent_id"]
        ].itertuples(index=False, name=None)
    ]
    missing_counts = result[
        ["fighter_prior_ufc_fights", "opponent_prior_ufc_fights"]
    ].isna().any(axis=1)
    if missing_counts.any():
        examples = result.loc[missing_counts, "fight_id"].astype(str).head(5).tolist()
        raise ValueError(f"could not reconstruct prior UFC counts for fights: {examples}")
    result["fighter_prior_ufc_fights"] = result[
        "fighter_prior_ufc_fights"
    ].astype(int)
    result["opponent_prior_ufc_fights"] = result[
        "opponent_prior_ufc_fights"
    ].astype(int)
    result["minimum_prior_ufc_fights"] = result[
        ["fighter_prior_ufc_fights", "opponent_prior_ufc_fights"]
    ].min(axis=1)
    return result


def _logit(values: pd.Series | np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def prepare_features(
    paired: pd.DataFrame,
    *,
    low_history_threshold: int = 3,
) -> pd.DataFrame:
    required = {
        "fight_id",
        "horizon",
        "market_probability",
        "model_probability",
        "book_probability_range",
        "minimum_prior_ufc_fights",
    }
    missing = required - set(paired.columns)
    if missing:
        raise ValueError(f"paired rows are missing columns: {sorted(missing)}")
    if low_history_threshold < 1:
        raise ValueError("low-history threshold must be at least one fight")

    result = paired.copy()
    opening = (
        result.loc[result["horizon"].eq("opening"), ["fight_id", "market_probability"]]
        .drop_duplicates("fight_id")
        .set_index("fight_id")["market_probability"]
    )
    result["opening_market_probability"] = result["fight_id"].map(opening)
    market_logit = _logit(result["market_probability"])
    model_logit = _logit(result["model_probability"])
    opening_logit = _logit(
        result["opening_market_probability"].fillna(result["market_probability"])
    )
    result["market_logit"] = market_logit
    result["model_disagreement"] = model_logit - market_logit
    result["opening_price_available"] = result[
        "opening_market_probability"
    ].notna()
    result["market_movement_from_opening"] = np.where(
        result["opening_price_available"], market_logit - opening_logit, 0.0
    )
    result["book_disagreement_market_strength"] = (
        market_logit * result["book_probability_range"].to_numpy(dtype=float)
    )
    result["low_history"] = (
        result["minimum_prior_ufc_fights"].to_numpy(dtype=int)
        < low_history_threshold
    )
    result["low_history_model_disagreement"] = (
        result["model_disagreement"].to_numpy(dtype=float)
        * result["low_history"].to_numpy(dtype=float)
    )
    values = result[["market_logit", *FEATURE_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("market-first features contain non-finite values")
    return result


def _metrics(frame: pd.DataFrame, column: str) -> dict[str, object]:
    return forecast_metrics(
        frame[column].astype(float).tolist(), frame["target"].astype(int).tolist()
    ).to_mapping()


def _fit_offset_logistic(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    l2: float,
) -> dict[str, object]:
    if not features:
        return {"features": [], "l2": None, "scales": [], "coefficients": []}
    if l2 <= 0.0:
        raise ValueError("L2 strength must be positive")
    x = frame[list(features)].to_numpy(dtype=float)
    y = frame["target"].to_numpy(dtype=float)
    offset = frame["market_logit"].to_numpy(dtype=float)
    scales = np.sqrt(np.mean(np.square(x), axis=0))
    if (scales <= 1e-12).any() or not np.isfinite(scales).all():
        raise ValueError("cannot fit a feature with no usable training variation")
    z = x / scales
    coefficients = np.zeros(z.shape[1], dtype=float)
    identity = np.eye(z.shape[1], dtype=float)
    for _ in range(60):
        probability = _sigmoid(offset + z @ coefficients)
        gradient = (z.T @ (probability - y)) / len(y) + l2 * coefficients
        weights = probability * (1.0 - probability)
        hessian = (z.T @ (z * weights[:, None])) / len(y) + l2 * identity
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    if not np.isfinite(coefficients).all():
        raise RuntimeError("market-first coefficient fit did not remain finite")
    return {
        "features": list(features),
        "l2": float(l2),
        "scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
    }


def _predict(frame: pd.DataFrame, fitted: Mapping[str, object]) -> np.ndarray:
    features = [str(value) for value in fitted["features"]]
    offset = frame["market_logit"].to_numpy(dtype=float)
    if not features:
        return _sigmoid(offset)
    scales = np.asarray(fitted["scales"], dtype=float)
    coefficients = np.asarray(fitted["coefficients"], dtype=float)
    values = frame[features].to_numpy(dtype=float) / scales
    return _sigmoid(offset + values @ coefficients)


def _all_subsets(features: Sequence[str]) -> list[tuple[str, ...]]:
    return [
        subset
        for size in range(len(features) + 1)
        for subset in combinations(features, size)
    ]


def _chronological_parts(
    frame: pd.DataFrame,
    *,
    minimum_event_dates: int,
    minimum_train_fights: int,
    minimum_selection_fights: int,
    minimum_test_fights: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dates = sorted(set(frame["event_date"].astype(str)))
    if len(dates) < minimum_event_dates:
        raise ValueError(
            f"only {len(dates)} event dates; need at least {minimum_event_dates}"
        )
    train_end = max(1, int(math.floor(len(dates) * 0.60)))
    selection_end = max(train_end + 1, int(math.floor(len(dates) * 0.80)))
    selection_end = min(selection_end, len(dates) - 1)
    training_dates = set(dates[:train_end])
    selection_dates = set(dates[train_end:selection_end])
    test_dates = set(dates[selection_end:])
    training = frame.loc[frame["event_date"].isin(training_dates)].copy()
    selection = frame.loc[frame["event_date"].isin(selection_dates)].copy()
    test = frame.loc[frame["event_date"].isin(test_dates)].copy()
    required_sizes = (
        ("training", training, minimum_train_fights),
        ("selection", selection, minimum_selection_fights),
        ("test", test, minimum_test_fights),
    )
    for label, rows, required in required_sizes:
        if len(rows) < required:
            raise ValueError(f"{label} has {len(rows)} fights; need at least {required}")
    if max(training_dates) >= min(selection_dates) or max(selection_dates) >= min(test_dates):
        raise RuntimeError("chronological partitions overlap")
    contract = {
        "rule": (
            "earliest 60% of event dates fit candidates; next 20% choose one; "
            "latest 20% are untouched final evaluation"
        ),
        "training": {
            "first_date": min(training_dates),
            "last_date": max(training_dates),
            "event_dates": len(training_dates),
            "events": int(training["event_id"].nunique()),
            "fights": len(training),
        },
        "selection": {
            "first_date": min(selection_dates),
            "last_date": max(selection_dates),
            "event_dates": len(selection_dates),
            "events": int(selection["event_id"].nunique()),
            "fights": len(selection),
        },
        "untouched_test": {
            "first_date": min(test_dates),
            "last_date": max(test_dates),
            "event_dates": len(test_dates),
            "events": int(test["event_id"].nunique()),
            "fights": len(test),
        },
    }
    return training, selection, test, contract


def _evaluate_horizon(
    rows: pd.DataFrame,
    *,
    minimum_event_dates: int,
    minimum_train_fights: int,
    minimum_selection_fights: int,
    minimum_test_fights: int,
    minimum_feature_support: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    training, selection, test, partition = _chronological_parts(
        rows,
        minimum_event_dates=minimum_event_dates,
        minimum_train_fights=minimum_train_fights,
        minimum_selection_fights=minimum_selection_fights,
        minimum_test_fights=minimum_test_fights,
    )
    support = {
        feature: int((training[feature].abs() > 1e-12).sum())
        for feature in FEATURE_COLUMNS
    }
    usable = [
        feature
        for feature in FEATURE_COLUMNS
        if support[feature] >= minimum_feature_support
        and float(training[feature].std(ddof=0)) > 1e-12
    ]
    skipped = {
        feature: (
            f"only {support[feature]} non-zero training fights; "
            f"need {minimum_feature_support}"
            if support[feature] < minimum_feature_support
            else "no usable variation in training fights"
        )
        for feature in FEATURE_COLUMNS
        if feature not in usable
    }

    candidates: list[dict[str, object]] = []
    for subset in _all_subsets(usable):
        strengths: Sequence[float | None] = L2_GRID if subset else (None,)
        for strength in strengths:
            fitted = _fit_offset_logistic(
                training, subset, l2=float(strength or 1.0)
            )
            scored = selection.copy()
            scored["candidate_probability"] = _predict(scored, fitted)
            metrics = _metrics(scored, "candidate_probability")
            candidates.append(
                {
                    "features": list(subset),
                    "l2": strength,
                    "selection_log_loss": float(metrics["log_loss"]),
                    "selection_brier_score": float(metrics["brier_score"]),
                    "selection_accuracy": float(metrics["accuracy"]),
                }
            )
    selected = min(
        candidates,
        key=lambda item: (
            round(float(item["selection_log_loss"]), 12),
            len(item["features"]),
            float(item["l2"] or 0.0),
        ),
    )
    fit_rows = pd.concat([training, selection], ignore_index=True)
    fitted = _fit_offset_logistic(
        fit_rows,
        selected["features"],
        l2=float(selected["l2"] or 1.0),
    )
    detail = test.copy()
    detail["candidate_probability"] = _predict(detail, fitted)
    detail["selected_features"] = "|".join(selected["features"])
    detail["selected_l2"] = selected["l2"] if selected["l2"] is not None else ""
    market_metrics = _metrics(detail, "market_probability")
    candidate_metrics = _metrics(detail, "candidate_probability")
    difference = event_block_difference_interval(
        detail, "candidate_probability", "market_probability"
    )
    if difference["ci_95_upper"] < 0.0:
        conclusion = (
            "The selected combination beat the market on the untouched fights, "
            "including the whole-event uncertainty range. Keep it frozen for a "
            "separate future test; do not promote it yet."
        )
    elif difference["point_difference"] < 0.0:
        conclusion = (
            "The selected combination was slightly better on the untouched fights, "
            "but the uncertainty range still includes no improvement."
        )
    else:
        conclusion = (
            "The selected combination did not beat the market on the untouched "
            "fights. Keep the market-only baseline."
        )
    leaderboard = sorted(
        candidates,
        key=lambda item: (
            float(item["selection_log_loss"]),
            len(item["features"]),
        ),
    )
    report = {
        "status": "evaluated",
        "partition": partition,
        "training_feature_support": support,
        "minimum_feature_support": minimum_feature_support,
        "features_with_enough_training_examples": usable,
        "features_skipped": skipped,
        "opening_price_coverage": {
            "all_fights": int(rows["opening_price_available"].sum()),
            "all_fight_count": len(rows),
            "untouched_test_fights": int(detail["opening_price_available"].sum()),
            "untouched_test_fight_count": len(detail),
        },
        "combinations_and_penalties_tested": len(candidates),
        "selection_winner": selected,
        "selection_top_10": leaderboard[:10],
        "final_fit": fitted,
        "untouched_test": {
            "market": market_metrics,
            "current_model": _metrics(detail, "model_probability"),
            "market_first_candidate": candidate_metrics,
            "candidate_minus_market_log_loss_interval": difference,
            "candidate_minus_current_model_log_loss_interval": (
                event_block_difference_interval(
                    detail, "candidate_probability", "model_probability"
                )
            ),
        },
        "plain_language_conclusion": conclusion,
    }
    return report, detail


def evaluate_market_first(
    paired: pd.DataFrame,
    *,
    low_history_threshold: int = 3,
    minimum_event_dates: int = 30,
    minimum_train_fights: int = 150,
    minimum_selection_fights: int = 50,
    minimum_test_fights: int = 50,
    minimum_feature_support: int = 40,
) -> tuple[dict[str, object], pd.DataFrame]:
    prepared = prepare_features(
        paired, low_history_threshold=low_history_threshold
    )
    results: dict[str, object] = {}
    details: list[pd.DataFrame] = []
    for horizon in HORIZON_ORDER:
        rows = prepared.loc[prepared["horizon"].eq(horizon)].copy()
        if rows.empty:
            continue
        try:
            result, detail = _evaluate_horizon(
                rows,
                minimum_event_dates=minimum_event_dates,
                minimum_train_fights=minimum_train_fights,
                minimum_selection_fights=minimum_selection_fights,
                minimum_test_fights=minimum_test_fights,
                minimum_feature_support=minimum_feature_support,
            )
        except ValueError as error:
            results[horizon] = {
                "status": "not_enough_history",
                "reason": str(error),
                "fights": len(rows),
                "event_dates": int(rows["event_date"].nunique()),
            }
            continue
        results[horizon] = result
        details.append(detail)
    if not details:
        reasons = {key: value.get("reason") for key, value in results.items()}
        raise ValueError(f"no horizon has enough history to evaluate: {reasons}")
    output = pd.concat(details, ignore_index=True).sort_values(
        ["horizon", "event_date", "event_id", "fight_id"], kind="stable"
    )
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "production_action": "none",
        "low_history_definition": (
            "at least one fighter had fewer than "
            f"{low_history_threshold} prior UFC fights before the event date"
        ),
        "plain_language_method": (
            "Start with the multi-book market probability. Test every combination "
            "of model disagreement, line movement, sportsbook disagreement, and "
            "limited UFC history. Fit on the earliest 60% of event dates, choose "
            "once on the next 20%, and score once on the untouched latest 20%."
        ),
        "feature_meanings": {
            "model_disagreement": (
                "how far the point-in-time UFC model disagrees with the market"
            ),
            "market_movement_from_opening": (
                "how far the market has moved from its first two-sided price"
            ),
            "book_disagreement_market_strength": (
                "the sportsbook probability range, allowed to adjust how strongly "
                "the consensus favorite is trusted"
            ),
            "low_history_model_disagreement": (
                "a separate model-versus-market adjustment when either fighter has "
                "little prior UFC evidence"
            ),
        },
        "important_limits": [
            "the historical odds backfill must be complete before treating this as the final retrospective result",
            "calendar-date price cutoffs are conservative substitutes for unavailable historical event start times",
            "this experiment chooses among many combinations, so any apparent winner still needs a frozen future test",
            "no candidate can alter production probabilities or place bets",
        ],
        "input_rows": len(prepared),
        "input_fights": int(prepared["fight_id"].nunique()),
        "input_events": int(prepared["event_id"].nunique()),
        "input_sha256": canonical_hash(prepared.to_dict("records")),
        "horizons": results,
    }
    return report, output


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
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--paired-input", type=Path)
    parser.add_argument("--raw-fights", type=Path, default=DEFAULT_RAW_FIGHTS)
    parser.add_argument("--minimum-consensus-books", type=int, default=3)
    parser.add_argument("--max-runtime-minutes", type=float, default=55.0)
    parser.add_argument("--low-history-threshold", type=int, default=3)
    parser.add_argument("--minimum-feature-support", type=int, default=40)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.paired_input is not None:
        paired = pd.read_csv(arguments.paired_input, low_memory=False)
        source = {"paired_input": str(arguments.paired_input)}
    else:
        baseline, paired = build_market_snapshot(
            database_path=arguments.database,
            minimum_consensus_books=arguments.minimum_consensus_books,
            raw_fights_path=arguments.raw_fights,
            max_runtime_minutes=arguments.max_runtime_minutes,
        )
        source = {
            "database_snapshot": baseline["database_snapshot"],
            "coverage": baseline["coverage"],
            "consensus_snapshot_sha256": baseline["consensus_snapshot_sha256"],
        }
    raw = pd.read_csv(arguments.raw_fights, low_memory=False)
    paired = attach_strict_prior_ufc_counts(paired, raw)
    report, detail = evaluate_market_first(
        paired,
        low_history_threshold=arguments.low_history_threshold,
        minimum_feature_support=arguments.minimum_feature_support,
    )
    report["source"] = source
    if not arguments.dry_run:
        _atomic_write(
            arguments.report,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        _atomic_write(
            arguments.detail,
            detail.to_csv(index=False, lineterminator="\n", float_format="%.15g"),
        )
    for horizon, result in report["horizons"].items():
        if result["status"] != "evaluated":
            print(f"{horizon}: skipped ({result['reason']})")
            continue
        test = result["untouched_test"]
        selected = result["selection_winner"]["features"] or ["market only"]
        print(
            f"{horizon}: {test['market']['count']} untouched fights; "
            f"market log loss={test['market']['log_loss']:.5f}; "
            f"candidate={test['market_first_candidate']['log_loss']:.5f}; "
            f"selected={', '.join(selected)}"
        )
    if not arguments.dry_run:
        print(f"Report: {arguments.report}")
        print(f"Fight detail: {arguments.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
