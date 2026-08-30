"""Causally compare historical method prop prices with the outcome model.

BestFightOdds history usually exposes three binary selections for one fighter:
that fighter by KO/TKO, submission, or decision. It does not usually expose a
single simultaneous six-way board for both fighters. This evaluator therefore
scores every quoted selection as a yes/no forecast. Mean prices are calibrated
using earlier calendar years only before comparison or blending.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from backfill_bestfightodds_method_history import default_database_path
from evaluate_historical_method_markets import (
    _atomic_write,
    _file_sha256,
    build_causal_outcome_predictions,
)


METHODS = ("ko_tko", "submission", "decision")
BLEND_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
FRESHNESS_LIMITS = (24.0, 72.0, 168.0)
DEFAULT_POINT_IN_TIME = Path(
    "src/content/data/processed/ufc_fights_point_in_time.csv"
)
DEFAULT_SELECTION_INPUT = (
    default_database_path().parent / "method_exports" / "horizon_method_prices.csv"
)
DEFAULT_ANALYSIS_DIRECTORY = default_database_path().parent / "analysis"
DEFAULT_REPORT = (
    DEFAULT_ANALYSIS_DIRECTORY / "historical_method_selection_evaluation.json"
)
DEFAULT_DETAIL = (
    DEFAULT_ANALYSIS_DIRECTORY / "historical_method_selection_evaluation.csv"
)


def _clip_probability(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("probability is not finite")
    return min(max(float(value), 1e-6), 1.0 - 1e-6)


def _logit(value: float) -> float:
    probability = _clip_probability(value)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def binary_logit_blend(market: float, model: float, model_weight: float) -> float:
    """Blend two binary probabilities with exact endpoints."""

    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model blend weight must be within [0, 1]")
    return _sigmoid(
        (1.0 - model_weight) * _logit(market)
        + model_weight * _logit(model)
    )


def align_method_selections(
    prices: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Align one-sided method prices to causal predictions by stable IDs."""

    required = {
        "ufc_event_date", "ufc_event_id", "ufc_fight_id", "source_matchup_id",
        "selected_fighter_id", "method", "book_key", "book_name", "horizon",
        "cutoff_utc", "observed_at_utc", "decimal_odds", "implied_probability",
    }
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"method prices are missing columns: {sorted(missing)}")
    prediction_lookup = {
        str(row["fight_id"]): row for row in predictions.to_dict("records")
    }
    logical_keys = [
        "ufc_fight_id", "book_key", "horizon", "selected_fighter_id", "method"
    ]
    output: list[dict[str, object]] = []
    ambiguous_groups = 0
    exact_duplicate_rows = 0
    unmatched_rows = 0
    for _, group in prices.groupby(logical_keys, sort=False, dropna=False):
        if len(group) > 1:
            distinct = group[
                ["observed_at_utc", "decimal_odds", "implied_probability"]
            ].drop_duplicates()
            if len(distinct) != 1:
                ambiguous_groups += 1
                continue
            exact_duplicate_rows += len(group) - 1
        row = group.iloc[0].to_dict()
        prediction = prediction_lookup.get(str(row["ufc_fight_id"]))
        if prediction is None or prediction.get("actual_outcome") is None:
            unmatched_rows += len(group)
            continue
        selected = str(row["selected_fighter_id"])
        if selected == str(prediction["fighter_id"]):
            side = "fighter"
        elif selected == str(prediction["opponent_id"]):
            side = "opponent"
        else:
            unmatched_rows += len(group)
            continue
        method = str(row["method"])
        if method not in METHODS:
            continue
        outcome = f"{side}_{method}"
        observed = pd.Timestamp(row["observed_at_utc"])
        cutoff = pd.Timestamp(row["cutoff_utc"])
        quote_age = (cutoff - observed).total_seconds() / 3600.0
        if quote_age < -1e-9:
            raise ValueError("method quote occurs after its declared cutoff")
        output.append(
            {
                **prediction,
                "source_matchup_id": str(row["source_matchup_id"]),
                "book_key": str(row["book_key"]),
                "book_name": str(row["book_name"]),
                "horizon": str(row["horizon"]),
                "selected_fighter_id": selected,
                "selected_side": side,
                "method": method,
                "selection_outcome": outcome,
                "selection_won": int(str(prediction["actual_outcome"]) == outcome),
                "observed_at_utc": str(row["observed_at_utc"]),
                "cutoff_utc": str(row["cutoff_utc"]),
                "quote_age_hours": max(float(quote_age), 0.0),
                "decimal_odds": float(row["decimal_odds"]),
                "raw_market_probability": _clip_probability(
                    float(row["implied_probability"])
                ),
                "model_probability": _clip_probability(
                    float(prediction[f"model_{outcome}_probability"])
                ),
            }
        )
    if not output:
        raise ValueError("no historical method selection aligned to a causal prediction")
    detail = pd.DataFrame(output).sort_values(
        ["event_date", "event_id", "fight_id", "selected_fighter_id", "method"],
        kind="stable",
    ).reset_index(drop=True)
    return detail, {
        "ambiguous_logical_selection_groups_excluded": int(ambiguous_groups),
        "exact_duplicate_rows_removed": int(exact_duplicate_rows),
        "unmatched_price_rows": int(unmatched_rows),
    }


def _calibration_features(frame: pd.DataFrame) -> np.ndarray:
    log_odds = np.asarray(
        [_logit(value) for value in frame["raw_market_probability"]], dtype=float
    )
    return np.column_stack(
        [
            log_odds,
            frame["method"].eq("submission").astype(float).to_numpy(),
            frame["method"].eq("decision").astype(float).to_numpy(),
        ]
    )


def add_causal_market_calibration(
    frame: pd.DataFrame,
    *,
    minimum_prior_selections: int = 300,
    minimum_prior_wins: int = 30,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Remove average price markup using selections from earlier years only."""

    result = frame.reset_index(drop=True).copy()
    years = result["event_date"].astype(str).str[:4].astype(int)
    calibrated = result["raw_market_probability"].to_numpy(dtype=float).copy()
    contracts: list[dict[str, object]] = []
    for year in sorted(years.unique().tolist()):
        prior = result.loc[years < year]
        current_mask = years.eq(year).to_numpy()
        positives = int(prior["selection_won"].sum())
        negatives = int(len(prior) - positives)
        if (
            len(prior) < minimum_prior_selections
            or positives < minimum_prior_wins
            or negatives < minimum_prior_wins
        ):
            contracts.append(
                {
                    "test_year": int(year),
                    "status": "raw_price_insufficient_earlier_history",
                    "prior_selections": int(len(prior)),
                    "prior_wins": positives,
                }
            )
            continue
        calibrator = LogisticRegression(C=1.0, max_iter=2_000, solver="lbfgs")
        calibrator.fit(
            _calibration_features(prior),
            prior["selection_won"].to_numpy(dtype=int),
        )
        calibrated[current_mask] = calibrator.predict_proba(
            _calibration_features(result.loc[years.eq(year)])
        )[:, 1]
        contracts.append(
            {
                "test_year": int(year),
                "status": "calibrated_on_earlier_years",
                "prior_selections": int(len(prior)),
                "prior_wins": positives,
                "intercept": float(calibrator.intercept_[0]),
                "coefficients": {
                    "raw_price_log_odds": float(calibrator.coef_[0, 0]),
                    "submission": float(calibrator.coef_[0, 1]),
                    "decision": float(calibrator.coef_[0, 2]),
                },
            }
        )
    result["calibrated_market_probability"] = np.clip(calibrated, 1e-6, 1 - 1e-6)
    return result, contracts


def _add_blend(
    frame: pd.DataFrame, prefix: str, weights: Sequence[float]
) -> pd.DataFrame:
    if len(frame) != len(weights):
        raise ValueError("blend weights do not match selection rows")
    result = frame.copy()
    result[f"{prefix}_probability"] = [
        binary_logit_blend(market, model, weight)
        for market, model, weight in zip(
            result["calibrated_market_probability"],
            result["model_probability"],
            weights,
        )
    ]
    return result


def _binary_metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, object]:
    truth = frame["selection_won"].to_numpy(dtype=int)
    probability = np.clip(
        frame[probability_column].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12
    )
    losses = -(truth * np.log(probability) + (1 - truth) * np.log(1 - probability))
    auc = None
    if len(set(truth.tolist())) == 2:
        auc = float(roc_auc_score(truth, probability))
    return {
        "selections": int(len(frame)),
        "fights": int(frame["fight_id"].nunique()),
        "events": int(frame["event_id"].nunique()),
        "observed_win_rate": float(truth.mean()),
        "mean_probability": float(probability.mean()),
        "binary_log_loss": float(losses.mean()),
        "brier": float(np.square(probability - truth).mean()),
        "roc_auc": auc,
    }


def add_rolling_binary_blend(
    frame: pd.DataFrame, *, minimum_prior_selections: int = 300
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    result = frame.reset_index(drop=True).copy()
    years = result["event_date"].astype(str).str[:4].astype(int)
    selected_weights = np.zeros(len(result), dtype=float)
    contracts: list[dict[str, object]] = []
    for year in sorted(years.unique().tolist()):
        prior = result.loc[years < year].reset_index(drop=True)
        if len(prior) < minimum_prior_selections:
            weight = 0.0
            losses: dict[str, float] = {}
            status = "market_only_insufficient_earlier_history"
        else:
            losses = {}
            for candidate in BLEND_GRID:
                scored = _add_blend(prior, "candidate", [candidate] * len(prior))
                losses[f"{candidate:g}"] = float(
                    _binary_metrics(scored, "candidate_probability")[
                        "binary_log_loss"
                    ]
                )
            weight = min(
                BLEND_GRID,
                key=lambda candidate: (round(losses[f"{candidate:g}"], 12), candidate),
            )
            status = "selected_on_earlier_years"
        selected_weights[years.eq(year).to_numpy()] = weight
        contracts.append(
            {
                "test_year": int(year),
                "status": status,
                "prior_selections": int(len(prior)),
                "selected_model_weight": float(weight),
                "prior_log_loss_by_weight": losses,
            }
        )
    result = _add_blend(result, "rolling_blend", selected_weights.tolist())
    result["rolling_model_weight"] = selected_weights
    return result, contracts


def _event_block_interval(
    frame: pd.DataFrame,
    candidate_column: str,
    reference_column: str,
    *,
    samples: int = 10_000,
    seed: int = 20260830,
) -> dict[str, object]:
    truth = frame["selection_won"].to_numpy(dtype=int)
    candidate = np.clip(frame[candidate_column].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    reference = np.clip(frame[reference_column].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    candidate_loss = -(truth * np.log(candidate) + (1 - truth) * np.log(1 - candidate))
    reference_loss = -(truth * np.log(reference) + (1 - truth) * np.log(1 - reference))
    differences = pd.DataFrame(
        {
            "event_id": frame["event_id"].astype(str).to_numpy(),
            "difference": candidate_loss - reference_loss,
        }
    )
    grouped = differences.groupby("event_id")["difference"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(samples, len(grouped)))
    sampled = grouped["sum"].to_numpy()[indices].sum(axis=1) / grouped[
        "count"
    ].to_numpy()[indices].sum(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return {
        "point_difference": float(differences["difference"].mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "events": int(len(grouped)),
        "selections": int(len(frame)),
        "definition": "candidate minus reference binary log loss; negative favors candidate",
    }


def _performance_report(frame: pd.DataFrame) -> dict[str, object]:
    columns = {
        "raw_market": "raw_market_probability",
        "calibrated_market": "calibrated_market_probability",
        "model": "model_probability",
        "blend_25": "blend_25_probability",
        "blend_50": "blend_50_probability",
        "rolling_blend": "rolling_blend_probability",
    }
    return {
        name: _binary_metrics(frame, column) for name, column in columns.items()
    }


def evaluate_historical_method_selections(
    *,
    selection_input: Path,
    point_in_time_path: Path,
    horizon: str = "safe_t24",
    max_runtime_minutes: float = 55.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not selection_input.is_file():
        raise FileNotFoundError(selection_input)
    if not point_in_time_path.is_file():
        raise FileNotFoundError(point_in_time_path)
    prices = pd.read_csv(selection_input, low_memory=False)
    prices = prices.loc[
        prices["horizon"].eq(horizon) & prices["book_key"].eq("mean")
    ].copy()
    if prices.empty:
        raise ValueError(f"method history contains no mean {horizon} selections")
    point = pd.read_csv(point_in_time_path, low_memory=False)
    predictions, folds = build_causal_outcome_predictions(
        point,
        fight_ids=set(prices["ufc_fight_id"].astype(str)),
        max_runtime_minutes=max_runtime_minutes,
    )
    aligned, exclusions = align_method_selections(prices, predictions)
    results: dict[str, object] = {}
    details: list[pd.DataFrame] = []
    for maximum_age in FRESHNESS_LIMITS:
        fresh = aligned.loc[aligned["quote_age_hours"] <= maximum_age].copy()
        if fresh.empty:
            results[f"{maximum_age:g}_hours"] = {"status": "no_fresh_selections"}
            continue
        fresh, calibration = add_causal_market_calibration(fresh)
        fresh = _add_blend(fresh, "blend_25", [0.25] * len(fresh))
        fresh = _add_blend(fresh, "blend_50", [0.50] * len(fresh))
        fresh, rolling = add_rolling_binary_blend(fresh)
        annual = {
            str(year): _performance_report(group.reset_index(drop=True))
            for year, group in fresh.groupby(
                fresh["event_date"].astype(str).str[:4].astype(int), sort=True
            )
        }
        method_slices = {
            method: _performance_report(
                fresh.loc[fresh["method"].eq(method)].reset_index(drop=True)
            )
            for method in METHODS
            if fresh["method"].eq(method).any()
        }
        results[f"{maximum_age:g}_hours"] = {
            "status": "evaluated",
            "coverage": {
                "selections": int(len(fresh)),
                "fights": int(fresh["fight_id"].nunique()),
                "events": int(fresh["event_id"].nunique()),
                "first_date": str(fresh["event_date"].min()),
                "last_date": str(fresh["event_date"].max()),
            },
            "probability_performance": _performance_report(fresh),
            "performance_by_year": annual,
            "performance_by_method": method_slices,
            "market_calibration": calibration,
            "rolling_blend_selection": rolling,
            "model_minus_calibrated_market_interval": _event_block_interval(
                fresh, "model_probability", "calibrated_market_probability"
            ),
            "blend_25_minus_calibrated_market_interval": _event_block_interval(
                fresh, "blend_25_probability", "calibrated_market_probability"
            ),
            "blend_50_minus_calibrated_market_interval": _event_block_interval(
                fresh, "blend_50_probability", "calibrated_market_probability"
            ),
            "rolling_blend_minus_calibrated_market_interval": _event_block_interval(
                fresh,
                "rolling_blend_probability",
                "calibrated_market_probability",
            ),
        }
        fresh["maximum_quote_age_hours"] = maximum_age
        details.append(fresh)
    if not details:
        raise ValueError("no freshness slice contains evaluable method selections")
    detail = pd.concat(details, ignore_index=True)
    report = {
        "report_schema_version": 1,
        "experiment_version": "historical-binary-method-selection-rolling-v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "production_action": "none",
        "plain_language_method": (
            "Treat each quoted fighter-by-method price as a yes/no prediction. "
            "Fit the fight model separately for each calendar year using only "
            "earlier fights. Remove the mean price source's average markup using "
            "only earlier price years, then compare market, model, and blends."
        ),
        "horizon": horizon,
        "causal_model_folds": folds,
        "alignment_exclusions": exclusions,
        "freshness_results": results,
        "inputs": {
            "method_selection_prices": str(selection_input),
            "method_selection_prices_sha256": _file_sha256(selection_input),
            "point_in_time": str(point_in_time_path),
            "point_in_time_sha256": _file_sha256(point_in_time_path),
        },
        "important_limits": [
            "BestFightOdds mean history is not an executable sportsbook price",
            "the historical source usually quotes only one fighter's three method props per fight",
            "raw implied probabilities include price markup; causal calibration estimates and removes its average effect",
            "historical event start times are unavailable, so safe T-24 uses the source event date at midnight UTC",
            "no profitability claim is made without individual-book prices and verified settlement contracts",
            "the simulator is not included because matching causal simulation forecasts do not exist for this full sample",
        ],
    }
    return report, detail


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-input", type=Path, default=DEFAULT_SELECTION_INPUT)
    parser.add_argument("--point-in-time", type=Path, default=DEFAULT_POINT_IN_TIME)
    parser.add_argument("--horizon", default="safe_t24")
    parser.add_argument("--max-runtime-minutes", type=float, default=55.0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not 0.0 < arguments.max_runtime_minutes <= 60.0:
        raise ValueError("max runtime must be within (0, 60] minutes")
    report, detail = evaluate_historical_method_selections(
        selection_input=arguments.selection_input,
        point_in_time_path=arguments.point_in_time,
        horizon=arguments.horizon,
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
    for freshness, result in report["freshness_results"].items():
        if result["status"] != "evaluated":
            print(f"{freshness}: {result['status']}")
            continue
        performance = result["probability_performance"]
        print(
            f"{freshness}: selections={result['coverage']['selections']}; "
            f"calibrated market log loss={performance['calibrated_market']['binary_log_loss']:.5f}; "
            f"model={performance['model']['binary_log_loss']:.5f}; "
            f"rolling blend={performance['rolling_blend']['binary_log_loss']:.5f}"
        )
    if not arguments.dry_run:
        print(f"Report: {arguments.report}")
        print(f"Detail: {arguments.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
