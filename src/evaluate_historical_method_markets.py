"""Chronologically evaluate historical fighter-by-method markets.

The outcome model is fitted separately for each calendar year using only
earlier fights.  Complete six-way method prices are aligned by stable fighter
IDs and compared with the model on the same fights.  Mean lines support
probability research; individual-book lines additionally support paper profit
settlement.  Nothing here can place bets or change production behavior.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from backfill_bestfightodds_method_history import default_database_path
from fight_predictor.outcome_model import (
    DiscreteTimeOutcomeModel,
    evaluate_outcome_model,
)
from fight_semantics import method_bucket, schedule_from_row
from market_tracker import forecast_metrics


PRIMARY_METHODS = ("ko_tko", "submission", "decision")
PRIMARY_OUTCOMES = tuple(
    f"{side}_{method}"
    for side in ("fighter", "opponent")
    for method in PRIMARY_METHODS
)
BLEND_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
FRESHNESS_LIMITS = (24.0, 72.0, 168.0)
THRESHOLD_GRID = (0.0, 0.025, 0.05, 0.075, 0.10)
DEFAULT_POINT_IN_TIME = Path(
    "src/content/data/processed/ufc_fights_point_in_time.csv"
)
DEFAULT_ANALYSIS_DIRECTORY = default_database_path().parent / "analysis"
DEFAULT_COHERENT_INPUT = (
    default_database_path().parent
    / "method_exports"
    / "coherent_method_probabilities.csv"
)
DEFAULT_REPORT = DEFAULT_ANALYSIS_DIRECTORY / "historical_method_market_evaluation.json"
DEFAULT_DETAIL = DEFAULT_ANALYSIS_DIRECTORY / "historical_method_market_evaluation.csv"
DEFAULT_BET_LEDGER = DEFAULT_ANALYSIS_DIRECTORY / "historical_method_market_bets.csv"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(probabilities: Mapping[str, float]) -> dict[str, float]:
    values = {key: float(probabilities.get(key, 0.0)) for key in PRIMARY_OUTCOMES}
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("primary outcome probabilities have no finite mass")
    result = {key: value / total for key, value in values.items()}
    if any(not 0.0 < value < 1.0 for value in result.values()):
        raise ValueError("primary outcome probability is outside (0, 1)")
    return result


def geometric_blend(
    market: Mapping[str, float],
    model: Mapping[str, float],
    model_weight: float,
) -> dict[str, float]:
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model blend weight must be within [0, 1]")
    market_values = _normalize(market)
    model_values = _normalize(model)
    log_values = {
        key: (1.0 - model_weight) * math.log(market_values[key])
        + model_weight * math.log(model_values[key])
        for key in PRIMARY_OUTCOMES
    }
    maximum = max(log_values.values())
    unscaled = {key: math.exp(value - maximum) for key, value in log_values.items()}
    total = sum(unscaled.values())
    return {key: value / total for key, value in unscaled.items()}


def _mapping_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{outcome}_probability" for outcome in PRIMARY_OUTCOMES]


def _mapping_from_row(row: Mapping[str, object], prefix: str) -> dict[str, float]:
    return {
        outcome: float(row[f"{prefix}_{outcome}_probability"])
        for outcome in PRIMARY_OUTCOMES
    }


def _actual_outcome(row: Mapping[str, object]) -> str | None:
    method = method_bucket(row.get("label_method"))
    if method not in PRIMARY_METHODS:
        return None
    side = "fighter" if int(row["target"]) == 1 else "opponent"
    return f"{side}_{method}"


def build_causal_outcome_predictions(
    point: pd.DataFrame,
    *,
    fight_ids: set[str],
    max_runtime_minutes: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Fit one outcome model per test year using only earlier labels."""

    started = time.monotonic()
    frame = point.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame = frame.sort_values(
        ["date", "event_id", "bout_order", "fight_id"], kind="stable"
    ).reset_index(drop=True)
    features = tuple(column for column in frame if column.endswith("_diff"))
    if not features:
        raise ValueError("point-in-time data has no difference features")
    selected = frame.loc[frame["fight_id"].isin(fight_ids)].copy()
    years = sorted(selected["date"].dt.year.unique().tolist())
    output: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for year in years:
        if time.monotonic() - started > max_runtime_minutes * 60.0:
            raise TimeoutError("method-market model fitting exceeded its runtime cap")
        cutoff = pd.Timestamp(year=int(year), month=1, day=1)
        train_start = cutoff - pd.DateOffset(years=10)
        training = frame.loc[
            (frame["date"] >= train_start) & (frame["date"] < cutoff)
        ].copy()
        test = selected.loc[selected["date"].dt.year.eq(int(year))].copy()
        if len(training) < 1_000 or test.empty:
            folds.append(
                {
                    "test_year": int(year),
                    "status": "insufficient_training_or_test",
                    "training_fights": int(len(training)),
                    "test_price_fights": int(len(test)),
                }
            )
            continue
        tuning_model, tuning_report = evaluate_outcome_model(training, features)
        selected_c = float(tuning_report["selected_c"])
        model = DiscreteTimeOutcomeModel(features, c_value=selected_c).fit(training)
        predicted = 0
        omitted_schedule = 0
        for row in test.to_dict("records"):
            rounds = schedule_from_row(pd.Series(row))[0]
            if rounds is None:
                omitted_schedule += 1
                continue
            prediction = model.predict(row, scheduled_rounds=int(rounds))
            primary = _normalize(prediction.terminal_probabilities)
            item = {
                "event_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "event_id": str(row["event_id"]),
                "fight_id": str(row["fight_id"]),
                "fighter_id": str(row["fighter_id"]),
                "opponent_id": str(row["opponent_id"]),
                "target": int(row["target"]),
                "actual_outcome": _actual_outcome(row),
                "scheduled_rounds": int(rounds),
                "model_training_start": training["date"].min().strftime("%Y-%m-%d"),
                "model_training_through": training["date"].max().strftime("%Y-%m-%d"),
                "model_training_fights": int(len(training)),
                "model_selected_c": selected_c,
            }
            item.update(
                {
                    f"model_{outcome}_probability": primary[outcome]
                    for outcome in PRIMARY_OUTCOMES
                }
            )
            output.append(item)
            predicted += 1
        folds.append(
            {
                "test_year": int(year),
                "status": "evaluated",
                "training_first_date": training["date"].min().strftime("%Y-%m-%d"),
                "training_last_date": training["date"].max().strftime("%Y-%m-%d"),
                "training_fights": int(len(training)),
                "test_price_fights": int(len(test)),
                "predicted_price_fights": predicted,
                "omitted_unknown_schedule": omitted_schedule,
                "selected_c": selected_c,
            }
        )
    if not output:
        raise ValueError("no historical method-price fight received a causal prediction")
    return pd.DataFrame(output), folds


def align_complete_markets(
    coherent: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "method_market_id",
        "ufc_event_date",
        "ufc_event_id",
        "ufc_fight_id",
        "book_key",
        "book_name",
        "horizon",
        "selected_fighter_id",
        "method",
        "observed_at_utc",
        "cutoff_utc",
        "decimal_odds",
        "no_vig_probability",
    }
    missing = required - set(coherent.columns)
    if missing:
        raise ValueError(f"coherent method prices are missing columns: {sorted(missing)}")
    prediction_lookup = {
        str(row["fight_id"]): row for row in predictions.to_dict("records")
    }
    output: list[dict[str, object]] = []
    for market_id, group in coherent.groupby("method_market_id", sort=False):
        if len(group) != 6:
            continue
        base = group.iloc[0].to_dict()
        prediction = prediction_lookup.get(str(base["ufc_fight_id"]))
        if prediction is None or prediction.get("actual_outcome") is None:
            continue
        market: dict[str, float] = {}
        odds: dict[str, float] = {}
        observed: list[pd.Timestamp] = []
        for row in group.to_dict("records"):
            selected = str(row["selected_fighter_id"])
            if selected == str(prediction["fighter_id"]):
                side = "fighter"
            elif selected == str(prediction["opponent_id"]):
                side = "opponent"
            else:
                market = {}
                break
            method = str(row["method"])
            key = f"{side}_{method}"
            if key not in PRIMARY_OUTCOMES or key in market:
                market = {}
                break
            market[key] = float(row["no_vig_probability"])
            odds[key] = float(row["decimal_odds"])
            observed.append(pd.Timestamp(row["observed_at_utc"]))
        if set(market) != set(PRIMARY_OUTCOMES):
            continue
        market = _normalize(market)
        cutoff = pd.Timestamp(base["cutoff_utc"])
        quote_age = max((cutoff - stamp).total_seconds() / 3600.0 for stamp in observed)
        if quote_age < -1e-9:
            raise ValueError("method quote occurs after its declared cutoff")
        item = dict(prediction)
        item.update(
            {
                "method_market_id": str(market_id),
                "book_key": str(base["book_key"]),
                "book_name": str(base["book_name"]),
                "horizon": str(base["horizon"]),
                "quote_age_hours": max(quote_age, 0.0),
            }
        )
        item.update(
            {
                f"market_{outcome}_probability": market[outcome]
                for outcome in PRIMARY_OUTCOMES
            }
        )
        item.update(
            {f"{outcome}_decimal_odds": odds[outcome] for outcome in PRIMARY_OUTCOMES}
        )
        output.append(item)
    if not output:
        raise ValueError("no complete method market aligned to a causal prediction")
    return pd.DataFrame(output).sort_values(
        ["event_date", "event_id", "fight_id", "book_key"], kind="stable"
    ).reset_index(drop=True)


def _multiclass_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
    truth = frame["actual_outcome"].astype(str).tolist()
    matrix = frame[_mapping_columns(prefix)].to_numpy(dtype=float)
    indices = np.asarray([PRIMARY_OUTCOMES.index(value) for value in truth], dtype=int)
    chosen = matrix.argmax(axis=1)
    joint_loss = float(-np.log(np.clip(matrix[np.arange(len(frame)), indices], 1e-12, 1.0)).mean())
    winner_truth = np.asarray([value.startswith("fighter_") for value in truth], dtype=int)
    winner_probability = matrix[:, :3].sum(axis=1)
    winner = forecast_metrics(winner_probability.tolist(), winner_truth.tolist()).to_mapping()
    method_truth = [value.split("_", 1)[1] for value in truth]
    method_matrix = np.column_stack(
        [matrix[:, index] + matrix[:, index + 3] for index in range(3)]
    )
    method_indices = np.asarray([PRIMARY_METHODS.index(value) for value in method_truth])
    method_loss = float(
        -np.log(
            np.clip(
                method_matrix[np.arange(len(frame)), method_indices], 1e-12, 1.0
            )
        ).mean()
    )
    return {
        "fights": int(len(frame)),
        "events": int(frame["event_id"].nunique()),
        "joint_side_method_log_loss": joint_loss,
        "joint_top_class_accuracy": float(
            np.mean(chosen == indices)
        ),
        "winner": winner,
        "method_log_loss": method_loss,
        "method_top_class_accuracy": float(
            np.mean(method_matrix.argmax(axis=1) == method_indices)
        ),
    }


def _add_blend_columns(
    frame: pd.DataFrame, prefix: str, weights: Sequence[float]
) -> pd.DataFrame:
    result = frame.copy()
    for row_index, row in result.iterrows():
        blended = geometric_blend(
            _mapping_from_row(row, "market"),
            _mapping_from_row(row, "model"),
            float(weights[row_index]),
        )
        for outcome, probability in blended.items():
            result.loc[row_index, f"{prefix}_{outcome}_probability"] = probability
    return result


def add_rolling_blend(frame: pd.DataFrame, *, minimum_prior_fights: int = 100) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    result = frame.reset_index(drop=True).copy()
    years = result["event_date"].astype(str).str[:4].astype(int)
    selected_weights = np.zeros(len(result), dtype=float)
    contracts: list[dict[str, object]] = []
    for year in sorted(years.unique().tolist()):
        prior = result.loc[years < year]
        if len(prior) < minimum_prior_fights:
            selected = 0.0
            status = "market_only_insufficient_prior_fights"
            losses: dict[str, float] = {}
        else:
            losses = {}
            for weight in BLEND_GRID:
                blended = _add_blend_columns(
                    prior.reset_index(drop=True), "candidate", [weight] * len(prior)
                )
                losses[f"{weight:g}"] = float(
                    _multiclass_metrics(blended, "candidate")[
                        "joint_side_method_log_loss"
                    ]
                )
            selected = min(
                BLEND_GRID,
                key=lambda weight: (round(losses[f"{weight:g}"], 12), weight),
            )
            status = "selected_on_earlier_fights"
        selected_weights[years.eq(year).to_numpy()] = selected
        contracts.append(
            {
                "test_year": int(year),
                "prior_fights": int(len(prior)),
                "selected_model_weight": float(selected),
                "status": status,
                "prior_log_loss_by_weight": losses,
            }
        )
    result = _add_blend_columns(result, "rolling_blend", selected_weights.tolist())
    result["rolling_model_weight"] = selected_weights
    return result, contracts


def _event_block_interval(
    frame: pd.DataFrame,
    candidate_prefix: str,
    reference_prefix: str,
    *,
    samples: int = 10_000,
    seed: int = 20260829,
) -> dict[str, object]:
    differences = []
    for row in frame.to_dict("records"):
        actual = str(row["actual_outcome"])
        candidate = float(row[f"{candidate_prefix}_{actual}_probability"])
        reference = float(row[f"{reference_prefix}_{actual}_probability"])
        differences.append(
            {
                "event_id": str(row["event_id"]),
                "difference": -math.log(max(candidate, 1e-12))
                + math.log(max(reference, 1e-12)),
            }
        )
    grouped = pd.DataFrame(differences).groupby("event_id")["difference"].agg(
        ["sum", "count"]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(samples, len(grouped)))
    values = grouped["sum"].to_numpy()[indices].sum(axis=1) / grouped[
        "count"
    ].to_numpy()[indices].sum(axis=1)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "point_difference": float(pd.DataFrame(differences)["difference"].mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "events": int(len(grouped)),
        "fights": int(len(frame)),
        "bootstrap_samples": samples,
        "definition": (
            f"{candidate_prefix} minus {reference_prefix} paired joint log loss; "
            "negative favors candidate"
        ),
    }


def _best_book_offer(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    offers: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        for outcome in PRIMARY_OUTCOMES:
            probability = float(row[f"{prefix}_{outcome}_probability"])
            odds = float(row[f"{outcome}_decimal_odds"])
            item = dict(row)
            item.update(
                {
                    "strategy": prefix,
                    "selection": outcome,
                    "fair_probability": probability,
                    "decimal_odds": odds,
                    "estimated_ev": probability * odds - 1.0,
                    "won": outcome == row["actual_outcome"],
                    "profit_units": (
                        odds - 1.0 if outcome == row["actual_outcome"] else -1.0
                    ),
                }
            )
            offers.append(item)
    return (
        pd.DataFrame(offers)
        .sort_values(
            ["fight_id", "estimated_ev", "book_name", "selection"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        .drop_duplicates("fight_id", keep="first")
        .sort_values(["event_date", "event_id", "fight_id"], kind="stable")
        .reset_index(drop=True)
    )


def _profit_summary(rows: pd.DataFrame) -> dict[str, object]:
    count = int(len(rows))
    profit = float(rows["profit_units"].sum()) if count else 0.0
    return {
        "bets": count,
        "events": int(rows["event_id"].nunique()) if count else 0,
        "wins": int(rows["won"].sum()) if count else 0,
        "profit_units": profit,
        "roi": profit / count if count else None,
    }


def rolling_profitability(
    frame: pd.DataFrame,
    *,
    minimum_prior_bets: int = 20,
    minimum_prior_events: int = 8,
) -> tuple[dict[str, object], pd.DataFrame]:
    book_rows = frame.loc[~frame["book_key"].eq("mean")].copy()
    if book_rows.empty:
        return {"status": "individual_book_history_unavailable"}, pd.DataFrame()
    reports: dict[str, object] = {}
    ledgers: list[pd.DataFrame] = []
    for strategy in ("model", "rolling_blend"):
        best = _best_book_offer(book_rows, strategy)
        years = best["event_date"].str[:4].astype(int)
        fold_reports = []
        for year in sorted(years.unique().tolist()):
            prior = best.loc[years < year]
            test = best.loc[years.eq(year)].copy()
            choices = []
            for threshold in THRESHOLD_GRID:
                selected = prior.loc[prior["estimated_ev"] >= threshold]
                summary = _profit_summary(selected)
                choices.append((threshold, summary))
            eligible = [
                item
                for item in choices
                if item[1]["bets"] >= minimum_prior_bets
                and item[1]["events"] >= minimum_prior_events
                and item[1]["profit_units"] > 0.0
            ]
            if eligible:
                threshold, _ = max(
                    eligible,
                    key=lambda item: (
                        item[1]["profit_units"],
                        item[1]["roi"],
                        item[0],
                    ),
                )
                status = "selected_on_earlier_profit"
            else:
                threshold = 0.05
                status = "fixed_5_percent_reference"
            test["test_year"] = year
            test["selected_threshold"] = threshold
            test["threshold_status"] = status
            test["qualifies"] = test["estimated_ev"] >= threshold
            ledgers.append(test)
            fold_reports.append(
                {
                    "test_year": int(year),
                    "prior_available_fights": int(len(prior)),
                    "test_available_fights": int(len(test)),
                    "selected_threshold": threshold,
                    "threshold_status": status,
                    "test_result": _profit_summary(test.loc[test["qualifies"]]),
                }
            )
        strategy_ledger = pd.concat(
            [item for item in ledgers if item["strategy"].eq(strategy).all()],
            ignore_index=True,
        )
        reports[strategy] = {
            "folds": fold_reports,
            "pooled_result": _profit_summary(
                strategy_ledger.loc[strategy_ledger["qualifies"]]
            ),
        }
    ledger = pd.concat(ledgers, ignore_index=True).sort_values(
        ["strategy", "event_date", "event_id", "fight_id"], kind="stable"
    )
    return {"status": "evaluated", "strategies": reports}, ledger


def evaluate_historical_method_markets(
    *,
    coherent_input: Path,
    point_in_time_path: Path,
    horizon: str = "safe_t24",
    max_runtime_minutes: float = 55.0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if not coherent_input.is_file():
        raise FileNotFoundError(coherent_input)
    if not point_in_time_path.is_file():
        raise FileNotFoundError(point_in_time_path)
    coherent = pd.read_csv(coherent_input, low_memory=False)
    coherent = coherent.loc[coherent["horizon"].eq(horizon)].copy()
    if coherent.empty:
        raise ValueError(f"method history contains no {horizon} complete markets")
    point = pd.read_csv(point_in_time_path, low_memory=False)
    fight_ids = set(coherent["ufc_fight_id"].astype(str))
    predictions, folds = build_causal_outcome_predictions(
        point,
        fight_ids=fight_ids,
        max_runtime_minutes=max_runtime_minutes,
    )
    aligned = align_complete_markets(coherent, predictions)
    results: dict[str, object] = {}
    details: list[pd.DataFrame] = []
    bet_ledgers: list[pd.DataFrame] = []
    for maximum_age in FRESHNESS_LIMITS:
        fresh = aligned.loc[aligned["quote_age_hours"] <= maximum_age].copy()
        mean = fresh.loc[fresh["book_key"].eq("mean")].copy()
        if mean.empty:
            results[f"{maximum_age:g}_hours"] = {
                "status": "no_complete_mean_markets",
                "all_complete_book_markets": int(len(fresh)),
            }
            continue
        mean = mean.sort_values(
            ["event_date", "event_id", "fight_id"], kind="stable"
        ).drop_duplicates("fight_id")
        mean = _add_blend_columns(mean.reset_index(drop=True), "blend_25", [0.25] * len(mean))
        mean = _add_blend_columns(mean, "blend_50", [0.5] * len(mean))
        mean, blend_contract = add_rolling_blend(mean)
        weight_by_year = {
            int(item["test_year"]): float(item["selected_model_weight"])
            for item in blend_contract
        }
        fresh = fresh.reset_index(drop=True)
        fresh_weights = [
            weight_by_year[int(str(event_date)[:4])]
            for event_date in fresh["event_date"]
        ]
        fresh = _add_blend_columns(fresh, "rolling_blend", fresh_weights)
        fresh["rolling_model_weight"] = fresh_weights
        profitability, bet_ledger = rolling_profitability(fresh)
        if not bet_ledger.empty:
            bet_ledger["maximum_quote_age_hours"] = maximum_age
            bet_ledgers.append(bet_ledger)
        results[f"{maximum_age:g}_hours"] = {
            "status": "evaluated",
            "coverage": {
                "mean_fights": int(len(mean)),
                "events": int(mean["event_id"].nunique()),
                "first_date": str(mean["event_date"].min()),
                "last_date": str(mean["event_date"].max()),
                "complete_individual_book_markets": int(
                    (~fresh["book_key"].eq("mean")).sum()
                ),
            },
            "probability_performance": {
                prefix: _multiclass_metrics(mean, prefix)
                for prefix in ("market", "model", "blend_25", "blend_50", "rolling_blend")
            },
            "rolling_blend_selection": blend_contract,
            "rolling_blend_minus_market_interval": _event_block_interval(
                mean, "rolling_blend", "market"
            ),
            "profitability": profitability,
        }
        mean["maximum_quote_age_hours"] = maximum_age
        details.append(mean)
    if not details:
        raise ValueError("no freshness slice has a complete mean method market")
    detail = pd.concat(details, ignore_index=True)
    bet_ledger = (
        pd.concat(bet_ledgers, ignore_index=True)
        if bet_ledgers
        else pd.DataFrame()
    )
    report = {
        "report_schema_version": 1,
        "experiment_version": "historical-method-market-rolling-v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "production_action": "none",
        "plain_language_method": (
            "Fit the outcome model separately for each year using only earlier "
            "fights. Compare its conditional six primary fighter-by-method "
            "probabilities with complete six-way historical markets. Choose any "
            "rolling blend weight and paper betting cutoff using earlier years only."
        ),
        "horizon": horizon,
        "primary_outcomes": list(PRIMARY_OUTCOMES),
        "other_result_contract": (
            "Fights with DQ, no contest, or another non-primary method are excluded; "
            "model probabilities are conditioned on the six primary outcomes."
        ),
        "causal_model_folds": folds,
        "freshness_results": results,
        "inputs": {
            "coherent_method_prices": str(coherent_input),
            "coherent_method_prices_sha256": _file_sha256(coherent_input),
            "point_in_time": str(point_in_time_path),
            "point_in_time_sha256": _file_sha256(point_in_time_path),
        },
        "important_limits": [
            "BestFightOdds mean history is not an executable sportsbook price",
            "profitability is unavailable until complete individual-book histories exist",
            "historical event start times are unavailable, so horizons use the source event calendar date at midnight UTC",
            "book limits, rejected bets, line latency, fees, and taxes are excluded",
            "prospective confirmation is required before any betting behavior changes",
        ],
    }
    return report, detail, bet_ledger


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
    parser.add_argument("--coherent-input", type=Path, default=DEFAULT_COHERENT_INPUT)
    parser.add_argument("--point-in-time", type=Path, default=DEFAULT_POINT_IN_TIME)
    parser.add_argument("--horizon", default="safe_t24")
    parser.add_argument("--max-runtime-minutes", type=float, default=55.0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--bet-ledger", type=Path, default=DEFAULT_BET_LEDGER)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not 0.0 < arguments.max_runtime_minutes <= 60.0:
        raise ValueError("max runtime must be within (0, 60] minutes")
    report, detail, bet_ledger = evaluate_historical_method_markets(
        coherent_input=arguments.coherent_input,
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
        _atomic_write(
            arguments.bet_ledger,
            bet_ledger.to_csv(
                index=False, lineterminator="\n", float_format="%.15g"
            ),
        )
    for freshness, result in report["freshness_results"].items():
        if result["status"] != "evaluated":
            print(f"{freshness}: {result['status']}")
            continue
        performance = result["probability_performance"]
        print(
            f"{freshness}: fights={result['coverage']['mean_fights']}; "
            f"market joint log loss={performance['market']['joint_side_method_log_loss']:.5f}; "
            f"model={performance['model']['joint_side_method_log_loss']:.5f}; "
            f"rolling blend={performance['rolling_blend']['joint_side_method_log_loss']:.5f}"
        )
    if not arguments.dry_run:
        print(f"Report: {arguments.report}")
        print(f"Detail: {arguments.detail}")
        print(f"Bet ledger: {arguments.bet_ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
