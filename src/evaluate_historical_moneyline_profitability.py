"""Test whether historical T-24 probability estimates translated into book profit.

This is retrospective, paper-only research. For every offered sportsbook price,
the fair-value consensus excludes that sportsbook and requires at least three
other books. The best positive edge is retained once per fight. A threshold is
chosen on the earlier selection period and scored once on the later period.
Nothing here changes production forecasts, thresholds, or betting behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from backfill_bestfightodds_history import (
    database_summary,
    default_database_path,
    derive_horizon_rows,
    open_database_readonly,
)
from market_tracker import symmetric_logit_blend
from market_tracker._common import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = default_database_path()
DEFAULT_ANALYSIS_DIRECTORY = DEFAULT_DATABASE.parent / "analysis"
DEFAULT_PAIRED_INPUT = (
    DEFAULT_ANALYSIS_DIRECTORY / "current_model_market_evaluation_2023_2026.csv"
)
DEFAULT_CHALLENGER_REPORT = (
    DEFAULT_ANALYSIS_DIRECTORY / "market_first_challenger_2023_2026.json"
)
DEFAULT_REPORT = (
    DEFAULT_ANALYSIS_DIRECTORY / "historical_moneyline_profitability_2023_2026.json"
)
DEFAULT_LEDGER = (
    DEFAULT_ANALYSIS_DIRECTORY / "historical_moneyline_profitability_2023_2026.csv"
)
DEFAULT_HORIZON = "safe_t24"
DEFAULT_THRESHOLDS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10)
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260829
DEFAULT_MAXIMUM_QUOTE_AGE_HOURS = 168.0
STRATEGY_PROBABILITIES = {
    "leave_one_out_market": "leave_one_out_market_probability",
    "current_model": "model_probability",
    "fixed_50_50_logit_blend": "fixed_50_50_probability",
    "market_first_candidate": "candidate_probability",
}


def _logit(probability: float) -> float:
    value = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-min(max(float(value), -40.0), 40.0)))


def candidate_probability(
    *,
    market_probability: float,
    model_probability: float,
    book_probability_range: float,
    fitted: Mapping[str, object],
) -> float:
    """Apply the frozen offset-logistic market-first fit."""

    feature_values = {
        "model_disagreement": _logit(model_probability)
        - _logit(market_probability),
        "book_disagreement_market_strength": _logit(market_probability)
        * float(book_probability_range),
    }
    features = [str(value) for value in fitted.get("features", [])]
    unsupported = set(features) - set(feature_values)
    if unsupported:
        raise ValueError(
            "profitability evaluator does not support frozen features: "
            f"{sorted(unsupported)}"
        )
    scales = [float(value) for value in fitted.get("scales", [])]
    coefficients = [float(value) for value in fitted.get("coefficients", [])]
    if len(features) != len(scales) or len(features) != len(coefficients):
        raise ValueError("frozen challenger fit dimensions do not agree")
    adjustment = 0.0
    for feature, scale, coefficient in zip(
        features, scales, coefficients, strict=True
    ):
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("frozen challenger scale must be finite and positive")
        adjustment += coefficient * feature_values[feature] / scale
    return _sigmoid(_logit(market_probability) + adjustment)


def _paired_by_fight(paired: pd.DataFrame, *, horizon: str) -> dict[str, dict[str, Any]]:
    required = {
        "event_date",
        "event_id",
        "fight_id",
        "fighter_id",
        "opponent_id",
        "fighter_name",
        "opponent_name",
        "target",
        "horizon",
        "model_probability",
    }
    missing = required - set(paired.columns)
    if missing:
        raise ValueError(f"paired input is missing columns: {sorted(missing)}")
    rows = paired.loc[paired["horizon"].eq(horizon)].copy()
    if rows.empty:
        raise ValueError(f"paired input has no {horizon} rows")
    if rows["fight_id"].astype(str).duplicated().any():
        raise ValueError(f"paired input has duplicate {horizon} fight rows")
    rows["target"] = pd.to_numeric(rows["target"], errors="raise").astype(int)
    rows["model_probability"] = pd.to_numeric(
        rows["model_probability"], errors="raise"
    )
    return {
        str(row["fight_id"]): row for row in rows.to_dict("records")
    }


def _orient_book_row(
    row: Mapping[str, object], paired: Mapping[str, object]
) -> dict[str, float]:
    first = str(row["ufc_fighter_1_id"])
    second = str(row["ufc_fighter_2_id"])
    fighter = str(paired["fighter_id"])
    opponent = str(paired["opponent_id"])
    first_probability = float(row["fighter_1_no_vig_probability"])
    first_odds = float(row["fighter_1_decimal_odds"])
    second_odds = float(row["fighter_2_decimal_odds"])
    if (first, second) == (fighter, opponent):
        return {
            "fighter_probability": first_probability,
            "fighter_odds": first_odds,
            "opponent_odds": second_odds,
        }
    if (first, second) == (opponent, fighter):
        return {
            "fighter_probability": 1.0 - first_probability,
            "fighter_odds": second_odds,
            "opponent_odds": first_odds,
        }
    raise ValueError(f"fighter identity mismatch for fight {paired['fight_id']}")


def _deduplicate_books(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    by_name: dict[str, Mapping[str, object]] = {}
    for row in rows:
        key = str(row["book_name"]).strip().casefold()
        if not key:
            raise ValueError("historical quote has an empty book name")
        current = by_name.get(key)
        if current is None or str(row["observed_at_utc"]) > str(
            current["observed_at_utc"]
        ):
            by_name[key] = row
    return list(by_name.values())


def build_leave_one_out_values(
    horizon_rows: Sequence[Mapping[str, object]],
    paired: pd.DataFrame,
    *,
    fitted: Mapping[str, object],
    horizon: str = DEFAULT_HORIZON,
    closing_horizon: str = "strict_latest_before_event_date",
    minimum_other_books: int = 3,
    maximum_quote_age_hours: float | None = DEFAULT_MAXIMUM_QUOTE_AGE_HOURS,
) -> pd.DataFrame:
    """Create one target-book value row while excluding it from fair value."""

    if minimum_other_books < 2:
        raise ValueError("at least two other books are required")
    paired_lookup = _paired_by_fight(paired, horizon=horizon)
    relevant = [
        dict(row)
        for row in horizon_rows
        if str(row.get("ufc_fight_id", "")) in paired_lookup
        and str(row.get("book_kind", "")) == "book"
        and str(row.get("horizon", "")) in {horizon, closing_horizon}
    ]
    for row in relevant:
        cutoff = pd.Timestamp(row["cutoff_utc"])
        observed = pd.Timestamp(row["observed_at_utc"])
        age_hours = (cutoff - observed).total_seconds() / 3600.0
        if age_hours < -1e-9:
            raise ValueError("historical quote occurs after its declared cutoff")
        row["_quote_age_hours"] = max(age_hours, 0.0)
    source_matchups: dict[str, set[str]] = defaultdict(set)
    for row in relevant:
        source_matchups[str(row["ufc_fight_id"])].add(
            str(row["source_matchup_id"])
        )
    duplicated_source_fights = {
        fight_id for fight_id, values in source_matchups.items() if len(values) != 1
    }
    if maximum_quote_age_hours is not None:
        if maximum_quote_age_hours <= 0.0:
            raise ValueError("maximum quote age must be positive")
        relevant = [
            row
            for row in relevant
            if float(row["_quote_age_hours"]) <= maximum_quote_age_hours
        ]

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in relevant:
        fight_id = str(row["ufc_fight_id"])
        if fight_id in duplicated_source_fights:
            continue
        grouped[(fight_id, str(row["horizon"]))].append(row)

    closing: dict[tuple[str, str], dict[str, float]] = {}
    for (fight_id, row_horizon), values in grouped.items():
        if row_horizon != closing_horizon:
            continue
        paired_row = paired_lookup[fight_id]
        for book in _deduplicate_books(values):
            key = str(book["book_name"]).strip().casefold()
            closing[(fight_id, key)] = _orient_book_row(book, paired_row)
            closing[(fight_id, key)]["quote_age_hours"] = float(
                book["_quote_age_hours"]
            )

    output: list[dict[str, object]] = []
    for (fight_id, row_horizon), values in sorted(grouped.items()):
        if row_horizon != horizon:
            continue
        paired_row = paired_lookup[fight_id]
        books = _deduplicate_books(values)
        oriented = [
            (book, _orient_book_row(book, paired_row)) for book in books
        ]
        for target_book, target in oriented:
            target_key = str(target_book["book_name"]).strip().casefold()
            other = [
                values
                for book, values in oriented
                if str(book["book_name"]).strip().casefold() != target_key
            ]
            if len(other) < minimum_other_books:
                continue
            probabilities = [float(item["fighter_probability"]) for item in other]
            market_probability = float(np.mean(probabilities))
            probability_range = max(probabilities) - min(probabilities)
            model_probability = float(paired_row["model_probability"])
            adjusted_probability = candidate_probability(
                market_probability=market_probability,
                model_probability=model_probability,
                book_probability_range=probability_range,
                fitted=fitted,
            )
            latest = closing.get((fight_id, target_key), {})
            output.append(
                {
                    "event_date": str(paired_row["event_date"]),
                    "event_id": str(paired_row["event_id"]),
                    "fight_id": fight_id,
                    "fighter_id": str(paired_row["fighter_id"]),
                    "opponent_id": str(paired_row["opponent_id"]),
                    "fighter_name": str(paired_row["fighter_name"]),
                    "opponent_name": str(paired_row["opponent_name"]),
                    "target": int(paired_row["target"]),
                    "book_key": str(target_book["book_key"]),
                    "book_name": str(target_book["book_name"]),
                    "book_count_excluding_target": len(other),
                    "observed_at_utc": str(target_book["observed_at_utc"]),
                    "cutoff_utc": str(target_book["cutoff_utc"]),
                    "quote_age_hours": float(target_book["_quote_age_hours"]),
                    "fighter_decimal_odds": float(target["fighter_odds"]),
                    "opponent_decimal_odds": float(target["opponent_odds"]),
                    "fighter_latest_decimal_odds": latest.get("fighter_odds"),
                    "opponent_latest_decimal_odds": latest.get("opponent_odds"),
                    "latest_quote_age_hours": latest.get("quote_age_hours"),
                    "leave_one_out_market_probability": market_probability,
                    "book_probability_range_excluding_target": probability_range,
                    "model_probability": model_probability,
                    "fixed_50_50_probability": symmetric_logit_blend(
                        market_probability, model_probability, 0.5
                    ),
                    "candidate_probability": adjusted_probability,
                }
            )
    if not output:
        raise ValueError("no fights have enough leave-one-out book prices")
    return pd.DataFrame(output).sort_values(
        ["event_date", "event_id", "fight_id", "book_name"], kind="stable"
    ).reset_index(drop=True)


def best_offer_per_fight(
    values: pd.DataFrame,
    *,
    probability_column: str,
    strategy: str,
) -> pd.DataFrame:
    """Retain the highest estimated return across both sides and all books."""

    if probability_column not in values:
        raise ValueError(f"missing probability column {probability_column}")
    offers: list[dict[str, object]] = []
    for row in values.to_dict("records"):
        fighter_probability = float(row[probability_column])
        for side, probability, odds_column, latest_column, won in (
            (
                "fighter",
                fighter_probability,
                "fighter_decimal_odds",
                "fighter_latest_decimal_odds",
                int(row["target"]) == 1,
            ),
            (
                "opponent",
                1.0 - fighter_probability,
                "opponent_decimal_odds",
                "opponent_latest_decimal_odds",
                int(row["target"]) == 0,
            ),
        ):
            decimal_odds = float(row[odds_column])
            latest_value = row.get(latest_column)
            latest_odds = (
                float(latest_value)
                if latest_value is not None and not pd.isna(latest_value)
                else math.nan
            )
            offer = dict(row)
            offer.update(
                {
                    "strategy": strategy,
                    "side": side,
                    "selection_name": (
                        row["fighter_name"] if side == "fighter" else row["opponent_name"]
                    ),
                    "fair_probability": probability,
                    "decimal_odds": decimal_odds,
                    "estimated_ev": probability * decimal_odds - 1.0,
                    "won": bool(won),
                    "profit_units": decimal_odds - 1.0 if won else -1.0,
                    "entry_break_even_probability": 1.0 / decimal_odds,
                    "latest_decimal_odds": latest_odds,
                    "closing_price_advantage": (
                        1.0 / latest_odds - 1.0 / decimal_odds
                        if math.isfinite(latest_odds) and latest_odds > 1.0
                        else math.nan
                    ),
                }
            )
            offers.append(offer)
    frame = pd.DataFrame(offers).sort_values(
        ["fight_id", "estimated_ev", "book_name", "side"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    return frame.drop_duplicates("fight_id", keep="first").sort_values(
        ["event_date", "event_id", "fight_id"], kind="stable"
    ).reset_index(drop=True)


def _maximum_drawdown(profits: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for profit in profits:
        cumulative += float(profit)
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def _event_bootstrap_roi(
    selected: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    if selected.empty or samples <= 0:
        return {
            "bootstrap_samples": 0,
            "ci_95_lower": None,
            "ci_95_upper": None,
        }
    grouped = selected.groupby("event_id", sort=True)["profit_units"].agg(
        ["sum", "count"]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(samples, len(grouped)))
    profits = grouped["sum"].to_numpy(dtype=float)[indices].sum(axis=1)
    counts = grouped["count"].to_numpy(dtype=float)[indices].sum(axis=1)
    roi = profits / counts
    lower, upper = np.quantile(roi, [0.025, 0.975])
    return {
        "bootstrap_samples": samples,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "method": "nonparametric whole-event block bootstrap",
        "event_count": int(len(grouped)),
    }


def threshold_metrics(
    best_offers: pd.DataFrame,
    *,
    threshold: float,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    selected = best_offers.loc[best_offers["estimated_ev"] >= threshold].copy()
    selected = selected.sort_values(
        ["event_date", "event_id", "fight_id"], kind="stable"
    )
    closing = pd.to_numeric(
        selected.get("closing_price_advantage", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    count = int(len(selected))
    profit = float(selected["profit_units"].sum()) if count else 0.0
    return {
        "threshold": float(threshold),
        "selections": count,
        "events": int(selected["event_id"].nunique()) if count else 0,
        "wins": int(selected["won"].sum()) if count else 0,
        "losses": int((~selected["won"]).sum()) if count else 0,
        "profit_units": profit,
        "risk_units": float(count),
        "roi": profit / count if count else None,
        "maximum_drawdown_units": (
            _maximum_drawdown(selected["profit_units"].tolist()) if count else 0.0
        ),
        "mean_estimated_ev": (
            float(selected["estimated_ev"].mean()) if count else None
        ),
        "closing_price_advantage": {
            "count": int(len(closing)),
            "mean": float(closing.mean()) if len(closing) else None,
            "positive_rate": float((closing > 0.0).mean()) if len(closing) else None,
        },
        "books": dict(Counter(selected["book_name"]).most_common()) if count else {},
        "sides": dict(Counter(selected["side"]).most_common()) if count else {},
        "roi_interval": _event_bootstrap_roi(
            selected,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }


def choose_threshold(
    best_offers: pd.DataFrame,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    minimum_bets: int = 30,
    minimum_events: int = 10,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> tuple[float, dict[str, dict[str, object]], str]:
    results = {
        f"{float(threshold):.3f}": threshold_metrics(
            best_offers,
            threshold=float(threshold),
            bootstrap_samples=bootstrap_samples,
        )
        for threshold in thresholds
    }
    eligible = [
        result
        for result in results.values()
        if int(result["selections"]) >= minimum_bets
        and int(result["events"]) >= minimum_events
    ]
    profitable = [
        result for result in eligible if float(result["profit_units"]) > 0.0
    ]
    if not profitable:
        fallback = min(thresholds, key=lambda value: abs(float(value) - 0.05))
        status = (
            "fallback_5_percent_insufficient_history"
            if not eligible
            else "fallback_5_percent_no_profitable_earlier_threshold"
        )
        return float(fallback), results, status
    selected = max(
        profitable,
        key=lambda item: (
            float(item["profit_units"]),
            float(item["roi"]),
            float(item["threshold"]),
        ),
    )
    return float(selected["threshold"]), results, "selected_on_earlier_flat_profit"


def _date_slice(frame: pd.DataFrame, contract: Mapping[str, object]) -> pd.DataFrame:
    first = str(contract["first_date"])
    last = str(contract["last_date"])
    return frame.loc[
        frame["event_date"].astype(str).between(first, last, inclusive="both")
    ].copy()


def _read_price_snapshot(
    database_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    connection = open_database_readonly(database_path, mode="both")
    try:
        connection.execute("BEGIN")
        rows = derive_horizon_rows(connection)
        summary = database_summary(connection, database_path=database_path)
        connection.execute("ROLLBACK")
    finally:
        connection.close()
    return rows, summary


def evaluate_profitability(
    *,
    database_path: Path,
    paired_input_path: Path,
    challenger_report_path: Path,
    horizon: str = DEFAULT_HORIZON,
    minimum_other_books: int = 3,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    minimum_selection_bets: int = 30,
    minimum_selection_events: int = 10,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    maximum_quote_age_hours: float | None = DEFAULT_MAXIMUM_QUOTE_AGE_HOURS,
) -> tuple[dict[str, object], pd.DataFrame]:
    for path in (database_path, paired_input_path, challenger_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    paired = pd.read_csv(paired_input_path, low_memory=False)
    challenger = json.loads(challenger_report_path.read_text(encoding="utf-8"))
    horizon_report = challenger.get("horizons", {}).get(horizon)
    if not horizon_report or horizon_report.get("status") != "evaluated":
        raise ValueError(f"challenger report has no evaluated {horizon} result")
    fitted = horizon_report["final_fit"]
    partition = horizon_report["partition"]
    price_rows, database = _read_price_snapshot(database_path)
    values = build_leave_one_out_values(
        price_rows,
        paired,
        fitted=fitted,
        horizon=horizon,
        minimum_other_books=minimum_other_books,
        maximum_quote_age_hours=maximum_quote_age_hours,
    )

    strategy_reports: dict[str, object] = {}
    ledgers: list[pd.DataFrame] = []
    for strategy, probability_column in STRATEGY_PROBABILITIES.items():
        best = best_offer_per_fight(
            values,
            probability_column=probability_column,
            strategy=strategy,
        )
        selection = _date_slice(best, partition["selection"])
        test = _date_slice(best, partition["untouched_test"])
        selected_threshold, selection_results, status = choose_threshold(
            selection,
            thresholds=thresholds,
            minimum_bets=minimum_selection_bets,
            minimum_events=minimum_selection_events,
            bootstrap_samples=bootstrap_samples,
        )
        test_results = {
            f"{float(threshold):.3f}": threshold_metrics(
                test,
                threshold=float(threshold),
                bootstrap_samples=bootstrap_samples,
            )
            for threshold in thresholds
        }
        selected_key = f"{selected_threshold:.3f}"
        strategy_reports[strategy] = {
            "probability_column": probability_column,
            "selection_period": {
                "first_date": partition["selection"]["first_date"],
                "last_date": partition["selection"]["last_date"],
                "available_fights": int(len(selection)),
                "threshold_results": selection_results,
            },
            "threshold_selection": {
                "status": status,
                "selected_threshold": selected_threshold,
                "minimum_bets": minimum_selection_bets,
                "minimum_events": minimum_selection_events,
                "objective": "maximum flat one-unit total profit on selection period",
            },
            "later_period": {
                "first_date": partition["untouched_test"]["first_date"],
                "last_date": partition["untouched_test"]["last_date"],
                "available_fights": int(len(test)),
                "selected_threshold_result": test_results[selected_key],
                "fixed_5_percent_result": test_results.get("0.050"),
                "all_thresholds_descriptive_only": test_results,
            },
        }
        marked = best.copy()
        marked["period"] = "outside_scored_periods"
        marked.loc[
            marked["event_date"].between(
                str(partition["selection"]["first_date"]),
                str(partition["selection"]["last_date"]),
                inclusive="both",
            ),
            "period",
        ] = "threshold_selection"
        marked.loc[
            marked["event_date"].between(
                str(partition["untouched_test"]["first_date"]),
                str(partition["untouched_test"]["last_date"]),
                inclusive="both",
            ),
            "period",
        ] = "later_evaluation"
        marked["selected_threshold"] = selected_threshold
        marked["qualifies_selected_threshold"] = (
            marked["estimated_ev"] >= selected_threshold
        )
        marked["qualifies_fixed_5_percent"] = marked["estimated_ev"] >= 0.05
        ledgers.append(marked)

    ledger = pd.concat(ledgers, ignore_index=True).sort_values(
        ["strategy", "event_date", "event_id", "fight_id"], kind="stable"
    )
    logical_ledger = ledger.to_csv(
        index=False, lineterminator="\n", float_format="%.15g", na_rep=""
    )
    report = {
        "report_schema_version": 1,
        "experiment_version": "historical-t24-leave-one-book-out-profitability-v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "execution_enabled": False,
        "production_action": "none",
        "horizon": horizon,
        "minimum_other_books": minimum_other_books,
        "maximum_quote_age_hours": maximum_quote_age_hours,
        "price_contract": {
            "one_best_offer_per_fight_and_strategy": True,
            "stake": "flat one unit",
            "fair_value_excludes_target_book": True,
            "minimum_other_books": minimum_other_books,
            "maximum_quote_age_hours": maximum_quote_age_hours,
            "gross_before_limits_slippage_and_account_access": True,
            "closing_price_advantage": (
                "latest same-book raw implied probability minus entry raw implied "
                "probability; positive favors the historical entry"
            ),
        },
        "threshold_contract": {
            "grid": [float(value) for value in thresholds],
            "selection_period": partition["selection"],
            "later_period": partition["untouched_test"],
            "minimum_selection_bets": minimum_selection_bets,
            "minimum_selection_events": minimum_selection_events,
            "fixed_reference_threshold": 0.05,
        },
        "important_limits": [
            "this exact profitability mapping was designed after earlier model research and is retrospective development evidence",
            "historical T-24 is measured from midnight UTC on the source event date, not the exact card start",
            "best-book results assume access to every listed sportsbook at the recorded price",
            "returns exclude limits, rejected bets, line latency, stake scaling, fees, and taxes",
            "the active odds backfill was read through one consistent snapshot but was not complete",
            "future paper tracking is required before changing any betting rule",
        ],
        "database_snapshot": database,
        "inputs": {
            "paired_input": str(paired_input_path),
            "challenger_report": str(challenger_report_path),
            "paired_input_sha256": canonical_hash(paired_input_path.read_text(encoding="utf-8")),
            "challenger_report_sha256": canonical_hash(
                challenger_report_path.read_text(encoding="utf-8")
            ),
        },
        "coverage": {
            "target_book_value_rows": int(len(values)),
            "fights_with_leave_one_out_prices": int(values["fight_id"].nunique()),
            "events_with_leave_one_out_prices": int(values["event_id"].nunique()),
            "books": int(values["book_name"].nunique()),
        },
        "strategies": strategy_reports,
        "ledger_sha256": canonical_hash(logical_ledger),
    }
    return report, ledger


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
    parser.add_argument("--paired-input", type=Path, default=DEFAULT_PAIRED_INPUT)
    parser.add_argument(
        "--challenger-report", type=Path, default=DEFAULT_CHALLENGER_REPORT
    )
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument("--minimum-other-books", type=int, default=3)
    parser.add_argument("--minimum-selection-bets", type=int, default=30)
    parser.add_argument("--minimum-selection-events", type=int, default=10)
    parser.add_argument(
        "--maximum-quote-age-hours",
        type=float,
        default=DEFAULT_MAXIMUM_QUOTE_AGE_HOURS,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report, ledger = evaluate_profitability(
        database_path=arguments.database,
        paired_input_path=arguments.paired_input,
        challenger_report_path=arguments.challenger_report,
        horizon=arguments.horizon,
        minimum_other_books=arguments.minimum_other_books,
        minimum_selection_bets=arguments.minimum_selection_bets,
        minimum_selection_events=arguments.minimum_selection_events,
        bootstrap_samples=arguments.bootstrap_samples,
        maximum_quote_age_hours=arguments.maximum_quote_age_hours,
    )
    if not arguments.dry_run:
        _atomic_write(
            arguments.report,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        _atomic_write(
            arguments.ledger,
            ledger.to_csv(
                index=False, lineterminator="\n", float_format="%.15g", na_rep=""
            ),
        )
    print(
        f"Leave-one-out prices: {report['coverage']['fights_with_leave_one_out_prices']} "
        f"fights across {report['coverage']['events_with_leave_one_out_prices']} events."
    )
    for strategy, result in report["strategies"].items():
        threshold = result["threshold_selection"]["selected_threshold"]
        later = result["later_period"]["selected_threshold_result"]
        roi = later["roi"]
        roi_text = "n/a" if roi is None else f"{100.0 * float(roi):.2f}%"
        print(
            f"{strategy}: threshold={100.0 * float(threshold):.1f}%, "
            f"later bets={later['selections']}, profit={later['profit_units']:.2f}u, "
            f"ROI={roi_text}"
        )
    if not arguments.dry_run:
        print(f"Report: {arguments.report}")
        print(f"Ledger: {arguments.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
