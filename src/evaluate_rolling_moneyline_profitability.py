"""Run a broad chronological model/market and moneyline profitability study.

Each test year is predicted using only earlier fights and prices.  The latest
25% of prior event dates chooses a betting threshold; the earlier 75% fits the
small market-first adjustment.  The adjustment is then refit on all prior
fights before the next calendar year is scored.  This is retrospective,
paper-only research and cannot place bets or change production behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from evaluate_historical_moneyline_profitability import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_DATABASE,
    DEFAULT_HORIZON,
    DEFAULT_MAXIMUM_QUOTE_AGE_HOURS,
    DEFAULT_THRESHOLDS,
    STRATEGY_PROBABILITIES,
    _event_bootstrap_roi,
    _maximum_drawdown,
    _read_price_snapshot,
    best_offer_per_fight,
    build_leave_one_out_values,
    candidate_probability,
    choose_threshold,
)
from evaluate_market_first_challenger import (
    L2_GRID,
    _fit_offset_logistic,
    _predict,
)
from evaluate_style_matchup_challenger import event_block_difference_interval
from market_tracker import forecast_metrics, symmetric_logit_blend
from market_tracker._common import canonical_hash


DEFAULT_ANALYSIS_DIRECTORY = DEFAULT_DATABASE.parent / "analysis"
DEFAULT_PAIRED_INPUT = (
    DEFAULT_ANALYSIS_DIRECTORY / "current_model_market_evaluation_2021_2026.csv"
)
DEFAULT_REPORT = (
    DEFAULT_ANALYSIS_DIRECTORY / "rolling_moneyline_profitability_2021_2026.json"
)
DEFAULT_LEDGER = (
    DEFAULT_ANALYSIS_DIRECTORY / "rolling_moneyline_profitability_2021_2026.csv"
)
DEFAULT_PREDICTION_LEDGER = (
    DEFAULT_ANALYSIS_DIRECTORY / "rolling_probability_comparison_2021_2026.csv"
)
CANDIDATE_FEATURES = (
    "model_disagreement",
    "book_disagreement_market_strength",
)
PROBABILITY_COLUMNS = {
    "leave_one_out_market": "market_probability",
    "current_model": "model_probability",
    "fixed_50_50_logit_blend": "fixed_50_50_probability",
    "market_first_candidate": "market_first_candidate_probability",
}


def _logit(values: pd.Series | np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def build_fight_level_fit_frame(values: pd.DataFrame) -> pd.DataFrame:
    """Collapse fresh leave-one-book-out rows to one equally weighted fight."""

    required = {
        "event_date",
        "event_id",
        "fight_id",
        "target",
        "model_probability",
        "leave_one_out_market_probability",
        "book_probability_range_excluding_target",
    }
    missing = required - set(values.columns)
    if missing:
        raise ValueError(f"value rows are missing columns: {sorted(missing)}")
    stable = ["event_date", "event_id", "target", "model_probability"]
    inconsistent = [
        column
        for column in stable
        if (values.groupby("fight_id")[column].nunique(dropna=False) > 1).any()
    ]
    if inconsistent:
        raise ValueError(f"fight rows disagree on stable fields: {inconsistent}")
    first = values.groupby("fight_id", sort=False)[stable].first()
    aggregate = values.groupby("fight_id", sort=False).agg(
        market_probability=("leave_one_out_market_probability", "mean"),
        book_probability_range=(
            "book_probability_range_excluding_target",
            "max",
        ),
        fresh_book_offers=("book_name", "nunique"),
    )
    frame = first.join(aggregate).reset_index()
    frame["market_logit"] = _logit(frame["market_probability"])
    frame["model_disagreement"] = (
        _logit(frame["model_probability"]) - frame["market_logit"]
    )
    frame["book_disagreement_market_strength"] = (
        frame["market_logit"] * frame["book_probability_range"]
    )
    if not np.isfinite(
        frame[["market_logit", *CANDIDATE_FEATURES]].to_numpy(dtype=float)
    ).all():
        raise ValueError("market-first fit frame contains non-finite values")
    return frame.sort_values(
        ["event_date", "event_id", "fight_id"], kind="stable"
    ).reset_index(drop=True)


def chronological_year_parts(
    frame: pd.DataFrame,
    test_year: int,
    *,
    validation_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.10 <= validation_fraction <= 0.50:
        raise ValueError("validation fraction must be within [0.10, 0.50]")
    year = frame["event_date"].astype(str).str[:4].astype(int)
    prior = frame.loc[year < test_year].copy()
    test = frame.loc[year.eq(test_year)].copy()
    dates = sorted(prior["event_date"].astype(str).unique().tolist())
    if len(dates) < 2:
        return prior.iloc[0:0].copy(), prior.copy(), test
    split = int(math.floor(len(dates) * (1.0 - validation_fraction)))
    split = min(max(split, 1), len(dates) - 1)
    development_dates = set(dates[:split])
    validation_dates = set(dates[split:])
    development = prior.loc[prior["event_date"].isin(development_dates)].copy()
    validation = prior.loc[prior["event_date"].isin(validation_dates)].copy()
    if development["event_date"].max() >= validation["event_date"].min():
        raise RuntimeError("development and validation dates overlap")
    if not validation.empty and not test.empty:
        if validation["event_date"].max() >= test["event_date"].min():
            raise RuntimeError("validation and test dates overlap")
    return development, validation, test


def _apply_candidate(values: pd.DataFrame, fitted: Mapping[str, object]) -> pd.DataFrame:
    result = values.copy()
    result["candidate_probability"] = [
        candidate_probability(
            market_probability=float(market),
            model_probability=float(model),
            book_probability_range=float(book_range),
            fitted=fitted,
        )
        for market, model, book_range in result[
            [
                "leave_one_out_market_probability",
                "model_probability",
                "book_probability_range_excluding_target",
            ]
        ].itertuples(index=False, name=None)
    ]
    return result


def _period_contract(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "first_date": str(frame["event_date"].min()),
        "last_date": str(frame["event_date"].max()),
        "events": int(frame["event_id"].nunique()),
        "fights": int(frame["fight_id"].nunique()),
    }


def _selected_summary(
    selected: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    ordered = selected.sort_values(
        ["event_date", "event_id", "fight_id"], kind="stable"
    )
    count = int(len(ordered))
    profit = float(ordered["profit_units"].sum()) if count else 0.0
    closing = pd.to_numeric(
        ordered.get("closing_price_advantage", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    return {
        "selections": count,
        "events": int(ordered["event_id"].nunique()) if count else 0,
        "wins": int(ordered["won"].sum()) if count else 0,
        "losses": int((~ordered["won"]).sum()) if count else 0,
        "profit_units": profit,
        "risk_units": float(count),
        "roi": profit / count if count else None,
        "maximum_drawdown_units": (
            _maximum_drawdown(ordered["profit_units"].tolist()) if count else 0.0
        ),
        "mean_estimated_ev": (
            float(ordered["estimated_ev"].mean()) if count else None
        ),
        "closing_price_advantage": {
            "count": int(len(closing)),
            "mean": float(closing.mean()) if len(closing) else None,
            "positive_rate": float((closing > 0.0).mean()) if len(closing) else None,
        },
        "books": dict(Counter(ordered["book_name"]).most_common()) if count else {},
        "roi_interval": _event_bootstrap_roi(
            ordered,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }


def evaluate_rolling_profitability(
    *,
    database_path: Path,
    paired_input_path: Path,
    horizon: str = DEFAULT_HORIZON,
    maximum_quote_age_hours: float = 24.0,
    minimum_other_books: int = 3,
    minimum_development_fights: int = 75,
    minimum_validation_fights: int = 20,
    minimum_test_fights: int = 20,
    minimum_selection_bets: int = 20,
    minimum_selection_events: int = 8,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    if not paired_input_path.is_file():
        raise FileNotFoundError(paired_input_path)
    paired = pd.read_csv(paired_input_path, low_memory=False)
    price_rows, database = _read_price_snapshot(database_path)
    base_values = build_leave_one_out_values(
        price_rows,
        paired,
        fitted={"features": [], "scales": [], "coefficients": []},
        horizon=horizon,
        minimum_other_books=minimum_other_books,
        maximum_quote_age_hours=maximum_quote_age_hours,
    )
    fit_frame = build_fight_level_fit_frame(base_values)
    years = sorted(fit_frame["event_date"].str[:4].astype(int).unique().tolist())
    folds: list[dict[str, object]] = []
    ledgers: list[pd.DataFrame] = []
    prediction_ledgers: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    for test_year in years[1:]:
        development, validation, test = chronological_year_parts(
            fit_frame, test_year
        )
        sizes = {
            "development": len(development),
            "validation": len(validation),
            "test": len(test),
        }
        requirements = {
            "development": minimum_development_fights,
            "validation": minimum_validation_fights,
            "test": minimum_test_fights,
        }
        too_small = [
            name for name, count in sizes.items() if count < requirements[name]
        ]
        if too_small:
            skipped.append(
                {
                    "test_year": test_year,
                    "reason": "not enough fights in " + ", ".join(too_small),
                    "counts": sizes,
                    "requirements": requirements,
                }
            )
            continue

        candidates: list[tuple[float, float, Mapping[str, object]]] = []
        for l2 in L2_GRID:
            fitted = _fit_offset_logistic(
                development, CANDIDATE_FEATURES, l2=float(l2)
            )
            probability = _predict(validation, fitted)
            score = forecast_metrics(
                probability.tolist(), validation["target"].astype(int).tolist()
            )
            candidates.append((float(score.log_loss), float(l2), fitted))
        _score, selected_l2, development_fit = min(
            candidates, key=lambda item: (round(item[0], 12), item[1])
        )
        prior = pd.concat([development, validation], ignore_index=True)
        final_fit = _fit_offset_logistic(
            prior, CANDIDATE_FEATURES, l2=selected_l2
        )

        validation_values = _apply_candidate(base_values, development_fit)
        test_values = _apply_candidate(base_values, final_fit)
        fold_strategies: dict[str, object] = {}
        for strategy, probability_column in STRATEGY_PROBABILITIES.items():
            validation_best = best_offer_per_fight(
                validation_values,
                probability_column=probability_column,
                strategy=strategy,
            )
            validation_best = validation_best.loc[
                validation_best["fight_id"].isin(validation["fight_id"])
            ].copy()
            selected_threshold, threshold_results, threshold_status = choose_threshold(
                validation_best,
                thresholds=thresholds,
                minimum_bets=minimum_selection_bets,
                minimum_events=minimum_selection_events,
                bootstrap_samples=bootstrap_samples,
            )
            test_best = best_offer_per_fight(
                test_values,
                probability_column=probability_column,
                strategy=strategy,
            )
            test_best = test_best.loc[test_best["fight_id"].isin(test["fight_id"])].copy()
            test_best["test_year"] = test_year
            test_best["selected_threshold"] = selected_threshold
            test_best["threshold_selection_status"] = threshold_status
            test_best["qualifies_selected_threshold"] = (
                test_best["estimated_ev"] >= selected_threshold
            )
            ledgers.append(test_best)
            selected_rows = test_best.loc[
                test_best["qualifies_selected_threshold"]
            ].copy()
            fold_strategies[strategy] = {
                "threshold_selection_status": threshold_status,
                "selected_threshold": selected_threshold,
                "validation_threshold_results": threshold_results,
                "test_available_fights": int(len(test_best)),
                "test_result": _selected_summary(
                    selected_rows,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED + test_year,
                ),
            }

        predictions = test.copy()
        predictions["test_year"] = test_year
        predictions["market_first_candidate_probability"] = _predict(
            predictions, final_fit
        )
        predictions["fixed_50_50_probability"] = [
            symmetric_logit_blend(float(market), float(model), 0.5)
            for market, model in predictions[
                ["market_probability", "model_probability"]
            ].itertuples(index=False, name=None)
        ]
        prediction_ledgers.append(predictions)
        fold_probability_performance = {
            name: forecast_metrics(
                predictions[column].astype(float).tolist(),
                predictions["target"].astype(int).tolist(),
            ).to_mapping()
            for name, column in PROBABILITY_COLUMNS.items()
        }
        folds.append(
            {
                "test_year": test_year,
                "development": _period_contract(development),
                "threshold_validation": _period_contract(validation),
                "test": _period_contract(test),
                "selected_candidate_l2": selected_l2,
                "candidate_validation_log_loss_by_l2": {
                    f"{l2:g}": score for score, l2, _fit in candidates
                },
                "probability_performance": fold_probability_performance,
                "strategies": fold_strategies,
            }
        )

    if not ledgers or not prediction_ledgers:
        raise ValueError(f"no yearly fold had enough history; skipped={skipped}")
    ledger = pd.concat(ledgers, ignore_index=True).sort_values(
        ["strategy", "event_date", "event_id", "fight_id"], kind="stable"
    )
    predictions = pd.concat(prediction_ledgers, ignore_index=True).sort_values(
        ["event_date", "event_id", "fight_id"], kind="stable"
    )
    probability_performance = {
        name: forecast_metrics(
            predictions[column].astype(float).tolist(),
            predictions["target"].astype(int).tolist(),
        ).to_mapping()
        for name, column in PROBABILITY_COLUMNS.items()
    }
    pooled_profitability = {
        strategy: _selected_summary(
            ledger.loc[
                ledger["strategy"].eq(strategy)
                & ledger["qualifies_selected_threshold"]
            ],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        )
        for strategy in STRATEGY_PROBABILITIES
    }
    supported_profitability = {
        strategy: _selected_summary(
            ledger.loc[
                ledger["strategy"].eq(strategy)
                & ledger["qualifies_selected_threshold"]
                & ~ledger["threshold_selection_status"].eq(
                    "fallback_5_percent_insufficient_history"
                )
            ],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        )
        for strategy in STRATEGY_PROBABILITIES
    }
    report = {
        "report_schema_version": 1,
        "experiment_version": "rolling-yearly-fresh-t24-moneyline-v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_only": True,
        "execution_enabled": False,
        "production_action": "none",
        "plain_language_method": (
            "For each test year, fit only on earlier fights. Use the latest 25% "
            "of those earlier event dates to choose the betting cutoff, refit on "
            "all earlier fights, then score the next year. Pool only these "
            "out-of-sample yearly predictions."
        ),
        "horizon": horizon,
        "maximum_quote_age_hours": maximum_quote_age_hours,
        "minimum_other_books": minimum_other_books,
        "available_data_years": years,
        "scored_years": [int(item["test_year"]) for item in folds],
        "skipped_years": skipped,
        "candidate_features": list(CANDIDATE_FEATURES),
        "threshold_grid": [float(value) for value in thresholds],
        "minimum_selection_bets": minimum_selection_bets,
        "minimum_selection_events": minimum_selection_events,
        "coverage": {
            "fresh_value_fights": int(base_values["fight_id"].nunique()),
            "fresh_value_events": int(base_values["event_id"].nunique()),
            "scored_fights": int(predictions["fight_id"].nunique()),
            "scored_events": int(predictions["event_id"].nunique()),
            "books": int(base_values["book_name"].nunique()),
        },
        "probability_performance": probability_performance,
        "probability_difference_intervals_vs_market": {
            name: event_block_difference_interval(
                predictions, column, "market_probability"
            )
            for name, column in PROBABILITY_COLUMNS.items()
            if name != "leave_one_out_market"
        },
        "pooled_profitability": pooled_profitability,
        "pooled_profitability_with_enough_prior_threshold_examples": (
            supported_profitability
        ),
        "yearly_folds": folds,
        "important_limits": [
            "the two market-first features were chosen during prior research, so this is broad retrospective stability evidence rather than a pristine confirmation",
            "historical T-24 means 24 hours before midnight UTC on the source event date, not exactly 24 hours before card start",
            "profit assumes access to the best listed book and excludes limits, rejected bets, line latency, fees, and taxes",
            "year-specific betting cutoffs are selected from earlier outcomes and may still be noisy",
            "pooled_profitability includes the fixed 5% reference fallback when earlier years lack enough qualifying bets; the separate supported summary excludes those years",
            "a frozen prospective paper record is required before changing any betting rule",
        ],
        "database_snapshot": database,
        "inputs": {
            "paired_input": str(paired_input_path),
            "paired_input_sha256": canonical_hash(
                paired_input_path.read_text(encoding="utf-8")
            ),
        },
    }
    return report, ledger, predictions


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
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument("--maximum-quote-age-hours", type=float, default=24.0)
    parser.add_argument("--minimum-other-books", type=int, default=3)
    parser.add_argument("--minimum-development-fights", type=int, default=75)
    parser.add_argument("--minimum-validation-fights", type=int, default=20)
    parser.add_argument("--minimum-test-fights", type=int, default=20)
    parser.add_argument("--minimum-selection-bets", type=int, default=20)
    parser.add_argument("--minimum-selection-events", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--prediction-ledger", type=Path, default=DEFAULT_PREDICTION_LEDGER
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report, ledger, predictions = evaluate_rolling_profitability(
        database_path=arguments.database,
        paired_input_path=arguments.paired_input,
        horizon=arguments.horizon,
        maximum_quote_age_hours=arguments.maximum_quote_age_hours,
        minimum_other_books=arguments.minimum_other_books,
        minimum_development_fights=arguments.minimum_development_fights,
        minimum_validation_fights=arguments.minimum_validation_fights,
        minimum_test_fights=arguments.minimum_test_fights,
        minimum_selection_bets=arguments.minimum_selection_bets,
        minimum_selection_events=arguments.minimum_selection_events,
        bootstrap_samples=arguments.bootstrap_samples,
    )
    if not arguments.dry_run:
        _atomic_write(
            arguments.report,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        _atomic_write(
            arguments.ledger,
            ledger.to_csv(index=False, lineterminator="\n", float_format="%.15g"),
        )
        _atomic_write(
            arguments.prediction_ledger,
            predictions.to_csv(
                index=False, lineterminator="\n", float_format="%.15g"
            ),
        )
    print(
        f"Scored {report['coverage']['scored_fights']} fights across years "
        f"{report['scored_years']} using prices no more than "
        f"{report['maximum_quote_age_hours']:g} hours old."
    )
    for strategy, metrics in report["probability_performance"].items():
        print(
            f"{strategy}: accuracy={100.0 * metrics['accuracy']:.2f}%, "
            f"log loss={metrics['log_loss']:.5f}"
        )
    for strategy, result in report["pooled_profitability"].items():
        roi = result["roi"]
        roi_text = "n/a" if roi is None else f"{100.0 * roi:.2f}%"
        print(
            f"{strategy} bets: {result['selections']}, "
            f"profit={result['profit_units']:.2f}u, ROI={roi_text}"
        )
    if not arguments.dry_run:
        print(f"Report: {arguments.report}")
        print(f"Bet ledger: {arguments.ledger}")
        print(f"Prediction ledger: {arguments.prediction_ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
