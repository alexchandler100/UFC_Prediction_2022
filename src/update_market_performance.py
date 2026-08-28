"""Settle immutable paper decisions from UFCStats and publish paper metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from statistics import median
import tempfile

import pandas as pd

from market_tracker import (
    BAYESIAN_FILTER_POLICY_VERSION,
    BETTING_STATUS,
    BayesianFilteredDecision,
    BayesianFilteredDecisionStore,
    PaperDecision,
    PaperDecisionStore,
    PaperSettlementStore,
    QuoteSnapshot,
    QuoteSnapshotStore,
    QuoteSourceMetadataStore,
    prospective_comparison_report,
    SimulationComparisonDecisionStore,
    build_simulation_comparison_decisions,
    simulation_comparison_report,
    TotalRoundsPaperDecision,
    TotalRoundsPaperDecisionStore,
    TotalRoundsPaperSettlementStore,
    TotalRoundsQuoteSnapshot,
    TotalRoundsQuoteStore,
    evaluate_timing_policies,
    forecast_metrics,
    settle_paper_decision,
    summarize_paper_settlements,
    settle_total_round_decision,
    summarize_total_round_performance,
)
from market_tracker._common import canonical_hash, implied_probability


ROOT = Path(__file__).resolve().parent
MARKET_ROOT = ROOT / "content" / "data" / "market"
RAW_PATH = ROOT / "content" / "data" / "processed" / "ufc_fights_reported_doubled.csv"
PREDICTION_HISTORY_PATH = ROOT / "content" / "data" / "external" / "prediction_history.json"
SIMULATION_FORECAST_PATH = (
    ROOT / "content" / "data" / "external" / "simulation_forecasts.json"
)
QUOTE_CSV_PATH = MARKET_ROOT / "quote_snapshots.csv"
QUOTE_JSONL_PATH = MARKET_ROOT / "quote_snapshots.jsonl"
SOURCE_METADATA_CSV_PATH = MARKET_ROOT / "quote_source_metadata.csv"
SOURCE_METADATA_JSONL_PATH = MARKET_ROOT / "quote_source_metadata.jsonl"
DECISION_CSV_PATH = MARKET_ROOT / "paper_decisions.csv"
DECISION_JSONL_PATH = MARKET_ROOT / "paper_decisions.jsonl"
BAYESIAN_FILTER_DECISION_CSV_PATH = (
    MARKET_ROOT / "bayesian_filtered_paper_decisions.csv"
)
BAYESIAN_FILTER_DECISION_JSONL_PATH = (
    MARKET_ROOT / "bayesian_filtered_paper_decisions.jsonl"
)
SETTLEMENT_CSV_PATH = MARKET_ROOT / "paper_settlements.csv"
SETTLEMENT_JSONL_PATH = MARKET_ROOT / "paper_settlements.jsonl"
SIMULATION_COMPARISON_CSV_PATH = MARKET_ROOT / "simulation_comparisons.csv"
SIMULATION_COMPARISON_JSONL_PATH = MARKET_ROOT / "simulation_comparisons.jsonl"
TOTAL_ROUNDS_QUOTE_CSV_PATH = MARKET_ROOT / "total_round_quote_snapshots.csv"
TOTAL_ROUNDS_QUOTE_JSONL_PATH = MARKET_ROOT / "total_round_quote_snapshots.jsonl"
TOTAL_ROUNDS_DECISION_CSV_PATH = MARKET_ROOT / "total_round_paper_decisions.csv"
TOTAL_ROUNDS_DECISION_JSONL_PATH = MARKET_ROOT / "total_round_paper_decisions.jsonl"
TOTAL_ROUNDS_SETTLEMENT_CSV_PATH = MARKET_ROOT / "total_round_paper_settlements.csv"
TOTAL_ROUNDS_SETTLEMENT_JSONL_PATH = MARKET_ROOT / "total_round_paper_settlements.jsonl"
REPORT_PATH = MARKET_ROOT / "performance_report.json"
REPORT_SIZE_LIMIT = 64 * 1024


def _identity(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().rstrip("/").rsplit("/", 1)[-1].casefold()


def _result_index(
    raw: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str, str], tuple[int | None, str]],
    set[str],
    set[tuple[str, str, str]],
]:
    """Index unambiguous results without collapsing same-card rematches."""

    required = {
        "event_url",
        "fight_url",
        "fighter_url",
        "opponent_url",
        "result",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"raw result data is missing columns: {missing}")
    outcomes: dict[tuple, tuple[int | None, str]] = {}
    completed_events: set[str] = set()
    ambiguous_matchups: set[tuple[str, str, str]] = set()
    for fight_url, group in raw.groupby("fight_url", sort=False, dropna=False):
        if pd.isna(fight_url) or len(group) != 2:
            continue
        event_ids = {_identity(value) for value in group["event_url"]}
        fighters = {_identity(value) for value in group["fighter_url"]}
        if len(event_ids) != 1 or len(fighters) != 2 or "" in fighters:
            continue
        event_id = next(iter(event_ids))
        completed_events.add(event_id)
        fighter_id, opponent_id = sorted(fighters)
        canonical_rows = group[
            group["fighter_url"].map(_identity).eq(fighter_id)
        ]
        if len(canonical_rows) != 1:
            continue
        raw_result = canonical_rows.iloc[0]["result"]
        result = (
            ""
            if pd.isna(raw_result)
            else str(raw_result).strip().upper()
        )
        target = 1 if result == "W" else 0 if result == "L" else None
        key = (event_id, fighter_id, opponent_id)
        value = (target, _identity(fight_url))
        prior = outcomes.get(key)
        if prior is not None and prior != value:
            # Early tournaments can contain two physical fights between the
            # same competitors on one card (for example Sakuraba-Silveira at
            # UFC Japan). Event/fighter IDs cannot distinguish those fights,
            # so quarantine the key rather than choosing one or aborting an
            # unrelated modern settlement run.
            outcomes.pop(key, None)
            ambiguous_matchups.add(key)
            continue
        if key not in ambiguous_matchups:
            outcomes[key] = value
    return outcomes, completed_events, ambiguous_matchups


def _total_duration_index(
    raw: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str, str], tuple[float, str]],
    set[tuple[str, str, str]],
]:
    """Index terminal W/L fight duration, quarantining same-card rematches."""

    required = {
        "event_url",
        "fight_url",
        "fighter_url",
        "opponent_url",
        "result",
        "total_fight_time",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"raw result data is missing total fields: {missing}")
    durations: dict[tuple[str, str, str], tuple[float, str]] = {}
    ambiguous: set[tuple[str, str, str]] = set()
    for fight_url, group in raw.groupby("fight_url", sort=False, dropna=False):
        if pd.isna(fight_url) or len(group) != 2:
            continue
        results = sorted(str(value).strip().upper() for value in group["result"])
        if results != ["L", "W"]:
            continue
        event_ids = {_identity(value) for value in group["event_url"]}
        fighters = {_identity(value) for value in group["fighter_url"]}
        if len(event_ids) != 1 or len(fighters) != 2 or "" in fighters:
            continue
        duration_values = pd.to_numeric(
            group["total_fight_time"], errors="coerce"
        ).dropna()
        if len(duration_values) != 2 or abs(
            float(duration_values.iloc[0]) - float(duration_values.iloc[1])
        ) > 1e-9:
            continue
        duration = float(duration_values.iloc[0])
        if not math.isfinite(duration) or not 0.0 < duration <= 25.0 * 300.0:
            continue
        event_id = next(iter(event_ids))
        fighter_id, opponent_id = sorted(fighters)
        key = (event_id, fighter_id, opponent_id)
        value = (duration, _identity(fight_url))
        prior = durations.get(key)
        if prior is not None and prior != value:
            durations.pop(key, None)
            ambiguous.add(key)
            continue
        if key not in ambiguous:
            durations[key] = value
    return durations, ambiguous


def _dataset_hash(records: object) -> str:
    return canonical_hash([record.to_mapping() for record in records])


def _latest_available_clv(
    decisions: tuple[PaperDecision, ...],
    quotes: tuple[QuoteSnapshot, ...],
) -> dict[str, object]:
    observations: list[tuple[str, float]] = []
    for decision in decisions:
        if decision.paper_action == "pass":
            continue
        reference = next(
            (item for item in quotes if item.quote_id == decision.reference_quote_id),
            None,
        )
        if reference is None:
            continue
        later = [
            item
            for item in quotes
            if item.matchup_id == decision.matchup_id
            and item.book.casefold() == reference.book.casefold()
            and item.observed_at_utc > decision.market_as_of_utc
        ]
        if not later:
            continue
        closing_proxy = max(
            later, key=lambda item: (item.observed_at_utc, item.quote_id)
        )
        if decision.paper_action == "fighter":
            closing_probability = implied_probability(
                closing_proxy.fighter_moneyline
            )
            decision_probability = decision.fighter_break_even_probability
        else:
            closing_probability = implied_probability(
                closing_proxy.opponent_moneyline
            )
            decision_probability = decision.opponent_break_even_probability
        observations.append(
            (decision.event_id, closing_probability - decision_probability)
        )
    values = [value for _, value in observations]
    grouped: dict[str, tuple[float, int]] = {}
    for event_id, value in observations:
        total, count = grouped.get(event_id, (0.0, 0))
        grouped[event_id] = (total + value, count + 1)
    result: dict[str, object] = {
        "definition": (
            "latest available same-book implied probability minus the locked "
            "decision break-even probability; positive favors the paper price"
        ),
        "count": len(values),
        "event_count": len(grouped),
        "mean_probability_edge": sum(values) / len(values) if values else None,
        "median_probability_edge": median(values) if values else None,
        "positive_rate": (
            sum(value > 0.0 for value in values) / len(values) if values else None
        ),
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(grouped) < 2:
        return result
    blocks = [grouped[key] for key in sorted(grouped)]
    seed = canonical_hash({"clv_blocks": blocks})
    generator = random.Random(int(seed[:16], 16))
    samples: list[float] = []
    for _ in range(10_000):
        selected = [generator.choice(blocks) for _ in blocks]
        count = sum(value[1] for value in selected)
        samples.append(sum(value[0] for value in selected) / count)
    samples.sort()
    result.update(
        {
            "bootstrap_samples": len(samples),
            "ci_95_lower": _quantile(samples, 0.025),
            "ci_95_upper": _quantile(samples, 0.975),
        }
    )
    return result


def _latest_available_total_clv(
    decisions: tuple[TotalRoundsPaperDecision, ...],
    quotes: tuple[TotalRoundsQuoteSnapshot, ...],
) -> dict[str, object]:
    """Latest same-book, same-line price movement after each locked decision."""

    observations: list[tuple[str, float]] = []
    for decision in decisions:
        if decision.paper_action == "pass":
            continue
        later = [
            item
            for item in quotes
            if item.matchup_id == decision.matchup_id
            and float(item.line) == float(decision.line)
            and item.source_book_key.casefold() == decision.target_book_key.casefold()
            and item.observed_at_utc > decision.market_as_of_utc
        ]
        if not later:
            continue
        closing_proxy = max(later, key=lambda item: (item.observed_at_utc, item.quote_id))
        if decision.paper_action == "over":
            later_probability = implied_probability(closing_proxy.over_moneyline)
            decision_probability = decision.over_break_even_probability
        else:
            later_probability = implied_probability(closing_proxy.under_moneyline)
            decision_probability = decision.under_break_even_probability
        observations.append(
            (decision.event_id, later_probability - float(decision_probability))
        )
    values = [value for _, value in observations]
    grouped: dict[str, list[float]] = {}
    for event_id, value in observations:
        grouped.setdefault(event_id, []).append(value)
    result: dict[str, object] = {
        "definition": (
            "latest available same-book, same-total-line implied probability "
            "minus locked break-even probability; positive favors the paper price"
        ),
        "count": len(values),
        "event_count": len(grouped),
        "mean_probability_edge": sum(values) / len(values) if values else None,
        "median_probability_edge": median(values) if values else None,
        "positive_rate": (
            sum(value > 0.0 for value in values) / len(values) if values else None
        ),
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(grouped) < 2:
        return result
    blocks = [grouped[key] for key in sorted(grouped)]
    seed = canonical_hash({"total_clv_blocks": blocks})
    generator = random.Random(int(seed[:16], 16))
    samples: list[float] = []
    for _ in range(10_000):
        selected = [generator.choice(blocks) for _ in blocks]
        sample = [value for block in selected for value in block]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    result.update(
        {
            "bootstrap_samples": len(samples),
            "ci_95_lower": _quantile(samples, 0.025),
            "ci_95_upper": _quantile(samples, 0.975),
        }
    )
    return result


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("a quantile requires observations")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bayesian_prediction_history_performance(
    history: pd.DataFrame,
) -> dict[str, object]:
    """Score frozen weekly Bayesian shadow rows after outcomes become known.

    This is a bounded first-pass monitor. The weekly publication retains the
    offered book/price and posterior before the result, but it is not yet the
    immutable T-24 ledger used by the market policy, so it cannot satisfy the
    CLV or execution promotion requirements.
    """

    policy_version = "bayesian-moneyline-shadow-v1"
    required = {
        "bayesian decision policy", "bayesian model id",
        "bayesian posterior mean", "bayesian paper action",
        "bayesian paper threshold met", "bayesian candidate odds",
        "bayesian candidate book", "bayesian candidate selection",
        "bayesian probability positive ev", "bayesian posterior mean ev",
        "bayesian ev lower", "bayesian ev upper", "forecast status",
        "actual result", "fighter id", "opponent id", "event id", "fight id",
    }
    if history.empty or not required.issubset(history.columns):
        return {
            "policy_version": policy_version,
            "paper_only": True,
            "execution_enabled": False,
            "source": "weekly_prediction_history",
            "source_limit": (
                "awaiting the first completed challenger forecast; not an immutable T-24 ledger"
            ),
            "scored_forecasts": 0,
            "settled_shadow_selections": 0,
            "wins": 0,
            "losses": 0,
            "hypothetical_profit_units": 0.0,
            "hypothetical_risk_units": 0.0,
            "hypothetical_roi": None,
            "forecast_metrics": None,
            "return_interval": {
                "event_count": 0,
                "bootstrap_samples": 0,
                "ci_95_lower": None,
                "ci_95_upper": None,
            },
            "dataset_sha256": canonical_hash([]),
            "promotion_gate": {
                "status": "collecting_prospective_evidence",
                "minimum_scored_fights": 500,
                "minimum_settled_events": 40,
                "minimum_shadow_selections": 100,
                "immutable_t24_ledger_requirement_met": False,
                "positive_clv_requirement_met": False,
                "positive_return_requirement_met": False,
                "execution_enabled": False,
            },
        }
    policy_rows = history[
        history["bayesian decision policy"].astype(str).eq(policy_version)
        & history["bayesian model id"].astype(str).str.strip().ne("")
    ].copy()
    scored = policy_rows[
        policy_rows["forecast status"].astype(str).eq("completed")
        & policy_rows["actual result"].astype(str).isin(["W", "L"])
    ].copy()
    probabilities = pd.to_numeric(
        scored.get("bayesian posterior mean"), errors="coerce"
    )
    valid_probability = probabilities.between(0, 1, inclusive="neither")
    scored = scored.loc[valid_probability].copy()
    probabilities = probabilities.loc[valid_probability]
    targets = scored["actual result"].astype(str).eq("W").astype(int)
    metrics = (
        forecast_metrics(probabilities.tolist(), targets.tolist()).to_mapping()
        if len(scored)
        else None
    )

    selected = scored[
        scored["bayesian paper threshold met"].astype(bool)
        & scored["bayesian paper action"].astype(str).isin(["fighter", "opponent"])
    ].copy()
    profits: list[float] = []
    event_profits: dict[str, list[float]] = {}
    audit_rows: list[dict[str, object]] = []
    wins = 0
    for _, row in selected.iterrows():
        try:
            line = int(float(row["bayesian candidate odds"]))
        except (TypeError, ValueError) as error:
            raise ValueError("Bayesian shadow selection has an invalid price") from error
        if abs(line) < 100 or line == 0:
            raise ValueError("Bayesian shadow selection has an invalid American price")
        fighter_won = str(row["actual result"]) == "W"
        selected_fighter = str(row["bayesian paper action"]) == "fighter"
        won = fighter_won == selected_fighter
        profit = (line / 100.0 if line > 0 else 100.0 / abs(line)) if won else -1.0
        wins += int(won)
        profits.append(float(profit))
        event_key = str(row.get("event id") or row.get("date") or "unknown")
        event_profits.setdefault(event_key, []).append(float(profit))
        audit_rows.append(
            {
                "fight_id": str(row.get("fight id") or ""),
                "event_id": str(row.get("event id") or ""),
                "fighter_id": str(row.get("fighter id") or ""),
                "opponent_id": str(row.get("opponent id") or ""),
                "model_id": str(row.get("bayesian model id") or ""),
                "action": str(row["bayesian paper action"]),
                "book": str(row["bayesian candidate book"]),
                "moneyline": line,
                "posterior_mean": float(row["bayesian posterior mean"]),
                "posterior_mean_ev": float(row["bayesian posterior mean ev"]),
                "probability_positive_ev": float(
                    row["bayesian probability positive ev"]
                ),
                "target": int(fighter_won),
                "profit_units": float(profit),
            }
        )
    interval: dict[str, object] = {
        "definition": "whole-card bootstrap of one-unit Bayesian shadow selections",
        "event_count": len(event_profits),
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(event_profits) >= 2 and profits:
        blocks = [event_profits[key] for key in sorted(event_profits)]
        generator = random.Random(
            int(canonical_hash({"bayesian_blocks": blocks})[:16], 16)
        )
        samples = []
        for _ in range(10_000):
            chosen = [generator.choice(blocks) for _ in blocks]
            flattened = [value for block in chosen for value in block]
            samples.append(sum(flattened) / len(flattened))
        samples.sort()
        interval.update(
            {
                "bootstrap_samples": len(samples),
                "ci_95_lower": _quantile(samples, 0.025),
                "ci_95_upper": _quantile(samples, 0.975),
            }
        )
    risk = float(len(profits))
    profit_total = float(sum(profits))
    settled_events = len(
        {
            str(row.get("event id") or row.get("date") or "unknown")
            for _, row in scored.iterrows()
        }
    )
    return {
        "policy_version": policy_version,
        "paper_only": True,
        "execution_enabled": False,
        "source": "weekly_prediction_history",
        "source_limit": (
            "timestamped weekly publication; not yet an immutable T-24 decision/CLV ledger"
        ),
        "scored_forecasts": len(scored),
        "settled_events": settled_events,
        "settled_shadow_selections": len(profits),
        "wins": wins,
        "losses": len(profits) - wins,
        "hypothetical_profit_units": profit_total,
        "hypothetical_risk_units": risk,
        "hypothetical_roi": profit_total / risk if risk else None,
        "forecast_metrics": metrics,
        "return_interval": interval,
        "dataset_sha256": canonical_hash(audit_rows),
        "promotion_gate": {
            "status": "collecting_prospective_evidence",
            "minimum_scored_fights": 500,
            "minimum_settled_events": 40,
            "minimum_shadow_selections": 100,
            "scored_fights": len(scored),
            "settled_events": settled_events,
            "shadow_selections": len(profits),
            "count_requirements_met": (
                len(scored) >= 500
                and settled_events >= 40
                and len(profits) >= 100
            ),
            "immutable_t24_ledger_requirement_met": False,
            "positive_clv_requirement_met": False,
            "positive_return_requirement_met": (
                interval["ci_95_lower"] is not None
                and float(interval["ci_95_lower"]) > 0.0
            ),
            "execution_enabled": False,
        },
    }


def _event_block_return_interval(
    decisions: tuple[PaperDecision, ...], settlements: tuple
) -> dict[str, object]:
    """Deterministic card-block bootstrap for one-unit paper-selection return."""

    event_by_decision = {item.decision_id: item.event_id for item in decisions}
    grouped: dict[str, tuple[float, float]] = {}
    for settlement in settlements:
        if float(settlement.hypothetical_risk_units) <= 0.0:
            continue
        event_id = event_by_decision[settlement.decision_id]
        profit, risk = grouped.get(event_id, (0.0, 0.0))
        grouped[event_id] = (
            profit + float(settlement.hypothetical_profit_units),
            risk + float(settlement.hypothetical_risk_units),
        )
    total_profit = sum(value[0] for value in grouped.values())
    total_risk = sum(value[1] for value in grouped.values())
    result: dict[str, object] = {
        "definition": "paired whole-card bootstrap of profit per one-unit paper selection",
        "event_count": len(grouped),
        "selection_count": int(total_risk),
        "observed_profit_per_selection": (
            total_profit / total_risk if total_risk else None
        ),
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(grouped) < 2 or total_risk == 0.0:
        return result
    blocks = [grouped[key] for key in sorted(grouped)]
    seed_payload = [
        item.to_mapping()
        for item in sorted(settlements, key=lambda value: value.settlement_id)
    ]
    generator = random.Random(int(canonical_hash(seed_payload)[:16], 16))
    samples: list[float] = []
    for _ in range(10_000):
        selected = [generator.choice(blocks) for _ in blocks]
        risk = sum(value[1] for value in selected)
        if risk:
            samples.append(sum(value[0] for value in selected) / risk)
    samples.sort()
    result.update(
        {
            "bootstrap_samples": len(samples),
            "ci_95_lower": _quantile(samples, 0.025),
            "ci_95_upper": _quantile(samples, 0.975),
        }
    )
    return result


def _paired_bayesian_filter_return_interval(
    rows: list[tuple[BayesianFilteredDecision, PaperDecision, object]],
) -> dict[str, object]:
    """Compare base and filtered ROI with paired whole-card resampling."""

    grouped: dict[str, list[float]] = {}
    for filtered, base, settlement in rows:
        values = grouped.setdefault(base.event_id, [0.0, 0.0, 0.0, 0.0])
        base_risk = float(settlement.hypothetical_risk_units)
        base_profit = float(settlement.hypothetical_profit_units)
        values[0] += base_profit
        values[1] += base_risk
        if filtered.filter_status == "qualified":
            values[2] += base_profit
            values[3] += base_risk
    base_profit = sum(values[0] for values in grouped.values())
    base_risk = sum(values[1] for values in grouped.values())
    filtered_profit = sum(values[2] for values in grouped.values())
    filtered_risk = sum(values[3] for values in grouped.values())
    base_roi = base_profit / base_risk if base_risk else None
    filtered_roi = filtered_profit / filtered_risk if filtered_risk else None
    result: dict[str, object] = {
        "definition": (
            "paired whole-card bootstrap of filtered ROI minus the existing "
            "moneyline policy ROI on the post-deployment cohort"
        ),
        "event_count": len(grouped),
        "base_selection_count": int(base_risk),
        "filtered_selection_count": int(filtered_risk),
        "base_roi": base_roi,
        "filtered_roi": filtered_roi,
        "point_difference": (
            filtered_roi - base_roi
            if filtered_roi is not None and base_roi is not None
            else None
        ),
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(grouped) < 2 or base_risk == 0.0 or filtered_risk == 0.0:
        return result
    blocks = [grouped[key] for key in sorted(grouped)]
    seed = canonical_hash(
        [
            {
                "filtered_decision_id": filtered.filtered_decision_id,
                "settlement_id": settlement.settlement_id,
            }
            for filtered, _, settlement in sorted(
                rows, key=lambda item: item[0].filtered_decision_id
            )
        ]
    )
    generator = random.Random(int(seed[:16], 16))
    samples: list[float] = []
    for _ in range(10_000):
        selected = [generator.choice(blocks) for _ in blocks]
        sampled_base_risk = sum(value[1] for value in selected)
        sampled_filtered_risk = sum(value[3] for value in selected)
        if sampled_base_risk and sampled_filtered_risk:
            samples.append(
                sum(value[2] for value in selected) / sampled_filtered_risk
                - sum(value[0] for value in selected) / sampled_base_risk
            )
    samples.sort()
    if samples:
        result.update(
            {
                "bootstrap_samples": len(samples),
                "ci_95_lower": _quantile(samples, 0.025),
                "ci_95_upper": _quantile(samples, 0.975),
            }
        )
    return result


def _bayesian_filtered_policy_performance(
    filtered_decisions: tuple[BayesianFilteredDecision, ...],
    base_decisions: tuple[PaperDecision, ...],
    settlements: tuple,
    quotes: tuple[QuoteSnapshot, ...],
) -> dict[str, object]:
    """Score the Bayesian veto and its unchanged-policy comparison cohort."""

    base_by_id = {item.decision_id: item for item in base_decisions}
    settlement_by_id = {item.decision_id: item for item in settlements}
    if len(base_by_id) != len(base_decisions):
        raise ValueError("base moneyline decision IDs are not unique")
    if len(settlement_by_id) != len(settlements):
        raise ValueError("moneyline settlement decision IDs are not unique")
    if len({item.base_decision_id for item in filtered_decisions}) != len(
        filtered_decisions
    ):
        raise ValueError("Bayesian filter contains duplicate base decisions")
    rows: list[tuple[BayesianFilteredDecision, PaperDecision, object]] = []
    for filtered in filtered_decisions:
        base = base_by_id.get(filtered.base_decision_id)
        if base is None:
            raise ValueError("Bayesian filter references an unknown base decision")
        settlement = settlement_by_id.get(base.decision_id)
        if settlement is not None:
            rows.append((filtered, base, settlement))

    def strategy_summary(*, filtered: bool) -> dict[str, object]:
        selected: list[object] = []
        for filter_decision, _, settlement in rows:
            if filtered and filter_decision.filter_status != "qualified":
                continue
            if float(settlement.hypothetical_risk_units) > 0.0:
                selected.append(settlement)
        profit = sum(float(item.hypothetical_profit_units) for item in selected)
        risk = sum(float(item.hypothetical_risk_units) for item in selected)
        return {
            "selections": len(selected),
            "wins": sum(float(item.hypothetical_profit_units) > 0.0 for item in selected),
            "losses": sum(float(item.hypothetical_profit_units) < 0.0 for item in selected),
            "hypothetical_profit_units": profit,
            "hypothetical_risk_units": risk,
            "hypothetical_roi": profit / risk if risk else None,
        }

    base_summary = strategy_summary(filtered=False)
    filtered_summary = strategy_summary(filtered=True)
    qualified_ids = {
        item.base_decision_id
        for item in filtered_decisions
        if item.filter_status == "qualified"
    }
    qualified_base = tuple(
        item for item in base_decisions if item.decision_id in qualified_ids
    )
    qualified_settlements = tuple(
        item for item in settlements if item.decision_id in qualified_ids
    )
    filtered_return = _event_block_return_interval(
        qualified_base, qualified_settlements
    )
    paired_return = _paired_bayesian_filter_return_interval(rows)
    filtered_clv = _latest_available_clv(qualified_base, quotes)
    settled_events = len({base.event_id for _, base, _ in rows})
    veto_counts = {
        status: sum(item.filter_status == status for item in filtered_decisions)
        for status in (
            "base_policy_pass",
            "bayesian_status_veto",
            "bayesian_mean_ev_veto",
            "bayesian_probability_veto",
            "qualified",
        )
    }
    count_requirements_met = (
        len(rows) >= 500
        and settled_events >= 40
        and int(filtered_summary["selections"]) >= 100
    )
    return {
        "policy_version": BAYESIAN_FILTER_POLICY_VERSION,
        "paper_only": True,
        "execution_enabled": False,
        "source": "immutable_t24_decision_ledger",
        "decision_count": len(filtered_decisions),
        "paired_settled_decisions": len(rows),
        "settled_events": settled_events,
        "veto_counts": veto_counts,
        "base_policy_on_same_cohort": base_summary,
        "bayesian_filtered_policy": filtered_summary,
        "filtered_return_interval": filtered_return,
        "paired_roi_difference": paired_return,
        "latest_available_price_clv": filtered_clv,
        "decision_dataset_sha256": _dataset_hash(filtered_decisions),
        "promotion_gate": {
            "status": "collecting_prospective_evidence",
            "minimum_paired_settled_decisions": 500,
            "minimum_settled_events": 40,
            "minimum_filtered_selections": 100,
            "paired_settled_decisions": len(rows),
            "settled_events": settled_events,
            "filtered_selections": filtered_summary["selections"],
            "count_requirements_met": count_requirements_met,
            "positive_filtered_return_requirement_met": (
                filtered_return["ci_95_lower"] is not None
                and float(filtered_return["ci_95_lower"]) > 0.0
            ),
            "improves_base_policy_roi_requirement_met": (
                paired_return["ci_95_lower"] is not None
                and float(paired_return["ci_95_lower"]) > 0.0
            ),
            "positive_clv_requirement_met": (
                filtered_clv["ci_95_lower"] is not None
                and float(filtered_clv["ci_95_lower"]) > 0.0
            ),
            "execution_enabled": False,
        },
    }


def _forecast_comparators(
    decisions: tuple[PaperDecision, ...], settlements: tuple
) -> dict[str, object]:
    """Score market, independent model, and locked blend on identical fights."""

    decision_by_id = {item.decision_id: item for item in decisions}
    scored = [item for item in settlements if item.target is not None]
    if not scored:
        return {
            "paired_fights": 0,
            "market": None,
            "independent_model": None,
            "locked_blend": None,
            "model_minus_market_log_loss": None,
            "blend_minus_market_log_loss": None,
        }
    targets = [int(item.target) for item in scored]
    market = forecast_metrics(
        [decision_by_id[item.decision_id].market_probability for item in scored],
        targets,
    )
    model = forecast_metrics(
        [decision_by_id[item.decision_id].model_probability for item in scored],
        targets,
    )
    blend = forecast_metrics(
        [decision_by_id[item.decision_id].blend_probability for item in scored],
        targets,
    )
    return {
        "paired_fights": len(scored),
        "market": market.to_mapping(),
        "independent_model": model.to_mapping(),
        "locked_blend": blend.to_mapping(),
        "model_minus_market_log_loss": model.log_loss - market.log_loss,
        "blend_minus_market_log_loss": blend.log_loss - market.log_loss,
    }


def _paired_market_log_loss_interval(
    decisions: tuple[PaperDecision, ...], settlements: tuple, probability_field: str
) -> dict[str, object]:
    """Card-block interval for candidate-minus-market paired log loss."""

    decision_by_id = {item.decision_id: item for item in decisions}
    blocks: dict[str, tuple[float, int]] = {}
    for settlement in settlements:
        if settlement.target is None:
            continue
        decision = decision_by_id[settlement.decision_id]
        target = int(settlement.target)
        market_probability = decision.market_probability
        candidate_probability = float(getattr(decision, probability_field))
        market_target_probability = (
            market_probability if target == 1 else 1.0 - market_probability
        )
        candidate_target_probability = (
            candidate_probability if target == 1 else 1.0 - candidate_probability
        )
        difference = -math.log(candidate_target_probability) + math.log(
            market_target_probability
        )
        total, count = blocks.get(decision.event_id, (0.0, 0))
        blocks[decision.event_id] = (total + difference, count + 1)
    total_count = sum(value[1] for value in blocks.values())
    point = (
        sum(value[0] for value in blocks.values()) / total_count
        if total_count
        else None
    )
    result: dict[str, object] = {
        "definition": "candidate minus market paired log loss; negative favors candidate",
        "event_count": len(blocks),
        "fight_count": total_count,
        "point_difference": point,
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(blocks) < 2 or total_count == 0:
        return result
    ordered = [blocks[key] for key in sorted(blocks)]
    seed = canonical_hash(
        {"probability_field": probability_field, "blocks": ordered}
    )
    generator = random.Random(int(seed[:16], 16))
    samples: list[float] = []
    for _ in range(10_000):
        selected = [generator.choice(ordered) for _ in ordered]
        count = sum(value[1] for value in selected)
        samples.append(sum(value[0] for value in selected) / count)
    samples.sort()
    result.update(
        {
            "bootstrap_samples": len(samples),
            "ci_95_lower": _quantile(samples, 0.025),
            "ci_95_upper": _quantile(samples, 0.975),
        }
    )
    return result


def _atomic_report(report: dict[str, object]) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if len(encoded.encode("utf-8")) > REPORT_SIZE_LIMIT:
        raise ValueError("paper performance report exceeded its size limit")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{REPORT_PATH.name}.", suffix=".tmp", dir=REPORT_PATH.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, REPORT_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def update_market_performance() -> dict[str, object]:
    decision_store = PaperDecisionStore(DECISION_CSV_PATH, DECISION_JSONL_PATH)
    settlement_store = PaperSettlementStore(
        SETTLEMENT_CSV_PATH, SETTLEMENT_JSONL_PATH
    )
    quote_store = QuoteSnapshotStore(QUOTE_CSV_PATH, QUOTE_JSONL_PATH)
    metadata_store = QuoteSourceMetadataStore(
        SOURCE_METADATA_CSV_PATH, SOURCE_METADATA_JSONL_PATH
    )
    decisions = decision_store.read()
    simulation_comparison_store = SimulationComparisonDecisionStore(
        SIMULATION_COMPARISON_CSV_PATH,
        SIMULATION_COMPARISON_JSONL_PATH,
    )
    existing_simulation_comparisons = simulation_comparison_store.read()
    simulation_publication: dict[str, object] | None = None
    if SIMULATION_FORECAST_PATH.is_file():
        try:
            loaded_simulation = json.loads(
                SIMULATION_FORECAST_PATH.read_text(encoding="utf-8")
            )
            simulation_publication = (
                loaded_simulation if isinstance(loaded_simulation, dict) else {}
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            simulation_publication = {}
    comparison_build = build_simulation_comparison_decisions(
        decisions,
        existing_simulation_comparisons,
        simulation_publication,
        comparison_issued_at_utc=datetime.now(timezone.utc),
    )
    simulation_comparison_store.append(comparison_build.decisions)
    simulation_comparisons = simulation_comparison_store.read()
    bayesian_filter_exists = (
        BAYESIAN_FILTER_DECISION_CSV_PATH.exists(),
        BAYESIAN_FILTER_DECISION_JSONL_PATH.exists(),
    )
    if any(bayesian_filter_exists) and not all(bayesian_filter_exists):
        raise ValueError("Bayesian filtered decision mirrors are incomplete")
    bayesian_filtered_decisions = (
        BayesianFilteredDecisionStore(
            BAYESIAN_FILTER_DECISION_CSV_PATH,
            BAYESIAN_FILTER_DECISION_JSONL_PATH,
        ).read()
        if all(bayesian_filter_exists)
        else ()
    )
    existing_settlements = settlement_store.read()
    settled_ids = {item.decision_id for item in existing_settlements}
    total_decision_contract = (
        TOTAL_ROUNDS_DECISION_CSV_PATH.exists()
        or TOTAL_ROUNDS_DECISION_JSONL_PATH.exists()
    )
    if total_decision_contract and not (
        TOTAL_ROUNDS_DECISION_CSV_PATH.exists()
        and TOTAL_ROUNDS_DECISION_JSONL_PATH.exists()
    ):
        raise ValueError("total-round paper decision mirrors are incomplete")
    raw_bytes = RAW_PATH.read_bytes()
    result_hash = sha256(raw_bytes).hexdigest()
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    prediction_history = (
        pd.read_json(PREDICTION_HISTORY_PATH)
        if PREDICTION_HISTORY_PATH.exists()
        else pd.DataFrame()
    )
    bayesian_performance = _bayesian_prediction_history_performance(
        prediction_history
    )
    outcomes, completed_events, ambiguous_matchups = _result_index(raw)
    total_durations, ambiguous_total_matchups = (
        _total_duration_index(raw) if total_decision_contract else ({}, set())
    )
    settled_at = datetime.now(timezone.utc).replace(microsecond=0)
    pending = []
    ambiguous_result_decisions = 0
    for decision in decisions:
        if decision.decision_id in settled_ids:
            continue
        key = (decision.event_id, decision.fighter_id, decision.opponent_id)
        if key in ambiguous_matchups:
            ambiguous_result_decisions += 1
            continue
        result = outcomes.get(key)
        if result is None and decision.event_id not in completed_events:
            continue
        target, fight_id = result if result is not None else (None, None)
        pending.append(
            settle_paper_decision(
                decision,
                target=target,
                fight_id=fight_id,
                settled_at_utc=settled_at,
                result_source_sha256=result_hash,
            )
        )
    settlement_store.append(pending)
    settlements = settlement_store.read()

    total_decisions = ()
    total_settlements = ()
    if total_decision_contract:
        total_decision_store = TotalRoundsPaperDecisionStore(
            TOTAL_ROUNDS_DECISION_CSV_PATH,
            TOTAL_ROUNDS_DECISION_JSONL_PATH,
        )
        total_settlement_store = TotalRoundsPaperSettlementStore(
            TOTAL_ROUNDS_SETTLEMENT_CSV_PATH,
            TOTAL_ROUNDS_SETTLEMENT_JSONL_PATH,
        )
        total_decisions = total_decision_store.read()
        existing_total_settlements = total_settlement_store.read()
        settled_total_ids = {
            item.decision_id for item in existing_total_settlements
        }
        total_pending = []
        for decision in total_decisions:
            if decision.decision_id in settled_total_ids:
                continue
            key = (decision.event_id, decision.fighter_id, decision.opponent_id)
            if key in ambiguous_total_matchups:
                continue
            result = total_durations.get(key)
            if result is None and decision.event_id not in completed_events:
                continue
            duration, fight_id = result if result is not None else (None, None)
            total_pending.append(
                settle_total_round_decision(
                    decision,
                    total_fight_seconds=duration,
                    fight_id=fight_id,
                    settled_at_utc=settled_at,
                    result_source_sha256=result_hash,
                )
            )
        total_settlement_store.append(total_pending)
        total_settlements = total_settlement_store.read()

    metrics = summarize_paper_settlements(decisions, settlements)
    quotes = quote_store.read()
    source_metadata = metadata_store.read()
    timing_experiment = evaluate_timing_policies(
        quotes, source_metadata, outcomes
    )
    model_vs_market = _paired_market_log_loss_interval(
        decisions, settlements, "model_probability"
    )
    blend_vs_market = _paired_market_log_loss_interval(
        decisions, settlements, "blend_probability"
    )
    return_interval = _event_block_return_interval(decisions, settlements)
    clv = _latest_available_clv(decisions, quotes)
    bayesian_filtered_performance = _bayesian_filtered_policy_performance(
        bayesian_filtered_decisions,
        decisions,
        settlements,
        quotes,
    )
    total_quote_exists = (
        TOTAL_ROUNDS_QUOTE_CSV_PATH.exists(),
        TOTAL_ROUNDS_QUOTE_JSONL_PATH.exists(),
    )
    if any(total_quote_exists) and not all(total_quote_exists):
        raise ValueError("total-round quote mirrors are incomplete")
    total_quote_contract = all(total_quote_exists)
    total_quotes = (
        TotalRoundsQuoteStore(
            TOTAL_ROUNDS_QUOTE_CSV_PATH,
            TOTAL_ROUNDS_QUOTE_JSONL_PATH,
        ).read()
        if total_quote_contract
        else ()
    )
    total_performance = summarize_total_round_performance(
        total_decisions, total_settlements
    )
    total_clv = _latest_available_total_clv(total_decisions, total_quotes)
    total_return_interval = _event_block_return_interval(
        total_decisions, total_settlements
    )
    total_settled_ids = {item.decision_id for item in total_settlements}
    total_settled_events = {
        decision.event_id
        for decision in total_decisions
        if decision.decision_id in total_settled_ids
    }
    total_official = total_performance["official_strategy"]
    total_residual_selection = total_performance["next_residual_weight_selection"]
    total_performance["latest_available_price_clv"] = total_clv
    total_performance["paper_return_interval"] = total_return_interval
    total_performance["decision_dataset_sha256"] = _dataset_hash(total_decisions)
    total_performance["settlement_dataset_sha256"] = _dataset_hash(total_settlements)
    total_performance["quote_dataset_sha256"] = _dataset_hash(total_quotes)
    total_performance["promotion_gate"] = {
        "status": "collecting_prospective_evidence",
        "minimum_scored_lines": 300,
        "minimum_settled_events": 30,
        "minimum_paper_selections": 100,
        "scored_lines": total_performance["scored_forecasts"],
        "settled_events": len(total_settled_events),
        "paper_selections": total_official["selections"],
        "count_requirements_met": (
            int(total_performance["scored_forecasts"]) >= 300
            and len(total_settled_events) >= 30
            and int(total_official["selections"]) >= 100
        ),
        "residual_market_log_loss_requirement_met": (
            total_residual_selection["selection_status"]
            == "residual_weight_promoted"
            and total_residual_selection["ci_95_upper"] is not None
            and float(total_residual_selection["ci_95_upper"]) < 0.0
        ),
        "paper_return_requirement_met": (
            total_return_interval["ci_95_lower"] is not None
            and float(total_return_interval["ci_95_lower"]) > 0.0
        ),
        "positive_clv_requirement_met": (
            total_clv["ci_95_lower"] is not None
            and float(total_clv["ci_95_lower"]) > 0.0
        ),
        "execution_enabled": False,
    }
    settled_decision_ids = {item.decision_id for item in settlements}
    settled_events = {
        decision.event_id
        for decision in decisions
        if decision.decision_id in settled_decision_ids
    }
    report_body: dict[str, object] = {
        "schema_version": 5,
        "betting_status": BETTING_STATUS,
        "paper_only": True,
        "execution_enabled": False,
        "as_of_utc": max(
            [
                *(item.settled_at_utc for item in settlements),
                *(item.decision_issued_at_utc for item in decisions),
                *(item.observed_at_utc for item in quotes),
                *(item.settled_at_utc for item in total_settlements),
                *(item.decision_issued_at_utc for item in total_decisions),
                *(item.observed_at_utc for item in total_quotes),
                *(
                    item.decision_issued_at_utc
                    for item in bayesian_filtered_decisions
                ),
            ],
            default=None,
        ),
        "result_source_sha256": result_hash,
        "decisions_total": len(decisions),
        "settlements_total": len(settlements),
        "unsettled_decisions": len(decisions) - len(settlements),
        "ambiguous_result_decisions": ambiguous_result_decisions,
        "ambiguous_historical_matchup_keys": len(ambiguous_matchups),
        "decision_dataset_sha256": _dataset_hash(decisions),
        "settlement_dataset_sha256": _dataset_hash(settlements),
        "quote_dataset_sha256": _dataset_hash(quotes),
        "source_metadata_dataset_sha256": _dataset_hash(source_metadata),
        "paper_metrics": metrics.to_mapping(),
        "forecast_comparators": _forecast_comparators(decisions, settlements),
        "prospective_model_market_comparison": prospective_comparison_report(
            decisions, settlements
        ),
        "prospective_simulation_comparison": simulation_comparison_report(
            simulation_comparisons,
            settlements,
            decisions,
        ),
        "simulation_comparison_dataset_sha256": _dataset_hash(
            simulation_comparisons
        ),
        "market_relative_log_loss_intervals": {
            "independent_model_vs_market": model_vs_market,
            "locked_blend_vs_market": blend_vs_market,
        },
        "paper_return_interval": return_interval,
        "latest_available_price_clv": clv,
        "entry_timing_experiment": timing_experiment,
        "bayesian_moneyline_challenger": bayesian_performance,
        "bayesian_filtered_moneyline_policy": bayesian_filtered_performance,
        "promotion_gate": {
            "status": "collecting_prospective_evidence",
            "minimum_scored_fights": 500,
            "minimum_settled_events": 40,
            "minimum_paper_selections": 100,
            "scored_fights": metrics.scored_forecasts,
            "settled_events": len(settled_events),
            "paper_selections": metrics.paper_selections,
            "count_requirements_met": (
                metrics.scored_forecasts >= 500
                and len(settled_events) >= 40
                and metrics.paper_selections >= 100
            ),
            "blend_market_log_loss_requirement_met": (
                blend_vs_market["point_difference"] is not None
                and float(blend_vs_market["point_difference"]) <= -0.005
                and blend_vs_market["ci_95_upper"] is not None
                and float(blend_vs_market["ci_95_upper"]) < 0.0
            ),
            "paper_return_requirement_met": (
                return_interval["ci_95_lower"] is not None
                and float(return_interval["ci_95_lower"]) > 0.0
            ),
            "positive_clv_requirement_met": (
                clv["ci_95_lower"] is not None
                and float(clv["ci_95_lower"]) > 0.0
            ),
            "execution_enabled": False,
        },
        "total_rounds": total_performance,
    }
    report_body["report_sha256"] = canonical_hash(report_body)
    _atomic_report(report_body)
    return report_body


def main() -> int:
    report = update_market_performance()
    print(
        "Paper performance updated: "
        f"{report['settlements_total']}/{report['decisions_total']} settled; "
        f"betting={BETTING_STATUS}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
