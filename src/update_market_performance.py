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
    BETTING_STATUS,
    PaperDecision,
    PaperDecisionStore,
    PaperSettlementStore,
    QuoteSnapshot,
    QuoteSnapshotStore,
    forecast_metrics,
    settle_paper_decision,
    summarize_paper_settlements,
)
from market_tracker._common import canonical_hash, implied_probability


ROOT = Path(__file__).resolve().parent
MARKET_ROOT = ROOT / "content" / "data" / "market"
RAW_PATH = ROOT / "content" / "data" / "processed" / "ufc_fights_reported_doubled.csv"
QUOTE_CSV_PATH = MARKET_ROOT / "quote_snapshots.csv"
QUOTE_JSONL_PATH = MARKET_ROOT / "quote_snapshots.jsonl"
DECISION_CSV_PATH = MARKET_ROOT / "paper_decisions.csv"
DECISION_JSONL_PATH = MARKET_ROOT / "paper_decisions.jsonl"
SETTLEMENT_CSV_PATH = MARKET_ROOT / "paper_settlements.csv"
SETTLEMENT_JSONL_PATH = MARKET_ROOT / "paper_settlements.jsonl"
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


def _result_index(raw: pd.DataFrame) -> tuple[dict[tuple, tuple[int | None, str]], set[str]]:
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
            raise ValueError("raw results contain conflicting stable matchup outcomes")
        outcomes[key] = value
    return outcomes, completed_events


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


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("a quantile requires observations")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


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
    decisions = decision_store.read()
    existing_settlements = settlement_store.read()
    settled_ids = {item.decision_id for item in existing_settlements}
    raw_bytes = RAW_PATH.read_bytes()
    result_hash = sha256(raw_bytes).hexdigest()
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    outcomes, completed_events = _result_index(raw)
    settled_at = datetime.now(timezone.utc).replace(microsecond=0)
    pending = []
    for decision in decisions:
        if decision.decision_id in settled_ids:
            continue
        key = (decision.event_id, decision.fighter_id, decision.opponent_id)
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
    metrics = summarize_paper_settlements(decisions, settlements)
    quotes = quote_store.read()
    model_vs_market = _paired_market_log_loss_interval(
        decisions, settlements, "model_probability"
    )
    blend_vs_market = _paired_market_log_loss_interval(
        decisions, settlements, "blend_probability"
    )
    return_interval = _event_block_return_interval(decisions, settlements)
    clv = _latest_available_clv(decisions, quotes)
    settled_decision_ids = {item.decision_id for item in settlements}
    settled_events = {
        decision.event_id
        for decision in decisions
        if decision.decision_id in settled_decision_ids
    }
    report_body: dict[str, object] = {
        "schema_version": 1,
        "betting_status": BETTING_STATUS,
        "paper_only": True,
        "execution_enabled": False,
        "as_of_utc": max(
            [
                *(item.settled_at_utc for item in settlements),
                *(item.decision_issued_at_utc for item in decisions),
            ],
            default=None,
        ),
        "result_source_sha256": result_hash,
        "decisions_total": len(decisions),
        "settlements_total": len(settlements),
        "unsettled_decisions": len(decisions) - len(settlements),
        "decision_dataset_sha256": _dataset_hash(decisions),
        "settlement_dataset_sha256": _dataset_hash(settlements),
        "paper_metrics": metrics.to_mapping(),
        "forecast_comparators": _forecast_comparators(decisions, settlements),
        "market_relative_log_loss_intervals": {
            "independent_model_vs_market": model_vs_market,
            "locked_blend_vs_market": blend_vs_market,
        },
        "paper_return_interval": return_interval,
        "latest_available_price_clv": clv,
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
