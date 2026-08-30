"""Capture and settle the frozen market-first prospective paper experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from market_tracker import (
    ForecastCaptureStore,
    QuoteSnapshotStore,
    QuoteSourceMetadataStore,
)
from market_tracker._common import canonical_hash
from market_tracker._storage import atomic_write_text
from market_tracker.market_first_paper import (
    FrozenMarketFirstPolicy,
    MarketFirstPaperDecision,
    MarketFirstPaperDecisionStore,
    PaperSettlementStore,
    build_market_first_decisions,
    settle_market_first_decision,
    summarize_market_first_paper,
)
from update_market_performance import _result_index


ROOT = Path(__file__).resolve().parent
MARKET_ROOT = ROOT / "content" / "data" / "market"
POLICY_PATH = (
    ROOT / "content" / "data" / "model_research" / "market_first_t24_policy.json"
)
RAW_PATH = ROOT / "content" / "data" / "processed" / "ufc_fights_reported_doubled.csv"
QUOTE_CSV_PATH = MARKET_ROOT / "quote_snapshots.csv"
QUOTE_JSONL_PATH = MARKET_ROOT / "quote_snapshots.jsonl"
FORECAST_CSV_PATH = MARKET_ROOT / "forecast_captures.csv"
FORECAST_JSONL_PATH = MARKET_ROOT / "forecast_captures.jsonl"
METADATA_CSV_PATH = MARKET_ROOT / "quote_source_metadata.csv"
METADATA_JSONL_PATH = MARKET_ROOT / "quote_source_metadata.jsonl"
DECISION_CSV_PATH = MARKET_ROOT / "market_first_paper_decisions.csv"
DECISION_JSONL_PATH = MARKET_ROOT / "market_first_paper_decisions.jsonl"
SETTLEMENT_CSV_PATH = MARKET_ROOT / "market_first_paper_settlements.csv"
SETTLEMENT_JSONL_PATH = MARKET_ROOT / "market_first_paper_settlements.jsonl"
REPORT_PATH = MARKET_ROOT / "market_first_paper_report.json"


def _stores() -> tuple[
    QuoteSnapshotStore,
    ForecastCaptureStore,
    QuoteSourceMetadataStore,
    MarketFirstPaperDecisionStore,
    PaperSettlementStore,
]:
    return (
        QuoteSnapshotStore(QUOTE_CSV_PATH, QUOTE_JSONL_PATH),
        ForecastCaptureStore(FORECAST_CSV_PATH, FORECAST_JSONL_PATH),
        QuoteSourceMetadataStore(METADATA_CSV_PATH, METADATA_JSONL_PATH),
        MarketFirstPaperDecisionStore(DECISION_CSV_PATH, DECISION_JSONL_PATH),
        PaperSettlementStore(SETTLEMENT_CSV_PATH, SETTLEMENT_JSONL_PATH),
    )


def _build_new_decisions(
    *,
    policy: FrozenMarketFirstPolicy,
    quotes: tuple,
    forecasts: tuple,
    metadata: tuple,
    existing: tuple[MarketFirstPaperDecision, ...],
) -> tuple[tuple[MarketFirstPaperDecision, ...], dict[str, int]]:
    quotes_by_capture: dict[str, list] = defaultdict(list)
    for quote in quotes:
        quotes_by_capture[quote.capture_id].append(quote)
    forecasts_by_capture: dict[str, list] = defaultdict(list)
    for forecast in forecasts:
        forecasts_by_capture[forecast.capture_id].append(forecast)
    metadata_by_capture: dict[str, list] = defaultdict(list)
    for item in metadata:
        metadata_by_capture[item.capture_id].append(item)

    pending: list[MarketFirstPaperDecision] = []
    counters = {
        "captures_considered": 0,
        "captures_at_locked_horizon": 0,
        "matchups_considered": 0,
        "matchups_already_frozen": 0,
        "matchups_without_fresh_quotes": 0,
        "matchups_without_forecast": 0,
    }
    ordered_captures = sorted(
        quotes_by_capture,
        key=lambda capture_id: (
            min(item.observed_at_utc for item in quotes_by_capture[capture_id]),
            capture_id,
        ),
    )
    frozen: tuple[MarketFirstPaperDecision, ...] = existing
    for capture_id in ordered_captures:
        counters["captures_considered"] += 1
        build = build_market_first_decisions(
            quotes_by_capture[capture_id],
            forecasts_by_capture.get(capture_id, ()),
            metadata_by_capture.get(capture_id, ()),
            policy=policy,
            existing_decisions=frozen,
        )
        counters["captures_at_locked_horizon"] += int(build.eligible_horizon)
        counters["matchups_considered"] += build.matchups_considered
        counters["matchups_already_frozen"] += build.matchups_already_frozen
        counters["matchups_without_fresh_quotes"] += (
            build.matchups_without_fresh_quotes
        )
        counters["matchups_without_forecast"] += build.matchups_without_forecast
        if build.decisions:
            pending.extend(build.decisions)
            frozen = (*frozen, *build.decisions)
    return tuple(pending), counters


def update_market_first_paper(*, validate_only: bool = False) -> dict[str, object]:
    policy = FrozenMarketFirstPolicy.load(POLICY_PATH)
    quote_store, forecast_store, metadata_store, decision_store, settlement_store = (
        _stores()
    )
    quotes = quote_store.read()
    forecasts = forecast_store.read()
    metadata = metadata_store.read()
    decisions = decision_store.read()
    settlements = settlement_store.read()
    build_counters: dict[str, int] = {
        "captures_considered": 0,
        "captures_at_locked_horizon": 0,
        "matchups_considered": 0,
        "matchups_already_frozen": 0,
        "matchups_without_fresh_quotes": 0,
        "matchups_without_forecast": 0,
    }

    if not validate_only:
        pending, build_counters = _build_new_decisions(
            policy=policy,
            quotes=quotes,
            forecasts=forecasts,
            metadata=metadata,
            existing=decisions,
        )
        decision_store.append(pending)
        decisions = decision_store.read()

        raw_bytes = RAW_PATH.read_bytes()
        raw = pd.read_csv(RAW_PATH, low_memory=False)
        outcomes, completed_events, ambiguous_matchups = _result_index(raw)
        settled_ids = {item.decision_id for item in settlements}
        settled_at = datetime.now(timezone.utc).replace(microsecond=0)
        pending_settlements = []
        for decision in decisions:
            if decision.decision_id in settled_ids:
                continue
            key = (decision.event_id, decision.fighter_id, decision.opponent_id)
            if key in ambiguous_matchups:
                continue
            result = outcomes.get(key)
            if result is None and decision.event_id not in completed_events:
                continue
            target, fight_id = result if result is not None else (None, None)
            pending_settlements.append(
                settle_market_first_decision(
                    decision,
                    target=target,
                    fight_id=fight_id,
                    settled_at_utc=settled_at,
                    result_source_sha256=sha256(raw_bytes).hexdigest(),
                )
            )
        settlement_store.append(pending_settlements)
        settlements = settlement_store.read()

    report = summarize_market_first_paper(decisions, settlements, quotes)
    report["policy"] = policy.artifact
    report["prospective_first_capture_utc"] = policy.prospective_first_capture_utc
    report["last_update_utc"] = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    report["last_build"] = build_counters
    report["report_sha256"] = canonical_hash(report)
    if not validate_only:
        atomic_write_text(
            REPORT_PATH,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record and settle the frozen market-first paper candidate. "
            "This cannot place a wager."
        )
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="read and validate existing records without appending or settling",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = update_market_first_paper(validate_only=arguments.validate_only)
    results = report["results"]
    print(
        "Market-first paper test: "
        f"{report['settlements_total']}/{report['decisions_total']} fights settled; "
        f"{results['recommended_bets']} recommendations; "
        f"profit={results['profit_units']:.2f} units; "
        f"betting={report['betting_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
