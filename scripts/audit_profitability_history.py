"""Audit saved historical betting research without retraining or changing policies.

Run from the repository root with ``python scripts/audit_profitability_history.py``.
Inputs are read only. Outputs go exclusively to --output-dir. Payout stresses
reduce net winning payouts, keep losing stakes unchanged, and never reselect bets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from backfill_bestfightodds_history import default_database_path  # noqa: E402

STEM = "rolling_moneyline_profitability_2021_2026"
STRATEGIES = (
    "current_model", "fixed_50_50_logit_blend", "leave_one_out_market",
    "market_first_candidate",
)
PAYOUT_REDUCTIONS = (0.0, 0.02, 0.05)
INSUFFICIENT_HISTORY = "fallback_5_percent_insufficient_history"
SEED = 20260904


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}


def _boolean(values: pd.Series, name: str) -> pd.Series:
    normalized = values.astype(str).str.lower()
    if not normalized.isin(("true", "false")).all():
        raise ValueError(f"{name} must contain only true/false")
    return normalized.eq("true")


def validate_ledger(frame: pd.DataFrame, source: dict) -> pd.DataFrame:
    required = {
        "event_date", "event_id", "fight_id", "strategy", "book_name", "side",
        "decimal_odds", "fair_probability", "estimated_ev", "won", "profit_units",
        "quote_age_hours", "selected_threshold", "threshold_selection_status",
        "qualifies_selected_threshold",
    }
    if required - set(frame):
        raise ValueError(f"missing ledger columns: {sorted(required - set(frame))}")
    result = frame.copy()
    for column in ("won", "qualifies_selected_threshold"):
        result[column] = _boolean(result[column], column)
    if result.duplicated(["strategy", "fight_id"]).any():
        raise ValueError("duplicate strategy/fight in saved ledger")
    if set(result.strategy) != set(STRATEGIES):
        raise ValueError("saved ledger does not contain exactly the four known strategies")
    numeric = ["decimal_odds", "fair_probability", "estimated_ev", "profit_units",
               "quote_age_hours", "selected_threshold"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[numeric].to_numpy()).all():
        raise ValueError("non-finite saved ledger value")
    if (result.decimal_odds <= 1).any() or not result.fair_probability.between(0, 1).all():
        raise ValueError("invalid odds or probability")
    if not np.allclose(result.profit_units, np.where(result.won, result.decimal_odds - 1, -1), atol=1e-9):
        raise ValueError("saved payouts do not reconcile to recorded odds and outcomes")
    if not np.allclose(result.estimated_ev, result.fair_probability * result.decimal_odds - 1, atol=1e-9):
        raise ValueError("estimated returns do not reconcile")
    if not (result.qualifies_selected_threshold == (result.estimated_ev >= result.selected_threshold)).all():
        raise ValueError("saved selection flags disagree with frozen thresholds")
    counts = source["coverage"]
    if result.fight_id.nunique() != counts["scored_fights"] or result.event_id.nunique() != counts["scored_events"]:
        raise ValueError("saved report coverage differs from ledger")
    for strategy, group in result.groupby("strategy"):
        if group.fight_id.nunique() != counts["scored_fights"]:
            raise ValueError("strategies do not share the reported fight cohort")
        selected = group[group.qualifies_selected_threshold]
        expected = source["pooled_profitability"][strategy]
        if len(selected) != expected["selections"] or selected.event_id.nunique() != expected["events"]:
            raise ValueError(f"saved report selection counts differ for {strategy}")
        if not np.isclose(selected.profit_units.sum(), expected["profit_units"], atol=1e-8):
            raise ValueError(f"saved report profit differs for {strategy}")
    return result.sort_values(["event_date", "event_id", "fight_id", "strategy"], kind="stable")


def stressed_profits(frame: pd.DataFrame, payout_reduction: float) -> np.ndarray:
    if not 0 <= payout_reduction < 1:
        raise ValueError("payout reduction must be in [0, 1)")
    return np.where(frame.won, (frame.decimal_odds - 1) * (1 - payout_reduction), -1.0)


def drawdown(profits: np.ndarray) -> float:
    equity = np.concatenate(([0.0], np.cumsum(profits)))
    return float(np.max(np.maximum.accumulate(equity) - equity))


def summarize(frame: pd.DataFrame, reduction: float, samples: int) -> dict:
    ordered = frame.sort_values(["event_date", "event_id", "fight_id"], kind="stable").copy()
    ordered["audit_profit"] = stressed_profits(ordered, reduction)
    events = ordered.groupby(["event_date", "event_id"], sort=True).audit_profit.agg(["sum", "count"])
    count = len(ordered)
    profit = float(ordered.audit_profit.sum())
    lower = upper = None
    if len(events) >= 2 and samples > 0:
        rng = np.random.default_rng(SEED)
        indices = rng.integers(0, len(events), size=(samples, len(events)))
        roi = events["sum"].to_numpy()[indices].sum(axis=1) / events["count"].to_numpy()[indices].sum(axis=1)
        lower, upper = map(float, np.quantile(roi, [0.025, 0.975]))
    return {
        "selections": count, "events": len(events), "wins": int(ordered.won.sum()),
        "losses": int((~ordered.won).sum()), "risk_units": float(count),
        "profit_units": profit, "roi": profit / count if count else None,
        "roi_ci_95_lower": lower, "roi_ci_95_upper": upper,
        "bootstrap_samples": samples if len(events) >= 2 else 0,
        "event_close_drawdown_units": drawdown(events["sum"].to_numpy()),
        "source_fight_id_order_drawdown_units": drawdown(ordered.audit_profit.to_numpy()),
        "mean_estimated_return_before_stress": float(ordered.estimated_ev.mean()) if count else None,
        "mean_selected_win_probability": float(ordered.fair_probability.mean()) if count else None,
        "observed_win_rate": float(ordered.won.mean()) if count else None,
        "underdog_bets": int((ordered.decimal_odds > 2).sum()),
        "median_decimal_odds": float(ordered.decimal_odds.median()) if count else None,
    }


def build_tables(frame: pd.DataFrame, samples: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = frame[frame.qualifies_selected_threshold].copy()
    selected["year"] = selected.event_date.astype(str).str[:4]
    selected["odds_band"] = pd.cut(selected.decimal_odds, [1, 2, 3, 5, 10, np.inf],
                                    labels=["1_to_2", "over_2_to_3", "over_3_to_5", "over_5_to_10", "over_10"]).astype(str)
    selected["quote_age_band"] = pd.cut(selected.quote_age_hours, [-1e-9, 1, 6, 12, 24, np.inf],
                                         labels=["0_to_1h", "over_1_to_6h", "over_6_to_12h", "over_12_to_24h", "over_24h"]).astype(str)
    for reduction in PAYOUT_REDUCTIONS:
        selected[f"profit_after_{int(reduction * 100)}pct_payout_reduction"] = stressed_profits(selected, reduction)
    summaries, breakdowns = [], []
    for cohort in ("all_scored_years", "enough_prior_threshold_examples"):
        cohort_rows = selected if cohort == "all_scored_years" else selected[selected.threshold_selection_status != INSUFFICIENT_HISTORY]
        for reduction in PAYOUT_REDUCTIONS:
            for strategy in STRATEGIES:
                group = cohort_rows[cohort_rows.strategy == strategy]
                summaries.append({"cohort": cohort, "strategy": strategy,
                                  "net_winning_payout_reduction": reduction, **summarize(group, reduction, samples)})
            summaries.append({"cohort": cohort, "strategy": "no_bets",
                              "net_winning_payout_reduction": reduction,
                              **summarize(selected.iloc[0:0], reduction, samples)})
    # These are contributions of original best-book selections, not independent
    # strategies or single-book replays. Do not choose a policy from these tables.
    for strategy, group in selected.groupby("strategy"):
        for dimension in ("year", "book_name", "odds_band", "quote_age_band", "threshold_selection_status"):
            for value, subset in group.groupby(dimension, observed=True):
                for reduction in PAYOUT_REDUCTIONS:
                    breakdowns.append({"strategy": strategy, "dimension": dimension, "value": str(value),
                                       "net_winning_payout_reduction": reduction,
                                       **summarize(subset, reduction, samples)})
    return pd.DataFrame(summaries), pd.DataFrame(breakdowns), selected


def findings(report: dict, summary: pd.DataFrame) -> str:
    lines = ["# Historical profitability audit", "", "The saved broad historical test does not establish a profitable betting rule. These are hypothetical one-unit bets using recorded prices, not executed bets or the exact website policy.", "", "## Reproduced results", "", "| Strategy | Bets | Profit | ROI | 2% payout reduction ROI | 5% payout reduction ROI |", "|---|---:|---:|---:|---:|---:|"]
    main = summary[summary.cohort == "all_scored_years"]
    for strategy in STRATEGIES:
        rows = main[main.strategy == strategy].set_index("net_winning_payout_reduction")
        base = rows.loc[0.0]
        lines.append(f"| {strategy} | {int(base.selections)} | {base.profit_units:+.2f}u | {base.roi:.2%} | {rows.loc[0.02].roi:.2%} | {rows.loc[0.05].roi:.2%} |")
    coverage = report["coverage"]
    lines += ["", f"The shared cohort contains {coverage['scored_fights']} fights across {coverage['scored_events']} events. Making no bets produces zero profit and zero exposure; ROI is undefined because nothing is risked.", "", "Payout reductions are deterministic stress scenarios: subtract 2% or 5% of net winnings from winning bets only. Losing stakes stay at -1 unit. Picks and thresholds remain fixed. These percentages are assumptions, not measured execution costs.", "", "## What limits the result", ""]
    for item in report["findings"]:
        lines.append(f"- **{item['title']}** {item['detail']} Source: {', '.join(item['sources'])}.")
    lines += ["", "## Reproduction and interpretation", "", "Run `python scripts/audit_profitability_history.py`; override `--analysis-dir`, `--database`, or `--output-dir` when needed. The database argument records availability only; this bounded audit never derives raw odds, fits a model, changes thresholds, or accesses the network.", "", "`summary.csv` includes event-level 95% return intervals; one-event samples receive no interval. Cards are resampled as whole units so bets on one card stay together. These intervals do not correct for all previous experiments or guarantee future results. `breakdowns.csv` contains hindsight diagnostics by year, book, odds, quote age, and threshold-selection status. Book rows only attribute original best-book bets; they are not single-book strategies.", "", "The 'enough_prior_threshold_examples' cohort reproduces the source report: it excludes insufficient-history fallbacks, but still includes 5% fallbacks when earlier cutoffs lost money. Both cohorts are therefore descriptive, not an automatically selected deployment policy.", "", "Event-close drawdown avoids inventing settlement order within a card. The separate fight-ID-order drawdown reproduces the historical deterministic ordering and is not a claim about intraday bankroll exposure. This ledger lacks execution and outstanding-bet timing, so capped bankroll, one-funded-bet-per-fight across market types, and 10% outstanding-exposure policies cannot be honestly replayed from it.", "", "All input fingerprints, source period contracts, and unavailable comparisons are recorded in `audit_report.json`. No production recommendation changed.", ""]
    return "\n".join(lines)


def run(analysis_dir: Path, database: Path, output_dir: Path, samples: int = 10_000) -> dict:
    if samples < 0:
        raise ValueError("bootstrap samples must be nonnegative")
    ledger_path, source_path = analysis_dir / f"{STEM}.csv", analysis_dir / f"{STEM}.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    ledger = validate_ledger(pd.read_csv(ledger_path, low_memory=False), source)
    summary, breakdowns, selected = build_tables(ledger, samples)
    market_selected = selected[selected.strategy == "leave_one_out_market"]
    market_summary = summarize(market_selected, 0, 0)
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_classification": "retrospective_existing_best_book_research_policy_replay",
        "executed_bets": False, "production_modified": False, "models_refitted": False,
        "inputs": {"ledger": fingerprint(ledger_path), "source_report": fingerprint(source_path),
                   "database": {"path": str(database.resolve()), "exists": database.is_file(), "read": False}},
        "coverage": source["coverage"], "source_experiment_version": source.get("experiment_version"),
        "source_created_at_utc": source.get("created_at_utc"),
        "periods": [{k: fold[k] for k in ("test_year", "development", "threshold_validation", "test")} for fold in source.get("yearly_folds", [])],
        "source_metrics_reconciled": True,
        "summaries": json.loads(summary.to_json(orient="records", double_precision=15)),
        "bootstrap": {"unit": "whole_event", "samples": samples, "seed": SEED, "minimum_events": 2,
                      "multiple_testing_adjustment": False},
        "payout_stress": {"reductions": list(PAYOUT_REDUCTIONS), "applies_to": "net_winning_payout_only", "thresholds_retuned": False},
        "available_historical_books": sorted(ledger.book_name.unique().tolist()),
        "single_book_policy_replay": {"status": "unavailable_in_bounded_audit", "reason": "The saved ledger keeps only the best offer per fight. The report omits fitted adjustment coefficients. Reconstructing alternatives requires raw-history derivation and small model fits; neither was run. Filtering saved winners by book is not a single-book policy.", "user_book_access_verified": False},
        "bankroll_policy_replay": {"status": "unavailable_from_this_ledger", "reason": "No actual stake/acceptance records or outstanding-bet timestamps; no simultaneous cross-market portfolio."},
        "findings": [
            {"title": "No profitable broad strategy", "detail": "All four saved strategies lose money before the added payout stresses. The narrower positive 35-bet snapshot was superseded by broader rolling evidence.", "sources": ["HISTORICAL_ODDS_BACKFILL.md:235", "HISTORICAL_ODDS_BACKFILL.md:263"]},
            {"title": "Older threshold selection used fitted outcomes", "detail": "The narrow evaluator loads an adjustment refitted on both training and selection outcomes, then selects betting thresholds from that same selection period. Its later test was separate at the time, but threshold-selection returns are optimistic training evidence. The broad rolling evaluator correctly uses a development-only fit for threshold selection.", "sources": ["src/evaluate_market_first_challenger.py:405", "src/evaluate_historical_moneyline_profitability.py:555", "src/evaluate_rolling_moneyline_profitability.py:311"]},
            {"title": "Abstention was not the historical fallback", "detail": "The historical evaluator uses a 5% cutoff even when no earlier cutoff was profitable or earlier examples were insufficient. This audit adds a zero-exposure comparison without rewriting old policies.", "sources": ["src/evaluate_historical_moneyline_profitability.py:492"]},
            {"title": "Selected long shots need calibration checks", "detail": f"The saved market-only rule selected {market_summary['selections']} bets, including {market_summary['underdog_bets']} underdogs, with median decimal odds {market_summary['median_decimal_odds']}. See summary.csv for estimated and realized returns. These hindsight diagnostics motivate a new independent calibration test, not a favorite-only rule selected from these results.", "sources": [f"{STEM}.csv", "breakdowns.csv"]},
            {"title": "Book access and timing were assumed", "detail": "The historical test chooses among seven books and excludes limits, rejected bets, fees, and latency. Historical T-24 is measured from midnight UTC of the source event date; prices can be up to 24 hours old. This is not an exact replay of the deployed website policy.", "sources": ["src/evaluate_rolling_moneyline_profitability.py:478", "HISTORICAL_ODDS_BACKFILL.md:230"]},
            {"title": "Closing movement is not verified closing fair value", "detail": "The source closing metric subtracts raw same-book implied probabilities. It does not remove the later bookmaker margin or prove that the entry quote was executable. Positive movement alone did not ensure profit in this ledger.", "sources": ["src/evaluate_historical_moneyline_profitability.py:359"]},
            {"title": "Historical research was reused", "detail": "Feature choices and model families were explored using some of these years. Chronological fitting prevents direct use of future training outcomes, but repeated design decisions make these retrospective development evidence rather than fresh confirmation. New comparisons must freeze rules before collecting later fights.", "sources": ["src/evaluate_rolling_moneyline_profitability.py:476", "MODEL_FAMILY_RESEARCH.md:181"]},
            {"title": "Prediction provenance can be missing", "detail": "The precomputed prediction loader checks training cutoffs only if supplied; its output still describes the file as causal. Require traceable training provenance or label imported predictions unverified in future evaluations.", "sources": ["src/evaluate_bestfightodds_history.py:117"]},
            {"title": "Method prices cannot establish executable returns", "detail": "Method research covers 7,755 selections over 2,586 fights using historical mean prices. The outcome model and the earlier-data-selected blend did not beat the market. Mean prices are not known executable offers, so no method profit replay is inferred here.", "sources": ["HISTORICAL_METHOD_PRICE_EVALUATION_2026-08-30.md:26", "HISTORICAL_METHOD_PRICE_EVALUATION_2026-08-30.md:36"]},
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    breakdowns.to_csv(output_dir / "breakdowns.csv", index=False)
    keep = ["event_date", "event_id", "fight_id", "strategy", "book_name", "side", "decimal_odds", "fair_probability", "estimated_ev", "won", "profit_units", "quote_age_hours", "selected_threshold", "threshold_selection_status", "year", "odds_band", "quote_age_band"]
    keep += [f"profit_after_{int(r * 100)}pct_payout_reduction" for r in PAYOUT_REDUCTIONS]
    selected[keep].to_csv(output_dir / "selected_bets.csv", index=False)
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "findings.md").write_text(findings(report, summary), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--database", type=Path, default=default_database_path())
    parser.add_argument("--output-dir", type=Path, default=ROOT / "audit/profitability/history")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    report = run(args.analysis_dir or args.database.parent / "analysis", args.database, args.output_dir, args.bootstrap_samples)
    print(f"Reconciled {report['coverage']['scored_fights']} fights across {report['coverage']['scored_events']} events; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
