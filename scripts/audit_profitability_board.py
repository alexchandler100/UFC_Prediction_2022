"""Offline, deterministic audit of the published betting board; never trades.

Only writes the requested audit output directory. No production imports, model
fitting, network calls, or changes to published data are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "src/content/data/market"
MAX_AGE_SECONDS = 1800


def utc(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def stamp(value):
    return value.isoformat().replace("+00:00", "Z") if value else None


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def decimal_odds(moneyline):
    price = float(moneyline)
    if abs(price) < 100:
        raise ValueError("Expected a valid American price")
    return 1 + (price / 100 if price > 0 else 100 / -price)


def expected_return(probability, moneyline):
    probability = number(probability)
    return None if probability is None else probability * decimal_odds(moneyline) - 1


def source_eligible(row, at):
    """Audit expiry rule: exact recoverable source time and future card start."""
    updated = utc(row.get("resolved_source_quote_updated_at_utc"))
    start = utc(row.get("resolved_event_start_utc"))
    published = utc(row.get("observed_at_utc"))
    return bool(updated and start and published and published <= at < start
                and -300 <= (at - updated).total_seconds() <= MAX_AGE_SECONDS)


def browser_visible(row, threshold=0.05, allowed_books=None):
    """Current script.js filter, deliberately without clock/expiry checks."""
    ev = number(row.get("estimated_expected_return"))
    return (row.get("threshold_met") is True and ev is not None and ev >= threshold
            and (allowed_books is None or row.get("target_book") in allowed_books))


def allocate(rows, as_of):
    """Illustrative caps on saved stakes, not an optimized betting strategy."""
    rank = sorted(rows, key=lambda row: (
        -(number(row.get("robust_lower_expected_return")) or 0),
        str(row.get("event_date")), str(row.get("matchup_id")),
        str(row.get("selection")), str(row.get("target_book")), str(row.get("bet_id")),
    ))
    by_card = defaultdict(float)
    funded_fights = set()
    total = 0.0
    result = []
    for row in rank:
        key = (row.get("event_id"), row.get("matchup_id"))
        floor_ev = number(row.get("robust_lower_expected_return"))
        saved_fraction = number(row.get("recommended_fraction")) or 0
        amount = 0.0
        if floor_ev is None or floor_ev <= 0 or saved_fraction <= 0:
            reason = "no_positive_conservative_edge_or_saved_stake"
        elif not source_eligible(row, as_of):
            reason = "source_not_fresh_or_card_not_confirmed_future"
        elif key in funded_fights:
            reason = "another_selection_funded_on_same_fight"
        else:
            amount = max(0.0, min(1.0, saved_fraction * 100,
                                  5.0 - by_card[row["event_id"]], 10.0 - total))
            reason = "funded_illustrative_only" if amount > 1e-12 else "card_or_outstanding_cap"
            if amount > 1e-12:
                funded_fights.add(key)
                by_card[row["event_id"]] += amount
                total += amount
        result.append({"bet_id": row["bet_id"], "event_id": row.get("event_id"),
                       "event_date": row.get("event_date"), "matchup_id": row.get("matchup_id"),
                       "fight": row.get("fight"), "selection": row.get("selection"),
                       "target_book": row.get("target_book"),
                       "offered_moneyline": row["offered_moneyline"],
                       "robust_lower_expected_return": floor_ev,
                       "published_stake_units": saved_fraction * 100,
                       "illustrative_stake_units": amount, "reason": reason})
    return {"bankroll_units": 100, "maximum_fight_units": 1, "maximum_card_units": 5,
            "maximum_outstanding_units": 10, "funded_selection_count": len(funded_fights),
            "total_stake_units": total, "rows": result,
            "policy": "One funded selection per fight; descending positive saved lower-bound EV; "
                      "cap saved stake at 1% per fight, 5% per card, 10% outstanding. "
                      "Require exact recoverable source timestamp and future card start. "
                      "Assume no existing open bets. Illustrative risk limits, not optimal stakes."}


def alternative_candidates(bet, quotes, as_of, minimum_return=0.05):
    """Reprice the published selection at other books, without hindsight."""
    fresh = []
    for quote in quotes:
        updated = utc(quote.get("source_quote_updated_at_utc"))
        if updated and -300 <= (as_of - updated).total_seconds() <= MAX_AGE_SECONDS:
            fresh.append(quote)
    result = []
    for target in fresh:
        other = [q for q in fresh if q["book"] != target["book"]]
        if bet["category"] == "Moneyline":
            if len(other) < 3:
                continue
            if number(bet.get("model_weight")) != 0:
                # A future policy requires its precise saved blend, never guess it.
                continue
            fighter_probability = sum(q["no_vig_fighter_probability"] for q in other) / len(other)
            probability = fighter_probability if bet["side"] == "fighter" else 1 - fighter_probability
            moneyline = target[f"{bet['side']}_moneyline"]
        else:
            probability = bet["estimated_win_probability"]
            moneyline = target[f"{bet['side']}_moneyline"]
        ev = expected_return(probability, moneyline)
        result.append({"bet_id": bet["bet_id"], "matchup_id": bet["matchup_id"],
                       "category": bet["category"], "selection": bet["selection"],
                       "published_book": bet["target_book"], "available_book": target["book"],
                       "offered_moneyline": moneyline, "estimated_win_probability": probability,
                       "estimated_expected_return": ev, "threshold_met": ev >= minimum_return,
                       "comparison_book_count": len(other),
                       "source_quote_updated_at_utc": target.get("source_quote_updated_at_utc"),
                       "different_from_published_book": target["book"] != bet["target_book"],
                       "eligibility_basis": "same published selection; fresh saved prices; raw policy EV"})
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return [{key: identity, "event_id": group[0]["event_id"],
             "event_date": group[0]["event_date"],
             "label": group[0]["fight"] if key == "matchup_id" else group[0]["event_title"],
             "bet_count": len(group),
             "published_stake_fraction": sum(row["recommended_fraction"] or 0 for row in group),
             "published_stake_units": 100 * sum(row["recommended_fraction"] or 0 for row in group)}
            for identity, group in sorted(grouped.items())]


def method_surface_rows(methods):
    rows = []
    for matchup in methods.get("method_markets", []):
        for book in matchup.get("book_quotes", []):
            for selection in book.get("selections", []):
                rows.append({"surface": "method_price_comparison", "matchup_id": matchup["matchup_id"],
                             "selection": selection["selection"], "book": book["book"],
                             "moneyline": selection.get("moneyline"),
                             "estimated_probability": selection.get("candidate_model_probability"),
                             "estimated_expected_return": selection.get("candidate_model_estimated_return"),
                             "observed_at_utc": book.get("observed_at_utc"),
                             "comparison_only": True, "on_qualified_board": False,
                             "price_contract_complete": book.get("is_complete_six_way"),
                             "settlement_status": methods.get("settlement_status")})
    return rows


def audit(as_of_text=None):
    paths = {name: MARKET / name for name in (
        "upcoming_bet_board.json", "current_opportunities.json", "current_method_markets.json",
        "total_round_quote_snapshots.jsonl", "total_round_forecast_captures.jsonl",
        "quote_snapshots.jsonl", "early_market_observations.jsonl", "published_bet_snapshots.json",
    )}
    paths["all_upcoming_forecasts.json"] = ROOT / "src/content/data/external/all_upcoming_forecasts.json"
    paths["script.js"] = ROOT / "script.js"
    paths["collector_workflow"] = ROOT / ".github/workflows/collect-market-snapshot.yml"
    board = read_json(paths["upcoming_bet_board.json"])
    current = read_json(paths["current_opportunities.json"])
    methods = read_json(paths["current_method_markets.json"])
    totals = read_jsonl(paths["total_round_quote_snapshots.jsonl"])
    total_forecasts = read_jsonl(paths["total_round_forecast_captures.jsonl"])
    moneylines = read_jsonl(paths["quote_snapshots.jsonl"])
    early = read_jsonl(paths["early_market_observations.jsonl"])
    forecasts = read_json(paths["all_upcoming_forecasts.json"])
    archive = read_json(paths["published_bet_snapshots.json"])
    archived_ids = {row["bet_id"] for row in archive["snapshots"]}
    forecast_index = {row["matchup_id"]: row for row in forecasts["matchups"]}
    market_index = {row["matchup_id"]: row for row in board.get("market_matchups", [])}
    captured = utc(board["observed_at_utc"])
    as_of = utc(as_of_text) if as_of_text else captured
    if as_of < captured:
        raise ValueError("--as-of cannot precede this board's publication timestamp")
    traced, alternatives = [], []
    for bet in board["bets"]:
        observed = utc(bet["observed_at_utc"])
        is_total = bet["category"] == "Total rounds"
        available = []
        matched = None
        forecast = forecast_index.get(bet["matchup_id"], {})
        if is_total:
            available = [q for q in totals if q["matchup_id"] == bet["matchup_id"]
                         and q["line"] == bet["line"] and utc(q["observed_at_utc"]) == observed]
            matched = next((q for q in available if q["book"] == bet["target_book"]
                            and q[f"{bet['side']}_moneyline"] == bet["offered_moneyline"]), None)
            if matched:
                forecast = next((f for f in total_forecasts if f["capture_id"] == matched["capture_id"]
                                 and f["matchup_id"] == bet["matchup_id"] and f["line"] == bet["line"]), {})
        else:
            available = market_index.get(bet["matchup_id"], {}).get("book_quotes", [])
            matched = next((q for q in moneylines if q["matchup_id"] == bet["matchup_id"]
                            and q["book"] == bet["target_book"] and utc(q["observed_at_utc"]) == observed
                            and q[f"{bet['side']}_moneyline"] == bet["offered_moneyline"]), None)
            if matched is None:
                # Future cards use source identities until official stable-ID capture.
                matched = next((q for q in early if q["book"] == bet["target_book"]
                                and q["source_quote_updated_at_utc"] == bet.get("source_quote_updated_at_utc")
                                and utc(q["first_observed_at_utc"]) <= observed
                                and {q["outcome_a"], q["outcome_b"]} == {bet["fighter_name"], bet["opponent_name"]}
                                and (q["outcome_a_moneyline"] if q["outcome_a"] == bet["selection"] else q["outcome_b_moneyline"])
                                == bet["offered_moneyline"]), None)
        matched = matched or {}
        updated = bet.get("source_quote_updated_at_utc") or matched.get("source_quote_updated_at_utc")
        start = bet.get("event_start_utc") or matched.get("event_start_utc") or matched.get("source_commence_time_utc")
        bayesian = bet.get("bayesian_kelly", {})
        probability_field = "over_probability" if is_total else "model_probability_for_fighter"
        source_probability = forecast.get(probability_field)
        if is_total and bet["side"] == "under" and source_probability is not None:
            source_probability = 1 - source_probability
        row = {key: bet.get(key) for key in (
            "bet_id", "event_id", "event_date", "event_title", "matchup_id", "category", "selection",
            "target_book", "side", "line", "offered_moneyline", "estimated_win_probability",
            "estimated_expected_return", "threshold_met", "probability_source", "model_weight",
            "consensus_book_count", "candidate_only", "observed_at_utc", "source",
            "source_quote_updated_at_utc", "event_start_utc",
        )}
        row.update({"fight": f"{bet['fighter_name']} vs {bet['opponent_name']}",
                    "recomputed_raw_expected_return": expected_return(bet["estimated_win_probability"], bet["offered_moneyline"]),
                    "calibration_status": bayesian.get("status"),
                    "calibration_artifact_sha256": bayesian.get("calibration_artifact_sha256"),
                    "calibration_training_fights": bayesian.get("calibration_training_fights"),
                    "calibration_training_events": bayesian.get("calibration_training_events"),
                    "calibration_trained_through": bayesian.get("calibration_trained_through"),
                    "calibrated_mean_probability": bayesian.get("posterior_mean_probability"),
                    "robust_lower_probability": bayesian.get("posterior_lower_probability"),
                    "calibrated_mean_expected_return": expected_return(bayesian.get("posterior_mean_probability"), bet["offered_moneyline"]),
                    "robust_lower_expected_return": expected_return(bayesian.get("posterior_lower_probability"), bet["offered_moneyline"]),
                    "recommended_fraction": number(bayesian.get("recommended_fraction")),
                    "quote_trace_matched": bool(matched),
                    "quote_or_observation_id": matched.get("quote_id", matched.get("observation_id")),
                    "capture_id": matched.get("capture_id", matched.get("first_capture_id")),
                    "source_payload_sha256": matched.get("source_payload_sha256"),
                    "resolved_source_quote_updated_at_utc": updated,
                    "resolved_event_start_utc": start,
                    "quote_age_seconds_as_of": (as_of - utc(updated)).total_seconds() if updated else None,
                    "published_exact_quote_timestamp_missing": not bool(bet.get("source_quote_updated_at_utc")),
                    "published_event_start_missing": not bool(bet.get("event_start_utc")),
                    "forecast_trace_matched": bool(forecast),
                    "forecast_issued_at_utc": forecast.get("forecast_issued_at_utc"),
                    "forecast_model_id": forecast.get("model_id"),
                    "forecast_model_trained_through": forecast.get("model_trained_through"),
                    "forecast_probability": source_probability,
                    "forecast_probability_is_selection_source": is_total,
                    "archived_exact_bet_id": bet["bet_id"] in archived_ids})
        row["browser_visible_at_as_of"] = browser_visible(bet, board["minimum_expected_return"])
        row["source_eligible_at_as_of"] = source_eligible(row, as_of)
        traced.append(row)
        alternatives.extend(alternative_candidates(bet, available, as_of, board["minimum_expected_return"]))
    expiry = []
    for row in traced:
        scenarios = {"as_of": as_of, "plus_30_minutes": as_of + timedelta(minutes=30),
                     "plus_6_hours": as_of + timedelta(hours=6),
                     "event_start": utc(row["resolved_event_start_utc"])}
        for scenario, at in scenarios.items():
            expiry.append({"bet_id": row["bet_id"], "selection": row["selection"],
                           "scenario": scenario, "at_utc": stamp(at),
                           "current_browser_visible_if_snapshot_unchanged": browser_visible(row),
                           "eligible_with_source_expiry": source_eligible(row, at) if at else False})
    books = sorted({row["available_book"] for row in alternatives} | {b["target_book"] for b in board["bets"]})
    book_scenarios = []
    for book in books:
        visible = {b["bet_id"] for b in board["bets"] if browser_visible(b, allowed_books={book})}
        qualifying = {a["bet_id"] for a in alternatives if a["available_book"] == book and a["threshold_met"]}
        book_scenarios.append({"only_accessible_book": book, "current_visible_count": len(visible),
                               "qualifying_selections_at_this_book": len(qualifying),
                               "hidden_qualifying_alternative_count": len(qualifying - visible),
                               "hidden_bet_ids": sorted(qualifying - visible)})
    surfaces = method_surface_rows(methods)
    for matchup in current.get("matchups", []):
        signal = matchup.get("current_signal") or {}
        surfaces.append({"surface": "current_moneyline_signal", "matchup_id": matchup["matchup_id"],
                         "selection": signal.get("best_candidate_name"), "book": signal.get("target_book"),
                         "moneyline": signal.get("offered_moneyline"),
                         "estimated_probability": signal.get("market_probability"),
                         "estimated_expected_return": signal.get("estimated_expected_return"),
                         "observed_at_utc": current.get("observed_at_utc"), "comparison_only": False,
                         "paper_action": signal.get("paper_action"),
                         "on_qualified_board": any(b["matchup_id"] == matchup["matchup_id"] and b["category"] == "Moneyline" for b in traced)})
    for market in current.get("prop_markets", {}).get("total_rounds", {}).get("markets", []):
        candidate = market.get("best_candidate") or {}
        surfaces.append({"surface": "current_total_comparison", "matchup_id": market["matchup_id"],
                         "selection": candidate.get("selection"), "book": candidate.get("target_book"),
                         "moneyline": candidate.get("offered_moneyline"),
                         "estimated_probability": candidate.get("model_probability"),
                         "estimated_expected_return": candidate.get("estimated_expected_return"),
                         "observed_at_utc": current.get("observed_at_utc"), "comparison_only": True,
                         "on_qualified_board": any(b["matchup_id"] == market["matchup_id"] and b["selection"] == candidate.get("selection") for b in traced)})
    script = paths["script.js"].read_text(encoding="utf-8")
    renderer = script.split("function renderQualifiedUpcomingBets()", 1)[1].split("function selectPerformanceRecords", 1)[0]
    allocation = allocate(traced, as_of)
    summary = {"board_bets": len(traced), "moneyline_bets": sum(r["category"] == "Moneyline" for r in traced),
               "total_round_bets": sum(r["category"] == "Total rounds" for r in traced),
               "published_stake_units_on_100_unit_bankroll": sum((r["recommended_fraction"] or 0) * 100 for r in traced),
               "zero_stake_qualified_bets": sum(r["recommended_fraction"] == 0 for r in traced),
               "negative_calibrated_mean_ev_bets": sum(r["calibrated_mean_expected_return"] is not None and r["calibrated_mean_expected_return"] < 0 for r in traced),
               "missing_published_quote_timestamp": sum(r["published_exact_quote_timestamp_missing"] for r in traced),
               "missing_published_event_start": sum(r["published_event_start_missing"] for r in traced),
               "matched_quote_traces": sum(r["quote_trace_matched"] for r in traced),
               "matched_forecast_traces": sum(r["forecast_trace_matched"] for r in traced),
               "archived_exact_bet_ids": sum(r["archived_exact_bet_id"] for r in traced),
               "eligible_at_as_of": sum(r["source_eligible_at_as_of"] for r in traced),
               "eligible_after_30_minutes": sum(r["eligible_with_source_expiry"] for r in expiry if r["scenario"] == "plus_30_minutes"),
               "hidden_book_alternatives": sum(r["hidden_qualifying_alternative_count"] for r in book_scenarios),
               "method_comparison_rows": sum(r["surface"] == "method_price_comparison" for r in surfaces),
               "illustrative_total_stake_units": allocation["total_stake_units"]}
    return {"schema_version": 1, "audit_only": True, "as_of_utc": stamp(as_of),
            "as_of_default": "published board observed_at_utc, not wall clock", "board_observed_at_utc": board["observed_at_utc"],
            "summary": summary, "bets": traced, "exposure_by_fight": aggregate(traced, "matchup_id"),
            "exposure_by_card": aggregate(traced, "event_id"), "expiry_scenarios": expiry,
            "book_access_scenarios": book_scenarios, "book_alternatives": alternatives,
            "illustrative_allocation": allocation, "surfaces": surfaces,
            "renderer_static_check": {"contains_clock_expression": any(s in renderer for s in ("Date.now", "new Date()")),
                                      "contains_event_start_filter": "event_start_utc" in renderer,
                                      "contains_quote_age_filter": "source_quote_age_seconds" in renderer,
                                      "basis": "Direct function inspection plus audit-only executable filter emulation; not a browser test."},
            "source_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in paths.values()},
            "limitations": ["No betting execution, live availability, limits, account restrictions, or future price changes are inferred.",
                            "Book alternatives cover existing board selections; moneyline probabilities are recomputed excluding each target book.",
                            "Expiry scenarios hold the stored snapshot fixed; they do not predict actual future prices.",
                            "Calibration uncertainty is not all matchup uncertainty. Expected returns are estimates, not realized profits.",
                            "Illustrative allocation assumes no existing open bets and does not prove profitability."]}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def findings(report):
    summary = report["summary"]
    return f"""# Published betting-board audit

This is an offline snapshot audit as of {report['as_of_utc']}. It changes no model,
published recommendation, price, calibration artifact, or staking default.

- {summary['board_bets']} displayed bets: {summary['total_round_bets']} totals and {summary['moneyline_bets']} moneylines.
- Published stakes total {summary['published_stake_units_on_100_unit_bankroll']:.2f} units on a 100-unit bankroll.
- {summary['zero_stake_qualified_bets']} qualified bets have zero recommended stake; {summary['negative_calibrated_mean_ev_bets']} have negative estimated return after probability calibration.
- Exact source prices trace for {summary['matched_quote_traces']}/{summary['board_bets']} rows; saved forecasts trace for {summary['matched_forecast_traces']}/{summary['board_bets']}.
- {summary['missing_published_quote_timestamp']} rows omit the source-update timestamp and {summary['missing_published_event_start']} omit card start in the website board. The ledger can supply these where the trace matches.
- {summary['eligible_after_30_minutes']} rows remain fresh after 30 minutes if this snapshot does not update.
- Across single-book access scenarios, {summary['hidden_book_alternatives']} qualifying book/selection alternatives are hidden by filtering only globally selected books.
- The illustrative capped allocation totals {summary['illustrative_total_stake_units']:.2f} units. It is a risk comparison, not an optimized or proven profitable strategy.

## Findings and proposed corrections

1. **Combined exposure is not limited.** `script.js:3033` acknowledges related bets; `script.js:3186` sizes all bets from the same card bankroll and only rescales above 100%. Add portfolio limits before presenting stakes as a combined strategy. Acceptance: one funded selection per fight, 1% fight / 5% card / 10% outstanding caps for this explicitly illustrative comparison; deterministic ties and unchanged entry odds.
2. **Qualification uses a different probability from staking.** `src/upcoming_bet_board.py:473` and `:508` select by unadjusted return, while `:492` attaches calibrated sizing. `script.js:3054` still displays zero-stake rows. Proposed correction: evaluate one declared calibrated selection policy prospectively and distinguish a rejected stake from an actionable recommendation. Acceptance: a nominally positive bet with negative calibrated return or zero stake cannot silently look actionable under that policy.
3. **Experimental totals are ranked with market-based moneylines.** `src/market_tracker/prop_opportunities.py:73` computes EV directly from the duration model. `src/bayesian_total_calibration.py:314` checks only whether calibration worsened probability errors, not betting profit. Require historical price-matched and prospective evidence before promotion; retain research comparisons. Acceptance: unsuccessful/missing performance evidence cannot become a promoted recommendation merely by exceeding raw EV.
4. **Stale prices remain displayed.** `script.js:3053` filters threshold and book without current-time/start checks; `src/update_and_rebuild_model.py:228` carries forward the prior capture time. `.github/workflows/collect-market-snapshot.yml:13` schedules separated captures. Proposed correction: explicit per-price expiry and card-start checks using preserved exact timestamps. Acceptance: stale, missing-timestamp and started-card cases are withheld; the stored audit is still available. Expiry scenarios here assume no intervening capture.
5. **Book filtering loses alternatives.** `src/upcoming_bet_board.py:500` and `src/market_tracker/prop_opportunities.py:144` retain only the globally best book; `script.js:3055` hides disallowed books. Proposed correction: retain qualifying per-book prices and select after the user's book choice, excluding each target from the comparison probability. Acceptance: removing the best book reveals an eligible second book, with its own price and recalculated moneyline probability; an ineligible price remains excluded.
6. **Method prices remain research comparisons.** `index.html:289` excludes methods from the qualified list and `script.js:2885` explains the unpassed performance check. `surface_trace.csv` records each method price/probability comparison, including incomplete book contracts. Preserve that distinction. Acceptance: method comparison rows do not enter the actionable board merely because their nominal EV is large.

## Reproduction and limits

Run `python scripts/audit_profitability_board.py --output-dir audit/profitability/board`.
Use `--as-of` with an explicit offset to test a later instant; the default is the
saved board timestamp for deterministic reproduction. `board_audit.json` contains
source hashes, each price/forecast trace, exposure, expiry and book-access scenarios.
CSV files expose the same calculations. No network calls or model fitting occur.
The browser check is direct code inspection and executable filter emulation, not
a browser automation test. Book access scenarios are hypothetical, not statements
about which accounts the user has. No profitability claim follows from the caps.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", help="UTC-offset timestamp; default is saved board capture time")
    args = parser.parse_args(argv)
    report = audit(args.as_of)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "board_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    for filename, rows in (("board_trace.csv", report["bets"]), ("book_alternatives.csv", report["book_alternatives"]),
                           ("surface_trace.csv", report["surfaces"]), ("expiry_scenarios.csv", report["expiry_scenarios"]),
                           ("illustrative_allocation.csv", report["illustrative_allocation"]["rows"])):
        write_csv(args.output_dir / filename, rows)
    (args.output_dir / "findings.md").write_text(findings(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
