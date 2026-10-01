"""Declared capture windows, price coverage, and a dependency-free workflow gate.

Final references are 15–90 minutes before the saved card start, not a claim of
the last tradable price of each bout. This module never creates betting records.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path

WINDOWS = {"early": (32 * 3600, 144 * 3600), "t24": (20 * 3600, 28 * 3600),
           "final": (15 * 60, 90 * 60)}
MAX_QUOTE_AGE = 1800
GATE_SCHEDULE = "7,37 * * * *"


def utc(value):
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp needs a timezone")
    return parsed.astimezone(timezone.utc)


def window_for(lead_seconds):
    return next((name for name, (lower, upper) in WINDOWS.items()
                 if lower <= lead_seconds <= upper), None)


def capture_due(card, previous, now):
    """Make at most one extra capture per window while a card's start is stable."""
    now = utc(now)
    if card.get("event_id") != previous.get("event_id") or not previous.get("event_start_utc"):
        return {"capture": False, "reason": "awaiting_card_start_from_regular_capture"}
    start = utc(previous["event_start_utc"])
    lead = (start - now).total_seconds()
    name = window_for(lead)
    if name not in ("t24", "final"):
        return {"capture": False, "reason": "outside_target_windows"}
    # Leave fifteen minutes for checkout/tests/capture in the final window.
    if name == "final" and lead < 30 * 60:
        return {"capture": False, "reason": "too_late_for_safe_final_capture"}
    observed = previous.get("captured_at_utc")
    if observed and utc(observed) <= now and window_for((start - utc(observed)).total_seconds()) == name:
        return {"capture": False, "reason": "window_already_captured", "window": name}
    return {"capture": True, "reason": "missing_window", "window": name}


def quote_rejection(quote, source, now):
    if source is None:
        return "missing_source_update"
    for key in ("quote_id", "capture_id", "matchup_id", "event_id", "book", "source", "observed_at_utc"):
        if source.get(key) != quote.get(key):
            return "source_identity_mismatch"
    try:
        observed, updated = utc(quote["observed_at_utc"]), utc(source["source_quote_updated_at_utc"])
        start, bout = utc(quote["event_start_utc"]), utc(source["source_commence_time_utc"])
    except (KeyError, ValueError, TypeError, AttributeError):
        return "missing_or_invalid_time"
    if quote.get("timing_precision") != "timestamp":
        return "missing_or_invalid_time"
    if observed > now:
        return "future_capture"
    if observed >= min(start, bout):
        return "card_or_bout_started"
    if not 0 <= (observed - updated).total_seconds() <= MAX_QUOTE_AGE:
        return "stale_or_future_source_quote"
    return None


def price_window_report(quote_sets, metadata, forecasts, decisions, now, *, additional_decisions=None):
    """Rebuild coverage and explicit missing references from saved evidence only."""
    now = utc(now)
    sources = {row["quote_id"]: row for row in metadata}
    if len(sources) != len(metadata):
        raise ValueError("duplicate source metadata")
    groups, valid_moneylines, rejected = {}, defaultdict(list), Counter()
    for market, rows in quote_sets.items():
        if market not in ("moneyline", "total_rounds"):
            continue
        for q in sorted(rows, key=lambda row: (row["observed_at_utc"], row["quote_id"])):
            if market == "total_rounds" and q.get("period") != "full_fight":
                continue
            line = q.get("line") if market == "total_rounds" else None
            key = (q["event_id"], q["matchup_id"], market, line)
            group = groups.setdefault(key, {"event_id": q["event_id"], "matchup_id": q["matchup_id"],
                "market": market, "line": line, "event_start_utc": q.get("event_start_utc"),
                "event_date": q.get("event_date"), "windows": defaultdict(list)})
            if utc(q["observed_at_utc"]) <= now:
                group["event_start_utc"] = q.get("event_start_utc")
            source = q if market == "total_rounds" else sources.get(q["quote_id"])
            reason = quote_rejection(q, source, now)
            if reason:
                rejected[reason] += 1
                continue
            name = window_for((utc(q["event_start_utc"]) - utc(q["observed_at_utc"])).total_seconds())
            if name:
                group["windows"][name].append(q)
                if market == "moneyline" and name == "final":
                    valid_moneylines[(q["event_id"], q["matchup_id"], q["capture_id"])].append(q)
    # A forecast with no quotes is still a missing observation, not a pass.
    for f in forecasts:
        key = (f["event_id"], f["matchup_id"], "moneyline", None)
        groups.setdefault(key, {"event_id": f["event_id"], "matchup_id": f["matchup_id"],
            "market": "moneyline", "line": None, "event_start_utc": f.get("event_start_utc"),
            "event_date": f.get("event_date"), "windows": defaultdict(list)})
    coverage, summary = [], defaultdict(Counter)
    for key in sorted(groups, key=str):
        group = groups[key]
        start = utc(group["event_start_utc"]) if group["event_start_utc"] else None
        windows = {}
        for name, (lower, upper) in WINDOWS.items():
            rows = group["windows"].get(name, [])
            if rows:
                status = "captured"
            elif start is None:
                status = "unknown_start"
            else:
                lead = (start - now).total_seconds()
                status = "not_due" if lead > upper else "due_missing" if lead >= lower else "missed"
            windows[name] = {"status": status, "fresh_quotes": len(rows),
                "books": sorted({q["book"] for q in rows}),
                "last_observed_at_utc": max((q["observed_at_utc"] for q in rows), default=None)}
            summary[group["market"] + ":" + name][status] += 1
        coverage.append({**group, "windows": windows})

    references = []
    decision_sets = {"locked_market": decisions, **(additional_decisions or {})}
    for strategy, d in ((name, d) for name, rows in decision_sets.items() for d in rows):
        if d["paper_action"] == "pass":
            continue
        entry_quote = next((q for q in quote_sets.get("moneyline", [])
                            if q["quote_id"] == d["reference_quote_id"]), None)
        row = {"decision_id": d["decision_id"], "event_id": d["event_id"], "matchup_id": d["matchup_id"],
            "book": entry_quote["book"] if entry_quote else None, "status": "missing_final_same_book_quote"}
        selected_id = d[d["paper_action"] + "_id"]
        row.update({"strategy": strategy, "event_date": d.get("event_date"), "selected_fighter_id": selected_id,
            "selection": next((entry_quote.get(side + "_name", selected_id) for side in ("fighter", "opponent")
                               if entry_quote[side + "_id"] == selected_id), selected_id) if entry_quote else selected_id})
        candidates = []
        for (event, matchup, capture), quotes in valid_moneylines.items():
            if (event, matchup) != (d["event_id"], d["matchup_id"]):
                continue
            for q in quotes:
                if (entry_quote and q["book"].casefold() == entry_quote["book"].casefold()
                        and utc(q["observed_at_utc"]) > utc(d["decision_issued_at_utc"])
                        and selected_id in (q["fighter_id"], q["opponent_id"])):
                    candidates.append((q, quotes))
        if candidates:
            q, peers = max(candidates, key=lambda pair: (pair[0]["observed_at_utc"], pair[0]["quote_id"]))
            selected_side = "fighter" if q["fighter_id"] == selected_id else "opponent"
            line = q[selected_side + "_moneyline"]
            implied = lambda price: 100 / (100 + price) if price > 0 else -price / (100 - price)
            entry_break_even = implied(d["action_reference_moneyline"])
            other_books = {}
            for peer in peers:
                if (peer["book"].casefold() == q["book"].casefold()
                        or set((peer["fighter_id"], peer["opponent_id"])) != set((q["fighter_id"], q["opponent_id"]))):
                    continue
                p = peer["no_vig_fighter_probability"]
                other_books[peer["book"].casefold()] = (peer, p if peer["fighter_id"] == selected_id else 1 - p)
            independent = (sum(p for _, p in other_books.values()) / len(other_books)
                           if len(other_books) >= 3 else None)
            row.update({"status": "available" if independent is not None else "fewer_than_three_other_books",
                "reference_quote_id": q["quote_id"], "observed_at_utc": q["observed_at_utc"],
                "source_quote_updated_at_utc": sources[q["quote_id"]]["source_quote_updated_at_utc"],
                "event_start_utc": q["event_start_utc"], "moneyline": line,
                "same_book_probability_movement": implied(line) - entry_break_even,
                "other_book_quote_ids": sorted(peer["quote_id"] for peer, _ in other_books.values()),
                "other_books": sorted(other_books), "independent_probability": independent,
                "independent_probability_advantage": None if independent is None else independent - entry_break_even})
        references.append(row)
    reference_summary = {}
    for strategy in decision_sets:
        rows = [r for r in references if r["strategy"] == strategy]
        independent = [r["independent_probability_advantage"] for r in rows
                       if r.get("independent_probability_advantage") is not None]
        same_book = [r["same_book_probability_movement"] for r in rows if "same_book_probability_movement" in r]
        reference_summary[strategy] = {"recorded_bets": len(rows), "same_book_references": len(same_book),
            "independent_references": len(independent), "missing_same_book_references": len(rows) - len(same_book),
            "mean_same_book_probability_movement": sum(same_book) / len(same_book) if same_book else None,
            "mean_independent_probability_advantage": sum(independent) / len(independent) if independent else None}
    return {"version": "declared-card-price-windows-v1", "windows_seconds_before_card": {k: list(v) for k, v in WINDOWS.items()},
        "maximum_source_quote_age_seconds": MAX_QUOTE_AGE,
        "interpretation": "Final reference: last fresh same-book quote 15–90 minutes before saved card start, after the decision; not an exact bout closing price. Missing references remain missing. Positive advantage favors the entry price.",
        "book_access": "Hypothetical recorded books; no account access or accepted wager is implied.",
        "coverage_scope": "Recorded moneyline forecasts and observed full-fight total lines; unquoted total lines are unknown.",
        "summary": {key: dict(value) for key, value in sorted(summary.items())},
        "rejected_quotes": dict(sorted(rejected.items())), "coverage": coverage,
        "moneyline_references": references, "moneyline_reference_summary": reference_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--now")
    args = parser.parse_args()
    decision = {"capture": True, "reason": "regular_or_manual_capture"}
    if os.environ.get("CAPTURE_SCHEDULE") == GATE_SCHEDULE:
        data = args.repo_root / "src/content/data"
        card = json.loads((data / "external/card_info.json").read_text())
        path = data / "market/capture_report.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        decision = capture_due(card, previous, args.now or datetime.now(timezone.utc))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write("capture=" + str(decision["capture"]).lower() + "\n")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
