"""Reproduce a bounded price diagnostic from a pinned git revision, offline.

No production files, decisions or policy parameters are written. The power
margin alternative is exploratory; these outcomes were already observed.
"""
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from market_capture_windows import price_window_report, quote_rejection, utc
from market_tracker.bayesian_kelly import BayesianKellyCalibrator
from market_tracker._common import implied_probability


def power_probability(first, second):
    """Unique positive k makes the two adjusted implied chances sum to one."""
    a, b = implied_probability(first), implied_probability(second)
    low, high = 0.0, 1.0
    while a ** high + b ** high > 1:
        high *= 2
    for _ in range(80):
        middle = (low + high) / 2
        if a ** middle + b ** middle > 1:
            low = middle
        else:
            high = middle
    return a ** ((low + high) / 2)


def probability_for(q, fighter_id, method="proportional"):
    if fighter_id not in (q["fighter_id"], q["opponent_id"]):
        raise ValueError("selected fighter absent from quote")
    p = (q["no_vig_fighter_probability"] if method == "proportional" else
         power_probability(q["fighter_moneyline"], q["opponent_moneyline"]))
    return p if q["fighter_id"] == fighter_id else 1 - p


def comparison_quotes(decision, target, quotes, metadata):
    """Match the original locked consensus eligibility, then audit freshness."""
    peers = [q for q in quotes
             if (q["event_id"], q["matchup_id"], q["capture_id"]) ==
                (decision["event_id"], decision["matchup_id"], decision["capture_id"])
             and q["book"].casefold() != target["book"].casefold()
             and q["quote_id"] in metadata
             and -300 <= metadata[q["quote_id"]]["source_quote_age_seconds"] <= 1800]
    if len({q["book"].casefold() for q in peers}) != len(peers) or len(peers) < 3:
        raise ValueError("missing or duplicate consensus books")
    peers.sort(key=lambda q: (q["book"].casefold(), q["quote_id"]))
    if any(set((q["fighter_id"], q["opponent_id"])) !=
           set((decision["fighter_id"], decision["opponent_id"])) for q in peers):
        raise ValueError("consensus fighter identity mismatch")
    p = statistics.mean(probability_for(q, decision["fighter_id"]) for q in peers)
    if not math.isclose(p, decision["market_probability"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("saved market probability cannot be reproduced")
    return peers


def book_omission_range(probabilities):
    if len(probabilities) < 4:
        return None, None
    removed = [(sum(probabilities) - p) / (len(probabilities) - 1) for p in probabilities]
    return min(removed), max(removed)


def scoring(rows, probability_key, target_key="target"):
    observed = [r for r in rows if r.get(probability_key) is not None and r.get(target_key) in (0, 1)]
    if not observed:
        return {"fights": 0}
    p = [min(max(r[probability_key], 1e-12), 1 - 1e-12) for r in observed]
    y = [r[target_key] for r in observed]
    return {"fights": len(p), "cards": len({r["event_id"] for r in observed}),
            "expected_wins": sum(p), "actual_wins": sum(y),
            "brier": statistics.mean((a - b) ** 2 for a, b in zip(p, y)),
            "log_loss": statistics.mean(-b * math.log(a) - (1 - b) * math.log(1 - a) for a, b in zip(p, y))}


def selected_summary(rows):
    scored = [r for r in rows if r["won"] is not None]
    return {"bets": len(rows), "settled_bets": len(scored), "cards": len({r["event_id"] for r in scored}),
            "wins": sum(r["won"] for r in scored),
            "profit_units": sum(r["profit_units"] for r in scored),
            "worse_winning_payout_profit_units": {str(h): sum((r["moneyline"] / 100 if r["moneyline"] > 0 else 100 / -r["moneyline"]) * (1-h)
                if r["won"] else -1 for r in scored) for h in (0, .02, .05)},
            "source_quality_failures": sum(bool(r["source_quality_failures"]) for r in rows),
            "retained_at_original_threshold": {key: sum(r[key] >= r["minimum_expected_return"] for r in rows)
                for key in ("proportional_market_ev", "power_market_ev")},
            "additional_book_omission_positive_edge": sum(r["omission_min_ev"] is not None and r["omission_min_ev"] > 0 for r in rows),
            "additional_book_omission_available": sum(r["omission_min_ev"] is not None for r in rows),
            "final_reference_statuses": dict(Counter(r["final_status"] for r in rows)),
            "mean_final_same_book_movement": mean_available(rows, "final_same_book_probability_movement"),
            "mean_final_independent_advantage": mean_available(rows, "final_independent_probability_advantage"),
            "mean_final_power_advantage": mean_available(rows, "final_power_probability_advantage"),
            "positive_final_independent_advantage": sum(r["final_independent_probability_advantage"] is not None and r["final_independent_probability_advantage"] > 0 for r in rows),
            "positive_final_power_advantage": sum(r["final_power_probability_advantage"] is not None and r["final_power_probability_advantage"] > 0 for r in rows),
            "probability_checks_on_selected_bets": {key: scoring(rows, key, "won") for key in
                ("recorded_probability", "proportional_market_probability", "power_market_probability", "calibrated_market_probability")}}


def mean_available(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return statistics.mean(values) if values else None


def main():
    design = json.loads((HERE / "selected_price_design.json").read_text())
    inputs = {}
    def read(name):
        path = "src/content/data/" + name
        data = subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(),
                                        "show", design["source_revision"] + ":" + path], cwd=ROOT)
        inputs[path] = sha256(data).hexdigest()
        return [json.loads(line) for line in data.splitlines() if line] if name.endswith(".jsonl") else json.loads(data)
    quotes = read("market/quote_snapshots.jsonl")
    metadata = read("market/quote_source_metadata.jsonl")
    calibration = read("market/equal_stake_experiment/policy.json")["calibration"]
    calibrator = BayesianKellyCalibrator(calibration)
    forecasts = read("market/forecast_captures.jsonl")
    captured = read("market/capture_report.json")["captured_at_utc"]
    qi = {q["quote_id"]: q for q in quotes}
    mi = {m["quote_id"]: m for m in metadata}
    selected, same_fight = [], []
    for strategy, prefix in (("locked_market", "paper"), ("market_first", "market_first_paper")):
        decisions = read("market/" + prefix + "_decisions.jsonl")
        settlements = {s["decision_id"]: s for s in read("market/" + prefix + "_settlements.jsonl")}
        finals = price_window_report({"moneyline": quotes}, metadata, forecasts, decisions, captured)
        final_by_id = {r["decision_id"]: r for r in finals["moneyline_references"]}
        for d in decisions:
            target = qi[d["reference_quote_id"]]
            peers = comparison_quotes(d, target, quotes, mi)
            settled = settlements.get(d["decision_id"], {})
            calibrated_available = utc(d["decision_issued_at_utc"]) >= utc(calibration["created_at_utc"])
            if strategy == "locked_market" and calibrated_available:
                p = d["market_probability"]
                assessment = calibrator.assessment(p, target["fighter_moneyline"], assessment_timing="historical_diagnostic")
                same_fight.append({"event_id": d["event_id"], "event_date": d["event_date"], "matchup_id": d["matchup_id"],
                    "target": settled.get("target"), "market": p, "calibrated_market": assessment["posterior_mean_probability"],
                    "winner_model": d["model_probability"], "power_market": statistics.mean(probability_for(q, d["fighter_id"], "power") for q in peers)})
            if d["paper_action"] == "pass":
                continue
            side = d["paper_action"]
            selected_id = d[side + "_id"]
            line = d["action_reference_moneyline"]
            decimal = 1 + (line / 100 if line > 0 else 100 / -line)
            p = [probability_for(q, selected_id) for q in peers]
            power = statistics.mean(probability_for(q, selected_id, "power") for q in peers)
            low, high = book_omission_range(p)
            calibrated = calibrator.assessment(statistics.mean(p), line, assessment_timing="historical_diagnostic") if calibrated_available else None
            final = final_by_id[d["decision_id"]]
            final_power = (statistics.mean(probability_for(qi[i], selected_id, "power") for i in final["other_book_quote_ids"])
                           if final.get("independent_probability") is not None else None)
            known_target = settled.get("target")
            won = (int(known_target == int(side == "fighter")) if known_target in (0, 1) else None)
            row = {"strategy": strategy, "event_date": d["event_date"], "event_id": d["event_id"],
                "matchup_id": d["matchup_id"], "decision_id": d["decision_id"], "selected_fighter_id": selected_id,
                "selection": target[side + "_name"], "book": target["book"], "moneyline": line,
                "recorded_probability": d["action_probability"], "minimum_expected_return": d["minimum_expected_return"],
                "recorded_ev": d[side + "_expected_return"], "proportional_market_probability": statistics.mean(p),
                "power_market_probability": power,
                "calibrated_market_probability": calibrated["posterior_mean_probability"] if calibrated else None,
                "calibrated_lower_probability": calibrated["posterior_lower_probability"] if calibrated else None,
                "proportional_market_ev": statistics.mean(p) * decimal - 1, "power_market_ev": power * decimal - 1,
                "calibrated_market_ev": calibrated["posterior_mean_probability"] * decimal - 1 if calibrated else None,
                "omission_min_ev": low * decimal - 1 if low is not None else None,
                "omission_max_ev": high * decimal - 1 if high is not None else None,
                "other_book_count": len(peers), "comparison_mean_overround": statistics.mean(q["overround"] for q in peers),
                "target_quote_age_seconds": mi[target["quote_id"]]["source_quote_age_seconds"],
                "source_quality_failures": [q["quote_id"] + ":" + reason for q in [target, *peers]
                    if (reason := quote_rejection(q, mi.get(q["quote_id"]), utc(d["decision_issued_at_utc"])))],
                "other_book_quote_ids": [q["quote_id"] for q in peers], "won": won,
                "profit_units": settled.get("hypothetical_profit_units"), "final_status": final["status"],
                "final_quote_id": final.get("reference_quote_id"), "final_observed_at_utc": final.get("observed_at_utc"),
                "final_same_book_probability_movement": final.get("same_book_probability_movement"),
                "final_independent_probability_advantage": final.get("independent_probability_advantage"),
                "final_power_probability_advantage": final_power - 1 / decimal if final_power is not None else None,
                "longshot": line >= 300}
            selected.append(row)
    summaries = {}
    for strategy in sorted({r["strategy"] for r in selected}):
        rows = [r for r in selected if r["strategy"] == strategy]
        summaries[strategy] = {"all": selected_summary(rows),
            "by_card": {date: selected_summary([r for r in rows if r["event_date"] == date]) for date in sorted({r["event_date"] for r in rows})},
            "by_book": {book: selected_summary([r for r in rows if r["book"] == book]) for book in sorted({r["book"] for r in rows})},
            "longshots": selected_summary([r for r in rows if r["longshot"]])}
    evidence = {"design": design, "input_sha256": inputs, "interpretation": "Selected-price sensitivity on already observed outcomes. Policies are separate, not additive. Book-omission ranges are not confidence intervals.",
        "strategy_summary": summaries, "same_fight_probability_scores": {key: scoring(same_fight, key) for key in
            ("market", "calibrated_market", "winner_model", "power_market")},
        "same_fight_scores_by_card": {date: {key: scoring([r for r in same_fight if r["event_date"] == date], key)
            for key in ("market", "calibrated_market", "winner_model", "power_market")}
            for date in sorted({r["event_date"] for r in same_fight})},
        "same_fight_rows": same_fight, "selected_rows": selected}
    (HERE / "selected_price_evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    fields = [key for key in selected[0] if key not in ("source_quality_failures", "other_book_quote_ids")]
    with (HERE / "selected_price_rows.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({"same_fight_probability_scores": evidence["same_fight_probability_scores"],
                      "selected_strategies": {key: value["all"] for key, value in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
