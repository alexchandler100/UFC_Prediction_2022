"""Read the pinned public snapshot; write audit evidence, never production data.

Run from the repository root with the project's Python environment. The source
manifest records the commit and hashes. Retrieve missing files from the raw
GitHub URL at that exact commit, preserving their paths under snapshot/.
"""
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "snapshot/src/content/data"


def read(relative):
    return json.loads((DATA / relative).read_text(encoding="utf-8"))


def ledger(relative):
    return [json.loads(line) for line in (DATA / relative).read_text().splitlines() if line]


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def metrics(probabilities, targets):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(targets, dtype=float)
    return {
        "fights": len(p),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier_score": float(((p - y) ** 2).mean()),
        "accuracy_with_half_credit_for_ties": float(
            (((p > .5) == y) * (p != .5) + .5 * (p == .5)).mean()
        ),
    }


def main():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for record in manifest["files"]:
        path = ROOT / "snapshot" / record["path"]
        assert sha256(path.read_bytes()).hexdigest() == record["sha256"], path
    result = {"remote_sha": manifest["remote_sha"], "verified_source_files": len(manifest["files"])}

    history = pd.DataFrame(read("external/prediction_history.json"))
    history["day"] = pd.to_datetime(history["date"], unit="ms").dt.strftime("%Y-%m-%d")
    paired = history[
        history["actual result"].isin(["W", "L"])
        & history["model probability"].notna()
        & history["market no-vig fighter probability"].notna()
        & (history.day >= "2026-08-01")
    ].copy()
    paired["target"] = (paired["actual result"] == "W").astype(int)
    pm, pq = paired["model probability"], paired["market no-vig fighter probability"]
    paired["equal_blend"] = 1 / (1 + np.exp(-.5 * (np.log(pm / (1 - pm)) + np.log(pq / (1 - pq)))))
    columns = {"model": "model probability", "market": "market no-vig fighter probability", "blend": "equal_blend"}
    result["weekly_published_probabilities"] = {
        day: {name: metrics(group[col], group.target) for name, col in columns.items()}
        for day, group in [*list(paired.groupby("day")), ("pooled", paired)]
    }
    paired[["day", "event id", "fight id", "fighter name", "opponent name", "target",
            *columns.values(), "forecast issued at", "odds observed at", "model trained through"]].to_csv(
        ROOT / "weekly_probability_rows.csv", index=False)

    decisions = ledger("market/paper_decisions.jsonl")
    settlements = {r["decision_id"]: r for r in ledger("market/paper_settlements.jsonl")}
    forecasts = {r["forecast_capture_id"]: r for r in ledger("market/forecast_captures.jsonl")}
    bets = []
    for d in decisions:
        if d["paper_action"] == "pass" or d["decision_id"] not in settlements:
            continue
        s, f = settlements[d["decision_id"]], forecasts[d["forecast_capture_id"]]
        bets.append({"day": d["event_date"], "selection": f[d["paper_action"] + "_name"],
                     "moneyline": d["action_reference_moneyline"], "model_weight": d["selected_gamma"],
                     "profit_units": s["hypothetical_profit_units"], "decision_id": d["decision_id"]})
    result["main_paper_bets"] = bets
    methods = read("market/method_paper/report.json")
    settled = [r for r in methods["recommendations"] if r.get("settlement_status") in ("win", "loss")]
    result["method_settled_bets"] = {
        "counts": dict(Counter(r["settlement_status"] for r in settled)),
        "profit_units": sum(r["profit_units"] for r in settled),
        "model_expected_wins": sum(r["probability"] for r in settled),
        "mean_claimed_expected_return": float(np.mean([r["expected_return"] for r in settled])),
    }

    quotes = ledger("market/quote_snapshots.jsonl")
    metadata = {r["quote_id"]: r for r in ledger("market/quote_source_metadata.jsonl")}
    counts, per_day, rejected, eligible = Counter(), defaultdict(Counter), set(), set()
    for q in quotes:
        if q["event_date"] < "2026-09-06":
            continue
        observed, start = utc(q["observed_at_utc"]), utc(q["event_start_utc"])
        if not 20 <= (start - observed).total_seconds() / 3600 <= 28:
            continue
        counts["t24_quotes"] += 1
        per_day[q["event_date"]]["t24_quotes"] += 1
        m = metadata.get(q["quote_id"])
        if m is None:
            counts["no_metadata"] += 1
            continue
        if utc(m["source_commence_time_utc"]) != start:
            counts["different_provider_start"] += 1
            per_day[q["event_date"]]["different_provider_start"] += 1
            rejected.add(q["matchup_id"])
        else:
            counts["same_start"] += 1
            eligible.add(q["matchup_id"])
    result["equal_stake_time_comparison"] = {
        "quotes": dict(counts), "by_day": dict(per_day),
        "matchups_with_rejected_quotes": len(rejected),
        "matchups_with_same_start_quotes": len(eligible),
        "note": "Static timing diagnostic; not a replacement decision ledger or a counterfactual return claim.",
    }

    raw = pd.read_csv(DATA / "processed/ufc_fights_reported_doubled.csv", low_memory=False)
    for source, dest in [("event_url", "event_id"), ("fighter_url", "fighter_id"), ("opponent_url", "opponent_id")]:
        raw[dest] = raw[source].str.rstrip("/").str.rsplit("/", n=1).str[-1]
    index = raw.set_index(["event_id", "fighter_id", "opponent_id"]).sort_index()
    sim_rows, exclusions, seen = [], Counter(), set()
    simulations = [json.loads(path.read_text()) for path in (DATA / "simulation/upcoming_matchups").glob("*.json")]
    for sim in sorted(simulations, key=lambda row: (row["forecast_issued_at_utc"], row["record_sha256"])):
        key = (sim["event_id"], sim["fighter_id"], sim["opponent_id"])
        if key not in index.index:
            exclusions["not_in_completed_results"] += 1
            continue
        actual = index.loc[key]
        if isinstance(actual, pd.DataFrame):
            assert len(actual) == 1, key
            actual = actual.iloc[0]
        assert isinstance(actual, pd.Series), key
        physical = (key[0], *sorted(key[1:]))
        if utc(sim["forecast_issued_at_utc"]).date().isoformat() >= actual["date"]:
            exclusions["not_provably_before_event_day"] += 1
            continue
        if actual["result"] not in ("W", "L"):
            exclusions["non_binary_result"] += 1
            continue
        if physical in seen:
            exclusions["later_record_of_same_physical_fight"] += 1
            continue
        seen.add(physical)
        probs = sim["aggregate"]["outcome_probabilities"]
        summaries = {r["statistic"]: r for r in sim["aggregate"]["statistic_summaries"]}
        red = sum(v for k, v in probs.items() if k.startswith("red_"))
        blue = sum(v for k, v in probs.items() if k.startswith("blue_"))
        # KO/SUB/decision counts use unconditional path probabilities. Winner
        # scoring conditions on a decisive outcome, matching the existing report.
        sim_rows.append({
            "event_id": sim["event_id"], "day": actual["date"], "fight_url": actual["fight_url"],
            "fighter": sim["fighter_name"], "opponent": sim["opponent_name"],
            "forecast_issued_at_utc": sim["forecast_issued_at_utc"],
            "mechanics_profile_id": sim["mechanics_profile_id"],
            "simulated_rounds": sim["scheduled_rounds"],
            "recorded_schedule": actual["time_format"],
            "schedule_matches": pd.notna(actual["time_format"]) and int(actual["time_format"].split()[0]) == sim["scheduled_rounds"],
            "target": int(actual["result"] == "W"), "winner_probability": red / (red + blue),
            "decision_probability": probs["red_decision"] + probs["blue_decision"],
            "actual_decision": int("DEC" in actual["method"]),
            "ko_probability": probs["red_ko_tko"] + probs["blue_ko_tko"],
            "actual_ko": int(actual["method"] == "KO/TKO"),
            "submission_probability": probs["red_submission"] + probs["blue_submission"],
            "actual_submission": int(actual["method"] == "SUB"),
            "predicted_seconds": summaries["duration_seconds"]["mean"],
            "actual_seconds": float(actual["total_fight_time"]),
        })
    frame = pd.DataFrame(sim_rows)
    frame.to_csv(ROOT / "simulation_result_rows.csv", index=False)
    result["retained_simulations"] = {"exclusions": dict(exclusions), "by_card": {}, "by_profile": {}}
    for group_type, groups in [("by_card", frame.groupby("day")), ("by_profile", frame.groupby("mechanics_profile_id")), ("pooled", [("all", frame)]), ("schedule_matched", [("matched", frame[frame.schedule_matches])])]:
        summaries = {}
        for name, group in groups:
            summaries[name] = {
                "fights": len(group), "cards": group.event_id.nunique(),
                "winner": metrics(group.winner_probability, group.target),
                "predicted_decisions": float(group.decision_probability.sum()),
                "actual_decisions": int(group.actual_decision.sum()),
                "predicted_kos": float(group.ko_probability.sum()), "actual_kos": int(group.actual_ko.sum()),
                "predicted_submissions": float(group.submission_probability.sum()),
                "actual_submissions": int(group.actual_submission.sum()),
                "mean_duration_error_seconds": float((group.predicted_seconds - group.actual_seconds).mean()),
            }
        result["retained_simulations"][group_type] = summaries
    result["retained_simulations"]["schedule_mismatches"] = frame.loc[
        ~frame.schedule_matches, ["day", "fighter", "opponent", "simulated_rounds", "recorded_schedule"]
    ].to_dict("records")
    (ROOT / "checkpoint_evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "weekly_published_probabilities"}, indent=2))


if __name__ == "__main__":
    main()
