"""Audit duration labels and published totals without changing production data.

Usage: python scripts/audit_profitability_duration.py --output-dir audit/profitability/duration
Only reports under --output-dir and an automatically removed temporary archive
are written. No model, production ledger, or website artifact is rebuilt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesian_total_calibration import fit_total_calibration  # noqa: E402
from fight_semantics import schedule_from_row  # noqa: E402
from market_tracker.bankroll import (  # noqa: E402
    archive_upcoming_bet_board,
    build_bet_performance_publication,
    validate_published_bet_archive,
)
from market_tracker.bayesian_kelly import (  # noqa: E402
    validate_bayesian_kelly_assessment,
)


INPUTS = {
    "training_data": "src/content/data/processed/ufc_fights_point_in_time.csv",
    "evaluation": "src/content/data/external/outcome_model_evaluation.json",
    "forecasts": "src/content/data/external/outcome_forecasts.json",
    "board": "src/content/data/market/upcoming_bet_board.json",
    "archive": "src/content/data/market/published_bet_snapshots.json",
    "moneyline_calibration": "src/content/data/model_research/bayesian_kelly_market_calibration.json",
}
CODE_INPUTS = [
    "src/fight_semantics.py", "src/fight_predictor/outcome_model.py",
    "src/bayesian_total_calibration.py", "src/market_tracker/bankroll.py",
    "src/market_tracker/bayesian_kelly.py", "src/upcoming_bet_board.py",
    "src/capture_market_snapshot.py", "src/update_bet_performance.py",
    "src/data_handler/data_handler.py", "tests/test_outcome_model.py",
    "scripts/audit_profitability_duration.py",
]


def evaluation_time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Reproduce evaluate_outcome_model's ten-year, whole-event 80/20 split."""
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(
        ["date", "event_id", "bout_order", "fight_id"], kind="stable"
    )
    frame = frame[frame["date"] >= frame["date"].max() - pd.DateOffset(years=10)]
    frame = frame.reset_index(drop=True)
    if len(frame) < 1000:
        raise ValueError("The production outcome evaluation requires at least 1,000 fights")
    cutoff = frame.iloc[int(len(frame) * 0.8)]["date"]
    frame["split"] = frame["date"].map(
        lambda value: "development" if value < cutoff else "holdout"
    )
    return frame, cutoff


def schedule_evidence(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count schedule sources and expose the exact rows behind the counts."""
    columns = [
        "split", "date", "event_id", "fight_id", "label_method",
        "label_finish_round", "label_total_fight_seconds", "label_time_format",
    ]
    evidence = frame[columns].copy()
    schedules = frame.apply(schedule_from_row, axis=1)
    evidence["scheduled_rounds"] = schedules.map(lambda value: value[0])
    evidence["schedule_basis"] = schedules.map(lambda value: value[1])
    evidence["over_3_5"] = evidence["label_total_fight_seconds"] > 1050
    evidence["finish_before_round_4"] = evidence["label_total_fight_seconds"] <= 900
    evidence["decision"] = evidence["label_method"].map(
        lambda value: "DEC" in str(value).upper()
    )
    evidence["explicit_schedule"] = evidence["schedule_basis"].eq("explicit_time_format")
    grouped = evidence.groupby(
        ["split", "scheduled_rounds", "schedule_basis"], dropna=False, sort=True
    ).agg(
        fights=("fight_id", "size"),
        over_3_5_fights=("over_3_5", "sum"),
        finish_before_round_4_fights=("finish_before_round_4", "sum"),
        decision_fights=("decision", "sum"),
    ).reset_index()
    grouped["over_3_5_rate"] = grouped["over_3_5_fights"] / grouped["fights"]
    return evidence, grouped


def _counts(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "fights": int(len(frame)),
        "events": int(frame["event_id"].nunique()),
        "explicit_schedule_fights": int(frame["explicit_schedule"].sum()),
        "inferred_or_unknown_schedule_fights": int((~frame["explicit_schedule"]).sum()),
        "over_3_5_fights": int(frame["over_3_5"].sum()),
        "over_3_5_rate": float(frame["over_3_5"].mean()) if len(frame) else None,
        "finish_before_round_4_fights": int(frame["finish_before_round_4"].sum()),
        "decision_fights": int(frame["decision"].sum()),
    }


def insufficient_check_reproduction() -> dict[str, object]:
    """Use uneven card sizes to meet overall support but leave two test fights."""
    rows = []
    for index in range(40):
        event = 0 if index < 33 else index - 32
        rows.append({
            "event_date": f"2025-01-{event + 1:02d}", "event_id": f"event-{event}",
            "fight_id": f"fight-{index}", "line": 1.5,
            "model_probability": 0.6, "target": index % 2,
        })
    line = fit_total_calibration(pd.DataFrame(rows))["lines"]["1.5"]
    return {
        "fixture": "40 synthetic fights on 8 dates; first event has 33 fights",
        "production_result": {key: value for key, value in line.items() if key != "posterior"},
        "insufficient_check_enabled": (
            line["status"] == "available"
            and line["chronological_check"]["status"] == "too_small"
        ),
        "scope": "Synthetic branch reproduction, not an additional historical sample",
    }


def _build_archive_result(archive: dict[str, object]) -> dict[str, object]:
    """Call the actual builder without its production writer or official ledgers."""
    try:
        publication = build_bet_performance_publication(
            decisions=(), settlements=(), quotes=(), forecasts=(),
            archive=copy.deepcopy(archive),
        )
    except ValueError as error:
        return {"status": "error", "error_type": type(error).__name__, "message": str(error)}
    return {"status": "success", "record_count": publication["record_count"]}


def archive_reachability(board: dict, archive: dict) -> dict[str, object]:
    """Compare the saved archive to its next-capture state in temporary storage."""
    validate_published_bet_archive(archive)
    total_rows = [bet for bet in board["bets"] if bet["category"] == "Total rounds"]
    isolated = {"status": "not_applicable", "reason": "No current total assessment"}
    if total_rows and total_rows[0].get("bayesian_kelly"):
        try:
            validate_bayesian_kelly_assessment(total_rows[0]["bayesian_kelly"])
        except ValueError as error:
            isolated = {"status": "error", "message": str(error)}
        else:
            isolated = {"status": "success"}
    existing = _build_archive_result(archive)
    with tempfile.TemporaryDirectory(prefix="ufc-duration-audit-") as scratch:
        temporary_archive = Path(scratch) / "published_bet_snapshots.json"
        temporary_archive.write_text(json.dumps(archive), encoding="utf-8")
        appended = archive_upcoming_bet_board(board, temporary_archive)
        after_append = _build_archive_result(appended)
    policies = Counter(
        (row.get("bayesian_kelly") or {}).get("policy_version", "missing")
        for row in archive["snapshots"] if row.get("category") == "Total rounds"
    )
    return {
        "saved_snapshot_count": archive["snapshot_count"],
        "saved_total_assessment_policies": dict(sorted(policies.items())),
        "saved_archive_builder": existing,
        "isolated_current_total_to_moneyline_validator": isolated,
        "temporary_archive_after_current_board_snapshot_count": appended["snapshot_count"],
        "temporary_archive_after_current_board_builder": after_append,
        "scope": (
            "Real archive_upcoming_bet_board and build_bet_performance_publication paths; "
            "temporary archive only, empty official inputs and no result/support enrichment. "
            "This isolates archive compatibility; it does not run the live updater or assert "
            "that the currently published performance file is broken."
        ),
        "call_path": [
            "capture_market_snapshot.capture -> archive_upcoming_bet_board",
            "update_bet_performance -> build_bet_performance_publication",
            "_archive_records copies the stored bayesian_kelly assessment",
            "_attach_bayesian_kelly validates it as a moneyline assessment before checking category",
        ],
    }


def forecast_impact(forecasts: dict, board: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rows = []
    for matchup in forecasts["matchups"]:
        rows.append({
            "matchup_id": matchup["matchup_id"],
            "fighter_name": matchup["fighter_name"], "opponent_name": matchup["opponent_name"],
            "scheduled_rounds": matchup.get("scheduled_rounds"),
            "schedule_basis": matchup.get("schedule_basis"),
            "method_probability_count": len(matchup.get("method_probabilities", {})),
            "total_probability_count": len(matchup.get("total_round_over_probabilities", {})),
            "decision_probability": matchup.get("method_probabilities", {}).get("decision"),
            "over_3_5_probability": matchup.get("total_round_over_probabilities", {}).get("3.5"),
            "forecast_status": matchup.get("forecast_status"),
        })
    bets = []
    for bet in board["bets"]:
        if bet["category"] != "Total rounds":
            continue
        assessment = bet.get("bayesian_kelly") or {}
        bets.append({
            "event_id": bet["event_id"], "matchup_id": bet["matchup_id"],
            "fighter_name": bet["fighter_name"], "opponent_name": bet["opponent_name"],
            "selection": bet["selection"], "scheduled_rounds": bet.get("scheduled_rounds"),
            "raw_probability": bet["estimated_win_probability"],
            "raw_expected_return": bet["estimated_expected_return"],
            "calibrated_mean_probability": assessment.get("posterior_mean_probability"),
            "recommended_fraction": assessment.get("recommended_fraction", 0.0),
            "model_id": bet.get("model_id"),
        })
    return {
        "forecast_model_id": forecasts.get("model_id"),
        "forecast_training_fights": forecasts.get("training_fights"),
        "forecast_issued_at_utc": forecasts.get("forecast_issued_at_utc"),
        "board_observed_at_utc": board.get("observed_at_utc"),
        "method_forecast_matchups": sum(row["method_probability_count"] > 0 for row in rows),
        "method_probability_values": sum(row["method_probability_count"] for row in rows),
        "total_forecast_matchups": sum(row["total_probability_count"] > 0 for row in rows),
        "total_probability_values": sum(row["total_probability_count"] for row in rows),
        "five_round_forecast_matchups": sum(row["scheduled_rounds"] == 5 for row in rows),
        "current_total_bets": len(bets),
        "current_total_bet_matchups": len({row["matchup_id"] for row in bets}),
        "five_round_current_total_bets": sum(row["scheduled_rounds"] == 5 for row in bets),
        "sum_current_total_recommended_fractions": sum(row["recommended_fraction"] for row in bets),
        "interpretation": (
            "These forecasts use the same duration/method model and require reevaluation. "
            "Counts describe affected provenance, not a measured correction to each probability "
            "or proof that every forecast is inaccurate. Board and forecast publications can "
            "have different issue times/model identities; each is preserved separately."
        ),
    }, pd.DataFrame(rows), pd.DataFrame(bets)


def _hashes() -> dict[str, str]:
    return {path: sha256((ROOT / path).read_bytes()).hexdigest()
            for path in [*INPUTS.values(), *CODE_INPUTS]}


def _source_link(output: Path, path: str, needle: str, label: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8").splitlines()
    line = next(index for index, value in enumerate(text, 1) if needle in value)
    relative = Path(os.path.relpath(ROOT / path, output)).as_posix()
    return f"[{label}]({relative}#L{line})"


def write_report(output: Path, summary: dict) -> None:
    schedules = summary["schedule_bias"]
    dev = schedules["development_five_round"]
    known = schedules["holdout_explicit_five_round"]
    unknown = schedules["holdout_inferred_five_round"]
    impact = summary["published_impact"]
    archive = summary["archive_reachability"]
    later = summary["insufficient_calibration_check"]["production_result"]
    long_line_support = "; ".join(
        f"{line}: {item['training_fights']} training fights and "
        f"{item.get('chronological_check', {}).get('holdout_fights', 0)} later test fights"
        for line, item in summary["calibration_line_support"].items()
        if float(line) > 3
    )
    link = lambda path, needle, label: _source_link(output, path, needle, label)
    report = f"""# Duration and totals profitability audit

The historical duration model learns an unrealistically durable five-round group because missing schedules are reconstructed from how fights ended. Correcting those source labels must precede any claim that the current large totals returns are real. This audit writes findings only; it does not change recommendations, models, or production ledgers.

## 1. Highest priority: schedules are inferred from outcomes

The reproduced development sample has {schedules['development']['fights']:,} fights; {schedules['development']['inferred_or_unknown_schedule_fights']:,} lack an explicit scheduled length. Of its {dev['fights']} five-round fights, {dev['decision_fights']} are decisions and {dev['fights'] - dev['decision_fights']} are finishes; **{dev['finish_before_round_4_fights']} end within the first three rounds**. {dev['over_3_5_fights']}/{dev['fights']} last over 3.5 rounds, reproducing the saved smoothed base rate of {schedules['reproduced_smoothed_development_over_3_5_rate']:.2%}.

In the later sample, only {known['fights']} fights have explicit five-round schedules and {known['over_3_5_fights']}/{known['fights']} ({known['over_3_5_rate']:.2%}) last over 3.5 rounds. The inferred five-round subset instead has {unknown['over_3_5_fights']}/{unknown['fights']} ({unknown['over_3_5_rate']:.2%}). These are differently selected groups, not a controlled estimate of the true change in UFC fight length.

Cause: {link('src/fight_semantics.py', 'if final_round > 3:', 'schedule inference')} assigns early finishes three rounds and late finishes five when true schedules are missing. {link('src/fight_predictor/outcome_model.py', 'rounds = _scheduled_rounds(fight)', 'training')} and {link('src/fight_predictor/outcome_model.py', 'for _, row in holdout.iterrows()', 'evaluation')} both use this result-dependent input. {link('src/data_handler/data_handler.py', "if 'time_format' not in old_ufc_fights_reported_doubled:", 'historical migration')} creates blank schedules, while {link('src/data_handler/data_handler.py', "time_formats = aggregate['time_format']", 'fight-page extraction')} already reads true schedules for new pages.

Consequence: scheduled five-round fights that finished early can be mislabeled as three rounds, biasing both groups. The live model shares duration and method predictions: {impact['method_forecast_matchups']} matchups have {impact['method_probability_values']} method probabilities and {impact['total_forecast_matchups']} matchups have {impact['total_probability_values']} totals probabilities requiring reevaluation. The current board has {impact['current_total_bets']} totals across {impact['current_total_bet_matchups']} fights, including {impact['five_round_current_total_bets']} bets on five-round fights. Their separate stakes sum to {impact['sum_current_total_recommended_fractions']:.0%} of bankroll. This does not measure corrected fair odds or prove every recommendation loses money.

Proposed correction: backfill scheduled lengths from independently recorded fight-page or event metadata, record its source, and exclude unresolved schedules from duration training and testing. Rebuild duration/method forecasts and their calibration only after that repair. Retain raw and calibrated estimates separately and calculate recommendations from the same probability used for stake decisions. A positive-slope-only calibration cannot move an over-50% estimate below 50%; calibration alone cannot fix missing early finishes.

Acceptance checks: an independently known five-round bout remains five rounds whether its result is an early knockout or a decision; a missing schedule plus finish round 1 remains unknown; both training and later evaluation include independently scheduled early finishes; development/holdout dates and identities stay separate; report results by schedule source and market line, with actual offered pre-fight prices before asserting profitability. The existing {link('tests/test_outcome_model.py', 'def test_unknown_schedule_is_not_silently_assumed', 'unknown-schedule test')} omits finish round and therefore misses the real-data branch.

## 2. High priority: small later calibration checks can enable staking

The synthetic reproduction meets the current 40-fight/eight-event minimum but leaves only {later['chronological_check']['holdout_fights']} later fights. Production returns line status **{later['status']}** with check status **{later['chronological_check']['status']}**. {link('src/bayesian_total_calibration.py', 'if chronological_check.get("status") == "complete"', 'The rejection branch')} only rejects completed checks that worsen probability accuracy. Real long-total support is {long_line_support}; this is probability checking, not an odds-based demonstration of betting profit.

Proposed correction: require a complete, sufficiently supported later check before assigning positive stakes, and reassess the minimum sample after repairing schedules. The available data do not establish an optimal minimum or stake cap. Acceptance checks: incomplete/too-small checks produce unavailable sizing; available calibrations meet documented fight/event counts and later-period requirements; the corrected probabilities must beat simple and market-based comparisons before release.

## 3. High priority: the next totals archive cannot feed the replay builder

The existing archive contains {archive['saved_snapshot_count']} snapshots and its isolated replay builder result is **{archive['saved_archive_builder']['status']}**. It does not yet contain the current totals calibration policy. After the actual archiver appends the current board to a temporary copy, there are {archive['temporary_archive_after_current_board_snapshot_count']} snapshots and the actual replay builder returns **{archive['temporary_archive_after_current_board_builder']['status']}**: `{archive['temporary_archive_after_current_board_builder'].get('message', '')}`.

This is a reachable next-capture compatibility failure, **not a claim that the saved website performance file is currently broken**. {link('src/capture_market_snapshot.py', 'archive_upcoming_bet_board(', 'The capture path')} archives the board; {link('src/market_tracker/bankroll.py', '"bayesian_kelly": item.get("bayesian_kelly")', 'archive conversion')} preserves totals assessments; {link('src/market_tracker/bankroll.py', 'record["bayesian_kelly"] = validate_bayesian_kelly_assessment(existing)', 'replay enrichment')} sends them to the moneyline validator before checking the category. The reproduction uses empty official inputs and does not run the live writer or modify any production archive.

Proposed correction: dispatch calibration validation by market/policy and preserve the exact assessment published before each fight. Acceptance checks: current totals pass archive-to-replay round-trip; historical unavailable assessments stay unavailable; delayed outcomes cannot replace a recorded stake; both current saved and newly appended archives remain valid.

## Reproduction and evidence

Run `python scripts/audit_profitability_duration.py --output-dir audit/profitability/duration` from the repository root. `summary.json` contains input/source SHA-256 hashes, split reproduction checks, counts, calibration support, current forecast identities, and both archive-builder results. `schedule_provenance.csv` summarizes groups; `schedule_fights.csv` lists every included fight; `current_forecasts.csv` and `current_total_bets.csv` identify affected published rows. No models are fit except a small synthetic calibration fixture held in memory. Production input hashes are verified unchanged before reports are written.
"""
    (output / "findings.md").write_text(report, encoding="utf-8")


def run(output: Path) -> dict[str, object]:
    before = _hashes()
    objects = {key: json.loads((ROOT / path).read_text(encoding="utf-8"))
               for key, path in INPUTS.items() if key != "training_data"}
    data = pd.read_csv(ROOT / INPUTS["training_data"], low_memory=False)
    frame, cutoff = evaluation_time_split(data)
    evidence, grouped = schedule_evidence(frame)
    dev = evidence[evidence["split"] == "development"]
    test = evidence[evidence["split"] == "holdout"]
    dev_five = dev[dev["scheduled_rounds"] == 5]
    test_five = test[test["scheduled_rounds"] == 5]
    impact, forecast_rows, bet_rows = forecast_impact(objects["forecasts"], objects["board"])
    evaluation = objects["evaluation"]
    baseline = float((dev_five["over_3_5"].sum() + 1) / (len(dev_five) + 2))
    checks = {
        "development_fight_count_matches_saved_evaluation": len(dev) == evaluation["development_fights"],
        "holdout_fight_count_matches_saved_evaluation": len(test) == evaluation["holdout_fights"],
        "holdout_start_matches_saved_evaluation": cutoff.date().isoformat() == evaluation["holdout_start"],
        "smoothed_five_round_base_rate_matches_saved_evaluation": abs(
            baseline - evaluation["total_rounds"]["over_3_5_rounds"]["development_base_rate"]
        ) < 1e-12,
        "forecast_and_evaluation_training_input_identity_matches": (
            objects["forecasts"].get("training_input_sha256") == evaluation.get("training_input_sha256")
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Saved evaluation no longer matches audit reproduction: {checks}")
    summary = {
        "schema_version": 1,
        "audit_only": True,
        "input_sha256": before,
        "reproduction_checks": checks,
        "schedule_bias": {
            "holdout_start": cutoff.date().isoformat(),
            "development": _counts(dev), "holdout": _counts(test),
            "development_five_round": _counts(dev_five),
            "holdout_explicit_five_round": _counts(test_five[test_five["explicit_schedule"]]),
            "holdout_inferred_five_round": _counts(test_five[~test_five["explicit_schedule"]]),
            "reproduced_smoothed_development_over_3_5_rate": baseline,
            "provenance": json.loads(grouped.to_json(orient="records")),
        },
        "calibration_line_support": {
            line: {key: value for key, value in item.items() if key != "posterior"}
            for line, item in evaluation["bayesian_total_calibration"]["lines"].items()
        },
        "published_impact": impact,
        "insufficient_calibration_check": insufficient_check_reproduction(),
        "archive_reachability": archive_reachability(objects["board"], objects["archive"]),
        "limitations": [
            "This audit does not retrain or estimate corrected odds, returns, or optimal stakes.",
            "Known versus inferred schedules form different selected groups; rate differences are diagnostic, not causal estimates.",
            "The synthetic fixture tests a branch, not historical profitability.",
            "The archive reproduction exercises actual helpers with temporary storage and empty official ledgers, not the live updater.",
        ],
    }
    if _hashes() != before:
        raise RuntimeError("An audit input changed during execution; rerun on a stable snapshot")
    summary["production_inputs_unchanged"] = True
    output.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(output / "schedule_fights.csv", index=False, date_format="%Y-%m-%d")
    grouped.to_csv(output / "schedule_provenance.csv", index=False)
    forecast_rows.to_csv(output / "current_forecasts.csv", index=False)
    bet_rows.to_csv(output / "current_total_bets.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(output, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output_dir.resolve())
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "development_five_round_fights": summary["schedule_bias"]["development_five_round"]["fights"],
        "development_five_round_early_finishes": summary["schedule_bias"]["development_five_round"]["finish_before_round_4_fights"],
        "production_inputs_unchanged": summary["production_inputs_unchanged"],
        "archive_after_append": summary["archive_reachability"]["temporary_archive_after_current_board_builder"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
