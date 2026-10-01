"""Offline eligibility comparison only: never write decisions or calculate returns."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from market_tracker.equal_stake_experiment import build_records
from market_tracker.bayesian_kelly import BayesianKellyCalibrator
from market_tracker._common import utc_datetime
from update_market_first_paper import _stores

BASE = "e8444907cd409a2aadaa1ba9f04b6c9aa9c6d47b"


def main():
    original = subprocess.check_output([
        "git", "-c", "safe.directory=" + ROOT.as_posix(), "show",
        BASE + ":src/market_tracker/equal_stake_experiment.py"], cwd=ROOT).decode("utf-8")
    legacy = types.ModuleType("market_tracker._timing_replay_legacy")
    exec(compile(original, "legacy_equal_stake_experiment.py", "exec"), legacy.__dict__)
    market = ROOT / "src/content/data/market"
    paths = list((market / "equal_stake_experiment").glob("*.json"))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    policy = json.loads((market / "equal_stake_experiment/policy.json").read_text())
    calibrator = BayesianKellyCalibrator(policy["calibration"])
    qs, fs, ms, _, _ = _stores()
    quotes, forecasts, metadata = qs.read(), fs.read(), ms.read()
    quotes = [q for q in quotes if q.event_date in ("2026-09-19", "2026-09-26")
              and q.event_start_utc and 20 * 3600 <= (
                  utc_datetime(q.event_start_utc, "start")
                  - utc_datetime(q.observed_at_utc, "observed")).total_seconds() <= 28 * 3600]
    output = {"source_revision": BASE, "interpretation": "Counterfactual eligibility only. No historical bets added and no returns inferred.",
              "quotes_in_card_windows": len(quotes), "captures": [], "by_card": {}}
    old_records, new_records = [], []
    for observed in sorted({q.observed_at_utc for q in quotes}):
        capture_quotes = [q for q in quotes if q.observed_at_utc == observed]
        old = legacy.build_records(capture_quotes, forecasts, metadata, old_records, policy, calibrator, observed)
        diagnostic = {}
        new = build_records(capture_quotes, forecasts, metadata, new_records, policy, calibrator, observed,
                            diagnostics=diagnostic)
        old_records.extend(old)
        new_records.extend(new)
        output["captures"].append({"observed_at_utc": observed,
            "legacy_eligible": len(old), "repaired_eligible": len(new), "diagnostics": diagnostic})
    for day in sorted({q.event_date for q in quotes}):
        old_ids = {r["matchup_id"] for r in old_records if r["event_date"] == day}
        new_ids = {r["matchup_id"] for r in new_records if r["event_date"] == day}
        output["by_card"][day] = {"legacy_eligible_fights": len(old_ids),
            "repaired_eligible_fights": len(new_ids), "recovered_matchup_ids": sorted(new_ids - old_ids)}
    output["legacy_eligible_fights"] = len(old_records)
    output["repaired_eligible_fights"] = len(new_records)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output["all_experiment_files_unchanged"] = True
    destination = Path(__file__).with_name("timing_repair_replay.json")
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("quotes_in_card_windows", "legacy_eligible_fights",
                                           "repaired_eligible_fights", "all_experiment_files_unchanged")}))


if __name__ == "__main__":
    main()
