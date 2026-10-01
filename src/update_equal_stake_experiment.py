"""Freeze new paper comparisons and settle them using existing local services."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from market_tracker._storage import atomic_write_text, exclusive_store_lock
from market_tracker.bayesian_kelly import BayesianKellyCalibrator
from market_tracker.equal_stake_experiment import (
    VERSION, CAPTURE_CONTRACT_VERSION, build_records, report, seal, verify,
)
from update_market_first_paper import _stores, RAW_PATH
from update_market_performance import _result_index

ROOT = Path(__file__).resolve().parent / "content" / "data" / "market" / "equal_stake_experiment"


def write(path, value):
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return
    atomic_write_text(path, rendered)


def load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def update(*, validate_only=False, root=ROOT):
    now = datetime.now(timezone.utc)
    policy_path = root / "policy.json"
    if not policy_path.exists():
        if validate_only:
            raise ValueError("experiment has not been initialized")
        calibrator = BayesianKellyCalibrator.load()
        with exclusive_store_lock(root / "write.lock"):
            if policy_path.exists():
                raise ValueError("experiment initialized by another writer; rerun")
            write(policy_path, seal({"version": VERSION, "activated_at_utc": now.isoformat(),
                "minimum_expected_return": .05, "risk_units_per_bet": 1,
                "maximum_selections_per_fight_per_strategy": 1,
                "horizon_hours": [20, 28], "maximum_quote_age_seconds": 1800,
                "maximum_capture_latency_seconds": 300,
                "book_access": "hypothetical_all_books_and_each_book_separately",
                "calibration": calibrator.artifact, "paper_only": True,
                "execution_enabled": False, "retuning_allowed": False,
                "review_after_at_least_fights": 200, "review_after_at_least_cards": 20}))
    policy = verify(load(policy_path, None))
    if policy["version"] != VERSION or policy["execution_enabled"] is not False:
        raise ValueError("unsupported experiment policy")
    calibrator = BayesianKellyCalibrator(policy["calibration"])
    if policy["calibration"]["training_last_event_date"] >= policy["activated_at_utc"][:10]:
        raise ValueError("calibration training must precede activation")
    with exclusive_store_lock(root / "write.lock"):
        contract_path = root / "capture_contract.json"
        contract = load(contract_path, None)
        if contract is None and not validate_only:
            contract = seal({"version": CAPTURE_CONTRACT_VERSION,
                "activated_at_utc": now.isoformat(), "policy_sha256": policy["record_sha256"],
                "decision_window_basis": "card_start", "require_before": ["card_start", "provider_bout_start"],
                "historical_decision_backfill": False})
            write(contract_path, contract)
        if contract is not None:
            verify(contract)
            if (contract["version"] != CAPTURE_CONTRACT_VERSION
                    or contract["policy_sha256"] != policy["record_sha256"]):
                raise ValueError("unsupported capture contract")
        records = load(root / "decisions.json", [])
        settlements = load(root / "settlements.json", [])
        for collection in (records, settlements):
            for row in collection:
                verify(row)
            if len({row["matchup_id"] for row in collection}) != len(collection):
                raise ValueError("duplicate experiment matchup")
        decisions = {row["matchup_id"]: row for row in records}
        for row in records:
            if row["policy_sha256"] != policy["record_sha256"]:
                raise ValueError("frozen decision policy mismatch")
            version = row.get("capture_contract_version")
            if version is not None:
                if (contract is None or version != contract["version"]
                        or datetime.fromisoformat(row["recorded_at_utc"].replace("Z", "+00:00"))
                        < datetime.fromisoformat(contract["activated_at_utc"].replace("Z", "+00:00"))):
                    raise ValueError("record predates or disagrees with its capture contract")
        for row in settlements:
            if row["matchup_id"] not in decisions or row["decision_sha256"] != decisions[row["matchup_id"]]["record_sha256"]:
                raise ValueError("settlement references unknown or changed decision")
            if row["target"] not in (0, 1, None):
                raise ValueError("invalid settlement target")
        if not validate_only:
            quotes, forecasts, metadata, _, _ = _stores()
            records += build_records(quotes.read(), forecasts.read(), metadata.read(),
                                     records, policy, calibrator, now, capture_contract=contract)
            # Preserve every existing decision value; additions only. Atomic replacement
            # prevents a crash from leaving a partial JSON ledger.
            write(root / "decisions.json", records)
            settled_ids = {row["matchup_id"] for row in settlements}
            if any(row["matchup_id"] not in settled_ids for row in records):
                raw_bytes = RAW_PATH.read_bytes()
                outcomes, _, ambiguous = _result_index(pd.read_csv(RAW_PATH, low_memory=False))
                for row in records:
                    key = (row["event_id"], row["fighter_id"], row["opponent_id"])
                    if (row["matchup_id"] in settled_ids or key in ambiguous or key not in outcomes
                            or datetime.fromisoformat(row["event_start_utc"].replace("Z", "+00:00")) >= now):
                        continue
                    target, fight_id = outcomes[key]
                    settlements.append(seal({"matchup_id": row["matchup_id"], "target": target,
                        "fight_id": fight_id, "settled_at_utc": now.isoformat(),
                        "decision_sha256": row["record_sha256"],
                        "result_source_sha256": sha256(raw_bytes).hexdigest()}))
            write(root / "settlements.json", settlements)
        result = report(records, settlements, policy)
        result["capture_contract"] = contract
        # Explain the latest observed capture without manufacturing decisions
        # from an old quote. Actual records are created only above, at now.
        quotes, forecasts, metadata, _, _ = _stores()
        quote_rows = [r for r in quotes.read()
                      if datetime.fromisoformat(r.observed_at_utc.replace("Z", "+00:00")) <= now]
        latest = max((r.observed_at_utc for r in quote_rows), default=None)
        diagnostics = {}
        if latest is not None:
            latest_rows = [r for r in quote_rows if r.observed_at_utc == latest]
            build_records(latest_rows, forecasts.read(), metadata.read(), records,
                          policy, calibrator, latest, capture_contract=contract, diagnostics=diagnostics)
        result["latest_capture_diagnostics"] = {"observed_at_utc": latest,
            "interpretation": "Quote eligibility at collection time; actual decisions also require creation within five minutes and after repair activation.",
            **diagnostics}
        if not validate_only:
            write(root / "report.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    result = update(validate_only=parser.parse_args().validate_only)
    print(f"Equal-stake experiment: {result['frozen_fights']} frozen fights, "
          f"{result['settled_fights']} settled. Paper only; execution disabled.")
