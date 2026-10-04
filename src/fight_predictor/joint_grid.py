"""Preserve raw joint forecasts without pretending they are settled-bet scenarios."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd

from .outcome_model import TERMINAL_OUTCOMES


GRID_VERSION = "raw-winner-method-time-grid-v1"
LIMITATIONS = ["interval_times_not_exact_finish_times", "decision_timing_not_validated",
              "draw_and_no_contest_probabilities_missing", "book_settlement_rules_not_mapped"]


def build_joint_grid(prediction):
    if not getattr(prediction, "interval_outcomes", ()):
        return None
    rows = [{"outcome": outcome, "start_seconds": start, "end_seconds": end, "probability": mass}
            for outcome, start, end, mass in prediction.interval_outcomes]
    return {"version": GRID_VERSION, "settlement_ready": False, "execution_enabled": False,
            "limitations": LIMITATIONS.copy(), "scenarios": rows,
            "decision_mass_before_scheduled_end": sum(row["probability"] for row in rows
                if row["outcome"].endswith("_decision") and row["end_seconds"] < prediction.scheduled_rounds * 300)}


def validate_joint_grid(grid, terminal, totals, rounds):
    if not isinstance(grid, dict) or set(terminal) != set(TERMINAL_OUTCOMES):
        raise ValueError("raw joint grid requires all winner/method outcomes")
    if (grid.get("version") != GRID_VERSION or grid.get("settlement_ready") is not False
            or grid.get("execution_enabled") is not False or grid.get("limitations") != LIMITATIONS):
        raise ValueError("raw joint grid must retain its research-only contract")
    rows = grid.get("scenarios")
    if not isinstance(rows, list) or not rows or len(rows) > 1000:
        raise ValueError("raw joint grid scenarios are missing or excessive")
    marginal = dict.fromkeys(terminal, 0.0)
    early_decisions = 0.0
    seen = set()
    horizon = rounds * 300
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("raw joint grid scenario must be an object")
        outcome, start, end = row.get("outcome"), row.get("start_seconds"), row.get("end_seconds")
        probability = row.get("probability")
        if (outcome not in marginal or type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= horizon
                or (start == end and (end != horizon or not outcome.endswith("_decision")))
                or type(probability) not in (int, float) or not math.isfinite(probability) or not 0 < probability <= 1):
            raise ValueError("invalid raw joint grid scenario")
        key = (outcome, start, end)
        if key in seen:
            raise ValueError("duplicate raw joint grid scenario")
        seen.add(key)
        marginal[outcome] += probability
        if outcome.endswith("_decision") and end < horizon:
            early_decisions += probability
    if any(abs(marginal[key] - float(terminal[key])) > 1e-9 for key in marginal):
        raise ValueError("joint grid disagrees with terminal probabilities")
    for line, expected in totals.items():
        threshold = float(line) * 300
        if any(row["start_seconds"] < threshold < row["end_seconds"] for row in rows):
            raise ValueError("total threshold falls inside an unresolved time interval")
        actual = sum(row["probability"] for row in rows if row["end_seconds"] > threshold)
        if abs(actual - float(expected)) > 1e-9:
            raise ValueError("joint grid disagrees with duration probabilities")
    if abs(early_decisions - float(grid.get("decision_mass_before_scheduled_end", -1))) > 1e-9:
        raise ValueError("joint grid decision timing diagnostic disagrees")


def archive_joint_publication(directory, publication, observed_at=None):
    """Archive new grids before the UTC event date; never recreate old forecasts.

    Without a verified start timestamp, same-day publications are deliberately
    ineligible. Repeated publication hashes do not create additional evidence.
    """
    from .outcome_publication import validate_outcome_forecast_publication
    validate_outcome_forecast_publication(publication)
    grids = [item for item in publication["matchups"] if item.get("joint_outcome_grid")]
    if not grids:
        return False
    observed = pd.Timestamp(observed_at or datetime.now(timezone.utc))
    if observed.tzinfo is None:
        raise ValueError("joint archive needs a timezone-aware collection time")
    observed = observed.tz_convert("UTC")
    issued = pd.Timestamp(publication["forecast_issued_at_utc"])
    event_day = pd.Timestamp(publication["event_date"]).tz_localize("UTC")
    if issued.tzinfo is None or issued > observed:
        raise ValueError("joint archive cannot record a future or untimed forecast")
    if observed >= event_day:
        return False
    for item in grids:
        validate_joint_grid(item["joint_outcome_grid"], item["terminal_probabilities"],
                            item["total_round_over_probabilities"], item["scheduled_rounds"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{publication['publication_sha256']}.json"
    envelope = {"archive_version": GRID_VERSION, "archived_at_utc": observed.isoformat(),
                "paper_only": True, "execution_enabled": False, "publication": publication}
    encoded = json.dumps(envelope, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".joint-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Publish only a complete file, without overwriting an existing hash.
            os.link(temporary, destination)
        except FileExistsError:
            saved = json.loads(destination.read_text(encoding="utf-8"))
            if saved.get("publication") != publication:
                raise ValueError("joint archive publication identity collision")
            validate_joint_archive(directory)
            return False
    finally:
        os.unlink(temporary)
    return True


def validate_joint_archive(directory):
    from .outcome_publication import validate_outcome_forecast_publication
    if Path(directory).exists() and not Path(directory).is_dir():
        raise ValueError("joint archive path must be a directory")
    files = list(Path(directory).glob("*.json"))
    for path in files:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(saved, dict) or saved.get("archive_version") != GRID_VERSION or saved.get("paper_only") is not True
                or saved.get("execution_enabled") is not False):
            raise ValueError("invalid joint archive contract")
        publication = validate_outcome_forecast_publication(saved.get("publication"))
        if path.stem != publication["publication_sha256"]:
            raise ValueError("joint archive filename disagrees with publication identity")
        if not any(item.get("joint_outcome_grid") for item in publication["matchups"]):
            raise ValueError("joint archive lacks a joint grid")
        observed = pd.Timestamp(saved.get("archived_at_utc"))
        issued = pd.Timestamp(publication["forecast_issued_at_utc"])
        event = pd.Timestamp(publication["event_date"]).tz_localize("UTC")
        if pd.isna(observed) or observed.tzinfo is None or not issued <= observed < event:
            raise ValueError("joint forecast archive was not recorded before the event")
    return len(files)
