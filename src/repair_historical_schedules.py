"""Recover explicit schedules from saved per-round data; dry-run unless --apply.

The repair matches stable fight/event/fighter IDs, verifies mirrored source
rows, and fills only blank raw aggregate time_format cells. It never infers a
schedule from the result or trains a model. Optional --apply-pit copies only
the repaired schedule label into existing point-in-time rows, preserving all
numeric features and other labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tempfile

from fight_semantics import (
    declared_round_lengths_seconds,
    scheduled_rounds_from_time_format,
    schedule_from_row,
    stable_ufcstats_id,
)


DATA = Path(__file__).resolve().parent / "content/data/processed"


def _read(payload: bytes) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    return list(reader.fieldnames or []), list(reader)


def plan_schedule_repair(raw_bytes: bytes, round_bytes: bytes) -> tuple[bytes, dict]:
    """Return repaired CSV bytes and evidence; reject conflicts before writing."""
    columns, raw = _read(raw_bytes)
    _, source = _read(round_bytes)
    if "time_format" not in columns:
        raise ValueError("Raw aggregate data must already have a time_format column")
    groups: dict[str, list[dict]] = defaultdict(list)
    targets: dict[str, list[dict]] = defaultdict(list)
    for row in source:
        fight_id = row["fight_id"].strip()
        if not fight_id or stable_ufcstats_id(row["fight_url"]) != fight_id:
            raise ValueError("Round data fight ID disagrees with its source URL")
        groups[fight_id].append(row)
    for row in raw:
        targets[stable_ufcstats_id(row["fight_url"])].append(row)
    changes = []
    unmatched = []
    for fight_id, rows in sorted(groups.items()):
        formats = {row["time_format"].strip() for row in rows}
        events = {row["event_id"].strip() for row in rows}
        if len(formats) != 1 or "" in formats or len(events) != 1 or "" in events:
            raise ValueError(f"Conflicting or missing source schedule/event for {fight_id}")
        time_format = next(iter(formats))
        scheduled = scheduled_rounds_from_time_format(time_format)
        if scheduled is None or not declared_round_lengths_seconds(time_format):
            raise ValueError(f"Unparseable explicit source schedule for {fight_id}")
        pairs: dict[str, list[dict]] = defaultdict(list)
        fighters = set()
        for row in rows:
            if int(row["scheduled_rounds"]) != scheduled:
                raise ValueError(f"Declared schedule count disagrees for {fight_id}")
            if stable_ufcstats_id(row["event_url"]) not in events:
                raise ValueError(f"Source event URL disagrees for {fight_id}")
            if any(stable_ufcstats_id(row[f"{side}_url"]) != row[f"{side}_id"]
                   for side in ("fighter", "opponent")):
                raise ValueError(f"Source fighter IDs disagree for {fight_id}")
            fighters.add(row["fighter_id"])
            pairs[row["round"]].append(row)
        if len(fighters) != 2:
            raise ValueError(f"Source does not identify two fighters for {fight_id}")
        for pair in pairs.values():
            if (len(pair) != 2 or pair[0]["fighter_id"] == pair[1]["fighter_id"]
                    or pair[0]["fighter_id"] != pair[1]["opponent_id"]
                    or pair[1]["fighter_id"] != pair[0]["opponent_id"]):
                raise ValueError(f"Source round perspectives are not mirrored for {fight_id}")
        aggregate = targets.get(fight_id)
        if aggregate is None:
            unmatched.append(fight_id)
            continue
        if (len(aggregate) != 2
                or {stable_ufcstats_id(row["event_url"]) for row in aggregate} != events
                or {stable_ufcstats_id(row["fighter_url"]) for row in aggregate} != fighters
                or any(stable_ufcstats_id(row["opponent_url"]) not in fighters
                       or row["fighter_url"] == row["opponent_url"] for row in aggregate)):
            raise ValueError(f"Aggregate perspectives disagree with source identity for {fight_id}")
        for row in aggregate:
            existing = row["time_format"].strip()
            if existing and existing != time_format:
                raise ValueError(f"Existing explicit schedule conflicts for {fight_id}")
        blanks = [row for row in aggregate if not row["time_format"].strip()]
        for row in blanks:
            row["time_format"] = time_format
        if blanks:
            changes.append({"fight_id": fight_id, "date": aggregate[0]["date"],
                            "time_format": time_format, "changed_side_cells": len(blanks)})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns,
                            lineterminator="\r\n" if b"\r\n" in raw_bytes else "\n")
    writer.writeheader()
    writer.writerows(raw)
    repaired = output.getvalue().encode("utf-8") if changes else raw_bytes
    # Assert every other parsed cell and row position survives the repair.
    _, original = _read(raw_bytes)
    _, verified = _read(repaired)
    if len(original) != len(verified) or any(
        before[column] != after[column]
        for before, after in zip(original, verified)
        for column in columns if column != "time_format"
    ):
        raise ValueError("Schedule repair unexpectedly changed unrelated cells")
    return repaired, {
        "schema_version": 1, "source": "saved_explicit_ufcstats_round_metadata",
        "raw_input_sha256": sha256(raw_bytes).hexdigest(),
        "round_input_sha256": sha256(round_bytes).hexdigest(),
        "proposed_raw_sha256": sha256(repaired).hexdigest(),
        "source_fights": len(groups), "source_round_pairs": sum(len({r['round'] for r in rows}) for rows in groups.values()),
        "repaired_fights": len(changes),
        "changed_side_cells": sum(item["changed_side_cells"] for item in changes),
        "unmatched_source_fight_ids": unmatched,
        "unrelated_cells_unchanged": True, "changes": changes,
    }


def plan_pit_schedule_repair(pit_bytes: bytes, raw_bytes: bytes) -> tuple[bytes, int]:
    """Fill only missing point-in-time schedule labels from repaired raw IDs."""
    columns, rows = _read(pit_bytes)
    _, raw = _read(raw_bytes)
    if "label_time_format" not in columns or "fight_id" not in columns:
        raise ValueError("Point-in-time data needs fight_id and label_time_format")
    formats: dict[str, set[str]] = defaultdict(set)
    for row in raw:
        if row["time_format"].strip():
            formats[stable_ufcstats_id(row["fight_url"])].add(row["time_format"].strip())
    if any(len(values) != 1 for values in formats.values()):
        raise ValueError("Repaired aggregate has conflicting schedules between perspectives")
    lookup = {fight_id: next(iter(values)) for fight_id, values in formats.items()}
    changed = 0
    for row in rows:
        proposed = lookup.get(row["fight_id"])
        existing = row["label_time_format"].strip()
        if proposed and existing and proposed != existing:
            raise ValueError(f"Existing PIT schedule conflicts for {row['fight_id']}")
        if proposed and not existing:
            row["label_time_format"] = proposed
            changed += 1
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns,
                            lineterminator="\r\n" if b"\r\n" in pit_bytes else "\n")
    writer.writeheader()
    writer.writerows(rows)
    repaired = output.getvalue().encode("utf-8") if changed else pit_bytes
    _, original = _read(pit_bytes)
    _, verified = _read(repaired)
    if len(original) != len(verified) or any(
        before[column] != after[column]
        for before, after in zip(original, verified)
        for column in columns if column != "label_time_format"
    ):
        raise ValueError("PIT repair unexpectedly changed numeric features or unrelated labels")
    return repaired, changed


def _atomic_write(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DATA / "ufc_fights_reported_doubled.csv")
    parser.add_argument("--rounds", type=Path, default=DATA / "ufc_fight_round_stats_doubled.csv")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--pit", type=Path, default=DATA / "ufc_fights_point_in_time.csv")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--apply-pit", action="store_true",
                        help="Also fill PIT schedule labels; requires --apply")
    args = parser.parse_args()
    if args.apply_pit and not args.apply:
        parser.error("--apply-pit requires --apply")
    if args.report and args.report.resolve() in {args.raw.resolve(), args.rounds.resolve(), args.pit.resolve()}:
        raise ValueError("Report path must not replace an input data file")
    raw_bytes = args.raw.read_bytes()
    repaired, report = plan_schedule_repair(raw_bytes, args.rounds.read_bytes())
    pit_bytes = proposed_pit = None
    if args.pit.exists():
        pit_bytes = args.pit.read_bytes()
        proposed_pit, pit_changes = plan_pit_schedule_repair(pit_bytes, repaired)
        _, pit = _read(pit_bytes)
        _, proposed_rows = _read(repaired)
        schedule_lookup = {stable_ufcstats_id(row["fight_url"]): row["time_format"]
                           for row in proposed_rows}
        report["pit_input_sha256"] = sha256(pit_bytes).hexdigest()
        report["proposed_pit_sha256"] = sha256(proposed_pit).hexdigest()
        report["proposed_changed_pit_cells"] = pit_changes
        report["pit_unrelated_cells_unchanged"] = True
        report["pit_existing_verified_schedule_fights"] = sum(
            schedule_from_row(row)[0] is not None for row in pit)
        report["pit_expected_verified_schedule_fights"] = sum(
            schedule_from_row({**row, "label_time_format": schedule_lookup.get(
                row["fight_id"], row.get("label_time_format", ""))})[0] is not None
            for row in pit)
    report["applied"] = False
    report["pit_applied"] = False
    report["changed_pit_cells"] = 0
    if args.apply_pit and pit_bytes is None:
        raise ValueError("--apply-pit needs an existing PIT file")
    if args.apply:
        if args.raw.read_bytes() != raw_bytes:
            raise RuntimeError("Raw data changed during planning; rerun the repair")
        if args.apply_pit and args.pit.read_bytes() != pit_bytes:
            raise RuntimeError("PIT data changed during planning; rerun the repair")
        raw_changed = bool(report["changed_side_cells"])
        try:
            if raw_changed:
                _atomic_write(args.raw, repaired)
            if args.apply_pit and report["proposed_changed_pit_cells"]:
                _atomic_write(args.pit, proposed_pit)
                report["pit_applied"] = True
                report["changed_pit_cells"] = report["proposed_changed_pit_cells"]
        except OSError:
            if raw_changed:
                _atomic_write(args.raw, raw_bytes)
            raise
        report["applied"] = raw_changed
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "changes"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
