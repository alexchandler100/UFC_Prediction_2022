"""Validate generated UFC datasets before the automation commits them.

The checks deliberately focus on contracts whose violation would make a
weekly publication misleading: complete doubled pairs, stable source IDs,
valid numeric domains, matching raw/derived rows, parseable JSON, and fresh
completed/upcoming snapshots.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


RAW_REQUIRED_COLUMNS = {
    "date",
    "fight_url",
    "event_url",
    "result",
    "fighter",
    "opponent",
    "fighter_url",
    "opponent_url",
    "division",
    "method",
    "round",
    "time",
    "total_fight_time",
}

FIGHTER_REQUIRED_COLUMNS = {"name", "height", "reach", "stance", "dob", "url"}

LANDED_ATTEMPTED_PAIRS = (
    ("sig_strikes_landed", "sig_strikes_attempts"),
    ("total_strikes_landed", "total_strikes_attempts"),
    ("takedowns_landed", "takedowns_attempts"),
    ("head_strikes_landed", "head_strikes_attempts"),
    ("body_strikes_landed", "body_strikes_attempts"),
    ("leg_strikes_landed", "leg_strikes_attempts"),
    ("distance_strikes_landed", "distance_strikes_attempts"),
    ("clinch_strikes_landed", "clinch_strikes_attempts"),
    ("ground_strikes_landed", "ground_strikes_attempts"),
)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def merge(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.facts.extend(other.facts)

    def raise_for_errors(self) -> None:
        if self.errors:
            formatted = "\n".join(f"- {message}" for message in self.errors)
            raise ValueError(f"Data validation failed:\n{formatted}")


def _require_columns(
    df: pd.DataFrame, required: set[str], dataset: str, report: ValidationReport
) -> bool:
    missing = sorted(required - set(df.columns))
    report.require(not missing, f"{dataset} is missing required columns: {missing}")
    return not missing


def validate_raw_fights(raw: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if not _require_columns(raw, RAW_REQUIRED_COLUMNS, "raw fights", report):
        return report

    report.require(len(raw) > 0, "raw fights is empty")
    report.require(len(raw) % 2 == 0, "raw fights must contain two rows per fight")
    if len(raw) == 0 or len(raw) % 2:
        return report

    parsed_dates = pd.to_datetime(raw["date"], errors="coerce")
    report.require(parsed_dates.notna().all(), "raw fights contains invalid dates")
    report.require(
        not (parsed_dates.dt.date > date.today()).any(),
        "raw fights contains a completed fight dated in the future",
    )

    left = raw.iloc[0::2].reset_index(drop=True)
    right = raw.iloc[1::2].reset_index(drop=True)
    report.require(
        (left["fight_url"] == right["fight_url"]).all(),
        "adjacent raw rows do not share a fight_url",
    )
    report.require(
        (left["event_url"] == right["event_url"]).all(),
        "adjacent raw rows do not share an event_url",
    )
    report.require(
        (left["date"] == right["date"]).all(),
        "adjacent raw rows do not share a date",
    )
    report.require(
        ((left["fighter_url"] == right["opponent_url"]) &
         (left["opponent_url"] == right["fighter_url"])).all(),
        "fighter IDs are not swapped within one or more doubled fights",
    )
    report.require(
        ((left["fighter"] == right["opponent"]) &
         (left["opponent"] == right["fighter"])).all(),
        "fighter display names are not swapped within one or more doubled fights",
    )
    complementary = (
        ((left["result"] == "W") & (right["result"] == "L"))
        | ((left["result"] == "L") & (right["result"] == "W"))
        | ((left["result"] == right["result"]) & left["result"].isin(["D", "NC"]))
    )
    report.require(complementary.all(), "one or more fight results are not complementary")
    report.require(
        not raw.duplicated(["fight_url", "fighter_url"]).any(),
        "raw fights contains a duplicate fight_url/fighter_url side",
    )
    fight_counts = raw["fight_url"].value_counts()
    report.require(
        (fight_counts == 2).all(), "every fight_url must occur exactly twice"
    )

    numeric_columns = {
        "round",
        "total_fight_time",
        "knockdowns",
        "sub_attempts",
        "reversals",
        "control",
    }
    numeric_columns.update(value for pair in LANDED_ATTEMPTED_PAIRS for value in pair)
    for column in sorted(numeric_columns & set(raw.columns)):
        values = pd.to_numeric(raw[column], errors="coerce")
        invalid = values.isna() & raw[column].notna()
        report.require(not invalid.any(), f"raw {column} contains non-numeric values")
        report.require(not (values.dropna() < 0).any(), f"raw {column} contains negatives")

    rounds = pd.to_numeric(raw["round"], errors="coerce")
    report.require(rounds.dropna().between(1, 5).all(), "round must be between 1 and 5")
    duration = pd.to_numeric(raw["total_fight_time"], errors="coerce")
    report.require(
        duration.dropna().between(1, 1500).all(),
        "total_fight_time must be between one second and 25 minutes",
    )
    if "control" in raw:
        control = pd.to_numeric(raw["control"], errors="coerce")
        report.require(
            not (control > duration).fillna(False).any(),
            "control time exceeds total fight time",
        )

    for landed_column, attempted_column in LANDED_ATTEMPTED_PAIRS:
        if landed_column not in raw or attempted_column not in raw:
            continue
        landed = pd.to_numeric(raw[landed_column], errors="coerce")
        attempted = pd.to_numeric(raw[attempted_column], errors="coerce")
        report.require(
            not (landed > attempted).fillna(False).any(),
            f"{landed_column} exceeds {attempted_column}",
        )

    report.facts.append(
        f"raw fights: {len(raw):,} rows / {raw['fight_url'].nunique():,} fights"
    )
    return report


def _metadata_counter(df: pd.DataFrame) -> Counter:
    columns = ["date", "fighter", "opponent", "result", "method", "division"]
    normalized = df[columns].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    return Counter(map(tuple, normalized.itertuples(index=False, name=None)))


def validate_derived(raw: pd.DataFrame, derived: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    required = {"date", "fighter", "opponent", "result", "method", "division"}
    if not _require_columns(derived, required, "derived fights", report):
        return report
    report.require(len(derived) == len(raw), "raw and derived row counts differ")
    report.require(len(derived) % 2 == 0, "derived fights must contain doubled pairs")
    report.require(
        _metadata_counter(derived) == _metadata_counter(raw),
        "raw and derived fight metadata do not describe the same rows",
    )

    if len(derived) and len(derived) % 2 == 0:
        left = derived.iloc[0::2].reset_index(drop=True)
        right = derived.iloc[1::2].reset_index(drop=True)
        report.require(
            ((left["fighter"] == right["opponent"]) &
             (left["opponent"] == right["fighter"])).all(),
            "derived fight pairs are not adjacent/symmetric",
        )

    numeric = derived.select_dtypes(include=[np.number])
    report.require(
        not np.isinf(numeric.to_numpy(dtype=float, copy=False)).any(),
        "derived features contain positive or negative infinity",
    )
    raw_max = pd.to_datetime(raw["date"], errors="coerce").max()
    derived_max = pd.to_datetime(derived["date"], errors="coerce").max()
    report.require(raw_max == derived_max, "raw and derived maximum fight dates differ")
    report.facts.append(f"derived fights: {derived.shape[0]:,} rows / {derived.shape[1]:,} columns")
    return report


def validate_fighters(fighters: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if not _require_columns(fighters, FIGHTER_REQUIRED_COLUMNS, "fighter stats", report):
        return report
    report.require(fighters["url"].notna().all(), "fighter stats contains a null URL")
    report.require(fighters["name"].notna().all(), "fighter stats contains a null name")
    report.require(not fighters["url"].duplicated().any(), "fighter URLs must be unique")
    duplicate_names = sorted(fighters.loc[fighters["name"].duplicated(False), "name"].unique())
    if duplicate_names:
        report.warnings.append(
            "Display names are not unique; ID-based joins are required: " + ", ".join(duplicate_names)
        )
    report.facts.append(f"fighter stats: {len(fighters):,} source IDs")
    return report


def _load_json(path: Path, report: ValidationReport):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        report.errors.append(f"{path.name} is not valid readable JSON: {error}")
        return None


def validate_publication(
    data_root: Path, raw: pd.DataFrame, *, allow_stale: bool = False
) -> ValidationReport:
    report = ValidationReport()
    external = data_root / "external"
    objects = {
        path.stem: _load_json(path, report) for path in sorted(external.glob("*.json"))
    }
    required_json = {
        "card_info",
        "fighter_stats",
        "prediction_history",
        "ufc_fight_data_for_website",
        "vegas_odds",
    }
    report.require(
        required_json.issubset(objects),
        f"missing publication JSON files: {sorted(required_json - set(objects))}",
    )
    if report.errors:
        return report

    website_data = objects["ufc_fight_data_for_website"]
    report.require(isinstance(website_data, dict), "website fight data must be a JSON object")
    if isinstance(website_data, dict):
        report.require(
            len(website_data) == len(raw),
            "website fight-data JSON row count differs from raw fights",
        )

    card_info = objects["card_info"]
    report.require(isinstance(card_info, dict), "card_info must be a JSON object")
    card_date = pd.NaT
    if isinstance(card_info, dict):
        report.require(bool(card_info.get("title")), "card_info title is blank")
        card_date = pd.to_datetime(card_info.get("date"), errors="coerce")
        report.require(pd.notna(card_date), "card_info date is invalid")

    vegas_object = objects["vegas_odds"]
    try:
        vegas = pd.DataFrame(vegas_object)
    except (TypeError, ValueError) as error:
        report.errors.append(f"vegas_odds cannot be loaded as a table: {error}")
        vegas = pd.DataFrame()
    if not vegas.empty:
        required = {"fighter name", "opponent name", "date"}
        if _require_columns(vegas, required, "vegas odds", report):
            vegas_dates = pd.to_datetime(vegas["date"], errors="coerce")
            report.require(vegas_dates.notna().all(), "vegas odds contains invalid dates")
            report.require(vegas["fighter name"].astype(bool).all(), "vegas odds has blank fighters")
            report.require(vegas["opponent name"].astype(bool).all(), "vegas odds has blank opponents")
            if pd.notna(card_date):
                report.require(
                    (vegas_dates.dt.normalize() == card_date.normalize()).all(),
                    "vegas odds dates do not match card_info",
                )

    completed_max = pd.to_datetime(raw["date"], errors="coerce").max()
    completed_age = (pd.Timestamp.today().normalize() - completed_max.normalize()).days
    report.facts.append(f"latest completed fight: {completed_max.date()} ({completed_age} days old)")
    if not allow_stale:
        report.require(completed_age <= 45, "completed fight data is more than 45 days stale")
        if pd.notna(card_date):
            report.require(
                card_date.normalize() >= pd.Timestamp.today().normalize(),
                "published upcoming card is already in the past",
            )
            report.require(
                card_date.normalize() <= pd.Timestamp.today().normalize() + pd.Timedelta(days=120),
                "published upcoming card is implausibly far in the future",
            )
    return report


def validate_repository(repo_root: Path, *, allow_stale: bool = False) -> ValidationReport:
    data_root = repo_root / "src" / "content" / "data"
    processed = data_root / "processed"
    raw = pd.read_csv(processed / "ufc_fights_reported_doubled.csv", low_memory=False)
    derived = pd.read_csv(
        processed / "ufc_fights_reported_derived_doubled.csv", low_memory=False
    )
    fighters = pd.read_csv(processed / "fighter_stats.csv", low_memory=False)

    report = ValidationReport()
    report.merge(validate_raw_fights(raw))
    report.merge(validate_derived(raw, derived))
    report.merge(validate_fighters(fighters))
    report.merge(validate_publication(data_root, raw, allow_stale=allow_stale))

    raw_dates = pd.to_datetime(raw["date"], errors="coerce")
    derived_dates = pd.to_datetime(derived["date"], errors="coerce")
    ordering_problems = []
    if not raw_dates.is_monotonic_decreasing:
        ordering_problems.append("raw fights are not sorted newest-to-oldest")
    if not derived_dates.is_monotonic_increasing:
        ordering_problems.append("derived fights are not sorted oldest-to-newest")
    if allow_stale:
        report.warnings.extend(ordering_problems)
    else:
        report.errors.extend(ordering_problems)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of src)",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="run structural checks without rejecting old completed/upcoming snapshots",
    )
    args = parser.parse_args()

    report = validate_repository(args.repo_root.resolve(), allow_stale=args.allow_stale)
    for fact in report.facts:
        print(f"FACT: {fact}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if report.errors:
        print(f"Validation failed with {len(report.errors)} error(s)")
        return 1
    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
