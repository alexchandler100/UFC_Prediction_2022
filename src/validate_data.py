"""Validate generated UFC datasets before the automation commits them.

The checks deliberately focus on contracts whose violation would make a
weekly publication misleading: complete doubled pairs, stable source IDs,
valid numeric domains, a causal point-in-time model contract, parseable JSON,
and fresh completed/upcoming snapshots. The obsolete derived notebook table is
intentionally outside the weekly publication gate.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re

import numpy as np
import pandas as pd

from build_fighter_explorer import SHARD_KEYS as FIGHTER_EXPLORER_SHARD_KEYS
from build_fighter_explorer import SHARD_SIZE_LIMIT as FIGHTER_SHARD_SIZE_LIMIT
from build_fighter_explorer import SIZE_LIMIT as FIGHTER_EXPLORER_SIZE_LIMIT
from build_fighter_explorer import validate_fighter_explorer

from fight_predictor.point_in_time import (
    MODEL_VERSION,
    REGULARIZATION_C_GRID,
    PointInTimeDatasetBuilder,
    training_fingerprint,
)
from external_mma import ExternalMmaStore, load_approved_auxiliary
from external_mma.schema import ExternalDataError
from market_tracker import (
    ForecastCaptureStore,
    MarketDataError,
    PaperDecisionStore,
    PaperSettlementStore,
    QuoteSnapshotStore,
    QuoteSourceMetadataStore,
    StoreIntegrityError,
    TIMING_POLICY_VERSION,
    consensus_as_of,
    summarize_paper_settlements,
    validate_current_opportunities,
)
from market_tracker._common import BETTING_STATUS, canonical_hash
from market_tracker.prospective import (
    DECISION_TARGET_LEAD_SECONDS,
    DECISION_WINDOW_SECONDS,
    MAX_SOURCE_QUOTE_AGE_SECONDS,
    MIN_CONSENSUS_BOOKS,
)


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

POINT_IN_TIME_REQUIRED_COLUMNS = {
    "schema_version",
    "date",
    "event_id",
    "fight_id",
    "fight_url",
    "event_url",
    "fighter_id",
    "opponent_id",
    "fighter_url",
    "opponent_url",
    "target",
    "bout_order",
    "label_method",
    "label_finish_round",
    "label_total_fight_seconds",
    "label_time_format",
}

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
    if {"source_card_index", "bout_order"} <= set(raw.columns):
        source_index = pd.to_numeric(raw["source_card_index"], errors="coerce")
        bout_order = pd.to_numeric(raw["bout_order"], errors="coerce")
        report.require(
            source_index.notna().all() and bout_order.notna().all(),
            "raw card-order metadata must be numeric",
        )
        report.require(
            (source_index.dropna() >= 0).all()
            and (bout_order.dropna() >= 0).all()
            and np.equal(source_index.dropna() % 1, 0).all()
            and np.equal(bout_order.dropna() % 1, 0).all(),
            "raw card-order metadata must contain nonnegative integers",
        )
        report.require(
            (source_index.iloc[0::2].reset_index(drop=True)
             == source_index.iloc[1::2].reset_index(drop=True)).all()
            and (bout_order.iloc[0::2].reset_index(drop=True)
                 == bout_order.iloc[1::2].reset_index(drop=True)).all(),
            "mirrored raw fight sides disagree on card order",
        )
        fight_order = raw.drop_duplicates("fight_url")
        for event_url, event in fight_order.groupby("event_url", sort=False):
            expected = set(range(len(event)))
            report.require(
                set(pd.to_numeric(event["source_card_index"], errors="coerce")) == expected,
                f"source_card_index is not contiguous for event {event_url}",
            )
            report.require(
                set(pd.to_numeric(event["bout_order"], errors="coerce")) == expected,
                f"bout_order is not contiguous for event {event_url}",
            )
            event_source = pd.to_numeric(
                event["source_card_index"], errors="coerce"
            )
            event_bout = pd.to_numeric(event["bout_order"], errors="coerce")
            report.require(
                (event_source + event_bout == len(event) - 1).all(),
                f"source_card_index and bout_order are not inverse for event {event_url}",
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
        report.require(
            np.isfinite(values.dropna().to_numpy(dtype=float)).all(),
            f"raw {column} contains non-finite values",
        )
        report.require(not (values.dropna() < 0).any(), f"raw {column} contains negatives")

    rounds = pd.to_numeric(raw["round"], errors="coerce")
    no_contest_rows = (
        raw["result"].astype(str).str.upper().eq("NC")
        | raw["method"].fillna("").astype(str).str.casefold().str.contains(
            "cnc|overturned", regex=True
        )
    )
    report.require(
        rounds[~no_contest_rows].notna().all()
        and rounds.dropna().between(1, 5).all()
        and np.equal(rounds.dropna() % 1, 0).all(),
        "round must be an integer between 1 and 5 except unknown no-contests",
    )
    duration = pd.to_numeric(raw["total_fight_time"], errors="coerce")
    report.require(
        duration.dropna().between(1, 7200).all(),
        "known total_fight_time must be between one second and two hours",
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
    raw_max = pd.to_datetime(
        raw.loc[raw["result"].isin(["W", "L"]), "date"], errors="coerce"
    ).max()
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
    report.require(
        fighters["name"].astype(str).str.strip().ne("").all(),
        "fighter display names must not be blank",
    )

    def placeholder(value: object) -> bool:
        return pd.isna(value) or str(value).strip().casefold() in {
            "", "--", "n/a", "nan", "none"
        }

    for index, value in fighters["height"].items():
        if placeholder(value):
            continue
        match = re.fullmatch(r"\s*(\d+)\s*'\s*(\d+)\s*\"?\s*", str(value))
        valid = bool(match)
        if match:
            inches = int(match.group(1)) * 12 + int(match.group(2))
            valid = 48 <= inches <= 96 and 0 <= int(match.group(2)) < 12
        report.require(valid, f"fighter height is implausible at row {index}: {value!r}")
    for index, value in fighters["reach"].items():
        if placeholder(value):
            continue
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        valid = bool(match) and 48 <= float(match.group()) <= 100
        report.require(valid, f"fighter reach is implausible at row {index}: {value!r}")
    known_dob = ~fighters["dob"].map(placeholder)
    parsed_dob = pd.to_datetime(fighters.loc[known_dob, "dob"], errors="coerce")
    report.require(parsed_dob.notna().all(), "fighter stats contains an invalid known DOB")
    if not parsed_dob.empty:
        report.require(
            (
                (parsed_dob >= pd.Timestamp("1930-01-01"))
                & (parsed_dob <= pd.Timestamp.today().normalize() - pd.DateOffset(years=18))
            ).all(),
            "fighter stats contains an implausible known DOB",
        )
    duplicate_names = sorted(fighters.loc[fighters["name"].duplicated(False), "name"].unique())
    if duplicate_names:
        report.warnings.append(
            "Display names are not unique; ID-based joins are required: " + ", ".join(duplicate_names)
        )
    report.facts.append(f"fighter stats: {len(fighters):,} source IDs")
    return report


def validate_point_in_time(raw: pd.DataFrame, point_in_time: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if not _require_columns(
        point_in_time,
        POINT_IN_TIME_REQUIRED_COLUMNS,
        "point-in-time fights",
        report,
    ):
        return report
    report.require(not point_in_time.empty, "point-in-time fights is empty")
    report.require(
        point_in_time["fight_id"].is_unique,
        "point-in-time fights must contain one row per fight_id",
    )
    report.require(
        point_in_time["schema_version"].eq(1).all(),
        "point-in-time fights has an unsupported schema version",
    )
    report.require(
        point_in_time["target"].isin([0, 1]).all(),
        "point-in-time target must contain only binary W/L labels",
    )
    report.require(
        (point_in_time["fighter_id"].astype(str) < point_in_time["opponent_id"].astype(str)).all(),
        "point-in-time matchup orientation is not canonical by fighter ID",
    )
    dates = pd.to_datetime(point_in_time["date"], errors="coerce")
    report.require(dates.notna().all(), "point-in-time fights contains invalid dates")
    report.require(dates.is_monotonic_increasing, "point-in-time fights is not chronological")
    expected_order = point_in_time.assign(_parsed_date=dates).sort_values(
        ["_parsed_date", "event_id", "bout_order", "fight_id"], kind="stable"
    )["fight_id"].astype(str).tolist()
    report.require(
        expected_order == point_in_time["fight_id"].astype(str).tolist(),
        "point-in-time fights are not in causal date/event/bout order",
    )
    feature_columns = [column for column in point_in_time if column.endswith("_diff")]
    report.require(bool(feature_columns), "point-in-time fights has no explicit diff features")
    numeric = point_in_time[feature_columns].apply(pd.to_numeric, errors="coerce")
    report.require(
        numeric.notna().all().all(),
        "point-in-time feature matrix contains null or non-numeric values",
    )
    report.require(
        np.isfinite(numeric.to_numpy(dtype=float)).all(),
        "point-in-time feature matrix contains infinity",
    )
    lineage_columns = {
        "date", "fight_url", "event_url", "fighter_url", "opponent_url",
        "result", "method", "round", "total_fight_time", "bout_order",
    }
    missing_raw = sorted(lineage_columns - set(raw.columns))
    if missing_raw:
        report.errors.append(
            f"raw fights cannot validate point-in-time lineage; missing: {missing_raw}"
        )
        return report
    raw = raw.copy()
    raw_bout_order = pd.to_numeric(raw["bout_order"], errors="coerce")
    if (
        raw_bout_order.isna().any()
        or not np.equal(raw_bout_order % 1, 0).all()
    ):
        report.errors.append(
            "raw bout_order must be integral before point-in-time lineage can be validated"
        )
        return report
    raw["bout_order"] = raw_bout_order.astype(int)

    def identity_token(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value).strip().rstrip("/").rsplit("/", 1)[-1]

    expected_rows: list[dict[str, object]] = []
    invalid_groups = []
    for fight_url, group in raw.groupby("fight_url", sort=False, dropna=False):
        if len(group) != 2:
            invalid_groups.append(str(fight_url))
            continue
        working = group.copy()
        working["_fighter_id"] = working["fighter_url"].map(identity_token)
        working["_opponent_id"] = working["opponent_url"].map(identity_token)
        results = working["result"].astype(str).str.upper()
        if sorted(results.tolist()) != ["L", "W"]:
            continue
        canonical = working.sort_values("_fighter_id", kind="stable").iloc[0]
        expected_rows.append(
            {
                "fight_id": identity_token(fight_url),
                "fight_url": canonical["fight_url"],
                "event_id": identity_token(canonical["event_url"]),
                "event_url": canonical["event_url"],
                "date": pd.to_datetime(canonical["date"], errors="coerce").normalize(),
                "fighter_id": canonical["_fighter_id"],
                "fighter_url": canonical["fighter_url"],
                "opponent_id": canonical["_opponent_id"],
                "opponent_url": canonical["opponent_url"],
                "target": 1 if str(canonical["result"]).upper() == "W" else 0,
                "bout_order": int(canonical["bout_order"]),
                "label_method": canonical["method"],
                "label_finish_round": canonical.get("round", np.nan),
                "label_total_fight_seconds": canonical.get(
                    "total_fight_time", np.nan
                ),
                "label_time_format": canonical.get("time_format", ""),
            }
        )
    report.require(
        not invalid_groups,
        "raw fights contain invalid groups, so point-in-time completeness cannot be proven",
    )
    if not expected_rows:
        report.errors.append("raw fights contain no terminal W/L rows for point-in-time validation")
        return report
    expected = pd.DataFrame(expected_rows).set_index("fight_id", drop=False)
    actual = point_in_time.copy()
    actual["date"] = dates
    actual["fight_id"] = actual["fight_id"].astype(str)
    actual = actual.set_index("fight_id", drop=False)
    expected_ids = set(expected.index.astype(str))
    actual_ids = set(actual.index.astype(str))
    report.require(
        actual_ids == expected_ids,
        "point-in-time fight IDs are not the exact set of terminal raw W/L fights",
    )
    if point_in_time["fight_id"].is_unique:
        for fight_id in sorted(actual_ids & expected_ids):
            expected_row = expected.loc[fight_id]
            actual_row = actual.loc[fight_id]
            for column in (
                "fight_url", "event_id", "event_url", "date", "fighter_id",
                "fighter_url", "opponent_id", "opponent_url", "target", "bout_order",
                "label_method", "label_finish_round",
                "label_total_fight_seconds", "label_time_format",
            ):
                both_missing = pd.isna(actual_row[column]) and pd.isna(
                    expected_row[column]
                )
                if not both_missing and actual_row[column] != expected_row[column]:
                    report.errors.append(
                        f"point-in-time {column} disagrees with raw lineage for fight {fight_id}"
                    )
    point_url_tokens = point_in_time["fighter_url"].map(identity_token)
    opponent_url_tokens = point_in_time["opponent_url"].map(identity_token)
    report.require(
        point_url_tokens.eq(point_in_time["fighter_id"].astype(str)).all()
        and opponent_url_tokens.eq(point_in_time["opponent_id"].astype(str)).all(),
        "point-in-time URL tokens do not match fighter IDs",
    )
    terminal_raw = raw[raw["result"].astype(str).str.upper().isin(["W", "L"])]
    raw_max = pd.to_datetime(terminal_raw["date"], errors="coerce").max()
    point_max = dates.max()
    report.require(
        raw_max == point_max,
        "latest terminal raw W/L and point-in-time label dates differ",
    )
    report.facts.append(
        f"point-in-time fights: {len(point_in_time):,} rows / {len(feature_columns):,} features"
    )
    return report


def validate_model_artifact(
    artifact: object,
    raw: pd.DataFrame,
    fighters: pd.DataFrame,
    point_in_time: pd.DataFrame | None,
    auxiliary_fights: pd.DataFrame | None = None,
) -> ValidationReport:
    report = ValidationReport()
    if not isinstance(artifact, dict):
        report.errors.append("winner_model.json must contain a JSON object")
        return report
    required = {
        "schema_version", "model_version", "model_id", "data_through",
        "source_data_through", "training_labels_through", "training_fights",
        "training_fingerprint_sha256", "state_fingerprint_sha256",
        "feature_columns", "scaler_scale", "coefficients", "intercept",
        "calibration_slope", "selected_c", "regularization_c_grid",
        "temporal_evaluation",
    }
    missing = sorted(required - set(artifact))
    report.require(not missing, f"winner model artifact is missing fields: {missing}")
    if missing:
        return report
    report.require(artifact["schema_version"] == 1, "winner model schema version must be 1")
    report.require(
        artifact["model_version"] == MODEL_VERSION,
        "winner model version is not supported by this code",
    )
    try:
        artifact_c_grid = tuple(
            float(value) for value in artifact["regularization_c_grid"]
        )
    except (TypeError, ValueError):
        artifact_c_grid = ()
        report.errors.append("winner model regularization grid is not numeric")
    report.require(
        artifact_c_grid == REGULARIZATION_C_GRID,
        "winner model regularization grid is not supported by this code",
    )
    features = artifact["feature_columns"]
    scales = artifact["scaler_scale"]
    coefficients = artifact["coefficients"]
    report.require(isinstance(features, list) and bool(features), "winner model features must be a list")
    report.require(isinstance(scales, list), "winner model scales must be a list")
    report.require(isinstance(coefficients, list), "winner model coefficients must be a list")
    if not isinstance(scales, list):
        scales = []
    if not isinstance(coefficients, list):
        coefficients = []
    if isinstance(features, list):
        report.require(len(features) == len(set(features)), "winner model features are duplicated")
        report.require(all(str(value).endswith("_diff") for value in features), "winner model has a non-difference feature")
        report.require(
            len(scales) == len(features) and len(coefficients) == len(features),
            "winner model vector lengths do not match its features",
        )
        if point_in_time is not None:
            report.require(
                features == [column for column in point_in_time if column.endswith("_diff")],
                "winner model feature order differs from the point-in-time matrix",
            )
    try:
        numeric = np.asarray(
            [
                *scales, *coefficients, artifact["intercept"],
                artifact["calibration_slope"], artifact["selected_c"],
            ],
            dtype=float,
        )
        report.require(np.isfinite(numeric).all(), "winner model contains non-finite parameters")
        report.require((np.asarray(scales, dtype=float) > 0).all(), "winner model scales must be positive")
        report.require(float(artifact["intercept"]) == 0.0, "winner model intercept must be zero")
        report.require(float(artifact["calibration_slope"]) > 0, "winner model calibration slope must be positive")
        report.require(float(artifact["selected_c"]) > 0, "winner model selected_c must be positive")
        report.require(
            float(artifact["selected_c"]) in artifact_c_grid,
            "winner model selected_c is not part of its regularization grid",
        )
    except (TypeError, ValueError):
        report.errors.append("winner model parameters are not numeric")
    unhashed = dict(artifact)
    supplied_model_id = unhashed.pop("model_id", None)
    try:
        canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected_model_id = sha256(canonical.encode("utf-8")).hexdigest()[:20]
        report.require(supplied_model_id == expected_model_id, "winner model_id does not match artifact contents")
    except (TypeError, ValueError):
        report.errors.append("winner model cannot be encoded canonically")
    state_max_timestamp = pd.to_datetime(raw["date"], errors="coerce").max()
    if auxiliary_fights is not None and not auxiliary_fights.empty:
        state_max_timestamp = max(
            state_max_timestamp,
            pd.to_datetime(auxiliary_fights["date"], errors="coerce").max(),
        )
    raw_max = pd.to_datetime(raw["date"], errors="coerce").max().strftime("%Y-%m-%d")
    state_max = state_max_timestamp.strftime("%Y-%m-%d")
    report.require(artifact["data_through"] == state_max, "winner model is not trained through latest replay state")
    report.require(
        artifact["source_data_through"] == state_max,
        "winner model source cutoff differs from latest replay state",
    )
    try:
        state_builder = PointInTimeDatasetBuilder(
            raw, fighters, auxiliary_fights=auxiliary_fights
        )
        prepared_raw = state_builder._validate_and_prepare_raw()
        expected_state_fingerprint = state_builder._state_source_fingerprint(
            prepared_raw
        )
        report.require(
            artifact["state_fingerprint_sha256"] == expected_state_fingerprint,
            "winner model state fingerprint differs from raw/profile source data",
        )
    except (KeyError, TypeError, ValueError) as error:
        report.errors.append(
            f"winner model state fingerprint could not be verified: {error}"
        )
    if point_in_time is not None:
        point_dates = pd.to_datetime(point_in_time["date"], errors="coerce")
        training_mask = point_dates >= (
            pd.to_datetime(raw_max, errors="raise") - pd.DateOffset(years=10)
        )
        expected_training_count = int(training_mask.sum())
        try:
            training_fights = int(artifact["training_fights"])
        except (TypeError, ValueError):
            training_fights = -1
            report.errors.append("winner model training_fights is not an integer")
        report.require(
            training_fights == expected_training_count,
            "winner model training count differs from its ten-year point-in-time window",
        )
        label_max = point_dates.max().strftime("%Y-%m-%d")
        report.require(
            artifact["training_labels_through"] == label_max,
            "winner model label cutoff differs from latest point-in-time W/L",
        )
        if isinstance(features, list) and set(features) <= set(point_in_time.columns):
            training = point_in_time.loc[training_mask].copy()
            training["date"] = point_dates.loc[training_mask]
            expected_fingerprint = training_fingerprint(training, features)
            report.require(
                artifact["training_fingerprint_sha256"] == expected_fingerprint,
                "winner model training fingerprint differs from point-in-time data",
            )
    report.facts.append(
        f"winner model: {artifact['model_id']} / {artifact['training_fights']} fights"
    )
    return report


def _load_json(path: Path, report: ValidationReport):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        report.errors.append(f"{path.name} is not valid readable JSON: {error}")
        return None


def validate_publication(
    data_root: Path,
    raw: pd.DataFrame,
    fighters: pd.DataFrame,
    *,
    allow_stale: bool = False,
    point_in_time: pd.DataFrame | None = None,
    auxiliary_fights: pd.DataFrame | None = None,
    require_model_artifact: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    external = data_root / "external"
    objects = {
        path.stem: _load_json(path, report) for path in sorted(external.glob("*.json"))
    }
    required_json = {
        "card_info",
        "fighter_explorer",
        "fighter_stats",
        "prediction_history",
        "ufc_fight_data_for_website",
        "vegas_odds",
        *(f"fighter_fights_{key}" for key in FIGHTER_EXPLORER_SHARD_KEYS),
    }
    if require_model_artifact:
        required_json.add("winner_model")
    report.require(
        required_json.issubset(objects),
        f"missing publication JSON files: {sorted(required_json - set(objects))}",
    )
    if report.errors:
        return report

    winner_model = objects.get("winner_model")
    if winner_model is not None and require_model_artifact:
        report.merge(
            validate_model_artifact(
                winner_model, raw, fighters, point_in_time, auxiliary_fights
            )
        )

    website_data = objects["ufc_fight_data_for_website"]
    report.require(isinstance(website_data, dict), "website fight data must be a JSON object")
    if isinstance(website_data, dict):
        report.require(
            len(website_data) == len(raw),
            "website fight-data JSON row count differs from raw fights",
        )

    vegas_object = objects["vegas_odds"]
    try:
        vegas = pd.DataFrame(vegas_object)
    except (TypeError, ValueError) as error:
        report.errors.append(f"vegas_odds cannot be loaded as a table: {error}")
        vegas = pd.DataFrame()

    explorer_path = external / "fighter_explorer.json"
    explorer = objects["fighter_explorer"]
    fight_shards = {
        key: objects[f"fighter_fights_{key}"]
        for key in FIGHTER_EXPLORER_SHARD_KEYS
    }
    try:
        report.require(
            explorer_path.stat().st_size <= FIGHTER_EXPLORER_SIZE_LIMIT,
            "fighter explorer index exceeds its 8 MiB limit",
        )
        for key in FIGHTER_EXPLORER_SHARD_KEYS:
            report.require(
                (external / f"fighter_fights_{key}.json").stat().st_size
                <= FIGHTER_SHARD_SIZE_LIMIT,
                f"fighter explorer shard {key} exceeds its 4 MiB limit",
            )
        validated_explorer = validate_fighter_explorer(
            explorer,
            raw,
            fighters,
            vegas,
            fight_shards,
        )
        report.facts.append(
            "fighter explorer: "
            f"{validated_explorer['counts']['fighters']:,} fighters / "
            f"{validated_explorer['counts']['unique_fights']:,} fights"
        )
    except (OSError, TypeError, ValueError) as error:
        report.errors.append(f"fighter explorer publication is invalid: {error}")

    card_info = objects["card_info"]
    report.require(isinstance(card_info, dict), "card_info must be a JSON object")
    card_date = pd.NaT
    if isinstance(card_info, dict):
        report.require(bool(card_info.get("title")), "card_info title is blank")
        card_date = pd.to_datetime(card_info.get("date"), errors="coerce")
        report.require(pd.notna(card_date), "card_info date is invalid")
        if require_model_artifact:
            event_url = str(card_info.get("event_url", "")).strip()
            event_id = str(card_info.get("event_id", "")).strip()
            report.require(bool(event_url), "generated card_info event_url is blank")
            report.require(bool(event_id), "generated card_info event_id is blank")
            if event_url and event_id:
                report.require(
                    event_url.rstrip("/").rsplit("/", 1)[-1].lower()
                    == event_id.lower(),
                    "card_info event_id does not match its UFCStats URL",
                )

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
            if (
                require_model_artifact
                and winner_model is not None
                and isinstance(winner_model, dict)
            ):
                model_id = winner_model.get("model_id")
                model_columns = {
                    "model id", "model version", "model trained through",
                    "model probability", "model status", "forecast probability",
                    "forecast source", "betting status", "odds observed at",
                    "forecast issued at", "forecast source commit",
                    "event id", "event url", "fighter id", "opponent id",
                }
                if _require_columns(vegas, model_columns, "vegas odds model", report):
                    report.require(
                        vegas["model id"].astype(str).eq(str(model_id)).all(),
                        "vegas odds model IDs do not match winner_model.json",
                    )
                    report.require(
                        vegas["model version"].astype(str).eq(
                            str(winner_model.get("model_version"))
                        ).all(),
                        "vegas odds model versions do not match winner_model.json",
                    )
                    report.require(
                        vegas["model trained through"].astype(str).eq(
                            str(winner_model.get("data_through"))
                        ).all(),
                        "vegas odds training cutoffs do not match winner_model.json",
                    )
                    issued_at = pd.to_datetime(
                        vegas["forecast issued at"], errors="coerce", utc=True
                    )
                    report.require(
                        issued_at.notna().all(),
                        "vegas odds forecasts require a UTC issuance timestamp",
                    )
                    report.require(
                        vegas["forecast source commit"]
                        .astype(str)
                        .str.strip()
                        .str.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
                        .fillna(False)
                        .all(),
                        "vegas odds forecasts require a 40- or 64-hex source revision",
                    )
                    if isinstance(card_info, dict):
                        report.require(
                            vegas["event id"].astype(str).eq(
                                str(card_info.get("event_id", ""))
                            ).all(),
                            "vegas odds event IDs do not match card_info",
                        )
                        report.require(
                            vegas["event url"].astype(str).eq(
                                str(card_info.get("event_url", ""))
                            ).all(),
                            "vegas odds event URLs do not match card_info",
                        )
                    model_probability = pd.to_numeric(
                        vegas["model probability"], errors="coerce"
                    )
                    forecast_probability = pd.to_numeric(
                        vegas["forecast probability"], errors="coerce"
                    )
                    resolved = vegas["model status"] != "abstain_unresolved_identity"
                    report.require(
                        vegas.loc[resolved, "fighter id"]
                        .astype(str).str.strip().ne("").all()
                        and vegas.loc[resolved, "opponent id"]
                        .astype(str).str.strip().ne("").all(),
                        "resolved vegas forecasts require stable fighter IDs",
                    )
                    report.require(
                        model_probability[resolved].between(0, 1, inclusive="neither").all(),
                        "resolved model probabilities must be strictly between zero and one",
                    )
                    report.require(
                        forecast_probability[resolved].between(0, 1, inclusive="neither").all(),
                        "resolved forecast probabilities must be strictly between zero and one",
                    )
                    matched = vegas.get(
                        "odds source status", pd.Series("", index=vegas.index)
                    ).eq("matched")
                    observed = pd.to_datetime(
                        vegas.loc[matched, "odds observed at"], errors="coerce", utc=True
                    )
                    report.require(
                        observed.notna().all(),
                        "matched odds rows require a parseable observation timestamp",
                    )
                    market_rows = vegas["forecast source"].eq(
                        "market_no_vig_consensus"
                    )
                    market_probability = pd.to_numeric(
                        vegas.get(
                            "market no-vig fighter probability",
                            pd.Series(np.nan, index=vegas.index),
                        ),
                        errors="coerce",
                    )
                    report.require(
                        market_probability[market_rows].between(
                            0, 1, inclusive="neither"
                        ).all(),
                        "market forecasts require a valid no-vig probability",
                    )
                    report.require(
                        vegas["betting status"].eq(
                            "disabled_pending_market_relative_validation"
                        ).all(),
                        "betting must remain disabled until market-relative validation",
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


def validate_market_data(
    market_root: Path,
    *,
    required: bool = False,
) -> ValidationReport:
    """Validate the immutable quote/model ledgers used for shadow research."""

    report = ValidationReport()
    quote_csv = market_root / "quote_snapshots.csv"
    quote_jsonl = market_root / "quote_snapshots.jsonl"
    forecast_csv = market_root / "forecast_captures.csv"
    forecast_jsonl = market_root / "forecast_captures.jsonl"
    expected = (quote_csv, quote_jsonl, forecast_csv, forecast_jsonl)
    existing = [path.exists() for path in expected]
    if not any(existing):
        if required:
            report.errors.append("market quote/forecast ledgers are missing")
        return report
    if not all(existing):
        missing = [path.name for path, present in zip(expected, existing) if not present]
        report.errors.append(f"market ledger mirrors are incomplete: {missing}")
        return report

    try:
        quotes = QuoteSnapshotStore(quote_csv, quote_jsonl).read()
        forecasts = ForecastCaptureStore(forecast_csv, forecast_jsonl).read()
    except (OSError, UnicodeError, MarketDataError, StoreIntegrityError) as error:
        report.errors.append(f"market ledgers failed integrity validation: {error}")
        return report

    report.require(bool(quotes), "market quote ledger is empty")
    report.require(bool(forecasts), "market forecast ledger is empty")
    if not quotes or not forecasts:
        return report

    capture_contracts: dict[str, set[tuple]] = {}
    capture_source_payloads: dict[tuple[str, str], set[str]] = {}
    for quote in quotes:
        capture_contracts.setdefault(quote.capture_id, set()).add(
            (
                quote.event_id,
                quote.event_date,
                quote.timing_precision,
                quote.event_start_utc,
                quote.observed_at_utc,
            )
        )
        capture_source_payloads.setdefault(
            (quote.capture_id, quote.source.casefold()), set()
        ).add(quote.source_payload_sha256)
    report.require(
        all(len(values) == 1 for values in capture_contracts.values()),
        "one market capture_id spans multiple event/timing/retrieval contracts",
    )
    report.require(
        all(len(values) == 1 for values in capture_source_payloads.values()),
        "one market capture/source has multiple payload hashes",
    )

    forecast_by_key: dict[tuple[str, str], list] = {}
    for forecast in forecasts:
        forecast_by_key.setdefault(
            (forecast.capture_id, forecast.matchup_id), []
        ).append(forecast)
        report.require(
            forecast.capture_id in capture_contracts
            and any(
                contract[0] == forecast.event_id
                for contract in capture_contracts[forecast.capture_id]
            ),
            "market forecast capture has no quote event with the same capture_id",
        )
    report.require(
        all(len(values) == 1 for values in forecast_by_key.values()),
        "one market capture/matchup has multiple frozen forecasts",
    )

    quote_keys = {(quote.capture_id, quote.matchup_id) for quote in quotes}
    forecast_keys = set(forecast_by_key)
    report.require(
        forecast_keys <= quote_keys,
        "a frozen market forecast has no quote for the same capture/matchup",
    )
    matched_forecast_keys = quote_keys & forecast_keys
    report.require(
        bool(matched_forecast_keys),
        "market quote and forecast ledgers have no matching capture/matchup",
    )
    for key in matched_forecast_keys:
        forecast = forecast_by_key[key][0]
        matchup_quotes = [
            quote
            for quote in quotes
            if (quote.capture_id, quote.matchup_id) == key
        ]
        observed = min(quote.observed_at_utc for quote in matchup_quotes)
        forecast_identity = (
            forecast.event_id,
            forecast.fighter_id,
            forecast.opponent_id,
            forecast.event_date,
            forecast.timing_precision,
            forecast.event_start_utc,
        )
        report.require(
            all(
                (
                    quote.event_id,
                    quote.fighter_id,
                    quote.opponent_id,
                    quote.event_date,
                    quote.timing_precision,
                    quote.event_start_utc,
                )
                == forecast_identity
                for quote in matchup_quotes
            ),
            "a market quote and frozen forecast disagree on matchup timing/identity",
        )
        known_fight_ids = {
            value
            for value in [
                forecast.fight_id,
                *(quote.fight_id for quote in matchup_quotes),
            ]
            if value is not None
        }
        report.require(
            len(known_fight_ids) <= 1,
            "a market quote and frozen forecast disagree on fight_id",
        )
        report.require(
            forecast.forecast_issued_at_utc <= observed,
            "a market quote was observed before its frozen model forecast existed",
        )

    quote_by_id = {quote.quote_id: quote for quote in quotes}
    forecast_by_id = {
        forecast.forecast_capture_id: forecast for forecast in forecasts
    }

    metadata_csv = market_root / "quote_source_metadata.csv"
    metadata_jsonl = market_root / "quote_source_metadata.jsonl"
    metadata_exists = (metadata_csv.exists(), metadata_jsonl.exists())
    metadata = ()
    if any(metadata_exists) and not all(metadata_exists):
        report.errors.append("quote source metadata mirrors are incomplete")
    elif all(metadata_exists):
        try:
            metadata = QuoteSourceMetadataStore(
                metadata_csv, metadata_jsonl
            ).read()
        except (OSError, UnicodeError, MarketDataError, StoreIntegrityError) as error:
            report.errors.append(
                f"quote source metadata failed integrity validation: {error}"
            )
    metadata_by_quote = {item.quote_id: item for item in metadata}
    for item in metadata:
        quote = quote_by_id.get(item.quote_id)
        report.require(
            quote is not None,
            "quote source metadata references an unknown quote_id",
        )
        if quote is None:
            continue
        report.require(
            (
                item.capture_id,
                item.matchup_id,
                item.event_id,
                item.source,
                item.book,
                item.observed_at_utc,
            )
            == (
                quote.capture_id,
                quote.matchup_id,
                quote.event_id,
                quote.source,
                quote.book,
                quote.observed_at_utc,
            ),
            "quote source metadata disagrees with its immutable quote",
        )
    api_quotes = [quote for quote in quotes if quote.source == "the-odds-api.com"]
    missing_api_metadata = [
        quote for quote in api_quotes if quote.quote_id not in metadata_by_quote
    ]
    if missing_api_metadata:
        report.warnings.append(
            f"market ledger has {len(missing_api_metadata):,} legacy API quote(s) "
            "without source-side update timestamps"
        )

    decision_csv = market_root / "paper_decisions.csv"
    decision_jsonl = market_root / "paper_decisions.jsonl"
    decision_exists = (decision_csv.exists(), decision_jsonl.exists())
    decisions = ()
    if any(decision_exists) and not all(decision_exists):
        report.errors.append("paper decision mirrors are incomplete")
    elif all(decision_exists):
        try:
            decisions = PaperDecisionStore(decision_csv, decision_jsonl).read()
        except (OSError, UnicodeError, MarketDataError, StoreIntegrityError) as error:
            report.errors.append(f"paper decisions failed integrity validation: {error}")

    report.require(
        len({item.matchup_id for item in decisions}) == len(decisions),
        "prospective policy froze more than one decision for a matchup",
    )
    for decision in decisions:
        reference = quote_by_id.get(decision.reference_quote_id)
        forecast = forecast_by_id.get(decision.forecast_capture_id)
        report.require(reference is not None, "paper decision references an unknown quote")
        report.require(
            forecast is not None, "paper decision references an unknown forecast"
        )
        if reference is None or forecast is None:
            continue
        if decision.timing_precision != "timestamp" or not decision.event_start_utc:
            report.errors.append("prospective paper decision lacks exact event timing")
            continue
        lead = (
            pd.Timestamp(decision.event_start_utc)
            - pd.Timestamp(decision.market_as_of_utc)
        ).total_seconds()
        report.require(
            abs(lead - DECISION_TARGET_LEAD_SECONDS) <= DECISION_WINDOW_SECONDS,
            "prospective paper decision is outside the locked T-24 window",
        )
        fresh_quotes = [
            quote
            for quote in quotes
            if quote.capture_id == decision.capture_id
            and quote.matchup_id == decision.matchup_id
            and quote.quote_id in metadata_by_quote
            and -300.0
            <= float(metadata_by_quote[quote.quote_id].source_quote_age_seconds)
            <= MAX_SOURCE_QUOTE_AGE_SECONDS
        ]
        try:
            market = consensus_as_of(
                fresh_quotes,
                capture_id=decision.capture_id,
                matchup_id=decision.matchup_id,
                as_of_utc=decision.market_as_of_utc,
                min_books=MIN_CONSENSUS_BOOKS,
                exclude_books=(reference.book,),
            )
            rebuilt = type(decision).create(
                market,
                reference,
                forecast,
                selected_gamma=decision.selected_gamma,
                decision_issued_at_utc=decision.decision_issued_at_utc,
                minimum_expected_return=decision.minimum_expected_return,
                maximum_quote_age_seconds=decision.maximum_quote_age_seconds,
                fight_id=decision.fight_id,
            )
            report.require(
                rebuilt == decision,
                "paper decision cannot be reproduced from its frozen inputs",
            )
        except (MarketDataError, StoreIntegrityError, ValueError) as error:
            report.errors.append(
                f"paper decision cannot be reconstructed: {error}"
            )

    opportunities_path = market_root / "current_opportunities.json"
    if opportunities_path.exists():
        try:
            if opportunities_path.stat().st_size > 256 * 1024:
                raise ValueError("current opportunity publication exceeds 256 KiB")
            opportunities = json.loads(
                opportunities_path.read_text(encoding="utf-8")
            )
            if not isinstance(opportunities, dict):
                raise ValueError("current opportunity publication is not an object")
            capture_id = str(opportunities.get("capture_id", "")).strip()
            if not capture_id:
                raise ValueError("current opportunity publication lacks capture_id")
            validate_current_opportunities(
                opportunities,
                quotes,
                forecasts,
                metadata,
                decisions,
                capture_id=capture_id,
            )
            report.require(
                opportunities.get("betting_status") == BETTING_STATUS
                and opportunities.get("paper_only") is True
                and opportunities.get("execution_enabled") is False,
                "current opportunity publication must keep execution disabled",
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
            MarketDataError,
            StoreIntegrityError,
        ) as error:
            report.errors.append(
                f"current opportunity publication is invalid: {error}"
            )

    settlement_csv = market_root / "paper_settlements.csv"
    settlement_jsonl = market_root / "paper_settlements.jsonl"
    settlement_exists = (settlement_csv.exists(), settlement_jsonl.exists())
    settlements = ()
    if any(settlement_exists) and not all(settlement_exists):
        report.errors.append("paper settlement mirrors are incomplete")
    elif all(settlement_exists):
        try:
            settlements = PaperSettlementStore(
                settlement_csv, settlement_jsonl
            ).read()
            summarize_paper_settlements(decisions, settlements)
        except (OSError, UnicodeError, MarketDataError, StoreIntegrityError) as error:
            report.errors.append(f"paper settlements failed integrity validation: {error}")

    performance_path = market_root / "performance_report.json"
    if performance_path.exists():
        try:
            if performance_path.stat().st_size > 64 * 1024:
                raise ValueError("performance report exceeds 64 KiB")
            performance = json.loads(performance_path.read_text(encoding="utf-8"))
            if not isinstance(performance, dict):
                raise ValueError("performance report is not an object")
            expected_report_hash = performance.pop("report_sha256", None)
            if expected_report_hash != canonical_hash(performance):
                raise ValueError("performance report hash does not match its contents")
            report.require(
                performance.get("betting_status") == BETTING_STATUS
                and performance.get("paper_only") is True
                and performance.get("execution_enabled") is False,
                "paper performance report must keep execution disabled",
            )
            report.require(
                performance.get("decision_dataset_sha256")
                == canonical_hash([item.to_mapping() for item in decisions]),
                "performance report decision hash is stale",
            )
            report.require(
                performance.get("settlement_dataset_sha256")
                == canonical_hash([item.to_mapping() for item in settlements]),
                "performance report settlement hash is stale",
            )
            if int(performance.get("schema_version", 1)) >= 2:
                report.require(
                    performance.get("quote_dataset_sha256")
                    == canonical_hash([item.to_mapping() for item in quotes]),
                    "performance report quote hash is stale",
                )
                report.require(
                    performance.get("source_metadata_dataset_sha256")
                    == canonical_hash([item.to_mapping() for item in metadata]),
                    "performance report source metadata hash is stale",
                )
                timing = performance.get("entry_timing_experiment")
                report.require(
                    isinstance(timing, dict)
                    and timing.get("policy_version") == TIMING_POLICY_VERSION
                    and timing.get("paper_only") is True
                    and timing.get("execution_enabled") is False,
                    "performance report timing experiment contract is invalid",
                )
            expected_metrics = summarize_paper_settlements(
                decisions, settlements
            ).to_mapping()
            report.require(
                performance.get("paper_metrics") == expected_metrics,
                "performance report metrics cannot be reproduced",
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
            MarketDataError,
            StoreIntegrityError,
        ) as error:
            report.errors.append(f"paper performance report is invalid: {error}")

    report.facts.append(
        "market ledger: "
        f"{len(quotes):,} quotes / {len(forecasts):,} forecasts / "
        f"{len(capture_contracts):,} captures / {len(decisions):,} paper decisions / "
        f"{len(settlements):,} settlements"
    )
    unmatched = len(quote_keys - forecast_keys)
    if unmatched:
        report.warnings.append(
            f"market ledger has {unmatched:,} quote matchup(s) without a model forecast"
        )
    return report


def validate_repository(
    repo_root: Path,
    *,
    allow_stale: bool = False,
    require_model_artifact: bool = False,
    require_market_data: bool = False,
) -> ValidationReport:
    data_root = repo_root / "src" / "content" / "data"
    processed = data_root / "processed"
    raw = pd.read_csv(processed / "ufc_fights_reported_doubled.csv", low_memory=False)
    fighters = pd.read_csv(processed / "fighter_stats.csv", low_memory=False)
    point_path = processed / "ufc_fights_point_in_time.csv"
    point_in_time = (
        pd.read_csv(point_path, low_memory=False) if point_path.exists() else None
    )
    report = ValidationReport()
    auxiliary_path = processed / "external_mma_auxiliary_doubled.csv"
    try:
        auxiliary_fights = load_approved_auxiliary(
            auxiliary_path,
            data_root / "external_mma" / "model_policy.json",
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        auxiliary_fights = None
        report.errors.append(f"external MMA model policy is invalid: {error}")

    external_mma_root = data_root / "external_mma"
    if external_mma_root.exists():
        try:
            external_report = ExternalMmaStore(external_mma_root).validate()
            report.facts.append(
                "external MMA: "
                f"{external_report['observations']:,} bouts / "
                f"{external_report['snapshots']:,} snapshots"
            )
        except (ExternalDataError, OSError, UnicodeError, ValueError) as error:
            report.errors.append(f"external MMA ledger is invalid: {error}")
    report.merge(
        validate_market_data(
            data_root / "market", required=require_market_data
        )
    )
    report.merge(validate_raw_fights(raw))
    report.merge(validate_fighters(fighters))
    raw_schema_valid = RAW_REQUIRED_COLUMNS <= set(raw.columns)
    if require_model_artifact:
        report.require(
            {"source_card_index", "bout_order", "time_format"} <= set(raw.columns),
            "generated raw fights require durable card order and time_format fields",
        )
    if raw_schema_valid:
        if point_in_time is not None and require_model_artifact:
            report.merge(validate_point_in_time(raw, point_in_time))
        elif require_model_artifact:
            report.errors.append("ufc_fights_point_in_time.csv is missing")
        elif point_in_time is not None:
            report.warnings.append(
                "point-in-time/model compatibility was deferred to the post-update validation"
            )
        report.merge(
            validate_publication(
                data_root,
                raw,
                fighters,
                allow_stale=allow_stale,
                point_in_time=point_in_time,
                auxiliary_fights=auxiliary_fights,
                require_model_artifact=require_model_artifact,
            )
        )
    else:
        report.errors.append(
            "dependent point-in-time/publication checks skipped because raw schema is invalid"
        )

    raw_dates = pd.to_datetime(raw["date"], errors="coerce")
    ordering_problems = []
    if not raw_dates.is_monotonic_decreasing:
        ordering_problems.append("raw fights are not sorted newest-to-oldest")
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
        "--require-market-data",
        action="store_true",
        help="require and validate immutable quote/model capture ledgers",
    )
    parser.add_argument(
        "--require-model-artifact",
        action="store_true",
        help="require and cross-check the point-in-time matrix and winner model",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="run structural checks without rejecting old completed/upcoming snapshots",
    )
    args = parser.parse_args()

    report = validate_repository(
        args.repo_root.resolve(),
        allow_stale=args.allow_stale,
        require_model_artifact=args.require_model_artifact,
        require_market_data=args.require_market_data,
    )
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
