"""Leakage-safe features and temporal modeling for UFC winner forecasts.

The raw UFCStats table stores two mirrored rows for every physical fight.  This
module reduces those rows to one stable-ID matchup, snapshots both fighters
immediately before each bout, and only then applies its result.  Explicit card
order lets a later tournament round see an earlier same-event result without
letting future bouts alter an old training vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


MODEL_SCHEMA_VERSION = 1
MODEL_VERSION = "point-in-time-elo-logistic-v2"
REGULARIZATION_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)
ELO_K_FACTORS = (32.0, 64.0, 128.0)
ELO_NAMES = ("elo_slow", "elo_medium", "elo_fast")
PIT_SORT_COLUMNS = ("date", "event_id", "bout_order", "fight_id")

COUNT_STATS = (
    "knockdowns",
    "sig_strikes_landed",
    "sig_strikes_attempts",
    "total_strikes_landed",
    "total_strikes_attempts",
    "takedowns_landed",
    "takedowns_attempts",
    "sub_attempts",
    "reversals",
    "control",
)


def _identity_token(url: object) -> str:
    """Return the stable UFCStats identifier portion of a URL."""
    if url is None:
        return ""
    missing = pd.isna(url)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return ""
    value = str(url).strip().rstrip("/")
    if value.casefold() in {"", "nan", "none", "<na>"}:
        return ""
    return value.rsplit("/", 1)[-1]


def _normalise_name(name: object) -> str:
    value = unicodedata.normalize("NFKD", str(name or ""))
    value = "".join(character for character in value if not unicodedata.combining(character))
    # Retain suffixes such as Jr/III: they can distinguish two real people.
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _parse_height(value: object) -> float:
    match = re.search(r"(\d+)\s*'\s*(\d+)", str(value or ""))
    if not match:
        return math.nan
    return float(int(match.group(1)) * 12 + int(match.group(2)))


def _parse_reach(value: object) -> float:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else math.nan


def _safe_number(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(number) else float(number)


def _optional_number(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return math.nan if pd.isna(number) else float(number)


@dataclass
class FightRecord:
    date: pd.Timestamp
    result: float | None
    method: str
    division: str
    opponent_rating: float
    seconds: float
    own_stats: dict[str, float]
    opponent_stats: dict[str, float]


@dataclass
class FighterState:
    ratings: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(ELO_NAMES, 1500.0)
    )
    division_ratings: dict[str, dict[str, float]] = field(default_factory=dict)
    records: list[FightRecord] = field(default_factory=list)


class PointInTimeDatasetBuilder:
    """Build UFC labels while optionally replaying non-label external history."""

    # Challenger builders may retain additional historical counters while the
    # production builder keeps its frozen 82-feature contract unchanged.
    STATE_COUNT_STATS = COUNT_STATS
    LANDED_ATTEMPTED_PAIRS = (
        ("sig_strikes_landed", "sig_strikes_attempts"),
        ("total_strikes_landed", "total_strikes_attempts"),
        ("takedowns_landed", "takedowns_attempts"),
    )

    def __init__(
        self,
        raw_fights: pd.DataFrame,
        fighter_stats: pd.DataFrame,
        auxiliary_fights: pd.DataFrame | None = None,
    ):
        self.raw_fights = raw_fights.copy()
        self.fighter_stats = fighter_stats.copy()
        self.auxiliary_fights = (
            auxiliary_fights.copy() if auxiliary_fights is not None else None
        )
        self.state_count_stats = tuple(self.STATE_COUNT_STATS)
        self.landed_attempted_pairs = tuple(self.LANDED_ATTEMPTED_PAIRS)
        self.states: dict[str, FighterState] = {}
        self.training_data: pd.DataFrame | None = None
        self._replayed_through: pd.Timestamp | None = None
        self.state_fingerprint_sha256: str | None = None
        self._profiles = self._build_profiles()
        self._name_to_ids = self._build_name_index()

        # Every side feature is differenced, so swapping fighter/opponent
        # negates the complete vector and guarantees p(A, B) = 1 - p(B, A)
        # for the zero-intercept logistic model.
        sample = self._side_features("__new_fighter__", pd.Timestamp("2020-01-01"), "")
        self.side_feature_names = tuple(sample)
        self.feature_columns = tuple(f"{name}_diff" for name in sample)

    def _build_profiles(self) -> dict[str, dict[str, object]]:
        required = {"url", "name", "dob", "height", "reach"}
        missing = required - set(self.fighter_stats.columns)
        if missing:
            raise ValueError(f"fighter_stats is missing required columns: {sorted(missing)}")
        profiles: dict[str, dict[str, object]] = {}
        for row in self.fighter_stats.to_dict("records"):
            fighter_id = _identity_token(row["url"])
            if not fighter_id:
                continue
            if fighter_id in profiles:
                raise ValueError(f"Duplicate fighter URL/ID in fighter_stats: {row['url']}")
            profiles[fighter_id] = {
                "name": str(row.get("name", "")).strip(),
                "dob": pd.to_datetime(row.get("dob"), errors="coerce"),
                "height": _parse_height(row.get("height")),
                "reach": _parse_reach(row.get("reach")),
            }
        return profiles

    def _build_name_index(self) -> dict[str, set[str]]:
        name_to_ids: dict[str, set[str]] = {}
        for fighter_id, profile in self._profiles.items():
            name_to_ids.setdefault(_normalise_name(profile["name"]), set()).add(fighter_id)
        required = {"fighter", "fighter_url"}
        if required <= set(self.raw_fights.columns):
            for name, url in self.raw_fights[["fighter", "fighter_url"]].itertuples(index=False):
                name_to_ids.setdefault(_normalise_name(name), set()).add(_identity_token(url))
        return name_to_ids

    def resolve_fighter_id(self, name: object, url: object = None) -> str | None:
        """Resolve an upcoming fighter without collapsing ambiguous names."""
        if url is not None and str(url).strip():
            return _identity_token(url)
        candidates = self._name_to_ids.get(_normalise_name(name), set())
        return next(iter(candidates)) if len(candidates) == 1 else None

    @staticmethod
    def _division_key(division: object) -> str:
        return str(division or "Unknown").strip() or "Unknown"

    @staticmethod
    def _rating_probability(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def _decayed_rating(
        rating: float,
        last_date: pd.Timestamp | None,
        fight_date: pd.Timestamp,
    ) -> float:
        if last_date is None:
            return float(rating)
        inactive_years = max((fight_date - last_date).days, 0) / 365.25
        # Gentle regression represents growing uncertainty after inactivity;
        # it was selected only from earlier temporal folds.
        return 1500.0 + (float(rating) - 1500.0) * (0.92 ** inactive_years)

    @staticmethod
    def _is_no_contest(rows: pd.DataFrame) -> bool:
        results = set(rows["result"].astype(str).str.upper())
        method = " ".join(rows["method"].fillna("").astype(str)).casefold()
        return "NC" in results or "cnc" in method or "overturned" in method

    @staticmethod
    def _score_for(row: pd.Series, rows: pd.DataFrame) -> float | None:
        if PointInTimeDatasetBuilder._is_no_contest(rows):
            return None
        result = str(row["result"]).upper()
        if result == "W":
            return 1.0
        if result == "L":
            return 0.0
        if result == "D":
            return 0.5
        return None

    @staticmethod
    def _smoothed_rate(numerator: float, denominator: float, prior: float, weight: float) -> float:
        return (numerator + prior * weight) / (denominator + weight)

    def _performance_features(self, records: list[FightRecord], prefix: str) -> dict[str, float]:
        def observed(side: str, column: str) -> tuple[float, float]:
            values_and_seconds = [
                (getattr(record, side)[column], max(record.seconds, 0.0))
                for record in records
                if math.isfinite(getattr(record, side)[column])
            ]
            total = sum(value for value, _seconds in values_and_seconds)
            # One 15-minute bout of prior exposure prevents sparse histories
            # from producing infinities. Unknown source stats do not count as
            # zero or contribute fight time to that statistic's denominator.
            exposure = sum(seconds for _value, seconds in values_and_seconds) / 900.0 + 1.0
            return total, exposure

        def observed_pair(side: str, landed: str, attempted: str) -> tuple[float, float]:
            pairs = [
                (getattr(record, side)[landed], getattr(record, side)[attempted])
                for record in records
                if math.isfinite(getattr(record, side)[landed])
                and math.isfinite(getattr(record, side)[attempted])
            ]
            return (
                sum(value_landed for value_landed, _value_attempted in pairs),
                sum(value_attempted for _value_landed, value_attempted in pairs),
            )

        own_sig_landed, own_sig_exposure = observed("own_stats", "sig_strikes_landed")
        absorbed_sig_landed, absorbed_sig_exposure = observed(
            "opponent_stats", "sig_strikes_landed"
        )
        own_sig_pair = observed_pair(
            "own_stats", "sig_strikes_landed", "sig_strikes_attempts"
        )
        absorbed_sig_pair = observed_pair(
            "opponent_stats", "sig_strikes_landed", "sig_strikes_attempts"
        )
        own_td_pair = observed_pair(
            "own_stats", "takedowns_landed", "takedowns_attempts"
        )
        absorbed_td_pair = observed_pair(
            "opponent_stats", "takedowns_landed", "takedowns_attempts"
        )
        sig_accuracy = self._smoothed_rate(
            own_sig_pair[0], own_sig_pair[1], 0.45, 40.0
        )
        sig_defence = 1.0 - self._smoothed_rate(
            absorbed_sig_pair[0],
            absorbed_sig_pair[1],
            0.45,
            40.0,
        )
        td_accuracy = self._smoothed_rate(
            own_td_pair[0], own_td_pair[1], 0.35, 8.0
        )
        td_defence = 1.0 - self._smoothed_rate(
            absorbed_td_pair[0],
            absorbed_td_pair[1],
            0.35,
            8.0,
        )

        def rate(side: str, column: str) -> float:
            total, exposure = observed(side, column)
            return total / exposure

        return {
            f"{prefix}_sig_landed_per15": own_sig_landed / own_sig_exposure,
            f"{prefix}_sig_absorbed_per15": absorbed_sig_landed / absorbed_sig_exposure,
            f"{prefix}_sig_accuracy": sig_accuracy,
            f"{prefix}_sig_defence": sig_defence,
            f"{prefix}_total_landed_per15": rate("own_stats", "total_strikes_landed"),
            f"{prefix}_total_absorbed_per15": rate("opponent_stats", "total_strikes_landed"),
            f"{prefix}_td_landed_per15": rate("own_stats", "takedowns_landed"),
            f"{prefix}_td_absorbed_per15": rate("opponent_stats", "takedowns_landed"),
            f"{prefix}_td_accuracy": td_accuracy,
            f"{prefix}_td_defence": td_defence,
            f"{prefix}_control_per15": rate("own_stats", "control"),
            f"{prefix}_control_absorbed_per15": rate("opponent_stats", "control"),
            f"{prefix}_knockdowns_per15": rate("own_stats", "knockdowns"),
            f"{prefix}_knockdowns_absorbed_per15": rate("opponent_stats", "knockdowns"),
            f"{prefix}_sub_attempts_per15": rate("own_stats", "sub_attempts"),
            f"{prefix}_reversals_per15": rate("own_stats", "reversals"),
        }

    @staticmethod
    def _record_features(records: list[FightRecord], prefix: str) -> dict[str, float]:
        wins = sum(record.result == 1.0 for record in records)
        losses = sum(record.result == 0.0 for record in records)
        draws = sum(record.result == 0.5 for record in records)
        decisions = wins + losses
        finishes = sum(
            record.result == 1.0 and ("KO" in record.method.upper() or "SUB" in record.method.upper())
            for record in records
        )
        ko_wins = sum(record.result == 1.0 and "KO" in record.method.upper() for record in records)
        sub_wins = sum(record.result == 1.0 and "SUB" in record.method.upper() for record in records)
        finish_losses = sum(
            record.result == 0.0 and ("KO" in record.method.upper() or "SUB" in record.method.upper())
            for record in records
        )
        return {
            f"{prefix}_fights_log": math.log1p(len(records)),
            f"{prefix}_wins_log": math.log1p(wins),
            f"{prefix}_losses_log": math.log1p(losses),
            f"{prefix}_win_rate": (wins + 0.5 * draws + 1.5) / (decisions + draws + 3.0),
            f"{prefix}_finish_win_rate": (finishes + 1.0) / (decisions + 4.0),
            f"{prefix}_ko_win_rate": (ko_wins + 0.75) / (decisions + 4.0),
            f"{prefix}_sub_win_rate": (sub_wins + 0.75) / (decisions + 4.0),
            f"{prefix}_finish_loss_rate": (finish_losses + 1.0) / (decisions + 4.0),
        }

    def _side_features(
        self,
        fighter_id: str,
        fight_date: pd.Timestamp,
        division: object,
    ) -> dict[str, float]:
        state = self.states.get(fighter_id, FighterState())
        division_key = self._division_key(division)
        division_ratings = state.division_ratings.get(
            division_key, dict.fromkeys(ELO_NAMES, 1500.0)
        )
        profile = self._profiles.get(fighter_id, {})
        dob = profile.get("dob", pd.NaT)
        age_known = float(pd.notna(dob))
        age = (fight_date - dob).days / 365.25 if age_known else 29.0
        height_known = float(pd.notna(profile.get("height", math.nan)))
        reach_known = float(pd.notna(profile.get("reach", math.nan)))
        height = float(profile.get("height", math.nan)) if height_known else 69.0
        reach = float(profile.get("reach", math.nan)) if reach_known else 70.0

        # State is updated only after a bout is emitted.  Allow already-played
        # same-day tournament rounds while still excluding any later date when
        # this helper is used for a historical what-if.
        prior_records = [record for record in state.records if record.date <= fight_date]
        recent_1y = [record for record in prior_records if (fight_date - record.date).days <= 365]
        recent_3y = [record for record in prior_records if (fight_date - record.date).days <= 3 * 365]
        division_records = [record for record in prior_records if record.division == division_key]
        last_date = prior_records[-1].date if prior_records else None
        last_division_date = division_records[-1].date if division_records else None
        effective_ratings = {
            name: self._decayed_rating(state.ratings[name], last_date, fight_date)
            for name in ELO_NAMES
        }
        effective_division_ratings = {
            name: self._decayed_rating(
                division_ratings[name], last_division_date, fight_date
            )
            for name in ELO_NAMES
        }
        if prior_records:
            days_since = min(max((fight_date - prior_records[-1].date).days, 0), 5 * 365)
            average_opponent_rating = (
                sum(record.opponent_rating for record in prior_records) + 2.0 * 1500.0
            ) / (len(prior_records) + 2.0)
        else:
            days_since = 2 * 365
            average_opponent_rating = 1500.0

        features: dict[str, float] = {}
        features.update(effective_ratings)
        features.update({
            f"division_{name}": effective_division_ratings[name]
            for name in ELO_NAMES
        })
        reliability = len(prior_records) / (len(prior_records) + 5.0)
        features["elo_medium_reliable"] = (
            1500.0 + (effective_ratings["elo_medium"] - 1500.0) * reliability
        )
        features["rating_uncertainty"] = 1.0 / math.sqrt(len(prior_records) + 1.0)
        features["average_opponent_elo"] = average_opponent_rating
        features["days_since_fight_log"] = math.log1p(days_since)
        features["has_history"] = float(bool(prior_records))
        features["age"] = float(age)
        features["age_squared"] = float(age) ** 2
        features["age_known"] = age_known
        features["height"] = height
        features["height_known"] = height_known
        features["reach"] = reach
        features["reach_known"] = reach_known
        features.update(self._record_features(prior_records, "career"))
        features.update(self._record_features(recent_1y, "recent_1y"))
        features.update(self._record_features(recent_3y, "recent_3y"))
        features.update(self._record_features(division_records, "division"))
        features.update(self._performance_features(prior_records, "career"))
        features.update(self._performance_features(recent_3y, "recent_3y"))
        return features

    def _matchup_features_from_current_state(
        self,
        fighter_id: str,
        opponent_id: str,
        timestamp: pd.Timestamp,
        division: object,
    ) -> pd.DataFrame:
        fighter = self._side_features(fighter_id, timestamp, division)
        opponent = self._side_features(opponent_id, timestamp, division)
        values = {
            f"{name}_diff": fighter[name] - opponent[name]
            for name in self.side_feature_names
        }
        return pd.DataFrame([values], columns=self.feature_columns)

    def matchup_features(
        self,
        fighter_id: str,
        opponent_id: str,
        fight_date: object,
        division: object,
    ) -> pd.DataFrame:
        """Build a future matchup from the fully replayed current state.

        Historical queries would otherwise combine date-filtered records with
        final Elo state. Historical rows must come from ``build()``, which
        snapshots them during replay.
        """
        timestamp = pd.to_datetime(fight_date, errors="raise").normalize()
        if self._replayed_through is not None and timestamp <= self._replayed_through:
            raise ValueError(
                "Historical matchup features must be read from the point-in-time "
                "dataset; ad-hoc queries must be after the replay cutoff"
            )
        return self._matchup_features_from_current_state(
            fighter_id, opponent_id, timestamp, division
        )

    def history_count(self, fighter_id: str) -> int:
        return len(self.states.get(fighter_id, FighterState()).records)

    def _state_source_fingerprint(self, raw: pd.DataFrame) -> str:
        columns = [
            "date", "event_id", "fight_id", "fighter_id", "opponent_id",
            "result", "method", "division", "total_fight_time", "bout_order",
            *self.state_count_stats,
        ]
        if (
            "emit_training_target" in raw
            and not raw["emit_training_target"].astype(bool).all()
        ):
            columns.insert(columns.index("bout_order") + 1, "emit_training_target")
        if "time_format" in raw:
            columns.insert(columns.index("total_fight_time"), "time_format")
        source_text = raw[columns].to_csv(
            index=False,
            date_format="%Y-%m-%d",
            float_format="%.17g",
            lineterminator="\n",
        )
        profiles = {
            fighter_id: {
                "name": profile["name"],
                "dob": (
                    profile["dob"].strftime("%Y-%m-%d")
                    if pd.notna(profile["dob"]) else None
                ),
                "height": (
                    float(profile["height"])
                    if math.isfinite(profile["height"]) else None
                ),
                "reach": (
                    float(profile["reach"])
                    if math.isfinite(profile["reach"]) else None
                ),
            }
            for fighter_id, profile in sorted(self._profiles.items())
        }
        profile_text = json.dumps(
            profiles, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return sha256(f"{source_text}\n{profile_text}".encode("utf-8")).hexdigest()

    def _validate_and_prepare_raw(self) -> pd.DataFrame:
        required = {
            "date", "fight_url", "event_url", "fighter_url", "opponent_url",
            "fighter", "opponent", "result", "method", "division", "round",
            "total_fight_time",
            *self.state_count_stats,
        }
        missing = required - set(self.raw_fights.columns)
        if missing:
            raise ValueError(f"raw fights are missing required columns: {sorted(missing)}")
        raw = self.raw_fights.copy()
        if "emit_training_target" in raw:
            supplied = raw["emit_training_target"].astype(str).str.casefold()
            if not supplied.isin({"true", "1"}).all():
                raise ValueError("primary UFC fights must emit training targets")
        raw["emit_training_target"] = True
        if self.auxiliary_fights is not None and not self.auxiliary_fights.empty:
            auxiliary = self.auxiliary_fights.copy()
            missing_auxiliary = required - set(auxiliary.columns)
            if missing_auxiliary:
                raise ValueError(
                    "auxiliary fights are missing required columns: "
                    f"{sorted(missing_auxiliary)}"
                )
            if "emit_training_target" not in auxiliary:
                raise ValueError("auxiliary fights require emit_training_target=False")
            supplied = auxiliary["emit_training_target"].astype(str).str.casefold()
            if not supplied.isin({"false", "0"}).all():
                raise ValueError("auxiliary fights must not emit training targets")
            auxiliary["emit_training_target"] = False
            raw = pd.concat([raw, auxiliary], ignore_index=True, sort=False)
        raw["_source_position"] = np.arange(len(raw))
        raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.normalize()
        raw["fighter_id"] = raw["fighter_url"].map(_identity_token)
        raw["opponent_id"] = raw["opponent_url"].map(_identity_token)
        raw["fight_id"] = raw["fight_url"].map(_identity_token)
        raw["event_id"] = raw["event_url"].map(_identity_token)
        for identity_column in (
            "fighter_id", "opponent_id", "fight_id", "event_id"
        ):
            if raw[identity_column].eq("").any():
                raise ValueError(f"raw fights contain a blank {identity_column}")
        if raw["fighter_id"].eq(raw["opponent_id"]).any():
            raise ValueError("raw fights contain a self-match or collapsed fighter ID")

        numeric_columns = ("total_fight_time", *self.state_count_stats)
        for column in numeric_columns:
            source = raw[column]
            parsed = pd.to_numeric(source, errors="coerce")
            malformed = parsed.isna() & source.notna() & source.astype(str).str.strip().ne("")
            if malformed.any():
                raise ValueError(f"raw {column} contains malformed non-null values")
            if (~np.isfinite(parsed.dropna().to_numpy(dtype=float))).any():
                raise ValueError(f"raw {column} contains non-finite values")
            if (parsed.dropna() < 0).any():
                raise ValueError(f"raw {column} contains negative values")
            raw[column] = parsed
        no_contest_rows = (
            raw["result"].astype(str).str.upper().eq("NC")
            | raw["method"].fillna("").astype(str).str.casefold().str.contains(
                "cnc|overturned", regex=True
            )
        )
        rounds = pd.to_numeric(raw["round"], errors="coerce")
        required_rounds = ~no_contest_rows & raw["emit_training_target"]
        if (
            rounds[required_rounds].isna().any()
            or (~np.isfinite(rounds.dropna().to_numpy(dtype=float))).any()
            or not rounds.dropna().between(1, 5).all()
            or not np.equal(rounds.dropna() % 1, 0).all()
        ):
            raise ValueError(
                "raw round must be an integer between 1 and 5 when the result is not NC"
            )
        raw["round"] = rounds
        duration = raw["total_fight_time"]
        if not duration.dropna().between(1, 7200).all():
            raise ValueError(
                "raw total_fight_time must be between 1 and 7200 seconds when known"
            )
        for landed, attempted in self.landed_attempted_pairs:
            invalid = raw[landed].notna() & raw[attempted].notna() & (
                raw[landed] > raw[attempted]
            )
            if invalid.any():
                raise ValueError(f"raw {landed} exceeds {attempted}")
        if (
            raw["control"].notna()
            & (raw["control"] > raw["total_fight_time"])
        ).any():
            raise ValueError("raw control exceeds total_fight_time")

        counts = raw.groupby("fight_id", dropna=False).size()
        invalid = counts[counts != 2]
        if not invalid.empty:
            raise ValueError(f"Every fight_id must have two sides; invalid: {invalid.head().to_dict()}")
        for fight_id, rows in raw.groupby("fight_id", sort=False):
            first, second = rows.iloc[0], rows.iloc[1]
            if (
                first["fighter_id"] != second["opponent_id"]
                or first["opponent_id"] != second["fighter_id"]
            ):
                raise ValueError(f"Mirrored fighter IDs do not agree for fight {fight_id}")
            if rows["date"].nunique() != 1 or rows["event_id"].nunique() != 1:
                raise ValueError(f"Mirrored event/date metadata do not agree for fight {fight_id}")
            results = rows["result"].astype(str).str.upper().tolist()
            if not (
                sorted(results) == ["L", "W"]
                or (len(set(results)) == 1 and results[0] in {"D", "NC"})
            ):
                raise ValueError(f"Mirrored results are not complementary for fight {fight_id}")
            for mirrored_column in (
                "method", "division", "round", "total_fight_time"
            ):
                if rows[mirrored_column].nunique(dropna=False) != 1:
                    raise ValueError(
                        f"Mirrored {mirrored_column} does not agree for fight {fight_id}"
                    )

        if "bout_order" in raw and raw["bout_order"].notna().all():
            parsed_bout_order = pd.to_numeric(raw["bout_order"], errors="raise")
            if (
                (~np.isfinite(parsed_bout_order.to_numpy(dtype=float))).any()
                or (parsed_bout_order < 0).any()
                or not np.equal(parsed_bout_order % 1, 0).all()
            ):
                raise ValueError("raw bout_order must contain nonnegative integers")
            raw["bout_order"] = parsed_bout_order.astype(int)
            if "source_card_index" in raw:
                parsed_source_index = pd.to_numeric(
                    raw["source_card_index"], errors="raise"
                )
                if (
                    parsed_source_index.isna().any()
                    or (~np.isfinite(parsed_source_index.to_numpy(dtype=float))).any()
                    or (parsed_source_index < 0).any()
                    or not np.equal(parsed_source_index % 1, 0).all()
                ):
                    raise ValueError(
                        "raw source_card_index must contain nonnegative integers"
                    )
                raw["source_card_index"] = parsed_source_index.astype(int)
            for fight_id, rows in raw.groupby("fight_id", sort=False):
                if rows["bout_order"].nunique() != 1:
                    raise ValueError(
                        f"Mirrored bout_order does not agree for fight {fight_id}"
                    )
                if (
                    "source_card_index" in raw
                    and rows["source_card_index"].nunique() != 1
                ):
                    raise ValueError(
                        f"Mirrored source_card_index does not agree for fight {fight_id}"
                    )
            physical_order = raw.drop_duplicates("fight_id")
            for event_id, event_rows in physical_order.groupby("event_id", sort=False):
                expected = list(range(len(event_rows)))
                observed_bout_order = sorted(event_rows["bout_order"].tolist())
                if observed_bout_order != expected:
                    raise ValueError(
                        f"raw bout_order is not contiguous for event {event_id}"
                    )
                if "source_card_index" in raw:
                    observed_source_order = sorted(
                        event_rows["source_card_index"].tolist()
                    )
                    if observed_source_order != expected:
                        raise ValueError(
                            f"raw source_card_index is not contiguous for event {event_id}"
                        )
                    event_size = len(event_rows)
                    if not (
                        event_rows["source_card_index"]
                        + event_rows["bout_order"]
                        == event_size - 1
                    ).all():
                        raise ValueError(
                            "raw source_card_index and bout_order are not inverse "
                            f"for event {event_id}"
                        )
        else:
            # UFCStats lists a card main-event first.  Existing snapshots did
            # not persist that position, but their stable row order still does;
            # reverse it so old same-night tournament rounds are chronological.
            first_positions = (
                raw.groupby(["event_id", "fight_id"], sort=False)["_source_position"]
                .min()
                .rename("first_position")
                .reset_index()
            )
            first_positions["source_card_index"] = first_positions.groupby(
                "event_id", sort=False
            )["first_position"].rank(method="dense").astype(int) - 1
            event_sizes = first_positions.groupby("event_id")["fight_id"].transform("size")
            first_positions["bout_order"] = event_sizes - 1 - first_positions["source_card_index"]
            order_lookup = first_positions.set_index(["event_id", "fight_id"])["bout_order"]
            raw["bout_order"] = [
                int(order_lookup.loc[(event_id, fight_id)])
                for event_id, fight_id in raw[["event_id", "fight_id"]].itertuples(index=False)
            ]
        return raw.sort_values(
            ["date", "event_id", "bout_order", "fight_id", "fighter_id"], kind="stable"
        )

    def _update_fight(self, rows: pd.DataFrame) -> None:
        first, second = rows.iloc[0], rows.iloc[1]
        first_id, second_id = first["fighter_id"], second["fighter_id"]
        first_state = self.states.setdefault(first_id, FighterState())
        second_state = self.states.setdefault(second_id, FighterState())
        first_score = self._score_for(first, rows)
        second_score = self._score_for(second, rows)
        division = self._division_key(first["division"])
        first_division = first_state.division_ratings.setdefault(
            division, dict.fromkeys(ELO_NAMES, 1500.0)
        )
        second_division = second_state.division_ratings.setdefault(
            division, dict.fromkeys(ELO_NAMES, 1500.0)
        )
        first_last_date = first_state.records[-1].date if first_state.records else None
        second_last_date = second_state.records[-1].date if second_state.records else None
        first_pre_rating = self._decayed_rating(
            first_state.ratings["elo_medium"], first_last_date, first["date"]
        )
        second_pre_rating = self._decayed_rating(
            second_state.ratings["elo_medium"], second_last_date, first["date"]
        )

        first_division_records = [
            record for record in first_state.records if record.division == division
        ]
        second_division_records = [
            record for record in second_state.records if record.division == division
        ]
        first_division_last = (
            first_division_records[-1].date if first_division_records else None
        )
        second_division_last = (
            second_division_records[-1].date if second_division_records else None
        )
        decision_weight = (
            0.75
            if str(first["method"]).upper() in {"S-DEC", "M-DEC"}
            else 1.0
        )
        for name, k_factor in zip(ELO_NAMES, ELO_K_FACTORS):
            first_rating = self._decayed_rating(
                first_state.ratings[name], first_last_date, first["date"]
            )
            second_rating = self._decayed_rating(
                second_state.ratings[name], second_last_date, first["date"]
            )
            first_division_rating = self._decayed_rating(
                first_division[name], first_division_last, first["date"]
            )
            second_division_rating = self._decayed_rating(
                second_division[name], second_division_last, first["date"]
            )
            change = 0.0
            division_change = 0.0
            if first_score is not None and second_score is not None:
                expected = self._rating_probability(first_rating, second_rating)
                change = decision_weight * k_factor * (first_score - expected)
                division_expected = self._rating_probability(
                    first_division_rating, second_division_rating
                )
                division_change = decision_weight * k_factor * (
                    first_score - division_expected
                )
            # Even a no-contest advances the rating's as-of clock. Without
            # this assignment, the next bout would decay an already-old value
            # from the NC date and silently skip part of the inactivity gap.
            first_state.ratings[name] = first_rating + change
            second_state.ratings[name] = second_rating - change
            first_division[name] = first_division_rating + division_change
            second_division[name] = second_division_rating - division_change

        def stats(row: pd.Series) -> dict[str, float]:
            if pd.isna(row["total_fight_time"]):
                return dict.fromkeys(self.state_count_stats, math.nan)
            return {
                column: _optional_number(row[column])
                for column in self.state_count_stats
            }

        fight_date = first["date"]
        seconds = max(_safe_number(first["total_fight_time"]), 0.0)
        first_state.records.append(
            FightRecord(
                date=fight_date,
                result=first_score,
                method=str(first["method"]),
                division=division,
                opponent_rating=second_pre_rating,
                seconds=seconds,
                own_stats=stats(first),
                opponent_stats=stats(second),
            )
        )
        second_state.records.append(
            FightRecord(
                date=fight_date,
                result=second_score,
                method=str(second["method"]),
                division=division,
                opponent_rating=first_pre_rating,
                seconds=seconds,
                own_stats=stats(second),
                opponent_stats=stats(first),
            )
        )

    def build(self, min_training_date: object | None = None) -> pd.DataFrame:
        """Build features and leave ``states`` positioned after all history."""
        raw = self._validate_and_prepare_raw()
        self.states = {}
        self.training_data = None
        self._replayed_through = None
        self.state_fingerprint_sha256 = self._state_source_fingerprint(raw)
        rows_out: list[dict[str, object]] = []
        min_date = (
            pd.to_datetime(min_training_date, errors="raise").normalize()
            if min_training_date is not None
            else None
        )

        # Replay bouts in their causal card order. Disjoint matchups are
        # unaffected by earlier results, while a later tournament round may
        # legitimately use a participant's earlier-round result.
        for (_fight_date, _event_id), event_rows in raw.groupby(
            ["date", "event_id"], sort=True
        ):
            fight_groups = [
                (fight_id, fight_rows)
                for fight_id, fight_rows in event_rows.groupby("fight_id", sort=False)
            ]
            fight_groups.sort(key=lambda item: int(item[1]["bout_order"].iloc[0]))
            for fight_id, fight_rows in fight_groups:
                ordered = fight_rows.sort_values("fighter_id", kind="stable")
                fighter, opponent = ordered.iloc[0], ordered.iloc[1]
                fight_date = fighter["date"]
                score = self._score_for(fighter, ordered)
                if bool(fighter["emit_training_target"]) and score in (0.0, 1.0) and (
                    min_date is None or fight_date >= min_date
                ):
                    feature_row = self._matchup_features_from_current_state(
                        fighter["fighter_id"],
                        opponent["fighter_id"],
                        fight_date,
                        fighter["division"],
                    ).iloc[0].to_dict()
                    feature_row.update(
                        {
                            "schema_version": MODEL_SCHEMA_VERSION,
                            "date": fight_date,
                            "event_id": fighter["event_id"],
                            "event_url": fighter["event_url"],
                            "fight_id": fight_id,
                            "fight_url": fighter["fight_url"],
                            "fighter_id": fighter["fighter_id"],
                            "fighter_url": fighter["fighter_url"],
                            "opponent_id": opponent["fighter_id"],
                            "opponent_url": opponent["fighter_url"],
                            "fighter": fighter["fighter"],
                            "opponent": opponent["fighter"],
                            "division": fighter["division"],
                            "bout_order": int(fighter["bout_order"]),
                            "label_method": fighter["method"],
                            "label_finish_round": fighter["round"],
                            "label_total_fight_seconds": fighter["total_fight_time"],
                            "label_time_format": fighter.get("time_format", ""),
                            "target": int(score),
                        }
                    )
                    rows_out.append(feature_row)
                self._update_fight(ordered)

        columns = [
            "schema_version", "date", "event_id", "event_url", "fight_id", "fight_url",
            "fighter_id", "fighter_url", "opponent_id", "opponent_url",
            "fighter", "opponent", "division", "bout_order", "label_method",
            "label_finish_round", "label_total_fight_seconds", "label_time_format",
            "target", *self.feature_columns,
        ]
        self.training_data = pd.DataFrame(rows_out, columns=columns).sort_values(
            list(PIT_SORT_COLUMNS), kind="stable"
        ).reset_index(drop=True)
        if self.training_data.empty:
            raise ValueError("No terminal W/L fights were available for model training")
        self._replayed_through = raw["date"].max()
        return self.training_data.copy()


def _expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            error += mask.mean() * abs(float(probability[mask].mean()) - float(y_true[mask].mean()))
    return float(error)


def _metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int | None]:
    truth = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    ties = np.isclose(probability, 0.5, rtol=0.0, atol=1e-12)
    correct = ((probability > 0.5).astype(int) == truth).astype(float)
    correct[ties] = 0.5
    return {
        "fights": int(len(truth)),
        "accuracy": float(correct.mean()),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(truth, probability)),
        "auc": (
            float(roc_auc_score(truth, probability))
            if len(np.unique(truth)) == 2 else None
        ),
        "ece_10_bin": _expected_calibration_error(truth, probability),
    }


def training_fingerprint(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
) -> str:
    """Hash the exact ordered training matrix and its stable lineage columns."""
    canonical = frame.copy()
    canonical["date"] = pd.to_datetime(canonical["date"], errors="raise")
    canonical = canonical.sort_values(list(PIT_SORT_COLUMNS), kind="stable")
    canonical[list(feature_columns)] = canonical[list(feature_columns)].apply(
        pd.to_numeric, errors="raise"
    ).round(12)
    fingerprint_columns = [
        "date", "fight_id", "fighter_id", "opponent_id", "target",
        *feature_columns,
    ]
    fingerprint_text = canonical[fingerprint_columns].to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.12f",
        lineterminator="\n",
    )
    return sha256(fingerprint_text.encode("utf-8")).hexdigest()


class TemporalFightPredictor:
    """Regularized, calibrated, exactly antisymmetric temporal predictor."""

    def __init__(
        self,
        training_data: pd.DataFrame,
        builder: PointInTimeDatasetBuilder,
        dh=None,
    ):
        self.point_in_time_data = training_data.sort_values(
            list(PIT_SORT_COLUMNS), kind="stable"
        ).reset_index(drop=True)
        parsed_dates = pd.to_datetime(self.point_in_time_data["date"], errors="raise")
        training_end = builder._replayed_through or parsed_dates.max()
        training_start = training_end - pd.DateOffset(years=10)
        self.training_data = self.point_in_time_data.loc[
            parsed_dates >= training_start
        ].reset_index(drop=True)
        self.builder = builder
        self.dh = dh
        self.feature_columns = list(builder.feature_columns)
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.model: LogisticRegression | None = None
        self.calibration_slope = 1.0
        self.best_c: float | None = None
        self.evaluation: dict[str, object] = {}
        self._artifact_scale: np.ndarray | None = None
        self._artifact_coefficients: np.ndarray | None = None
        self._loaded_artifact: dict[str, object] | None = None

    @staticmethod
    def _fit_pipeline(X: pd.DataFrame, y: pd.Series, c_value: float):
        if pd.Series(y).nunique(dropna=True) < 2:
            raise ValueError("Logistic training data must contain both W and L labels")
        imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
        imputed = imputer.fit_transform(X)
        scaler = StandardScaler(with_mean=False)
        scaled = scaler.fit_transform(imputed)
        model = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=c_value,
            fit_intercept=False,
            max_iter=30_000,
            random_state=48,
        )
        model.fit(scaled, y)
        return imputer, scaler, model

    @staticmethod
    def _pipeline_probability(pipeline, X: pd.DataFrame) -> np.ndarray:
        imputer, scaler, model = pipeline
        return model.predict_proba(scaler.transform(imputer.transform(X)))[:, 1]

    @staticmethod
    def _rolling_splits(dates: pd.Series, n_splits: int = 4):
        parsed = pd.to_datetime(dates).reset_index(drop=True)
        unique_dates = np.array(sorted(parsed.unique()))
        if len(unique_dates) < n_splits + 2:
            raise ValueError("Not enough distinct event dates for rolling validation")
        initial = max(1, len(unique_dates) // 2)
        validation_dates = unique_dates[initial:]
        chunks = [chunk for chunk in np.array_split(validation_dates, n_splits) if len(chunk)]
        for chunk in chunks:
            train_mask = parsed < chunk[0]
            test_mask = parsed.isin(chunk)
            if train_mask.any() and test_mask.any():
                yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)

    def _rolling_probabilities(
        self,
        frame: pd.DataFrame,
        c_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities: list[np.ndarray] = []
        truth: list[np.ndarray] = []
        for train_indices, test_indices in self._rolling_splits(frame["date"]):
            train = frame.iloc[train_indices]
            test = frame.iloc[test_indices]
            pipeline = self._fit_pipeline(
                train[self.feature_columns], train["target"], c_value
            )
            probabilities.append(
                self._pipeline_probability(pipeline, test[self.feature_columns])
            )
            truth.append(test["target"].to_numpy())
        return np.concatenate(truth), np.concatenate(probabilities)

    @staticmethod
    def _fit_symmetric_calibration_slope(y_true: np.ndarray, probability: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            raise ValueError("Calibration data must contain both W and L labels")
        probability = np.clip(probability, 1e-6, 1 - 1e-6)
        logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
        calibrator = LogisticRegression(
            solver="lbfgs", C=1_000_000.0, fit_intercept=False, max_iter=10_000
        ).fit(logits, y_true)
        # A positive slope preserves ordering and exact matchup symmetry.
        return float(np.clip(calibrator.coef_[0, 0], 0.25, 2.0))

    @staticmethod
    def _calibrate(probability: np.ndarray, slope: float) -> np.ndarray:
        probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
        logits = np.log(probability / (1.0 - probability))
        return 1.0 / (1.0 + np.exp(-np.clip(slope * logits, -709, 709)))

    def train(self) -> dict[str, object]:
        # This grid is part of the production artifact contract. Candidate
        # experiments must run separately rather than creating an artifact
        # that the production loader cannot reproduce.
        c_grid = REGULARIZATION_C_GRID
        frame = self.training_data.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if len(frame) < 100:
            raise ValueError(f"At least 100 fights are required to train, got {len(frame)}")

        row_cutoff = max(1, int(len(frame) * 0.8))
        holdout_date = frame.iloc[row_cutoff]["date"]
        development = frame[frame["date"] < holdout_date].reset_index(drop=True)
        holdout = frame[frame["date"] >= holdout_date].reset_index(drop=True)
        if development.empty or holdout.empty:
            raise ValueError("Temporal holdout requires at least two distinct fight dates")

        cv_scores: dict[str, float] = {}
        for c_value in c_grid:
            y_cv, p_cv = self._rolling_probabilities(development, float(c_value))
            cv_scores[str(c_value)] = float(log_loss(y_cv, p_cv, labels=[0, 1]))
        self.best_c = min((float(value) for value in c_grid), key=lambda value: cv_scores[str(value)])

        y_oof, p_oof = self._rolling_probabilities(development, self.best_c)
        evaluation_slope = self._fit_symmetric_calibration_slope(y_oof, p_oof)
        evaluation_pipeline = self._fit_pipeline(
            development[self.feature_columns], development["target"], self.best_c
        )
        raw_holdout_probability = self._pipeline_probability(
            evaluation_pipeline, holdout[self.feature_columns]
        )
        calibrated_holdout_probability = self._calibrate(
            raw_holdout_probability, evaluation_slope
        )
        elo_probability = 1.0 / (
            1.0
            + 10.0
            ** (-holdout["elo_medium_diff"].to_numpy(dtype=float) / 400.0)
        )

        # Re-estimate calibration from rolling predictions across all history,
        # then refit the selected feature contract on every eligible fight.
        y_all_oof, p_all_oof = self._rolling_probabilities(frame, self.best_c)
        self.calibration_slope = self._fit_symmetric_calibration_slope(y_all_oof, p_all_oof)
        self.imputer, self.scaler, self.model = self._fit_pipeline(
            frame[self.feature_columns], frame["target"], self.best_c
        )
        self._artifact_scale = np.asarray(self.scaler.scale_, dtype=float)
        self._artifact_coefficients = np.asarray(self.model.coef_[0], dtype=float)
        self.evaluation = {
            "holdout_start": holdout["date"].min().strftime("%Y-%m-%d"),
            "holdout_end": holdout["date"].max().strftime("%Y-%m-%d"),
            "development_fights": int(len(development)),
            "selected_c": self.best_c,
            "rolling_cv_log_loss_by_c": cv_scores,
            "evaluation_calibration_slope": evaluation_slope,
            "raw_model": _metrics(holdout["target"], raw_holdout_probability),
            "calibrated_model": _metrics(holdout["target"], calibrated_holdout_probability),
            "elo_only": _metrics(holdout["target"], elo_probability),
            "coin_flip": _metrics(holdout["target"], np.full(len(holdout), 0.5)),
        }
        self.evaluation["walk_forward"] = self.walk_forward_evaluation()
        print(
            "Point-in-time temporal holdout: "
            f"{self.evaluation['calibrated_model']['accuracy']:.3f} accuracy, "
            f"{self.evaluation['calibrated_model']['log_loss']:.3f} log loss, "
            f"{self.evaluation['calibrated_model']['brier']:.3f} Brier"
        )
        return self.evaluation.copy()

    def walk_forward_evaluation(
        self,
        years: tuple[int, ...] | None = None,
    ) -> dict[str, object]:
        """Nested expanding-year evaluation without consulting a future fold."""
        predictions, folds = self._walk_forward_predictions_and_folds(years)
        aggregate = _metrics(
            predictions["target"],
            predictions["model_probability"].to_numpy(dtype=float),
        )
        return {"folds": folds, "aggregate": aggregate}

    def walk_forward_predictions(
        self,
        years: tuple[int, ...] | None = None,
    ) -> pd.DataFrame:
        """Return lineage-bearing out-of-fold probabilities for whole years.

        Every probability is produced by a model whose training window ends
        before January 1 of that row's calendar year. Hyperparameter and
        calibration selection are nested inside that earlier window. This is
        the safe bridge for joining the current algorithm to historical market
        captures without applying today's fully fitted artifact retroactively.
        """

        predictions, _folds = self._walk_forward_predictions_and_folds(years)
        return predictions.copy()

    def _walk_forward_predictions_and_folds(
        self,
        years: tuple[int, ...] | None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        c_grid = REGULARIZATION_C_GRID
        frame = self.point_in_time_data.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        available_years = sorted(frame["date"].dt.year.unique())
        if years is None:
            years = tuple(available_years[-4:])
        prediction_frames: list[pd.DataFrame] = []
        folds: dict[str, object] = {}
        for year in years:
            test_start = pd.Timestamp(year=year, month=1, day=1)
            train_start = test_start - pd.DateOffset(years=10)
            train = frame[
                (frame["date"] >= train_start) & (frame["date"] < test_start)
            ].reset_index(drop=True)
            test = frame[frame["date"].dt.year == year].reset_index(drop=True)
            if len(train) < 500 or test.empty:
                continue
            scores: dict[float, float] = {}
            for c_value in c_grid:
                y_cv, p_cv = self._rolling_probabilities(train, float(c_value))
                scores[float(c_value)] = float(log_loss(y_cv, p_cv, labels=[0, 1]))
            selected_c = min(scores, key=scores.get)
            y_oof, p_oof = self._rolling_probabilities(train, selected_c)
            slope = self._fit_symmetric_calibration_slope(y_oof, p_oof)
            pipeline = self._fit_pipeline(
                train[self.feature_columns], train["target"], selected_c
            )
            probability = self._calibrate(
                self._pipeline_probability(pipeline, test[self.feature_columns]), slope
            )
            lineage_columns = [
                "date",
                "event_id",
                "fight_id",
                "fighter_id",
                "opponent_id",
                "fighter",
                "opponent",
                "target",
            ]
            prediction_frame = test[lineage_columns].copy()
            prediction_frame["evaluation_year"] = int(year)
            prediction_frame["training_start"] = train["date"].min().strftime(
                "%Y-%m-%d"
            )
            prediction_frame["training_through"] = train["date"].max().strftime(
                "%Y-%m-%d"
            )
            prediction_frame["selected_c"] = float(selected_c)
            prediction_frame["calibration_slope"] = float(slope)
            prediction_frame["model_probability"] = probability
            prediction_frames.append(prediction_frame)
            folds[str(year)] = {
                "train_fights": int(len(train)),
                "selected_c": float(selected_c),
                "calibration_slope": float(slope),
                **_metrics(test["target"], probability),
            }
        if not prediction_frames:
            raise ValueError("No eligible calendar-year folds for walk-forward evaluation")
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions = predictions.sort_values(
            ["date", "event_id", "fight_id"], kind="stable"
        ).reset_index(drop=True)
        return predictions, folds

    def probability(self, diff_row: pd.DataFrame) -> float:
        if self._artifact_scale is None or self._artifact_coefficients is None:
            raise RuntimeError("The temporal model must be trained before prediction")
        values = (
            diff_row[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        logits = (values / self._artifact_scale) @ self._artifact_coefficients
        raw = 1.0 / (1.0 + np.exp(-np.clip(logits, -709, 709)))
        return float(self._calibrate(raw, self.calibration_slope)[0])

    @staticmethod
    def probability_to_american_odds(probability: float) -> str:
        probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
        if probability >= 0.5:
            return str(-round(100.0 * probability / (1.0 - probability)))
        return f"+{round(100.0 * (1.0 - probability) / probability)}"

    def predict_upcoming_fights(
        self,
        prediction_history: pd.DataFrame,
        fighter_stats: pd.DataFrame,
        fights_list: list,
        card_date: str,
    ) -> pd.DataFrame:
        del fighter_stats  # identity/profile data is already frozen in the builder
        columns = list(prediction_history.columns)
        output = pd.DataFrame("", index=range(len(fights_list)), columns=columns)
        artifact = self.artifact()
        for required in (
            "fighter name", "opponent name", "date", "division",
            "event id", "event url", "fighter id", "opponent id",
            "model id", "model version", "model trained through", "model probability", "model status",
            "forecast probability", "forecast source", "forecast fighter odds",
            "forecast opponent odds",
            "fighter prior fights", "opponent prior fights",
        ):
            if required not in output:
                output[required] = ""

        for index, fight in enumerate(fights_list):
            fighter_name, opponent_name = fight[0], fight[1]
            division = fight[2] if len(fight) > 2 else "Unknown"
            fighter_url = fight[3] if len(fight) > 3 else None
            opponent_url = fight[4] if len(fight) > 4 else None
            event_url = fight[5] if len(fight) > 5 else None
            fighter_id = self.builder.resolve_fighter_id(fighter_name, fighter_url)
            opponent_id = self.builder.resolve_fighter_id(opponent_name, opponent_url)
            output.loc[
                index,
                [
                    "fighter name", "opponent name", "date", "division",
                    "event id", "event url",
                ],
            ] = [
                fighter_name,
                opponent_name,
                card_date,
                division,
                _identity_token(event_url) if event_url else "",
                str(event_url or ""),
            ]
            output.at[index, "model version"] = MODEL_VERSION
            output.at[index, "model id"] = artifact["model_id"]
            output.at[index, "model trained through"] = artifact["data_through"]
            if not fighter_id or not opponent_id or fighter_id == opponent_id:
                output.at[index, "model status"] = "abstain_unresolved_identity"
                output.at[index, "forecast source"] = "abstain_unresolved_identity"
                continue

            features = self.builder.matchup_features(
                fighter_id, opponent_id, card_date, division
            )
            probability = self.probability(features)
            fighter_fights = self.builder.history_count(fighter_id)
            opponent_fights = self.builder.history_count(opponent_id)
            output.at[index, "fighter id"] = fighter_id
            output.at[index, "opponent id"] = opponent_id
            output.at[index, "fighter prior fights"] = fighter_fights
            output.at[index, "opponent prior fights"] = opponent_fights
            output.at[index, "model probability"] = probability
            output.at[index, "predicted fighter odds"] = self.probability_to_american_odds(probability)
            output.at[index, "predicted opponent odds"] = self.probability_to_american_odds(1.0 - probability)
            output.at[index, "model status"] = (
                "model" if min(fighter_fights, opponent_fights) >= 2
                else "low_history"
            )
            output.at[index, "forecast probability"] = probability
            output.at[index, "forecast source"] = (
                "stats_model"
                if output.at[index, "model status"] == "model"
                else "stats_model_low_history"
            )
            output.at[index, "forecast fighter odds"] = output.at[
                index, "predicted fighter odds"
            ]
            output.at[index, "forecast opponent odds"] = output.at[
                index, "predicted opponent odds"
            ]
            print(
                f"predicting {fighter_name} vs {opponent_name}: "
                f"p={probability:.3f} ({output.at[index, 'model status']})"
            )
        return output

    def artifact(self) -> dict[str, object]:
        if self._loaded_artifact is not None:
            return json.loads(json.dumps(self._loaded_artifact))
        if (
            self.model is None
            or self.imputer is None
            or self.scaler is None
            or self._artifact_scale is None
            or self._artifact_coefficients is None
        ):
            raise RuntimeError("Train the model before exporting an artifact")
        if self.builder._replayed_through is None or not self.builder.state_fingerprint_sha256:
            raise RuntimeError("Point-in-time builder state was not fully replayed")
        frame = self.training_data
        source_data_through = self.builder._replayed_through.strftime("%Y-%m-%d")
        training_labels_through = pd.to_datetime(frame["date"]).max().strftime(
            "%Y-%m-%d"
        )
        artifact = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "model_type": "zero-intercept L2 logistic with symmetric logit calibration",
            "identity_contract": "UFCStats fighter_url and fight_url IDs",
            "point_in_time_contract": "features snapshot before each causal bout",
            # ``data_through`` remains the public compatibility field and now
            # means replayed source state. A trailing draw/NC may be later than
            # the newest binary training label.
            "data_through": source_data_through,
            "source_data_through": source_data_through,
            "training_labels_through": training_labels_through,
            "training_window_start": pd.to_datetime(frame["date"]).min().strftime("%Y-%m-%d"),
            "training_fights": int(len(frame)),
            "training_fingerprint_sha256": training_fingerprint(
                frame, self.feature_columns
            ),
            "state_fingerprint_sha256": self.builder.state_fingerprint_sha256,
            "feature_columns": self.feature_columns,
            "imputer": {"strategy": "constant", "fill_value": 0.0},
            "scaler_scale": [float(value) for value in self._artifact_scale],
            "coefficients": [float(value) for value in self._artifact_coefficients],
            "intercept": 0.0,
            "calibration_slope": float(self.calibration_slope),
            "selected_c": float(self.best_c),
            "regularization_c_grid": list(REGULARIZATION_C_GRID),
            "rating_config": {
                "initial_rating": 1500.0,
                "k_factors": list(ELO_K_FACTORS),
                "annual_regression_to_mean": 0.08,
                "split_majority_decision_weight": 0.75,
                "event_batch_updates": False,
                "sequential_bout_order_updates": True,
            },
            "temporal_evaluation": self.evaluation,
            "runtime_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
        canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
        artifact["model_id"] = sha256(canonical.encode("utf-8")).hexdigest()[:20]
        return artifact

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        builder: PointInTimeDatasetBuilder,
        dh=None,
    ) -> "TemporalFightPredictor":
        if (
            builder.training_data is None
            or builder.training_data.empty
            or builder._replayed_through is None
            or not builder.state_fingerprint_sha256
        ):
            raise ValueError(
                "Model artifacts require a fully built point-in-time replay state"
            )
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {
            "schema_version", "model_version", "model_id", "data_through",
            "source_data_through", "training_labels_through", "training_fights",
            "training_fingerprint_sha256", "state_fingerprint_sha256",
            "feature_columns", "scaler_scale", "coefficients", "intercept",
            "calibration_slope", "selected_c", "regularization_c_grid",
        }
        missing = sorted(required - set(artifact))
        if missing:
            raise ValueError(f"Model artifact is missing required fields: {missing}")
        if artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError(f"Unsupported model artifact schema: {artifact.get('schema_version')!r}")
        if artifact.get("model_version") != MODEL_VERSION:
            raise ValueError(f"Unsupported model version: {artifact.get('model_version')!r}")
        features = artifact.get("feature_columns")
        if features != list(builder.feature_columns):
            raise ValueError("Model artifact feature order does not match the point-in-time builder")
        scales = np.asarray(artifact.get("scaler_scale"), dtype=float)
        coefficients = np.asarray(artifact.get("coefficients"), dtype=float)
        expected_length = len(features)
        if scales.shape != (expected_length,) or coefficients.shape != (expected_length,):
            raise ValueError("Model artifact scaler/coefficient vector length is invalid")
        if (
            not np.isfinite(scales).all()
            or not np.isfinite(coefficients).all()
            or (scales <= 0).any()
        ):
            raise ValueError("Model artifact contains non-finite or non-positive parameters")
        if float(artifact.get("intercept", math.nan)) != 0.0:
            raise ValueError("Point-in-time model artifact must have a zero intercept")
        slope = float(artifact.get("calibration_slope", math.nan))
        if not math.isfinite(slope) or slope <= 0:
            raise ValueError("Model artifact calibration slope must be finite and positive")
        selected_c = float(artifact.get("selected_c", math.nan))
        if not math.isfinite(selected_c) or selected_c <= 0:
            raise ValueError("Model artifact selected_c must be finite and positive")
        try:
            artifact_c_grid = tuple(
                float(value) for value in artifact.get("regularization_c_grid", [])
            )
        except (TypeError, ValueError):
            raise ValueError("Model artifact regularization grid must be numeric")
        if artifact_c_grid != REGULARIZATION_C_GRID:
            raise ValueError(
                "Model artifact regularization grid is not supported by this code"
            )
        if selected_c not in artifact_c_grid:
            raise ValueError(
                "Model artifact selected_c is not part of its regularization grid"
            )
        supplied_model_id = artifact.get("model_id")
        unhashed = dict(artifact)
        unhashed.pop("model_id", None)
        canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected_model_id = sha256(canonical.encode("utf-8")).hexdigest()[:20]
        if supplied_model_id != expected_model_id:
            raise ValueError("Model artifact model_id does not match its contents")

        if builder.training_data is not None and not builder.training_data.empty:
            source = builder.training_data.copy()
            source["date"] = pd.to_datetime(source["date"], errors="raise")
            source = source.sort_values(
                list(PIT_SORT_COLUMNS), kind="stable"
            ).reset_index(drop=True)
            training_start = builder._replayed_through - pd.DateOffset(years=10)
            eligible = source.loc[source["date"] >= training_start].reset_index(
                drop=True
            )
            if len(eligible) != int(artifact.get("training_fights", -1)):
                raise ValueError(
                    "Model artifact training count does not match current point-in-time data"
                )
            if training_fingerprint(eligible, features) != artifact.get(
                "training_fingerprint_sha256"
            ):
                raise ValueError(
                    "Model artifact training fingerprint does not match current point-in-time data"
                )
            expected_label_date = eligible["date"].max().strftime("%Y-%m-%d")
            if artifact.get("training_labels_through") != expected_label_date:
                raise ValueError(
                    "Model artifact training label cutoff does not match current data"
                )
        if builder._replayed_through is not None:
            replayed_through = builder._replayed_through.strftime("%Y-%m-%d")
            if artifact.get("source_data_through") != replayed_through:
                raise ValueError(
                    "Model artifact source cutoff does not match current replay state"
                )
            if artifact.get("data_through") != replayed_through:
                raise ValueError("Model artifact public data cutoff is inconsistent")
            if artifact.get("state_fingerprint_sha256") != (
                builder.state_fingerprint_sha256
            ):
                raise ValueError(
                    "Model artifact state fingerprint does not match current replay state"
                )

        instance = cls.__new__(cls)
        instance.training_data = eligible.copy()
        instance.point_in_time_data = builder.training_data.copy()
        instance.builder = builder
        instance.dh = dh
        instance.feature_columns = features
        instance.imputer = None
        instance.scaler = None
        instance.model = None
        instance.calibration_slope = slope
        instance.best_c = selected_c
        instance.evaluation = artifact.get("temporal_evaluation", {})
        instance._artifact_scale = scales
        instance._artifact_coefficients = coefficients
        instance._loaded_artifact = artifact
        return instance

    def save_artifact(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.artifact(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as temporary:
                temporary.write(text)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
