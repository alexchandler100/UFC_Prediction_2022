"""Causal round-level pace-retention challenger for winner prediction.

Only pairs of complete five-minute UFCStats rounds are compared.  The
fighter-specific changes are strongly shrunk toward no round effect and are
updated only after a bout has been emitted, preserving point-in-time order.
The frozen production feature contract is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ufc_round_data import validate_normalized_round_stats

from .point_in_time import PointInTimeDatasetBuilder


CARDIO_LATE_ROUNDS = (2, 3)
CARDIO_METRICS = (
    "sig_attempt_retention_log",
    "sig_accuracy_change",
    "opponent_sig_attempt_retention_log",
    "sig_defence_change",
    "control_share_change",
)
CARDIO_PRIOR_FIGHTS = 4.0
PACE_PSEUDO_ATTEMPTS = 10.0
ACCURACY_PRIOR_LANDED = 9.0
ACCURACY_PRIOR_ATTEMPTS = 20.0


@dataclass(frozen=True)
class CardioObservation:
    date: pd.Timestamp
    round_number: int
    values: dict[str, float]


class RoundCardioDatasetBuilder(PointInTimeDatasetBuilder):
    """Build the baseline plus 12 bounded round-cardio side features."""

    def __init__(
        self,
        raw_fights: pd.DataFrame,
        fighter_stats: pd.DataFrame,
        round_stats: pd.DataFrame,
        auxiliary_fights: pd.DataFrame | None = None,
    ):
        self.round_stats = self._prepare_round_stats(round_stats)
        self._rounds_by_fight = {
            str(fight_id): rows.copy()
            for fight_id, rows in self.round_stats.groupby("fight_id", sort=False)
            if rows["reconciliation_status"].astype(str).eq("matched").all()
            and pd.to_numeric(
                rows["reconciliation_issue_count"], errors="coerce"
            ).fillna(0).eq(0).all()
        }
        self._cardio_observations: dict[str, list[CardioObservation]] = {}
        super().__init__(raw_fights, fighter_stats, auxiliary_fights=auxiliary_fights)

    @staticmethod
    def _prepare_round_stats(round_stats: pd.DataFrame) -> pd.DataFrame:
        frame = round_stats.copy()
        validate_normalized_round_stats(frame)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        numeric = (
            "round",
            "round_seconds",
            "sig_strikes_landed",
            "sig_strikes_attempts",
            "control",
            "reconciliation_issue_count",
        )
        for column in numeric:
            source = frame[column]
            parsed = pd.to_numeric(source, errors="coerce")
            malformed = (
                parsed.isna()
                & source.notna()
                & source.astype(str).str.strip().ne("")
            )
            if malformed.any():
                raise ValueError(f"round {column} contains malformed values")
            if (parsed.dropna() < 0).any() or (
                ~np.isfinite(parsed.dropna().to_numpy(dtype=float))
            ).any():
                raise ValueError(f"round {column} must be finite and nonnegative")
            frame[column] = parsed
        invalid_control = (
            frame["control"].notna()
            & frame["round_seconds"].notna()
            & (frame["control"] > frame["round_seconds"])
        )
        if invalid_control.any():
            raise ValueError("round control exceeds round exposure")
        return frame

    @staticmethod
    def _smoothed_accuracy(landed: float, attempted: float) -> float:
        return (landed + ACCURACY_PRIOR_LANDED) / (
            attempted + ACCURACY_PRIOR_ATTEMPTS
        )

    @classmethod
    def _observation_values(
        cls,
        early: pd.Series,
        late: pd.Series,
        early_opponent: pd.Series,
        late_opponent: pd.Series,
    ) -> dict[str, float] | None:
        required = (
            early["sig_strikes_landed"],
            early["sig_strikes_attempts"],
            early["control"],
            late["sig_strikes_landed"],
            late["sig_strikes_attempts"],
            late["control"],
            early_opponent["sig_strikes_landed"],
            early_opponent["sig_strikes_attempts"],
            late_opponent["sig_strikes_landed"],
            late_opponent["sig_strikes_attempts"],
        )
        if any(pd.isna(value) or not math.isfinite(float(value)) for value in required):
            return None

        early_attempts = float(early["sig_strikes_attempts"])
        late_attempts = float(late["sig_strikes_attempts"])
        early_opponent_attempts = float(early_opponent["sig_strikes_attempts"])
        late_opponent_attempts = float(late_opponent["sig_strikes_attempts"])
        early_accuracy = cls._smoothed_accuracy(
            float(early["sig_strikes_landed"]), early_attempts
        )
        late_accuracy = cls._smoothed_accuracy(
            float(late["sig_strikes_landed"]), late_attempts
        )
        early_opponent_accuracy = cls._smoothed_accuracy(
            float(early_opponent["sig_strikes_landed"]), early_opponent_attempts
        )
        late_opponent_accuracy = cls._smoothed_accuracy(
            float(late_opponent["sig_strikes_landed"]), late_opponent_attempts
        )
        return {
            "sig_attempt_retention_log": math.log(
                (late_attempts + PACE_PSEUDO_ATTEMPTS)
                / (early_attempts + PACE_PSEUDO_ATTEMPTS)
            ),
            "sig_accuracy_change": late_accuracy - early_accuracy,
            "opponent_sig_attempt_retention_log": math.log(
                (late_opponent_attempts + PACE_PSEUDO_ATTEMPTS)
                / (early_opponent_attempts + PACE_PSEUDO_ATTEMPTS)
            ),
            "sig_defence_change": (
                (1.0 - late_opponent_accuracy)
                - (1.0 - early_opponent_accuracy)
            ),
            "control_share_change": (
                float(late["control"]) - float(early["control"])
            ) / 300.0,
        }

    def _record_completed_fight_rounds(self, fight_rows: pd.DataFrame) -> None:
        fight_id = str(fight_rows["fight_id"].iloc[0])
        rounds = self._rounds_by_fight.get(fight_id)
        if rounds is None:
            return
        fight_date = pd.Timestamp(fight_rows["date"].iloc[0]).normalize()
        if not rounds["date"].eq(fight_date).all():
            raise ValueError(f"round date does not match aggregate fight {fight_id}")
        participants = set(fight_rows["fighter_id"].astype(str))
        if set(rounds["fighter_id"].astype(str)) != participants:
            raise ValueError(
                f"round participants do not match aggregate fight {fight_id}"
            )

        for fighter_id in sorted(participants):
            fighter_rounds = rounds[rounds["fighter_id"].astype(str).eq(fighter_id)]
            opponent_rounds = rounds[rounds["opponent_id"].astype(str).eq(fighter_id)]
            early = fighter_rounds[fighter_rounds["round"].eq(1)]
            early_opponent = opponent_rounds[opponent_rounds["round"].eq(1)]
            if len(early) != 1 or len(early_opponent) != 1:
                continue
            early_row = early.iloc[0]
            early_opponent_row = early_opponent.iloc[0]
            if (
                float(early_row["round_seconds"]) != 300.0
                or float(early_opponent_row["round_seconds"]) != 300.0
            ):
                continue

            for round_number in CARDIO_LATE_ROUNDS:
                late = fighter_rounds[fighter_rounds["round"].eq(round_number)]
                late_opponent = opponent_rounds[
                    opponent_rounds["round"].eq(round_number)
                ]
                if len(late) != 1 or len(late_opponent) != 1:
                    continue
                late_row = late.iloc[0]
                late_opponent_row = late_opponent.iloc[0]
                if (
                    float(late_row["round_seconds"]) != 300.0
                    or float(late_opponent_row["round_seconds"]) != 300.0
                ):
                    continue
                values = self._observation_values(
                    early_row, late_row, early_opponent_row, late_opponent_row
                )
                if values is not None:
                    self._cardio_observations.setdefault(fighter_id, []).append(
                        CardioObservation(
                            date=fight_date,
                            round_number=round_number,
                            values=values,
                        )
                    )

    def _side_features(
        self,
        fighter_id: str,
        fight_date: pd.Timestamp,
        division: object,
    ) -> dict[str, float]:
        features = super()._side_features(fighter_id, fight_date, division)
        observations = self._cardio_observations.get(fighter_id, [])
        for round_number in CARDIO_LATE_ROUNDS:
            eligible = [
                item
                for item in observations
                if item.date <= fight_date and item.round_number == round_number
            ]
            count = len(eligible)
            prefix = f"cardio_r{round_number}"
            features[f"{prefix}_samples_log"] = math.log1p(count)
            for metric in CARDIO_METRICS:
                features[f"{prefix}_{metric}"] = (
                    sum(item.values[metric] for item in eligible)
                    / (count + CARDIO_PRIOR_FIGHTS)
                )
        return features

    def cardio_sample_count(
        self,
        fighter_id: str,
        as_of: object,
        round_number: int = 2,
        *,
        strictly_before: bool = True,
    ) -> int:
        """Return qualifying prior observations for coverage reporting."""

        timestamp = pd.to_datetime(as_of, errors="raise").normalize()
        return sum(
            item.round_number == round_number
            and (item.date < timestamp if strictly_before else item.date <= timestamp)
            for item in self._cardio_observations.get(str(fighter_id), [])
        )

    def _update_fight(self, rows: pd.DataFrame) -> None:
        super()._update_fight(rows)
        self._record_completed_fight_rounds(rows)

    def build(self, min_training_date: object | None = None) -> pd.DataFrame:
        self._cardio_observations = {}
        return super().build(min_training_date=min_training_date)
