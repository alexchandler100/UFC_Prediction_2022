"""Causal style-matchup feature challenger for winner prediction.

This module intentionally does not alter the production feature contract.  It
adds one predeclared group of UFCStats target/position tendencies and nonlinear
offense-versus-vulnerability interactions for a temporal challenger test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .point_in_time import COUNT_STATS, FightRecord, FighterState, PointInTimeDatasetBuilder


STYLE_CATEGORIES = (
    "head",
    "body",
    "leg",
    "distance",
    "clinch",
    "ground",
)
STYLE_COUNT_STATS = tuple(
    stat
    for category in STYLE_CATEGORIES
    for stat in (
        f"{category}_strikes_landed",
        f"{category}_strikes_attempts",
    )
)
STYLE_LANDED_ATTEMPTED_PAIRS = tuple(
    (f"{category}_strikes_landed", f"{category}_strikes_attempts")
    for category in STYLE_CATEGORIES
)

TARGET_SHARE_PRIORS = {"head": 0.70, "body": 0.20, "leg": 0.10}
POSITION_SHARE_PRIORS = {"distance": 0.75, "clinch": 0.15, "ground": 0.10}
SHARE_PRIOR_ATTEMPTS = 30.0


class StyleMatchupDatasetBuilder(PointInTimeDatasetBuilder):
    """Build the frozen baseline plus a bounded 30-feature style group."""

    STATE_COUNT_STATS = (*COUNT_STATS, *STYLE_COUNT_STATS)
    LANDED_ATTEMPTED_PAIRS = (
        *PointInTimeDatasetBuilder.LANDED_ATTEMPTED_PAIRS,
        *STYLE_LANDED_ATTEMPTED_PAIRS,
    )

    AGGREGATE_INTERACTION_KEYS = (
        "sig_pace_vs_absorption",
        "sig_accuracy_vs_defence",
        "td_pace_vs_absorption",
        "td_accuracy_vs_defence",
        "control_vs_control_absorption",
        "power_vs_knockdown_absorption",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interaction_feature_names = tuple(
            [
                f"{window}_{key}_matchup"
                for window in ("career", "recent_3y")
                for key in self.AGGREGATE_INTERACTION_KEYS
            ]
            + [f"career_{category}_style_matchup" for category in STYLE_CATEGORIES]
        )
        self.feature_columns = (
            *self.feature_columns,
            *self.interaction_feature_names,
        )

    @staticmethod
    def _records_as_of(
        state: FighterState,
        fight_date: pd.Timestamp,
    ) -> list[FightRecord]:
        return [record for record in state.records if record.date <= fight_date]

    @staticmethod
    def _smoothed_attempt_shares(
        records: list[FightRecord],
        side: str,
        priors: dict[str, float],
    ) -> dict[str, float]:
        categories = tuple(priors)
        totals = dict.fromkeys(categories, 0.0)
        for record in records:
            stats = getattr(record, side)
            values = [stats[f"{category}_strikes_attempts"] for category in categories]
            if not all(math.isfinite(value) for value in values):
                continue
            for category, value in zip(categories, values):
                totals[category] += value
        denominator = sum(totals.values()) + SHARE_PRIOR_ATTEMPTS
        return {
            category: (
                totals[category] + priors[category] * SHARE_PRIOR_ATTEMPTS
            )
            / denominator
            for category in categories
        }

    @staticmethod
    def _landed_per15(
        records: list[FightRecord],
        side: str,
        category: str,
    ) -> float:
        values_and_seconds = [
            (
                getattr(record, side)[f"{category}_strikes_landed"],
                max(record.seconds, 0.0),
            )
            for record in records
            if math.isfinite(
                getattr(record, side)[f"{category}_strikes_landed"]
            )
        ]
        total = sum(value for value, _seconds in values_and_seconds)
        exposure = sum(seconds for _value, seconds in values_and_seconds) / 900.0 + 1.0
        return total / exposure

    def _side_features(
        self,
        fighter_id: str,
        fight_date: pd.Timestamp,
        division: object,
    ) -> dict[str, float]:
        features = super()._side_features(fighter_id, fight_date, division)
        state = self.states.get(fighter_id, FighterState())
        records = self._records_as_of(state, fight_date)
        own_target = self._smoothed_attempt_shares(
            records, "own_stats", TARGET_SHARE_PRIORS
        )
        absorbed_target = self._smoothed_attempt_shares(
            records, "opponent_stats", TARGET_SHARE_PRIORS
        )
        own_position = self._smoothed_attempt_shares(
            records, "own_stats", POSITION_SHARE_PRIORS
        )
        absorbed_position = self._smoothed_attempt_shares(
            records, "opponent_stats", POSITION_SHARE_PRIORS
        )
        for category in STYLE_CATEGORIES:
            own = own_target if category in TARGET_SHARE_PRIORS else own_position
            absorbed = (
                absorbed_target
                if category in TARGET_SHARE_PRIORS
                else absorbed_position
            )
            features[f"career_{category}_attempt_share"] = own[category]
            features[f"career_{category}_absorbed_attempt_share"] = absorbed[
                category
            ]
        return features

    @staticmethod
    def _cross_product(
        fighter: dict[str, float],
        opponent: dict[str, float],
        offense_key: str,
        vulnerability_key: str,
    ) -> float:
        return (
            fighter[offense_key] * opponent[vulnerability_key]
            - opponent[offense_key] * fighter[vulnerability_key]
        )

    def _aggregate_interactions(
        self,
        fighter: dict[str, float],
        opponent: dict[str, float],
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for prefix in ("career", "recent_3y"):
            values[f"{prefix}_sig_pace_vs_absorption_matchup"] = self._cross_product(
                fighter,
                opponent,
                f"{prefix}_sig_landed_per15",
                f"{prefix}_sig_absorbed_per15",
            )
            values[f"{prefix}_sig_accuracy_vs_defence_matchup"] = (
                fighter[f"{prefix}_sig_accuracy"]
                * (1.0 - opponent[f"{prefix}_sig_defence"])
                - opponent[f"{prefix}_sig_accuracy"]
                * (1.0 - fighter[f"{prefix}_sig_defence"])
            )
            values[f"{prefix}_td_pace_vs_absorption_matchup"] = self._cross_product(
                fighter,
                opponent,
                f"{prefix}_td_landed_per15",
                f"{prefix}_td_absorbed_per15",
            )
            values[f"{prefix}_td_accuracy_vs_defence_matchup"] = (
                fighter[f"{prefix}_td_accuracy"]
                * (1.0 - opponent[f"{prefix}_td_defence"])
                - opponent[f"{prefix}_td_accuracy"]
                * (1.0 - fighter[f"{prefix}_td_defence"])
            )
            values[
                f"{prefix}_control_vs_control_absorption_matchup"
            ] = self._cross_product(
                fighter,
                opponent,
                f"{prefix}_control_per15",
                f"{prefix}_control_absorbed_per15",
            )
            values[
                f"{prefix}_power_vs_knockdown_absorption_matchup"
            ] = self._cross_product(
                fighter,
                opponent,
                f"{prefix}_knockdowns_per15",
                f"{prefix}_knockdowns_absorbed_per15",
            )
        return values

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
        values.update(self._aggregate_interactions(fighter, opponent))

        fighter_records = self._records_as_of(
            self.states.get(fighter_id, FighterState()), timestamp
        )
        opponent_records = self._records_as_of(
            self.states.get(opponent_id, FighterState()), timestamp
        )
        for category in STYLE_CATEGORIES:
            fighter_share = fighter[f"career_{category}_attempt_share"]
            opponent_share = opponent[f"career_{category}_attempt_share"]
            opponent_absorption = self._landed_per15(
                opponent_records, "opponent_stats", category
            )
            fighter_absorption = self._landed_per15(
                fighter_records, "opponent_stats", category
            )
            values[f"career_{category}_style_matchup"] = (
                fighter_share * opponent_absorption
                - opponent_share * fighter_absorption
            )
        return pd.DataFrame([values], columns=self.feature_columns)

    def _validate_and_prepare_raw(self) -> pd.DataFrame:
        raw = super()._validate_and_prepare_raw()
        partition_groups = (
            ("head", "body", "leg"),
            ("distance", "clinch", "ground"),
        )
        for suffix in ("landed", "attempts"):
            significant = raw[f"sig_strikes_{suffix}"]
            for categories in partition_groups:
                components = raw[
                    [f"{category}_strikes_{suffix}" for category in categories]
                ]
                comparable = significant.notna() & components.notna().all(axis=1)
                component_sum = components.sum(axis=1)
                invalid = comparable & ~np.isclose(
                    significant.to_numpy(dtype=float),
                    component_sum.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1e-9,
                )
                if invalid.any():
                    joined = "/".join(categories)
                    raise ValueError(
                        f"raw significant strikes do not equal {joined} {suffix} partition"
                    )
        return raw

