"""Bounded stance-matchup challenger for causal winner-model experiments.

The production winner model deliberately keeps its frozen 82-feature contract.
This builder adds only three profile-stance indicators and five open-stance
interactions.  Every added value is antisymmetric under a fighter swap so the
zero-intercept model retains ``p(A, B) = 1 - p(B, A)``.
"""

from __future__ import annotations

import pandas as pd

from .point_in_time import PointInTimeDatasetBuilder, _identity_token


STANDARD_STANCES = frozenset({"orthodox", "southpaw", "switch"})
OPEN_STANCE_INTERACTION_KEYS = (
    "reach",
    "career_sig_accuracy",
    "career_sig_defence",
    "recent_3y_sig_accuracy",
    "recent_3y_sig_defence",
)


def normalize_stance(value: object) -> str:
    """Return a supported stance label or ``unknown``.

    UFCStats currently also contains a handful of ``Open Stance`` and
    ``Sideways`` labels.  Their semantics do not map reliably to the three
    standard profile categories, so the experiment does not guess.
    """

    normalized = str(value or "").strip().casefold()
    return normalized if normalized in STANDARD_STANCES else "unknown"


class StanceMatchupDatasetBuilder(PointInTimeDatasetBuilder):
    """Build the frozen baseline plus a predeclared eight-feature stance group."""

    def __init__(
        self,
        raw_fights: pd.DataFrame,
        fighter_stats: pd.DataFrame,
        auxiliary_fights: pd.DataFrame | None = None,
    ):
        required = {"url", "stance"}
        missing = required - set(fighter_stats.columns)
        if missing:
            raise ValueError(
                f"fighter_stats is missing stance columns: {sorted(missing)}"
            )
        self._stance_by_id: dict[str, str] = {}
        for row in fighter_stats[["url", "stance"]].to_dict("records"):
            fighter_id = _identity_token(row["url"])
            if fighter_id:
                self._stance_by_id[fighter_id] = normalize_stance(row.get("stance"))

        super().__init__(raw_fights, fighter_stats, auxiliary_fights=auxiliary_fights)
        self.interaction_feature_names = tuple(
            f"open_stance_{key}_matchup" for key in OPEN_STANCE_INTERACTION_KEYS
        )
        self.feature_columns = (
            *self.feature_columns,
            *self.interaction_feature_names,
        )

    def stance_for(self, fighter_id: str) -> str:
        """Expose the normalized profile category for coverage reporting."""

        return self._stance_by_id.get(fighter_id, "unknown")

    def _side_features(
        self,
        fighter_id: str,
        fight_date: pd.Timestamp,
        division: object,
    ) -> dict[str, float]:
        features = super()._side_features(fighter_id, fight_date, division)
        stance = self.stance_for(fighter_id)
        features["stance_southpaw"] = float(stance == "southpaw")
        features["stance_switch"] = float(stance == "switch")
        features["stance_known"] = float(stance in STANDARD_STANCES)
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

        stances = {self.stance_for(fighter_id), self.stance_for(opponent_id)}
        open_stance = float(stances == {"orthodox", "southpaw"})
        for key in OPEN_STANCE_INTERACTION_KEYS:
            values[f"open_stance_{key}_matchup"] = open_stance * (
                fighter[key] - opponent[key]
            )
        return pd.DataFrame([values], columns=self.feature_columns)
