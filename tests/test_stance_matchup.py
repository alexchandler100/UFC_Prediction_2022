import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_predictor import (  # noqa: E402
    PointInTimeDatasetBuilder,
    StanceMatchupDatasetBuilder,
)
from fight_predictor.stance_matchup import normalize_stance  # noqa: E402
from test_style_matchup import make_fight, make_profiles  # noqa: E402


class StanceMatchupFeatureTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(
            make_fight("f1", "e1", "2020-01-01", "a", "b")
            + make_fight("f2", "e2", "2021-01-01", "a", "c")
        )
        self.profiles = make_profiles("a", "b", "c")
        self.profiles.loc[self.profiles["name"] == "A", "stance"] = "Southpaw"
        self.profiles.loc[self.profiles["name"] == "C", "stance"] = "Switch"

    def test_challenger_adds_exactly_eight_features(self):
        production = PointInTimeDatasetBuilder(self.raw, self.profiles)
        challenger = StanceMatchupDatasetBuilder(self.raw, self.profiles)

        self.assertEqual(len(production.feature_columns), 82)
        self.assertEqual(len(challenger.feature_columns), 90)
        self.assertEqual(
            tuple(challenger.feature_columns[:82]), production.feature_columns
        )
        self.assertEqual(len(set(challenger.feature_columns)), 90)

    def test_every_stance_feature_is_antisymmetric(self):
        builder = StanceMatchupDatasetBuilder(self.raw, self.profiles)
        builder.build()
        date = pd.Timestamp("2022-01-01")

        forward = builder._matchup_features_from_current_state(
            "a", "b", date, "Lightweight"
        )
        reverse = builder._matchup_features_from_current_state(
            "b", "a", date, "Lightweight"
        )

        np.testing.assert_allclose(
            forward.to_numpy(dtype=float),
            -reverse.to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        )
        self.assertEqual(float(forward["stance_southpaw_diff"].iloc[0]), 1.0)
        self.assertTrue(
            np.isfinite(float(forward["open_stance_reach_matchup"].iloc[0]))
        )

    def test_switch_and_nonstandard_labels_are_not_declared_open(self):
        builder = StanceMatchupDatasetBuilder(self.raw, self.profiles)
        builder.build()
        date = pd.Timestamp("2022-01-01")

        switch = builder._matchup_features_from_current_state(
            "a", "c", date, "Lightweight"
        )
        self.assertEqual(float(switch["open_stance_reach_matchup"].iloc[0]), 0.0)
        self.assertEqual(normalize_stance("Open Stance"), "unknown")
        self.assertEqual(normalize_stance("Sideways"), "unknown")
        self.assertEqual(normalize_stance(None), "unknown")

    def test_appending_future_fight_does_not_change_prior_features(self):
        before = StanceMatchupDatasetBuilder(self.raw, self.profiles).build()
        future = pd.DataFrame(
            make_fight("f3", "e3", "2022-01-01", "b", "c")
        )
        after = StanceMatchupDatasetBuilder(
            pd.concat([self.raw, future], ignore_index=True), self.profiles
        ).build()

        pd.testing.assert_frame_equal(
            before,
            after[after["fight_id"].isin(before["fight_id"])].reset_index(drop=True),
            check_exact=True,
        )


if __name__ == "__main__":
    unittest.main()
