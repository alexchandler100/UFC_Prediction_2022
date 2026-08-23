import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_predictor import (  # noqa: E402
    PointInTimeDatasetBuilder,
    StyleMatchupDatasetBuilder,
)
from evaluate_style_matchup_challenger import (  # noqa: E402
    event_block_difference_interval,
)


def make_profiles(*fighter_ids):
    return pd.DataFrame(
        [
            {
                "name": fighter_id.upper(),
                "height": "5' 10\"",
                "reach": '70"',
                "stance": "Orthodox",
                "dob": "Jan 01, 1990",
                "url": f"http://ufcstats.test/fighter-details/{fighter_id}",
            }
            for fighter_id in fighter_ids
        ]
    )


def make_fight(
    fight_id,
    event_id,
    fight_date,
    fighter_id,
    opponent_id,
    *,
    fighter_result="W",
    fighter_sig=24,
    opponent_sig=12,
):
    def stats(sig_landed, *, aggressive):
        attempts = 40
        if aggressive:
            target_landed = (16, 5, sig_landed - 21)
            position_landed = (18, 3, sig_landed - 21)
            target_attempts = (28, 8, 4)
            position_attempts = (30, 6, 4)
        else:
            target_landed = (8, 3, sig_landed - 11)
            position_landed = (9, 2, sig_landed - 11)
            target_attempts = (24, 10, 6)
            position_attempts = (25, 9, 6)
        values = {
            "knockdowns": int(aggressive),
            "sig_strikes_landed": sig_landed,
            "sig_strikes_attempts": attempts,
            "total_strikes_landed": sig_landed,
            "total_strikes_attempts": attempts,
            "takedowns_landed": int(not aggressive),
            "takedowns_attempts": 2,
            "sub_attempts": 0,
            "reversals": 0,
            "control": 30 if aggressive else 60,
        }
        for category, landed, attempted in zip(
            ("head", "body", "leg"), target_landed, target_attempts
        ):
            values[f"{category}_strikes_landed"] = landed
            values[f"{category}_strikes_attempts"] = attempted
        for category, landed, attempted in zip(
            ("distance", "clinch", "ground"),
            position_landed,
            position_attempts,
        ):
            values[f"{category}_strikes_landed"] = landed
            values[f"{category}_strikes_attempts"] = attempted
        return values

    common = {
        "date": fight_date,
        "fight_url": f"http://ufcstats.test/fight-details/{fight_id}",
        "event_url": f"http://ufcstats.test/event-details/{event_id}",
        "division": "Lightweight",
        "method": "U-DEC",
        "round": 3,
        "time": "5:00",
        "total_fight_time": 900,
        "source_card_index": 0,
        "bout_order": 0,
    }
    first = {
        **common,
        **stats(fighter_sig, aggressive=True),
        "fighter": fighter_id.upper(),
        "opponent": opponent_id.upper(),
        "fighter_url": f"http://ufcstats.test/fighter-details/{fighter_id}",
        "opponent_url": f"http://ufcstats.test/fighter-details/{opponent_id}",
        "result": fighter_result,
    }
    second = {
        **common,
        **stats(opponent_sig, aggressive=False),
        "fighter": opponent_id.upper(),
        "opponent": fighter_id.upper(),
        "fighter_url": f"http://ufcstats.test/fighter-details/{opponent_id}",
        "opponent_url": f"http://ufcstats.test/fighter-details/{fighter_id}",
        "result": "L" if fighter_result == "W" else "W",
    }
    return [first, second]


class StyleMatchupFeatureTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(
            make_fight("f1", "e1", "2020-01-01", "a", "b")
            + make_fight("f2", "e2", "2021-01-01", "a", "c")
        )
        self.profiles = make_profiles("a", "b", "c")

    def test_challenger_adds_30_features_without_changing_production_contract(self):
        production = PointInTimeDatasetBuilder(self.raw, self.profiles)
        challenger = StyleMatchupDatasetBuilder(self.raw, self.profiles)

        self.assertEqual(len(production.feature_columns), 82)
        self.assertEqual(len(challenger.feature_columns), 112)
        self.assertEqual(
            tuple(challenger.feature_columns[:82]), production.feature_columns
        )
        self.assertEqual(len(set(challenger.feature_columns)), 112)

    def test_every_challenger_feature_is_antisymmetric(self):
        builder = StyleMatchupDatasetBuilder(self.raw, self.profiles)
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
        self.assertNotEqual(
            float(forward["career_head_style_matchup"].iloc[0]), 0.0
        )

    def test_appending_a_future_fight_does_not_change_prior_style_features(self):
        before_builder = StyleMatchupDatasetBuilder(self.raw, self.profiles)
        before = before_builder.build()
        future = pd.DataFrame(
            make_fight("f3", "e3", "2022-01-01", "b", "c")
        )
        after_builder = StyleMatchupDatasetBuilder(
            pd.concat([self.raw, future], ignore_index=True), self.profiles
        )
        after = after_builder.build()

        pd.testing.assert_frame_equal(
            before,
            after[after["fight_id"].isin(before["fight_id"])].reset_index(drop=True),
            check_exact=True,
        )

    def test_rejects_a_broken_significant_strike_partition(self):
        corrupt = self.raw.copy()
        corrupt.loc[0, "head_strikes_attempts"] += 1
        with self.assertRaisesRegex(ValueError, "significant strikes do not equal"):
            StyleMatchupDatasetBuilder(corrupt, self.profiles).build()

    def test_event_block_comparison_is_paired_and_deterministic(self):
        frame = pd.DataFrame(
            {
                "event_id": ["one", "one", "two", "two"],
                "target": [1, 0, 1, 0],
                "candidate": [0.7, 0.3, 0.65, 0.35],
                "reference": [0.6, 0.4, 0.55, 0.45],
            }
        )
        first = event_block_difference_interval(
            frame, "candidate", "reference", seed=4, replicates=100
        )
        second = event_block_difference_interval(
            frame, "candidate", "reference", seed=4, replicates=100
        )
        self.assertEqual(first, second)
        self.assertLess(first["point_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
