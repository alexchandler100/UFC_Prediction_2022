import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_predictor import PointInTimeDatasetBuilder  # noqa: E402
from fight_predictor.round_cardio import RoundCardioDatasetBuilder  # noqa: E402
from test_style_matchup import make_fight, make_profiles  # noqa: E402
from ufc_round_data import ROUND_DATA_COLUMNS  # noqa: E402


def make_round_data(fight_id, event_id, date, fighter_id, opponent_id):
    rows = []
    profiles = {
        fighter_id: {
            1: (20, 40, 30),
            2: (16, 30, 20),
            3: (12, 24, 15),
        },
        opponent_id: {
            1: (15, 30, 20),
            2: (17, 34, 35),
            3: (18, 36, 40),
        },
    }
    for round_number in (1, 2, 3):
        for own, other in ((fighter_id, opponent_id), (opponent_id, fighter_id)):
            landed, attempted, control = profiles[own][round_number]
            row = dict.fromkeys(ROUND_DATA_COLUMNS)
            row.update(
                {
                    "schema_version": 1,
                    "round_stat_id": f"{fight_id}:{own}:r{round_number}",
                    "event_id": event_id,
                    "event_url": f"http://ufcstats.test/event-details/{event_id}",
                    "fight_id": fight_id,
                    "fight_url": f"http://ufcstats.test/fight-details/{fight_id}",
                    "date": date,
                    "source_card_index": 0,
                    "bout_order": 0,
                    "division": "Lightweight",
                    "time_format": "3 Rnd (5-5-5)",
                    "scheduled_rounds": 3,
                    "finish_round": 3,
                    "finish_time": "5:00",
                    "total_fight_seconds": 900,
                    "round": round_number,
                    "round_seconds": 300,
                    "fighter_id": own,
                    "fighter_url": f"http://ufcstats.test/fighter-details/{own}",
                    "fighter": own.upper(),
                    "opponent_id": other,
                    "opponent_url": f"http://ufcstats.test/fighter-details/{other}",
                    "opponent": other.upper(),
                    "result": "W" if own == fighter_id else "L",
                    "method": "U-DEC",
                    "knockdowns": 0,
                    "sig_strikes_landed": landed,
                    "sig_strikes_attempts": attempted,
                    "total_strikes_landed": landed,
                    "total_strikes_attempts": attempted,
                    "takedowns_landed": 0,
                    "takedowns_attempts": 0,
                    "sub_attempts": 0,
                    "reversals": 0,
                    "control": control,
                    "head_strikes_landed": landed,
                    "head_strikes_attempts": attempted,
                    "body_strikes_landed": 0,
                    "body_strikes_attempts": 0,
                    "leg_strikes_landed": 0,
                    "leg_strikes_attempts": 0,
                    "distance_strikes_landed": landed,
                    "distance_strikes_attempts": attempted,
                    "clinch_strikes_landed": 0,
                    "clinch_strikes_attempts": 0,
                    "ground_strikes_landed": 0,
                    "ground_strikes_attempts": 0,
                    "reconciliation_status": "matched",
                    "reconciliation_issue_count": 0,
                }
            )
            rows.append(row)
    return rows


class RoundCardioFeatureTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(
            make_fight("f1", "e1", "2020-01-01", "a", "b")
            + make_fight("f2", "e2", "2021-01-01", "a", "c")
        )
        self.profiles = make_profiles("a", "b", "c")
        self.rounds = pd.DataFrame(
            make_round_data("f1", "e1", "2020-01-01", "a", "b")
            + make_round_data("f2", "e2", "2021-01-01", "a", "c")
        )

    def test_challenger_adds_exactly_twelve_features(self):
        baseline = PointInTimeDatasetBuilder(self.raw, self.profiles)
        cardio = RoundCardioDatasetBuilder(self.raw, self.profiles, self.rounds)

        self.assertEqual(len(baseline.feature_columns), 82)
        self.assertEqual(len(cardio.feature_columns), 94)
        self.assertEqual(tuple(cardio.feature_columns[:82]), baseline.feature_columns)
        self.assertEqual(len(set(cardio.feature_columns)), 94)

    def test_features_use_only_rounds_completed_before_the_matchup(self):
        builder = RoundCardioDatasetBuilder(self.raw, self.profiles, self.rounds)
        points = builder.build()
        first = points[points["fight_id"].eq("f1")].iloc[0]
        second = points[points["fight_id"].eq("f2")].iloc[0]

        self.assertEqual(float(first["cardio_r2_samples_log_diff"]), 0.0)
        self.assertNotEqual(float(second["cardio_r2_samples_log_diff"]), 0.0)
        self.assertNotEqual(
            float(second["cardio_r2_sig_attempt_retention_log_diff"]), 0.0
        )

    def test_every_cardio_feature_is_antisymmetric(self):
        builder = RoundCardioDatasetBuilder(self.raw, self.profiles, self.rounds)
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

    def test_partial_late_round_is_not_used(self):
        rounds = self.rounds.copy()
        mask = rounds["fight_id"].eq("f1") & rounds["round"].eq(2)
        rounds.loc[mask, "round_seconds"] = 120
        builder = RoundCardioDatasetBuilder(self.raw, self.profiles, rounds)
        builder.build()

        self.assertEqual(builder.cardio_sample_count("a", "2021-01-01"), 0)
        self.assertEqual(builder.cardio_sample_count("b", "2021-01-01"), 0)

    def test_appending_future_raw_and_rounds_cannot_change_prior_features(self):
        before = RoundCardioDatasetBuilder(
            self.raw, self.profiles, self.rounds
        ).build()
        future_raw = pd.DataFrame(
            make_fight("f3", "e3", "2022-01-01", "b", "c")
        )
        future_rounds = pd.DataFrame(
            make_round_data("f3", "e3", "2022-01-01", "b", "c")
        )
        after = RoundCardioDatasetBuilder(
            pd.concat([self.raw, future_raw], ignore_index=True),
            self.profiles,
            pd.concat([self.rounds, future_rounds], ignore_index=True),
        ).build()

        pd.testing.assert_frame_equal(
            before,
            after[after["fight_id"].isin(before["fight_id"])].reset_index(drop=True),
            check_exact=True,
        )


if __name__ == "__main__":
    unittest.main()
