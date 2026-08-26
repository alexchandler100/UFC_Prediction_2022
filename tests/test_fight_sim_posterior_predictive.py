import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fight_sim.domain import FighterStats
from fight_sim.posterior_predictive import (
    score_observation,
    validate_completed_fight,
    write_validation_report,
)


class PosteriorPredictiveScoreTests(unittest.TestCase):
    def test_discrete_percentile_tail_probability_and_crps(self):
        score = score_observation(
            statistic="red_knockdowns",
            label="Red knockdowns",
            observed=1.0,
            support=np.asarray([0.0, 1.0, 2.0]),
            mass=np.asarray([0.25, 0.5, 0.25]),
            unit="count",
            definition_alignment="exact",
            predictive_sample_size=4,
        )

        self.assertEqual(score["mean"], 1.0)
        self.assertEqual(score["mid_pit_percentile"], 0.5)
        self.assertEqual(score["two_sided_tail_probability"], 1.0)
        self.assertAlmostEqual(score["crps"], 0.125)
        self.assertTrue(score["central_intervals"]["p90"]["contains_observed"])

        beyond = score_observation(
            statistic="red_knockdowns",
            label="Red knockdowns",
            observed=3.0,
            support=np.asarray([0.0, 1.0, 2.0]),
            mass=np.asarray([0.25, 0.5, 0.25]),
            unit="count",
            definition_alignment="exact",
            predictive_sample_size=4,
        )
        self.assertEqual(beyond["two_sided_tail_probability"], 0.0)
        self.assertGreater(beyond["two_sided_tail_upper_95_when_zero"], 0.0)

    def test_phase_attempt_partition_invariant(self):
        stats = FighterStats(
            strike_attempts=3,
            significant_strike_attempts=3,
            distance_attempts=2,
            clinch_attempts=1,
        )
        self.assertEqual(stats.distance_attempts + stats.clinch_attempts, 3)
        with self.assertRaisesRegex(ValueError, "attempt partitions"):
            FighterStats(
                strike_attempts=3,
                significant_strike_attempts=3,
                distance_attempts=1,
                clinch_attempts=1,
            )


class CompletedFightValidationTests(unittest.TestCase):
    def test_extracts_observed_sides_and_writes_self_contained_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            (run / "specs.json").write_text(
                json.dumps(
                    {
                        "specs": [
                            {
                                "red": {"fighter_id": "red-id", "fighter_name": "Red"},
                                "blue": {"fighter_id": "blue-id", "fighter_name": "Blue"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (run / "aggregate.json").write_text(
                json.dumps(
                    {
                        "aggregate": {
                            "matchup_id": "matchup-1",
                            "total_paths": 4,
                            "bootstrap_members": 1,
                            "outcome_probabilities": {"blue_decision": 0.25},
                            "duration_bins": [
                                {"upper_seconds": 300, "count": 1},
                                {"upper_seconds": 900, "count": 3},
                            ],
                            "statistic_distributions": [
                                {
                                    "statistic": "red_significant_strikes",
                                    "counts": [
                                        {"value": 10, "count": 1},
                                        {"value": 20, "count": 3},
                                    ],
                                },
                                {
                                    "statistic": "blue_significant_strikes",
                                    "counts": [
                                        {"value": 20, "count": 2},
                                        {"value": 30, "count": 2},
                                    ],
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            observed = root / "observed.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-01-01",
                        "fight_url": "http://ufcstats.com/fight-details/fight-1",
                        "fighter_url": "http://ufcstats.com/fighter-details/red-id",
                        "fighter": "Red",
                        "opponent": "Blue",
                        "result": "L",
                        "method": "U-DEC",
                        "total_fight_time": 900,
                        "sig_strikes_landed": 20,
                    },
                    {
                        "date": "2026-01-01",
                        "fight_url": "http://ufcstats.com/fight-details/fight-1",
                        "fighter_url": "http://ufcstats.com/fighter-details/blue-id",
                        "fighter": "Blue",
                        "opponent": "Red",
                        "result": "W",
                        "method": "U-DEC",
                        "total_fight_time": 900,
                        "sig_strikes_landed": 30,
                    },
                ]
            ).to_csv(observed, index=False)

            report = validate_completed_fight(
                run, observed_path=observed, fight_id="fight-1"
            )

            self.assertEqual(report["actual_outcome"], "blue_decision")
            self.assertEqual(report["actual_outcome_probability"], 0.25)
            self.assertEqual(report["summary"]["scored_marginals"], 3)
            self.assertIn(
                "single_bootstrap_member_excludes_parameter_model_uncertainty",
                report["coverage_warnings"],
            )
            json_path, html_path = write_validation_report(
                report,
                json_path=root / "validation.json",
                html_path=root / "validation.html",
            )
            self.assertTrue(json_path.is_file())
            self.assertIn("Embedded authoritative JSON", html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
