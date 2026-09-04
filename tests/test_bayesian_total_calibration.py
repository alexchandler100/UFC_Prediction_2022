from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayesian_total_calibration import (  # noqa: E402
    BayesianTotalCalibrator,
    fit_total_calibration,
    validate_total_bayesian_kelly_assessment,
    validate_total_calibration_artifact,
)


def _predictions() -> pd.DataFrame:
    rows = []
    for index in range(80):
        probability = 0.35 if index % 2 else 0.65
        target = int(index % 5 not in {0, 1})
        rows.append(
            {
                "event_date": pd.Timestamp("2020-01-01")
                + pd.Timedelta(days=7 * index),
                "event_id": f"event-{index:03d}",
                "fight_id": f"fight-{index:03d}",
                "line": 1.5,
                "model_probability": probability,
                "target": target,
            }
        )
    return pd.DataFrame(rows)


class BayesianTotalCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = fit_total_calibration(_predictions())
        cls.calibrator = BayesianTotalCalibrator(cls.artifact)

    def test_artifact_has_a_later_fight_check(self):
        artifact = validate_total_calibration_artifact(self.artifact)
        line = artifact["lines"]["1.5"]
        self.assertEqual(line["status"], "available")
        self.assertEqual(line["chronological_check"]["status"], "complete")
        self.assertGreater(line["chronological_check"]["holdout_fights"], 0)

    def test_over_and_under_use_complementary_probability_distributions(self):
        over = self.calibrator.assessment(0.62, "over", 1.5, 110)
        under = self.calibrator.assessment(0.62, "under", 1.5, 110)
        validate_total_bayesian_kelly_assessment(over)
        validate_total_bayesian_kelly_assessment(under)
        self.assertAlmostEqual(
            over["posterior_mean_probability"],
            1.0 - under["posterior_mean_probability"],
        )
        self.assertAlmostEqual(
            over["posterior_lower_probability"],
            1.0 - under["posterior_upper_probability"],
        )
        self.assertLessEqual(over["recommended_fraction"], 0.05)

    def test_artifact_tampering_is_rejected(self):
        changed = copy.deepcopy(self.artifact)
        changed["lines"]["1.5"]["posterior"]["slope_draws"][0] = 3.0
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_total_calibration_artifact(changed)


if __name__ == "__main__":
    unittest.main()
