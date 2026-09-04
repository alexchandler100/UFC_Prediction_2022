import copy
from hashlib import sha256
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_tracker.bayesian_kelly import (  # noqa: E402
    BayesianKellyCalibrator,
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_REPLAY_PATH,
    expected_log_growth,
    validate_market_calibration_artifact,
)


class BayesianKellyTests(unittest.TestCase):
    def setUp(self):
        self.calibrator = BayesianKellyCalibrator.load()

    def test_frozen_artifact_is_valid_and_matches_its_source_file(self):
        artifact = validate_market_calibration_artifact(
            json.loads(DEFAULT_ARTIFACT_PATH.read_text(encoding="utf-8"))
        )
        self.assertEqual(artifact["training_fights"], 503)
        self.assertEqual(artifact["training_events"], 58)
        self.assertEqual(
            artifact["source_file_sha256"],
            sha256(DEFAULT_REPLAY_PATH.read_bytes()).hexdigest(),
        )
        check = artifact["chronological_check"]
        self.assertEqual(check["development_fights"], 363)
        self.assertEqual(check["holdout_fights"], 140)
        self.assertLess(check["log_loss_change"], 0.0)
        self.assertLess(check["brier_change"], 0.0)

    def test_expected_log_kelly_uses_only_the_posterior_mean(self):
        probabilities = (0.42, 0.54, 0.67, 0.73)
        fraction = 0.08
        averaged_growth = sum(
            expected_log_growth(fraction, probability, 140)
            for probability in probabilities
        ) / len(probabilities)
        growth_at_mean = expected_log_growth(
            fraction, sum(probabilities) / len(probabilities), 140
        )
        self.assertAlmostEqual(averaged_growth, growth_at_mean, places=14)

    def test_robust_assessment_uses_lower_chance_and_cap(self):
        result = self.calibrator.assessment(0.60, 110)
        self.assertEqual(result["status"], "available")
        self.assertLess(
            result["posterior_lower_probability"],
            result["posterior_mean_probability"],
        )
        self.assertLessEqual(
            result["robust_uncapped_kelly_fraction"],
            result["posterior_mean_full_kelly_fraction"],
        )
        self.assertEqual(result["recommended_fraction"], 0.05)
        self.assertTrue(result["cap_applied"])

    def test_probability_calibration_is_symmetric_when_fighters_swap(self):
        first = self.calibrator.assessment(0.60, 100)
        swapped = self.calibrator.assessment(0.40, 100)
        self.assertAlmostEqual(
            first["posterior_mean_probability"],
            1.0 - swapped["posterior_mean_probability"],
        )
        self.assertAlmostEqual(
            first["posterior_lower_probability"],
            1.0 - swapped["posterior_upper_probability"],
        )

    def test_artifact_tampering_is_rejected(self):
        changed = copy.deepcopy(self.calibrator.artifact)
        changed["posterior"]["slope_draws"][0] = 3.0
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_market_calibration_artifact(changed)


if __name__ == "__main__":
    unittest.main()
