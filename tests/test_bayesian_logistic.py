from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fight_predictor.bayesian_logistic import (  # noqa: E402
    BayesianLogisticConfig,
    bayesian_logistic_predict,
    feature_group,
    feature_groups,
)


class BayesianLogisticTests(unittest.TestCase):
    @staticmethod
    def _training() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        matchups = (
            ("a", "b", 0.9, 0.3, 1),
            ("b", "a", -0.9, -0.3, 0),
            ("c", "d", -0.7, 0.2, 0),
            ("d", "c", 0.7, -0.2, 1),
            ("a", "c", 0.5, 0.4, 1),
            ("c", "a", -0.5, -0.4, 0),
        )
        for _ in range(4):
            for fighter, opponent, rating, striking, target in matchups:
                rows.append(
                    {
                        "fighter_id": fighter,
                        "opponent_id": opponent,
                        "elo_slow_diff": rating,
                        "career_sig_accuracy_diff": striking,
                        "target": target,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _prediction() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fighter_id": "a",
                    "opponent_id": "b",
                    "elo_slow_diff": 0.8,
                    "career_sig_accuracy_diff": 0.3,
                },
                {
                    "fighter_id": "b",
                    "opponent_id": "a",
                    "elo_slow_diff": -0.8,
                    "career_sig_accuracy_diff": -0.3,
                },
            ]
        )

    @staticmethod
    def _config() -> BayesianLogisticConfig:
        return BayesianLogisticConfig(
            burn_in=40,
            posterior_draws=40,
            chains=2,
            variance_prior_shape=3.0,
            variance_prior_scale=0.08,
            seed=37,
        )

    def test_prediction_is_deterministic_and_exactly_swap_symmetric(self):
        features = ["elo_slow_diff", "career_sig_accuracy_diff"]
        first = bayesian_logistic_predict(
            self._training(), self._prediction(), features, config=self._config()
        )
        second = bayesian_logistic_predict(
            self._training(), self._prediction(), features, config=self._config()
        )
        np.testing.assert_allclose(first.probability, second.probability, atol=0.0)
        self.assertAlmostEqual(first.probability[0] + first.probability[1], 1.0)
        self.assertTrue(np.all(first.lower_probability <= first.probability))
        self.assertTrue(np.all(first.probability <= first.upper_probability))
        self.assertEqual(first.diagnostics["total_retained_draws"], 80)
        self.assertEqual(
            first.diagnostics["sampler"],
            "Laplace-preconditioned Hamiltonian Monte Carlo",
        )
        for chain in first.diagnostics["sampler_chains"]:
            self.assertGreater(chain["retained_acceptance_rate"], 0.0)
            self.assertLessEqual(chain["retained_acceptance_rate"], 1.0)
        self.assertTrue(
            np.isfinite(
                first.diagnostics["mean_lag_one_probability_autocorrelation"]
            )
        )

    def test_prediction_labels_are_never_used(self):
        features = ["elo_slow_diff", "career_sig_accuracy_diff"]
        first_frame = self._prediction()
        second_frame = self._prediction()
        first_frame["target"] = [0, 0]
        second_frame["target"] = [1, 1]
        first = bayesian_logistic_predict(
            self._training(), first_frame, features, config=self._config()
        )
        second = bayesian_logistic_predict(
            self._training(), second_frame, features, config=self._config()
        )
        np.testing.assert_allclose(first.probability, second.probability, atol=0.0)

    def test_group_scales_are_sampled_and_reported(self):
        result = bayesian_logistic_predict(
            self._training(),
            self._prediction(),
            ["elo_slow_diff", "career_sig_accuracy_diff"],
            config=self._config(),
        )
        scales = result.diagnostics["posterior_group_coefficient_scales"]
        self.assertEqual(set(scales), {"rating", "striking"})
        for values in scales.values():
            self.assertGreater(values["mean"], 0.0)
            self.assertLessEqual(values["q05"], values["mean"])
            self.assertLessEqual(values["mean"], values["q95"])
            self.assertEqual(values["feature_count"], 1)

    def test_feature_group_contract_is_complete_and_global_mode_is_available(self):
        features = (
            "elo_slow_diff",
            "age_diff",
            "days_since_fight_log_diff",
            "career_win_rate_diff",
            "career_sig_landed_per15_diff",
            "career_td_landed_per15_diff",
        )
        self.assertEqual(
            tuple(feature_group(value) for value in features),
            (
                "rating",
                "physical",
                "activity_experience",
                "record_results",
                "striking",
                "grappling",
            ),
        )
        self.assertEqual(
            feature_groups(features, grouped=False),
            tuple("all_features" for _ in features),
        )

    def test_invalid_hyperprior_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite mean"):
            BayesianLogisticConfig(variance_prior_shape=1.0)
        with self.assertRaisesRegex(ValueError, "acceptance target"):
            BayesianLogisticConfig(hmc_target_acceptance=0.4)


if __name__ == "__main__":
    unittest.main()
