from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fight_predictor.dynamic_bayes import (  # noqa: E402
    DynamicBayesConfig,
    coefficient_prior_scales,
    dynamic_bayes_predict,
    without_elo_features,
)
from fight_predictor.hierarchical_bayes import (  # noqa: E402
    CoefficientBayesConfig,
    HierarchicalBayesConfig,
    coefficient_bayes_predict,
    hierarchical_bayes_predict,
)


class DynamicBayesianFightModelTests(unittest.TestCase):
    @staticmethod
    def _training() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        matchups = (
            ("a", "b", 0.8, 1),
            ("c", "d", -0.7, 0),
            ("a", "c", 0.4, 1),
            ("b", "d", -0.3, 0),
            ("a", "d", 0.6, 1),
            ("b", "c", -0.5, 0),
        )
        for index, (fighter, opponent, feature, target) in enumerate(matchups):
            rows.append(
                {
                    "date": pd.Timestamp("2018-01-01")
                    + pd.Timedelta(days=180 * index),
                    "event_id": f"event-{index}",
                    "fight_id": f"fight-{index}",
                    "bout_order": 1,
                    "fighter_id": fighter,
                    "opponent_id": opponent,
                    "strength_diff": feature,
                    "target": target,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _config() -> DynamicBayesConfig:
        return DynamicBayesConfig(
            burn_in=20,
            posterior_draws=20,
            chains=2,
            seed=29,
        )

    def test_predictions_are_deterministic_and_swap_symmetric(self):
        prediction = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2021-01-01"),
                    "fighter_id": "a",
                    "opponent_id": "b",
                    "strength_diff": 0.7,
                },
                {
                    "date": pd.Timestamp("2021-01-01"),
                    "fighter_id": "b",
                    "opponent_id": "a",
                    "strength_diff": -0.7,
                },
            ]
        )
        first = dynamic_bayes_predict(
            self._training(), prediction, ["strength_diff"], config=self._config()
        )
        second = dynamic_bayes_predict(
            self._training(), prediction, ["strength_diff"], config=self._config()
        )
        np.testing.assert_allclose(first.probability, second.probability, atol=0.0)
        self.assertAlmostEqual(first.probability[0] + first.probability[1], 1.0)
        self.assertTrue(np.all(first.lower_probability <= first.probability))
        self.assertTrue(np.all(first.probability <= first.upper_probability))
        self.assertEqual(first.diagnostics["fighter_appearance_states"], 12)
        self.assertEqual(first.diagnostics["total_retained_draws"], 40)

    def test_more_inactivity_adds_uncertainty_and_moves_toward_even(self):
        prediction = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2021-01-01"),
                    "fighter_id": "a",
                    "opponent_id": "b",
                    "strength_diff": 0.7,
                },
                {
                    "date": pd.Timestamp("2026-01-01"),
                    "fighter_id": "a",
                    "opponent_id": "b",
                    "strength_diff": 0.7,
                },
            ]
        )
        result = dynamic_bayes_predict(
            self._training(), prediction, ["strength_diff"], config=self._config()
        )
        self.assertLessEqual(
            abs(float(result.probability[1]) - 0.5),
            abs(float(result.probability[0]) - 0.5),
        )

    def test_prediction_labels_are_not_used_and_past_dates_are_rejected(self):
        prediction = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2021-01-01"),
                    "fighter_id": "a",
                    "opponent_id": "b",
                    "strength_diff": 0.7,
                    "target": 0,
                }
            ]
        )
        changed = prediction.copy()
        changed["target"] = 1
        first = dynamic_bayes_predict(
            self._training(), prediction, ["strength_diff"], config=self._config()
        )
        second = dynamic_bayes_predict(
            self._training(), changed, ["strength_diff"], config=self._config()
        )
        np.testing.assert_allclose(first.probability, second.probability, atol=0.0)

        too_early = prediction.copy()
        too_early["date"] = pd.Timestamp("2017-01-01")
        with self.assertRaisesRegex(ValueError, "precedes"):
            dynamic_bayes_predict(
                self._training(), too_early, ["strength_diff"], config=self._config()
            )

    def test_feature_and_prior_variants_are_predeclared(self):
        features = (
            "elo_slow_diff",
            "division_elo_fast_diff",
            "rating_uncertainty_diff",
            "age_diff",
            "career_win_rate_diff",
            "career_sig_landed_per15_diff",
        )
        self.assertEqual(
            without_elo_features(features),
            (
                "age_diff",
                "career_win_rate_diff",
                "career_sig_landed_per15_diff",
            ),
        )
        grouped = coefficient_prior_scales(
            without_elo_features(features), base_scale=0.35, grouped=True
        )
        np.testing.assert_allclose(grouped, [0.25, 0.20, 0.15])

    def test_coefficient_only_and_grouped_static_models_preserve_symmetry(self):
        prediction = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2021-01-01"),
                    "fighter_id": "a",
                    "opponent_id": "b",
                    "strength_diff": 0.7,
                },
                {
                    "date": pd.Timestamp("2021-01-01"),
                    "fighter_id": "b",
                    "opponent_id": "a",
                    "strength_diff": -0.7,
                },
            ]
        )
        coefficient = coefficient_bayes_predict(
            self._training(),
            prediction,
            ["strength_diff"],
            config=CoefficientBayesConfig(
                burn_in=20,
                posterior_draws=20,
                chains=2,
                grouped_coefficient_priors=True,
                seed=31,
            ),
        )
        static = hierarchical_bayes_predict(
            self._training(),
            prediction,
            ["strength_diff"],
            config=HierarchicalBayesConfig(
                burn_in=20,
                posterior_draws=20,
                chains=2,
                grouped_coefficient_priors=True,
                seed=33,
            ),
        )
        self.assertAlmostEqual(
            coefficient.probability[0] + coefficient.probability[1], 1.0
        )
        self.assertAlmostEqual(static.probability[0] + static.probability[1], 1.0)
        self.assertEqual(
            coefficient.diagnostics["model"], "coefficient_only_bayesian_probit"
        )


if __name__ == "__main__":
    unittest.main()
