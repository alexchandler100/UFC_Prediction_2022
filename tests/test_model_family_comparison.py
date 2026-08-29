from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_model_families import (  # noqa: E402
    FittedClassifier,
    _chronological_inner_split,
    _symmetrized_probability,
)
from fight_predictor.hierarchical_bayes import (  # noqa: E402
    HierarchicalBayesConfig,
    hierarchical_bayes_predict,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


class HierarchicalBayesianFightModelTests(unittest.TestCase):
    @staticmethod
    def _training() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        matchups = [
            ("a", "b", 1.0, 1),
            ("b", "a", -1.0, 0),
            ("c", "d", -0.8, 0),
            ("d", "c", 0.8, 1),
            ("a", "c", 0.5, 1),
            ("c", "a", -0.5, 0),
        ]
        for _ in range(4):
            for fighter, opponent, feature, target in matchups:
                rows.append(
                    {
                        "fighter_id": fighter,
                        "opponent_id": opponent,
                        "strength_diff": feature,
                        "target": target,
                    }
                )
        return pd.DataFrame(rows)

    def test_posterior_is_deterministic_symmetric_and_shrinks_new_fighters(self):
        prediction = pd.DataFrame(
            [
                {"fighter_id": "a", "opponent_id": "b", "strength_diff": 0.7},
                {"fighter_id": "b", "opponent_id": "a", "strength_diff": -0.7},
                {"fighter_id": "new-1", "opponent_id": "new-2", "strength_diff": 0.0},
                {"fighter_id": "new-2", "opponent_id": "new-1", "strength_diff": 0.0},
            ]
        )
        config = HierarchicalBayesConfig(
            burn_in=20, posterior_draws=20, chains=2, seed=17
        )
        first = hierarchical_bayes_predict(
            self._training(), prediction, ["strength_diff"], config=config
        )
        second = hierarchical_bayes_predict(
            self._training(), prediction, ["strength_diff"], config=config
        )
        np.testing.assert_allclose(first.probability, second.probability, atol=0.0)
        self.assertAlmostEqual(first.probability[0] + first.probability[1], 1.0)
        self.assertAlmostEqual(first.probability[2] + first.probability[3], 1.0)
        self.assertLess(abs(first.probability[2] - 0.5), 0.15)
        self.assertTrue(np.all(first.lower_probability <= first.probability))
        self.assertTrue(np.all(first.probability <= first.upper_probability))
        self.assertEqual(first.diagnostics["total_retained_draws"], 40)
        self.assertEqual(
            first.diagnostics[
                "unseen_prediction_fighters_integrated_over_population_prior"
            ],
            2,
        )

    def test_prediction_labels_are_never_consulted(self):
        prediction = pd.DataFrame(
            [{"fighter_id": "a", "opponent_id": "b", "strength_diff": 0.7}]
        )
        changed = prediction.copy()
        prediction["target"] = 0
        changed["target"] = 1
        config = HierarchicalBayesConfig(
            burn_in=10, posterior_draws=10, chains=1, seed=19
        )
        first = hierarchical_bayes_predict(
            self._training(), prediction, ["strength_diff"], config=config
        )
        second = hierarchical_bayes_predict(
            self._training(), changed, ["strength_diff"], config=config
        )
        np.testing.assert_allclose(first.probability, second.probability, atol=0.0)


class ModelFamilyEvaluationContractTests(unittest.TestCase):
    def test_nonlinear_probabilities_are_forced_to_swap_complements(self):
        training = pd.DataFrame(
            {
                "x": [-2.0, -1.0, -0.5, 0.4, 1.0, 2.0],
                "target": [0, 0, 0, 1, 1, 1],
            }
        )
        imputer = SimpleImputer(
            strategy="constant", fill_value=0.0, keep_empty_features=True
        )
        values = imputer.fit_transform(training[["x"]])
        scaler = StandardScaler(with_mean=False)
        values = scaler.fit_transform(values)
        classifier = GaussianNB().fit(values, training["target"])
        fitted = FittedClassifier(imputer, scaler, classifier)
        test = pd.DataFrame({"x": [0.7, -0.7]})
        probability = _symmetrized_probability(fitted, test, ["x"])
        self.assertAlmostEqual(probability[0] + probability[1], 1.0)

    def test_inner_validation_is_the_latest_complete_training_year(self):
        rows: list[dict[str, object]] = []
        for year, count in ((2020, 300), (2021, 300), (2022, 120)):
            for index in range(count):
                rows.append(
                    {
                        "date": pd.Timestamp(year=year, month=1, day=1)
                        + pd.Timedelta(days=index % 300),
                        "target": index % 2,
                    }
                )
        fit, validation, contract = _chronological_inner_split(pd.DataFrame(rows))
        self.assertEqual(set(fit["date"].dt.year), {2020, 2021})
        self.assertEqual(set(validation["date"].dt.year), {2022})
        self.assertEqual(contract["validation_year"], 2022)
        self.assertLess(fit["date"].max(), validation["date"].min())


if __name__ == "__main__":
    unittest.main()
