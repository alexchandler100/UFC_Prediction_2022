import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_feature_selection import (  # noqa: E402
    ModelConfig,
    _correlation_components,
    _derived_combinations,
    _fit_pipeline,
    _pipeline_probability,
    _rolling_splits,
)


class FeatureSelectionTests(unittest.TestCase):
    def test_search_includes_every_pair_and_triple_even_when_singles_fail(self):
        families = {name: (f"{name}_feature",) for name in "abcdefg"}
        combinations = _derived_combinations(families)

        self.assertEqual(len(combinations), 64)
        self.assertIn(("a", "b"), combinations)
        self.assertIn(("a", "b", "c"), combinations)
        self.assertTrue(all(len(item) <= 3 for item in combinations))

    def test_correlation_groups_are_built_from_training_values(self):
        frame = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [2.0, 4.0, 6.0, 8.0],
                "c": [1.0, -1.0, 1.0, -1.0],
            }
        )
        groups = [set(item) for item in _correlation_components(frame, ["a", "b", "c"])]

        self.assertIn({"a", "b"}, groups)
        self.assertIn({"c"}, groups)

    def test_inner_validation_never_trains_on_the_test_date_or_future(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=20, freq="30D"))

        for train_index, test_index in _rolling_splits(dates, n_splits=4):
            self.assertLess(
                dates.iloc[train_index].max(),
                dates.iloc[test_index].min(),
            )

    def test_standard_robust_elastic_and_svd_preserve_swap_symmetry(self):
        rng = np.random.default_rng(7)
        positive = rng.normal(size=(80, 6))
        values = np.vstack([positive, -positive])
        target = np.array([1] * 80 + [0] * 80)
        columns = [f"f{index}" for index in range(values.shape[1])]
        frame = pd.DataFrame(values, columns=columns)
        frame["target"] = target
        matchup = pd.DataFrame([positive[0], -positive[0]], columns=columns)
        configs = (
            ModelConfig("standard", 0.03),
            ModelConfig("robust", 0.03),
            ModelConfig("elastic", 0.1, l1_ratio=0.5),
            ModelConfig("svd", 0.03, components=4),
        )

        for config in configs:
            with self.subTest(config=config.key):
                pipeline = _fit_pipeline(frame, columns, config)
                probability = _pipeline_probability(pipeline, matchup, columns)
                self.assertAlmostEqual(float(probability.sum()), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
