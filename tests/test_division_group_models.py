from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_division_group_models import (  # noqa: E402
    _fit_group_fold,
    division_group,
    pooled_probability,
    select_pooling_design,
)


class DivisionGroupingTests(unittest.TestCase):
    def test_predeclared_division_groups(self):
        self.assertEqual(
            division_group("Flyweight"), "mens_below_welterweight"
        )
        self.assertEqual(
            division_group("Middleweight"), "mens_welter_to_middle"
        )
        self.assertEqual(
            division_group("Heavyweight"), "mens_light_heavy_plus"
        )
        self.assertEqual(division_group("Women's Flyweight"), "womens_all")
        self.assertEqual(division_group("Catch Weight"), "unclassified")

    def test_log_odds_pooling_preserves_side_swap_symmetry(self):
        shared = np.array([0.7, 0.3])
        grouped = np.array([0.8, 0.2])
        pooled = pooled_probability(shared, grouped, 0.4)
        self.assertAlmostEqual(float(pooled[0] + pooled[1]), 1.0)
        np.testing.assert_allclose(
            pooled_probability(shared, grouped, 0.0), shared
        )
        np.testing.assert_allclose(
            pooled_probability(shared, grouped, 1.0), grouped
        )


class DivisionModelSelectionTests(unittest.TestCase):
    def test_group_specific_weights_are_selected_from_earlier_fights(self):
        rows: list[dict[str, object]] = []
        for target in (0, 1) * 20:
            rows.append(
                {
                    "target": target,
                    "division_group": "mens_below_welterweight",
                    "current_logistic_probability": 0.9 if target else 0.1,
                    "group_model_probability": 0.1 if target else 0.9,
                }
            )
            rows.append(
                {
                    "target": target,
                    "division_group": "mens_welter_to_middle",
                    "current_logistic_probability": 0.5,
                    "group_model_probability": 0.9 if target else 0.1,
                }
            )
        selection, _ = select_pooling_design(pd.DataFrame(rows))
        self.assertEqual(
            selection["selected_design"], "group_specific_pooling_weights"
        )
        self.assertEqual(
            selection["selected_group_weights"]["mens_below_welterweight"],
            0.0,
        )
        self.assertEqual(
            selection["selected_group_weights"]["mens_welter_to_middle"],
            1.0,
        )

    def test_group_fit_is_deterministic_and_does_not_read_test_labels(self):
        rows = []
        for index in range(260):
            feature = -1.0 if index % 2 == 0 else 1.0
            rows.append(
                {
                    "date": pd.Timestamp("2010-01-01")
                    + pd.Timedelta(days=index * 14),
                    "strength": feature,
                    "target": int(feature > 0),
                }
            )
        training = pd.DataFrame(rows)
        test = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
                "strength": [0.6, -0.6],
                "target": [0, 1],
            }
        )
        changed = test.copy()
        changed["target"] = 1 - changed["target"]
        first, _ = _fit_group_fold(training, test, ["strength"])
        second, _ = _fit_group_fold(training, changed, ["strength"])
        np.testing.assert_allclose(first, second, atol=0.0)
        self.assertAlmostEqual(float(first[0] + first[1]), 1.0)


if __name__ == "__main__":
    unittest.main()
