import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fight_predictor.nonlinear_research import NonlinearResearchBuilder, nonlinear_terms
from fight_predictor.point_in_time import PointInTimeDatasetBuilder, TemporalFightPredictor
from evaluate_nonlinear_features import select_family
from test_style_matchup import make_profiles, make_fight


class NonlinearResearchTests(unittest.TestCase):
    def test_unknown_age_and_reach_cannot_create_interactions(self):
        values = {"age": 40, "age_known": 0, "reach": 80, "reach_known": 0,
                  "career_fights_log": 3, "recent_3y_sig_landed_per15": 60,
                  "recent_3y_td_landed_per15": 4, "career_sig_absorbed_per15": 50,
                  "career_knockdowns_absorbed_per15": 1}
        self.assertTrue(all(value == 0 for value in nonlinear_terms(values).values()))

    def test_features_preserve_baseline_and_do_not_use_future_fights(self):
        rows = make_fight("one", "e1", "2020-01-01", "a", "b")
        rows += make_fight("two", "e2", "2020-04-01", "a", "b")
        profiles = make_profiles("a", "b")
        profiles.loc[0, "dob"] = "Jan 01, 1982"
        profiles.loc[0, "reach"] = '76"'
        base = PointInTimeDatasetBuilder(pd.DataFrame(rows), profiles)
        base_points = base.build()
        builder = NonlinearResearchBuilder(pd.DataFrame(rows), profiles)
        points = builder.build()
        np.testing.assert_allclose(points[list(base.feature_columns)], base_points[list(base.feature_columns)])
        future = NonlinearResearchBuilder(pd.DataFrame(rows + make_fight(
            "three", "e3", "2021-01-01", "a", "b")), profiles).build()
        np.testing.assert_allclose(points[list(builder.feature_columns)],
                                   future.iloc[:2][list(builder.feature_columns)])
        a = builder.matchup_features("a", "b", "2020-06-01", "Lightweight")
        b = builder.matchup_features("b", "a", "2020-06-01", "Lightweight")
        np.testing.assert_allclose(a.to_numpy(), -b.to_numpy())
        self.assertGreater(abs(a["reach_striking_diff"].iloc[0]), 0)
        self.assertGreater(abs(a["age_after_35_diff"].iloc[0]), 0)
        # Verify the fitted probability too, not only the feature signs.
        rng = np.random.default_rng(48)
        x = pd.DataFrame(rng.normal(size=(100, len(builder.feature_columns))), columns=builder.feature_columns)
        model = TemporalFightPredictor._fit_pipeline(x, np.arange(100) % 2, .01)
        pa = TemporalFightPredictor._pipeline_probability(model, a)[0]
        pb = TemporalFightPredictor._pipeline_probability(model, b)[0]
        self.assertAlmostEqual(pa + pb, 1)

    def test_selection_requires_both_metrics_in_both_years(self):
        base = {"overall": {"log_loss": .63}, "per_year": {
            str(y): {"log_loss": .63, "brier": .22} for y in (2023, 2024)}}
        good = {"overall": {"log_loss": .62}, "per_year": {
            str(y): {"log_loss": .62, "brier": .21} for y in (2023, 2024)}}
        comparisons = {"baseline": base, "age_experience": good}
        self.assertEqual(select_family(comparisons), "age_experience")
        bad = copy.deepcopy(comparisons)
        bad["age_experience"]["per_year"]["2024"]["brier"] = .23
        self.assertEqual(select_family(bad), "baseline")
        tied = copy.deepcopy(comparisons)
        tied["age_curve"] = copy.deepcopy(good)
        self.assertEqual(select_family(tied), "age_experience")


if __name__ == "__main__":
    unittest.main()
