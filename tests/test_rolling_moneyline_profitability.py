from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_rolling_moneyline_profitability import (  # noqa: E402
    build_fight_level_fit_frame,
    chronological_year_parts,
)


class RollingMoneylineProfitabilityTests(unittest.TestCase):
    def test_year_parts_never_train_or_select_on_test_year(self):
        frame = pd.DataFrame(
            [
                {
                    "event_date": date,
                    "event_id": f"event-{index}",
                    "fight_id": f"fight-{index}",
                }
                for index, date in enumerate(
                    (
                        "2021-01-01",
                        "2021-02-01",
                        "2021-03-01",
                        "2021-04-01",
                        "2022-01-01",
                        "2022-02-01",
                    )
                )
            ]
        )
        development, validation, test = chronological_year_parts(frame, 2022)
        self.assertEqual(development["event_date"].max(), "2021-03-01")
        self.assertEqual(validation["event_date"].tolist(), ["2021-04-01"])
        self.assertTrue(test["event_date"].str.startswith("2022").all())

    def test_book_rows_are_collapsed_to_one_equally_weighted_fight(self):
        values = pd.DataFrame(
            [
                {
                    "event_date": "2022-01-01",
                    "event_id": "event",
                    "fight_id": "fight",
                    "target": 1,
                    "model_probability": 0.7,
                    "leave_one_out_market_probability": probability,
                    "book_probability_range_excluding_target": spread,
                    "book_name": book,
                }
                for probability, spread, book in (
                    (0.55, 0.10, "A"),
                    (0.65, 0.20, "B"),
                )
            ]
        )
        frame = build_fight_level_fit_frame(values)
        self.assertEqual(len(frame), 1)
        self.assertAlmostEqual(frame.iloc[0]["market_probability"], 0.60)
        self.assertAlmostEqual(frame.iloc[0]["book_probability_range"], 0.20)
        self.assertEqual(frame.iloc[0]["fresh_book_offers"], 2)


if __name__ == "__main__":
    unittest.main()
