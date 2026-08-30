from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_historical_method_selections import (  # noqa: E402
    add_causal_market_calibration,
    add_rolling_binary_blend,
    align_method_selections,
    binary_logit_blend,
)


class HistoricalMethodSelectionTests(unittest.TestCase):
    def test_binary_blend_has_exact_endpoints(self):
        self.assertAlmostEqual(binary_logit_blend(0.2, 0.7, 0.0), 0.2)
        self.assertAlmostEqual(binary_logit_blend(0.2, 0.7, 1.0), 0.7)

    def test_one_sided_price_aligns_by_stable_fighter_id(self):
        price = pd.DataFrame(
            [
                {
                    "ufc_event_date": "2025-01-11",
                    "ufc_event_id": "event",
                    "ufc_fight_id": "fight",
                    "source_matchup_id": 123,
                    "selected_fighter_id": "a",
                    "method": "ko_tko",
                    "book_key": "mean",
                    "book_name": "Mean",
                    "horizon": "safe_t24",
                    "cutoff_utc": "2025-01-10T00:00:00Z",
                    "observed_at_utc": "2025-01-09T12:00:00Z",
                    "decimal_odds": 4.0,
                    "implied_probability": 0.25,
                }
            ]
        )
        prediction = {
            "event_date": "2025-01-11",
            "event_id": "event",
            "fight_id": "fight",
            "fighter_id": "b",
            "opponent_id": "a",
            "actual_outcome": "opponent_ko_tko",
            "model_opponent_ko_tko_probability": 0.3,
        }
        aligned, exclusions = align_method_selections(
            price, pd.DataFrame([prediction])
        )
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned.iloc[0]["selected_side"], "opponent")
        self.assertEqual(aligned.iloc[0]["selection_won"], 1)
        self.assertAlmostEqual(aligned.iloc[0]["quote_age_hours"], 12.0)
        self.assertEqual(
            exclusions["ambiguous_logical_selection_groups_excluded"], 0
        )

    @staticmethod
    def _rows(year: int, *, market: float, model: float, won: int, count: int):
        return [
            {
                "event_date": f"{year}-01-01",
                "event_id": f"event-{year}-{index}",
                "fight_id": f"fight-{year}-{index}",
                "method": "ko_tko",
                "selection_won": won,
                "raw_market_probability": market,
                "calibrated_market_probability": market,
                "model_probability": model,
            }
            for index in range(count)
        ]

    def test_calibration_and_blend_never_use_future_years(self):
        rows = self._rows(2024, market=0.2, model=0.8, won=1, count=248)
        rows += self._rows(2024, market=0.2, model=0.8, won=0, count=62)
        rows += self._rows(2025, market=0.2, model=0.8, won=1, count=1)
        frame = pd.DataFrame(rows)
        calibrated, contracts = add_causal_market_calibration(frame)
        self.assertEqual(
            contracts[0]["status"], "raw_price_insufficient_earlier_history"
        )
        self.assertEqual(contracts[1]["status"], "calibrated_on_earlier_years")
        blended, blend_contract = add_rolling_binary_blend(calibrated)
        self.assertEqual(blend_contract[0]["selected_model_weight"], 0.0)
        self.assertEqual(blend_contract[1]["selected_model_weight"], 1.0)
        original_2025 = blended.loc[
            blended["event_date"].str.startswith("2025"),
            "rolling_model_weight",
        ].iloc[0]
        extended = pd.concat(
            [
                frame,
                pd.DataFrame(
                    self._rows(2026, market=0.8, model=0.2, won=0, count=310)
                ),
            ],
            ignore_index=True,
        )
        recalibrated, _ = add_causal_market_calibration(extended)
        reblended, _ = add_rolling_binary_blend(recalibrated)
        self.assertEqual(
            reblended.loc[
                reblended["event_date"].str.startswith("2025"),
                "rolling_model_weight",
            ].iloc[0],
            original_2025,
        )


if __name__ == "__main__":
    unittest.main()
