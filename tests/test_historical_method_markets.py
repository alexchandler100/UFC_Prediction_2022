from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_historical_method_markets import (  # noqa: E402
    PRIMARY_OUTCOMES,
    add_rolling_blend,
    align_complete_markets,
    geometric_blend,
)


class HistoricalMethodMarketTests(unittest.TestCase):
    @staticmethod
    def _evaluation_row(*, year, number, actual, market, model):
        row = {
            "event_date": f"{year}-01-01",
            "event_id": f"event-{year}-{number}",
            "fight_id": f"fight-{year}-{number}",
            "actual_outcome": actual,
        }
        row.update(
            {
                f"market_{outcome}_probability": market[index]
                for index, outcome in enumerate(PRIMARY_OUTCOMES)
            }
        )
        row.update(
            {
                f"model_{outcome}_probability": model[index]
                for index, outcome in enumerate(PRIMARY_OUTCOMES)
            }
        )
        return row

    def test_geometric_blend_has_exact_market_and_model_endpoints(self):
        market = {
            outcome: probability
            for outcome, probability in zip(
                PRIMARY_OUTCOMES, (0.30, 0.10, 0.20, 0.15, 0.05, 0.20)
            )
        }
        model = {
            outcome: probability
            for outcome, probability in zip(
                PRIMARY_OUTCOMES, (0.20, 0.20, 0.10, 0.10, 0.10, 0.30)
            )
        }
        for outcome in PRIMARY_OUTCOMES:
            self.assertAlmostEqual(
                geometric_blend(market, model, 0.0)[outcome], market[outcome]
            )
            self.assertAlmostEqual(
                geometric_blend(market, model, 1.0)[outcome], model[outcome]
            )
        self.assertAlmostEqual(sum(geometric_blend(market, model, 0.25).values()), 1.0)

    def test_complete_market_aligns_by_stable_fighter_id(self):
        rows = []
        selections = (
            ("a", "ko_tko", 0.30),
            ("a", "submission", 0.10),
            ("a", "decision", 0.20),
            ("b", "ko_tko", 0.15),
            ("b", "submission", 0.05),
            ("b", "decision", 0.20),
        )
        for fighter_id, method, probability in selections:
            rows.append(
                {
                    "method_market_id": "market",
                    "ufc_event_date": "2025-01-11",
                    "ufc_event_id": "event",
                    "ufc_fight_id": "fight",
                    "book_key": "mean",
                    "book_name": "Mean",
                    "horizon": "safe_t24",
                    "selected_fighter_id": fighter_id,
                    "method": method,
                    "observed_at_utc": "2025-01-09T12:00:00Z",
                    "cutoff_utc": "2025-01-10T00:00:00Z",
                    "decimal_odds": 1.0 / probability,
                    "no_vig_probability": probability,
                }
            )
        prediction = {
            "event_date": "2025-01-11",
            "event_id": "event",
            "fight_id": "fight",
            # Reverse the source market's displayed fighter order deliberately.
            "fighter_id": "b",
            "opponent_id": "a",
            "target": 0,
            "actual_outcome": "opponent_decision",
            "scheduled_rounds": 3,
        }
        prediction.update(
            {
                f"model_{outcome}_probability": probability
                for outcome, probability in zip(
                    PRIMARY_OUTCOMES, (0.15, 0.05, 0.20, 0.30, 0.10, 0.20)
                )
            }
        )
        aligned = align_complete_markets(
            pd.DataFrame(rows), pd.DataFrame([prediction])
        )
        self.assertEqual(len(aligned), 1)
        self.assertAlmostEqual(aligned.iloc[0]["market_fighter_ko_tko_probability"], 0.15)
        self.assertAlmostEqual(aligned.iloc[0]["market_opponent_ko_tko_probability"], 0.30)
        self.assertAlmostEqual(aligned.iloc[0]["quote_age_hours"], 12.0)

        without_cutoff = pd.DataFrame(rows).drop(columns=["cutoff_utc"])
        with self.assertRaisesRegex(ValueError, "cutoff_utc"):
            align_complete_markets(without_cutoff, pd.DataFrame([prediction]))

    def test_rolling_blend_uses_only_earlier_years(self):
        market = (0.10, 0.18, 0.18, 0.18, 0.18, 0.18)
        model = (0.90, 0.02, 0.02, 0.02, 0.02, 0.02)
        rows = [
            self._evaluation_row(
                year=2024,
                number=index,
                actual="fighter_ko_tko",
                market=market,
                model=model,
            )
            for index in range(120)
        ]
        rows.append(
            self._evaluation_row(
                year=2025,
                number=0,
                actual="fighter_ko_tko",
                market=market,
                model=model,
            )
        )
        evaluated, contract = add_rolling_blend(pd.DataFrame(rows))
        self.assertEqual(
            evaluated.loc[evaluated["event_date"].str.startswith("2024"), "rolling_model_weight"].unique().tolist(),
            [0.0],
        )
        self.assertEqual(
            evaluated.loc[evaluated["event_date"].str.startswith("2025"), "rolling_model_weight"].unique().tolist(),
            [1.0],
        )
        extended = pd.concat(
            [
                pd.DataFrame(rows),
                pd.DataFrame(
                    [
                        self._evaluation_row(
                            year=2026,
                            number=0,
                            actual="opponent_decision",
                            market=market,
                            model=model,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )
        reevaluated, _ = add_rolling_blend(extended)
        self.assertEqual(
            reevaluated.loc[reevaluated["event_date"].str.startswith("2025"), "rolling_model_weight"].unique().tolist(),
            [1.0],
        )


if __name__ == "__main__":
    unittest.main()
