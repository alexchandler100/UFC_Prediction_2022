from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_historical_moneyline_profitability import (  # noqa: E402
    best_offer_per_fight,
    build_leave_one_out_values,
    candidate_probability,
    choose_threshold,
    threshold_metrics,
)


def _decimal_pair(probability: float) -> tuple[float, float]:
    overround = 1.05
    return (
        1.0 / (probability * overround),
        1.0 / ((1.0 - probability) * overround),
    )


def _book_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (book, probability) in enumerate(
        (("A", 0.55), ("B", 0.60), ("C", 0.65), ("D", 0.70)),
        start=1,
    ):
        fighter_odds, opponent_odds = _decimal_pair(probability)
        for horizon, suffix in (
            ("safe_t24", "00:00:00Z"),
            ("strict_latest_before_event_date", "12:00:00Z"),
        ):
            rows.append(
                {
                    "ufc_event_date": "2025-01-11",
                    "ufc_event_id": "event",
                    "ufc_fight_id": "fight",
                    "ufc_fighter_1_id": "a",
                    "ufc_fighter_2_id": "b",
                    "source_matchup_id": 1,
                    "book_key": f"book-{index}",
                    "book_name": book,
                    "book_kind": "book",
                    "horizon": horizon,
                    "cutoff_utc": (
                        "2025-01-10T00:00:00Z"
                        if horizon == "safe_t24"
                        else "2025-01-11T00:00:00Z"
                    ),
                    "observed_at_utc": f"2025-01-10T{suffix}",
                    "fighter_1_decimal_odds": fighter_odds,
                    "fighter_2_decimal_odds": opponent_odds,
                    "fighter_1_no_vig_probability": probability,
                }
            )
    return rows


def _paired(*, reversed_sides: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_date": "2025-01-11",
                "event_id": "event",
                "fight_id": "fight",
                "fighter_id": "b" if reversed_sides else "a",
                "opponent_id": "a" if reversed_sides else "b",
                "fighter_name": "B" if reversed_sides else "A",
                "opponent_name": "A" if reversed_sides else "B",
                "target": 0 if reversed_sides else 1,
                "horizon": "safe_t24",
                "model_probability": 0.40 if reversed_sides else 0.60,
            }
        ]
    )


class HistoricalMoneylineProfitabilityTests(unittest.TestCase):
    def test_target_book_is_excluded_from_fair_probability(self):
        values = build_leave_one_out_values(
            _book_rows(),
            _paired(),
            fitted={"features": [], "scales": [], "coefficients": []},
        )
        book_a = values.loc[values["book_name"].eq("A")].iloc[0]
        self.assertAlmostEqual(
            book_a["leave_one_out_market_probability"],
            (0.60 + 0.65 + 0.70) / 3.0,
        )
        self.assertEqual(book_a["book_count_excluding_target"], 3)
        self.assertAlmostEqual(
            book_a["candidate_probability"],
            book_a["leave_one_out_market_probability"],
        )

    def test_reversed_fighter_orientation_swaps_prices_and_probability(self):
        values = build_leave_one_out_values(
            _book_rows(),
            _paired(reversed_sides=True),
            fitted={"features": [], "scales": [], "coefficients": []},
        )
        book_a = values.loc[values["book_name"].eq("A")].iloc[0]
        expected_probability = 1.0 - (0.60 + 0.65 + 0.70) / 3.0
        self.assertAlmostEqual(
            book_a["leave_one_out_market_probability"], expected_probability
        )
        original_fighter_odds, original_opponent_odds = _decimal_pair(0.55)
        self.assertAlmostEqual(
            book_a["fighter_decimal_odds"], original_opponent_odds
        )
        self.assertAlmostEqual(
            book_a["opponent_decimal_odds"], original_fighter_odds
        )

    def test_best_offer_settles_only_once_per_fight(self):
        values = build_leave_one_out_values(
            _book_rows(),
            _paired(),
            fitted={"features": [], "scales": [], "coefficients": []},
        )
        best = best_offer_per_fight(
            values,
            probability_column="leave_one_out_market_probability",
            strategy="market",
        )
        self.assertEqual(len(best), 1)
        self.assertIn(best.iloc[0]["side"], {"fighter", "opponent"})
        expected_profit = (
            best.iloc[0]["decimal_odds"] - 1.0
            if best.iloc[0]["won"]
            else -1.0
        )
        self.assertAlmostEqual(best.iloc[0]["profit_units"], expected_profit)

    def test_threshold_is_selected_only_from_rows_passed_to_selector(self):
        selection = pd.DataFrame(
            [
                {
                    "event_date": f"2025-01-{index + 1:02d}",
                    "event_id": f"event-{index}",
                    "fight_id": f"fight-{index}",
                    "estimated_ev": 0.06 if index < 10 else 0.01,
                    "profit_units": 1.0 if index < 10 else -1.0,
                    "won": index < 10,
                    "closing_price_advantage": 0.01,
                    "book_name": "A",
                    "side": "fighter",
                }
                for index in range(20)
            ]
        )
        threshold, _results, status = choose_threshold(
            selection,
            thresholds=(0.0, 0.05),
            minimum_bets=5,
            minimum_events=5,
            bootstrap_samples=100,
        )
        self.assertEqual(threshold, 0.05)
        self.assertEqual(status, "selected_on_earlier_flat_profit")

        later = selection.copy()
        later["profit_units"] = -1.0
        later_metrics = threshold_metrics(
            later, threshold=threshold, bootstrap_samples=100
        )
        self.assertEqual(threshold, 0.05)
        self.assertLess(later_metrics["profit_units"], 0.0)

    def test_frozen_candidate_formula_matches_offset_definition(self):
        probability = candidate_probability(
            market_probability=0.60,
            model_probability=0.70,
            book_probability_range=0.10,
            fitted={
                "features": ["model_disagreement"],
                "scales": [1.0],
                "coefficients": [1.0],
            },
        )
        self.assertAlmostEqual(probability, 0.70)

    def test_unprofitable_selection_period_uses_fixed_reference_threshold(self):
        rows = pd.DataFrame(
            [
                {
                    "event_date": f"2025-02-{index + 1:02d}",
                    "event_id": f"event-{index}",
                    "fight_id": f"fight-{index}",
                    "estimated_ev": 0.10,
                    "profit_units": -1.0,
                    "won": False,
                    "closing_price_advantage": 0.0,
                    "book_name": "A",
                    "side": "fighter",
                }
                for index in range(12)
            ]
        )
        threshold, _results, status = choose_threshold(
            rows,
            thresholds=(0.0, 0.05),
            minimum_bets=10,
            minimum_events=10,
            bootstrap_samples=100,
        )
        self.assertEqual(threshold, 0.05)
        self.assertEqual(
            status, "fallback_5_percent_no_profitable_earlier_threshold"
        )
        threshold, _, status = choose_threshold(
            rows, thresholds=(0.0, 0.05), minimum_bets=10,
            minimum_events=10, bootstrap_samples=100, allow_abstention=True,
        )
        self.assertIsNone(threshold)
        self.assertEqual(status, "no_bet_no_profitable_earlier_threshold")
        threshold, _, status = choose_threshold(
            rows.iloc[:1], thresholds=(0.0, 0.05), minimum_bets=10,
            minimum_events=10, bootstrap_samples=100, allow_abstention=True,
        )
        self.assertIsNone(threshold)
        self.assertEqual(status, "no_bet_insufficient_history")


if __name__ == "__main__":
    unittest.main()
