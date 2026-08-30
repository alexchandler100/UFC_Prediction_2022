import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_current_model_vs_market import (  # noqa: E402
    evaluate_prior_card_blend,
    event_block_log_loss_interval,
    latest_eligible_consensus,
)
from fight_predictor import TemporalFightPredictor  # noqa: E402
from market_tracker import QuoteSnapshot  # noqa: E402


def make_quote(capture, observed, book, fighter_line, opponent_line):
    return QuoteSnapshot.create(
        capture_id=capture,
        event_id="event-one",
        fight_id="fight-one",
        fighter_id="fighter-a",
        opponent_id="fighter-b",
        fighter_name="Fighter A",
        opponent_name="Fighter B",
        event_date="2026-02-10",
        timing_precision="date",
        event_start_utc=None,
        observed_at_utc=observed,
        source="fixture",
        book=book,
        fighter_moneyline=fighter_line,
        opponent_moneyline=opponent_line,
        source_payload={"capture": capture},
    )


class CurrentModelMarketReplayTests(unittest.TestCase):
    def test_latest_consensus_requires_enough_books_and_selects_latest_capture(self):
        quotes = []
        for capture, observed, lines in (
            ("early", "2026-02-01T12:00:00Z", (-120, 100)),
            ("late", "2026-02-08T12:00:00Z", (-150, 125)),
        ):
            for book in ("A", "B", "C"):
                quotes.append(make_quote(capture, observed, book, *lines))
        # A still later capture with only two books is ineligible.
        for book in ("A", "B"):
            quotes.append(
                make_quote("thin", "2026-02-09T12:00:00Z", book, -200, 160)
            )

        selected = latest_eligible_consensus(quotes, minimum_books=3)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].capture_id, "late")
        self.assertEqual(selected[0].book_count, 3)

    def test_prior_card_blend_never_uses_current_date_for_weight_selection(self):
        rows = []
        for card_index in range(4):
            for fight_index in range(5):
                rows.append(
                    {
                        "event_date": f"2026-01-{card_index + 1:02d}",
                        "event_id": f"event-{card_index}",
                        "fight_id": f"fight-{card_index}-{fight_index}",
                        "market_probability": 0.55,
                        "model_probability": 0.80,
                        "target": 1,
                    }
                )
        paired = pd.DataFrame(rows)

        evaluated = evaluate_prior_card_blend(
            paired,
            gamma_grid=(0.0, 1.0),
            minimum_prior_cards=2,
            minimum_prior_fights=10,
            lookback_cards=None,
        )

        first_two = evaluated[evaluated["event_id"].isin(["event-0", "event-1"])]
        later = evaluated[evaluated["event_id"].isin(["event-2", "event-3"])]
        self.assertTrue((first_two["blend_status"] == "insufficient_prior_history").all())
        self.assertTrue((later["blend_status"] == "evaluated").all())
        self.assertTrue((later["selected_gamma"] == 1.0).all())
        self.assertTrue((later["prior_card_count"] >= 2).all())

    def test_event_block_interval_is_deterministic(self):
        frame = pd.DataFrame(
            {
                "event_id": ["a", "a", "b", "b"],
                "target": [1, 0, 1, 0],
                "market_probability": [0.6, 0.4, 0.55, 0.45],
                "model_probability": [0.7, 0.3, 0.65, 0.35],
            }
        )
        first = event_block_log_loss_interval(
            frame, "model_probability", seed=9, replicates=100
        )
        second = event_block_log_loss_interval(
            frame, "model_probability", seed=9, replicates=100
        )
        self.assertEqual(first, second)
        self.assertLess(first["point_difference"], 0.0)

    def test_walk_forward_predictions_preserve_lineage_and_use_prior_years(self):
        class FastPredictor(TemporalFightPredictor):
            def _rolling_probabilities(self, frame, c_value):
                return frame["target"].to_numpy(), np.full(len(frame), 0.55)

            @staticmethod
            def _fit_symmetric_calibration_slope(y_true, probability):
                return 1.0

            @staticmethod
            def _fit_pipeline(X, y, c_value):
                return len(X)

            @staticmethod
            def _pipeline_probability(pipeline, X):
                return np.full(len(X), 0.60)

        dates = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=index // 2) for index in range(600)]
        dates.extend([pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-09")])
        frame = pd.DataFrame(
            {
                "date": dates,
                "event_id": [f"event-{index // 2}" for index in range(602)],
                "fight_id": [f"fight-{index}" for index in range(602)],
                "fighter_id": [f"fighter-{index}-a" for index in range(602)],
                "opponent_id": [f"fighter-{index}-b" for index in range(602)],
                "fighter": [f"A {index}" for index in range(602)],
                "opponent": [f"B {index}" for index in range(602)],
                "target": [index % 2 for index in range(602)],
                "feature_diff": np.linspace(-1.0, 1.0, 602),
            }
        )
        predictor = object.__new__(FastPredictor)
        predictor.point_in_time_data = frame
        predictor.feature_columns = ["feature_diff"]

        predictions = predictor.walk_forward_predictions((2021,))

        self.assertEqual(predictions["fight_id"].tolist(), ["fight-600", "fight-601"])
        self.assertEqual(set(predictions["training_through"]), {"2020-10-26"})
        self.assertEqual(set(predictions["training_fights"]), {600})
        self.assertEqual(set(predictions["evaluation_year"]), {2021})
        self.assertTrue(np.allclose(predictions["model_probability"], 0.60))


if __name__ == "__main__":
    unittest.main()
