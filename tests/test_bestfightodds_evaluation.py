from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_bestfightodds_history import (  # noqa: E402
    evaluate_paired_snapshot,
    load_precomputed_predictions,
    pair_consensus_with_predictions,
)


def _consensus(*, reversed_ids: bool = False) -> list[dict[str, object]]:
    first, second = ("b", "a") if reversed_ids else ("a", "b")
    probability = 0.40 if reversed_ids else 0.60
    minimum = 0.35 if reversed_ids else 0.55
    maximum = 0.45 if reversed_ids else 0.65
    return [
        {
            "ufc_event_date": "2025-01-11",
            "ufc_event_id": "event",
            "ufc_fight_id": "fight",
            "ufc_fighter_1_id": first,
            "ufc_fighter_2_id": second,
            "fighter_1_name": "First",
            "fighter_2_name": "Second",
            "horizon": horizon,
            "cutoff_basis": "source_event_calendar_date_at_00_utc",
            "actual_event_start_time_known": False,
            "book_count": 4,
            "fighter_1_market_probability": probability + shift,
            "minimum_book_probability": minimum + shift,
            "maximum_book_probability": maximum + shift,
        }
        for horizon, shift in (("opening", 0.0), ("safe_t24", 0.02))
    ]


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-01-11",
                "event_id": "event",
                "fight_id": "fight",
                "fighter_id": "a",
                "opponent_id": "b",
                "fighter": "A",
                "opponent": "B",
                "target": 1,
                "model_probability": 0.70,
                "training_through": "2024-12-31",
                "selected_c": 0.1,
                "calibration_slope": 0.9,
            }
        ]
    )


class BestFightOddsEvaluationTests(unittest.TestCase):
    def test_precomputed_current_logistic_alias_is_validated(self):
        source = _predictions().drop(
            columns=["model_probability", "training_through"]
        )
        source["current_logistic_probability"] = 0.70
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            source.to_csv(path, index=False)
            loaded, metadata = load_precomputed_predictions(path)
        self.assertAlmostEqual(loaded.iloc[0]["model_probability"], 0.70)
        self.assertEqual(metadata["fights"], 1)
        self.assertEqual(len(metadata["sha256"]), 64)

    def test_precomputed_future_trained_prediction_is_rejected(self):
        source = _predictions()
        source["training_through"] = source["date"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            source.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "training reaches"):
                load_precomputed_predictions(path)

    def test_stable_ids_align_reversed_market_orientation(self):
        paired, coverage = pair_consensus_with_predictions(
            _consensus(reversed_ids=True), _predictions()
        )
        self.assertAlmostEqual(paired.iloc[0]["market_probability"], 0.60)
        self.assertAlmostEqual(paired.iloc[0]["minimum_book_probability"], 0.55)
        self.assertAlmostEqual(paired.iloc[0]["maximum_book_probability"], 0.65)
        self.assertEqual(coverage["paired_fights"], 1)
        self.assertEqual(coverage["paired_rows"], 2)

    def test_duplicate_fight_horizon_is_rejected(self):
        duplicate = [*_consensus(), _consensus()[0]]
        with self.assertRaisesRegex(ValueError, "duplicate fight/horizon"):
            pair_consensus_with_predictions(duplicate, _predictions())

    def test_harmless_floating_point_range_roundoff_is_accepted(self):
        rows = _consensus()
        rows[0]["fighter_1_market_probability"] = 0.6 - 1e-16
        rows[0]["minimum_book_probability"] = 0.6
        rows[0]["maximum_book_probability"] = 0.6
        paired, _ = pair_consensus_with_predictions(rows, _predictions())
        self.assertEqual(len(paired), 2)

    def test_reports_each_horizon_and_common_fight_market_movement(self):
        paired, _ = pair_consensus_with_predictions(_consensus(), _predictions())
        report = evaluate_paired_snapshot(paired)
        self.assertEqual(set(report["horizons"]), {"opening", "safe_t24"})
        self.assertEqual(
            report["horizons"]["opening"]["earlier_cards_selected_blend"]
            ["model_weight_fight_counts"],
            {},
        )
        self.assertEqual(
            report["same_fights_at_every_available_horizon"]["fights"], 1
        )
        movement = report["market_movement_from_opening"][
            "safe_t24_minus_opening"
        ]
        self.assertEqual(movement["fight_count"], 1)
        self.assertLess(movement["point_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
