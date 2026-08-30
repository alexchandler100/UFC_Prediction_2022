from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capture_method_market_snapshot import _build_method_forecast_captures  # noqa: E402
from market_tracker import (  # noqa: E402
    MethodForecastCapture,
    MethodForecastStore,
    MethodMarketSnapshot,
)


TERMINAL = {
    "fighter_ko_tko": 0.10,
    "fighter_submission": 0.05,
    "fighter_decision": 0.25,
    "fighter_other": 0.01,
    "opponent_ko_tko": 0.20,
    "opponent_submission": 0.10,
    "opponent_decision": 0.28,
    "opponent_other": 0.01,
}


def _forecast(**overrides):
    values = {
        "capture_id": "capture-1",
        "event_id": "event-1",
        "fighter_id": "b-fighter",
        "opponent_id": "a-fighter",
        "fighter_name": "Fighter B",
        "opponent_name": "Fighter A",
        "event_date": "2026-09-05",
        "event_start_utc": "2026-09-05T18:00:00Z",
        "observed_at_utc": "2026-09-04T18:00:00Z",
        "horizon": "t24",
        "forecast_issued_at_utc": "2026-09-01T12:00:00Z",
        "model_id": "outcome-model-1",
        "model_version": "candidate-v1",
        "model_trained_through": "2026-08-29",
        "source_commit_sha": "a" * 40,
        "training_input_sha256": "b" * 64,
        "source_publication_sha256": "c" * 64,
        "scheduled_rounds": 3,
        "terminal_probabilities": TERMINAL,
    }
    values.update(overrides)
    return MethodForecastCapture.create(**values)


def _price_snapshot(**overrides):
    values = {
        "capture_id": "capture-1",
        "event_id": "event-1",
        "fighter_id": "a-fighter",
        "opponent_id": "b-fighter",
        "fighter_name": "Fighter A",
        "opponent_name": "Fighter B",
        "event_date": "2026-09-05",
        "timing_precision": "timestamp",
        "event_start_utc": "2026-09-05T18:00:00Z",
        "observed_at_utc": "2026-09-04T18:00:00Z",
        "source": "bestfightodds.com",
        "source_event_id": "matchup-1",
        "source_book_key": "book-1",
        "book": "Book One",
        "horizon": "t24",
        "fighter_prices": {"ko_tko": 200},
        "opponent_prices": {"decision": 150},
        "source_payload_sha256": "d" * 64,
    }
    values.update(overrides)
    return MethodMarketSnapshot.create(**values)


class MethodForecastCaptureTests(unittest.TestCase):
    def test_weekly_workflow_publishes_forecast_mirrors(self):
        workflow = (ROOT / ".github/workflows/collect-market-snapshot.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("method_forecast_captures.csv", workflow)
        self.assertIn("method_forecast_captures.jsonl", workflow)

    def test_canonical_orientation_swaps_probabilities_and_round_trips(self):
        forecast = _forecast()
        self.assertEqual(forecast.fighter_id, "a-fighter")
        self.assertAlmostEqual(forecast.fighter_ko_tko_probability, 0.20)
        self.assertAlmostEqual(forecast.opponent_ko_tko_probability, 0.10)
        self.assertEqual(
            MethodForecastCapture.from_mapping(forecast.to_mapping()), forecast
        )

    def test_builder_freezes_one_forecast_per_matchup_horizon(self):
        first = _price_snapshot()
        second = _price_snapshot(source_book_key="book-2", book="Book Two")
        publication = {
            "forecast_issued_at_utc": "2026-09-01T12:00:00Z",
            "model_id": "outcome-model-1",
            "model_version": "candidate-v1",
            "model_trained_through": "2026-08-29",
            "source_commit_sha": "a" * 40,
            "training_input_sha256": "b" * 64,
            "publication_sha256": "c" * 64,
            "matchups": [
                {
                    "fighter_id": "b-fighter",
                    "opponent_id": "a-fighter",
                    "fighter_name": "Fighter B",
                    "opponent_name": "Fighter A",
                    "scheduled_rounds": 3,
                    "terminal_probabilities": TERMINAL,
                }
            ],
        }
        rows, missing = _build_method_forecast_captures(
            (first, second), outcome_forecasts=publication, existing=()
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(missing, 0)
        self.assertEqual(rows[0].capture_id, first.capture_id)
        self.assertEqual(rows[0].horizon, "t24")

    def test_store_is_append_only(self):
        forecast = _forecast()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = MethodForecastStore(root / "forecasts.csv", root / "forecasts.jsonl")
            first = store.append((forecast,))
            duplicate = store.append((forecast,))
            self.assertEqual(first.total_records, 1)
            self.assertEqual(duplicate.duplicate_ids, (forecast.forecast_id,))
            with self.assertRaisesRegex(Exception, "natural key"):
                store.append(
                    (
                        _forecast(
                            model_id="different-model",
                            terminal_probabilities={
                                **TERMINAL,
                                "fighter_ko_tko": 0.11,
                                "opponent_decision": 0.27,
                            },
                        ),
                    )
                )

    def test_forecast_must_precede_capture_and_remain_paper_only(self):
        with self.assertRaisesRegex(Exception, "issued after"):
            _forecast(forecast_issued_at_utc="2026-09-04T19:00:00Z")
        mapping = _forecast().to_mapping()
        mapping["execution_enabled"] = True
        with self.assertRaisesRegex(Exception, "paper-only"):
            MethodForecastCapture.from_mapping(mapping)


if __name__ == "__main__":
    unittest.main()
