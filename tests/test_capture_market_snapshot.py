import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import capture_market_snapshot as collector  # noqa: E402
from market_tracker import (  # noqa: E402
    ForecastCaptureStore,
    QuoteSnapshotStore,
    matchup_id_for,
)


CAPTURE_STARTED = datetime(2026, 8, 14, 17, 0, 0, tzinfo=timezone.utc)
CAPTURE_FINISHED = datetime(2026, 8, 14, 17, 0, 5, tzinfo=timezone.utc)
EVENT_DATE = "2026-08-16"
FORECAST_ISSUED = "2026-08-13T21:00:00Z"
SOURCE_COMMIT = "a" * 40


class _CaptureClock:
    calls = 0

    @classmethod
    def now(cls, tz=None):
        cls.calls += 1
        value = CAPTURE_STARTED if cls.calls == 1 else CAPTURE_FINISHED
        return value if tz is None else value.astimezone(tz)


def _published_matchup(index: int) -> collector.PublishedMatchup:
    event_id = "event-card"
    fighter_id = f"fighter-{index}"
    opponent_id = f"opponent-{index}"
    return collector.PublishedMatchup(
        fighter_name=f"Fighter {index}",
        opponent_name=f"Opponent {index}",
        fighter_id=fighter_id,
        opponent_id=opponent_id,
        matchup_id=matchup_id_for(event_id, fighter_id, opponent_id),
        fight_id=None,
        model_probability=0.55,
        model_status="model",
        forecast_issued_at_utc=FORECAST_ISSUED,
        forecast_source_commit=SOURCE_COMMIT,
    )


class CaptureMarketSnapshotTests(unittest.TestCase):
    def test_offline_capture_preserves_publication_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external"
            market = root / "market"
            external.mkdir()
            market.mkdir()

            event_id = "event-fixture"
            event_url = f"http://ufcstats.test/event/{event_id}"
            artifact = {
                "model_id": "winner-model-fixture",
                "model_version": "2",
                "data_through": "2026-08-08",
            }
            card = {
                "date": EVENT_DATE,
                "title": "UFC Fixture Card",
                "event_url": event_url,
                "event_id": event_id,
            }
            # Upcoming fights do not have a UFCStats fight ID. Its deliberate
            # absence here protects the collector's optional-fight-ID contract.
            vegas = pd.DataFrame(
                [
                    {
                        "fighter name": "Alpha One",
                        "opponent name": "Beta Two",
                        "date": EVENT_DATE,
                        "event id": event_id,
                        "event url": event_url,
                        "fighter id": "fighter-alpha",
                        "opponent id": "fighter-beta",
                        "model id": artifact["model_id"],
                        "model version": artifact["model_version"],
                        "model trained through": artifact["data_through"],
                        "model probability": 0.61,
                        "model status": "model",
                        "forecast issued at": FORECAST_ISSUED,
                        "forecast source commit": SOURCE_COMMIT,
                        "betting status": (
                            "disabled_pending_market_relative_validation"
                        ),
                    }
                ]
            )
            self.assertNotIn("fight id", vegas.columns)

            card_path = external / "card_info.json"
            vegas_path = external / "vegas_odds.json"
            model_path = external / "winner_model.json"
            card_path.write_text(json.dumps(card), encoding="utf-8")
            vegas_path.write_text(vegas.to_json(), encoding="utf-8")
            model_path.write_text(json.dumps(artifact), encoding="utf-8")
            publication_paths = (card_path, vegas_path, model_path)
            publication_before = {
                path: path.read_bytes() for path in publication_paths
            }

            fresh_odds = pd.DataFrame(
                [
                    {
                        "fighter name": "Alpha One",
                        "opponent name": "Beta Two",
                        "fighter BookA": "-120",
                        "opponent BookA": "+105",
                        "fighter BookB": "-118",
                        "opponent BookB": "+103",
                        "fighter BookC": "-115",
                        "opponent BookC": "+100",
                        "fighter BookD": "-117",
                        "opponent BookD": "+102",
                    },
                    {
                        "fighter name": "Other Card Fighter",
                        "opponent name": "Other Card Opponent",
                        "fighter BookA": "-110",
                        "opponent BookA": "-110",
                        "fighter BookB": "-110",
                        "opponent BookB": "-110",
                        "fighter BookC": "-110",
                        "opponent BookC": "-110",
                        "fighter BookD": "-110",
                        "opponent BookD": "-110",
                    },
                ]
            )

            output_paths = {
                "CARD_PATH": card_path,
                "VEGAS_PATH": vegas_path,
                "MODEL_PATH": model_path,
                "QUOTE_CSV_PATH": market / "quote_snapshots.csv",
                "QUOTE_JSONL_PATH": market / "quote_snapshots.jsonl",
                "FORECAST_CSV_PATH": market / "forecast_captures.csv",
                "FORECAST_JSONL_PATH": market / "forecast_captures.jsonl",
                "REPORT_PATH": market / "capture_report.json",
            }
            scrape_order = []

            def fake_scrape(_odds_getter):
                scrape_order.append("scrape")
                self.assertEqual(_CaptureClock.calls, 1)
                return fresh_odds.copy(deep=True)

            _CaptureClock.calls = 0
            with ExitStack() as stack:
                for attribute, path in output_paths.items():
                    stack.enter_context(patch.object(collector, attribute, path))
                stack.enter_context(patch.object(collector, "datetime", _CaptureClock))
                scrape_mock = stack.enter_context(
                    patch.object(
                        collector.OddsGetter,
                        "make_odds_df",
                        autospec=True,
                        side_effect=fake_scrape,
                    )
                )
                stack.enter_context(
                    patch.dict(
                        os.environ,
                        {
                            "GITHUB_SHA": "b" * 40,
                            "GITHUB_STEP_SUMMARY": "",
                            "MARKET_CAPTURE_ID": "capture-fixture",
                        },
                    )
                )

                report = collector.capture_market_snapshot()
                round_tripped = collector.validate_generated_capture()

            scrape_mock.assert_called_once_with(ANY)
            self.assertEqual(scrape_order, ["scrape"])
            self.assertEqual(_CaptureClock.calls, 2)
            self.assertEqual(report, round_tripped)
            self.assertEqual(
                report["capture_started_at_utc"], "2026-08-14T17:00:00.000000Z"
            )
            self.assertEqual(
                report["captured_at_utc"], "2026-08-14T17:00:05.000000Z"
            )
            self.assertEqual(report["source_unmatched_matchup_count"], 1)
            self.assertEqual(report["quote_records_in_capture"], 4)
            self.assertEqual(report["forecast_records_in_capture"], 1)

            quotes = QuoteSnapshotStore(
                output_paths["QUOTE_CSV_PATH"],
                output_paths["QUOTE_JSONL_PATH"],
            ).read()
            forecasts = ForecastCaptureStore(
                output_paths["FORECAST_CSV_PATH"],
                output_paths["FORECAST_JSONL_PATH"],
            ).read()
            self.assertEqual(len(quotes), 4)
            self.assertEqual(len(forecasts), 1)
            self.assertEqual(
                {quote.observed_at_utc for quote in quotes},
                {report["captured_at_utc"]},
            )
            self.assertEqual(
                {quote.source_payload_sha256 for quote in quotes},
                {report["source_payload_sha256"]},
            )
            self.assertEqual(
                report["source_payload_sha256"],
                collector._source_payload_sha256(fresh_odds),
            )
            self.assertEqual(
                {path: path.read_bytes() for path in publication_paths},
                publication_before,
            )
            self.assertEqual(
                json.loads(output_paths["REPORT_PATH"].read_text(encoding="utf-8")),
                report,
            )

    def test_unrelated_rows_are_skipped_but_half_card_coverage_is_required(self):
        published = tuple(_published_matchup(index) for index in range(5))
        source_rows = [
            {
                "fighter name": matchup.fighter_name,
                "opponent name": matchup.opponent_name,
            }
            for matchup in published[:3]
        ]
        unrelated = {
            "fighter name": "Unrelated Fighter",
            "opponent name": "Unrelated Opponent",
        }

        matches, unmatched = collector._map_source_rows(
            pd.DataFrame([*source_rows, unrelated]), published
        )
        self.assertEqual(len(matches), 3)
        self.assertEqual(unmatched, 1)

        with self.assertRaisesRegex(
            collector.CaptureError, r"2/5 \(required 3\)"
        ):
            collector._map_source_rows(
                pd.DataFrame([*source_rows[:2], unrelated]), published
            )

    def test_browser_retrieval_retries_with_fresh_instances(self):
        odds = pd.DataFrame(
            [{"fighter name": "A", "opponent name": "B"}]
        )
        with patch.object(
            collector.OddsGetter,
            "make_odds_df",
            autospec=True,
            side_effect=[RuntimeError("first"), RuntimeError("second"), odds],
        ) as scrape, patch.object(collector.time, "sleep") as sleep:
            result = collector._retrieve_fresh_odds()
        self.assertTrue(result.equals(odds))
        self.assertEqual(scrape.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [15.0, 60.0])


if __name__ == "__main__":
    unittest.main()
