import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import capture_market_snapshot as collector  # noqa: E402
from market_tracker import (  # noqa: E402
    ForecastCaptureStore,
    PaperDecisionStore,
    QuoteSnapshotStore,
    QuoteSourceMetadataStore,
    matchup_id_for,
)
from odds_getter import OddsApiResponse  # noqa: E402


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
                        "source event id": "api-event-fixture",
                        "source commence time": "2026-08-16T17:00:00Z",
                        "fighter BookA": "-120",
                        "opponent BookA": "+105",
                        "source BookA key": "book-a",
                        "source BookA last update": "2026-08-14T16:59:30Z",
                        "fighter BookB": "-118",
                        "opponent BookB": "+103",
                        "source BookB key": "book-b",
                        "source BookB last update": "2026-08-14T16:59:30Z",
                        "fighter BookC": "-115",
                        "opponent BookC": "+100",
                        "source BookC key": "book-c",
                        "source BookC last update": "2026-08-14T16:59:30Z",
                        "fighter BookD": "-117",
                        "opponent BookD": "+102",
                        "source BookD key": "book-d",
                        "source BookD last update": "2026-08-14T16:59:30Z",
                    },
                    {
                        "fighter name": "Other Card Fighter",
                        "opponent name": "Other Card Opponent",
                        "source event id": "another-api-event",
                        "source commence time": "2026-08-16T18:00:00Z",
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
                "SOURCE_METADATA_CSV_PATH": market / "quote_source_metadata.csv",
                "SOURCE_METADATA_JSONL_PATH": market / "quote_source_metadata.jsonl",
                "DECISION_CSV_PATH": market / "paper_decisions.csv",
                "DECISION_JSONL_PATH": market / "paper_decisions.jsonl",
                "REPORT_PATH": market / "capture_report.json",
            }
            scrape_order = []

            def fake_scrape():
                scrape_order.append("scrape")
                self.assertEqual(_CaptureClock.calls, 1)
                frame = fresh_odds.copy(deep=True)
                return collector.RetrievedOdds(
                    source=collector.ODDS_API_SOURCE,
                    frame=frame,
                    source_payload_sha256=collector._source_payload_sha256(
                        frame, source=collector.ODDS_API_SOURCE
                    ),
                    request_metadata={
                        "sport": "mma_mixed_martial_arts",
                        "market": "h2h",
                        "regions": "us,us2",
                        "odds_format": "american",
                        "requests_remaining": 499,
                        "requests_used": 1,
                        "request_cost": 2,
                    },
                )

            _CaptureClock.calls = 0
            with ExitStack() as stack:
                for attribute, path in output_paths.items():
                    stack.enter_context(patch.object(collector, attribute, path))
                stack.enter_context(patch.object(collector, "datetime", _CaptureClock))
                scrape_mock = stack.enter_context(
                    patch.object(
                        collector,
                        "_retrieve_market_odds",
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

            scrape_mock.assert_called_once_with()
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
            metadata = QuoteSourceMetadataStore(
                output_paths["SOURCE_METADATA_CSV_PATH"],
                output_paths["SOURCE_METADATA_JSONL_PATH"],
            ).read()
            self.assertEqual(len(metadata), 4)
            self.assertEqual(
                {item.source_quote_age_seconds for item in metadata}, {35.0}
            )
            self.assertEqual(
                PaperDecisionStore(
                    output_paths["DECISION_CSV_PATH"],
                    output_paths["DECISION_JSONL_PATH"],
                ).read(),
                (),
            )
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
                collector._source_payload_sha256(
                    fresh_odds, source=collector.ODDS_API_SOURCE
                ),
            )
            self.assertEqual(report["source"], collector.ODDS_API_SOURCE)
            self.assertEqual(
                report["source_request"]["requests_remaining"], 499
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

    def test_api_matchup_outside_card_date_window_is_rejected(self):
        matchup = _published_matchup(1)
        source = pd.DataFrame(
            [
                {
                    "fighter name": matchup.fighter_name,
                    "opponent name": matchup.opponent_name,
                    "source commence time": "2026-09-20T02:00:00Z",
                }
            ]
        )
        with self.assertRaisesRegex(collector.CaptureError, "0/1"):
            collector._map_source_rows(
                source, (matchup,), event_day="2026-08-22"
            )

    def test_browser_retrieval_retries_with_fresh_instances(self):
        odds = pd.DataFrame(
            [{"fighter name": "A", "opponent name": "B"}]
        )
        with patch.object(
            collector.OddsGetter,
            "make_fightodds_df",
            autospec=True,
            side_effect=[RuntimeError("first"), RuntimeError("second"), odds],
        ) as scrape, patch.object(collector.time, "sleep") as sleep:
            result = collector._retrieve_fresh_odds()
        self.assertTrue(result.equals(odds))
        self.assertEqual(scrape.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [15.0, 60.0])

    def test_default_market_source_uses_api_and_records_quota(self):
        frame = pd.DataFrame(
            [
                {
                    "fighter name": "A",
                    "opponent name": "B",
                    "fighter Book": -120,
                    "opponent Book": 105,
                }
            ]
        )
        response = OddsApiResponse(
            frame=frame,
            payload=[{"id": "event"}],
            requests_remaining=498,
            requests_used=2,
            request_cost=2,
        )
        with patch.dict(
            os.environ,
            {
                "MARKET_ODDS_SOURCE": "the-odds-api",
                "THE_ODDS_API_KEY": "secret-fixture",
                "ODDS_API_REGIONS": "us,us2",
            },
        ), patch.object(
            collector.TheOddsApiClient,
            "fetch",
            autospec=True,
            return_value=response,
        ) as fetch:
            retrieved = collector._retrieve_market_odds()
        self.assertEqual(retrieved.source, collector.ODDS_API_SOURCE)
        self.assertTrue(retrieved.frame.equals(frame))
        self.assertEqual(retrieved.request_metadata["requests_remaining"], 498)
        self.assertEqual(retrieved.request_metadata["request_cost"], 2)
        self.assertNotIn("secret-fixture", json.dumps(retrieved.request_metadata))
        fetch.assert_called_once()

    def test_api_request_metadata_rejects_credentials(self):
        report = {
            "source": collector.ODDS_API_SOURCE,
            "source_request": {
                "sport": "mma_mixed_martial_arts",
                "market": "h2h",
                "regions": "us,us2",
                "odds_format": "american",
                "requests_remaining": 499,
                "requests_used": 1,
                "request_cost": 2,
                "api_key": "must-never-be-published",
            },
        }
        with self.assertRaisesRegex(collector.CaptureError, "credential"):
            collector._validate_source_request(report)

    def test_post_commencement_retry_is_a_successful_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "capture_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "event_id": "event-card",
                        "event_url": "https://ufcstats.test/event-card",
                        "timing_precision": "timestamp",
                        "event_start_utc": "2026-08-14T16:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(collector, "REPORT_PATH", report_path):
                with self.assertRaisesRegex(collector.CaptureSkipped, "commenced"):
                    collector._skip_if_prior_capture_card_started(
                        {
                            "event_id": "event-card",
                            "event_url": "https://ufcstats.test/event-card",
                        },
                        CAPTURE_STARTED,
                    )

    def test_workflows_stage_the_complete_paper_only_market_contract(self):
        collector_workflow = (
            REPO_ROOT / ".github" / "workflows" / "collect-market-snapshot.yml"
        ).read_text(encoding="utf-8")
        for filename in (
            "quote_snapshots.csv",
            "quote_snapshots.jsonl",
            "forecast_captures.csv",
            "forecast_captures.jsonl",
            "quote_source_metadata.csv",
            "quote_source_metadata.jsonl",
            "paper_decisions.csv",
            "paper_decisions.jsonl",
            "paper_settlements.csv",
            "paper_settlements.jsonl",
            "performance_report.json",
            "capture_report.json",
        ):
            self.assertIn(filename, collector_workflow)
        self.assertIn('cron: "17 12,18 * * 4"', collector_workflow)
        self.assertIn('cron: "17 12,18,23 * * 5"', collector_workflow)
        self.assertIn('cron: "17 9,12,15,18 * * 6"', collector_workflow)
        self.assertNotIn("git add .", collector_workflow)

        updater_workflow = (
            REPO_ROOT / ".github" / "workflows" / "update-data.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("python update_market_performance.py", updater_workflow)
        self.assertIn("paper_settlements.jsonl", updater_workflow)
        self.assertIn("performance_report.json", updater_workflow)


if __name__ == "__main__":
    unittest.main()
