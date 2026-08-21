import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import update_market_performance as updater  # noqa: E402
from market_tracker import (  # noqa: E402
    ForecastCapture,
    ForecastCaptureStore,
    PaperDecision,
    PaperDecisionStore,
    PaperSettlementStore,
    QuoteSnapshot,
    QuoteSnapshotStore,
    QuoteSourceMetadata,
    QuoteSourceMetadataStore,
    consensus_as_of,
)
from validate_data import validate_market_data  # noqa: E402
from market_tracker._common import canonical_hash  # noqa: E402


class MarketPerformanceTests(unittest.TestCase):
    @staticmethod
    def _quote(book, fighter_line, opponent_line, *, capture, observed):
        return QuoteSnapshot.create(
            capture_id=capture,
            event_id="event-one",
            fighter_id="fighter-a",
            opponent_id="fighter-b",
            fighter_name="Fighter A",
            opponent_name="Fighter B",
            event_date="2026-01-10",
            timing_precision="timestamp",
            event_start_utc="2026-01-10T12:00:00Z",
            observed_at_utc=observed,
            source="fixture",
            book=book,
            fighter_moneyline=fighter_line,
            opponent_moneyline=opponent_line,
            source_payload={"capture": capture},
        )

    def test_settles_once_and_publishes_reproducible_return_report(self):
        decision_quotes = [
            self._quote("BookA", -110, -110, capture="c1", observed="2026-01-09T12:00:00Z"),
            self._quote("BookB", -105, -115, capture="c1", observed="2026-01-09T12:00:00Z"),
            self._quote("BookC", -115, -105, capture="c1", observed="2026-01-09T12:00:00Z"),
            self._quote("Target", +200, -250, capture="c1", observed="2026-01-09T12:00:00Z"),
        ]
        later_target = self._quote(
            "Target", +150, -175, capture="c2", observed="2026-01-10T10:00:00Z"
        )
        market = consensus_as_of(
            decision_quotes,
            capture_id="c1",
            matchup_id=decision_quotes[0].matchup_id,
            as_of_utc="2026-01-09T12:00:00Z",
            min_books=3,
            exclude_books=("Target",),
        )
        forecast = ForecastCapture.create(
            capture_id="c1",
            event_id="event-one",
            fighter_id="fighter-a",
            opponent_id="fighter-b",
            fighter_name="Fighter A",
            opponent_name="Fighter B",
            event_date="2026-01-10",
            timing_precision="timestamp",
            event_start_utc="2026-01-10T12:00:00Z",
            forecast_issued_at_utc="2026-01-08T12:00:00Z",
            model_probability=0.60,
            model_id="model-one",
            model_version="fixture-v1",
            model_trained_through="2026-01-03",
            model_training_cutoff_precision="date",
            source_commit_sha="a" * 40,
        )
        decision = PaperDecision.create(
            market,
            decision_quotes[-1],
            forecast,
            selected_gamma=0.0,
            decision_issued_at_utc="2026-01-09T12:00:00Z",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_root = root / "market"
            market_root.mkdir()
            raw_path = root / "raw.csv"
            pd.DataFrame(
                [
                    {
                        "event_url": "https://ufcstats.test/event-one",
                        "fight_url": "https://ufcstats.test/fight-one",
                        "fighter_url": "https://ufcstats.test/fighter-a",
                        "opponent_url": "https://ufcstats.test/fighter-b",
                        "result": "W",
                    },
                    {
                        "event_url": "https://ufcstats.test/event-one",
                        "fight_url": "https://ufcstats.test/fight-one",
                        "fighter_url": "https://ufcstats.test/fighter-b",
                        "opponent_url": "https://ufcstats.test/fighter-a",
                        "result": "L",
                    },
                    {
                        "event_url": "https://ufcstats.test/tournament-event",
                        "fight_url": "https://ufcstats.test/tournament-fight-one",
                        "fighter_url": "https://ufcstats.test/fighter-c",
                        "opponent_url": "https://ufcstats.test/fighter-d",
                        "result": "W",
                    },
                    {
                        "event_url": "https://ufcstats.test/tournament-event",
                        "fight_url": "https://ufcstats.test/tournament-fight-one",
                        "fighter_url": "https://ufcstats.test/fighter-d",
                        "opponent_url": "https://ufcstats.test/fighter-c",
                        "result": "L",
                    },
                    {
                        "event_url": "https://ufcstats.test/tournament-event",
                        "fight_url": "https://ufcstats.test/tournament-fight-two",
                        "fighter_url": "https://ufcstats.test/fighter-c",
                        "opponent_url": "https://ufcstats.test/fighter-d",
                        "result": "NC",
                    },
                    {
                        "event_url": "https://ufcstats.test/tournament-event",
                        "fight_url": "https://ufcstats.test/tournament-fight-two",
                        "fighter_url": "https://ufcstats.test/fighter-d",
                        "opponent_url": "https://ufcstats.test/fighter-c",
                        "result": "NC",
                    },
                ]
            ).to_csv(raw_path, index=False)
            paths = {
                "RAW_PATH": raw_path,
                "QUOTE_CSV_PATH": market_root / "quote_snapshots.csv",
                "QUOTE_JSONL_PATH": market_root / "quote_snapshots.jsonl",
                "SOURCE_METADATA_CSV_PATH": market_root / "quote_source_metadata.csv",
                "SOURCE_METADATA_JSONL_PATH": market_root / "quote_source_metadata.jsonl",
                "DECISION_CSV_PATH": market_root / "paper_decisions.csv",
                "DECISION_JSONL_PATH": market_root / "paper_decisions.jsonl",
                "SETTLEMENT_CSV_PATH": market_root / "paper_settlements.csv",
                "SETTLEMENT_JSONL_PATH": market_root / "paper_settlements.jsonl",
                "REPORT_PATH": market_root / "performance_report.json",
            }
            QuoteSnapshotStore(
                paths["QUOTE_CSV_PATH"], paths["QUOTE_JSONL_PATH"]
            ).append([*decision_quotes, later_target])
            PaperDecisionStore(
                paths["DECISION_CSV_PATH"], paths["DECISION_JSONL_PATH"]
            ).append([decision])
            ForecastCaptureStore(
                market_root / "forecast_captures.csv",
                market_root / "forecast_captures.jsonl",
            ).append([forecast])
            QuoteSourceMetadataStore(
                market_root / "quote_source_metadata.csv",
                market_root / "quote_source_metadata.jsonl",
            ).append(
                [
                    QuoteSourceMetadata.create(
                        quote,
                        source_book_key=quote.book.casefold(),
                        source_event_id="source-event-one",
                        source_quote_updated_at_utc="2026-01-09T11:55:00Z",
                        source_commence_time_utc="2026-01-10T12:00:00Z",
                    )
                    for quote in decision_quotes
                ]
            )
            patches = [patch.object(updater, key, value) for key, value in paths.items()]
            for active in patches:
                active.start()
            try:
                first = updater.update_market_performance()
                second = updater.update_market_performance()
            finally:
                for active in reversed(patches):
                    active.stop()

            settlements = PaperSettlementStore(
                paths["SETTLEMENT_CSV_PATH"], paths["SETTLEMENT_JSONL_PATH"]
            ).read()
            self.assertEqual(len(settlements), 1)
            self.assertEqual(settlements[0].settlement_status, "paper_win")
            self.assertEqual(first, second)
            self.assertEqual(first["paper_metrics"]["hypothetical_roi"], 2.0)
            self.assertEqual(first["ambiguous_historical_matchup_keys"], 1)
            self.assertEqual(first["ambiguous_result_decisions"], 0)
            self.assertEqual(first["forecast_comparators"]["paired_fights"], 1)
            self.assertEqual(
                first["forecast_comparators"]["blend_minus_market_log_loss"],
                0.0,
            )
            self.assertEqual(first["paper_return_interval"]["event_count"], 1)
            self.assertIsNone(first["paper_return_interval"]["ci_95_lower"])
            self.assertEqual(
                first["latest_available_price_clv"]["mean_probability_edge"],
                second["latest_available_price_clv"]["mean_probability_edge"],
            )
            persisted = json.loads(paths["REPORT_PATH"].read_text(encoding="utf-8"))
            self.assertEqual(persisted, second)
            self.assertFalse(persisted["execution_enabled"])
            self.assertEqual(persisted["schema_version"], 2)
            self.assertIn("entry_timing_experiment", persisted)
            validation = validate_market_data(market_root, required=True)
            self.assertEqual(validation.errors, [], validation.errors)

            tampered = dict(persisted)
            tampered.pop("report_sha256")
            tampered["paper_metrics"] = dict(tampered["paper_metrics"])
            tampered["paper_metrics"]["hypothetical_roi"] = -999.0
            tampered["report_sha256"] = canonical_hash(tampered)
            paths["REPORT_PATH"].write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            rejected = validate_market_data(market_root, required=True)
            self.assertTrue(
                any("metrics cannot be reproduced" in error for error in rejected.errors),
                rejected.errors,
            )


if __name__ == "__main__":
    unittest.main()
