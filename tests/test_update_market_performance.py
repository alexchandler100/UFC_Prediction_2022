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
    BayesianFilteredDecision,
    BayesianFilteredDecisionStore,
    ForecastCapture,
    ForecastCaptureStore,
    PaperDecision,
    PaperDecisionStore,
    PaperSettlementStore,
    QuoteSnapshot,
    QuoteSnapshotStore,
    QuoteSourceMetadata,
    QuoteSourceMetadataStore,
    TotalRoundsForecastCapture,
    TotalRoundsPaperDecisionStore,
    TotalRoundsPaperSettlementStore,
    TotalRoundsQuoteSnapshot,
    TotalRoundsQuoteStore,
    build_locked_total_round_decisions,
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

    def test_bayesian_shadow_history_is_scored_without_enabling_execution(self):
        rows = []
        for index, action in enumerate(("fighter", "opponent"), start=1):
            rows.append(
                {
                    "bayesian decision policy": "bayesian-moneyline-shadow-v1",
                    "bayesian model id": f"model-{index}",
                    "bayesian posterior mean": 0.60,
                    "bayesian paper action": action,
                    "bayesian paper threshold met": True,
                    "bayesian candidate odds": +150,
                    "bayesian candidate book": "BookA",
                    "bayesian candidate selection": f"Selection {index}",
                    "bayesian probability positive ev": 0.85,
                    "bayesian posterior mean ev": 0.50,
                    "bayesian ev lower": -0.10,
                    "bayesian ev upper": 1.00,
                    "forecast status": "completed",
                    "actual result": "W",
                    "fighter id": f"fighter-{index}",
                    "opponent id": f"opponent-{index}",
                    "event id": f"event-{index}",
                    "fight id": f"fight-{index}",
                }
            )
        report = updater._bayesian_prediction_history_performance(
            pd.DataFrame(rows)
        )
        self.assertEqual(report["scored_forecasts"], 2)
        self.assertEqual(report["settled_shadow_selections"], 2)
        self.assertEqual(report["wins"], 1)
        self.assertEqual(report["losses"], 1)
        self.assertAlmostEqual(report["hypothetical_profit_units"], 0.5)
        self.assertFalse(report["execution_enabled"])
        self.assertFalse(
            report["promotion_gate"]["immutable_t24_ledger_requirement_met"]
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
        filtered_decision = BayesianFilteredDecision.create(
            decision,
            source_vegas_sha256="b" * 64,
            bayesian_artifact_sha256="c" * 64,
            bayesian_model_id="bayes-one",
            bayesian_status="paper_only_challenger",
            credible_level=0.9,
            fighter_posterior_mean=0.55,
            fighter_posterior_median=0.55,
            fighter_probability_lower=0.5254727972575093,
            fighter_probability_upper=0.5742865237557964,
            fighter_calibrated_logit_location=0.2006706954621514,
            calibrated_logit_scale=0.06,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_root = root / "market"
            market_root.mkdir()
            external_root = root / "external"
            external_root.mkdir()
            prediction_history_path = external_root / "prediction_history.json"
            pd.DataFrame().to_json(prediction_history_path)
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
                "PREDICTION_HISTORY_PATH": prediction_history_path,
                "SIMULATION_FORECAST_PATH": external_root / "simulation_forecasts.json",
                "QUOTE_CSV_PATH": market_root / "quote_snapshots.csv",
                "QUOTE_JSONL_PATH": market_root / "quote_snapshots.jsonl",
                "SOURCE_METADATA_CSV_PATH": market_root / "quote_source_metadata.csv",
                "SOURCE_METADATA_JSONL_PATH": market_root / "quote_source_metadata.jsonl",
                "DECISION_CSV_PATH": market_root / "paper_decisions.csv",
                "DECISION_JSONL_PATH": market_root / "paper_decisions.jsonl",
                "BAYESIAN_FILTER_DECISION_CSV_PATH": market_root / "bayesian_filtered_paper_decisions.csv",
                "BAYESIAN_FILTER_DECISION_JSONL_PATH": market_root / "bayesian_filtered_paper_decisions.jsonl",
                "SETTLEMENT_CSV_PATH": market_root / "paper_settlements.csv",
                "SETTLEMENT_JSONL_PATH": market_root / "paper_settlements.jsonl",
                "SIMULATION_COMPARISON_CSV_PATH": market_root / "simulation_comparisons.csv",
                "SIMULATION_COMPARISON_JSONL_PATH": market_root / "simulation_comparisons.jsonl",
                "TOTAL_ROUNDS_QUOTE_CSV_PATH": market_root / "total_round_quote_snapshots.csv",
                "TOTAL_ROUNDS_QUOTE_JSONL_PATH": market_root / "total_round_quote_snapshots.jsonl",
                "TOTAL_ROUNDS_DECISION_CSV_PATH": market_root / "total_round_paper_decisions.csv",
                "TOTAL_ROUNDS_DECISION_JSONL_PATH": market_root / "total_round_paper_decisions.jsonl",
                "TOTAL_ROUNDS_SETTLEMENT_CSV_PATH": market_root / "total_round_paper_settlements.csv",
                "TOTAL_ROUNDS_SETTLEMENT_JSONL_PATH": market_root / "total_round_paper_settlements.jsonl",
                "REPORT_PATH": market_root / "performance_report.json",
            }
            QuoteSnapshotStore(
                paths["QUOTE_CSV_PATH"], paths["QUOTE_JSONL_PATH"]
            ).append([*decision_quotes, later_target])
            PaperDecisionStore(
                paths["DECISION_CSV_PATH"], paths["DECISION_JSONL_PATH"]
            ).append([decision])
            BayesianFilteredDecisionStore(
                paths["BAYESIAN_FILTER_DECISION_CSV_PATH"],
                paths["BAYESIAN_FILTER_DECISION_JSONL_PATH"],
            ).append([filtered_decision])
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
            self.assertEqual(persisted["schema_version"], 5)
            self.assertEqual(
                persisted["prospective_model_market_comparison"]["status"],
                "collecting_results",
            )
            self.assertEqual(
                persisted["prospective_model_market_comparison"]["scored_fights"],
                0,
            )
            self.assertEqual(
                persisted["prospective_simulation_comparison"]["scored_fights"],
                0,
            )
            self.assertIn("entry_timing_experiment", persisted)
            self.assertIn("total_rounds", persisted)
            self.assertEqual(
                persisted["bayesian_filtered_moneyline_policy"]
                ["bayesian_filtered_policy"]["selections"],
                1,
            )
            self.assertEqual(
                persisted["bayesian_filtered_moneyline_policy"]
                ["bayesian_filtered_policy"]["hypothetical_roi"],
                2.0,
            )
            self.assertEqual(persisted["total_rounds"]["decisions"], 0)
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

    def test_total_round_decision_settles_with_same_book_clv(self):
        def total_quote(
            *, capture, observed, updated, book, book_key, over, under
        ):
            return TotalRoundsQuoteSnapshot.create(
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
                source="the-odds-api.com",
                source_event_id="source-event-one",
                source_book_key=book_key,
                source_quote_updated_at_utc=updated,
                source_commence_time_utc="2026-01-10T12:30:00Z",
                book=book,
                line=2.5,
                over_moneyline=over,
                under_moneyline=under,
                source_payload={"capture": capture},
            )

        decision_quotes = (
            total_quote(
                capture="tc1", observed="2026-01-09T12:00:00Z",
                updated="2026-01-09T11:59:30Z", book="Target",
                book_key="target", over=120, under=-140,
            ),
            total_quote(
                capture="tc1", observed="2026-01-09T12:00:00Z",
                updated="2026-01-09T11:59:30Z", book="Book A",
                book_key="book-a", over=-110, under=-110,
            ),
            total_quote(
                capture="tc1", observed="2026-01-09T12:00:00Z",
                updated="2026-01-09T11:59:30Z", book="Book B",
                book_key="book-b", over=-110, under=-110,
            ),
        )
        later_target = total_quote(
            capture="tc2", observed="2026-01-10T10:00:00Z",
            updated="2026-01-10T09:59:30Z", book="Target",
            book_key="target", over=100, under=-120,
        )
        forecast = TotalRoundsForecastCapture.create(
            capture_id="tc1",
            event_id="event-one",
            fighter_id="fighter-a",
            opponent_id="fighter-b",
            fighter_name="Fighter A",
            opponent_name="Fighter B",
            event_date="2026-01-10",
            timing_precision="timestamp",
            event_start_utc="2026-01-10T12:00:00Z",
            forecast_issued_at_utc="2026-01-08T12:00:00Z",
            scheduled_rounds=3,
            schedule_basis="fixture",
            line=2.5,
            over_probability=0.70,
            model_id="outcome-model",
            model_version="candidate-v1",
            model_trained_through="2026-01-03",
            source_commit_sha="a" * 40,
            source_publication_sha256="b" * 64,
        )
        decision = build_locked_total_round_decisions(
            decision_quotes, (forecast,)
        ).decisions[0]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_root = root / "market"
            market_root.mkdir()
            external_root = root / "external"
            external_root.mkdir()
            prediction_history_path = external_root / "prediction_history.json"
            pd.DataFrame().to_json(prediction_history_path)
            raw_path = root / "raw.csv"
            pd.DataFrame(
                [
                    {
                        "event_url": "https://ufcstats.test/event-one",
                        "fight_url": "https://ufcstats.test/fight-one",
                        "fighter_url": "https://ufcstats.test/fighter-a",
                        "opponent_url": "https://ufcstats.test/fighter-b",
                        "result": "W",
                        "total_fight_time": 800.0,
                    },
                    {
                        "event_url": "https://ufcstats.test/event-one",
                        "fight_url": "https://ufcstats.test/fight-one",
                        "fighter_url": "https://ufcstats.test/fighter-b",
                        "opponent_url": "https://ufcstats.test/fighter-a",
                        "result": "L",
                        "total_fight_time": 800.0,
                    },
                ]
            ).to_csv(raw_path, index=False)
            paths = {
                "RAW_PATH": raw_path,
                "PREDICTION_HISTORY_PATH": prediction_history_path,
                "SIMULATION_FORECAST_PATH": external_root / "simulation_forecasts.json",
                "QUOTE_CSV_PATH": market_root / "quote_snapshots.csv",
                "QUOTE_JSONL_PATH": market_root / "quote_snapshots.jsonl",
                "SOURCE_METADATA_CSV_PATH": market_root / "quote_source_metadata.csv",
                "SOURCE_METADATA_JSONL_PATH": market_root / "quote_source_metadata.jsonl",
                "DECISION_CSV_PATH": market_root / "paper_decisions.csv",
                "DECISION_JSONL_PATH": market_root / "paper_decisions.jsonl",
                "BAYESIAN_FILTER_DECISION_CSV_PATH": market_root / "bayesian_filtered_paper_decisions.csv",
                "BAYESIAN_FILTER_DECISION_JSONL_PATH": market_root / "bayesian_filtered_paper_decisions.jsonl",
                "SETTLEMENT_CSV_PATH": market_root / "paper_settlements.csv",
                "SETTLEMENT_JSONL_PATH": market_root / "paper_settlements.jsonl",
                "SIMULATION_COMPARISON_CSV_PATH": market_root / "simulation_comparisons.csv",
                "SIMULATION_COMPARISON_JSONL_PATH": market_root / "simulation_comparisons.jsonl",
                "TOTAL_ROUNDS_QUOTE_CSV_PATH": market_root / "total_round_quote_snapshots.csv",
                "TOTAL_ROUNDS_QUOTE_JSONL_PATH": market_root / "total_round_quote_snapshots.jsonl",
                "TOTAL_ROUNDS_DECISION_CSV_PATH": market_root / "total_round_paper_decisions.csv",
                "TOTAL_ROUNDS_DECISION_JSONL_PATH": market_root / "total_round_paper_decisions.jsonl",
                "TOTAL_ROUNDS_SETTLEMENT_CSV_PATH": market_root / "total_round_paper_settlements.csv",
                "TOTAL_ROUNDS_SETTLEMENT_JSONL_PATH": market_root / "total_round_paper_settlements.jsonl",
                "REPORT_PATH": market_root / "performance_report.json",
            }
            TotalRoundsQuoteStore(
                paths["TOTAL_ROUNDS_QUOTE_CSV_PATH"],
                paths["TOTAL_ROUNDS_QUOTE_JSONL_PATH"],
            ).append((*decision_quotes, later_target))
            TotalRoundsPaperDecisionStore(
                paths["TOTAL_ROUNDS_DECISION_CSV_PATH"],
                paths["TOTAL_ROUNDS_DECISION_JSONL_PATH"],
            ).append((decision,))
            patches = [patch.object(updater, key, value) for key, value in paths.items()]
            for active in patches:
                active.start()
            try:
                first = updater.update_market_performance()
                second = updater.update_market_performance()
            finally:
                for active in reversed(patches):
                    active.stop()

            self.assertEqual(first, second)
            settlements = TotalRoundsPaperSettlementStore(
                paths["TOTAL_ROUNDS_SETTLEMENT_CSV_PATH"],
                paths["TOTAL_ROUNDS_SETTLEMENT_JSONL_PATH"],
            ).read()
            self.assertEqual(len(settlements), 1)
            self.assertEqual(settlements[0].settlement_status, "paper_win")
            totals = first["total_rounds"]
            self.assertEqual(totals["scored_forecasts"], 1)
            self.assertAlmostEqual(
                totals["official_strategy"]["hypothetical_roi"], 1.2
            )
            self.assertGreater(
                totals["latest_available_price_clv"]["mean_probability_edge"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
