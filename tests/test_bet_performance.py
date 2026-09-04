import json
import sys
import tempfile
import unittest
import pandas as pd
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_tracker._common import canonical_hash  # noqa: E402
from market_tracker.bankroll import (  # noqa: E402
    archive_upcoming_bet_board,
    bet_support_key,
    build_bet_performance_publication,
    kelly_fraction,
    validate_bet_performance_publication,
    validate_published_bet_archive,
)


class BetPerformanceTests(unittest.TestCase):
    def test_verified_total_assessment_and_allocated_stake_survive_archive_and_settlement(self):
        from bayesian_total_calibration import BayesianTotalCalibrator, fit_total_calibration
        rows = pd.DataFrame([{
            "event_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=7 * i),
            "event_id": f"event-{i:03d}", "fight_id": f"fight-{i:03d}",
            "line": 1.5, "model_probability": 0.35 if i % 2 else 0.65,
            "target": int(i % 5 not in {0, 1}),
        } for i in range(80)])
        assessment = BayesianTotalCalibrator(fit_total_calibration(rows)).assessment(0.62, "over", 1.5, 1000)
        self.assertEqual(assessment["status"], "available")
        probability = assessment["posterior_mean_probability"]
        allocation = min(0.01, assessment["recommended_fraction"])
        self.assertGreater(allocation, 0)
        bet = {
            "category": "Total rounds", "event_id": "event", "event_date": "2026-09-05",
            "matchup_id": "matchup", "fighter_id": "alpha", "opponent_id": "beta",
            "fighter_name": "Alpha", "opponent_name": "Beta", "selection": "Over 1.5 rounds",
            "side": "over", "target_book": "Book", "offered_moneyline": 1000,
            "estimated_win_probability": probability, "estimated_expected_return": probability * 11 - 1,
            "raw_estimated_win_probability": 0.62, "minimum_expected_return": 0.05,
            "threshold_met": True, "candidate_only": True, "probability_source": "candidate_duration_model",
            "observed_at_utc": "2026-09-01T12:00:00Z", "paper_only": True, "execution_enabled": False,
            "bayesian_kelly": assessment, "allocated_fraction": allocation,
            "allocation_policy_version": "capped-paper-portfolio-v1",
        }
        bet["bet_id"] = canonical_hash(bet)
        with tempfile.TemporaryDirectory() as directory:
            archive = archive_upcoming_bet_board({"publication_sha256": "a" * 64, "bets": [bet]}, Path(directory) / "archive.json")
        publication = build_bet_performance_publication(
            decisions=(), settlements=(), quotes=(), forecasts=(), archive=archive,
            outcomes={("event", "alpha", "beta"): 1}, durations={("event", "alpha", "beta"): 600.0},
        )
        validate_bet_performance_publication(publication)
        record = publication["records"][0]
        self.assertEqual(record["bayesian_kelly"], assessment)
        self.assertEqual(record["allocated_fraction"], allocation)
        self.assertEqual(record["allocation_policy_version"], bet["allocation_policy_version"])
        self.assertEqual(record["raw_estimated_win_probability"], 0.62)
        self.assertEqual(record["status"], "won")

    def test_kelly_fraction_uses_probability_and_american_odds(self):
        self.assertAlmostEqual(kelly_fraction(0.60, 100), 0.20)
        self.assertAlmostEqual(kelly_fraction(0.60, -150), 0.0)
        self.assertAlmostEqual(kelly_fraction(0.75, -150), 0.375)

    def test_archive_is_idempotent_and_rejects_tampering(self):
        bet = {
            "category": "Moneyline",
            "event_id": "event",
            "event_date": "2026-09-05",
            "matchup_id": "matchup",
            "fighter_id": "alpha",
            "opponent_id": "beta",
            "fighter_name": "Alpha",
            "opponent_name": "Beta",
            "selection": "Alpha",
            "side": "fighter",
            "target_book": "Book",
            "offered_moneyline": 120,
            "estimated_win_probability": 0.52,
            "estimated_expected_return": 0.144,
            "minimum_expected_return": 0.05,
            "threshold_met": True,
            "candidate_only": False,
            "probability_source": "test",
            "observed_at_utc": "2026-09-01T12:00:00Z",
            "paper_only": True,
            "execution_enabled": False,
        }
        bet["bet_id"] = canonical_hash(bet)
        board = {"publication_sha256": "a" * 64, "bets": [bet]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            first = archive_upcoming_bet_board(board, path)
            second = archive_upcoming_bet_board(board, path)
            self.assertEqual(first, second)
            self.assertEqual(second["snapshot_count"], 1)
            validate_published_bet_archive(json.loads(path.read_text()))
            changed = json.loads(path.read_text())
            changed["snapshots"][0]["offered_moneyline"] = 200
            with self.assertRaisesRegex(ValueError, "hash"):
                validate_published_bet_archive(changed)

    def test_publication_keeps_exact_official_result_and_kelly_input(self):
        decision = SimpleNamespace(
            decision_id="decision", paper_action="fighter",
            forecast_capture_id="forecast", reference_quote_id="quote",
            matchup_id="matchup", event_id="event", event_date="2026-09-05",
            event_start_utc="2026-09-05T20:00:00Z", fighter_id="alpha",
            opponent_id="beta", action_probability=0.60,
            action_reference_moneyline=100, fighter_expected_return=0.20,
            opponent_expected_return=-0.20, minimum_expected_return=0.05,
            market_as_of_utc="2026-09-04T20:00:00Z", selected_gamma=0.0,
            to_mapping=lambda: {"decision_id": "decision"},
        )
        settlement = SimpleNamespace(
            decision_id="decision", settlement_status="paper_win",
            hypothetical_profit_units=1.0,
            settled_at_utc="2026-09-06T01:00:00Z",
            to_mapping=lambda: {"settlement_id": "settlement"},
        )
        quote = SimpleNamespace(quote_id="quote", book="Book")
        forecast = SimpleNamespace(
            forecast_capture_id="forecast", fighter_name="Alpha",
            opponent_name="Beta", model_probability=0.64,
            forecast_issued_at_utc="2026-09-04T18:00:00Z",
        )
        archive = {
            "archive_sha256": "b" * 64,
            "snapshots": [],
        }
        publication = build_bet_performance_publication(
            decisions=(decision,), settlements=(settlement,), quotes=(quote,),
            forecasts=(forecast,), archive=archive,
        )
        validate_bet_performance_publication(publication)
        self.assertEqual(publication["official_wins"], 1)
        self.assertEqual(publication["official_losses"], 0)
        self.assertAlmostEqual(publication["records"][0]["kelly_fraction"], 0.20)
        self.assertAlmostEqual(
            publication["records"][0]["model_support_probability"], 0.64
        )
        self.assertEqual(publication["research_support"]["model_supported_records"], 1)
        self.assertEqual(publication["records"][0]["status"], "won")
        self.assertEqual(
            publication["records"][0]["bayesian_kelly"]["status"], "available"
        )
        self.assertIn("robust_bayesian_kelly", publication["staking_strategies"])
        self.assertEqual(
            publication["research_support"]["bayesian_kelly_supported_records"], 1
        )

    def test_research_support_is_attached_only_when_available_before_publication(self):
        bet = {
            "category": "Moneyline", "event_id": "event",
            "event_date": "2026-09-05", "matchup_id": "matchup",
            "fighter_id": "alpha", "opponent_id": "beta",
            "fighter_name": "Alpha", "opponent_name": "Beta",
            "selection": "Beta", "side": "opponent",
            "target_book": "Book", "offered_moneyline": 140,
            "estimated_win_probability": 0.48,
            "estimated_expected_return": 0.152,
            "minimum_expected_return": 0.05, "threshold_met": True,
            "candidate_only": False, "probability_source": "test",
            "observed_at_utc": "2026-09-01T12:00:00Z",
            "paper_only": True, "execution_enabled": False,
        }
        bet["bet_id"] = canonical_hash(bet)
        with tempfile.TemporaryDirectory() as directory:
            archive = archive_upcoming_bet_board(
                {"publication_sha256": "a" * 64, "bets": [bet]},
                Path(directory) / "archive.json",
            )
        key = bet_support_key(
            event_id="event", fighter_id="alpha", opponent_id="beta",
            category="Moneyline", side="opponent", selection="Beta",
        )
        publication = build_bet_performance_publication(
            decisions=(), settlements=(), quotes=(), forecasts=(),
            archive=archive,
            model_support={key: {
                "probability": 0.55, "source": "production_winner_model",
                "issued_at_utc": "2026-09-01T11:00:00Z",
            }},
            simulation_support={key: {
                "probability": 0.58, "source": "frozen_pre_event_monte_carlo",
                "issued_at_utc": "2026-09-01T13:00:00Z",
            }},
        )
        record = publication["records"][0]
        self.assertEqual(record["model_support_probability"], 0.55)
        self.assertIsNone(record["simulation_support_probability"])
        self.assertEqual(publication["research_support"]["model_supported_records"], 1)
        self.assertEqual(publication["research_support"]["simulation_supported_records"], 0)
        self.assertEqual(record["bayesian_kelly"]["status"], "unavailable")

    def test_total_ending_exactly_on_line_is_void(self):
        bet = {
            "category": "Total rounds", "event_id": "event",
            "event_date": "2026-09-05", "matchup_id": "matchup",
            "fighter_id": "alpha", "opponent_id": "beta",
            "fighter_name": "Alpha", "opponent_name": "Beta",
            "selection": "Under 2.5 rounds", "side": "under",
            "target_book": "Book", "offered_moneyline": 100,
            "estimated_win_probability": 0.55,
            "estimated_expected_return": 0.10,
            "minimum_expected_return": 0.05, "threshold_met": True,
            "candidate_only": True, "probability_source": "test",
            "observed_at_utc": "2026-09-01T12:00:00Z",
            "paper_only": True, "execution_enabled": False,
        }
        bet["bet_id"] = canonical_hash(bet)
        with tempfile.TemporaryDirectory() as directory:
            archive = archive_upcoming_bet_board(
                {"publication_sha256": "a" * 64, "bets": [bet]},
                Path(directory) / "archive.json",
            )
        key = ("event", "alpha", "beta")
        publication = build_bet_performance_publication(
            decisions=(), settlements=(), quotes=(), forecasts=(),
            archive=archive, outcomes={key: 1}, durations={key: 750.0},
        )
        self.assertEqual(publication["records"][0]["status"], "void")
        self.assertEqual(publication["records"][0]["unit_profit"], 0.0)
        self.assertEqual(
            publication["records"][0]["bayesian_kelly"]["status"], "unavailable"
        )


if __name__ == "__main__":
    unittest.main()
