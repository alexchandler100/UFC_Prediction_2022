from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.aggregate import aggregate_paths, validate_aggregate_coherence  # noqa: E402
from fight_sim.domain import (  # noqa: E402
    DecisionType,
    FightResult,
    FighterStats,
    OutcomeMethod,
    Side,
    SimulationPath,
)
from fight_sim.evaluation import (  # noqa: E402
    BacktestConfig,
    evaluate_simulation_ledger,
    run_chronological_backtest,
)


def _stats(
    significant_strikes: int,
    *,
    knockdowns: int,
    takedowns: int,
    submission_attempts: int,
    control_seconds: int,
) -> FighterStats:
    return FighterStats(
        strike_attempts=significant_strikes,
        strikes_landed=significant_strikes,
        significant_strike_attempts=significant_strikes,
        significant_strikes_landed=significant_strikes,
        head_landed=significant_strikes,
        distance_landed=significant_strikes,
        knockdowns=knockdowns,
        takedown_attempts=takedowns,
        takedowns_landed=takedowns,
        submission_attempts=submission_attempts,
        control_time_us=control_seconds * 1_000_000,
    )


def _path(
    member: int,
    index: int,
    red_significant_strikes: int,
) -> SimulationPath:
    return SimulationPath(
        matchup_id="red--blue",
        scheduled_rounds=3,
        bootstrap_member=member,
        simulation_index=index,
        result=FightResult(
            winner=Side.RED,
            method=OutcomeMethod.DECISION,
            round_number=3,
            fight_time_us=900_000_000,
            round_time_us=300_000_000,
            reason="test decision",
            decision_type=DecisionType.UNANIMOUS,
        ),
        red_stats=_stats(
            red_significant_strikes,
            knockdowns=member,
            takedowns=index,
            submission_attempts=member + index,
            control_seconds=10 * red_significant_strikes,
        ),
        blue_stats=_stats(
            red_significant_strikes + 1,
            knockdowns=index,
            takedowns=member,
            submission_attempts=member + index + 1,
            control_seconds=5 * red_significant_strikes,
        ),
        final_state_hash="0" * 64,
        phase_time_us=(
            ("distance", 600_000_000),
            ("clinch", 100_000_000),
            ("ground", 150_000_000),
            ("scramble", 50_000_000),
        ),
    )


def _forecast():
    return aggregate_paths(
        (
            _path(0, 0, 1),
            _path(0, 1, 3),
            _path(1, 0, 5),
            _path(1, 1, 7),
        ),
        3,
    )


class StatisticDistributionAggregationTests(unittest.TestCase):
    def test_exact_aggregate_and_member_counts_are_authoritative(self):
        forecast = _forecast()
        validate_aggregate_coherence(forecast)

        red = next(
            item
            for item in forecast.statistic_distributions
            if item.statistic == "red_significant_strikes"
        )
        self.assertEqual(
            [(item.value, item.count) for item in red.counts],
            [(1.0, 1), (3.0, 1), (5.0, 1), (7.0, 1)],
        )
        distance_time = next(
            item
            for item in forecast.statistic_distributions
            if item.statistic == "distance_time_seconds"
        )
        self.assertEqual(
            [(item.value, item.count) for item in distance_time.counts],
            [(600.0, 4)],
        )
        total_strikes = next(
            item
            for item in forecast.statistic_distributions
            if item.statistic == "total_significant_strikes"
        )
        self.assertEqual(
            [(item.value, item.count) for item in total_strikes.counts],
            [(3.0, 1), (7.0, 1), (11.0, 1), (15.0, 1)],
        )
        strike_differential = next(
            item
            for item in forecast.statistic_distributions
            if item.statistic == "significant_strike_differential"
        )
        self.assertEqual(
            [(item.value, item.count) for item in strike_differential.counts],
            [(-1.0, 4)],
        )
        member_zero = next(
            item
            for item in forecast.bootstrap_statistic_distributions
            if item.bootstrap_member == 0
            and item.statistic == "red_significant_strikes"
        )
        self.assertEqual(
            [(item.value, item.count) for item in member_zero.counts],
            [(1.0, 1), (3.0, 1)],
        )

        uncertainty = next(
            item
            for item in forecast.statistic_uncertainty
            if item.statistic == "red_significant_strikes"
        )
        self.assertEqual(uncertainty.conditional_means, ((0, 2.0), (1, 6.0)))
        self.assertAlmostEqual(uncertainty.estimate_mean, 4.0)
        self.assertAlmostEqual(uncertainty.process_mcse_mean, 0.5)
        self.assertAlmostEqual(uncertainty.parameter_model_median, 4.0)

        serialized = forecast.to_dict()
        self.assertIn("statistic_distributions", serialized)
        self.assertIn("bootstrap_statistic_distributions", serialized)
        self.assertIn("statistic_uncertainty", serialized)
        self.assertEqual(serialized["outcome_counts"], {"red_decision": 4})


class StatisticDistributionEvaluationTests(unittest.TestCase):
    def test_crps_coverage_integrated_brier_and_available_total(self):
        ledger = pd.DataFrame(
            [
                {
                    "event_id": "event",
                    "fight_id": "fight",
                    "date": "2026-08-01",
                    "actual_outcome": "red_decision",
                    "actual_duration_seconds": 900.0,
                    "actual_red_significant_strikes": 4.0,
                    "market_total_line_rounds": 2.5,
                    "forecast": _forecast(),
                }
            ]
        )

        metrics = evaluate_simulation_ledger(ledger)

        # Uniform support {1, 3, 5, 7} at y=4 has CRPS
        # 2 - 0.5 * 2.5 = 0.75.
        red = metrics["count_distribution_predictive_checks"][
            "red_significant_strikes"
        ]
        self.assertEqual(red["n"], 1)
        self.assertAlmostEqual(red["count_distribution_crps"], 0.75)
        self.assertAlmostEqual(red["interval_90_coverage"], 1.0)
        self.assertAlmostEqual(red["predictive_minus_observed_mean"], 0.0)
        self.assertEqual(red["interval_50_coverage"], 1.0)
        posterior = metrics["posterior_predictive_checks"][
            "red_significant_strikes"
        ]
        self.assertEqual(posterior["n"], 1)
        self.assertEqual(len(posterior["pit_histogram_10"]), 10)
        self.assertIsNone(posterior["pit_cvm_statistic"])
        self.assertEqual(metrics["posterior_predictive_diagnostic_rows"], 2)
        self.assertAlmostEqual(metrics["duration_integrated_brier"], 0.0)
        self.assertAlmostEqual(metrics["duration_crps_seconds"], 0.0)
        self.assertLess(metrics["available_totals_log_loss"], 2e-12)
        self.assertEqual(metrics["available_totals_scored_fights"], 1)

    def test_old_publication_without_exact_stat_counts_remains_scorable(self):
        forecast = _forecast().to_dict()
        forecast.pop("statistic_distributions")
        forecast.pop("bootstrap_statistic_distributions")
        forecast.pop("statistic_uncertainty")
        ledger = pd.DataFrame(
            [
                {
                    "event_id": "event",
                    "fight_id": "fight",
                    "date": "2026-08-01",
                    "actual_outcome": "red_decision",
                    "actual_duration_seconds": 900.0,
                    "actual_red_significant_strikes": 4.0,
                    "forecast": forecast,
                }
            ]
        )

        metrics = evaluate_simulation_ledger(ledger)

        self.assertEqual(metrics["count_distribution_predictive_checks"], {})
        self.assertIn("duration_seconds", metrics["posterior_predictive_checks"])
        self.assertIsNone(metrics["available_totals_log_loss"])
        self.assertAlmostEqual(metrics["duration_integrated_brier"], 0.0)

    def test_backtest_reports_zero_coverage_and_joint_forecast_comparators(self):
        physical = pd.DataFrame(
            [
                {
                    "event_id": f"event-{year}",
                    "fight_id": f"fight-{year}-{index}",
                    "date": f"{year}-06-0{index + 1}",
                    "actual_outcome": (
                        "red_decision" if index == 0 else "blue_decision"
                    ),
                    "market_red_win_probability": None,
                }
                for year in (2019, 2020)
                for index in range(2)
            ]
        )

        def forecast(red_probability: float) -> dict[str, object]:
            red_count = int(round(100 * red_probability))
            return {
                "total_paths": 100,
                "outcome_counts": {
                    "red_decision": red_count,
                    "blue_decision": 100 - red_count,
                },
                "outcome_probabilities": {
                    "red_decision": red_probability,
                    "blue_decision": 1.0 - red_probability,
                },
            }

        def predict(train, test, cutoff):
            del train, cutoff
            rows = []
            for row in test.to_dict("records"):
                red_won = str(row["actual_outcome"]).startswith("red_")
                rows.append(
                    {
                        "fight_id": row["fight_id"],
                        "forecast": forecast(0.8 if red_won else 0.2),
                        # A bare outcome mapping and a full forecast mapping are
                        # both supported causal-comparator contracts.
                        "population_forecast": {
                            "red_decision": 0.6,
                            "blue_decision": 0.4,
                        },
                        "division_forecast": forecast(0.7),
                        "outcome_model_forecast": forecast(0.65),
                    }
                )
            return pd.DataFrame(rows)

        _, report = run_chronological_backtest(
            physical,
            predict,
            config=BacktestConfig(
                first_test_year=2020,
                last_test_year=2020,
                min_training_fights=2,
                card_bootstrap_replicates=20,
                random_seed=5,
            ),
        )

        market = report.comparisons["timestamped_market"]
        self.assertEqual(market["n_covered"], 0)
        self.assertEqual(market["coverage"], 0.0)
        self.assertIsNone(market["paired_event_card_interval"])
        for name in (
            "competing_risk_joint",
            "population_joint",
            "division_joint",
        ):
            comparison = report.comparisons[name]
            self.assertEqual(comparison["metric"], "joint_side_by_method_log_loss")
            self.assertEqual(comparison["n_covered"], 2)
            self.assertEqual(comparison["coverage"], 1.0)
            self.assertEqual(comparison["baseline"]["n"], 2)
            self.assertLess(
                comparison["paired_event_card_interval"][
                    "challenger_minus_baseline_log_loss"
                ],
                0.0,
            )

    def test_market_totals_compare_log_loss_and_brier_on_identical_rows(self):
        rows = []
        for year in (2019, 2020):
            for index, (duration, market_probability) in enumerate(
                (
                    (900.0, 0.55),
                    (600.0, 0.55),
                    (750.0, 0.55),
                    (900.0, None),
                )
            ):
                rows.append(
                    {
                        "event_id": f"event-{year}-{index // 2}",
                        "fight_id": f"fight-{year}-{index}",
                        "date": f"{year}-06-{index + 1:02d}",
                        "actual_outcome": "red_decision",
                        "actual_duration_seconds": duration,
                        "market_total_line_rounds": 2.5,
                        "market_total_over_probability": market_probability,
                    }
                )
        physical = pd.DataFrame(rows)

        def forecast(over_probability: float) -> dict[str, object]:
            over = int(round(100 * over_probability))
            return {
                "total_paths": 100,
                "outcome_counts": {"red_decision": 100},
                "outcome_probabilities": {"red_decision": 1.0},
                "total_lines": [
                    {
                        "half_rounds": 2.5,
                        "threshold_seconds": 750.0,
                        "over": over,
                        "under": 100 - over,
                        "push": 0,
                        "no_action": 0,
                    }
                ],
            }

        def predict(train, test, cutoff):
            del train, cutoff
            return pd.DataFrame(
                [
                    {
                        "fight_id": row["fight_id"],
                        "forecast": forecast(
                            0.8
                            if float(row["actual_duration_seconds"]) > 750.0
                            else 0.2
                        ),
                    }
                    for row in test.to_dict("records")
                ]
            )

        _, report = run_chronological_backtest(
            physical,
            predict,
            config=BacktestConfig(
                first_test_year=2020,
                last_test_year=2020,
                min_training_fights=2,
                card_bootstrap_replicates=100,
                random_seed=17,
            ),
        )

        comparison = report.comparisons["timestamped_market_totals"]
        self.assertEqual(comparison["n_eligible"], 2)
        self.assertEqual(comparison["n_covered"], 2)
        self.assertEqual(comparison["coverage"], 1.0)
        self.assertEqual(comparison["omitted_push_or_no_contest"], 1)
        self.assertAlmostEqual(
            comparison["baseline"]["log_loss"],
            (-math.log(0.55) - math.log(0.45)) / 2.0,
        )
        self.assertAlmostEqual(comparison["baseline"]["brier"], 0.2525)
        self.assertAlmostEqual(
            comparison["simulation_same_fights"]["log_loss"],
            -math.log(0.8),
        )
        self.assertAlmostEqual(
            comparison["simulation_same_fights"]["brier"], 0.04
        )
        paired = comparison["paired_event_card_interval"]
        self.assertLess(paired["challenger_minus_baseline_log_loss"], 0.0)
        self.assertLess(paired["log_loss_interval_p975"], 0.0)
        self.assertLess(paired["challenger_minus_baseline_brier"], 0.0)
        self.assertLess(paired["brier_interval_p975"], 0.0)


if __name__ == "__main__":
    unittest.main()
