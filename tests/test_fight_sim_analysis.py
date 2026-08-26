from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.analysis import render_analysis_report, write_analysis_report  # noqa: E402
from fight_sim.research import execute_analyze  # noqa: E402


class FightSimulationAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aggregate = {
            "schema_version": "fight-sim.v1",
            "matchup_id": "matchup-test",
            "scheduled_rounds": 3,
            "total_paths": 100,
            "bootstrap_members": 2,
            "outcome_counts": {"red_ko_tko": 30, "blue_decision": 70},
            "outcome_probabilities": {"red_ko_tko": 0.3, "blue_decision": 0.7},
            "duration_bins": [{"upper_seconds": 300, "count": 30}, {"upper_seconds": 900, "count": 70}],
            "survival": [
                {"seconds": 0.0, "probability": 1.0},
                {"seconds": 450.0, "probability": 0.7},
                {"seconds": 900.0, "probability": 0.0},
            ],
            "uncertainty": [
                {
                    "metric": "red_win",
                    "estimate": 0.3,
                    "process_mcse": 0.02,
                    "parameter_p025": 0.2,
                    "parameter_median": 0.3,
                    "parameter_p975": 0.4,
                }
            ],
            "total_lines": [
                {
                    "half_rounds": 1.5,
                    "threshold_seconds": 450.0,
                    "over": 70,
                    "under": 30,
                    "push": 0,
                    "no_action": 0,
                }
            ],
            "statistic_summaries": [
                {"statistic": "red.sig_strikes_landed", "mean": 40.0, "p05": 10.0, "median": 38.0, "p95": 80.0}
            ],
        }

    def test_report_is_self_contained_and_labels_uncertainty(self):
        html = render_analysis_report(
            self.aggregate,
            run_spec={"warnings": ["Sparse fighter history"]},
            evaluation={"metrics": {"winner_log_loss": {"simulation": 0.62}}},
            traces=[{"simulation_index": 3, "selection_reasons": ["red_ko_tko"], "events": []}],
        )

        self.assertIn("Candidate-only local research report", html)
        self.assertIn("Process MCSE", html)
        self.assertIn("Parameter 2.5%", html)
        self.assertIn("Sparse fighter history", html)
        self.assertIn("Simulation 3", html)
        self.assertIn('type="application/json"', html)
        self.assertNotIn("https://", html)

        payload_text = html.split(
            '<script type="application/json" id="fight-sim-report-data">', 1
        )[1].split("</script>", 1)[0]
        payload = json.loads(payload_text)
        self.assertEqual(payload["aggregate"]["total_paths"], 100)

    def test_report_write_is_atomic_and_creates_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "report.html"
            written = write_analysis_report(target, self.aggregate)
            self.assertEqual(written, target.resolve())
            self.assertTrue(written.exists())
            self.assertIn("matchup-test", written.read_text(encoding="utf-8"))
            self.assertFalse(list(written.parent.glob("*.tmp")))

    def test_analyze_accepts_a_chronological_backtest_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "backtest_report.json"
            source.write_text(
                json.dumps(
                    {
                        "primary_metric": "joint_side_by_method_log_loss",
                        "folds": [],
                        "aggregate": {
                            "primary_joint_side_method_log_loss": 0.91,
                            "winner": {
                                "log_loss": 0.64,
                                "brier": 0.22,
                                "calibration_intercept": 0.01,
                                "calibration_slope": 0.97,
                            },
                        },
                        "comparisons": {
                            "population_joint": {
                                "coverage": 1.0,
                                "paired_event_card_interval": {
                                    "interval_p025": -0.03,
                                    "interval_p975": 0.01,
                                },
                            }
                        },
                        "coverage_warnings": ["timestamped_market_coverage_below_half"],
                    }
                ),
                encoding="utf-8",
            )
            target = root / "backtest.html"
            execute_analyze(source, output=target)
            html = target.read_text(encoding="utf-8")
            self.assertIn("Fight simulation chronological backtest", html)
            self.assertIn("winner.calibration_slope", html)
            self.assertIn("population_joint.coverage", html)
            self.assertIn("timestamped_market_coverage_below_half", html)


if __name__ == "__main__":
    unittest.main()
