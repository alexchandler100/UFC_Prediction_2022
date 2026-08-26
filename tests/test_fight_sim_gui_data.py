from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.gui_data import (  # noqa: E402
    RunBundleError,
    load_run_bundle,
    load_trace_timeline,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FightSimulationGuiDataTests(unittest.TestCase):
    def test_loads_authoritative_distributions_and_validation_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root / "aggregate.json",
                {
                    "candidate_only": True,
                    "aggregate": {
                        "matchup_id": "red-blue",
                        "total_paths": 4,
                        "statistic_distributions": [
                            {
                                "statistic": "red_knockdowns",
                                "total_paths": 4,
                                "counts": [
                                    {"value": 0.0, "count": 3},
                                    {"value": 1.0, "count": 1},
                                ],
                            }
                        ],
                        "duration_bins": [
                            {"upper_seconds": 5, "count": 1},
                            {"upper_seconds": 10, "count": 3},
                        ],
                    },
                },
            )
            _write(
                root / "specs.json",
                {
                    "specs": [
                        {
                            "red": {"fighter_name": "Red Name"},
                            "blue": {"fighter_name": "Blue Name"},
                        }
                    ]
                },
            )
            _write(root / "convergence.json", {"converged": True, "convergence": []})
            _write(
                root / "validation.json",
                {
                    "statistics": [
                        {
                            "statistic": "red_knockdowns",
                            "label": "Red Name: Knockdowns",
                            "observed": 1,
                            "unit": "count",
                            "mid_pit_percentile": 0.875,
                        },
                        {
                            "statistic": "duration_seconds",
                            "observed": 8,
                            "unit": "seconds",
                        },
                    ]
                },
            )
            bundle = load_run_bundle(root)
            self.assertEqual(bundle.red_name, "Red Name")
            self.assertEqual(bundle.blue_name, "Blue Name")
            self.assertEqual(bundle.total_paths, 4)
            knockdowns = bundle.distribution("red_knockdowns")
            self.assertEqual(knockdowns.counts, (3, 1))
            self.assertEqual(knockdowns.probabilities, (0.75, 0.25))
            self.assertEqual(knockdowns.observed, 1)
            self.assertEqual(knockdowns.mean, 0.25)
            duration = bundle.distribution("duration_seconds")
            self.assertEqual(duration.values, (5.0, 10.0))
            self.assertEqual(duration.observed, 8)

    def test_trace_timeline_applies_replacement_dynamics_and_additive_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.json"
            empty_delta = {
                "red_stamina": None,
                "blue_stamina": None,
                "red_hurt": None,
                "blue_hurt": None,
                "red_damage": None,
                "blue_damage": None,
                "red_stats_delta": None,
                "blue_stats_delta": None,
            }
            _write(
                trace,
                {
                    "simulation_index": 17,
                    "bootstrap_member": 2,
                    "result": {"winner": "red", "method": "decision"},
                    "events": [
                        {
                            "sequence": 0,
                            "fight_time_us": 0,
                            "phase_after": "distance",
                            "delta": empty_delta,
                        },
                        {
                            "sequence": 1,
                            "fight_time_us": 5_000_000,
                            "phase_after": "clinch",
                            "delta": {
                                **empty_delta,
                                "red_stamina": 0.9,
                                "red_stats_delta": {
                                    "significant_strike_attempts": 2,
                                    "significant_strikes_landed": 1,
                                },
                            },
                        },
                        {
                            "sequence": 2,
                            "fight_time_us": 7_000_000,
                            "phase_after": "ground",
                            "delta": {
                                **empty_delta,
                                "red_stamina": 0.8,
                                "red_stats_delta": {
                                    "significant_strike_attempts": 1,
                                    "significant_strikes_landed": 1,
                                    "control_time_us": 2_000_000,
                                },
                            },
                        },
                    ],
                },
            )
            timeline = load_trace_timeline(trace)
            self.assertEqual(timeline.simulation_index, 17)
            self.assertEqual(timeline.seconds, (0.0, 5.0, 7.0))
            self.assertEqual(timeline.phases, ("distance", "clinch", "ground"))
            self.assertEqual(timeline.positions, ("distance", "clinch", "ground"))
            self.assertEqual(timeline.red_stamina, (1.0, 0.9, 0.8))
            self.assertEqual(timeline.red_stats["significant_strike_attempts"], (0.0, 2.0, 3.0))
            self.assertEqual(timeline.red_stats["control_time_us"], (0.0, 0.0, 2_000_000.0))

    def test_rejects_non_run_directory_with_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RunBundleError, "aggregate.json"):
                load_run_bundle(directory)


if __name__ == "__main__":
    unittest.main()
