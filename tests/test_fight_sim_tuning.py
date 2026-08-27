from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fight_sim.parameters import canonical_sha256  # noqa: E402
from fight_sim.tuning import (  # noqa: E402
    select_finish_profile,
    validate_finish_profile,
    validate_knockdown_observation_profile,
)


def finish_metrics(
    *,
    config_value: float,
    joint: float = 1.5,
    method: float = 1.0,
    winner: float = 0.65,
    duration: float = 200.0,
    duration_bias: float = -40.0,
    action: float = 0.3,
) -> dict[str, object]:
    return {
        "report_path": "population-summary.json",
        "report_sha256": "report",
        "event_ids": ["event-a", "event-b"],
        "fight_ids": ["fight-a", "fight-b"],
        "joint_log_loss": joint,
        "method_log_loss": method,
        "winner_log_loss": winner,
        "duration_crps_seconds": duration,
        "duration_mean_bias_seconds": duration_bias,
        "knockdown_crps": 0.4,
        "knockdown_observed_mean": 0.5,
        "knockdown_predictive_mean": 0.9,
        "knockdown_mean_bias": 0.4,
        "observable_action_error": action,
        "observable_action_errors": {},
        "simulator_config": {
            "schema_version": "fight-sim.v1",
            "ko_tko_finish_probability_multiplier": config_value,
        },
    }


class FinishTuningTests(unittest.TestCase):
    def test_selection_uses_duration_only_after_all_preservation_gates(self):
        baseline = finish_metrics(config_value=1.0)
        eligible = finish_metrics(
            config_value=0.4,
            joint=1.49,
            method=0.98,
            winner=0.64,
            duration=170.0,
            duration_bias=5.0,
            action=0.31,
        )
        ineligible = finish_metrics(
            config_value=0.2,
            joint=1.54,
            duration=150.0,
            duration_bias=20.0,
        )
        values = {"baseline": baseline, "eligible": eligible, "ineligible": ineligible}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "selection.json"
            with patch(
                "fight_sim.tuning._finish_selection_metrics",
                side_effect=lambda path: values[str(path)],
            ):
                report = select_finish_profile(
                    "baseline",
                    {"ko040": "eligible", "ko020": "ineligible"},
                    output=output,
                )
            self.assertEqual(report["selection_status"], "selected")
            self.assertEqual(report["selected_label"], "ko040")
            self.assertEqual(
                report["simulator_config"]["ko_tko_finish_probability_multiplier"],
                0.4,
            )
            self.assertFalse(next(row for row in report["candidates"] if row["label"] == "ko020")["eligible"])
            stored = json.loads(output.read_text(encoding="utf-8"))
            unhashed = dict(stored)
            supplied = unhashed.pop("selection_sha256")
            self.assertEqual(supplied, canonical_sha256(unhashed))

    def test_joint_selection_uses_primary_metric_with_duration_safety_gates(self):
        baseline = finish_metrics(
            config_value=1.0,
            joint=1.50,
            duration=200.0,
            duration_bias=10.0,
        )
        best_joint = finish_metrics(
            config_value=0.5,
            joint=1.47,
            duration=204.0,
            duration_bias=20.0,
        )
        best_duration = finish_metrics(
            config_value=0.7,
            joint=1.48,
            duration=190.0,
            duration_bias=5.0,
        )
        unsafe_duration = finish_metrics(
            config_value=0.3,
            joint=1.40,
            duration=206.0,
            duration_bias=5.0,
        )
        values = {
            "baseline": baseline,
            "best-joint": best_joint,
            "best-duration": best_duration,
            "unsafe-duration": unsafe_duration,
        }
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "fight_sim.tuning._finish_selection_metrics",
                side_effect=lambda path: values[str(path)],
            ):
                report = select_finish_profile(
                    "baseline",
                    {
                        "best-joint": "best-joint",
                        "best-duration": "best-duration",
                        "unsafe-duration": "unsafe-duration",
                    },
                    output=Path(temporary) / "selection.json",
                    objective="joint",
                )
        self.assertEqual(report["objective"], "joint")
        self.assertEqual(report["selected_label"], "best-joint")
        unsafe = next(
            row for row in report["candidates"] if row["label"] == "unsafe-duration"
        )
        self.assertFalse(unsafe["eligible"])
        self.assertFalse(
            unsafe["gates"]["duration_crps_not_worse_by_more_than_5_seconds"]
        )

    def test_selection_rejects_unknown_objective(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "objective"):
                select_finish_profile(
                    "baseline",
                    {"candidate": "candidate"},
                    output=Path(temporary) / "selection.json",
                    objective="accuracy",
                )

    def test_validation_falls_back_when_any_holdout_gate_fails(self):
        baseline = finish_metrics(config_value=0.8)
        candidate = finish_metrics(
            config_value=0.4,
            duration=180.0,
            duration_bias=5.0,
            action=0.33,
        )
        values = {"baseline": baseline, "candidate": candidate}
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "fight_sim.tuning._finish_selection_metrics",
                side_effect=lambda path: values[str(path)],
            ):
                report = validate_finish_profile(
                    "baseline",
                    "candidate",
                    output=Path(temporary) / "validation.json",
                )
        self.assertEqual(report["validation_status"], "rejected_baseline_fallback")
        self.assertFalse(report["gates"]["observable_action_error_not_worse_by_more_than_0.02"])
        self.assertEqual(
            report["simulator_config"]["ko_tko_finish_probability_multiplier"],
            0.8,
        )

    def test_validation_retains_candidate_only_when_every_gate_passes(self):
        baseline = finish_metrics(config_value=0.8)
        candidate = finish_metrics(
            config_value=0.4,
            joint=1.49,
            method=0.99,
            winner=0.64,
            duration=180.0,
            duration_bias=5.0,
            action=0.29,
        )
        values = {"baseline": baseline, "candidate": candidate}
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "fight_sim.tuning._finish_selection_metrics",
                side_effect=lambda path: values[str(path)],
            ):
                report = validate_finish_profile(
                    "baseline",
                    "candidate",
                    output=Path(temporary) / "validation.json",
                )
        self.assertEqual(report["validation_status"], "retained")
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(
            report["simulator_config"]["ko_tko_finish_probability_multiplier"],
            0.4,
        )

    def test_knockdown_observation_validation_requires_invariant_outcomes(self):
        baseline = finish_metrics(config_value=0.4, action=0.40)
        baseline["simulator_config"].update(
            {
                "knockdown_probability_multiplier": 0.8,
                "official_knockdown_observation_probability": 1.0,
            }
        )
        candidate = dict(baseline)
        candidate["simulator_config"] = dict(baseline["simulator_config"])
        candidate["simulator_config"][
            "official_knockdown_observation_probability"
        ] = 0.59
        candidate["observable_action_errors"] = dict(
            baseline["observable_action_errors"]
        )
        candidate.update(
            {
                "knockdown_crps": 0.3,
                "knockdown_predictive_mean": 0.55,
                "knockdown_mean_bias": 0.05,
                "observable_action_error": 0.32,
            }
        )
        values = {"baseline": baseline, "candidate": candidate}
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "fight_sim.tuning._finish_selection_metrics",
                side_effect=lambda path: values[str(path)],
            ):
                report = validate_knockdown_observation_profile(
                    "baseline",
                    "candidate",
                    output=Path(temporary) / "validation.json",
                )
        self.assertEqual(
            report["validation_status"], "retained_for_prospective_shadow"
        )
        self.assertTrue(all(report["gates"].values()))

        candidate["winner_log_loss"] = 0.64
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "fight_sim.tuning._finish_selection_metrics",
                side_effect=lambda path: values[str(path)],
            ):
                rejected = validate_knockdown_observation_profile(
                    "baseline",
                    "candidate",
                    output=Path(temporary) / "validation.json",
                )
        self.assertEqual(rejected["validation_status"], "rejected_baseline_fallback")
        self.assertFalse(
            rejected["gates"]["outcome_and_duration_metrics_are_identical"]
        )


if __name__ == "__main__":
    unittest.main()
