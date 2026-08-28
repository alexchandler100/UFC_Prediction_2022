import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_round_cardio_challenger import (  # noqa: E402
    _causal_counts,
    _factorial_summary,
)


class RoundCardioEvaluationTests(unittest.TestCase):
    def test_factorial_effects_preserve_candidate_direction(self):
        metrics = {
            "baseline": {"log_loss": 0.630},
            "external_history": {"log_loss": 0.628},
            "cardio": {"log_loss": 0.629},
            "external_cardio": {"log_loss": 0.626},
        }
        effects = _factorial_summary(metrics)

        self.assertAlmostEqual(effects["external_effect_without_cardio"], -0.002)
        self.assertAlmostEqual(effects["cardio_effect_without_external"], -0.001)
        self.assertAlmostEqual(effects["external_effect_with_cardio"], -0.003)
        self.assertAlmostEqual(effects["cardio_effect_with_external"], -0.002)
        self.assertAlmostEqual(effects["interaction"], -0.001)

    def test_market_coverage_uses_event_date(self):
        class Builder:
            @staticmethod
            def cardio_sample_count(fighter_id, as_of, round_number):
                self = (fighter_id, pd.Timestamp(as_of), round_number)
                return int(self[0] == "a" and self[1].year == 2026 and self[2] == 2)

        frame = pd.DataFrame(
            {
                "fighter_id": ["a"],
                "opponent_id": ["b"],
                "event_date": ["2026-01-01"],
            }
        )
        counts = _causal_counts(frame, Builder(), 2)
        self.assertEqual(counts.to_dict("records"), [
            {"fighter_samples": 1, "opponent_samples": 0}
        ])


if __name__ == "__main__":
    unittest.main()
