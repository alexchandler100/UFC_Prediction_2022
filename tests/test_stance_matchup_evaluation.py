import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_stance_matchup_challenger import _factorial_summary  # noqa: E402


class StanceMatchupEvaluationTests(unittest.TestCase):
    def test_factorial_effects_preserve_candidate_direction(self):
        metrics = {
            "baseline": {"log_loss": 0.630},
            "external_history": {"log_loss": 0.628},
            "stance": {"log_loss": 0.629},
            "external_stance": {"log_loss": 0.626},
        }
        effects = _factorial_summary(metrics)

        self.assertAlmostEqual(effects["external_effect_without_stance"], -0.002)
        self.assertAlmostEqual(effects["stance_effect_without_external"], -0.001)
        self.assertAlmostEqual(effects["external_effect_with_stance"], -0.003)
        self.assertAlmostEqual(effects["stance_effect_with_external"], -0.002)
        self.assertAlmostEqual(effects["interaction"], -0.001)


if __name__ == "__main__":
    unittest.main()
