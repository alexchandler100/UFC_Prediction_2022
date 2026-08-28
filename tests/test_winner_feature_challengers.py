import json
from hashlib import sha256
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_winner_feature_challengers import (  # noqa: E402
    _factorial_summary,
    load_frozen_research_auxiliary,
)
from evaluate_external_mma_outcome import _deltas  # noqa: E402
from fight_predictor.point_in_time import COUNT_STATS  # noqa: E402


def auxiliary_frame() -> pd.DataFrame:
    common = {
        "fight_url": "https://external.test/fight/one",
        "emit_training_target": False,
        **dict.fromkeys(COUNT_STATS, np.nan),
    }
    return pd.DataFrame(
        [
            {
                **common,
                "fighter_url": "https://external.test/fighter/a",
                "opponent_url": "https://external.test/fighter/b",
            },
            {
                **common,
                "fighter_url": "https://external.test/fighter/b",
                "opponent_url": "https://external.test/fighter/a",
            },
        ]
    )


class WinnerFeatureChallengerTests(unittest.TestCase):
    def _write_fixture(self, root: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
        auxiliary = root / "auxiliary.csv"
        report = root / "evaluation.json"
        frame.to_csv(auxiliary, index=False)
        report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "auxiliary_sha256": sha256(auxiliary.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        return auxiliary, report

    def test_research_loader_requires_evaluated_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            auxiliary, report = self._write_fixture(
                Path(directory), auxiliary_frame()
            )
            loaded, metadata = load_frozen_research_auxiliary(auxiliary, report)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(
                metadata["auxiliary_sha256"],
                sha256(auxiliary.read_bytes()).hexdigest(),
            )
            auxiliary.write_text(
                auxiliary.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "evaluation hash"):
                load_frozen_research_auxiliary(auxiliary, report)

    def test_research_loader_rejects_fabricated_detailed_stats(self):
        frame = auxiliary_frame()
        frame.loc[0, "sig_strikes_landed"] = 10
        with tempfile.TemporaryDirectory() as directory:
            auxiliary, report = self._write_fixture(Path(directory), frame)
            with self.assertRaisesRegex(ValueError, "fabricate"):
                load_frozen_research_auxiliary(auxiliary, report)

    def test_factorial_summary_reports_marginal_and_interaction_effects(self):
        metrics = {
            "baseline": {"log_loss": 0.63},
            "external_history": {"log_loss": 0.62},
            "style": {"log_loss": 0.625},
            "external_style": {"log_loss": 0.61},
        }
        summary = _factorial_summary(metrics)
        self.assertAlmostEqual(summary["external_effect_without_style"], -0.01)
        self.assertAlmostEqual(summary["style_effect_with_external"], -0.01)
        self.assertAlmostEqual(summary["interaction"], -0.005)

    def test_outcome_comparison_deltas_keep_market_directions(self):
        baseline = {
            "joint_outcome": {"log_loss": 1.6},
            "winner": {"log_loss": 0.63},
            "method": {"log_loss": 1.03},
            "total_rounds": {"over_1_5_rounds": {"log_loss": 0.64}},
        }
        external = {
            "joint_outcome": {"log_loss": 1.59},
            "winner": {"log_loss": 0.62},
            "method": {"log_loss": 1.04},
            "total_rounds": {"over_1_5_rounds": {"log_loss": 0.63}},
        }
        result = _deltas(baseline, external)
        self.assertAlmostEqual(result["joint_outcome_log_loss"], -0.01)
        self.assertAlmostEqual(result["method_log_loss"], 0.01)
        self.assertAlmostEqual(
            result["total_round_log_loss"]["over_1_5_rounds"], -0.01
        )


if __name__ == "__main__":
    unittest.main()
