from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.shadow import validate_research_status  # noqa: E402


class SimulationShadowBoundaryTests(unittest.TestCase):
    def test_status_requires_explicit_nonproduction_gates(self):
        valid = {
            "schema_version": 1,
            "candidate_only": True,
            "paper_only": True,
            "production_enabled": False,
            "execution_enabled": False,
            "integrity_gate_passed": True,
            "causal_backtest_gate_passed": True,
            "shadow_enabled": False,
        }
        self.assertFalse(validate_research_status(valid)["shadow_enabled"])
        invalid = dict(valid, production_enabled=True)
        with self.assertRaisesRegex(ValueError, "candidate-only"):
            validate_research_status(invalid)
        invalid = dict(valid, integrity_gate_passed=False)
        with self.assertRaisesRegex(ValueError, "integrity"):
            validate_research_status(invalid)


if __name__ == "__main__":
    unittest.main()
