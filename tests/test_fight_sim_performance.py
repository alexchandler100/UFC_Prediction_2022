from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.performance import execute_benchmark  # noqa: E402
from fight_sim.research import atomic_write_json  # noqa: E402
from test_fight_sim_core import _spec  # noqa: E402


class FightSimulationPerformanceTests(unittest.TestCase):
    def test_benchmark_measures_complete_deterministic_streaming_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = root / "specs.json"
            output = root / "benchmark.json"
            atomic_write_json(
                specs,
                {"schema_version": 1, "specs": [_spec().to_dict()]},
            )
            result = execute_benchmark(
                specs,
                paths_per_member=2,
                worker_counts=(1,),
                chunk_size=1,
                repeats=2,
                output=output,
            )
            written = output.is_file()
        self.assertTrue(result["worker_invariant"])
        self.assertEqual(result["results"][0]["total_paths"], 2)
        self.assertGreater(result["results"][0]["paths_per_second"], 0.0)
        self.assertEqual(len(result["results"][0]["forecast_sha256"]), 64)
        self.assertTrue(written)


if __name__ == "__main__":
    unittest.main()
