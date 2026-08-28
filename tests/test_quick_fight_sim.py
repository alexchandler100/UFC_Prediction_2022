import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "quick_fight_sim.py"
SPEC = importlib.util.spec_from_file_location("quick_fight_sim", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
QUICK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUICK)


class QuickFightSimulationTests(unittest.TestCase):
    def setUp(self):
        self.publication = {
            "matchups": [
                {
                    "matchup_id": "fight-1",
                    "fighter_name": "Alpha Fighter",
                    "opponent_name": "Beta Fighter",
                },
                {
                    "matchup_id": "fight-2",
                    "fighter_name": "Gamma Fighter",
                    "opponent_name": "Delta Fighter",
                },
            ]
        }

    def test_names_resolve_case_insensitively_in_either_order(self):
        row, reversed_order = QUICK._matchup(
            self.publication, "beta", "ALPHA FIGHTER"
        )
        self.assertEqual(row["matchup_id"], "fight-1")
        self.assertTrue(reversed_order)

    def test_ambiguous_or_missing_names_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "no current upcoming fight"):
            QUICK._matchup(self.publication, "Unknown", "Beta")
        with self.assertRaisesRegex(ValueError, "more than one"):
            QUICK._matchup(self.publication, "Fighter", "Fighter")

    def test_exact_path_total_uses_the_largest_available_divisor(self):
        self.assertEqual(QUICK._largest_divisor_at_most(100, 200), 100)
        self.assertEqual(QUICK._largest_divisor_at_most(1000, 200), 200)
        self.assertEqual(QUICK._largest_divisor_at_most(101, 64), 1)


if __name__ == "__main__":
    unittest.main()
