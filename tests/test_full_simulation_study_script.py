import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FullSimulationStudyScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            REPO_ROOT / "scripts" / "run_full_simulation_study.sh"
        ).read_text(encoding="utf-8")
        cls.progress_script = (
            REPO_ROOT / "scripts" / "simulation_study_progress.py"
        ).read_text(encoding="utf-8")

    def test_study_is_broad_precise_and_excludes_low_history_fights(self):
        self.assertIn("LAST_EVENTS=100", self.script)
        self.assertIn("MIN_PRIOR_FIGHTS=3", self.script)
        self.assertIn("BOOTSTRAP_MEMBERS=64", self.script)
        self.assertIn("PATHS_PER_MATCHUP=1024", self.script)
        self.assertIn("SEED_REPEATS=2", self.script)
        self.assertIn("--snapshot-parameter-mode full", self.script)
        self.assertIn("SIMULATION_MECHANICS_BASELINE_V1.json", self.script)

    def test_large_outputs_and_code_snapshot_live_outside_git(self):
        self.assertIn("$HOME/.ufc-data-lab/simulation-studies", self.script)
        self.assertIn("Refusing to store the study inside the Git repository", self.script)
        self.assertIn('cp -R "$REPO_ROOT/src/fight_sim"', self.script)
        self.assertIn('cp -R "$REPO_ROOT/src/market_tracker"', self.script)
        self.assertIn('cp -p "$REPO_ROOT"/src/*.py "$CODE_DIR/"', self.script)
        self.assertIn("fight_semantics.py", self.script)
        self.assertIn("ufc_round_data.py", self.script)
        self.assertIn("ufcstats_client.py", self.script)
        self.assertIn("input-snapshot", self.script)
        self.assertIn("causal-fit-cache", self.script)
        self.assertIn("max_traces=0", (
            REPO_ROOT / "src" / "fight_sim" / "research.py"
        ).read_text(encoding="utf-8"))

    def test_runtime_storage_and_resume_are_hard_bounded(self):
        self.assertIn("SIM_MAX_WALL_HOURS:-22", self.script)
        self.assertIn("SIM_MAX_STORAGE_GB:-4", self.script)
        self.assertIn("SLICE_SECONDS=3300", self.script)
        self.assertIn("COMMAND_HARD_LIMIT_SECONDS=3900", self.script)
        self.assertIn("timeout --foreground --signal=INT", self.script)
        self.assertIn("--resume", self.script)
        self.assertIn("completed_fight_seed_pairs", self.script)
        self.assertIn("planned_fight_seed_pairs", self.script)

    def test_preflight_loads_the_exact_snapshotted_cli(self):
        self.assertIn("--prepare-only", self.script)
        self.assertIn(
            '"$PYTHON_BIN" -m fight_sim posterior-backtest --help', self.script
        )

    def test_progress_printer_does_not_use_invalid_escaped_f_string_keys(self):
        self.assertIn("simulation_study_progress.py", self.script)
        self.assertNotIn('print(f"Progress:', self.script)
        self.assertIn("fight/seed pairs checkpointed", self.progress_script)


if __name__ == "__main__":
    unittest.main()
