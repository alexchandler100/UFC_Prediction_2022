from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "simulation-research.yml"
README_PATH = REPO_ROOT / "README.md"
ARCHITECTURE_PATH = REPO_ROOT / "SIMULATION_ARCHITECTURE.md"


class SimulationResearchWorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        cls.readme = README_PATH.read_text(encoding="utf-8")
        cls.architecture = ARCHITECTURE_PATH.read_text(encoding="utf-8")

    def test_workflow_is_manual_read_only_and_uses_a_standard_runner(self):
        trigger_block = self.workflow.split("\npermissions:", 1)[0]
        self.assertIn("\non:\n  workflow_dispatch:\n", trigger_block)
        for automatic_trigger in ("push", "pull_request", "schedule"):
            self.assertNotRegex(
                trigger_block,
                rf"(?m)^  {re.escape(automatic_trigger)}:\s*$",
            )

        self.assertIn("permissions:\n  contents: read", self.workflow)
        self.assertIn("runs-on: ubuntu-24.04", self.workflow)
        self.assertIn("timeout-minutes: 360", self.workflow)
        self.assertIn('OMP_NUM_THREADS: "1"', self.workflow)
        self.assertIn('OPENBLAS_NUM_THREADS: "1"', self.workflow)
        self.assertNotIn("contents: write", self.workflow)
        self.assertNotIn("self-hosted", self.workflow)
        self.assertNotIn("services:", self.workflow)
        self.assertNotIn("secrets.", self.workflow)
        self.assertNotIn("git push", self.workflow)
        self.assertNotIn("git commit", self.workflow)

    def test_all_research_modes_use_the_package_cli(self):
        for mode in ("tests", "round-backfill", "fit", "backtest"):
            self.assertRegex(self.workflow, rf"(?m)^          - {re.escape(mode)}$")

        self.assertIn("python -B -m unittest discover -s tests -v", self.workflow)
        self.assertIn("python -m fight_sim backfill", self.workflow)
        self.assertIn("python -m fight_sim fit", self.workflow)
        self.assertIn("python -m fight_sim backtest", self.workflow)
        self.assertIn("--summary-output \"$RESEARCH_OUTPUT_DIR/round-backfill-summary.json\"", self.workflow)
        self.assertIn("--output \"$RESEARCH_OUTPUT_DIR/parameter_model.json.gz\"", self.workflow)
        self.assertIn("--output \"$RESEARCH_OUTPUT_DIR/backtest_report.json\"", self.workflow)
        self.assertIn('--workers "$WORKERS"', self.workflow)
        self.assertIn("--chunk-size 64", self.workflow)
        self.assertNotIn("--ledger-output", self.workflow)
        self.assertNotIn("research_status.json", self.workflow)

    def test_expensive_inputs_are_hard_bounded(self):
        expected_bounds = (
            'backfill_max_fights "$BACKFILL_MAX_FIGHTS" 1 100',
            'backfill_checkpoint_every "$BACKFILL_CHECKPOINT_EVERY" 1 25',
            'fit_bootstrap_members "$FIT_BOOTSTRAP_MEMBERS" 1 200',
            'backtest_bootstrap_members "$BACKTEST_BOOTSTRAP_MEMBERS" 1 64',
            'backtest_paths_per_matchup "$BACKTEST_PATHS_PER_MATCHUP" 1 16384',
            'backtest_max_fights "$BACKTEST_MAX_FIGHTS" 1 500',
            'backtest_workers "$BACKTEST_WORKERS" 1 2',
        )
        for bound in expected_bounds:
            self.assertIn(bound, self.workflow)
        self.assertIn("MAX_COMPACT_ARTIFACT_BYTES: \"5242880\"", self.workflow)

    def test_upload_is_explicit_short_lived_and_compact_only(self):
        upload_input = re.search(
            r"(?ms)^      upload_compact_artifact:\n(.*?)(?=^permissions:)",
            self.workflow,
        )
        self.assertIsNotNone(upload_input)
        self.assertIn("type: boolean", upload_input.group(1))
        self.assertIn("default: false", upload_input.group(1))

        opt_in_condition = "inputs.upload_compact_artifact && inputs.mode != 'tests'"
        self.assertGreaterEqual(self.workflow.count(opt_in_condition), 2)
        self.assertIn("uses: actions/upload-artifact@v4", self.workflow)
        self.assertIn("retention-days: 3", self.workflow)
        self.assertIn("if-no-files-found: error", self.workflow)
        self.assertIn('"round-backfill": "round-backfill-summary.json"', self.workflow)
        self.assertIn('"fit": "parameter_model.json.gz"', self.workflow)
        self.assertIn('"backtest": "backtest_report.json"', self.workflow)
        self.assertIn("inspect_parameter_artifact(source).validate()", self.workflow)
        self.assertIn("inspection.bootstrap_members <= 0", self.workflow)
        self.assertIn("load_backtest_report(source).validate()", self.workflow)

    def test_readme_documents_cli_and_nonproduction_boundary(self):
        self.assertIn(
            "[SIMULATION_ARCHITECTURE.md](SIMULATION_ARCHITECTURE.md)",
            self.readme,
        )
        for command in (
            "backfill",
            "fit",
            "backtest",
            "run",
            "replay",
            "reduce",
            "diff",
            "analyze",
        ):
            self.assertIn(f"python -m fight_sim {command} --help", self.readme)

        section = self.readme.split(
            "## Evidence-first fight simulation research", 1
        )[1].split("\n## Running scripts locally", 1)[0]
        normalized_section = " ".join(section.split())
        for contract in (
            "candidate-only",
            "does not replace or blend with the production winner model",
            "manual-only",
            "upload_compact_artifact",
            "5 MiB hard cap",
            "three-day retention",
            "never commits, pushes, enables shadows, or changes production",
            "parameter_model.json.gz",
            "backtest_report.json",
            "research_status.json",
            'production_influence: "none"',
        ):
            self.assertIn(contract, normalized_section)

    def test_docs_match_the_frozen_bundle_and_shadow_status_schema(self):
        for document in (self.readme, self.architecture):
            for path in (
                "src/content/data/processed/ufc_fight_round_stats_doubled.csv",
                "parameter_model.json.gz",
                "backtest_report.json",
                "research_status.json",
                "shadow_forecasts/<date>_<event>_<publication_sha256>.json",
            ):
                self.assertIn(path, document)
            for field in (
                "candidate_only",
                "paper_only",
                "production_enabled",
                "execution_enabled",
                "integrity_gate_passed",
                "causal_backtest_gate_passed",
                "shadow_enabled",
                "parameter_artifact_sha256",
                "backtest_report_sha256",
            ):
                self.assertIn(field, document)
            self.assertIn("--require-simulation-artifact", document)

        self.assertIn("No frozen bundle is checked in at present", self.readme)
        self.assertIn("at most two", self.architecture)
        self.assertIn("hard-fails above 5 MiB", self.architecture)


if __name__ == "__main__":
    unittest.main()
