from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.__main__ import build_parser, main  # noqa: E402
from fight_sim.evaluation import load_backtest_report  # noqa: E402
from fight_sim.parameters import CausalParameterFitter, load_parameter_artifact  # noqa: E402
from fight_sim.research import (  # noqa: E402
    _repeat_forecast_sha256,
    execute_backtest,
)


def _side(
    date: str,
    event: str,
    fight: str,
    fighter: str,
    opponent: str,
    result: str,
    method: str,
    landed: int,
    attempted: int,
) -> dict[str, object]:
    return {
        "date": date,
        "event_url": f"http://ufcstats.test/event-details/{event}",
        "fight_url": f"http://ufcstats.test/fight-details/{fight}",
        "fighter_url": f"http://ufcstats.test/fighter-details/{fighter}",
        "opponent_url": f"http://ufcstats.test/fighter-details/{opponent}",
        "fighter": fighter.upper(),
        "opponent": opponent.upper(),
        "division": "Lightweight",
        "result": result,
        "method": method,
        "round": 3,
        "time": "5:00",
        "total_fight_time": 900,
        "time_format": "3 Rnd (5-5-5)",
        "knockdowns": int(method == "KO/TKO" and result == "W"),
        "sig_strikes_landed": landed,
        "sig_strikes_attempts": attempted,
        "takedowns_landed": 1,
        "takedowns_attempts": 3,
        "sub_attempts": int(method == "SUB" and result == "W"),
        "reversals": 0,
        "control": 60 if result == "W" else 30,
        "distance_strikes_attempts": int(attempted * 0.75),
        "clinch_strikes_attempts": int(attempted * 0.15),
        "ground_strikes_attempts": attempted
        - int(attempted * 0.75)
        - int(attempted * 0.15),
    }


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    fights = (
        ("2018-06-01", "event-2018", "fight-2018", "a", "b", "KO/TKO"),
        ("2019-06-01", "event-2019", "fight-2019", "c", "a", "SUB"),
        ("2020-06-01", "event-2020", "fight-2020", "b", "c", "U-DEC"),
        ("2021-06-01", "event-2021", "fight-2021", "a", "b", "U-DEC"),
    )
    rows = []
    for date, event, fight, winner, loser, method in fights:
        rows.append(_side(date, event, fight, winner, loser, "W", method, 70, 130))
        rows.append(_side(date, event, fight, loser, winner, "L", method, 50, 120))
    raw = root / "raw.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    profiles = root / "profiles.csv"
    pd.DataFrame(
        [
            {
                "url": f"http://ufcstats.test/fighter-details/{fighter}",
                "name": fighter.upper(),
                "dob": dob,
            }
            for fighter, dob in (
                ("a", "1990-01-01"),
                ("b", "1991-01-01"),
                ("c", "1992-01-01"),
            )
        ]
    ).to_csv(profiles, index=False)
    return raw, profiles, root / "missing-rounds.csv"


class FightSimulationCliTests(unittest.TestCase):
    def test_every_documented_subcommand_parses_and_workflow_flags_are_exact(self):
        parser = build_parser()
        for command in (
            "backfill",
            "fit",
            "backtest",
            "posterior-backtest",
            "derive-mechanics",
            "select-mechanics",
            "select-finishing",
            "validate-mechanics",
            "validate-finishing",
            "upcoming-card",
            "run",
            "replay",
            "reduce",
            "diff",
            "analyze",
            "benchmark",
            "validate-fight",
            "gui",
        ):
            self.assertIn(command, parser._subparsers._group_actions[0].choices)
        parsed = parser.parse_args(
            [
                "backtest",
                "--bootstrap-members",
                "16",
                "--paths-per-matchup",
                "4096",
                "--first-test-year",
                "2017",
                "--last-test-year",
                "2026",
                "--max-fights",
                "200",
                "--output",
                "summary.json",
            ]
        )
        self.assertEqual(parsed.bootstrap_members, 16)
        self.assertEqual(parsed.paths_per_matchup, 4096)
        self.assertIsNone(parsed.ledger_output)
        self.assertEqual(parsed.seed_repeats, 2)
        self.assertFalse(parsed.skip_borderline_rerun)
        posterior = parser.parse_args(["posterior-backtest"])
        self.assertEqual(posterior.min_prior_ufc_fights, 3)
        self.assertEqual(posterior.last_events, 20)
        self.assertEqual(posterior.skip_latest_events, 0)
        self.assertFalse(posterior.resume)
        self.assertFalse(posterior.quick_screen)
        screen = parser.parse_args(["posterior-backtest", "--quick-screen"])
        self.assertTrue(screen.quick_screen)
        benchmark = parser.parse_args(
            ["benchmark", "specs.json", "--workers", "1,4,4,8"]
        )
        self.assertEqual(benchmark.workers, (1, 4, 8))
        upcoming = parser.parse_args(["upcoming-card"])
        self.assertEqual(upcoming.minimum_prior_ufc_fights, 3)
        self.assertEqual(upcoming.bootstrap_members, 200)
        tune = parser.parse_args(["derive-mechanics", "population-run"])
        self.assertEqual(tune.holdout_latest_events, 5)
        self.assertEqual(parser.parse_args(["gui", "run-dir"]).run, "run-dir")

        run_parser = parser.parse_args(
            [
                "run",
                "--red-fighter-id",
                "red",
                "--blue-fighter-id",
                "blue",
                "--division",
                "Lightweight",
                "--launch-gui",
            ]
        )
        self.assertTrue(run_parser.launch_gui)

    def test_fit_run_trace_diagnostics_and_analysis_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, profiles, rounds = _inputs(root)
            parameters = root / "parameters.json.gz"
            self.assertEqual(
                main(
                    [
                        "fit",
                        "--raw",
                        str(raw),
                        "--profiles",
                        str(profiles),
                        "--round-stats",
                        str(rounds),
                        "--as-of",
                        "2022-01-01T00:00:00Z",
                        "--bootstrap-members",
                        "1",
                        "--output",
                        str(parameters),
                    ]
                ),
                0,
            )
            self.assertEqual(len(load_parameter_artifact(parameters).members), 1)

            run_dir = root / "run"
            run_args = [
                        "run",
                        "--parameters",
                        str(parameters),
                        "--raw",
                        str(raw),
                        "--profiles",
                        str(profiles),
                        "--round-stats",
                        str(rounds),
                        "--red-fighter-id",
                        "a",
                        "--blue-fighter-id",
                        "b",
                        "--division",
                        "Lightweight",
                        "--initial-paths-per-member",
                        "2",
                        "--max-paths-per-member",
                        "2",
                        "--winner-mcse-target",
                        "1",
                        "--parameter-quantile-tolerance",
                        "1",
                        "--max-traces",
                        "1",
                        "--allow-nonconverged-research",
                        "--output-dir",
                        str(run_dir),
                    ]
            self.assertEqual(
                main(run_args),
                0,
            )
            for relative in (
                "specs.json",
                "aggregate.json",
                "convergence.json",
                "trace-manifest.json",
                "analysis.html",
            ):
                self.assertTrue((run_dir / relative).is_file(), relative)
            convergence_payload = json.loads(
                (run_dir / "convergence.json").read_text(encoding="utf-8")
            )
            self.assertTrue(convergence_payload["convergence"])
            self.assertIn(
                "Paths/member", (run_dir / "analysis.html").read_text(encoding="utf-8")
            )
            original_aggregate = (run_dir / "aggregate.json").read_bytes()
            self.assertEqual(main(run_args), 2)
            self.assertEqual(
                (run_dir / "aggregate.json").read_bytes(), original_aggregate
            )
            trace = next((run_dir / "traces").glob("*.json"))
            trace_payload = json.loads(trace.read_text(encoding="utf-8"))

            self.assertEqual(main(["reduce", str(trace)]), 0)
            self.assertEqual(main(["replay", "--trace", str(trace)]), 0)
            self.assertEqual(
                main(
                    [
                        "replay",
                        "--spec",
                        str(run_dir / "specs.json"),
                        "--bootstrap-member",
                        "0",
                        "--simulation-index",
                        str(trace_payload["simulation_index"]),
                        "--expected-trace",
                        str(trace),
                    ]
                ),
                0,
            )
            self.assertEqual(main(["diff", str(trace), str(trace)]), 0)
            report = root / "analysis-copy.html"
            self.assertEqual(
                main(["analyze", str(run_dir), "--output", str(report)]), 0
            )
            self.assertIn(
                "Candidate-only local research report",
                report.read_text(encoding="utf-8"),
            )

    def test_backtest_executes_real_causal_fit_and_simulation_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, profiles, rounds = _inputs(root)
            summary = root / "backtest.json"
            ledger = root / "ledger.jsonl"
            fit_calls = []
            original_fit = CausalParameterFitter.fit

            def counted_fit(instance, *args, **kwargs):
                fit_calls.append(args[0] if args else kwargs.get("as_of"))
                return original_fit(instance, *args, **kwargs)

            with patch.object(CausalParameterFitter, "fit", new=counted_fit):
                code = main(
                    [
                    "backtest",
                    "--raw",
                    str(raw),
                    "--profiles",
                    str(profiles),
                    "--round-stats",
                    str(rounds),
                    "--bootstrap-members",
                    "1",
                    "--paths-per-matchup",
                    "2",
                    "--first-test-year",
                    "2020",
                    "--last-test-year",
                    "2021",
                    "--max-fights",
                    "2",
                    "--seed-repeats",
                    "3",
                    "--skip-borderline-rerun",
                    "--min-training-fights",
                    "2",
                    "--output",
                    str(summary),
                    "--ledger-output",
                    str(ledger),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(len(fit_calls), 2)
            report = load_backtest_report(summary)
            self.assertTrue(report.candidate_only)
            self.assertEqual(len(report.folds), 2)
            self.assertEqual(report.aggregate["n_fights"], 2)
            self.assertEqual(report.simulation_noise["seed_repeats"], 3)
            self.assertEqual(len(report.simulation_noise["repeat_forecast_sha256"]), 3)
            self.assertEqual(report.config["seed_repeats"], 3)
            self.assertFalse(report.config["precision"]["borderline_rerun_triggered"])
            lines = ledger.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(all("forecast" in json.loads(line) for line in lines))
            authoritative = pd.DataFrame([json.loads(line) for line in lines])
            self.assertEqual(
                report.simulation_noise["repeat_forecast_sha256"][0],
                _repeat_forecast_sha256(authoritative),
            )

    def test_borderline_rerun_reuses_each_fitted_fold_and_does_not_recurse(self):
        from fight_sim.monte_carlo import run_nested as actual_run_nested

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, profiles, rounds = _inputs(root)
            fit_calls = []
            requested_inner_paths = []
            original_fit = CausalParameterFitter.fit

            def counted_fit(instance, *args, **kwargs):
                fit_calls.append(args[0] if args else kwargs.get("as_of"))
                return original_fit(instance, *args, **kwargs)

            def bounded_nested(specs, paths_per_member, **_kwargs):
                requested_inner_paths.append(paths_per_member)
                return actual_run_nested(
                    specs,
                    1,
                    workers=1,
                    chunk_size=1,
                    max_traces=0,
                )

            output = root / "precision.json"
            with (
                patch.object(CausalParameterFitter, "fit", new=counted_fit),
                patch("fight_sim.research.run_nested", side_effect=bounded_nested),
                patch(
                    "fight_sim.research._is_default_repository_backtest",
                    return_value=True,
                ),
                patch(
                    "fight_sim.research._borderline_joint_comparisons",
                    return_value=("population_joint",),
                ),
            ):
                ledger, report = execute_backtest(
                    raw_path=raw,
                    profiles_path=profiles,
                    round_path=rounds,
                    output=output,
                    bootstrap_members=1,
                    paths_per_matchup=2,
                    first_test_year=2020,
                    last_test_year=2021,
                    max_fights=2,
                    min_training_fights=2,
                    include_baselines=False,
                    seed_repeats=2,
                )
            self.assertEqual(len(ledger), 2)
            self.assertEqual(len(fit_calls), 2)
            self.assertEqual(set(requested_inner_paths), {2, 16384})
            precision = report.config["precision"]
            self.assertTrue(precision["borderline_rerun_triggered"])
            self.assertEqual(precision["final_paths_per_matchup"], 16384)
            self.assertEqual(report.simulation_noise["paths_per_matchup"], 16384)
            self.assertEqual(load_backtest_report(output).to_dict(), report.to_dict())


if __name__ == "__main__":
    unittest.main()
