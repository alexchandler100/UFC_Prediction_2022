from __future__ import annotations

import builtins
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.domain import FighterParameters  # noqa: E402
from fight_sim.evaluation import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    EVALUATION_VERSION,
    BacktestReport,
    write_backtest_report,
)
from fight_sim.parameters import (  # noqa: E402
    PARAMETER_MODEL_VERSION,
    PARAMETER_SCHEMA_VERSION,
    BootstrapParameterMember,
    ParameterEnsembleArtifact,
    ParameterFitConfig,
    canonical_sha256,
    save_parameter_artifact,
)
from fight_sim.publication import (  # noqa: E402
    append_shadow_forecast_publication,
    build_shadow_forecast_publication,
)
from ufc_round_data import (  # noqa: E402
    ROUND_STAT_COLUMNS,
    normalize_round_stats,
    reconcile_round_stats,
)
from validate_data import validate_simulation_artifacts  # noqa: E402


def _raw_fight() -> pd.DataFrame:
    rows = []
    for fighter, opponent, result in (("Alpha", "Beta", "W"), ("Beta", "Alpha", "L")):
        row = {
            "date": "2025-12-01",
            "event_url": "http://ufcstats.test/event-details/event-one",
            "fight_url": "http://ufcstats.test/fight-details/fight-one",
            "fighter_url": f"http://ufcstats.test/fighter-details/{fighter.casefold()}",
            "opponent_url": f"http://ufcstats.test/fighter-details/{opponent.casefold()}",
            "fighter": fighter,
            "opponent": opponent,
            "result": result,
            "division": "Lightweight",
            "method": "U-DEC",
            "round": 1,
            "time": "5:00",
            "time_format": "3 Rnd (5-5)",
            "total_fight_time": 300,
            "source_card_index": 0,
            "bout_order": 0,
        }
        row.update({column: 0 for column in ROUND_STAT_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def _round_frame(raw: pd.DataFrame) -> pd.DataFrame:
    partial = []
    for row in raw.to_dict("records"):
        value = {
            "fight_url": row["fight_url"],
            "fighter_url": row["fighter_url"],
            "fighter": row["fighter"],
            "round": 1,
        }
        value.update({column: 0 for column in ROUND_STAT_COLUMNS})
        partial.append(value)
    normalized = normalize_round_stats(pd.DataFrame(partial), raw)
    annotated, issues = reconcile_round_stats(normalized, raw)
    if not issues.empty:
        raise AssertionError("test fixture should reconcile exactly")
    return annotated


def _write_round_data(data_root: Path, raw: pd.DataFrame) -> Path:
    destination = data_root / "processed" / "ufc_fight_round_stats_doubled.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(raw).to_csv(destination, index=False)
    return destination


def _parameter_artifact() -> ParameterEnsembleArtifact:
    config = ParameterFitConfig(bootstrap_members=1, random_seed=7)
    member = BootstrapParameterMember(
        member_index=0,
        bootstrap_seed=7,
        sampled_event_count=1,
        context_parameters={"__global__": FighterParameters().to_dict()},
        fighter_parameters={},
        covariate_effects={},
    )
    artifact = ParameterEnsembleArtifact(
        schema_version=PARAMETER_SCHEMA_VERSION,
        model_version=PARAMETER_MODEL_VERSION,
        as_of_utc="2026-01-01T00:00:00+00:00",
        trained_through="2025-12-01",
        input_sha256="a" * 64,
        config=config,
        members=(member,),
        observed_fights=1,
        observed_fighter_sides=2,
        observed_round_sides=2,
        round_reconciliation_counts={"matched": 2},
        created_at_utc="2026-01-02T00:00:00+00:00",
        artifact_sha256="",
    )
    return replace(
        artifact, artifact_sha256=canonical_sha256(artifact.content_dict())
    ).validate()


def _backtest_report() -> BacktestReport:
    body = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "primary_metric": "joint_side_by_method_log_loss",
        "config": {},
        "folds": [],
        "aggregate": {"n_fights": 1},
        "slices": {},
        "comparisons": {},
        "coverage_warnings": [],
        "ledger_sha256": "b" * 64,
    }
    return BacktestReport(
        schema_version=EVALUATION_SCHEMA_VERSION,
        evaluation_version=EVALUATION_VERSION,
        candidate_only=True,
        production_enabled=False,
        execution_enabled=False,
        primary_metric="joint_side_by_method_log_loss",
        config={},
        folds=(),
        aggregate={"n_fights": 1},
        slices={},
        comparisons={},
        coverage_warnings=(),
        ledger_sha256="b" * 64,
        report_sha256=canonical_sha256(body),
    ).validate()


def _write_required_bundle(data_root: Path) -> tuple[ParameterEnsembleArtifact, BacktestReport]:
    root = data_root / "simulation"
    root.mkdir(parents=True, exist_ok=True)
    artifact = _parameter_artifact()
    backtest = _backtest_report()
    save_parameter_artifact(root / "parameter_model.json.gz", artifact)
    write_backtest_report(root / "backtest_report.json", backtest)
    status = {
        "schema_version": 1,
        "candidate_only": True,
        "paper_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "integrity_gate_passed": True,
        "causal_backtest_gate_passed": True,
        "shadow_enabled": False,
        "parameter_artifact_sha256": artifact.artifact_sha256,
        "backtest_report_sha256": backtest.report_sha256,
    }
    (root / "research_status.json").write_text(
        json.dumps(status, sort_keys=True), encoding="utf-8"
    )
    return artifact, backtest


class OptionalSimulationValidationTests(unittest.TestCase):
    def test_importing_production_validator_does_not_import_fight_sim(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPO_ROOT / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; import validate_data; "
                    "assert not any(name == 'fight_sim' or "
                    "name.startswith('fight_sim.') or name == 'ufc_round_data' "
                    "for name in sys.modules)"
                ),
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_normal_validation_does_not_import_or_parse_simulation_research(self):
        raw = _raw_fight()
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            _write_round_data(data_root, raw)
            _write_required_bundle(data_root)
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if (
                    name == "fight_sim"
                    or name.startswith("fight_sim.")
                    or name == "ufc_round_data"
                ):
                    raise AssertionError(
                        "normal validation imported simulation research code"
                    )
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=guarded_import):
                report = validate_simulation_artifacts(data_root, raw)
        self.assertFalse(report.errors, report.errors)
        self.assertTrue(
            any("present but not validated" in fact for fact in report.facts)
        )

    def test_normal_validation_warns_when_research_inputs_are_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_simulation_artifacts(Path(directory), _raw_fight())
        self.assertFalse(report.errors)
        self.assertTrue(any("per-round data" in item for item in report.warnings))
        self.assertTrue(any("simulation research directory" in item for item in report.warnings))

    def test_required_mode_requires_round_and_frozen_research_triple(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_simulation_artifacts(
                Path(directory), _raw_fight(), required=True
            )
        self.assertGreaterEqual(len(report.errors), 4)
        self.assertTrue(any("parameter artifact" in item for item in report.errors))
        self.assertTrue(any("research-status" in item for item in report.errors))

    def test_source_discrepancy_is_warning_but_structural_corruption_is_error(self):
        raw = _raw_fight()
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            round_path = _write_round_data(data_root, raw)
            _write_required_bundle(data_root)
            altered = raw.copy()
            altered.loc[0, "knockdowns"] = 1
            discrepancy = validate_simulation_artifacts(
                data_root, altered, required=True
            )
            self.assertFalse(discrepancy.errors)
            self.assertTrue(
                any("source discrepancy" in item for item in discrepancy.warnings)
            )

            corrupt = pd.read_csv(round_path)
            corrupt = corrupt.iloc[:1]
            corrupt.to_csv(round_path, index=False)
            structural = validate_simulation_artifacts(
                data_root, raw, required=True
            )
            self.assertTrue(
                any("per-round data is invalid" in item for item in structural.errors)
            )

    def test_repository_round_reconciliation_handles_multiple_physical_fights(self):
        first = _raw_fight()
        second = _raw_fight().replace(
            {
                "fight-one": "fight-two",
                "event-one": "event-two",
                "Alpha": "Gamma",
                "Beta": "Delta",
                "alpha": "gamma",
                "beta": "delta",
            },
            regex=True,
        )
        raw = pd.concat((first, second), ignore_index=True)
        rounds = pd.concat(
            (_round_frame(first), _round_frame(second)), ignore_index=True
        )
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            round_path = data_root / "processed" / "ufc_fight_round_stats_doubled.csv"
            round_path.parent.mkdir(parents=True)
            rounds.to_csv(round_path, index=False)
            _write_required_bundle(data_root)
            report = validate_simulation_artifacts(data_root, raw, required=True)
        self.assertFalse(report.errors, report.errors)
        self.assertTrue(any("4 matched sides" in item for item in report.facts))

    def test_required_bundle_and_optional_shadow_are_cross_hashed(self):
        raw = _raw_fight()
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            _write_round_data(data_root, raw)
            artifact, _backtest = _write_required_bundle(data_root)
            publication = build_shadow_forecast_publication(
                {},
                [],
                {
                    "event_id": "event-upcoming",
                    "event_url": "http://ufcstats.test/event-details/event-upcoming",
                    "date": "2099-07-01T00:00:00Z",
                    "title": "UFC Test",
                },
                artifact,
                forecast_issued_at_utc="2026-06-01T00:00:00Z",
                source_commit_sha="c" * 40,
            )
            shadow_path = append_shadow_forecast_publication(
                data_root / "simulation" / "shadow_forecasts", publication
            )
            report = validate_simulation_artifacts(data_root, raw, required=True)
            self.assertFalse(report.errors, report.errors)
            self.assertTrue(any("parameter artifact" in item for item in report.facts))
            self.assertTrue(any("shadow forecasts: 1/1 valid" in item for item in report.facts))

            tampered = json.loads(shadow_path.read_text(encoding="utf-8"))
            tampered["parameter_artifact_sha256"] = "d" * 64
            shadow_path.write_text(json.dumps(tampered), encoding="utf-8")
            invalid = validate_simulation_artifacts(data_root, raw, required=True)
            self.assertTrue(any("shadow forecast is invalid" in item for item in invalid.errors))

    def test_archived_shadows_keep_their_own_parameter_commitment(self):
        raw = _raw_fight()
        historical_artifact = {
            "artifact_sha256": "d" * 64,
            "input_sha256": "e" * 64,
            "as_of_utc": "2019-01-01T00:00:00Z",
            "trained_through": "2018-12-01",
            "bootstrap_members": 200,
        }
        publication = build_shadow_forecast_publication(
            {},
            [],
            {
                "event_id": "historical-event",
                "event_url": "http://ufcstats.test/event-details/historical-event",
                "date": "2020-01-10T00:00:00Z",
                "title": "Historical UFC Test",
            },
            historical_artifact,
            forecast_issued_at_utc="2019-12-01T00:00:00Z",
            source_commit_sha="c" * 40,
        )
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            _write_round_data(data_root, raw)
            current_artifact, _ = _write_required_bundle(data_root)
            self.assertNotEqual(
                publication["parameter_artifact_sha256"],
                current_artifact.artifact_sha256,
            )
            append_shadow_forecast_publication(
                data_root / "simulation" / "shadow_forecasts", publication
            )
            report = validate_simulation_artifacts(data_root, raw, required=True)
        self.assertFalse(report.errors, report.errors)
        self.assertTrue(any("0 current / 1 archived" in fact for fact in report.facts))

    def test_current_upcoming_shadow_must_name_the_frozen_status_artifact(self):
        raw = _raw_fight()
        today = pd.Timestamp.now(tz="UTC").normalize()
        event_date = today + pd.Timedelta(days=30)
        issued = today - pd.Timedelta(days=1)
        stale_artifact = {
            "artifact_sha256": "d" * 64,
            "input_sha256": "e" * 64,
            "as_of_utc": (today - pd.Timedelta(days=20)).isoformat(),
            "trained_through": (today - pd.Timedelta(days=21)).date().isoformat(),
            "bootstrap_members": 200,
        }
        publication = build_shadow_forecast_publication(
            {},
            [],
            {
                "event_id": "upcoming-event",
                "event_url": "http://ufcstats.test/event-details/upcoming-event",
                "date": event_date.isoformat(),
                "title": "Upcoming UFC Test",
            },
            stale_artifact,
            forecast_issued_at_utc=issued.isoformat(),
            source_commit_sha="c" * 40,
        )
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            _write_round_data(data_root, raw)
            _write_required_bundle(data_root)
            append_shadow_forecast_publication(
                data_root / "simulation" / "shadow_forecasts", publication
            )
            report = validate_simulation_artifacts(data_root, raw, required=True)
        self.assertTrue(
            any(
                "current upcoming shadow names a different frozen" in error
                for error in report.errors
            ),
            report.errors,
        )

    def test_older_upcoming_shadow_survives_when_newer_forecast_is_active(self):
        raw = _raw_fight()
        today = pd.Timestamp.now(tz="UTC").normalize()
        event = {
            "event_id": "rotated-upcoming-event",
            "event_url": (
                "http://ufcstats.test/event-details/rotated-upcoming-event"
            ),
            "date": (today + pd.Timedelta(days=30)).isoformat(),
            "title": "Rotated Upcoming UFC Test",
        }
        stale_artifact = {
            "artifact_sha256": "d" * 64,
            "input_sha256": "e" * 64,
            "as_of_utc": (today - pd.Timedelta(days=20)).isoformat(),
            "trained_through": (today - pd.Timedelta(days=21)).date().isoformat(),
            "bootstrap_members": 200,
        }
        stale = build_shadow_forecast_publication(
            {},
            [],
            event,
            stale_artifact,
            forecast_issued_at_utc=(today - pd.Timedelta(days=2)).isoformat(),
            source_commit_sha="c" * 40,
        )
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            _write_round_data(data_root, raw)
            current_artifact, _ = _write_required_bundle(data_root)
            current = build_shadow_forecast_publication(
                {},
                [],
                event,
                current_artifact,
                forecast_issued_at_utc=(today - pd.Timedelta(days=1)).isoformat(),
                source_commit_sha="f" * 40,
            )
            shadow_root = data_root / "simulation" / "shadow_forecasts"
            append_shadow_forecast_publication(shadow_root, stale)
            append_shadow_forecast_publication(shadow_root, current)
            report = validate_simulation_artifacts(data_root, raw, required=True)
        self.assertFalse(report.errors, report.errors)
        self.assertTrue(any("2 current / 0 archived" in fact for fact in report.facts))

    def test_research_status_must_name_exact_parameter_and_backtest_hashes(self):
        raw = _raw_fight()
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            _write_round_data(data_root, raw)
            _write_required_bundle(data_root)
            status_path = data_root / "simulation" / "research_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["backtest_report_sha256"] = "0" * 64
            status_path.write_text(json.dumps(status), encoding="utf-8")
            report = validate_simulation_artifacts(data_root, raw, required=True)
        self.assertTrue(any("research-status gate is invalid" in item for item in report.errors))

    def test_enabled_shadow_cannot_rely_on_gate_booleans_without_evidence(self):
        raw = _raw_fight()
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            _write_round_data(data_root, raw)
            _write_required_bundle(data_root)
            status_path = data_root / "simulation" / "research_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["shadow_enabled"] = True
            status_path.write_text(json.dumps(status), encoding="utf-8")
            report = validate_simulation_artifacts(data_root, raw, required=True)
        self.assertTrue(
            any("shadow evidence gate failed" in item for item in report.errors),
            report.errors,
        )


if __name__ == "__main__":
    unittest.main()
