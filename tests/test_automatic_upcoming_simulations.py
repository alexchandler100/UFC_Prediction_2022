from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fight_sim.catalog import (  # noqa: E402
    AVAILABLE,
    PENDING_NEW,
    build_automatic_website_publication,
    execute_upcoming_catalog,
    load_automatic_records,
    validate_automatic_website_publication,
)
from fight_sim.upcoming import compact_website_aggregate  # noqa: E402
from upcoming_bet_board import build_upcoming_forecast_publication  # noqa: E402


def _catalog() -> dict[str, object]:
    frame = pd.DataFrame(
        [
            {
                "date": "2030-01-05",
                "event id": "aaaaaaaaaaaaaaaa",
                "event url": "http://ufcstats.com/event-details/aaaaaaaaaaaaaaaa",
                "event title": "Future UFC event",
                "bout order": 0,
                "fighter id": "1111111111111111",
                "opponent id": "2222222222222222",
                "fighter name": "Red Fighter",
                "opponent name": "Blue Fighter",
                "division": "Lightweight",
                "model id": "model",
                "model version": "v1",
                "model trained through": "2029-12-01",
                "model probability": 0.6,
                "model status": "available",
                "forecast issued at": "2029-12-02T00:00:00Z",
                "forecast source commit": "a" * 40,
            }
        ]
    )
    return build_upcoming_forecast_publication(
        frame, generated_at_utc="2029-12-02T00:00:00Z"
    )


def _raw_history() -> pd.DataFrame:
    rows = []
    for fighter_id in ("1111111111111111", "2222222222222222"):
        for number in range(3):
            rows.append(
                {
                    "date": f"2029-0{number + 1}-01",
                    "fight_url": (
                        "http://ufcstats.com/fight-details/"
                        f"{fighter_id[:4]}{number:012d}"
                    ),
                    "fighter_url": (
                        "http://ufcstats.com/fighter-details/" + fighter_id
                    ),
                }
            )
    return pd.DataFrame(rows)


class AutomaticUpcomingSimulationTests(unittest.TestCase):
    def test_new_matchup_is_recorded_once_and_reused(self):
        catalog = _catalog()
        raw = _raw_history()
        matchup_id = str(catalog["matchups"][0]["matchup_id"])
        artifact = SimpleNamespace(
            artifact_sha256="a" * 64,
            input_sha256="b" * 64,
        )
        fitter = SimpleNamespace(fit=lambda *args, **kwargs: artifact)
        forecast = SimpleNamespace(
            total_paths=8,
            to_dict=lambda: {
                "matchup_id": matchup_id,
                "total_paths": 8,
                "bootstrap_members": 2,
                "scheduled_rounds": 5,
                "outcome_counts": {"red_decision": 8},
                "outcome_probabilities": {"red_decision": 1.0},
                "total_lines": [],
                "survival": [],
            },
        )
        result = SimpleNamespace(forecast=forecast)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.json"
            records = root / "records"
            website = root / "website.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            patches = (
                patch(
                    "fight_sim.catalog.load_research_inputs",
                    return_value=(raw, object(), object()),
                ),
                patch("fight_sim.catalog.CausalParameterFitter", return_value=fitter),
                patch("fight_sim.catalog.build_specs", return_value=(object(), object())),
                patch("fight_sim.catalog.run_nested", return_value=result),
            )
            with patches[0], patches[1], patches[2], patches[3] as run_mock:
                _, first = execute_upcoming_catalog(
                    catalog_path=catalog_path,
                    record_directory=records,
                    website_output=website,
                    bootstrap_members=2,
                    paths_per_member=4,
                    issued_at_utc="2029-12-03T00:00:00Z",
                )
            self.assertEqual(run_mock.call_count, 1)
            self.assertEqual(first["available_matchups"], 1)
            self.assertEqual(first["pending_matchups"], 0)
            self.assertEqual(first["matchups"][0]["status"], AVAILABLE)
            self.assertEqual(len(load_automatic_records(records)), 1)

            with patch(
                "fight_sim.catalog.load_research_inputs",
                return_value=(raw, object(), object()),
            ), patch("fight_sim.catalog.CausalParameterFitter") as fitter_mock, patch(
                "fight_sim.catalog.run_nested"
            ) as run_mock:
                _, second = execute_upcoming_catalog(
                    catalog_path=catalog_path,
                    record_directory=records,
                    website_output=website,
                    bootstrap_members=2,
                    paths_per_member=4,
                    issued_at_utc="2029-12-03T00:00:00Z",
                )
            fitter_mock.assert_not_called()
            run_mock.assert_not_called()
            self.assertEqual(second["available_matchups"], 1)

    def test_publication_reports_queued_matchups_without_fake_results(self):
        catalog = _catalog()
        from fight_sim.catalog import _catalog_rows
        from fight_sim.domain import SimulatorConfig

        rows = _catalog_rows(
            catalog,
            simulator_config=SimulatorConfig(),
            mechanics_profile_id="mechanics-test",
            bootstrap_members=2,
            paths_per_member=4,
        )
        publication = build_automatic_website_publication(
            catalog,
            rows,
            {},
            {"1111111111111111": 3, "2222222222222222": 3},
            generated_at_utc="2029-12-03T00:00:00Z",
            minimum_prior_ufc_fights=3,
            bootstrap_members=2,
            paths_per_member=4,
            mechanics_profile_id="mechanics-test",
        )
        validated = validate_automatic_website_publication(publication)
        self.assertEqual(validated["pending_matchups"], 1)
        self.assertEqual(validated["matchups"][0]["status"], PENDING_NEW)
        self.assertNotIn("aggregate", validated["matchups"][0])

    def test_workflow_runs_incremental_simulator_without_review_file_gate(self):
        workflow = (ROOT / ".github/workflows/update-data.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("python src/update_upcoming_simulations.py", workflow)
        self.assertIn("--paths-per-member 64", workflow)
        self.assertIn("src/content/data/simulation/upcoming_matchups", workflow)
        simulation_job = workflow.split("  upcoming_simulations:", 1)[1]
        self.assertNotIn("research_status.json", simulation_job)
        self.assertNotIn("shadow_gate", simulation_job)

    def test_compact_fixture_is_valid_for_archive_contract(self):
        aggregate = compact_website_aggregate(
            {
                "matchup_id": "matchup-test",
                "total_paths": 8,
                "bootstrap_members": 2,
                "scheduled_rounds": 3,
                "outcome_counts": {"red_decision": 8},
                "outcome_probabilities": {"red_decision": 1.0},
                "total_lines": [],
                "survival": [],
            }
        )
        self.assertEqual(aggregate["total_paths"], 8)
        self.assertIn("local_aggregate_sha256", aggregate)


if __name__ == "__main__":
    unittest.main()
