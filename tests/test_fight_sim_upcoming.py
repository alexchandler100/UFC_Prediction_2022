from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import json
import gzip
import tempfile
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fight_sim.parameters import canonical_json, canonical_sha256  # noqa: E402
from fight_sim.monte_carlo import ConvergenceDiagnostics  # noqa: E402
from fight_sim.upcoming import (  # noqa: E402
    AVAILABLE,
    WITHHELD_HISTORY,
    compact_website_aggregate,
    execute_upcoming_card,
    prior_ufc_exposure,
    validate_website_simulation_publication,
    _write_authority,
)
from market_tracker import matchup_id_for  # noqa: E402


class UpcomingSimulationTests(unittest.TestCase):
    def test_prior_exposure_is_distinct_and_strictly_before_cutoff(self):
        raw = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "fight_url": "http://ufcstats.com/fight-details/f1",
                    "fighter_url": "http://ufcstats.com/fighter-details/a",
                },
                {
                    "date": "2025-01-01",
                    "fight_url": "http://ufcstats.com/fight-details/f1",
                    "fighter_url": "http://ufcstats.com/fighter-details/a",
                },
                {
                    "date": "2025-02-01",
                    "fight_url": "http://ufcstats.com/fight-details/f2",
                    "fighter_url": "http://ufcstats.com/fighter-details/a",
                },
                {
                    "date": "2025-03-01",
                    "fight_url": "http://ufcstats.com/fight-details/f3",
                    "fighter_url": "http://ufcstats.com/fighter-details/a",
                },
            ]
        )
        exposure = prior_ufc_exposure(raw, "2025-03-01T00:00:00Z")
        self.assertEqual(exposure, {"a": 2})

    def test_withheld_matchup_is_explicit_and_contains_no_forecast(self):
        publication = {
            "schema_version": 1,
            "candidate_only": True,
            "paper_only": True,
            "execution_enabled": False,
            "production_influence": "none",
            "matchup_count": 1,
            "available_matchups": 0,
            "excluded_matchups": 1,
            "matchups": [
                {
                    "matchup_id": "matchup-test",
                    "status": WITHHELD_HISTORY,
                    "unavailable_reason": "both fighters require three prior bouts",
                }
            ],
        }
        publication["publication_sha256"] = canonical_sha256(publication)
        validated = validate_website_simulation_publication(publication)
        self.assertEqual(validated["excluded_matchups"], 1)
        self.assertNotIn("aggregate", validated["matchups"][0])

    def test_publication_hash_detects_tampering(self):
        publication = {
            "schema_version": 1,
            "candidate_only": True,
            "paper_only": True,
            "execution_enabled": False,
            "production_influence": "none",
            "matchup_count": 1,
            "available_matchups": 0,
            "excluded_matchups": 1,
            "matchups": [
                {
                    "matchup_id": "matchup-test",
                    "status": WITHHELD_HISTORY,
                    "unavailable_reason": "insufficient history",
                }
            ],
        }
        publication["publication_sha256"] = canonical_sha256(publication)
        publication["excluded_matchups"] = 2
        with self.assertRaisesRegex(ValueError, "excluded matchup count"):
            validate_website_simulation_publication(publication)

    def test_available_matchup_requires_compact_exact_count_authority(self):
        aggregate = compact_website_aggregate(
            {
                "matchup_id": "matchup-test",
                "total_paths": 2,
                "bootstrap_members": 1,
                "scheduled_rounds": 3,
                "outcome_counts": {"red_decision": 2},
                "outcome_probabilities": {"red_decision": 1.0},
                "total_lines": [],
                "survival": [],
            }
        )
        publication = {
            "schema_version": 1,
            "candidate_only": True,
            "paper_only": True,
            "execution_enabled": False,
            "production_influence": "none",
            "matchup_count": 1,
            "available_matchups": 1,
            "excluded_matchups": 0,
            "matchups": [
                {
                    "matchup_id": "matchup-test",
                    "status": AVAILABLE,
                    "aggregate": aggregate,
                }
            ],
        }
        publication["publication_sha256"] = canonical_sha256(publication)
        self.assertEqual(
            validate_website_simulation_publication(publication)["available_matchups"],
            1,
        )

    def test_website_compaction_omits_unused_member_and_statistic_detail(self):
        aggregate = compact_website_aggregate(
            {
                "matchup_id": "matchup-test",
                "total_paths": 2,
                "bootstrap_members": 1,
                "scheduled_rounds": 3,
                "outcome_counts": {"red_decision": 2},
                "outcome_probabilities": {"red_decision": 1.0},
                "total_lines": [],
                "survival": [],
                "bootstrap_outcome_counts": [{"bootstrap_member": 0}],
                "statistic_distributions": [{"statistic": "red_knockdowns"}],
                "statistic_uncertainty": [{"statistic": "red_knockdowns"}],
                "uncertainty": [
                    {
                        "metric": "red_win",
                        "estimate": 1.0,
                        "conditional_probabilities": {"0": 1.0},
                    }
                ],
            }
        )
        self.assertNotIn("bootstrap_outcome_counts", aggregate)
        self.assertNotIn("statistic_distributions", aggregate)
        self.assertNotIn("statistic_uncertainty", aggregate)
        self.assertNotIn("conditional_probabilities", aggregate["uncertainty"][0])
        self.assertEqual(compact_website_aggregate(aggregate), aggregate)

    def test_authority_hash_is_reproducible_after_json_round_trip(self):
        full = {
            "matchup_id": "matchup-test",
            "total_paths": 2,
            "bootstrap_members": 1,
            "scheduled_rounds": 3,
            "outcome_counts": {"red_decision": 2},
            "outcome_probabilities": {"red_decision": 1.0},
            "total_lines": [],
            "survival": [],
            # Numeric keys expose pre/post-JSON sort-order differences (2, 10
            # versus "10", "2") unless normalization precedes hashing.
            "numeric_key_counts": {2: 1, 10: 1},
        }
        normalized = json.loads(canonical_json(full))
        with tempfile.TemporaryDirectory() as directory:
            authority = _write_authority(Path(directory), full)
            stored = json.loads(gzip.decompress(authority.read_bytes()))
            digest = canonical_sha256(stored)

        self.assertEqual(stored, normalized)
        self.assertIn(digest, authority.name)
        self.assertEqual(
            compact_website_aggregate(full)["local_aggregate_sha256"],
            digest,
        )

    def test_upcoming_card_resumes_adaptive_checkpoint_then_skips_completed_matchup(self):
        event_id = "9d61d8cb1c354867"
        red_id = "1111111111111111"
        blue_id = "2222222222222222"
        matchup_id = matchup_id_for(event_id, red_id, blue_id)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            card_path = root / "card.json"
            outcome_path = root / "outcomes.json"
            raw_path = root / "raw.csv"
            profiles_path = root / "profiles.csv"
            round_path = root / "rounds.csv"
            parameter_path = root / "parameters.json.gz"
            output_dir = root / "run"
            website_path = root / "simulation.json"
            card_path.write_text(
                json.dumps(
                    {
                        "date": "2026-08-29",
                        "title": "Test card",
                        "event_id": event_id,
                        "event_url": f"http://ufcstats.com/event-details/{event_id}",
                    }
                ),
                encoding="utf-8",
            )
            outcome_path.write_text(
                json.dumps(
                    {
                        "event_id": event_id,
                        "matchups": [
                            {
                                "bout_order": 0,
                                "fighter_id": red_id,
                                "fighter_name": "Red Fighter",
                                "opponent_id": blue_id,
                                "opponent_name": "Blue Fighter",
                                "division": "Lightweight",
                                "scheduled_rounds": 3,
                                "matchup_id": matchup_id,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            # Round data is optional for a self-contained reused parameter
            # artifact; the immutable run contract records its absence.
            for path in (raw_path, profiles_path, parameter_path):
                path.write_text("stable-test-input\n", encoding="utf-8")
            raw = pd.DataFrame(
                [
                    {
                        "date": f"2024-0{number + 1}-01",
                        "fight_url": f"http://ufcstats.com/fight-details/red{number}",
                        "fighter_url": f"http://ufcstats.com/fighter-details/{red_id}",
                    }
                    for number in range(3)
                ]
                + [
                    {
                        "date": f"2024-0{number + 1}-02",
                        "fight_url": f"http://ufcstats.com/fight-details/blue{number}",
                        "fighter_url": f"http://ufcstats.com/fighter-details/{blue_id}",
                    }
                    for number in range(3)
                ]
            )
            artifact = SimpleNamespace(
                members=(object(), object()),
                artifact_sha256="a" * 64,
                input_sha256="b" * 64,
                as_of_utc="2026-08-25T00:00:00+00:00",
                validate=lambda: None,
            )
            diagnostic = ConvergenceDiagnostics(
                paths_per_member=4,
                total_paths=8,
                winner_process_mcse=0.01,
                split_estimate_difference=0.0,
                split_combined_mcse=0.02,
                parameter_quantile_max_shift=0.0,
                mcse_within_target=True,
                headline_batches_stable=True,
                parameter_quantiles_stable=True,
            )

            class FakeForecast:
                total_paths = 8

                @staticmethod
                def to_dict():
                    return {
                        "matchup_id": matchup_id,
                        "total_paths": 8,
                        "bootstrap_members": 2,
                        "scheduled_rounds": 3,
                        "outcome_counts": {"red_decision": 8},
                        "outcome_probabilities": {"red_decision": 1.0},
                        "total_lines": [],
                        "survival": [],
                    }

            completed = SimpleNamespace(
                forecast=FakeForecast(),
                convergence=(diagnostic,),
                converged=True,
                invariant_failures=(),
            )
            captured = []

            class Interrupted(RuntimeError):
                pass

            def interrupted_run(*args, **kwargs):
                self.assertIsNone(kwargs["resume_checkpoint"])
                checkpoint = {"paths_per_member": 2, "test": "exact-state"}
                captured.append(checkpoint)
                kwargs["checkpoint_callback"](checkpoint)
                raise Interrupted

            common_patches = (
                patch(
                    "fight_sim.upcoming.load_research_inputs",
                    return_value=(raw, object(), object()),
                ),
                patch(
                    "fight_sim.upcoming.CausalParameterFitter",
                    return_value=object(),
                ),
                patch(
                    "fight_sim.upcoming.load_parameter_artifact_cached",
                    return_value=(artifact, True, root / "cached.json.gz"),
                ),
                patch("fight_sim.upcoming.save_parameter_artifact"),
                patch("fight_sim.upcoming.build_specs", return_value=(object(),)),
            )
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4], patch(
                "fight_sim.upcoming.run_adaptive_nested", side_effect=interrupted_run
            ):
                with self.assertRaises(Interrupted):
                    execute_upcoming_card(
                        card_path=card_path,
                        outcome_path=outcome_path,
                        raw_path=raw_path,
                        profiles_path=profiles_path,
                        round_path=round_path,
                        output_dir=output_dir,
                        website_output=website_path,
                        parameter_artifact_path=parameter_path,
                        bootstrap_members=2,
                        initial_paths_per_member=2,
                        max_paths_per_member=4,
                        issued_at_utc="2026-08-26T00:00:00Z",
                    )

            checkpoint_path = output_dir / "adaptive-checkpoints" / f"{matchup_id}.json.gz"
            self.assertTrue((output_dir / "run-manifest.json").is_file())
            self.assertTrue(checkpoint_path.is_file())
            self.assertFalse(website_path.exists())

            def resumed_run(*args, **kwargs):
                self.assertEqual(kwargs["resume_checkpoint"], captured[0])
                return completed

            with patch(
                "fight_sim.upcoming.load_research_inputs",
                return_value=(raw, object(), object()),
            ), patch(
                "fight_sim.upcoming.CausalParameterFitter", return_value=object()
            ), patch(
                "fight_sim.upcoming.load_parameter_artifact_cached",
                return_value=(artifact, True, root / "cached.json.gz"),
            ), patch("fight_sim.upcoming.save_parameter_artifact"), patch(
                "fight_sim.upcoming.build_specs", return_value=(object(),)
            ), patch(
                "fight_sim.upcoming.run_adaptive_nested", side_effect=resumed_run
            ) as resumed_mock:
                _, publication = execute_upcoming_card(
                    card_path=card_path,
                    outcome_path=outcome_path,
                    raw_path=raw_path,
                    profiles_path=profiles_path,
                    round_path=round_path,
                    output_dir=output_dir,
                    website_output=website_path,
                    parameter_artifact_path=parameter_path,
                    bootstrap_members=2,
                    initial_paths_per_member=2,
                    max_paths_per_member=4,
                    resume=True,
                )
                self.assertEqual(resumed_mock.call_count, 1)

            self.assertEqual(publication["available_matchups"], 1)
            self.assertTrue(website_path.is_file())
            self.assertFalse(checkpoint_path.exists())
            with patch(
                "fight_sim.upcoming.load_research_inputs",
                return_value=(raw, object(), object()),
            ), patch(
                "fight_sim.upcoming.CausalParameterFitter", return_value=object()
            ), patch(
                "fight_sim.upcoming.load_parameter_artifact_cached",
                return_value=(artifact, True, root / "cached.json.gz"),
            ), patch("fight_sim.upcoming.save_parameter_artifact"), patch(
                "fight_sim.upcoming.build_specs", return_value=(object(),)
            ), patch("fight_sim.upcoming.run_adaptive_nested") as skipped_mock:
                execute_upcoming_card(
                    card_path=card_path,
                    outcome_path=outcome_path,
                    raw_path=raw_path,
                    profiles_path=profiles_path,
                    round_path=round_path,
                    output_dir=output_dir,
                    website_output=website_path,
                    parameter_artifact_path=parameter_path,
                    bootstrap_members=2,
                    initial_paths_per_member=2,
                    max_paths_per_member=4,
                    resume=True,
                )
                skipped_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
