from __future__ import annotations

from dataclasses import replace
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.parameters import (  # noqa: E402
    CausalParameterFitter,
    PARAMETER_STORAGE_FORMAT,
    TAKEDOWN_CONTROL_PARAMETER_MODEL_VERSION,
    ParameterFitConfig,
    canonical_json,
    canonical_sha256,
    inspect_parameter_artifact,
    load_parameter_artifact,
    load_parameter_artifact_cached,
    save_parameter_artifact,
)
from test_fight_sim_research import _profiles, _raw  # noqa: E402


def _round_rows(*, reconciled: bool, mismatched: bool = False) -> pd.DataFrame:
    raw = _raw().loc[lambda frame: frame["fight_url"].str.endswith("fight-1")]
    rows: list[dict[str, object]] = []
    for source in raw.to_dict("records"):
        fighter_url = source["fighter_url"]
        if mismatched and not rows:
            fighter_url = "http://ufcstats.com/fighter-details/not-in-bout"
        row: dict[str, object] = {
            "date": source["date"],
            "fight_url": source["fight_url"],
            "event_url": source["event_url"],
            "fighter_url": fighter_url,
            "opponent_url": source["opponent_url"],
            "round": 1,
            "round_seconds": 300,
            "sig_strikes_attempts": source["sig_strikes_attempts"],
            "division": source["division"],
            "takedowns_landed": source["takedowns_landed"],
            "control": source["control"],
        }
        if reconciled:
            row["reconciliation_status"] = "matched"
        rows.append(row)
    return pd.DataFrame(rows)


def _sufficient_stats() -> dict[str, float]:
    return {
        "minutes": 30.0,
        "sig_attempts": 300.0,
        "distance_minutes": 30.0,
        "clinch_minutes": 30.0,
        "ground_minutes": 30.0,
        "distance_attempts": 240.0,
        "clinch_attempts": 30.0,
        "ground_attempts": 30.0,
        "sig_landed": 135.0,
        "head_landed": 80.0,
        "body_landed": 35.0,
        "leg_landed": 20.0,
        "opp_sig_attempts": 280.0,
        "opp_sig_landed": 120.0,
        "knockdowns": 1.0,
        "opp_knockdowns": 1.0,
        "td_minutes": 30.0,
        "td_attempts": 12.0,
        "td_landed": 5.0,
        "opp_td_attempts": 10.0,
        "opp_td_landed": 4.0,
        "reversals": 1.0,
        "sub_minutes": 30.0,
        "sub_attempts": 4.0,
        "opp_sub_attempts": 3.0,
        "sub_wins": 1.0,
        "ko_wins": 1.0,
        "ko_losses": 0.0,
        "sub_losses": 0.0,
        "control_seconds": 240.0,
        "control_exposure": 1800.0,
        "opp_control_seconds": 180.0,
        "opp_control_exposure": 1800.0,
    }


class ParameterArtifactTests(unittest.TestCase):
    def test_takedown_control_candidate_is_explicit_and_recipe_reproducible(self):
        config = ParameterFitConfig(bootstrap_members=1, random_seed=13)
        rounds = _round_rows(reconciled=True)
        baseline = CausalParameterFitter(_raw(), _profiles(), rounds).fit(
            "2021-01-01", config=config, created_at_utc="2021-01-01T00:00:00Z"
        )
        candidate = CausalParameterFitter(
            _raw(),
            _profiles(),
            rounds,
            use_takedown_control_association=True,
        ).fit(
            "2021-01-01", config=config, created_at_utc="2021-01-01T00:00:00Z"
        )
        self.assertEqual(
            candidate.model_version, TAKEDOWN_CONTROL_PARAMETER_MODEL_VERSION
        )
        self.assertNotEqual(candidate.input_sha256, baseline.input_sha256)
        self.assertNotEqual(
            candidate.members[0].fighter_parameters["a"]["ground_control_rate"],
            baseline.members[0].fighter_parameters["a"]["ground_control_rate"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate-recipe.json.gz"
            save_parameter_artifact(path, candidate)
            restored = load_parameter_artifact(path)
        self.assertEqual(restored.to_dict(), candidate.to_dict())

    def test_recipe_artifact_populates_and_reuses_content_addressed_cache(self):
        artifact = CausalParameterFitter(_raw(), _profiles()).fit(
            "2022-01-01",
            config=ParameterFitConfig(bootstrap_members=2, random_seed=31),
            created_at_utc="2022-01-01T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recipe.json.gz"
            cache = root / "cache"
            save_parameter_artifact(source, artifact)
            first, first_hit, cache_path = load_parameter_artifact_cached(
                source, cache
            )
            self.assertFalse(first_hit)
            self.assertTrue(cache_path.is_file())
            with patch.object(
                CausalParameterFitter,
                "fit",
                side_effect=AssertionError("cache hit must not refit"),
            ):
                second, second_hit, second_path = load_parameter_artifact_cached(
                    source, cache
                )
        self.assertTrue(second_hit)
        self.assertEqual(second_path, cache_path)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_materialized_local_cache_loads_without_refitting(self):
        artifact = CausalParameterFitter(_raw(), _profiles()).fit(
            "2022-01-01",
            config=ParameterFitConfig(bootstrap_members=3, random_seed=29),
            created_at_utc="2022-01-01T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "materialized-cache.json.gz"
            save_parameter_artifact(path, artifact, materialized=True)
            physical = json.loads(gzip.decompress(path.read_bytes()))
            self.assertEqual(physical["codec"], "exact-columnar-v1")
            with patch.object(
                CausalParameterFitter,
                "fit",
                side_effect=AssertionError("materialized cache must not refit"),
            ):
                loaded = load_parameter_artifact(path)
        self.assertEqual(loaded.to_dict(), artifact.to_dict())

    def test_compact_round_trip_is_exact_and_physical_bytes_are_deterministic(self):
        artifact = CausalParameterFitter(_raw(), _profiles()).fit(
            "2022-01-01",
            config=ParameterFitConfig(bootstrap_members=3, random_seed=91),
            created_at_utc="2022-01-02T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json.gz"
            second = Path(directory) / "second.json.gz"
            save_parameter_artifact(first, artifact)
            save_parameter_artifact(second, artifact)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            physical = json.loads(gzip.decompress(first.read_bytes()))
            self.assertEqual(physical["storage_format"], PARAMETER_STORAGE_FORMAT)
            self.assertEqual(physical["codec"], "self-contained-causal-fit-v1")
            with patch.object(
                CausalParameterFitter,
                "fit",
                side_effect=AssertionError("fast inspection must not refit"),
            ):
                inspection = inspect_parameter_artifact(first)
            self.assertEqual(inspection.artifact_sha256, artifact.artifact_sha256)
            self.assertEqual(inspection.bootstrap_members, 3)
            self.assertEqual(
                inspection.members_sha256,
                physical["commitments"]["member_values_sha256"],
            )
            loaded = load_parameter_artifact(first)
        self.assertEqual(loaded.to_dict(), artifact.to_dict())

    def test_fast_inspection_rejects_wrapper_commitment_corruption(self):
        artifact = CausalParameterFitter(_raw(), _profiles()).fit(
            "2022-01-01",
            config=ParameterFitConfig(bootstrap_members=2, random_seed=33),
            created_at_utc="2022-01-02T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json.gz"
            save_parameter_artifact(path, artifact)
            physical = json.loads(gzip.decompress(path.read_bytes()))
            physical["commitments"]["member_count"] = 999
            path.write_bytes(
                gzip.compress(
                    (canonical_json(physical) + "\n").encode("utf-8"),
                    compresslevel=9,
                    mtime=0,
                )
            )
            with self.assertRaisesRegex(ValueError, "storage hash"):
                inspect_parameter_artifact(path)

    def test_original_row_oriented_artifact_remains_loadable(self):
        current = CausalParameterFitter(_raw(), _profiles()).fit(
            "2022-01-01",
            config=ParameterFitConfig(bootstrap_members=1, random_seed=19),
            created_at_utc="2022-01-02T00:00:00Z",
        )
        legacy = replace(
            current,
            schema_version=2,
            model_version="fight-sim-empirical-bayes-card-bootstrap-v2",
            artifact_sha256="",
        )
        legacy = replace(
            legacy,
            artifact_sha256=canonical_sha256(legacy.unhashed_dict()),
        ).validate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json.gz"
            path.write_bytes(
                gzip.compress(
                    (canonical_json(legacy.to_dict()) + "\n").encode("utf-8"),
                    compresslevel=9,
                    mtime=0,
                )
            )
            loaded = load_parameter_artifact(path)
        self.assertEqual(loaded.to_dict(), legacy.to_dict())

    def test_two_hundred_member_compact_artifact_stays_below_workflow_cap(self):
        artifact = CausalParameterFitter(_raw(), _profiles()).fit(
            "2022-01-01",
            config=ParameterFitConfig(bootstrap_members=200, random_seed=71),
            created_at_utc="2022-01-02T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameter_model.json.gz"
            save_parameter_artifact(path, artifact)
            size = path.stat().st_size
        self.assertLessEqual(size, 5 * 1024 * 1024)

    def test_artifact_identity_excludes_creation_time_and_future_profiles(self):
        config = ParameterFitConfig(bootstrap_members=2, random_seed=17)
        baseline = CausalParameterFitter(_raw(), _profiles()).fit(
            "2022-01-01",
            config=config,
            created_at_utc="2022-01-02T00:00:00Z",
        )
        appended_profiles = pd.concat(
            [
                _profiles().assign(name=lambda frame: frame["name"] + " ignored"),
                pd.DataFrame(
                    [
                        {
                            "url": "http://ufcstats.com/fighter-details/future",
                            "name": "Future Fighter",
                            "dob": "2000-01-01",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        changed_provenance = CausalParameterFitter(
            _raw(), appended_profiles
        ).fit(
            "2022-01-01",
            config=config,
            created_at_utc="2025-05-06T07:08:09Z",
        )
        self.assertNotEqual(
            baseline.created_at_utc, changed_provenance.created_at_utc
        )
        self.assertEqual(baseline.input_sha256, changed_provenance.input_sha256)
        self.assertEqual(baseline.members, changed_provenance.members)
        self.assertEqual(
            baseline.artifact_sha256, changed_provenance.artifact_sha256
        )

    def test_strike_phase_mix_cannot_inflate_td_or_submission_rate(self):
        first = _sufficient_stats()
        second = dict(first)
        second.update(
            {
                "distance_attempts": 296.0,
                "clinch_attempts": 2.0,
                "ground_attempts": 2.0,
            }
        )
        config = ParameterFitConfig(bootstrap_members=1)
        first_fit = CausalParameterFitter._parameters_from_sufficient(
            first, None, config
        )
        second_fit = CausalParameterFitter._parameters_from_sufficient(
            second, None, config
        )
        self.assertEqual(
            first_fit["takedown_attempt_rate"],
            second_fit["takedown_attempt_rate"],
        )
        self.assertEqual(
            first_fit["submission_attempt_rate"],
            second_fit["submission_attempt_rate"],
        )

    def test_round_fitting_is_matched_only_and_identity_checked(self):
        legacy = _round_rows(reconciled=False)
        strict = CausalParameterFitter(_raw(), _profiles(), legacy).fit(
            "2021-01-01",
            config=ParameterFitConfig(bootstrap_members=1),
            created_at_utc="2021-01-02T00:00:00Z",
        )
        self.assertEqual(strict.observed_round_sides, 0)
        self.assertEqual(
            strict.round_reconciliation_counts["legacy_unlabeled_excluded"], 2
        )

        research_override = CausalParameterFitter(
            _raw(),
            _profiles(),
            legacy,
            allow_legacy_unreconciled_rounds=True,
        ).fit(
            "2021-01-01",
            config=ParameterFitConfig(bootstrap_members=1),
            created_at_utc="2021-01-02T00:00:00Z",
        )
        self.assertEqual(research_override.observed_round_sides, 2)
        self.assertEqual(
            research_override.round_reconciliation_counts[
                "legacy_unlabeled_eligible"
            ],
            2,
        )

        mismatched = _round_rows(reconciled=True, mismatched=True)
        with self.assertRaisesRegex(ValueError, "identity is absent"):
            CausalParameterFitter(_raw(), _profiles(), mismatched).fit(
                "2021-01-01",
                config=ParameterFitConfig(bootstrap_members=1),
                created_at_utc="2021-01-02T00:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
