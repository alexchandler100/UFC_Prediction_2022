from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fight_sim.parameters import canonical_sha256  # noqa: E402
from fight_sim.upcoming import (  # noqa: E402
    AVAILABLE,
    WITHHELD_HISTORY,
    compact_website_aggregate,
    prior_ufc_exposure,
    validate_website_simulation_publication,
)


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


if __name__ == "__main__":
    unittest.main()
