import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_tracker import (  # noqa: E402
    MarketDataError,
    SimulationComparisonDecision,
    SimulationComparisonDecisionStore,
    build_simulation_comparison_decisions,
    equal_logit_pool,
    simulation_comparison_report,
)
from market_tracker._common import canonical_hash  # noqa: E402


def _base(*, fighter="red", opponent="blue"):
    return SimpleNamespace(
        decision_id="base-decision",
        matchup_id="matchup-one",
        event_id="event-one",
        fighter_id=fighter,
        opponent_id=opponent,
        event_date="2026-09-05",
        timing_precision="timestamp",
        event_start_utc="2026-09-05T02:00:00Z",
        decision_issued_at_utc="2026-09-04T02:00:00.000000Z",
        market_probability=0.55,
        model_probability=0.65,
    )


def _publication(*, issued="2026-09-03T12:00:00Z"):
    body = {
        "schema_version": 1,
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "production_influence": "none",
        "event_id": "event-one",
        "event_date": "2026-09-05T00:00:00+00:00",
        "forecast_issued_at_utc": issued,
        "parameter_artifact_sha256": "a" * 64,
        "mechanics_profile_id": "mechanics-fixture",
        "matchups": [
            {
                "matchup_id": "matchup-one",
                "fighter_id": "red",
                "opponent_id": "blue",
                "status": "available",
                "aggregate": {
                    "outcome_probabilities": {
                        "red_ko_tko": 0.40,
                        "red_decision": 0.25,
                        "blue_ko_tko": 0.20,
                        "blue_decision": 0.10,
                        "draw": 0.05,
                    }
                },
            }
        ],
    }
    body["publication_sha256"] = canonical_hash(body)
    return body


class SimulationComparisonTests(unittest.TestCase):
    def test_fixed_probability_pool_is_swap_symmetric(self):
        pooled = equal_logit_pool((0.55, 0.65, 0.70))
        swapped = equal_logit_pool((0.45, 0.35, 0.30))
        self.assertAlmostEqual(pooled, 1.0 - swapped, places=15)

    def test_freezes_all_fixed_blends_and_round_trips_store(self):
        base = _base()
        record = SimulationComparisonDecision.create(
            base,
            _publication(),
            comparison_issued_at_utc="2026-09-04T02:01:00Z",
        )
        self.assertAlmostEqual(record.simulation_probability, 0.65 / 0.95)
        self.assertAlmostEqual(
            record.equal_three_probability,
            equal_logit_pool((0.55, 0.65, 0.65 / 0.95)),
        )
        record.validate_integrity()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SimulationComparisonDecisionStore(
                root / "comparisons.csv", root / "comparisons.jsonl"
            )
            result = store.append((record,))
            self.assertEqual(result.total_records, 1)
            self.assertEqual(store.read(), (record,))

        settlement = SimpleNamespace(decision_id=base.decision_id, target=1)
        report = simulation_comparison_report((record,), (settlement,), (base,))
        self.assertEqual(report["scored_fights"], 1)
        self.assertEqual(len(report["scores"]), 7)
        self.assertEqual(
            report["scores"]["market_model_simulation_thirds"]["count"], 1
        )
        self.assertFalse(report["execution_enabled"])

    def test_requires_simulation_to_exist_by_t24_and_comparison_before_fight(self):
        base = _base()
        build = build_simulation_comparison_decisions(
            (base,),
            (),
            _publication(issued="2026-09-04T03:00:00Z"),
            comparison_issued_at_utc="2026-09-04T03:01:00Z",
        )
        self.assertEqual(build.decisions, ())
        self.assertEqual(build.unavailable_or_mismatched, 1)

        with self.assertRaisesRegex(MarketDataError, "strictly earlier"):
            SimulationComparisonDecision.create(
                base,
                _publication(),
                comparison_issued_at_utc="2026-09-05T02:00:00Z",
            )

    def test_publication_hash_is_checked(self):
        publication = json.loads(json.dumps(_publication()))
        publication["mechanics_profile_id"] = "changed-after-hash"
        build = build_simulation_comparison_decisions(
            (_base(),),
            (),
            publication,
            comparison_issued_at_utc="2026-09-04T02:01:00Z",
        )
        self.assertEqual(build.publication_status, "invalid")
        self.assertEqual(build.decisions, ())


if __name__ == "__main__":
    unittest.main()
