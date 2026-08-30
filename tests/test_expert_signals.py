import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_tracker import (  # noqa: E402
    ExpertPick,
    ExpertPickStore,
    ExpertSourcePolicy,
    MarketDataError,
    StoreIntegrityError,
    load_expert_source_registry,
    validate_expert_pick,
)


def _pick(**overrides):
    values = {
        "observed_at_utc": "2026-08-28T12:00:00Z",
        "issued_at_utc": "2026-08-28T11:00:00Z",
        "event_date": "2026-08-30",
        "timing_precision": "timestamp",
        "event_start_utc": "2026-08-30T18:00:00Z",
        "analyst_id": "public-analyst",
        "source_url": "https://example.com/picks/123",
        "source_record_id": "pick-123",
        "source_text_sha256": "a" * 64,
        "event_id": "event-1",
        "matchup_id": "matchup-1",
        "selected_fighter_id": "fighter-a",
        "opponent_id": "fighter-b",
        "selected_fighter_name": "Fighter A",
        "opponent_name": "Fighter B",
        "posted_moneyline": "+125",
    }
    values.update(overrides)
    return ExpertPick.create(**values)


class ExpertSignalTests(unittest.TestCase):
    def test_pick_is_prospective_and_paper_only(self):
        pick = _pick()
        self.assertTrue(pick.paper_only)
        self.assertFalse(pick.execution_enabled)
        self.assertEqual(pick.posted_moneyline, "125")
        with self.assertRaisesRegex(MarketDataError, "strictly earlier"):
            _pick(observed_at_utc="2026-08-30T18:00:00Z")
        with self.assertRaisesRegex(MarketDataError, "later than observed"):
            _pick(issued_at_utc="2026-08-28T13:00:00Z")

    def test_enabled_source_policy_checks_public_host(self):
        policy = ExpertSourcePolicy.from_mapping(
            {
                "analyst_id": "public-analyst",
                "display_name": "Public Analyst",
                "enabled": True,
                "free_public": True,
                "timestamp_verifiable": True,
                "allowed_hosts": ["example.com"],
            }
        )
        validate_expert_pick(policy, _pick())
        with self.assertRaisesRegex(MarketDataError, "not allowed"):
            validate_expert_pick(policy, _pick(source_url="https://invalid.test/pick"))

    def test_disabled_source_is_rejected(self):
        policy = ExpertSourcePolicy.from_mapping(
            {
                "analyst_id": "public-analyst",
                "display_name": "Public Analyst",
                "enabled": False,
                "free_public": True,
                "timestamp_verifiable": True,
                "allowed_hosts": ["example.com"],
            }
        )
        with self.assertRaisesRegex(MarketDataError, "not enabled"):
            validate_expert_pick(policy, _pick())

    def test_append_only_mirrors_reject_rewrites(self):
        original = _pick()
        rewritten = _pick(issued_at_utc="2026-08-28T10:00:00Z")
        self.assertEqual(original.pick_id, rewritten.pick_id)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ExpertPickStore(root / "picks.csv", root / "picks.jsonl")
            result = store.append((original,))
            self.assertEqual(result.total_records, 1)
            duplicate = store.append((original,))
            self.assertEqual(duplicate.duplicate_ids, (original.pick_id,))
            with self.assertRaisesRegex(StoreIntegrityError, "rewritten"):
                store.append((rewritten,))

    def test_registry_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            source = {
                "sources": [
                    {
                        "analyst_id": "same",
                        "display_name": "One",
                        "enabled": False,
                        "free_public": False,
                        "timestamp_verifiable": False,
                        "allowed_hosts": ["example.com"],
                    },
                    {
                        "analyst_id": "same",
                        "display_name": "Two",
                        "enabled": False,
                        "free_public": False,
                        "timestamp_verifiable": False,
                        "allowed_hosts": ["example.org"],
                    },
                ]
            }
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(MarketDataError, "duplicate"):
                load_expert_source_registry(path)


if __name__ == "__main__":
    unittest.main()
