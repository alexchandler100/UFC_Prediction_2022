import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_tracker import (  # noqa: E402
    EarlyMarketLink,
    EarlyMarketLinkStore,
    EarlyMarketObservation,
    EarlyMarketObservationStore,
    MarketDataError,
    StoreIntegrityError,
    matchup_id_for,
)
import capture_market_snapshot as collector  # noqa: E402


OBSERVED = datetime(2026, 8, 14, 17, 0, tzinfo=timezone.utc)
COMMENCE = "2026-08-30T18:00:00Z"
PAYLOAD_HASH = "a" * 64


def _observation(**overrides):
    values = {
        "first_capture_id": "capture-one",
        "first_observed_at_utc": OBSERVED,
        "source": "the-odds-api.com",
        "source_payload_sha256": PAYLOAD_HASH,
        "source_event_id": "source-event-one",
        "source_commence_time_utc": COMMENCE,
        "source_fighter_name": "Alpha One",
        "source_opponent_name": "Beta Two",
        "book": "Book A",
        "source_book_key": "book-a",
        "source_quote_updated_at_utc": "2026-08-14T16:59:30Z",
        "market": "h2h",
        "outcome_a": "Alpha One",
        "outcome_b": "Beta Two",
        "outcome_a_moneyline": -120,
        "outcome_b_moneyline": 105,
    }
    values.update(overrides)
    return EarlyMarketObservation.create(**values)


class EarlyMarketTests(unittest.TestCase):
    def test_repeated_unchanged_response_reuses_first_sighting(self):
        frame = pd.DataFrame(
            [
                {
                    "source event id": "source-event-one",
                    "source commence time": COMMENCE,
                    "fighter name": "Alpha One",
                    "opponent name": "Beta Two",
                    "fighter Book A": -120,
                    "opponent Book A": 105,
                    "source Book A key": "book-a",
                    "source Book A last update": "2026-08-14T16:59:30Z",
                }
            ]
        )
        first, _ = collector._build_early_market_observations(
            frame,
            None,
            collector._book_columns(frame),
            (),
            capture_id="capture-one",
            observed_at=OBSERVED,
            source=collector.ODDS_API_SOURCE,
            source_payload_sha256=PAYLOAD_HASH,
            published_event_day="2026-08-16",
        )
        repeated, _ = collector._build_early_market_observations(
            frame,
            None,
            collector._book_columns(frame),
            first,
            capture_id="capture-two",
            observed_at=datetime(2026, 8, 15, 17, 0, tzinfo=timezone.utc),
            source=collector.ODDS_API_SOURCE,
            source_payload_sha256="b" * 64,
            published_event_day="2026-08-16",
        )
        self.assertEqual(repeated, first)

    def test_distinct_states_are_append_only_and_mirrored(self):
        first = _observation()
        changed = _observation(
            source_quote_updated_at_utc="2026-08-14T17:02:00Z",
            outcome_a_moneyline=-125,
            outcome_b_moneyline=110,
        )
        self.assertNotEqual(first.observation_id, changed.observation_id)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = EarlyMarketObservationStore(
                root / "early.csv", root / "early.jsonl"
            )
            first_result = store.append((first,))
            duplicate_result = store.append((first,))
            changed_result = store.append((changed,))
            self.assertEqual(len(first_result.added_ids), 1)
            self.assertEqual(duplicate_result.duplicate_ids, (first.observation_id,))
            self.assertEqual(len(changed_result.added_ids), 1)
            self.assertEqual(store.read(), (first, changed))

    def test_same_state_cannot_rewrite_its_first_seen_provenance(self):
        first = _observation()
        later_copy = _observation(
            first_capture_id="capture-two",
            first_observed_at_utc="2026-08-15T17:00:00Z",
            source_payload_sha256="b" * 64,
        )
        self.assertEqual(first.observation_id, later_copy.observation_id)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = EarlyMarketObservationStore(
                root / "early.csv", root / "early.jsonl"
            )
            store.append((first,))
            with self.assertRaisesRegex(StoreIntegrityError, "rewritten"):
                store.append((later_copy,))

    def test_total_round_contract_and_pre_fight_cutoff(self):
        total = _observation(
            market="total_rounds",
            line=2.5,
            outcome_a="Over",
            outcome_b="Under",
            outcome_a_moneyline=-110,
            outcome_b_moneyline=-105,
        )
        self.assertEqual(total.line, "2.5")
        with self.assertRaisesRegex(MarketDataError, "before commence"):
            _observation(first_observed_at_utc=COMMENCE)

    def test_official_link_preserves_source_orientation(self):
        event_id = "ufc-event"
        fighter_id = "fighter-alpha"
        opponent_id = "fighter-beta"
        link = EarlyMarketLink.create(
            first_linked_at_utc=OBSERVED,
            first_capture_id="capture-one",
            source="the-odds-api.com",
            source_event_id="source-event-one",
            source_commence_time_utc=COMMENCE,
            source_fighter_name="Beta Two",
            source_opponent_name="Alpha One",
            ufc_event_id=event_id,
            matchup_id=matchup_id_for(event_id, fighter_id, opponent_id),
            source_fighter_ufcstats_id=opponent_id,
            source_opponent_ufcstats_id=fighter_id,
            source_is_reversed=True,
        )
        self.assertTrue(link.source_is_reversed)
        self.assertTrue(link.paper_only)
        self.assertFalse(link.execution_enabled)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = EarlyMarketLinkStore(root / "links.csv", root / "links.jsonl")
            store.append((link,))
            self.assertEqual(store.read(), (link,))


if __name__ == "__main__":
    unittest.main()
