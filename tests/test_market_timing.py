import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_tracker import (  # noqa: E402
    QuoteSnapshot,
    QuoteSourceMetadata,
    evaluate_timing_policies,
)


class MarketTimingPolicyTests(unittest.TestCase):
    EVENT_START = "2026-01-10T12:00:00Z"

    @staticmethod
    def _utc(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _capture(
        self,
        *,
        capture,
        observed,
        fighter,
        opponent,
        lines,
    ):
        quotes = []
        metadata = []
        for book, fighter_line, opponent_line in lines:
            quote = QuoteSnapshot.create(
                capture_id=capture,
                event_id="event-one",
                fighter_id=fighter,
                opponent_id=opponent,
                event_date="2026-01-10",
                timing_precision="timestamp",
                event_start_utc=self.EVENT_START,
                observed_at_utc=observed,
                source="fixture",
                book=book,
                fighter_moneyline=fighter_line,
                opponent_moneyline=opponent_line,
                source_payload={"capture": capture},
            )
            quotes.append(quote)
            metadata.append(
                QuoteSourceMetadata.create(
                    quote,
                    source_book_key=book.casefold(),
                    source_event_id="source-event-one",
                    source_quote_updated_at_utc=(
                        self._utc(observed) - timedelta(seconds=60)
                    ),
                    source_commence_time_utc=self.EVENT_START,
                )
            )
        return quotes, metadata

    def test_frozen_timing_policies_are_causal_and_reproducible(self):
        regular = [
            ("BookA", -150, 130),
            ("BookB", -145, 125),
            ("BookC", -155, 135),
            ("Target", -150, 130),
        ]
        quotes = []
        metadata = []

        # Matchup one has a valuable favorite at the first observation.  The
        # combined policy must lock it and cannot later replace it with a dog.
        stages_one = (
            (
                "m1-early",
                "2026-01-06T12:00:00Z",
                [*regular[:3], ("Target", 110, -130)],
            ),
            (
                "m1-t24",
                "2026-01-09T12:00:00Z",
                [*regular[:3], ("Target", -250, 200)],
            ),
            (
                "m1-late",
                "2026-01-10T09:00:00Z",
                [*regular[:3], ("Target", -300, 250)],
            ),
        )
        for capture, observed, lines in stages_one:
            new_quotes, new_metadata = self._capture(
                capture=capture,
                observed=observed,
                fighter="fighter-a",
                opponent="opponent-b",
                lines=lines,
            )
            quotes.extend(new_quotes)
            metadata.extend(new_metadata)

        # Matchup two has no early edge, so the combined policy waits for the
        # frozen early underdog and takes it only in the late window.
        for capture, observed, lines in (
            ("m2-early", "2026-01-06T12:00:00Z", regular),
            ("m2-t24", "2026-01-09T12:00:00Z", regular),
            (
                "m2-late",
                "2026-01-10T09:00:00Z",
                [*regular[:3], ("Target", -350, 275)],
            ),
        ):
            new_quotes, new_metadata = self._capture(
                capture=capture,
                observed=observed,
                fighter="fighter-c",
                opponent="opponent-d",
                lines=lines,
            )
            quotes.extend(new_quotes)
            metadata.extend(new_metadata)

        outcomes = {
            ("event-one", "fighter-a", "opponent-b"): (1, "fight-one"),
            ("event-one", "fighter-c", "opponent-d"): (0, "fight-two"),
        }
        first = evaluate_timing_policies(quotes, metadata, outcomes)
        second = evaluate_timing_policies(reversed(quotes), reversed(metadata), outcomes)
        self.assertEqual(first, second)
        self.assertEqual(first["ledger_counts"]["frozen_classifications"], 2)
        self.assertEqual(first["coverage"]["matchups_with_early_capture"], 2)
        self.assertEqual(first["coverage"]["matchups_with_t24_capture"], 2)
        self.assertEqual(first["coverage"]["matchups_with_late_capture"], 2)
        combined = first["shadow_policies"]["favorite_early_underdog_late"]
        self.assertEqual(combined["selection_count"], 2)
        self.assertEqual(combined["scored_selection_count"], 2)
        self.assertEqual(combined["wins"], 2)
        self.assertEqual(combined["entry_window_counts"]["early"], 1)
        self.assertEqual(combined["entry_window_counts"]["late"], 1)
        self.assertGreater(combined["hypothetical_roi"], 0.0)
        prices = first["price_timing_test"]["best_available_price"]
        self.assertGreater(
            prices["favorite_early_minus_late"]["mean_probability_price_advantage"],
            0.0,
        )
        self.assertGreater(
            prices["underdog_late_minus_early"]["mean_probability_price_advantage"],
            0.0,
        )

    def test_quotes_without_source_timing_are_not_eligible(self):
        quotes, _ = self._capture(
            capture="missing-metadata",
            observed="2026-01-06T12:00:00Z",
            fighter="fighter-a",
            opponent="opponent-b",
            lines=[
                ("BookA", -150, 130),
                ("BookB", -145, 125),
                ("BookC", -155, 135),
                ("Target", 110, -130),
            ],
        )
        result = evaluate_timing_policies(quotes, (), {})
        self.assertEqual(result["ledger_counts"]["eligible_captures"], 0)
        self.assertEqual(
            result["shadow_policies"]["favorite_early_underdog_late"]["selection_count"],
            0,
        )

    def test_late_only_capture_is_reported_without_an_early_price_pair(self):
        quotes, metadata = self._capture(
            capture="late-only",
            observed="2026-01-10T09:00:00Z",
            fighter="fighter-a",
            opponent="opponent-b",
            lines=[
                ("BookA", -150, 130),
                ("BookB", -145, 125),
                ("BookC", -155, 135),
                ("Target", -350, 275),
            ],
        )
        result = evaluate_timing_policies(quotes, metadata, {})
        self.assertEqual(result["coverage"]["matchups_with_early_capture"], 0)
        self.assertEqual(result["coverage"]["matchups_with_late_capture"], 1)
        self.assertEqual(result["price_timing_test"]["paired_matchups"], 0)
        self.assertEqual(
            result["shadow_policies"]["favorite_early_underdog_late"]["selection_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
