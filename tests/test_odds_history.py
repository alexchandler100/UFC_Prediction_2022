import unittest

from src.market_tracker._common import MarketDataError
from src.market_tracker.odds_history import (
    ODDS_HISTORY_CONTRACT,
    build_odds_history,
    validate_odds_history,
)
from src.market_tracker.quotes import QuoteSnapshot


class OddsHistoryPublicationTests(unittest.TestCase):
    @staticmethod
    def quote(capture, observed, book, fighter_line, opponent_line):
        return QuoteSnapshot.create(
            capture_id=capture,
            event_id="event-1",
            fighter_id="fighter-a",
            opponent_id="fighter-b",
            fighter_name="Alpha",
            opponent_name="Bravo",
            event_date="2026-09-05",
            timing_precision="date",
            event_start_utc=None,
            observed_at_utc=observed,
            source="test-api",
            book=book,
            fighter_moneyline=fighter_line,
            opponent_moneyline=opponent_line,
            source_payload=f"payload-{capture}",
        )

    def test_publishes_each_book_and_a_consensus_at_every_capture(self):
        quotes = (
            self.quote("capture-1", "2026-09-01T12:00:00Z", "Book A", 120, -140),
            self.quote("capture-1", "2026-09-01T12:00:00Z", "Book B", 110, -130),
            self.quote("capture-2", "2026-09-02T12:00:00Z", "Book A", 100, -120),
            self.quote("capture-2", "2026-09-02T12:00:00Z", "Book B", -105, -115),
        )
        card = {"event_id": "event-1", "date": "September 05, 2026", "title": "Test card"}

        publication = build_odds_history(quotes, card)

        self.assertEqual(publication["contract"], ODDS_HISTORY_CONTRACT)
        self.assertEqual(publication["capture_count"], 2)
        self.assertEqual(publication["quote_count"], 4)
        self.assertEqual(publication["matchup_count"], 1)
        series = publication["matchups"][0]["series"]
        self.assertEqual([item["label"] for item in series], ["Consensus", "Book A", "Book B"])
        self.assertEqual([len(item["points"]) for item in series], [2, 2, 2])
        expected = sum(quote.no_vig_fighter_probability for quote in quotes[:2]) / 2
        self.assertAlmostEqual(series[0]["points"][0]["fighter_probability"], expected)
        self.assertAlmostEqual(
            series[0]["points"][0]["fighter_probability"]
            + series[0]["points"][0]["opponent_probability"],
            1.0,
        )
        validate_odds_history(publication, quotes, card)

    def test_excludes_other_events_and_detects_publication_changes(self):
        current = self.quote("capture-1", "2026-09-01T12:00:00Z", "Book A", 120, -140)
        other = QuoteSnapshot.create(
            capture_id="capture-old",
            event_id="event-old",
            fighter_id="fighter-a",
            opponent_id="fighter-b",
            event_date="2026-08-29",
            timing_precision="date",
            event_start_utc=None,
            observed_at_utc="2026-08-25T12:00:00Z",
            source="test-api",
            book="Book A",
            fighter_moneyline=120,
            opponent_moneyline=-140,
            source_payload="old-payload",
        )
        card = {"event_id": "event-1", "date": "2026-09-05", "title": "Test card"}
        publication = build_odds_history((current, other), card)
        self.assertEqual(publication["quote_count"], 1)
        self.assertEqual([item["label"] for item in publication["matchups"][0]["series"]], ["Book A"])
        publication["quote_count"] = 2
        with self.assertRaisesRegex(MarketDataError, "does not reproduce"):
            validate_odds_history(publication, (current, other), card)


if __name__ == "__main__":
    unittest.main()
