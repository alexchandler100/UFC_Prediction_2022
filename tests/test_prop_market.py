from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_tracker import (  # noqa: E402
    TotalRoundsForecastCapture,
    TotalRoundsForecastStore,
    MarketDataError,
    StoreIntegrityError,
    TotalRoundsQuoteSnapshot,
    TotalRoundsQuoteStore,
    build_prop_market_view,
)


def _quote(
    *,
    capture_id="capture-one",
    first_seen=None,
    book="DraftKings",
    book_key="draftkings",
    over=-110,
    under=-105,
):
    return TotalRoundsQuoteSnapshot.create(
        capture_id=capture_id,
        event_id="event-one",
        fighter_id="fighter-z",
        opponent_id="fighter-a",
        fighter_name="Zed Fighter",
        opponent_name="Alpha Fighter",
        event_date="2026-08-23",
        timing_precision="timestamp",
        event_start_utc="2026-08-23T23:00:00Z",
        observed_at_utc="2026-08-22T12:00:00Z",
        quote_first_seen_at_utc=first_seen,
        source="the-odds-api.com",
        source_event_id="source-event",
        source_book_key=book_key,
        source_quote_updated_at_utc="2026-08-22T11:59:30Z",
        source_commence_time_utc="2026-08-23T23:30:00Z",
        book=book,
        line=2.5,
        over_moneyline=over,
        under_moneyline=under,
        source_payload={"fixture": 1},
    )


class TotalRoundsMarketTests(unittest.TestCase):
    def test_quote_is_canonical_and_round_trips_through_both_mirrors(self):
        quote = _quote(first_seen="2026-08-22T10:00:00Z")
        self.assertEqual(quote.fighter_id, "fighter-a")
        self.assertEqual(quote.fighter_name, "Alpha Fighter")
        self.assertEqual(quote.market, "total_rounds")
        self.assertEqual(quote.period, "full_fight")
        self.assertAlmostEqual(
            quote.no_vig_over_probability,
            quote.over_implied_probability / quote.overround,
        )
        self.assertEqual(quote.source_quote_age_seconds, 30.0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = TotalRoundsQuoteStore(root / "quotes.csv", root / "quotes.jsonl")
            result = store.append([quote])
            self.assertEqual(result.total_records, 1)
            self.assertEqual(store.read(), (quote,))
            duplicate = store.append([quote])
            self.assertEqual(duplicate.duplicate_ids, (quote.quote_id,))
            with (root / "quotes.csv").open(encoding="utf-8", newline="") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 1)
            self.assertEqual(
                len((root / "quotes.jsonl").read_text(encoding="utf-8").splitlines()),
                1,
            )

    def test_invalid_line_timing_and_overround_fail_closed(self):
        base = dict(
            capture_id="capture-one",
            event_id="event-one",
            fighter_id="fighter-a",
            opponent_id="fighter-b",
            event_date="2026-08-23",
            timing_precision="timestamp",
            event_start_utc="2026-08-23T23:00:00Z",
            observed_at_utc="2026-08-22T12:00:00Z",
            source="source",
            source_event_id="source-event",
            source_book_key="book",
            source_quote_updated_at_utc="2026-08-22T11:59:00Z",
            source_commence_time_utc="2026-08-23T23:00:00Z",
            book="Book",
            line=2.5,
            over_moneyline=-110,
            under_moneyline=-105,
            source_payload={"fixture": 1},
        )
        for field, value in (
            ("line", 0),
            ("observed_at_utc", "2026-08-24T00:00:00Z"),
            ("over_moneyline", -10000),
        ):
            with self.subTest(field=field), self.assertRaises(MarketDataError):
                TotalRoundsQuoteSnapshot.create(**{**base, field: value})

    def test_tampered_jsonl_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = TotalRoundsQuoteStore(root / "quotes.csv", root / "quotes.jsonl")
            quote = _quote()
            store.append([quote])
            payload = json.loads((root / "quotes.jsonl").read_text(encoding="utf-8"))
            payload["line"] = 1.5
            (root / "quotes.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaises(StoreIntegrityError):
                store.read()

    def test_frozen_total_forecast_builds_positive_ev_candidate(self):
        quotes = (
            _quote(book="DraftKings", book_key="draftkings", over=110, under=-130),
            _quote(book="FanDuel", book_key="fanduel", over=100, under=-120),
            _quote(book="BetMGM", book_key="betmgm", over=-105, under=-115),
        )
        forecast = TotalRoundsForecastCapture.create(
            capture_id="capture-one",
            event_id="event-one",
            fighter_id="fighter-z",
            opponent_id="fighter-a",
            fighter_name="Zed Fighter",
            opponent_name="Alpha Fighter",
            event_date="2026-08-23",
            timing_precision="timestamp",
            event_start_utc="2026-08-23T23:00:00Z",
            forecast_issued_at_utc="2026-08-21T12:00:00Z",
            scheduled_rounds=3,
            schedule_basis="test_schedule",
            line=2.5,
            over_probability=0.58,
            model_id="outcome-model-one",
            model_version="candidate-v1",
            model_trained_through="2026-08-15",
            source_commit_sha="a" * 40,
            source_publication_sha256="b" * 64,
        )
        view = build_prop_market_view(
            quotes, (forecast,), capture_id="capture-one"
        )
        candidates = view["total_rounds"]["positive_candidates"]
        self.assertGreater(len(candidates), 0)
        self.assertEqual(candidates[0]["selection"], "Over 2.5 rounds")
        self.assertEqual(candidates[0]["target_book"], "DraftKings")
        self.assertGreater(candidates[0]["estimated_expected_return"], 0.20)
        self.assertEqual(
            view["method_of_victory"]["expected_value_status"],
            "unavailable_without_book_price",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = TotalRoundsForecastStore(
                root / "forecasts.csv", root / "forecasts.jsonl"
            )
            self.assertEqual(store.append((forecast,)).total_records, 1)
            self.assertEqual(store.read(), (forecast,))


if __name__ == "__main__":
    unittest.main()
