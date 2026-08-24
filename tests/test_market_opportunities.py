import math
import sys
import unittest
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_tracker import (  # noqa: E402
    ForecastCapture,
    QuoteSnapshot,
    QuoteSourceMetadata,
    StoreIntegrityError,
    build_current_opportunities,
    validate_current_opportunities,
)
from market_tracker._common import canonical_hash  # noqa: E402


SOURCE_SHA = "a" * 64
COMMIT_SHA = "b" * 40


def _fixture(lines):
    quotes = []
    metadata = []
    for book, fighter_line, opponent_line in lines:
        quote = QuoteSnapshot.create(
            capture_id="capture-opportunity",
            event_id="event-opportunity",
            fighter_id="fighter-alpha",
            opponent_id="fighter-beta",
            fighter_name="Alpha",
            opponent_name="Beta",
            event_date="2026-01-11",
            timing_precision="timestamp",
            event_start_utc="2026-01-11T20:00:00Z",
            observed_at_utc="2026-01-10T12:00:00Z",
            source="fixture-source",
            book=book,
            fighter_moneyline=fighter_line,
            opponent_moneyline=opponent_line,
            source_payload_sha256=SOURCE_SHA,
        )
        quotes.append(quote)
        metadata.append(
            QuoteSourceMetadata.create(
                quote,
                source_book_key=book.lower(),
                source_event_id="source-event",
                source_quote_updated_at_utc="2026-01-10T11:59:30Z",
                source_commence_time_utc="2026-01-11T20:00:00Z",
            )
        )
    forecast = ForecastCapture.create(
        capture_id="capture-opportunity",
        event_id="event-opportunity",
        fighter_id="fighter-alpha",
        opponent_id="fighter-beta",
        fighter_name="Alpha",
        opponent_name="Beta",
        event_date="2026-01-11",
        timing_precision="timestamp",
        event_start_utc="2026-01-11T20:00:00Z",
        forecast_issued_at_utc="2026-01-09T18:00:00Z",
        model_probability=0.90,
        model_id="model-fixture",
        model_version="fixture-v1",
        model_trained_through="2026-01-03",
        model_training_cutoff_precision="date",
        source_commit_sha=COMMIT_SHA,
    )
    return tuple(quotes), (forecast,), tuple(metadata)


class CurrentOpportunityPublicationTests(unittest.TestCase):
    def test_selects_executable_outlier_and_excludes_target_book(self):
        quotes, forecasts, metadata = _fixture(
            [
                ("TargetBook", 100, -120),
                ("BookB", -150, 130),
                ("BookC", -150, 130),
                ("BookD", -150, 130),
            ]
        )
        publication = build_current_opportunities(
            quotes,
            forecasts,
            metadata,
            capture_id="capture-opportunity",
        )
        matchup = publication["matchups"][0]
        signal = matchup["current_signal"]
        self.assertEqual(signal["paper_action"], "fighter")
        self.assertEqual(signal["action_name"], "Alpha")
        self.assertEqual(signal["target_book"], "TargetBook")
        self.assertEqual(signal["offered_moneyline"], 100)
        self.assertGreater(signal["estimated_expected_return"], 0.05)
        self.assertEqual(signal["consensus_book_count"], 3)
        self.assertNotIn("TargetBook", signal["consensus_books"])
        self.assertEqual(signal["model_weight"], 0.0)
        self.assertAlmostEqual(signal["market_probability"], 0.5798319328)
        self.assertNotAlmostEqual(
            signal["market_probability"], signal["model_probability_for_fighter"]
        )
        self.assertFalse(publication["execution_enabled"])
        self.assertEqual(publication["timing_status"], "before_t24_decision_window")

    def test_pass_still_exposes_best_candidate_and_reason(self):
        quotes, forecasts, metadata = _fixture(
            [
                ("BookA", -110, -110),
                ("BookB", -110, -110),
                ("BookC", -110, -110),
                ("BookD", -110, -110),
            ]
        )
        publication = build_current_opportunities(
            quotes,
            forecasts,
            metadata,
            capture_id="capture-opportunity",
        )
        signal = publication["matchups"][0]["current_signal"]
        self.assertEqual(signal["paper_action"], "pass")
        self.assertEqual(signal["target_book"], "BookA")
        self.assertEqual(signal["offered_moneyline"], -110)
        self.assertLess(signal["estimated_expected_return"], 0.05)
        self.assertIn("below", signal["reason"])

    def test_publication_validation_detects_display_tampering(self):
        quotes, forecasts, metadata = _fixture(
            [
                ("BookA", -110, -110),
                ("BookB", -110, -110),
                ("BookC", -110, -110),
                ("BookD", -110, -110),
            ]
        )
        publication = build_current_opportunities(
            quotes,
            forecasts,
            metadata,
            capture_id="capture-opportunity",
        )
        tampered = deepcopy(publication)
        tampered["matchups"][0]["current_signal"]["offered_moneyline"] = 999
        tampered_body = dict(tampered)
        tampered_body.pop("publication_sha256")
        tampered["publication_sha256"] = canonical_hash(tampered_body)
        with self.assertRaisesRegex(
            StoreIntegrityError,
            r"cannot be reproduced.*\.offered_moneyline",
        ):
            validate_current_opportunities(
                tampered,
                quotes,
                forecasts,
                metadata,
                (),
                capture_id="capture-opportunity",
            )

    def test_validation_accepts_cross_platform_float_roundoff(self):
        quotes, forecasts, metadata = _fixture(
            [
                ("BookA", -110, -110),
                ("BookB", -110, -110),
                ("BookC", -110, -110),
                ("BookD", -110, -110),
            ]
        )
        publication = build_current_opportunities(
            quotes,
            forecasts,
            metadata,
            capture_id="capture-opportunity",
        )
        probability = publication["matchups"][0]["full_market_consensus"][
            "fighter_probability"
        ]
        publication["matchups"][0]["full_market_consensus"][
            "fighter_probability"
        ] = math.nextafter(probability, math.inf)
        publication_body = dict(publication)
        publication_body.pop("publication_sha256")
        publication["publication_sha256"] = canonical_hash(publication_body)

        validate_current_opportunities(
            publication,
            quotes,
            forecasts,
            metadata,
            (),
            capture_id="capture-opportunity",
        )

    def test_validation_rejects_a_tampered_publication_fingerprint(self):
        quotes, forecasts, metadata = _fixture(
            [
                ("BookA", -110, -110),
                ("BookB", -110, -110),
                ("BookC", -110, -110),
                ("BookD", -110, -110),
            ]
        )
        publication = build_current_opportunities(
            quotes,
            forecasts,
            metadata,
            capture_id="capture-opportunity",
        )
        publication["matchups"][0]["current_signal"]["offered_moneyline"] = 999
        with self.assertRaisesRegex(StoreIntegrityError, "fingerprint"):
            validate_current_opportunities(
                publication,
                quotes,
                forecasts,
                metadata,
                (),
                capture_id="capture-opportunity",
            )


if __name__ == "__main__":
    unittest.main()
