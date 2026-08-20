import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_tracker import (  # noqa: E402
    BlendObservation,
    ForecastCapture,
    ForecastCaptureStore,
    MarketDataError,
    PaperDecision,
    PaperDecisionStore,
    PaperSettlementStore,
    PriorCardBlendEvaluator,
    QuoteSnapshot,
    QuoteSnapshotStore,
    StoreIntegrityError,
    consensus_as_of,
    settle_paper_decision,
    select_latest_observations_by_horizon,
    summarize_paper_settlements,
    symmetric_logit_blend,
)
from market_tracker._common import canonical_hash  # noqa: E402
from validate_data import validate_market_data  # noqa: E402


EVENT_DATE = "2026-01-10"
OBSERVED = "2026-01-09T12:00:00Z"
ISSUED = "2026-01-09T10:00:00Z"
SOURCE_SHA = "a" * 40


def make_quote(
    book,
    fighter_line,
    opponent_line,
    *,
    capture_id="capture-one",
    observed_at=OBSERVED,
    first_seen=None,
):
    return QuoteSnapshot.create(
        capture_id=capture_id,
        event_id="event-one",
        fighter_id="fighter-a",
        opponent_id="fighter-b",
        fighter_name="Fighter A",
        opponent_name="Fighter B",
        event_date=EVENT_DATE,
        timing_precision="date",
        event_start_utc=None,
        observed_at_utc=observed_at,
        quote_first_seen_at_utc=first_seen,
        source="fixture",
        book=book,
        fighter_moneyline=fighter_line,
        opponent_moneyline=opponent_line,
        source_payload={"capture": capture_id},
    )


def make_forecast(*, capture_id="capture-one", probability=0.60):
    return ForecastCapture.create(
        capture_id=capture_id,
        event_id="event-one",
        fighter_id="fighter-a",
        opponent_id="fighter-b",
        fighter_name="Fighter A",
        opponent_name="Fighter B",
        event_date=EVENT_DATE,
        timing_precision="date",
        event_start_utc=None,
        forecast_issued_at_utc=ISSUED,
        model_probability=probability,
        model_id="model-one",
        model_version="fixture-v1",
        model_trained_through="2026-01-03",
        model_training_cutoff_precision="date",
        source_commit_sha=SOURCE_SHA,
    )


class MarketTrackerTests(unittest.TestCase):
    def test_date_only_contract_rejects_same_day_quote(self):
        with self.assertRaisesRegex(MarketDataError, "strictly before"):
            make_quote(
                "BookA",
                -110,
                -110,
                observed_at="2026-01-10T00:00:00Z",
            )

    def test_canonical_orientation_and_blend_are_symmetric(self):
        normal = make_quote("BookA", -150, +130)
        reversed_quote = QuoteSnapshot.create(
            capture_id="capture-one",
            event_id="event-one",
            fighter_id="fighter-b",
            opponent_id="fighter-a",
            fighter_name="Fighter B",
            opponent_name="Fighter A",
            event_date=EVENT_DATE,
            timing_precision="date",
            event_start_utc=None,
            observed_at_utc=OBSERVED,
            source="fixture",
            book="BookA",
            fighter_moneyline=+130,
            opponent_moneyline=-150,
            source_payload={"capture": "capture-one"},
        )
        self.assertEqual(normal, reversed_quote)
        probability = symmetric_logit_blend(0.42, 0.61, 0.25)
        self.assertAlmostEqual(
            probability,
            1.0 - symmetric_logit_blend(0.58, 0.39, 0.25),
            places=15,
        )

    def test_reversed_legacy_forecast_round_trips_exactly(self):
        forecast = ForecastCapture.from_legacy_american_odds(
            predicted_american_odds=+884,
            capture_id="capture-one",
            event_id="event-one",
            fighter_id="fighter-b",
            opponent_id="fighter-a",
            fighter_name="Fighter B",
            opponent_name="Fighter A",
            event_date=EVENT_DATE,
            timing_precision="date",
            event_start_utc=None,
            forecast_issued_at_utc=ISSUED,
            model_id="legacy-model",
            model_version="legacy-v1",
            model_trained_through="2026-01-03",
            model_training_cutoff_precision="date",
            source_commit_sha=SOURCE_SHA,
        )
        rebuilt = ForecastCapture.from_mapping(forecast.to_mapping())
        self.assertEqual(rebuilt, forecast)
        self.assertEqual(rebuilt.forecast_capture_id, forecast.forecast_capture_id)

    def test_market_validator_rejects_quote_forecast_timing_mismatch(self):
        quote = make_quote("BookA", -110, -110)
        mismatched = ForecastCapture.create(
            capture_id="capture-one",
            event_id="event-one",
            fighter_id="fighter-a",
            opponent_id="fighter-b",
            fighter_name="Fighter A",
            opponent_name="Fighter B",
            event_date="2026-01-11",
            timing_precision="date",
            event_start_utc=None,
            forecast_issued_at_utc=ISSUED,
            model_probability=0.60,
            model_id="model-one",
            model_version="fixture-v1",
            model_trained_through="2026-01-03",
            model_training_cutoff_precision="date",
            source_commit_sha=SOURCE_SHA,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            QuoteSnapshotStore(
                root / "quote_snapshots.csv", root / "quote_snapshots.jsonl"
            ).append([quote])
            ForecastCaptureStore(
                root / "forecast_captures.csv",
                root / "forecast_captures.jsonl",
            ).append([mismatched])
            report = validate_market_data(root, required=True)
        self.assertTrue(
            any("timing/identity" in error for error in report.errors),
            report.errors,
        )

    def test_prior_card_blend_uses_strictly_earlier_dates(self):
        def observation(event_id, event_date, observed_at, model_probability, target):
            capture_id = f"capture-{event_id}"
            fighter_id = f"fighter-{event_id}-a"
            opponent_id = f"fighter-{event_id}-b"
            quotes = [
                QuoteSnapshot.create(
                    capture_id=capture_id,
                    event_id=event_id,
                    fighter_id=fighter_id,
                    opponent_id=opponent_id,
                    fighter_name="Fighter A",
                    opponent_name="Fighter B",
                    event_date=event_date,
                    timing_precision="date",
                    event_start_utc=None,
                    observed_at_utc=observed_at,
                    source="fixture",
                    book=book,
                    fighter_moneyline=-110,
                    opponent_moneyline=-110,
                    source_payload={"capture": capture_id},
                )
                for book in ("BookA", "BookB", "BookC")
            ]
            market = consensus_as_of(
                quotes,
                capture_id=capture_id,
                matchup_id=quotes[0].matchup_id,
                as_of_utc=observed_at,
                min_books=3,
            )
            forecast = ForecastCapture.create(
                capture_id=capture_id,
                event_id=event_id,
                fighter_id=fighter_id,
                opponent_id=opponent_id,
                fighter_name="Fighter A",
                opponent_name="Fighter B",
                event_date=event_date,
                timing_precision="date",
                event_start_utc=None,
                forecast_issued_at_utc=observed_at,
                model_probability=model_probability,
                model_id="model-one",
                model_version="fixture-v1",
                model_trained_through="2025-12-31",
                model_training_cutoff_precision="date",
                source_commit_sha=SOURCE_SHA,
            )
            return BlendObservation.from_captures(
                market, forecast, target=target, fight_id=f"fight-{event_id}"
            )

        records = [
            observation("event-1", "2026-01-10", "2026-01-09T12:00:00Z", 0.5, 1),
            observation("event-2", "2026-01-17", "2026-01-16T12:00:00Z", 0.5, 0),
            # If the first same-day event leaked into the second, it would favor
            # gamma=1. Both must retain the gamma=0 tie-break learned only from
            # the two strictly earlier cards.
            observation("event-3a", "2026-01-24", "2026-01-23T12:00:00Z", 0.9, 1),
            observation("event-3b", "2026-01-24", "2026-01-23T12:00:00Z", 0.1, 0),
        ]
        evaluator = PriorCardBlendEvaluator(
            gamma_grid=(0.0, 1.0),
            min_prior_cards=2,
            min_prior_fights=2,
            lookback_cards=None,
        )
        result = evaluator.evaluate(reversed(records))
        self.assertEqual(result.evaluation_id, evaluator.evaluate(records).evaluation_id)
        by_event = {prediction.event_id: prediction for prediction in result.predictions}
        self.assertEqual(by_event["event-1"].status, "insufficient_prior_history")
        self.assertEqual(by_event["event-2"].status, "insufficient_prior_history")
        for event_id in ("event-3a", "event-3b"):
            self.assertEqual(by_event[event_id].status, "evaluated")
            self.assertEqual(by_event[event_id].selected_gamma, 0.0)
            self.assertEqual(by_event[event_id].prior_card_count, 2)
            self.assertEqual(
                by_event[event_id].selection_training_through_event_date,
                "2026-01-17",
            )

    def test_capture_selection_uses_predeclared_horizon_not_outcome(self):
        def captured(
            capture_id,
            observed_at,
            model_probability,
            forecast_issued_at=None,
        ):
            quotes = [
                QuoteSnapshot.create(
                    capture_id=capture_id,
                    event_id="event-select",
                    fighter_id="fighter-select-a",
                    opponent_id="fighter-select-b",
                    fighter_name="Fighter A",
                    opponent_name="Fighter B",
                    event_date="2026-01-24",
                    timing_precision="date",
                    event_start_utc=None,
                    observed_at_utc=observed_at,
                    source="fixture",
                    book=book,
                    fighter_moneyline=-110,
                    opponent_moneyline=-110,
                    source_payload={"capture": capture_id},
                )
                for book in ("BookA", "BookB", "BookC")
            ]
            market = consensus_as_of(
                quotes,
                capture_id=capture_id,
                matchup_id=quotes[0].matchup_id,
                as_of_utc=observed_at,
                min_books=3,
            )
            forecast = ForecastCapture.create(
                capture_id=capture_id,
                event_id="event-select",
                fighter_id="fighter-select-a",
                opponent_id="fighter-select-b",
                fighter_name="Fighter A",
                opponent_name="Fighter B",
                event_date="2026-01-24",
                timing_precision="date",
                event_start_utc=None,
                forecast_issued_at_utc=forecast_issued_at or observed_at,
                model_probability=model_probability,
                model_id="model-one",
                model_version="fixture-v1",
                model_trained_through="2025-12-31",
                model_training_cutoff_precision="date",
                source_commit_sha=SOURCE_SHA,
            )
            return BlendObservation.from_captures(
                market, forecast, target=1, fight_id="fight-select"
            )

        early = captured("capture-early", "2026-01-23T12:00:00Z", 0.55)
        late = captured("capture-late", "2026-01-23T14:00:00Z", 0.95)
        cutoffs = {"event-select": "2026-01-23T13:00:00Z"}
        chosen = select_latest_observations_by_horizon([late, early], cutoffs)
        self.assertEqual([item.capture_id for item in chosen], ["capture-early"])
        late_forecast = captured(
            "capture-quote-before-model-after",
            "2026-01-23T12:30:00Z",
            0.90,
            forecast_issued_at="2026-01-23T13:30:00Z",
        )
        chosen = select_latest_observations_by_horizon(
            [early, late_forecast], cutoffs
        )
        self.assertEqual([item.capture_id for item in chosen], ["capture-early"])
        with self.assertRaisesRegex(MarketDataError, "available by its cutoff"):
            select_latest_observations_by_horizon(
                [early, late],
                {"event-select": "2026-01-23T11:00:00Z"},
            )

    def test_consensus_cannot_mix_capture_runs(self):
        old_quotes = [
            make_quote("BookA", -110, -110, capture_id="old-capture"),
            make_quote("BookB", -105, -115, capture_id="old-capture"),
            make_quote("BookC", -115, -105, capture_id="old-capture"),
        ]
        new_quote = make_quote(
            "BookA",
            -120,
            +100,
            capture_id="new-capture",
            observed_at="2026-01-09T18:00:00Z",
        )
        with self.assertRaisesRegex(MarketDataError, "only 1 distinct books"):
            consensus_as_of(
                [*old_quotes, new_quote],
                capture_id="new-capture",
                matchup_id=new_quote.matchup_id,
                as_of_utc="2026-01-09T18:00:00Z",
                min_books=3,
            )

    def test_consensus_cannot_mix_retrieval_times_within_capture(self):
        early = [
            make_quote("Target", +200, -250),
            make_quote("BookA", -110, -110),
        ]
        later = [
            make_quote(
                "BookB", -105, -115,
                observed_at="2026-01-09T12:10:00Z",
            ),
            make_quote(
                "BookC", -115, -105,
                observed_at="2026-01-09T12:10:00Z",
            ),
        ]
        with self.assertRaisesRegex(StoreIntegrityError, "retrieval timestamps"):
            consensus_as_of(
                [*early, *later],
                capture_id="capture-one",
                matchup_id=early[0].matchup_id,
                as_of_utc="2026-01-09T12:10:00Z",
                min_books=3,
                exclude_books=["Target"],
            )

    def test_consensus_cannot_backdate_or_carry_forward_a_capture(self):
        quotes = [
            make_quote("BookA", -110, -110),
            make_quote("BookB", -105, -115),
            make_quote("BookC", -115, -105),
        ]
        with self.assertRaisesRegex(StoreIntegrityError, "must equal"):
            consensus_as_of(
                quotes,
                capture_id="capture-one",
                matchup_id=quotes[0].matchup_id,
                as_of_utc="2026-01-09T18:00:00Z",
                min_books=3,
            )

    def test_quote_store_is_idempotent_and_rejects_rewrites(self):
        quote = make_quote("BookA", -110, -110)
        changed_same_key = make_quote("BookA", -120, +100)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = QuoteSnapshotStore(root / "quotes.csv", root / "quotes.jsonl")
            first = store.append([quote])
            second = store.append([quote])
            self.assertEqual(len(first.added_ids), 1)
            self.assertEqual(second.duplicate_ids, (quote.quote_id,))
            self.assertEqual(store.read(), (quote,))
            with self.assertRaisesRegex(StoreIntegrityError, "conflicting quote"):
                store.append([changed_same_key])

    def test_first_seen_and_current_observation_are_separate(self):
        quote = make_quote(
            "BookA",
            -110,
            -110,
            capture_id="later-capture",
            observed_at="2026-01-09T18:00:00Z",
            first_seen="2026-01-08T18:00:00Z",
        )
        self.assertEqual(quote.observed_at_utc, "2026-01-09T18:00:00.000000Z")
        self.assertEqual(
            quote.quote_first_seen_at_utc,
            "2026-01-08T18:00:00.000000Z",
        )

    def test_paper_decision_uses_leave_one_book_out_consensus_and_ev(self):
        quotes = [
            make_quote("BookA", -110, -110),
            make_quote("BookB", -105, -115),
            make_quote("BookC", -115, -105),
            make_quote("Target", +200, -250),
        ]
        target = quotes[-1]
        market = consensus_as_of(
            quotes,
            capture_id="capture-one",
            matchup_id=target.matchup_id,
            as_of_utc=OBSERVED,
            min_books=3,
            exclude_books=["Target"],
        )
        forecast = make_forecast()
        decision = PaperDecision.create(
            market,
            target,
            forecast,
            selected_gamma=0.0,
            decision_issued_at_utc=OBSERVED,
        )
        self.assertEqual(decision.paper_action, "fighter")
        self.assertGreaterEqual(decision.fighter_expected_return, 0.05)
        self.assertTrue(decision.paper_only)

        delayed = PaperDecision.create(
            market,
            target,
            forecast,
            selected_gamma=0.0,
            decision_issued_at_utc="2026-01-09T12:04:00Z",
        )
        self.assertEqual(delayed.quote_age_seconds_at_decision, 240.0)
        with self.assertRaisesRegex(MarketDataError, "maximum allowed quote age"):
            PaperDecision.create(
                market,
                target,
                forecast,
                selected_gamma=0.0,
                decision_issued_at_utc="2026-01-09T12:06:00Z",
            )

        self_including_market = consensus_as_of(
            quotes,
            capture_id="capture-one",
            matchup_id=target.matchup_id,
            as_of_utc=OBSERVED,
            min_books=4,
        )
        with self.assertRaisesRegex(StoreIntegrityError, "excluded"):
            PaperDecision.create(
                self_including_market,
                target,
                forecast,
                selected_gamma=0.0,
                decision_issued_at_utc=OBSERVED,
            )

        too_thin_market = consensus_as_of(
            [quotes[0], quotes[1], target],
            capture_id="capture-one",
            matchup_id=target.matchup_id,
            as_of_utc=OBSERVED,
            min_books=2,
            exclude_books=["Target"],
        )
        with self.assertRaisesRegex(StoreIntegrityError, "at least three"):
            PaperDecision.create(
                too_thin_market,
                target,
                forecast,
                selected_gamma=0.0,
                decision_issued_at_utc=OBSERVED,
            )

    def test_paper_decision_and_settlement_round_trip(self):
        quotes = [
            make_quote("BookA", -110, -110),
            make_quote("BookB", -105, -115),
            make_quote("BookC", -115, -105),
            make_quote("Target", +200, -250),
        ]
        market = consensus_as_of(
            quotes,
            capture_id="capture-one",
            matchup_id=quotes[0].matchup_id,
            as_of_utc=OBSERVED,
            min_books=3,
            exclude_books=["Target"],
        )
        decision = PaperDecision.create(
            market,
            quotes[-1],
            make_forecast(),
            selected_gamma=0.0,
            decision_issued_at_utc=OBSERVED,
        )
        settlement = settle_paper_decision(
            decision,
            target=1,
            fight_id="fight-one",
            settled_at_utc="2026-01-11T00:00:00Z",
            result_source_sha256="b" * 64,
        )
        self.assertEqual(settlement.settlement_status, "paper_win")
        self.assertEqual(settlement.hypothetical_profit_units, 2.0)
        metrics = summarize_paper_settlements([decision], [settlement])
        self.assertEqual(metrics.wins, 1)

        tampered_body = settlement.to_mapping()
        tampered_body.pop("settlement_id")
        tampered_body["capture_id"] = "another-capture"
        tampered = type(settlement).from_mapping(
            {"settlement_id": canonical_hash(tampered_body), **tampered_body}
        )
        with self.assertRaisesRegex(StoreIntegrityError, "matching decision"):
            summarize_paper_settlements([decision], [tampered])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decisions = PaperDecisionStore(
                root / "decisions.csv", root / "decisions.jsonl"
            )
            settlements = PaperSettlementStore(
                root / "settlements.csv", root / "settlements.jsonl"
            )
            decisions.append([decision])
            settlements.append([settlement])
            self.assertEqual(decisions.read(), (decision,))
            self.assertEqual(settlements.read(), (settlement,))


if __name__ == "__main__":
    unittest.main()
