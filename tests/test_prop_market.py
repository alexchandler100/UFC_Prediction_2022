from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_tracker import (  # noqa: E402
    TotalRoundsForecastCapture,
    TotalRoundsForecastStore,
    MarketDataError,
    StoreIntegrityError,
    TotalRoundsQuoteSnapshot,
    TotalRoundsQuoteStore,
    TotalRoundsPaperDecisionStore,
    TotalRoundsPaperSettlementStore,
    build_locked_total_round_decisions,
    build_prop_market_view,
    settle_total_round_decision,
    summarize_total_round_performance,
)
import market_tracker.prop_paper as prop_paper  # noqa: E402


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


def _paper_quote(
    *,
    book,
    book_key,
    over,
    under,
    capture_id="paper-capture",
    event_id="event-one",
    event_date="2026-08-23",
    event_start="2026-08-23T23:00:00Z",
    observed="2026-08-22T23:00:00Z",
):
    return TotalRoundsQuoteSnapshot.create(
        capture_id=capture_id,
        event_id=event_id,
        fighter_id="fighter-z",
        opponent_id="fighter-a",
        fighter_name="Zed Fighter",
        opponent_name="Alpha Fighter",
        event_date=event_date,
        timing_precision="timestamp",
        event_start_utc=event_start,
        observed_at_utc=observed,
        source="the-odds-api.com",
        source_event_id="source-event",
        source_book_key=book_key,
        source_quote_updated_at_utc=observed,
        source_commence_time_utc=event_start,
        book=book,
        line=2.5,
        over_moneyline=over,
        under_moneyline=under,
        source_payload={"paper": 1},
    )


def _paper_forecast(
    *,
    capture_id="paper-capture",
    event_id="event-one",
    event_date="2026-08-23",
    event_start="2026-08-23T23:00:00Z",
    issued="2026-08-21T12:00:00Z",
):
    return TotalRoundsForecastCapture.create(
        capture_id=capture_id,
        event_id=event_id,
        fighter_id="fighter-z",
        opponent_id="fighter-a",
        fighter_name="Zed Fighter",
        opponent_name="Alpha Fighter",
        event_date=event_date,
        timing_precision="timestamp",
        event_start_utc=event_start,
        forecast_issued_at_utc=issued,
        scheduled_rounds=3,
        schedule_basis="test_schedule",
        line=2.5,
        over_probability=0.70,
        model_id="outcome-model-one",
        model_version="candidate-v1",
        model_trained_through="2026-08-15",
        source_commit_sha="a" * 40,
        source_publication_sha256="b" * 64,
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
        self.assertEqual(candidates[0]["model_trained_through"], "2026-08-15")
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

    def test_t24_total_decision_settles_and_reports_shadow_thresholds(self):
        quotes = (
            _paper_quote(
                book="Target", book_key="target", over=120, under=-140
            ),
            _paper_quote(
                book="Consensus A", book_key="consensus-a", over=-110, under=-110
            ),
            _paper_quote(
                book="Consensus B", book_key="consensus-b", over=-110, under=-110
            ),
        )
        built = build_locked_total_round_decisions(quotes, (_paper_forecast(),))
        self.assertTrue(built.eligible_horizon)
        self.assertEqual(len(built.decisions), 1)
        decision = built.decisions[0]
        self.assertEqual(decision.paper_action, "over")
        self.assertEqual(decision.target_book, "Target")
        self.assertEqual(decision.selected_residual_weight, 0.0)
        self.assertEqual(
            decision.residual_selection_status,
            "market_only_insufficient_history",
        )
        self.assertAlmostEqual(decision.market_over_probability, 0.5)
        self.assertAlmostEqual(decision.residual_over_probability, 0.5)
        view = build_prop_market_view(
            quotes,
            (_paper_forecast(),),
            capture_id="paper-capture",
            decisions=(decision,),
        )
        locked = view["total_rounds"]["markets"][0]["locked_t24_decision"]
        self.assertEqual(locked["paper_action"], "over")
        self.assertEqual(locked["target_book"], "Target")

        settlement = settle_total_round_decision(
            decision,
            total_fight_seconds=800.0,
            fight_id="fight-one",
            settled_at_utc="2026-08-24T01:00:00Z",
            result_source_sha256="c" * 64,
        )
        self.assertEqual(settlement.target, 1)
        self.assertEqual(settlement.settlement_status, "paper_win")
        self.assertAlmostEqual(settlement.hypothetical_profit_units, 1.2)
        performance = summarize_total_round_performance(
            (decision,), (settlement,)
        )
        self.assertEqual(performance["scored_forecasts"], 1)
        self.assertAlmostEqual(
            performance["official_strategy"]["hypothetical_roi"], 1.2
        )
        self.assertEqual(
            set(performance["shadow_threshold_strategies"]),
            {"independent_model", "market_residual"},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision_store = TotalRoundsPaperDecisionStore(
                root / "decisions.csv", root / "decisions.jsonl"
            )
            settlement_store = TotalRoundsPaperSettlementStore(
                root / "settlements.csv", root / "settlements.jsonl"
            )
            self.assertEqual(decision_store.append((decision,)).total_records, 1)
            self.assertEqual(
                settlement_store.append((settlement,)).total_records, 1
            )
            self.assertEqual(decision_store.read(), (decision,))
            self.assertEqual(settlement_store.read(), (settlement,))

    def test_exact_total_boundary_is_voided_fail_closed(self):
        quotes = (
            _paper_quote(
                book="Target", book_key="target", over=120, under=-140
            ),
            _paper_quote(
                book="Consensus A", book_key="consensus-a", over=-110, under=-110
            ),
            _paper_quote(
                book="Consensus B", book_key="consensus-b", over=-110, under=-110
            ),
        )
        decision = build_locked_total_round_decisions(
            quotes, (_paper_forecast(),)
        ).decisions[0]
        settlement = settle_total_round_decision(
            decision,
            total_fight_seconds=750.0,
            settled_at_utc="2026-08-24T01:00:00Z",
            result_source_sha256="d" * 64,
        )
        self.assertIsNone(settlement.target)
        self.assertEqual(settlement.settlement_status, "void")
        self.assertEqual(settlement.hypothetical_risk_units, 0.0)

    def test_residual_weight_uses_only_prior_settled_evidence(self):
        first_quotes = (
            _paper_quote(book="Target", book_key="target", over=120, under=-140),
            _paper_quote(book="Book A", book_key="book-a", over=-110, under=-110),
            _paper_quote(book="Book B", book_key="book-b", over=-110, under=-110),
        )
        first = build_locked_total_round_decisions(
            first_quotes, (_paper_forecast(),)
        ).decisions[0]
        first_settlement = settle_total_round_decision(
            first,
            total_fight_seconds=800.0,
            settled_at_utc="2026-08-24T01:00:00Z",
            result_source_sha256="e" * 64,
        )
        second_kwargs = {
            "capture_id": "paper-capture-two",
            "event_id": "event-two",
            "event_date": "2026-09-01",
            "event_start": "2026-09-01T23:00:00Z",
            "observed": "2026-08-31T23:00:00Z",
        }
        second_quotes = (
            _paper_quote(
                book="Target", book_key="target", over=120, under=-140,
                **second_kwargs,
            ),
            _paper_quote(
                book="Book A", book_key="book-a", over=-110, under=-110,
                **second_kwargs,
            ),
            _paper_quote(
                book="Book B", book_key="book-b", over=-110, under=-110,
                **second_kwargs,
            ),
        )
        second_forecast = _paper_forecast(
            capture_id="paper-capture-two",
            event_id="event-two",
            event_date="2026-09-01",
            event_start="2026-09-01T23:00:00Z",
            issued="2026-08-30T12:00:00Z",
        )
        second = build_locked_total_round_decisions(
            second_quotes,
            (second_forecast,),
            (first,),
            (first_settlement,),
        ).decisions[0]
        second_settlement = settle_total_round_decision(
            second,
            total_fight_seconds=800.0,
            settled_at_utc="2026-09-02T01:00:00Z",
            result_source_sha256="f" * 64,
        )
        third_kwargs = {
            "capture_id": "paper-capture-three",
            "event_id": "event-three",
            "event_date": "2026-09-08",
            "event_start": "2026-09-08T23:00:00Z",
            "observed": "2026-09-07T23:00:00Z",
        }
        third_quotes = (
            _paper_quote(
                book="Target", book_key="target", over=120, under=-140,
                **third_kwargs,
            ),
            _paper_quote(
                book="Book A", book_key="book-a", over=-110, under=-110,
                **third_kwargs,
            ),
            _paper_quote(
                book="Book B", book_key="book-b", over=-110, under=-110,
                **third_kwargs,
            ),
        )
        third_forecast = _paper_forecast(
            capture_id="paper-capture-three",
            event_id="event-three",
            event_date="2026-09-08",
            event_start="2026-09-08T23:00:00Z",
            issued="2026-09-06T12:00:00Z",
        )
        with (
            patch.object(prop_paper, "RESIDUAL_MIN_SCORED_LINES", 2),
            patch.object(prop_paper, "RESIDUAL_MIN_SETTLED_EVENTS", 2),
            patch.object(prop_paper, "RESIDUAL_BOOTSTRAP_SAMPLES", 100),
        ):
            third = build_locked_total_round_decisions(
                third_quotes,
                (third_forecast,),
                (first, second),
                (first_settlement, second_settlement),
            ).decisions[0]
        self.assertGreater(third.selected_residual_weight, 0.0)
        self.assertEqual(
            third.residual_selection_status, "residual_weight_promoted"
        )
        self.assertEqual(third.residual_training_scored_lines, 2)


if __name__ == "__main__":
    unittest.main()
