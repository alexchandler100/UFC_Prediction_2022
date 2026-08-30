import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_tracker import (  # noqa: E402
    ForecastCapture,
    QuoteSnapshot,
    QuoteSourceMetadata,
    StoreIntegrityError,
)
from market_tracker.market_first_paper import (  # noqa: E402
    FrozenMarketFirstPolicy,
    MarketFirstPaperDecision,
    MarketFirstPaperDecisionStore,
    PaperSettlementStore,
    build_market_first_decisions,
    settle_market_first_decision,
    summarize_market_first_paper,
)


POLICY_PATH = (
    REPO_ROOT
    / "src"
    / "content"
    / "data"
    / "model_research"
    / "market_first_t24_policy.json"
)
OBSERVED = "2026-09-04T12:00:00Z"
EVENT_START = "2026-09-05T12:00:00Z"


def quote(book, fighter_line, opponent_line, *, observed=OBSERVED):
    return QuoteSnapshot.create(
        capture_id="capture-future-t24",
        event_id="future-event",
        fighter_id="fighter-a",
        opponent_id="fighter-b",
        fighter_name="Fighter A",
        opponent_name="Fighter B",
        event_date="2026-09-05",
        timing_precision="timestamp",
        event_start_utc=EVENT_START,
        observed_at_utc=observed,
        source="fixture",
        book=book,
        fighter_moneyline=fighter_line,
        opponent_moneyline=opponent_line,
        source_payload={"capture": "capture-future-t24", "observed": observed},
    )


def forecast():
    return ForecastCapture.create(
        capture_id="capture-future-t24",
        event_id="future-event",
        fighter_id="fighter-a",
        opponent_id="fighter-b",
        fighter_name="Fighter A",
        opponent_name="Fighter B",
        event_date="2026-09-05",
        timing_precision="timestamp",
        event_start_utc=EVENT_START,
        forecast_issued_at_utc="2026-09-04T10:00:00Z",
        model_probability=0.60,
        model_id="model-one",
        model_version="fixture-v1",
        model_trained_through="2026-08-29",
        model_training_cutoff_precision="date",
        source_commit_sha="a" * 40,
    )


def metadata(quotes):
    return tuple(
        QuoteSourceMetadata.create(
            item,
            source_book_key=item.book.casefold(),
            source_event_id="source-future-event",
            source_quote_updated_at_utc="2026-09-04T11:55:00Z",
            source_commence_time_utc=EVENT_START,
        )
        for item in quotes
    )


class MarketFirstPaperTests(unittest.TestCase):
    def setUp(self):
        self.policy = FrozenMarketFirstPolicy.load(POLICY_PATH)
        self.quotes = (
            quote("BookA", -110, -110),
            quote("BookB", -105, -115),
            quote("BookC", -115, -105),
            quote("Target", +200, -250),
        )

    def test_builds_one_leave_one_out_future_decision(self):
        built = build_market_first_decisions(
            self.quotes,
            (forecast(),),
            metadata(self.quotes),
            policy=self.policy,
        )
        self.assertTrue(built.eligible_horizon)
        self.assertEqual(len(built.decisions), 1)
        decision = built.decisions[0]
        self.assertEqual(decision.reference_book, "Target")
        self.assertEqual(decision.other_book_count, 3)
        self.assertEqual(decision.paper_action, "fighter")
        self.assertGreater(decision.candidate_probability, decision.market_probability)
        self.assertEqual(decision.minimum_expected_return, 0.025)
        self.assertTrue(decision.paper_only)
        self.assertFalse(decision.execution_enabled)

        repeated = build_market_first_decisions(
            self.quotes,
            (forecast(),),
            metadata(self.quotes),
            policy=self.policy,
            existing_decisions=built.decisions,
        )
        self.assertEqual(repeated.decisions, ())
        self.assertEqual(repeated.matchups_already_frozen, 1)

    def test_rejects_predeployment_capture(self):
        old_observed = "2026-08-28T12:00:00Z"
        old_start = "2026-08-29T12:00:00Z"
        old_quotes = tuple(
            QuoteSnapshot.create(
                capture_id="old-capture",
                event_id="old-event",
                fighter_id="fighter-a",
                opponent_id="fighter-b",
                event_date="2026-08-29",
                timing_precision="timestamp",
                event_start_utc=old_start,
                observed_at_utc=old_observed,
                source="fixture",
                book=f"Book{index}",
                fighter_moneyline=-110,
                opponent_moneyline=-110,
                source_payload={"index": index},
            )
            for index in range(4)
        )
        built = build_market_first_decisions(
            old_quotes, (), (), policy=self.policy
        )
        self.assertFalse(built.eligible_horizon)
        self.assertEqual(built.decisions, ())

    def test_decision_settlement_and_report_round_trip(self):
        decision = build_market_first_decisions(
            self.quotes,
            (forecast(),),
            metadata(self.quotes),
            policy=self.policy,
        ).decisions[0]
        settlement = settle_market_first_decision(
            decision,
            target=1,
            fight_id="fight-one",
            settled_at_utc="2026-09-06T00:00:00Z",
            result_source_sha256="b" * 64,
        )
        self.assertEqual(settlement.settlement_status, "paper_win")
        self.assertEqual(settlement.hypothetical_profit_units, 2.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision_store = MarketFirstPaperDecisionStore(
                root / "decisions.csv", root / "decisions.jsonl"
            )
            settlement_store = PaperSettlementStore(
                root / "settlements.csv", root / "settlements.jsonl"
            )
            decision_store.append((decision,))
            settlement_store.append((settlement,))
            self.assertEqual(decision_store.read(), (decision,))
            self.assertEqual(settlement_store.read(), (settlement,))
        report = summarize_market_first_paper(
            (decision,), (settlement,), self.quotes, bootstrap_samples=10
        )
        self.assertEqual(report["results"]["recommended_bets"], 1)
        self.assertEqual(report["results"]["profit_units"], 2.0)
        self.assertEqual(report["results"]["roi"], 2.0)
        self.assertFalse(report["execution_enabled"])

    def test_tampering_with_candidate_probability_is_detected(self):
        decision = build_market_first_decisions(
            self.quotes,
            (forecast(),),
            metadata(self.quotes),
            policy=self.policy,
        ).decisions[0]
        changed = decision.to_mapping()
        changed["candidate_probability"] = 0.99
        with self.assertRaises(StoreIntegrityError):
            MarketFirstPaperDecision.from_mapping(changed)

    def test_workflows_update_and_publish_the_separate_ledger(self):
        collector = (
            REPO_ROOT / ".github" / "workflows" / "collect-market-snapshot.yml"
        ).read_text(encoding="utf-8")
        updater = (
            REPO_ROOT / ".github" / "workflows" / "update-data.yml"
        ).read_text(encoding="utf-8")
        for workflow in (collector, updater):
            self.assertIn("update_market_first_paper.py", workflow)
            self.assertIn("market_first_paper_decisions.jsonl", workflow)
            self.assertIn("market_first_paper_settlements.jsonl", workflow)
            self.assertIn("market_first_paper_report.json", workflow)


if __name__ == "__main__":
    unittest.main()
