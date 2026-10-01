import sys
import tempfile
import unittest
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_market_first_paper import quote, forecast, metadata, OBSERVED
from market_tracker.bayesian_kelly import BayesianKellyCalibrator
from market_tracker import StoreIntegrityError
from market_tracker.equal_stake_experiment import (
    CAPTURE_CONTRACT_VERSION, LEGACY_CAPTURE_CONTRACT, build_records, choose, report, seal, verify,
)
import update_equal_stake_experiment as runner


class EqualStakeTests(unittest.TestCase):
    def setUp(self):
        self.quotes = (quote("A", -110, -110), quote("B", -105, -115),
                       quote("C", -115, -105), quote("Target", 200, -250))
        self.calibrator = BayesianKellyCalibrator.load()
        self.policy = seal({"activated_at_utc": "2026-09-04T11:00:00Z",
                            "minimum_expected_return": .05})

    def build(self, *, quotes=None, forecasts=None, source=None, existing=(), now=OBSERVED, policy=None):
        quotes = self.quotes if quotes is None else quotes
        return build_records(quotes, [forecast()] if forecasts is None else forecasts,
                             metadata(quotes) if source is None else source,
                             existing, policy or self.policy, self.calibrator, now)

    def test_freezes_both_sides_all_books_and_never_duplicates(self):
        rows = self.build()
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["offers"]), 8)
        self.assertEqual(self.build(existing=rows), [])
        self.assertEqual(rows, self.build(quotes=tuple(reversed(self.quotes))))
        for strategy in ("market", "adjusted_market", "production_model"):
            selected = choose(rows[0], strategy, None, .05)
            self.assertEqual(selected["book"], "Target")
            self.assertEqual(selected["side"], "fighter")
        selected = choose(rows[0], "production_model", "A", .05)
        self.assertEqual(selected["book"], "A")
        self.assertIsNone(choose(rows[0], "market", "Unavailable", .05))

    def test_rejects_stale_future_missing_and_pre_activation_prices(self):
        self.assertEqual(self.build(now="2026-09-04T12:06:00Z"), [])
        self.assertEqual(self.build(now="2026-09-04T11:59:00Z"), [])
        self.assertEqual(self.build(source=[]), [])
        later_policy = seal({**self.policy, "activated_at_utc": "2026-09-04T12:01:00Z"})
        self.assertEqual(self.build(policy=later_policy), [])
        for updated in ("2026-09-04T11:00:00Z", "2026-09-04T12:01:00Z"):
            source = [replace(item, source_quote_updated_at_utc=updated) for item in metadata(self.quotes)]
            self.assertEqual(self.build(source=source), [])
        self.assertEqual(self.build(quotes=[replace(q, event_start_utc=None) for q in self.quotes]), [])

    def test_forecasts_must_exist_before_prices_and_be_native(self):
        self.assertEqual(self.build(forecasts=[]), [])
        self.assertEqual(self.build(forecasts=[replace(forecast(), forecast_issued_at_utc="2026-09-04T12:01:00Z")]), [])
        self.assertEqual(self.build(forecasts=[replace(forecast(), probability_provenance="legacy_reconstructed_american_odds")]), [])

    def test_outside_window_does_not_claim_a_shortage_of_books(self):
        observed = "2026-09-04T18:00:00Z"
        quotes = [quote(book, -110, -110, observed=observed) for book in ('A', 'B', 'C', 'D')]
        diagnostic = {}
        records = build_records(quotes, [forecast()], metadata(quotes), [], self.policy,
                                self.calibrator, observed, diagnostics=diagnostic)
        self.assertEqual(records, [])
        self.assertEqual(diagnostic['outcomes'], {'outside_card_decision_window': 1})
        self.assertEqual(diagnostic['quote_exclusions'], {'outside_card_decision_window': 4})

    def test_later_bout_starts_keep_card_window_and_preserve_both_times(self):
        source = [replace(m, source_commence_time_utc="2026-09-05T18:00:00Z")
                  for m in metadata(self.quotes)]
        rows = self.build(source=source)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_start_utc"], "2026-09-05T12:00:00.000000Z")
        self.assertTrue(all(o["source_metadata"]["source_commence_time_utc"] ==
                            "2026-09-05T18:00:00Z" for o in rows[0]["offers"]))
        self.assertEqual(rows[0]["capture_contract_version"], CAPTURE_CONTRACT_VERSION)

    def test_started_bout_is_rejected_even_if_card_clock_is_future(self):
        source = [replace(m, source_commence_time_utc=OBSERVED) for m in metadata(self.quotes)]
        diagnostic = {}
        rows = build_records(self.quotes, [forecast()], source, (), self.policy,
                             self.calibrator, OBSERVED, diagnostics=diagnostic)
        self.assertEqual(rows, [])
        self.assertEqual(diagnostic["quote_exclusions"], {"card_or_bout_started": 4})
        self.assertEqual(diagnostic["outcomes"], {"fewer_than_four_eligible_books": 1})

    def test_source_identity_and_forecast_card_time_are_still_checked(self):
        source = list(metadata(self.quotes))
        source[0] = replace(source[0], event_id="wrong-event")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.build(source=source)
        with self.assertRaises(StoreIntegrityError):
            self.build(forecasts=[replace(forecast(), event_start_utc="2026-09-05T18:00:00Z")])

    def test_repair_cannot_backfill_pre_activation_captures(self):
        contract = {"activated_at_utc": "2026-09-04T12:01:00Z"}
        self.assertEqual(build_records(self.quotes, [forecast()], metadata(self.quotes), (),
                         self.policy, self.calibrator, "2026-09-04T12:02:00Z",
                         capture_contract=contract), [])

    def test_legacy_and_repaired_results_are_reported_separately(self):
        new = self.build()[0]
        old = dict(new, matchup_id="legacy")
        old.pop("capture_contract_version")
        result = report([old, new], [{"matchup_id": "legacy", "target": 1},
                                    {"matchup_id": new["matchup_id"], "target": 0}], self.policy)
        cohorts = {r["capture_contract_version"]: r for r in result["capture_cohorts"]}
        for version, profit in [(LEGACY_CAPTURE_CONTRACT, 2), (CAPTURE_CONTRACT_VERSION, -1)]:
            row = next(r for r in cohorts[version]["results"] if r["strategy"] == "market"
                       and r["book"] == "all_books_hypothetical" and r["winning_payout_reduction"] == 0)
            self.assertEqual(row["profit_units"], profit)
        self.assertNotIn("capture_contract_version", old)

    def test_target_book_does_not_influence_own_consensus(self):
        first = self.build()[0]
        changed = self.build(quotes=(*self.quotes[:3], quote("Target", 300, -400)))[0]
        probabilities = lambda row: next(o["probabilities"] for o in row["offers"] if o["book"] == "Target")
        self.assertEqual(probabilities(first), probabilities(changed))

    def test_equal_stake_settlement_stress_voids_and_pending(self):
        row = self.build()[0]
        def summary(target, haircut=0):
            settlements = [] if target == "pending" else [{"matchup_id": row["matchup_id"], "target": target}]
            result = report([row], settlements, self.policy)
            return next(r for r in result["results"] if r["strategy"] == "market"
                        and r["book"] == "all_books_hypothetical" and r["winning_payout_reduction"] == haircut)
        self.assertEqual(summary(1)["profit_units"], 2)
        self.assertEqual(summary(1)["risk_units"], 1)
        self.assertEqual(summary(1, .05)["profit_units"], 1.9)
        self.assertEqual(summary(0)["profit_units"], -1)
        self.assertEqual(summary(0)["max_card_end_drawdown"], .01)
        self.assertEqual(summary(None)["risk_units"], 0)
        self.assertEqual(summary(None)["void_bets"], 1)
        self.assertEqual(summary("pending")["pending_bets"], 1)
        self.assertIsNone(summary(1)["card_bootstrap_roi_95"])
        baseline = report([], [], self.policy)["results"][0]
        self.assertEqual(baseline["ending_normalized_bankroll"], 100)
        self.assertIsNone(baseline["return_per_unit"])

    def test_hash_detects_changed_records(self):
        row = self.build()[0]
        verify(row)
        row["fighter_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify(row)

    def test_related_bets_stay_together_in_card_accounting(self):
        first = self.build()[0]
        second = {**first, "matchup_id": "second"}
        third = {**first, "matchup_id": "third", "event_id": "later",
                 "event_start_utc": "2026-09-12T12:00:00Z"}
        outcomes = [{"matchup_id": first["matchup_id"], "target": 1},
                    {"matchup_id": "second", "target": 0},
                    {"matchup_id": "third", "target": 0}]
        result = report([first, second, third], outcomes, self.policy)
        row = next(r for r in result["results"] if r["strategy"] == "market"
                   and r["book"] == "all_books_hypothetical" and r["winning_payout_reduction"] == 0)
        self.assertEqual(row["settled_cards"], 2)
        self.assertEqual(row["settled_bets"], 3)
        self.assertEqual(row["profit_units"], 0)
        self.assertAlmostEqual(row["max_card_end_drawdown"], 1 / 101)
        self.assertEqual(row["card_bootstrap_roi_95"], [-1, .5])

    def test_capture_then_settlement_is_idempotent_and_keeps_decisions(self):
        class Store:
            def __init__(self, rows=()):
                self.rows = rows
            def read(self):
                return self.rows
        class Clock(datetime):
            moment = datetime(2026, 9, 4, 11, tzinfo=timezone.utc)
            @classmethod
            def now(cls, tz=None):
                return cls.moment
        stores = (Store(self.quotes), Store([forecast()]), Store(metadata(self.quotes)), Store(), Store())
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "datetime", Clock), patch.object(runner, "_stores", return_value=stores):
            root = Path(directory)
            raw = root / "raw.csv"
            raw.write_text("fixture\n1\n")
            with patch.object(runner, "RAW_PATH", raw), patch.object(runner, "_result_index", return_value=({}, set(), set())):
                self.assertEqual(runner.update(root=root)["frozen_fights"], 0)
                Clock.moment = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)
                captured = runner.update(root=root)
                self.assertEqual(captured["frozen_fights"], 1)
                self.assertEqual(captured["settled_fights"], 0)
                decision_bytes = (root / "decisions.json").read_bytes()
                # A no-op must preserve the original file even on a CRLF checkout.
                decision_bytes = decision_bytes.replace(b"\n", b"\r\n")
                (root / "decisions.json").write_bytes(decision_bytes)
            key = ("future-event", "fighter-a", "fighter-b")
            Clock.moment = datetime(2026, 9, 5, 18, tzinfo=timezone.utc)
            with patch.object(runner, "RAW_PATH", raw), patch.object(runner, "_result_index", return_value=({key: (1, "fight")}, set(), set())):
                settled = runner.update(root=root)
                self.assertEqual(settled["settled_fights"], 1)
                self.assertEqual(settled, runner.update(root=root))
                self.assertEqual(settled, runner.update(root=root, validate_only=True))
                self.assertEqual(decision_bytes, (root / "decisions.json").read_bytes())

    def test_runner_initializes_once_and_preserves_policy_and_records(self):
        class Store:
            def read(self):
                return ()
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "_stores", return_value=(Store(),) * 5):
            root = Path(directory)
            first = runner.update(root=root)
            policy_bytes = (root / "policy.json").read_bytes()
            self.assertEqual(first, runner.update(root=root))
            self.assertEqual(first, runner.update(root=root, validate_only=True))
            self.assertEqual(policy_bytes, (root / "policy.json").read_bytes())
            self.assertEqual(first["frozen_fights"], 0)


if __name__ == "__main__":
    unittest.main()
