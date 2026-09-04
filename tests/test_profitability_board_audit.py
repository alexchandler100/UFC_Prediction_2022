"""Audit-only arithmetic, expiry, access and exposure acceptance scenarios."""

import importlib.util
from datetime import timedelta
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("profitability_board_audit", ROOT / "scripts/audit_profitability_board.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
AS_OF = audit.utc("2026-09-04T12:00:00Z")


def row(identity, *, event="event-a", fight=None, ev=0.2, stake=0.05):
    return {"bet_id": identity, "event_id": event, "event_date": "2026-09-05",
            "matchup_id": fight or identity, "fight": "A vs B", "selection": identity,
            "target_book": "Book A", "offered_moneyline": 200,
            "robust_lower_expected_return": ev, "recommended_fraction": stake,
            "observed_at_utc": "2026-09-04T12:00:00Z",
            "resolved_source_quote_updated_at_utc": "2026-09-04T11:59:00Z",
            "resolved_event_start_utc": "2026-09-05T12:00:00Z",
            "estimated_expected_return": 0.25, "threshold_met": True}


class ProfitabilityBoardAuditTests(unittest.TestCase):
    def test_price_math_can_reverse_unadjusted_edge(self):
        self.assertAlmostEqual(audit.expected_return(0.20, 500), 0.20)
        self.assertAlmostEqual(audit.expected_return(0.15, 500), -0.10)
        self.assertAlmostEqual(audit.expected_return(0.70, -200), 0.05)
        self.assertIsNone(audit.expected_return(None, 500))

    def test_expiry_uses_source_update_not_retrieval(self):
        bet = row("one")
        self.assertTrue(audit.source_eligible(bet, AS_OF))
        self.assertTrue(audit.source_eligible(bet, AS_OF + timedelta(minutes=29)))
        self.assertFalse(audit.source_eligible(bet, AS_OF + timedelta(minutes=29, seconds=1)))
        self.assertTrue(audit.browser_visible(bet))
        bet["resolved_source_quote_updated_at_utc"] = None
        self.assertFalse(audit.source_eligible(bet, AS_OF))

    def test_fresh_quote_cannot_authorize_started_or_untimed_card(self):
        bet = row("one")
        bet["resolved_event_start_utc"] = audit.stamp(AS_OF)
        self.assertFalse(audit.source_eligible(bet, AS_OF))
        bet["resolved_event_start_utc"] = None
        self.assertFalse(audit.source_eligible(bet, AS_OF))

    def test_one_selection_per_fight_and_all_exposure_caps(self):
        rows = [row(f"{event}-{i}", event=event) for event in ("a", "b", "c") for i in range(7)]
        rows.append(row("better-same-fight", event="a", fight="a-0", ev=0.9))
        result = audit.allocate(rows, AS_OF)
        self.assertEqual(result["total_stake_units"], 10)
        funded = [r for r in result["rows"] if r["illustrative_stake_units"] > 0]
        self.assertEqual(len(funded), 10)
        self.assertTrue(all(r["illustrative_stake_units"] <= 1 for r in funded))
        self.assertEqual(len({(r["event_id"], r["matchup_id"]) for r in funded}), 10)
        for event in ("a", "b", "c"):
            self.assertLessEqual(sum(r["illustrative_stake_units"] for r in funded if r["event_id"] == event), 5)
        self.assertIn("better-same-fight", {r["bet_id"] for r in funded})
        self.assertNotIn("a-0", {r["bet_id"] for r in funded})
        self.assertEqual(result, audit.allocate(list(reversed(rows)), AS_OF))
        self.assertTrue(all(r["offered_moneyline"] == 200 for r in result["rows"]))

    def test_allocation_rejects_zero_negative_unknown_stale_and_preserves_small_stake(self):
        rows = [row("negative", ev=-0.1), row("zero", stake=0), row("unknown", ev=None),
                row("small", stake=0.003), row("stale")]
        rows[-1]["resolved_source_quote_updated_at_utc"] = "2026-09-04T10:00:00Z"
        result = audit.allocate(rows, AS_OF)
        self.assertAlmostEqual(result["total_stake_units"], 0.3)
        self.assertEqual(result["funded_selection_count"], 1)

    def test_accessible_second_book_is_repriced_not_hidden(self):
        bet = {**row("total"), "category": "Total rounds", "side": "over",
               "estimated_win_probability": 0.60}
        quotes = [{"book": "Book A", "over_moneyline": 120, "under_moneyline": -140,
                   "source_quote_updated_at_utc": "2026-09-04T11:59:00Z"},
                  {"book": "Book B", "over_moneyline": 100, "under_moneyline": -120,
                   "source_quote_updated_at_utc": "2026-09-04T11:59:00Z"}]
        self.assertFalse(audit.browser_visible(bet, allowed_books={"Book B"}))
        alternate = next(r for r in audit.alternative_candidates(bet, quotes, AS_OF) if r["available_book"] == "Book B")
        self.assertTrue(alternate["threshold_met"])
        self.assertAlmostEqual(alternate["estimated_expected_return"], 0.20)

    def test_each_moneyline_alternative_excludes_its_own_book(self):
        bet = {**row("moneyline"), "category": "Moneyline", "side": "fighter", "model_weight": 0}
        quotes = [{"book": name, "no_vig_fighter_probability": p,
                   "fighter_moneyline": 200, "opponent_moneyline": -250,
                   "source_quote_updated_at_utc": "2026-09-04T11:59:00Z"}
                  for name, p in (("Book A", .8), ("Book B", .4), ("Book C", .4), ("Book D", .4))]
        candidates = {r["available_book"]: r for r in audit.alternative_candidates(bet, quotes, AS_OF)}
        self.assertAlmostEqual(candidates["Book A"]["estimated_win_probability"], .4)
        self.assertAlmostEqual(candidates["Book B"]["estimated_win_probability"], 1.6 / 3)
        self.assertEqual(audit.alternative_candidates(bet, quotes[:3], AS_OF), [])

    def test_method_rows_cannot_become_qualified_board_entries(self):
        methods = {"settlement_status": "unverified", "method_markets": [{"matchup_id": "m",
                   "book_quotes": [{"book": "A", "is_complete_six_way": False,
                   "selections": [{"selection": "A by decision", "moneyline": 1000,
                                   "candidate_model_probability": .8,
                                   "candidate_model_estimated_return": 7.8}]}]}]}
        result = audit.method_surface_rows(methods)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["comparison_only"])
        self.assertFalse(result[0]["on_qualified_board"])


if __name__ == "__main__":
    unittest.main()
