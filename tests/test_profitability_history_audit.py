from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_profitability_history import STRATEGIES, stressed_profits, summarize, validate_ledger


def sample():
    rows = []
    for strategy in STRATEGIES:
        for index, (won, odds) in enumerate(((True, 3.0), (False, 2.0))):
            rows.append({"event_date": f"2025-01-0{index + 1}", "event_id": str(index),
                         "fight_id": str(index), "strategy": strategy, "book_name": "A", "side": "fighter",
                         "decimal_odds": odds, "fair_probability": 0.6, "estimated_ev": 0.6 * odds - 1,
                         "won": won, "profit_units": odds - 1 if won else -1,
                         "quote_age_hours": 1, "selected_threshold": 0.05,
                         "threshold_selection_status": "selected_on_earlier_flat_profit", "qualifies_selected_threshold": True})
    report = {"coverage": {"scored_fights": 2, "scored_events": 2},
              "pooled_profitability": {s: {"selections": 2, "events": 2, "profit_units": 1} for s in STRATEGIES}}
    return pd.DataFrame(rows), report


class ProfitabilityHistoryAuditTests(unittest.TestCase):
    def test_payout_stress_reduces_only_net_winnings(self):
        frame, _ = sample()
        np.testing.assert_allclose(stressed_profits(frame.iloc[:2], 0.05), [1.9, -1])

    def test_saved_report_reconciliation_rejects_altered_profit(self):
        frame, report = sample()
        validate_ledger(frame, report)
        report["pooled_profitability"][STRATEGIES[0]]["profit_units"] = 7
        with self.assertRaisesRegex(ValueError, "report profit differs"):
            validate_ledger(frame, report)

    def test_duplicate_fight_and_changed_selection_are_rejected(self):
        frame, report = sample()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_ledger(pd.concat([frame, frame.iloc[:1]]), report)
        frame.loc[0, "qualifies_selected_threshold"] = False
        with self.assertRaisesRegex(ValueError, "selection flags"):
            validate_ledger(frame, report)

    def test_no_bets_has_zero_exposure_and_undefined_roi(self):
        frame, _ = sample()
        result = summarize(frame.iloc[:0], 0, 100)
        self.assertEqual(result["profit_units"], 0)
        self.assertEqual(result["risk_units"], 0)
        self.assertIsNone(result["roi"])
        self.assertIsNone(result["roi_ci_95_lower"])

    def test_one_card_does_not_produce_false_precision(self):
        frame, _ = sample()
        rows = frame.iloc[:2].copy()
        rows["event_id"] = "same"
        rows["event_date"] = "2025-01-01"
        result = summarize(rows, 0, 1000)
        self.assertIsNone(result["roi_ci_95_lower"])
        self.assertEqual(result["event_close_drawdown_units"], 0)
        self.assertEqual(result["source_fight_id_order_drawdown_units"], 1)


if __name__ == "__main__":
    unittest.main()
