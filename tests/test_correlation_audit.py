"""Analytic checks for the offline correlation audit, not a live staking policy."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('correlation_audit', ROOT / 'audit/profitability/2026-10-03/correlation_audit.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class CorrelationAuditTests(unittest.TestCase):
    def test_binary_case_matches_closed_form_kelly_and_keeps_unused_cash(self):
        probabilities = np.array([.505, .495])
        allocation = audit.joint_allocation(probabilities, np.array([[1.0], [-1.0]]), budget=.02)
        self.assertAlmostEqual(allocation[0] * .02, .01, places=6)

    def test_identical_outcomes_share_one_exposure_instead_of_doubling_it(self):
        probabilities = np.array([.505, .495])
        returns = np.array([[1.0, 1.0], [-1.0, -1.0]])
        allocation = audit.joint_allocation(probabilities, returns, budget=.02)
        self.assertAlmostEqual(allocation.sum() * .02, .01, places=6)

    def test_mutually_exclusive_good_prices_can_both_improve_log_growth(self):
        probabilities = np.array([.45, .45, .10])
        returns = np.array([[2.0, -1.0], [-1.0, 2.0], [-1.0, -1.0]])
        allocation = audit.joint_allocation(probabilities, returns)
        np.testing.assert_allclose(allocation, [.5, .5], atol=1e-6)
        combined = probabilities @ np.log(1 + .01 * returns @ allocation)
        single = probabilities @ np.log(1 + .01 * returns[:, 0])
        self.assertGreater(combined, single)

    def test_no_positive_edge_can_leave_the_entire_budget_unspent(self):
        allocation = audit.joint_allocation(np.array([.2, .8]), np.array([[2.0], [-1.0]]))
        np.testing.assert_allclose(allocation, [0], atol=1e-7)

    def test_same_method_at_multiple_books_retains_only_best_price(self):
        a = {'fighter_id': 'a', 'method': 'ko_tko', 'moneyline': 200}
        b = {**a, 'moneyline': 250}
        c = {**a, 'method': 'submission', 'moneyline': 500}
        result = audit.best_unique_offers({'offers': [a, b, c]})
        self.assertEqual(result, [b, c])


if __name__ == '__main__':
    unittest.main()
