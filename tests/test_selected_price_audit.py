"""Independent invariants for the offline selected-price diagnostic."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('selected_price_audit', ROOT / 'audit/profitability/2026-10-01/selected_price_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from test_market_first_paper import quote, metadata


class SelectedPriceAuditTests(unittest.TestCase):
    def test_power_margin_is_symmetric_bounded_and_removes_more_longshot_margin(self):
        self.assertAlmostEqual(audit.power_probability(-110, -110), .5)
        for first, second in ((1000, -2000), (-130, 110), (110, 110), (-100, 100)):
            p = audit.power_probability(first, second)
            self.assertGreater(p, 0)
            self.assertLess(p, 1)
            self.assertAlmostEqual(p + audit.power_probability(second, first), 1, places=13)
        self.assertLess(audit.power_probability(1000, -2000), (1/11) / (1/11 + 20/21))

    def test_consensus_keeps_other_books_and_capture_identity(self):
        objects = [quote(book, -110, -110) for book in ('A', 'B', 'C', 'Target')]
        qs = [q.to_mapping() for q in objects]
        ms = {m.quote_id: m.to_mapping() for m in metadata(objects)}
        target = qs[-1]
        d = {**target, 'market_probability': .5}
        peers = audit.comparison_quotes(d, target, qs, ms)
        self.assertEqual({q['book'] for q in peers}, {'A', 'B', 'C'})
        altered_target = {**target, 'no_vig_fighter_probability': .99}
        self.assertEqual(audit.comparison_quotes(d, altered_target, qs[:-1] + [altered_target], ms), peers)
        wrong_capture = [{**q, 'capture_id': 'another'} for q in qs[:3]]
        with self.assertRaises(ValueError):
            audit.comparison_quotes(d, target, wrong_capture + [target], ms)
        with self.assertRaises(ValueError):
            audit.comparison_quotes({**d, 'market_probability': .6}, target, qs, ms)

    def test_selected_fighter_orientation_and_omission_minimum(self):
        q = quote('A', 300, -400).to_mapping()
        flipped = {**q, 'fighter_id': q['opponent_id'], 'opponent_id': q['fighter_id'],
                   'fighter_moneyline': -400, 'opponent_moneyline': 300,
                   'no_vig_fighter_probability': 1 - q['no_vig_fighter_probability']}
        for method in ('proportional', 'power'):
            self.assertAlmostEqual(audit.probability_for(q, 'fighter-a', method), audit.probability_for(flipped, 'fighter-a', method))
        self.assertEqual(audit.book_omission_range([.5, .5, .5]), (None, None))
        low, high = audit.book_omission_range([.1, .2, .3, .9])
        self.assertAlmostEqual(low, .2)
        self.assertAlmostEqual(high, 1.4 / 3)


if __name__ == '__main__':
    unittest.main()
