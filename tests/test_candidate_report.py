import sys
import unittest
import tempfile
from unittest.mock import patch
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from update_candidate_report import build_report

NOW = '2026-09-04T12:00:00Z'


class Calibration:
    artifact = {'artifact_sha256': 'frozen'}
    mean = .5
    lower = .4
    def assessment(self, p, price):
        return {'status': 'available', 'posterior_mean_probability': self.mean,
                'posterior_lower_probability': self.lower,
                'recommended_fraction': .01 if self.lower * (1 + price / 100) > 1 else 0}


class CandidateReportTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration()
        self.forecasts = {'publication_sha256': 'forecast', 'matchups': [
            {'matchup_id': 'fight', 'fighter_name': 'A', 'opponent_name': 'B',
             'model_probability_for_fighter': .7, 'event_id': 'card'}]}
        self.board = {'publication_sha256': 'board', 'forecast_publication_sha256': 'forecast',
                      'observed_at_utc': NOW, 'bets': [], 'market_matchups': [
            {'matchup_id': 'fight', 'event_start_utc': '2026-09-05T12:00:00Z', 'book_quotes': [
                {'book': name, 'book_key': name, 'no_vig_fighter_probability': .5,
                 'fighter_moneyline': 200 if name == 'Target' else 100, 'opponent_moneyline': -110,
                 'source_quote_updated_at_utc': '2026-09-04T11:59:00Z'}
                for name in ('Target', 'B', 'C', 'D')]}]}

    def result(self, now=NOW):
        return build_report(self.forecasts, self.board, {}, self.calibration, generated_at=now)

    def target(self, result):
        return next(row for row in result['rows'] if row['book'] == 'Target' and row['side'] == 'fighter')

    def test_reports_all_offers_and_missing_markets_without_mutation(self):
        self.forecasts['matchups'].append({'matchup_id': 'unpriced', 'model_probability_for_fighter': .6})
        before = deepcopy(self.board)
        result = self.result()
        self.assertEqual(len(result['rows']), 9)
        self.assertEqual(self.board, before)
        self.assertEqual(result['rows'][-1]['reasons_now'], ['no_prices'])
        row = self.target(result)
        self.assertAlmostEqual(row['raw_market_ev'], .5)
        self.assertAlmostEqual(row['model_ev'], 1.1)
        self.assertNotIn('Target', row['consensus_books'])
        self.assertIn('portfolio', row['reasons_at_capture'])

    def test_distinguishes_adjustment_from_uncertainty_and_expiry(self):
        self.calibration.mean = .3
        self.calibration.lower = .25
        row = self.target(self.result())
        self.assertIn('below_threshold', row['reasons_now'])
        self.assertIn('uncertainty', row['reasons_now'])
        self.calibration.mean = .5
        row = self.target(self.result('2026-09-04T13:00:00Z'))
        self.assertNotIn('below_threshold', row['reasons_now'])
        self.assertIn('uncertainty', row['reasons_now'])
        self.assertIn('expired', row['reasons_now'])
        self.assertNotIn('expired', row['reasons_at_capture'])

    def test_excludes_stale_other_books_and_rejects_mismatched_inputs(self):
        self.board['market_matchups'][0]['book_quotes'][1]['source_quote_updated_at_utc'] = '2026-09-04T10:00:00Z'
        row = self.target(self.result())
        self.assertIsNone(row['raw_market_probability'])
        self.assertIn('no_consensus', row['reasons_now'])
        self.board['forecast_publication_sha256'] = 'different'
        with self.assertRaises(ValueError):
            self.result()

    def test_unfunded_verified_totals_still_capture_and_persist_forecasts(self):
        import capture_market_snapshot as collector
        from test_prop_market import _quote
        from market_tracker import TotalRoundsForecastStore
        quote = _quote()
        publication = {'model_version': 'candidate-discrete-time-competing-risks-v2-verified-schedules',
            'schedule_contract_version': 'verified-pre-fight-schedule-v1',
            'forecast_matchup_count': 1, 'event_id': quote.event_id, 'event_date': quote.event_date,
            'forecast_issued_at_utc': '2026-08-21T12:00:00Z', 'model_id': 'verified-duration',
            'model_trained_through': '2026-08-15', 'source_commit_sha': 'a' * 40,
            'publication_sha256': 'b' * 64, 'betting_performance_validated': False,
            'matchups': [{'matchup_id': quote.matchup_id, 'scheduled_rounds': 3,
                          'schedule_basis': 'explicit_time_format', 'total_round_over_probabilities': {'2.5': .65}}]}
        with patch.object(collector, 'validate_outcome_forecast_publication', return_value=publication):
            captured, counts = collector._build_total_round_forecasts((quote,), publication)
        self.assertEqual(counts['total_round_forecast_lines'], 1)
        self.assertEqual(captured[0].over_probability, .65)
        with tempfile.TemporaryDirectory() as directory:
            store = TotalRoundsForecastStore(Path(directory) / 'forecasts.csv', Path(directory) / 'forecasts.jsonl')
            store.append(captured)
            self.assertEqual(store.read(), captured)

    def test_unpaired_totals_prices_remain_visible_without_inventing_probabilities(self):
        current = {'prop_markets': {'total_rounds': {'quote_count': 1, 'forecast_count': 0,
            'markets': [{'matchup_id': 'fight', 'line': 2.5, 'book_quotes': [
                {'book': 'A', 'over_moneyline': 120, 'under_moneyline': -140,
                 'source_quote_updated_at_utc': NOW, 'event_start_utc': '2026-09-05T12:00:00Z'}]}]}}}
        result = build_report(self.forecasts, self.board, current, self.calibration, generated_at=NOW)
        totals = [row for row in result['rows'] if row['market'] == 'Total rounds']
        self.assertEqual(len(totals), 2)
        self.assertTrue(all(row['model_ev'] is None and 'totals_research' in row['reasons_now'] for row in totals))


if __name__ == '__main__':
    unittest.main()
