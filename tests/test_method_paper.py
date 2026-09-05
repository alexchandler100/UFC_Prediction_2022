import sys
import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_method_forecast_capture import _forecast, _price_snapshot
from fight_predictor.outcome_publication import OUTCOME_MODEL_VERSION
from update_method_paper import build_decisions, settle, summarize, update
from market_tracker.equal_stake_experiment import seal

NOW = datetime(2026, 9, 4, 18, tzinfo=timezone.utc)


class MethodPaperTests(unittest.TestCase):
    def setUp(self):
        self.policy = seal({'activated_at_utc': '2026-09-04T17:00:00Z', 'minimum_expected_return': .05})
        self.forecast = _forecast(model_version=OUTCOME_MODEL_VERSION)
        self.quote = _price_snapshot(fighter_prices={'ko_tko': 1000}, opponent_prices={'decision': 300})

    def build(self, **kwargs):
        return build_decisions(kwargs.get('quotes', [self.quote]), kwargs.get('forecasts', [self.forecast]),
            kwargs.get('existing', []), kwargs.get('policy', self.policy), kwargs.get('now', NOW))

    def test_highest_ev_one_selection_and_canonical_fighter_orientation(self):
        rows = self.build()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['risk_units'], 1)
        self.assertEqual(rows[0]['selection']['fighter_id'], 'a-fighter')
        self.assertEqual(rows[0]['selection']['method'], 'ko_tko')
        self.assertAlmostEqual(rows[0]['selection']['probability'], .20)
        self.assertAlmostEqual(rows[0]['selection']['expected_return'], 1.2)
        self.assertEqual(self.build(existing=rows), [])

    def test_no_backfill_stale_future_legacy_or_mismatched_forecast(self):
        self.assertEqual(self.build(policy={**self.policy, 'activated_at_utc': '2026-09-04T18:01:00Z'}), [])
        self.assertEqual(self.build(now=datetime(2026, 9, 4, 19, tzinfo=timezone.utc)), [])
        self.assertEqual(self.build(now=datetime(2026, 9, 4, 17, tzinfo=timezone.utc)), [])
        self.assertEqual(self.build(forecasts=[_forecast()]), [])
        self.assertEqual(self.build(forecasts=[_forecast(model_version=OUTCOME_MODEL_VERSION,capture_id='other')]), [])

    def test_first_eligible_pass_stays_frozen(self):
        rows = self.build(quotes=[_price_snapshot(fighter_prices={'submission': -110}, opponent_prices={})])
        self.assertEqual(rows[0]['risk_units'], 0)
        self.assertIsNone(rows[0]['selection'])
        self.assertEqual(self.build(existing=rows), [])

    def test_win_loss_void_unknown_and_schedule_change(self):
        decision = self.build()[0]
        def sides(method='KO/TKO', rounds='3 Rnd (5-5-5)', winner='a-fighter'):
            return [{'fighter_id': name, 'result': 'W' if name == winner else 'L',
                     'method': method, 'time_format': rounds} for name in ('a-fighter','b-fighter')]
        self.assertEqual(settle(decision,sides())['profit_units'],10)
        self.assertEqual(settle(decision,sides(winner='b-fighter'))['profit_units'],-1)
        self.assertEqual(settle(decision,sides('SUB'))['status'],'loss')
        self.assertEqual(settle(decision,sides('DQ'))['status'],'void')
        self.assertEqual(settle(decision,sides(rounds='5 Rnd (5-5-5-5-5)'))['status'],'void')
        self.assertIsNone(settle(decision,sides('UNRECOGNIZED')))
        self.assertIsNone(settle(decision,sides(rounds='')))
        self.assertIsNone(settle(decision,sides()[:1]))

    def test_pending_and_expired_recommendations_are_retained(self):
        rows = self.build()
        result = summarize(rows, [], self.policy, datetime(2026,9,4,19,tzinfo=timezone.utc))
        self.assertEqual(result['recommendations'][0]['status'],'price_expired')
        self.assertEqual(result['settled_risk_units'],0)
        self.assertIsNone(result['return_per_unit'])

    def test_empty_initialization_is_idempotent(self):
        class Store:
            def __init__(self,*args): pass
            def read(self): return ()
        with tempfile.TemporaryDirectory() as directory, patch('update_method_paper.MethodMarketStore',Store), patch('update_method_paper.MethodForecastStore',Store):
            root=Path(directory)
            first=update(root=root)
            policy=(root/'policy.json').read_bytes()
            self.assertEqual(update(root=root)['frozen_fights'],0)
            self.assertEqual(update(root=root,validate_only=True)['paper_recommendations'],0)
            self.assertEqual((root/'policy.json').read_bytes(),policy)
            self.assertFalse(first['execution_enabled'])

    def test_settled_recommendation_keeps_original_price_and_probability(self):
        rows = self.build()
        result = summarize(rows, [{'matchup_id': rows[0]['matchup_id'], 'event_id': rows[0]['event_id'],
            'status': 'win', 'risk_units': 1, 'profit_units': 10}], self.policy, NOW)
        self.assertEqual(len(result['recommendations']), 1)
        self.assertEqual(result['recommendations'][0]['settlement_status'], 'win')
        self.assertEqual(result['recommendations'][0]['moneyline'], 1000)
        self.assertEqual(result['recommendations'][0]['probability'], rows[0]['selection']['probability'])


if __name__ == '__main__': unittest.main()
