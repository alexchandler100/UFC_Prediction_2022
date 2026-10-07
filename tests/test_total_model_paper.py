import copy
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from market_tracker._common import canonical_hash
from market_tracker.equal_stake_experiment import seal
from update_total_model_paper import build_decisions, settle, summarize, update
import tempfile

NOW = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
POLICY = seal({'activated_at_utc': '2026-10-07T11:00:00Z', 'minimum_expected_return': .05})

def publication(**changes):
    offer = dict(matchup_id='fight', model_version='candidate-discrete-time-competing-risks-v2-verified-schedules',
        schedule_contract_version='verified-pre-fight-schedule-v1', side='over', line=1.5, scheduled_rounds=3,
        capture_id='capture', observed_at_utc='2026-10-07T12:00:00Z', forecast_issued_at_utc='2026-10-07T10:00:00Z',
        event_start_utc='2026-10-10T20:00:00Z', source_quote_updated_at_utc='2026-10-07T11:59:00Z',
        model_probability=.6, offered_moneyline=100, target_book='Book', quote_id='q', selection='Over 1.5 rounds',
        fighter_name='A', opponent_name='B')
    offer.update(changes)
    body = dict(observed_at_utc='2026-10-07T12:00:00Z', capture_id='capture', event_id='card', event_date='2026-10-10',
        matchups=[dict(matchup_id='fight',fighter_id='a',opponent_id='b',fighter_name='A',opponent_name='B')],
        prop_markets={'total_rounds': {'candidate_offers': [offer]}})
    body['publication_sha256'] = canonical_hash(body)
    return body

class TotalModelPaperTests(unittest.TestCase):
    def test_freezes_pick_and_pass_and_never_reselects(self):
        picks=build_decisions(publication(), [], POLICY, NOW)
        self.assertAlmostEqual(picks[0]['selection']['expected_return'], .2)
        self.assertEqual(build_decisions(publication(model_probability=.9),picks,POLICY,NOW),[])
        passes=build_decisions(publication(model_probability=.51),[],POLICY,NOW)
        self.assertIsNone(passes[0]['selection'])
        self.assertEqual(build_decisions(publication(),passes,POLICY,NOW),[])

    def test_no_backfill_stale_future_post_start_or_unpaired_inputs(self):
        self.assertEqual(build_decisions(publication(),[],{**POLICY,'activated_at_utc':'2026-10-07T12:01:00Z'},NOW),[])
        for changes in [dict(source_quote_updated_at_utc='2026-10-07T11:00:00Z'),
                        dict(source_quote_updated_at_utc='2026-10-07T12:01:00Z'),
                        dict(forecast_issued_at_utc='2026-10-07T12:01:00Z'),
                        dict(event_start_utc='2026-10-07T12:00:00Z'),dict(capture_id='wrong'),dict(model_version='legacy')]:
            self.assertEqual(build_decisions(publication(**changes),[],POLICY,NOW),[])
        self.assertEqual(build_decisions(publication(),[],POLICY,datetime(2026,10,7,13,tzinfo=timezone.utc)),[])
        p=publication();p['capture_id']='tampered'
        with self.assertRaises(ValueError):build_decisions(p,[],POLICY,NOW)

    def test_competing_books_and_lines_pick_only_best_per_fight(self):
        p=publication();offer=copy.deepcopy(p['prop_markets']['total_rounds']['candidate_offers'][0])
        offer.update(offered_moneyline=200,quote_id='q2',line=2.5,selection='Over 2.5 rounds')
        p['prop_markets']['total_rounds']['candidate_offers'].append(offer)
        p.pop('publication_sha256');p['publication_sha256']=canonical_hash(p)
        rows=build_decisions(p,[],POLICY,NOW)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['selection']['quote_id'],'q2')

    def test_total_without_moneyline_uses_saved_forecast_identity(self):
        p=publication();identities=p.pop('matchups');p['matchups']=[]
        p.pop('publication_sha256');p['publication_sha256']=canonical_hash(p)
        self.assertEqual(build_decisions(p,[],POLICY,NOW),[])
        rows=build_decisions(p,[],POLICY,NOW,identities)
        self.assertEqual(rows[0]['fighter_id'],'a')
        self.assertEqual(rows[0]['opponent_id'],'b')

    def test_settlement_clock_boundary_voids_and_unresolved_results(self):
        decision=build_decisions(publication(),[],POLICY,NOW)[0]
        def sides(duration=500,method='KO/TKO',schedule='3 Rnd (5-5-5)'):
            return [dict(fighter_id=f,result=r,total_fight_time=duration,method=method,time_format=schedule)
                    for f,r in [('a','W'),('b','L')]]
        self.assertEqual(settle(decision,sides())['status'],'win')
        self.assertEqual(settle(decision,sides(400))['profit_units'],-1)
        self.assertEqual(settle(decision,sides(450))['status'],'void')
        self.assertEqual(settle(decision,sides(method='DQ'))['status'],'void')
        self.assertEqual(settle(decision,sides(schedule='5 Rnd (5-5-5-5-5)'))['status'],'void')
        self.assertIsNone(settle(decision,sides(method='UNKNOWN')))
        bad=sides();bad[1]['total_fight_time']=600
        self.assertIsNone(settle(decision,bad))
        self.assertIsNone(settle(decision,sides(schedule='')))
        under=build_decisions(publication(side='under',selection='Under 1.5 rounds'),[],POLICY,NOW)[0]
        self.assertEqual(settle(under,sides(400))['status'],'win')
        for result in ['D','NC']:
            pair=sides()
            for r in pair:r['result']=result
            self.assertEqual(settle(decision,pair)['risk_units'],0)

    def test_report_retains_original_price_fighters_and_resolved_profit(self):
        d=build_decisions(publication(),[],POLICY,NOW)
        r=summarize(d,[dict(matchup_id='fight',event_id='card',status='win',risk_units=1,profit_units=1)],POLICY,NOW)
        self.assertEqual(r['recommendations'][0]['fighter_name'],'A')
        self.assertEqual(r['recommendations'][0]['opponent_name'],'B')
        self.assertEqual(r['recommendations'][0]['moneyline'],100)
        self.assertEqual(r['return_per_unit'],1)

    def test_initialization_validation_and_policy_idempotence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            update(root=root);before=(root/'policy.json').read_bytes()
            update(root=root,validate_only=True);update(root=root)
            self.assertEqual(before,(root/'policy.json').read_bytes())
