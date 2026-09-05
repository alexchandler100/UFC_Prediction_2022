import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from update_simulation_conditions import new_policy, indicators, history_indicators, build_records, report, price_for, update
from market_tracker._common import canonical_hash
from market_tracker import SimulationComparisonDecision
from test_simulation_comparison import _base, _catalog_publication

NOW = datetime(2026,9,4,3,tzinfo=timezone.utc)


class ConditionalSimulationTests(unittest.TestCase):
    def setUp(self):
        self.policy=new_policy(datetime(2026,9,4,tzinfo=timezone.utc))
        self.pub=_catalog_publication()
        self.item=self.pub['matchups'][0]
        self.item['bootstrap_members']=64
        self.item['aggregate']['uncertainty']=[{'metric':side,'parameter_p025':.4,'parameter_p975':.55,'process_mcse':.01} for side in ('red_win','blue_win')]
        self.pub['publication_sha256']=canonical_hash({k:v for k,v in self.pub.items() if k!='publication_sha256'})
        self.c=SimulationComparisonDecision.create(_base(),self.pub,comparison_issued_at_utc='2026-09-04T02:01:00Z').to_mapping()
        self.raw=pd.DataFrame([{'fighter_url':f'http://ufcstats.com/fighter-details/{side}','fight_url':f'fight-{i}',
            'date':f'2026-0{i+1}-01','distance_strikes_attempts':30,'ground_strikes_attempts':12,'takedowns_attempts':5}
            for side in ('red','blue') for i in range(5)])

    def build(self, **kwargs):
        return build_records(kwargs.get('comparisons',[self.c]),kwargs.get('existing',[]),kwargs.get('publication',self.pub),
            self.raw,'a'*64,{}, {}, {}, self.policy,kwargs.get('now',NOW))

    def test_indicators_are_prefight_and_same_day_data_is_excluded(self):
        result=indicators(self.raw,self.c,self.item,self.policy)
        self.assertTrue(all(result['groups'].values()))
        extra=self.raw.iloc[[0]].copy();extra['date']='2026-09-03';extra['fight_url']='future'
        history=history_indicators(pd.concat([self.raw,extra]),'red',datetime(2026,9,3,12,tzinfo=timezone.utc),730)
        self.assertEqual(history['prior_fights'],5)
        self.assertEqual(history['ground_attempts'],60)

    def test_missing_stats_or_parameter_precision_are_unavailable(self):
        self.raw['ground_strikes_attempts']=None
        self.item['bootstrap_members']=1
        groups=indicators(self.raw,self.c,self.item,self.policy)['groups']
        self.assertIsNone(groups['substantial_relevant_history'])
        self.assertIsNone(groups['narrow_simulation_range'])
        self.assertTrue(groups['recent_history'])

    def test_prospective_only_exact_forecast_and_one_record_per_fight(self):
        records=self.build();self.assertEqual(len(records),1)
        self.assertEqual(self.build(existing=records),[])
        self.assertEqual(self.build(now=datetime(2026,9,6,tzinfo=timezone.utc)),[])
        self.assertEqual(self.build(comparisons=[{**self.c,'comparison_issued_at_utc':'2026-09-03T23:00:00Z'}]),[])
        self.assertEqual(self.build(comparisons=[{**self.c,'simulation_publication_sha256':'b'*64}]),[])
        self.assertEqual(len(self.build(comparisons=[self.c,{**self.c,'comparison_id':'second'}])),1)

    def test_quote_requires_exact_fight_source_time_and_capture(self):
        quote={k:self.c[k] for k in ('matchup_id','event_id','fighter_id','opponent_id')}
        quote.update(quote_id='q',capture_id='cap',observed_at_utc='2026-09-04T02:00:00Z',fighter_moneyline=100,opponent_moneyline=-110)
        meta={'quote_id':'q','source_quote_updated_at_utc':'2026-09-04T01:59:00Z'}
        base={'reference_quote_id':'q','capture_id':'cap'}
        self.assertIsNotNone(price_for(self.c,base,{'q':quote},{'q':meta}))
        self.assertIsNone(price_for(self.c,base,{'q':quote},{}))
        self.assertIsNone(price_for(self.c,base,{'q':quote},{'q':{**meta,'source_quote_updated_at_utc':'2026-09-04T01:00:00Z'}}))

    def test_same_fight_scores_card_intervals_and_fixed_stake_returns(self):
        row=self.build()[0]
        row['price']={'quote':{'fighter_moneyline':100,'opponent_moneyline':100}}
        other=copy.deepcopy(row);other['comparison'].update(base_decision_id='second',event_id='card-two')
        outcomes=[{'decision_id':self.c['base_decision_id'],'target':1},{'decision_id':'second','target':0}]
        result=report([row,other],outcomes,self.policy)
        group=result['results'][0]
        self.assertEqual((group['scored_fights'],group['scored_cards']),(2,2))
        self.assertEqual(group['scores']['market']['fights'],group['scores']['simulation']['fights'])
        self.assertIsNotNone(group['brier_difference_vs_market']['simulation']['card_bootstrap_95'])
        self.assertEqual(group['paper_returns']['simulation'][0]['profit_units'],0)
        self.assertAlmostEqual(group['paper_returns']['simulation'][1]['profit_units'],-.02)
        self.assertEqual(result,report([row,other],outcomes,self.policy))
        void=report([row],[{'decision_id':self.c['base_decision_id'],'target':None}],self.policy)['results'][0]
        self.assertEqual(void['scored_fights'],0);self.assertEqual(void['void_results'],1)

    def test_missing_prices_do_not_look_like_profitable_abstention(self):
        row=self.build()[0]
        group=report([row],[{'decision_id':self.c['base_decision_id'],'target':1}],self.policy)['results'][0]
        self.assertIsNone(group['paper_returns']['simulation'][0]['profit_units'])
        self.assertFalse(group['review_sample_reached'])

    def test_empty_initialization_preserves_policy_and_validates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);update(root);saved=(root/'policy.json').read_bytes();update(root);update(root,True)
            self.assertEqual(saved,(root/'policy.json').read_bytes())


if __name__ == '__main__': unittest.main()
