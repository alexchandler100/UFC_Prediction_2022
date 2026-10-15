import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from update_career_shadow import record, settle, digest, update, write_json
from link_series_history import propose, identity_name, extend_archive_crosswalk
from test_style_matchup import make_fight
from test_style_matchup import make_profiles
from fight_predictor.point_in_time import PointInTimeDatasetBuilder


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,10,15,tzinfo=timezone.utc)
        self.model={'feature_columns':['x'],'scale':[1.], 'coefficients':[.4],'calibration_slope':1.}
        self.model['model_id']=digest(self.model)
        self.builder=SimpleNamespace(_profiles={'a':{},'b':{}},
            matchup_features=lambda *args:pd.DataFrame({'x':[1.]}),
            career_history=SimpleNamespace(features=lambda *args:(1,)))
        self.matchup={'matchup_id':'m','event_id':'e1','event_date':'2026-10-17',
            'fighter_id':'a','opponent_id':'b','fighter_name':'A','opponent_name':'B'}

    def test_capture_is_idempotent_and_changed_model_preserves_first_row(self):
        first,_=record([self.matchup],self.builder,self.model,[],self.now)
        again,_=record([self.matchup],self.builder,self.model,first,self.now)
        self.assertEqual(first,again)
        changed={**self.model,'coefficients':[.8]}
        changed['model_id']=digest({k:v for k,v in changed.items() if k!='model_id'})
        added,_=record([self.matchup],self.builder,changed,first,self.now)
        self.assertEqual(len(added),2)
        self.assertEqual(first[0],added[0])
        self.assertNotEqual(added[0]['probability_for_fighter'],added[1]['probability_for_fighter'])

    def test_reject_same_day_and_unresolved_identity(self):
        rows,skipped=record([{**self.matchup,'event_date':'2026-10-15'},
                             {**self.matchup,'fighter_id':'unknown'}],self.builder,self.model,[],self.now)
        self.assertEqual(rows,[])
        self.assertEqual(len(skipped),1)

    def test_market_reference_rejects_future_and_old_quotes_and_reverses_side(self):
        quote={'event_id':'e1','event_date':'2026-10-17','fighter_id':'b','opponent_id':'a',
            'book':'book','quote_id':'q','no_vig_fighter_probability':.3,'observed_at_utc':'2026-10-14T23:00:00Z'}
        quotes=[quote,{**quote,'book':'old','observed_at_utc':'2026-10-10T00:00:00Z'},
                {**quote,'book':'future','observed_at_utc':'2026-10-15T01:00:00Z'}]
        rows,_=record([self.matchup],self.builder,self.model,[],self.now,quotes)
        self.assertEqual(len(rows[0]['market_quotes']),1)
        self.assertAlmostEqual(rows[0]['market_quotes'][0]['no_vig_probability_for_fighter'],.7)

    def test_settlement_requires_same_event_pair_and_date_and_handles_reversal(self):
        rows,_=record([self.matchup],self.builder,self.model,[],self.now)
        raw=pd.DataFrame(make_fight('f','e1','2026-10-17','b','a',fighter_result='W'))
        later=datetime(2026,10,18,tzinfo=timezone.utc)
        outcomes=settle(rows,raw,[],later)
        self.assertEqual(outcomes[0]['target'],0)
        self.assertEqual(outcomes,settle(rows,raw,outcomes,later))
        self.assertEqual(settle([{**rows[0],'event_id':'wrong'}],raw,[],later),[])
        unrelated=pd.DataFrame(make_fight('old1','old','1997-12-21','c','d')+
                               make_fight('old2','old','1997-12-21','c','d'))
        self.assertEqual(settle(rows,pd.concat([raw,unrelated]),[],later),outcomes)

    def test_saved_model_reproduces_capture_and_detects_tampering(self):
        rows,_=record([self.matchup],self.builder,self.model,[],self.now)
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder);root=data/'market/career_shadow'
            write_json(root/'models'/f"{self.model['model_id']}.json",self.model)
            path=root/'forecasts.jsonl'
            path.write_text(json.dumps(rows[0])+'\n')
            update(data,validate_only=True)
            rows[0]['probability_for_fighter']=.99
            path.write_text(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError,'reproduced'):update(data,validate_only=True)

    def test_complete_update_records_once_and_reuses_archived_fit(self):
        raw=pd.DataFrame(make_fight('f','old','2026-10-10','a','b'))
        profiles=make_profiles('a','b')
        points=PointInTimeDatasetBuilder(raw,profiles).build()
        def train(predictor):
            predictor._artifact_scale=np.ones(len(predictor.feature_columns))
            predictor._artifact_coefficients=np.zeros(len(predictor.feature_columns))
            predictor.calibration_slope=1.
            predictor.best_c=.01
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)
            (data/'processed').mkdir()
            raw.to_csv(data/'processed/ufc_fights_reported_doubled.csv',index=False)
            profiles.to_csv(data/'processed/fighter_stats.csv',index=False)
            points.to_csv(data/'processed/ufc_fights_point_in_time.csv',index=False)
            for root in (data/'external_mma',data/'external_mma/event_results'):
                root.mkdir(parents=True,exist_ok=True)
                for name in ('bouts.jsonl','identity_map.csv'):(root/name).write_text('')
            write_json(data/'external/all_upcoming_forecasts.json',{'generated_at_utc':'2026-10-14T00:00:00Z','matchups':[self.matchup]})
            with patch('update_career_shadow.load_career_bouts',return_value=([('2026-10-10','a','b','W',False),('2026-10-16','a','b','W',True)],{})), \
                 patch('update_career_shadow.TemporalFightPredictor.train',autospec=True,side_effect=train) as fit:
                first=update(data,self.now)
                second=update(data,self.now)
                self.assertEqual(fit.call_count,1)
            self.assertEqual(first['new_captures'],1)
            self.assertEqual(second['new_captures'],0)
            self.assertEqual(first['coverage']['source_bouts_deferred_until_next_day'],1)
            update(data,validate_only=True)


class SeriesLinkTests(unittest.TestCase):
    def test_asian_name_order_requires_a_dated_two_person_ufc_witness(self):
        self.assertEqual(identity_name('HyunSung Park'),identity_name('Park Hyun-sung'))
        raw=pd.DataFrame(make_fight('f','e1','2025-01-01','a','b'))
        raw.loc[raw.fighter_url.str.endswith('/a'),'fighter']='HyunSung Park'
        raw.loc[raw.fighter_url.str.endswith('/b'),'opponent']='HyunSung Park'
        other=raw.loc[raw.fighter_url.str.endswith('/b'),'fighter'].iloc[0]
        archive=SimpleNamespace(promotion='UFC',event_date='2025-01-01',fighter_name='Park Hyun-sung',
            opponent_name=other,fighter_source_id='park',opponent_source_id='other',result='W',source='archive',source_bout_id='witness')
        from external_mma.identity import IDENTITY_COLUMNS
        empty=pd.DataFrame(columns=IDENTITY_COLUMNS)
        result=extend_archive_crosswalk([archive],raw,empty,datetime.now(timezone.utc))
        self.assertEqual(result.loc[result.source_fighter_id.eq('park'),'canonical_fighter_id'].tolist(),['a'])
        archive.event_date='2025-01-02'
        self.assertTrue(extend_archive_crosswalk([archive],raw,empty,datetime.now(timezone.utc)).empty)

    def test_dated_bout_and_ufc_witness_required_and_conflict_rejected(self):
        def bout(source, fid, oid):
            row=SimpleNamespace(event_date='2020-01-01',fighter_name='A',opponent_name='B',
                fighter_source_id=fid,opponent_source_id=oid,promotion='Road to UFC',
                result='W',source=source,observation_id=source,source_url='https://example.org/'+source)
            row.to_dict=lambda:{'observation_id':source,'event_date':row.event_date,'source_url':row.source_url}
            return row
        series=bout('wiki','unresolved:a','unresolved:b')
        archive=bout('archive','stable-a','stable-b')
        crosswalk=pd.DataFrame([{'source':'archive','source_fighter_id':'stable-a','canonical_fighter_id':'ufc-a','notes':'separate UFC date/pair'}])
        now=datetime(2026,10,15,tzinfo=timezone.utc)
        mapping,evidence,conflicts=propose([series],[archive],crosswalk,{},now)
        self.assertEqual(len(mapping),1)
        self.assertEqual(mapping.iloc[0].canonical_fighter_id,'ufc-a')
        self.assertTrue(evidence[0]['witnesses'][0]['ufc_crosswalk_evidence'])
        mapping,_,conflicts=propose([series],[archive],crosswalk,{('wiki','unresolved:a'):'wrong'},now)
        self.assertTrue(mapping.empty)
        self.assertEqual(len(conflicts),1)
        archive.event_date='2020-01-02'
        self.assertTrue(propose([series],[archive],crosswalk,{},now)[0].empty)


if __name__=='__main__':unittest.main()
