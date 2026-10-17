import sys
from pathlib import Path
from datetime import date, datetime, timezone
import unittest
import json
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from link_series_history import identity_witnesses, extend_archive_crosswalk, propose
from external_mma.identity import IDENTITY_COLUMNS


class IdentityWitnessTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([dict(fight_id='one',organization='ufc',event_name='UFC',
            event_date='2025-01-01',fighter_1='Hyun Sung Park',fighter_2='Unknown Opponent',
            winner='Hyun Sung Park',method='Decision')])

    def test_unknown_opponent_does_not_remove_known_identity(self):
        masters=pd.DataFrame([dict(fighter_id='park',fighter_name='Park Hyun-sung')])
        bouts,audit=identity_witnesses(self.rows(),masters,'a'*64,date(2026,1,1))
        self.assertEqual(len(bouts),1)
        self.assertIn('park',[bouts[0].fighter_source_id,bouts[0].opponent_source_id])
        self.assertEqual(audit['bout_local_participant'],1)
        raw=pd.DataFrame([
            dict(date='2025-01-01',fight_url='fight',fighter='HyunSung Park',opponent='Unknown Opponent',fighter_url='ufc/park',result='W'),
            dict(date='2025-01-01',fight_url='fight',fighter='Unknown Opponent',opponent='HyunSung Park',fighter_url='ufc/other',result='L')])
        links=extend_archive_crosswalk(bouts,raw,pd.DataFrame(columns=IDENTITY_COLUMNS),datetime.now(timezone.utc))
        self.assertEqual(links.source_fighter_id.tolist(),['park'])
        raw['result']=['L','W']
        self.assertTrue(extend_archive_crosswalk(bouts,raw,pd.DataFrame(columns=IDENTITY_COLUMNS),datetime.now(timezone.utc)).empty)

    def test_ambiguous_directory_names_stay_bout_local_and_future_excluded(self):
        masters=pd.DataFrame([dict(fighter_id='one',fighter_name='Park Hyun-sung'),
                              dict(fighter_id='two',fighter_name='Hyun Sung Park')])
        rows=self.rows()
        rows['fighter_1']='HyunSung Park'
        rows['winner']='HyunSung Park'
        bouts,_=identity_witnesses(rows,masters,'a'*64,date(2026,1,1))
        self.assertTrue(bouts[0].fighter_source_id.startswith('unresolved:'))
        self.assertTrue(bouts[0].opponent_source_id.startswith('unresolved:'))
        self.assertEqual(identity_witnesses(rows,masters,'a'*64,date(2024,1,1))[0],[])

    def test_duplicate_dated_pair_cannot_grant_link(self):
        masters=pd.DataFrame([dict(fighter_id='park',fighter_name='Park Hyun-sung')])
        bouts,_=identity_witnesses(self.rows(),masters,'a'*64,date(2026,1,1))
        from dataclasses import replace
        series=[replace(bouts[0],source='wiki',promotion='Road to UFC')]
        crosswalk=pd.DataFrame([dict(source=bouts[0].source,source_fighter_id='park',canonical_fighter_id='ufcpark')])
        self.assertTrue(propose(series,bouts*2,crosswalk,{},datetime.now(timezone.utc))[0].empty)

    def test_series_bout_cannot_be_its_own_only_ufc_witness(self):
        from dataclasses import replace
        masters=pd.DataFrame([dict(fighter_id='park',fighter_name='Park Hyun-sung')])
        bouts,_=identity_witnesses(self.rows(),masters,'a'*64,date(2026,1,1))
        series=[replace(bouts[0],source='wiki',promotion='Road to UFC')]
        link=dict(source=bouts[0].source,source_fighter_id='park',canonical_fighter_id='ufcpark',
                  match_method='dated_ufc_pair_name_token_order',notes=json.dumps([{'archive_bout_id':'one'}]))
        self.assertTrue(propose(series,bouts,pd.DataFrame([link]),{},datetime.now(timezone.utc))[0].empty)
        link['notes']=json.dumps([{'archive_bout_id':'another-fight'}])
        self.assertEqual(len(propose(series,bouts,pd.DataFrame([link]),{},datetime.now(timezone.utc))[0]),1)


if __name__=='__main__':unittest.main()
