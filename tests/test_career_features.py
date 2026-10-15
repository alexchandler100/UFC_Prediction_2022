import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fight_predictor.career_features import CareerHistory, CareerDatasetBuilder, load_career_bouts, FEATURES
from fight_predictor.point_in_time import PointInTimeDatasetBuilder
from test_style_matchup import make_profiles, make_fight


class CareerFeatureTests(unittest.TestCase):
    def test_future_and_same_day_results_cannot_change_features(self):
        prior = [('2019-01-01','a','regional','W',True)]
        history = CareerHistory(prior)
        changed = CareerHistory(prior + [('2020-01-01','a','b','L',True),
                                        ('2021-01-01','a','b','L',True)])
        self.assertEqual(history.features('a','2020-01-01'), changed.features('a','2020-01-01'))
        self.assertEqual(history.features('a','2019-01-01'), (0.,)*len(FEATURES))
        self.assertAlmostEqual(changed.features('a','2020-01-02')[1], np.log1p(2), places=14)

    def test_original_inputs_labels_and_swapped_orientation(self):
        raw = pd.DataFrame(make_fight('one','e1','2020-01-01','a','b'))
        profiles = make_profiles('a','b')
        baseline = PointInTimeDatasetBuilder(raw,profiles)
        points = baseline.build()
        history = CareerHistory([('2019-01-01','a','regional','W',True)])
        candidate = CareerDatasetBuilder(raw,profiles,history)
        extended = candidate.extend_baseline(points)
        pd.testing.assert_frame_equal(points,extended[points.columns])
        built = candidate.build()
        np.testing.assert_allclose(extended[list(candidate.feature_columns)],built[list(candidate.feature_columns)],equal_nan=True)
        a = candidate.matchup_features('a','b','2020-02-01','Lightweight')
        b = candidate.matchup_features('b','a','2020-02-01','Lightweight')
        np.testing.assert_allclose(a,-b,equal_nan=True)
        self.assertTrue(extended.external_history_available.all())

    def test_duplicate_ufc_and_cross_source_results_excluded_without_name_merge(self):
        raw = pd.DataFrame(make_fight('one','e1','2020-01-01','a','b'))
        first = raw.iloc[0]
        def observation(date, result='W', source='one', promotion='Regional'):
            return SimpleNamespace(event_date=date, fighter_name=first.fighter,
                opponent_name=first.opponent, result=result, source=source,
                fighter_source_id='a',opponent_source_id='b',promotion=promotion,
                discipline='mma',professional=True)
        rows = [observation('2020-01-01',promotion='Road to UFC'),
                observation('2019-01-01'),observation('2019-01-01',source='two')]
        with patch('fight_predictor.career_features.load_identity_map',return_value={('one','a'):'verified-a'}), \
             patch('fight_predictor.career_features.ExternalMmaStore') as store:
            store.return_value.observations.return_value = rows
            bouts, report = load_career_bouts(raw,[Path('.')])
            self.assertEqual(report['external_bouts'],1)
            self.assertEqual(report['duplicate_bouts_excluded'],2)
            self.assertEqual(bouts[-1][1:3],('verified-a','external:one:b'))
            store.return_value.observations.return_value = [observation('2019-01-01'),observation('2019-01-01','L','two')]
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                load_career_bouts(raw,[Path('.')])

    def test_opponent_strength_is_measured_before_result(self):
        history = CareerHistory([('2018-01-01','b','c','W',False),
                                 ('2019-01-01','a','b','W',True)])
        features = history.features('a','2020-01-01')
        self.assertAlmostEqual(features[3],16/5/400)
        nc = CareerHistory([('2019-01-01','a','b','NC',True)])
        self.assertEqual(nc.features('a','2020-01-01'),(0.,)*len(FEATURES))

    def test_evaluation_preserves_market_orientation_and_publishes_predictions(self):
        from evaluate_career_features import evaluate
        raw=pd.DataFrame(make_fight('one','e1','2023-01-01','a','b')+
                         make_fight('two','e2','2023-02-01','a','b',fighter_result='L'))
        profiles=make_profiles('a','b')
        points=PointInTimeDatasetBuilder(raw,profiles).build()
        predictions=points[['date','event_id','fight_id','fighter_id','opponent_id','target']].copy()
        predictions['model_probability']=[.4,.6]
        market=predictions.rename(columns={'date':'event_date','model_probability':'market_probability'}).copy()
        market['fighter_id'],market['opponent_id']=market.opponent_id.copy(),market.fighter_id.copy()
        market['target']=1-market.target
        market['market_probability']=1-market.market_probability
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'repo/src/content/data'
            (data/'processed').mkdir(parents=True)
            raw.to_csv(data/'processed/ufc_fights_reported_doubled.csv',index=False)
            profiles.to_csv(data/'processed/fighter_stats.csv',index=False)
            points.to_csv(data/'processed/ufc_fights_point_in_time.csv',index=False)
            (data/'market_history_backfill').mkdir()
            market.to_csv(data/'market_history_backfill/current_model_market_replay.csv',index=False)
            for root in (data/'external_mma',data/'external_mma/event_results'):
                root.mkdir(parents=True,exist_ok=True)
                for name in ('bouts.jsonl','identity_map.csv'):
                    (root/name).write_text('')
            with patch('evaluate_career_features.load_career_bouts',return_value=([],{})), \
                 patch('evaluate_career_features.TemporalFightPredictor.walk_forward_predictions',return_value=predictions):
                result=evaluate(data)
            self.assertEqual(result['market_paired']['fights'],2)
            self.assertAlmostEqual(result['market_paired']['career_minus_market']['point_difference'],0)
            self.assertEqual(result['fights_with_external_history']['fights'],0)
            self.assertFalse(result['production_enabled'])
            self.assertTrue((data/'model_research/career_feature_predictions.csv').exists())


if __name__ == '__main__':
    unittest.main()
