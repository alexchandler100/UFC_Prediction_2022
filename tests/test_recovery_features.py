import sys
import unittest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fight_predictor.point_in_time import PointInTimeDatasetBuilder
from fight_predictor.recovery_features import RecoveryDatasetBuilder
from test_style_matchup import make_profiles, make_fight


class RecoveryFeatureTests(unittest.TestCase):
    def dataset(self, later=False):
        a=make_fight('one','e1','2020-01-01','a','b',fighter_result='L')
        for row in a:row['method']='KO/TKO'
        rows=a+make_fight('two','e2','2020-03-01','a','b')
        if later:rows+=make_fight('future','e3','2021-01-01','a','b')
        return pd.DataFrame(rows)

    def test_original_features_stay_identical_and_future_results_cannot_rewrite_them(self):
        raw=self.dataset();profiles=make_profiles('a','b')
        base=PointInTimeDatasetBuilder(raw,profiles);bp=base.build()
        candidate=RecoveryDatasetBuilder(raw,profiles);cp=candidate.build()
        self.assertEqual(cp.target.tolist(),bp.target.tolist())
        np.testing.assert_allclose(bp[list(base.feature_columns)],cp[list(base.feature_columns)],equal_nan=True)
        future=RecoveryDatasetBuilder(self.dataset(True),profiles).build()
        np.testing.assert_allclose(cp[list(candidate.feature_columns)],future.iloc[:len(cp)][list(candidate.feature_columns)],equal_nan=True)

    def test_recovery_is_prior_history_only_and_fighter_swap_negates_features(self):
        candidate=RecoveryDatasetBuilder(self.dataset(),make_profiles('a','b'));points=candidate.build()
        first=points.iloc[0]
        self.assertEqual(first['prior_ko_loss_diff'],0)
        second=points.iloc[1]
        # Canonical orientation can be either corner; the difference must be nonzero.
        self.assertEqual(abs(second['prior_ko_loss_diff']),1)
        a=candidate.matchup_features('a','b','2020-04-01','Lightweight')
        b=candidate.matchup_features('b','a','2020-04-01','Lightweight')
        np.testing.assert_allclose(a.to_numpy(),-b.to_numpy(),equal_nan=True)

    def test_evaluation_serializes_date_columns_and_aligns_reversed_market_identity(self):
        from evaluate_recovery_features import evaluate
        raw=self.dataset();raw['date']=raw['date'].str.replace('2020','2023')
        profiles=make_profiles('a','b')
        points=PointInTimeDatasetBuilder(raw,profiles).build()
        prediction=points[['date','event_id','fight_id','fighter_id','opponent_id','target']].copy()
        prediction['model_probability']=[.4,.6]
        market=prediction.rename(columns={'date':'event_date','model_probability':'market_probability'}).copy()
        market['fighter_id'],market['opponent_id']=market.opponent_id.copy(),market.fighter_id.copy()
        market['target']=1-market.target;market['market_probability']=1-market.market_probability
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'repo/src/content/data';(data/'processed').mkdir(parents=True)
            raw.to_csv(data/'processed/ufc_fights_reported_doubled.csv',index=False)
            profiles.to_csv(data/'processed/fighter_stats.csv',index=False)
            points.to_csv(data/'processed/ufc_fights_point_in_time.csv',index=False)
            (data/'market_history_backfill').mkdir()
            market.to_csv(data/'market_history_backfill/current_model_market_replay.csv',index=False)
            with patch('evaluate_recovery_features.TemporalFightPredictor.walk_forward_predictions',return_value=prediction.copy()):
                result=evaluate(data)
            self.assertEqual(result['scored_fights'],2)
            self.assertEqual(result['market_paired']['fights'],2)
            self.assertEqual(result['first_test_date'],'2023-01-01')
            self.assertFalse(json.loads((data/'model_research/recovery_feature_comparison.json').read_text())['production_enabled'])


if __name__=='__main__':unittest.main()
