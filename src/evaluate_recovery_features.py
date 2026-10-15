"""Test one frozen recovery feature group using all valid expanding-year coverage."""
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from fight_predictor.point_in_time import PointInTimeDatasetBuilder, TemporalFightPredictor
from fight_predictor.recovery_features import RecoveryDatasetBuilder
from evaluate_style_matchup_challenger import _align_predictions, _metric, event_block_difference_interval, _validate_candidate_baseline_features
from market_tracker._storage import atomic_write_text

DATA=Path(__file__).resolve().parent/'content/data'


def evaluate(data=DATA):
    raw=pd.read_csv(data/'processed/ufc_fights_reported_doubled.csv',low_memory=False)
    fighters=pd.read_csv(data/'processed/fighter_stats.csv',low_memory=False)
    baseline=PointInTimeDatasetBuilder(raw,fighters)
    recovery=RecoveryDatasetBuilder(raw,fighters)
    print('Replaying full source history and verifying the checked baseline',flush=True)
    base_points=pd.read_csv(data/'processed/ufc_fights_point_in_time.csv',low_memory=False)
    files=[data/'processed'/name for name in ('ufc_fights_reported_doubled.csv','fighter_stats.csv','ufc_fights_point_in_time.csv')]
    files += [Path(__file__),Path(__file__).parent/'fight_predictor/recovery_features.py',Path(__file__).parent/'fight_predictor/point_in_time.py']
    source_key=sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    cache=data.parents[2]/'artifacts/recovery_research'/source_key
    cache.mkdir(parents=True,exist_ok=True)
    def cached_frame(name, produce):
        path=cache/(name+'.csv');digest=path.with_suffix('.sha256')
        if path.exists() and digest.exists() and sha256(path.read_bytes()).hexdigest()==digest.read_text():
            return pd.read_csv(path,low_memory=False)
        frame=produce()
        atomic_write_text(path,frame.to_csv(index=False,float_format='%.17g',lineterminator='\n'))
        atomic_write_text(digest,sha256(path.read_bytes()).hexdigest())
        return frame
    recovery_points=cached_frame('recovery_points',recovery.build)
    _validate_candidate_baseline_features(base_points,recovery_points,baseline.feature_columns)
    def predictions(frame,builder,name):
        def produce():
            result=TemporalFightPredictor(frame,builder).walk_forward_predictions(years)
            result['event_date']=pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
            return result
        return cached_frame(name,produce)
    years=tuple(range(2022,int(raw['date'].astype(str).str[:4].max())+1))
    print('Baseline expanding-year tests',years,flush=True)
    a=predictions(base_points,baseline,'baseline')
    print('Recovery expanding-year tests',years,flush=True)
    b=predictions(recovery_points,recovery,'recovery')
    paired=_align_predictions(a,b,probability_name='recovery_probability').rename(columns={'model_probability':'baseline_probability'})
    interval=event_block_difference_interval(paired,'recovery_probability','baseline_probability')
    report={'version':'predeclared-recovery-group-v1','production_enabled':False,
        'source_hash_protocol':'SHA-256 of UTF-8 source CSV with CRLF normalized to LF',
        'source_data_sha256':{name:sha256((data/'processed'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for name in ('ufc_fights_reported_doubled.csv','fighter_stats.csv')},
        'feature_columns_added':[c for c in recovery.feature_columns if c not in baseline.feature_columns],
        'evaluation_years':list(years),'scored_fights':len(paired),'scored_events':paired.event_id.nunique(),
        'first_test_date':paired.event_date.min(),'last_test_date':paired.event_date.max(),
        'baseline':_metric(paired,'baseline_probability'),'recovery':_metric(paired,'recovery_probability'),
        'paired_log_loss_interval':interval,
        'per_year':{str(y):{'fights':len(g),'baseline':_metric(g,'baseline_probability'),'recovery':_metric(g,'recovery_probability')}
            for y,g in paired.groupby(paired.event_date.astype(str).str[:4])},
        'method':'For each test year, train and select settings/calibration only on earlier fights. Both variants replay identical source history; no external research results or new production features are enabled.',
        'decision':'Retain for prospective testing only; no automatic promotion.' if interval['ci_95_upper']<0 else 'No convincing improvement; retain production feature contract.',
        'limitations':['This grouped experiment cannot identify which individual feature caused a difference.',
            'A historical comparison remains development evidence, not a new prospective profit record.']}
    market_path=data/'market_history_backfill/current_model_market_replay.csv'
    if market_path.exists():
        market=pd.read_csv(market_path,low_memory=False)
        reference=market.loc[market.fight_id.isin(paired.fight_id)].copy()
        paired_market=_align_predictions(reference,paired.rename(columns={'baseline_probability':'model_probability'}),probability_name='baseline_probability')
        paired_market=_align_predictions(paired_market,paired.rename(columns={'recovery_probability':'model_probability'}),probability_name='recovery_probability')
        if len(paired_market):
            report['market_paired']={'fights':len(paired_market),'events':paired_market.event_id.nunique(),
                'baseline':_metric(paired_market,'baseline_probability'),'recovery':_metric(paired_market,'recovery_probability'),
                'market':_metric(paired_market,'market_probability'),
                'recovery_minus_market':event_block_difference_interval(paired_market,'recovery_probability','market_probability')}
    out=data/'model_research/recovery_feature_comparison.json'
    atomic_write_text(out,json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    evaluate()
