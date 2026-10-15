"""Test one frozen career-history group against later UFC fights and market prices."""
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from fight_predictor.point_in_time import PointInTimeDatasetBuilder, TemporalFightPredictor
from fight_predictor.career_features import CareerDatasetBuilder, CareerHistory, load_career_bouts
from evaluate_style_matchup_challenger import _align_predictions, _metric, event_block_difference_interval, _validate_candidate_baseline_features
from market_tracker._storage import atomic_write_text

DATA=Path(__file__).resolve().parent/'content/data'


def evaluate(data=DATA):
    raw=pd.read_csv(data/'processed/ufc_fights_reported_doubled.csv',low_memory=False)
    fighters=pd.read_csv(data/'processed/fighter_stats.csv',low_memory=False)
    baseline=PointInTimeDatasetBuilder(raw,fighters)
    roots=[data/'external_mma',data/'external_mma/event_results']
    bouts, coverage=load_career_bouts(raw,roots)
    career=CareerDatasetBuilder(raw,fighters,CareerHistory(bouts))
    print('Replaying full source history and verifying the checked baseline',flush=True)
    base_points=pd.read_csv(data/'processed/ufc_fights_point_in_time.csv',low_memory=False)
    files=[data/'processed'/name for name in ('ufc_fights_reported_doubled.csv','fighter_stats.csv','ufc_fights_point_in_time.csv')]
    files += [Path(__file__),Path(__file__).parent/'fight_predictor/career_features.py',Path(__file__).parent/'fight_predictor/point_in_time.py']
    files += [root/name for root in roots for name in ('bouts.jsonl','identity_map.csv')]
    files += [root/'series_identity_map.csv' for root in roots if (root/'series_identity_map.csv').exists()]
    files += [Path(__file__).parent/'external_mma'/name for name in ('identity.py','integration.py','schema.py','storage.py')]
    files += [Path(__file__).parent/'evaluate_style_matchup_challenger.py']
    source_key=sha256(b'\0'.join(p.read_bytes().replace(b'\r\n',b'\n') for p in files)).hexdigest()
    cache=data.parents[2]/'artifacts/career_research'/source_key
    cache.mkdir(parents=True,exist_ok=True)
    def cached_frame(name, produce):
        path=cache/(name+'.csv');digest=path.with_suffix('.sha256')
        if path.exists() and digest.exists() and sha256(path.read_bytes()).hexdigest()==digest.read_text():
            return pd.read_csv(path,low_memory=False)
        frame=produce()
        atomic_write_text(path,frame.to_csv(index=False,float_format='%.17g',lineterminator='\n'))
        atomic_write_text(digest,sha256(path.read_bytes()).hexdigest())
        return frame
    career_points=cached_frame('career_points',lambda: career.extend_baseline(base_points))
    _validate_candidate_baseline_features(base_points,career_points,baseline.feature_columns)
    def predictions(frame,builder,name):
        def produce():
            result=TemporalFightPredictor(frame,builder).walk_forward_predictions(years)
            result['event_date']=pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
            return result
        return cached_frame(name,produce)
    years=tuple(range(2022,int(raw['date'].astype(str).str[:4].max())+1))
    print('Baseline expanding-year tests',years,flush=True)
    a=predictions(base_points,baseline,'baseline')
    print('Career expanding-year tests',years,flush=True)
    b=predictions(career_points,career,'career')
    paired=_align_predictions(a,b,probability_name='career_probability').rename(columns={'model_probability':'baseline_probability'})
    interval=event_block_difference_interval(paired,'career_probability','baseline_probability')
    report={'version':'predeclared-career-group-v1','production_enabled':False,'coverage':coverage,'input_and_code_sha256':source_key,
        'source_hash_protocol':'SHA-256 with CRLF normalized to LF; combined input/code digest uses NUL-separated file contents in script order',
        'source_data_sha256':{name:sha256((data/'processed'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for name in ('ufc_fights_reported_doubled.csv','fighter_stats.csv')},
        'feature_columns_added':[c for c in career.feature_columns if c not in baseline.feature_columns],
        'evaluation_years':list(years),'scored_fights':len(paired),'scored_events':paired.event_id.nunique(),
        'first_test_date':paired.event_date.min(),'last_test_date':paired.event_date.max(),
        'baseline':_metric(paired,'baseline_probability'),'career':_metric(paired,'career_probability'),
        'paired_log_loss_interval':interval,
        'per_year':{str(y):{'fights':len(g),'baseline':_metric(g,'baseline_probability'),'career':_metric(g,'career_probability')}
            for y,g in paired.groupby(paired.event_date.astype(str).str[:4])},
        'method':'For each test year, train and select settings/calibration only on earlier fights. Original inputs are identical. Added features use only results before each fight date; all same-day results are excluded. External bouts supply history only, never UFC training labels. Identities use approved crosswalks; no name-only career merges.',
        'decision':'Retain for prospective testing only; no automatic promotion.' if interval['ci_95_upper']<0 else 'No convincing improvement; retain production feature contract.',
        'limitations':['This grouped experiment cannot identify which individual feature caused a difference.',
            'A historical comparison remains development evidence, not a new prospective profit record.',
            'Sources and identity links were assembled retrospectively; this is not an as-collected historical betting replay.',
            'Uneven promotion coverage and unresolved identities mean recorded experience is incomplete.',
            'Elo begins at 1500 with fixed K=32; disconnected regional opponent groups have no independently established common strength.']}
    market_path=data/'market_history_backfill/current_model_market_replay.csv'
    if market_path.exists():
        market=pd.read_csv(market_path,low_memory=False)
        reference=market.loc[market.fight_id.isin(paired.fight_id)].copy()
        paired_market=_align_predictions(reference,paired.rename(columns={'baseline_probability':'model_probability'}),probability_name='baseline_probability')
        paired_market=_align_predictions(paired_market,paired.rename(columns={'career_probability':'model_probability'}),probability_name='career_probability')
        if len(paired_market):
            report['market_paired']={'fights':len(paired_market),'events':paired_market.event_id.nunique(),
                'baseline':_metric(paired_market,'baseline_probability'),'career':_metric(paired_market,'career_probability'),
                'market':_metric(paired_market,'market_probability'),
                'career_minus_market':event_block_difference_interval(paired_market,'career_probability','market_probability')}
    eligible=set(career_points.loc[career_points.external_history_available.astype(str).str.lower().eq('true'),'fight_id'])
    covered=paired.loc[paired.fight_id.isin(eligible)]
    report['fights_with_external_history']={'fights':len(covered),'events':covered.event_id.nunique()}
    if len(covered):
        report['fights_with_external_history'].update({'baseline':_metric(covered,'baseline_probability'),'career':_metric(covered,'career_probability'),'paired_log_loss_interval':event_block_difference_interval(covered,'career_probability','baseline_probability')})
    paired['external_history_available']=paired.fight_id.isin(eligible)
    prediction_text=paired.to_csv(index=False,float_format='%.17g',lineterminator='\n')
    atomic_write_text(data/'model_research/career_feature_predictions.csv',prediction_text)
    report['predictions_sha256']=sha256(prediction_text.encode('utf-8')).hexdigest()
    out=data/'model_research/career_feature_comparison.json'
    atomic_write_text(out,json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    evaluate()
