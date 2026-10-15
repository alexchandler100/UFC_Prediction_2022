"""Record immutable, pre-event career-model predictions; never place bets."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fight_predictor.career_features import CareerDatasetBuilder, CareerHistory, load_career_bouts
from fight_predictor.point_in_time import TemporalFightPredictor
from market_tracker._storage import atomic_write_text

DATA = Path(__file__).resolve().parent/'content/data'
VERSION = 'career-shadow-v1'


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line] if path.exists() else []


def write_json(path,value):
    atomic_write_text(path,json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def score(features, model):
    values=np.asarray(features,dtype=float)
    if len(values)!=len(model['feature_columns']) or not np.isfinite(values).all():
        raise ValueError('Invalid shadow features')
    logit=float((values/np.asarray(model['scale'])) @ np.asarray(model['coefficients']))
    raw=1/(1+np.exp(-np.clip(logit,-709,709)))
    return float(TemporalFightPredictor._calibrate(np.array([raw]),model['calibration_slope'])[0])


def record(matchups, builder, model, existing, now, quotes=()):
    """A changed model/input gets a new row; repeated identical capture is a no-op."""
    rows=list(existing)
    keys={r['forecast_id'] for r in rows}
    skipped=[]
    for matchup in matchups:
        date=matchup['event_date']
        if date <= now.date().isoformat():
            continue  # No exact start time: reject all same-day captures.
        a,b=matchup.get('fighter_id'),matchup.get('opponent_id')
        if not a or not b or a==b or a not in builder._profiles or b not in builder._profiles:
            skipped.append({'matchup_id':matchup.get('matchup_id'),'reason':'unresolved_identity'})
            continue
        values=builder.matchup_features(a,b,date,matchup.get('division','Unknown')).iloc[0].fillna(0).tolist()
        books={}
        for q in quotes:
            if q.get('event_id')!=matchup['event_id'] or q.get('event_date')!=date or {q.get('fighter_id'),q.get('opponent_id')}!={a,b}:continue
            observed=datetime.fromisoformat(q['observed_at_utc'].replace('Z','+00:00'))
            if not 0 <= (now-observed).total_seconds() <= 86400:continue
            if q['book'] not in books or observed > datetime.fromisoformat(books[q['book']]['observed_at_utc'].replace('Z','+00:00')):
                books[q['book']]=q
        market=[{'quote_id':q['quote_id'],'book':book,'observed_at_utc':q['observed_at_utc'],
            'no_vig_probability_for_fighter':q['no_vig_fighter_probability'] if q['fighter_id']==a else 1-q['no_vig_fighter_probability']}
            for book,q in sorted(books.items())]
        body={k:matchup[k] for k in ('matchup_id','event_id','event_date','fighter_id','opponent_id','fighter_name','opponent_name')}
        body.update({'model_id':model['model_id'],'features':values,
            'market_quotes':market,
            'probability_for_fighter':score(values,model),
            'baseline_probability_for_fighter':matchup.get('model_probability_for_fighter'),
            'baseline_model_id':matchup.get('model_id'),
            'baseline_issued_at_utc':matchup.get('forecast_issued_at_utc'),
            'schedule_generated_at_utc':matchup.get('schedule_generated_at_utc'),
            'fighter_external_history':bool(builder.career_history.features(a,date)[0]),
            'opponent_external_history':bool(builder.career_history.features(b,date)[0])})
        identifier=digest(body)
        if identifier not in keys:
            rows.append({**body,'forecast_id':identifier,'recorded_at_utc':now.isoformat(),'execution_enabled':False})
            keys.add(identifier)
    return rows,skipped


def settle(rows, raw, existing, now):
    output=list(existing)
    settled={r['forecast_id'] for r in output}
    wanted={(r['event_date'],r['event_id'],frozenset((r['fighter_id'],r['opponent_id']))) for r in rows if r['forecast_id'] not in settled}
    index={}
    for _,group in raw.groupby('fight_url',sort=False):
        if len(group)!=2:continue
        r=group.iloc[0]
        a=str(r.fighter_url).rstrip('/').rsplit('/',1)[-1]
        b=str(r.opponent_url).rstrip('/').rsplit('/',1)[-1]
        event=str(r.event_url).rstrip('/').rsplit('/',1)[-1]
        key=(str(r.date)[:10],event,frozenset((a,b)))
        if key not in wanted:continue
        if key in index:raise ValueError('Ambiguous settlement fight')
        index[key]=(a,r.result,str(r.fight_url))
    for row in rows:
        if row['forecast_id'] in settled:continue
        result=index.get((row['event_date'],row['event_id'],frozenset((row['fighter_id'],row['opponent_id']))))
        if result is None or row['event_date']>now.date().isoformat():continue
        a,outcome,url=result
        target={'W':1,'L':0}.get(outcome)
        if target is not None and a!=row['fighter_id']:target=1-target
        output.append({'forecast_id':row['forecast_id'],'settled_at_utc':now.isoformat(),
            'fight_url':url,'target':target,'status':'settled' if target is not None else 'void'})
    return output


def update(data=DATA, now=None, validate_only=False):
    live_clock=now is None
    now=now or datetime.now(timezone.utc)
    root=data/'market/career_shadow'
    rows=read_rows(root/'forecasts.jsonl')
    settlements=read_rows(root/'settlements.jsonl')
    if validate_only:
        ids=set()
        for row in rows:
            if row['forecast_id'] in ids:raise ValueError('Duplicate shadow capture')
            ids.add(row['forecast_id'])
            if datetime.fromisoformat(row['recorded_at_utc']).date().isoformat()>=row['event_date']:
                raise ValueError('Capture was not before event date')
            model=json.loads((root/'models'/f"{row['model_id']}.json").read_text())
            if digest({k:v for k,v in model.items() if k!='model_id'})!=model['model_id']:
                raise ValueError('Model archive checksum mismatch')
            if abs(score(row['features'],model)-row['probability_for_fighter'])>1e-12:
                raise ValueError('Archived probability cannot be reproduced')
            if digest({k:v for k,v in row.items() if k not in ('forecast_id','recorded_at_utc','execution_enabled')})!=row['forecast_id']:
                raise ValueError('Forecast checksum mismatch')
        if len({r['forecast_id'] for r in settlements})!=len(settlements) or any(r['forecast_id'] not in ids for r in settlements):
            raise ValueError('Invalid shadow settlements')
        print(f'Career shadow validated: {len(rows)} captures, {len(settlements)} settlements')
        return
    if rows:update(data,validate_only=True)
    raw=pd.read_csv(data/'processed/ufc_fights_reported_doubled.csv',low_memory=False)
    fighters=pd.read_csv(data/'processed/fighter_stats.csv',low_memory=False)
    points=pd.read_csv(data/'processed/ufc_fights_point_in_time.csv',low_memory=False)
    if str(raw.date.max())[:10]>=now.date().isoformat():
        raise ValueError('Shadow requires completed source history before capture date')
    roots=[data/'external_mma',data/'external_mma/event_results']
    bouts,coverage=load_career_bouts(raw,roots)
    deferred=sum(b[0]>=now.date().isoformat() for b in bouts)
    bouts=[b for b in bouts if b[0]<now.date().isoformat()]
    coverage['source_bouts_deferred_until_next_day']=deferred
    builder=CareerDatasetBuilder(raw,fighters,CareerHistory(bouts))
    candidate=builder.extend_baseline(points)
    files=[data/'processed'/name for name in ('ufc_fights_reported_doubled.csv','fighter_stats.csv','ufc_fights_point_in_time.csv')]
    files += [root/name for root in roots for name in ('bouts.jsonl','identity_map.csv','series_identity_map.csv') if (root/name).exists()]
    files += [Path(__file__),Path(__file__).parent/'fight_predictor/career_features.py',Path(__file__).parent/'fight_predictor/point_in_time.py']
    files += [Path(__file__).parent/'external_mma'/name for name in ('identity.py','integration.py','schema.py','storage.py')]
    hashes={str(p.relative_to(data)) if p.is_relative_to(data) else p.name:sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in files}
    hashes['eligible_history_sha256']=digest(bouts)
    input_id=digest(hashes)
    model=None
    for path in sorted((root/'models').glob('*.json')):
        saved=json.loads(path.read_text())
        if saved['input_id']==input_id:model=saved;break
    if model is None:
        print('Training career shadow on completed fights',flush=True)
        predictor=TemporalFightPredictor(candidate,builder)
        predictor.train()
        model={'version':VERSION,'input_id':input_id,'input_hashes':hashes,
            'feature_columns':list(builder.feature_columns),'scale':predictor._artifact_scale.tolist(),
            'coefficients':predictor._artifact_coefficients.tolist(),'calibration_slope':predictor.calibration_slope,
            'selected_c':predictor.best_c,'training_fights':len(predictor.training_data),
            'trained_through':str(raw.date.max())[:10],'coverage':coverage}
        model['model_id']=digest(model)
        write_json(root/'models'/f"{model['model_id']}.json",model)
    # State replay only; historical feature rows already come from the checked snapshot.
    print('Replaying current fighter state',flush=True)
    builder.build(min_training_date=str(points.date.max())[:10])
    if live_clock:now=datetime.now(timezone.utc)
    upcoming=json.loads((data/'external/all_upcoming_forecasts.json').read_text())
    if datetime.fromisoformat(upcoming['generated_at_utc'].replace('Z','+00:00'))>now:
        raise ValueError('Upcoming publication is from the future')
    matchups=[{**r,'schedule_generated_at_utc':upcoming['generated_at_utc']} for r in upcoming['matchups']]
    before=len(rows)
    rows,skipped=record(matchups,builder,model,rows,now,read_rows(data/'market/quote_snapshots.jsonl'))
    settlements=settle(rows,raw,settlements,now)
    for name,values in [('forecasts',rows),('settlements',settlements)]:
        atomic_write_text(root/f'{name}.jsonl',''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in values))
    first={}
    for row in sorted(rows,key=lambda r:r['recorded_at_utc']):first.setdefault(row['matchup_id'],row)
    latest={r['matchup_id']:r for r in sorted(rows,key=lambda r:r['recorded_at_utc'])}
    outcomes={r['forecast_id']:r for r in settlements}
    scored=[(r,outcomes[r['forecast_id']]['target']) for r in first.values() if r['forecast_id'] in outcomes and outcomes[r['forecast_id']]['target'] is not None]
    report={'version':VERSION,'generated_at_utc':now.isoformat(),'execution_enabled':False,
        'captures':len(rows),'unique_matchups':len(first),'models':len({r['model_id'] for r in rows}),
        'latest_model_id':model['model_id'],'new_captures':len(rows)-before,'skipped':skipped,
        'upcoming_matchups':sum(r['event_date']>now.date().isoformat() for r in first.values()),
        'first_capture_settled_fights':len(scored),
        'upcoming':[{'matchup_id':r['matchup_id'],'event_date':r['event_date'],
            'fighter_name':r['fighter_name'],'opponent_name':r['opponent_name'],
            'probability_for_fighter':r['probability_for_fighter'],'model_id':r['model_id'],
            'recorded_at_utc':r['recorded_at_utc'],'first_probability_for_fighter':first[r['matchup_id']]['probability_for_fighter']}
            for r in sorted(latest.values(),key=lambda r:(r['event_date'],r['matchup_id'])) if r['event_date']>now.date().isoformat()],
        'first_capture_log_loss':float(np.mean([-np.log(np.clip(r['probability_for_fighter'] if y else 1-r['probability_for_fighter'],1e-12,1)) for r,y in scored])) if scored else None,
        'evaluation_rule':'Score the first genuinely pre-event capture per matchup. Preserve later versions separately; never pick the best version after results.',
        'coverage':coverage}
    write_json(root/'report.json',report)
    write_json(root/'contract.json',{'version':VERSION,'execution_enabled':False,
        'same_day_captures':False,'feature_columns':list(builder.feature_columns),
        'training_rule':'Existing ten-year temporal logistic training and earlier-only calibration; unchanged five career features. Refit on changed source data, archive each model.',
        'evaluation_rule':report['evaluation_rule']})
    print(f'Career shadow: {len(rows)-before} new captures; {len(first)} matchups; {len(scored)} first captures settled',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only',action='store_true')
    update(validate_only=parser.parse_args().validate_only)
