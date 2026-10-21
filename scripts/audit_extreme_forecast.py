"""Reproduce the October 17 forecast from frozen Git inputs; run from repo root."""
import sys,json,subprocess,io
from pathlib import Path
import pandas as pd,numpy as np
sys.path.insert(0,'src')
from fight_predictor.point_in_time import PointInTimeDatasetBuilder
REF='e452b32e'  # Saved pre-event artifact; never use the post-fight model.
def blob(p):return subprocess.check_output(['git','-c','safe.directory=C:/Users/Alex/Projects/GitHub/UFC_Prediction_2022','show',REF+':src/content/data/'+p],text=True)
m=json.loads(blob('external/winner_model.json'))
raw=pd.read_csv(io.StringIO(blob('processed/ufc_fights_reported_doubled.csv')),low_memory=False)
profiles=pd.read_csv(io.StringIO(blob('processed/fighter_stats.csv')))
b=PointInTimeDatasetBuilder(raw,profiles);b.build(min_training_date='2026-10-10')
x=b.matchup_features('3c4320a1b6106611','a474aade8eb3a8f0','2026-10-17','Bantamweight').iloc[0]
coef=np.array(m['coefficients']);scale=np.array(m['scaler_scale']);slope=m['calibration_slope'];con=x.to_numpy()/scale*coef*slope
training=pd.read_csv(io.StringIO(blob('processed/ufc_fights_point_in_time.csv')))
training=training[pd.to_datetime(training.date)>=pd.Timestamp(m['training_window_start'])]
rows=[dict(feature=k,value=float(x[k]),scale=float(scale[i]),coefficient=float(coef[i]),logit_contribution=float(con[i]),training_min=float(training[k].min()),training_max=float(training[k].max())) for i,k in enumerate(m['feature_columns'])]
rows.sort(key=lambda r:abs(r['logit_contribution']),reverse=True)
result={'revision':REF,'model_id':m['model_id'],'calibration_slope':slope,'logit':float(sum(con)),'probability':float(1/(1+np.exp(-sum(con)))),'contributions':rows}

flags={'age_known_diff','height_known_diff','reach_known_diff'}
flag_effect=sum(r['logit_contribution'] for r in rows if r['feature'] in flags)
result['training_fights']=len(training)
result['missingness_support']={f:int(training[f].ne(0).sum()) for f in sorted(flags)}
result['without_missingness_flag_effects_probability']=float(1/(1+np.exp(-(sum(con)-flag_effect))))
# A diagnostic using later profile data, NOT a replacement pre-fight forecast.
profile_ref='c6cc0e21'
later_profiles=pd.read_csv(io.StringIO(subprocess.check_output(['git','-c','safe.directory=C:/Users/Alex/Projects/GitHub/UFC_Prediction_2022','show',profile_ref+':src/content/data/processed/fighter_stats.csv'],text=True)))
fid='3c4320a1b6106611'
old_side=b._side_features(fid,pd.Timestamp('2026-10-17'),'Bantamweight')
profile_builder=PointInTimeDatasetBuilder(raw,later_profiles)
new_side=profile_builder._side_features(fid,pd.Timestamp('2026-10-17'),'Bantamweight')
delta=sum((new_side[r['feature'][:-5]]-old_side[r['feature'][:-5]])/r['scale']*r['coefficient']*slope for r in rows)
result['later_profile_only_diagnostic']={'profile_revision':profile_ref,
    'probability':float(1/(1+np.exp(-(sum(con)+delta)))),
    'note':'Frozen pre-event model and opponent history; only debutant profile changes. Later-collected profile is not proven available at issuance. No post-fight result used.',
    'profile':later_profiles.loc[later_profiles.url.str.endswith(fid)].to_dict('records')}
publication=json.loads(blob('external/all_upcoming_forecasts.json'))
ids=set(profiles.url.str.rsplit('/',n=1).str[-1])
result['missing_profile_matchups']=[]
for row in publication['matchups']:
    missing=[row[side+'_name'] for side in ('fighter','opponent') if row[side+'_id'] not in ids]
    if missing:result['missing_profile_matchups'].append({**{k:row[k] for k in ('event_date','fighter_name','opponent_name','model_probability_for_fighter','model_status')},'missing_profiles':missing})
result['publication_matchups']=len(publication['matchups'])
result['publication_low_history_matchups']=sum(r['model_status']=='low_history' for r in publication['matchups'])
saved=json.loads(blob('external/vegas_odds.json'))
i=next(i for i,name in saved['fighter name'].items() if name=='Cody Chovancek')
result['saved_prediction']={k:saved[k].get(i) for k in saved if k.startswith('bayesian ') or k in ('forecast probability','forecast issued at','forecast source','fighter prior fights','opponent prior fights','model probability','betting status')}
assert abs(result['probability']-saved['model probability'][i])<1e-9
result['diagnosis']=[
 'Profile refresh only collects fighters appearing in completed bouts; upcoming debutants can have no profile.',
 'Missing age and height indicators each differ between fighters in only one training bout; scaling by their tiny standard deviations amplifies their coefficients.',
 'Three missing-profile flags supply most of the extreme logit. Removing their effects is a sensitivity diagnostic, not a validated replacement model.',
 'Low-history tagging does not shrink or suppress the primary forecast; the separate Bayesian policy did abstain.',
 'Historical features use the assembled profile table rather than snapshots of profiles available at each historic forecast date. This understates deployment missingness.'
]
result['recommended_next_steps']=[
 'Fetch announced fighter profiles before freezing model inputs; retain retrieval times and source evidence.',
 'Withhold the primary stats forecast when a fighter profile is absent; preserve the raw diagnostic and use market-only context if available.',
 'Evaluate unscaled binary indicators or explicit minimum evidence requirements for missingness effects using earlier-only validation.',
 'Evaluate newcomer calibration separately and simulate missing profiles in validation. Do not deploy a fitted probability correction based on this one fight.'
]
out=Path('audit/model_confidence/extreme_forecast_2026-10-17.json');out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('probability','training_fights','missingness_support','without_missingness_flag_effects_probability')},indent=2))
