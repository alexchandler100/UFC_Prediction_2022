"""Inventory the published winner model's actual inputs and unused data sources."""
import argparse
import json
from pathlib import Path

from market_tracker._common import canonical_hash
from market_tracker._storage import atomic_write_text

DATA = Path(__file__).resolve().parent / 'content/data'


def build(data=DATA):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    model = read(data / 'external/winner_model.json')
    features = model['feature_columns']
    policy_path = data / 'external_mma/model_policy.json'
    policy = read(policy_path) if policy_path.exists() else {}
    artifacts = []
    for name in ('winner_feature_factorial', 'round_cardio_factorial', 'stance_matchup_factorial', 'outcome_feature_comparison'):
        path = data / f'external_mma/{name}.json'
        if path.exists():
            d = read(path)
            artifacts.append({'artifact': str(path.relative_to(data)).replace('\\', '/'),
                'sha256': canonical_hash(d), 'sample': d.get('sample'), 'decision': d.get('decision')})
    external_path = data / 'external_mma/bouts.jsonl'
    recovery_path=data/'model_research/recovery_feature_comparison.json'
    recovery=read(recovery_path) if recovery_path.exists() else {}
    if recovery:
        artifacts.append({'artifact':'model_research/recovery_feature_comparison.json','sha256':canonical_hash(recovery),
            'sample':{'scored_fights':recovery['scored_fights'],'scored_events':recovery['scored_events']},
            'decision':recovery['decision']})
    career_path=data/'model_research/career_feature_comparison.json'
    if career_path.exists():
        career=read(career_path)
        artifacts.append({'artifact':'model_research/career_feature_comparison.json','sha256':canonical_hash(career),
            'sample':{'scored_fights':career['scored_fights'],'scored_events':career['scored_events']},
            'decision':career['decision']})
    event_path=data/'external_mma/event_results/coverage.json'
    events=read(event_path) if event_path.exists() else {}
    bayes_path=data/'external/bayesian_winner_challenger.json'
    bayes=read(bayes_path) if bayes_path.exists() else {}
    return {'version': 'published-model-input-inventory-v1', 'model_id': model['model_id'],
        'model_sha256': canonical_hash(model), 'model_version': model['model_version'],
        'training_fights': model['training_fights'], 'training_window_start': model['training_window_start'],
        'data_through': model['data_through'], 'feature_count': len(features), 'feature_columns': features,
        'uncertainty_scope':{'production_winner':model.get('model_type'),
            'published_bayesian_winner':{'model_type':bayes.get('model_type'), 'posterior_approximation':bayes.get('posterior_approximation'),
                'calibration_uncertainty':bayes.get('calibration_uncertainty'),'credible_level':bayes.get('credible_level')},
            'robust_kelly':'80% central calibration-slope posterior band; lower 10th percentile used for sizing. Conditional on the model and recorded data; not all sources of fight uncertainty.'},
        'layoff_feature_present': 'days_since_fight_log_diff' in features,
        'layoff_contract': 'For this point-in-time feature contract: log(1+days) per fighter, capped at five years, then fighter minus opponent. No recorded history uses two years plus has_history. This is time since the latest replayed source fight, not necessarily the latest MMA fight anywhere.',
        'external_history_enabled': policy.get('enable_auxiliary_replay', False),
        'external_history_records_available': sum(bool(line.strip()) for line in external_path.open(encoding='utf-8')) if external_path.exists() else 0,
        'researched_groups': artifacts,
        'additional_event_results': {'production_enabled':False, 'records':events.get('validation',{}).get('observations',0),
            'approved_ufcstats_identity_links':events.get('approved_ufcstats_identity_links',0), 'promotions':events.get('promotions',[]),
            'note':'Separate revision-pinned secondary-source research ledger. Unlinked identities do not merge into production history.'},
        'coverage_note': 'A dataset being stored does not mean production uses it. Winner, outcome and simulation models have different contracts. Historical results, post-fight stats, and later market prices must not enter earlier forecasts.',
        'next_feature_candidates': [
            {'group':'Layoff and recovery', 'inputs':'Layoff buckets; age-by-layoff; time since KO/TKO loss; recent activity count.',
             'source':'Existing dated source fight history; broader MMA history needed to correct regional-fight gaps.',
             'status':recovery.get('decision','Proposed chronological research; not production inputs.')},
            {'group':'Fight duration context', 'inputs':'Both fighters\' absolute activity and age; division finish rates; explicit three/five-round schedule.',
             'source':'Existing history and verified schedule data. Relative-only features can hide shared context.',
             'status':'Proposed duration-model research; compare against market totals at identical lines.'},
            {'group':'Pre-fight changes', 'inputs':'Official weigh-in results and weight misses; timestamped opponent changes and announcement lead time.',
             'source':'Official UFC event announcements/weigh-in reports; record first observation and stable identities.',
             'status':'Not yet a structured training source; no retrospectively guessed short-notice flags.'}],
        'evaluation_rule': 'Use all valid chronological coverage; select settings on earlier fights, test on later untouched cards, and compare probability quality and price-matched returns with the market. No automatic promotion.'}


def update(data=DATA, validate_only=False):
    report = build(data)
    path = data / 'model_research/model_input_inventory.json'
    if validate_only:
        if not path.exists() or json.loads(path.read_text(encoding='utf-8')) != report:
            raise ValueError('model input inventory does not match published artifacts')
    else:
        atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(f"Model inventory: {report['feature_count']} inputs, {report['training_fights']} training fights; layoff used={report['layoff_feature_present']}, external history enabled={report['external_history_enabled']}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    update(validate_only=parser.parse_args().validate_only)
