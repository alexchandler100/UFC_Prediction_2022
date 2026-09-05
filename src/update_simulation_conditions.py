"""Freeze three conditional-model hypotheses before fights; offline, paper only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random

import pandas as pd

from market_tracker._common import canonical_hash
from market_tracker._storage import atomic_write_text, exclusive_store_lock
from market_tracker.equal_stake_experiment import seal, verify
from market_tracker.paper import _profit_for_one_unit_risk
from market_tracker.simulation_comparison import _validate_simulation_publication
from market_tracker import SimulationComparisonDecision

DATA = Path(__file__).resolve().parent / 'content/data'
MARKET = DATA / 'market'
ROOT = MARKET / 'simulation_conditions'
VERSION = 'simulation-conditions-v1'
RULES = ('substantial_relevant_history', 'recent_history', 'narrow_simulation_range')


def utc(value):
    d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('timezone required')
    return d.astimezone(timezone.utc)


def read(path, default):
    if not path.exists():
        return default
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    atomic_write_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def new_policy(now):
    return seal({'version': VERSION, 'activated_at_utc': now.isoformat(), 'paper_only': True,
        'production_influence': 'none', 'retuning_allowed': False,
        'one_record_per_physical_fight': True,
        'rules': {'substantial_relevant_history': {'minimum_fights_each': 5,
            'minimum_distance_attempts_each': 100, 'minimum_ground_attempts_each': 50,
            'minimum_takedown_attempts_each': 20, 'minimum_complete_fraction_each': .8},
            'recent_history': {'lookback_days': 730, 'minimum_fights_each': 2},
            'narrow_simulation_range': {'maximum_95pct_width_each': .20,
                'maximum_process_mcse_each': .02, 'minimum_bootstrap_members': 32}},
        'betting': {'minimum_ev': .05, 'risk_units': 1, 'one_selection_per_fight_per_strategy': True,
            'price': 'exact_base_decision_reference_book_only', 'winning_payout_reductions': [0, .02, .05]},
        'review_minimum_fights_per_group': 200, 'review_minimum_cards_per_group': 20,
        'bootstrap_samples': 1000, 'bootstrap_seed': 20260905,
        'limitations': ['Thresholds are illustrative and were not optimized on results.',
            'Groups overlap; intervals are descriptive, not adjusted for multiple comparisons.',
            'Each mechanics version is reported separately; no automatic model promotion.',
            'Indicators describe prior records available at capture, not proven exact training inputs.',
            'Ground/distance attempts are activity evidence, not minutes spent in each phase.']})


def history_indicators(raw, fighter_id, cutoff, lookback_days):
    ids = raw.fighter_url.astype(str).str.rstrip('/').str.rsplit('/').str[-1]
    dates = pd.to_datetime(raw.date, utc=True, errors='coerce')
    # Date-only records cannot prove availability earlier on the forecast day.
    prior = raw.loc[ids.eq(fighter_id) & dates.lt(pd.Timestamp(cutoff.date(), tz='UTC'))].copy()
    if prior.fight_url.duplicated().any():
        raise ValueError('duplicate fighter/fight history')
    numeric = prior[['distance_strikes_attempts', 'ground_strikes_attempts', 'takedowns_attempts']].apply(pd.to_numeric, errors='coerce')
    valid = numeric.notna().all(axis=1) & numeric.ge(0).all(axis=1)
    count = len(prior)
    return {'prior_fights': count,
        'recent_fights': int(pd.to_datetime(prior.date, utc=True).ge(cutoff - timedelta(days=lookback_days)).sum()),
        'complete_fights': int(valid.sum()), 'complete_fraction': float(valid.mean()) if count else None,
        **{name: float(numeric.loc[valid, column].sum()) if valid.any() else None for name, column in
           [('distance_attempts','distance_strikes_attempts'), ('ground_attempts','ground_strikes_attempts'), ('takedown_attempts','takedowns_attempts')]}}


def indicators(raw, comparison, item, policy):
    rules = policy['rules']
    sides = [history_indicators(raw, comparison[f'{side}_id'], utc(comparison['simulation_forecast_issued_at_utc']),
        rules['recent_history']['lookback_days']) for side in ('fighter', 'opponent')]
    h = rules['substantial_relevant_history']
    supported = None if any(s['complete_fraction'] is None or s['complete_fraction'] < h['minimum_complete_fraction_each'] for s in sides) else all(
        s['prior_fights'] >= h['minimum_fights_each'] and s['distance_attempts'] >= h['minimum_distance_attempts_each']
        and s['ground_attempts'] >= h['minimum_ground_attempts_each'] and s['takedown_attempts'] >= h['minimum_takedown_attempts_each'] for s in sides)
    u = {x['metric']: x for x in item.get('aggregate', {}).get('uncertainty', [])}
    ranges = []
    for side in ('red_win', 'blue_win'):
        row = u.get(side, {})
        values = [row.get(k) for k in ('parameter_p025', 'parameter_p975', 'process_mcse')]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            ranges = []; break
        lower, upper, mcse = values
        if not 0 <= lower <= upper <= 1 or mcse < 0:
            ranges = []; break
        ranges.append({'metric': side, 'width': upper - lower, 'process_mcse': mcse})
    agreement = rules['narrow_simulation_range']
    members = item.get('bootstrap_members')
    narrow = None if len(ranges) != 2 or not isinstance(members, int) or members < agreement['minimum_bootstrap_members'] else all(
        r['width'] <= agreement['maximum_95pct_width_each'] and r['process_mcse'] <= agreement['maximum_process_mcse_each'] for r in ranges)
    return {'fighter': sides[0], 'opponent': sides[1], 'winner_ranges': ranges, 'bootstrap_members': members,
        'groups': {RULES[0]: supported, RULES[1]: all(s['recent_fights'] >= rules['recent_history']['minimum_fights_each'] for s in sides), RULES[2]: narrow}}


def price_for(comparison, base, quotes, metadata):
    quote = quotes.get(base.get('reference_quote_id'))
    if not quote:
        return None
    meta = metadata.get(quote['quote_id'])
    if not meta or not meta.get('source_quote_updated_at_utc'):
        return None
    at = utc(comparison['base_decision_issued_at_utc'])
    if any(quote[k] != comparison[k] for k in ('matchup_id','event_id','fighter_id','opponent_id')):
        return None
    if quote.get('capture_id') != base.get('capture_id') or meta.get('quote_id') != quote['quote_id']:
        return None
    if not (0 <= (at - utc(quote['observed_at_utc'])).total_seconds() <= 300 and
            0 <= (at - utc(meta['source_quote_updated_at_utc'])).total_seconds() <= 1800):
        return None
    return {'quote': quote, 'source_metadata': meta}


def build_records(comparisons, existing, publication, raw, raw_hash, bases, quotes, metadata, policy, now):
    frozen = {row['comparison']['comparison_id'] for row in existing}
    fight_key = lambda c: (c['event_id'], *sorted((c['fighter_id'], c['opponent_id'])))
    frozen_fights = {fight_key(r['comparison']) for r in existing}
    pending = [c for c in comparisons if c['comparison_id'] not in frozen
        and utc(policy['activated_at_utc']) <= utc(c['comparison_issued_at_utc']) <= now < utc(c['event_start_utc'])]
    if not pending or not publication:
        return []
    publication = _validate_simulation_publication(publication)
    items = {m['matchup_id']: m for m in publication['matchups']}
    records = []
    for c in sorted(pending, key=lambda c: (c['comparison_issued_at_utc'], c['comparison_id'])):
        if fight_key(c) in frozen_fights:
            continue
        item = items.get(c['matchup_id'])
        if (not item or item.get('status') != 'available' or publication['publication_sha256'] != c['simulation_publication_sha256']
                or item.get('parameter_artifact_sha256') != c['simulation_parameter_artifact_sha256']
                or utc(item['forecast_issued_at_utc']) != utc(c['simulation_forecast_issued_at_utc'])):
            continue
        records.append(seal({'policy_sha256': policy['record_sha256'], 'recorded_at_utc': now.isoformat(),
            'comparison': c, 'comparison_sha256': canonical_hash(c), 'raw_data_sha256': raw_hash,
            'indicators': indicators(raw, c, item, policy),
            'price': price_for(c, bases.get(c['base_decision_id'], {}), quotes, metadata)}))
        frozen_fights.add(fight_key(c))
    return records


def metrics(rows, field):
    if not rows:
        return {'fights': 0, 'brier': None, 'log_loss': None, 'accuracy': None}
    probabilities = [min(1-1e-12, max(1e-12, r['comparison'][field])) for r, _ in rows]
    targets = [t for _, t in rows]
    return {'fights': len(rows), 'brier': sum((p-t)**2 for p,t in zip(probabilities,targets))/len(rows),
        'log_loss': sum(-math.log(p if t else 1-p) for p,t in zip(probabilities,targets))/len(rows),
        'accuracy': sum(.5 if p==.5 else float((p>.5)==bool(t)) for p,t in zip(probabilities,targets))/len(rows)}


def interval(rows, field, policy):
    cards = defaultdict(list)
    for row, target in rows:
        c = row['comparison']
        cards[c['event_id']].append((c[field]-target)**2 - (c['market_probability']-target)**2)
    if len(cards) < 2:
        return None
    blocks = [(sum(v), len(v)) for v in cards.values()]
    generator = random.Random(policy['bootstrap_seed'])
    samples = []
    for _ in range(policy['bootstrap_samples']):
        picked = generator.choices(blocks, k=len(blocks))
        samples.append(sum(v[0] for v in picked)/sum(v[1] for v in picked))
    samples.sort()
    return [samples[int(.025*(len(samples)-1))], samples[int(.975*(len(samples)-1))]]


def returns(rows, field, policy):
    output = []
    for haircut in policy['betting']['winning_payout_reductions']:
        risk = profit = available = 0
        for row, target in rows:
            if not row['price']:
                continue
            available += 1
            quote = row['price']['quote']; p = row['comparison'][field]
            options = [(p*(1+_profit_for_one_unit_risk(quote['fighter_moneyline']))-1, 1, quote['fighter_moneyline']),
                ((1-p)*(1+_profit_for_one_unit_risk(quote['opponent_moneyline']))-1, 0, quote['opponent_moneyline'])]
            ev, side, price = max(options)
            if ev < policy['betting']['minimum_ev']:
                continue
            risk += 1
            profit += _profit_for_one_unit_risk(price)*(1-haircut) if target==side else -1
        output.append({'winning_payout_reduction': haircut, 'fights_with_prices': available, 'bets': risk,
            'risk_units': risk, 'profit_units': profit if available else None,
            'return_per_unit': profit/risk if risk else None})
    return output


def report(records, settlements, policy):
    outcomes = {s['decision_id']: s.get('target') for s in settlements}
    reports = []
    profiles = sorted({r['comparison']['mechanics_profile_id'] for r in records})
    for profile in profiles:
        same = [r for r in records if r['comparison']['mechanics_profile_id']==profile]
        for rule, state in [('all', None), *[(rule, state) for rule in RULES for state in (True, False, None)]]:
            group = [r for r in same if rule=='all' or r['indicators']['groups'][rule] is state]
            rows = [(r, outcomes[r['comparison']['base_decision_id']]) for r in group if outcomes.get(r['comparison']['base_decision_id']) in (0,1)]
            cards = len({r['comparison']['event_id'] for r,_ in rows})
            scores = {name: metrics(rows, field) for name, field in [('market','market_probability'),('model','model_probability'),('simulation','simulation_probability')]}
            reports.append({'mechanics_profile_id': profile, 'condition': rule, 'matches': state, 'recorded_fights': len(group),
                'scored_fights': len(rows), 'scored_cards': cards,
                'pending_results': sum(r['comparison']['base_decision_id'] not in outcomes for r in group),
                'void_results': sum(r['comparison']['base_decision_id'] in outcomes and outcomes[r['comparison']['base_decision_id']] is None for r in group),
                'review_sample_reached': len(rows)>=policy['review_minimum_fights_per_group'] and cards>=policy['review_minimum_cards_per_group'],
                'scores': scores, 'brier_difference_vs_market': {name: {'difference': scores[name]['brier']-scores['market']['brier'] if rows else None,
                    'card_bootstrap_95': interval(rows, field, policy)} for name,field in [('model','model_probability'),('simulation','simulation_probability')]},
                'paper_returns': {name: returns(rows,field,policy) for name,field in [('market','market_probability'),('model','model_probability'),('simulation','simulation_probability')]},
                'no_bet_profit_units': 0})
    return {'policy': policy, 'recorded_comparisons': len(records), 'status': 'collecting_evidence',
        'records_sha256': canonical_hash(records), 'results': reports,
        'interpretation': 'Lower probability error than market favors the model; this is not proof of profitable betting. No automatic promotion.'}


def update(root=ROOT, validate_only=False):
    now = datetime.now(timezone.utc)
    with exclusive_store_lock(root/'write.lock'):
        policy = read(root/'policy.json', None)
        if policy is None:
            if validate_only:
                raise ValueError('conditional experiment not initialized')
            policy = new_policy(now); write(root/'policy.json', policy)
        verify(policy)
        if policy['version'] != VERSION:
            raise ValueError('unsupported conditional experiment policy')
        records = read(root/'records.json', [])
        comparisons = [SimulationComparisonDecision.from_mapping(c).to_mapping() for c in read(MARKET/'simulation_comparisons.jsonl', [])]
        current = {c['comparison_id']: c for c in comparisons}
        for row in records:
            verify(row)
            if row['policy_sha256'] != policy['record_sha256'] or row['comparison_sha256'] != canonical_hash(row['comparison']):
                raise ValueError('conditional evidence provenance mismatch')
            if current.get(row['comparison']['comparison_id']) != row['comparison']:
                raise ValueError('conditional evidence references missing or changed comparison')
        if len({r['comparison']['comparison_id'] for r in records}) != len(records):
            raise ValueError('duplicate conditional comparison')
        if not validate_only:
            if any(utc(c['comparison_issued_at_utc'])>=utc(policy['activated_at_utc']) and now<utc(c['event_start_utc']) for c in comparisons):
                raw_path = DATA/'processed/ufc_fights_reported_doubled.csv'
                from hashlib import sha256
                raw = pd.read_csv(raw_path, low_memory=False)
                records += build_records(comparisons, records, read(DATA/'external/simulation_forecasts.json', None), raw,
                    sha256(raw_path.read_bytes()).hexdigest(),
                    {r['decision_id']: r for r in read(MARKET/'paper_decisions.jsonl', [])},
                    {r['quote_id']: r for r in read(MARKET/'quote_snapshots.jsonl', [])},
                    {r['quote_id']: r for r in read(MARKET/'quote_source_metadata.jsonl', [])}, policy, now)
            write(root/'records.json', records)
        value = report(records, read(MARKET/'paper_settlements.jsonl', []), policy)
        eligible = {c['comparison_id'] for c in comparisons if utc(c['comparison_issued_at_utc'])>=utc(policy['activated_at_utc'])}
        value['coverage'] = {'comparisons_since_activation': len(eligible),
            'comparisons_without_indicators': len(eligible - {r['comparison']['comparison_id'] for r in records}),
            'legacy_comparisons_excluded': len(comparisons)-len(eligible)}
        if validate_only:
            if read(root/'report.json', None) != value:
                raise ValueError('conditional report does not reproduce')
        else:
            write(root/'report.json', value)
        print(f"Conditional simulation experiment: {len(records)} recorded comparisons; recommendations unchanged.")
        return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    update(validate_only=parser.parse_args().validate_only)
