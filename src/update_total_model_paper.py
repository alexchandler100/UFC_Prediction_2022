"""Prospective model-only total-round paper recommendations; no execution API."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import pandas as pd

from fight_predictor.outcome_publication import OUTCOME_MODEL_VERSION
from market_tracker._storage import atomic_write_text, exclusive_store_lock
from market_tracker.equal_stake_experiment import seal, verify
from market_tracker.paper import _profit_for_one_unit_risk

DATA = Path(__file__).resolve().parent / 'content/data'
ROOT = DATA / 'market/total_model_paper'
VERSION = 'prospective-total-model-flat-stake-v1'


def utc(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timezone required')
    return parsed.astimezone(timezone.utc)


def build_decisions(publication, existing, policy, now, identity_matchups=()):
    """Freeze one best total per fight on the first fresh post-activation capture."""
    from market_tracker._common import canonical_hash
    if not publication:
        return []
    unhashed = dict(publication)
    supplied = unhashed.pop('publication_sha256', None)
    if supplied != canonical_hash(unhashed):
        raise ValueError('total paper source publication hash mismatch')
    observed = utc(publication['observed_at_utc'])
    if not (utc(policy['activated_at_utc']) <= observed <= now
            and (now - observed).total_seconds() <= 1800):
        return []
    frozen = {row['matchup_id'] for row in existing}
    groups = defaultdict(list)
    for offer in publication.get('prop_markets', {}).get('total_rounds', {}).get('candidate_offers', []):
        if offer['matchup_id'] in frozen:
            continue
        if (offer.get('model_version') != OUTCOME_MODEL_VERSION
                or offer.get('schedule_contract_version') != 'verified-pre-fight-schedule-v1'
                or offer.get('side') not in ('over', 'under')
                or offer.get('observed_at_utc') != publication['observed_at_utc']
                or offer.get('capture_id') != publication['capture_id']):
            continue
        if not (utc(offer['forecast_issued_at_utc']) <= observed < utc(offer['event_start_utc'])
                and now < utc(offer['event_start_utc'])
                and 0 <= (observed - utc(offer['source_quote_updated_at_utc'])).total_seconds() <= 1800):
            continue
        probability, price, line = float(offer['model_probability']), float(offer['offered_moneyline']), float(offer['line'])
        if not (math.isfinite(probability) and 0 < probability < 1 and math.isfinite(price)
                and price.is_integer() and abs(price) >= 100 and math.isfinite(line)
                and line % 1 == .5 and 0 < line < offer['scheduled_rounds']):
            raise ValueError('invalid total paper offer')
        groups[offer['matchup_id']].append({**offer, 'book': offer['target_book'],
            'moneyline': int(price), 'probability': probability,
            'expected_return': probability * (1 + _profit_for_one_unit_risk(int(price))) - 1})
    pending = []
    for matchup, offers in sorted(groups.items()):
        identities = {(r['fighter_name'], r['opponent_name'], r['scheduled_rounds'], r['event_start_utc']) for r in offers}
        if len(identities) != 1:
            raise ValueError('total paper offers disagree on fight identity or schedule')
        offers.sort(key=lambda r: (-r['expected_return'], r['book'].casefold(), r['selection'], r['quote_id']))
        top = offers[0]
        market = next((r for r in [*publication.get('matchups', []), *identity_matchups]
                       if r['matchup_id'] == matchup), None)
        if not market:
            continue
        if (market['fighter_name'], market['opponent_name']) != (top['fighter_name'], top['opponent_name']):
            raise ValueError('total paper source fight identity mismatch')
        selection = top if top['expected_return'] >= policy['minimum_expected_return'] else None
        pending.append(seal({'policy_sha256': policy['record_sha256'], 'matchup_id': matchup,
            'event_id': publication['event_id'], 'event_date': publication['event_date'],
            'event_start_utc': top['event_start_utc'], 'recorded_at_utc': now.isoformat(),
            'fighter_id': market['fighter_id'], 'opponent_id': market['opponent_id'],
            'fighter_name': top['fighter_name'], 'opponent_name': top['opponent_name'],
            'forecast': {'scheduled_rounds': top['scheduled_rounds']}, 'offers': offers,
            'source_publication_sha256': supplied, 'selection': selection,
            'risk_units': 1 if selection else 0}))
    return pending


def settle(decision, sides):
    from fight_semantics import scheduled_rounds_from_time_format
    if len(sides) != 2 or {r['fighter_id'] for r in sides} != {decision['fighter_id'], decision['opponent_id']}:
        return None
    selection = decision['selection']
    def outcome(status, profit=0, risk=0):
        return {'status': status if selection else 'pass', 'profit_units': profit if selection else 0,
                'risk_units': risk if selection else 0}
    results = sorted(str(r['result']).strip().upper() for r in sides)
    methods = {str(r['method']).strip().upper() for r in sides}
    if len(methods) != 1:
        return None
    method = next(iter(methods))
    if results in (['D', 'D'], ['NC', 'NC']) or method in ('DQ', 'DISQUALIFICATION', 'CNC', 'OVERTURNED'):
        return outcome('void')
    if results != ['L', 'W'] or method not in ('KO/TKO','KO','TKO','SUB','SUBMISSION','U-DEC','S-DEC','M-DEC','DEC'):
        return None
    schedules = {str(r.get('time_format') or '').strip() for r in sides}
    if len(schedules) != 1:
        return None
    rounds = scheduled_rounds_from_time_format(next(iter(schedules)))
    if rounds is None:
        return None
    if rounds != decision['forecast']['scheduled_rounds']:
        return outcome('void')
    try:
        durations = [float(r['total_fight_time']) for r in sides]
    except (ValueError, TypeError, KeyError):
        return None
    if any(not math.isfinite(d) or not 0 < d <= rounds * 300 for d in durations) or abs(durations[0]-durations[1]) > 1e-9:
        return None
    if selection is None:
        return outcome('pass')
    boundary = selection['line'] * 300
    if durations[0] == boundary:
        return outcome('void')
    won = (durations[0] > boundary) if selection['side'] == 'over' else (durations[0] < boundary)
    return outcome('win' if won else 'loss', _profit_for_one_unit_risk(selection['moneyline']) if won else -1, 1)


def summarize(decisions, settlements, policy, now):
    settled = {row['matchup_id']: row for row in settlements}
    recommendations = []
    for row in decisions:
        if row['selection']:
            offer = row['selection']
            recommendations.append({**offer, 'matchup_id': row['matchup_id'],
                'matchup_fighter_id': row['fighter_id'], 'matchup_opponent_id': row['opponent_id'],
                'fighter_name': row['fighter_name'], 'opponent_name': row['opponent_name'],
                'event_id': row['event_id'], 'event_date': row['event_date'],
                'event_start_utc': row['event_start_utc'], 'risk_units': 1,
                'decision_sha256': row['record_sha256'],
                'settlement_status': settled.get(row['matchup_id'], {}).get('status'),
                'profit_units': settled.get(row['matchup_id'], {}).get('profit_units'),
                'status': 'awaiting_result' if utc(row['event_start_utc']) <= now else
                    ('recently_collected' if (now - utc(offer['observed_at_utc'])).total_seconds() <= 1800 else 'price_expired')})
    recommendations.sort(key=lambda row: (-row['expected_return'], row['matchup_id']))
    risk = sum(row['risk_units'] for row in settlements)
    profit = sum(row['profit_units'] for row in settlements)
    cards = defaultdict(float)
    for row in settlements:
        cards[row['event_id']] += row['profit_units']
    return {'policy': policy, 'generated_at_utc': now.isoformat(), 'paper_only': True,
        'execution_enabled': False, 'frozen_fights': len(decisions),
        'paper_recommendations': sum(row['selection'] is not None for row in decisions),
        'settled_fights': len(settlements), 'settled_risk_units': risk, 'profit_units': profit,
        'return_per_unit': profit / risk if risk else None, 'card_profits': dict(cards),
        'no_bet_baseline_profit_units': 0,
        'recommendations': recommendations, 'settlements': settlements,
        'decisions_sha256': sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()}


def update(*, validate_only=False, root=ROOT):
    now = datetime.now(timezone.utc)
    read = lambda path, default: json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
    def write(name, value):
        atomic_write_text(root / name, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with exclusive_store_lock(root / 'write.lock'):
        policy = read(root / 'policy.json', None)
        if policy is None:
            if validate_only:
                raise ValueError('total model paper experiment not initialized')
            policy = seal({'version': VERSION, 'activated_at_utc': now.isoformat(),
                'minimum_expected_return': .05, 'risk_units': 1, 'one_selection_per_fight': True,
                'freeze': 'first_eligible_capture_including_passes', 'execution_enabled': False,
                'quote_freshness_basis': 'provider_update_within_30_minutes',
                'settlement_convention': 'active_seconds_half_round_draw_NC_DQ_schedule_change_exact_boundary_void_v1',
                'bookmaker_specific_returns_verified': False,
                'special_results': 'unknown_or_ambiguous_results_remain_pending_review',
                'probability_source': 'independent_duration_model', 'historical_backfill': False})
            write('policy.json', policy)
        verify(policy)
        if (policy['version'] != VERSION or policy['execution_enabled'] is not False
                or policy['minimum_expected_return'] != .05 or policy['risk_units'] != 1
                or policy['one_selection_per_fight'] is not True or policy['historical_backfill'] is not False):
            raise ValueError('unsupported total model paper policy')
        decisions = read(root / 'decisions.json', [])
        settlements = read(root / 'settlements.json', [])
        for records in (decisions, settlements):
            for row in records:
                verify(row)
            if len({row['matchup_id'] for row in records}) != len(records):
                raise ValueError('duplicate total model paper matchup')
        for row in decisions:
            observed = utc(row['offers'][0]['observed_at_utc'])
            if not (utc(policy['activated_at_utc']) <= observed <= utc(row['recorded_at_utc']) < utc(row['event_start_utc'])):
                raise ValueError('total model decision is outside its prospective window')
            ordered = sorted(row['offers'], key=lambda r: (-r['expected_return'], r['book'].casefold(), r['selection'], r['quote_id']))
            expected_selection = ordered[0] if ordered[0]['expected_return'] >= .05 else None
            if row['offers'] != ordered or row['selection'] != expected_selection or row['risk_units'] != (1 if expected_selection else 0):
                raise ValueError('total model decision differs from frozen policy')
        indexed = {row['matchup_id']: row for row in decisions}
        if any(row['policy_sha256'] != policy['record_sha256'] for row in decisions):
            raise ValueError('total model paper policy mismatch')
        if any(row['matchup_id'] not in indexed or row['decision_sha256'] != indexed[row['matchup_id']]['record_sha256'] for row in settlements):
            raise ValueError('total model settlement references unknown or changed decision')
        if not validate_only:
            market = DATA / 'market'
            publication = read(market / 'current_opportunities.json', None)
            identities = read(DATA / 'external/outcome_forecasts.json', {}).get('matchups', [])
            decisions += build_decisions(publication, decisions, policy, now, identities)
            write('decisions.json', decisions)
            settled_ids = {row['matchup_id'] for row in settlements}
            unresolved = [row for row in decisions if row['matchup_id'] not in settled_ids and utc(row['event_start_utc']) < now]
            if unresolved:
                path = DATA / 'processed/ufc_fights_reported_doubled.csv'
                raw = pd.read_csv(path, low_memory=False).fillna('')
                for name in ('fighter', 'event', 'fight'):
                    raw[f'{name}_id'] = raw[f'{name}_url'].astype(str).str.rstrip('/').str.rsplit('/').str[-1]
                source_hash = sha256(path.read_bytes()).hexdigest()
                for decision in unresolved:
                    pair = raw.loc[raw.event_id.eq(decision['event_id']) & raw.fighter_id.isin([decision['fighter_id'], decision['opponent_id']])]
                    if len(pair) != 2 or pair.fight_id.nunique() != 1:
                        continue
                    outcome = settle(decision, pair.to_dict('records'))
                    if outcome is not None:
                        settlements.append(seal({**outcome, 'matchup_id': decision['matchup_id'],
                            'event_id': decision['event_id'], 'fight_id': pair.iloc[0].fight_id,
                            'decision_sha256': decision['record_sha256'], 'settled_at_utc': now.isoformat(),
                            'result_source_sha256': source_hash}))
            write('settlements.json', settlements)
        published = read(root / 'report.json', None) if validate_only else None
        if validate_only and published is None:
            raise ValueError('total model paper report missing')
        report = summarize(decisions, settlements, policy, utc(published['generated_at_utc']) if published else now)
        if validate_only and published != report:
            raise ValueError('total model paper report does not match its frozen records')
        if not validate_only:
            write('report.json', report)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    result = update(validate_only=parser.parse_args().validate_only)
    print(f"Total model paper: {result['paper_recommendations']} recommendations, {result['settled_fights']} settled; execution disabled.")
