"""Prospective method-of-victory paper recommendations; no execution API."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from fight_predictor.outcome_publication import OUTCOME_MODEL_VERSION
from market_tracker import MethodMarketStore, MethodForecastStore
from market_tracker._storage import atomic_write_text, exclusive_store_lock
from market_tracker.equal_stake_experiment import seal, verify
from market_tracker.paper import _profit_for_one_unit_risk

DATA = Path(__file__).resolve().parent / 'content/data'
ROOT = DATA / 'market/method_paper'
VERSION = 'prospective-method-flat-stake-v1'
METHODS = ('ko_tko', 'submission', 'decision')


def utc(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timezone required')
    return parsed.astimezone(timezone.utc)


def build_decisions(quotes, forecasts, existing, policy, now):
    """Freeze first eligible capture, including passes, once per matchup."""
    start = utc(policy['activated_at_utc'])
    frozen = {row['matchup_id'] for row in existing}
    by_forecast = {}
    for forecast in forecasts:
        key = (forecast.capture_id, forecast.matchup_id, forecast.horizon)
        if key in by_forecast:
            raise ValueError('duplicate method forecast')
        by_forecast[key] = forecast
    groups = defaultdict(list)
    for quote in quotes:
        observed = utc(quote.observed_at_utc)
        if observed >= start and 0 <= (now - observed).total_seconds() <= 1800:
            groups[(quote.observed_at_utc, quote.capture_id, quote.matchup_id, quote.horizon)].append(quote)
    pending = []
    for (_, capture, matchup, horizon), group in sorted(groups.items()):
        if matchup in frozen:
            continue
        forecast = by_forecast.get((capture, matchup, horizon))
        if forecast is None or forecast.model_version != OUTCOME_MODEL_VERSION:
            continue
        if not (utc(forecast.forecast_issued_at_utc) <= utc(group[0].observed_at_utc)
                and utc(forecast.event_start_utc) > now):
            continue
        offers = []
        for quote in sorted(group, key=lambda row: (row.book.casefold(), row.quote_id)):
            if (quote.fighter_id, quote.opponent_id, quote.event_id, quote.event_start_utc) != (
                    forecast.fighter_id, forecast.opponent_id, forecast.event_id, forecast.event_start_utc):
                raise ValueError('method quote and forecast identity or start mismatch')
            if quote.timing_precision != 'timestamp' or quote.market != 'fighter_method_of_victory':
                continue
            for side in ('fighter', 'opponent'):
                for method in METHODS:
                    price = getattr(quote, f'{side}_{method}_moneyline')
                    if price is None:
                        continue
                    probability = getattr(forecast, f'{side}_{method}_probability')
                    expected = probability * (1 + _profit_for_one_unit_risk(price)) - 1
                    offers.append({'book': quote.book, 'quote_id': quote.quote_id, 'side': side,
                        'fighter_id': getattr(quote, f'{side}_id'), 'method': method,
                        'selection': f"{getattr(quote, f'{side}_name')} by {method.replace('_', '/').upper()}",
                        'moneyline': price, 'probability': probability, 'expected_return': expected,
                        'observed_at_utc': quote.observed_at_utc,
                        'source_quote_updated_at_utc': None,
                        'source_payload_sha256': quote.source_payload_sha256})
        if not offers:
            continue
        offers.sort(key=lambda row: (-row['expected_return'], row['book'].casefold(), row['selection'], row['quote_id']))
        selection = offers[0] if offers[0]['expected_return'] >= policy['minimum_expected_return'] else None
        pending.append(seal({'policy_sha256': policy['record_sha256'], 'matchup_id': matchup,
            'event_id': forecast.event_id, 'event_date': forecast.event_date,
            'event_start_utc': forecast.event_start_utc, 'recorded_at_utc': now.isoformat(),
            'fighter_id': forecast.fighter_id, 'opponent_id': forecast.opponent_id,
            'fighter_name': forecast.fighter_name, 'opponent_name': forecast.opponent_name,
            'forecast': forecast.to_mapping(), 'offers': offers, 'selection': selection,
            'risk_units': 1 if selection else 0}))
        frozen.add(matchup)
    return pending


def settle(decision, sides):
    """Apply a declared paper convention, not a claim of bookmaker settlement."""
    if len(sides) != 2:
        return None
    if {row['fighter_id'] for row in sides} != {decision['fighter_id'], decision['opponent_id']}:
        return None
    methods = {str(row['method']).strip().upper() for row in sides}
    if len(methods) != 1:
        return None
    method = next(iter(methods))
    results = sorted(str(row['result']).strip().upper() for row in sides)
    selection = decision['selection']
    if results in (['D', 'D'], ['NC', 'NC']) or method in ('DQ', 'DISQUALIFICATION', 'CNC', 'OVERTURNED'):
        return {'status': 'void' if selection else 'pass', 'profit_units': 0, 'risk_units': 0}
    if results != ['L', 'W']:
        return None
    schedules = {str(row.get('time_format') or '').strip() for row in sides}
    from fight_semantics import scheduled_rounds_from_time_format
    if len(schedules) != 1:
        return None
    rounds = scheduled_rounds_from_time_format(next(iter(schedules)))
    if rounds is None:
        return None
    if rounds != decision['forecast']['scheduled_rounds']:
        return {'status': 'void' if selection else 'pass', 'profit_units': 0, 'risk_units': 0}
    category = {'KO/TKO': 'ko_tko', 'KO': 'ko_tko', 'TKO': 'ko_tko',
                'SUB': 'submission', 'SUBMISSION': 'submission',
                'U-DEC': 'decision', 'S-DEC': 'decision', 'M-DEC': 'decision',
                'DEC': 'decision', 'T-DEC': 'decision'}.get(method)
    if category is None:
        return None
    if selection is None:
        return {'status': 'pass', 'profit_units': 0, 'risk_units': 0}
    winner = next(row['fighter_id'] for row in sides if str(row['result']).strip().upper() == 'W')
    won = selection['fighter_id'] == winner and selection['method'] == category
    return {'status': 'win' if won else 'loss', 'risk_units': 1,
            'profit_units': _profit_for_one_unit_risk(selection['moneyline']) if won else -1}


def summarize(decisions, settlements, policy, now):
    settled = {row['matchup_id']: row for row in settlements}
    recommendations = []
    for row in decisions:
        if row['selection'] and row['matchup_id'] not in settled:
            offer = row['selection']
            recommendations.append({**offer, 'matchup_id': row['matchup_id'],
                'event_id': row['event_id'], 'event_date': row['event_date'],
                'event_start_utc': row['event_start_utc'], 'risk_units': 1,
                'decision_sha256': row['record_sha256'],
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
                raise ValueError('method paper experiment not initialized')
            policy = seal({'version': VERSION, 'activated_at_utc': now.isoformat(),
                'minimum_expected_return': .05, 'risk_units': 1, 'one_selection_per_fight': True,
                'freeze': 'first_eligible_capture_including_passes', 'execution_enabled': False,
                'quote_freshness_basis': 'collection_time_only_provider_update_unknown',
                'settlement_convention': 'standard_KO_SUB_DEC_draw_NC_DQ_schedule_change_void_v1',
                'bookmaker_specific_returns_verified': False,
                'special_results': 'unknown_or_ambiguous_results_remain_pending_review',
                'rule_reference': 'https://www.fanduel.com/fanduel-sportsbook-house-rules-nj'})
            write('policy.json', policy)
        verify(policy)
        if policy['version'] != VERSION or policy['execution_enabled'] is not False:
            raise ValueError('unsupported method paper policy')
        decisions = read(root / 'decisions.json', [])
        settlements = read(root / 'settlements.json', [])
        for records in (decisions, settlements):
            for row in records:
                verify(row)
            if len({row['matchup_id'] for row in records}) != len(records):
                raise ValueError('duplicate method paper matchup')
        indexed = {row['matchup_id']: row for row in decisions}
        if any(row['policy_sha256'] != policy['record_sha256'] for row in decisions):
            raise ValueError('method paper policy mismatch')
        if any(row['matchup_id'] not in indexed or row['decision_sha256'] != indexed[row['matchup_id']]['record_sha256'] for row in settlements):
            raise ValueError('method settlement references unknown or changed decision')
        if not validate_only:
            market = DATA / 'market'
            quotes = MethodMarketStore(market / 'method_market_snapshots.csv', market / 'method_market_snapshots.jsonl').read()
            forecasts = MethodForecastStore(market / 'method_forecast_captures.csv', market / 'method_forecast_captures.jsonl').read()
            decisions += build_decisions(quotes, forecasts, decisions, policy, now)
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
            raise ValueError('method paper report missing')
        report = summarize(decisions, settlements, policy, utc(published['generated_at_utc']) if published else now)
        if validate_only and published != report:
            raise ValueError('method paper report does not match its frozen records')
        if not validate_only:
            write('report.json', report)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    result = update(validate_only=parser.parse_args().validate_only)
    print(f"Method paper: {result['paper_recommendations']} recommendations, {result['settled_fights']} settled; execution disabled.")
