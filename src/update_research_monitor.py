"""Publish experiment health and exact-selection odds histories; offline only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from market_tracker._storage import atomic_write_text

ROOT = Path(__file__).resolve().parent / 'content/data/market'
QUOTES = {'moneyline': 'quote_snapshots.jsonl', 'total_rounds': 'total_round_quote_snapshots.jsonl',
          'method': 'method_market_snapshots.jsonl'}


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timestamp requires timezone')
    return parsed.astimezone(timezone.utc)


def read(root, name, default=None):
    path = root / name
    if not path.exists():
        return default
    if name.endswith('.jsonl'):
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return json.loads(path.read_text(encoding='utf-8'))


def histories(quote_sets, metadata):
    """Keep exact fighter/method/line identity; never average or interpolate prices."""
    source_times = {row['quote_id']: row.get('source_quote_updated_at_utc') for row in metadata}
    groups = {}
    for market, quotes in quote_sets.items():
        for q in quotes:
            selections = []
            if market == 'total_rounds':
                if q.get('period') != 'full_fight':
                    continue
                selections = [(side, f"{side.title()} {q['line']:g} rounds", q.get(f'{side}_moneyline')) for side in ('over', 'under')]
            else:
                for side in ('fighter', 'opponent'):
                    for method in (('ko_tko', 'submission', 'decision') if market == 'method' else ('',)):
                        key = f'{side}_{method}_moneyline' if method else f'{side}_moneyline'
                        label = q.get(f'{side}_name') or q[f'{side}_id']
                        if method:
                            label += ' by ' + method.replace('_', '/').upper()
                        selections.append((q[f'{side}_id'] + (':' + method if method else ''), label, q.get(key)))
            for selection_id, label, price in selections:
                if price is None:
                    continue
                line = q.get('line') if market == 'total_rounds' else None
                key = '|'.join((q['matchup_id'], market, str(line), selection_id))
                group = groups.setdefault(key, {'key': key, 'matchup_id': q['matchup_id'], 'event_id': q['event_id'],
                    'fighter_id': q['fighter_id'], 'opponent_id': q['opponent_id'], 'market': market,
                    'line': line, 'selection_id': selection_id, 'label': label, 'points': {}})
                point = {'book': q['book'], 'observed_at_utc': q['observed_at_utc'], 'moneyline': price,
                    'source_quote_updated_at_utc': q.get('source_quote_updated_at_utc') or source_times.get(q['quote_id'])}
                # Method horizons can contain the same observation twice.
                group['points'][(q['book'], q['observed_at_utc'], price)] = point
    result = []
    for _, group in sorted(groups.items()):
        group['points'] = sorted(group['points'].values(), key=lambda p: (utc(p['observed_at_utc']), p['book'], p['moneyline']))
        result.append(group)
    return {'version': 1, 'time_basis': 'collection_time', 'series': result}


def experiment(name, decisions, settlements, now, key='decision_id'):
    settled = {row[key] for row in settlements}
    pending = [row for row in decisions if row[key] not in settled]
    return {'name': name, 'recorded': len(decisions), 'settled': len(settlements),
        'recorded_last_7_days': sum(now - timedelta(days=7) <= utc(row.get('recorded_at_utc') or row['decision_issued_at_utc']) <= now for row in decisions),
        'recommendations': sum(row.get('selection') is not None if 'selection' in row else row.get('hypothetical_risk_units', 0) > 0 for row in decisions),
        'pending': len(pending),
        'overdue_review': sum(bool(row.get('event_start_utc')) and utc(row['event_start_utc']) < now - timedelta(days=2) for row in pending),
        'risk_units': sum(row.get('risk_units', row.get('hypothetical_risk_units', 0)) for row in settlements),
        'profit_units': sum(row.get('profit_units', row.get('hypothetical_profit_units', 0)) for row in settlements)}


def build(root, now):
    quote_sets = {market: read(root, filename, []) for market, filename in QUOTES.items()}
    feeds = []
    for market, rows in quote_sets.items():
        recent = [q for q in rows if now - timedelta(days=7) <= utc(q['observed_at_utc']) <= now]
        feeds.append({'market': market, 'quotes': len(rows), 'captures_last_7_days': len({q['capture_id'] for q in recent}),
            'ledger_available': (root / QUOTES[market]).exists(),
            'fights_last_7_days': len({q['matchup_id'] for q in recent}),
            'last_collected_at_utc': max((q['observed_at_utc'] for q in rows), key=utc, default=None),
            'cadence': 'opening / about 72h, 24h, 6h' if market == 'method' else 'scheduled throughout the week'})
    experiments = [experiment(label, read(root, prefix + '_decisions.jsonl', []),
        read(root, prefix + '_settlements.jsonl', []), now) for label, prefix in
        [('Winner locked paper', 'paper'), ('Totals locked paper', 'total_round_paper'), ('Market-first paper', 'market_first_paper')]]
    experiments.append(experiment('Method paper', read(root, 'method_paper/decisions.json', []),
        read(root, 'method_paper/settlements.json', []), now, 'matchup_id'))
    equal = read(root, 'equal_stake_experiment/report.json', {})
    report = {'version': 1, 'generated_at_utc': now.isoformat(), 'feeds': feeds, 'experiments': experiments,
        'equal_stake': {'recorded': equal.get('frozen_fights', 0), 'settled': equal.get('settled_fights', 0),
            'review_fights': 200, 'review_cards': 20,
            'results': equal.get('results', [])},
        'workflow_runs_url': 'https://github.com/alexchandler100/UFC_Prediction_2022/actions',
        'workflow_failure_status': 'unavailable_from_ledgers',
        'note': 'Zero recommendations can be a valid pass. Missing collections and overdue results require review. Experiments are separate; do not add their profits together.'}
    return report, histories(quote_sets, read(root, 'quote_source_metadata.jsonl', []))


def update(root=ROOT, validate_only=False):
    saved = read(root, 'research_monitor.json')
    if validate_only and saved is None:
        raise ValueError('research monitor missing')
    now = utc(saved['generated_at_utc']) if validate_only else datetime.now(timezone.utc)
    report, history = build(root, now)
    for filename, value in [('research_monitor.json', report), ('bet_odds_history.json', history)]:
        if validate_only:
            if read(root, filename) != value:
                raise ValueError(f'{filename} does not reproduce from ledgers')
        else:
            atomic_write_text(root / filename, json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    print(f"Research monitor: {len(history['series'])} selection histories; "
          f"{sum(row['overdue_review'] for row in report['experiments'])} decisions overdue for review.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    update(validate_only=parser.parse_args().validate_only)
