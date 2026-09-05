"""Explain stored betting candidates without changing recommendations or ledgers."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from market_tracker._common import canonical_hash
from market_tracker._storage import atomic_write_text
from market_tracker.bayesian_kelly import BayesianKellyCalibrator
from bayesian_total_calibration import BayesianTotalCalibrator
from upcoming_bet_board import _current_opportunity_bets

DATA = Path(__file__).resolve().parent / 'content/data'
OUTPUT = DATA / 'market/candidate_report.json'
REASONS = {
    'no_prices': 'No matched bookmaker prices saved',
    'no_model': 'Model probability unavailable',
    'no_consensus': 'Fewer than three other fresh books at capture',
    'no_calibration': 'Probability adjustment unavailable',
    'below_threshold': 'Adjusted expected return below 5%',
    'uncertainty': 'Conservative probability does not support a positive return',
    'zero_stake': 'Conservative stake is zero',
    'expired': 'Price older than 30 minutes',
    'missing_time': 'Quote update or event start missing',
    'future_quote': 'Quote update is in the future',
    'event_started': 'Event has started',
    'totals_research': 'Totals excluded from conservative stakes pending betting evidence',
    'portfolio': 'Not selected in the saved capped portfolio',
}


def moment(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, TypeError):
        return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def ev(probability, price):
    if probability is None or price is None or abs(price) < 100:
        return None
    decimal = 1 + (price / 100 if price > 0 else 100 / -price)
    return probability * decimal - 1


def funding_reasons(row, assessment, at):
    reasons = []
    if row['market'] == 'Total rounds':
        reasons.append('totals_research')
    if row['raw_market_probability'] is None and row['market'] == 'Moneyline':
        reasons.append('no_consensus')
    if row['model_probability'] is None:
        reasons.append('no_model')
    if assessment is None or assessment.get('status') != 'available':
        reasons.append('no_calibration')
    else:
        if row['adjusted_ev'] < .05:
            reasons.append('below_threshold')
        if row['conservative_ev'] <= 0:
            reasons.append('uncertainty')
        if assessment['recommended_fraction'] <= 0:
            reasons.append('zero_stake')
    updated, start = moment(row.get('quote_updated_at_utc')), moment(row.get('event_start_utc'))
    if updated is None or start is None:
        reasons.append('missing_time')
    if updated:
        age = (at - updated).total_seconds()
        if age < 0:
            reasons.append('future_quote')
        elif age > 1800:
            reasons.append('expired')
    if start and start <= at:
        reasons.append('event_started')
    return reasons


def build_report(forecasts, board, current, calibrator, *, generated_at, total_calibrator=None):
    """Reproduce capture-time moneyline inputs, with explicit present-day expiry."""
    captured = moment(board['observed_at_utc'])
    if captured is None:
        raise ValueError('board capture timestamp missing')
    at = moment(generated_at)
    if at is None:
        raise ValueError('report timestamp missing')
    if board.get('forecast_publication_sha256') != forecasts['publication_sha256']:
        raise ValueError('board and forecasts do not match')
    markets = {row['matchup_id']: row for row in board.get('market_matchups', [])}
    current_by_id = {row['matchup_id']: row for row in current.get('matchups', [])}
    funded = {(row['matchup_id'], row['category'], row['target_book'], row['side'], row.get('line'))
              for row in board.get('bets', [])}
    rows = []

    def finish(row, assessment):
        row['adjusted_probability'] = (assessment.get('posterior_mean_probability')
                                        if assessment and assessment.get('status') == 'available' else None)
        row['conservative_probability'] = (assessment.get('posterior_lower_probability')
                                            if assessment and assessment.get('status') == 'available' else None)
        for field, probability in [('raw_market_ev', 'raw_market_probability'), ('model_ev', 'model_probability'),
                                    ('adjusted_ev', 'adjusted_probability'), ('conservative_ev', 'conservative_probability')]:
            row[field] = ev(row[probability], row['moneyline'])
        row['reasons_at_capture'] = funding_reasons(row, assessment, captured)
        row['reasons_now'] = funding_reasons(row, assessment, at)
        row['funded_in_saved_board'] = (row['matchup_id'], row['market'], row['book'], row['side'], row.get('line')) in funded
        if not row['reasons_at_capture'] and not row['funded_in_saved_board']:
            row['reasons_at_capture'].append('portfolio')
            row['reasons_now'].append('portfolio')
        row['row_id'] = canonical_hash(row)
        rows.append(row)

    for matchup in forecasts['matchups']:
        base = {key: matchup.get(key) for key in ('matchup_id', 'event_id', 'event_date', 'event_title',
                                                  'fighter_name', 'opponent_name', 'model_id', 'forecast_issued_at_utc')}
        stored = markets.get(matchup['matchup_id'], {})
        quotes = stored.get('book_quotes', [])
        start = stored.get('event_start_utc') or current_by_id.get(matchup['matchup_id'], {}).get('event_start_utc')
        if not start and matchup.get('event_id') == current.get('event_id'):
            start = current.get('event_start_utc')
        if not quotes:
            rows.append({**base, 'market': 'Moneyline', 'book': None, 'selection': None,
                         'model_probability': matchup.get('model_probability_for_fighter'),
                         'reasons_at_capture': ['no_prices'], 'reasons_now': ['no_prices'],
                         'funded_in_saved_board': False, 'row_id': canonical_hash(base)})
            continue
        for quote in quotes:
            others = [q for q in quotes if q['book_key'] != quote['book_key'] and
                      moment(q.get('source_quote_updated_at_utc')) is not None and
                      -300 <= (captured - moment(q['source_quote_updated_at_utc'])).total_seconds() <= 1800]
            raw = sum(q['no_vig_fighter_probability'] for q in others) / len(others) if len(others) >= 3 else None
            independent = number(matchup.get('model_probability_for_fighter'))
            for side in ('fighter', 'opponent'):
                p = raw if side == 'fighter' or raw is None else 1 - raw
                model = independent if side == 'fighter' or independent is None else 1 - independent
                price = number(quote[f'{side}_moneyline'])
                assessment = calibrator.assessment(p, price) if p is not None else None
                finish({**base, 'market': 'Moneyline', 'book': quote['book'], 'side': side,
                        'selection': matchup[f'{side}_name'], 'moneyline': price, 'line': None,
                        'raw_market_probability': p, 'model_probability': model,
                        'event_start_utc': start, 'quote_updated_at_utc': quote.get('source_quote_updated_at_utc'),
                        'quote_first_observed_at_utc': quote.get('first_observed_at_utc'),
                        'consensus_books': [q['book'] for q in others]}, assessment)
    indexed = {row['matchup_id']: row for row in forecasts['matchups']}
    for candidate in _current_opportunity_bets(current, indexed, calibrator, total_calibrator):
        if candidate['category'] != 'Total rounds':
            continue
        finish({**{key: candidate.get(key) for key in ('matchup_id', 'event_id', 'event_date', 'event_title',
                                                       'fighter_name', 'opponent_name', 'model_id', 'forecast_issued_at_utc')},
                'market': 'Total rounds', 'book': candidate['target_book'], 'side': candidate['side'],
                'selection': candidate['selection'], 'moneyline': candidate['offered_moneyline'],
                'line': candidate['line'], 'raw_market_probability': None,
                'model_probability': candidate['estimated_win_probability'],
                'quote_updated_at_utc': candidate.get('source_quote_updated_at_utc'),
                'event_start_utc': candidate.get('event_start_utc')}, candidate.get('bayesian_kelly'))
    recorded_totals = {(row['matchup_id'], row['book'], row.get('side'), row.get('line'))
                       for row in rows if row['market'] == 'Total rounds'}
    totals_view = current.get('prop_markets', {}).get('total_rounds', {})
    for market in totals_view.get('markets', []):
        matchup = indexed.get(market['matchup_id'])
        if not matchup:
            continue
        for quote in market.get('book_quotes', []):
            for side in ('over', 'under'):
                key = (market['matchup_id'], quote['book'], side, market['line'])
                if key in recorded_totals:
                    continue
                finish({**{key: matchup.get(key) for key in ('matchup_id', 'event_id', 'event_date',
                        'event_title', 'fighter_name', 'opponent_name')},
                        'market': 'Total rounds', 'book': quote['book'], 'side': side,
                        'selection': f"{side.title()} {market['line']:g} rounds", 'line': market['line'],
                        'moneyline': quote[f'{side}_moneyline'], 'raw_market_probability': None,
                        'model_probability': None, 'forecast_issued_at_utc': None, 'model_id': None,
                        'forecast_unavailable_reason': market.get('forecast_unavailable_reason'),
                        'quote_updated_at_utc': quote.get('source_quote_updated_at_utc'),
                        'event_start_utc': quote.get('event_start_utc') or current.get('event_start_utc')}, None)
    summary = {label: sum(row.get(field) is not None and row[field] >= .05 for row in rows)
               for label, field in [('raw_market_above_5pct', 'raw_market_ev'),
                                    ('adjusted_above_5pct', 'adjusted_ev'), ('model_above_5pct', 'model_ev')]}
    return {'schema_version': 1, 'generated_at_utc': at.isoformat(), 'captured_at_utc': board['observed_at_utc'],
            'paper_only': True, 'execution_enabled': False, 'retrospective_diagnostic_only': True,
            'reason_labels': REASONS, 'summary': summary, 'rows': rows,
            'totals_coverage': {key: totals_view.get(key, 0) for key in ('quote_count', 'forecast_count')},
            'reason_counts_now': dict(Counter(reason for row in rows for reason in row['reasons_now'])),
            'inputs': {'forecasts': forecasts['publication_sha256'], 'board': board['publication_sha256'],
                       'opportunities': current.get('publication_sha256'),
                       'calibration': calibrator.artifact['artifact_sha256']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    existing = read(OUTPUT) if args.validate_only else None
    try:
        totals = BayesianTotalCalibrator.load()
    except (OSError, ValueError):
        totals = None
    result = build_report(read(DATA / 'external/all_upcoming_forecasts.json'),
        read(DATA / 'market/upcoming_bet_board.json'), read(DATA / 'market/current_opportunities.json'),
        BayesianKellyCalibrator.load(), total_calibrator=totals,
        generated_at=existing['generated_at_utc'] if existing else datetime.now(timezone.utc).isoformat())
    if args.validate_only:
        if result != existing:
            raise ValueError('candidate report cannot be reproduced from current inputs')
    else:
        atomic_write_text(OUTPUT, json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'rows': len(result['rows']), **result['summary'], 'reasons': result['reason_counts_now']}))


if __name__ == '__main__':
    main()
