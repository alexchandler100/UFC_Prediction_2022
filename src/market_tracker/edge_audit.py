"""Read-only audit of frozen bets against declared final pre-card prices."""
from collections import Counter, defaultdict
from statistics import mean

from market_capture_windows import WINDOWS, quote_rejection, utc, window_for
from ._common import canonical_hash, implied_probability
from .equal_stake_experiment import choose


def normalize_picks(read, quote_sets):
    quotes = {q['quote_id']: q for rows in quote_sets.values() for q in rows}
    picks = []
    for strategy, prefix, market in [('Locked market winner', 'paper', 'moneyline'),
            ('Market-first winner', 'market_first_paper', 'moneyline'),
            ('Locked market totals', 'total_round_paper', 'total_rounds')]:
        results = {r['decision_id']: r for r in read(prefix + '_settlements.jsonl', [])}
        for d in read(prefix + '_decisions.jsonl', []):
            if d['paper_action'] == 'pass':
                continue
            q = quotes.get(d['reference_quote_id'], {})
            result = results.get(d['decision_id'], {})
            picks.append({**{k: d.get(k) for k in ('event_id', 'event_date', 'matchup_id',
                'fighter_id', 'opponent_id', 'event_start_utc')},
                'strategy': strategy, 'pick_id': d['decision_id'], 'market': market,
                'fighter_name': d.get('fighter_name') or q.get('fighter_name'),
                'opponent_name': d.get('opponent_name') or q.get('opponent_name'),
                'book': q.get('book') or d.get('reference_book') or d.get('target_book'),
                'side': d['paper_action'], 'line': d.get('line'),
                'selected_fighter_id': d.get(d['paper_action'] + '_id'),
                'probability': d['action_probability'], 'moneyline': d['action_reference_moneyline'],
                'issued_at_utc': d['decision_issued_at_utc'],
                'entry_quote_id': d['reference_quote_id'],
                'settlement_status': {'paper_win':'win','paper_loss':'loss'}.get(result.get('settlement_status'), result.get('settlement_status')),
                'profit_units': result.get('hypothetical_profit_units'),
                'risk_units': result.get('hypothetical_risk_units')})
    for strategy, folder, market in [('Independent model totals', 'total_model_paper', 'total_rounds'),
                                     ('Method experiment', 'method_paper', 'method')]:
        results = {r['matchup_id']: r for r in read(folder + '/settlements.json', [])}
        for d in read(folder + '/decisions.json', []):
            offer = d.get('selection')
            if not offer:
                continue
            result = results.get(d['matchup_id'], {})
            picks.append({**{k: d.get(k) for k in ('event_id', 'event_date', 'matchup_id',
                'fighter_id', 'opponent_id', 'fighter_name', 'opponent_name', 'event_start_utc')},
                'strategy': strategy, 'pick_id': d['record_sha256'], 'market': market,
                'book': offer['book'], 'side': offer['side'], 'line': offer.get('line'),
                'selection': offer['selection'], 'selected_fighter_id': offer.get('fighter_id'),
                'probability': offer['probability'], 'moneyline': offer['moneyline'],
                'issued_at_utc': d['recorded_at_utc'], 'entry_quote_id': offer['quote_id'],
                'settlement_status': result.get('status'), 'profit_units': result.get('profit_units'),
                'risk_units': result.get('risk_units')})
    # Preserve the equal-stake experiment's original rules and collection periods.
    results = {r['matchup_id']: r for r in read('equal_stake_experiment/settlements.json', [])}
    policy = read('equal_stake_experiment/policy.json', {})
    for d in read('equal_stake_experiment/decisions.json', []):
        for strategy in ('market', 'adjusted_market', 'production_model'):
            offer = choose(d, strategy, None, policy['minimum_expected_return'])
            if offer is None:
                continue
            result = results.get(d['matchup_id'])
            target = result.get('target') if result else None
            won = target == (1 if offer['side'] == 'fighter' else 0)
            price = offer['moneyline']
            profit = (price / 100 if price > 0 else 100 / -price) if won else -1
            picks.append({**{k: d.get(k) for k in ('event_id', 'event_date', 'matchup_id',
                'fighter_id', 'opponent_id', 'event_start_utc')},
                'fighter_name': d['forecast']['fighter_name'], 'opponent_name': d['forecast']['opponent_name'],
                'strategy': 'Equal stake: ' + strategy, 'pick_id': d['record_sha256'] + ':' + strategy,
                'collection_period': d.get('capture_contract_version', 'equal-stake-identical-starts-v1'),
                'market': 'moneyline', 'side': offer['side'], 'selected_fighter_id': d[offer['side'] + '_id'],
                'line': None, 'book': offer['book'], 'moneyline': price,
                'probability': offer['probabilities'][strategy], 'issued_at_utc': d['recorded_at_utc'],
                'entry_quote_id': offer['quote_id'],
                'settlement_status': None if result is None else 'void' if target is None else 'win' if won else 'loss',
                'risk_units': None if result is None else 0 if target is None else 1,
                'profit_units': None if result is None else 0 if target is None else profit})
    return picks


def summarize(rows):
    settled = [r for r in rows if r['settlement_status'] in ('win', 'loss', 'won', 'lost')
               and r.get('risk_units', 0) and r['risk_units'] > 0]
    profit = sum(r['profit_units'] for r in settled)
    risk = sum(r['risk_units'] for r in settled)
    largest = max(settled, key=lambda r: r['profit_units'], default=None)
    positive = largest if largest and largest['profit_units'] > 0 else None
    comparable = [r for r in rows if r.get('final_fair_expected_return') is not None]
    cards = defaultdict(float)
    for r in settled:
        cards[(r['event_date'], r['event_id'])] += r['profit_units']
    balance = peak = drawdown = 0
    for _, amount in sorted(cards.items()):
        balance += amount
        peak = max(peak, balance)
        drawdown = max(drawdown, peak - balance)
    return {'recorded_bets': len(rows), 'settled_bets': len(settled),
        'settled_cards': len(cards), 'pending_bets': sum(r['settlement_status'] is None for r in rows),
        'void_bets': sum(r['settlement_status'] == 'void' for r in rows),
        'profit_units': profit, 'risk_units': risk, 'return_per_unit': profit / risk if risk else None,
        'card_end_drawdown_units': drawdown,
        'largest_win_units': positive['profit_units'] if positive else None,
        'largest_win_pick_id': positive['pick_id'] if positive else None,
        'profit_without_largest_win_units': profit - (positive['profit_units'] if positive else 0),
        'roi_without_largest_win': ((profit-positive['profit_units'])/(risk-positive['risk_units'])
            if positive and risk > positive['risk_units'] else None),
        'final_reference_bets': len(comparable),
        'mean_claimed_expected_return': mean(r['claimed_expected_return'] for r in rows) if rows else None,
        'mean_claimed_expected_return_on_final_sample': mean(r['claimed_expected_return'] for r in comparable) if comparable else None,
        'mean_final_fair_expected_return': mean(r['final_fair_expected_return'] for r in comparable) if comparable else None,
        'mean_probability_overstatement': mean(r['probability']-r['final_fair_probability'] for r in comparable) if comparable else None,
        'final_status_counts': dict(sorted(Counter(r['final_status'] for r in rows).items()))}


def build_edge_audit(picks, quote_sets, metadata, now):
    """Use exact market identities and simultaneous other books; no interpolation."""
    sources = {r['quote_id']: r for r in metadata}
    groups = defaultdict(list)
    for market in ('moneyline', 'total_rounds'):
        for q in quote_sets.get(market, []):
            if market == 'total_rounds' and q.get('period') != 'full_fight':
                continue
            source = q if market == 'total_rounds' else sources.get(q['quote_id'])
            if quote_rejection(q, source, now) is not None:
                continue
            start = utc(q['event_start_utc'])
            if window_for((start-utc(q['observed_at_utc'])).total_seconds()) != 'final':
                continue
            groups[(q['event_id'], q['matchup_id'], market, q.get('line'), q['capture_id'])].append(q)
    records = []
    for p in picks:
        decimal = 1 / implied_probability(p['moneyline'])
        row = {**p, 'claimed_expected_return': p['probability'] * decimal - 1,
            'final_status': 'missing_final_same_book_quote', 'final_fair_probability': None,
            'final_fair_expected_return': None, 'same_book_probability_movement': None}
        if p['market'] == 'method':
            row['final_status'] = 'method_final_source_timing_unavailable'
        elif not p.get('event_start_utc'):
            row['final_status'] = 'missing_card_start'
        elif utc(p['event_start_utc']) > now:
            row['final_status'] = 'not_due'
        else:
            candidates = []
            for key, qs in groups.items():
                if key[:4] != (p['event_id'], p['matchup_id'], p['market'], p.get('line')):
                    continue
                for q in qs:
                    if (q['book'].casefold() == str(p['book']).casefold()
                            and set((q['fighter_id'],q['opponent_id'])) == set((p['fighter_id'],p['opponent_id']))
                            and utc(p['issued_at_utc']) < utc(q['observed_at_utc']) < utc(p['event_start_utc'])):
                        candidates.append((q, qs))
            if candidates:
                q, peers = max(candidates, key=lambda pair: (utc(pair[0]['observed_at_utc']),pair[0]['quote_id']))
                def side_values(quote):
                    if p['market'] == 'total_rounds':
                        probability = quote['no_vig_over_probability']
                        return (probability if p['side'] == 'over' else 1-probability, quote[p['side']+'_moneyline'])
                    selected = p['selected_fighter_id']
                    side = 'fighter' if quote['fighter_id'] == selected else 'opponent'
                    probability = quote['no_vig_fighter_probability']
                    return (probability if side == 'fighter' else 1-probability, quote[side+'_moneyline'])
                others = {}
                for peer in sorted(peers, key=lambda q:(utc(q['observed_at_utc']),q['quote_id'])):
                    if (peer['book'].casefold() != q['book'].casefold()
                            and peer['observed_at_utc'] == q['observed_at_utc']
                            and set((peer['fighter_id'],peer['opponent_id'])) == set((p['fighter_id'],p['opponent_id']))):
                        others[peer['book'].casefold()] = peer
                fair = mean(side_values(peer)[0] for peer in others.values()) if len(others) >= 3 else None
                row.update({'final_status': 'available' if fair is not None else 'fewer_than_three_other_books',
                    'final_quote_id': q['quote_id'], 'final_observed_at_utc': q['observed_at_utc'],
                    'final_source_updated_at_utc': (q if p['market']=='total_rounds' else sources[q['quote_id']])['source_quote_updated_at_utc'],
                    'final_moneyline': side_values(q)[1],
                    'other_book_quote_ids': sorted(peer['quote_id'] for peer in others.values()),
                    'other_books': sorted(others), 'final_fair_probability': fair,
                    'final_fair_expected_return': fair*decimal-1 if fair is not None else None,
                    'same_book_probability_movement': implied_probability(side_values(q)[1])-implied_probability(p['moneyline'])})
        row['odds_range'] = 'Favorite' if p['moneyline'] < 0 else 'Underdog +100–399' if p['moneyline'] < 400 else 'Longshot +400 or higher'
        lead = (utc(p['event_start_utc'])-utc(p['issued_at_utc'])).total_seconds() if p.get('event_start_utc') else None
        row['entry_window'] = window_for(lead) or 'outside_declared_windows' if lead is not None else 'unknown'
        records.append(row)
    records.sort(key=lambda r:(r['strategy'],r['event_date'],r['pick_id']))
    breakdown = []
    for field in ('strategy','book','odds_range','entry_window','collection_period'):
        grouped = defaultdict(list)
        for row in records:
            if field == 'collection_period' and field not in row:
                continue
            grouped[(row['strategy'], str(row.get(field, 'unknown')))].append(row)
        for (strategy, label), subset in sorted(grouped.items()):
            breakdown.append({'dimension':field,'strategy':strategy,'group':label,**summarize(subset)})
    return {'version': 'frozen-bet-edge-audit-v1', 'paper_only': True, 'execution_enabled': False,
        'reference_definition': 'Last fresh same-book observation 15–90 minutes before recorded card start, after entry; NOT exact bout closing odds. Fair probability averages at least three other simultaneous books with margin removed.',
        'interpretation': 'A final market estimate is a comparison, not the true win probability or proof of profit. Strategies overlap; never add their profits together. Missing prices remain missing.',
        'maximum_source_quote_age_seconds':1800, 'final_window_seconds':list(WINDOWS['final']),
        'records':records,'breakdowns':breakdown,'source_picks_sha256':canonical_hash(picks)}
