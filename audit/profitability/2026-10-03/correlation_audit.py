"""Offline audit of one-pick rules; never edits forecasts, picks or live policies.

Alternative allocations are retrospective diagnostics using frozen offers and
the same one-unit budget per fight. They are not new prospective track records.
Run from the repository root with .venv/Scripts/python.exe <this file>.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from update_method_paper import settle
from market_tracker.equal_stake_experiment import verify

DATA = ROOT / 'src/content/data'
STATES = tuple(f'{side}_{method}' for side in ('fighter', 'opponent')
               for method in ('ko_tko', 'submission', 'decision', 'other'))
BUDGET = .01


def load(relative):
    return json.loads((DATA / relative).read_text(encoding='utf-8'))


def decimal(odds):
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def identity(row):
    return (row['event_id'], *sorted((row['fighter_id'], row['opponent_id'])))


def best_unique_offers(decision):
    best = {}
    for offer in decision['offers']:
        key = (offer['fighter_id'], offer['method'])
        if key not in best or decimal(offer['moneyline']) > decimal(best[key]['moneyline']):
            best[key] = offer
    return [best[key] for key in sorted(best)]


def return_matrix(offers, other_void=True):
    returns = np.full((len(STATES), len(offers)), -1.0)
    if other_void:
        for i, state in enumerate(STATES):
            if state.endswith('_other'):
                returns[i, :] = 0
    for j, offer in enumerate(offers):
        returns[STATES.index(f"{offer['side']}_{offer['method']}"), j] = decimal(offer['moneyline']) - 1
    return returns


def joint_allocation(probabilities, returns, budget=BUDGET):
    """Optimize stakes together under the frozen, uncalibrated method model.

This isolated diagnostic deliberately makes no claim that those probabilities
are accurate or that historical quotes were executable. Residual 'other' states
are stress-tested both as refunds and as losses, not silently renormalized away.
"""
    count = returns.shape[1]
    if not count:
        return np.array([])
    def value(weights):
        wealth = 1 + budget * returns @ weights
        return -float(probabilities @ np.log(wealth)) / budget
    def gradient(weights):
        return -(returns.T @ (probabilities / (1 + budget * returns @ weights)))
    result = minimize(value, np.zeros(count), jac=gradient, method='SLSQP',
        bounds=[(0, 1)] * count,
        constraints=[{'type': 'ineq', 'fun': lambda w: 1 - w.sum(), 'jac': lambda w: -np.ones(count)}],
        options={'ftol': 1e-12, 'maxiter': 500})
    if not result.success:
        raise ValueError(result.message)
    weights = np.maximum(result.x, 0)
    assert weights.sum() <= 1 + 1e-8
    # First-order optimality for the simplex, including its cash vertex.
    growth_gradient = -gradient(weights)
    gap = max(0.0, float(growth_gradient.max())) - float(weights @ growth_gradient)
    assert gap < 1e-5, gap
    return weights / max(1.0, weights.sum())


def main():
    inputs = ['market/method_paper/decisions.json', 'market/method_paper/settlements.json',
        'market/method_paper/report.json', 'market/method_paper/policy.json',
        'market/upcoming_bet_board.json', 'market/bet_performance.json',
        'processed/ufc_fights_reported_doubled.csv', 'external/outcome_forecasts.json']
    hashes = {name: sha256((DATA / name).read_bytes()).hexdigest() for name in inputs}
    decisions = load(inputs[0]); settlements = load(inputs[1]); report = load(inputs[2])
    for record in decisions + settlements:
        verify(record)
    settled_by_id = {row['matchup_id']: row for row in settlements}
    raw = pd.read_csv(DATA / inputs[6], low_memory=False).fillna('')
    for field in ('fighter', 'event', 'fight'):
        raw[f'{field}_id'] = raw[f'{field}_url'].str.rstrip('/').str.rsplit('/').str[-1]
    strategies = defaultdict(list)
    details = []
    for decision in decisions:
        offers = best_unique_offers(decision)
        probabilities = np.array([decision['forecast'][f'{state}_probability'] for state in STATES])
        assert abs(probabilities.sum() - 1) < 1e-8
        for offer in offers:
            assert abs(offer['probability'] - probabilities[STATES.index(f"{offer['side']}_{offer['method']}")]) < 1e-8
        chosen = decision['selection']
        qualifying = [i for i, offer in enumerate(offers) if offer['expected_return'] >= .05]
        allocations = {}
        allocations['frozen_max_ev'] = np.array([float(chosen is not None and
            (offer['fighter_id'], offer['method']) == (chosen['fighter_id'], chosen['method'])) for offer in offers])
        allocations['equal_split_positive_ev'] = np.array([1 / len(qualifying) if i in qualifying else 0 for i in range(len(offers))])
        matrix = return_matrix(offers)
        growth = [float(probabilities @ np.log(1 + BUDGET * matrix[:, i])) for i in range(len(offers))]
        best = max(qualifying, key=lambda i: growth[i]) if qualifying else None
        allocations['best_single_log_growth'] = np.array([float(i == best) for i in range(len(offers))])
        allocations['joint_log_growth'] = joint_allocation(probabilities, matrix)
        allocations['joint_other_as_loss'] = joint_allocation(probabilities, return_matrix(offers, False))
        settlement = settled_by_id.get(decision['matchup_id'])
        outcome = None
        if settlement:
            sides = raw.loc[raw.event_id.eq(decision['event_id']) & raw.fight_id.eq(settlement['fight_id'])]
            assert len(sides) == 2
            actual = settle(decision, sides.to_dict('records'))
            assert actual is not None
            assert all(actual[field] == settlement[field] for field in ('status', 'risk_units', 'profit_units'))
            alternative_results = [settle({**decision, 'selection': offer}, sides.to_dict('records')) for offer in offers]
            assert all(item is not None for item in alternative_results)
            outcome = next((offer['selection'] for offer, result in zip(offers, alternative_results) if result['status'] == 'win'), 'Void or winning method not quoted')
            for name, weights in allocations.items():
                active = [i for i, weight in enumerate(weights) if weight > 1e-6]
                profit = sum(float(weight) * result['profit_units'] for weight, result in zip(weights, alternative_results))
                risk = sum(float(weight) * result['risk_units'] for weight, result in zip(weights, alternative_results))
                strategies[name].append({'event_id': decision['event_id'], 'matchup_id': decision['matchup_id'],
                    'profit_units': profit, 'risk_units': risk, 'positions': len(active)})
        details.append({'event_date': decision['event_date'], 'matchup_id': decision['matchup_id'],
            'fight': f"{decision['fighter_name']} vs {decision['opponent_name']}",
            'status': settlement['status'] if settlement else 'pending',
            'selected': chosen, 'unique_quoted_methods': len(offers), 'qualifying_unique_methods': len(qualifying),
            'alternative_shorter_prices': sum(decimal(offers[i]['moneyline']) < decimal(chosen['moneyline']) for i in qualifying) if chosen else 0,
            'best_single_growth_selection': offers[best]['selection'] if best is not None else None,
            'joint': [{**offer, 'budget_share': float(weight)} for offer, weight in zip(offers, allocations['joint_log_growth']) if weight > 1e-6],
            'known_outcome': outcome})
    selections = [d['selection'] for d in decisions if d['selection']]
    nonvoid = [d for d in decisions if settled_by_id.get(d['matchup_id'], {}).get('status') in ('win', 'loss')]
    summaries = {}
    for name, rows in strategies.items():
        profit = sum(row['profit_units'] for row in rows); risk = sum(row['risk_units'] for row in rows)
        cards = defaultdict(float)
        for row in rows:
            cards[row['event_id']] += row['profit_units']
        summaries[name] = {'settled_fights_including_passes_and_voids': len(rows),
            'fights_with_stakes': sum(row['positions'] > 0 for row in rows),
            'multi_position_fights': sum(row['positions'] > 1 for row in rows),
            'risk_units': risk, 'profit_units': profit, 'return_per_risk_unit': profit / risk if risk else None,
            'card_profits': dict(cards)}
    assert abs(summaries['frozen_max_ev']['profit_units'] - report['profit_units']) < 1e-8
    assert abs(summaries['frozen_max_ev']['risk_units'] - report['settled_risk_units']) < 1e-8
    board = load(inputs[4]); perf = load(inputs[5])
    main_pairs = {identity(row) for row in perf['records']}
    overlaps = [d for d in decisions if d['selection'] and identity(d) in main_pairs]
    outcome_publication = load(inputs[7])
    chosen_details = [d for d in details if d['selected']]
    result = {
        'source_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'input_hashes': hashes,
        'method_report_as_of_utc': report['generated_at_utc'], 'method_policy': load(inputs[3]),
        'live_board': {'captured_at_utc': board['observed_at_utc'], 'eligible_offers_before_one_pick_limit': len(board['offers']),
            'funded_picks': len(board['bets']), 'allocation_policy': board['allocation_policy']},
        'method_summary': {'frozen_fights': len(decisions), 'selected': len(selections),
            'median_american_odds': float(np.median([s['moneyline'] for s in selections])),
            'at_least_plus_1000': sum(s['moneyline'] >= 1000 for s in selections),
            'at_least_plus_2000': sum(s['moneyline'] >= 2000 for s in selections),
            'nonvoid_settled_picks': len(nonvoid),
            'wins': sum(settled_by_id[d['matchup_id']]['status'] == 'win' for d in nonvoid),
            'model_expected_wins_on_same_picks': sum(d['selection']['probability'] for d in nonvoid),
            'fights_with_multiple_claimed_positive_ev_methods': sum(d['qualifying_unique_methods'] > 1 for d in details),
            'selected_fights_with_shorter_positive_ev_alternative': sum(d['alternative_shorter_prices'] > 0 for d in details),
            'best_single_growth_changes': sum(d['best_single_growth_selection'] != d['selected']['selection'] for d in chosen_details),
            'joint_fights_with_multiple_positions_all_records': sum(len(d['joint']) > 1 for d in details)},
        'cross_experiment_overlap': {'fights': len(overlaps),
            'examples': [f"{d['event_date']}: {d['fighter_name']} vs {d['opponent_name']}" for d in overlaps[:8]],
            'meaning': 'Recorded experiments overlap; their stake limits are not enforced as one combined portfolio.'},
        'counterfactuals': summaries,
        'counterfactual_limits': ['Retrospective diagnostic, not a newly earned prospective result or a strategy selection test.',
            'One unit total budget per fight; joint log optimizer treats one unit as 1% of bankroll. No compounding.',
            'Same frozen first-capture forecasts and quote set; no later price replacements or outcome-based choice.',
            'Best quote per distinct method, assuming access to every quoted book; actual execution/limits unverified.',
            'Six standard method states plus residual other states. Other states modeled as refunds and stress-tested as losses.',
            'The joint optimizer uses the original uncalibrated probabilities; it is not ready for betting decisions.',
            'Joint optimization can consider every quoted method; the equal-split and best-single diagnostics retain the original 5% standalone EV threshold.',
            'Only settled method experiment fights can be scored. A full moneyline/method/totals optimizer is not replayed.'],
        'joint_data_gap': {'published_outcome_fields': list(outcome_publication['matchups'][0]),
            'finding': 'Winner/method probabilities and totals probabilities are published separately; method-by-duration joint cells and posterior draws are not archived by this publication.'},
        'fight_details': details,
    }
    assert hashes == {name: sha256((DATA / name).read_bytes()).hexdigest() for name in inputs}
    destination = Path(__file__).with_name('correlation_audit.json')
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['method_summary', 'cross_experiment_overlap', 'counterfactuals']}, indent=2))


if __name__ == '__main__':
    main()
