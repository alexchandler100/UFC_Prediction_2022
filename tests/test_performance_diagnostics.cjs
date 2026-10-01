const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '..', 'script.js'), 'utf8');
function extract(name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, name);
  const end = source.indexOf('\nfunction ', start + 1);
  return source.slice(start, end < 0 ? source.length : end);
}
const s = {};
vm.createContext(s);
vm.runInContext(['finite', 'decimalOdds', 'kellyFraction', 'probabilityLogit', 'equalLogitPool',
  'performanceStakePlan', 'simulatePaperBankroll', 'fundedPerformanceCounts',
  'performanceDecision', 'performanceCoverage', 'sharedPerformanceComparison', 'performanceComparisonGroups'].map(extract).join('\n')
  + '\nfunction formatDate(value) { return value; }', s);
function record(id, overrides = {}) {
  return { record_id: id, event_id: 'card', event_date: '2026-09-05', fighter_id: id, opponent_id: 'opponent',
    category: 'Moneyline', estimated_win_probability: 0.6, offered_moneyline: 100, kelly_fraction: 0.2,
    status: 'won', unit_profit: 1, model_support_probability: 0.6, simulation_support_probability: 0.6,
    bayesian_kelly: { status: 'available', recommended_fraction: 0.05, posterior_lower_probability: 0.55 },
    ...overrides };
}
const plain = value => JSON.parse(JSON.stringify(value));

test('coverage separates settled and pending missing estimates from zero-stake passes', () => {
  const rows = [record('bet'), record('pass', { bayesian_kelly: { status: 'available', recommended_fraction: 0, posterior_lower_probability: 0.4 }}),
    record('missing', { bayesian_kelly: { status: 'unavailable' }, category: 'Total rounds' }),
    record('pending', { status: 'pending' }), record('pending-pass', { status: 'pending', allocated_fraction: 0 }),
    record('pending-missing', { status: 'pending', bayesian_kelly: null })];
  const coverage = s.performanceCoverage(rows, 'robust_bayesian_kelly');
  assert.deepEqual(plain(coverage.settled), { total: 3, bet: 1, pass: 1, missing: 1 });
  assert.deepEqual(plain(coverage.pending), { total: 3, bet: 1, pass: 1, missing: 1 });
  assert.equal(coverage.decisions[4].reason, 'Saved portfolio allocation is zero');
});

test('blends explain incompatible totals separately and identify missing predictions', () => {
  assert.equal(s.performanceDecision(record('total', { category: 'Total rounds', model_support_probability: null }), 'half_kelly_model_blend').reason,
    'Winner model does not predict total rounds');
  const missing = s.performanceDecision(record('missing', { model_support_probability: null, simulation_support_probability: null }), 'half_kelly_model_sim_blend');
  assert.equal(missing.reason, 'No prediction saved by pick time: winner model and simulation');
  assert.equal(s.performanceDecision(record('edge', { simulation_support_probability: 0.1 }), 'half_kelly_sim_blend').decision, 'pass');
});

test('shared comparison retains passes, excludes pending and missing, and uses identical record IDs', () => {
  const records = [record('shared-pass', { bayesian_kelly: { status: 'available', recommended_fraction: 0, posterior_lower_probability: 0.4 } }),
    record('missing', { bayesian_kelly: null }), record('pending', { status: 'pending' }), record('shared-bet')];
  const strategies = ['robust_bayesian_kelly', 'half_kelly', 'half_kelly_model_sim_blend'];
  const result = s.sharedPerformanceComparison(records, strategies, 1000);
  assert.equal(result.records.length, 2);
  assert.equal(result.excluded, 1);
  assert.equal(result.fights, 2);
  assert.equal(result.cards, 1);
  for (const comparison of result.comparisons) {
    assert.deepEqual(plain(comparison.result.rows.map(r => r.record_id)), ['shared-pass', 'shared-bet']);
  }
  assert.equal(result.comparisons[0].result.totalStaked, 50);
  assert.equal(result.comparisons[1].result.totalStaked, 200);
  const changed = s.sharedPerformanceComparison(records.map(r => ({ ...r, status: r.status === 'pending' ? 'pending' : 'lost', unit_profit: -1 })), strategies, 1000);
  assert.deepEqual(plain(changed.records.map(r => r.record_id)), plain(result.records.map(r => r.record_id)));
});

test('empty shared cohort and all-pass strategies do not invent returns', () => {
  const strategies = ['robust_bayesian_kelly', 'half_kelly'];
  const empty = s.sharedPerformanceComparison([record('missing', { bayesian_kelly: null })], strategies, 1000);
  assert.equal(empty.records.length, 0);
  assert.equal(empty.comparisons[0].result.endingBankroll, 1000);
  assert.equal(empty.comparisons[0].result.roi, null);
  const pass = s.sharedPerformanceComparison([record('pass', { allocated_fraction: 0 })], strategies, 1000);
  assert.equal(pass.records.length, 1);
  assert.equal(pass.comparisons[0].result.roi, null);
  assert.equal(pass.comparisons[0].result.profit, 0);
});

test('shared fight counts deduplicate multiple selections and both fighter orientations', () => {
  const a = record('one', { fighter_id: 'alpha', opponent_id: 'beta' });
  const b = record('two', { fighter_id: 'beta', opponent_id: 'alpha' });
  const result = s.sharedPerformanceComparison([a, b], ['half_kelly'], 1000);
  assert.equal(result.records.length, 2);
  assert.equal(result.fights, 1);
});

test('expected-return tiers use the saved probability and odds with inclusive 5, 10 and 20 percent boundaries', () => {
  for (const [edge, fraction] of [[-.1, 0], [0, 0], [.049999, 0], [.05, .01], [.099999, .01], [.10, .02], [.199999, .02], [.20, .03], [.60, .03]]) {
    // p * decimal odds - 1 = expected net return per dollar staked.
    for (const odds of [100, 250, -200]) {
      const probability = (1 + edge) / s.decimalOdds(odds);
      if (probability >= 1) continue;
      const row = record('tier', { estimated_win_probability: probability, offered_moneyline: odds,
        estimated_expected_return: 10, kelly_fraction: 0, model_support_probability: null, simulation_support_probability: null });
      const plan = s.performanceStakePlan(row, 'tiered_expected_return');
      assert.equal(plan.fraction, fraction, `edge ${edge}, odds ${odds}`);
      assert.ok(Math.abs(plan.expectedReturn - edge) < 1e-12);
      assert.equal(plan.probability, probability);
      assert.deepEqual(plain(s.performanceStakePlan({...row, status: 'lost', unit_profit: -1}, 'tiered_expected_return')), plain(plan));
    }
  }
  assert.equal(s.performanceDecision(record('small-edge', {estimated_win_probability: .52}), 'tiered_expected_return').reason,
    'Estimated return is below the 5% minimum');
});

test('tiered stakes separate unavailable inputs from an estimated return below the minimum', () => {
  for (const value of [null, undefined, '', NaN, Infinity, 0, 1, -.1, 1.1]) {
    assert.equal(s.performanceStakePlan(record('invalid', {estimated_win_probability: value}), 'tiered_expected_return'), null);
  }
  for (const value of [null, undefined, '', NaN, Infinity, 0, 90, -90]) {
    assert.equal(s.performanceStakePlan(record('invalid', {offered_moneyline: value}), 'tiered_expected_return'), null);
  }
  const coverage = s.performanceCoverage([record('missing', {offered_moneyline: null}),
    record('pass', {estimated_win_probability: .5}), record('pending', {status: 'pending'})], 'tiered_expected_return');
  assert.deepEqual(plain(coverage.settled), {total: 2, bet: 0, pass: 1, missing: 1});
  assert.deepEqual(plain(coverage.pending), {total: 1, bet: 1, pass: 0, missing: 0});
});

test('tiered replay uses card-start bankroll, includes losses and voids, and excludes pending returns', () => {
  const rows = [record('one', {estimated_win_probability: .525}),
    record('two', {estimated_win_probability: .55, status: 'lost', unit_profit: -1}),
    record('three', {estimated_win_probability: .6, status: 'void', unit_profit: 0}),
    record('pending', {status: 'pending'}),
    record('next', {event_id: 'later', event_date: '2026-09-12', estimated_win_probability: .6})];
  const result = s.simulatePaperBankroll(rows, 1000, 'tiered_expected_return');
  assert.deepEqual(plain(result.rows.map(row => row.stake)), [10, 20, 30, 29.7]);
  assert.ok(Math.abs(result.endingBankroll - 1019.7) < 1e-9);
  assert.equal(result.pending.length, 1);
  const doubled = s.simulatePaperBankroll(rows, 2000, 'tiered_expected_return');
  assert.equal(doubled.endingBankroll, result.endingBankroll * 2);
  const crowded = s.simulatePaperBankroll(Array.from({length: 40}, (_, i) => record(String(i))), 1000, 'tiered_expected_return');
  assert.ok(crowded.totalStaked <= 1000 + 1e-9, 'existing card cash limit still applies');
});

test('missing advanced estimates do not shrink stake-only or unrelated pairwise comparisons', () => {
  const rows = [
    record('all'),
    record('model-only', {simulation_support_probability: null}),
    record('sim-only-pass', {model_support_probability: null, simulation_support_probability: .05}),
    record('total', {category: 'Total rounds', model_support_probability: null, simulation_support_probability: null, bayesian_kelly: null}),
    record('pending', {status: 'pending'}),
  ];
  const before = JSON.stringify(rows);
  const groups = s.performanceComparisonGroups(rows, 1000);
  assert.equal(groups.sizing.records.length, 4);
  assert.equal(groups.sizing.comparisons.length, 5);
  assert.deepEqual(plain(groups.alternatives.map(row => row.comparison.records.length)), [3, 2, 2, 1]);
  assert.equal(groups.alternatives[2].comparison.comparisons[1].result.rows[1].stake, 0);
  for (const {comparison} of groups.alternatives) {
    assert.deepEqual(plain(comparison.comparisons[0].result.rows.map(row => row.record_id)),
      plain(comparison.comparisons[1].result.rows.map(row => row.record_id)));
  }
  assert.equal(JSON.stringify(rows), before);
  const changedResults = s.performanceComparisonGroups(rows.map(row => ({...row, status: row.status === 'pending' ? 'pending' : 'lost', unit_profit: -1})), 1000);
  assert.deepEqual(plain(changedResults.alternatives.map(row => row.comparison.records.map(record => record.record_id))),
    plain(groups.alternatives.map(row => row.comparison.records.map(record => record.record_id))));
  const totals = s.performanceComparisonGroups([rows[3]], 1000);
  assert.equal(totals.sizing.records.length, 1);
  assert.ok(totals.alternatives.every(row => row.comparison.records.length === 0));
});
