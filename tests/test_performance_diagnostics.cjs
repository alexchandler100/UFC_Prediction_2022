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
  'performanceDecision', 'performanceCoverage', 'sharedPerformanceComparison'].map(extract).join('\n')
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
