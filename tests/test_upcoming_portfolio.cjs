const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '..', 'script.js'), 'utf8');
function extract(name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `Missing ${name}`);
  const end = source.indexOf('\nfunction ', start + 1);
  return source.slice(start, end < 0 ? source.length : end);
}
const sandbox = {};
vm.createContext(sandbox);
vm.runInContext(['finite', 'decimalOdds', 'evaluateUpcomingPaperOffers', 'fundedPerformanceCounts'].map(extract).join('\n'), sandbox);
const now = Date.parse('2026-09-05T12:00:00Z');
function offer(id = 'fight', overrides = {}) {
  const row = { event_id: 'card', event_date: '2026-09-05', matchup_id: id,
    fighter_id: `${id}-a`, opponent_id: `${id}-b`, selection: `${id}-a`, side: 'fighter',
    category: 'Moneyline', target_book: 'A', offered_moneyline: 100,
    source_quote_updated_at_utc: '2026-09-05T11:55:00Z', event_start_utc: '2026-09-05T14:00:00Z',
    estimated_win_probability: 0.6, estimated_expected_return: 0.2, robust_lower_expected_return: 0.12,
    bayesian_kelly: { status: 'available', posterior_mean_probability: 0.6,
      posterior_lower_probability: 0.56, recommended_fraction: 0.05 }, ...overrides };
  return row;
}
function board(offers) { return { schema_version: 2, paper_only: true, execution_enabled: false, minimum_expected_return: 0.05, offers }; }
const evaluate = (offers, books = null, at = now) => sandbox.evaluateUpcomingPaperOffers(board(offers), books, at);

test('expired, future and missing source times never receive a stake', () => {
  for (const time of [null, '2026-09-05T11:29:59Z', '2026-09-05T12:00:01Z', '2026-09-05T11:55:00']) {
    assert.equal(evaluate([offer('f', { source_quote_updated_at_utc: time })]).bets.length, 0);
  }
  assert.equal(evaluate([offer('f', { source_quote_updated_at_utc: '2026-09-05T11:30:00Z' })]).bets.length, 1);
  assert.equal(evaluate([offer()], null, now + 26 * 60 * 1000).bets.length, 0);
});

test('card start and schema1 publications fail closed', () => {
  for (const start of [null, '2026-09-05T12:00:00Z', '2026-09-05T11:59:59Z']) {
    assert.equal(evaluate([offer('f', { event_start_utc: start })]).bets.length, 0);
  }
  assert.equal(sandbox.evaluateUpcomingPaperOffers({ ...board([offer()]), schema_version: 1, bets: [offer()] }, null, now).bets.length, 0);
});

test('calibrated edge and conservative stake govern eligibility, not raw EV', () => {
  const row = offer(); row.raw_estimated_expected_return = 0.8;
  row.estimated_win_probability = 0.4; row.estimated_expected_return = -0.2;
  row.bayesian_kelly.posterior_mean_probability = 0.4;
  assert.equal(evaluate([row]).bets.length, 0);
  const zero = offer(); zero.bayesian_kelly.recommended_fraction = 0;
  assert.equal(evaluate([zero]).bets.length, 0);
  assert.equal(evaluate([offer('f', { estimated_expected_return: 1 })]).bets.length, 0);
});

test('book changes select the next accessible offer before allocating', () => {
  const better = offer('f', { target_book: 'A', offered_moneyline: 110, estimated_expected_return: 0.26, robust_lower_expected_return: 0.176 });
  const alternate = offer('f', { target_book: 'B' });
  const rows = [alternate, better]; const saved = JSON.stringify(rows);
  assert.equal(evaluate(rows).bets[0].target_book, 'A');
  assert.equal(evaluate(rows, new Set(['B'])).bets[0].target_book, 'B');
  assert.equal(evaluate(rows, new Set()).bets.length, 0);
  assert.equal(JSON.stringify(rows), saved);
});

test('one physical fight receives only one allocation across reversed identities and markets', () => {
  const moneyline = offer('f');
  const total = offer('another-market', { fighter_id: 'f-b', opponent_id: 'f-a', category: 'Total rounds' });
  assert.equal(evaluate([total]).bets.length, 0);
  total.bayesian_kelly.schedule_contract_version = 'verified-pre-fight-schedule-v1';
  total.schedule_contract_version = 'verified-pre-fight-schedule-v1';
  total.model_version = 'candidate-discrete-time-competing-risks-v2-verified-schedules';
  assert.equal(evaluate([total]).bets.length, 0); // Corrected model still lacks betting evidence.
  total.betting_performance_validated = true;
  assert.equal(evaluate([total]).bets.length, 1);
  assert.equal(evaluate([total, moneyline]).bets.length, 1);
});

test('fight, card and snapshot caps all apply', () => {
  const offers = Array.from({ length: 24 }, (_, index) => offer(`f${index}`, { event_id: `card${Math.floor(index / 8)}` }));
  const result = evaluate(offers);
  assert.equal(result.bets.length, 10);
  assert.ok(result.allocatedFraction <= 0.10 + 1e-12);
  const cards = new Map();
  for (const row of result.bets) {
    assert.ok(row.allocated_fraction <= 0.01);
    cards.set(row.event_id, (cards.get(row.event_id) || 0) + row.allocated_fraction);
  }
  for (const fraction of cards.values()) assert.ok(fraction <= 0.05 + 1e-12);
  assert.equal(JSON.stringify(evaluate([...offers].reverse()).bets), JSON.stringify(result.bets));
});

test('zero-stake outcomes are excluded from funded performance record', () => {
  const result = sandbox.fundedPerformanceCounts([{ stake: 0, status: 'won' }, { stake: 0, status: 'lost' }, { stake: 1, status: 'lost' }]);
  assert.equal(result.funded, 1); assert.equal(result.zeroStake, 2);
  assert.equal(result.wins, 0); assert.equal(result.losses, 1);
});
