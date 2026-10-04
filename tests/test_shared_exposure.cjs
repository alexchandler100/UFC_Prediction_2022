const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../script.js'), 'utf8');
const s = {};
vm.createContext(s);
vm.runInContext(['finite', 'decimalOdds', 'kellyFraction', 'probabilityLogit', 'equalLogitPool',
  'performanceStakePlan', 'simulatePaperBankroll', 'portfolioRecordValid', 'portfolioFightKey',
  'portfolioContractKey', 'allocatePortfolioBatch', 'simulatePortfolioBankroll',
  'recordedPaperStatus', 'upcomingMarketExplanation'].map(name => {
    const start = source.indexOf(`function ${name}(`);
    assert.ok(start >= 0, name);
    return source.slice(start, source.indexOf('\nfunction ', start + 1));
  }).join('\n'), s);
const row = (id, overrides = {}) => ({record_id: id, event_id: 'card', event_date: '2026-10-10',
  fighter_id: id, opponent_id: 'opponent', selection: id, category: 'Moneyline', side: 'fighter',
  published_at_utc: '2026-10-08T12:00:00Z', event_start_utc: '2026-10-10T20:00:00Z',
  settled_at_utc: '2026-10-11T10:00:00Z', status: 'lost', unit_profit: -1,
  estimated_win_probability: .6, offered_moneyline: 100, kelly_fraction: .2, ...overrides});
const replay = rows => s.simulatePaperBankroll(rows, 1000, 'portfolio_tiered_v1');
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} != ${expected}`);

test('two 3% proposals share a 3% fight budget across reversed identities and markets', () => {
  const r = replay([row('a'), row('b', {fighter_id: 'opponent', opponent_id: 'a', category: 'Total rounds', selection: 'Over 2.5'})]);
  assert.equal(r.rows.length, 2);
  for (const item of r.rows) { close(item.planned_stake, 30); close(item.stake, 15); }
  close(r.totalStaked, 30); close(r.profit, -30); close(r.maximumFightFraction, .03);
});

test('fight/card caps apply before outstanding funds, without picking winners', () => {
  const rows = Array.from({length: 10}, (_, i) => row(String(i)));
  const loss = replay(rows), win = replay(rows.map(r => ({...r, status: 'won', unit_profit: 1})));
  close(loss.totalStaked, 50); close(loss.maximumCardFraction, .05);
  assert.deepEqual(loss.rows.map(r => r.stake), win.rows.map(r => r.stake));
  const conservative = s.simulatePaperBankroll(rows, 1000, 'portfolio_half_kelly_v1');
  close(conservative.maximumCardFraction, .05);
  assert.ok(conservative.maximumFightFraction <= .01 + 1e-12);
});

test('later captures, multiple books and formatting cannot buy an identical selection twice', () => {
  const r = replay([row('a', {target_book: 'A', category: 'Total rounds', selection: 'Over 2.5'}),
    row('better', {fighter_id: 'a', target_book: 'B', offered_moneyline: 150, category: 'Total rounds', selection: 'Over 2.5'}),
    row('later', {fighter_id: 'a', target_book: 'C', category: 'Total rounds', selection: 'Over 2.50', published_at_utc: '2026-10-09T12:00:00Z'})]);
  close(r.totalStaked, 30);
  assert.equal(r.rows.filter(r => r.stake > 0).length, 1);
  assert.equal(r.rows.find(r => r.stake > 0).record_id, 'better');
  assert.match(r.rows.find(r => r.record_id === 'later').allocation_reason, /duplicate/);
});

test('different total lines consume the same fight account, including later captures', () => {
  const r = replay([row('a', {category: 'Total rounds', selection: 'Over 1.5'}),
    row('b', {fighter_id: 'a', category: 'Total rounds', selection: 'Over 2.5', published_at_utc: '2026-10-09T12:00:00Z'})]);
  close(r.rows[0].stake, 30); close(r.rows[1].stake, 0);
  assert.equal(r.rows[1].allocation_reason, 'Shared fight budget');
});

test('a later stronger tier can only use the remaining total fight budget', () => {
  const r = replay([row('a', {estimated_win_probability: .525}),
    row('b', {fighter_id: 'a', category: 'Total rounds', selection: 'Over 2.5', published_at_utc: '2026-10-09T12:00:00Z'})]);
  close(r.rows[0].stake, 10); close(r.rows[1].stake, 20); close(r.maximumFightFraction, .03);
});

test('pending earlier cards reserve cash and prevent 10% being spent again', () => {
  const rows = ['first', 'second', 'third'].flatMap((event, day) => Array.from({length: 5}, (_, i) => row(event + i,
    {event_id: event, event_date: `2026-10-${12 + day}`, status: 'pending', unit_profit: null, settled_at_utc: null,
      published_at_utc: `2026-10-0${5 + day}T12:00:00Z`}))) ;
  const r = replay(rows);
  assert.equal(r.rows.length, 0); close(r.profit, 0); close(r.reservedStake, 100);
  close(r.peakOutstandingFraction, .10);
  assert.ok(r.pending.filter(row => row.event_id === 'third').every(row => row.stake === 0));
});

test('settlement releases funds only when known; unused fight/card budgets are not refilled', () => {
  const rows = ['first', 'second'].flatMap(event => Array.from({length: 5}, (_, i) => row(event + i,
    {event_id: event, settled_at_utc: '2026-10-09T10:00:00Z', status: 'void', unit_profit: 0})));
  const r = replay([...rows, row('new', {event_id: 'third', published_at_utc: '2026-10-09T12:00:00Z'}),
    row('old', {event_id: 'first', published_at_utc: '2026-10-09T12:00:00Z'})]);
  close(r.rows.find(row => row.record_id === 'new').stake, 30);
  close(r.rows.find(row => row.record_id === 'old').stake, 0);
});

test('same-time allocation is invariant to input ordering', () => {
  const rows = [row('a'), row('b'), row('c', {fighter_id: 'a', category: 'Total rounds', selection: 'Under 2.5'})];
  const stakes = result => Object.fromEntries(result.rows.map(row => [row.record_id, row.stake]));
  assert.deepEqual(stakes(replay(rows)), stakes(replay([...rows].reverse())));
});

test('missing or late publication identities fail closed; legacy settlement times are explicit', () => {
  const r = replay([row('a', {settled_at_utc: null}), row('missing', {fighter_id: ''}),
    row('late', {published_at_utc: '2026-10-11T12:00:00Z'}), row('bad', {estimated_win_probability: 1.2})]);
  assert.equal(r.unsupported.length, 3); assert.equal(r.approximateSettlements, 1);
  assert.equal(r.rows[0].release_at, Date.parse('2026-10-12T00:00:00Z'));
});

test('replay sizes against booking-time funds, not future results or future prices', () => {
  const base = row('a');
  const original = replay([base]);
  const changed = replay([base, row('future', {fighter_id: 'a', category: 'Total rounds', selection: 'Under 1.5',
    published_at_utc: '2026-10-09T12:00:00Z', offered_moneyline: 10000, status: 'won', unit_profit: 100})]);
  close(original.rows[0].stake, changed.rows[0].stake);
});

test('market empty state identifies price coverage and threshold instead of blaming filters', () => {
  const forecasts = {matchups: [row('a'), row('b')]};
  const candidates = {captured_at_utc: '2026-10-03T18:00:00Z', rows: [
    {...row('a'), matchup_id: 'a', market: 'Moneyline', book: 'A', moneyline: 100, adjusted_ev: .0149},
    {...row('a'), matchup_id: 'a', market: 'Moneyline', book: 'B', moneyline: 100, adjusted_ev: -.01}]};
  const now = Date.parse('2026-10-04T06:00:00Z');
  const info = s.upcomingMarketExplanation(forecasts, candidates, [], 'all', null, now);
  assert.equal(info.pricedFights, 1); assert.equal(info.totalFights, 2);
  assert.match(info.message, /1\.5%; the minimum is 5%/);
  assert.match(s.upcomingMarketExplanation(forecasts, candidates, [], 'Method', null, now).message, /No method picks/);
  assert.match(s.upcomingMarketExplanation(forecasts, candidates, [], 'Total rounds', null, now).message, /duration predictions/);
  assert.match(s.upcomingMarketExplanation(forecasts, candidates, [], 'all', new Set(['C']), now).message, /0 of 2/);
  assert.match(s.upcomingMarketExplanation(forecasts, candidates, [{latest: row('a')}], 'all', new Set(['C']), now).message, /hidden by/);
  assert.match(s.upcomingMarketExplanation(null, candidates, [], 'all', null, now).message, /next card/);
  const onFightDay = s.upcomingMarketExplanation(forecasts, candidates, [], 'all', null, Date.parse('2026-10-10T12:00:00Z'));
  assert.equal(onFightDay.totalFights, 2, 'a known later start keeps today\'s card upcoming');
});
