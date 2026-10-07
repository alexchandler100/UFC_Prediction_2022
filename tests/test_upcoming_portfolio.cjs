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
vm.runInContext(['finite', 'decimalOdds', 'evaluateUpcomingPaperOffers', 'recordedPaperStatus', 'recordedPaperGroups', 'marketRecommendationGroups', 'partitionMarketRecommendations', 'fundedPerformanceCounts'].map(extract).join('\n'), sandbox);
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

test('recorded picks survive price expiry, card start, board rollover and settlement', () => {
  const saved={...offer(),bet_id:'saved',snapshot_id:'snapshot',threshold_met:true,paper_only:true,execution_enabled:false,observed_at_utc:'2026-09-05T12:00:00Z',allocated_fraction:.01};
  const archive={paper_only:true,execution_enabled:false,snapshots:[saved]};
  const before=JSON.stringify(archive);
  const read=(performance={})=>sandbox.recordedPaperGroups(archive,{paper_only:true,execution_enabled:false,bets:[]},performance);
  assert.equal(read().length,1);
  assert.equal(sandbox.recordedPaperStatus(read()[0].latest,now+31*60000),'Upcoming');
  assert.equal(sandbox.recordedPaperStatus(read()[0].latest,now+2*3600000),'Awaiting result');
  const settled=read({records:[{record_type:'published_snapshot',record_id:'snapshot',status:'won',unit_profit:1}]});
  assert.equal(sandbox.recordedPaperStatus(settled[0].latest,now+86400000),'Correct');
  assert.equal(settled[0].latest.offered_moneyline,100);
  assert.equal(settled[0].latest.estimated_expected_return,.2);
  assert.equal(JSON.stringify(archive),before);
});

test('archive groups snapshots, filters saved books and never promotes an unrecorded offer',()=>{
  const saved={...offer(),bet_id:'one',threshold_met:true,paper_only:true,execution_enabled:false,observed_at_utc:'2026-09-05T12:00:00Z'};
  const second={...saved,bet_id:'two',observed_at_utc:'2026-09-05T12:15:00Z',offered_moneyline:110};
  const archive={paper_only:true,execution_enabled:false,snapshots:[saved,second]};
  const publication={...board([offer('never-recorded')]),bets:[second]};
  const groups=sandbox.recordedPaperGroups(archive,publication,{});
  assert.equal(groups.length,1);assert.equal(groups[0].versions.length,2);
  assert.equal(groups[0].latest.bet_id,'two');
  assert.equal(sandbox.recordedPaperGroups(archive,publication,{},new Set(['B'])).length,0);
  assert.equal(sandbox.recordedPaperGroups(null,board([offer()]),{}).length,0);
});

test('saved picks separate future, unresolved past and confirmed outcomes without guessing results', () => {
  const groups = [
    ['future', {event_date: '2026-09-06'}],
    ['starting-now', {event_start_utc: new Date(now).toISOString()}],
    ['past', {event_date: '2026-09-04'}],
    ['unknown-time', {}],
    ['win', {settlement_status: 'won'}],
    ['loss', {settlement_status: 'loss'}],
    ['void', {settlement_status: 'void'}],
  ].map(([key, latest]) => ({key, latest}));
  const sections = sandbox.partitionMarketRecommendations(groups, now);
  assert.deepEqual(Array.from(sections.upcoming, row => row.key), ['future']);
  assert.deepEqual(Array.from(sections.awaiting, row => row.key).sort(), ['past', 'starting-now', 'unknown-time']);
  assert.deepEqual(Array.from(sections.resolved, row => sandbox.recordedPaperStatus(row.latest, now)).sort(), ['Correct', 'Incorrect', 'Void']);
  assert.equal(sandbox.recordedPaperStatus({event_date: '2026-09-05', event_start_utc: '2026-09-05T14:00:00Z'}, now), 'Upcoming');
  assert.equal(sandbox.partitionMarketRecommendations(groups, now + 2 * 86400000).upcoming.length, 0);
});

test('repeated prices and official copies count once, with both totals fighters preserved', () => {
  const saved = {...offer(), bet_id: 'saved', snapshot_id: 'snapshot', threshold_met: true,
    fighter_name: 'Fighter A', opponent_name: 'Fighter B', category: 'Total rounds', selection: 'Over 2.5 rounds',
    paper_only: true, execution_enabled: false, observed_at_utc: '2026-09-05T11:00:00Z'};
  const later = {...saved, bet_id: 'later', snapshot_id: 'later-snapshot', target_book: 'B', offered_moneyline: 120, observed_at_utc: '2026-09-05T12:00:00Z'};
  const archive = {paper_only: true, execution_enabled: false, snapshots: [saved, later]};
  const performance = {records: [
    {...saved, official: true, record_id: 'official', published_at_utc: '2026-09-05T11:30:00Z', status: 'lost'},
    {record_type: 'published_snapshot', record_id: 'snapshot', status: 'lost'},
    {record_type: 'published_snapshot', record_id: 'later-snapshot', status: 'lost'},
  ]};
  const before = JSON.stringify({archive, performance});
  const groups = sandbox.recordedPaperGroups(archive, null, performance);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].versions.length, 3);
  assert.equal(groups[0].latest.target_book, 'B');
  assert.equal(groups[0].latest.offered_moneyline, 120);
  assert.equal(groups[0].latest.fighter_name, 'Fighter A');
  assert.equal(groups[0].latest.opponent_name, 'Fighter B');
  assert.equal(sandbox.recordedPaperStatus(groups[0].latest, now), 'Incorrect');
  const filtered = sandbox.recordedPaperGroups(archive, null, performance, new Set(['a']));
  assert.equal(filtered.length, 1);
  assert.equal(filtered[0].latest.target_book, 'A');
  assert.equal(sandbox.recordedPaperGroups(null, null, performance).length, 1);
  assert.equal(JSON.stringify({archive, performance}), before);
});

test('method picks retain the selected opponent and canonical matchup names', () => {
  const methods = {paper_only: true, execution_enabled: false, recommendations: [{
    decision_sha256: 'frozen', matchup_id: 'fight', fighter_id: 'b', matchup_fighter_id: 'a', matchup_opponent_id: 'b',
    fighter_name: 'Fighter A', opponent_name: 'Fighter B', book: 'Book', selection: 'Fighter B by KO/TKO',
    method: 'ko_tko', moneyline: 250, probability: .3, expected_return: .05, settlement_status: 'loss',
  }]};
  const before = JSON.stringify(methods);
  const groups = sandbox.marketRecommendationGroups(null, null, null, methods);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].latest.selected_fighter_id, 'b');
  assert.equal(groups[0].latest.fighter_id, 'a');
  assert.equal(groups[0].latest.opponent_name, 'Fighter B');
  assert.equal(groups[0].latest.offered_moneyline, 250);
  assert.equal(sandbox.partitionMarketRecommendations(groups, now).resolved.length, 1);
  assert.equal(sandbox.marketRecommendationGroups(null, null, null, methods, new Set(['Other'])).length, 0);
  assert.equal(JSON.stringify(methods), before);
});


test('total model experiment preserves names, book filtering and resolved results separately', () => {
  const row = {book: 'Book', moneyline: 150, probability: .6, expected_return: .5,
    fighter_name: 'A', opponent_name: 'B', matchup_fighter_id: 'a', matchup_opponent_id: 'b',
    decision_sha256: 'total-decision', event_id: 'card', event_date: '2026-09-05',
    event_start_utc: '2026-09-05T14:00:00Z', selection: 'Over 1.5 rounds', settlement_status: 'win', profit_units: 1.5};
  const report = {paper_only: true, execution_enabled: false, recommendations: [row]};
  const groups = sandbox.marketRecommendationGroups(null,null,null,null,null,report);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].latest.category, 'Total rounds');
  assert.equal(groups[0].latest.experimental, true);
  assert.equal(groups[0].latest.fighter_name, 'A');
  assert.equal(groups[0].latest.opponent_name, 'B');
  assert.equal(groups[0].latest.unit_profit, 1.5);
  assert.equal(sandbox.partitionMarketRecommendations(groups, now).resolved.length,1);
  assert.equal(sandbox.marketRecommendationGroups(null,null,null,null,new Set(['Other']),report).length,0);
  assert.equal(sandbox.marketRecommendationGroups(null,null,null,null,null,{...report,execution_enabled:true}).length,0);
});
