// Reproduce the shared-budget diagnostic from the website's actual functions.
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm'), crypto = require('node:crypto');
const root = path.resolve(__dirname, '../../..');
const source = fs.readFileSync(path.join(root, 'script.js'), 'utf8');
const read = name => JSON.parse(fs.readFileSync(path.join(root, 'src/content/data', name), 'utf8'));
const s = {};
vm.createContext(s);
vm.runInContext(['finite', 'decimalOdds', 'kellyFraction', 'probabilityLogit', 'equalLogitPool',
  'selectPerformanceRecords', 'performanceStakePlan', 'simulatePaperBankroll', 'portfolioRecordValid',
  'portfolioFightKey', 'portfolioContractKey', 'allocatePortfolioBatch', 'simulatePortfolioBankroll',
  'recordedPaperStatus', 'upcomingMarketExplanation'].map(name => {
    const start = source.indexOf(`function ${name}(`);
    if (start < 0) throw Error(name);
    return source.slice(start, source.indexOf('\nfunction ', start + 1));
  }).join('\n') + '\nfunction formatDate(value) { return value; }', s);
const performance = read('market/bet_performance.json');
const selected = s.selectPerformanceRecords(performance.records, 'first_qualifying');
const results = {};
for (const rule of ['portfolio_flat_v1', 'portfolio_half_kelly_v1', 'portfolio_tiered_v1']) {
  const r = s.simulatePaperBankroll(selected, 1000, rule);
  const funded = r.rows.filter(row => row.stake > 0);
  const fights = new Map();
  funded.forEach(row => fights.set(row.fight, (fights.get(row.fight) || 0) + 1));
  results[rule] = {settled_bets: funded.length, settled_passes: r.rows.length - funded.length,
    missing_inputs: r.unsupported.length, total_staked: r.totalStaked, profit: r.profit, return_on_stakes: r.roi,
    maximum_drawdown: r.maxDrawdown, maximum_fight_fraction: r.maximumFightFraction,
    maximum_card_fraction: r.maximumCardFraction, peak_outstanding_at_entry_fraction: r.peakOutstandingFraction,
    pending_reserve: r.reservedStake, multi_pick_fights: [...fights.values()].filter(count => count > 1).length,
    approximate_settlement_times: r.approximateSettlements};
}
const evidence = {as_of_utc: '2026-10-04T06:00:00Z', initial_bankroll: 1000, timing: 'first_qualifying',
  performance_publication_sha256: performance.publication_sha256,
  website_sha256: crypto.createHash('sha256').update(source).digest('hex'), results,
  market_empty_state: s.upcomingMarketExplanation(read('external/all_upcoming_forecasts.json'),
    read('market/candidate_report.json'), [], 'all', null, Date.parse('2026-10-04T06:00:00Z')),
  interpretation: 'Retrospective diagnostics, not evidence of improved expected profitability. Methods remain a separate experiment.',
  settlement_convention: 'Exact saved settlement timestamps when present; otherwise UTC event date plus 48 hours for already settled records. Pending stays reserved.',
  allocation: 'Chronological bookings; same-time proposals scaled proportionally; no duplicate logical selections; existing reservations survive new captures.',
  raw_joint_forecasts: 'New model publications export and archive the raw winner/method/time grid before the event date. Historical grids are not reconstructed. Decision timing, draws/NC and book settlement rules remain unresolved; settlement_ready is always false.'};
fs.writeFileSync(path.join(__dirname, 'shared_budget_validation.json'), JSON.stringify(evidence, null, 2) + '\n');
console.log(JSON.stringify(evidence, null, 2));
