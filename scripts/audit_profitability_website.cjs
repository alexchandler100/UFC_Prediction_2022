/* Run the actual website's pure bankroll functions offline, without a DOM. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'script.js'), 'utf8');
function extract(name) {
  const start = source.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`Missing website function: ${name}`);
  const next = source.indexOf('\nfunction ', start + 1);
  return source.slice(start, next < 0 ? source.length : next);
}
const sandbox = {};
vm.createContext(sandbox);
vm.runInContext(
  ['finite', 'decimalOdds', 'kellyFraction', 'probabilityLogit', 'equalLogitPool',
    'selectPerformanceRecords', 'performanceStakePlan', 'simulatePaperBankroll']
    .map(extract).join('\n') + '\nfunction formatDate(value) { return value; }',
  sandbox,
  { timeout: 5000 },
);
const publication = JSON.parse(fs.readFileSync(path.join(root,
  'src/content/data/market/bet_performance.json'), 'utf8'));
const summaries = [];
const rows = [];
for (const timing of publication.timing_strategies) {
  for (const market of ['All', 'Moneyline', 'Total rounds']) {
    const selected = sandbox.selectPerformanceRecords(publication.records, timing)
      .filter(record => market === 'All' || record.category === market);
    for (const staking of publication.staking_strategies) {
      for (const reduction of [0, 0.02, 0.05]) {
        const adjusted = selected.map(record => ({ ...record,
          unit_profit: record.unit_profit > 0
            ? record.unit_profit * (1 - reduction) : record.unit_profit,
        }));
        const result = sandbox.simulatePaperBankroll(adjusted, 100, staking);
        const funded = result.rows.filter(row => row.stake > 1e-12);
        summaries.push({ timing, market, staking, payout_reduction: reduction,
          selected_records: selected.length, shown_settled_records: result.rows.length,
          funded_settled_bets: funded.length, pending_records: result.pending.length,
          unsupported_records: result.unsupported.length,
          funded_wins: funded.filter(row => row.status === 'won').length,
          funded_losses: funded.filter(row => row.status === 'lost').length,
          settled_events: new Set(result.rows.map(row => row.event_id)).size,
          initial_bankroll: 100, ending_bankroll: result.endingBankroll,
          bankroll_growth: result.endingBankroll / 100 - 1,
          risk_units: result.totalStaked, profit_units: result.profit,
          roi: result.roi, maximum_drawdown_fraction: result.maxDrawdown,
        });
        if (market === 'All' && reduction === 0 &&
            ['official_t24', 'first_qualifying'].includes(timing)) {
          for (const row of result.rows) rows.push({ timing, staking,
            record_id: row.record_id, market_key: row.market_key,
            event_id: row.event_id, event_date: row.event_date,
            category: row.category, selection: row.selection,
            target_book: row.target_book, offered_moneyline: row.offered_moneyline,
            published_at_utc: row.published_at_utc,
            settled_at_utc: row.settled_at_utc,
            status: row.status, stake: row.stake, profit: row.profit,
            sizing_probability: row.sizing_probability,
            assessment_timing: row.bayesian_kelly?.assessment_timing ?? null,
          });
        }
      }
    }
  }
}
process.stdout.write(JSON.stringify({
  source: 'Actual unmodified script.js pure functions executed in Node vm',
  publication_as_of_utc: publication.as_of_utc,
  no_bet_baseline: { initial_bankroll: 100, ending_bankroll: 100,
    risk_units: 0, profit_units: 0, bankroll_growth: 0, roi: null,
    maximum_drawdown_fraction: 0 },
  limitations: [
    'All returns are paper replays; there are no records of placed or accepted wagers.',
    'Website simulation settles whole cards by event date and ignores pending capital; it is not an execution-timed portfolio replay.',
    'Missing settlement times on archived snapshots prevent a verified outstanding-capital replay.',
    'Historical robust assessments marked retrospective were not necessarily published before the fight; they are not prospective policy evidence.',
    'One or two settled cards cannot supply meaningful resampling uncertainty; no interval is reported for this website archive.',
    'Alternative timing views can use later snapshots; their results are descriptive, not new independently tested policies.',
    'Payout stresses leave original selections and stake calculations fixed, reducing only positive net winning payouts.',
    'Shown settled counts include zero-stake rows; funded counts count only stakes greater than zero.',
  ], summaries, rows,
}));
