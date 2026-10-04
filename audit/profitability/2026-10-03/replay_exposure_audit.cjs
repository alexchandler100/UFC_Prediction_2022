// Read-only replay diagnostic using the actual website functions and saved data.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '../../..');
const source = fs.readFileSync(path.join(root, 'script.js'), 'utf8');
const raw = fs.readFileSync(path.join(root, 'src/content/data/market/bet_performance.json'));
const s = {};
vm.createContext(s);
vm.runInContext(['finite', 'decimalOdds', 'kellyFraction', 'probabilityLogit', 'equalLogitPool',
  'performanceStakePlan', 'selectPerformanceRecords', 'simulatePaperBankroll'].map(name => {
    const start = source.indexOf(`function ${name}(`);
    if (start < 0) throw Error(`Missing ${name}`);
    return source.slice(start, source.indexOf('\nfunction ', start + 1));
  }).join('\n') + '\nfunction formatDate(value) { return value; }', s);
const selected = s.selectPerformanceRecords(JSON.parse(raw).records, 'first_qualifying');
const strategies = {};
for (const staking of ['tiered_expected_return', 'half_kelly', 'flat_one_percent', 'robust_bayesian_kelly']) {
  const result = s.simulatePaperBankroll(selected, 1000, staking);
  const fights = new Map(), cards = new Map();
  for (const row of result.rows) {
    const key = JSON.stringify([row.event_id, [row.fighter_id, row.opponent_id].sort()]);
    if (!fights.has(key)) fights.set(key, {fight: `${row.fighter_name} vs ${row.opponent_name}`, event: row.event_date, fraction: 0, selections: []});
    const fight = fights.get(key);
    fight.fraction += row.stake_fraction;
    if (row.stake > 0) fight.selections.push({selection: row.selection, fraction: row.stake_fraction, category: row.category});
    cards.set(row.event_date, (cards.get(row.event_date) || 0) + row.stake_fraction);
  }
  strategies[staking] = {settled_selections: result.rows.length, settled_fights: fights.size,
    multi_pick_fights: [...fights.values()].filter(fight => fight.selections.length > 1),
    maximum_fight_stake_fraction: Math.max(0, ...[...fights.values()].map(fight => fight.fraction)),
    card_stake_fractions: Object.fromEntries(cards)};
}
const evidence = {timing: 'first_qualifying', source_as_of_utc: JSON.parse(raw).as_of_utc,
  source_sha256: crypto.createHash('sha256').update(raw).digest('hex'),
  website_sha256: crypto.createHash('sha256').update(source).digest('hex'), strategies,
  limitation: 'Hypothetical recorded selections; methods are a separate experiment, not included in these bankroll replays.'};
fs.writeFileSync(path.join(__dirname, 'replay_exposure_audit.json'), JSON.stringify(evidence, null, 2) + '\n');
console.log(JSON.stringify(Object.fromEntries(Object.entries(strategies).map(([key, value]) => [key,
  {...value, multi_pick_fights: value.multi_pick_fights.length}])) , null, 2));
