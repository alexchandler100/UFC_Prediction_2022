const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const root = path.join(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'script.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const extract = name => {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, name);
  return source.slice(start, source.indexOf('\nfunction ', start + 1));
};
const s = {formatDate: value => value};
vm.createContext(s);
vm.runInContext(['finite', 'decimalOdds', 'kellyFraction', 'probabilityLogit', 'equalLogitPool',
  'performanceStakePlan', 'simulatePaperBankroll', 'portfolioRecordValid', 'portfolioFightKey',
  'portfolioContractKey', 'allocatePortfolioBatch', 'simulatePortfolioBankroll',
  'selectPerformanceRecords', 'fundedPerformanceCounts', 'rankPerformancePresets', 'applyPerformancePreset'].map(extract).join('\n'), s);
const options = id => Array.from(html.match(new RegExp(`<select id="${id}"[^>]*>([\\s\\S]*?)</select>`))[1]
  .matchAll(/<option value="([^"]+)"[^>]*>([^<]+)<\/option>/g), m => ({value:m[1],label:m[2]}));
const choices = {strategies:options('performance-staking'), timings:options('performance-timing'), markets:options('performance-market')};
const records = JSON.parse(fs.readFileSync(path.join(root, 'src/content/data/market/bet_performance.json'), 'utf8')).records;

test('every full preset is ranked by the same bankroll replay used in Results', () => {
  const before = JSON.stringify(records);
  const ranked = s.rankPerformancePresets(records, 1000, choices);
  assert.equal(ranked.length, choices.strategies.length * choices.timings.length * choices.markets.length);
  assert.equal(new Set(ranked.map(row=>row.id)).size, ranked.length);
  for (let i=0;i<ranked.length;i++) {
    const preset = ranked[i];
    if (i) assert.ok(ranked[i-1].endingBankroll >= preset.endingBankroll);
    const filtered = records.filter(row=>preset.market === 'all' || row.category === preset.market);
    const result = s.simulatePaperBankroll(s.selectPerformanceRecords(filtered,preset.timing),1000,preset.staking);
    assert.equal(preset.endingBankroll,result.endingBankroll);
    assert.equal(preset.funded,s.fundedPerformanceCounts(result.rows).funded);
  }
  assert.equal(JSON.stringify(records),before);
});
test('changing initial bankroll recalculates all preset balances', () => {
  const small = s.rankPerformancePresets(records,1000,choices);
  const large = new Map(s.rankPerformancePresets(records,2000,choices).map(row=>[row.id,row]));
  for (const preset of small) assert.ok(Math.abs(large.get(preset.id).endingBankroll - preset.endingBankroll*2) < 1e-7);
});
test('selecting a preset overrides strategy, timing, market and refreshes Results', () => {
  const preset = s.rankPerformancePresets(records,1000,choices)[0];
  const controls = Object.fromEntries(['presets','staking','timing','market','bankroll','show-zero'].map(id=>[`#performance-${id}`,{value:'unchanged'}]));
  controls['#performance-presets'].value = preset.id;
  let renders=0;
  s.$ = key=>controls[key]; s.state={performancePresetCache:{presets:[preset]}}; s.renderBetPerformance=()=>renders++;
  s.applyPerformancePreset();
  assert.equal(controls['#performance-staking'].value,preset.staking);
  assert.equal(controls['#performance-timing'].value,preset.timing);
  assert.equal(controls['#performance-market'].value,preset.market);
  assert.equal(controls['#performance-bankroll'].value,'unchanged');
  assert.equal(controls['#performance-show-zero'].value,'unchanged');
  assert.equal(renders,1);
});
