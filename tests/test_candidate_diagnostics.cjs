const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');
const source = fs.readFileSync(require('node:path').join(__dirname, '..', 'script.js'), 'utf8');
function extract(name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0);
  const end = source.indexOf('\nfunction ', start + 1);
  return source.slice(start, end);
}
const context = {};
vm.createContext(context);
vm.runInContext(['finite', 'candidateDiagnosticReasons', 'compareCandidateDiagnostics', 'methodPaperStatus'].map(extract).join('\n'), context);
const now = Date.parse('2026-09-04T12:00:00Z');
const offer = (id, extra = {}) => ({row_id: id, book: 'A', market: 'Moneyline',
  quote_updated_at_utc: '2026-09-04T11:59:00Z', event_start_utc: '2026-09-05T12:00:00Z',
  reasons_now: [], adjusted_ev: .1, conservative_ev: .03, model_ev: .2, ...extra});
test('best-supported ranking prioritizes usable positive conservative edges', () => {
  const rows = [offer('stale', {adjusted_ev: 2, quote_updated_at_utc: '2026-09-04T10:00:00Z'}),
    offer('unsupported', {adjusted_ev: 1, conservative_ev: -.2, reasons_now: ['uncertainty']}),
    offer('best', {adjusted_ev: .3}), offer('less')];
  rows.sort((a,b) => context.compareCandidateDiagnostics(a,b,'supported',now));
  assert.deepEqual(rows.map(r=>r.row_id), ['best','less','unsupported','stale']);
});
test('model sorting highlights independent-model opportunities', () => {
  const a=offer('a',{model_ev:.8,adjusted_ev:-.1}); const b=offer('b',{model_ev:.1,adjusted_ev:.5});
  assert.ok(context.compareCandidateDiagnostics(a,b,'model',now)<0);
  assert.ok(context.compareCandidateDiagnostics(a,b,'adjusted',now)>0);
});
test('price expiry and event start update without reloading the report', () => {
  const reasons=context.candidateDiagnosticReasons(offer('a'),Date.parse('2026-09-05T12:01:00Z'));
  assert.ok(reasons.includes('expired')); assert.ok(reasons.includes('event_started'));
  assert.ok(!context.candidateDiagnosticReasons(offer('a'),now).includes('expired'));
});
test('method paper prices expire and remain visible for result review', () => {
  const row = {observed_at_utc: '2026-09-04T11:59:00Z', event_start_utc: '2026-09-05T12:00:00Z'};
  assert.equal(context.methodPaperStatus(row, now), 'Recently collected');
  assert.equal(context.methodPaperStatus(row, now + 31 * 60000), 'Recorded price expired');
  assert.equal(context.methodPaperStatus(row, Date.parse(row.event_start_utc)), 'Awaiting result / review');
});
