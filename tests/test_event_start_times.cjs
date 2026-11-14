const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '..', 'script.js'), 'utf8');
const start = source.indexOf('function eventStartTimeText(');
const end = source.indexOf('\nfunction ', start + 1);
const sandbox = {};
vm.createContext(sandbox);
vm.runInContext(source.slice(start, end), sandbox);
const schedule = {event_date: '2026-11-14', starts: {
  early_prelims: '2026-11-14T22:00:00Z', prelims: '2026-11-15T00:00:00Z', main_card: '2026-11-15T02:00:00Z'
}};

test('all broadcast stages display in local time with a timezone label', () => {
  assert.equal(sandbox.eventStartTimeText(schedule, 'America/Chicago'),
    'Early prelims 4:00 PM CST · Prelims 6:00 PM CST · Main card 8:00 PM CST');
  assert.match(sandbox.eventStartTimeText(schedule, 'America/Los_Angeles'), /Main card 6:00 PM PST/);
});
test('next-day starts include their local calendar date', () => {
  assert.match(sandbox.eventStartTimeText(schedule, 'Europe/London'), /Main card Nov 15, 2:00 AM GMT/);
});
test('daylight saving time follows the event date', () => {
  assert.match(sandbox.eventStartTimeText({event_date: '2026-07-04', starts: {main_card: '2026-07-05T02:00:00Z'}}, 'America/Chicago'), /Main card 9:00 PM CDT/);
});
test('unpublished times are explicit and early prelims are optional', () => {
  assert.equal(sandbox.eventStartTimeText(null, 'America/Chicago'), 'Main card time TBA · Prelims time TBA');
  assert.equal(sandbox.eventStartTimeText({event_date: schedule.event_date, starts: {main_card: schedule.starts.main_card}}, 'America/Chicago'), 'Main card 8:00 PM CST · Prelims time TBA');
});
