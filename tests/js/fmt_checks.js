/* The header's clock arithmetic, under Node.
 *
 * The venue publishes its calendar in UTC. An operator in Johannesburg or
 * New York reading "opens Fri 13:30 UTC" has to convert it every time they
 * glance at the header, which is why the countdown leads: it is the same
 * sentence in every timezone.
 */
const fs = require('fs');
const path = require('path');

const g = {};
eval(fs.readFileSync(
  path.join(__dirname, '..', '..', 'src', 'imperium', 'server', 'static', 'fmt.js'),
  'utf8').replace('typeof window !== \'undefined\' ? window : globalThis', 'g'));

const results = [];
function check(name, pass, detail) {
  results.push({ name, pass: !!pass, detail: detail || '' });
}

const NOW = Date.parse('2026-09-10T09:18:00Z');
const OPEN = '2026-09-10T13:30:00Z';

check('a countdown is rendered', g.Fmt.fmtUntil(OPEN, NOW) === '4h 12m',
      g.Fmt.fmtUntil(OPEN, NOW));
check('under an hour drops the hours',
      g.Fmt.fmtUntil('2026-09-10T09:45:00Z', NOW) === '27m',
      g.Fmt.fmtUntil('2026-09-10T09:45:00Z', NOW));
/* A negative countdown would render as "-1h -5m", which reads as a fault in
 * the one place an operator looks to see whether anything is wrong. */
check('a past instant never renders negative',
      g.Fmt.fmtUntil('2026-09-10T08:00:00Z', NOW) === 'now',
      g.Fmt.fmtUntil('2026-09-10T08:00:00Z', NOW));
check('the exact instant reads as now',
      g.Fmt.fmtUntil('2026-09-10T09:18:00Z', NOW) === 'now');

/* Missing and malformed inputs must return empty rather than "Invalid Date",
 * which is what the header showed for a venue that had not answered yet. */
check('a missing timestamp is empty', g.Fmt.fmtUntil(null, NOW) === '' &&
      g.Fmt.fmtLocalTime(null) === '');
check('a malformed timestamp is empty', g.Fmt.fmtUntil('not a date', NOW) === '',
      g.Fmt.fmtUntil('not a date', NOW));
check('a malformed timestamp does not reach the header as Invalid Date',
      g.Fmt.fmtLocalTime('not a date') === '',
      g.Fmt.fmtLocalTime('not a date'));

/* The local rendering must actually differ from UTC where the viewer does.
 * Run under a fixed non-UTC zone, which is the case the change exists for. */
const local = g.Fmt.fmtLocalTime(OPEN);
check('a local time is produced at all', /\d{1,2}:\d{2}/.test(local), local);
/* Under TZ=Africa/Johannesburg (UTC+2) 13:30 UTC is 15:30 local. Asserting
 * only that it "looks like a time" passes for a UTC fallback, which is exactly
 * the bug -- the operator would be reading the venue's clock and believing it
 * was their own. */
const utcHour = new Date(OPEN).getUTCHours();
const localHour = new Date(OPEN).getHours();
check('the viewer sees their own clock, not UTC', localHour !== utcHour,
      'local hour ' + localHour + ' vs UTC ' + utcHour);
check('the rendered string carries the local hour',
      local.indexOf(String(localHour > 12 ? localHour - 12 : localHour)) === 0,
      local + ' (expected to start with local hour ' + localHour + ')');

console.log(JSON.stringify(results, null, 2));
process.exit(results.every(r => r.pass) ? 0 : 1);
