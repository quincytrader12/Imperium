/* Exercises cluster.js under Node with a stub canvas.
 *
 * These check the four properties the brief calls out that are invisible in a
 * screenshot: dedupe on the sequence number, a bounded replay, an orb budget
 * measured against the bloom rather than the core, and jittered speed/offset.
 */
'use strict';
const fs = require('fs');
const path = require('path');

function stubCtx() {
  const noop = () => {};
  return new Proxy({}, {
    get(_, k) {
      if (k === 'createRadialGradient' || k === 'createLinearGradient') {
        return () => ({ addColorStop: noop });
      }
      if (k === 'canvas') return { width: 0, height: 0 };
      return noop;
    },
    set() { return true; }
  });
}

global.window = global;
global.performance = { now: () => Date.now() };
global.devicePixelRatio = 1;
global.document = {
  createElement: () => ({ width: 0, height: 0, getContext: stubCtx })
};

const src = fs.readFileSync(
  path.join(__dirname, '..', '..', 'src', 'imperium', 'server', 'static', 'cluster.js'),
  'utf8');
eval(src);

const canvas = {
  width: 0, height: 0,
  getContext: stubCtx,
  getBoundingClientRect: () => ({ width: 1200, height: 700 }),
  addEventListener: () => {}
};

const results = [];
function check(name, cond, detail) {
  results.push({ name, pass: !!cond, detail: detail || '' });
}

const c = new window.Cluster(canvas, null);
c.resize();
c.setUniverse(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']);

/* 1. Dedupe on the monotonic sequence number. Overlapping snapshot windows must
 *    spawn each orb exactly once. */
const window1 = [];
for (let i = 1; i <= 10; i++) {
  window1.push({ seq: i, symbol: 'BTCUSDT', kind: 'scan', reason: 'r', intensity: 0.5 });
}
c.ingest(window1);
const afterFirst = c.orbs.length;
c.ingest(window1);                       // the identical window again
check('dedupe_identical_window', c.orbs.length === afterFirst,
      `${afterFirst} -> ${c.orbs.length}`);

const overlapping = [];
for (let i = 6; i <= 15; i++) {
  overlapping.push({ seq: i, symbol: 'BTCUSDT', kind: 'scan', reason: 'r', intensity: 0.5 });
}
c.ingest(overlapping);                   // 5 overlap, 5 are new
check('dedupe_overlapping_window', c.orbs.length === afterFirst + 5,
      `expected ${afterFirst + 5}, got ${c.orbs.length}`);

/* 2. The replay queue is bounded. A backgrounded tab may have missed thousands
 *    of pulses; replaying them all is a stampede describing work already
 *    missed. */
const c2 = new window.Cluster(canvas, null);
c2.resize();
c2.setUniverse(['BTCUSDT']);
const flood = [];
for (let i = 1; i <= 5000; i++) {
  flood.push({ seq: i, symbol: 'BTCUSDT', kind: 'scan', reason: 'r', intensity: 0.5 });
}
c2.ingest(flood);
/* Asserted against a fixed literal, not against MAX_REPLAY itself. A mutation
 * test found that comparing to the exported constant made this check pass when
 * MAX_REPLAY was raised to 100000 -- a test that moves with the thing it is
 * testing is not a test. */
const REPLAY_CEILING = 250;
check('replay_is_bounded', c2.orbs.length <= REPLAY_CEILING,
      `${c2.orbs.length} orbs from a 5000-pulse backlog (ceiling ${REPLAY_CEILING})`);
check('replay_advances_seq', c2.seenSeq === 5000, `seq=${c2.seenSeq}`);

/* 3. The orb budget is measured against the bloom radius, not the 2px core, and
 *    scales with the panel's area. Bound on the core and the field saturates
 *    into a wash at any real pulse rate. */
const big = new window.Cluster({
  ...canvas, getBoundingClientRect: () => ({ width: 1600, height: 900 })
}, null);
big.resize();
const small = new window.Cluster({
  ...canvas, getBoundingClientRect: () => ({ width: 400, height: 300 })
}, null);
small.resize();
check('budget_scales_with_area', big.orbBudget() > small.orbBudget(),
      `${big.orbBudget()} vs ${small.orbBudget()}`);

// A core-based budget would be (BLOOM_MULTIPLE^2) ~= 30x larger. Assert the
// budget is far below that, i.e. the bloom is what was measured.
const bloomBudget = big.orbBudget();
const coreEquivalent = Math.floor((1600 * 900 * 0.22) / (Math.PI * 2 * 2));
check('budget_uses_bloom_not_core', bloomBudget < coreEquivalent / 10,
      `bloom-based ${bloomBudget} vs core-based ${coreEquivalent}`);

/* 4. Speed and start offset are jittered. Identical speed from an identical
 *    start makes a burst travel as one rigid smear. */
const c3 = new window.Cluster(canvas, null);
c3.resize();
c3.setUniverse(['BTCUSDT']);
const burst = [];
for (let i = 1; i <= 40; i++) {
  burst.push({ seq: i, symbol: 'BTCUSDT', kind: 'order', reason: 'r', intensity: 1 });
}
c3.ingest(burst);
const speeds = new Set(c3.orbs.map(o => o.speed.toFixed(6)));
const starts = new Set(c3.orbs.map(o => o.t.toFixed(6)));
const depths = new Set(c3.orbs.map(o => o.depth.toFixed(6)));
check('speeds_are_jittered', speeds.size > 30, `${speeds.size} distinct of 40`);
check('starts_are_jittered', starts.size > 30, `${starts.size} distinct of 40`);
check('depth_varies', depths.size > 30, `${depths.size} distinct of 40`);

/* 5. A pulse for a symbol that is not on the ring must still be shown, not
 *    silently dropped -- BOOK-level halts have no node of their own. */
const before = c3.orbs.length;
c3.ingest([{ seq: 999, symbol: 'BOOK', kind: 'halt', reason: 'daily loss', intensity: 1 }]);
check('unknown_symbol_still_renders', c3.orbs.length === before + 1,
      `${before} -> ${c3.orbs.length}`);

/* 6. The network sprite is cached and regenerated only when the universe
 *    changes -- rebuilding the filament geometry every frame is the most
 *    expensive thing this canvas could do. */
const c4 = new window.Cluster(canvas, null);
c4.resize();
c4.setUniverse(['BTCUSDT', 'ETHUSDT']);
c4.buildNetworkSprite();
const sprite = c4.networkSprite;
c4.setUniverse(['BTCUSDT', 'ETHUSDT']);        // same universe
check('sprite_kept_when_universe_unchanged', c4.networkSprite === sprite);
c4.setUniverse(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']);
check('sprite_invalidated_when_universe_changes', c4.networkSprite === null);

console.log(JSON.stringify(results, null, 2));
process.exit(results.every(r => r.pass) ? 0 : 1);
