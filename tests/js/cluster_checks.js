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
const sprite = c4.networkLayers;
c4.setUniverse(['BTCUSDT', 'ETHUSDT']);        // same universe
check('sprite_kept_when_universe_unchanged', c4.networkLayers === sprite);
c4.setUniverse(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']);
check('sprite_invalidated_when_universe_changes', c4.networkLayers === null);

/* The network is split across layers so each can drift on its own phase. One
 * layer would make the drift a picture being slid about. */
c4.setUniverse(['A', 'B', 'C', 'D', 'E', 'F']);
c4.buildNetworkSprite();
check('network_is_layered_for_independent_drift',
      Array.isArray(c4.networkLayers) && c4.networkLayers.length > 1,
      'layers: ' + (c4.networkLayers || []).length);
const phases = (c4.networkLayers || []).map(l => l.phase + ':' + l.rate);
check('no_two_layers_drift_together',
      new Set(phases).size === phases.length, phases.join(' '));

/* The drift has to actually move, and stay small: a ring that wanders far
 * detaches its nodes from the filaments they sit on. */
const d0 = c4.nodeDrift(0, 0), d1 = c4.nodeDrift(0, 2500);
check('nodes_drift_over_time', Math.abs(d0.x - d1.x) + Math.abs(d0.y - d1.y) > 0.05,
      JSON.stringify([d0, d1]));
let maxDrift = 0;
for (let i = 0; i < 40; i++) {
  for (let t = 0; t < 20000; t += 250) {
    const d = c4.nodeDrift(i, t);
    maxDrift = Math.max(maxDrift, Math.hypot(d.x, d.y));
  }
}
check('drift_stays_bounded', maxDrift < 8, 'max ' + maxDrift.toFixed(2));
check('neighbouring_nodes_do_not_drift_in_lockstep',
      Math.abs(c4.nodeDrift(0, 1000).x - c4.nodeDrift(1, 1000).x) > 0.01);

/* The ring grows out of how often each symbol has been scanned. Without this
 * the field looks the same after six hours as it did at startup, which is the
 * complaint it answers. */
const c5 = new window.Cluster(canvas, null);
c5.resize();
c5.setUniverse(['A', 'B']);
c5.spawn({ symbol: 'A', kind: 'scan', intensity: 1 }, 1000);
check('a_pulse_is_counted_against_its_symbol', c5.scans.A === 1);
check('an_unscanned_symbol_stays_at_zero', !c5.scans.B);
for (let i = 0; i < 50; i++) c5.spawn({ symbol: 'A', kind: 'scan', intensity: 1 }, 1000);
check('scanning_accumulates', c5.scans.A === 51, 'scans ' + c5.scans.A);
check('the_last_verdict_is_remembered', c5.lastKind.A === 'scan');
c5.spawn({ symbol: 'A', kind: 'order', intensity: 1 }, 2000);
check('a_new_verdict_replaces_the_last', c5.lastKind.A === 'order');

/* Growth must be gradual and bounded, or the first cohort saturates within a
 * minute and the ring stops meaning anything for the rest of the session. */
const m = window.Cluster.maturity;
check('maturity_is_zero_before_anything_is_scanned', m(0) === 0);
check('maturity_never_exceeds_one', m(1e9) <= 1);
check('maturity_rises_with_scanning', m(200) > m(20) && m(20) > m(2));
check('maturity_is_not_saturated_by_one_cohort', m(60) < 0.85,
      'm(60)=' + m(60).toFixed(3));

/* The legend is painted from this palette rather than carrying its own copy,
 * so the two cannot drift apart. */
const dots = {};
const fakeLegend = {
  querySelectorAll: () => ['scan', 'decision', 'cap', 'order'].map(k => ({
    getAttribute: () => k,
    querySelector: () => ({ style: (dots[k] = { background: '' }) })
  }))
};
const painted = window.Cluster.paintLegend(fakeLegend);
check('the_legend_is_painted_from_the_palette', painted === 4, 'painted ' + painted);
check('decision_is_neon_green', /^rgb\(57, ?255, ?140\)$/.test(dots.decision.background),
      dots.decision.background);
check('cap_is_neon_purple', /^rgb\(190, ?60, ?255\)$/.test(dots.cap.background),
      dots.cap.background);
check('order_is_bright_orange', /^rgb\(255, ?150, ?40\)$/.test(dots.order.background),
      dots.order.background);
check('the_three_loud_kinds_are_distinct',
      new Set([dots.decision.background, dots.cap.background,
               dots.order.background, dots.scan.background]).size === 4);

/* The half-second freeze.
 *
 * The cohort rotates every twenty seconds, which changes the symbol list and
 * invalidated the network sprite. Rebuilding it allocated three fresh
 * canvases: measured at a realistic 1100x700 on a 2x display, allocating them
 * costs 193ms, clearing them costs 0ms, and drawing all 150 filaments costs
 * 7ms. The work was never the drawing -- it was asking for sixteen megapixels
 * of backing store twice a minute on the thread that paints the frame.
 */
const c6 = new window.Cluster(canvas, null);
c6.resize();
c6.setUniverse(Array.from({length: 60}, (_, i) => 'A' + i));
c6.buildNetworkSprite();
const pooled = c6.networkLayers.map(l => l.canvas);
c6.setUniverse(Array.from({length: 60}, (_, i) => 'B' + i));
c6.buildNetworkSprite();
check('a rotation reuses the canvases rather than allocating new ones',
      c6.networkLayers.every((l, i) => l.canvas === pooled[i]),
      'the sprite canvases were re-created on a rotation');
check('the redraw still produced the new universe',
      c6.symbols.length === 60 && c6.symbols[0] === 'B0');

console.log(JSON.stringify(results, null, 2));
process.exit(results.every(r => r.pass) ? 0 : 1);
