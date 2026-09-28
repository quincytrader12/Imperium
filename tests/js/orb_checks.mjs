/* The orb's arithmetic, checked under Node and reported to pytest.
 *
 * These are the properties that decide whether the orb reads as alive or as
 * broken, and every one of them is invisible in a screenshot: damping that is
 * frame-rate dependent looks fine at 60fps and wrong at 144, a mode band with
 * no hysteresis looks fine until the activity sits on a boundary, and a symbol
 * whose position is not derived from its name looks fine until you watch it
 * for a minute.
 *
 * Emits JSON on stdout; tests/test_orb_js.py runs this and fails on any
 * `pass: false`.
 */
import {
  clamp, damp, nextMode, MODE_BANDS, fibonacciSphere, symbolDirection,
  activityFor, FULL_RATE, heartbeatAt, heartRate
} from '../../src/imperium/server/static/orb.math.js';

const results = [];
function check(name, ok, detail) {
  results.push({ name, pass: !!ok, detail: detail || '' });
}

// -- damping ---------------------------------------------------------------

{
  // The same elapsed time must produce the same result however it is sliced.
  // A frame-based lerp fails this, and the failure looks like the orb moving
  // at a different speed on a different monitor.
  const oneStep = damp(0, 1, 3, 0.1);
  let many = 0;
  for (let i = 0; i < 10; i++) many = damp(many, 1, 3, 0.01);
  check('damping is frame-rate independent',
        Math.abs(oneStep - many) < 1e-9,
        `one 100ms step ${oneStep.toFixed(9)} vs ten 10ms steps ${many.toFixed(9)}`);

  // It approaches but never overshoots, however large the step.
  const huge = damp(0, 1, 3, 100);
  check('damping never overshoots its target',
        huge <= 1 && huge > 0.999, `dt=100s gave ${huge}`);

  const backwards = damp(1, 0, 3, 0.5);
  check('damping works downward too',
        backwards < 1 && backwards > 0, `${backwards}`);
}

// -- mode hysteresis -------------------------------------------------------

{
  // Bands must overlap, or a value on the boundary flips every frame.
  let overlapping = true;
  for (let i = 0; i < MODE_BANDS.length - 1; i++) {
    if (MODE_BANDS[i + 1].enter >= MODE_BANDS[i].exit) overlapping = false;
  }
  check('mode bands overlap, so a boundary value cannot flicker', overlapping,
        JSON.stringify(MODE_BANDS));

  // In the overlap, the current mode is kept rather than recomputed.
  check('a value inside the overlap keeps the mode it is already in',
        nextMode('idle', 0.09) === 'idle' && nextMode('active', 0.09) === 'active',
        `idle->${nextMode('idle', 0.09)}, active->${nextMode('active', 0.09)}`);

  check('a value past the exit edge does change the mode',
        nextMode('idle', 0.5) !== 'idle' && nextMode('active', 0.9) === 'intense',
        `idle@0.5->${nextMode('idle', 0.5)}, active@0.9->${nextMode('active', 0.9)}`);

  // The specific flicker this exists to prevent: a rate jittering by a hair
  // around the active/intense edge must not produce two different modes.
  const edge = MODE_BANDS[2].enter;
  let mode = 'active';
  const seen = new Set();
  for (let i = 0; i < 200; i++) {
    mode = nextMode(mode, edge + (i % 2 ? 0.004 : -0.004));
    seen.add(mode);
  }
  check('jitter around the intense edge does not oscillate the mode',
        seen.size === 1, `modes seen: ${[...seen].join(',')}`);
}

// -- the particle skin -----------------------------------------------------

{
  const { positions, phases } = fibonacciSphere(4000);
  check('every skin point lands on the unit sphere', (() => {
    for (let i = 0; i < 4000; i++) {
      const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
      if (Math.abs(Math.hypot(x, y, z) - 1) > 1e-4) return false;
    }
    return true;
  })(), '');

  // Even coverage, which is the reason for the golden-angle spiral: random
  // directions clump at this count and the clumps read as a defect.
  const bands = new Array(10).fill(0);
  for (let i = 0; i < 4000; i++) {
    const y = positions[i * 3 + 1];
    bands[Math.min(9, Math.floor((y + 1) / 2 * 10))]++;
  }
  const lo = Math.min(...bands), hi = Math.max(...bands);
  check('skin points are evenly spread, not clumped',
        hi / lo < 1.25, `band counts ${bands.join(',')}`);

  check('per-point phases are spread across the cycle', (() => {
    const buckets = new Array(8).fill(0);
    for (let i = 0; i < 4000; i++) buckets[Math.min(7, Math.floor(phases[i] * 8))]++;
    return Math.min(...buckets) > 4000 / 8 * 0.5;
  })(), '');

  // Deterministic: a quality-tier change rebuilds the cloud, and a reshuffled
  // shimmer would pop.
  const again = fibonacciSphere(4000);
  check('the skin is deterministic across rebuilds',
        again.phases[123] === phases[123] && again.positions[369] === positions[369],
        '');

  check('a single point does not divide by zero',
        Number.isFinite(fibonacciSphere(1).positions[1]), '');
}

// -- symbol positions ------------------------------------------------------

{
  const spy = symbolDirection('SPY');
  const again = symbolDirection('SPY');
  check('a symbol always maps to the same point',
        spy.x === again.x && spy.y === again.y && spy.z === again.z, '');

  check('symbol points are unit vectors',
        Math.abs(Math.hypot(spy.x, spy.y, spy.z) - 1) < 1e-6,
        `|SPY| = ${Math.hypot(spy.x, spy.y, spy.z)}`);

  // Different tickers must land apart, or two symbols ripple from one spot
  // and the panel says something false about which one fired.
  const names = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'XLF', 'XLK',
                 'XLE', 'XLV', 'TSLA', 'META', 'GOOG', 'BTC/USD', 'ETH/USD'];
  let closest = Infinity, pair = '';
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const a = symbolDirection(names[i]), b = symbolDirection(names[j]);
      const arc = Math.acos(Math.min(1, Math.max(-1,
        a.x * b.x + a.y * b.y + a.z * b.z)));
      if (arc < closest) { closest = arc; pair = `${names[i]}/${names[j]}`; }
    }
  }
  check('different symbols land at distinguishable points',
        closest > 0.12, `closest pair ${pair} at ${closest.toFixed(3)} rad`);

  check('an empty symbol still yields a valid direction',
        Number.isFinite(symbolDirection('').y), '');
}

// -- activity scaling ------------------------------------------------------

{
  // Each kind is scaled against what it considers busy. Without that, one
  // order a day is indistinguishable from no orders at all.
  check('a single order in the window is clearly visible',
        activityFor('order', 1, 3) > 0.5,
        `${activityFor('order', 1, 3)}`);
  check('a single scan in the window is barely visible',
        activityFor('scan', 1, 3) < 0.05,
        `${activityFor('scan', 1, 3)}`);
  check('activity is capped at one',
        activityFor('scan', 100000, 3) === 1, '');
  check('no activity is zero',
        activityFor('scan', 0, 3) === 0, '');
  check('an unknown kind still scales rather than dividing by zero',
        Number.isFinite(activityFor('nonsense', 4, 3)), '');
  check('every palette kind has a rate calibrated for it',
        ['scan', 'refused', 'warmup', 'decision', 'cap', 'order', 'halt']
          .every((k) => FULL_RATE[k] > 0),
        JSON.stringify(FULL_RATE));
}

// -- heartbeat -------------------------------------------------------------

{
  // Two beats per cycle, unequal: a single sine reads as a pulsing light
  // rather than as something with a chest.
  const samples = [];
  for (let i = 0; i < 1000; i++) samples.push(heartbeatAt(i / 1000));
  let peaks = 0;
  for (let i = 1; i < samples.length - 1; i++) {
    if (samples[i] > samples[i - 1] && samples[i] >= samples[i + 1]
        && samples[i] > 0.15) peaks++;
  }
  check('the heartbeat is a double pulse', peaks === 2, `${peaks} peaks`);

  const lub = Math.max(...samples.slice(0, 150));
  const dub = Math.max(...samples.slice(150, 350));
  check('the second beat is softer than the first',
        dub < lub && dub > 0.3 * lub, `lub ${lub.toFixed(3)} dub ${dub.toFixed(3)}`);

  check('the heart never stops, however quiet it gets',
        heartRate(0) > 0, `${heartRate(0)} bpm at rest`);
  check('the heart rate rises with activity',
        heartRate(1) > heartRate(0.2) && heartRate(0.2) > heartRate(0),
        `${heartRate(0)} / ${heartRate(0.2)} / ${heartRate(1)} bpm`);
  check('the heart rate is bounded at the top',
        heartRate(1000) === heartRate(1.4), '');
}

// -- clamp -----------------------------------------------------------------

check('clamp holds both ends',
      clamp(-5, 0, 1) === 0 && clamp(5, 0, 1) === 1 && clamp(0.5, 0, 1) === 0.5, '');

process.stdout.write(JSON.stringify(results));
