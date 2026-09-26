/* The orb's arithmetic, with no three.js and no canvas in it.
 *
 * Split out so it can be tested. Everything below decides how the orb behaves
 * -- how fast a value chases live data, when a mode flips, where a symbol's
 * ripple appears -- and none of it needs a GPU to be wrong. The parts that do
 * need a GPU are checked in a browser; these are checked under Node, in the
 * same pytest run as everything else.
 */

export function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

/* Frame-rate independent damping.
 *
 * The naive `a += (b - a) * k` moves a fixed fraction of the gap *per frame*,
 * so it settles roughly twice as fast on a 120Hz monitor as on a 60Hz one and
 * jumps when a frame is late. This is the exponential form: the result depends
 * on elapsed time and not on how that time was sliced, which is the whole
 * reason the orb looks the same on any machine.
 */
export function damp(current, target, rate, dt) {
  return target + (current - target) * Math.exp(-rate * dt);
}

/* Mode thresholds on total activity.
 *
 * Each band is left only when the value crosses the *other* edge, which is
 * what stops a reading sitting on a boundary from flipping between two morph
 * states several times a second. Without the overlap a terminal scanning at
 * exactly the active/intense boundary would visibly shudder.
 */
export const MODE_BANDS = [
  { mode: 'idle',    enter: 0.00, exit: 0.10 },
  { mode: 'active',  enter: 0.08, exit: 0.52 },
  { mode: 'intense', enter: 0.45, exit: Infinity }
];

export function nextMode(current, activity) {
  for (const band of MODE_BANDS) {
    if (band.mode === current && activity >= band.enter && activity < band.exit) {
      return current;
    }
  }
  if (activity >= MODE_BANDS[2].enter) return 'intense';
  if (activity >= MODE_BANDS[1].enter) return 'active';
  return 'idle';
}

/* An even spread of points over a sphere.
 *
 * The golden-angle spiral, not random directions: random points on a sphere
 * clump visibly at this count, and the clumps read as a defect in the mesh
 * rather than as texture.
 */
export function fibonacciSphere(count) {
  const positions = new Float32Array(count * 3);
  const phases = new Float32Array(count);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = count === 1 ? 0 : 1 - (i / (count - 1)) * 2;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * i;
    positions[i * 3] = Math.cos(theta) * radius;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = Math.sin(theta) * radius;
    // Deterministic per-point phase, so a rebuild at a different quality tier
    // does not reshuffle the shimmer into a visible pop.
    phases[i] = ((Math.sin(i * 12.9898) * 43758.5453) % 1 + 1) % 1;
  }
  return { positions, phases };
}

/* A stable point on the unit sphere for a symbol.
 *
 * The same ticker must always ripple from the same place. This is the one
 * piece of information the 2D pulse field carried that a body cannot spell
 * out, and it only survives if the position is derived from the name rather
 * than assigned on arrival: an operator who has watched the panel for an hour
 * knows roughly where SPY lives, and that is only true if SPY does not move
 * between frames, between reconnects, or between sessions.
 */
export function symbolDirection(symbol) {
  let h = 2166136261;
  const text = String(symbol || '');
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const a = ((h >>> 0) % 100000) / 100000;
  const b = ((Math.imul(h ^ 0x5f3759df, 2654435761) >>> 0) % 100000) / 100000;
  const y = a * 2 - 1;
  const r = Math.sqrt(Math.max(0, 1 - y * y));
  const theta = b * Math.PI * 2;
  return { x: Math.cos(theta) * r, y: y, z: Math.sin(theta) * r };
}

/* Pulses per second, per kind, that counts as an activity of 1.0.
 *
 * Calibrated from the rates this terminal actually produces rather than
 * chosen for looks: a full universe sweep runs scans at tens per second,
 * while an order is a handful a day. Normalising every kind against one
 * denominator would make everything except scanning invisible, so each is
 * scaled against what *it* considers busy.
 */
export const FULL_RATE = {
  scan: 26,
  refused: 6,
  warmup: 8,
  decision: 3,
  cap: 3,
  order: 0.6,
  halt: 0.3
};

export function activityFor(kind, count, seconds) {
  const per = count / Math.max(seconds, 0.001);
  return clamp(per / (FULL_RATE[kind] || 4), 0, 1);
}

/* The heartbeat: two unequal beats, not one sine.
 *
 * A single sine reads as a pulsing light. A sharp first beat, a short gap and
 * a softer second reads as something with a chest, which is the entire point
 * of putting a heartbeat on a panel nobody is staring at.
 */
export function heartbeatAt(phase) {
  const lub = Math.exp(-Math.pow((phase - 0.06) / 0.05, 2));
  const dub = Math.exp(-Math.pow((phase - 0.22) / 0.07, 2)) * 0.6;
  return lub + dub;
}

/* Beats per minute from total activity. Never zero: a resting rate is still a
 * rate, and a panel whose heart stops looks broken rather than quiet. */
export function heartRate(activity) {
  return 42 + 58 * clamp(activity, 0, 1.4);
}
