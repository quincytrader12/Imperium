/* Wiring the orb to the terminal's live state.
 *
 * This is the only file that knows both about ProcessOrb and about this
 * particular terminal. The component itself takes generic processes; the
 * translation from "pulses on a websocket" into "activity per process" lives
 * here, so the orb stays reusable and the terminal-specific decisions stay in
 * one readable place.
 *
 * It is a module because three.js is, and app.js is a classic script, so the
 * two talk through window.orb rather than through imports.
 *
 * WHAT IT MUST NOT LOSE. This replaced a 2D pulse field, and that field was
 * load-bearing: an operator read the kind of work, the symbol that fired and
 * the overall rate off it at a glance. The first is the orb's colour zones,
 * the second is a ripple at that symbol's own point, the third is the morph
 * state and heartbeat. The invariants that field had -- dedupe on the
 * monotonic sequence number, a bounded replay after a backgrounded tab -- are
 * kept here, because they were never about drawing. They were about not
 * lying: an orb replayed twice says twice as much happened.
 */
import { ProcessOrb, MAX_PROCESSES } from './orb.js';
import { activityFor } from './orb.math.js';

/* A pulse arriving twice must not count twice, and a tab that was hidden for
 * an hour must not stampede. Both were true of the field this replaces. */
const MAX_REPLAY = 240;

/* The window over which a process's activity is measured, in milliseconds.
 *
 * Three seconds. Shorter and a quiet gap between scan sweeps reads as the
 * terminal having stopped; longer and an order placed now is still glowing
 * long after it stopped being news. */
const RATE_WINDOW = 3000;

/* Kinds that mean something has gone wrong. Any of them puts the orb into
 * its alert state regardless of the overall rate -- a halt is not a quiet
 * moment just because nothing else is happening. */
const ALERT_KINDS = { halt: true };

/* How long an alert holds after the last one, in ms. The melt has to last
 * long enough to be seen and understood; a halt that flashed for 400ms and
 * recovered would be missed by anybody not staring at the panel. */
const ALERT_HOLD = 12000;

export class OrbBridge {
  constructor(canvas, options) {
    const opts = options || {};
    this.orb = new ProcessOrb(canvas, opts);
    this.palette = window.Palette;
    this.seenSeq = 0;
    this.droppedReplay = 0;
    this.stamps = {};          // kind -> array of arrival times
    // -Infinity, not 0. performance.now() starts near zero on a fresh page,
    // so `now - 0 < ALERT_HOLD` is true for the first twelve seconds and the
    // orb booted into its melting alert state having seen nothing at all.
    this.lastAlert = -Infinity;
    this.pulseRate = 0;
    this._allStamps = [];
    this.debug = null;
    this.debugValues = null;

    const order = (this.palette && this.palette.KIND_ORDER) || [];
    this.kinds = order.slice(0, MAX_PROCESSES);
    this.kinds.forEach((k) => { this.stamps[k] = []; });
    this._push();
  }

  /* One snapshot from the websocket. Returns how many pulses were new, which
   * is what the panel's own readout reports. */
  ingest(pulses) {
    let fresh = [];
    for (let i = 0; i < (pulses || []).length; i++) {
      // Dedupe on the monotonic sequence number, so overlapping snapshot
      // windows deliver each event exactly once and a dropped frame costs
      // nothing.
      if (pulses[i].seq > this.seenSeq) fresh.push(pulses[i]);
    }
    if (!fresh.length) { this._expire(); this._push(); return 0; }
    this.seenSeq = fresh[fresh.length - 1].seq;

    // A backgrounded tab may have missed thousands. Replaying all of them
    // would spend a second of animation on events that are already history.
    if (fresh.length > MAX_REPLAY) {
      this.droppedReplay += fresh.length - MAX_REPLAY;
      fresh = fresh.slice(-MAX_REPLAY);
    }

    const now = performance.now();
    for (const p of fresh) {
      const kind = this.stamps[p.kind] ? p.kind : 'scan';
      this.stamps[kind].push(now);
      this._allStamps.push(now);
      this.orb.event(kind, p.symbol, p.intensity);
      if (ALERT_KINDS[p.kind]) this.lastAlert = now;
    }
    this._expire();
    this._push();
    return fresh.length;
  }

  _expire() {
    const cut = performance.now() - RATE_WINDOW;
    for (const kind of this.kinds) {
      const arr = this.stamps[kind];
      while (arr.length && arr[0] < cut) arr.shift();
    }
    while (this._allStamps.length && this._allStamps[0] < cut) {
      this._allStamps.shift();
    }
    this.pulseRate = this._allStamps.length / (RATE_WINDOW / 1000);
  }

  /* Rate -> activity, and hand the whole set to the orb. */
  _push() {
    if (this.debugValues) { this._pushDebug(); return; }
    if (this.manual) return;
    const seconds = RATE_WINDOW / 1000;
    const list = this.kinds.map((kind) => ({
      id: kind,
      label: (this.palette.KIND_LABEL || {})[kind] || kind,
      color: this.palette.css(kind),
      activity: activityFor(kind, this.stamps[kind].length, seconds)
    }));
    this.orb.setProcesses(list);

    const alerting = performance.now() - this.lastAlert < ALERT_HOLD;
    this.orb.setMode(this.forced || (alerting ? 'alert' : null));
  }

  /* Halt state arrives on the snapshot as well as through pulses, because a
   * book that halted before this tab connected emitted its pulse to nobody. */
  setHalted(halted) {
    if (halted) this.lastAlert = performance.now();
    else this.lastAlert = -Infinity;
  }

  /* The component's own API, forwarded. window.orb is this bridge, so
   * setProcesses/setMode/dispose have to work here or the documented surface
   * is a surface nobody can reach. Setting processes by hand parks the live
   * feed, the same way the debug panel does -- otherwise the next snapshot
   * would overwrite the caller a fraction of a second later, which looks
   * exactly like the call having been ignored. */
  setProcesses(list) {
    this.manual = true;
    this.orb.setProcesses(list);
  }

  setMode(mode) {
    this.forced = mode || null;
    this.orb.setMode(this.forced);
  }

  resumeLive() {
    this.manual = false;
    this.forced = null;
    this._push();
  }

  frame(now) { this.orb.frame(now); }
  resize() { this.orb.resize(); }
  dispose() {
    if (this.debug) { this.debug.destroy(); this.debug = null; }
    this.orb.dispose();
  }

  /* What the panel's caption says. The same three facts the field it replaced
   * put in its caption, so nothing an operator was reading has gone away. */
  caption() {
    const tier = this.orb.tierIndex;
    const names = ['high', 'medium', 'low', 'floor'];
    const dominant = this.orb.dominantId;
    let text = this.pulseRate.toFixed(1) + ' pulses/s · ' + this.orb.mode;
    if (dominant) text += ' · ' + dominant;
    if (tier > 0) text += ' · quality ' + names[tier];
    if (this.droppedReplay) text += ' · ' + this.droppedReplay + ' replayed';
    return text;
  }

  // -- the dev-only debug panel -------------------------------------------

  /* Sliders for every process and a mode selector, so every state can be seen
   * without waiting for the market to produce one.
   *
   * Loaded on demand and only when asked for: lil-gui is 30KB that an
   * operator never needs, and a debug panel that ships open is a debug panel
   * somebody ships by accident. Enable with ?debug=orb or Ctrl+Shift+O.
   */
  async openDebug() {
    if (this.debug) return;
    const { default: GUI } = await import('./vendor/lil-gui.module.js');
    const gui = new GUI({ title: 'process orb' });
    this.debug = gui;

    const values = { mode: 'auto', _live: false };
    this.kinds.forEach((k) => { values[k] = 0; });
    this.debugValues = values;

    gui.add(values, 'mode',
            ['auto', 'idle', 'active', 'intense', 'alert'])
       .onChange(() => this._pushDebug());
    this.kinds.forEach((kind) => {
      gui.add(values, kind, 0, 1, 0.01).onChange(() => this._pushDebug());
    });
    gui.add({ burst: () => {
      // Fire ripples from real tickers, so the surface behaviour under load
      // can be seen without a live feed.
      const syms = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'XLF', 'XLK'];
      for (let i = 0; i < 10; i++) {
        this.orb.event(this.kinds[i % this.kinds.length],
                       syms[i % syms.length], 0.8);
      }
    } }, 'burst').name('fire 10 ripples');
    gui.add({ live: () => {
      this.debugValues = null;
      gui.destroy();
      this.debug = null;
      this._push();
    } }, 'live').name('back to live data');

    this._pushDebug();
  }

  _pushDebug() {
    const v = this.debugValues;
    this.orb.setProcesses(this.kinds.map((kind) => ({
      id: kind,
      label: (this.palette.KIND_LABEL || {})[kind] || kind,
      color: this.palette.css(kind),
      activity: v[kind]
    })));
    this.orb.setMode(v.mode === 'auto' ? null : v.mode);
  }
}

/* Mount it. app.js is a classic script and cannot import, so the instance is
 * published on window and a flag says it is ready -- app.js checks the flag
 * every frame rather than racing the module's load. */
function mount() {
  const canvas = document.getElementById('orb');
  if (!canvas) return;
  try {
    const bridge = new OrbBridge(canvas);
    window.orb = bridge;
    window.__orbReady = true;
    window.addEventListener('resize', () => bridge.resize());

    const wantsDebug = /(\?|&)debug=orb\b/.test(location.search);
    if (wantsDebug) bridge.openDebug();
    window.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'O' || e.key === 'o')) {
        e.preventDefault();
        bridge.openDebug();
      }
    });
  } catch (err) {
    // WebGL can be absent, blocked, or software-emulated into uselessness.
    // The terminal must still work: every other panel carries the same facts
    // in text, so the orb failing is a missing ornament, not a broken tool.
    window.__orbError = String((err && err.message) || err);
    const note = document.getElementById('cluster-note');
    if (note) note.textContent = 'the orb could not start: ' + window.__orbError;
    if (canvas) canvas.style.display = 'none';
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mount);
} else {
  mount();
}
