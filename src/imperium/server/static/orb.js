/* The process orb: what this terminal is doing, as one living object.
 *
 * It replaces the 2D pulse field that used to sit here, and it has to carry
 * everything that field carried, because an operator glancing at the centre
 * of the screen was reading three things off it at once:
 *
 *   what kind of work is happening   -> the core's colour zones, one per kind
 *   which symbol just fired          -> a ripple at that symbol's own point
 *   how hard it is working           -> morph state, heartbeat rate, bloom
 *
 * The third of those is why the orb is a body and not a chart. A number
 * telling you the scan rate is something you have to read; a thing that
 * breathes faster is something you notice while looking at something else,
 * and this panel is looked at out of the corner of an eye for hours.
 *
 * ARCHITECTURE. Three layers over one shared displacement field (see
 * orb.shaders.js): a glossy liquid core, a frosted glass shell that refracts
 * it, and a point-cloud skin just outside. They deform together because they
 * evaluate the same function, not because their animations are kept in step.
 *
 * EVERY INPUT IS DAMPED. Live telemetry is spiky -- a scan burst can take the
 * rate from two to forty in one frame -- and a body that tracked that exactly
 * would twitch. Activity, colour, mode weights and heartbeat rate are all
 * critically damped towards their targets, so the data can be as jumpy as it
 * likes and the orb still moves like something alive.
 *
 * API
 *   orb.setProcesses([{ id, label, color, activity }])   colour + zone sizes
 *   orb.setMode('idle'|'active'|'intense'|'alert'|null)  null = automatic
 *   orb.event(kind, symbol, intensity)                   one surface ripple
 *   orb.frame(nowMs)                                     drive from the host loop
 *   orb.dispose()
 */
import * as THREE from './vendor/three/three.module.js';
import { EffectComposer } from './vendor/three/postprocessing/EffectComposer.js';
import { RenderPass } from './vendor/three/postprocessing/RenderPass.js';
import { UnrealBloomPass } from './vendor/three/postprocessing/UnrealBloomPass.js';
import {
  NOISE, DISPLACE, CORE_VERT, CORE_FRAG, SHELL_VERT, SHELL_FRAG,
  SKIN_VERT, SKIN_FRAG, SHADOW_VERT, SHADOW_FRAG
} from './orb.shaders.js';
import {
  clamp, damp, nextMode, fibonacciSphere, symbolDirection as dirFor,
  heartbeatAt, heartRate
} from './orb.math.js';

/* Hard ceilings, compiled into the shaders as #defines.
 *
 * GLSL loops need a constant bound, so these are limits rather than
 * preferences. MAX_PROCESSES is seven because that is how many pulse kinds
 * the server can emit; a test asserts the two agree, because a kind added
 * server-side and not here would render with no colour at all. */
export const MAX_PROCESSES = 8;
export const MAX_RIPPLES = 12;

/* Quality tiers. The orb measures its own frame times and steps down through
 * these, exactly as the pulse field it replaces did -- the lag on this
 * operator's machine was never reproducible here, so the honest answer is a
 * panel that finds its own level rather than a number I guessed. */
const TIERS = [
  { name: 'high',   detail: 64, points: 14000, bloom: true,  shells: 2 },
  { name: 'medium', detail: 48, points: 7000,  bloom: true,  shells: 2 },
  { name: 'low',    detail: 32, points: 3000,  bloom: true,  shells: 1 },
  { name: 'floor',  detail: 20, points: 0,     bloom: false, shells: 1 }
];

/* Frame budget. A tier is dropped after sustained slow frames and regained
 * only after a long clean run: quick to cut, slow to relax, because a panel
 * that recovers eagerly oscillates between two tiers and the flicker is worse
 * than simply running at the lower one. */
const SLOW_FRAME_MS = 22;
const SLOW_STREAK_TO_DROP = 45;
const FAST_STREAK_TO_RAISE = 600;

const MODE_BLEND_SECONDS = 1.5;

/* How quickly damped values chase their targets, in "fraction of the gap per
 * second". Colour is slower than activity on purpose: a dominance change is
 * meant to read as ink spreading through water, which takes about a second. */
const RATE_ACTIVITY = 3.2;
const RATE_COLOR = 1.1;
const RATE_MODE = 1.0 / MODE_BLEND_SECONDS;

const RIPPLE_SECONDS = 1.6;

/* The shared hash, as a three vector. The maths lives in orb.math.js so it
 * can be tested without a GPU; this is the only place it meets three. */
function symbolDirection(symbol) {
  const d = dirFor(symbol);
  return new THREE.Vector3(d.x, d.y, d.z);
}

function prefersReducedMotion() {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch (e) {
    return false;
  }
}

export class ProcessOrb {
  constructor(canvas, options) {
    const opts = options || {};
    this.canvas = canvas;
    this.disposed = false;
    this.reduced = prefersReducedMotion();

    this.processes = [];
    this.forcedMode = null;
    this.mode = 'idle';
    this.modeWeights = { idle: 1, active: 0, intense: 0, alert: 0 };
    this.totalActivity = 0;
    this.dominantId = null;

    this.heartPhase = 0;
    this.heartbeat = 0;
    this.bobPhase = Math.random() * 6.28;
    this.spinAxis = new THREE.Vector3(0.18, 1, 0.07).normalize();
    this.spin = 0;
    this.axisDrift = 0;

    this.ripples = [];
    this.tierIndex = this.reduced ? 2 : 0;
    this.slowStreak = 0;
    this.fastStreak = 0;
    this.lastFrame = 0;
    this.fps = 60;

    this._initRenderer(opts);
    this._initScene();
    this._build();
    this.resize();
  }

  // -- setup ---------------------------------------------------------------

  _initRenderer(opts) {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: false,      // bloom hides the aliasing and MSAA is not free
      alpha: true,           // the panel's own background shows through
      powerPreference: 'high-performance'
    });
    // Capped at 2: beyond that the pixel count doubles again for a difference
    // nobody can see on a panel this size.
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.86;
    this.background = opts.background || null;
  }

  _initScene() {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    this.camera.position.set(0, 0.30, 6.4);
    this.camera.lookAt(0, 0, 0);

    this.root = new THREE.Group();
    this.scene.add(this.root);

    // Lighting for the glass. Key from upper front, a cool fill opposite, and
    // a rim behind so the shell's edge separates from a black background --
    // without the rim a dark orb on a dark panel loses its silhouette.
    this.key = new THREE.DirectionalLight(0xffffff, 0.42);
    this.key.position.set(2.4, 3.2, 2.6);
    this.fill = new THREE.DirectionalLight(0x8fa0b8, 0.30);
    this.fill.position.set(-2.8, -0.6, 1.4);
    this.rim = new THREE.DirectionalLight(0xffffff, 0.55);
    this.rim.position.set(-1.2, 0.8, -3.0);
    this.ambient = new THREE.AmbientLight(0x2a2f38, 0.35);
    this.scene.add(this.key, this.fill, this.rim, this.ambient);

    this.composer = null;
    this.bloom = null;
  }

  /* Uniforms shared by every layer. One object, referenced by all three
   * materials, so there is exactly one place a morph weight is written and no
   * way for the layers to disagree about the shape they are drawing. */
  _sharedUniforms() {
    const ripples = [];
    for (let i = 0; i < MAX_RIPPLES; i++) ripples.push(new THREE.Vector4(0, 1, 0, 0));
    return {
      uTime: { value: 0 },
      uIdle: { value: 1 },
      uActive: { value: 0 },
      uIntense: { value: 0 },
      uAlert: { value: 0 },
      uBreath: { value: 0 },
      uFlow: { value: 0.5 },
      uAmp: { value: this.reduced ? 0.35 : 1 },
      uRipples: { value: ripples },
      uRippleStrength: { value: this.reduced ? 0.02 : 0.055 },
      uShellFollow: { value: 0.92 }
    };
  }

  _defines() {
    return {
      MAX_PROCESSES: MAX_PROCESSES,
      MAX_RIPPLES: MAX_RIPPLES
    };
  }

  _build() {
    const tier = TIERS[this.tierIndex];
    this.shared = this.shared || this._sharedUniforms();

    const geo = new THREE.IcosahedronGeometry(1, tier.detail);

    // -- core
    //
    // The zone uniforms are built once and reused across rebuilds, not
    // recreated here. A quality-tier change rebuilds every mesh, and when
    // these were rebuilt with them the orb lost its dominant colour, its
    // damped zone weights and its attractor positions -- so stepping down a
    // tier on a slow machine flashed the body back to grey and re-converged
    // in front of the operator, which is exactly the moment it should be
    // drawing least attention to itself.
    if (!this.coreUniforms) {
      this.coreUniforms = {
        uCount: { value: 0 },
        uColors: { value: this._fillVectors(MAX_PROCESSES,
                     () => new THREE.Color(0.2, 0.2, 0.24)) },
        uWeights: { value: new Array(MAX_PROCESSES).fill(0) },
        uSeeds: { value: this._fillVectors(MAX_PROCESSES,
                    () => new THREE.Vector3(0, 1, 0)) },
        uDominant: { value: new THREE.Color(0.85, 0.89, 0.93) },
        uHeartbeat: { value: 0 },
        uGlow: { value: 1 }
      };
    }
    const coreUniforms = Object.assign({}, this.shared, this.coreUniforms);
    this.coreMat = new THREE.ShaderMaterial({
      uniforms: coreUniforms,
      defines: this._defines(),
      vertexShader: NOISE + DISPLACE + CORE_VERT,
      fragmentShader: NOISE + CORE_FRAG
    });
    this.core = new THREE.Mesh(geo, this.coreMat);
    this.core.scale.setScalar(0.86);
    this.core.renderOrder = 1;
    this.root.add(this.core);

    // -- shell: two membranes, so the folds overlap
    //
    // Both double-sided with no depth write, so a fold shows the layer behind
    // it instead of culling into a hole. They share the core's zone uniforms,
    // which is how the colour inside appears through the glass without a
    // second render of the scene.
    this.shells = [];
    const shellRadii = tier.shells > 1 ? [1.0, 1.085] : [1.02];
    const shellAlphas = tier.shells > 1 ? [0.30, 0.16] : [0.36];
    for (let i = 0; i < shellRadii.length; i++) {
      const uniforms = Object.assign({}, this.shared, {
        uCount: this.coreUniforms.uCount,
        uColors: this.coreUniforms.uColors,
        uWeights: this.coreUniforms.uWeights,
        uSeeds: this.coreUniforms.uSeeds,
        uDominant: this.coreUniforms.uDominant,
        uShellFollow: { value: i === 0 ? 0.88 : 0.62 },
        uShellRadius: { value: shellRadii[i] },
        uShellAlpha: { value: shellAlphas[i] },
        uIor: { value: 1.4 }
      });
      const mat = new THREE.ShaderMaterial({
        uniforms: uniforms,
        defines: this._defines(),
        vertexShader: NOISE + DISPLACE + SHELL_VERT,
        fragmentShader: NOISE + SHELL_FRAG,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        blending: THREE.NormalBlending
      });
      const mesh = new THREE.Mesh(geo.clone(), mat);
      mesh.renderOrder = 2 + i;
      this.root.add(mesh);
      this.shells.push(mesh);
    }
    // Kept for the isolation harness and for anything that reaches for "the"
    // shell; the outer membrane is the one that reads as the surface.
    this.shell = this.shells[0];

    // -- particle skin
    this.skin = null;
    if (tier.points > 0) {
      const { positions, phases } = fibonacciSphere(tier.points);
      const skinGeo = new THREE.BufferGeometry();
      skinGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      skinGeo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));
      this.skinUniforms = Object.assign({}, this.shared, {
        uSize: { value: 1.35 },
        uPixelRatio: { value: this.renderer.getPixelRatio() },
        uSkinOffset: { value: 1.07 },
        uSkinFollow: { value: 0.38 },
        uTint: { value: new THREE.Color(0.62, 0.68, 0.78) },
        uOpacity: { value: 0.16 }
      });
      this.skinMat = new THREE.ShaderMaterial({
        uniforms: this.skinUniforms,
        defines: this._defines(),
        vertexShader: NOISE + DISPLACE + SKIN_VERT,
        fragmentShader: SKIN_FRAG,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending
      });
      this.skin = new THREE.Points(skinGeo, this.skinMat);
      this.skin.renderOrder = 5;
      this.root.add(this.skin);
    }

    // -- contact shadow
    if (!this.shadow) {
      this.shadowUniforms = {
        uOpacity: { value: 0.55 },
        uSpread: { value: 1.0 }
      };
      this.shadowMat = new THREE.ShaderMaterial({
        uniforms: this.shadowUniforms,
        vertexShader: SHADOW_VERT,
        fragmentShader: SHADOW_FRAG,
        transparent: true,
        depthWrite: false
      });
      this.shadow = new THREE.Mesh(new THREE.PlaneGeometry(3.6, 3.6), this.shadowMat);
      this.shadow.rotation.x = -Math.PI / 2;
      this.shadow.position.y = -1.45;
      this.scene.add(this.shadow);
    }

    this._buildComposer(tier);
    this._pushProcessUniforms();
  }

  _fillVectors(n, make) {
    const out = [];
    for (let i = 0; i < n; i++) out.push(make());
    return out;
  }

  _buildComposer(tier) {
    if (this.composer) { this.composer.dispose(); this.composer = null; }
    if (!tier.bloom) { this.bloom = null; return; }
    const size = this.renderer.getSize(new THREE.Vector2());
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(size, 0.30, 0.55, 0.78);
    this.composer.addPass(this.bloom);
    this.composer.setPixelRatio(this.renderer.getPixelRatio());
    this.composer.setSize(Math.max(1, size.x), Math.max(1, size.y));
  }

  // -- public API ----------------------------------------------------------

  /* The processes to draw, each 0..1.
   *
   * Values are targets, not positions: nothing here moves the orb this frame.
   * That is what lets the caller push jittery live telemetry straight in --
   * the damping between target and current is the whole reason the body reads
   * as alive rather than as a bar chart in a sphere. */
  setProcesses(list) {
    const incoming = (list || []).slice(0, MAX_PROCESSES);
    const byId = {};
    this.processes.forEach((p) => { byId[p.id] = p; });

    this.processes = incoming.map((entry, index) => {
      const existing = byId[entry.id];
      const target = clamp(Number(entry.activity) || 0, 0, 1);
      const color = new THREE.Color(entry.color || '#d6e0ee');
      return {
        id: entry.id,
        label: entry.label || entry.id,
        target: target,
        // A brand new process starts where it is rather than swelling from
        // nothing; one that was already here keeps its damped value, so a
        // re-send of the same list never makes the orb jump.
        value: existing ? existing.value : target,
        color: color,
        shown: existing ? existing.shown : color.clone(),
        seed: existing ? existing.seed : symbolDirection(entry.id + '::zone'),
        index: index
      };
    });
    this._pushProcessUniforms();
  }

  /* Force a mode, or pass null to hand control back to the activity level. */
  setMode(mode) {
    this.forcedMode = mode || null;
    if (mode) this.mode = mode;
  }

  /* One process event: a ripple from that symbol's own point on the surface.
   *
   * Dropped silently when all ripple slots are busy. Not queued: an event
   * that shows up two seconds after it happened is telling the operator
   * something false about now, and at a high pulse rate a queue would do
   * nothing but lag. */
  event(kind, symbol, intensity) {
    if (this.reduced) return;
    if (this.ripples.length >= MAX_RIPPLES) {
      // Reuse the oldest rather than refusing outright, so a sustained burst
      // still shows movement instead of freezing at twelve.
      this.ripples.shift();
    }
    const dir = symbolDirection(symbol || kind);
    this.ripples.push({
      dir: dir,
      age: 0,
      life: RIPPLE_SECONDS,
      strength: clamp(Number(intensity) || 0.5, 0.1, 1)
    });
  }

  /* Drive one frame. Takes the host's clock so the orb shares a timebase with
   * the rest of the page rather than running its own. */
  frame(nowMs) {
    if (this.disposed) return;
    const now = nowMs === undefined ? performance.now() : nowMs;
    let dt = this.lastFrame ? (now - this.lastFrame) / 1000 : 1 / 60;
    this.lastFrame = now;
    // A tab that was backgrounded returns a delta of minutes. Clamped, or the
    // orb integrates the whole gap in one step and snaps across the screen.
    dt = clamp(dt, 0.001, 0.1);

    this._observeFrame(now);
    this._stepActivity(dt);
    this._stepMode(dt);
    this._stepRipples(dt);
    this._stepBody(dt);
    this._render();
  }

  dispose() {
    this.disposed = true;
    this._disposeLayers();
    if (this.shadow) {
      this.shadow.geometry.dispose();
      this.shadowMat.dispose();
      this.scene.remove(this.shadow);
      this.shadow = null;
    }
    if (this.composer) { this.composer.dispose(); this.composer = null; }
    this.scene.traverse((o) => {
      if (o.isLight && o.dispose) o.dispose();
    });
    this.renderer.dispose();
    this.renderer.forceContextLoss();
  }

  _disposeLayers() {
    const meshes = [this.core, this.skin].concat(this.shells || []);
    meshes.forEach((mesh) => {
      if (!mesh) return;
      mesh.geometry.dispose();
      if (mesh.material.dispose) mesh.material.dispose();
      this.root.remove(mesh);
    });
    this.core = this.shell = this.skin = null;
    this.shells = [];
  }

  resize() {
    if (this.disposed) return;
    const w = Math.max(1, this.canvas.clientWidth || 1);
    const h = Math.max(1, this.canvas.clientHeight || 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.composer) {
      this.composer.setPixelRatio(this.renderer.getPixelRatio());
      this.composer.setSize(w, h);
    }
    if (this.skinUniforms) {
      this.skinUniforms.uPixelRatio.value = this.renderer.getPixelRatio();
    }
  }

  // -- per-frame steps -----------------------------------------------------

  _observeFrame(now) {
    const gap = this._lastObserved ? now - this._lastObserved : 16;
    this._lastObserved = now;
    this.fps = damp(this.fps, 1000 / Math.max(gap, 1), 2, 0.05);

    if (gap > SLOW_FRAME_MS) {
      this.slowStreak++;
      this.fastStreak = 0;
    } else {
      this.fastStreak++;
      this.slowStreak = 0;
    }

    if (this.slowStreak >= SLOW_STREAK_TO_DROP && this.tierIndex < TIERS.length - 1) {
      this.slowStreak = 0;
      this._setTier(this.tierIndex + 1);
    } else if (this.fastStreak >= FAST_STREAK_TO_RAISE && this.tierIndex > 0
               && !this.reduced) {
      this.fastStreak = 0;
      this._setTier(this.tierIndex - 1);
    }
  }

  _setTier(index) {
    this.tierIndex = clamp(index, 0, TIERS.length - 1);
    this._disposeLayers();
    this._build();
    this.resize();
  }

  _stepActivity(dt) {
    let total = 0;
    let best = null;
    for (const p of this.processes) {
      p.value = damp(p.value, p.target, RATE_ACTIVITY, dt);
      p.shown.lerp(p.color, 1 - Math.exp(-RATE_COLOR * dt));
      total += p.value;
      if (!best || p.value > best.value) best = p;
    }
    this.totalActivity = clamp(total, 0, 4);
    this.dominantId = best && best.value > 0.01 ? best.id : null;

    if (best) {
      const dom = this.coreUniforms.uDominant.value;
      dom.lerp(best.shown, 1 - Math.exp(-RATE_COLOR * dt));
      if (this.bloom) {
        // Bloom leans on the dominant process, so the glow around the orb is
        // the colour of whatever it is mostly doing.
        this.bloom.strength = damp(this.bloom.strength,
          0.18 + 0.42 * clamp(this.totalActivity, 0, 1.4), 1.4, dt);
      }
      this.rim.color.lerp(best.shown, 1 - Math.exp(-0.9 * dt));
      if (this.skinUniforms) {
        this.skinUniforms.uTint.value.lerp(best.shown, 1 - Math.exp(-0.7 * dt));
      }
    }
    this._pushProcessUniforms();
  }

  _pushProcessUniforms() {
    if (!this.coreUniforms) return;
    this.coreUniforms.uCount.value = this.processes.length;
    for (let i = 0; i < MAX_PROCESSES; i++) {
      const p = this.processes[i];
      if (p) {
        this.coreUniforms.uColors.value[i].copy(p.shown);
        this.coreUniforms.uWeights.value[i] = p.value;
        this.coreUniforms.uSeeds.value[i].copy(p.seed);
      } else {
        this.coreUniforms.uWeights.value[i] = 0;
      }
    }
  }

  /* Pick a mode from the activity level, with hysteresis.
   *
   * A band is left only when the value passes the *other* edge, so a reading
   * hovering on a boundary stays where it is instead of flipping between two
   * morph states several times a second. */
  _autoMode() {
    return nextMode(this.mode, this.totalActivity);
  }

  _stepMode(dt) {
    const wanted = this.forcedMode || this._autoMode();
    this.mode = wanted;
    const w = this.modeWeights;
    for (const key of ['idle', 'active', 'intense', 'alert']) {
      w[key] = damp(w[key], key === wanted ? 1 : 0, RATE_MODE * 3, dt);
    }
    const sum = w.idle + w.active + w.intense + w.alert || 1;
    this.shared.uIdle.value = w.idle / sum;
    this.shared.uActive.value = w.active / sum;
    this.shared.uIntense.value = w.intense / sum;
    this.shared.uAlert.value = w.alert / sum;

    // The noise field itself runs faster when there is more going on, so
    // "busy" is legible from the texture's motion and not only its shape.
    this.shared.uFlow.value = damp(this.shared.uFlow.value,
      0.35 + 0.85 * clamp(this.totalActivity, 0, 1.5), 1.2, dt);
  }

  _stepRipples(dt) {
    const slots = this.shared.uRipples.value;
    for (let i = this.ripples.length - 1; i >= 0; i--) {
      this.ripples[i].age += dt / this.ripples[i].life;
      if (this.ripples[i].age >= 1) this.ripples.splice(i, 1);
    }
    for (let i = 0; i < MAX_RIPPLES; i++) {
      const r = this.ripples[i];
      if (r) slots[i].set(r.dir.x, r.dir.y, r.dir.z, r.age);
      else slots[i].set(0, 1, 0, 0);
    }
  }

  /* Heartbeat, bob, spin -- the parts that must never stop.
   *
   * The heartbeat is a double pulse, lub-dub: one sharp beat, a short gap, a
   * softer second. A single sine reads as a pulsing light; two unequal beats
   * read as something with a chest. Its rate rises with activity, which is
   * the fastest way to feel how hard the terminal is working without reading
   * a single number. */
  _stepBody(dt) {
    const t = (this.shared.uTime.value += dt);

    const bpm = heartRate(this.totalActivity);
    this.heartPhase = (this.heartPhase + dt * (bpm / 60)) % 1;
    this.heartbeat = heartbeatAt(this.heartPhase);
    this.coreUniforms.uHeartbeat.value = this.heartbeat;

    // Breathing is the idle state's whole signal of life, so it is strongest
    // there and mostly swamped by the noise once things get busy.
    const breath = Math.sin(t * (Math.PI * 2) / 4) * 0.02 * this.shared.uIdle.value;
    this.shared.uBreath.value = breath + this.heartbeat * 0.012;

    // A drifting axis, so the rotation never settles into an obvious loop.
    this.axisDrift += dt * 0.05;
    this.spinAxis.set(
      Math.sin(this.axisDrift * 0.7) * 0.25,
      1,
      Math.cos(this.axisDrift * 0.5) * 0.2
    ).normalize();
    this.spin += dt * (0.10 + 0.22 * clamp(this.totalActivity, 0, 1.2));
    this.root.quaternion.setFromAxisAngle(this.spinAxis, this.spin);

    this.bobPhase += dt;
    const bob = this.reduced ? 0 : Math.sin(this.bobPhase * 0.8) * 0.055;
    this.root.position.y = bob;

    // The shadow tracks the float: tighter and darker when the orb is low,
    // wider and fainter when it rises.
    if (this.shadowUniforms) {
      this.shadowUniforms.uSpread.value = 0.86 + bob * 1.6;
      this.shadowUniforms.uOpacity.value = 0.50 - bob * 0.9;
    }

    // Alert melts downward, so the body sinks a little as it does.
    this.root.position.y -= this.shared.uAlert.value * 0.12;
  }

  _render() {
    if (this.composer) this.composer.render();
    else this.renderer.render(this.scene, this.camera);
  }
}

export default ProcessOrb;
