/* The neural cluster.
 *
 * Neurons (one per position), pathways (the watched universe wired to a central
 * hub) and orbs (one per pulse, travelling along the pathways).
 *
 * Four things that look wrong if they are skipped, all handled here:
 *
 * 1. Every orb's speed and starting offset is jittered. Identical speed from an
 *    identical start makes a burst on one symbol travel as a rigid body — ten
 *    orbs stacked into one elongated smear that reads as a rendering artefact
 *    rather than as ten events.
 * 2. Depth varies. Some orbs near and sharp, some far and dim. A field where
 *    every orb is the same size and brightness looks like stickers on glass
 *    however well each one is drawn.
 * 3. The orb budget is measured against the *bloom* radius, not the 2px core.
 *    Bound on the core and the field saturates into a wash at any real pulse
 *    rate, because what fills the panel is the glow.
 * 4. The replay queue is bounded. A backgrounded tab may have missed thousands
 *    of pulses, and replaying them all is a stampede describing work already
 *    missed.
 *
 * Neurons and the pathway network are cached as sprites and redrawn only when
 * the universe changes; per-frame work is orbs only.
 */

(function (global) {
  'use strict';

  /* One colour per pulse kind, and they have to be told apart at a glance in a
   * field of several hundred. The three that matter most to an operator are
   * the loudest: a decision was reached, the cost cap refused it, an order
   * went out. Scanning stays the quiet blue underneath them -- it is the
   * background hum, and at twenty pulses a second anything brighter would
   * drown the three events worth looking at. */
  var KIND_COLOR = {
    scan:     [53, 167, 255],     // blue — the background hum
    decision: [57, 255, 140],     // neon green — a verdict was reached
    refused:  [107, 123, 145],    // grey — looked at, nothing there
    cap:      [190, 60, 255],     // neon purple — the cost gate said no
    order:    [255, 150, 40],     // bright orange — money actually moved
    warmup:   [232, 180, 68],
    halt:     [255, 92, 108]
  };

  /* The visible radius of an orb is its bloom, not its core. The budget below
   * is computed against this, which is what actually fills the panel. */
  var CORE_RADIUS = 2.0;
  var BLOOM_MULTIPLE = 5.5;
  /* Fraction of the panel the bloom discs may cover before the field reads as a
   * wash rather than as distinct events. */
  var COVERAGE_LIMIT = 0.22;
  var MIN_ORBS = 40;
  var MAX_ORBS = 900;
  /* A backgrounded tab wakes with a large seq gap; replaying it all is a
   * stampede describing work already missed. */
  var MAX_REPLAY = 120;

  function rnd(a, b) { return a + Math.random() * (b - a); }

  function cssColor(kind) {
    var c = KIND_COLOR[kind];
    return c ? 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')' : '';
  }

  /* Paint the legend from this palette, so the two cannot drift apart. */
  function paintLegend(root) {
    if (!root) return 0;
    var spans = root.querySelectorAll('[data-kind]'), painted = 0;
    for (var i = 0; i < spans.length; i++) {
      var dot = spans[i].querySelector('i');
      var col = cssColor(spans[i].getAttribute('data-kind'));
      if (dot && col) { dot.style.background = col; painted++; }
    }
    return painted;
  }

  function Cluster(canvas, tooltipEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.tooltip = tooltipEl;
    this.dpr = Math.min(global.devicePixelRatio || 1, 2);
    this.w = 0; this.h = 0;

    this.symbols = [];
    this.nodes = {};          // symbol -> {x, y, angle, radius}
    this.hub = { x: 0, y: 0 };
    this.paths = {};          // symbol -> [{x, y}, ...] control points
    /* The network is built as several sprite layers rather than one image so
     * each can drift on its own phase. One image sliding about reads as a
     * picture being moved; layers at different phases read as depth. */
    this.networkLayers = null;
    //: Allocated once and redrawn. Re-creating these was a 193ms stall
    //: every time the cohort rotated. See buildNetworkSprite.
    this._layerPool = null;
    this.neuronSprites = {};  // cache key -> canvas
    this.positions = {};      // symbol -> weight (drives neurons)
    /* How many times each symbol has been looked at, and what the last look
     * concluded. The ring grows out of this: a symbol the scanner has been
     * over many times is drawn as something larger and more alive than one it
     * has never reached, so the field visibly matures as the sweep works
     * through the universe instead of looking identical at minute one and
     * hour six. */
    this.scans = {};
    this.lastKind = {};
    this.lastPulseAt = {};
    this.totalPulses = 0;

    this.orbs = [];
    this.seenSeq = 0;         // monotonic dedupe: overlapping windows are free
    this.hover = null;
    this.mouse = { x: -1, y: -1, inside: false };
    this.lastFrame = performance.now();
    this.pulseRate = 0;
    this._pulseTimes = [];

    var self = this;
    canvas.addEventListener('mousemove', function (e) {
      var r = canvas.getBoundingClientRect();
      self.mouse.x = e.clientX - r.left;
      self.mouse.y = e.clientY - r.top;
      self.mouse.inside = true;
    });
    canvas.addEventListener('mouseleave', function () {
      self.mouse.inside = false;
      self.hover = null;
      if (self.tooltip) self.tooltip.hidden = true;
    });
  }

  Cluster.prototype.orbBudget = function () {
    var bloom = CORE_RADIUS * BLOOM_MULTIPLE;
    var discArea = Math.PI * bloom * bloom;
    var panelArea = Math.max(1, this.w * this.h);
    var budget = Math.floor((panelArea * COVERAGE_LIMIT) / discArea);
    return Math.max(MIN_ORBS, Math.min(MAX_ORBS, budget));
  };

  Cluster.prototype.resize = function () {
    var rect = this.canvas.getBoundingClientRect();
    var w = Math.max(80, Math.floor(rect.width));
    var h = Math.max(80, Math.floor(rect.height));
    if (w === this.w && h === this.h) return;
    this.w = w; this.h = h;
    this.canvas.width = Math.floor(w * this.dpr);
    this.canvas.height = Math.floor(h * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.layout();
  };

  /* ---------- the pathway network ---------- */

  Cluster.prototype.setUniverse = function (symbols) {
    var same = symbols.length === this.symbols.length &&
      symbols.every(function (s, i) { return s === this.symbols[i]; }, this);
    if (same) return;
    this.symbols = symbols.slice();
    this.layout();
  };

  Cluster.prototype.layout = function () {
    if (!this.w || !this.symbols.length) { this.networkLayers = null; return; }
    var cx = this.w * 0.5, cy = this.h * 0.5;
    this.hub = { x: cx, y: cy };
    var rx = Math.min(this.w * 0.42, this.h * 0.86);
    var ry = this.h * 0.40;
    var n = this.symbols.length;

    this.nodes = {};
    this.paths = {};
    for (var i = 0; i < n; i++) {
      var sym = this.symbols[i];
      var a = (i / n) * Math.PI * 2 - Math.PI / 2;
      /* A deterministic per-symbol wobble, so the ring is organic rather than a
       * perfect circle, but stable across frames and reloads. */
      var seed = hashString(sym);
      var jitter = 0.86 + ((seed % 100) / 100) * 0.28;
      var node = {
        x: cx + Math.cos(a) * rx * jitter,
        y: cy + Math.sin(a) * ry * jitter,
        angle: a,
        radius: 3 + (seed % 3)
      };
      this.nodes[sym] = node;
      this.paths[sym] = growPath(node, this.hub, seed);
    }
    this.networkLayers = null;   // regenerate on the next frame
  };

  function hashString(s) {
    var h = 2166136261;
    for (var i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = (h * 16777619) >>> 0;
    }
    return h >>> 0;
  }

  /* Grow a branching filament from a node toward the hub. Procedural, seeded by
   * the symbol, so the network is stable but not geometric. */
  function growPath(node, hub, seed) {
    var pts = [{ x: node.x, y: node.y }];
    var steps = 5;
    var rand = mulberry(seed);
    for (var i = 1; i < steps; i++) {
      var t = i / steps;
      var bx = node.x + (hub.x - node.x) * t;
      var by = node.y + (hub.y - node.y) * t;
      /* Deviate most in the middle, so the filament leaves the node and enters
       * the hub cleanly and bows in between. */
      var bow = Math.sin(t * Math.PI) * 34;
      pts.push({
        x: bx + (rand() - 0.5) * bow,
        y: by + (rand() - 0.5) * bow
      });
    }
    pts.push({ x: hub.x, y: hub.y });
    return pts;
  }

  function mulberry(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Position along a path, as a Catmull-Rom-ish walk through the control points. */
  function pointAt(pts, t) {
    var n = pts.length - 1;
    var scaled = Math.max(0, Math.min(0.9999, t)) * n;
    var i = Math.floor(scaled);
    var f = scaled - i;
    var p0 = pts[Math.max(0, i - 1)], p1 = pts[i];
    var p2 = pts[Math.min(n, i + 1)], p3 = pts[Math.min(n, i + 2)];
    var f2 = f * f, f3 = f2 * f;
    return {
      x: 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * f +
        (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * f2 +
        (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * f3),
      y: 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * f +
        (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * f2 +
        (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * f3)
    };
  }

  //: How many independent drifting layers the pathway network is split across.
  //
  // Three, not one: a single sprite nudged each frame is a picture being slid
  // about, which the eye reads as exactly that. Split across layers that drift
  // on different phases, the parallax between them reads as depth and the
  // filaments look suspended rather than painted on.
  var NETWORK_LAYERS = 3;

  Cluster.prototype.buildNetworkSprite = function () {
    /* The canvases are allocated once and redrawn, never re-created.
     *
     * This was the half-second freeze. The cohort rotates every twenty
     * seconds, which changes the symbol list, which invalidated the sprite --
     * and rebuilding it allocated three fresh canvases. Measured at a
     * realistic 1100x700 on a 2x display: allocating the three costs 193ms,
     * clearing them costs 0ms, and drawing all hundred and fifty filaments
     * into them costs 7ms. The work was never the drawing. It was asking the
     * browser for sixteen megapixels of backing store, twice a minute, on the
     * thread that also paints the frame.
     */
    var pw = Math.max(1, Math.floor(this.w * this.dpr));
    var ph = Math.max(1, Math.floor(this.h * this.dpr));
    var layers = this._layerPool;
    if (!layers || layers.length !== NETWORK_LAYERS ||
        layers[0].canvas.width !== pw || layers[0].canvas.height !== ph) {
      layers = [];
      for (var L = 0; L < NETWORK_LAYERS; L++) {
        var c = document.createElement('canvas');
        c.width = pw;
        c.height = ph;
        layers.push({ canvas: c, ctx: c.getContext('2d'),
                      // A phase and a rate per layer, so no two drift together.
                      phase: L * 2.2, rate: 0.055 + L * 0.021,
                      ax: 3.5 + L * 1.6, ay: 2.4 + L * 1.1 });
      }
      this._layerPool = layers;
    }
    for (var R = 0; R < layers.length; R++) {
      // setTransform resets on a resize; re-applied here because clearRect
      // and every draw below are in CSS pixels.
      layers[R].ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      layers[R].ctx.clearRect(0, 0, this.w, this.h);
    }

    for (var i = 0; i < this.symbols.length; i++) {
      var sym = this.symbols[i];
      var pts = this.paths[sym];
      if (!pts) continue;
      var lg = layers[i % NETWORK_LAYERS].ctx;
      /* Two passes: a wide dim halo, then a thin bright core. One pass with a
       * shadow blur is far more expensive and looks flatter. */
      lg.strokeStyle = 'rgba(40, 92, 140, 0.16)';
      lg.lineWidth = 2.4;
      strokePath(lg, pts);
      lg.strokeStyle = 'rgba(90, 170, 235, 0.30)';
      lg.lineWidth = 0.7;
      strokePath(lg, pts);
      /* The ring node is NOT baked in any more -- it is drawn per frame from
       * how often the scanner has been over that symbol. See drawNodes. */
    }

    // The hub, on the middle layer.
    var hgctx = layers[Math.floor(NETWORK_LAYERS / 2)].ctx;
    var hg = hgctx.createRadialGradient(this.hub.x, this.hub.y, 0,
                                        this.hub.x, this.hub.y, 26);
    hg.addColorStop(0, 'rgba(124, 224, 255, 0.30)');
    hg.addColorStop(1, 'rgba(124, 224, 255, 0)');
    hgctx.fillStyle = hg;
    hgctx.beginPath();
    hgctx.arc(this.hub.x, this.hub.y, 26, 0, Math.PI * 2);
    hgctx.fill();

    this.networkLayers = layers;
  };

  /* How mature a symbol looks, 0..1, from how many times it has been scanned.
   *
   * Logarithmic on purpose. Linear growth would have the first cohort dwarf
   * everything that follows within a minute and then stop meaning anything;
   * on a log curve the difference between one look and ten is as visible as
   * between ten and a hundred, which is the comparison an operator actually
   * makes. */
  function maturity(count) {
    if (!count) return 0;
    // Saturates over a few hundred looks rather than a few dozen, so the field
    // keeps growing across a session instead of reaching its final appearance
    // in the first minute and then standing still again.
    return Math.min(1, Math.log(1 + count) / Math.log(1 + 400));
  }

  /* The ring, drawn per frame rather than baked into the sprite.
   *
   * This is what makes the field show its work: a symbol nothing has reached
   * yet is a bare point, and one the sweep has been over many times has grown
   * a bright soma with a halo. Without it the cluster looks identical after
   * six hours of scanning as it did at startup, which is the complaint this
   * answers -- the panel was busy but never *changed*.
   *
   * Cheap: one arc and one gradient per symbol, and the gradient only for the
   * ones that have earned a halo. */
  Cluster.prototype.drawNodes = function (ctx, now) {
    for (var i = 0; i < this.symbols.length; i++) {
      var sym = this.symbols[i];
      var node = this.nodes[sym];
      if (!node) continue;
      var d = this.nodeDrift(i, now);
      var x = node.x + d.x, y = node.y + d.y;
      var m = maturity(this.scans[sym] || 0);

      // Recency, so a symbol that was just looked at flares briefly and then
      // settles back to its accumulated size. A field where everything is the
      // same brightness cannot show where the sweep is right now.
      var since = now - (this.lastPulseAt[sym] || -1e9);
      var fresh = since < 1400 ? Math.pow(1 - since / 1400, 2) : 0;

      var col = KIND_COLOR[this.lastKind[sym]] || [120, 190, 240];
      var r = node.radius * (0.72 + m * 0.95) + fresh * 1.8;

      if (m > 0.02 || fresh > 0) {
        /* Restrained on purpose. The halo is what fills the panel, and at a
         * hundred and fifty nodes a generous one turns the ring into a band of
         * light that hides both the filaments under it and the orbs crossing
         * them. Growth has to be legible, not loud. */
        var halo = r * (1.8 + m * 1.3);
        var g = ctx.createRadialGradient(x, y, 0, x, y, halo);
        var a = (0.04 + m * 0.10 + fresh * 0.26);
        g.addColorStop(0, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + a + ')');
        g.addColorStop(1, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(x, y, halo, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' +
                      (0.32 + m * 0.40 + fresh * 0.26) + ')';
      ctx.fill();
    }
  };

  /* A slow per-node wander. Seeded off the index so neighbours never move
   * together -- a ring drifting in unison is a rotating picture, not a set of
   * suspended cells. */
  Cluster.prototype.nodeDrift = function (i, now) {
    var t = now / 1000;
    return {
      x: Math.sin(t * (0.17 + (i % 7) * 0.013) + i * 1.7) * (1.6 + (i % 3) * 0.7),
      y: Math.cos(t * (0.13 + (i % 5) * 0.011) + i * 2.3) * (1.4 + (i % 4) * 0.6)
    };
  };

  function strokePath(g, pts) {
    g.beginPath();
    g.moveTo(pts[0].x, pts[0].y);
    for (var t = 0.02; t <= 1; t += 0.02) {
      var p = pointAt(pts, t);
      g.lineTo(p.x, p.y);
    }
    g.stroke();
  }

  /* ---------- neurons ---------- */

  /* One per position. Cached as a sprite keyed by its size bucket: redrawing
   * the cytoplasm, limbs and contour banding every frame for every position is
   * the single most expensive thing this canvas could do. */
  Cluster.prototype.neuronSprite = function (bucket, seed) {
    var key = bucket + ':' + (seed % 8);
    if (this.neuronSprites[key]) return this.neuronSprites[key];

    var R = 14 + bucket * 5;
    var pad = R * 2.1;
    var size = Math.ceil(pad * 2);
    var c = document.createElement('canvas');
    c.width = c.height = size;
    var g = c.getContext('2d');
    var cx = size / 2, cy = size / 2;
    var rand = mulberry(seed + 7919);

    // Outer glow. Deliberately restrained: a bright halo washes out the
    // cytoplasm, limbs and banding drawn underneath it, and the neuron reads as
    // a featureless star rather than as a cell.
    var glow = g.createRadialGradient(cx, cy, R * 0.55, cx, cy, R * 2.0);
    glow.addColorStop(0, 'rgba(70, 190, 255, 0.13)');
    glow.addColorStop(0.5, 'rgba(70, 190, 255, 0.05)');
    glow.addColorStop(1, 'rgba(70, 190, 255, 0)');
    g.fillStyle = glow;
    g.fillRect(0, 0, size, size);

    // Several irregular limbs that brighten toward the soma.
    var limbs = 5 + Math.floor(rand() * 4);
    for (var i = 0; i < limbs; i++) {
      var a = (i / limbs) * Math.PI * 2 + rand() * 0.6;
      var len = R * (1.25 + rand() * 0.75);
      var grad = g.createLinearGradient(cx, cy,
        cx + Math.cos(a) * len, cy + Math.sin(a) * len);
      grad.addColorStop(0, 'rgba(150, 225, 255, 0.55)');
      grad.addColorStop(1, 'rgba(60, 150, 220, 0)');
      g.strokeStyle = grad;
      g.lineWidth = 2.2;
      g.lineCap = 'round';
      g.beginPath();
      g.moveTo(cx, cy);
      // A kink, so limbs are not straight spokes.
      var mx = cx + Math.cos(a + (rand() - 0.5) * 0.5) * len * 0.55;
      var my = cy + Math.sin(a + (rand() - 0.5) * 0.5) * len * 0.55;
      g.quadraticCurveTo(mx, my, cx + Math.cos(a) * len, cy + Math.sin(a) * len);
      g.stroke();
    }

    // Translucent cytoplasm.
    var body = g.createRadialGradient(cx - R * 0.25, cy - R * 0.25, R * 0.1,
                                      cx, cy, R);
    body.addColorStop(0, 'rgba(180, 240, 255, 0.50)');
    body.addColorStop(0.55, 'rgba(70, 165, 230, 0.30)');
    body.addColorStop(1, 'rgba(30, 90, 150, 0.16)');
    g.fillStyle = body;
    g.beginPath();
    // An irregular outline rather than a circle.
    for (var t = 0; t <= Math.PI * 2 + 0.01; t += 0.22) {
      var rr = R * (0.9 + Math.sin(t * 3 + seed) * 0.06 + rand() * 0.03);
      var px = cx + Math.cos(t) * rr, py = cy + Math.sin(t) * rr;
      if (t === 0) g.moveTo(px, py); else g.lineTo(px, py);
    }
    g.closePath();
    g.fill();

    // Soft contour banding.
    for (var b = 1; b <= 3; b++) {
      g.strokeStyle = 'rgba(150, 225, 255, ' + (0.14 - b * 0.03) + ')';
      g.lineWidth = 1;
      g.beginPath();
      g.arc(cx, cy, R * (0.30 + b * 0.20), 0, Math.PI * 2);
      g.stroke();
    }

    // Visible internal structure.
    for (var k = 0; k < 6; k++) {
      var ia = rand() * Math.PI * 2, ir = rand() * R * 0.55;
      g.fillStyle = 'rgba(200, 245, 255, ' + (0.10 + rand() * 0.16) + ')';
      g.beginPath();
      g.arc(cx + Math.cos(ia) * ir, cy + Math.sin(ia) * ir,
            1 + rand() * 2.2, 0, Math.PI * 2);
      g.fill();
    }

    // The soma.
    g.fillStyle = 'rgba(225, 250, 255, 0.42)';
    g.beginPath();
    g.arc(cx, cy, R * 0.22, 0, Math.PI * 2);
    g.fill();

    this.neuronSprites[key] = c;
    return c;
  };

  Cluster.prototype.setPositions = function (positions) {
    this.positions = positions || {};
  };

  /* ---------- orbs ---------- */

  Cluster.prototype.ingest = function (pulses) {
    if (!pulses || !pulses.length) return;
    var fresh = [];
    for (var i = 0; i < pulses.length; i++) {
      /* Dedupe on the monotonic sequence number, so overlapping snapshot
       * windows spawn each orb exactly once and a dropped frame costs
       * nothing. */
      if (pulses[i].seq > this.seenSeq) fresh.push(pulses[i]);
    }
    if (!fresh.length) return;
    this.seenSeq = fresh[fresh.length - 1].seq;

    /* Bound the replay. A backgrounded tab may have missed thousands. */
    var dropped = 0;
    if (fresh.length > MAX_REPLAY) {
      dropped = fresh.length - MAX_REPLAY;
      fresh = fresh.slice(-MAX_REPLAY);
    }

    var now = performance.now();
    for (var j = 0; j < fresh.length; j++) {
      this.spawn(fresh[j], now);
      this._pulseTimes.push(now);
    }
    while (this._pulseTimes.length && now - this._pulseTimes[0] > 3000) {
      this._pulseTimes.shift();
    }
    this.pulseRate = this._pulseTimes.length / 3;
    this.droppedReplay = (this.droppedReplay || 0) + dropped;

    var budget = this.orbBudget();
    if (this.orbs.length > budget) this.orbs.splice(0, this.orbs.length - budget);
  };

  Cluster.prototype.spawn = function (pulse, now) {
    var pts = this.paths[pulse.symbol];
    if (!pts) {
      // A pulse for a symbol not on the ring (e.g. BOOK-level halts) travels
      // the shortest path we have, rather than being dropped silently.
      var keys = Object.keys(this.paths);
      if (!keys.length) return;
      pts = this.paths[keys[hashString(pulse.symbol) % keys.length]];
    }
    /* The ring is built out of this: every pulse is another look at that
     * symbol, and the node grows and takes the colour of the last verdict. */
    this.scans[pulse.symbol] = (this.scans[pulse.symbol] || 0) + 1;
    this.lastKind[pulse.symbol] = pulse.kind;
    this.lastPulseAt[pulse.symbol] = now;
    this.totalPulses++;

    var depth = rnd(0.35, 1.0);            // near and sharp, or far and dim
    this.orbs.push({
      pts: pts,
      symbol: pulse.symbol,
      kind: pulse.kind,
      reason: pulse.reason,
      intensity: Math.max(0.05, Math.min(1, pulse.intensity)),
      // Jittered speed and start offset: identical speed from an identical
      // start makes a burst travel as one rigid smear.
      t: rnd(-0.10, 0.02),
      speed: rnd(0.16, 0.42) * (0.6 + depth * 0.6),
      depth: depth,
      phase: Math.random() * Math.PI * 2,
      born: now,
      x: 0, y: 0
    });
  };

  /* ---------- the frame ---------- */

  Cluster.prototype.frame = function (now) {
    var dt = Math.min(0.1, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    this.resize();
    var ctx = this.ctx;
    if (!this.w) return;

    // Trails: fade rather than clear, so orbs leave a short tail.
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(6, 8, 12, 0.34)';
    ctx.fillRect(0, 0, this.w, this.h);

    if (!this.networkLayers && this.symbols.length) this.buildNetworkSprite();
    if (this.networkLayers) {
      /* Each layer drifts on its own phase. A translate around a cached image
       * costs nothing -- the geometry is not rebuilt -- so the whole network
       * breathes for the price of three drawImage calls.
       *
       * The network also brightens with how hard the scanner is working: at
       * rest it is a faint skeleton, under a heavy sweep it lights up. That is
       * the difference between a panel that is running and one that is merely
       * displayed. */
      var load = Math.min(1, this.pulseRate / 25);
      var t = now / 1000;
      for (var L = 0; L < this.networkLayers.length; L++) {
        var lay = this.networkLayers[L];
        var dx = Math.sin(t * lay.rate * Math.PI * 2 + lay.phase) * lay.ax;
        var dy = Math.cos(t * lay.rate * Math.PI * 2 * 0.83 + lay.phase) * lay.ay;
        ctx.globalAlpha = 0.62 + 0.38 * load;
        ctx.drawImage(lay.canvas, dx, dy, this.w, this.h);
      }
      ctx.globalAlpha = 1;
    }

    // The ring, sized by how often each symbol has been scanned.
    this.drawNodes(ctx, now);

    this.drawNeurons(ctx, now);

    // Orbs: additive, so overlapping glows brighten rather than occlude.
    ctx.globalCompositeOperation = 'lighter';
    var hover = null, hoverDist = 14;
    for (var i = this.orbs.length - 1; i >= 0; i--) {
      var o = this.orbs[i];
      o.t += o.speed * dt;
      if (o.t > 1.02) { this.orbs.splice(i, 1); continue; }
      if (o.t < 0) continue;

      var p = pointAt(o.pts, o.t);
      o.x = p.x; o.y = p.y;

      var col = KIND_COLOR[o.kind] || KIND_COLOR.scan;
      // Gentle pulsing, plus fade at the very end of the run.
      var pulse = 0.72 + 0.28 * Math.sin(now / 320 + o.phase);
      var tail = o.t > 0.9 ? Math.max(0, (1.02 - o.t) / 0.12) : 1;
      var alpha = o.intensity * o.depth * pulse * tail;

      var isHover = false;
      if (this.mouse.inside) {
        var d = Math.hypot(this.mouse.x - p.x, this.mouse.y - p.y);
        if (d < hoverDist) { hoverDist = d; hover = o; isHover = true; }
      }

      /* An orb on a well-worked symbol is a little more substantial than one
       * on a symbol the sweep has just reached. Bounded tightly -- the orb
       * budget is measured against the bloom, so letting these grow freely
       * would saturate the field at exactly the moment it gets busy. */
      var grow = 1 + 0.30 * maturity(this.scans[o.symbol] || 0);
      var core = CORE_RADIUS * (0.6 + o.depth * 0.7) * grow * (isHover ? 1.7 : 1);
      var bloom = core * BLOOM_MULTIPLE;
      var a = isHover ? Math.min(1, alpha * 2.1) : alpha;

      var g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, bloom);
      g.addColorStop(0, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + (a * 0.85) + ')');
      g.addColorStop(0.35, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + (a * 0.22) + ')');
      g.addColorStop(1, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(p.x, p.y, bloom, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = 'rgba(240,252,255,' + Math.min(1, a * 1.1) + ')';
      ctx.beginPath();
      ctx.arc(p.x, p.y, core * 0.6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';

    this.hover = hover;
    this.updateTooltip();
  };

  Cluster.prototype.drawNeurons = function (ctx, now) {
    var syms = Object.keys(this.positions);
    for (var i = 0; i < syms.length; i++) {
      var sym = syms[i];
      var weight = this.positions[sym];
      if (!weight) continue;
      var node = this.nodes[sym];
      if (!node) continue;
      var bucket = Math.max(0, Math.min(3, Math.round(Math.abs(weight) * 18)));
      var sprite = this.neuronSprite(bucket, hashString(sym));
      // A slow breath, so a held position reads as alive without animating the
      // geometry itself.
      var breath = 1 + 0.05 * Math.sin(now / 900 + hashString(sym) % 10);
      // Grown by attention as well as by size: a position the scanner keeps
      // returning to is drawn larger than one it has looked at twice.
      var grown = 1 + 0.22 * maturity(this.scans[sym] || 0);
      var w = sprite.width * breath * grown, h = sprite.height * breath * grown;
      // Drifts with its node, so the cell and the filament it sits on stay
      // together. A neuron pinned to a moving ring would swim off its own
      // pathway.
      var idx = this.symbols.indexOf(sym);
      var d = this.nodeDrift(idx < 0 ? i : idx, now);
      ctx.globalAlpha = 0.82;
      ctx.drawImage(sprite, node.x + d.x - w / 2, node.y + d.y - h / 2, w, h);
      ctx.globalAlpha = 1;
    }
  };

  Cluster.prototype.updateTooltip = function () {
    var tip = this.tooltip;
    if (!tip) return;
    if (!this.hover) { tip.hidden = true; return; }
    var o = this.hover;
    tip.innerHTML = '<span class="sym"></span> <span class="kind"></span><br><span class="why"></span>';
    tip.querySelector('.sym').textContent = o.symbol;
    tip.querySelector('.kind').textContent = o.kind;
    tip.querySelector('.why').textContent = o.reason || '';
    tip.hidden = false;
    var tw = tip.offsetWidth || 200, th = tip.offsetHeight || 40;
    var x = Math.min(this.w - tw - 6, Math.max(4, o.x + 12));
    var y = Math.min(this.h - th - 6, Math.max(4, o.y - th - 8));
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  };

  global.Cluster = Cluster;
  global.Cluster.MAX_REPLAY = MAX_REPLAY;
  global.Cluster.KIND_COLOR = KIND_COLOR;
  global.Cluster.paintLegend = paintLegend;
  global.Cluster.maturity = maturity;
})(window);
