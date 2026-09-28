/* The console's equity chart.
 *
 * Two views on one time axis, never two y-scales on one plot:
 *
 *   Equity       the book's value, with its drawdown drawn underneath as a
 *                second, smaller plot -- a fall from the high is a different
 *                measure from the value, so it gets its own axis rather than
 *                sharing one it does not belong on;
 *   By strategy  each strategy's running profit in dollars, one line each,
 *                in a colour that belongs to the strategy and never moves.
 *
 * Trades are marked on the equity line, shaped by side and coloured by the
 * strategy that placed them. A new one pulses once where it lands.
 *
 * Drawn on a 2D canvas, and only when something changed: the orb beside it
 * already owns the GPU, and a chart repainting sixty times a second to show a
 * line that moves every five minutes would be the terminal fighting itself.
 */
(function () {
  'use strict';

  /* Fixed per strategy, never by rank or order of appearance. The dark
   * steps of a validated categorical palette, checked against this panel's
   * surface for colour-blind separation and contrast; identity is never
   * colour alone -- the legend and the end labels name every line. */
  var STRATEGY_COLOR = {
    trend: '#3987e5',
    cross_section: '#d95926',
    overnight: '#199e70',
    intraday: '#c98500',
    sector: '#d55181',
    unattributed: '#9085e9'
  };
  var STRATEGY_LABEL = {
    intraday: 'Intraday', trend: 'Multi-day trend', overnight: 'Overnight drift',
    cross_section: 'Crypto ranking', sector: 'Sector trend',
    unattributed: 'Unattributed'
  };

  var INK = {
    line: '#dfe5ec', text: '#dbe0e6', dim: '#aeb4bc', dimmer: '#8d939b',
    grid: '#15171b', axis: '#22252b', surface: '#0a0b0d',
    good: '#2fe0a4', bad: '#ff5c6c'
  };
  var REFRESH_MS = 30000;
  var PULSE_MS = 1600;
  var RIGHT_AXIS = 62, BOTTOM_AXIS = 18, TOP_PAD = 26, LEFT_PAD = 12;

  /* The time axis runs a little past the present, the way a trading chart
   * leaves room to its right: the grid, the axis and the last value carry
   * on across it, so the plot reaches the panel's edge instead of stopping
   * short, and the orb sits in the part of it no data can ever occupy. As
   * wide as the orb needs beyond the price axis, and no wider. */
  function future() {
    var orb = document.getElementById('cluster-wrap');
    var w = orb && orb.offsetWidth ? orb.offsetWidth : 0;
    // The orb sits left of the price axis, wholly inside this space, so it
    // can never cover a price label or the live value's tag.
    return Math.max(24, w + 20);
  }

  function $(id) { return document.getElementById(id); }

  function money(v, signed) {
    if (v == null || !isFinite(v)) return '—';
    var a = Math.abs(v);
    var digits = a >= 10000 ? 0 : 2;
    var s = '$' + a.toLocaleString(undefined, {minimumFractionDigits: digits,
                                                maximumFractionDigits: digits});
    if (v < 0) return '-' + s;
    return signed ? '+' + s : s;
  }
  function pct(v, signed) {
    if (v == null || !isFinite(v)) return '—';
    var s = (v * 100).toFixed(Math.abs(v) < 0.1 ? 2 : 1) + '%';
    return (signed && v > 0 ? '+' : '') + s;
  }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
    });
  }

  /* Round tick steps: 1, 2, 2.5, 5 times a power of ten. */
  function niceTicks(lo, hi, count) {
    var span = hi - lo;
    if (!(span > 0)) return [lo];
    var raw = span / Math.max(1, count);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var steps = [1, 2, 2.5, 5, 10];
    var step = mag;
    for (var i = 0; i < steps.length; i++) {
      if (steps[i] * mag >= raw) { step = steps[i] * mag; break; }
    }
    var out = [];
    for (var t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
      out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    }
    return out;
  }

  function axisMoney(v, step) {
    var a = Math.abs(v);
    var s;
    if (a >= 1e6) s = (a / 1e6).toFixed(step >= 1e5 ? 1 : 2) + 'M';
    else if (a >= 1e4) s = (a / 1e3).toFixed(step >= 1000 ? 0 : 1) + 'k';
    else if (step < 1) s = a.toFixed(2);
    else s = a.toFixed(step < 10 ? 1 : 0);
    return (v < 0 ? '-$' : '$') + s;
  }

  var DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
                'Oct', 'Nov', 'Dec'];
  function two(n) { return (n < 10 ? '0' : '') + n; }
  function timeLabel(ts, span) {
    var d = new Date(ts * 1000);
    if (span === '1D') return two(d.getHours()) + ':' + two(d.getMinutes());
    if (span === '1W') return DAYS[d.getDay()] + ' ' + d.getDate();
    return d.getDate() + ' ' + MONTHS[d.getMonth()];
  }
  function fullTime(ts, span) {
    var d = new Date(ts * 1000);
    var day = DAYS[d.getDay()] + ' ' + d.getDate() + ' ' + MONTHS[d.getMonth()];
    if (span === '1D' || span === '1W') {
      return day + ' · ' + two(d.getHours()) + ':' + two(d.getMinutes());
    }
    return day + ' ' + d.getFullYear();
  }

  function Chart() {
    this.canvas = $('equity-chart');
    this.wrap = $('chart-wrap');
    this.tip = $('chart-tip');
    this.legend = $('chart-legend');
    this.empty = $('chart-empty');
    this.stats = $('chart-stats');
    this.view = 'equity';
    this.span = '1W';
    this.data = null;
    this.live = null;
    this.hover = null;
    this.pulses = [];
    this.layout = null;
    this.frame = 0;
    this.fetching = false;
    this.error = '';
    this.reduced = window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    try {
      this.view = localStorage.getItem('imperium.chart.view') || this.view;
      this.span = localStorage.getItem('imperium.chart.span') || this.span;
    } catch (e) { /* private window: the defaults stand */ }
    this._wire();
    this.refresh();
    var self = this;
    setInterval(function () { self.refresh(); }, REFRESH_MS);
  }

  Chart.prototype._wire = function () {
    var self = this;
    function seg(id, attr, key) {
      var root = $(id);
      if (!root) return;
      Array.prototype.forEach.call(root.querySelectorAll('button'), function (b) {
        b.classList.toggle('on', b.getAttribute(attr) === self[key]);
        b.setAttribute('aria-selected', b.getAttribute(attr) === self[key]);
        b.addEventListener('click', function (e) {
          e.stopPropagation();
          self[key] = b.getAttribute(attr);
          try { localStorage.setItem('imperium.chart.' + key, self[key]); } catch (e) {}
          Array.prototype.forEach.call(root.querySelectorAll('button'), function (o) {
            o.classList.toggle('on', o === b);
            o.setAttribute('aria-selected', o === b);
          });
          if (key === 'span') self.refresh(); else self.draw();
        });
      });
    }
    seg('chart-view', 'data-view', 'view');
    seg('chart-span', 'data-span', 'span');

    this.canvas.addEventListener('mousemove', function (e) {
      var r = self.canvas.getBoundingClientRect();
      self.hover = {x: e.clientX - r.left, y: e.clientY - r.top};
      self.draw();
    });
    this.canvas.addEventListener('mouseleave', function () {
      self.hover = null;
      self.tip.hidden = true;
      self.draw();
    });
    if (window.ResizeObserver) {
      new ResizeObserver(function () { self.draw(); }).observe(this.wrap);
    } else {
      window.addEventListener('resize', function () { self.draw(); });
    }
  };

  Chart.prototype.refresh = function () {
    var self = this;
    if (this.fetching) return;
    this.fetching = true;
    var span = this.span;
    fetch('/api/chart?span=' + encodeURIComponent(span), {cache: 'no-store'})
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (d) {
        if (d.span === self.span) { self.data = d; self.error = ''; }
        self.draw();
      })
      .catch(function (e) { self.error = 'chart data unavailable (' + e + ')'; self.draw(); })
      .then(function () {
        self.fetching = false;
        if (self.span !== span) self.refresh();
      });
  };

  /* The book's value right now, from the once-a-second snapshot, so the
   * line's end moves with the account between the half-minute refreshes. */
  Chart.prototype.setLive = function (value) {
    if (value == null || !isFinite(value) || value <= 0) return;
    var changed = !this.live || Math.abs(this.live - value) > 1e-6;
    this.live = value;
    if (changed) this.draw();
  };

  /* A trade just happened: pulse where it lands, and fetch it into the
   * markers now rather than in up to thirty seconds. */
  Chart.prototype.pulse = function (fill) {
    if (!fill) return;
    this.pulses.push({ts: fill.ts, strategy: fill.strategy, side: fill.side,
                      born: performance.now()});
    var self = this;
    setTimeout(function () { self.refresh(); }, 800);
    this._animate();
  };

  Chart.prototype._animate = function () {
    if (this.frame) return;
    var self = this;
    function step() {
      self.frame = 0;
      var now = performance.now();
      self.pulses = self.pulses.filter(function (p) { return now - p.born < PULSE_MS; });
      self.draw();
      if (self.pulses.length && !self.reduced) self.frame = requestAnimationFrame(step);
    }
    this.frame = requestAnimationFrame(step);
  };

  Chart.prototype._series = function () {
    var d = this.data;
    if (!d) return [];
    var pts = (d.equity || []).map(function (p) { return [p[0], p[1]]; });
    if (this.live && pts.length) {
      var now = Date.now() / 1000;
      if (now - pts[pts.length - 1][0] < 600) pts[pts.length - 1] = [pts[pts.length - 1][0], this.live];
      else pts.push([now, this.live]);
    }
    return pts;
  };

  Chart.prototype._fit = function () {
    var c = this.canvas;
    var w = Math.max(1, c.clientWidth), h = Math.max(1, c.clientHeight);
    if (!c.clientWidth && !c.offsetParent) return null;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (c.width !== bw || c.height !== bh) { c.width = bw; c.height = bh; }
    var ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return {ctx: ctx, w: w, h: h};
  };

  Chart.prototype.draw = function () {
    var fit = this._fit();
    if (!fit) return;
    var pts = this._series();
    if (this.view === 'strategies') this._drawStrategies(fit);
    else this._drawEquity(fit, pts);
  };

  Chart.prototype._showEmpty = function (title, body) {
    this.empty.hidden = false;
    this.empty.innerHTML = '<div><b>' + esc(title) + '</b>' + esc(body) + '</div>';
    this.legend.innerHTML = '';
    this.tip.hidden = true;
  };

  Chart.prototype._statCells = function (cells) {
    this.stats.innerHTML = cells.map(function (c) {
      return '<div title="' + esc(c.title || '') + '"><span class="k">' + esc(c.k) +
        '</span><span class="v ' + (c.tone || '') + '">' + c.v + '</span></div>';
    }).join('');
  };

  function xScale(t0, t1, x0, x1) {
    var span = Math.max(1e-9, t1 - t0);
    return function (t) { return x0 + (t - t0) / span * (x1 - x0); };
  }
  function yScale(lo, hi, y0, y1) {
    var span = Math.max(1e-9, hi - lo);
    return function (v) { return y1 - (v - lo) / span * (y1 - y0); };
  }

  Chart.prototype._timeAxis = function (ctx, t0, t1, X, y, x0, x1) {
    ctx.fillStyle = INK.dimmer;
    ctx.font = '9.5px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    var width = x1 - x0;
    var n = Math.max(2, Math.min(7, Math.floor(width / 90)));
    var last = -1e9;
    for (var i = 0; i <= n; i++) {
      var t = t0 + (t1 - t0) * i / n;
      var x = X(t);
      if (x - last < 70) continue;
      var label = timeLabel(t, this.span);
      ctx.fillText(label, Math.min(x1 - 20, Math.max(x0 + 20, x)), y + 4);
      last = x;
    }
  };

  /* Gridlines and their labels. ``opts.left`` puts the labels left of the
   * plot, for the view whose right edge carries the line-end labels;
   * ``opts.avoid`` is a y the live-value pill occupies, which a tick label
   * gives way to rather than printing under it. */
  Chart.prototype._valueAxis = function (ctx, lo, hi, Y, x0, x1, count, fmt, opts) {
    opts = opts || {};
    var ticks = niceTicks(lo, hi, count);
    var step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
    ctx.font = '9.5px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = opts.left ? 'right' : 'left';
    ctx.textBaseline = 'middle';
    ticks.forEach(function (t) {
      var y = Math.round(Y(t)) + 0.5;
      ctx.strokeStyle = INK.grid;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(opts.gridTo || x1, y); ctx.stroke();
      if (opts.avoid != null && Math.abs(y - opts.avoid) < 14) return;
      ctx.fillStyle = INK.dimmer;
      ctx.fillText(fmt(t, step), opts.left ? x0 - 8 : x1 + 8, y);
    });
  };

  /* A value pill on the right axis, for the live end of a line. */
  function pill(ctx, x, y, text, color) {
    ctx.font = '600 10px ' + getComputedStyle(document.body).fontFamily;
    var w = ctx.measureText(text).width + 10;
    ctx.fillStyle = color;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x + 2, y - 8, w, 16, 3); else ctx.rect(x + 2, y - 8, w, 16);
    ctx.fill();
    ctx.fillStyle = '#07080a';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + 7, y + 0.5);
  }

  Chart.prototype._drawEquity = function (fit, pts) {
    var ctx = fit.ctx, w = fit.w, h = fit.h;
    this.legend.innerHTML = '';
    if (pts.length < 2) {
      this._statCells([{k: 'Change', v: '—'}, {k: 'High', v: '—'},
                       {k: 'Max drawdown', v: '—'}, {k: 'Trades', v: '—'}]);
      this._showEmpty(this.error || 'No equity history yet',
        this.error ? '' : 'The line starts with the first account reading after ' +
        'Start. A point is kept every five minutes and saved, so it survives a restart.');
      return;
    }
    this.empty.hidden = true;

    var t0 = pts[0][0], t1 = pts[pts.length - 1][0];
    var vals = pts.map(function (p) { return p[1]; });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var pad = Math.max((hi - lo) * 0.12, hi * 0.0015, 0.01);
    lo -= pad; hi += pad;

    // Drawdown: the fall from the running high, as a fraction of it. The
    // running high itself is kept, because it is drawn: the high-water mark.
    var peak = -Infinity, dd = [], highs = [], worst = 0, worstAt = 0;
    pts.forEach(function (p, i) {
      peak = Math.max(peak, p[1]);
      highs.push(peak);
      var d = peak > 0 ? p[1] / peak - 1 : 0;
      dd.push(d);
      if (d < worst) { worst = d; worstAt = i; }
    });

    // x1 is the plot's right edge, beside the price axis; xe is the present.
    // Between them is the room ahead of the line.
    var x0 = LEFT_PAD, x1 = w - RIGHT_AXIS, xe = x1 - future();
    // One plot, the full height: the drawdown is drawn inside it, as the
    // distance between the line and its own high, not as a second chart.
    var mainTop = TOP_PAD, mainBottom = h - BOTTOM_AXIS - 8;
    var ddBottom = mainBottom;
    var X = xScale(t0, t1, x0, xe);
    var Y = yScale(lo, hi, mainTop, mainBottom);
    this.layout = {X: X, Y: Y, pts: pts, dd: dd, x0: x0, x1: xe};

    this._valueAxis(ctx, lo, hi, Y, x0, x1, Math.max(3, Math.floor((mainBottom - mainTop) / 46)),
                    axisMoney, {avoid: Y(vals[vals.length - 1])});

    // Where the period started: the line everything is measured against.
    var base = pts[0][1];
    var by = Math.round(Y(base)) + 0.5;
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = '#3a3f48';
    ctx.beginPath(); ctx.moveTo(x0, by); ctx.lineTo(x1, by); ctx.stroke();
    ctx.restore();

    // The area under the line: a soft wash of the accent, fading out.
    var up = vals[vals.length - 1] >= base;
    var grad = ctx.createLinearGradient(0, mainTop, 0, mainBottom);
    grad.addColorStop(0, up ? 'rgba(47, 224, 164, 0.16)' : 'rgba(255, 92, 108, 0.14)');
    grad.addColorStop(1, 'rgba(10, 11, 13, 0)');
    ctx.beginPath();
    ctx.moveTo(X(pts[0][0]), mainBottom);
    pts.forEach(function (p) { ctx.lineTo(X(p[0]), Y(p[1])); });
    ctx.lineTo(X(t1), mainBottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Past the present the wash carries on at the last level and fades out
    // across the room ahead, so the fill never stops on a hard vertical edge.
    var endY = Y(vals[vals.length - 1]);
    var top_ = Math.min(endY, mainBottom), height = Math.abs(mainBottom - endY);
    // One-pixel columns on whole pixels: overlapping strips doubled their
    // alpha where they met and striped the fade.
    var fx0 = Math.round(xe), span = Math.max(1, Math.round(x1) - fx0);
    ctx.save();
    ctx.fillStyle = grad;
    for (var k = 0; k < span; k++) {
      ctx.globalAlpha = Math.pow(1 - (k + 0.5) / span, 1.6);
      ctx.fillRect(fx0 + k, top_, 1, height);
    }
    ctx.restore();

    // The present, as a faint rule through both plots.
    ctx.strokeStyle = 'rgba(223, 229, 236, 0.10)';
    ctx.lineWidth = 1;
    var nx = Math.round(xe) + 0.5;
    ctx.beginPath(); ctx.moveTo(nx, mainTop); ctx.lineTo(nx, ddBottom); ctx.stroke();

    // The drawdown: a red veil between the high-water mark and the line,
    // present only while the book is below its high. It reads as what it is
    // -- the distance still to climb back -- and it is nothing at all while
    // the book is making new highs, which is when there is nothing to show.
    ctx.beginPath();
    pts.forEach(function (p, i) {
      if (i === 0) ctx.moveTo(X(p[0]), Y(highs[i])); else ctx.lineTo(X(p[0]), Y(highs[i]));
    });
    for (var r = pts.length - 1; r >= 0; r--) ctx.lineTo(X(pts[r][0]), Y(pts[r][1]));
    ctx.closePath();
    var veil = ctx.createLinearGradient(0, mainTop, 0, mainBottom);
    veil.addColorStop(0, 'rgba(255, 92, 108, 0.20)');
    veil.addColorStop(1, 'rgba(255, 92, 108, 0.10)');
    ctx.fillStyle = veil;
    ctx.fill();

    // The high-water mark: a fine dotted rule that only ever steps up, and
    // carries on to the axis as the level the book has to beat.
    ctx.save();
    ctx.setLineDash([1, 3]);
    ctx.strokeStyle = 'rgba(223, 229, 236, 0.38)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    pts.forEach(function (p, i) {
      var y = Math.round(Y(highs[i])) + 0.5;
      if (i === 0) ctx.moveTo(X(p[0]), y); else ctx.lineTo(X(p[0]), y);
    });
    if (dd[dd.length - 1] < -1e-9) ctx.lineTo(x1, Math.round(Y(highs[highs.length - 1])) + 0.5);
    ctx.stroke();
    ctx.restore();

    // The line. Neutral ink, so the markers on it can carry colour.
    ctx.beginPath();
    pts.forEach(function (p, i) {
      if (i === 0) ctx.moveTo(X(p[0]), Y(p[1])); else ctx.lineTo(X(p[0]), Y(p[1]));
    });
    ctx.strokeStyle = INK.line;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.shadowColor = 'rgba(223, 229, 236, 0.35)';
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Trades, on the line at the moment they filled.
    var trades = (this.data && this.data.trades) || [];
    var marks = [];
    var self = this;
    trades.forEach(function (tr) {
      if (tr.ts < t0 || tr.ts > t1 + 60) return;
      var v = valueAt(pts, tr.ts);
      marks.push({x: X(tr.ts), y: Y(v), tr: tr});
    });
    marks.forEach(function (m) { drawMarker(ctx, m.x, m.y, m.tr); });

    // Pulses for trades that just happened.
    var now = performance.now();
    this.pulses.forEach(function (p) {
      var age = (now - p.born) / PULSE_MS;
      if (age >= 1) return;
      var x = X(Math.min(t1, p.ts)), y = Y(valueAt(pts, p.ts));
      var color = STRATEGY_COLOR[p.strategy] || INK.line;
      ctx.beginPath();
      ctx.arc(x, y, 5 + age * 22, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.globalAlpha = (1 - age) * 0.9;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    // The live end: a dot, a dashed line carrying its level across the room
    // ahead to the axis, and its value there.
    var lx = X(t1), ly = Y(vals[vals.length - 1]);
    ctx.save();
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = 'rgba(223, 229, 236, 0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(lx + 6, Math.round(ly) + 0.5);
    ctx.lineTo(x1, Math.round(ly) + 0.5); ctx.stroke();
    ctx.restore();
    ctx.beginPath(); ctx.arc(lx, ly, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(223, 229, 236, 0.16)'; ctx.fill();
    ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2);
    ctx.fillStyle = INK.line; ctx.fill();
    pill(ctx, x1 + 4, ly, axisMoney(vals[vals.length - 1], 0.5), INK.line);

    // The deepest fall, called out once: a fine bracket from the high to the
    // trough, and its size beside it. The only place the chart names a
    // drawdown in numbers; everywhere else the veil says it.
    if (worst < -1e-4) {
      var wx = Math.round(X(pts[worstAt][0])) + 0.5;
      var wyTop = Y(highs[worstAt]), wyBot = Y(pts[worstAt][1]);
      ctx.strokeStyle = 'rgba(255, 92, 108, 0.75)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(wx - 3, wyTop); ctx.lineTo(wx + 3, wyTop);
      ctx.moveTo(wx, wyTop); ctx.lineTo(wx, wyBot);
      ctx.moveTo(wx - 3, wyBot); ctx.lineTo(wx + 3, wyBot);
      ctx.stroke();
      var label = pct(worst);
      ctx.font = '600 9.5px ' + getComputedStyle(document.body).fontFamily;
      var lw = ctx.measureText(label).width + 10;
      var onLeft = wx + 8 + lw > xe;
      var lx0 = onLeft ? wx - 8 - lw : wx + 8;
      var lyc = Math.max(mainTop + 9, Math.min(mainBottom - 9, (wyTop + wyBot) / 2));
      ctx.fillStyle = 'rgba(9, 10, 13, 0.92)';
      ctx.strokeStyle = 'rgba(255, 92, 108, 0.55)';
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(lx0, lyc - 8, lw, 16, 3); else ctx.rect(lx0, lyc - 8, lw, 16);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#ff8a96';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(label, lx0 + 5, lyc + 0.5);
    }

    this._timeAxis(ctx, t0, t1, X, h - BOTTOM_AXIS, x0, xe);

    // The period in numbers.
    var last = vals[vals.length - 1];
    var change = last - base;
    var realised = trades.reduce(function (a, t) { return a + (t.realised || 0); }, 0);
    this._statCells([
      {k: 'Change', v: money(change, true) + ' <small>' + pct(base > 0 ? change / base : 0, true) + '</small>',
       tone: change > 0 ? 'up' : change < 0 ? 'down' : '',
       title: 'From ' + money(base) + ' at the start of the range to ' + money(last)},
      {k: 'High', v: money(Math.max.apply(null, vals)),
       title: 'The highest reading in this range'},
      {k: 'Max drawdown', v: pct(worst) + ' <small>now ' +
         (dd[dd.length - 1] < -1e-6 ? pct(dd[dd.length - 1]) : 'at the high') + '</small>',
       tone: worst < -0.02 ? 'down' : '',
       title: 'The largest fall from a high inside this range' +
              (worst < 0 ? ', reached ' + fullTime(pts[worstAt][0], this.span) : '')},
      {k: 'Trades', v: String(trades.length) + (trades.length ? ' <small>' + money(realised, true) + ' realised</small>' : ''),
       title: 'Fills in this range, and the profit or loss they closed'}
    ]);

    this._hoverEquity(ctx, marks, mainTop, ddBottom);
  };

  function ends4(series) { return series.length <= 4; }

  function valueAt(pts, ts) {
    if (ts <= pts[0][0]) return pts[0][1];
    for (var i = 1; i < pts.length; i++) {
      if (pts[i][0] >= ts) {
        var a = pts[i - 1], b = pts[i];
        var f = (ts - a[0]) / Math.max(1e-9, b[0] - a[0]);
        return a[1] + (b[1] - a[1]) * f;
      }
    }
    return pts[pts.length - 1][1];
  }

  function drawMarker(ctx, x, y, tr) {
    var color = STRATEGY_COLOR[tr.strategy] || INK.dim;
    var buy = String(tr.side).toUpperCase() === 'BUY';
    var s = 5;
    ctx.beginPath();
    if (buy) {       // below the line, pointing up at it
      ctx.moveTo(x, y + 4); ctx.lineTo(x - s, y + 4 + s * 1.5); ctx.lineTo(x + s, y + 4 + s * 1.5);
    } else {         // above the line, pointing down at it
      ctx.moveTo(x, y - 4); ctx.lineTo(x - s, y - 4 - s * 1.5); ctx.lineTo(x + s, y - 4 - s * 1.5);
    }
    ctx.closePath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = INK.surface;   // a ring of surface, so markers never merge
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fill();
  }

  Chart.prototype._hoverEquity = function (ctx, marks, top, bottom) {
    var hv = this.hover, L = this.layout;
    if (!hv || !L || hv.x < L.x0 || hv.x > L.x1) { this.tip.hidden = true; return; }
    // A trade under the pointer wins over the line.
    var hit = null, best = 12;
    marks.forEach(function (m) {
      var d = Math.hypot(m.x - hv.x, m.y + (String(m.tr.side).toUpperCase() === 'BUY' ? 9 : -9) - hv.y);
      if (d < best) { best = d; hit = m; }
    });
    var pts = L.pts, idx = 0, bd = Infinity;
    for (var i = 0; i < pts.length; i++) {
      var d2 = Math.abs(L.X(pts[i][0]) - hv.x);
      if (d2 < bd) { bd = d2; idx = i; }
    }
    var p = pts[idx], x = Math.round(L.X(p[0])) + 0.5, y = L.Y(p[1]);
    ctx.strokeStyle = 'rgba(223, 229, 236, 0.28)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = INK.line; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = INK.surface; ctx.stroke();

    var html;
    if (hit) {
      var tr = hit.tr;
      html = '<div class="t">' + esc(fullTime(tr.ts, '1D')) + '</div>' +
        '<div class="row"><span><i class="sw" style="background:' +
        (STRATEGY_COLOR[tr.strategy] || INK.dim) + '"></i>' +
        esc(STRATEGY_LABEL[tr.strategy] || tr.strategy || 'trade') + '</span></div>' +
        '<div class="row"><span>' + esc(tr.side) + ' ' + esc(tr.symbol) + '</span><b>' +
        esc(tr.quantity) + ' @ ' + money(tr.price) + '</b></div>' +
        (tr.realised ? '<div class="row"><span>realised</span><b class="' +
          (tr.realised > 0 ? 'up' : 'down') + '">' + money(tr.realised, true) + '</b></div>' : '') +
        (tr.order ? '<div class="row"><span>order</span><b>' + esc(tr.order) + '</b></div>' : '');
    } else {
      var base = pts[0][1];
      html = '<div class="t">' + esc(fullTime(p[0], this.span)) + '</div>' +
        '<div class="row"><span>equity</span><b>' + money(p[1]) + '</b></div>' +
        '<div class="row"><span>since start</span><b class="' +
        (p[1] >= base ? 'up' : 'down') + '">' + money(p[1] - base, true) + '</b></div>' +
        '<div class="row"><span>from high</span><b>' + pct(L.dd[idx]) + '</b></div>';
    }
    this._placeTip(html, hv);
  };

  Chart.prototype._placeTip = function (html, hv) {
    var tip = this.tip;
    tip.innerHTML = html;
    tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var W = this.wrap.clientWidth, H = this.wrap.clientHeight;
    var left = hv.x + 14, top = hv.y + 14;
    if (left + w > W - 4) left = hv.x - w - 14;
    if (top + h > H - 4) top = hv.y - h - 14;
    tip.style.left = Math.max(4, left) + 'px';
    tip.style.top = Math.max(4, top) + 'px';
  };

  Chart.prototype._drawStrategies = function (fit) {
    var ctx = fit.ctx, w = fit.w, h = fit.h;
    var d = this.data;
    var series = [];
    var order = ['trend', 'cross_section', 'overnight', 'intraday', 'sector', 'unattributed'];
    var names = Object.keys((d && d.strategies) || {}).sort(function (a, b) {
      var ia = order.indexOf(a), ib = order.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    names.forEach(function (n) {
      var pts = d.strategies[n] || [];
      if (pts.length) series.push({name: n, pts: pts});
    });
    var have = series.filter(function (s) { return s.pts.length >= 2; });
    this.empty.hidden = true;
    if (!have.length) {
      this._statCells([{k: 'Strategies', v: String(series.length)},
                       {k: 'Best', v: '—'}, {k: 'Worst', v: '—'}]);
      this._showEmpty(this.error || 'No strategy has a daily record yet',
        this.error ? '' : 'Each strategy is marked once a day, at the evening brief. ' +
        'Its line starts with its first mark.');
      return;
    }
    var t0 = Infinity, t1 = -Infinity, lo = 0, hi = 0;
    have.forEach(function (s) {
      s.pts.forEach(function (p) {
        t0 = Math.min(t0, p[0]); t1 = Math.max(t1, p[0]);
        lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]);
      });
    });
    var pad = Math.max((hi - lo) * 0.12, 0.5);
    lo -= pad; hi += pad;
    // The value axis on the left here: the right edge is where each line's
    // end is named, and the two sets of labels cannot share one column.
    var x0 = 58, x1 = w - (ends4(have) ? 170 : 24);
    var top = TOP_PAD + 8, bottom = h - BOTTOM_AXIS;
    var X = xScale(t0, t1, x0, x1), Y = yScale(lo, hi, top, bottom);
    this._valueAxis(ctx, lo, hi, Y, x0, x1, Math.max(3, Math.floor((bottom - top) / 46)),
                    axisMoney, {left: true, gridTo: w - 8});
    var zy = Math.round(Y(0)) + 0.5;
    ctx.strokeStyle = '#3a3f48'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, zy); ctx.lineTo(w - 8, zy); ctx.stroke();

    var ends = [];
    have.forEach(function (s) {
      var color = STRATEGY_COLOR[s.name] || INK.dim;
      ctx.beginPath();
      s.pts.forEach(function (p, i) {
        if (i === 0) ctx.moveTo(X(p[0]), Y(p[1])); else ctx.lineTo(X(p[0]), Y(p[1]));
      });
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
      var last = s.pts[s.pts.length - 1];
      ctx.beginPath(); ctx.arc(X(last[0]), Y(last[1]), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = INK.surface; ctx.stroke();
      ends.push({name: s.name, y: Y(last[1]), v: last[1], color: color});
    });

    // Direct labels at the line ends, nudged apart so none collide. More
    // than four and the legend alone carries the names.
    if (ends.length <= 4) {
      ends.sort(function (a, b) { return a.y - b.y; });
      for (var i = 1; i < ends.length; i++) {
        if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
      }
      ctx.font = '9.5px ' + getComputedStyle(document.body).fontFamily;
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ends.forEach(function (e) {
        ctx.fillStyle = INK.dim;
        ctx.fillText((STRATEGY_LABEL[e.name] || e.name) + ' ' + money(e.v, true),
                     x1 + 8, e.y);
      });
    }
    this._timeAxis(ctx, t0, t1, X, bottom, x0, x1);

    this.legend.innerHTML = have.map(function (s) {
      return '<span><i style="background:' + (STRATEGY_COLOR[s.name] || INK.dim) +
        '"></i>' + esc(STRATEGY_LABEL[s.name] || s.name) + '</span>';
    }).join('');

    var totals = have.map(function (s) {
      return {name: s.name, v: s.pts[s.pts.length - 1][1]};
    }).sort(function (a, b) { return b.v - a.v; });
    var sum = totals.reduce(function (a, t) { return a + t.v; }, 0);
    this._statCells([
      {k: 'All strategies', v: money(sum, true), tone: sum > 0 ? 'up' : sum < 0 ? 'down' : ''},
      {k: 'Best', v: esc(STRATEGY_LABEL[totals[0].name] || totals[0].name) +
        ' <small>' + money(totals[0].v, true) + '</small>'},
      {k: 'Worst', v: esc(STRATEGY_LABEL[totals[totals.length - 1].name] || totals[totals.length - 1].name) +
        ' <small>' + money(totals[totals.length - 1].v, true) + '</small>'}
    ]);

    // Hover: every strategy's value at the nearest mark.
    var hv = this.hover;
    if (!hv || hv.x < x0 || hv.x > x1) { this.tip.hidden = true; return; }
    var t = t0 + (hv.x - x0) / Math.max(1, x1 - x0) * (t1 - t0);
    var cx = Math.round(hv.x) + 0.5;
    ctx.strokeStyle = 'rgba(223, 229, 236, 0.28)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, top); ctx.lineTo(cx, bottom); ctx.stroke();
    var rows = have.map(function (s) {
      var v = valueAt(s.pts, t);
      ctx.beginPath(); ctx.arc(cx, Y(v), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = STRATEGY_COLOR[s.name] || INK.dim; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = INK.surface; ctx.stroke();
      return {name: s.name, v: v};
    }).sort(function (a, b) { return b.v - a.v; });
    this._placeTip('<div class="t">' + esc(fullTime(t, this.span)) + '</div>' +
      rows.map(function (r) {
        return '<div class="row"><span><i class="sw" style="background:' +
          (STRATEGY_COLOR[r.name] || INK.dim) + '"></i>' +
          esc(STRATEGY_LABEL[r.name] || r.name) + '</span><b>' + money(r.v, true) + '</b></div>';
      }).join(''), hv);
  };

  /* The header's thirty-day sparkline. Here because it is the same line at
   * a glance, drawn by the same hand. */
  function drawSpark(canvas, values) {
    if (!canvas) return;
    var w = canvas.clientWidth || 120, h = canvas.clientHeight || 34;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(w * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!values || values.length < 2) {
      ctx.strokeStyle = '#2a2e35'; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(2, h / 2); ctx.lineTo(w - 2, h / 2); ctx.stroke();
      ctx.setLineDash([]);
      return;
    }
    var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
    if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
    var up = values[values.length - 1] >= values[0];
    var color = up ? INK.good : INK.bad;
    var X = function (i) { return 2 + i / (values.length - 1) * (w - 8); };
    var Y = function (v) { return h - 4 - (v - lo) / (hi - lo) * (h - 8); };
    var g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, up ? 'rgba(47, 224, 164, 0.22)' : 'rgba(255, 92, 108, 0.20)');
    g.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.beginPath(); ctx.moveTo(X(0), h);
    values.forEach(function (v, i) { ctx.lineTo(X(i), Y(v)); });
    ctx.lineTo(X(values.length - 1), h); ctx.closePath();
    ctx.fillStyle = g; ctx.fill();
    ctx.beginPath();
    values.forEach(function (v, i) { if (i) ctx.lineTo(X(i), Y(v)); else ctx.moveTo(X(i), Y(v)); });
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.lineJoin = 'round'; ctx.stroke();
    var lx = X(values.length - 1), ly = Y(values[values.length - 1]);
    ctx.beginPath(); ctx.arc(lx, ly, 2.2, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  }

  window.ImperiumChart = {
    start: function () {
      if (!window.__chart && $('equity-chart')) window.__chart = new Chart();
      return window.__chart;
    },
    spark: drawSpark,
    color: STRATEGY_COLOR
  };
})();
