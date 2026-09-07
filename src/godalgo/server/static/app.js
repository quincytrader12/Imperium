/* The terminal's client.
 *
 * The question this UI exists to answer: a bot that is working and a bot that
 * has silently stopped look identical, and "nothing trading" is the normal case
 * — a scanner that refuses everything it sees is behaving correctly.
 *
 * Two rules drive most of what follows:
 *
 * - Rows are mutated in place, never rebuilt. Rebuilding innerHTML once a
 *   second drops scroll position and text selection, which makes the watchlist
 *   unusable exactly when someone is trying to read it.
 * - "Warming up" and "seeing no opportunity" are shown differently everywhere.
 *   One resolves itself; the other is the bot deciding not to trade.
 */

(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  var state = {
    snapshot: null,
    rows: {},          // symbol -> {tr, cells}
    ws: null,
    retry: 1000,
    ecg: [],
    pendingMode: null
  };

  var cluster = new Cluster($('cluster'), $('tooltip'));

  /* ---------- formatting ---------- */

  function fmtNum(v, dp) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return Number(v).toLocaleString(undefined, {
      minimumFractionDigits: dp, maximumFractionDigits: dp
    });
  }
  function fmtPrice(v) {
    if (!v) return '—';
    var dp = v >= 1000 ? 2 : v >= 1 ? 4 : 6;
    return fmtNum(v, dp);
  }
  function fmtPct(v) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%';
  }
  function fmtMoney(v) { return (v < 0 ? '-' : '') + '$' + fmtNum(Math.abs(v || 0), 2); }
  function fmtTime(ts) {
    var d = new Date(ts * 1000);
    return String(d.getHours()).padStart(2, '0') + ':' +
           String(d.getMinutes()).padStart(2, '0') + ':' +
           String(d.getSeconds()).padStart(2, '0');
  }

  /* ---------- header ---------- */

  function renderHeader(s) {
    $('h-venue').textContent = s.venue || '—';
    var mode = $('h-mode');
    mode.textContent = s.mode === 'live' ? 'LIVE — REAL ORDERS'
      : s.mode === 'paper' ? 'paper (simulated)' : 'dry run (no orders)';
    mode.className = 'v mode-' + s.mode;
    $('h-equity').textContent = fmtMoney(s.equity);
    var r = $('h-realised');
    r.textContent = fmtMoney(s.realised_pnl);
    r.className = 'v num ' + (s.realised_pnl > 0 ? 'up' : s.realised_pnl < 0 ? 'down' : '');
    var u = $('h-unrealised');
    u.textContent = fmtMoney(s.unrealised_pnl);
    u.className = 'v num ' + (s.unrealised_pnl > 0 ? 'up' : s.unrealised_pnl < 0 ? 'down' : '');
    $('h-gross').textContent = (s.gross_exposure * 100).toFixed(1) + '% / ' +
                               (s.gross_ceiling * 100).toFixed(0) + '%';

    var trading = s.watchlist.filter(function (w) { return w.verdict === 'trading'; }).length;
    $('h-trading').textContent = trading + ' / ' + s.watchlist.length;

    ['link', 'venue', 'data', 'key', 'session'].forEach(function (k) {
      var el = $('lamp-' + k);
      el.className = 'lamp ' + (s.lamps[k] || 'off');
    });

    $('btn-start').disabled = s.running;
    $('btn-stop').disabled = !s.running;
    if ($('sel-mode').value !== s.mode && state.pendingMode === null) {
      $('sel-mode').value = s.mode;
    }

    // Banners: the things that stop the bot working, at the top, not buried.
    var banner = $('banner');
    var messages = [];
    if (s.calibration_error) messages.push(['bad', s.calibration_error]);
    if (s.halted) messages.push(['bad', 'BOOK HALTED — ' + s.halt_reason]);
    if (s.venue_error) messages.push(['warn', s.venue_error]);
    if (!messages.length) { banner.hidden = true; banner.textContent = ''; return; }
    banner.hidden = false;
    banner.innerHTML = '';
    messages.forEach(function (m) {
      var d = document.createElement('div');
      d.className = 'notice' + (m[0] === 'bad' ? ' bad' : '');
      d.textContent = m[1];
      banner.appendChild(d);
    });
  }

  /* ---------- watchlist ---------- */

  /* Sort by verdict first, then turnover. Burying a READY behind two NOs makes
   * the list read as unsorted, because the eye scans the top. */
  var VERDICT_ORDER = { trading: 0, not_admitted: 1, rejected: 2, unscanned: 3 };

  function verdictLabel(row) {
    if (row.verdict === 'rejected' && row.decision && row.decision.warming_up) {
      return { cls: 'v-warmup', text: 'warming up' };
    }
    return {
      cls: 'v-' + row.verdict,
      text: row.verdict === 'not_admitted' ? 'queued' : row.verdict.replace('_', ' ')
    };
  }

  function makeRow(symbol) {
    var tr = document.createElement('tr');
    var cells = {};
    ['sym', 'price', 'chg', 'verdict'].forEach(function (k) {
      var td = document.createElement('td');
      if (k === 'price' || k === 'chg') td.className = 'num';
      cells[k] = td;
      tr.appendChild(td);
    });
    cells.sym.textContent = symbol;
    var chip = document.createElement('span');
    chip.className = 'v-chip';
    cells.verdict.appendChild(chip);
    cells.chip = chip;
    return { tr: tr, cells: cells };
  }

  function renderWatchlist(s) {
    var body = $('wl-body');
    var rows = s.watchlist.slice().sort(function (a, b) {
      var va = VERDICT_ORDER[a.verdict] === undefined ? 9 : VERDICT_ORDER[a.verdict];
      var vb = VERDICT_ORDER[b.verdict] === undefined ? 9 : VERDICT_ORDER[b.verdict];
      if (va !== vb) return va - vb;
      return (b.turnover || 0) - (a.turnover || 0);
    });

    rows.forEach(function (row, i) {
      var entry = state.rows[row.symbol];
      if (!entry) {
        entry = makeRow(row.symbol);
        state.rows[row.symbol] = entry;
      }
      // Mutate text, never rebuild innerHTML: rebuilding drops scroll position
      // and any text the operator has selected.
      var c = entry.cells;
      var price = fmtPrice(row.price);
      if (c.price.textContent !== price) c.price.textContent = price;
      var chg = fmtPct(row.change_pct);
      if (c.chg.textContent !== chg) {
        c.chg.textContent = chg;
        c.chg.className = 'num ' + (row.change_pct > 0 ? 'up' : row.change_pct < 0 ? 'down' : 'muted');
      }
      var v = verdictLabel(row);
      if (c.chip.textContent !== v.text) c.chip.textContent = v.text;
      if (c.chip.className !== 'v-chip ' + v.cls) c.chip.className = 'v-chip ' + v.cls;
      entry.tr.title = row.reason || '';

      // Reorder by moving the existing node, which preserves it.
      if (body.children[i] !== entry.tr) {
        body.insertBefore(entry.tr, body.children[i] || null);
      }
    });

    $('wl-note').textContent = rows.length + ' symbols';
  }

  /* ---------- reasoning ---------- */

  function renderReasoning(s) {
    var host = $('reasoning');
    var items = s.watchlist.map(function (r) { return r.decision; })
      .filter(function (d) { return d && d.symbol; });

    // Ordered by closeness to trading, with any halt pinned to the top.
    items.sort(function (a, b) { return (a.distance || 0) - (b.distance || 0); });

    var html = [];
    if (s.halted) {
      html.push('<div class="reason-row halt"><span class="sym">BOOK HALTED</span>' +
        '<span class="why">' + esc(s.halt_reason) + '</span></div>');
    }
    items.slice(0, 24).forEach(function (d) {
      var cls = 'reason-row';
      var metrics;
      if (d.verdict === 'unscanned') {
        /* An unscanned symbol has no edge and no cost. Printing "edge 0.0bp vs
         * 0.0bp needed" for it invents a measurement that was never taken, and
         * reads as a refusal to trade rather than as "no bar has arrived yet". */
        metrics = '<span class="dimmer">no bar has been evaluated yet</span>';
      } else if (d.warming_up) {
        var pct = Math.round(100 * d.bars_seen / Math.max(1, d.warmup_bars));
        metrics = 'warming up — ' + d.bars_seen + ' of ' + d.warmup_bars +
          ' bars (' + pct + '%). This resolves itself.' +
          '<div class="bar warm"><i style="width:' + pct + '%"></i></div>';
      } else {
        var edge = d.expected_edge_bps, need = d.required_bps;
        var frac = need > 0 ? Math.max(0, Math.min(100, 100 * edge / need)) : 0;
        metrics = 'edge ' + edge.toFixed(1) + 'bp vs ' + need.toFixed(1) +
          'bp needed (round trip ' + d.round_trip_cost_bps.toFixed(1) + 'bp) · ' +
          d.regime +
          '<div class="bar"><i style="width:' + frac + '%"></i></div>';
      }
      html.push(
        '<div class="' + cls + '">' +
          '<span class="sym">' + esc(d.symbol) + '</span>' +
          '<span class="dimmer">' + (d.target_weight ?
            (d.target_weight * 100).toFixed(1) + '%' : '') + '</span>' +
          '<span class="why">' + esc(d.reason) + '</span>' +
          '<span class="metrics">' + metrics + '</span>' +
        '</div>');
    });
    host.innerHTML = html.join('');
  }

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ---------- positions, fills, log ---------- */

  function renderPositions(s) {
    var body = $('pos-body');
    if (!s.positions.length) {
      body.innerHTML = '<tr><td colspan="4" class="dimmer">no open positions</td></tr>';
      return;
    }
    body.innerHTML = s.positions.map(function (p) {
      var cls = p.unrealised > 0 ? 'up' : p.unrealised < 0 ? 'down' : 'muted';
      return '<tr><td>' + esc(p.symbol) + '</td>' +
        '<td class="num">' + fmtNum(p.quantity, 6) + '</td>' +
        '<td class="num">' + fmtMoney(p.value) + '</td>' +
        '<td class="num ' + cls + '">' + fmtMoney(p.unrealised) + '</td></tr>';
    }).join('');
  }

  function renderFills(s) {
    var body = $('fill-body');
    if (!s.fills.length) {
      body.innerHTML = '<tr><td colspan="5" class="dimmer">no fills yet</td></tr>';
      return;
    }
    body.innerHTML = s.fills.map(function (f) {
      return '<tr><td class="dimmer">' + fmtTime(f.ts) + '</td>' +
        '<td>' + esc(f.symbol) + '</td>' +
        '<td class="' + (f.side === 'BUY' ? 'up' : 'down') + '">' + esc(f.side) +
        (f.simulated ? ' <span class="dimmer">sim</span>' : '') + '</td>' +
        '<td class="num">' + esc(f.quantity) + '</td>' +
        '<td class="num">' + esc(f.price) + '</td></tr>';
    }).join('');
  }

  function renderLog(s) {
    $('log').innerHTML = s.events.map(function (e) {
      return '<div class="log-line ' + e.level + '">' +
        '<span class="t">' + fmtTime(e.ts) + '</span>' +
        '<span class="m">' + esc(e.message) +
        (e.detail ? ' <span class="d">— ' + esc(e.detail) + '</span>' : '') +
        '</span></div>';
    }).join('');
  }

  /* ---------- health ECG ---------- */

  function renderHealth(s) {
    var h = s.health;
    $('health-note').textContent = 'score ' + (h.score * 100).toFixed(0) + '%';
    $('hz-age').textContent = h.data_age === null ? 'no data' : h.data_age + 's';
    $('hz-reconnects').textContent = h.reconnects;
    $('hz-errors').textContent = h.errors;

    state.ecg.push(h.score);
    if (state.ecg.length > 240) state.ecg.shift();

    var c = $('ecg'), ctx = c.getContext('2d');
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = c.clientWidth || 320, ht = 46;
    if (c.width !== Math.floor(w * dpr)) {
      c.width = Math.floor(w * dpr); c.height = Math.floor(ht * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    ctx.clearRect(0, 0, w, ht);
    ctx.strokeStyle = '#131c29'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, ht / 2); ctx.lineTo(w, ht / 2); ctx.stroke();

    var col = h.score > 0.75 ? '#35d69b' : h.score > 0.45 ? '#e8b444' : '#ff5c6c';
    ctx.strokeStyle = col; ctx.lineWidth = 1.4;
    ctx.shadowBlur = 7; ctx.shadowColor = col;
    ctx.beginPath();
    var n = state.ecg.length;
    for (var i = 0; i < n; i++) {
      var x = w - (n - 1 - i) * (w / 240);
      // A beat shape rather than a plain line, so a flatline reads as a
      // flatline at a glance.
      var beat = Math.sin(i * 1.3) * (0.10 + state.ecg[i] * 0.34);
      var y = ht - 4 - (state.ecg[i] * 0.55 + beat + 0.2) * (ht - 8);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  function renderPnl(s) {
    var c = $('pnl'), ctx = c.getContext('2d');
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = c.clientWidth || 300, h = c.clientHeight || 150;
    if (c.width !== Math.floor(w * dpr)) {
      c.width = Math.floor(w * dpr); c.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    ctx.clearRect(0, 0, w, h);
    var pts = s.equity_curve || [];
    if (pts.length < 2) {
      ctx.fillStyle = '#46536a'; ctx.font = '11px monospace';
      ctx.fillText('no equity history yet', 10, h / 2);
      return;
    }
    var vals = pts.map(function (p) { return p[1]; });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (hi - lo < 1e-9) { hi = lo + 1; }
    var first = vals[0], last = vals[vals.length - 1];
    var col = last >= first ? '#35d69b' : '#ff5c6c';
    ctx.strokeStyle = col; ctx.lineWidth = 1.4;
    ctx.beginPath();
    vals.forEach(function (v, i) {
      var x = (i / (vals.length - 1)) * (w - 8) + 4;
      var y = h - 6 - ((v - lo) / (hi - lo)) * (h - 20);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = '#6b7b91'; ctx.font = '10px monospace';
    ctx.fillText(fmtMoney(last), 6, 12);
  }

  /* ---------- websocket ---------- */

  function connect() {
    var proto = location.protocol === 'https:' ? 'wss' : 'ws';
    var ws = new WebSocket(proto + '://' + location.host + '/ws');
    state.ws = ws;

    ws.onopen = function () { state.retry = 1000; };
    ws.onmessage = function (ev) {
      var s;
      try { s = JSON.parse(ev.data); } catch (e) { return; }
      if (s.error) return;
      state.snapshot = s;
      try { render(s); } catch (e) {
        // A render failure must not take down the stream; the next snapshot
        // gets another chance.
        console.error('render failed', e);
      }
    };
    ws.onclose = function () {
      $('lamp-link').className = 'lamp bad';
      setTimeout(connect, state.retry);
      state.retry = Math.min(10000, state.retry * 2);
    };
    ws.onerror = function () { try { ws.close(); } catch (e) {} };
  }

  function render(s) {
    renderHeader(s);
    renderWatchlist(s);
    renderReasoning(s);
    renderPositions(s);
    renderFills(s);
    renderLog(s);
    renderHealth(s);
    renderPnl(s);

    cluster.setUniverse(s.watchlist.map(function (r) { return r.symbol; }));
    var pos = {};
    s.positions.forEach(function (p) {
      pos[p.symbol] = s.equity > 0 ? Math.abs(p.value) / s.equity : 0;
    });
    cluster.setPositions(pos);
    cluster.ingest(s.pulses);

    $('cluster-note').textContent =
      cluster.orbs.length + ' orbs / ' + cluster.orbBudget() + ' budget · ' +
      cluster.pulseRate.toFixed(1) + ' pulses/s';
  }

  function animate(now) {
    cluster.frame(now);
    requestAnimationFrame(animate);
  }

  /* ---------- controls ---------- */

  function post(url, body) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) throw new Error(j.detail || ('HTTP ' + r.status));
        return j;
      });
    });
  }

  $('btn-start').onclick = function () { post('/api/session/start').catch(alertErr); };
  $('btn-stop').onclick = function () { post('/api/session/stop').catch(alertErr); };

  $('sel-mode').onchange = function (e) {
    var mode = e.target.value;
    if (mode === 'live') {
      state.pendingMode = 'live';
      $('live-error').innerHTML = '';
      $('live-phrase').value = '';
      $('live-modal').hidden = false;
      $('live-phrase').focus();
      return;
    }
    post('/api/session/mode', { mode: mode }).catch(function (err) {
      alertErr(err);
      if (state.snapshot) e.target.value = state.snapshot.mode;
    });
  };

  $('live-cancel').onclick = function () {
    $('live-modal').hidden = true;
    state.pendingMode = null;
    if (state.snapshot) $('sel-mode').value = state.snapshot.mode;
  };

  $('live-go').onclick = function () {
    post('/api/session/mode', {
      mode: 'live', confirmation: $('live-phrase').value
    }).then(function () {
      $('live-modal').hidden = true;
      state.pendingMode = null;
    }).catch(function (err) {
      $('live-error').innerHTML = '<div class="notice bad"></div>';
      $('live-error').firstChild.textContent = err.message;
    });
  };

  $('btn-conn').onclick = function () {
    $('conn-modal').hidden = false;
    loadConnections();
  };
  $('c-close').onclick = function () { $('conn-modal').hidden = true; };

  function loadConnections() {
    fetch('/api/connections').then(function (r) { return r.json(); }).then(function (j) {
      var perm = $('conn-perm');
      if (!j.permissions_ok) {
        perm.innerHTML = '<div class="notice bad"></div>';
        perm.firstChild.textContent =
          'INSECURE CREDENTIAL FILE: ' + j.permissions_detail +
          (j.permissions_remedy ? '  Fix: ' + j.permissions_remedy : '');
      } else {
        perm.innerHTML = '<div class="dimmer" style="font-size:10px"></div>';
        perm.firstChild.textContent = j.path + ' — ' + j.permissions_detail;
      }
      var body = $('conn-body');
      if (!j.credentials.length) {
        body.innerHTML = '<tr><td colspan="5" class="dimmer">no keys stored</td></tr>';
        return;
      }
      body.innerHTML = '';
      j.credentials.forEach(function (c) {
        var tr = document.createElement('tr');
        tr.innerHTML =
          '<td></td><td class="dimmer"></td><td class="dimmer"></td>' +
          '<td><input type="checkbox"' + (c.trade_enabled ? ' checked' : '') + '></td>' +
          '<td><button class="attach">use</button></td>';
        tr.children[0].textContent = c.name + (j.attached === c.name ? ' *' : '');
        tr.children[1].textContent = c.api_key_masked;
        tr.children[2].textContent = c.secret_masked;
        tr.querySelector('input').onchange = function (e) {
          post('/api/connections/' + encodeURIComponent(c.name) + '/trade',
               { trade_enabled: e.target.checked })
            .then(loadConnections).catch(alertErr);
        };
        tr.querySelector('.attach').onclick = function () {
          post('/api/attach', { name: c.name }).then(loadConnections).catch(alertErr);
        };
        body.appendChild(tr);
      });
    }).catch(alertErr);
  }

  $('c-add').onclick = function () {
    var err = $('conn-error');
    err.innerHTML = '';
    post('/api/connections', {
      name: $('c-name').value.trim(),
      api_key: $('c-key').value.trim(),
      secret: $('c-secret').value.trim()
    }).then(function () {
      $('c-name').value = ''; $('c-key').value = ''; $('c-secret').value = '';
      loadConnections();
    }).catch(function (e) {
      err.innerHTML = '<div class="notice bad"></div>';
      err.firstChild.textContent = e.message;
    });
  };

  function alertErr(err) {
    var banner = $('banner');
    banner.hidden = false;
    banner.innerHTML = '<div class="notice bad"></div>';
    banner.firstChild.textContent = err.message || String(err);
  }

  connect();
  requestAnimationFrame(animate);
  window.addEventListener('resize', function () { cluster.resize(); });
})();
