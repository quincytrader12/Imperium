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
    /* The event log is a client-side ring now: the stream sends only what is
     * new, so replacing this from a frame would leave the panel showing the
     * last second of history instead of the last hour. */
    events: [],
    pendingMode: null
  };

  var EVENT_RING = 200;
  var ECG_SAMPLES = 240;
  /* Bounded by what the stream sends full reasoning for (DETAIL_ROWS). */
  var REASON_ROWS = 40;

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
    /* The account balance is the headline, not the simulated book's. A
     * terminal showing a made-up default where the money goes is worse than
     * showing nothing, and "—" is the honest reading before a key is
     * attached. The book is still shown next to it whenever the two differ,
     * because in dry run and paper they are different numbers on purpose. */
    var acct = s.account || {};
    var eq = $('h-equity');
    if (acct.known) {
      eq.textContent = fmtMoney(acct.equity);
      eq.title = 'account ' + fmtMoney(acct.equity) + ' · cash ' +
        fmtMoney(acct.cash) + ' · buying power ' + fmtMoney(acct.buying_power) +
        (acct.simulated ? '\nsimulated book: ' + fmtMoney(acct.book_equity) : '');
    } else {
      eq.textContent = '—';
      eq.title = acct.error || 'attach a key to read the account balance';
    }
    var book = $('h-book');
    /* Shown only when it says something the headline does not. */
    var drift = acct.known && acct.simulated &&
                Math.abs(acct.book_equity - acct.equity) >= 0.01;
    book.parentNode.hidden = !drift;
    if (drift) book.textContent = fmtMoney(acct.book_equity);
    var r = $('h-realised');
    r.textContent = fmtMoney(s.realised_pnl);
    r.className = 'v num ' + (s.realised_pnl > 0 ? 'up' : s.realised_pnl < 0 ? 'down' : '');
    var u = $('h-unrealised');
    u.textContent = fmtMoney(s.unrealised_pnl);
    u.className = 'v num ' + (s.unrealised_pnl > 0 ? 'up' : s.unrealised_pnl < 0 ? 'down' : '');
    $('h-gross').textContent = (s.gross_exposure * 100).toFixed(1) + '% / ' +
                               (s.gross_ceiling * 100).toFixed(0) + '%';

    var market = s.market || {};
    var mk = $('h-market');
    mk.textContent = market.describe || '-';
    mk.className = 'v ' + (market.is_open ? 'up' : market.crypto_only ? '' : 'muted');

    var trading = s.watchlist.filter(function (w) { return w.verdict === 'trading'; }).length;
    $('h-trading').textContent = trading + ' / ' + s.watchlist.length;

    ['link', 'venue', 'data', 'key', 'session'].forEach(function (k) {
      var el = $('lamp-' + k);
      el.className = 'lamp ' + (s.lamps[k] || 'off');
    });
    /* A dark lamp is ambiguous on its own: a shut market, a refused
     * subscription and a socket that never connected all read the same. The
     * sentence that separates them was already arriving in the payload. */
    if (s.feed && s.feed.reason) $('lamp-data').title = s.feed.reason;

    $('btn-start').disabled = s.running;
    $('btn-stop').disabled = !s.running;
    if ($('sel-mode').value !== s.mode && state.pendingMode === null) {
      $('sel-mode').value = s.mode;
    }

    // Banners: the things that stop the bot working, at the top, not buried.
    var banner = $('banner');
    var messages = [];
    if (s.calibration_error) messages.push(['bad', s.calibration_error]);
    if (s.store_error) messages.push(['bad', 'Credentials: ' + s.store_error]);
    if (s.limits && s.limits.pdt_blocked) {
      messages.push(['warn', 'Pattern day trader: ' + s.limits.pdt_blocked +
        '. Exits still pass; no new equity exposure is opened.']);
    }
    if (market && !market.is_open && !market.crypto_only) {
      messages.push(['warn', 'Equity market is closed (' + (market.describe || '') +
        '). Crypto continues; equities take no new exposure.']);
    }
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
    ['sym', 'cls', 'price', 'chg', 'verdict'].forEach(function (k) {
      var td = document.createElement('td');
      if (k === 'price' || k === 'chg') td.className = 'num';
      cells[k] = td;
      tr.appendChild(td);
    });
    cells.sym.textContent = symbol;
    cells.cls.className = 'cls-tag';
    var chip = document.createElement('span');
    chip.className = 'v-chip';
    cells.verdict.appendChild(chip);
    cells.chip = chip;
    return { tr: tr, cells: cells };
  }

  function renderWatchlist(s) {
    var body = $('wl-body');
    var scan = s.universe_scan || {};
    /* The table is a ranked window on the universe, not the universe. Saying
     * so beats a row count that quietly contradicts the scanner's own note. */
    $('wl-note').textContent = scan.omitted
      ? scan.shown + ' of ' + scan.size + ' shown · ' + scan.omitted +
        ' more scanned below the line'
      : (scan.size || s.watchlist.length) + ' symbols';
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
      /* The asset class drives the calendar, the costs and the calibration, so
       * it is shown rather than left for the reader to infer from the ticker. */
      var cls = row.asset_class || (row.decision && row.decision.asset_class) || '';
      var short = cls === 'crypto' ? 'CR' : cls === 'us_equity' ? 'EQ'
                : cls === 'us_option' ? 'OP' : '';
      if (c.cls.textContent !== short) {
        c.cls.textContent = short;
        c.cls.className = 'cls-tag cls-' + (cls || 'none');
      }
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
  }

  /* ---------- reasoning ---------- */

  function renderReasoning(s) {
    var host = $('reasoning');

    /* Say that a scan is happening. Without this the panel is a list of
     * verdicts with no indication that anything is still working through the
     * market behind them -- and a symbol that has not been reached yet looks
     * identical to one that was looked at and had nothing to say. */
    /* The cohort-wide answer to "why is nothing trading". A scrolling list of
     * per-symbol prose cannot answer it: every line is about one symbol, and
     * the thing worth knowing is what is holding up all of them. */
    var bl = $('blockers');
    if (bl) {
      var b = s.blockers || {};
      var bits = (b.counts || []).map(function (c) {
        return c.symbols + ' ' + c.blocker;
      });
      bl.innerHTML =
        '<b class="' + (b.trading ? 'trading' : '') + '">' +
        (b.summary || 'nothing evaluated yet') + '</b>' +
        (bits.length ? ' <span class="tally">· ' + bits.join(' · ') + '</span>' : '');
    }

    var us = s.universe_scan || {};
    var note = $('reason-note');
    if (note) {
      if (us.ranked) {
        var pct = Math.round((us.cohort_progress || 0) * 100);
        note.innerHTML =
          '<span class="scanning"><i></i>scanning</span> ' +
          fmtNum(us.cohort_at || 0, 0) + ' of ' + fmtNum(us.ranked, 0) +
          ' ranked · ' + pct + '% of this pass · ' +
          fmtCount(us.cohort_passes || 0) + ' complete';
        note.title =
          'The ranking covers the whole market; a cohort of ' +
          fmtNum(us.size || 0, 0) + ' carries engines at a time and rotates ' +
          'through it. Symbols that were scanned and had nothing to say are ' +
          'retired and come round again on the next pass; anything holding a ' +
          'position or worth trading stays.';
      } else {
        note.textContent = 'why each symbol is or is not trading';
        note.title = '';
      }
    }
    var items = s.watchlist.map(function (r) { return r.decision; })
      .filter(function (d) { return d && d.symbol; });

    // Ordered by closeness to trading, with any halt pinned to the top.
    items.sort(function (a, b) { return (a.distance || 0) - (b.distance || 0); });

    var html = [];
    if (s.halted) {
      html.push('<div class="reason-row halt"><span class="sym">BOOK HALTED</span>' +
        '<span class="why">' + esc(s.halt_reason) + '</span></div>');
    }
    items.slice(0, REASON_ROWS).forEach(function (d) {
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
    $('pos-note').textContent = s.positions.length
      ? s.positions.length + ' open' : 'flat';
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
      var slip = f.slippage_bps || 0;
      var slipTone = slip > 5 ? 'down' : slip <= 0 ? 'up' : 'muted';
      return '<tr><td class="dimmer">' + fmtTime(f.ts) + '</td>' +
        '<td>' + esc(f.symbol) + '</td>' +
        '<td class="' + (f.side === 'BUY' ? 'up' : 'down') + '">' + esc(f.side) +
        (f.simulated ? ' <span class="dimmer">sim</span>' : '') + '</td>' +
        '<td class="num">' + esc(f.quantity) + '</td>' +
        '<td class="num">' + esc(f.price) + '</td>' +
        '<td class="num ' + slipTone + '">' + slip.toFixed(1) + '</td></tr>';
    }).join('');
  }

  /* Did crossing cost what the cost gate assumed? The gate admits a symbol on a
   * modelled crossing cost; measuring the realised one is the only way to catch
   * a mis-calibrated model before the P&L does. */
  function renderExecution(s) {
    var x = s.execution || {count: 0};
    if (!x.count) {
      $('exec-grid').innerHTML =
        '<div style="grid-column:1/-1"><span class="k">execution quality</span>' +
        '<span class="v dimmer" style="font-size:10px">no fills yet — measured ' +
        'against the modelled crossing cost once there are</span></div>';
      $('exec-note').textContent = 'fills';
      return;
    }
    var drift = x.avg_slippage_bps - x.modelled_bps;
    var tone = Math.abs(drift) > 3 ? 'bad' : Math.abs(drift) > 1 ? 'warn' : 'good';
    $('exec-grid').innerHTML =
      cell('fills', String(x.count), '', x.simulated ? x.simulated + ' sim' : 'live') +
      cell('notional', fmtMoney(x.notional)) +
      cell('buy / sell', x.buys + ' / ' + x.sells) +
      cell('avg slip', x.avg_slippage_bps.toFixed(1) + 'bp', tone) +
      cell('modelled', x.modelled_bps.toFixed(1) + 'bp') +
      cell('worst slip', x.worst_slippage_bps.toFixed(1) + 'bp',
           x.worst_slippage_bps > 10 ? 'bad' : '');
    $('exec-note').textContent = 'realised ' + x.avg_slippage_bps.toFixed(1) +
      'bp vs modelled ' + x.modelled_bps.toFixed(1) + 'bp';
  }

  function absorbEvents(s) {
    /* A frame that is not a delta is the instruction to start over: a
     * reconnect, or a client that fell behind the ring and cannot be caught up
     * by one. Anything else is prepended, newest first, as the server sends. */
    var incoming = s.events || [];
    if (!s.delta) {
      state.events = incoming.slice(0, EVENT_RING);
      return;
    }
    if (!incoming.length) return;
    state.events = incoming.concat(state.events).slice(0, EVENT_RING);
  }

  function renderLog(s) {
    absorbEvents(s);
    /* Rebuilt only when something arrived. innerHTML on a 200-row list every
     * second is a layout pass the terminal cannot afford, and for most seconds
     * there is nothing new to show. */
    if (s.delta && !(s.events || []).length && $('log').childElementCount) return;
    $('log').innerHTML = state.events.map(function (e) {
      return '<div class="log-line ' + e.level + '">' +
        '<span class="t">' + fmtTime(e.ts) + '</span>' +
        '<span class="m">' + esc(e.message) +
        (e.detail ? ' <span class="d">— ' + esc(e.detail) + '</span>' : '') +
        '</span></div>';
    }).join('');
  }

  /* ---------- canvas ---------- */

  /* Every canvas here had the same two bugs, so they are fixed in one place.
   *
   * The resize check compared only the width, so a panel that changed height
   * without changing width kept its old backing store and drew a stretched
   * picture into it -- which is what a flex layout does constantly.
   *
   * And setting canvas.width or .height resets the 2D context, transform
   * included. Any code that sets one without immediately re-applying the
   * transform draws at the wrong scale on a high-DPI screen, so the two are
   * done together and nowhere else. */
  function fitCanvas(c, cssHeight) {
    var ctx = c.getContext('2d');
    // A folded panel gives its canvas no box at all. Drawing into it would fit
    // a 1px backing store that the next unfold would render at the wrong scale.
    if (!c.clientWidth && !c.offsetParent) return null;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = Math.max(1, c.clientWidth || 300);
    var h = Math.max(1, cssHeight || c.clientHeight || 150);
    var bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (c.width !== bw || c.height !== bh) {
      c.width = bw; c.height = bh;
    }
    // Re-applied every frame: cheap, and it cannot drift out of step with a
    // backing store that something else resized.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  /* ---------- health ECG ---------- */

  function renderHealth(s) {
    var h = s.health;
    $('health-note').textContent = 'score ' + (h.score * 100).toFixed(0) + '%' +
      (state.ecg.length < 240 ? ' · ' + state.ecg.length + 's history' : '');
    $('hz-age').textContent = h.data_age === null ? 'no data' : h.data_age + 's';

    /* Why there is no data, in words. Without it "no data" is a dead end: it
     * is the same text whether the market is shut, the plan refused the
     * subscription, or the socket never connected at all. */
    var fr = $('hz-feed');
    var f = s.feed || {};
    if (fr) {
      fr.textContent = f.reason || '—';
      var level = '';
      if (!s.running) level = '';
      else if (f.connected === false) level = 'bad';
      else if (h.data_age === null || h.data_age >= 20) {
        /* Idle because the market is shut is expected, not a fault. */
        level = (s.market && !s.market.is_open && !s.market.crypto_only) ? '' : 'warn';
      }
      fr.className = 'feed-reason ' + level;
      fr.title = f.last_error || '';
    }
    $('hz-reconnects').textContent = h.reconnects;
    $('hz-errors').textContent = h.errors;

    /* Continuous operation, made visible. A loop that stopped and was restarted
     * looks identical to one that never stopped from every other indicator on
     * screen, and a terminal quietly restarting itself all night is a terminal
     * with a problem worth seeing. */
    var hz = $('hz-uptime');
    if (hz) {
      hz.textContent = s.running ? fmtDuration(s.uptime) : 'stopped';
      var bits = [];
      if (s.restarts) bits.push(s.restarts + ' restart' + (s.restarts > 1 ? 's' : ''));
      var ka = s.keep_awake || {};
      if (ka.active) bits.push('sleep held off');
      else if (s.running && ka.supported === false) bits.push('sleep not managed');
      var note = $('hz-uptime-note');
      if (note) {
        note.textContent = bits.join(' · ');
        note.className = s.restarts ? 'warn' : '';
      }
      hz.title = (ka.note || '') + (s.loop_age !== null && s.loop_age !== undefined
        ? '\nlast loop tick ' + s.loop_age.toFixed(1) + 's ago' : '');
    }

    state.ecg.push(h.score);
    if (state.ecg.length > ECG_SAMPLES) state.ecg.shift();

    var fit = fitCanvas($('ecg'), 46);
    if (!fit) return;
    var ctx = fit.ctx, w = fit.w, ht = fit.h;

    ctx.strokeStyle = '#131c29'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, ht / 2); ctx.lineTo(w, ht / 2); ctx.stroke();

    var n = state.ecg.length;
    if (!n) return;
    var col = h.score > 0.75 ? '#35d69b' : h.score > 0.45 ? '#e8b444' : '#ff5c6c';

    /* Never plot more points than there are pixels to plot them on. Beyond
     * that every extra point is sub-pixel detail nobody can see, and a dense
     * zigzag drawn under a glow is what made this read as a smear rather than
     * as an instrument. */
    var stride = Math.max(1, Math.ceil(n / w));
    var step = n > 1 ? w / (n - 1) : w;

    function pointAt(i) {
      return { x: w - (n - 1 - i) * step,
               y: ht - 3 - state.ecg[i] * (ht - 8) };
    }

    /* A soft area under the line instead of shadowBlur. The glow was the most
     * expensive thing on this canvas -- recomputed over the whole path every
     * second -- and a fill is close to free. */
    /* Anchored to the trace, not to the canvas. Run from y=0 and the bright
     * end of the gradient sits above the line, in the region that is never
     * filled -- so the fill under a low score was drawn almost entirely in the
     * transparent tail and the area read as empty. */
    var top = ht, bot = 0;
    for (var g0 = 0; g0 < n; g0++) {
      var yy = ht - 3 - state.ecg[g0] * (ht - 8);
      if (yy < top) top = yy;
      if (yy > bot) bot = yy;
    }
    var grad = ctx.createLinearGradient(0, top, 0, Math.max(bot + 6, ht));
    grad.addColorStop(0, col + '4c');
    // A mid stop, so a trace that spends most of its range low still has a
    // visible body under it. With two stops the fill under a falling line sits
    // entirely in the transparent tail.
    grad.addColorStop(0.6, col + '22');
    grad.addColorStop(1, col + '00');
    ctx.beginPath();
    ctx.moveTo(pointAt(0).x, ht);
    for (var i = 0; i < n; i += stride) { var p = pointAt(i); ctx.lineTo(p.x, p.y); }
    var last = pointAt(n - 1);
    ctx.lineTo(last.x, last.y);
    ctx.lineTo(last.x, ht);
    ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    ctx.strokeStyle = col; ctx.lineWidth = 1.4;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (var k = 0, first = true; k < n; k += stride, first = false) {
      var q = pointAt(k);
      if (first) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
    }
    ctx.lineTo(last.x, last.y);
    ctx.stroke();

    // The live end, so a flatline still reads as something running.
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(last.x - 1, last.y, 1.8, 0, Math.PI * 2); ctx.fill();
  }

  function renderPnl(s) {
    var fit = fitCanvas($('pnl'));
    if (!fit) return;
    var ctx = fit.ctx, w = fit.w, h = fit.h;
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

  /* ---------- risk, limits and the halt control ---------- */

  /* A gauge shows a spent fraction against its own ceiling, with the ceiling
   * marked. "60% of equity deployed" means nothing without the 80% ceiling
   * beside it. */
  function gauge(label, value, ceiling, text, tone) {
    var pct = ceiling > 0 ? Math.min(100, (value / ceiling) * 100) : 0;
    return '<div class="gauge ' + (tone || '') + '">' +
      '<span class="g-label">' + esc(label) + '</span>' +
      '<span class="g-value">' + esc(text) + '</span>' +
      '<span class="g-track"><i class="g-fill" style="width:' + pct.toFixed(1) + '%"></i>' +
      '<i class="g-mark" style="right:0"></i></span>' +
      '</div>';
  }

  function cell(k, v, tone, sub) {
    return '<div><span class="k">' + esc(k) + '</span><span class="v ' +
      (tone || '') + '">' + esc(v) +
      (sub ? ' <small>' + esc(sub) + '</small>' : '') + '</span></div>';
  }

  function renderRisk(s) {
    var L = s.limits, dd = s.drawdown;

    var grossTone = s.gross_exposure >= L.max_gross_exposure * 0.95 ? 'bad'
      : s.gross_exposure >= L.max_gross_exposure * 0.75 ? 'warn' : '';
    var slotTone = L.slots_used >= L.slots_max ? 'warn' : '';
    var ddTone = dd.used >= 0.75 ? 'bad' : dd.used >= 0.4 ? 'warn' : 'good';

    $('gauges').innerHTML =
      gauge('gross exposure', s.gross_exposure, L.max_gross_exposure,
            (s.gross_exposure * 100).toFixed(1) + '% of ' +
            (L.max_gross_exposure * 100).toFixed(0) + '%', grossTone) +
      gauge('concurrency slots', L.slots_used, L.slots_max,
            L.slots_used + ' of ' + L.slots_max, slotTone) +
      gauge('daily loss budget', dd.used, 1,
            (dd.pct * 100).toFixed(2) + '% of ' + (dd.limit * 100).toFixed(0) + '%',
            ddTone);

    $('limit-grid').innerHTML =
      cell('budget / sym', (s.per_symbol_budget * 100).toFixed(1) + '%') +
      cell('position cap', (L.max_position_weight * 100).toFixed(0) + '%') +
      cell('buying power', fmtMoney(L.buying_power),
           '', (L.buying_power_reserve * 100).toFixed(0) + '% held') +
      cell('risk / trade', (L.risk_per_trade * 100).toFixed(2) + '%') +
      cell('target vol', (L.target_volatility * 100).toFixed(0) + '%') +
      cell('ATR stop', L.atr_stop_multiple + 'x') +
      cell('day trades', L.day_trade_count + ' / ' + L.pdt_max_day_trades,
           L.pdt_blocked ? 'bad' : '',
           s.equity < L.pdt_floor ? 'under $25k' : 'no PDT limit');

    /* On a small account "why is it not trading" is almost always the size of
     * the account, and the answer is arithmetic rather than a fault. Shown
     * next to the limits it produced, because a 40% position cap looks reckless
     * until you know it is $28. */
    var sc = s.account_scale || {};
    var sn = $('scale-note');
    if (sn) {
      if (sc.scaled) {
        sn.textContent = sc.positions + ' × $' + fmtNum(sc.max_position_value, 2) +
          ' max · floor $' + fmtNum(sc.position_floor, 0) +
          ' · overnight needs a share under $' +
          fmtNum(sc.overnight_max_share_price, 2);
        sn.className = 'notice compact scaled';
        sn.hidden = false;
        sn.title = sc.note || '';
      } else {
        sn.hidden = true;
      }
    }

    $('risk-note').textContent = s.halted ? 'HALTED'
      : (sc.scaled ? 'scaled to $' + fmtNum(sc.equity, 2) : 'live bounds');
    var btn = $('btn-halt');
    btn.textContent = s.halted ? 'Release book' : 'Halt book';
    btn.className = s.halted ? 'armed' : 'danger';
    $('halt-state').textContent = s.halted
      ? s.halt_reason
      : 'a halt blocks new exposure; exits always pass';
  }

  /* ---------- regime census ---------- */

  var REGIME_LABEL = {
    trending: 'trending', mean_reverting: 'mean reverting',
    indeterminate: 'indeterminate', contradicted: 'contradicted',
    warming_up: 'warming up'
  };
  var REGIME_COLOR = {
    trending: '#35d69b', mean_reverting: '#c678f0', indeterminate: '#46536a',
    contradicted: '#e8b444', warming_up: '#35a7ff'
  };

  function renderCensus(s) {
    var census = s.regime_census || {};
    var total = Object.keys(census).reduce(function (a, k) { return a + census[k]; }, 0);
    var order = ['trending', 'mean_reverting', 'indeterminate', 'contradicted',
                 'warming_up'];
    $('census').innerHTML = order.map(function (k) {
      var n = census[k] || 0;
      var pct = total ? (100 * n / total) : 0;
      return '<div class="census-row">' +
        '<span class="c-name">' + REGIME_LABEL[k] + '</span>' +
        '<span class="c-track"><i class="c-fill" style="width:' + pct.toFixed(1) +
        '%;background:' + REGIME_COLOR[k] + '"></i></span>' +
        '<span class="c-count">' + n + '</span></div>';
    }).join('');
    // "Warming up" resolves itself; the rest are decisions. Say which is which.
    var warming = census.warming_up || 0;
    $('census-note').textContent = warming
      ? warming + ' still warming up'
      : total + ' evaluated';
  }

  /* ---------- multi-day trend ---------- */

  function renderTrend(s) {
    var t = s.trend;
    if (!t) return;
    var o = t.options || {};

    $('tr-note').textContent = t.credible
      ? t.beta_bps.toFixed(2) + 'bp/day · t=' + t.t_stat.toFixed(1)
      : (t.measured ? 'premium not measurable yet' : 'not yet measured');

    /* The day-trade budget sits in this panel on purpose: it is the reason
     * this strategy is the one running on a small account, not a side note. */
    var dtTone = t.day_trades_available ? '' : 'warn';
    $('tr-grid').innerHTML =
      cell('premium', t.beta_bps.toFixed(2) + 'bp', t.credible ? 'good' : '',
           'per unit of trend') +
      cell('t-stat', (t.t_stat >= 0 ? '+' : '') + t.t_stat.toFixed(2),
           Math.abs(t.t_stat) >= 2 ? 'good' : '',
           t.credible ? 'credible' : 'not yet') +
      cell('sample', fmtNum(t.observations, 0), '', t.symbols + ' symbols') +
      cell('carrying', String((t.holdings || []).length), '', 'multi-day') +
      cell('day trades', t.day_trades_left + ' left', dtTone,
           t.day_trades_available ? 'intraday open' : 'intraday closed') +
      cell('options', o.affordable ? 'reachable' : 'need $' +
           fmtNum(o.min_equity, 0), '',
           '1 contract = ' + o.contract_multiplier + ' sh');

    var why = $('tr-why');
    if (!t.day_trades_available) {
      why.textContent = t.why;
    } else if (!t.credible) {
      why.textContent = t.note;
    } else {
      var held = (t.holdings || []).map(function (h) {
        return h.symbol + ' ' + h.days.toFixed(1) + '/' + h.min_days + 'd';
      });
      why.textContent = held.length ? 'carrying ' + held.join(' · ') : t.note;
    }
    why.title = (o.note || '');
  }

  /* ---------- overnight drift ---------- */

  /* The four windows the strategy moves through, in clock order. The two that
   * matter are the ones where an order is actually lodged; the strategy does
   * nothing at all in the other two, and showing that is the point. */
  var PHASES = [
    { k: 'closed',   label: 'closed',   hint: 'market shut, nothing to lodge' },
    { k: 'preopen',  label: 'pre-open', hint: 'market-on-open exits go in here' },
    { k: 'intraday', label: 'intraday', hint: 'the intraday blend has the book' },
    { k: 'closing',  label: 'closing',  hint: 'market-on-close entries go in here' }
  ];

  function renderOvernight(s) {
    var o = s.overnight;
    if (!o) return;
    $('on-note').textContent = o.credible
      ? o.mean_bps.toFixed(2) + 'bp/night · t=' + o.t_stat.toFixed(1)
      : (o.measured ? 'sample too thin to act on' : 'not yet measured');

    $('on-phase').innerHTML = PHASES.map(function (p) {
      var on = o.phase === p.k;
      return '<span class="phase' + (on ? ' on' : '') + '" title="' +
        esc(p.hint) + '">' + esc(p.label) + '</span>';
    }).join('');

    /* The drift is shown against the cost of capturing it, never alone. On its
     * own a few basis points a night reads as free money; next to a round trip
     * of the same order it reads as what it is. */
    var equity = (s.costs.by_asset_class || {}).us_equity;
    var rt = equity ? equity.median_round_trip_bps : 0;
    var driftTone = !o.credible ? '' : (o.mean_bps > rt ? 'good' : 'warn');
    var tTone = Math.abs(o.t_stat) >= 2 ? 'good' : '';

    $('on-grid').innerHTML =
      cell('drift / night', o.mean_bps.toFixed(2) + 'bp', driftTone,
           rt ? 'cost ' + rt.toFixed(1) + 'bp' : 'cost unknown') +
      cell('intraday', o.intraday_bps.toFixed(2) + 'bp', '', 'same sessions') +
      cell('t-stat', (o.t_stat >= 0 ? '+' : '') + o.t_stat.toFixed(2), tTone,
           o.credible ? 'credible' : 'not yet') +
      cell('sample', fmtNum(o.observations, 0), '',
           o.symbols + ' symbols') +
      cell('eligible', o.eligible + ' / ' + o.candidates, '', 'this close') +
      cell('held overnight', String(o.holdings.length), '',
           o.exempt_from_pdt ? 'not day trades' : '');

    var v = $('on-verdict');
    if (!o.measured) {
      v.textContent = o.note;
    } else if (!o.credible) {
      v.textContent = o.note + ' — too little to act on, so nothing is traded on it';
    } else if (o.mean_bps <= rt) {
      /* The honest headline for this anomaly, and the most common state. */
      v.textContent = 'the drift is real and smaller than the ' + rt.toFixed(1) +
        'bp it costs to capture — refusing is the correct answer, not a fault';
    } else {
      v.textContent = o.note;
    }
  }

  /* ---------- execution and costs ---------- */

  function renderCosts(s) {
    var c = s.costs;
    var feeTone = c.fees_assumed ? 'warn' : 'good';
    /* Reported per asset class, because the two are not comparable: an equity
     * round trip is almost all spread, a crypto one almost all commission. A
     * blended figure would describe neither. */
    var byClass = c.by_asset_class || {};
    var perClass = '';
    ['us_equity', 'crypto'].forEach(function (k) {
      var b = byClass[k];
      if (!b) return;
      perClass += cell(b.display_name + ' RT',
                       b.median_round_trip_bps.toFixed(1) + 'bp', '',
                       b.symbols + ' sym');
      perClass += cell(b.display_name + ' fees',
                       b.commission_bps.toFixed(1) + '+' +
                       b.sell_side_bps.toFixed(1) + 'bp', feeTone,
                       b.assumed ? 'assumed' : 'confirmed');
    });
    $('cost-grid').innerHTML = perClass +
      cell('safety multiple', c.safety_multiple + 'x') +
      cell('adv. selection', c.adverse_selection_fraction.toFixed(2),
           'warn', 'assumed') +
      cell('spreads live', c.spreads_measured + '/' + c.spreads_total,
           c.spreads_measured < c.spreads_total ? 'warn' : 'good');

    var parts = [];
    if (c.cheapest) parts.push('cheapest ' + c.cheapest.symbol + ' ' +
                               c.cheapest.bps.toFixed(1) + 'bp');
    if (c.dearest) parts.push('dearest ' + c.dearest.symbol + ' ' +
                              c.dearest.bps.toFixed(1) + 'bp');
    $('cost-note').textContent = parts.join(' · ') || '—';

    // An assumption nobody is told about becomes a fact by default.
    /* One line, expandable. The warning has to stay -- an assumed fee moves the
     * trade/no-trade line directly, and an assumption nobody is told about
     * becomes a fact by default -- but it is a standing condition, not news,
     * and six permanent lines of it were being paid for out of the reasoning
     * panel every second of every session. */
    var warn = $('cost-warn');
    if (!c.fees_assumed) { warn.innerHTML = ''; warn.dataset.built = ''; return; }
    if (warn.dataset.built !== 'yes') {
      warn.innerHTML =
        '<div class="notice compact" id="fee-notice" role="button" tabindex="0">' +
        '<span class="fee-head">fees assumed — affects the trade/no-trade line' +
        '<i class="chev">\u25be</i></span>' +
        '<span class="fee-body" id="fee-body" hidden></span></div>';
      var box = $('fee-notice');
      var toggle = function () {
        var body = $('fee-body');
        body.hidden = !body.hidden;
        box.querySelector('.chev').textContent = body.hidden ? '\u25be' : '\u25b4';
      };
      box.addEventListener('click', toggle);
      box.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
      });
      warn.dataset.built = 'yes';
    }
    var body = $('fee-body');
    var full = 'Not confirmed for this account (' + c.fee_source +
               '). It moves the trade/no-trade line directly.';
    if (body.textContent !== full) body.textContent = full;
    $('fee-notice').title = full;
  }

  /* ---------- session statistics ---------- */

  function fmtDuration(sec) {
    if (!sec || sec < 1) return '—';
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60),
        x = Math.floor(sec % 60);
    return h ? h + 'h ' + m + 'm' : m ? m + 'm ' + x + 's' : x + 's';
  }
  function fmtCount(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
    return String(n || 0);
  }

  function renderStats(s) {
    var c = s.counters || {};
    var evaluated = (c.decision || 0) + (c.refused || 0) + (c.cap || 0) +
                    (c.warmup || 0);
    var acted = c.order || 0;
    var refusalRate = evaluated ? (100 * (c.refused || 0) / evaluated) : 0;

    $('stat-grid').innerHTML =
      '<div><span class="k">bars eval</span><span class="v">' +
        fmtCount(evaluated) + '</span></div>' +
      '<div><span class="k">decisions</span><span class="v">' +
        fmtCount(c.decision || 0) + '</span></div>' +
      '<div><span class="k">refused</span><span class="v">' +
        fmtCount(c.refused || 0) + '</span></div>' +
      '<div><span class="k">capped</span><span class="v">' +
        fmtCount(c.cap || 0) + '</span></div>' +
      '<div><span class="k">orders</span><span class="v">' +
        fmtCount(acted) + '</span></div>' +
      '<div><span class="k">refusal rate</span><span class="v">' +
        refusalRate.toFixed(0) + '%</span></div>';

    // "Nothing trading" is the normal case, so say so rather than leaving a
    // zero that reads as a fault.
    $('perf-note').textContent = acted
      ? fmtCount(acted) + ' orders · ' + fmtDuration(s.uptime)
      : (evaluated ? 'scanning, nothing has cleared its costs yet'
                   : 'no bars evaluated yet');
  }

  /* ---------- footer ---------- */

  function renderFooter(s) {
    var v = s.venue_budget || {};
    $('foot-venue').textContent = s.venue + ' ' + (s.environment || '') +
      ' · ' + s.mode + ' · ' + ((s.market && s.market.feed) || '');
    /* Alpaca reports what is left, not what was spent. Shown that way round
     * rather than converted, so it matches the header it came from. */
    $('foot-weight').textContent = v.limit
      ? 'requests ' + v.remaining + '/' + v.limit + ' left' +
        (v.throttled ? ' · throttled ' + v.retry_after.toFixed(1) + 's' : '')
      : 'requests —';
    /* The scan note alone reads as a stall: it is the same sentence for
     * fifteen minutes between full re-ranks. Saying when the next one lands,
     * and that the evaluation sweep is still turning over in between, is the
     * difference between "hung" and "on a schedule". */
    var us = s.universe_scan || {};
    var bits = [us.note || ''];
    if (us.next_scan_in > 0) bits.push('re-rank in ' + fmtDuration(us.next_scan_in));
    if (us.sweeps !== undefined) {
      bits.push('sweep ' + (us.sweep_at || 0) + '/' + (us.size || 0) +
                ' · ' + fmtCount(us.sweeps) + ' passes');
    }
    $('foot-clock').textContent = bits.filter(Boolean).join(' · ');
    var c = s.counters || {};
    $('foot-work').textContent =
      fmtCount(c.scan || 0) + ' scans · ' + fmtCount(c.decision || 0) +
      ' decisions · ' + fmtCount(c.order || 0) + ' orders';
    $('foot-uptime').textContent = s.running
      ? 'up ' + fmtDuration(s.uptime) : 'stopped';
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
      /* Absorbed even while hidden, so the log is complete on return; only the
       * rendering is skipped. Skipping the absorb instead would silently
       * discard delta frames the server will never send again. */
      if (document.hidden) { absorbEvents(s); cluster.ingest(s.pulses); return; }
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
    renderRisk(s);
    renderCensus(s);
    renderTrend(s);
    renderOvernight(s);
    renderCosts(s);
    renderStats(s);
    renderExecution(s);
    renderFooter(s);

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
    /* The cluster animates continuously, which is the right behaviour for a
     * panel someone is watching and pure waste for one nobody is. On a machine
     * left running for days this is the difference between a warm laptop and a
     * hot one -- and browsers throttle background rAF unevenly, so relying on
     * them to do it produces stutter on return rather than a clean resume. */
    if (!document.hidden) cluster.frame(now);
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

  $('btn-halt').onclick = function () {
    var halted = state.snapshot && state.snapshot.halted;
    post('/api/session/halt', {
      halted: !halted, reason: 'halted by the operator from the terminal'
    }).catch(alertErr);
  };

  $('btn-conn').onclick = function () {
    $('conn-modal').hidden = false;
    loadConnections();
  };
  $('c-close').onclick = function () { $('conn-modal').hidden = true; };

  function loadConnections() {
    fetch('/api/connections').then(function (r) { return r.json(); }).then(function (j) {
      var perm = $('conn-perm');
      if (j.error) {
        perm.innerHTML = '<div class="notice bad"></div>';
        perm.firstChild.textContent = j.error;
      } else if (!j.permissions_ok) {
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
  /* ---------- collapsible panels ---------- */

  /* Remembered per panel, because a terminal meant to run for days should not
   * make the operator re-fold it every time the page reloads. localStorage can
   * throw outright in a private window or with site data blocked, so every
   * access is guarded: a layout preference is never worth a dead script. */
  var FOLD_KEY = 'imperium.folded';

  function foldedSet() {
    try {
      return new Set(JSON.parse(localStorage.getItem(FOLD_KEY) || '[]'));
    } catch (e) { return new Set(); }
  }

  function rememberFolds(set) {
    try {
      localStorage.setItem(FOLD_KEY, JSON.stringify(Array.from(set)));
    } catch (e) { /* nothing to do, and nothing worth breaking over */ }
  }

  function panelKey(panel, i) {
    var h = panel.querySelector('h2');
    return panel.id || (h ? h.textContent.trim().split(' ')[0] : 'p' + i);
  }

  function initFolding() {
    var folded = foldedSet();
    Array.prototype.forEach.call(document.querySelectorAll('.panel'),
      function (panel, i) {
        var h2 = panel.querySelector('h2');
        if (!h2) return;
        var key = panelKey(panel, i);
        if (folded.has(key)) panel.classList.add('collapsed');
        h2.tabIndex = 0;
        h2.title = 'click to fold this panel';
        var toggle = function () {
          panel.classList.toggle('collapsed');
          var now = foldedSet();
          if (panel.classList.contains('collapsed')) now.add(key); else now.delete(key);
          rememberFolds(now);
          // Canvases inside a panel that just changed size need re-fitting, and
          // the cluster owns its own backing store.
          cluster.resize();
          if (state.snapshot) { try { render(state.snapshot); } catch (e) {} }
        };
        h2.addEventListener('click', toggle);
        h2.addEventListener('keydown', function (e) {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
        });
      });
  }

  initFolding();

  /* The legend takes its swatches from the canvas palette rather than carrying
   * its own copy of the same hex codes. */
  if (window.Cluster && window.Cluster.paintLegend) {
    window.Cluster.paintLegend(document.getElementById('cluster-legend'));
  }

  window.addEventListener('resize', function () { cluster.resize(); });
  document.addEventListener('visibilitychange', function () {
    // Repaint on return rather than showing a second of stale panel.
    if (!document.hidden && state.snapshot) {
      cluster.resize();
      try { render(state.snapshot); } catch (e) { console.error(e); }
    }
  });
})();
