/* The process palette, and the only copy of it.
 *
 * One colour per process kind. Everything that draws a process -- the orb's
 * internal zones, its surface ripples, the legend under it, the bloom tint --
 * reads from here, so a colour cannot be changed in one place and stay stale
 * in another. The kinds themselves are validated server-side in
 * imperium/telemetry/streams.py (PULSE_KINDS); a kind that exists there and
 * not here would render untinted, so the two sets are asserted equal by a
 * test rather than kept in step by memory.
 *
 * Silver for the hum, colour only for meaning. The scan pulse is by far the
 * commonest thing the terminal emits, so whatever colour it is, that is the
 * colour of the terminal. It is silver, and the display stays monochrome
 * until something actually happens. The four that mean something keep their
 * colours: green for a verdict, purple for the cost gate, orange for money
 * moving, red for a halt.
 */
(function (global) {
  'use strict';

  var KIND_COLOR = {
    scan:     [214, 224, 238],    // silver — the background hum
    decision: [57, 255, 140],     // neon green — a verdict was reached
    refused:  [104, 110, 120],    // graphite — looked at, nothing there
    cap:      [190, 60, 255],     // neon purple — the cost gate said no
    order:    [255, 150, 40],     // bright orange — money actually moved
    warmup:   [232, 180, 68],     // amber — not enough history yet
    halt:     [255, 92, 108]      // red — the book stopped itself
  };

  /* What each kind is called out loud, in the legend and the orb's readout.
   * The wording is the operator's, not the code's: "refused" is what the
   * cost gate does, "nothing there" is what it means. */
  var KIND_LABEL = {
    scan: 'scan',
    decision: 'decision',
    refused: 'refused',
    cap: 'cap',
    order: 'order',
    warmup: 'warmup',
    halt: 'halt'
  };

  /* The order kinds are listed in. Not alphabetical and not the object's own
   * key order: it runs from the quietest and commonest to the loudest and
   * rarest, so a legend read top to bottom is read in order of how much it
   * should worry you. */
  var KIND_ORDER = ['scan', 'refused', 'warmup', 'decision', 'cap', 'order',
                    'halt'];

  function css(kind, alpha) {
    var c = KIND_COLOR[kind] || KIND_COLOR.scan;
    return alpha === undefined
      ? 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')'
      : 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + alpha + ')';
  }

  /* 0..1 floats, which is what a shader uniform wants. */
  function unit(kind) {
    var c = KIND_COLOR[kind] || KIND_COLOR.scan;
    return [c[0] / 255, c[1] / 255, c[2] / 255];
  }

  /* Paint the legend from this palette, so the two cannot drift apart.
   *
   * Built rather than written into the markup: a legend typed by hand beside
   * a palette defined in code is a legend that disagrees with the field it
   * explains the first time a colour changes, and a legend that lies is worse
   * than no legend. */
  function paintLegend(host) {
    if (!host) return;
    host.innerHTML = '';
    for (var i = 0; i < KIND_ORDER.length; i++) {
      var kind = KIND_ORDER[i];
      var item = document.createElement('span');
      item.className = 'legend-item';
      var dot = document.createElement('i');
      dot.style.background = css(kind);
      item.appendChild(dot);
      item.appendChild(document.createTextNode(KIND_LABEL[kind] || kind));
      host.appendChild(item);
    }
  }

  global.Palette = {
    KIND_COLOR: KIND_COLOR,
    KIND_LABEL: KIND_LABEL,
    KIND_ORDER: KIND_ORDER,
    css: css,
    unit: unit,
    paintLegend: paintLegend
  };
})(window);
