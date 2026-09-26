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
 * until something actually happens. The ones that mean something carry a
 * neon: green for a verdict, purple for the cost gate, orange for money
 * moving, red for a halt.
 *
 * THE COLOURS ARE CHOSEN BY MEASUREMENT, NOT BY EYE.
 *
 * "It is difficult to tell scan from refused, and warmup and order look
 * almost the same." Both were true, and both were measurable. Converted to
 * CIELAB and compared as a distance, the palette that reading came from had a
 * closest pair of dE 25.8 (order against warmup) and a second at 42.6 (scan
 * against refused) -- the two the operator named, in the order they named
 * them. Everything else sat above 54.
 *
 * Two of them were confusable because they were the same colour twice: scan
 * and refused were both greys separated only by brightness, and warmup and
 * order were both oranges about twenty degrees of hue apart.
 *
 * So the meanings were given hues rather than shades of one:
 *
 *   refused  deep indigo. Looked at, nothing there -- cold and dim, and no
 *            longer a darker version of the scan it follows.
 *   warmup   electric cyan. Not enough history yet: something incubating,
 *            which is a cold idea and never was an amber one.
 *   order    hot orange, alone in its half of the wheel now that warmup has
 *            left it. Money actually moved.
 *
 * The search was over hue, saturation and value within those meanings,
 * maximising the *smallest* pairwise distance rather than the average -- the
 * average is what a palette with one invisible pair still scores well on. The
 * result separates every pair by at least dE 50, against 25.8 before, and the
 * floor is asserted by a test so it cannot quietly drift back.
 */
(function (global) {
  'use strict';

  var KIND_COLOR = {
    scan:     [196, 208, 226],    // cool silver — the background hum
    decision: [57, 255, 140],     // neon green — a verdict was reached
    refused:  [57, 75, 128],      // deep indigo — looked at, nothing there
    cap:      [190, 60, 255],     // neon purple — the cost gate said no
    order:    [255, 134, 13],     // hot orange — money actually moved
    warmup:   [0, 242, 226],      // electric cyan — not enough history yet
    halt:     [255, 92, 108]      // red — the book stopped itself
  };

  /* The floor the palette was built to, in CIELAB dE76. Exported so the test
   * that guards it reads the number from here rather than carrying its own
   * copy of it. */
  var MIN_SEPARATION = 45;

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
      /* Lit from its own colour. A flat 6px dot on a dark panel is a shade;
       * the same dot with its own colour bleeding out of it is a light, and a
       * light is what the eye picks a kind out by at a glance. Scan is left
       * nearly unlit on purpose -- it is the hum, and a glowing hum would
       * make the legend as loud as the things that matter. */
      dot.style.boxShadow = kind === 'scan'
        ? '0 0 3px ' + css(kind, 0.35)
        : '0 0 5px ' + css(kind, 0.95) + ', 0 0 11px ' + css(kind, 0.45);
      item.appendChild(dot);
      item.appendChild(document.createTextNode(KIND_LABEL[kind] || kind));
      host.appendChild(item);
    }
  }

  global.Palette = {
    KIND_COLOR: KIND_COLOR,
    MIN_SEPARATION: MIN_SEPARATION,
    KIND_LABEL: KIND_LABEL,
    KIND_ORDER: KIND_ORDER,
    css: css,
    unit: unit,
    paintLegend: paintLegend
  };
})(window);
