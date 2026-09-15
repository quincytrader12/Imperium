/* Time formatting for the terminal header.
 *
 * Its own file so it can be exercised under Node by the test suite. The venue
 * publishes its calendar in UTC, which is right for a log and useless on
 * screen: an operator outside UTC has to convert it in their head every time
 * they glance at the header. These render the viewer's own clock, and lead
 * with a countdown, which needs no timezone at all.
 */
(function (global) {
  'use strict';

  function fmtLocalTime(iso, now) {
    if (!iso) return '';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    // 'numeric' rather than '2-digit': on a 12-hour locale the latter
    // renders "08:25 PM", which reads as a padded field rather than a
    // time. 24-hour locales are unaffected in practice.
    return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }

  /* A countdown to an ISO instant. Timezone-free by construction, which is the
   * point: "opens in 4h 12m" is the same sentence in every country. */
  function fmtUntil(iso, now) {
    if (!iso) return '';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    var sec = Math.round((d.getTime() - (now === undefined ? Date.now() : now)) / 1000);
    if (sec <= 0) return 'now';
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
    return h ? h + 'h ' + m + 'm' : m + 'm';
  }

  global.Fmt = { fmtLocalTime: fmtLocalTime, fmtUntil: fmtUntil };
})(typeof window !== 'undefined' ? window : globalThis);
