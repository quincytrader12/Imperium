"""The equity chart as a picture, for Telegram.

Telegram is the terminal's phone view -- the terminal itself binds to this
machine only -- and a line says in a glance what a paragraph of numbers takes
a minute to. So the brief carries one.

Drawn with numpy and written as PNG by hand, deliberately. The Windows build
excludes Pillow and matplotlib: together they would add tens of megabytes to
an executable that Defender scans file by file on first launch, for one image
a day. Everything a chart needs -- a smooth line, a filled area, a few labels
-- is a few arrays:

* lines are stamped as discs along the path, at twice the final size, and the
  image is averaged down by two, which is anti-aliasing by supersampling;
* labels use a small built-in pixel font, which suits an instrument panel and
  is the one thing here that is legible at any size Telegram shows it;
* the PNG is a zlib stream with the three chunks the format requires.

The palette is the terminal's own: the same background, the same neutral line,
green and red for the day's direction only.
"""

from __future__ import annotations

import datetime as dt
import math
import struct
import zlib
from typing import Sequence

import numpy as np

BG = (10, 11, 13)
GRID = (24, 26, 31)
LINE = (223, 229, 236)
TEXT = (219, 224, 230)
DIM = (141, 147, 155)
GOOD = (47, 224, 164)
BAD = (255, 92, 108)

SCALE = 2          # supersampling factor

# A 5x7 pixel font, rows top to bottom. Only what the chart prints.
_FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "$": ["00100", "01111", "10100", "01110", "00101", "11110", "00100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "00100", "01000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "·": ["00000", "00000", "00000", "01100", "01100", "00000", "00000"],
    " ": ["00000"] * 7,
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}
_GLYPHS = {k: np.array([[c == "1" for c in row] for row in v], dtype=bool)
           for k, v in _FONT.items()}


# -- the canvas -----------------------------------------------------------------


class Canvas:
    """An RGB float image at supersampled resolution."""

    def __init__(self, width: int, height: int) -> None:
        self.w, self.h = width * SCALE, height * SCALE
        self.px = np.empty((self.h, self.w, 3), dtype=np.float32)
        self.px[:] = BG

    def blend(self, mask: np.ndarray, color: Sequence[int],
              alpha: float | np.ndarray = 1.0) -> None:
        """Paint ``color`` wherever ``mask`` (0..1 coverage) is set.

        Only inside the mask's bounding box: a label is a few hundred pixels
        of a two-million-pixel canvas, and blending all of it for each of
        thirty labels was most of the time the chart took to draw."""
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if not rows.size:
            return
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        a = mask[r0:r1, c0:c1].astype(np.float32)
        if isinstance(alpha, np.ndarray):
            a = a * alpha[r0:r1, c0:c1]
        else:
            a = a * alpha
        a = a[..., None]
        region = self.px[r0:r1, c0:c1]
        region[:] = region * (1 - a) + np.asarray(color, np.float32) * a

    def hline(self, y: float, x0: float, x1: float, color, width: float = 1.0,
              dash: int = 0) -> None:
        y0 = int(round(y * SCALE))
        t = max(1, int(round(width * SCALE)))
        mask = np.zeros((self.h, self.w), dtype=np.float32)
        a, b = int(x0 * SCALE), int(x1 * SCALE)
        mask[max(0, y0):min(self.h, y0 + t), max(0, a):min(self.w, b)] = 1
        if dash:
            cols = np.arange(self.w)
            mask[:, (cols // (dash * SCALE)) % 2 == 1] = 0
        self.blend(mask, color)

    def polyline(self, xs: np.ndarray, ys: np.ndarray, color,
                 width: float = 2.0, alpha: float = 1.0) -> None:
        """A smooth line: discs stamped every half pixel along the path."""
        xs = np.asarray(xs, float) * SCALE
        ys = np.asarray(ys, float) * SCALE
        r = width * SCALE / 2
        # Densify so consecutive stamps overlap.
        px, py = [xs[:1]], [ys[:1]]
        for i in range(1, len(xs)):
            d = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            n = max(1, int(d / 0.5))
            t = np.linspace(0, 1, n + 1)[1:]
            px.append(xs[i - 1] + (xs[i] - xs[i - 1]) * t)
            py.append(ys[i - 1] + (ys[i] - ys[i - 1]) * t)
        px_, py_ = np.concatenate(px), np.concatenate(py)
        mask = np.zeros((self.h, self.w), dtype=np.float32)
        R = int(math.ceil(r)) + 1
        oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
        disc = np.clip(r + 0.5 - np.hypot(ox, oy), 0, 1).astype(np.float32)
        for cx, cy in zip(np.round(px_).astype(int), np.round(py_).astype(int)):
            y0, y1 = cy - R, cy + R + 1
            x0, x1 = cx - R, cx + R + 1
            if y1 <= 0 or x1 <= 0 or y0 >= self.h or x0 >= self.w:
                continue
            sy0, sx0 = max(0, -y0), max(0, -x0)
            sy1 = disc.shape[0] - max(0, y1 - self.h)
            sx1 = disc.shape[1] - max(0, x1 - self.w)
            region = mask[max(0, y0):min(self.h, y1), max(0, x0):min(self.w, x1)]
            np.maximum(region, disc[sy0:sy1, sx0:sx1], out=region)
        self.blend(mask, color, alpha)

    def area(self, xs: np.ndarray, ys: np.ndarray, base: float, color,
             top_alpha: float, bottom_alpha: float = 0.0) -> None:
        """Fill between a line and ``base``, fading from top to bottom."""
        cols = np.arange(self.w) / SCALE
        line = np.interp(cols, xs, ys, left=np.nan, right=np.nan)
        rows = (np.arange(self.h) / SCALE)[:, None]
        lo = np.minimum(line, base)[None, :]
        hi = np.maximum(line, base)[None, :]
        inside = (rows >= lo) & (rows <= hi) & np.isfinite(line)[None, :]
        # Faded by height, the same at every column: a wash, not a stain
        # that darkens wherever the line happens to come close to the base.
        finite = line[np.isfinite(line)]
        far = float(finite.min() if base >= finite.max() else finite.max()) if finite.size else 0.0
        span = max(1e-6, abs(base - far))
        depth = np.clip(np.abs(rows - far) / span, 0, 1)
        alpha = np.broadcast_to(top_alpha + (bottom_alpha - top_alpha) * depth,
                                (self.h, self.w))
        self.blend(inside.astype(np.float32), color, alpha)

    def between(self, xs: np.ndarray, y_top: np.ndarray, y_bottom: np.ndarray,
                color, alpha: float) -> None:
        """Fill between two lines that share their x positions."""
        cols = np.arange(self.w) / SCALE
        a = np.interp(cols, xs, y_top, left=np.nan, right=np.nan)
        b = np.interp(cols, xs, y_bottom, left=np.nan, right=np.nan)
        rows = (np.arange(self.h) / SCALE)[:, None]
        inside = ((rows >= np.minimum(a, b)[None, :])
                  & (rows <= np.maximum(a, b)[None, :])
                  & (np.abs(b - a) > 0.25)[None, :])
        self.blend(inside.astype(np.float32), color, alpha)

    def text(self, x: float, y: float, s: str, color, size: int = 2,
             align: str = "left") -> float:
        """Pixel-font text; ``size`` is output pixels per font pixel.
        Returns the text's width in output pixels."""
        s = s.upper()
        step = 6 * size
        width = step * len(s) - size
        if align == "right":
            x -= width
        elif align == "center":
            x -= width / 2
        mask = np.zeros((self.h, self.w), dtype=np.float32)
        k = size * SCALE
        for i, ch in enumerate(s):
            glyph = _GLYPHS.get(ch, _GLYPHS[" "])
            big = np.kron(glyph, np.ones((k, k), dtype=bool))
            gx, gy = int((x + i * step) * SCALE), int(y * SCALE)
            h, w = big.shape
            if gy < 0 or gx < 0 or gy + h > self.h or gx + w > self.w:
                continue
            mask[gy:gy + h, gx:gx + w] = np.maximum(mask[gy:gy + h, gx:gx + w], big)
        self.blend(mask, color)
        return width

    def rgb(self) -> np.ndarray:
        """Averaged down by the supersampling factor: the anti-aliasing."""
        h, w = self.h // SCALE, self.w // SCALE
        small = self.px[:h * SCALE, :w * SCALE].reshape(h, SCALE, w, SCALE, 3)
        return np.clip(small.mean(axis=(1, 3)), 0, 255).astype(np.uint8)


def png(rgb: np.ndarray) -> bytes:
    """An RGB array as PNG bytes: signature, IHDR, one IDAT, IEND."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


# -- the chart ------------------------------------------------------------------


def _money(v: float, signed: bool = False) -> str:
    digits = 0 if abs(v) >= 10_000 else 2
    s = f"${abs(v):,.{digits}f}"
    if v < 0:
        return "-" + s
    return ("+" + s) if signed else s


def _axis(v: float) -> str:
    """Axis labels short enough for the margin: whole dollars from $1,000."""
    return f"${v:,.0f}" if abs(v) >= 1000 else f"${v:,.2f}"


def _date(ts: float) -> str:
    d = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    return f"{d.day} {d.strftime('%b').upper()}"


def render_equity(points: Sequence[tuple[float, float]], *, title: str,
                  width: int = 960, height: int = 520) -> bytes | None:
    """The equity line over ``points`` as PNG bytes, or None with fewer than
    two points -- a chart of one reading is not a chart."""
    pts = [(float(t), float(v)) for t, v in points
           if math.isfinite(t) and math.isfinite(v) and v > 0]
    if len(pts) < 2:
        return None
    ts = np.array([p[0] for p in pts])
    vs = np.array([p[1] for p in pts])
    first, last = vs[0], vs[-1]
    up = last >= first
    tone = GOOD if up else BAD

    c = Canvas(width, height)
    left, right, top = 28, width - 128, 108
    main_bottom = height - 52

    # Header: what, the value, the change.
    c.text(left, 26, "IMPERIUM", DIM, size=2)
    c.text(left + 120, 26, "· " + title, DIM, size=2)
    c.text(left, 52, _money(last), TEXT, size=5)
    change = last - first
    pct = change / first if first else 0.0
    c.text(width - 28, 58, f"{_money(change, True)}  ({pct:+.2%})", tone,
           size=3, align="right")

    lo, hi = float(vs.min()), float(vs.max())
    pad = max((hi - lo) * 0.10, hi * 0.002, 0.01)
    lo, hi = lo - pad, hi + pad
    X = left + (ts - ts[0]) / max(1e-9, ts[-1] - ts[0]) * (right - left)
    Y = main_bottom - (vs - lo) / (hi - lo) * (main_bottom - top)

    # Grid and axis labels: high, middle, low.
    for frac in (0.0, 0.5, 1.0):
        value = lo + (hi - lo) * (1 - frac)
        y = top + (main_bottom - top) * frac
        c.hline(y, left, right, GRID)
        c.text(right + 12, y - 7, _axis(value), DIM, size=2)
    base_y = main_bottom - (first - lo) / (hi - lo) * (main_bottom - top)
    c.hline(base_y, left, right, (58, 63, 72), dash=4)

    c.area(X, Y, main_bottom, tone, 0.20, 0.0)

    # The drawdown as the terminal draws it: a red veil between the
    # high-water mark and the line, only while the book is below its high,
    # and the high itself as a fine rule above.
    peak = np.maximum.accumulate(vs)
    dd = vs / peak - 1.0
    worst = float(dd.min())
    HY = main_bottom - (peak - lo) / (hi - lo) * (main_bottom - top)
    c.between(X, HY, Y, BAD, 0.22)
    c.polyline(X, HY, DIM, width=1.0, alpha=0.6)
    c.polyline(X, Y, LINE, width=2.6)
    c.polyline(X[-1:], Y[-1:], LINE, width=9)
    if worst < -1e-4:
        at = int(np.argmin(dd))
        c.polyline(np.array([X[at], X[at]]), np.array([HY[at], Y[at]]), BAD,
                   width=1.4, alpha=0.9)
        label = f"{worst:.1%}"
        lx = X[at] + 10
        if lx + 6 * 2 * len(label) > right:
            lx = X[at] - 10 - 6 * 2 * len(label)
        c.text(lx, (HY[at] + Y[at]) / 2 - 7, label, BAD, size=2)

    c.text(left, height - 26, _date(ts[0]), DIM, size=2)
    c.text(right, height - 26, _date(ts[-1]), DIM, size=2, align="right")
    return png(c.rgb())
