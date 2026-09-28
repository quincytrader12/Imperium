"""Bar aggregation.

Klines seed history; the live trade stream extends it. The seam between the two
is where duplicate and out-of-order bars come from, so bars are keyed by their
open time and a repeated key updates in place rather than appending.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bar:
    open_time: int          # milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


class BarSeries:
    """A bounded, de-duplicated series of bars for one symbol."""

    def __init__(self, symbol: str, bar_seconds: int = 60, capacity: int = 1500) -> None:
        self.symbol = symbol
        self.bar_seconds = bar_seconds
        self.capacity = capacity
        self._bars: list[Bar] = []
        self._index: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def bars(self) -> list[Bar]:
        return self._bars

    @property
    def last(self) -> Bar | None:
        return self._bars[-1] if self._bars else None

    def add(self, bar: Bar) -> bool:
        """Append or update. Returns True if this closed a *new* bar.

        Only a newly closed bar is a decision point. Re-evaluating on every tick
        of an open bar would emit a pulse per tick, and a pulse must be a fact
        about a bar that was actually evaluated.
        """
        existing = self._index.get(bar.open_time)
        if existing is not None:
            was_closed = self._bars[existing].closed
            self._bars[existing] = bar
            return bar.closed and not was_closed
        if self._bars and bar.open_time < self._bars[-1].open_time:
            return False        # a late bar older than the series; ignore
        self._bars.append(bar)
        self._index[bar.open_time] = len(self._bars) - 1
        if len(self._bars) > self.capacity:
            drop = len(self._bars) - self.capacity
            self._bars = self._bars[drop:]
            self._index = {b.open_time: i for i, b in enumerate(self._bars)}
        return bar.closed

    def ingest_klines(self, rows: list[list]) -> int:
        """Seed from ``/api/v3/klines``.

        The final row is the *currently forming* bar and is marked open, because
        treating it as closed makes every strategy act on a partial bar.
        """
        added = 0
        for i, row in enumerate(rows):
            closed = i < len(rows) - 1
            bar = Bar(int(row[0]), float(row[1]), float(row[2]), float(row[3]),
                      float(row[4]), float(row[5]), closed=closed)
            if self.add(bar):
                added += 1
        return added

    # -- views used by the strategies ------------------------------------

    def closes(self, n: int | None = None) -> np.ndarray:
        vals = [b.close for b in self._bars if b.closed]
        arr = np.asarray(vals, dtype=float)
        return arr if n is None else arr[-n:]

    def highs(self, n: int | None = None) -> np.ndarray:
        arr = np.asarray([b.high for b in self._bars if b.closed], dtype=float)
        return arr if n is None else arr[-n:]

    def lows(self, n: int | None = None) -> np.ndarray:
        arr = np.asarray([b.low for b in self._bars if b.closed], dtype=float)
        return arr if n is None else arr[-n:]

    def closed_count(self) -> int:
        return sum(1 for b in self._bars if b.closed)

    def open_times(self, n: int | None = None) -> np.ndarray:
        arr = np.asarray([b.open_time for b in self._bars if b.closed], dtype=float)
        return arr if n is None else arr[-n:]

    def log_returns(self, n: int | None = None, *,
                    exclude_session_gaps: bool = False) -> np.ndarray:
        """Log returns, optionally dropping the ones that span a session gap.

        This is the single most important difference between an equity series
        and a crypto one. Consecutive *bars* are not consecutive *minutes* for
        an equity: between 16:00 and 09:30 the next day there is a seam, and
        across a weekend a much larger one. Treating close-to-open as a
        one-minute return does two things, both bad:

        * it inflates the volatility estimate, because a gap is typically many
          times a one-minute move -- and volatility is the denominator of the
          position sizer, so the whole book is then sized too small;
        * it corrupts the variance ratio, because a handful of huge
          pseudo-returns dominate both the numerator and the denominator and
          push the statistic toward the null regardless of what the market did.

        A gap is any interval longer than 1.5 bars, which separates a real
        missing-bar seam from ordinary jitter in bar timestamps.
        """
        closes = self.closes(n)
        if closes.size < 2:
            return np.zeros(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.diff(np.log(closes))
        finite = np.isfinite(rets)
        if not exclude_session_gaps:
            return rets[finite]

        times = self.open_times(n)
        if times.size != closes.size:
            return rets[finite]
        gaps_ms = np.diff(times)
        contiguous = gaps_ms <= (self.bar_seconds * 1000 * 1.5)
        keep = finite & contiguous
        return rets[keep]
