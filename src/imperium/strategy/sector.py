"""Sector Trend: Donchian/Keltner breakouts on liquid industry ETFs.

After Zarattini and Antonacci, *A Century of Profitable Industry Trends*. The
claim the paper rests on is narrow and worth stating plainly, because it is the
reason this strategy is allowed to be simple: industry indices trend, the
effect survives a century of out-of-sample data, and a breakout rule with a
volatility-scaled stop captures most of it. It is not a high-Sharpe strategy.
The paper's own numbers over 2005-2024 are a CAGR near 7.7% at a Sharpe near
0.6 with a 24% drawdown -- a slow, long-only, low-beta sleeve, and any backtest
of this code that reports much better than that is far more likely to have a
lookahead bug than an edge.

**Everything here is pure.** Arrays in, numbers out, no clock and no venue. The
live job and the backtester import these same functions, which is the only way
the two can be compared: a backtest that reimplements the signal is a backtest
of different code.

**The one subtlety that decides whether the results are real.** An entry
compares today's close against *yesterday's* upper band. The bands include the
current bar by construction -- ``DonchianUp20_t`` is the highest close of the
last twenty *including today* -- so comparing today's close against today's
band would be comparing a number with itself and would "enter" on every new
high with perfect foresight. :func:`entry_signal` takes the previous index
deliberately and :func:`bands` returns whole series so a caller cannot
accidentally align them by one bar in the wrong direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: The published parameters. Named rather than inlined so a reader can see at a
#: glance that nothing here was fitted by this program: these are the paper's
#: numbers, and :mod:`imperium.strategy.sector_config` is the only place they
#: may be changed.
DONCHIAN_UP_DAYS = 20
DONCHIAN_DOWN_DAYS = 40
KELTNER_UP_DAYS = 20
KELTNER_DOWN_DAYS = 40

#: The Keltner multiplier, applied to twice the mean absolute change.
#:
#: The paper writes the band as ``EMA ± 1.4 × 2 × MAD``. The doubling is kept
#: explicit rather than folded into 2.8 because the two factors mean different
#: things: the 2 converts a mean absolute deviation into something comparable
#: to a standard deviation, and the 1.4 is the band width the paper chose.
KELTNER_MULTIPLE = 1.4

#: Days of returns behind the volatility estimate that sizes a position.
SIGMA_DAYS = 14

#: The fewest daily bars a symbol needs before it may be traded.
#:
#: The paper's own floor. Note that it is a floor, not a recommendation: a
#: 40-day EMA computed from 60 bars is still substantially its own seed, so the
#: live job asks the venue for far more history than this and uses the minimum
#: only to refuse symbols that genuinely have no past.
MIN_BARS = 60

#: Bars requested in live use, so the 40-day EMA is converged rather than
#: seeded. Ten times the longest window is comfortably past the point where the
#: seeding convention stops mattering.
PREFERRED_BARS = 400


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average, seeded with the first ``span`` values' mean.

    The seed is stated because it is a choice and it is visible in short
    series: a recursive EMA has to start somewhere, and starting at the first
    observation rather than its window's mean puts a transient in the first few
    dozen bars. Positions ``< span - 1`` are NaN rather than a partial average,
    so a caller cannot read a number that had no window behind it.
    """
    values = np.asarray(values, dtype=float)
    out = np.full(values.size, np.nan, dtype=float)
    if values.size < span or span <= 0:
        return out
    alpha = 2.0 / (span + 1.0)
    level = float(np.mean(values[:span]))
    out[span - 1] = level
    for i in range(span, values.size):
        level = alpha * float(values[i]) + (1.0 - alpha) * level
        out[i] = level
    return out


def mean_absolute_change(closes: np.ndarray, window: int) -> np.ndarray:
    """``mean(|ΔP|)`` over the last ``window`` days, aligned to each close.

    Absolute *price* change, not return: the paper's band is drawn in price
    units around a price EMA, and a percentage measure would widen the band on
    a cheap ETF and narrow it on an expensive one for no reason connected to
    how far it moves.
    """
    closes = np.asarray(closes, dtype=float)
    out = np.full(closes.size, np.nan, dtype=float)
    if closes.size < window + 1 or window <= 0:
        return out
    deltas = np.abs(np.diff(closes))
    # deltas[i] is |P_{i+1} - P_i|, so the window ending at close t is
    # deltas[t-window:t]. The first usable t is therefore `window`.
    for t in range(window, closes.size):
        out[t] = float(np.mean(deltas[t - window:t]))
    return out


def rolling_max(closes: np.ndarray, window: int) -> np.ndarray:
    """Highest close of the last ``window`` days, including today."""
    closes = np.asarray(closes, dtype=float)
    out = np.full(closes.size, np.nan, dtype=float)
    for t in range(window - 1, closes.size):
        out[t] = float(np.max(closes[t - window + 1:t + 1]))
    return out


def rolling_min(closes: np.ndarray, window: int) -> np.ndarray:
    """Lowest close of the last ``window`` days, including today."""
    closes = np.asarray(closes, dtype=float)
    out = np.full(closes.size, np.nan, dtype=float)
    for t in range(window - 1, closes.size):
        out[t] = float(np.min(closes[t - window + 1:t + 1]))
    return out


@dataclass(frozen=True)
class Bands:
    """The entry and stop bands, as whole series aligned to ``closes``.

    Returned as series rather than as the latest pair on purpose. The entry
    rule needs yesterday's upper band and today's close; handing back only
    "the current bands" is what makes an off-by-one lookahead easy to write
    and impossible to see.
    """

    upper: np.ndarray
    lower: np.ndarray
    donchian_up: np.ndarray
    donchian_down: np.ndarray
    keltner_up: np.ndarray
    keltner_down: np.ndarray

    @property
    def usable_from(self) -> int:
        """First index where both bands are finite."""
        both = np.isfinite(self.upper) & np.isfinite(self.lower)
        found = np.flatnonzero(both)
        return int(found[0]) if found.size else len(self.upper)


def bands(closes: np.ndarray) -> Bands:
    """The paper's two bands.

    ``UpperBand = min(DonchianUp20, KeltnerUp20)`` -- the *lower* of the two
    entry triggers, so a breakout qualifies on whichever condition is reached
    first. ``LowerBand = max(DonchianDown40, KeltnerDown40)`` -- the *higher*
    of the two exits, so the stop is the tighter one. Both are "whichever binds
    sooner", which is why one is a min and the other a max.
    """
    closes = np.asarray(closes, dtype=float)
    up = rolling_max(closes, DONCHIAN_UP_DAYS)
    down = rolling_min(closes, DONCHIAN_DOWN_DAYS)

    mad_up = mean_absolute_change(closes, KELTNER_UP_DAYS)
    mad_down = mean_absolute_change(closes, KELTNER_DOWN_DAYS)
    kelt_up = ema(closes, KELTNER_UP_DAYS) + KELTNER_MULTIPLE * 2.0 * mad_up
    kelt_down = ema(closes, KELTNER_DOWN_DAYS) - KELTNER_MULTIPLE * 2.0 * mad_down

    # np.minimum / np.maximum, never np.fmin / np.fmax.
    #
    # The f-variants ignore a NaN operand and return the other one, which here
    # would mean quietly publishing a band computed from a single leg while the
    # other still had no window behind it -- a Donchian-only band for the
    # twenty bars before the Keltner leg exists, presented as though both had
    # agreed. These propagate the NaN instead: no band until both legs are
    # real. A test caught this; the difference is invisible on a chart.
    return Bands(
        upper=np.minimum(up, kelt_up),
        lower=np.maximum(down, kelt_down),
        donchian_up=up, donchian_down=down,
        keltner_up=kelt_up, keltner_down=kelt_down,
    )


def daily_sigma(closes: np.ndarray, window: int = SIGMA_DAYS) -> float:
    """Standard deviation of the last ``window`` daily returns.

    Simple returns rather than log: the sizing divides a target volatility by
    this, and the target is expressed as a fraction of equity per day, which is
    a simple return.
    """
    closes = np.asarray(closes, dtype=float)
    if closes.size < window + 1:
        return float("nan")
    tail = closes[-(window + 1):]
    if np.any(tail <= 0):
        return float("nan")
    returns = np.diff(tail) / tail[:-1]
    return float(np.std(returns, ddof=1))


def entry_signal(closes: np.ndarray, band: Bands, t: int) -> bool:
    """Whether a flat symbol breaks out on day ``t``.

    Compares ``closes[t]`` against ``band.upper[t - 1]``. The previous index is
    the whole point: the band includes day ``t``'s own close, so testing
    against ``band.upper[t]`` compares a number against a maximum that already
    contains it and fires on every new high with perfect foresight.
    """
    if t <= 0 or t >= len(closes):
        return False
    yesterday = band.upper[t - 1]
    if not math.isfinite(float(yesterday)):
        return False
    return float(closes[t]) >= float(yesterday)


def trail_stop(previous_stop: float, lower_band_today: float) -> float:
    """The stop for tomorrow: never lower than it was.

    A stop that can fall is not a stop. This is the single line that turns a
    breakout rule into a strategy with a bounded loss per trade, and it is
    tested directly rather than implicitly.
    """
    today = float(lower_band_today)
    if not math.isfinite(today):
        return float(previous_stop)
    if not math.isfinite(float(previous_stop)):
        return today
    return max(float(previous_stop), today)


def exit_signal(close_today: float, stop_carried_in: float) -> bool:
    """Whether a held position is stopped out on today's close.

    ``stop_carried_in`` is yesterday's stop, not one recomputed from today's
    band. Updating the stop before testing it would let a position that should
    have been closed survive on a band that only moved because of the very fall
    that should have ended it.
    """
    stop = float(stop_carried_in)
    if not math.isfinite(stop):
        return False
    return float(close_today) < stop


@dataclass
class SleeveWeights:
    """Target weights for one rebalance, and what bound them."""

    weights: dict[str, float] = field(default_factory=dict)
    gross: float = 0.0
    #: Gross before the leverage cap was applied, so the panel can say the cap
    #: bound rather than leaving the operator to infer it.
    gross_before_cap: float = 0.0
    capped: bool = False


def target_weights(sigmas: dict[str, float], longs: list[str], *,
                   universe_size: int, target_vol: float,
                   max_leverage: float) -> SleeveWeights:
    """Volatility-target weights for the symbols currently long.

    ``w_j = (target_vol / N) / sigma_j``, where N is the size of the *active
    universe* rather than the number of positions. That is the paper's
    formulation and it matters: dividing by the number of longs would make each
    position bigger precisely when few symbols qualify, which is when the
    market is least hospitable. Dividing by the universe keeps a thin signal a
    small book.
    """
    result = SleeveWeights()
    if universe_size <= 0 or target_vol <= 0:
        return result

    raw: dict[str, float] = {}
    for symbol in longs:
        sigma = float(sigmas.get(symbol, float("nan")))
        # A symbol whose volatility cannot be estimated gets no weight rather
        # than a default: an invented sigma is an invented position size.
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        raw[symbol] = (target_vol / universe_size) / sigma

    gross = float(sum(raw.values()))
    result.gross_before_cap = gross
    if gross > max_leverage > 0:
        scale = max_leverage / gross
        raw = {s: w * scale for s, w in raw.items()}
        result.capped = True
        gross = float(sum(raw.values()))
    result.weights = raw
    result.gross = gross
    return result


def needs_rebalance(current_qty: float, target_qty: float,
                    threshold: float) -> bool:
    """Whether an existing position has drifted far enough to be worth resizing.

    Entries and exits always execute; only a *held* symbol is subject to this.
    The threshold exists because the alternative is paying a spread every day
    to correct a weight by a percent, which turns a low-turnover strategy into
    a high-turnover one without changing a single signal.

    A position that is currently flat is an entry, not a rebalance, so it is
    never suppressed here.
    """
    current = abs(float(current_qty))
    if current <= 0:
        return True
    return abs(float(target_qty) - float(current_qty)) / current >= threshold
