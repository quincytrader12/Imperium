"""Short-term mean reversion in the broad index ETFs.

The opposite bet to everything else here. Trend-following owns what has been
rising; this buys what has just fallen hard -- but only inside a long-run
uptrend -- and sells it on the bounce. It earns in choppy, sideways markets,
which are exactly the markets where the trend strategies bleed, so the two
together smooth each other rather than compound each other.

**The evidence.** Short-term reversal is one of the oldest documented effects
in equity returns (Jegadeesh 1990; Lehmann 1990): losers over days to a week
tend to recover over the next few days. Connors and Alvarez (2008) gave it the
tradeable form used here for index ETFs -- buy when a two-day RSI is deeply
oversold while the price is above its 200-day average, sell when it closes
back above a short average -- and it has been studied out of sample since.
The effect is strongest in the most liquid index products, which is why the
universe is four of them and not the whole market.

THE RULE
--------

* **Only in an uptrend.** The close must be above its 200-day average. A sharp
  fall inside a long decline is not a dip, it is the decline.
* **Entry.** The two-day RSI below :data:`ENTRY_RSI`: two days of losses
  sharp enough that almost nothing was gained against them.
* **Exit.** The close back above its 5-day average -- the bounce has come --
  or :data:`MAX_HOLD_DAYS` trading days held, whichever is first. The time
  stop is what keeps a reversal trade from quietly becoming a buy-and-hope.
* **Size.** At most :data:`MAX_POSITIONS` at once, each an equal share of the
  sleeve. Signals are rare -- a few a year per ETF -- so the sleeve is mostly
  in cash, and the share it holds when it is in is what makes it matter.

All multi-day, so none of it is a day trade under the pattern-day-trader rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

UNIVERSE: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA")

TREND_DAYS = 200
RSI_DAYS = 2
EXIT_AVERAGE_DAYS = 5
ENTRY_RSI = 10.0
MAX_HOLD_DAYS = 10
MAX_POSITIONS = 2
MIN_BARS = TREND_DAYS + 1


def rsi(closes: np.ndarray, period: int = RSI_DAYS) -> float:
    """Wilder's relative strength index of the last close, 0..100.

    Smoothed from the start of the series, as Wilder defined it, so the value
    depends on more than the last ``period`` days -- which is why it is
    computed over the whole history rather than a window.
    """
    closes = np.asarray(closes, dtype=float)
    if closes.size < period + 1:
        return float("nan")
    change = np.diff(closes)
    gains = np.clip(change, 0, None)
    losses = np.clip(-change, 0, None)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))


@dataclass
class Targets:
    weights: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    #: symbol -> trading days held, carried to tomorrow.
    held_days: dict[str, int] = field(default_factory=dict)
    note: str = ""


def targets(closes: dict[str, np.ndarray], held: set[str],
            held_days: dict[str, int]) -> Targets:
    """Today's targets. ``held`` is what the sleeve holds now; ``held_days``
    how many trading days it has held each, counting today (bought yesterday
    is one)."""
    out = Targets()
    keep: list[str] = []
    for symbol in sorted(held):
        series = closes.get(symbol)
        days = int(held_days.get(symbol, 0))
        if series is None or series.size < EXIT_AVERAGE_DAYS:
            keep.append(symbol)
            out.held_days[symbol] = days
            out.reasons[symbol] = "holding: no fresh history to judge the exit"
            continue
        average = float(np.mean(series[-EXIT_AVERAGE_DAYS:]))
        if series[-1] > average:
            out.reasons[symbol] = (f"bounced: closed {series[-1]:,.2f} above its "
                                   f"{EXIT_AVERAGE_DAYS}-day average {average:,.2f}")
            continue
        if days >= MAX_HOLD_DAYS:
            out.reasons[symbol] = (f"time stop: {days} trading days without the "
                                   f"bounce")
            continue
        keep.append(symbol)
        out.held_days[symbol] = days
        out.reasons[symbol] = (f"holding, day {days} of at most {MAX_HOLD_DAYS}: "
                               f"waiting for a close above {average:,.2f}")

    candidates: list[tuple[float, str]] = []
    for symbol, series in closes.items():
        if symbol in held or series.size < MIN_BARS:
            continue
        trend = float(np.mean(series[-TREND_DAYS:]))
        value = rsi(series)
        if not math.isfinite(value):
            continue
        if series[-1] <= trend:
            out.reasons.setdefault(symbol, f"below its {TREND_DAYS}-day average: "
                                           f"no dips bought in a decline")
            continue
        if value >= ENTRY_RSI:
            out.reasons.setdefault(symbol, f"not oversold: 2-day RSI {value:.0f}")
            continue
        candidates.append((value, symbol))

    # The most oversold first, into whatever room is left.
    room = MAX_POSITIONS - len(keep)
    for value, symbol in sorted(candidates)[:max(0, room)]:
        keep.append(symbol)
        out.held_days[symbol] = 0
        out.reasons[symbol] = (f"oversold in an uptrend: 2-day RSI {value:.0f} "
                               f"above its {TREND_DAYS}-day average")
    for value, symbol in sorted(candidates)[max(0, room):]:
        out.reasons[symbol] = (f"oversold (RSI {value:.0f}) but the sleeve already "
                               f"holds {MAX_POSITIONS}")

    for symbol in keep:
        out.weights[symbol] = 1.0 / MAX_POSITIONS
    out.note = (f"holding {len(keep)} of at most {MAX_POSITIONS}"
                if keep else "no oversold dip in an uptrend; in cash")
    return out
