"""Capital follows the market's regime, where a strategy's record says it should.

The evidence allocator asks whether a strategy earns. This asks a narrower
question: whether it earns *differently* depending on what the market is doing,
and if so, sizes it for the market it is in now. Trend-following that earns in
a rising, calm market and bleeds in a falling, volatile one should not be sized
the same in both, and after enough of each its own record says so.

THE STATE OF THE MARKET
-----------------------

Read from the benchmark's daily closes, in two parts, each a convention rather
than a fitted number:

* **Direction**: above or below its own 200-day average. The long-horizon
  trend filter every allocator uses, and slow enough that it changes a few
  times a year rather than a few times a week.
* **Volatility**: 20-day realised volatility above or below its own median
  over the last year. A median splits the year's days in half by
  construction, so "volatile" means volatile *for this market*, not above a
  number someone chose.

Four states. The state as of a day's close is what the allocator acts on for
the next day, and it is what that next day's return is filed under -- so a
strategy's record by state is exactly the record of how it did when the
allocator would have known the state. Nothing is filed under a state it could
only have known afterwards.

THE TEST
--------

For the state the market is in now, a strategy's daily returns split in two:
the days filed under this state, and every other day. Two questions, and a
tilt needs both:

* Is it **different** here? Welch's test on the difference in mean daily
  return, as a probability that the mean here is higher.
* Does it **earn** here at all? The probabilistic Sharpe ratio of the days
  here alone, against zero.

More capital takes 99% that it does better here *and* 95% that it earns here;
less takes 99% that it does worse here *and* a Sharpe here that is negative.
Worse-but-still-earning is not cut: a strategy that earns everywhere and a
little less in this state is still earning. The bars are strict for the same
reason the evidence bands are asymmetric: across four states every threshold
is tried four times, and a tilt on noise is money moved on nothing. Measured
on a strategy with no edge, checked in every state: about 1.8% of checks tilt
it, half up and half down. A real dependence of a third of a daily Sharpe
either way is found in 99% of years.

Nothing tilts on fewer than :data:`MIN_STATE_DAYS` days on each side.

HOW IT IS APPLIED
-----------------

A factor on the evidence allocator's target, never a replacement for it, and
never past its bounds. A strategy the decay rule has cut is not lifted by the
regime: a strategy that stopped working does not get more capital because the
market looks like one it used to work in. The applied multiplier still moves
at most one step a day, so a state that flips back and forth across its line
cannot churn the book.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable

from imperium.execution.evidence import moments, probabilistic_sharpe

#: The trend filter's average, in trading days.
TREND_DAYS = 200

#: Realised volatility's window, and the history its median is taken over.
VOL_DAYS = 20
VOL_HISTORY = 252
#: Days of volatility history before "above its median" means anything.
VOL_MIN_HISTORY = 60

#: Days filed under the current state, and outside it, before any tilt.
MIN_STATE_DAYS = 20

#: The bars for a tilt. See the module docstring.
DIFFERENT_BAR = 0.99
EARNS_BAR = 0.95
MORE = 1.25
LESS = 0.5

RISING_CALM = "rising_calm"
RISING_VOLATILE = "rising_volatile"
FALLING_CALM = "falling_calm"
FALLING_VOLATILE = "falling_volatile"

LABEL = {
    RISING_CALM: "rising and calm",
    RISING_VOLATILE: "rising and volatile",
    FALLING_CALM: "falling and calm",
    FALLING_VOLATILE: "falling and volatile",
}

_NORMAL = NormalDist()


@dataclass(frozen=True)
class MarketState:
    """What the market was doing as of one day's close."""

    day: str
    state: str
    #: Close over its 200-day average, less one.
    trend: float
    #: 20-day volatility over its median for the year.
    vol_ratio: float

    @property
    def label(self) -> str:
        return LABEL.get(self.state, self.state)

    @property
    def reason(self) -> str:
        where = "above" if self.trend >= 0 else "below"
        return (f"the benchmark is {abs(self.trend):.1%} {where} its "
                f"{TREND_DAYS}-day average, with volatility "
                f"{self.vol_ratio:.2f}x its median for the year")

    def as_dict(self) -> dict[str, Any]:
        return {"day": self.day, "state": self.state, "label": self.label,
                "trend": round(self.trend, 4),
                "vol_ratio": round(self.vol_ratio, 3), "reason": self.reason}


def _day_of(bar: Any) -> str:
    # The UTC date, deliberately not the Eastern one. The venue stamps a daily
    # bar at midnight Eastern -- 04:00 or 05:00 UTC -- and other sources at
    # midnight UTC; both fall on the bar's own date in UTC. Converted to
    # Eastern, a 04:00 UTC stamp in winter is 23:00 the evening before, and
    # every session would be filed a day early.
    stamp = dt.datetime.fromtimestamp(int(bar.open_time) / 1000,
                                      tz=dt.timezone.utc)
    return stamp.date().isoformat()


def states_by_day(bars: Iterable[Any]) -> dict[str, MarketState]:
    """The market's state as of each day's close, for every day it can be read.

    Each day's state uses closes up to and including that day and nothing
    after: this is replayed over history, and a state that peeked at the
    future would file returns under a state the allocator could not have
    known.
    """
    rows = [(_day_of(b), float(b.close)) for b in bars
            if getattr(b, "close", 0) and float(b.close) > 0]
    rows.sort()
    deduped: list[tuple[str, float]] = []
    for day, close in rows:
        if deduped and deduped[-1][0] == day:
            deduped[-1] = (day, close)
        else:
            deduped.append((day, close))
    closes = [c for _, c in deduped]
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:])]

    vols: list[float | None] = [None] * len(closes)
    for i in range(VOL_DAYS, len(closes)):
        window = rets[i - VOL_DAYS:i]
        vols[i] = statistics.stdev(window)

    out: dict[str, MarketState] = {}
    running = sum(closes[:TREND_DAYS])
    for i in range(TREND_DAYS - 1, len(closes)):
        if i >= TREND_DAYS:
            running += closes[i] - closes[i - TREND_DAYS]
        vol = vols[i]
        if vol is None:
            continue
        history = [v for v in vols[max(0, i - VOL_HISTORY + 1):i + 1]
                   if v is not None]
        if len(history) < VOL_MIN_HISTORY:
            continue
        median = statistics.median(history)
        if median <= 0:
            continue
        average = running / TREND_DAYS
        trend = closes[i] / average - 1.0
        ratio = vol / median
        rising = trend >= 0
        volatile = ratio > 1.0
        state = ((RISING_VOLATILE if volatile else RISING_CALM) if rising else
                 (FALLING_VOLATILE if volatile else FALLING_CALM))
        out[deduped[i][0]] = MarketState(deduped[i][0], state, trend, ratio)
    return out


def current(states: dict[str, MarketState], day: str | None = None
            ) -> MarketState | None:
    """The latest state as of ``day`` (or the latest of all)."""
    days = sorted(d for d in states if day is None or d <= day)
    return states[days[-1]] if days else None


def state_on(states: dict[str, MarketState], day: str) -> str | None:
    """The state as of ``day``'s close; the last close before it on a day the
    market was shut, so a weekend mark is filed under Friday's state."""
    found = current(states, day)
    return found.state if found else None


def filed_returns(marks: list[tuple[str, float]], equity: dict[str, float],
                  states: dict[str, MarketState]) -> list[tuple[str, float]]:
    """Each daily return, filed under the state as of the day it started.

    The same returns ``evidence.daily_returns`` computes -- a day with no
    known equity before it is skipped -- and a day with no known state is
    left out here too rather than filed under a guess.
    """
    out: list[tuple[str, float]] = []
    for (prev_day, prev), (_, value) in zip(marks, marks[1:]):
        base = equity.get(prev_day, 0.0)
        if base <= 0:
            continue
        state = state_on(states, prev_day)
        if state is None:
            continue
        out.append((state, (value - prev) / base))
    return out


@dataclass(frozen=True)
class Tilt:
    factor: float
    reason: str
    #: P(mean here > mean elsewhere), and P(Sharpe here > 0). None when not
    #: measured.
    different: float | None = None
    earns: float | None = None
    here: int = 0
    elsewhere: int = 0


def tilt(filed: list[tuple[str, float]], state: str | None) -> Tilt:
    """How much more or less to size a strategy in ``state``, from its record."""
    if state is None:
        return Tilt(1.0, "market state not yet known")
    label = LABEL.get(state, state)
    here = [r for s, r in filed if s == state]
    rest = [r for s, r in filed if s != state]
    if len(here) < MIN_STATE_DAYS or len(rest) < MIN_STATE_DAYS:
        return Tilt(1.0, f"{min(len(here), MIN_STATE_DAYS)} of {MIN_STATE_DAYS} "
                         f"days measured while the market was {label}",
                    here=len(here), elsewhere=len(rest))
    var_here = statistics.variance(here)
    var_rest = statistics.variance(rest)
    se = math.sqrt(var_here / len(here) + var_rest / len(rest))
    if se <= 0 or not math.isfinite(se):
        return Tilt(1.0, "no variation to measure", here=len(here),
                    elsewhere=len(rest))
    gap = statistics.fmean(here) - statistics.fmean(rest)
    different = _NORMAL.cdf(gap / se)
    m = moments(here)
    earns = probabilistic_sharpe(m, 0.0) if m is not None else 0.5
    counts = f"over {len(here)} days {label} and {len(rest)} otherwise"
    if different >= DIFFERENT_BAR and earns >= EARNS_BAR:
        return Tilt(MORE, f"earns more when the market is {label}: "
                          f"{different:.0%} likely better than otherwise, "
                          f"{counts}", different, earns, len(here), len(rest))
    if different <= 1.0 - DIFFERENT_BAR and earns < 0.5:
        return Tilt(LESS, f"loses when the market is {label}: "
                          f"{1 - different:.0%} likely worse than otherwise, "
                          f"{counts}", different, earns, len(here), len(rest))
    return Tilt(1.0, f"no difference measured when the market is {label}, "
                     f"{counts}", different, earns, len(here), len(rest))
