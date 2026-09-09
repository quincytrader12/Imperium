"""The overnight drift, and the session decomposition it rests on.

US equity returns split into two economically distinct pieces, and almost the
entire historical equity premium lives in one of them:

* **Overnight** (previous close to today's open) has been strongly positive.
* **Intraday** (open to close) has been close to zero and often negative.

Cooper, Cliff & Gulen measured this on S&P 500 constituents from 1993 to 2006
and found average night returns of **2.82 to 4.76 bps** against day returns of
**-2.85 to +0.22 bps**. Lou, Polk & Skouras (JFE 2019) showed the split is not
noise: firm-level overnight and intraday returns are each persistent for years
and offset one another -- a "tug of war" between clienteles that trade at
different times of day. For *large* stocks, momentum profits accrue mostly
overnight; for small stocks, mostly intraday, which is why the universe this
runs on matters.

**The reason this is gated rather than traded every night.** The edge is 3-5 bps
and a round trip costs something of the same order. Alpha Architect's replication
found that adding one cent per share of cost cuts the Sharpe to 0.31; two ETFs
launched in 2022 to harvest it (NSPY, NIWM) closed within a year. So the
arithmetic that decides whether to trade is the cost gate, not the signal --
and the signal's job is only to say how large the expected drift is so the gate
can judge it.

That produces one concrete selection rule that falls straight out of the
research: the drift is roughly a **fixed number of basis points**, while the
cost in basis points is the spread divided by the price. A one-cent spread is
0.25bp on a $400 name and 10bp on a $10 name. High-priced, tight-spread symbols
are where this survives, and the cost gate discovers that on its own.

**Why this does not extend to options.** The drift belongs to the underlying;
an option rents delta exposure to it and pays theta on *calendar* days. At the
same delta-equivalent exposure, one night of theta on an at-the-money call needs
a drift of about 53bp at 1 day to expiry, 20bp at a week and 10bp at a month --
against an actual drift near 4bp. Only a deep in-the-money contract with delta
near one breaks even, and that is a stock substitute carrying a much wider
spread. The measurement is in ``scripts/overnight_option_arithmetic.py``; the
conclusion is that this strategy is equity-only, and options remain untraded.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from imperium.execution.bars import Bar, BarSeries

#: A gap larger than this many bar intervals is treated as a session boundary.
GAP_FACTOR = 1.5

#: Minimum overnight observations before the estimate means anything. Below
#: this the standard error swamps a 3-5bp effect entirely.
MIN_NIGHTS = 20

#: Alpaca refuses a market-on-close order lodged inside the last ten minutes of
#: the session, and a market-on-open order lodged inside the last two minutes
#: before the open. These are venue rules, not preferences: an order sent at
#: 15:51 ET is not an order that fills on the close, it is an order that is
#: rejected. Each carries a minute of margin, because the clock we compare
#: against was fetched over a network.
MOC_CUTOFF_MINUTES = 10 + 1
MOO_CUTOFF_MINUTES = 2 + 1


class SessionPhase(str, Enum):
    """Where the clock is, relative to the trade this strategy wants to make."""

    #: Near the close: the entry window. Buying earlier than this holds
    #: intraday risk the strategy is not trying to take.
    CLOSING = "closing"
    #: At or just after the open: the exit window.
    OPENING = "opening"
    #: Before the open, inside the window a market-on-open order is still
    #: accepted. This is where the overnight position is *exited*, not at the
    #: open itself: the strategy is paid the opening print, and an order sent
    #: after the bell has already missed it.
    PREOPEN = "preopen"
    #: Mid-session. The overnight trade is neither entered nor exited here.
    INTRADAY = "intraday"
    #: Outside regular hours.
    CLOSED = "closed"


@dataclass(frozen=True)
class SessionSplit:
    """A symbol's return decomposed into its overnight and intraday parts."""

    overnight: np.ndarray
    intraday: np.ndarray
    nights: int

    @property
    def enough(self) -> bool:
        return self.nights >= MIN_NIGHTS


def split_daily(bars: list[Bar]) -> SessionSplit:
    """Separate close-to-open from open-to-close using **daily** bars.

    Daily bars, not minute bars, and the reason is not convenience. A 3-5bp
    effect needs dozens of observations before its standard error is smaller
    than the effect itself, and a bounded ring of one-minute bars holds only a
    few sessions -- the first attempt at this measured 14 nights from what
    should have been 80, because the ring had silently truncated the history.
    One daily bar carries exactly the open and close this needs, so a hundred
    nights costs a hundred rows.
    """
    usable = [b for b in bars if b.open > 0 and b.close > 0]
    if len(usable) < 2:
        return SessionSplit(np.zeros(0), np.zeros(0), 0)

    intraday = [math.log(b.close / b.open) for b in usable]
    overnight = [math.log(current.open / previous.close)
                 for previous, current in zip(usable, usable[1:])]
    return SessionSplit(np.asarray(overnight), np.asarray(intraday),
                        len(overnight))


def split_sessions(series: BarSeries) -> SessionSplit:
    """Derive the same split from an intraday series, by finding its seams.

    Kept for a series that has no daily history yet. Sessions are found from the
    bar timestamps rather than a hardcoded clock, so early closes and half days
    fall out correctly -- a mislabelled boundary would put an intraday return
    into the overnight bucket, which is the one thing this module must not do.
    """
    bars = [b for b in series.bars if b.closed]
    if len(bars) < 3:
        return SessionSplit(np.zeros(0), np.zeros(0), 0)

    threshold = series.bar_seconds * 1000 * GAP_FACTOR
    sessions: list[list[Bar]] = [[bars[0]]]
    for previous, current in zip(bars, bars[1:]):
        if current.open_time - previous.open_time > threshold:
            sessions.append([current])
        else:
            sessions[-1].append(current)

    daily = [Bar(open_time=s[0].open_time, open=s[0].open,
                 high=max(b.high for b in s), low=min(b.low for b in s),
                 close=s[-1].close, volume=sum(b.volume for b in s), closed=True)
             for s in sessions]
    return split_daily(daily)


@dataclass(frozen=True)
class PooledDrift:
    """The overnight drift estimated across the whole universe at once.

    This exists because of a measurement, not a preference. A single symbol's
    overnight returns have a standard deviation around 40bp a night; over 90
    nights the standard error of its mean is therefore about 4bp -- the same
    size as the entire published effect. Estimated one symbol at a time, a real
    3-5bp drift is indistinguishable from zero, and the only symbols that would
    ever clear a t-test are the ones whose noise happened to look like drift.
    That is a machine for selecting outliers.

    Pooling across N symbols multiplies the observations by N and divides the
    standard error by sqrt(N). Forty symbols over ninety nights is 3,600
    symbol-nights, which resolves a 4bp effect comfortably -- and it matches
    what the literature actually claims, which is a market-wide premium rather
    than a property of any one ticker.
    """

    mean_bps: float
    t_stat: float
    observations: int
    symbols: int
    vol_bps: float
    intraday_bps: float = 0.0

    @property
    def credible(self) -> bool:
        """Enough evidence to use as a prior at all."""
        return self.observations >= 200 and self.symbols >= 5

    def describe(self) -> str:
        return (f"market overnight drift {self.mean_bps:+.2f}bp/night "
                f"(t={self.t_stat:+.2f}) from {self.observations:,} symbol-nights "
                f"across {self.symbols} symbols; intraday "
                f"{self.intraday_bps:+.2f}bp")

    def explain(self) -> str:
        """The same measurement, in words, and what follows from it.

        The compact form above is for people who already know what a
        t-statistic on a pooled overnight decomposition is. This one is for
        reading at a glance on a screen at seven in the morning, and it says
        what the program will *do*, which is the part that actually matters.
        """
        if not self.observations:
            return ("No overnight history measured yet. Nothing will be held "
                    "overnight until there is — this resolves itself as daily "
                    "bars accumulate, it is not a refusal to trade.")

        if not self.credible:
            missing = []
            if self.observations < 200:
                missing.append(f"{self.observations:,} of 200 symbol-nights")
            if self.symbols < 5:
                missing.append(f"{self.symbols} of 5 symbols")
            if abs(self.t_stat) < 2.0:
                missing.append(f"a t-statistic of {self.t_stat:+.2f}, "
                               f"where 2.0 is the bar for calling it real")
            return ("Not enough overnight history yet to tell a real drift "
                    "from noise: " + "; ".join(missing) + ". Nothing will be "
                    "carried overnight until it is measurable.")

        direction = "up" if self.mean_bps > 0 else "down"
        return (
            f"Overnight, these stocks have drifted {direction} "
            f"{abs(self.mean_bps):.2f} basis points a night — about "
            f"{abs(self.mean_bps) / 100:.3f}% — against {self.intraday_bps:+.2f}bp "
            f"during the session. Measured across {self.observations:,} "
            f"symbol-nights on {self.symbols} symbols, which is enough to be "
            f"confident it is a real pattern rather than noise "
            f"(t={self.t_stat:+.2f}). A position is only carried overnight "
            f"where that drift beats what the trade costs, and on most nights "
            f"it does not — that refusal is the strategy working, not failing.")


def pool(splits: dict[str, SessionSplit]) -> PooledDrift:
    """Pool every symbol's overnight returns into one market estimate."""
    overnight: list[np.ndarray] = []
    intraday: list[np.ndarray] = []
    symbols = 0
    for split in splits.values():
        if split.overnight.size >= 5:
            overnight.append(split.overnight)
            intraday.append(split.intraday)
            symbols += 1
    if not overnight:
        return PooledDrift(0.0, 0.0, 0, 0, 0.0)
    stacked = np.concatenate(overnight)
    stacked_id = np.concatenate(intraday) if intraday else np.zeros(0)
    return PooledDrift(
        mean_bps=float(np.mean(stacked)) * 10_000,
        t_stat=_t_stat(stacked),
        observations=int(stacked.size),
        symbols=symbols,
        vol_bps=float(np.std(stacked, ddof=1)) * 10_000 if stacked.size > 1 else 0.0,
        intraday_bps=(float(np.mean(stacked_id)) * 10_000
                      if stacked_id.size else 0.0),
    )


def _shrink(symbol_mean: float, symbol_se: float,
            prior_mean: float, prior_se: float) -> float:
    """Combine a noisy symbol estimate with the market prior by precision.

    Straightforward inverse-variance weighting. A symbol with 90 nights of its
    own data barely moves off the market estimate, which is the correct
    behaviour: at this effect size its own data carries very little information.
    """
    if symbol_se <= 0 or prior_se <= 0:
        return prior_mean if symbol_se <= 0 else symbol_mean
    ws, wp = 1.0 / (symbol_se ** 2), 1.0 / (prior_se ** 2)
    return (ws * symbol_mean + wp * prior_mean) / (ws + wp)


@dataclass(frozen=True)
class OvernightSignal:
    """What the overnight decomposition concluded for one symbol."""

    value: float                 # [-1, +1], the direction and conviction
    expected_edge_bps: float     # the drift the cost gate must clear
    mean_overnight_bps: float
    mean_intraday_bps: float
    t_stat: float
    nights: int
    overnight_vol_bps: float
    reason: str
    eligible: bool = False
    #: The market-wide estimate this symbol was shrunk toward, if there was one.
    pooled_bps: float = 0.0
    shrunk_bps: float = 0.0
    #: Set when a recent gap is large enough that an event -- earnings, a halt,
    #: a guide -- is the likelier explanation than drift.
    event_risk: bool = False

    @property
    def flat(self) -> bool:
        return self.value == 0.0


def _t_stat(sample: np.ndarray) -> float:
    if sample.size < 2:
        return 0.0
    sd = float(np.std(sample, ddof=1))
    if sd <= 0:
        return 0.0
    return float(np.mean(sample)) / (sd / math.sqrt(sample.size))


def evaluate(
    daily_bars: list[Bar],
    *,
    pooled: PooledDrift | None = None,
    min_t_stat: float = 2.0,
    event_gap_sigma: float = 3.0,
    max_edge_bps: float = 60.0,
) -> OvernightSignal:
    """Decide whether to hold this symbol overnight.

    The decision rests on the **pooled** market estimate, adjusted by whatever
    little the symbol's own history adds. Trading a symbol because its own
    ninety nights happened to average well is selecting on noise -- see
    :class:`PooledDrift` for the measurement that forced this design.
    """
    split = split_daily(daily_bars)
    if not split.enough:
        return OvernightSignal(
            0.0, 0.0, 0.0, 0.0, 0.0, split.nights, 0.0,
            f"warming up: {split.nights} of {MIN_NIGHTS} overnight observations",
        )

    overnight, intraday = split.overnight, split.intraday
    mean_on = float(np.mean(overnight)) * 10_000
    mean_id = float(np.mean(intraday)) * 10_000 if intraday.size else 0.0
    vol_on = float(np.std(overnight, ddof=1)) * 10_000
    t = _t_stat(overnight)

    # An event -- earnings, a halt, a guidance change -- produces a gap that is
    # not drift, and one of them can dominate the mean on its own. Alpaca's
    # basic plan publishes no earnings calendar, so this is a statistical stand
    # in and is reported as such rather than as a real earnings check.
    last_gap = abs(float(overnight[-1])) * 10_000
    event_risk = bool(vol_on > 0 and last_gap > event_gap_sigma * vol_on)
    if event_risk:
        return OvernightSignal(
            0.0, 0.0, mean_on, mean_id, t, split.nights, vol_on,
            (f"last night's gap was {last_gap:.0f}bp, over {event_gap_sigma:.0f} "
             f"sigma of this symbol's {vol_on:.0f}bp overnight volatility — an "
             f"event is the likelier explanation than drift"),
            event_risk=True,
        )

    if pooled is None or not pooled.credible:
        return OvernightSignal(
            0.0, 0.0, mean_on, mean_id, t, split.nights, vol_on,
            ("no market-wide overnight estimate yet — a single symbol's own "
             "history cannot resolve an effect this small, so nothing is traded "
             "on it alone"),
            pooled_bps=pooled.mean_bps if pooled else 0.0,
        )

    if pooled.mean_bps <= 0 or abs(pooled.t_stat) < min_t_stat:
        return OvernightSignal(
            0.0, 0.0, mean_on, mean_id, t, split.nights, vol_on,
            (f"the market overnight drift is {pooled.mean_bps:+.2f}bp "
             f"(t={pooled.t_stat:+.2f}) over {pooled.observations:,} "
             f"symbol-nights — no premium to harvest right now"),
            pooled_bps=pooled.mean_bps,
        )

    symbol_se = vol_on / math.sqrt(max(1, split.nights))
    prior_se = (pooled.vol_bps / math.sqrt(max(1, pooled.observations))
                if pooled.vol_bps > 0 else 0.0)
    shrunk = _shrink(mean_on, symbol_se, pooled.mean_bps, prior_se)

    if shrunk <= 0:
        return OvernightSignal(
            0.0, 0.0, mean_on, mean_id, t, split.nights, vol_on,
            (f"this symbol's own {mean_on:+.1f}bp pulls the market's "
             f"{pooled.mean_bps:+.2f}bp below zero once combined — nothing to "
             f"harvest here"),
            pooled_bps=pooled.mean_bps, shrunk_bps=shrunk,
        )

    edge = min(shrunk, max_edge_bps)
    # Conviction comes from the pooled t-statistic, which is the estimate that
    # actually carries information, saturating at t=6.
    value = float(np.clip((abs(pooled.t_stat) - min_t_stat) / (6.0 - min_t_stat),
                          0.0, 1.0))
    return OvernightSignal(
        value, edge, mean_on, mean_id, t, split.nights, vol_on,
        (f"market overnight drift {pooled.mean_bps:+.2f}bp (t={pooled.t_stat:+.2f}, "
         f"{pooled.observations:,} symbol-nights); this symbol's own "
         f"{mean_on:+.1f}bp shrinks the estimate to {shrunk:+.2f}bp"),
        eligible=True, pooled_bps=pooled.mean_bps, shrunk_bps=shrunk,
    )


def phase_from_clock(now: dt.datetime, next_close: dt.datetime | None,
                     is_open: bool, *, next_open: dt.datetime | None = None,
                     entry_minutes: int = 25,
                     exit_lead_minutes: int = 90) -> SessionPhase:
    """Where we are relative to the entry and exit windows.

    Both windows end *before* the event they aim at, because both orders are
    lodged with the venue ahead of time. That is also why a wide window costs
    nothing: a market-on-close order fills at the closing print whether it was
    sent twenty minutes before the bell or twelve, so the only thing that
    matters is being inside the window at all, and a wider one survives a
    missed tick.
    """
    if is_open:
        if next_close is None:
            return SessionPhase.INTRADAY
        minutes_left = (next_close - now).total_seconds() / 60.0
        if MOC_CUTOFF_MINUTES <= minutes_left <= entry_minutes:
            return SessionPhase.CLOSING
        return SessionPhase.INTRADAY
    if next_open is None:
        return SessionPhase.CLOSED
    minutes_to_open = (next_open - now).total_seconds() / 60.0
    if MOO_CUTOFF_MINUTES <= minutes_to_open <= exit_lead_minutes:
        return SessionPhase.PREOPEN
    return SessionPhase.CLOSED


def minutes_since_session_open(series: BarSeries) -> float:
    """How long the current session has been running, from the bars themselves.

    Read from the data rather than a clock so that a half day, a late open or a
    delayed feed cannot put the strategy in the wrong window.
    """
    bars = [b for b in series.bars if b.closed]
    if len(bars) < 2:
        return float("inf")
    threshold = series.bar_seconds * 1000 * GAP_FACTOR
    last_boundary = bars[0].open_time
    for previous, current in zip(bars, bars[1:]):
        if current.open_time - previous.open_time > threshold:
            last_boundary = current.open_time
    return (bars[-1].open_time - last_boundary) / 60_000.0
