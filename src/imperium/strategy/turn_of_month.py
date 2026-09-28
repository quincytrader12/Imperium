"""The turn of the month: hold the S&P 500 for the few days it earns the most.

A calendar strategy, the only one here with no price signal at all. It holds
the index over the last trading day of a month and the first three of the
next, and is in cash the rest of the time. Its exposure is to a different
cause of returns from everything else in this terminal -- when money arrives,
not what the price has done -- so its good days are not the trend
strategies' good days.

**The evidence.** Ariel (1987) and Lakonishok and Smidt (1988) found that
nearly all of the US market's return over decades arrived in a window of
about four trading days around the turn of each month. McConnell and Xu
(2008) re-tested it out of sample on 1987-2005 and in 34 other countries: in
the US the window from the last trading day to the third of the next month
still carried essentially all of the market's excess return, and the effect
was present in 30 of the 35 countries. The usual explanation is the calendar
of cash: salaries, pension contributions and fund inflows arrive at the month
end and are invested in the days after.

THE RULE
--------

* **The window** is day -1 (the month's last trading day) through day +3 (the
  third trading day of the next month), as McConnell and Xu define it.
* **Entry** at the close of day -2, so the position is held through the
  whole of day -1. A terminal that was off on day -2 enters on any later day
  inside the window rather than skip the month: less of the window, but the
  rest of it.
* **Exit** at the close of day +3.
* **The instrument** is IVV, an S&P 500 fund. Not SPY: SPY belongs to the
  mean-reversion sleeve, and one owner per symbol is what lets each sleeve
  size against the broker's position as its own.

Held for about five trading days, entered and exited on different days, so it
is never a day trade under the pattern-day-trader rule.

**The calendar** is the NYSE's published holiday rules, computed here rather
than fetched, because the strategy has to know *in advance* which day will be
the month's second-to-last. An unscheduled closure (a national day of
mourning) is not in any rule; on such a month the window starts a day early
or late, which costs a fraction of one month's edge.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache

SYMBOL = "IVV"
UNIVERSE: tuple[str, ...] = (SYMBOL,)

#: Offsets held at the close: entered by day -2, out at the close of +3.
HOLD = (-2, -1, 1, 2)
EXIT = 3


def _easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (the anonymous Meeus/Jones/Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    nxt = dt.date(year + month // 12, month % 12 + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: dt.date) -> dt.date:
    """Saturday holidays are taken on the Friday, Sunday ones on the Monday."""
    if day.weekday() == 5:
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:
        return day + dt.timedelta(days=1)
    return day


@lru_cache(maxsize=64)
def nyse_holidays(year: int) -> frozenset[dt.date]:
    """The NYSE's full-day closures in ``year`` under its standing rules."""
    days = set()
    new_year = dt.date(year, 1, 1)
    # A Saturday New Year is not moved back into December: the exchange's
    # rule is not to close on the last trading day of a year for it.
    if new_year.weekday() != 5:
        days.add(_observed(new_year))
    days.add(_nth_weekday(year, 1, 0, 3))            # Martin Luther King Jr.
    days.add(_nth_weekday(year, 2, 0, 3))            # Washington's Birthday
    days.add(_easter(year) - dt.timedelta(days=2))   # Good Friday
    days.add(_last_weekday(year, 5, 0))              # Memorial Day
    if year >= 2022:
        days.add(_observed(dt.date(year, 6, 19)))    # Juneteenth
    days.add(_observed(dt.date(year, 7, 4)))         # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))            # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))           # Thanksgiving
    days.add(_observed(dt.date(year, 12, 25)))       # Christmas
    return frozenset(days)


def is_trading_day(day: dt.date) -> bool:
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


@lru_cache(maxsize=256)
def trading_days(year: int, month: int) -> tuple[dt.date, ...]:
    day = dt.date(year, month, 1)
    out = []
    while day.month == month:
        if is_trading_day(day):
            out.append(day)
        day += dt.timedelta(days=1)
    return tuple(out)


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def offset(day: dt.date) -> int | None:
    """Where ``day`` sits relative to the turn of the month: -1 is the last
    trading day, +1 the first, and so on; None off the edges of the window
    that matter (-2..+3) or on a day the exchange is shut."""
    if not is_trading_day(day):
        return None
    days = trading_days(day.year, day.month)
    index = days.index(day)
    if index < 3:
        return index + 1
    from_end = index - len(days)                     # -1 on the last day
    return from_end if from_end >= -2 else None


def next_entry(day: dt.date) -> dt.date:
    """The next day -2 on or after ``day``."""
    year, month = day.year, day.month
    while True:
        entry = trading_days(year, month)[-2]
        if entry >= day:
            return entry
        year, month = _next_month(year, month)


def exit_day(day: dt.date) -> dt.date:
    """The day +3 of the window ``day`` is in (or approaching)."""
    days = trading_days(day.year, day.month)
    if day <= days[2]:
        return days[2]
    return trading_days(*_next_month(day.year, day.month))[2]


@dataclass
class Targets:
    weights: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    note: str = ""


def _fmt(day: dt.date) -> str:
    return f"{day:%a} {day.day} {day:%b}"


def targets(day: dt.date) -> Targets:
    """Today's target, from the date alone."""
    out = Targets()
    where = offset(day)
    if where in HOLD:
        leave = exit_day(day)
        out.weights[SYMBOL] = 1.0
        label = f"day {where:+d}"
        out.reasons[SYMBOL] = (f"turn of the month, {label} of -1..+3: held "
                               f"to the close of {_fmt(leave)}")
        out.note = f"in the window; out at the close of {_fmt(leave)}"
        return out
    entry = next_entry(day + dt.timedelta(days=1) if where == EXIT else day)
    if where == EXIT:
        out.reasons[SYMBOL] = "turn of the month over: day +3, out at the close"
    else:
        out.reasons[SYMBOL] = f"outside the window; next entry {_fmt(entry)}"
    out.note = f"in cash until the close of {_fmt(entry)}"
    return out
