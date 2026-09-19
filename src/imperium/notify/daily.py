"""One message a day: what the book made, what it holds, what it did.

The brief exists because the notification stream this program produces is
event-shaped -- a fill, a halt, an arm -- and none of those answer the only
question an operator actually has at the end of a day, which is whether the
week is going well. Six fills and two halts tell you nothing about the
balance.

**It is sent once.** That is not a nicety, it is the whole design constraint.
Every repeat-notification bug this program has had came from a level-triggered
check firing every cycle: the same "BOOK HALTED, daily loss 33.93%" arriving
five times with the same figure. A daily message that does that is worse than
no daily message, because it trains the operator to ignore the one notice
they were meant to read.

So the guard is a **date, written to disk**. In memory would be enough for a
process that runs all day; this one is started by a batch file that relaunches
it whenever it exits, so an in-memory flag would send a fresh brief after
every crash, restart and overnight reboot. The date on disk is what makes
"once a day" mean once.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

#: Hour of the US Eastern day, at or after which the brief is due.
#:
#: Five in the afternoon: the equity close is 16:00 ET and late prints settle
#: shortly after, so a brief written before then reports a day that has not
#: finished. Crypto never closes and never will have a natural cut, so it is
#: measured against the same boundary as everything else rather than given one
#: of its own -- one brief a day, not one per asset class.
DUE_HOUR_ET = 17

#: Positions listed in full before the tail is summarised. Long enough for
#: every book this program can hold at the balances it is built for.
MAX_LISTED = 8

GREEN = "\U0001F7E2"
RED = "\U0001F534"
FLAT = "⚪"


def dot(value: float) -> str:
    """Green for a gain, red for a loss, white for neither.

    Zero gets its own mark rather than being rounded into one of the others: a
    position that has not moved is a different fact from one that is up a
    hundredth of a cent, and colouring it green would be the message making a
    claim the number does not support.
    """
    if not math.isfinite(value) or value == 0:
        return FLAT
    return GREEN if value > 0 else RED


def money(amount: float, currency: str = "USD") -> str:
    sign = "-" if amount < 0 else "+"
    symbol = "$" if currency == "USD" else f"{currency} "
    return f"{sign}{symbol}{abs(amount):,.2f}"


def percent(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.2%}"


@dataclass(frozen=True)
class Position:
    symbol: str
    value: float
    unrealised: float
    #: What the position cost, used for the percentage. Zero where it is not
    #: known, in which case no percentage is shown rather than a wrong one.
    basis: float = 0.0

    @property
    def change(self) -> float:
        if self.basis <= 0:
            return float("nan")
        return self.unrealised / self.basis


@dataclass(frozen=True)
class Activity:
    """What the book did, counted rather than narrated.

    Only what the program actually records. There is no submitted-order
    counter anywhere in it, so there is no "4 of 6 filled" line here -- a
    denominator invented for the sake of a nicer sentence is the kind of
    number an operator would go on to make a decision with.
    """

    buys: int = 0
    sells: int = 0
    protected: int = 0
    halted: bool = False
    halt_reason: str = ""

    @property
    def fills(self) -> int:
        return self.buys + self.sells


@dataclass(frozen=True)
class Brief:
    day: str
    equity: float
    day_start_equity: float
    cash: float
    positions: list[Position] = field(default_factory=list)
    activity: Activity = field(default_factory=Activity)
    currency: str = "USD"
    mode: str = ""

    @property
    def day_pnl(self) -> float:
        return self.equity - self.day_start_equity

    @property
    def day_change(self) -> float:
        if self.day_start_equity <= 0:
            return float("nan")
        return self.day_pnl / self.day_start_equity


def is_due(*, last_sent_day: str, now: dt.datetime, today: str) -> bool:
    """Whether the brief should go out.

    Two conditions, and the second is the one that matters. The hour keeps the
    brief from reporting an unfinished day; ``last_sent_day`` keeps it from
    reporting a finished one twice, across restarts as well as across ticks.
    """
    if last_sent_day == today:
        return False
    return now.hour >= DUE_HOUR_ET


def build(brief: Brief) -> str:
    """The message, as Telegram will show it.

    Plain text on purpose. Telegram's Markdown parser rejects a message with
    an unbalanced underscore or asterisk in it, and ticker symbols contain
    both -- a formatted brief is one delisted ticker away from silently
    failing to send.
    """
    pnl, change = brief.day_pnl, brief.day_change
    lines = [
        f"\U0001F4CA DAILY BRIEF — {brief.day}",
        f"{dot(pnl)} Day {money(pnl, brief.currency)}"
        + (f"  ({percent(change)})" if math.isfinite(change) else ""),
    ]

    held = sum(p.value for p in brief.positions)
    where = f"Equity ${brief.equity:,.2f} · cash ${brief.cash:,.2f}"
    if held > 0 and brief.equity > 0:
        where += f" · {held / brief.equity:.0%} invested"
    lines.append(where)
    if brief.mode:
        lines.append(f"Mode: {brief.mode}")

    lines.append("")
    if not brief.positions:
        lines.append("No open positions.")
    else:
        lines.append(f"Positions ({len(brief.positions)})")
        # Worst first. A brief read on a phone is read from the top, and the
        # position that needs a decision is the one losing money.
        ordered = sorted(brief.positions, key=lambda p: p.unrealised)
        for p in ordered[:MAX_LISTED]:
            tail = (f"  ({percent(p.change)})"
                    if math.isfinite(p.change) else "")
            lines.append(
                f"{dot(p.unrealised)} {p.symbol:<10} ${p.value:>9,.2f}   "
                f"{money(p.unrealised, brief.currency)}{tail}")
        if len(ordered) > MAX_LISTED:
            rest = ordered[MAX_LISTED:]
            lines.append(
                f"{dot(sum(p.unrealised for p in rest))} and "
                f"{len(rest)} more, {money(sum(p.unrealised for p in rest))} "
                f"between them")

    lines.append("")
    lines.append("Activity")
    a = brief.activity
    if a.fills:
        parts = []
        if a.buys:
            parts.append(f"{a.buys} {'buy' if a.buys == 1 else 'buys'}")
        if a.sells:
            parts.append(f"{a.sells} {'sell' if a.sells == 1 else 'sells'}")
        lines.append(f"{a.fills} filled — " + ", ".join(parts))
    if a.protected:
        lines.append(
            f"{a.protected} closed to protect a gain")
    if a.halted:
        lines.append(f"{RED} Book halted: {a.halt_reason}")
    if lines[-1] == "Activity":
        lines.append("Nothing traded today.")
    return "\n".join(lines)
