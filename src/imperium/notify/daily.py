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
class StrategyLine:
    """One strategy's standing, as the brief reports it."""

    name: str
    total: float
    round_trips: int | None = None
    hit_rate: float | None = None
    #: What the evidence allocator scales it by. Shown only when it is not 1,
    #: so the brief says when capital has moved and is silent when it has not.
    multiplier: float = 1.0


#: What each strategy is called in the brief. The code's names are for the
#: code; a phone screen at the end of the day wants words.
STRATEGY_LABEL = {
    "intraday": "Intraday",
    "trend": "Multi-day trend",
    "overnight": "Overnight drift",
    "cross_section": "Crypto ranking",
    "sector": "Sector trend",
    "global_trend": "Global trend",
    "mean_reversion": "Mean reversion",
    "unattributed": "Unattributed",
}


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
    #: Best first. Empty until any strategy has traded, and then the section is
    #: left out rather than printed with nothing in it.
    strategies: tuple[StrategyLine, ...] = ()
    #: The book measured as a whole, in one sentence, or empty when it holds
    #: nothing measurable.
    book_risk: str = ""
    #: What the market is doing, in a phrase, or empty before it can be read.
    market: str = ""
    #: (label, verdict, reason) for every strategy the research desk could
    #: measure last night.
    research: tuple[tuple[str, str, str], ...] = ()

    @property
    def day_pnl(self) -> float:
        """The day's profit, or NaN when there is nothing to measure it from.

        Not ``equity - 0``. A session that has not yet marked an opening
        equity -- one started mid-session, or restarted before the first
        account read -- would otherwise report the entire account as the day's
        gain, with a green dot next to it. A brief whose headline number can
        be the whole balance is worse than a brief that says it does not know.
        """
        if self.day_start_equity <= 0:
            return float("nan")
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
    lines = [f"\U0001F4CA DAILY BRIEF — {brief.day}"]
    if math.isfinite(pnl):
        lines.append(
            f"{dot(pnl)} Day {money(pnl, brief.currency)}"
            + (f"  ({percent(change)})" if math.isfinite(change) else ""))
    else:
        lines.append(f"{FLAT} Day — no opening mark to measure against")

    held = sum(p.value for p in brief.positions)
    where = f"Equity ${brief.equity:,.2f} · cash ${brief.cash:,.2f}"
    if held > 0 and brief.equity > 0:
        where += f" · {held / brief.equity:.0%} invested"
    lines.append(where)
    if brief.book_risk:
        lines.append(f"Book: {brief.book_risk}")
    if brief.market:
        lines.append(f"Market: {brief.market}")
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

    if brief.strategies:
        # Which strategy earned it. The question the rest of the brief could
        # not answer: a green day made by one strategy while another bled is
        # a different day from one they all contributed to.
        lines.append("")
        lines.append("By strategy (to date)")
        for line in brief.strategies:
            label = STRATEGY_LABEL.get(line.name, line.name)
            tail = ""
            if line.round_trips:
                tail = f"  · {line.round_trips} closed"
                if line.hit_rate is not None:
                    tail += f", {line.hit_rate:.0%} won"
            if abs(line.multiplier - 1.0) > 1e-6:
                tail += f"  · sized x{line.multiplier:.2f}"
            lines.append(f"{dot(line.total)} {label:<16} "
                         f"{money(line.total, brief.currency)}{tail}")

    if brief.research:
        # Last night's research desk. A holding edge is one word; one that is
        # fading or has reversed gets its numbers, because that is the line
        # the operator will want to check.
        lines.append("")
        lines.append("Research (is each edge still there?)")
        for label, verdict, reason in brief.research:
            if verdict == "holding":
                lines.append(f"✅ {label}: holding")
            else:
                icon = "🛑" if verdict == "reversed" else "⚠️"
                lines.append(f"{icon} {label}: {reason}")

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


# -- the weekly summary ----------------------------------------------------------

#: Trades named at each end of the week.
WEEK_LISTED = 3


@dataclass(frozen=True)
class WeekTrade:
    """One closing fill, and what it realised."""

    symbol: str
    strategy: str
    realised: float
    ts: float


@dataclass
class Week:
    """Seven days, from the saved equity history and the trade journal."""

    label: str
    start_equity: float = 0.0
    end_equity: float = 0.0
    #: The largest fall from a high inside the week, as a fraction (<= 0).
    max_drawdown: float = 0.0
    fills: int = 0
    #: (strategy, change in its running profit over the week), best first.
    strategies: tuple[tuple[str, float], ...] = ()
    #: Every fill that realised something, for the best and the worst.
    closes: tuple[WeekTrade, ...] = ()
    currency: str = "USD"

    @property
    def change(self) -> float:
        if self.start_equity <= 0:
            return float("nan")
        return self.end_equity - self.start_equity


def build_weekly(week: Week) -> str:
    """The Sunday message: the week's result, who made it, and the trades at
    either end of it. Plain text, for the same reason the brief is."""
    lines = [f"📅 WEEKLY SUMMARY — {week.label}", ""]
    change = week.change
    if math.isfinite(change):
        pct = change / week.start_equity
        lines.append(f"{dot(change)} Week {money(change, week.currency)}  "
                     f"({percent(pct)})")
        lines.append(f"Equity ${week.end_equity:,.2f} · from "
                     f"${week.start_equity:,.2f}")
        if week.max_drawdown < 0:
            lines.append(f"Deepest fall from a high: {week.max_drawdown:.1%}")
    else:
        lines.append(f"{FLAT} Week — no equity history for the whole week yet")

    closes = list(week.closes)
    wins = sum(1 for t in closes if t.realised > 0)
    lines.append("")
    lines.append(f"{week.fills} fills · {len(closes)} closed"
                 + (f" · {wins / len(closes):.0%} won" if closes else ""))

    if week.strategies:
        lines.append("")
        lines.append("By strategy (this week)")
        for name, value in week.strategies:
            label = STRATEGY_LABEL.get(name, name)
            lines.append(f"{dot(value)} {label:<16} {money(value, week.currency)}")

    ranked = sorted(closes, key=lambda t: t.realised, reverse=True)
    best = [t for t in ranked if t.realised > 0][:WEEK_LISTED]
    worst = [t for t in reversed(ranked) if t.realised < 0][:WEEK_LISTED]

    def line(t: WeekTrade) -> str:
        day = dt.datetime.fromtimestamp(t.ts, tz=dt.timezone.utc).strftime("%a")
        label = STRATEGY_LABEL.get(t.strategy, t.strategy or "—")
        return (f"{dot(t.realised)} {t.symbol:<9} {money(t.realised, week.currency)}"
                f"  · {label}, {day}")

    if best:
        lines.append("")
        lines.append("Best trades")
        lines.extend(line(t) for t in best)
    if worst:
        lines.append("")
        lines.append("Worst trades")
        lines.extend(line(t) for t in worst)
    if not closes:
        lines.append("Nothing closed this week.")
    return "\n".join(lines)
