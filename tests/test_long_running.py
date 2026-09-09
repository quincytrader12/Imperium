"""What breaks when the terminal is left running for a week.

None of this shows up in a session that lasts an afternoon, which is why it is
tested rather than observed: every failure here is silent, and each one stops
the bot trading without stopping the bot.
"""

from __future__ import annotations

import datetime as dt

import pytest

from imperium.execution.broker import FILL_HISTORY, PaperBroker
from imperium.execution.portfolio import PortfolioAllocator
from imperium.execution.risk import RiskLimits
from imperium.session import TradingSession
from imperium.venues import registry
from imperium.venues.alpaca.client import MarketClock

UTC = dt.timezone.utc


def _at(day: int, hour: int = 15) -> MarketClock:
    return MarketClock(is_open=True,
                       timestamp=dt.datetime(2026, 3, day, hour, tzinfo=UTC),
                       next_close=dt.datetime(2026, 3, day, 21, tzinfo=UTC))


@pytest.mark.asyncio
async def test_the_daily_loss_halt_does_not_outlive_the_day_that_caused_it():
    """The single thing most likely to stop a week-long run, silently.

    The daily-loss reference was taken once at startup and never moved, so by
    Thursday the "daily" loss was measured against Monday's equity. And the
    halt it applied was permanent: nothing cleared it. A terminal left running
    would stop trading after its first bad afternoon and never start again --
    while still showing a running session, a live feed and a green health
    score.
    """
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.market_clock = _at(4)
    session.broker.cash = __import__("decimal").Decimal("10000")

    await session._tick()
    assert session.day_start_equity == pytest.approx(10_000.0)

    # A bad afternoon: down through the 4% limit.
    session.broker.cash = __import__("decimal").Decimal("9500")
    await session._tick()
    assert session.allocator.halted
    assert session.allocator.halt_source == "daily_loss"

    # The next day. The halt belonged to yesterday.
    session.market_clock = _at(5)
    await session._tick()

    assert not session.allocator.halted
    assert session.day_start_equity == pytest.approx(9_500.0)


@pytest.mark.asyncio
async def test_a_new_day_never_releases_a_halt_the_operator_applied():
    """A halt a human applied is not scoped to a day. Releasing it because
    midnight passed would be the program overruling the person who stopped
    it."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.market_clock = _at(4)
    await session._tick()

    session.allocator.set_halt(True, "stopped by hand", source="operator")
    session.market_clock = _at(5)
    await session._tick()

    assert session.allocator.halted
    assert session.allocator.halt_reason == "stopped by hand"


@pytest.mark.asyncio
async def test_startup_is_not_treated_as_a_rollover():
    """The first tick establishes the day; it does not end one. Treating it as
    a rollover would clear a halt restored from a previous run."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.market_clock = _at(4)
    session.allocator.set_halt(True, "carried in", source="daily_loss")

    await session._tick()

    assert session.allocator.halted, "the first tick is not a day boundary"


@pytest.mark.asyncio
async def test_the_day_boundary_comes_from_the_venue_not_the_local_clock():
    """A laptop in another timezone must not roll the book mid-session. The
    trading day this rule is about is the venue's."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.market_clock = _at(4, hour=14)
    await session._tick()
    first = session._trading_day

    # Hours pass locally; the venue is still on the same date.
    session.market_clock = _at(4, hour=20)
    await session._tick()
    assert session._trading_day == first


@pytest.mark.asyncio
async def test_the_fill_journal_is_bounded_but_the_count_is_not():
    """A week of trading must not be a leak, and bounding the ring must not
    make the session's own history appear to reset. The execution panel
    re-averages this list on every frame, so an unbounded one is a cost that
    grows for as long as the process runs."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = __import__("decimal").Decimal("10000000")
    session.feed.quote("AAPL").last = 100.0

    for i in range(FILL_HISTORY + 200):
        await session.broker.apply_target("AAPL", 0.001 * (i % 5 + 1),
                                          100.0, 1_000_000.0)

    assert len(session.broker.fills) == FILL_HISTORY
    snap = session.snapshot()
    assert snap["execution"]["count"] > FILL_HISTORY
    assert snap["execution"]["window"] == FILL_HISTORY
    assert len(snap["fills"]) == 40


@pytest.mark.asyncio
async def test_the_equity_curve_stays_bounded_across_days():
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.market_clock = _at(4)
    for _ in range(3000):
        await session._tick()
    assert len(session._equity_curve) <= 2000
    assert len(session.snapshot()["equity_curve"]) <= 240


# ------------------------------------------------- the loop that stops

@pytest.mark.asyncio
async def test_a_trading_loop_that_died_is_restarted():
    """The worst failure this program has, because everything says it is fine.

    The loop catches everything inside its body, so it does not die by raising
    -- it dies by the task ending. The session then reports a running session,
    a live websocket and a green health score while evaluating no bars at all.
    Nothing on screen contradicts it, so nothing prompts anyone to look.
    """
    session = TradingSession()
    session.running = True
    session._loop_beat = __import__("time").time()

    async def _dies():
        return None

    import asyncio
    session._loop_task = asyncio.create_task(_dies())
    await asyncio.sleep(0.01)
    assert session._loop_task.done()

    await session.supervise()

    assert session.restarts == 1
    assert session._loop_task is not None and not session._loop_task.done()
    assert any("restarted" in e["message"] for e in session.telemetry.events(20))

    session.running = False
    session._loop_task.cancel()


@pytest.mark.asyncio
async def test_a_loop_that_stopped_ticking_is_restarted_even_though_it_is_alive():
    """A task that is alive but wedged -- blocked on a request that never
    returns -- looks healthy to a liveness check. The heartbeat is what
    distinguishes running from merely existing."""
    import asyncio
    import time as _t

    from imperium.session import LOOP_STALL_SECONDS

    session = TradingSession()
    session.running = True

    async def _wedged():
        await asyncio.sleep(3600)

    session._loop_task = asyncio.create_task(_wedged())
    session._loop_beat = _t.time() - LOOP_STALL_SECONDS - 5

    await session.supervise()

    assert session.restarts == 1
    assert any("has not ticked" in e["message"] for e in session.telemetry.events(20))

    session.running = False
    if session._loop_task:
        session._loop_task.cancel()


@pytest.mark.asyncio
async def test_a_healthy_loop_is_left_alone():
    """A supervisor that restarts a working loop is worse than none: it would
    drop the bar ring and re-warm every symbol, once a second."""
    import asyncio
    import time as _t

    session = TradingSession()
    session.running = True

    async def _fine():
        await asyncio.sleep(3600)

    session._loop_task = asyncio.create_task(_fine())
    session._loop_beat = _t.time()
    task = session._loop_task

    for _ in range(5):
        await session.supervise()

    assert session.restarts == 0
    assert session._loop_task is task

    session.running = False
    task.cancel()


@pytest.mark.asyncio
async def test_a_stopped_session_is_not_restarted_by_the_supervisor():
    """Stop means stop. A supervisor that restarts a session the operator
    stopped is a bot that will not turn off."""
    session = TradingSession()
    session.running = False
    session._loop_task = None

    await session.supervise()

    assert session.restarts == 0
    assert session._loop_task is None


def test_keeping_the_machine_awake_reports_what_it_actually_did():
    """A no-op that reports success would leave the operator believing the
    laptop was being held awake for an overnight position while it slept."""
    import sys

    from imperium.keepalive import KeepAwake

    ka = KeepAwake()
    assert ka.supported == (sys.platform == "win32")
    acquired = ka.acquire()
    assert acquired == ka.active
    if sys.platform != "win32":
        assert not ka.active
        assert "not supported" in ka.note
        assert sys.platform in ka.note
    ka.release()
    assert not ka.active
