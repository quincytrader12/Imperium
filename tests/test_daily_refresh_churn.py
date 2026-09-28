"""Why the terminal kept restarting its own trading loop.

From a live run, over and over: "the trading loop has not ticked for 90s,
restarting it."

The loop was not dead. It was busy, and it was busy because of this: the
six-hourly daily-bar pull decides whether it is due by asking whether any
symbol in the universe is missing its daily history. A symbol the venue has
no daily rows for -- newly listed, long halted, a pair the daily endpoint
does not cover -- gets an empty list, and an empty list reads as missing. So
the condition is true again on the next pass, and the next, for as long as
that symbol is in the universe.

The result is a full history pull, every sixty seconds, forever: a hundred
and fifty symbols times four hundred daily rows, tens of thousands of rows
parsed on the event loop, against a request budget shared with everything
else. That is what stopped the loop ticking, and the supervisor did exactly
what it was told to do about a loop that has stopped ticking.
"""

from __future__ import annotations

import time

import pytest

from imperium.session import OVERNIGHT_REFRESH_SECONDS, TradingSession
from imperium.venues.alpaca.client import AlpacaClient
from mock_venue import KEY, SECRET, MockVenue


class _CountingClient:
    """Answers with rows for one symbol and nothing for the other."""

    def __init__(self, silent: set[str], rows: int = 30) -> None:
        self.silent = silent
        self.rows = rows
        self.calls = 0
        #: The symbol list of each call, so a test can assert on what was
        #: asked for and not only on how often.
        self.asked: list[list[str]] = []

    async def bars(self, symbols, **kw):
        self.calls += 1
        self.asked.append(list(symbols))
        out = {}
        for s in symbols:
            if s in self.silent:
                out[s] = []          # the venue has nothing for it
                continue
            # A walk, not a ramp: the trend work this feeds does real
            # numerical effort per symbol, and a straight line is not a
            # representative amount of it.
            out[s] = []
            price = 100.0
            for i in range(self.rows):
                price *= 1.0 + (0.013 if i % 3 else -0.011)
                out[s].append(
                    {"t": 1_600_000_000_000 + i * 86_400_000,
                     "o": price * 0.998, "h": price * 1.006,
                     "l": price * 0.994, "c": price, "v": 1000.0})
        return out


async def _session(silent: set[str], universe: list[str],
                   rows: int = 30) -> tuple:
    session = TradingSession()
    client = _CountingClient(silent, rows=rows)
    session.client = client
    session.universe = list(universe)
    return session, client


@pytest.mark.asyncio
async def test_a_symbol_the_venue_has_no_history_for_does_not_pull_every_minute():
    """The bug itself. One silent symbol must not make every later pass think
    the whole history is due again."""
    session, client = await _session({"QUIET"}, ["AAPL", "QUIET"])

    await session.refresh_daily_history()
    assert client.calls == 1, "the first pull should happen"

    # Whatever the loop does next, the venue was asked a moment ago and
    # nothing has changed. Ten more passes, one a minute.
    for _ in range(10):
        await session.refresh_daily_history()

    assert client.calls == 1, (
        f"the full daily history was pulled {client.calls} times in eleven "
        f"passes because one symbol has no rows; that is the whole universe "
        f"re-fetched every sixty seconds, forever")


@pytest.mark.asyncio
async def test_a_symbol_admitted_since_the_last_pull_is_still_fetched():
    """The behaviour the condition exists for, which the fix must not lose: a
    symbol the scanner admits between pulls would otherwise carry no history
    for six hours and refuse every night in that window as 'warming up'."""
    session, client = await _session(set(), ["AAPL"])
    await session.refresh_daily_history()
    assert client.calls == 1

    session.universe.append("MSFT")
    await session.refresh_daily_history()
    assert client.calls == 2, (
        "a newly admitted symbol did not trigger a pull, so it has no daily "
        "history and will refuse to trade until the six-hourly refresh")
    assert client.asked[1] == ["MSFT"], (
        f"the whole universe was re-read to learn about one new symbol: "
        f"{client.asked[1]}. The cohort rotates every twenty seconds, so that "
        f"is the entire universe pulled and re-parsed about once a minute")


@pytest.mark.asyncio
async def test_the_six_hourly_refresh_still_happens():
    """The silent symbol must not be remembered so well that its history is
    never asked for again -- a symbol with no rows today may have rows
    tomorrow."""
    session, client = await _session({"QUIET"}, ["AAPL", "QUIET"])
    await session.refresh_daily_history()
    assert client.calls == 1

    session._daily_loaded_at = time.time() - OVERNIGHT_REFRESH_SECONDS - 1
    await session.refresh_daily_history()
    assert client.calls == 2, "the six-hourly refresh stopped happening"


@pytest.mark.asyncio
async def test_a_failed_pull_is_retried_rather_than_marked_done():
    """A venue error is not an answer. Marking those symbols as asked would
    leave the book with no daily history until the six-hourly timer."""
    from imperium.venues.alpaca.client import VenueError

    session = TradingSession()

    class _Broken:
        calls = 0

        async def bars(self, symbols, **kw):
            type(self).calls += 1
            raise VenueError("the venue is unavailable")

    session.client = _Broken()
    session.universe = ["AAPL"]

    await session.refresh_daily_history()
    await session.refresh_daily_history()
    assert _Broken.calls == 2, (
        "a failed pull was recorded as done, so the history will not be "
        "asked for again until the six-hourly refresh")


# -- what the restart says about itself ------------------------------------


@pytest.mark.asyncio
async def test_a_stall_names_the_step_it_was_stuck_in():
    """"The trading loop has not ticked for 90s" is true of every stall and
    says nothing about which one this was. It sent the last investigation
    looking for a dead loop when the loop was alive and stuck inside one slow
    call."""
    session = TradingSession()
    session.running = True
    session._loop_beat = time.time() - 200
    session._loop_where = "pulling daily history"

    async def forever():
        import asyncio

        await asyncio.sleep(3600)

    import asyncio

    session._loop_task = asyncio.create_task(forever())
    try:
        await session.supervise()
    finally:
        session.running = False
        session._loop_task.cancel()

    said = [e for e in session.telemetry.events()
            if "has not ticked" in e.get("message", "")]
    assert said, "the stall was not reported at all"
    assert "pulling daily history" in said[0]["message"], said[0]["message"]


@pytest.mark.asyncio
async def test_a_long_pull_does_not_block_the_whole_process():
    """Parsing the universe's daily bars is 28ms a symbol and 150 symbols of
    it. Held in one unbroken stretch, nothing else in this process runs for
    four and a half seconds -- including the pong the browser's connection
    depends on."""
    import asyncio

    # The real shape of the work: four hundred daily rows a symbol, which is
    # what a year and a half of history costs and what the live universe
    # actually pulls.
    session, client = await _session(set(), [f"SYM{i:03d}" for i in range(60)],
                                     rows=400)

    worst = 0.0

    async def probe() -> None:
        nonlocal worst
        while True:
            at = time.perf_counter()
            await asyncio.sleep(0.005)
            worst = max(worst, time.perf_counter() - at - 0.005)

    watcher = asyncio.create_task(probe())
    # Given a turn before the work starts. Without this the probe's first
    # measurement begins *after* the blocking stretch, and the test passes by
    # never looking at it -- which is how this test first went green against a
    # version with the yield removed.
    await asyncio.sleep(0.02)
    try:
        await session.refresh_daily_history(force=True)
        # And a turn afterwards. The probe records a span when it *resumes*;
        # cancelling the moment the work returns means the longest span of all
        # -- the one that ran to the end of the work -- is never recorded, and
        # the test passes by never looking at it. That is how this test first
        # went green against a version with the yield removed.
        await asyncio.sleep(0.02)
    finally:
        watcher.cancel()

    assert worst < 1.0, (
        f"the event loop was blocked for {worst * 1000:.0f}ms in one stretch "
        f"while parsing daily bars")


@pytest.mark.asyncio
async def test_a_rotating_cohort_costs_only_the_symbols_that_are_new():
    """The live shape of it. The cohort walks the ranking, so a handful of
    names is replaced every twenty seconds and there is almost always
    something new. Under the old rule each of those was a reason to re-read
    every symbol in the universe."""
    start = [f"OLD{i:02d}" for i in range(40)]
    session, client = await _session(set(), list(start))
    await session.refresh_daily_history()
    assert len(client.asked[0]) == 40

    # Five rotations, three names retired and three admitted each time.
    carried = list(start)
    for turn in range(5):
        arrivals = [f"NEW{turn}{i}" for i in range(3)]
        carried = carried[3:] + arrivals
        session.universe = list(carried)
        await session.refresh_daily_history()

    pulled = sum(len(batch) for batch in client.asked[1:])
    assert pulled == 15, (
        f"{pulled} symbol-fetches to admit fifteen new names; the universe is "
        f"being re-read on every rotation")


@pytest.mark.asyncio
async def test_a_top_up_does_not_postpone_the_six_hourly_re_read():
    """The trap in pulling incrementally: a new symbol arrives well inside
    every six-hour window, so letting a one-symbol top-up restart the clock
    would mean the universe is never fully re-read again -- and every series
    quietly goes stale a day at a time."""
    session, client = await _session(set(), ["AAPL"])
    await session.refresh_daily_history()

    # Thirty seconds left in the window.
    session._daily_loaded_at = time.time() - OVERNIGHT_REFRESH_SECONDS + 30
    due_at = session._daily_loaded_at
    session.universe.append("MSFT")
    await session.refresh_daily_history()          # the top-up
    assert client.asked[-1] == ["MSFT"]
    assert session._daily_loaded_at == due_at, (
        "a one-symbol top-up restarted the six-hourly clock. A new symbol "
        "arrives well inside every window, so the full re-read would be "
        "pushed back forever and every series would go stale a day at a time")

    # And when it does come due, it is the whole universe.
    session._daily_loaded_at = time.time() - OVERNIGHT_REFRESH_SECONDS - 1
    await session.refresh_daily_history()
    assert sorted(client.asked[-1]) == ["AAPL", "MSFT"]
