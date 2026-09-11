"""Scanning the whole market without carrying the whole market.

The scan ranks every tradable listing; the engine reasons bar-by-bar about the
top of that ranking. The split is memory, measured rather than assumed: one
engine's bar ring costs about 286KB once full, so an engine per listed symbol
would be hundreds of megabytes on a laptop that is also running a browser.
"""

from __future__ import annotations

import contextlib

import pytest

from imperium.execution.broker import PaperBroker
from imperium.session import TRADED_UNIVERSE, TradingSession
from imperium.venues import registry
from imperium.venues.alpaca.client import SNAPSHOT_BATCH
from mock_venue import MAX_URL_BYTES, KEY, SECRET, MockVenue


@contextlib.asynccontextmanager
async def _session(venue: MockVenue):
    from imperium.venues.alpaca.client import AlpacaClient

    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    try:
        yield session
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_a_market_sized_sweep_is_batched_rather_than_sent_as_one_url():
    """Prevents the sweep failing in the one way that looks like success.

    The symbol list travels in the query string. Asked for the whole listing in
    one request, the URL is tens of kilobytes and is refused before it reaches
    the venue -- and the refusal surfaces as "the scanner found nothing to
    trade", which reads as a quiet market rather than as a broken request.
    """
    venue = MockVenue()
    venue.list_extra_equities(3000)

    async with _session(venue) as session:
        await session.scan_universe()

    assert venue.snapshot_batches, "the sweep must actually have asked"
    assert max(venue.snapshot_batches) <= SNAPSHOT_BATCH
    assert len(venue.snapshot_batches) >= 15, "3,000 symbols is many batches"
    longest = max(len(str(r.url)) for r in venue.requests)
    assert longest <= MAX_URL_BYTES


@pytest.mark.asyncio
async def test_the_whole_listing_is_ranked_but_only_the_top_is_carried():
    """The split the design rests on.

    Every listed symbol is priced and ranked. Only the top of that ranking gets
    an engine, because that is what costs memory. Reporting both numbers is
    what makes "it scans everything" checkable rather than a claim.
    """
    venue = MockVenue()
    venue.list_extra_equities(1200)

    async with _session(venue) as session:
        await session.scan_universe()
        snap = session.snapshot()

    scan = snap["universe_scan"]
    assert scan["considered"] >= 1200
    assert scan["priced"] >= 1200
    assert scan["size"] == TRADED_UNIVERSE
    assert len(session.engines) <= TRADED_UNIVERSE + 5

    # The constant's own contract, in literals rather than in terms of itself:
    # a bound written as `== TRADED_UNIVERSE` moves with the constant and would
    # hold just as well at the old value of 40.
    assert TRADED_UNIVERSE >= 120, "the point of the sweep is breadth"
    # And the ceiling that breadth is traded against. A full bar ring measured
    # at ~286KB, so this is the memory the terminal carries in bar history.
    assert TRADED_UNIVERSE * 286_000 < 150e6, "bar history must stay under 150MB"


@pytest.mark.asyncio
async def test_state_for_symbols_that_fell_out_of_the_universe_is_released():
    """Prevents unbounded growth on a terminal meant to run for days.

    The sweep sees the whole market every quarter of an hour, and rankings
    move. Without pruning, a symbol that was briefly interesting on a Tuesday
    keeps its engine and its bar ring for as long as the process lives.
    """
    venue = MockVenue()
    venue.list_extra_equities(600)

    async with _session(venue) as session:
        await session.scan_universe()
        for symbol in list(session.universe)[:20]:
            session.engine(symbol)
        # Symbols from an earlier sweep that are no longer ranked.
        for i in range(300):
            session.engine(f"GONE{i}")
            session.feed.quote(f"GONE{i}").last = 1.0
        assert len(session.engines) > TRADED_UNIVERSE

        await session.scan_universe()

        assert len(session.engines) <= TRADED_UNIVERSE + 5
        assert not [s for s in session.engines if s.startswith("GONE")]
        assert not [s for s in session.feed.quotes if s.startswith("GONE")]


@pytest.mark.asyncio
async def test_a_symbol_holding_a_position_is_never_pruned():
    """Its engine is the thing managing that position. Dropping it because the
    symbol fell down a turnover ranking would leave a real position with
    nothing behind it -- no stop, no exit, and nothing on screen."""
    venue = MockVenue()
    venue.list_extra_equities(600)

    async with _session(venue) as session:
        session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
        session.engine("HELD")
        session.feed.quote("HELD").last = 100.0
        await session.broker.apply_target("HELD", 0.1, 100.0, 10_000.0)

        await session.scan_universe()

        assert "HELD" in session.engines
        assert "HELD" in session.universe
        assert "HELD" in session.feed.quotes

        # And when the ranking itself does not save it. scan_universe appends
        # held symbols to the kept list, which masks the guard inside _prune --
        # so the guard is exercised directly, with a keep list that excludes it.
        session._prune(["SOMETHING", "ELSE"])
        assert "HELD" in session.engines
        assert "HELD" in session.feed.quotes


@pytest.mark.asyncio
async def test_an_overnight_hold_is_never_pruned_either():
    """It has an opening exit still to be lodged. Losing its engine between the
    close and the next open is losing the trade."""
    venue = MockVenue()
    venue.list_extra_equities(600)

    async with _session(venue) as session:
        session.engine("NIGHT")
        session.overnight_holdings["NIGHT"] = 0.05
        await session.scan_universe()
        assert "NIGHT" in session.engines


def test_the_allocator_never_forgets_a_symbol_that_carries_weight():
    """Gross exposure is summed from these states. Dropping one that carries
    weight would make the book look emptier than it is, and the gross-exposure
    ceiling is enforced against that sum."""
    from imperium.execution.portfolio import PortfolioAllocator
    from imperium.execution.risk import RiskLimits

    allocator = PortfolioAllocator(RiskLimits())
    allocator.observe("FLAT").current_weight = 0.0
    allocator.observe("HELD").current_weight = 0.08
    before = allocator.gross_exposure

    dropped = allocator.forget({"FLAT", "HELD"})

    assert dropped == 1
    assert "HELD" in allocator.states and "FLAT" not in allocator.states
    assert allocator.gross_exposure == before
