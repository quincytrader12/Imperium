"""The live book against the venue's own positions.

Everything the live broker records is optimistic: an order is booked when the
venue *accepts* it, because that is the only moment a market order yields a
number. Several things can happen afterwards that the book never hears about,
and every one of them leaves the terminal reporting a position it believes in
completely while sizing every later decision against a fiction.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from imperium.execution.broker import LiveBroker
from imperium.session import TradingSession
from imperium.venues import registry
from imperium.venues.alpaca.client import VenueError


class _Venue:
    """A venue with its own opinion about what is held."""

    def __init__(self, positions=None, open_orders=None, cash="1000") -> None:
        self._positions = positions or []
        self._open = open_orders or []
        self._cash = cash
        self.calls = 0

    @staticmethod
    def new_client_order_id(prefix: str = "imp") -> str:
        return prefix + "-1"

    async def positions(self):
        self.calls += 1
        return self._positions

    async def open_orders(self, symbols=None):
        return self._open

    async def account(self):
        return {"cash": self._cash, "equity": self._cash}


def _live(venue: _Venue) -> LiveBroker:
    broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), venue, "k")
    broker.arm("GO LIVE", True)
    return broker


@pytest.mark.asyncio
async def test_a_position_the_venue_does_not_hold_is_corrected():
    """An accepted order can still be rejected afterwards -- for buying power,
    a locate failure, a halt, a wash-trade block. The book would otherwise
    carry that position for the rest of the session."""
    venue = _Venue(positions=[])
    broker = _live(venue)
    broker.position("AAPL").quantity = Decimal("10")

    drift = await broker.reconcile()

    assert drift == [("AAPL", Decimal("10"), Decimal("0"))]
    assert broker.positions["AAPL"].is_flat
    assert broker.position("AAPL").avg_price == 0


@pytest.mark.asyncio
async def test_a_partial_fill_is_corrected_to_what_actually_filled():
    venue = _Venue(positions=[{"symbol": "AAPL", "qty": "30",
                               "avg_entry_price": "101.50"}])
    broker = _live(venue)
    broker.position("AAPL").quantity = Decimal("100")

    drift = await broker.reconcile()

    assert drift == [("AAPL", Decimal("100"), Decimal("30"))]
    assert broker.position("AAPL").quantity == Decimal("30")
    assert broker.position("AAPL").avg_price == Decimal("101.50")


@pytest.mark.asyncio
async def test_a_position_opened_outside_this_program_is_adopted():
    """The account may be traded by hand or by something else. A position the
    book does not know about is still risk the book is carrying."""
    venue = _Venue(positions=[{"symbol": "MSFT", "qty": "5",
                               "avg_entry_price": "400"}])
    broker = _live(venue)

    drift = await broker.reconcile()

    assert drift == [("MSFT", Decimal("0"), Decimal("5"))]
    assert broker.position("MSFT").quantity == Decimal("5")


@pytest.mark.asyncio
async def test_a_symbol_with_an_order_still_open_is_left_alone():
    """The single most important exclusion.

    A market-on-close order is accepted now and filled at an auction hours
    later. Between those two moments the venue holds no position -- and
    "correcting" the book to flat would flatten it and immediately re-submit,
    turning one intended trade into an endless loop of them.
    """
    venue = _Venue(positions=[], open_orders=[{"symbol": "AAPL", "id": "1"}])
    broker = _live(venue)
    broker.position("AAPL").quantity = Decimal("10")

    drift = await broker.reconcile()

    assert drift == []
    assert broker.position("AAPL").quantity == Decimal("10")


@pytest.mark.asyncio
async def test_an_agreeing_book_reports_nothing():
    venue = _Venue(positions=[{"symbol": "AAPL", "qty": "10",
                               "avg_entry_price": "100"}])
    broker = _live(venue)
    broker.position("AAPL").quantity = Decimal("10")

    assert await broker.reconcile() == []


@pytest.mark.asyncio
async def test_it_waits_rather_than_guessing_when_the_order_list_is_unavailable():
    """Without the open-order list this cannot tell "not filled yet" from "did
    not happen". Guessing would flatten live positions on a transient error."""
    class _Broken(_Venue):
        async def open_orders(self, symbols=None):
            raise VenueError("orders unavailable", status=500)

    venue = _Broken(positions=[])
    broker = _live(venue)
    broker.position("AAPL").quantity = Decimal("10")

    assert await broker.reconcile() == []
    assert broker.position("AAPL").quantity == Decimal("10")


@pytest.mark.asyncio
async def test_cash_is_taken_from_the_account_too():
    venue = _Venue(positions=[], cash="4321.00")
    broker = _live(venue)
    await broker.reconcile()
    assert broker.cash == Decimal("4321.00")


# --------------------------------------------------------- through the session

@pytest.mark.asyncio
async def test_a_simulated_book_is_never_reconciled():
    """There is no venue for a paper book to disagree with, and reading one
    would spend requests comparing a simulation against somebody else's
    positions -- then "correcting" the simulation to them.

    Tested on a simulated broker that *does* have a reconcile method, because
    the default ones do not: without that the hasattr check catches this case
    first and the guard that matters could be deleted unnoticed.
    """
    from imperium.execution.broker import PaperBroker

    called = []

    class _SimulatedWithReconcile(PaperBroker):
        async def reconcile(self):
            called.append(1)
            return [("AAPL", Decimal("1"), Decimal("0"))]

    session = TradingSession()
    session.broker = _SimulatedWithReconcile(registry.get(registry.DEFAULT_VENUE))
    assert session.broker.simulated
    session._reconciled_at = 0.0

    await session._reconcile_book()

    assert called == [], "a simulated book has no venue to be corrected against"
    assert session.reconciliations == 0


@pytest.mark.asyncio
async def test_the_tick_reconciles_rather_than_only_the_method_existing():
    """Prevents a correct reconciliation nothing ever calls.

    The whole point is that it happens on its own, on a timer, without anybody
    asking. A method that works and is never reached is the same as no method.
    """
    venue = _Venue(positions=[])
    session = TradingSession()
    session.broker = _live(venue)
    session.broker.position("AAPL").quantity = Decimal("10")
    session.feed.quote("AAPL").last = 100.0
    session._reconciled_at = 0.0

    await session._tick()

    assert venue.calls >= 1, "the tick must reach the venue"
    assert session.reconciliations == 1
    assert session.broker.positions["AAPL"].is_flat


@pytest.mark.asyncio
async def test_a_correction_is_reported_loudly_not_absorbed():
    """A difference means an order did not do what this program was told it
    did. That is the single most important thing an operator can be shown, so
    it is an error-level event rather than a silent fix."""
    venue = _Venue(positions=[])
    session = TradingSession()
    session.broker = _live(venue)
    session.broker.position("AAPL").quantity = Decimal("10")
    session.feed.quote("AAPL").last = 100.0
    session._reconciled_at = 0.0

    await session._reconcile_book()

    assert session.reconciliations == 1
    events = session.telemetry.events(10)
    assert any(e["level"] == "error" and "AAPL" in e["message"] for e in events)
    assert any("venue holds" in e["message"] for e in events)
    assert session.snapshot()["reconciliations"] == 1
