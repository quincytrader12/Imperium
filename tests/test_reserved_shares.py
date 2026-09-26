"""Shares a working sell order has already spoken for.

From a live run: "could not lodge the opening exit for AG: the account is not
permitted to do this -- usually insufficient buying power, or an attempt to
sell more than is held."

Nothing had drifted. The venue reserves stock against an *open* sell order;
the local book does not, and correctly so, because nothing has sold yet. The
overnight exit lodges a market-on-open sell that sits until the auction, and
anything that then decides to exit the same position -- the give-back
ratchet, or a retry of the overnight exit itself -- sizes against shares the
venue has already set aside and is rejected.

The rejection was not the whole cost. The overnight exit only drops a symbol
once apply_target returns; a raised VenueError left it in the set, so the
same doomed order went out on every pre-open tick, and the operator got the
same alarming sentence over and over about a position that was already on its
way out.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from imperium.execution import protect
from imperium.execution.broker import (
    MARKET_ON_OPEN, LiveBroker, Mode, Position,
)
from imperium.venues import registry
from imperium.venues.alpaca.client import AlpacaClient, VenueError
from mock_venue import KEY, SECRET, MockVenue

from test_order_path import _orders


def _working_sell(venue: MockVenue, symbol: str, qty: str,
                  filled: str = "0", *, side: str = "sell",
                  coid: str | None = None) -> None:
    """Put an order on the venue's book that has not filled yet.

    MockVenue._place fills everything immediately, which is what makes it
    useful elsewhere and useless here: the state this is about is an order the
    venue has accepted and not yet executed.
    """
    coid = coid or f"working-{symbol}-{len(venue.orders)}"
    venue.orders[coid] = {
        "id": f"ord-{coid}", "client_order_id": coid, "symbol": symbol,
        "side": side, "qty": qty, "filled_qty": filled,
        "type": "market", "time_in_force": "opg", "status": "new",
    }


def _broker(venue: MockVenue, client: AlpacaClient, symbol: str,
            quantity: str) -> LiveBroker:
    broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), client, "k",
                        mode=Mode.PAPER)
    broker.arm_for_paper()
    broker.positions[symbol] = Position(symbol, Decimal(quantity),
                                        Decimal("100"))
    return broker


# -- what the venue has set aside -----------------------------------------


@pytest.mark.asyncio
async def test_a_working_sell_reserves_its_unfilled_remainder():
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = _broker(venue, client, "AAPL", "10")
        _working_sell(venue, "AAPL", "4")
        assert await broker.reserved_for_sale("AAPL") == Decimal("4")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_a_partial_fill_only_reserves_what_is_left():
    """The venue is holding the rest, not the whole order. Counting the full
    quantity would refuse an exit for shares that are free to sell."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = _broker(venue, client, "AAPL", "10")
        _working_sell(venue, "AAPL", "6", filled="4")
        assert await broker.reserved_for_sale("AAPL") == Decimal("2")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_buys_and_other_symbols_reserve_nothing():
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = _broker(venue, client, "AAPL", "10")
        _working_sell(venue, "AAPL", "3", side="buy")
        _working_sell(venue, "SPY", "5")
        assert await broker.reserved_for_sale("AAPL") == Decimal("0")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_an_unreachable_venue_reserves_nothing():
    """Fails open on purpose. Refusing to sell because the order list could
    not be read would trap the position -- and the venue rejects a genuine
    duplicate anyway, which is the outcome this is only trying to explain."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = _broker(venue, client, "AAPL", "10")
        venue.fail_next.append(
            lambda request: __import__("httpx").Response(500, json={
                "code": 50010000, "message": "server error"}))
        assert await broker.reserved_for_sale("AAPL") == Decimal("0")
    finally:
        await client.aclose()


# -- what that does to an order -------------------------------------------


@pytest.mark.asyncio
async def test_a_second_exit_is_refused_rather_than_rejected():
    """The bug as the operator met it. The whole position is already on its
    way out; sending another sell for it cannot fill."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = _broker(venue, client, "AAPL", "5")
        _working_sell(venue, "AAPL", "5")
        before = len(_orders(venue))

        fill = await broker.apply_target("AAPL", 0.0, 100.0, 10_000.0,
                                         order=MARKET_ON_OPEN)

        assert fill is None
        assert len(_orders(venue)) == before, (
            "an order the venue can only reject was sent anyway")
        said = broker._refusals["AAPL"][0]
        assert "already working" in said, said
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_an_exit_is_sized_to_what_is_left_unreserved():
    """Half the position is spoken for, half is not. The unreserved half is a
    real order and must still go."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = _broker(venue, client, "AAPL", "10")
        _working_sell(venue, "AAPL", "6")

        await broker.apply_target("AAPL", 0.0, 100.0, 10_000.0)

        sent = _orders(venue)[-1]
        assert sent["side"] == "sell"
        assert Decimal(str(sent["qty"])) == Decimal("4"), (
            f"sized against the local book of 10 rather than the 4 the venue "
            f"has not reserved: {sent['qty']}")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_the_first_exit_still_goes_out():
    """The ordinary case, which is every exit until one is working. Guards
    against a check that refuses everything."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = _broker(venue, client, "AAPL", "5")
        await broker.apply_target("AAPL", 0.0, 100.0, 10_000.0)
        sent = _orders(venue)[-1]
        assert sent["side"] == "sell"
        assert Decimal(str(sent["qty"])) == Decimal("5")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_a_short_is_not_blocked_by_its_own_working_order():
    """A sell against a flat or short book opens or extends a short; it
    disposes of nothing, and the venue reserves no stock against it. Treating
    an open sell as a reservation there would refuse a legitimate order."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = _broker(venue, client, "AAPL", "-5")
        _working_sell(venue, "AAPL", "5")
        before = len(_orders(venue))

        await broker.apply_target("AAPL", -0.10, 100.0, 10_000.0)

        assert len(_orders(venue)) == before + 1, (
            "extending a short was refused because an open sell was read as a "
            "reservation against stock that is not held")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_a_buy_never_asks_the_venue_for_open_orders():
    """A buy reserves nothing and needs no round trip. Asking anyway would put
    an extra request on every entry, on a client that is rate limited."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = _broker(venue, client, "AAPL", "0")
        listed = [r for r in venue.requests
                  if r.method == "GET" and r.url.path == "/v2/orders"]
        await broker.apply_target("AAPL", 0.10, 100.0, 10_000.0)
        after = [r for r in venue.requests
                 if r.method == "GET" and r.url.path == "/v2/orders"]
        assert len(after) == len(listed)
    finally:
        await client.aclose()


# -- and to the overnight exit that reported it ---------------------------


@pytest.mark.asyncio
async def test_the_overnight_exit_stops_retrying_once_one_is_working():
    """Why the message repeated. The exit pops the symbol when apply_target
    returns and leaves it in place when it raises, so a venue rejection meant
    the same order every pre-open tick. A refusal returns."""
    from imperium.session import TradingSession

    venue = MockVenue()
    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    try:
        session.broker = _broker(venue, session.client, "AAPL", "5")
        session.overnight_holdings["AAPL"] = 100.0
        _working_sell(venue, "AAPL", "5")

        await session._exit_overnight_holdings()

        assert "AAPL" not in session.overnight_holdings, (
            "the exit is already working at the venue, and this will send "
            "another one on the next tick")
    finally:
        await session.detach_client()


def test_the_venue_error_the_operator_saw_is_still_caught():
    """The refusal above is not a reason to stop handling a real one: an exit
    can be rejected for restriction or buying power too, and that must still
    reach the operator rather than the console."""
    import inspect

    from imperium import session as session_module

    source = inspect.getsource(session_module.TradingSession
                               ._exit_overnight_holdings)
    assert "could not lodge the opening exit" in source
    assert "VenueError" in source
    assert VenueError is session_module.VenueError


@pytest.mark.asyncio
async def test_a_protective_exit_that_was_not_sent_is_not_counted():
    """The daily brief reports how many positions the ratchet closed today.
    A refused order closes nothing, and a brief that says otherwise is worse
    than no brief."""
    from imperium.session import TradingSession

    venue = MockVenue()
    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    try:
        session.broker = _broker(venue, session.client, "AAPL", "5")
        # Bought at 100, marked at 200, and the whole position is already on
        # its way out on a working sell.
        session.broker.positions["AAPL"] = Position("AAPL", Decimal("5"),
                                                    Decimal("100"))
        session.feed.quote("AAPL").last = 200.0
        _working_sell(venue, "AAPL", "5")
        # Take it to its high-water mark, then hand most of it back.
        session._peaks.setdefault("AAPL", protect.PositionPeak()).observe(2.0)
        session.feed.quote("AAPL").last = 110.0

        before = session._today_protected
        await session._protect_positions()

        assert session._today_protected == before, (
            "an exit that was never sent was counted as a protected close")
    finally:
        await session.detach_client()
