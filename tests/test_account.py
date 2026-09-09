"""The balance the terminal shows must be the balance the account holds.

Every limit in this program is a fraction of equity, so a wrong equity is not a
cosmetic problem: it is a sizer working from a number nobody has.
"""

from __future__ import annotations

import time

import pytest

from imperium.execution.broker import LiveBroker, PaperBroker
from imperium.session import TradingSession
from imperium.venues import registry
from mock_venue import KEY, SECRET, MockVenue


def _session(venue: MockVenue) -> TradingSession:
    from imperium.venues.alpaca.client import AlpacaClient

    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    return session


@pytest.mark.asyncio
async def test_the_header_shows_the_account_not_the_simulated_default():
    """Prevents the terminal reporting a number it invented.

    The simulated brokers start with a round $10,000 of make-believe cash. That
    is a reasonable default for a book with no account behind it and a
    completely unreasonable thing to print in the place where the operator's
    balance goes.
    """
    venue = MockVenue()
    venue.equity = 742.19
    session = _session(venue)
    try:
        assert float(session.broker.cash) == 10_000.0     # the default
        assert session.snapshot()["account"]["known"] is False

        session.absorb_account(await session.client.account())
        account = session.snapshot()["account"]

        assert account["known"] is True
        assert account["equity"] == pytest.approx(742.19)
        assert account["cash"] == pytest.approx(742.19)
        assert account["currency"] == "USD"
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_a_simulated_book_is_seeded_from_the_real_balance():
    """Prevents a paper book that sizes against money the account does not have.

    Every limit is a fraction of equity, so running a $742 account against a
    $10,000 book does not produce smaller versions of the same decisions -- it
    produces different ones. A paper run whose sizing does not match the live
    account it is standing in for is not a rehearsal of anything.
    """
    venue = MockVenue()
    venue.equity = 742.19
    session = _session(venue)
    try:
        session.absorb_account(await session.client.account())
        assert float(session.broker.cash) == pytest.approx(742.19)
        assert session.snapshot()["account"]["seeded"] is True
        assert session.day_start_equity == pytest.approx(742.19)
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_a_book_that_has_already_traded_is_never_re_seeded():
    """Prevents silently rewriting a running book's P&L.

    Seeding is a starting condition, not a correction. Applying it to a book
    that already holds positions would move the cash line underneath trades
    that have already happened, and every P&L figure derived from it would
    become fiction.
    """
    venue = MockVenue()
    venue.equity = 5_000.0
    session = _session(venue)
    try:
        session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
        session.feed.quote("AAPL").last = 100.0
        await session.broker.apply_target("AAPL", 0.1, 100.0, 10_000.0)
        traded_cash = float(session.broker.cash)

        session.absorb_account(await session.client.account())

        assert float(session.broker.cash) == pytest.approx(traded_cash)
        assert session.snapshot()["account"]["seeded"] is False
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_one_malformed_field_does_not_discard_the_whole_account():
    """Alpaca sends these as strings. An account whose buying power came back
    unparseable still knows its own equity, and refusing all of it would blank
    the balance over a field nothing here depends on."""
    session = TradingSession()
    session.absorb_account({
        "equity": "1234.56", "cash": "1000.00",
        "buying_power": "not a number", "currency": "USD",
        "status": "ACTIVE", "daytrade_count": 1,
    })
    account = session.snapshot()["account"]
    assert account["equity"] == pytest.approx(1234.56)
    assert account["cash"] == pytest.approx(1000.0)
    assert account["buying_power"] == 0.0
    assert session.allocator.day_trade_count == 1


@pytest.mark.asyncio
async def test_the_account_is_read_when_the_key_is_attached_not_a_minute_later():
    """The key check already spends this request, so the balance arrives with
    it. An operator who has just attached a key and sees a dash is looking at a
    terminal that appears not to have worked."""
    from imperium.security.credentials import Credential, CredentialStore

    venue = MockVenue()
    venue.equity = 3_333.33
    session = TradingSession()
    session.paper_endpoint = True

    class _Store:
        @staticmethod
        def require(name):
            return Credential(name=name, venue=registry.DEFAULT_VENUE,
                              api_key=KEY, secret=SECRET, trade_enabled=False)

    import imperium.session as session_mod
    real_client = session_mod.AlpacaClient

    def _patched(*args, **kwargs):
        kwargs["transport"] = venue.transport
        return real_client(*args, **kwargs)

    session_mod.AlpacaClient = _patched
    try:
        await session.attach_credential(_Store(), "k")
        assert session.snapshot()["account"]["equity"] == pytest.approx(3333.33)
    finally:
        session_mod.AlpacaClient = real_client
        await session.detach_client()


@pytest.mark.asyncio
async def test_a_live_book_is_not_seeded_because_it_is_the_account():
    """In live mode the broker syncs from the venue itself. Seeding it here
    would overwrite a real reconciliation with a second, cruder one."""
    venue = MockVenue()
    venue.equity = 9_100.0
    session = _session(venue)
    try:
        broker = LiveBroker(registry.get(registry.DEFAULT_VENUE),
                            session.client, "k")
        broker.arm("GO LIVE", True)
        session.broker = broker
        session.absorb_account(await session.client.account())

        assert session.snapshot()["account"]["seeded"] is False
        assert session.snapshot()["account"]["simulated"] is False
    finally:
        await session.detach_client()
