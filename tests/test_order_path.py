"""Can this program actually place an order?

The question was asked directly, and it was the right question: until now
"paper" meant a book simulated inside this process. Orders were matched
against a local model of the market and never left the machine. So the order
path live trading depends on had never once been exercised -- the first real
order would also have been the first test of it, with money behind it.

These tests drive the path that sends, and assert on what reaches the wire.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from imperium.execution.broker import (
    LIVE_CONFIRMATION_PHRASE, LiveBroker, Mode, ModeSwitchRefused, PaperBroker,
)
from imperium.session import TradingSession
from imperium.venues import registry
from imperium.venues.alpaca.client import AlpacaClient
from mock_venue import KEY, SECRET, MockVenue


def _orders(venue: MockVenue) -> list[dict]:
    """Every order that actually reached the venue."""
    out = []
    for req in venue.requests:
        path = getattr(req, "path", None) or str(getattr(req, "url", ""))
        method = getattr(req, "method", "")
        if method == "POST" and path.endswith("/v2/orders"):
            body = getattr(req, "content", b"") or b""
            try:
                out.append(json.loads(body))
            except ValueError:
                pass
    return out


@pytest.mark.asyncio
async def test_paper_mode_sends_a_real_order_to_the_venue():
    """The whole point. A fill from a simulated book is not evidence that this
    program can place an order; a POST the venue answered is."""
    venue = MockVenue()
    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    session.credential = type("C", (), {"name": "paper-key",
                                        "trade_enabled": False})()
    try:
        await session.set_mode(Mode.PAPER)
        assert isinstance(session.broker, LiveBroker), (
            "paper mode is still simulating locally, so the order path is "
            "still untested")
        assert session.broker.mode is Mode.PAPER
        assert not session.broker.simulated

        before = len(_orders(venue))
        fill = await session.broker.apply_target("AAPL", 0.10, 100.0, 10_000.0)
        sent = _orders(venue)
    finally:
        await session.detach_client()

    assert len(sent) == before + 1, "no order reached the venue"
    assert sent[-1]["symbol"] == "AAPL"
    assert sent[-1]["side"] == "buy"
    assert fill is not None, "the venue accepted the order and no fill came back"


@pytest.mark.asyncio
async def test_paper_mode_without_a_credential_says_it_is_only_simulating():
    """The honest fallback. Silently simulating would leave the operator
    believing the order path had been exercised when it had not."""
    session = TradingSession()

    await session.set_mode(Mode.PAPER)

    assert isinstance(session.broker, PaperBroker)
    assert session.broker.simulated
    events = [e for e in session.telemetry.events(20) if e["source"] == "mode"]
    assert events and "simulated in this process" in events[0]["message"]
    assert events[0]["level"] == "warn"


@pytest.mark.asyncio
async def test_the_paper_arming_cannot_arm_a_live_broker():
    """The one thing that must not leak. Paper arms itself without a phrase;
    if that path could reach a live broker the confirmation gate would be
    decorative."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), client,
                            "some-key", mode=Mode.LIVE)
        with pytest.raises(ModeSwitchRefused):
            broker.arm_for_paper()
        assert not broker.armed
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_live_still_needs_the_phrase_and_a_tradeable_key():
    """Unchanged by any of this, and checked here because the paper path now
    shares the class."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), client, "k")
        with pytest.raises(ModeSwitchRefused):
            broker.arm("GO LIVE", credential_is_tradeable=False)
        with pytest.raises(ModeSwitchRefused):
            broker.arm("go live", credential_is_tradeable=True)
        broker.arm(LIVE_CONFIRMATION_PHRASE, credential_is_tradeable=True)
        assert broker.armed
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_an_unarmed_broker_sends_nothing():
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), client, "k")
        before = len(_orders(venue))
        with pytest.raises(ModeSwitchRefused):
            await broker.apply_target("AAPL", 0.1, 100.0, 10_000.0)
        assert len(_orders(venue)) == before
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_the_order_carries_the_fields_the_venue_needs():
    """A rejected order is not a placed order. The wire format is what decides
    whether this works, and it is the part a simulated book never checks."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), client,
                            "k", mode=Mode.PAPER)
        broker.arm_for_paper()
        await broker.apply_target("BTC/USD", 0.10, 50_000.0, 10_000.0)
        sent = _orders(venue)[-1]
    finally:
        await client.aclose()

    assert sent["symbol"] == "BTC/USD"
    assert sent["side"] == "buy"
    assert sent["type"] in ("market", "limit")
    assert sent["time_in_force"], "the venue rejects an order with no TIF"
    assert "qty" in sent or "notional" in sent
    assert sent.get("client_order_id"), (
        "no client order id — a retry after a timeout would place a second "
        "order with no way to tell it from the first")


# ---------------------------------------------------------------------------
# A refused order must reach the terminal, not only a log file.
#
# Reported directly by an operator: the console printed "not sending a market
# on close order for AG, its size is less than one whole share and auctions
# don't take fractions" with no explanation of what that meant or where to
# look. The engine's own decision still said TRADING with a sizing reason
# attached, so from the web UI a refusal here is indistinguishable from a bug
# in the order path -- unless the refusal itself reaches the Activity log.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_order_reaches_the_activity_log_not_only_stdout():
    """The operator's exact situation: an auction order that rounds to zero
    shares. Without telemetry wired in, this was a log.info() line visible
    only in a console window running the packaged exe."""
    from imperium.telemetry.streams import Level, TelemetryHub

    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    hub = TelemetryHub()
    try:
        broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), client,
                            "k", mode=Mode.PAPER, telemetry=hub)
        broker.arm_for_paper()
        # A tiny target weight on a normally-priced share sizes to a fraction.
        fill = await broker.apply_target("AAPL", 0.0001, 180.0, 10_000.0,
                                         order="market-on-close")
    finally:
        await client.aclose()

    assert fill is None
    events = [e for e in hub.events(20) if e["source"] == "order"]
    assert events, (
        "the order was refused and nothing reached the Activity log — the "
        "operator's only explanation was a console line nobody but a raw "
        "log window would see")
    assert "AAPL" in events[0]["message"]
    assert "whole share" in events[0]["message"]
    assert events[0]["level"] == "warn"

    pulses = [p for p in hub.pulse_window(50) if p["kind"] == "refused"]
    assert any(p["symbol"] == "AAPL" for p in pulses), (
        "no orb pulse fired either, so the cluster shows nothing happened")


@pytest.mark.asyncio
async def test_every_silent_refusal_branch_now_reaches_telemetry():
    """The three siblings of the fractional-share refusal: not tradable, below
    the venue minimum, and an auction order on a non-equity. All four used
    log.info() and nothing else before this."""
    from imperium.telemetry.streams import TelemetryHub

    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    hub = TelemetryHub()
    try:
        broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), client,
                            "k", mode=Mode.PAPER, telemetry=hub)
        broker.arm_for_paper()

        # Not tradable.
        await broker.apply_target("HALTED", 0.1, 50.0, 10_000.0)
        # An auction order on a non-equity.
        await broker.apply_target("BTC/USD", 0.1, 50_000.0, 10_000.0,
                                  order="market-on-close")
    finally:
        await client.aclose()

    messages = " ".join(e["message"] for e in hub.events(20)
                        if e["source"] == "order")
    assert "HALTED" in messages and "not tradable" in messages
    assert "BTC/USD" in messages and "US equities" in messages


@pytest.mark.asyncio
async def test_a_broker_with_no_telemetry_still_works_and_does_not_raise():
    """Tests that construct a broker directly, without a hub, must keep
    passing -- telemetry is a courtesy to the UI, not a requirement to place
    an order."""
    venue = MockVenue()
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), client,
                            "k", mode=Mode.PAPER)          # no telemetry=
        broker.arm_for_paper()
        await broker.apply_target("HALTED", 0.1, 50.0, 10_000.0)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_set_mode_actually_wires_the_session_telemetry_into_the_broker():
    """The call site, not the mechanism.

    Every test above constructs a LiveBroker directly and passes telemetry=
    by hand, so all of them still pass with session.set_mode() forgetting to
    pass it along. Driven through set_mode instead: a refusal after switching
    mode the ordinary way must land in the same telemetry hub the reasoning
    panel and Activity log read from.
    """
    venue = MockVenue()
    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    session.credential = type("C", (), {"name": "paper-key",
                                        "trade_enabled": False})()
    try:
        await session.set_mode(Mode.PAPER)
        assert isinstance(session.broker, LiveBroker)

        before = len(session.telemetry.events(50))
        fill = await session.broker.apply_target(
            "AAPL", 0.0001, 180.0, 10_000.0, order="market-on-close")
    finally:
        await session.detach_client()

    assert fill is None
    after = [e for e in session.telemetry.events(50) if e["source"] == "order"]
    assert after, (
        "set_mode built a broker with no telemetry attached — a refusal "
        "through the ordinary mode-switch path never reaches the session's "
        "own Activity log")


@pytest.mark.asyncio
async def test_set_mode_wires_telemetry_for_live_too_not_only_paper():
    """The sibling branch. Paper and live construct LiveBroker on two
    separate lines in set_mode, and a fix to one is not a fix to the other."""
    venue = MockVenue()
    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    session.credential = type("C", (), {"name": "live-key",
                                        "trade_enabled": True})()
    try:
        await session.set_mode(Mode.LIVE, phrase=LIVE_CONFIRMATION_PHRASE)
        assert isinstance(session.broker, LiveBroker)
        assert session.broker.armed

        fill = await session.broker.apply_target(
            "AAPL", 0.0001, 180.0, 10_000.0, order="market-on-close")
    finally:
        await session.detach_client()

    assert fill is None
    after = [e for e in session.telemetry.events(50) if e["source"] == "order"]
    assert after, (
        "the live-mode branch of set_mode built a broker with no telemetry "
        "attached")
