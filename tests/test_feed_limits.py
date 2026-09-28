"""The data plan's subscription cap, and the silence it causes.

Alpaca's basic plan limits concurrent websocket subscriptions. Two properties
of that limit make it dangerous rather than merely restrictive:

* an over-limit request is rejected **whole** -- error 405, with previous
  subscriptions left untouched, which on a fresh connection means none at all;
* the rejection arrives as a message with no symbol attached, which the symbol
  filter used to discard.

Together those produce a connected socket delivering nothing, with the one
message that explains it thrown away. It reads as a quiet market.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import pytest

from imperium.execution.broker import PaperBroker
from imperium.session import TradingSession
from imperium.telemetry.streams import TelemetryHub
from imperium.venues import registry
from imperium.venues.alpaca.feed import (
    MIN_STREAM_SYMBOLS, STREAM_SYMBOL_LIMIT, MarketFeed,
)
from imperium.venues.assets import AssetClass


def _feed() -> MarketFeed:
    return MarketFeed(registry.get(registry.DEFAULT_VENUE), TelemetryHub())


class _FakeSocket:
    """Records what was sent, and replays a scripted server."""

    def __init__(self, script: list[dict]) -> None:
        self.sent: list[dict] = []
        self._script = list(script)

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def recv(self) -> str:
        return json.dumps([self._script.pop(0)]) if self._script else json.dumps([])


@pytest.mark.asyncio
async def test_the_subscription_is_capped_at_what_the_plan_allows():
    """Prevents asking for more than the plan will give.

    The request is rejected whole rather than trimmed, so asking for 150
    symbols on a 30-symbol plan does not get 30 -- it gets nothing.
    """
    feed = _feed()
    ws = _FakeSocket([{"T": "success", "msg": "authenticated"}])
    symbols = [f"S{i:03d}" for i in range(150)]

    from imperium.venues.assets import AssetClass
    await feed._handshake(ws, AssetClass.US_EQUITY, symbols)

    subscribe = [m for m in ws.sent if m.get("action") == "subscribe"]
    assert len(subscribe) == 1
    assert len(subscribe[0]["bars"]) == STREAM_SYMBOL_LIMIT
    assert len(subscribe[0]["quotes"]) == STREAM_SYMBOL_LIMIT


@pytest.mark.asyncio
async def test_a_rejected_subscription_is_reported_not_discarded():
    """The message that explains the silence used to be the one thrown away:
    it carries no symbol, and the handler filtered on symbols."""
    feed = _feed()
    feed.symbols = [f"S{i}" for i in range(150)]
    feed._subscribed = {AssetClass.US_EQUITY: feed.symbols[:30]}

    feed._handle(json.dumps([{"T": "error", "code": 405,
                              "msg": "symbol limit exceeded"}]))

    assert "405" in feed.last_error
    assert feed.errors == 1
    events = [e["message"] for e in feed.telemetry.events(10)]
    assert any("refused" in m for m in events)


@pytest.mark.asyncio
async def test_the_cap_is_negotiated_down_rather_than_guessed():
    """The real limit depends on a data subscription this program cannot read,
    so it is discovered by halving until the venue accepts."""
    feed = _feed()
    feed.symbols = [f"S{i}" for i in range(150)]
    feed._subscribed = {AssetClass.US_EQUITY: feed.symbols[:30]}

    feed._handle(json.dumps([{"T": "error", "code": 405, "msg": "limit"}]))
    assert feed.symbol_limit == 15
    assert feed._resubscribe.is_set()

    feed._subscribed = {AssetClass.US_EQUITY: feed.symbols[:15]}
    feed._handle(json.dumps([{"T": "error", "code": 405, "msg": "limit"}]))
    assert feed.symbol_limit == 7


@pytest.mark.asyncio
async def test_negotiation_never_converges_on_nothing():
    """Halving without a floor reaches zero, and a stream of nothing is not a
    stream. A handful of symbols is still worth having."""
    feed = _feed()
    feed.symbols = [f"S{i}" for i in range(150)]
    for _ in range(12):
        feed._subscribed = {AssetClass.US_EQUITY: feed.symbols[:feed.symbol_limit]}
        feed._handle(json.dumps([{"T": "error", "code": 405, "msg": "limit"}]))
    assert feed.symbol_limit == MIN_STREAM_SYMBOLS


@pytest.mark.asyncio
async def test_an_unrelated_stream_error_is_surfaced_without_changing_the_cap():
    feed = _feed()
    feed.symbols = ["AAPL"]
    before = feed.symbol_limit

    feed._handle(json.dumps([{"T": "error", "code": 400, "msg": "bad message"}]))

    assert feed.symbol_limit == before
    assert not feed._resubscribe.is_set()
    assert "400" in feed.last_error


@pytest.mark.asyncio
async def test_a_held_position_is_first_in_the_queue_for_a_stream():
    """The cap decides who gets a live price. A carried position is the one
    symbol where a stale price means a stop that does not fire and an exit
    sized on a number from several minutes ago."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("100000")
    session.universe = [f"S{i:03d}" for i in range(150)]
    laggard = session.universe[-1]
    session.feed.quote(laggard).last = 100.0
    await session.broker.apply_target(laggard, 0.1, 100.0, 100_000.0)

    order = session._stream_priority()

    assert order[0] == laggard
    assert order[:STREAM_SYMBOL_LIMIT].count(laggard) == 1
    assert sorted(order) == sorted(session.universe)


@pytest.mark.asyncio
async def test_the_snapshot_reports_streamed_against_scanned():
    """The gap between them is the plan's cap, not a fault, and an operator
    who cannot see it will read a symbol with no live price as a broken one."""
    session = TradingSession()
    session.universe = [f"S{i:03d}" for i in range(150)]
    session.feed.symbols = list(session.universe)
    session.feed.dropped = 120

    feed = session.snapshot()["feed"]
    assert feed["streamed"] == STREAM_SYMBOL_LIMIT
    assert feed["symbol_limit"] == STREAM_SYMBOL_LIMIT
    assert feed["dropped"] == 120


@pytest.mark.asyncio
async def test_the_session_hands_the_feed_the_prioritised_order():
    """Prevents the priority being computed and then not used.

    _stream_priority can be perfectly correct and never reach the subscription.
    The feed takes the first N of whatever list it is given, so the order it
    actually receives is the thing that decides who gets a live price.
    """
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("100000")
    session.universe = [f"S{i:03d}" for i in range(60)]
    laggard = session.universe[-1]
    session.feed.quote(laggard).last = 100.0
    await session.broker.apply_target(laggard, 0.1, 100.0, 100_000.0)

    handed: list[list[str]] = []

    async def _record(symbols):
        handed.append(list(symbols))

    session.feed.start = _record            # type: ignore[assignment]
    session.market_clock = session.market_clock
    await session.start()
    try:
        assert handed, "the session must actually start the feed"
        assert handed[0][0] == laggard, (
            "the held position must be first in what the feed is given, not "
            "merely first in a list nobody passed on")
    finally:
        await session.stop()


class _ScriptedSocket:
    """A socket that authenticates, then closes so the loop reconnects."""

    def __init__(self, recorder: list[list[str]]) -> None:
        self._recorder = recorder
        self._auth_sent = False

    async def send(self, raw: str) -> None:
        msg = json.loads(raw)
        if msg.get("action") == "subscribe":
            self._recorder.append(list(msg.get("bars", [])))

    async def recv(self) -> str:
        return json.dumps([{"T": "success", "msg": "authenticated"}])

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration          # the socket drops immediately

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_a_reconnecting_socket_subscribes_the_symbols_wanted_now(monkeypatch):
    """Not the ones its task was created with.

    The cohort rotates every twenty seconds, so by the time a dropped socket
    comes back the symbols worth streaming have moved on. A task that captured
    its list at creation reconnects to a set of retired names and stays there
    for the rest of the session -- a connected, healthy, entirely useless
    stream.
    """
    subscribed: list[list[str]] = []

    def _connect(url, **kwargs):
        return _ScriptedSocket(subscribed)

    monkeypatch.setattr("imperium.venues.alpaca.feed.websockets.connect", _connect)

    feed = _feed()
    feed.set_credentials("k", "s")
    await feed.start(["AAPL", "MSFT"])
    # Let it connect, drop, and come back at least once.
    for _ in range(200):
        await asyncio.sleep(0.01)
        if subscribed:
            break
    assert subscribed and subscribed[0] == ["AAPL", "MSFT"]

    # The cohort rotates underneath it.
    feed.symbols = ["TSLA", "NVDA"]
    before = len(subscribed)
    for _ in range(400):
        await asyncio.sleep(0.01)
        if len(subscribed) > before:
            break
    await feed.stop()

    assert len(subscribed) > before, "the socket never reconnected"
    assert subscribed[-1] == ["TSLA", "NVDA"], (
        f"the socket reconnected to {subscribed[-1]} — the list it was "
        f"created with, not the symbols being reasoned about now")
