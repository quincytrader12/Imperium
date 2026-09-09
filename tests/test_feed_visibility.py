"""A dark data lamp has to say why it is dark.

The operator's report was "the data window is not firing, even the data light
is not on, but everything else is running" -- and from the terminal there was
no way to tell which of five situations that was: the session not started, the
socket still connecting, the venue refusing the key, the plan refusing the
subscription, or a perfectly healthy socket on a market that is shut and has
nothing to send.

Every fact needed to separate them was already in the payload and thrown away
in the browser, so the lamp was the only signal and it was ambiguous.
"""

from __future__ import annotations

import datetime as dt
import time
from decimal import Decimal

import pytest

from imperium.execution.broker import PaperBroker
from imperium.session import STALE_AFTER_SECONDS, TradingSession
from imperium.venues import registry
from imperium.venues.alpaca.client import MarketClock

UTC = dt.timezone.utc


def _session(*, open_market: bool) -> TradingSession:
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.universe = ["AAPL", "MSFT", "SPY"]
    session.market_clock = MarketClock(
        is_open=open_market,
        next_open=None if open_market else dt.datetime.now(UTC) + dt.timedelta(hours=14),
    )
    return session


def test_a_shut_market_is_reported_as_expected_rather_than_as_a_fault():
    """The operator's actual situation, and the one the lamp handled worst.

    A connected socket on a closed equity market sends nothing, correctly.
    Reported as a dark lamp and the words "no data", that is indistinguishable
    from a feed that is broken -- so the terminal has to say the market is
    shut and that this is not a fault.
    """
    session = _session(open_market=False)
    session.running = True
    session.feed.connected = True
    session.feed.symbols = list(session.universe)
    session.feed.last_message_at = 0.0        # nothing has ever arrived

    reason = session._feed_reason()

    assert "shut" in reason
    assert "not a fault" in reason
    assert "3" in reason, "it must say how many symbols are subscribed"


def test_a_socket_that_never_connected_names_the_error_rather_than_going_quiet():
    """The failure the dark lamp was hiding. The venue's own words are the
    only thing that distinguishes a bad key from a blocked port."""
    session = _session(open_market=True)
    session.running = True
    session.feed.connected = False
    session.feed.last_error = "InvalidStatusCode: server rejected WebSocket connection"

    reason = session._feed_reason()

    assert "not connected" in reason
    assert "server rejected WebSocket connection" in reason


def test_a_connected_socket_with_nothing_subscribed_says_so():
    """The quietest failure there is: the plan rejects an over-limit request
    whole, so the socket is up, healthy, and subscribed to nothing at all. It
    reads exactly like a market with no activity."""
    session = _session(open_market=True)
    session.running = True
    session.feed.connected = True
    session.feed.symbols = []

    reason = session._feed_reason()

    assert "nothing is subscribed" in reason


def test_silence_on_an_open_market_is_flagged_rather_than_excused():
    """The one case that really is worth an operator's attention. It must not
    be worded like the closed-market case, which is the whole point."""
    session = _session(open_market=True)
    session.running = True
    session.feed.connected = True
    session.feed.symbols = list(session.universe)
    session.feed.last_message_at = time.time() - (STALE_AFTER_SECONDS + 60)

    reason = session._feed_reason()

    assert "market is open" in reason
    assert "not a fault" not in reason


def test_a_stopped_session_says_it_is_stopped_rather_than_reporting_a_data_fault():
    session = _session(open_market=True)
    session.running = False

    assert "not started" in session._feed_reason()


def test_a_live_stream_reports_how_many_symbols_are_flowing():
    session = _session(open_market=True)
    session.running = True
    session.feed.connected = True
    session.feed.symbols = list(session.universe)
    session.feed.last_message_at = time.time()

    reason = session._feed_reason()

    assert "streaming" in reason
    assert "3" in reason


@pytest.mark.asyncio
async def test_the_explanation_actually_reaches_the_browser():
    """Driven through the snapshot rather than the method.

    The reason existing is not the fix -- it was the *rendering* that was
    missing, and a sentence the payload does not carry cannot be rendered.
    """
    session = _session(open_market=False)
    session.running = True
    session.feed.connected = True
    session.feed.symbols = list(session.universe)

    snapshot = session.snapshot()

    assert "reason" in snapshot["feed"], (
        "the explanation must be in the payload, or the browser has nothing "
        "to show")
    assert snapshot["feed"]["reason"] == session._feed_reason()
    assert snapshot["lamps"]["data"] == "off", (
        "the lamp is still honest — the sentence explains it, it does not "
        "dress it up as working")
