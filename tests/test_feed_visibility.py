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


# ---------------------------------------------------------------------------
# The health score, which had the same defect one layer down: it scored
# expected silence as failure and sat flat and red all night.
# ---------------------------------------------------------------------------


def test_a_shut_market_does_not_drag_the_health_score_down():
    """Reported as "the health scanner is not functioning".

    It was functioning. ``data`` and ``link`` are 55% of the weight between
    them, and with the equity market shut and nothing to stream both scored
    zero -- so a program doing exactly the right thing was pinned at 45%, red
    and flat, every night. A gauge that reads failure whenever the market is
    closed measures the clock, not health, and an operator learns to ignore it
    within a day.
    """
    session = _session(open_market=False)
    session.running = True
    session.universe = ["AAPL", "MSFT"]          # no crypto: nothing to stream
    session.feed.connected = False
    # Held steady so this measures the market's hours and nothing else. An
    # unattached key is its own (real) deduction, and would otherwise be the
    # thing the assertion below is actually reading.
    session.lamps.venue = "ok"

    health = session._health_score()

    assert "data" in health["not_applicable"]
    assert "link" in health["not_applicable"]
    assert health["score"] > 0.8, (
        f"a correctly idle session still scores {health['score']} — the gauge "
        f"is reporting the market's hours as the program's health")


def test_crypto_in_the_cohort_means_silence_is_still_a_fault():
    """The distinction that makes the rule safe rather than an excuse.

    Crypto trades around the clock, so a cohort holding any means the feed
    should be delivering whatever the equity clock says. Excusing silence here
    would hide a dead socket every evening -- and since crypto is now pinned
    resident, that is most evenings.
    """
    session = _session(open_market=False)
    session.running = True
    session.universe = ["AAPL", "BTC/USD"]
    session.feed.connected = False

    health = session._health_score()

    assert health["not_applicable"] == [], (
        "the cohort holds crypto, which trades all night — a silent feed is a "
        "real fault and must count against the score")
    assert health["score"] < 0.6
    assert "not a fault" not in session._feed_reason()


def test_the_two_readouts_agree_about_whether_silence_is_expected():
    """The lamp's sentence and the health score must not contradict each other.

    They were computed from different tests -- one asked whether *everything*
    was crypto, the other whether *anything* was -- so a mixed cohort would
    have had the panel calling the same silence expected and unhealthy at once.
    """
    for open_market, universe in ((False, ["AAPL"]), (False, ["AAPL", "BTC/USD"]),
                                  (True, ["AAPL"])):
        session = _session(open_market=open_market)
        session.running = True
        session.universe = list(universe)
        session.feed.connected = True
        session.feed.symbols = list(universe)

        excused_by_lamp = "not a fault" in session._feed_reason()
        excused_by_score = bool(session._health_score()["not_applicable"])
        assert excused_by_lamp == excused_by_score, (
            f"open={open_market} universe={universe}: the lamp and the health "
            f"score disagree about whether this silence is expected")


def test_a_stopped_session_is_not_scored_as_unhealthy():
    """Nothing is subscribed because nothing was asked for. That is the Start
    button not having been pressed, not a fault to report."""
    session = _session(open_market=True)
    session.running = False
    session.lamps.venue = "ok"

    assert session._health_score()["score"] > 0.8


# ---------------------------------------------------------------------------
# Sentences an operator actually reads, in a clock they actually keep.
# ---------------------------------------------------------------------------


def test_the_closed_market_sentence_is_not_two_sentences_spliced_together():
    """A real defect, and one only reading the output in place would catch.

    ``describe`` returns a whole clause -- "market closed, opens Fri 13:30
    UTC" -- and it was embedded mid-sentence, producing "the equity market is
    shut, which opens market closed, opens Fri 13:30 UTC". Every unit test
    passed: they checked for the phrases around it, and none read the finished
    line.
    """
    session = _session(open_market=False)
    session.running = True
    session.universe = ["AAPL", "MSFT"]
    session.feed.connected = True
    session.feed.symbols = list(session.universe)

    reason = session._feed_reason()

    assert "opens market closed" not in reason, reason
    assert reason.count("market") <= 2, f"the clause is spliced twice: {reason}"
    assert "UTC" in reason, "the operator still needs to know when"


def test_the_clock_offers_a_bare_time_for_composing_sentences():
    import datetime as dt

    clock = MarketClock(is_open=False,
                        next_open=dt.datetime(2026, 9, 11, 13, 30, tzinfo=UTC))
    assert clock.next_change_text() == "Fri 13:30 UTC"
    # And the whole-clause form is still available for standalone display.
    assert clock.describe().startswith("market closed")


@pytest.mark.asyncio
async def test_the_per_symbol_note_does_not_repeat_the_timestamp():
    """The note is stamped onto every equity row the market guard refuses. A
    hundred and fifty copies of the same UTC time is noise, and the header
    already carries the clock with a countdown.

    Driven through the clock refresh rather than assigned here: a note this
    test composes itself proves only that this test can compose one.
    """
    from imperium.venues.alpaca.client import AlpacaClient
    from mock_venue import KEY, SECRET, MockVenue

    venue = MockVenue()
    venue.market_open = False
    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    try:
        await session.refresh_clock()
    finally:
        await session.detach_client()

    note = session.allocator.market_note
    assert note, "the note must still say something"
    assert "UTC" not in note, f"the timestamp is repeated per symbol: {note}"
    assert "crypto continues" in note
