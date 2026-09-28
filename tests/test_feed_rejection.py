"""What the terminal says when the data feed refuses to connect.

This file exists because of a real operator report. The handshake printed one
guessed cause for every rejection Alpaca could send -- "the key must be valid
and the plan must include the 'iex' feed" -- and an operator hit it, checked
their key, checked their plan, and found nothing wrong with either, because
nothing was. The message named two things it had not verified and omitted the
one thing the venue had actually said.

A diagnostic that guesses is worse than no diagnostic. It does not merely fail
to help; it spends the operator's time on the wrong thing and makes them
distrust the next message too.
"""

from __future__ import annotations

import pytest

from imperium.venues.alpaca.feed import (
    CONNECTION_LIMIT_BACKOFF, STREAM_ERRORS, FeedRejected,
)


def test_a_connection_limit_is_not_reported_as_a_key_or_plan_problem():
    """THE test in this file -- the exact regression the operator hit.

    406 means something else already holds the one market data connection the
    account is allowed. The key is fine and the plan is fine. Sending someone
    to regenerate a working key is worse than saying nothing.
    """
    rejected = FeedRejected(406, "connection limit exceeded")
    said = (rejected.operator_text() + " " + rejected.remedy).lower()

    assert "already has a live market data connection" in said
    assert "one at a time" in said
    # The remedy must name what to actually close.
    assert "imperium" in said and "task manager" in said
    # And must not send them after the two things that are not the problem.
    assert "regenerat" not in rejected.cause.lower()
    assert "plan must include" not in said


def test_the_free_feed_is_not_described_as_something_to_buy():
    """A factual correction the old message had backwards. IEX is included
    with every Alpaca account, paper and live; only SIP needs a subscription.
    Telling someone their plan lacks the free feed sends them to a billing
    page to fix a problem that is not there."""
    _, remedy = STREAM_ERRORS[409]
    assert "'iex' feed is free" in remedy
    assert "'sip' feed needs a paid" in remedy


def test_a_bad_key_says_the_key_and_says_paper_keys_are_fine():
    """402 is the case the old message described -- and it still has to say
    that a paper key is not the fault, because that is the first thing an
    operator suspects."""
    rejected = FeedRejected(402, "auth failed")
    said = (rejected.operator_text() + " " + rejected.remedy).lower()
    assert "not accepted" in said
    assert "paper keys stream market data" in said
    assert "shown only once" in said


@pytest.mark.parametrize("code", sorted(STREAM_ERRORS))
def test_every_known_code_has_a_cause_and_something_to_do(code):
    """Prevents a code being added with a cause and no remedy, which is how
    this file's original bug would come back one entry at a time."""
    cause, remedy = STREAM_ERRORS[code]
    assert cause and not cause.endswith(".")
    assert remedy, f"code {code} names a cause but nothing to do about it"


def test_an_unknown_code_admits_it_does_not_know():
    """Prevents the failure mode this whole file is about: inventing a cause
    for a code nobody has seen. Saying "the venue refused it, here is what it
    said" is the honest answer and is more useful than a guess."""
    rejected = FeedRejected(4242, "something new")
    assert rejected.cause == "the data feed refused the connection"
    assert rejected.remedy == ""
    assert "something new" in rejected.operator_text()
    assert "4242" in rejected.operator_text()


def test_the_venues_own_words_are_always_quoted():
    """Prevents a paraphrase replacing the thing the operator needs to search
    for or quote to Alpaca's support."""
    rejected = FeedRejected(408, "v2 not enabled")
    assert '"v2 not enabled"' in rejected.operator_text()


def test_the_operator_never_sees_a_python_exception_class():
    """Prevents "RuntimeError: ..." on a trading terminal. It tells an
    operator nothing except that something inside broke, which is both
    unhelpful and, for a venue saying no, untrue."""
    rejected = FeedRejected(406, "connection limit exceeded")
    text = rejected.operator_text()
    for leak in ("RuntimeError", "Exception", "Traceback", "FeedRejected"):
        assert leak not in text


def test_a_connection_limit_backs_off_far_longer_than_a_dropped_socket():
    """Prevents a tight retry loop against a limit that cannot clear by
    asking again. It buries the one line explaining what to close under a wall
    of identical errors, and looks like a misbehaving client from the venue's
    side."""
    assert CONNECTION_LIMIT_BACKOFF >= 30.0


# -- when the thing holding the connection is us -------------------------

def _feed():
    from imperium.telemetry.streams import TelemetryHub
    from imperium.venues import registry
    from imperium.venues.alpaca.feed import MarketFeed
    return MarketFeed(registry.get(registry.DEFAULT_VENUE), TelemetryHub())


def test_our_own_second_stream_losing_the_race_is_named_as_such():
    """Prevents the same wild goose chase in a new costume.

    Alpaca serves equities and crypto from different endpoints, so this program
    opens two sockets. Where the account allows one connection, the second of
    our own two is refused by the first -- and "close the other copy of
    IMPERIUM" is then useless advice, because there is no other copy. The
    operator would go hunting a process that does not exist.
    """
    from imperium.venues.assets import AssetClass

    feed = _feed()
    feed._live = {AssetClass.US_EQUITY: True, AssetClass.CRYPTO: False}
    feed._report_self_contention(AssetClass.CRYPTO)

    said = (feed.last_error + " " + feed.last_remedy).lower()
    assert "imperium's own us_equity stream is using it" in said
    assert "do not go looking for a second copy" in said
    # And it must say what is actually lost, which is less than it sounds.
    assert "daily bars" in said and "snapshot sweep" in said


def test_a_first_stream_refused_with_nothing_of_ours_live_is_not_blamed_on_us():
    """Prevents the opposite error: telling an operator the program is
    competing with itself when it holds no connection at all, which would hide
    a genuine third-party or stale-connection 406."""
    from imperium.venues.assets import AssetClass

    feed = _feed()
    feed._live = {AssetClass.US_EQUITY: False, AssetClass.CRYPTO: False}
    feed.last_error = "original"
    feed.last_remedy = "original remedy"
    feed._report_self_contention(AssetClass.US_EQUITY)
    assert feed.last_error == "original"
    assert feed.last_remedy == "original remedy"
