"""The greeting on Start.

Small feature, and the tests are mostly about honesty rather than mechanics: a
terminal whose whole claim is that it reports what it measured must not open
by telling the operator something false, and trading quotations are passed
around in a state of near-total attribution collapse.
"""

from __future__ import annotations

import datetime as dt
import random
import re

import pytest

from imperium.notify import greeting


def _client():
    from fastapi.testclient import TestClient

    from imperium.server.app import create_app
    from imperium.session import TradingSession

    return TestClient(create_app(TradingSession()))


# -- the quotes themselves --------------------------------------------------


def test_every_quote_names_who_said_it():
    """An unattributed quote is a sentence the terminal is claiming as its own.

    Which, for a line somebody else wrote and is famous for, is the one thing
    this file must not do.
    """
    for quote in greeting.QUOTES:
        assert quote.who.strip(), f"no attribution for {quote.text!r}"


def test_the_commonly_misattributed_one_is_corrected():
    """The most famous line here belongs to A. Gary Shilling.

    It is given to Keynes almost everywhere, including by people who should
    know better, and he never wrote it. Repeating that would be the terminal
    passing on a thing it had not checked, which is the habit the rest of this
    program exists to avoid.
    """
    irrational = [q for q in greeting.QUOTES if "irrational" in q.text]
    assert len(irrational) == 1
    quote = irrational[0]
    assert "Shilling" in quote.who
    assert "Keynes" in quote.note, (
        "the quote is corrected but does not say what it is being corrected "
        "from, which helps nobody")


def test_no_quote_is_written_in_a_way_a_voice_would_mangle():
    """These are read aloud before they are read.

    A numeral becomes "one nine eight seven" in some voices, an ampersand
    becomes "ampersand", and an em dash lands as a pause in the wrong place.
    """
    for quote in greeting.QUOTES:
        spoken = quote.spoken()
        for glyph in ("&", "%", "$", "—", "–", "..."):
            assert glyph not in spoken, f"{glyph!r} in {spoken!r}"
        assert not re.search(r"\d", spoken), f"a numeral in {spoken!r}"


def test_there_are_enough_quotes_to_not_feel_like_a_loop():
    assert len(greeting.QUOTES) >= 12
    assert len({q.text for q in greeting.QUOTES}) == len(greeting.QUOTES), \
        "a duplicated quote makes the rotation shorter than it looks"


# -- picking ----------------------------------------------------------------


def test_it_never_says_the_same_thing_twice_in_a_row():
    """The only repeat that reads as a bug.

    A quote coming round again next week is a rotation. The same one twice in
    a row is a program that is not really choosing.
    """
    rng = random.Random(1234)
    previous = -1
    for _ in range(400):
        index = greeting.pick(previous, rng)
        assert index != previous
        previous = index


def test_over_many_draws_it_uses_the_whole_rotation():
    rng = random.Random(99)
    seen, previous = set(), -1
    for _ in range(2000):
        previous = greeting.pick(previous, rng)
        seen.add(previous)
    assert seen == set(range(len(greeting.QUOTES))), (
        f"{len(greeting.QUOTES) - len(seen)} quotes are unreachable")


def test_a_single_quote_would_not_hang():
    """pick() loops until it draws something new, so a one-entry rotation is
    the obvious way to write an infinite loop."""
    original = greeting.QUOTES
    try:
        greeting.QUOTES = (original[0],)
        assert greeting.pick(0) == 0
    finally:
        greeting.QUOTES = original


# -- the opening ------------------------------------------------------------


def test_it_greets_the_operator_by_name():
    opening = greeting.opening()
    assert "Mr Gininda" in opening.spoken
    assert "Mr Gininda" in opening.hello


def test_the_name_can_be_changed(monkeypatch):
    monkeypatch.setenv("IMPERIUM_OPERATOR", "Captain Ahab")
    assert "Captain Ahab" in greeting.opening().spoken


def test_an_empty_name_falls_back_rather_than_greeting_nobody(monkeypatch):
    monkeypatch.setenv("IMPERIUM_OPERATOR", "   ")
    assert greeting.DEFAULT_OPERATOR in greeting.opening().spoken


@pytest.mark.parametrize("hour,expected", [
    (7, "Good morning"), (13, "Good afternoon"), (21, "Good evening")])
def test_the_greeting_follows_the_operators_own_clock(hour, expected):
    at = dt.datetime(2026, 9, 16, hour, 30)
    assert greeting.opening(now=at).spoken.startswith(expected)


def test_the_spoken_and_written_forms_are_the_same_quote():
    """Two draws would have the terminal say one thing and print another.

    Which is a small bug that reads as the program not knowing what it is
    doing -- and it is exactly what the first version of this did, because
    speaking and printing were two functions that each picked.
    """
    for seed in range(50):
        opening = greeting.opening(rng=random.Random(seed))
        # The written quote keeps its full stop; the spoken form runs the
        # attribution on. Compare the words rather than the punctuation.
        assert opening.quote.rstrip(".") in opening.spoken
        if opening.who:
            assert opening.who in opening.spoken


@pytest.mark.parametrize("mode,says", [
    ("live", "live"), ("paper", "paper"), ("dry_run", "No orders")])
def test_the_mode_is_named_out_loud(mode, says):
    """The one moment it matters most.

    "Running live" and "running in dry run" are one glance apart in the header
    and a completely different fact about the next hour.
    """
    assert says in greeting.opening(mode=mode).spoken


def test_an_unknown_mode_does_not_invent_one():
    spoken = greeting.opening(mode="something-new").spoken
    assert "something-new" not in spoken
    assert "Mr Gininda" in spoken


# -- the wiring -------------------------------------------------------------


def test_starting_the_session_returns_a_greeting():
    with _client() as client:
        reply = client.post("/api/session/start")
        assert reply.status_code == 200
        opening = reply.json()["opening"]
        assert opening is not None
        assert "Mr Gininda" in opening["hello"]
        assert opening["quote"]
        assert opening["who"]
        client.post("/api/session/stop")


def test_stopping_and_starting_again_draws_a_different_quote():
    with _client() as client:
        first = client.post("/api/session/start").json()["opening"]["quote"]
        client.post("/api/session/stop")
        second = client.post("/api/session/start").json()["opening"]["quote"]
        client.post("/api/session/stop")
        assert first != second


def test_the_greeting_cannot_be_spoken_before_the_session_starts():
    """409 rather than a drawn-on-demand greeting: the endpoint speaks what is
    already on screen, and before Start there is nothing on screen."""
    with _client() as client:
        reply = client.post("/api/voice/greeting")
        assert reply.status_code == 409
        assert "not been started" in reply.json()["detail"]


def test_speaking_the_greeting_needs_a_voice_and_says_so():
    with _client() as client:
        client.post("/api/session/start")
        reply = client.post("/api/voice/greeting")
        assert reply.status_code == 400
        assert "ElevenLabs" in reply.json()["detail"]
        client.post("/api/session/stop")


def test_the_endpoint_speaks_the_greeting_already_on_screen(monkeypatch):
    """Not a fresh draw, and not arbitrary text.

    The browser asks for *the* greeting and the server decides what that is,
    so a page on this machine cannot run up a bill a character at a time, and
    the spoken line is guaranteed to be the one the operator is reading.
    """
    from fastapi.testclient import TestClient

    from imperium.server.app import create_app
    from imperium.session import TradingSession

    sent: list[str] = []
    session = TradingSession()

    async def capture(text):
        sent.append(text)
        return b"audio"

    session.speaker.speak = capture
    monkeypatch.setattr(type(session.speaker), "enabled",
                        property(lambda self: True))

    with TestClient(create_app(session)) as client:
        shown = client.post("/api/session/start").json()["opening"]
        assert client.post("/api/voice/greeting").status_code == 200
        client.post("/api/session/stop")

    assert len(sent) == 1
    assert sent[0] == shown["spoken"]
    assert shown["quote"].rstrip(".") in sent[0]


def test_a_start_that_is_already_running_does_not_redraw():
    """Start is disabled while running, but the endpoint is reachable anyway,
    and a second press should not swap the quote out from under the operator
    mid-read."""
    with _client() as client:
        first = client.post("/api/session/start").json()["opening"]["quote"]
        second = client.post("/api/session/start").json()["opening"]["quote"]
        client.post("/api/session/stop")
        assert first == second
