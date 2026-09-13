"""Speech, and the two things it must never do.

It must never leak the ElevenLabs key — a bearer credential billed per
character, so anyone holding it can spend the account's quota. And it must
never raise into the trading loop or the UI handler: a terminal that stops
because a text-to-speech API is down is a far worse outcome than a briefing
nobody hears.
"""

from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest

from imperium.notify import briefing, voice

KEY = "sk_0123456789abcdef0123456789abcdef0123456789abcdef"


def _transport(handler):
    return httpx.MockTransport(handler)


def _voices(*names):
    return httpx.Response(200, json={"voices": [
        {"voice_id": f"v{i}", "name": n} for i, n in enumerate(names)]})


# ------------------------------------------------------------------ the key


@pytest.mark.asyncio
async def test_the_key_is_checked_by_listing_the_accounts_own_voices():
    """Listing rather than asking for a hard-coded id: voice ids differ per
    account, so a default baked into the source works on one machine only."""
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("xi-api-key")
        return _voices("Rachel", "Adam")

    found = await voice.list_voices(KEY, transport=_transport(handler))

    assert [v.name for v in found] == ["Rachel", "Adam"]
    assert seen["url"].endswith("/v1/voices")
    assert seen["key"] == KEY


@pytest.mark.asyncio
async def test_a_rejected_key_says_where_to_get_a_new_one():
    def handler(request):
        return httpx.Response(401, json={"detail": "Unauthorized"})

    with pytest.raises(voice.VoiceError) as caught:
        await voice.list_voices("nope", transport=_transport(handler))

    assert "rejected the API key" in caught.value.message
    assert "dashboard" in caught.value.remedy


@pytest.mark.asyncio
async def test_an_account_with_no_voices_is_reported_rather_than_empty():
    def handler(request):
        return httpx.Response(200, json={"voices": []})

    with pytest.raises(voice.VoiceError) as caught:
        await voice.list_voices(KEY, transport=_transport(handler))

    assert "no voices" in caught.value.message


# ------------------------------------------------------------- the synthesis


@pytest.mark.asyncio
async def test_speech_is_requested_with_the_voice_and_the_script():
    sent = {}

    def handler(request):
        sent["url"] = str(request.url)
        sent["key"] = request.headers.get("xi-api-key")
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"ID3fake-mp3-bytes",
                              headers={"content-type": "audio/mpeg"})

    audio = await voice.synthesize(KEY, "v0", "Good morning.",
                                   transport=_transport(handler))

    assert audio == b"ID3fake-mp3-bytes"
    assert sent["url"].endswith("/v1/text-to-speech/v0")
    assert sent["key"] == KEY
    assert sent["body"]["text"] == "Good morning."
    assert sent["body"]["model_id"] == voice.DEFAULT_MODEL


@pytest.mark.asyncio
async def test_an_overlong_script_is_cut_at_a_sentence_not_mid_word():
    """ElevenLabs bills per character, so an unbounded briefing is an
    unbounded bill — and one cut mid-word sounds like the program failed
    rather than like a limit."""
    sent = {}

    def handler(request):
        sent["text"] = json.loads(request.content)["text"]
        return httpx.Response(200, content=b"mp3")

    script = ("This is a sentence. " * 400)          # well past the cap
    await voice.synthesize(KEY, "v0", script, transport=_transport(handler))

    assert len(sent["text"]) <= voice.MAX_CHARACTERS
    assert sent["text"].endswith("."), sent["text"][-40:]


@pytest.mark.asyncio
async def test_an_empty_script_is_refused_before_a_request_is_made():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, content=b"mp3")

    with pytest.raises(voice.VoiceError):
        await voice.synthesize(KEY, "v0", "   ", transport=_transport(handler))
    assert not calls, "an empty script still cost a request"


@pytest.mark.asyncio
async def test_a_spent_quota_is_reported_as_a_quota_not_a_mystery():
    def handler(request):
        return httpx.Response(429, json={"detail": "quota exceeded"})

    with pytest.raises(voice.VoiceError) as caught:
        await voice.synthesize(KEY, "v0", "hello",
                               transport=_transport(handler))

    assert "quota" in caught.value.message
    assert "quota" in caught.value.remedy


# -------------------------------------------------------------- the speaker


@pytest.mark.asyncio
async def test_the_speaker_never_raises_into_its_caller():
    """The contract. Every caller is a UI handler or the trading loop."""
    def handler(request):
        raise httpx.ConnectError("no route to host")

    speaker = voice.Speaker(KEY, "v0", "Rachel", transport=_transport(handler))

    assert await speaker.speak("hello") is None
    assert speaker.failed == 1
    assert "could not reach ElevenLabs" in speaker.last_error
    # The remedy, not just the failure. A generic handler would record
    # "VoiceError: ..." and drop the sentence saying what to do about it,
    # which is the difference between a log line and a diagnosis.
    assert "api.elevenlabs.io" in speaker.last_error


@pytest.mark.asyncio
async def test_an_unconfigured_speaker_says_nothing_and_costs_nothing():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, content=b"mp3")

    speaker = voice.Speaker("", "", transport=_transport(handler))

    assert not speaker.enabled
    assert await speaker.speak("hello") is None
    assert not calls


def test_the_status_never_contains_the_key():
    """What the panel renders. A key in a status payload is a key in every
    screenshot of the panel and in every browser's memory."""
    speaker = voice.Speaker(KEY, "v0", "Rachel")

    status = json.dumps(speaker.status())

    assert KEY not in status
    assert "sk_" not in status
    assert "v0" not in status


@pytest.mark.asyncio
async def test_the_key_is_never_returned_by_any_voice_endpoint():
    """The rule the whole credential design rests on, applied to this key."""
    from fastapi.testclient import TestClient

    from imperium.security.credentials import CredentialStore
    from imperium.server import app as app_mod

    store = CredentialStore()
    store.put_token(app_mod.VOICE_NAME, app_mod.VOICE_VENUE, KEY,
                    note="Rachel", chat="v0")
    try:
        with TestClient(app_mod.create_app()) as client:
            status = client.get("/api/voice").text
            script = client.get("/api/voice/script").text
    finally:
        store.remove(app_mod.VOICE_NAME)
        store.save()

    assert KEY not in status and KEY not in script
    assert "api_key" not in json.loads(status)


@pytest.mark.asyncio
async def test_the_voice_survives_a_restart():
    """This program restarts itself. A voice that silently unlinks on restart
    leaves the operator pressing a dead button."""
    from fastapi.testclient import TestClient

    from imperium.security.credentials import CredentialStore
    from imperium.server import app as app_mod

    store = CredentialStore()
    store.put_token(app_mod.VOICE_NAME, app_mod.VOICE_VENUE, KEY,
                    note="Rachel", chat="v0")
    try:
        with TestClient(app_mod.create_app()) as client:
            status = client.get("/api/voice").json()
    finally:
        store.remove(app_mod.VOICE_NAME)
        store.save()

    assert status["enabled"], "the voice was not restored from the store"
    assert status["voice"] == "Rachel"


# --------------------------------------------------------------- the script
#
# What it says is the feature. The synthesis above is plumbing.


def _snapshot(**overrides):
    base = {
        "running": True, "simulated": False, "mode": "paper",
        "equity": 70.0, "unrealised_pnl": 0.0, "realised_pnl": 0.0,
        "account": {"currency": "USD"},
        "positions": [], "watchlist": [], "limits": {},
        "market": {"is_open": False}, "universe_scan": {},
        "blockers": {}, "cross_section": {}, "feed": {},
    }
    base.update(overrides)
    return base


def test_it_greets_by_the_operators_own_clock():
    morning = dt.datetime(2026, 9, 13, 7, 0)
    evening = dt.datetime(2026, 9, 13, 21, 0)

    assert briefing.greeting(morning) == "Good morning"
    assert briefing.greeting(evening) == "Good evening"
    assert briefing.build(_snapshot(), now=morning).startswith(
        "Good morning. This is IMPERIUM.")


def test_money_is_said_the_way_a_person_says_it():
    """"$70.00" read literally is "dollar seventy point zero zero"."""
    assert briefing.say_money(70.0) == "70 dollars"
    assert briefing.say_money(1.0) == "1 dollar"
    assert briefing.say_money(70.55) == "70.55 dollars"
    assert "$" not in briefing.build(_snapshot(equity=70.0))


def test_a_ticker_is_spelled_but_its_digits_are_not():
    """"AAPL" unspaced is read as a word, and the word is not the company.
    But "CO007" spaced becomes "C O zero zero seven", which is not how anyone
    says a number."""
    assert briefing.say_ticker("AAPL") == "A A P L"
    assert briefing.say_ticker("BTC/USD") == "B T C against U S D"
    assert briefing.say_ticker("CO007/USD") == "C O 007 against U S D"


def test_the_script_carries_no_characters_a_voice_would_stumble_over():
    """Symbols that read as noise: a currency sign, a percent sign, an em
    dash, a slash in a pair."""
    script = briefing.build(_snapshot(
        equity=70.0,
        positions=[{"symbol": "BTC/USD"}],
        watchlist=[{"symbol": "BTC/USD", "verdict": "trading"}],
        limits={"max_concurrent_positions": 2},
    ))
    for bad in ("$", "%", "—", "BTC/USD", "bp"):
        assert bad not in script, f"{bad!r} survives into the spoken script"


def test_it_says_what_was_admitted_and_why_each_one_holds_its_slot():
    """Asked for directly. Admission is the allocation of a fixed number of
    slots, and the two reasons a symbol holds one are different facts."""
    script = briefing.build(_snapshot(
        positions=[{"symbol": "AAPL"}],
        watchlist=[
            {"symbol": "AAPL", "verdict": "trading"},
            {"symbol": "MSFT", "verdict": "trading"},
            {"symbol": "NVDA", "verdict": "rejected"},
        ],
        limits={"max_concurrent_positions": 2},
    ))

    assert "2 symbols hold a position slot, out of 2" in script
    assert "A A P L keeps a slot because it is already holding a position" in script
    assert "never displaced to chase a better score" in script
    assert "M S F T was admitted on score" in script
    # A rejected symbol is not an admission and must not be named as one.
    assert "N V D A" not in script


def test_an_empty_book_says_so_plainly_rather_than_trailing_off():
    """The normal state of this program most days. "Nothing is being traded"
    followed by the reason is the most useful thing it can say."""
    script = briefing.build(_snapshot(
        limits={"max_concurrent_positions": 2},
        blockers={"trading": 0, "counts": [
            {"blocker": "costs", "symbols": 120},
            {"blocker": "warming up", "symbols": 20},
        ]},
    ))

    assert "The book is flat" in script
    assert "Nothing holds one of the 2 position slots" in script
    assert "Nothing is being traded." in script
    # The labels are written for a table column and must be turned into
    # clauses, or they land as fragments when read aloud.
    assert "120 because the edge does not cover the cost of trading" in script
    assert "20 because they are still warming up" in script


def test_a_halt_is_said_early_because_nothing_below_it_matters():
    script = briefing.build(_snapshot(
        halted=True, halt_reason="daily loss limit reached",
        blockers={"trading": 0, "counts": [{"blocker": "costs", "symbols": 5}]},
    ))

    assert "The book is halted" in script
    assert script.index("halted") < script.index("Nothing is being traded")


def test_a_simulated_session_is_never_described_as_really_trading():
    """The single most important sentence in the briefing to get right."""
    simulated = briefing.build(_snapshot(simulated=True))
    real = briefing.build(_snapshot(simulated=False))

    assert "simulated in this process and never reach the venue" in simulated
    assert "really sent to the venue" not in simulated
    assert "really sent to the venue" in real


def test_a_stopped_session_says_so_before_anything_else():
    script = briefing.build(_snapshot(running=False))
    assert "The session is stopped" in script
    assert "Running in" not in script


def test_the_briefing_stays_short_enough_to_listen_to():
    """A spoken summary that runs past a minute is one nobody waits through."""
    script = briefing.build(_snapshot(
        positions=[{"symbol": f"SYM{i}"} for i in range(20)],
        watchlist=[{"symbol": f"SYM{i}", "verdict": "trading"}
                   for i in range(20)],
        limits={"max_concurrent_positions": 20},
        universe_scan={"ranked": 11005, "size": 150, "cohort_passes": 3},
        blockers={"trading": 20, "counts": [
            {"blocker": "costs", "symbols": 120}]},
    ))

    assert len(script) < voice.MAX_CHARACTERS, len(script)
    # The tail of a long list is counted, not read.
    assert "and 15 others" in script


@pytest.mark.asyncio
async def test_the_session_builds_its_briefing_from_its_own_snapshot():
    """Driven through the session, so the spoken version cannot drift from
    the screen: one description of the state, two ways of presenting it."""
    from imperium.session import TradingSession

    session = TradingSession()
    script = session.briefing()

    assert script.startswith("Good ")
    assert "IMPERIUM" in script
