"""The Telegram notifier, and the two things it must never do.

It must never leak the bot token — a token is a bearer credential, and anyone
holding it can send as the bot and read everything sent to it. And it must
never raise into the trading loop: a book that stops because a messaging API
is down is a worse outcome than a message that never arrives.
"""

from __future__ import annotations

import json

import httpx
import pytest

from imperium.notify import telegram as tg

TOKEN = "123456789:AAEabcdefghijklmnopqrstuvwxyz012345678"


def _transport(handler):
    return httpx.MockTransport(handler)


def _ok(payload):
    return httpx.Response(200, json={"ok": True, "result": payload})


# ---------------------------------------------------------------- the setup


@pytest.mark.asyncio
async def test_the_token_is_checked_before_anything_is_stored():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return _ok({"id": 42, "username": "imperium_bot", "first_name": "IMPERIUM"})

    identity = await tg.identify(TOKEN, transport=_transport(handler))

    assert identity.username == "imperium_bot"
    assert "getMe" in seen["url"]


@pytest.mark.asyncio
async def test_a_bad_token_is_reported_with_what_to_do_about_it():
    def handler(request):
        return httpx.Response(401, json={"ok": False, "description": "Unauthorized"})

    with pytest.raises(tg.TelegramError) as caught:
        await tg.identify("nope", transport=_transport(handler))

    assert "rejected the bot token" in caught.value.message
    assert "BotFather" in caught.value.remedy


@pytest.mark.asyncio
async def test_the_chat_is_read_out_of_the_message_the_operator_sent():
    """The step every other integration makes people do by hand.

    Asking someone to message their own bot is both the simplest instruction
    and the proof that the account on the other end is theirs.
    """
    def handler(request):
        return _ok([
            {"update_id": 1, "message": {"chat": {"id": 555111222}}},
        ])

    chat = await tg.discover_chat(TOKEN, transport=_transport(handler))

    assert chat == "555111222"


@pytest.mark.asyncio
async def test_the_most_recent_chat_wins():
    """A bot that has been linked before carries older updates. Taking the
    first would link the previous owner's chat."""
    def handler(request):
        return _ok([
            {"update_id": 1, "message": {"chat": {"id": 111}}},
            {"update_id": 2, "message": {"chat": {"id": 222}}},
        ])

    assert await tg.discover_chat(TOKEN, transport=_transport(handler)) == "222"


@pytest.mark.asyncio
async def test_no_message_yet_says_exactly_what_to_do():
    """The one place this setup can stall, so it must not be a bare failure:
    Telegram only lets a bot see chats that spoke to it first."""
    def handler(request):
        return _ok([])

    with pytest.raises(tg.TelegramError) as caught:
        await tg.discover_chat(TOKEN, transport=_transport(handler))

    assert "no message from you has reached the bot" in caught.value.message
    assert "press Start" in caught.value.remedy


# ------------------------------------------------------------- the sending


@pytest.mark.asyncio
async def test_a_message_is_sent_with_the_chat_and_the_text():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return _ok({"message_id": 1})

    notifier = tg.Notifier(TOKEN, "555", transport=_transport(handler))
    assert await notifier.send("BUY 1 AAPL @ 100")

    assert sent[0]["chat_id"] == "555"
    assert sent[0]["text"] == "BUY 1 AAPL @ 100"
    assert notifier.sent == 1


@pytest.mark.asyncio
async def test_an_unlinked_notifier_sends_nothing_and_says_so():
    notifier = tg.Notifier("", "")
    assert not notifier.enabled
    assert await notifier.send("anything") is False


@pytest.mark.asyncio
async def test_a_telegram_outage_never_raises_into_the_caller():
    """The contract. Every caller is inside the trading loop."""
    def handler(request):
        raise httpx.ConnectError("no route to host")

    notifier = tg.Notifier(TOKEN, "555", transport=_transport(handler))

    assert await notifier.send("hello") is False
    assert notifier.failed == 1
    assert "could not reach Telegram" in notifier.last_error
    # The remedy, not just the failure. A generic exception handler would
    # record "TelegramError: ..." and drop the sentence that says what to do,
    # which is the whole difference between a log line and a diagnosis.
    assert "api.telegram.org" in notifier.last_error


@pytest.mark.asyncio
async def test_a_refusal_from_telegram_never_raises_either():
    def handler(request):
        return httpx.Response(400, json={"ok": False,
                                         "description": "chat not found"})

    notifier = tg.Notifier(TOKEN, "555", transport=_transport(handler))

    assert await notifier.send("hello") is False
    assert "chat not found" in notifier.last_error


@pytest.mark.asyncio
async def test_the_status_never_contains_the_token_or_the_chat():
    """What the panel renders. A token in a status payload is a token in every
    screenshot of the panel and in every browser's memory."""
    notifier = tg.Notifier(TOKEN, "555111222")

    status = json.dumps(notifier.status())

    assert TOKEN not in status
    assert "AAEabcdef" not in status
    assert "555111222" not in status
    assert '"linked": true' in status.lower()


# --------------------------------------------------------------- the session


@pytest.mark.asyncio
async def test_the_session_notifies_on_a_fill_and_survives_the_notifier_failing():
    """Driven through the session, because a notifier nothing calls is the
    same as no notifier -- and one that can stop the book is worse than one."""
    from imperium.session import TradingSession

    session = TradingSession()

    def handler(request):
        raise httpx.ConnectError("down")

    session.notifier = tg.Notifier(TOKEN, "555", transport=_transport(handler))

    # Must not raise, and must record the failure rather than swallow it.
    assert await session.notify("BUY 1 AAPL") is False
    assert session.notifier.failed == 1


@pytest.mark.asyncio
async def test_the_token_is_never_returned_by_the_status_endpoint():
    """The rule the whole credential design rests on, applied to this token."""
    from fastapi.testclient import TestClient

    from imperium.server.app import create_app

    with TestClient(create_app()) as client:
        body = client.get("/api/telegram").text

    assert TOKEN not in body
    assert "token" not in json.loads(body)


@pytest.mark.asyncio
async def test_the_link_survives_a_restart():
    """This program is meant to run for weeks and to restart itself when it
    crashes. A notifier that silently unlinks on restart is worse than none:
    the silence afterwards reads as "nothing happened"."""
    from fastapi.testclient import TestClient

    from imperium.security.credentials import CredentialStore
    from imperium.server import app as app_mod

    store = CredentialStore()
    store.put_token(app_mod.TELEGRAM_NAME, app_mod.TELEGRAM_VENUE, TOKEN,
                    note="@imperium_bot", chat="555111222")
    try:
        # A fresh process: the app loads the store from disk at startup.
        with TestClient(app_mod.create_app()) as client:
            status = client.get("/api/telegram").json()
    finally:
        store.remove(app_mod.TELEGRAM_NAME)
        store.save()

    assert status["linked"], (
        "the Telegram link was not restored from the credential file, so it "
        "is lost every time the terminal restarts")
    assert status["enabled"]
