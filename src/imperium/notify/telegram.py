"""Telegram notifications, with the shortest setup that is still honest.

**The setup problem.** Every Telegram integration asks for two things: a bot
token and a chat id. The token you get from @BotFather in thirty seconds. The
chat id is where people give up -- the usual instructions involve messaging a
second bot, or calling ``getUpdates`` by hand in a browser and reading a JSON
blob for a number.

So this does not ask for it. Paste the token, send the bot any message from
your own Telegram, and press Link: the program reads the chat id out of the
update itself. Two steps, no numbers to copy, and the message you send is the
proof that the account on the other end is yours.

**Where the token lives.** In the same owner-only credential file as the venue
keys, behind the same permission checks, and masked by the same code on the way
out. A bot token is a bearer credential -- anyone holding it can send messages
as your bot and read everything sent to it -- so it gets the treatment the
venue keys get rather than a second, less careful store invented for it.

**What it sends.** Fills, halts, going live, and the loop dying. Not refusals:
this program refuses thousands of times an hour by design, and a notifier that
relays those is one the operator mutes within a day -- after which it cannot
tell them the one thing that mattered.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger("imperium.telegram")

API = "https://api.telegram.org"

#: How long to wait for Telegram. Short: this is a notifier, and a notifier
#: that blocks the trading loop is worse than no notifier.
TIMEOUT = 10.0

#: The least this will wait between messages, in seconds.
#:
#: Telegram's documented limit is about thirty messages a second, which this
#: will never approach. The bound exists for the failure where something in
#: the loop starts emitting the same event repeatedly: the notifier must not
#: turn a bug into a flood the operator has to mute.
MIN_INTERVAL = 1.0


class TelegramError(Exception):
    """A Telegram call that failed, with something an operator can act on."""

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    def operator_text(self) -> str:
        return f"{self.message} — {self.remedy}" if self.remedy else self.message


@dataclass(frozen=True)
class BotIdentity:
    """Who the token belongs to. Shown so the operator can see they pasted the
    right one before anything is stored."""

    id: int
    username: str
    name: str


async def _call(token: str, method: str, payload: dict | None = None,
                *, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    """One Telegram API call. Failures are returned as TelegramError, never
    raised as a transport exception into the trading loop."""
    url = f"{API}/bot{token}/{method}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT,
                                     transport=transport) as client:
            response = await client.post(url, json=payload or {})
    except httpx.HTTPError as exc:
        raise TelegramError(
            f"could not reach Telegram ({type(exc).__name__})",
            "Check the machine's internet connection. Telegram is reached "
            "over HTTPS on api.telegram.org.") from None

    if response.status_code == 401:
        raise TelegramError(
            "Telegram rejected the bot token",
            "Copy the token from @BotFather again — it looks like "
            "123456789:AAE... and is invalid the moment it is regenerated.")
    try:
        body = response.json()
    except ValueError:
        raise TelegramError(
            f"Telegram returned something that is not JSON "
            f"(HTTP {response.status_code})", "") from None
    if not body.get("ok"):
        raise TelegramError(
            body.get("description", "Telegram refused the request"),
            f"HTTP {response.status_code}")
    return body.get("result") or {}


async def identify(token: str, *,
                   transport: httpx.AsyncBaseTransport | None = None
                   ) -> BotIdentity:
    """Confirm the token works, and say which bot it belongs to."""
    result = await _call(token, "getMe", transport=transport)
    return BotIdentity(id=int(result.get("id", 0)),
                       username=str(result.get("username", "")),
                       name=str(result.get("first_name", "")))


async def discover_chat(token: str, *,
                        transport: httpx.AsyncBaseTransport | None = None
                        ) -> str:
    """Find the chat to send to, from a message the operator just sent.

    This is the step every other integration makes the operator do by hand.
    The most recent update carries the chat id; asking the operator to message
    their own bot is both the simplest instruction and the proof that the
    account is theirs.
    """
    updates = await _call(token, "getUpdates", {"limit": 10, "timeout": 0},
                          transport=transport)
    if not isinstance(updates, list):
        updates = []
    for update in reversed(updates):
        for key in ("message", "edited_message", "channel_post"):
            chat = (update.get(key) or {}).get("chat") or {}
            if chat.get("id") is not None:
                return str(chat["id"])
    raise TelegramError(
        "no message from you has reached the bot yet",
        "Open Telegram, find the bot by its @username, press Start or send it "
        "any message, then press Link again. Telegram only lets a bot see "
        "chats that spoke to it first.")


class Notifier:
    """Sends the few things worth interrupting someone for.

    Holds no thread and no background task: each send is awaited by the caller
    and every failure is swallowed into a log line. A notifier that can raise
    into the trading loop is a notifier that can stop the book.
    """

    def __init__(self, token: str = "", chat_id: str = "", *,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.token = token
        self.chat_id = chat_id
        self.transport = transport
        self.enabled = bool(token and chat_id)
        self.sent = 0
        self.failed = 0
        self.last_error = ""
        self._last_sent_at = 0.0
        self._lock = asyncio.Lock()

    def configure(self, token: str, chat_id: str) -> None:
        self.token, self.chat_id = token, chat_id
        self.enabled = bool(token and chat_id)

    async def send(self, text: str) -> bool:
        """Best effort. Returns whether it went; never raises."""
        if not self.enabled:
            return False
        async with self._lock:
            loop = asyncio.get_running_loop()
            wait = MIN_INTERVAL - (loop.time() - self._last_sent_at)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                await _call(self.token, "sendMessage",
                            {"chat_id": self.chat_id, "text": text,
                             "disable_web_page_preview": True},
                            transport=self.transport)
            except TelegramError as exc:
                # Recorded, not raised. The book keeps running when the
                # notifier cannot reach Telegram; that is the whole contract.
                self.failed += 1
                self.last_error = exc.operator_text()
                log.warning("telegram send failed: %s", self.last_error)
                return False
            except Exception as exc:                      # pragma: no cover
                self.failed += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("telegram send raised")
                return False
            self.sent += 1
            self.last_error = ""
            self._last_sent_at = loop.time()
            return True

    def status(self) -> dict[str, object]:
        """What the panel shows. Never the token."""
        return {
            "enabled": self.enabled,
            "linked": bool(self.chat_id),
            "sent": self.sent,
            "failed": self.failed,
            "last_error": self.last_error,
        }
