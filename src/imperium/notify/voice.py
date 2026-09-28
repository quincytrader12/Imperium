"""Speech, through ElevenLabs.

**Where the key lives.** In the same owner-only credential file as the venue
keys and the Telegram token, behind the same permission checks, and masked by
the same code on the way out. An ElevenLabs key is a bearer credential billed
per character: anyone holding it can spend the account's quota. It gets the
treatment the venue keys get.

**Where the audio is made.** Here, in the server process, never in the browser.
The page asks for a briefing and receives audio bytes; it never sees the key.
Putting the key in the page to save a hop would put it in every browser cache,
every devtools session and every screenshot of the terminal.

**What it will not do.** It will not speak on a timer, and it will not speak
every event. This program emits thousands of refusals an hour by design, and a
voice that reads them is one the operator mutes on the first day -- after which
it cannot say the one thing that mattered. Speech happens when asked for, and
on the handful of moments that already warrant a Telegram message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger("imperium.voice")

API = "https://api.elevenlabs.io"

#: Long enough for a paragraph of speech to be synthesised, short enough that a
#: hung request cannot hold a UI handler open indefinitely.
TIMEOUT = 30.0

#: The default model.
#:
#: Turbo is the low-latency family, which matters here: this is asked for
#: interactively and a briefing that takes four seconds to start is a briefing
#: the operator stops asking for. Overridable, because the account's plan
#: decides which models it may use.
DEFAULT_MODEL = "eleven_turbo_v2_5"

#: The longest script that will be sent, in characters.
#:
#: ElevenLabs bills per character, so an unbounded briefing is an unbounded
#: bill. It is also unlistenable: a spoken summary that runs past a minute is
#: one nobody waits through. The builder in :mod:`imperium.notify.briefing`
#: aims well under this; the cap is the backstop for the day it does not.
MAX_CHARACTERS = 2_500


class VoiceError(Exception):
    """A failed call, with something an operator can act on."""

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    def operator_text(self) -> str:
        return f"{self.message} — {self.remedy}" if self.remedy else self.message


@dataclass(frozen=True)
class Voice:
    """One voice on the account."""

    voice_id: str
    name: str


async def list_voices(api_key: str, *,
                      transport: httpx.AsyncBaseTransport | None = None
                      ) -> list[Voice]:
    """The account's voices, which doubles as the check that the key works.

    Listing rather than asking for a hard-coded voice id: the ids differ per
    account, and a default baked in here would be a default that works on the
    machine it was written on and nowhere else.
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT,
                                     transport=transport) as client:
            response = await client.get(f"{API}/v1/voices",
                                        headers={"xi-api-key": api_key})
    except httpx.HTTPError as exc:
        raise VoiceError(
            f"could not reach ElevenLabs ({type(exc).__name__})",
            "Check the machine's internet connection. ElevenLabs is reached "
            "over HTTPS on api.elevenlabs.io.") from None

    if response.status_code == 401:
        raise VoiceError(
            "ElevenLabs rejected the API key",
            "Copy it again from the ElevenLabs dashboard under Profile → API "
            "Key. A key is invalid the moment it is regenerated.")
    if response.status_code >= 400:
        raise VoiceError(
            f"ElevenLabs refused the request (HTTP {response.status_code})",
            _detail(response))

    try:
        payload = response.json()
    except ValueError:
        raise VoiceError("ElevenLabs returned something that is not JSON",
                         "") from None

    voices = []
    for row in payload.get("voices") or []:
        vid, name = row.get("voice_id"), row.get("name")
        if vid:
            voices.append(Voice(str(vid), str(name or vid)))
    if not voices:
        raise VoiceError(
            "the key works but the account has no voices",
            "Add or subscribe to a voice in the ElevenLabs dashboard first.")
    return voices


async def synthesize(api_key: str, voice_id: str, text: str, *,
                     model_id: str = DEFAULT_MODEL,
                     transport: httpx.AsyncBaseTransport | None = None
                     ) -> bytes:
    """Turn a script into MP3 bytes. Raises VoiceError, never a transport one."""
    script = (text or "").strip()
    if not script:
        raise VoiceError("there is nothing to say", "")
    if len(script) > MAX_CHARACTERS:
        # Truncated at a sentence boundary where possible: a briefing cut
        # mid-word sounds like a fault in the program rather than a limit.
        cut = script[:MAX_CHARACTERS]
        stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        script = cut[:stop + 1] if stop > MAX_CHARACTERS // 2 else cut

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT,
                                     transport=transport) as client:
            response = await client.post(
                f"{API}/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key,
                         "accept": "audio/mpeg",
                         "content-type": "application/json"},
                json={"text": script, "model_id": model_id},
            )
    except httpx.HTTPError as exc:
        raise VoiceError(
            f"could not reach ElevenLabs ({type(exc).__name__})",
            "Check the machine's internet connection. ElevenLabs is reached "
            "over HTTPS on api.elevenlabs.io.") from None

    if response.status_code == 401:
        raise VoiceError("ElevenLabs rejected the API key",
                         "Paste the key again in Connections.")
    if response.status_code == 422:
        raise VoiceError(
            "ElevenLabs refused the voice or model",
            _detail(response) or "The chosen voice may not exist on this "
                                 "account any more; pick another in "
                                 "Connections.")
    if response.status_code == 429:
        raise VoiceError(
            "ElevenLabs is rate limiting or the quota is spent",
            "Check the character quota on the ElevenLabs dashboard.")
    if response.status_code >= 400:
        raise VoiceError(
            f"ElevenLabs refused the request (HTTP {response.status_code})",
            _detail(response))

    audio = response.content
    if not audio:
        raise VoiceError("ElevenLabs returned no audio", "")
    return audio


def _detail(response: httpx.Response) -> str:
    """ElevenLabs' own words, when it gives any."""
    try:
        body = response.json()
    except ValueError:
        return ""
    detail = body.get("detail")
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("status") or "")
    if isinstance(detail, str):
        return detail
    return ""


class Speaker:
    """Holds the key and the chosen voice, and reports what it has done.

    Never raises into a caller: a terminal that stops because a
    text-to-speech API is down is a far worse outcome than a briefing nobody
    hears.
    """

    def __init__(self, api_key: str = "", voice_id: str = "",
                 voice_name: str = "", *,
                 model_id: str = DEFAULT_MODEL,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.voice_name = voice_name
        self.model_id = model_id
        self.transport = transport
        self.spoken = 0
        self.failed = 0
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.voice_id)

    def configure(self, api_key: str, voice_id: str = "",
                  voice_name: str = "") -> None:
        self.api_key = api_key
        if voice_id:
            self.voice_id = voice_id
        if voice_name:
            self.voice_name = voice_name

    async def speak(self, text: str) -> bytes | None:
        """MP3 bytes, or None with the reason recorded."""
        if not self.enabled:
            return None
        try:
            audio = await synthesize(self.api_key, self.voice_id, text,
                                     model_id=self.model_id,
                                     transport=self.transport)
        except VoiceError as exc:
            self.failed += 1
            self.last_error = exc.operator_text()
            log.warning("speech failed: %s", self.last_error)
            return None
        except Exception as exc:                            # pragma: no cover
            self.failed += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.exception("speech raised")
            return None
        self.spoken += 1
        self.last_error = ""
        return audio

    def status(self) -> dict[str, object]:
        """What the panel shows. Never the key."""
        return {
            "enabled": self.enabled,
            "voice": self.voice_name,
            "spoken": self.spoken,
            "failed": self.failed,
            "last_error": self.last_error,
        }
