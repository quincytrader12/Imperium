"""The local server.

Binds **127.0.0.1 only**, and validates it rather than defaulting to it. This
process holds API keys and has no authentication of its own; a bind to 0.0.0.0
would publish an unauthenticated trading control plane to the local network.

The UI is read-only with respect to the venue. It starts and stops sessions and
switches mode. It never places an order. A UI that can trade is a second,
untested path to the exchange.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import logging
import re
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (FastAPI, HTTPException, Request, Response, WebSocket,
                     WebSocketDisconnect)
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from imperium import config, logging_setup, loopwatch
from imperium.diagnostics.layers import NetworkDiagnostic
from imperium.notify import ask as ask_mod
from imperium.notify import telegram, voice
from imperium.execution.broker import LIVE_CONFIRMATION_PHRASE, Mode, ModeSwitchRefused
from imperium.security.credentials import CredentialError, CredentialStore
from imperium.execution.sleeve_ledger import trading_day
from imperium.session import LOOP_STALL_SECONDS, TradingSession
from imperium.telemetry.streams import Level
from imperium.venues import registry
from imperium.venues.alpaca.client import VenueError

log = logging.getLogger("imperium.server")

#: Fixed cadence, not on-change. The cluster animates continuously, so a steady
#: frame rate is what the client needs, and it bounds the server's work
#: regardless of how busy the book is.
SNAPSHOT_HZ = 1.0

#: Memoised by asset_version(). The files cannot change under a running
#: process: in a frozen build they are unpacked once, and from source a
#: developer restarts.
_ASSET_VERSION = ""


class VersionedStatic(StaticFiles):
    """Static files that cannot serve a stale build, and still cache well.

    Versioning the URLs in the page closes most of the hole but not all of
    it: the orb is ES modules, and ``import { ProcessOrb } from './orb.js'``
    resolves to an unversioned URL that no rewrite of the markup can reach.
    Leaving those to heuristic caching is how a new build renders an old orb.

    So the rule is per-request rather than per-file:

    * **With a ``?v=`` token** the URL changes whenever the bytes do, so the
      response is safe to keep forever and is marked immutable. This is the
      fast path and it covers every entry point the page names.
    * **Without one** -- a relative import between modules -- the response
      must be revalidated before use. ``no-cache`` does not mean "do not
      store"; it means "ask first", and with the ETag that Starlette already
      sends the answer is a 304 with no body. Over loopback that is free.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        query = scope.get("query_string", b"")
        versioned = b"v=" in query
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if versioned else "no-cache")
        return response


def asset_version() -> str:
    """A token that changes exactly when the static files change.

    The bug this fixes, which would have recurred on every build shipped:
    ``/`` is regenerated from disk on each request, so the markup was always
    current, while every asset it names -- app.js, styles.css, palette.js,
    orb.boot.js, the vendored three.js -- was an unversioned URL served with
    no Cache-Control header at all.

    With no cache directive a browser is free to apply heuristic freshness,
    and a normal refresh does not revalidate subresources; only a hard reload
    does. The result is new HTML running old JavaScript, which presents as
    "the terminal is showing an old build even after refreshing" and is
    impossible to distinguish from a build that did not install.

    A content hash rather than the version string, because two builds of the
    same version are exactly the case that goes wrong -- and rather than an
    mtime, because PyInstaller's unpack sets its own timestamps and a file
    restored from a backup can have an older one than the file it replaced.

    Read once. The cost is one pass over a few hundred kilobytes at startup,
    and it buys a URL that is safe to cache forever.
    """
    global _ASSET_VERSION
    if _ASSET_VERSION:
        return _ASSET_VERSION
    digest = hashlib.sha256()
    try:
        for path in sorted(static_dir().rglob("*")):
            if path.is_file() and path.name != "index.html":
                digest.update(path.name.encode("utf-8"))
                digest.update(path.read_bytes())
    except OSError:
        # Never fatal. A version that cannot be computed becomes the process
        # start time, which is still correct across a restart -- the only way
        # the files change in a packaged build.
        _ASSET_VERSION = f"{int(time.time()):x}"
        return _ASSET_VERSION
    _ASSET_VERSION = digest.hexdigest()[:12]
    return _ASSET_VERSION


#: Matches every /static/… URL the page names, in a tag or in the import map.
_ASSET_URL = re.compile(r'(["\'])(/static/[^"\'?]+)\1')


def version_assets(markup: str, token: str) -> str:
    """Stamp every /static URL in the page with the build token.

    Rewritten on the way out rather than written into index.html, because the
    import map names files no tag points at -- a hand-maintained list would go
    stale exactly when a new module was added, which is when it matters.
    """
    return _ASSET_URL.sub(
        lambda m: f"{m.group(1)}{m.group(2)}?v={token}{m.group(1)}", markup)


def static_dir() -> Path:
    """Locate the bundled static files, in a frozen build or from source.

    PyInstaller unpacks --add-data under sys._MEIPASS. A build that silently
    loses the static files starts, serves the API, and 404s its own page, so
    this raises loudly instead.
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidate = base / "imperium" / "server" / "static"
        if candidate.is_dir():
            return candidate
        candidate = base / "static"
        if candidate.is_dir():
            return candidate
        raise RuntimeError(
            "the packaged build is missing its static files. The executable was "
            "built without --add-data, so the API works but the page cannot be "
            "served. Rebuild with the data files included."
        )
    return Path(__file__).parent / "static"


def validate_bind_host(host: str) -> str:
    """Raise on any bind address other than loopback.

    Not a default -- a rule. There is no flag that relaxes it.
    """
    if host in config.ALLOWED_BIND_HOSTS:
        return host
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(
            f"refusing to bind to {host!r}: this process holds API keys and has "
            f"no authentication. Only {sorted(config.ALLOWED_BIND_HOSTS)} are "
            f"permitted."
        ) from None
    if not addr.is_loopback:
        raise ValueError(
            f"refusing to bind to {host!r}: that address is reachable from the "
            f"network, and this process holds API keys and has no "
            f"authentication. Use 127.0.0.1."
        )
    return host


# -- request models -------------------------------------------------------

#: The credential-store entry the Telegram token lives under.
#:
#: A fixed name and a distinct venue, so it can never be confused with a
#: trading key: nothing that walks the venue credentials picks it up, and the
#: connections panel lists it separately.
TELEGRAM_NAME = "telegram"
TELEGRAM_VENUE = "telegram"


#: Where the ElevenLabs key lives in the credential store.
VOICE_NAME = "elevenlabs"
VOICE_VENUE = "elevenlabs"


class VoiceKeyRequest(BaseModel):
    # ElevenLabs keys are prefixed and of a known rough length; bounded so a
    # paste of the wrong thing entirely is refused before it reaches the
    # network.
    api_key: str = Field(min_length=20, max_length=256)


class VoiceChoiceRequest(BaseModel):
    voice_id: str = Field(min_length=1, max_length=128)
    voice_name: str = Field(default="", max_length=128)


class ArmRequest(BaseModel):
    # Set once the operator has been told what arming below the threshold
    # costs. The server asks rather than the page, so the warning cannot be
    # skipped by calling the endpoint directly.
    acknowledged: bool = False


class AskRequest(BaseModel):
    # A spoken question, transcribed. Bounded because it is echoed back in the
    # reply and, when spoken, billed per character: an unbounded question is an
    # unbounded bill and an unbounded thing to read out.
    question: str = Field(min_length=1, max_length=300)


class TelegramRequest(BaseModel):
    # BotFather tokens look like 123456789:AAE... — bounded so a paste of the
    # wrong thing entirely is refused before it reaches the network.
    token: str = Field(min_length=20, max_length=256)


class AddKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    venue: str = "alpaca"
    api_key: str = Field(min_length=8, max_length=256)
    secret: str = Field(min_length=8, max_length=256)
    note: str = ""


class EnableKeyRequest(BaseModel):
    trade_enabled: bool


class ModeRequest(BaseModel):
    mode: str
    confirmation: str = ""


class AttachRequest(BaseModel):
    name: str | None = None


class HaltRequest(BaseModel):
    halted: bool
    reason: str = "halted by the operator"


#: The RuntimeError messages a send raises when the client has already gone.
#:
#: Matched on text because that is all the server gives: neither layer raises a
#: typed disconnect on the send path. Both layers are listed because a closing
#: socket can fail at either -- Starlette's own check fires when it has already
#: written the close frame, uvicorn's when the close reached the protocol first
#: -- and an earlier version of this matched only uvicorn's, which left the
#: commoner of the two still printing a traceback.
_CLOSED_SOCKET_MARKERS = (
    "websocket.close",          # uvicorn: send after 'websocket.close'
    "websocket.disconnect",     # uvicorn: send after the client's disconnect
    "once a close message",     # starlette: send after it sent the close
    "is not connected",         # starlette: the socket is no longer accepted
)


def _is_closed_socket(exc: RuntimeError) -> bool:
    """Whether this RuntimeError is just a socket the client already closed.

    Narrow on purpose. Any other RuntimeError from a send is a real fault and
    is re-raised: swallowing those would trade a noisy log for a silent one,
    which is the worse of the two.
    """
    text = str(exc).lower()
    return any(marker in text for marker in _CLOSED_SOCKET_MARKERS)


def create_app(session: TradingSession | None = None) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["session"] = session or TradingSession()

        # The credential store must never prevent the terminal from starting.
        # This is the whole point of the program: a bot that will not run and a
        # bot that cannot explain why look identical, and the UI is what tells
        # them apart -- so it has to come up even when something is broken.
        #
        # A malformed credentials file did take the whole application down once:
        # a file that parsed as JSON but was shaped wrongly raised a TypeError
        # out of CredentialStore, only CredentialError was caught here, and
        # uvicorn aborted startup. Catching the broad Exception is deliberate.
        try:
            state["store"] = CredentialStore(quarantine_corrupt=True)
        except CredentialError as exc:
            state["store"] = None
            state["store_error"] = f"{exc}" + (f" — {exc.remedy}" if exc.remedy else "")
        except Exception as exc:  # noqa: BLE001 - see above
            log.exception("the credential store failed to load")
            state["store"] = None
            state["store_error"] = (
                f"the credential store could not be read ({type(exc).__name__}: "
                f"{exc}). This is a bug -- the terminal is running without it."
            )
        else:
            store = state["store"]
            if store.quarantined_to is not None:
                # Resolved, not merely reported: say so once in the activity log
                # and move on, rather than showing a standing error for a file
                # that is no longer in the way.
                state["session"].telemetry.event(
                    Level.WARN, "security",
                    "the previous credentials file could not be read and was "
                    "moved aside; starting with an empty one",
                    detail=f"the old file is kept as {store.quarantined_to.name}")
            report = store.permission_report
            if not report.ok:
                state["session"].telemetry.event(
                    Level.ERROR, "security",
                    f"INSECURE CREDENTIAL FILE: {report.detail}",
                    detail=report.remedy)
            # Restore the Telegram link. Without this it is lost on every
            # restart, and this program is meant to run for weeks and to
            # restart itself when it crashes -- a notifier that silently
            # unlinks on the one event worth notifying about is worse than
            # none, because the silence reads as "nothing happened".
            token = store.token_for(TELEGRAM_NAME)
            if token:
                state["session"].notifier.configure(
                    token, store.chat_for(TELEGRAM_NAME))
            # Same for the voice: a terminal that restarts itself must come
            # back able to speak, or the silence after a restart reads as
            # nothing having happened.
            # Re-attach the key that was attached last time.
            #
            # Stored keys always persisted; what did not was the *attachment*,
            # so every restart came back with no account, no balance and no
            # live data until somebody opened Connections and clicked attach.
            # On a program built to restart itself unattended that is not a
            # small annoyance, it is a terminal that quietly stops trading
            # until a human notices.
            remembered = TradingSession.remembered_credential()
            venue_keys = [c["name"] for c in store.masked_list()
                          if c.get("venue") not in (TELEGRAM_VENUE, VOICE_VENUE)]
            wanted = (remembered if remembered in venue_keys
                      else venue_keys[0] if len(venue_keys) == 1 else "")
            if wanted:
                try:
                    await state["session"].attach_credential(store, wanted)
                    state["session"].telemetry.event(
                        Level.INFO, "security",
                        f"re-attached the credential {wanted!r} from the last "
                        f"session")
                except Exception as exc:            # noqa: BLE001
                    # A key that no longer works must not stop the terminal
                    # starting; it must say so and carry on unattached.
                    state["session"].telemetry.event(
                        Level.WARN, "security",
                        f"could not re-attach {wanted!r}: {exc}",
                        detail="Attach it again in Connections, or store a "
                               "new key if this one was revoked.")

            voice_key = store.token_for(VOICE_NAME)
            if voice_key:
                chosen = next((c for c in store.masked_list()
                               if c.get("name") == VOICE_NAME), {})
                state["session"].speaker.configure(
                    voice_key, store.chat_for(VOICE_NAME),
                    str(chosen.get("note") or ""))

        if state.get("store_error"):
            state["session"].store_error = state["store_error"]
            state["session"].telemetry.event(
                Level.ERROR, "security",
                "the credentials file could not be loaded",
                detail=state["store_error"])
        # Started here rather than with the trading loop, because the
        # question it answers -- did this process stall? -- is asked most
        # often when nothing is running.
        session_ = state["session"]
        watch = loopwatch.LoopWatch()
        session_.loop_watch = watch

        def _say(lag: float) -> None:
            session_.telemetry.event(
                Level.WARN, "health",
                f"the terminal was busy or blocked for {lag:.1f}s and could "
                f"not do anything else in that time",
                detail="Long enough to drop the browser link if it passes "
                       "20s. Recorded so 'the link went red' has evidence "
                       "behind it.")

        watcher = asyncio.create_task(loopwatch.watch(watch, on_stall=_say),
                                      name="loop-watch")

        async def heartbeat() -> None:
            """Restart a trading loop that has stopped beating.

            This used to run on the websocket's frame, which made the
            terminal's own watchdog depend on a browser being connected to
            it. That is backwards for a program built to run unattended for
            weeks: the moment the link went red -- a closed tab, a refresh, a
            dropped socket -- the one thing watching for a dead trading loop
            stopped watching, and it stayed stopped for as long as the link
            did. Nothing about a browser is load-bearing here.
            """
            while True:
                await asyncio.sleep(1.0)
                with contextlib.suppress(Exception):
                    await session_.supervise()

        pulse = asyncio.create_task(heartbeat(), name="session-heartbeat")
        try:
            yield
        finally:
            for task in (watcher, pulse):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        await state["session"].stop()
        await state["session"].detach_client()

    app = FastAPI(title="IMPERIUM", lifespan=lifespan, docs_url=None, redoc_url=None)

    def get_session() -> TradingSession:
        return state["session"]

    def get_store() -> CredentialStore:
        store = state.get("store")
        if store is None:
            exc = state.get("store_error")
            raise HTTPException(500, detail=str(exc) if exc else "no credential store")
        return store

    # -- pages -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        page = static_dir() / "index.html"
        # Explicit utf-8: the platform default is cp1252 on Windows, and one
        # box-drawing character would fail the Windows build while passing
        # everywhere it was tested.
        markup = version_assets(
            page.read_text(encoding=config.TEXT_ENCODING), asset_version())
        # The page itself must never be cached. It is the only thing that
        # carries the current asset version, so a stale copy of it pins the
        # browser to a stale build no matter how new the files on disk are.
        return HTMLResponse(markup, headers={"Cache-Control": "no-store"})

    @app.get("/diagnose", response_class=PlainTextResponse)
    async def diagnose(venue: str = registry.DEFAULT_VENUE) -> PlainTextResponse:
        """Plain text, at a URL as well as behind a button.

        A button in a scrolling panel is a control people cannot find, and a URL
        survives any layout change and can be pasted into a report without a
        screenshot.
        """
        try:
            spec = registry.get(venue)
        except KeyError as exc:
            return PlainTextResponse(str(exc), status_code=404)
        result = await NetworkDiagnostic(spec.base_url).run()
        return PlainTextResponse(result.as_text())

    @app.get("/api/diagnose")
    async def diagnose_json(venue: str = registry.DEFAULT_VENUE) -> JSONResponse:
        spec = registry.get(venue)
        result = await NetworkDiagnostic(spec.base_url).run()
        return JSONResponse(result.as_dict())

    # -- credentials -----------------------------------------------------

    @app.get("/api/connections")
    async def connections() -> JSONResponse:
        """Masked views and nothing else. Not once, not for debugging."""
        store = state.get("store")
        if store is None:
            # The panel still renders, and says what is wrong and where. A raw
            # 500 here would tell the operator only that something failed.
            return JSONResponse({
                "credentials": [],
                "path": str(config.credentials_path()),
                "permissions_ok": True,
                "permissions_detail": "",
                "permissions_remedy": "",
                "attached": None,
                "error": state.get("store_error", "the credential store is unavailable"),
            })
        report = store.permission_report
        return JSONResponse({
            "credentials": store.masked_list(),
            "path": str(store.path),
            "permissions_ok": report.ok,
            "permissions_detail": report.detail,
            "permissions_remedy": report.remedy,
            "attached": get_session().credential.name
            if get_session().credential else None,
        })

    @app.post("/api/connections")
    async def add_connection(body: AddKeyRequest) -> JSONResponse:
        store = get_store()
        try:
            cred = store.add(body.name, body.venue, body.api_key, body.secret,
                             body.note)
        except CredentialError as exc:
            raise HTTPException(400, detail=f"{exc} — {exc.remedy}") from None
        get_session().telemetry.event(
            Level.INFO, "security",
            f"credential {cred.name!r} stored (not enabled for trading)")
        return JSONResponse(cred.masked(), status_code=201)

    @app.post("/api/connections/{name}/trade")
    async def set_trade_enabled(name: str, body: EnableKeyRequest) -> JSONResponse:
        store = get_store()
        try:
            cred = store.set_trade_enabled(name, body.trade_enabled)
        except CredentialError as exc:
            raise HTTPException(404, detail=str(exc)) from None
        session = get_session()
        if session.credential and session.credential.name == name:
            session.credential = cred
        session.telemetry.event(
            Level.WARN if cred.trade_enabled else Level.INFO, "security",
            f"credential {name!r} is now "
            f"{'ENABLED for trading' if cred.trade_enabled else 'disabled for trading'}")
        return JSONResponse(cred.masked())

    @app.delete("/api/connections/{name}")
    async def delete_connection(name: str) -> JSONResponse:
        get_store().remove(name)
        return JSONResponse({"removed": name})

    @app.get("/api/voice")
    async def voice_status() -> JSONResponse:
        """Status only. The key is never returned, masked or otherwise."""
        session = get_session()
        store = state.get("store")
        stored = False
        if store is not None:
            stored = any(c.get("venue") == VOICE_VENUE
                         for c in store.masked_list())
        return JSONResponse({**session.speaker.status(), "stored": stored})

    @app.post("/api/voice")
    async def voice_connect(body: VoiceKeyRequest) -> JSONResponse:
        """Check the key and remember it, and hand back the account's voices.

        Listing the voices rather than baking in an id: they differ per
        account, so a default written here would work on one machine and
        nowhere else.
        """
        session = get_session()
        key = body.api_key.strip()
        try:
            voices = await voice.list_voices(key)
        except voice.VoiceError as exc:
            raise HTTPException(400, detail=exc.operator_text()) from None
        store = get_store()
        chosen = store.chat_for(VOICE_NAME)          # the previously picked id
        store.put_token(VOICE_NAME, VOICE_VENUE, key,
                        note=session.speaker.voice_name, chat=chosen)
        session.speaker.configure(key)
        session.telemetry.event(
            Level.INFO, "security",
            f"ElevenLabs key stored ({len(voices)} voices available)")
        return JSONResponse({
            "voices": [{"voice_id": v.voice_id, "name": v.name}
                       for v in voices],
        })

    @app.post("/api/voice/select")
    async def voice_select(body: VoiceChoiceRequest) -> JSONResponse:
        session = get_session()
        store = get_store()
        key = store.token_for(VOICE_NAME)
        if not key:
            raise HTTPException(400, detail="paste the ElevenLabs key first")
        store.set_chat(VOICE_NAME, body.voice_id)
        store.put_token(VOICE_NAME, VOICE_VENUE, key,
                        note=body.voice_name, chat=body.voice_id)
        session.speaker.configure(key, body.voice_id, body.voice_name)
        return JSONResponse(session.speaker.status())

    @app.get("/api/voice/script")
    async def voice_script() -> JSONResponse:
        """The briefing as text, without spending a character of quota.

        Worth its own endpoint: it is how an operator checks what it would say
        before paying to hear it, and how this gets debugged without audio.
        """
        return JSONResponse({"script": get_session().briefing()})

    @app.post("/api/voice/speak")
    async def voice_speak() -> Response:
        """The briefing as audio.

        Synthesised here rather than in the page: the browser never sees the
        key. Putting it in the page to save a hop would put it in every
        browser cache, devtools session and screenshot of this terminal.
        """
        session = get_session()
        if not session.speaker.enabled:
            raise HTTPException(
                400, detail="no ElevenLabs key and voice are set up yet")
        audio = await session.speaker.speak(session.briefing())
        if audio is None:
            raise HTTPException(
                502, detail=session.speaker.last_error or "speech failed")
        return Response(content=audio, media_type="audio/mpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/sector/arm")
    async def sector_arm(body: ArmRequest) -> JSONResponse:
        """Switch the Sector Trend sleeve on now.

        Allowed below the arming threshold, because it is the operator's money
        and their call -- but not silently. An undersized sleeve cannot afford
        its most volatile names, and the ones it drops are exactly the ones
        the strategy's returns come from, so the first call gets a refusal
        explaining that and the second one carries ``acknowledged``.
        """
        session = get_session()
        sector = session.sector
        if sector.enabled:
            return JSONResponse({"armed": True, "note": "already armed"})

        equity = session.arming_equity()
        if equity <= 0:
            raise HTTPException(
                409, detail="the account balance is not known yet, so there "
                            "is nothing to size a sleeve against. Attach a "
                            "key first.")

        threshold = float(sector.config.arm_at_equity or 0.0)
        if threshold > 0 and equity < threshold and not body.acknowledged:
            ceiling = sector.volatility_ceiling(equity)
            raise HTTPException(409, detail=(
                f"${equity:,.2f} is under the ${threshold:,.0f} this sleeve "
                f"wants. At this balance it can only afford ETFs quieter than "
                f"{ceiling * 100:.2f}% a day, and the ones it would drop are "
                f"the most volatile -- which is where the strategy's return "
                f"comes from. It would be trading the calm half of its "
                f"universe, which is a different strategy with no backtest "
                f"behind it. Arm anyway to accept that."))

        message = sector.arm_by_hand(equity, trading_day())
        if message:
            session.telemetry.event(Level.WARN, "sector", message)
        return JSONResponse({"armed": True, "note": message})

    @app.post("/api/sector/disarm")
    async def sector_disarm() -> JSONResponse:
        """Switch it off. Refused while it is holding anything."""
        session = get_session()
        refusal = session.sector.disarm()
        if refusal:
            raise HTTPException(409, detail=refusal)
        session.telemetry.event(Level.INFO, "sector",
                                "Sector Trend disarmed by hand")
        return JSONResponse({"armed": False})

    @app.post("/api/voice/greeting")
    async def voice_greeting() -> Response:
        """Speak the greeting drawn at the last Start.

        Deliberately not an endpoint that speaks arbitrary text. The browser
        asks for *the* greeting and the server decides what that is, so a page
        on this machine cannot run up an ElevenLabs bill a character at a time,
        and the spoken line is guaranteed to be the one already on screen.
        """
        session = get_session()
        if session.opening is None:
            raise HTTPException(409, detail="the session has not been started")
        if not session.speaker.enabled:
            raise HTTPException(
                400, detail="no ElevenLabs key and voice are set up yet")
        audio = await session.speaker.speak(session.opening.spoken)
        if audio is None:
            raise HTTPException(
                502, detail=session.speaker.last_error or "speech failed")
        return Response(content=audio, media_type="audio/mpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/ask")
    async def ask_question(body: AskRequest) -> JSONResponse:
        """Answer a question from the snapshot. Text only, no quota spent.

        Separate from the speaking endpoint on purpose: this is how the
        question is answered when the voice is not set up, how it is tested,
        and how an operator sees what it *would* say before paying to hear it.
        """
        session = get_session()
        answer = ask_mod.respond(body.question, session.snapshot())
        return JSONResponse(answer.as_dict())

    @app.post("/api/ask/speak")
    async def ask_aloud(body: AskRequest) -> Response:
        """The answer as audio.

        Synthesised here rather than in the page, for the same reason as the
        briefing: the browser never sees the ElevenLabs key.

        The answer is computed first and the question is not sent anywhere. A
        question is whatever the operator said in their own room, and the only
        thing that leaves this machine is the sentence built from their own
        snapshot.
        """
        session = get_session()
        answer = ask_mod.respond(body.question, session.snapshot())
        if not session.speaker.enabled:
            raise HTTPException(
                400, detail="no ElevenLabs key and voice are set up yet")
        audio = await session.speaker.speak(answer.text)
        if audio is None:
            raise HTTPException(
                502, detail=session.speaker.last_error or "speech failed")
        return Response(content=audio, media_type="audio/mpeg",
                        headers={"Cache-Control": "no-store",
                                 "X-Imperium-Intent": answer.intent})

    @app.delete("/api/voice")
    async def voice_forget() -> JSONResponse:
        session = get_session()
        store = state.get("store")
        if store is not None:
            try:
                store.remove(VOICE_NAME)
            except Exception:
                pass
        session.speaker.configure("", "", "")
        session.speaker.voice_id = ""
        session.speaker.api_key = ""
        return JSONResponse({"removed": True})

    @app.get("/api/telegram")
    async def telegram_status() -> JSONResponse:
        """Status only. The token is never returned, masked or otherwise."""
        session = get_session()
        store = state.get("store")
        cred = None
        if store is not None:
            cred = next((c for c in store.masked_list()
                         if c.get("venue") == TELEGRAM_VENUE), None)
        return JSONResponse({
            **session.notifier.status(),
            "bot": (cred or {}).get("note", ""),
            "stored": cred is not None,
        })

    @app.post("/api/telegram")
    async def telegram_connect(body: TelegramRequest) -> JSONResponse:
        """Step one: check the token and remember it. No chat yet."""
        session = get_session()
        try:
            identity = await telegram.identify(body.token.strip())
        except telegram.TelegramError as exc:
            raise HTTPException(400, detail=exc.operator_text()) from None
        store = get_store()
        # Stored beside the venue keys on purpose: same owner-only file, same
        # permission checks, same masking on the way out. A bot token is a
        # bearer credential and deserves the treatment the venue keys get,
        # not a second and less careful store invented for it.
        keep_chat = store.chat_for(TELEGRAM_NAME)
        store.put_token(TELEGRAM_NAME, TELEGRAM_VENUE, body.token.strip(),
                        note=f"@{identity.username}", chat=keep_chat)
        session.notifier.configure(body.token.strip(), keep_chat)
        session.telemetry.event(
            Level.INFO, "security",
            f"Telegram bot @{identity.username} stored (not linked to a chat yet)")
        return JSONResponse({"bot": f"@{identity.username}",
                             "name": identity.name, "linked": False})

    @app.post("/api/telegram/link")
    async def telegram_link() -> JSONResponse:
        """Step two: read the chat id out of the message the operator sent.

        This is the step every other integration makes people do by hand.
        """
        session = get_session()
        store = get_store()
        token = store.token_for(TELEGRAM_NAME)
        if not token:
            raise HTTPException(400, detail="paste the bot token first")
        try:
            chat_id = await telegram.discover_chat(token)
        except telegram.TelegramError as exc:
            raise HTTPException(400, detail=exc.operator_text()) from None
        store.set_chat(TELEGRAM_NAME, chat_id)
        session.notifier.configure(token, chat_id)
        await session.notify(
            "✅ IMPERIUM is linked.\nYou will get a message when an order "
            "fills, when the book halts itself, and when live trading is armed."
            "\nRefusals are not sent — this program refuses thousands of times "
            "an hour by design.")
        session.telemetry.event(Level.GOOD, "security",
                                "Telegram linked to a chat")
        return JSONResponse({"linked": True, **session.notifier.status()})

    @app.post("/api/telegram/test")
    async def telegram_test() -> JSONResponse:
        session = get_session()
        sent = await session.notify("IMPERIUM test message — the link works.")
        return JSONResponse({"sent": sent,
                             "error": session.notifier.last_error})

    @app.delete("/api/telegram")
    async def telegram_forget() -> JSONResponse:
        session = get_session()
        store = state.get("store")
        if store is not None:
            try:
                store.remove(TELEGRAM_NAME)
            except Exception:
                pass
        session.notifier.configure("", "")
        return JSONResponse({"removed": True})

    @app.post("/api/attach")
    async def attach(body: AttachRequest) -> JSONResponse:
        session = get_session()
        try:
            await session.attach_credential(get_store(), body.name)
        except CredentialError as exc:
            raise HTTPException(404, detail=str(exc)) from None
        except VenueError as exc:
            raise HTTPException(502, detail=exc.operator_text()) from None
        return JSONResponse({"attached": body.name, "key_lamp": session.lamps.key,
                             "error": session.venue_error})

    @app.get("/api/balances")
    async def balances() -> JSONResponse:
        session = get_session()
        if session.client is None or not session.client.authenticated:
            raise HTTPException(400, detail="no credential is attached")
        try:
            rows = await session.client.balances()
        except VenueError as exc:
            raise HTTPException(502, detail=exc.operator_text()) from None
        return JSONResponse({"balances": [
            {"asset": r["asset"], "free": str(r["free"]), "locked": str(r["locked"])}
            for r in rows]})

    # -- session control -------------------------------------------------

    @app.post("/api/session/start")
    async def start_session() -> JSONResponse:
        session = get_session()
        await session.start()
        # The greeting comes back with the response rather than arriving on the
        # next snapshot, so it is on screen the instant the button is released.
        opening = session.opening.as_dict() if session.opening else None
        return JSONResponse({"running": True, "opening": opening})

    @app.post("/api/session/stop")
    async def stop_session() -> JSONResponse:
        await get_session().stop()
        return JSONResponse({"running": False})

    @app.post("/api/session/mode")
    async def set_mode(body: ModeRequest) -> JSONResponse:
        session = get_session()
        try:
            mode = Mode(body.mode)
        except ValueError:
            raise HTTPException(400, detail=f"unknown mode {body.mode!r}") from None
        try:
            await session.set_mode(mode, body.confirmation, get_store())
        except ModeSwitchRefused as exc:
            raise HTTPException(403, detail=str(exc)) from None
        except VenueError as exc:
            raise HTTPException(502, detail=exc.operator_text()) from None
        return JSONResponse({"mode": session.broker.mode.value,
                             "phrase_required": LIVE_CONFIRMATION_PHRASE})

    @app.post("/api/session/halt")
    async def set_halt(body: HaltRequest) -> JSONResponse:
        """Stop the book taking new or increased exposure.

        This is a *reduction* control, not an order path: a halt blocks new and
        increased exposure and explicitly still lets reductions through, so it
        cannot trap a position. Nothing here submits an order, which is what
        keeps the "the UI never places an order" rule intact -- flattening
        happens through a mode switch, which already exists.
        """
        session = get_session()
        session.allocator.set_halt(body.halted, body.reason if body.halted else "")
        session.telemetry.event(
            Level.WARN if body.halted else Level.GOOD, "risk",
            f"the book was {'HALTED' if body.halted else 'released'} by the operator",
            detail=body.reason if body.halted else "")
        if body.halted:
            session.telemetry.pulse("BOOK", "halt", body.reason, 1.0)
        return JSONResponse({"halted": session.allocator.halted,
                             "reason": session.allocator.halt_reason})

    @app.get("/api/snapshot")
    async def snapshot() -> JSONResponse:
        return JSONResponse(get_session().snapshot())

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True, "version": __import__("imperium").__version__})

    # -- the stream ------------------------------------------------------

    @app.websocket("/ws")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        session = get_session()
        session.lamps.link = "ok"
        # Per connection, not per session: two browsers open on the same
        # terminal each need their own place in the rings, and a shared cursor
        # would have them stealing each other's rows.
        since_pulse = 0
        since_event = 0
        opened_at = time.time()

        # Drained, rather than left to pile up.
        #
        # This loop only ever sent. Starlette buffers whatever the client
        # sends until the application receives it, and the client now sends a
        # small keepalive -- so without a reader those frames accumulate for
        # as long as the tab is open.
        #
        # It also fixes the thing the comment below used to describe rather
        # than solve: a socket that closes is noticed here immediately, on the
        # disconnect message, instead of on the next send several hundred
        # milliseconds later.
        gone = asyncio.Event()

        async def drain() -> None:
            try:
                while True:
                    message = await ws.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
            except Exception:
                pass
            finally:
                gone.set()

        reader = asyncio.create_task(drain())
        try:
            while not gone.is_set():
                start = time.perf_counter()
                try:
                    # A client whose cursor has fallen behind the ring cannot be
                    # caught up by a delta -- the rows it missed are gone. It is
                    # sent the whole window instead, which is what a frame with
                    # delta=False tells it to expect.
                    oldest = session.telemetry.oldest_pulse_seq
                    if since_pulse and oldest and since_pulse < oldest - 1:
                        since_pulse = since_event = 0
                    payload = session.snapshot(since_pulse=since_pulse,
                                               since_event=since_event)
                    since_pulse = payload.get("pulse_seq") or since_pulse
                    since_event = payload.get("event_seq") or since_event
                except Exception as exc:
                    # A snapshot that raises must not close the socket, or the
                    # UI goes dark for a reason it cannot show. The cursor is
                    # left where it was so the next frame re-sends what this one
                    # failed to deliver rather than skipping past it.
                    log.exception("snapshot failed")
                    payload = {"ts": time.time(), "error": str(exc)}

                # A closed tab is not an error.
                #
                # This loop only ever sends; it never awaits receive(), so it
                # is never handed the `websocket.disconnect` that would raise
                # WebSocketDisconnect. The first it learns of a closed socket
                # is the next send, and uvicorn answers that with a bare
                # RuntimeError -- "Unexpected ASGI message 'websocket.send',
                # after sending 'websocket.close'". Left uncaught it printed a
                # five-frame traceback at ERROR every time an operator
                # refreshed the page or closed the tab, which trains the eye
                # to skip the log that real faults are written to.
                if (ws.client_state is not WebSocketState.CONNECTED
                        or ws.application_state is not WebSocketState.CONNECTED):
                    break
                try:
                    await ws.send_json(payload)
                except RuntimeError as exc:
                    # The race the state check above cannot close: the client
                    # can go between the check and the send.
                    if _is_closed_socket(exc):
                        break
                    raise
                elapsed = time.perf_counter() - start
                # Woken by the drain task on disconnect rather than sleeping
                # out the rest of the frame and discovering it on the next
                # send.
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        gone.wait(), max(0.05, (1.0 / SNAPSHOT_HZ) - elapsed))
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("the websocket loop failed")
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            session.lamps.link = "off"
            # Both ends of the story. The browser records its close code on
            # the lamp's tooltip; this records how long the connection lasted
            # and whether this end knew it was ending. A socket that closes
            # with code 1006 in the browser and no disconnect message here was
            # dropped underneath both of them -- which is the difference
            # between a bug in this program and something outside it, and it
            # cannot be told apart from one side alone.
            log.info("link closed after %.0fs (%s)",
                     time.time() - opened_at,
                     "the client said goodbye" if gone.is_set()
                     else "no disconnect reached us")

    app.mount("/static", VersionedStatic(directory=str(static_dir())),
              name="static")
    return app


def port_is_free(host: str, port: int) -> bool:
    """True if nothing is already listening on this address.

    **Do not set SO_REUSEADDR here.** Its meaning differs between platforms in
    exactly the way that breaks this check: on POSIX it only permits reusing a
    socket in TIME_WAIT, but on Windows it permits binding to a port that is
    *actively listening*. With it set, this function returned True for a busy
    port on Windows, so :func:`choose_port` handed back the port that was
    already taken and the server died with the bare ``WinError 10048`` this
    whole mechanism exists to avoid.

    It is invisible on Linux, which is why the test suite runs on Windows too.

    ``SO_EXCLUSIVEADDRUSE`` (Windows only) makes the probe stricter still: it
    refuses the bind if anything else could also claim the port.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:                      # Windows
            try:
                sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
            except OSError:
                pass
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def choose_port(host: str, preferred: int, attempts: int = 20) -> int:
    """Return the preferred port, or the next free one above it.

    A second copy of the program, or anything else holding the port, otherwise
    kills the process with a bare ``WinError 10048`` before it prints a URL --
    which reads as "the program is broken" rather than "that port is taken".
    """
    for offset in range(attempts):
        candidate = preferred + offset
        if candidate > 65535:
            break
        if port_is_free(host, candidate):
            return candidate
    raise RuntimeError(
        f"no free port between {preferred} and {preferred + attempts - 1}. "
        f"Close whatever is holding them, or pass --port with a free one."
    )


def run_server(host: str = "127.0.0.1", port: int = config.DEFAULT_PORT,
               open_browser: bool = True) -> int:
    import uvicorn

    host = validate_bind_host(host)
    logging_setup.configure()
    config.ensure_home()

    # Fail loudly now rather than 404-ing the page later.
    static = static_dir()
    if not (static / "index.html").exists():
        raise RuntimeError(f"static files are missing from {static}")

    chosen = choose_port(host, port)
    if chosen != port:
        print(f"  note: port {port} is already in use, so this instance is on "
              f"{chosen} instead.")

    url = f"http://{host}:{chosen}/"
    if open_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()

    # A banner rather than a log line: this is the one piece of information the
    # operator needs, and it must survive being scrolled past.
    bar = "=" * 62
    print(bar)
    print("  IMPERIUM — built by Quincy Gininda")
    print(bar)
    print(f"  Open:        {url}")
    print(f"  Diagnostics: {url}diagnose")
    print(f"  Bound to {host} only — not reachable from your network.")
    print("  Press Ctrl+C to stop.")
    print(bar, flush=True)

    uvicorn.run(
        create_app(), host=host, port=chosen, log_level="warning",
        # uvicorn's defaults are 20 seconds to ping and 20 more to give up.
        # Giving up means dropping the TCP connection without a close frame,
        # which is exactly the close code 1006 an operator sees when the link
        # lamp goes red -- and it is the wrong trade here. The default is sized
        # for a public server with thousands of peers, where a client that
        # stops answering is a resource leak. This server has one client, on
        # the loopback interface, where there is no network to lose packets on:
        # a late pong means one end was briefly too busy to answer, and tearing
        # the connection down blanks the terminal for something that was about
        # to recover.
        #
        # Still pinged, so a browser that really has gone is still reaped --
        # just not before the trading loop itself would be declared stalled.
        ws_ping_interval=20.0,
        ws_ping_timeout=float(LOOP_STALL_SECONDS),
    )
    return 0


def _open_when_ready(url: str, timeout: float = 20.0) -> None:
    """Open the browser only once the server actually answers.

    Opening on a timer races the server and shows the operator a connection
    error on a program that is about to work perfectly.
    """
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "api/health", timeout=1) as resp:
                if resp.status == 200:
                    webbrowser.open(url)
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    log.warning("the server did not answer within %.0fs; not opening a browser",
                timeout)
