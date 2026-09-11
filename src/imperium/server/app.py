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
import ipaddress
import logging
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from imperium import config, logging_setup
from imperium.diagnostics.layers import NetworkDiagnostic
from imperium.notify import telegram
from imperium.execution.broker import LIVE_CONFIRMATION_PHRASE, Mode, ModeSwitchRefused
from imperium.security.credentials import CredentialError, CredentialStore
from imperium.session import TradingSession
from imperium.telemetry.streams import Level
from imperium.venues import registry
from imperium.venues.alpaca.client import VenueError

log = logging.getLogger("imperium.server")

#: Fixed cadence, not on-change. The cluster animates continuously, so a steady
#: frame rate is what the client needs, and it bounds the server's work
#: regardless of how busy the book is.
SNAPSHOT_HZ = 1.0


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

        if state.get("store_error"):
            state["session"].store_error = state["store_error"]
            state["session"].telemetry.event(
                Level.ERROR, "security",
                "the credentials file could not be loaded",
                detail=state["store_error"])
        yield
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
        return HTMLResponse(page.read_text(encoding=config.TEXT_ENCODING))

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
        await get_session().start()
        return JSONResponse({"running": True})

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
        try:
            while True:
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
                await ws.send_json(payload)
                # Checked on the frame the operator is already paying for. A
                # dead trading loop cannot notice itself, and every other
                # indicator on screen -- session running, feed live, health
                # green -- keeps saying the terminal is fine while it evaluates
                # nothing at all.
                with contextlib.suppress(Exception):
                    await session.supervise()
                elapsed = time.perf_counter() - start
                await asyncio.sleep(max(0.05, (1.0 / SNAPSHOT_HZ) - elapsed))
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("the websocket loop failed")
        finally:
            session.lamps.link = "off"

    app.mount("/static", StaticFiles(directory=str(static_dir())), name="static")
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

    uvicorn.run(create_app(), host=host, port=chosen, log_level="warning")
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
