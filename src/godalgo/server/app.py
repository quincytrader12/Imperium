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

from godalgo import config, logging_setup
from godalgo.diagnostics.layers import NetworkDiagnostic
from godalgo.execution.broker import LIVE_CONFIRMATION_PHRASE, Mode, ModeSwitchRefused
from godalgo.security.credentials import CredentialError, CredentialStore
from godalgo.session import TradingSession
from godalgo.telemetry.streams import Level
from godalgo.venues import registry
from godalgo.venues.binance.client import VenueError

log = logging.getLogger("godalgo.server")

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
        candidate = base / "godalgo" / "server" / "static"
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

class AddKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    venue: str = "binance_spot"
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
        try:
            state["store"] = CredentialStore()
        except CredentialError as exc:
            state["store"] = None
            state["store_error"] = exc
        else:
            report = state["store"].permission_report
            if not report.ok:
                state["session"].telemetry.event(
                    Level.ERROR, "security",
                    f"INSECURE CREDENTIAL FILE: {report.detail}",
                    detail=report.remedy)
        yield
        await state["session"].stop()
        await state["session"].detach_client()

    app = FastAPI(title="GODALGO", lifespan=lifespan, docs_url=None, redoc_url=None)

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
        store = get_store()
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
        return JSONResponse({"ok": True, "version": __import__("godalgo").__version__})

    # -- the stream ------------------------------------------------------

    @app.websocket("/ws")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        session = get_session()
        session.lamps.link = "ok"
        try:
            while True:
                start = time.perf_counter()
                try:
                    payload = session.snapshot()
                except Exception as exc:
                    # A snapshot that raises must not close the socket, or the
                    # UI goes dark for a reason it cannot show.
                    log.exception("snapshot failed")
                    payload = {"ts": time.time(), "error": str(exc)}
                await ws.send_json(payload)
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
    """True if nothing is already listening on this address."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
    print("  GODALGO — built by Quincy Gininda")
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
