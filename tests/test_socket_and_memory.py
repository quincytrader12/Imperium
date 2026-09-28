"""The websocket's disconnect handling, and remembering the attached key.

Two faults this file pins down, both of which reached an operator's screen:

* a closed browser tab printed a five-frame ``RuntimeError`` traceback at
  ERROR on every frame the server tried to send afterwards;
* every restart of the terminal required the Alpaca key to be pasted in
  again, because nothing recorded which stored key had been attached.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from imperium.server import app as app_mod
from imperium.server.app import create_app, _is_closed_socket
from imperium.session import TradingSession

KEY = "PK" + "A" * 62
SECRET = "S3cr3t" + "x" * 58


# -- the socket ----------------------------------------------------------

def test_the_ws_route_is_the_streaming_loop_and_not_a_helper():
    """Prevents: a helper defined between ``@app.websocket("/ws")`` and the
    coroutine it was meant to decorate, which silently registers the helper as
    the route. FastAPI then tries to build a request model from the helper's
    parameters and the whole app fails to construct -- so this is checked on
    the route table rather than on a response."""
    app = create_app(TradingSession())
    routes = {r.path: r for r in app.routes if getattr(r, "path", "") == "/ws"}
    assert "/ws" in routes, "the stream route is not registered"
    assert routes["/ws"].endpoint.__name__ == "stream"


def test_the_socket_delivers_a_snapshot_frame():
    """Prevents: a stream that accepts and then sends nothing."""
    with TestClient(create_app(TradingSession())) as client:
        with client.websocket_connect("/ws") as ws:
            frame = ws.receive_json()
    assert "ts" in frame
    assert "watchlist" in frame


@pytest.mark.parametrize("message", [
    "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'.",
    "Cannot call \"send\" once a close message has been sent.",
    "WebSocket is not connected. Need to call \"accept\" first.",
    "Unexpected ASGI message 'websocket.send', after sending "
    "'websocket.disconnect'.",
])
def test_a_closed_tab_is_recognised_rather_than_logged_as_a_fault(message):
    """Prevents: an operator refreshing the page and being shown a traceback.
    The server is given no typed disconnect on the send path -- uvicorn raises
    a bare RuntimeError -- so the message is all there is to match on."""
    assert _is_closed_socket(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "dictionary changed size during iteration",
    "cannot reuse already awaited coroutine",
    "Event loop is running",
])
def test_a_real_fault_on_the_send_path_is_still_raised(message):
    """Prevents: over-broad matching that swallows genuine bugs. The narrow
    read is the point: anything that is not a closed socket must still reach
    the log and the operator."""
    assert not _is_closed_socket(RuntimeError(message))


# -- remembering the key -------------------------------------------------

@pytest.mark.asyncio
async def test_attaching_a_key_records_its_name_for_the_next_run(tmp_path,
                                                                monkeypatch):
    """Prevents: the operator pasting the same Alpaca key in on every launch."""
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    from imperium import config
    from imperium.security.credentials import CredentialStore

    store = CredentialStore(tmp_path / "credentials.json")
    store.add("alpaca-paper", "alpaca", KEY, SECRET)

    session = TradingSession()
    await session.attach_credential(store, "alpaca-paper")

    assert TradingSession.remembered_credential() == "alpaca-paper"
    assert config.state_path().exists()
    await session.detach_client()


@pytest.mark.asyncio
async def test_remembering_a_key_does_not_lose_the_overnight_book(tmp_path,
                                                                  monkeypatch):
    """Prevents: the read-modify-write becoming an overwrite. The same file
    carries positions held across a restart; clobbering them to save a name
    would silently drop the book."""
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    from imperium import config
    from imperium.security.credentials import CredentialStore

    config.ensure_home()
    config.state_path().write_text(
        json.dumps({"overnight": {"AAPL": {"weight": 0.2}}}),
        encoding=config.TEXT_ENCODING)

    store = CredentialStore(tmp_path / "credentials.json")
    store.add("alpaca-paper", "alpaca", KEY, SECRET)

    session = TradingSession()
    await session.attach_credential(store, "alpaca-paper")

    saved = json.loads(config.state_path().read_text(
        encoding=config.TEXT_ENCODING))
    assert saved["attached"] == "alpaca-paper"
    assert saved["overnight"] == {"AAPL": {"weight": 0.2}}
    await session.detach_client()


def test_no_remembered_name_when_nothing_was_ever_attached(tmp_path,
                                                           monkeypatch):
    """Prevents: a first run trying to attach a key that does not exist."""
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    assert TradingSession.remembered_credential() == ""


def test_a_revoked_remembered_key_does_not_stop_the_terminal_starting(
        tmp_path, monkeypatch):
    """Prevents: the worst failure mode of remembering a key -- a key that has
    since been revoked or deleted taking the whole terminal down at startup.
    It must say so and carry on unattached."""
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    from imperium import config
    config.ensure_home()
    config.state_path().write_text(json.dumps({"attached": "long-gone"}),
                                   encoding=config.TEXT_ENCODING)

    session = TradingSession()
    with TestClient(create_app(session)) as client:
        assert client.get("/api/health").json()["ok"] is True
    assert session.credential is None
