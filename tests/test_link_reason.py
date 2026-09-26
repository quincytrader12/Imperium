"""Which end let go of the websocket.

From three days of a live run, several thousand times:

    link closed after 214s (the client said goodbye)

It could not have said anything else. The close-down path cancels the reader
task and awaits it, which runs the reader's ``finally`` and sets the event --
and the event was then what the message was built from. The field meant to
settle where the link drops come from was hardcoded true by its own cleanup,
so three days of evidence turned out to be three days of one sentence.

The reason now comes from what the receive side actually recorded, and the
recording is what is tested here: a cancellation must leave it empty, or the
same bug is back under a different variable name.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient

from imperium.server.app import create_app, link_reason
from imperium.session import TradingSession


def test_a_real_disconnect_is_reported_as_the_client():
    said = link_reason({"code": 1001})
    assert "client closed it" in said and "1001" in said, said


def test_a_receive_failure_says_so_rather_than_blaming_the_client():
    said = link_reason({"error": "RuntimeError: boom"})
    assert "receive side failed" in said, said
    assert "RuntimeError" in said, said


def test_nothing_recorded_means_this_end_let_go_first():
    """The case the old message could never report, and the one that matters:
    a socket dropped underneath both ends looks exactly like this, and calling
    it a polite goodbye sends the next investigation to the wrong machine."""
    said = link_reason({})
    assert "this end let go first" in said, said
    assert "client" not in said, said


@pytest.mark.asyncio
async def test_cancelling_the_reader_records_no_farewell():
    """The mechanism of the bug itself. Closing down cancels the reader and
    awaits it, which runs its cleanup -- and the cleanup must leave nothing
    behind that reads as the client having spoken."""
    farewell: dict = {}
    started = asyncio.Event()

    async def drain() -> None:
        try:
            started.set()
            await asyncio.sleep(3600)          # stands in for ws.receive()
        except asyncio.CancelledError:
            raise
        except Exception as exc:               # noqa: BLE001
            farewell["error"] = f"{type(exc).__name__}: {exc}"

    reader = asyncio.create_task(drain())
    await started.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader

    assert farewell == {}, (
        f"the close-down path wrote {farewell}, so the reason is being read "
        f"off this end\'s own cleanup again")
    assert "this end let go first" in link_reason(farewell)


# -- and against a running server -----------------------------------------


def _link_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if "link #" in r.getMessage()]


def test_a_client_that_closes_is_reported_as_the_client(caplog):
    caplog.set_level(logging.INFO, logger="imperium.server")
    with TestClient(create_app(TradingSession())) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()

    said = _link_lines(caplog)
    assert said, "the close was not logged at all"
    assert "the client closed it" in said[-1], said[-1]


def test_the_log_counts_the_links_that_are_still_open(caplog):
    """Two tabs left open look exactly like one tab dropping twice as often,
    and a log of durations alone cannot tell them apart."""
    caplog.set_level(logging.INFO, logger="imperium.server")
    with TestClient(create_app(TradingSession())) as client:
        with client.websocket_connect("/ws") as first:
            first.receive_json()
            with client.websocket_connect("/ws") as second:
                second.receive_json()

    said = _link_lines(caplog)
    assert said, "nothing was logged"
    assert "1 still open" in said[0], said[0]


def test_each_link_is_numbered_so_they_can_be_told_apart(caplog):
    caplog.set_level(logging.INFO, logger="imperium.server")
    with TestClient(create_app(TradingSession())) as client:
        for _ in range(2):
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()

    said = _link_lines(caplog)
    assert len(said) >= 2
    assert "link #1" in said[0], said[0]
    assert "link #2" in said[1], said[1]


def test_the_handler_reads_the_recording_and_not_its_own_cleanup():
    """A source-level guard, and deliberately so.

    The behavioural version of this test -- drive the handler until *it* lets
    go, then read the log -- deadlocks under Starlette's TestClient, which has
    no way to observe a server-initiated failure on a websocket without the
    client blocking on a frame that will never come. Two attempts at it hung
    for the full timeout.

    So this asserts the one thing that was actually wrong: the reason must be
    built from what the receive side recorded, and the close-down path's own
    event must not be what decides it. That event is set by the cleanup that
    runs immediately before this line, which is why it read true on every
    close for three days.
    """
    import inspect

    from imperium.server import app as app_module

    source = inspect.getsource(app_module.create_app)
    tail = source[source.index("link #"):]
    assert "link_reason(farewell)" in source, (
        "the close message no longer comes from what the receive side saw")
    assert "gone.is_set()" not in tail, (
        "the close message is being decided by an event the close-down path "
        "sets itself, which is the bug this replaces")

    drain = source[source.index("async def drain"):source.index("reader =")]
    # Just the cancellation branch: the one after it legitimately records a
    # receive failure, and slicing to the end of the function swept that in.
    after = drain[drain.index("except asyncio.CancelledError"):]
    cancelled = after[:after.index("except Exception")]
    assert "farewell[" not in cancelled, (
        "the cancellation path writes a farewell, so this end closing will "
        "again be reported as the client having spoken")
