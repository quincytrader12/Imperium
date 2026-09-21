"""Did the terminal stall, or did something else drop the link?

The link lamp went red with close code 1006 -- a socket that died without a
close frame from either end. On loopback there is no network between the two
ends, so one of the short list of things that can do that is this process
being too blocked to answer uvicorn's keepalive ping.

That was a theory, and it stayed a theory because nothing measured it. These
tests cover the measurement: sleep for a known interval, see how much longer
it actually took, and keep the worst case rather than an average -- an average
over hours hides exactly the event being looked for.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from imperium import loopwatch
from imperium.loopwatch import LoopWatch


def test_a_sleep_that_returned_on_time_is_not_lateness():
    w = LoopWatch(interval=1.0)
    assert w.observe(1.0) == 0.0
    assert w.worst == 0.0


def test_lateness_is_what_the_sleep_cost_beyond_the_interval():
    w = LoopWatch(interval=1.0)
    assert w.observe(1.25) == pytest.approx(0.25)
    assert w.last == pytest.approx(0.25)


def test_a_sleep_that_returned_early_is_not_negative_lateness():
    """Some platforms return from a sleep a hair early. Recorded as a negative
    number it would sit in ``worst`` forever and make it meaningless."""
    w = LoopWatch(interval=1.0)
    assert w.observe(0.998) == 0.0
    assert w.worst == 0.0


def test_the_worst_case_is_kept_not_the_last_one():
    w = LoopWatch(interval=1.0)
    w.observe(9.0)
    w.observe(1.01)
    assert w.worst == pytest.approx(8.0)
    assert w.last == pytest.approx(0.01)


def test_the_worst_case_is_stamped_so_it_can_be_lined_up_with_the_link():
    """A stall is only evidence if it can be matched against when the lamp
    went red."""
    w = LoopWatch(interval=1.0, _now=lambda: 1234.0)
    w.observe(7.0)
    assert w.worst_at == 1234.0


def test_ordinary_jitter_is_not_counted_as_a_stall():
    w = LoopWatch(interval=1.0, stall_after=5.0)
    for _ in range(50):
        w.observe(1.2)
    assert w.stalls == 0
    assert w.samples == 50


def test_a_stall_long_enough_to_drop_a_socket_is_counted_separately():
    """Under uvicorn's ping timeout it is a hiccup the operator never sees.
    Over it, the loop closed the link by itself, and that is the answer to the
    question this whole file exists for."""
    w = LoopWatch(interval=1.0, stall_after=5.0)
    w.observe(1.0 + 6.0)
    assert (w.stalls, w.socket_killers) == (1, 0)
    w.observe(1.0 + loopwatch.PING_TIMEOUT_SECONDS)
    assert (w.stalls, w.socket_killers) == (2, 1)


def test_the_verdict_says_no_stall_when_there_was_none():
    w = LoopWatch(interval=1.0)
    w.observe(1.03)
    assert "no stall" in w.verdict()


def test_the_verdict_names_a_stall_that_could_have_dropped_the_link():
    w = LoopWatch(interval=1.0)
    w.observe(1.0 + 25.0)
    said = w.verdict()
    assert "drop the link" in said, said


def test_nothing_measured_yet_says_so_rather_than_claiming_health():
    assert LoopWatch().verdict() == "not measured yet"


def test_the_report_is_json_safe():
    import json

    w = LoopWatch()
    w.observe(1.5)
    json.dumps(w.report())


# -- against a real event loop --------------------------------------------


@pytest.mark.asyncio
async def test_a_blocked_loop_is_actually_caught():
    """The measurement, end to end. A synchronous sleep inside a coroutine is
    the exact fault being looked for: nothing else on the loop can run, the
    process is idle rather than busy, and no traceback is produced anywhere."""
    seen: list[float] = []
    record = LoopWatch(interval=0.05, stall_after=0.2)
    task = asyncio.create_task(loopwatch.watch(record, on_stall=seen.append))
    try:
        await asyncio.sleep(0.12)
        time.sleep(0.45)            # the block
        await asyncio.sleep(0.2)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert record.samples > 0, "the watch never took a measurement"
    assert record.worst >= 0.3, (
        f"a 0.45s block on the event loop was not noticed: worst was "
        f"{record.worst:.3f}s")
    assert seen, "the stall was measured but nothing was told about it"


@pytest.mark.asyncio
async def test_a_loop_that_is_not_blocked_reports_no_stall():
    """The other half. A measurement that fires on a healthy loop would be
    worse than none -- it would send the next investigation the wrong way."""
    record = LoopWatch(interval=0.05, stall_after=0.2)
    task = asyncio.create_task(loopwatch.watch(record))
    try:
        for _ in range(8):
            await asyncio.sleep(0.05)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert record.stalls == 0, (
        f"an idle loop was reported as stalling; worst {record.worst:.3f}s")


@pytest.mark.asyncio
async def test_a_reporting_failure_does_not_kill_the_watch():
    """The watch is evidence for a fault that has already happened once. It
    must not be the thing that stops working next time."""
    def boom(lag: float) -> None:
        raise RuntimeError("no")

    record = LoopWatch(interval=0.05, stall_after=0.1)
    task = asyncio.create_task(loopwatch.watch(record, on_stall=boom))
    try:
        time.sleep(0.25)
        await asyncio.sleep(0.2)
        assert not task.done(), "the watch died with the handler"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# -- wired into the running server ----------------------------------------


def test_the_server_measures_its_own_loop_and_publishes_it():
    """Wired in, not merely written. The point of the measurement is that it
    is running in the build the operator has, before the next time the link
    goes red."""
    from fastapi.testclient import TestClient

    from imperium.server.app import create_app
    from imperium.session import TradingSession

    with TestClient(create_app(TradingSession())) as client:
        body = client.get("/api/snapshot").json()

    assert "loop_lag" in body, (
        "the terminal reports how old its trading loop's heartbeat is, but "
        "not whether the event loop under it can run at all")
    assert body["loop_lag"] is not None, "the watch was never started"
    assert set(body["loop_lag"]) >= {"worst_ms", "stalls", "socket_killers"}
    assert body["loop_lag_verdict"]


def test_a_session_without_a_server_says_it_has_not_measured():
    """Builds of the session outside a server -- every test in this suite,
    and the backtest -- must not claim a healthy loop they never watched."""
    from imperium.session import TradingSession

    body = TradingSession().snapshot()
    assert body["loop_lag"] is None
    assert body["loop_lag_verdict"] == ""


# -- the other end of a 1006 ----------------------------------------------


def test_the_server_does_not_drop_a_live_socket_over_a_late_pong(monkeypatch):
    """uvicorn's defaults are 20 seconds to ping and 20 more to give up, and
    giving up means dropping the TCP connection with no close frame -- which
    is the close code 1006 the operator saw.

    That default is sized for a public server with thousands of peers, where a
    client that stops answering is a resource leak. This one has a single
    client on the loopback interface, where there is no network to lose a
    packet on: a late pong means one end was briefly busy, and tearing the
    connection down blanks the terminal for something about to recover. It is
    still pinged, so a browser that really has gone is still reaped."""
    import uvicorn

    from imperium.server import app as app_module
    from imperium.session import LOOP_STALL_SECONDS

    seen: dict = {}

    def _fake_run(application, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr(app_module, "choose_port", lambda host, port: port)
    monkeypatch.setattr(app_module, "create_app", lambda: object())
    app_module.run_server(host="127.0.0.1", port=8765, open_browser=False)

    assert seen["ws_ping_interval"], "a peer that has gone must still be reaped"
    assert seen["ws_ping_timeout"] > 20.0, (
        "still on uvicorn's default, which closes a live local socket "
        "abruptly -- the 1006 this is about")
    assert seen["ws_ping_timeout"] == float(LOOP_STALL_SECONDS), (
        "the link should not be given up on sooner than the trading loop "
        "itself would be declared stalled")


def test_the_watchdog_does_not_need_a_browser():
    """A dead trading loop is the worst failure this program has: the session
    still says running, the health score stays green, and nothing evaluates a
    single bar. The watchdog for it used to run on the websocket's frame --
    so the moment the link went red, the one thing watching for that failure
    stopped watching, for as long as the link stayed down.

    Asserted on the running app rather than on the source, because what
    matters is that it ticks without a socket ever being opened."""
    import time as _time

    from fastapi.testclient import TestClient

    from imperium.server.app import create_app
    from imperium.session import TradingSession

    session = TradingSession()
    calls: list[float] = []
    original = session.supervise

    async def counted() -> None:
        calls.append(_time.time())
        await original()

    session.supervise = counted

    with TestClient(create_app(session)) as client:
        # No websocket is ever opened. Only an ordinary request, so the app is
        # up and its lifespan has run.
        assert client.get("/api/health").json()["ok"]
        _time.sleep(2.5)

    assert calls, (
        "the trading loop's watchdog never ran with no browser attached; a "
        "loop that dies while the link is red would stay dead")
