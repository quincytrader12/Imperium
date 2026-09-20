"""The client a broker keeps after the session replaces it.

From a live run: "the trading loop raised RuntimeError: cannot send a request,
as the client has been closed". That is httpx refusing a closed AsyncClient,
and it repeated on every tick, because every tick reconciles the book through
the broker and the broker was holding a client the session had already closed.

LiveBroker takes its client at construction. attach_credential closes the old
client and builds a new one. Nothing re-pointed the broker, so it kept the
dead one -- and there was no way back short of restarting the terminal.
"""

from __future__ import annotations

import pytest

from imperium.execution.broker import LiveBroker, Mode
from imperium.session import TradingSession
from imperium.security.credentials import Credential


class _FakeStore:
    """The two methods attach_credential uses."""

    def __init__(self, cred: Credential) -> None:
        self._cred = cred

    def require(self, name: str) -> Credential:
        return self._cred


def _credential() -> Credential:
    return Credential(name="test", venue="alpaca", api_key="k",
                      secret="s", trade_enabled=False)


@pytest.mark.asyncio
async def test_replacing_the_credential_re_points_the_broker(monkeypatch):
    """The bug, reproduced at the level it actually happened.

    Not by asserting on httpx -- by asserting the broker is not left holding
    the object the session just closed. Any call through that object raises,
    and the one the trading loop makes every tick is the reconcile.
    """
    session = TradingSession()
    await session.attach_credential(_FakeStore(_credential()), "test")
    first = session.client
    assert first is not None

    # A live broker, built the way switch_mode builds one.
    session.broker = LiveBroker(session.spec, first, "test", mode=Mode.PAPER,
                                telemetry=session.telemetry)
    assert session.broker.client is first

    # Attach again -- reconnecting a key, or restoring one on start.
    await session.attach_credential(_FakeStore(_credential()), "test")

    assert session.client is not first, "the session did not build a new client"
    assert session.broker.client is session.client, (
        "the broker is still holding the client the session closed; every "
        "tick's reconcile would raise 'cannot send a request, as the client "
        "has been closed'")


@pytest.mark.asyncio
async def test_the_closed_client_is_not_the_one_left_in_use(monkeypatch):
    """Named separately because the two facts are independent.

    A version that handed the broker a *third* client would satisfy the test
    above and still be wrong.
    """
    closed: list[object] = []

    session = TradingSession()
    await session.attach_credential(_FakeStore(_credential()), "test")
    first = session.client

    async def record(self=first):
        closed.append(self)

    monkeypatch.setattr(type(first), "aclose", lambda self: record(self))
    session.broker = LiveBroker(session.spec, first, "test", mode=Mode.PAPER,
                                telemetry=session.telemetry)

    await session.attach_credential(_FakeStore(_credential()), "test")

    assert first in closed, "the old client was never closed"
    assert session.broker.client not in closed, (
        "the broker was re-pointed at a client that has been closed")


@pytest.mark.asyncio
async def test_a_simulated_broker_is_left_alone():
    """Only a broker that holds a client needs re-pointing, and reaching into
    one that does not is how an unrelated broker gets a stray attribute."""
    session = TradingSession()
    before = session.broker
    await session.attach_credential(_FakeStore(_credential()), "test")
    assert session.broker is before
    assert not hasattr(before, "client") or before.client is None
