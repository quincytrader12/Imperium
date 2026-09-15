from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point the whole application at a temporary configuration directory.

    Autouse and unconditional: a test that writes to the operator's real
    ``~/.imperium/credentials.json`` would destroy their keys, and that must not
    depend on a test remembering to opt in.
    """
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path / "home"))
    yield


@pytest.fixture
def venue():
    from mock_venue import MockVenue

    return MockVenue()


@pytest.fixture
def client(venue):
    from imperium.venues.alpaca.client import AlpacaClient
    from mock_venue import KEY, SECRET

    return AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport,
                        max_retries=0)


@pytest.fixture(autouse=True)
def offline_news(monkeypatch):
    """No test reaches the real news feed.

    Autouse and unconditional, for the same reason ``isolated_home`` is. The
    default news source is a public RSS feed on the open internet; a suite that
    touches it is a suite whose result depends on somebody else's uptime, and
    one that would hammer a stranger's server every time it ran. Tests that
    exercise the source pass their own transport and are unaffected by this.
    """
    import httpx
    from imperium.execution import newsdesk as newsdesk_mod

    empty = httpx.MockTransport(
        lambda request: httpx.Response(200, text="<rss><channel></channel></rss>"))
    original = newsdesk_mod.yahoo_source

    def offline(*, transport=None):
        return original(transport=transport or empty)

    monkeypatch.setattr(newsdesk_mod, "yahoo_source", offline)
    yield


@pytest.fixture(autouse=True)
def offline_fx(monkeypatch):
    """No test reaches the real exchange-rate service.

    The same rule as ``offline_news`` and for the same reason: the rate comes
    from a public endpoint on the open internet, and a suite that touches it
    depends on a stranger's uptime and hits their server on every run. No test
    reaches it today -- the ones that exercise ``FxDesk`` pass their own
    transport -- but the guard is here so that the first test which calls
    ``refresh`` or runs the trading loop cannot quietly start doing so.

    The guard is on ``FxDesk.refresh`` and not on ``httpx``. The first version
    of this patched ``AsyncClient`` through ``fx.httpx``, which is not a local
    alias -- it is the httpx module -- so it forced a mock transport into every
    client the whole program builds, and four tests in two unrelated files
    started failing. A guard that reaches beyond the thing it is guarding is
    worse than no guard.
    """
    from imperium.venues import fx as fx_mod

    original = fx_mod.FxDesk.refresh

    async def guarded(self):
        if self.enabled and not self.manual and self.transport is None:
            raise AssertionError(
                "a test tried to fetch a live exchange rate; pass a transport "
                "to FxDesk instead")
        await original(self)

    monkeypatch.setattr(fx_mod.FxDesk, "refresh", guarded)
    yield


@pytest.fixture(autouse=True)
def clean_settings_environment():
    """No test may leak a setting into the next one.

    ``config.load_settings`` writes into the real ``os.environ``, which is
    process-wide and outlives the test that caused it. Without this, a test
    that enables the sector sleeve silently arms it for every test that runs
    afterwards, and a test asserting the default behaviour fails or passes
    depending on alphabetical order — which is exactly the kind of failure
    that gets "fixed" by weakening the assertion.

    Autouse and unconditional, like ``isolated_home``, and for the same
    reason: correctness here must not depend on a test remembering to opt in.
    """
    import os

    from imperium import config

    watched = set(config.SETTABLE) | {"IMPERIUM_HOME"}
    before = {name: os.environ.get(name) for name in watched}
    for name in config.SETTABLE:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
