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
    from imperium.venues.binance.client import BinanceSpotClient
    from mock_venue import API_KEY, SECRET

    return BinanceSpotClient(API_KEY, SECRET, transport=venue.transport,
                             max_retries=0)
