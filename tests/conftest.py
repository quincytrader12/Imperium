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
