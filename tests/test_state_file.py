"""One file, several writers, and none of them may erase the others.

The overnight save wrote its five keys as a fresh file. The same file carries
the Sector Trend sleeve's section -- its positions, its trailing stops and the
record of it arming -- so every overnight save deleted it, several times a day.
The sleeve's own docstring warned that "a forgotten stop is an unbounded
position".

Two of the three writers already merged correctly. The fix was to stop letting
each one implement the read-modify-write for itself.
"""

from __future__ import annotations

import json

import pytest

from imperium import config
from imperium.execution.sleeve_ledger import STATE_KEY, SleeveLedger
from imperium.session import TradingSession


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    return tmp_path


def _sleeve_with_a_stop() -> SleeveLedger:
    ledger = SleeveLedger()
    ledger.open_position("XLK", 3.0, 200.0, stop=190.0, day="2026-09-25")
    ledger.armed_at_equity = 500.0
    ledger.armed_by = "equity"
    return ledger


def test_an_overnight_save_keeps_the_sleeves_stops(home):
    """The bug itself, reproduced exactly as it happened."""
    _sleeve_with_a_stop().save()

    session = TradingSession()
    session.overnight_holdings["SPY"] = 0.1
    session._save_overnight_state()

    reloaded = SleeveLedger.load()
    assert "XLK" in reloaded.positions, (
        "an overnight save deleted the sleeve's positions")
    assert reloaded.positions["XLK"].stop == 190.0, (
        "the sleeve's trailing stop was lost")
    assert reloaded.armed_by == "equity", "the sleeve forgot it had armed"


def test_a_sleeve_save_keeps_the_overnight_book(home):
    session = TradingSession()
    session.overnight_holdings["SPY"] = 0.1
    session._save_overnight_state()

    _sleeve_with_a_stop().save()

    payload = config.read_state()
    assert payload.get("overnight_holdings") == {"SPY": 0.1}


def test_remembering_the_key_keeps_everything_else(home):
    _sleeve_with_a_stop().save()
    session = TradingSession()
    session.overnight_holdings["SPY"] = 0.1
    session._save_overnight_state()

    session._remember_attached("paper-key")

    payload = config.read_state()
    assert payload["attached"] == "paper-key"
    assert STATE_KEY in payload
    assert payload["overnight_holdings"] == {"SPY": 0.1}


def test_every_writer_goes_through_the_one_helper():
    """Guards the root cause rather than the instance of it. A new writer that
    opens the file and writes a dict is the same bug waiting to happen."""
    import inspect

    from imperium import session as session_module
    from imperium.execution import sleeve_ledger

    for module in (session_module, sleeve_ledger):
        source = inspect.getsource(module)
        assert "state_path().write_text" not in source, module.__name__
        assert "path.write_text(json.dumps" not in source, module.__name__


def test_a_failed_write_leaves_the_previous_file_intact(home, monkeypatch):
    """Written beside the real file and moved over it. A crash mid-write must
    leave the last good file, not a truncated one every reader treats as empty
    -- which would lose the whole book at once."""
    config.update_state(overnight_holdings={"SPY": 0.1})
    good = config.state_path().read_text(encoding="utf-8")

    import os

    def refuse(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError):
        config.update_state(overnight_holdings={"QQQ": 0.2})

    assert config.state_path().read_text(encoding="utf-8") == good


def test_a_corrupt_file_reads_as_empty_rather_than_raising(home):
    config.ensure_home()
    config.state_path().write_text("{not json", encoding="utf-8")
    assert config.read_state() == {}
    # And the next write replaces it cleanly.
    config.update_state(attached="k")
    assert json.loads(config.state_path().read_text(encoding="utf-8")) == {
        "attached": "k"}
