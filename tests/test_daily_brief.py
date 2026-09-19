"""One message a day, and exactly one.

The repeat is the whole risk. Every notification bug this program has had was
a level-triggered check firing every cycle -- the same "BOOK HALTED, daily
loss 33.93%" five times, same figure, same minute -- and a daily brief that
does that is worse than no brief, because it teaches the operator to ignore
the one notice they were meant to read.

So most of this file is about the guard rather than the text.
"""

from __future__ import annotations

import datetime as dt
import json

from imperium import config
from imperium.notify import daily


def _at(hour: int, day: int = 19) -> dt.datetime:
    return dt.datetime(2026, 9, day, hour, 30)


# -- the guard --------------------------------------------------------------


def test_it_is_not_due_before_the_close():
    """A brief written at noon reports a day that has not happened yet."""
    assert not daily.is_due(last_sent_day="", now=_at(11), today="2026-09-19")


def test_it_is_due_after_the_close():
    assert daily.is_due(last_sent_day="", now=_at(17), today="2026-09-19")


def test_it_is_not_due_twice_in_a_day():
    """The one that matters. Every later tick of the same evening must be
    silent, or the operator gets a brief a second for the rest of the night."""
    for hour in (17, 18, 21, 23):
        assert not daily.is_due(last_sent_day="2026-09-19",
                                now=_at(hour), today="2026-09-19")


def test_it_is_due_again_the_next_day():
    assert daily.is_due(last_sent_day="2026-09-19",
                        now=_at(17, day=20), today="2026-09-20")


def test_the_guard_survives_a_restart(tmp_path, monkeypatch):
    """The reason the day is written to disk.

    IMPERIUM is started by a batch file that relaunches it whenever it exits.
    A guard held only in memory answers "have I sent one?" with "no" after
    every crash, every restart and every overnight reboot, and the operator
    gets a fresh brief each time -- which is precisely the failure this whole
    file exists to prevent.
    """
    from imperium.session import TradingSession

    monkeypatch.setattr(config, "home_dir", lambda: tmp_path)

    first = TradingSession()
    first._brief_sent_day = "2026-09-19"
    first._save_overnight_state()

    saved = json.loads(config.state_path().read_text(encoding="utf-8"))
    assert saved["brief_sent_day"] == "2026-09-19", (
        "the day was not persisted, so a restart would send a second brief")

    after_restart = TradingSession()
    after_restart._load_overnight_state()
    assert after_restart._brief_sent_day == "2026-09-19"
    assert not daily.is_due(last_sent_day=after_restart._brief_sent_day,
                            now=_at(20), today="2026-09-19")


def test_a_state_file_without_the_key_does_not_crash(tmp_path, monkeypatch):
    """Upgrading into this feature must not fail to start."""
    from imperium.session import TradingSession

    monkeypatch.setattr(config, "home_dir", lambda: tmp_path)
    config.ensure_home()
    config.state_path().write_text(
        json.dumps({"overnight_holdings": {}}), encoding="utf-8")

    session = TradingSession()
    session._load_overnight_state()
    assert session._brief_sent_day == ""


# -- the dots ---------------------------------------------------------------


def test_a_winner_is_green_and_a_loser_is_red():
    assert daily.dot(7.79) == daily.GREEN
    assert daily.dot(-0.01) == daily.RED


def test_a_position_that_has_not_moved_is_neither():
    """Colouring a flat position green would be the message making a claim the
    number does not support."""
    assert daily.dot(0.0) == daily.FLAT
    assert daily.dot(float("nan")) == daily.FLAT


def test_every_position_carries_its_own_dot():
    text = daily.build(daily.Brief(
        day="Fri 19 Sep", equity=78.78, day_start_equity=74.84, cash=27.13,
        positions=[daily.Position("FILUSD", 31.94, 7.79, 24.94),
                   daily.Position("AG", 19.84, -0.01, 19.85)]))
    for line in text.splitlines():
        if line.startswith((daily.GREEN, daily.RED, daily.FLAT)):
            continue
        assert "FILUSD" not in line and "AG " not in line, (
            f"a position line with no dot: {line!r}")
    assert daily.GREEN in text and daily.RED in text


def test_the_days_own_result_gets_a_dot_too():
    down = daily.build(daily.Brief(day="x", equity=70.0, day_start_equity=80.0,
                                   cash=70.0))
    assert down.splitlines()[1].startswith(daily.RED)


# -- the text ---------------------------------------------------------------


def test_losers_are_listed_first():
    """A brief is read from the top on a phone, and the position that needs a
    decision is the one losing money."""
    text = daily.build(daily.Brief(
        day="x", equity=100.0, day_start_equity=100.0, cash=10.0,
        positions=[daily.Position("WIN", 50.0, 9.0, 41.0),
                   daily.Position("LOSE", 40.0, -6.0, 46.0)]))
    assert text.index("LOSE") < text.index("WIN")


def test_an_empty_book_says_so_rather_than_printing_a_heading():
    text = daily.build(daily.Brief(day="x", equity=70.0,
                                   day_start_equity=70.0, cash=70.0))
    assert "No open positions." in text
    assert "Nothing traded today." in text


def test_a_halt_is_reported_in_the_brief():
    text = daily.build(daily.Brief(
        day="x", equity=70.0, day_start_equity=74.0, cash=70.0,
        activity=daily.Activity(halted=True, halt_reason="daily loss 5.4%")))
    assert "halted" in text.lower() and "daily loss 5.4%" in text


def test_the_message_carries_no_markdown_that_telegram_could_reject():
    """Telegram refuses a message with an unbalanced underscore or asterisk,
    and ticker symbols contain both. A formatted brief is one odd ticker away
    from silently failing to send."""
    text = daily.build(daily.Brief(
        day="x", equity=100.0, day_start_equity=99.0, cash=1.0,
        positions=[daily.Position("BRK.B", 50.0, 1.0, 49.0),
                   daily.Position("A_B*C", 20.0, -1.0, 21.0)]))
    assert "**" not in text and "__" not in text
    assert "`" not in text


def test_a_position_with_no_basis_shows_no_percentage():
    """Better a missing number than an invented one."""
    text = daily.build(daily.Brief(
        day="x", equity=100.0, day_start_equity=100.0, cash=50.0,
        positions=[daily.Position("X", 50.0, 5.0, basis=0.0)]))
    assert "%" not in text.split("Positions")[1].split("Activity")[0]
