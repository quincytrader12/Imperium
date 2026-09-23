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

import pytest

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


# -- knowing what time it is in New York, without 605 files ------------------


def test_the_computed_rule_matches_the_real_database_exactly():
    """The fallback has to be right, because it is what the .exe uses.

    Bundling the IANA database cost 605 files and six seconds of first-launch
    startup, and the build that shipped it did not answer inside the twenty
    seconds its own launcher waits. So US Eastern is computed from the
    statutory rule -- second Sunday in March to first Sunday in November --
    and this checks it against the database hour by hour rather than at the
    four transitions, because an off-by-one in the nth-Sunday arithmetic moves
    a boundary by a week and still passes a spot check.
    """
    import datetime as dt

    zoneinfo = pytest.importorskip("zoneinfo")
    from imperium.execution.sleeve_ledger import eastern_offset

    try:
        ny = zoneinfo.ZoneInfo("America/New_York")
    except zoneinfo.ZoneInfoNotFoundError:          # pragma: no cover
        pytest.skip("no IANA database on this machine to check against")

    moment = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2031, 1, 1, tzinfo=dt.timezone.utc)
    mismatches = []
    while moment < end:
        if moment.astimezone(ny).utcoffset() != eastern_offset(moment):
            mismatches.append(moment.isoformat())
        moment += dt.timedelta(hours=1)
    assert not mismatches, (
        f"{len(mismatches)} hours disagree with the real database, first at "
        f"{mismatches[0]}")


def test_the_transitions_land_on_the_right_sundays():
    """Named explicitly, so a failure says which end moved."""
    import datetime as dt

    from imperium.execution.sleeve_ledger import _nth_sunday

    # 2026: March 8 is the second Sunday, November 1 the first.
    assert _nth_sunday(2026, 3, 2) == 8
    assert _nth_sunday(2026, 11, 1) == 1
    # 2027: March 14 and November 7.
    assert _nth_sunday(2027, 3, 2) == 14
    assert _nth_sunday(2027, 11, 1) == 7
    # A month starting on a Sunday must not skip a week.
    assert _nth_sunday(2026, 2, 1) == 1
    assert dt.date(2026, 2, 1).weekday() == 6


# -- the whole path, on a real session ------------------------------------
#
# The gap that let the brief never arrive once. Every test above builds a
# Brief by hand and checks the text; not one of them asked the session to
# build its own. So a session attribute that did not exist -- `self.mode`,
# which is `self.broker.mode` -- raised out of _todays_brief, through the
# tick, into the trading loop's catch-all, where it read as "the trading loop
# raised AttributeError" and looked like anything at all. The day had already
# been claimed a line earlier, so it was never retried, and every other
# Telegram message kept arriving as normal.


import datetime as dt

import pytest

from imperium.session import TradingSession


class _Notifier:
    linked = True

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, text: str) -> bool:
        self.sent.append(text)
        return True


def _evening_session() -> tuple[TradingSession, _Notifier]:
    session = TradingSession()
    session.running = True
    notifier = _Notifier()
    session.notifier = notifier
    # 21:30 UTC is 17:30 in New York, which is when the brief is due.
    session.market_clock.timestamp = dt.datetime(2026, 9, 23, 21, 30,
                                                 tzinfo=dt.timezone.utc)
    return session, notifier


@pytest.mark.asyncio
async def test_the_session_can_actually_build_and_send_its_own_brief():
    session, notifier = _evening_session()
    await session._maybe_send_daily_brief()
    assert notifier.sent, "no brief was sent at half past five in the evening"
    assert "DAILY BRIEF" in notifier.sent[0]


@pytest.mark.asyncio
async def test_a_brief_that_cannot_be_built_does_not_take_the_tick_with_it():
    """The second half of the same fault. Raising out of here aborted the rest
    of the tick -- the account refresh, the reconcile, the sleeve's arming
    check -- and reported itself as a trading-loop failure."""
    session, notifier = _evening_session()

    def boom() -> None:
        raise ValueError("a number that is not there")

    session._todays_brief = boom

    await session._maybe_send_daily_brief()          # must not raise

    assert not notifier.sent
    said = [e for e in session.telemetry.events()
            if e.get("source") == "brief"]
    assert said, "the failure was swallowed without a word"
    assert "could not be built" in said[0]["message"], said[0]["message"]
    assert "ValueError" in said[0]["message"]


@pytest.mark.asyncio
async def test_it_is_still_only_sent_once_a_day():
    session, notifier = _evening_session()
    for _ in range(5):
        await session._maybe_send_daily_brief()
    assert len(notifier.sent) == 1, (
        f"{len(notifier.sent)} briefs in five ticks; this is the "
        f"repeat-notification bug the whole design exists to avoid")


@pytest.mark.asyncio
async def test_nothing_goes_out_before_the_close():
    session, notifier = _evening_session()
    # 14:30 UTC is 10:30 in New York: the day has not finished.
    session.market_clock.timestamp = dt.datetime(2026, 9, 23, 14, 30,
                                                 tzinfo=dt.timezone.utc)
    await session._maybe_send_daily_brief()
    assert not notifier.sent


@pytest.mark.asyncio
async def test_a_session_with_no_opening_mark_does_not_claim_the_account():
    """A session restarted before its first account read has no opening
    equity. Subtracting from zero reported the whole balance as the day's
    profit, with a green dot beside it."""
    session, notifier = _evening_session()
    session.day_start_equity = 0.0
    await session._maybe_send_daily_brief()

    said = notifier.sent[0]
    assert "no opening mark" in said, said
    assert "🟢" not in said.splitlines()[1], said
