"""The turn-of-the-month sleeve: the calendar it trades on, and the trade.

What it has to get right is the calendar: which day is a month's
second-to-last *trading* day is decided by the exchange's holidays, and a
window that starts a day wrong is a window that misses its best day.
"""

from __future__ import annotations

import datetime as dt
import time
from decimal import Decimal

import numpy as np
import pytest

from imperium.execution import sleeves as sl
from imperium.strategy import turn_of_month as tom

D = dt.date


@pytest.fixture(autouse=True)
def _sleeve_on(monkeypatch):
    monkeypatch.setenv("TURN_OF_MONTH_ENABLED", "true")


# -- the calendar -------------------------------------------------------------------


def test_the_2026_and_2027_holidays_are_the_exchanges():
    assert tom.nyse_holidays(2026) == {
        D(2026, 1, 1), D(2026, 1, 19), D(2026, 2, 16), D(2026, 4, 3),
        D(2026, 5, 25), D(2026, 6, 19), D(2026, 7, 3), D(2026, 9, 7),
        D(2026, 11, 26), D(2026, 12, 25)}
    assert tom.nyse_holidays(2027) == {
        D(2027, 1, 1), D(2027, 1, 18), D(2027, 2, 15), D(2027, 3, 26),
        D(2027, 5, 31), D(2027, 6, 18), D(2027, 7, 5), D(2027, 9, 6),
        D(2027, 11, 25), D(2027, 12, 24)}


def test_the_rules_edge_cases():
    # A Saturday New Year is not taken on the Friday before.
    assert D(2021, 12, 31) not in tom.nyse_holidays(2021)
    assert not any(d.month == 1 and d.day < 3 for d in tom.nyse_holidays(2022))
    assert all(d.year == 2022 for d in tom.nyse_holidays(2022))
    # Thanksgiving is the fourth Thursday, not the last: 2029 has five.
    assert D(2029, 11, 22) in tom.nyse_holidays(2029)
    assert D(2029, 11, 29) not in tom.nyse_holidays(2029)
    # A Sunday New Year is taken on the Monday.
    assert D(2023, 1, 2) in tom.nyse_holidays(2023)
    # Juneteenth only from 2022.
    assert D(2021, 6, 18) not in tom.nyse_holidays(2021)
    # Good Friday follows Easter.
    for good_friday in (D(2024, 3, 29), D(2025, 4, 18), D(2019, 4, 19)):
        assert good_friday in tom.nyse_holidays(good_friday.year)


def test_trading_days_skip_weekends_and_holidays():
    september = tom.trading_days(2026, 9)
    assert D(2026, 9, 7) not in september and D(2026, 9, 5) not in september
    assert len(september) == 21 and september[-1] == D(2026, 9, 30)


# -- the window -----------------------------------------------------------------------


@pytest.mark.parametrize("day,where", [
    (D(2026, 9, 28), None),      # -3: not yet
    (D(2026, 9, 29), -2),        # entry at the close
    (D(2026, 9, 30), -1),
    (D(2026, 10, 1), 1),
    (D(2026, 10, 2), 2),
    (D(2026, 10, 5), 3),         # a Monday: the weekend is not a trading day
    (D(2026, 10, 6), None),
    (D(2026, 10, 3), None),      # Saturday
])
def test_the_window_counts_trading_days(day, where):
    assert tom.offset(day) == where


def test_the_window_across_thanksgiving_and_the_new_year():
    # Thanksgiving Thursday is shut; the Friday after it is day -2.
    assert tom.offset(D(2026, 11, 27)) == -2
    assert tom.offset(D(2026, 11, 30)) == -1
    # The first of January is shut; the window's +1 is the fourth.
    assert tom.offset(D(2026, 12, 30)) == -2
    assert tom.offset(D(2027, 1, 4)) == 1
    assert tom.offset(D(2027, 1, 6)) == 3


def test_held_from_minus_two_to_the_close_of_plus_three():
    held = [d for d in (D(2026, 9, 28) + dt.timedelta(days=i) for i in range(10))
            if tom.targets(d).weights]
    assert held == [D(2026, 9, 29), D(2026, 9, 30), D(2026, 10, 1), D(2026, 10, 2)]
    out = tom.targets(D(2026, 10, 5))
    assert not out.weights and "day +3" in out.reasons["IVV"]
    # 31 October 2026 is a Saturday: the last trading day is Friday the 30th.
    assert out.note == "in cash until the close of Thu 29 Oct"


def test_the_reasons_name_the_dates():
    t = tom.targets(D(2026, 9, 28))
    assert t.reasons["IVV"] == "outside the window; next entry Tue 29 Sep"
    t = tom.targets(D(2026, 9, 29))
    assert t.weights == {"IVV": 1.0}
    assert "day -2" in t.reasons["IVV"] and "Mon 5 Oct" in t.reasons["IVV"]
    assert tom.targets(D(2026, 10, 2)).note == "in the window; out at the close of Mon 5 Oct"
    assert tom.next_entry(D(2026, 12, 31)) == D(2027, 1, 28)
    assert tom.next_entry(D(2026, 9, 29)) == D(2026, 9, 29), "the day itself counts"
    assert tom.exit_day(D(2026, 10, 5)) == D(2026, 10, 5), "the day itself counts"
    assert tom.exit_day(D(2026, 12, 31)) == D(2027, 1, 6)


def test_its_symbol_is_not_the_mean_reversion_sleeves():
    from imperium.strategy import mean_reversion
    assert not set(tom.UNIVERSE) & set(mean_reversion.UNIVERSE)


# -- when it runs ------------------------------------------------------------------------


def test_on_an_early_close_the_decision_moves_before_the_bell():
    s = next(x for x in sl.build_all() if x.name == "turn_of_month")
    close = dt.datetime(2026, 11, 27, 13, 0)
    assert not s.due(dt.datetime(2026, 11, 27, 12, 44), True, close)
    assert s.due(dt.datetime(2026, 11, 27, 12, 45), True, close)
    # An ordinary day: the configured time stands.
    close = dt.datetime(2026, 9, 29, 16, 0)
    assert not s.due(dt.datetime(2026, 9, 29, 15, 44), True, close)
    assert s.due(dt.datetime(2026, 9, 29, 15, 45), True, close)
    # A close on another day -- a stale clock -- says nothing about today.
    yesterday = dt.datetime(2026, 9, 28, 16, 0)
    assert not s.due(dt.datetime(2026, 9, 29, 9, 31), True, yesterday)


# -- in the session ------------------------------------------------------------------------


class _Client:
    authenticated = True

    async def bars(self, symbols, **kw):
        return {s: [{"c": 600.0 + i} for i in range(10)] for s in symbols}


def _session(equity=73.0):
    from imperium.execution.broker import PaperBroker
    from imperium.session import TradingSession
    from imperium.venues import registry

    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal(str(equity))
    session._attribution_loaded = True
    return session


@pytest.mark.asyncio
async def test_in_on_minus_two_and_out_on_plus_three_at_73_dollars():
    session = _session(73.0)
    session.client = _Client()
    q = session.feed.quote("IVV")
    q.last, q.updated_at = 610.0, time.time()
    sleeve = next(x for x in session.sleeves if x.name == "turn_of_month")
    assert session.reserved_symbols()["IVV"] == "Turn of the month"

    await session._plan_sleeve(sleeve, "2026-09-29")
    session.client = None
    await session._tick()
    ivv = session.broker.position("IVV")
    assert float(ivv.quantity) * 610.0 == pytest.approx(73.0 * 0.15, rel=0.05)

    # Held through the window without trading again...
    session.client = _Client()
    await session._plan_sleeve(sleeve, "2026-10-01")
    assert sleeve.pending == []
    session.client = None
    await session._tick()
    # ...and out on +3.
    session.client = _Client()
    q.updated_at = time.time()
    await session._plan_sleeve(sleeve, "2026-10-05")
    assert [o.side for o in sleeve.pending] == ["sell"]
    session.client = None
    await session._tick()
    assert session.broker.position("IVV").is_flat
    rows = session.trade_journal.read()
    assert len(rows) == 2 and all(r.strategy == "turn_of_month" for r in rows)
