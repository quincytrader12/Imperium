"""The mean-reversion sleeve: buy a sharp dip in an uptrend, sell the bounce.

What makes it safe on a $73 account: it only buys inside a long-run uptrend,
it always leaves -- on the bounce or on the time stop, counted from the entry
date so a day the terminal was closed still counts -- it holds at most two,
and it decides on today's price, not yesterday's close.
"""

from __future__ import annotations

import datetime as dt
import time
from decimal import Decimal

import numpy as np
import pytest

from imperium.execution import sleeves as sl
from imperium.strategy import mean_reversion as mr


@pytest.fixture(autouse=True)
def _sleeve_on(monkeypatch):
    monkeypatch.setenv("MEAN_REVERSION_ENABLED", "true")


def _uptrend(n=260, start=100.0, end=150.0):
    return np.linspace(start, end, n)


def _dip(series, *falls):
    out = list(series)
    for f in falls:
        out.append(out[-1] * (1 - f))
    return np.asarray(out)


def _sleeve():
    return next(s for s in sl.build_all() if s.name == "mean_reversion")


# -- the indicator --------------------------------------------------------------


def test_rsi_is_wilders_smoothing_worked_by_hand():
    # Changes +1 -.5 +1 -.5 +1. Seed 0.5/0.25, then Wilder's (prev + x) / 2:
    # gains .75 .375 .6875, losses .125 .3125 .15625 -> RS 4.4 -> 81.48.
    closes = np.array([10, 11, 10.5, 11.5, 11, 12], dtype=float)
    assert mr.rsi(closes, 2) == pytest.approx(100 - 100 / 5.4)
    assert mr.rsi(np.linspace(1, 2, 10)) == 100.0
    assert mr.rsi(np.full(10, 5.0)) == 50.0
    assert np.isnan(mr.rsi(np.array([1.0, 2.0])))


def test_two_sharp_down_days_are_deeply_oversold():
    assert mr.rsi(_dip(_uptrend(), 0.03, 0.03)) < 5


# -- entries ----------------------------------------------------------------------


def test_a_sharp_dip_inside_an_uptrend_is_bought():
    t = mr.targets({"SPY": _dip(_uptrend(), 0.03, 0.03)}, set(), {})
    assert t.weights == {"SPY": pytest.approx(1 / mr.MAX_POSITIONS)}
    assert "oversold in an uptrend" in t.reasons["SPY"]
    assert t.held_days == {"SPY": 0}


def test_a_dip_below_the_long_average_is_not_a_dip():
    falling = _dip(_uptrend(start=150, end=100), 0.03, 0.03)
    t = mr.targets({"SPY": falling}, set(), {})
    assert not t.weights and "below its 200-day" in t.reasons["SPY"]
    assert "in cash" in t.note


def test_an_ordinary_day_in_an_uptrend_is_not_bought():
    t = mr.targets({"SPY": _uptrend()}, set(), {})
    assert not t.weights and "not oversold" in t.reasons["SPY"]


def test_the_threshold_is_strict():
    series = _dip(_uptrend(), 0.03, 0.03)
    value = mr.rsi(series)
    assert value < mr.ENTRY_RSI
    old = mr.ENTRY_RSI
    try:
        mr.ENTRY_RSI = value
        assert not mr.targets({"SPY": series}, set(), {}).weights
    finally:
        mr.ENTRY_RSI = old


def test_too_little_history_is_never_bought():
    short = _dip(_uptrend(n=mr.MIN_BARS - 3), 0.03, 0.03)
    assert short.size == mr.MIN_BARS - 1
    assert not mr.targets({"SPY": short}, set(), {}).weights
    assert mr.targets({"SPY": _dip(_uptrend(n=mr.MIN_BARS - 2), 0.03, 0.03)},
                      set(), {}).weights


def test_at_most_two_and_the_most_oversold_first():
    closes = {"SPY": _dip(_uptrend(), 0.02, 0.02),
              "QQQ": _dip(_uptrend(), 0.05, 0.05),
              "IWM": _dip(_uptrend(), 0.03, 0.03)}
    t = mr.targets(closes, set(), {})
    assert set(t.weights) == {"QQQ", "IWM"}
    assert "already holds 2" in t.reasons["SPY"]
    # One slot taken by a position still waiting: only the deepest gets in.
    closes["DIA"] = _dip(_uptrend(), 0.01, 0.01)          # held, not bounced
    t = mr.targets(closes, {"DIA"}, {"DIA": 2})
    assert set(t.weights) == {"DIA", "QQQ"}
    assert sum(t.weights.values()) == pytest.approx(1.0)


# -- exits ------------------------------------------------------------------------


def test_the_bounce_is_sold():
    bounced = np.append(_dip(_uptrend(), 0.03, 0.03), 150.0)
    t = mr.targets({"SPY": bounced}, {"SPY"}, {"SPY": 1})
    assert "SPY" not in t.weights and "bounced" in t.reasons["SPY"]
    assert "SPY" not in t.held_days


def test_a_close_at_its_average_is_not_yet_the_bounce():
    level = np.append(_uptrend(), [140.0] * mr.EXIT_AVERAGE_DAYS)
    t = mr.targets({"SPY": level}, {"SPY"}, {"SPY": 2})
    assert "SPY" in t.weights, "the bounce is a close above the average"


def test_a_position_waiting_for_its_bounce_is_kept_and_counted():
    waiting = _dip(_uptrend(), 0.03, 0.03)
    t = mr.targets({"SPY": waiting}, {"SPY"}, {"SPY": 3})
    assert t.weights == {"SPY": pytest.approx(0.5)}
    assert "day 3 of at most 10" in t.reasons["SPY"]
    assert t.held_days == {"SPY": 3}


def test_the_time_stop_ends_a_trade_that_never_bounced():
    waiting = _dip(_uptrend(), 0.03, 0.03)
    kept = mr.targets({"SPY": waiting}, {"SPY"}, {"SPY": mr.MAX_HOLD_DAYS - 1})
    assert "SPY" in kept.weights
    out = mr.targets({"SPY": waiting}, {"SPY"}, {"SPY": mr.MAX_HOLD_DAYS})
    assert "SPY" not in out.weights and "time stop" in out.reasons["SPY"]


def test_a_holding_is_never_sold_on_missing_data():
    t = mr.targets({}, {"SPY"}, {"SPY": 4})
    assert "SPY" in t.weights and "no fresh history" in t.reasons["SPY"]


def test_an_exit_frees_its_slot_for_todays_dip_the_same_day():
    closes = {"SPY": np.append(_dip(_uptrend(), 0.03, 0.03), 150.0),
              "DIA": _dip(_uptrend(), 0.01, 0.01),
              "QQQ": _dip(_uptrend(), 0.04, 0.04)}
    t = mr.targets(closes, {"SPY", "DIA"}, {"SPY": 2, "DIA": 2})
    assert set(t.weights) == {"DIA", "QQQ"}


# -- the sleeve's memory ------------------------------------------------------------


def test_trading_days_are_counted_from_the_entry_date():
    assert sl.trading_days_between("2026-09-25", "2026-09-28") == 1   # Fri -> Mon
    assert sl.trading_days_between("2026-09-28", "2026-09-28") == 0
    assert sl.trading_days_between("2026-09-14", "2026-09-28") == 10
    assert sl.trading_days_between("garbage", "2026-09-28") == 0
    assert sl.trading_days_between("2026-09-28", "2026-09-25") == 0


def test_the_entry_date_is_remembered_and_the_time_stop_uses_it():
    sleeve = _sleeve()
    waiting = {"SPY": _dip(_uptrend(), 0.03, 0.03)}
    sleeve.decide(waiting, sleeve.memory, "2026-09-14", set())
    assert sleeve.memory["entered"] == {"SPY": "2026-09-14"}
    # Days the terminal was closed still count: nine weekdays on, kept...
    t = sleeve.decide(waiting, sleeve.memory, "2026-09-25", {"SPY"})
    assert "SPY" in t.weights and sleeve.memory["entered"] == {"SPY": "2026-09-14"}
    # ...ten, and out, and forgotten.
    t = sleeve.decide(waiting, sleeve.memory, "2026-09-28", {"SPY"})
    assert "SPY" not in t.weights and "time stop" in t.reasons["SPY"]
    assert sleeve.memory["entered"] == {}


def test_an_entry_that_never_filled_is_not_remembered_as_held():
    sleeve = _sleeve()
    dip = {"SPY": _dip(_uptrend(), 0.03, 0.03)}
    sleeve.decide(dip, sleeve.memory, "2026-09-14", set())
    # Not held the next day (the buy was refused): a fresh candidate, dated anew.
    sleeve.decide(dip, sleeve.memory, "2026-09-15", set())
    assert sleeve.memory["entered"] == {"SPY": "2026-09-15"}


def test_a_holding_without_memory_is_counted_from_today():
    sleeve = _sleeve()
    sleeve.memory = {"entered": "not a dict"}
    t = sleeve.decide({"SPY": _dip(_uptrend(), 0.03, 0.03)}, sleeve.memory,
                      "2026-09-28", {"SPY"})
    assert "SPY" in t.weights and sleeve.memory["entered"] == {"SPY": "2026-09-28"}


def test_the_decision_sees_what_the_sleeve_holds():
    seen = []
    s = sl.Sleeve(name="x", label="X", universe=("SPY", "QQQ"), allocation=0.2,
                  decide=lambda c, m, d, h: seen.append(h) or sl.Targets())
    s.plan({}, "2026-09-28", prices={"SPY": 1.0, "QQQ": 1.0},
           held={"SPY": 2.0, "QQQ": 0.0}, account_equity=73.0)
    assert seen == [{"SPY"}]


def test_configuration_is_read_from_the_environment_and_bounded(monkeypatch):
    monkeypatch.setenv("MEAN_REVERSION_ALLOCATION", "0.9")
    monkeypatch.setenv("MEAN_REVERSION_ENABLED", "off")
    s = _sleeve()
    assert s.allocation == 0.6 and not s.enabled
    monkeypatch.delenv("MEAN_REVERSION_ALLOCATION")
    monkeypatch.delenv("MEAN_REVERSION_ENABLED")
    s = _sleeve()
    assert s.allocation == pytest.approx(0.20) and s.enabled


# -- today's price ------------------------------------------------------------------


def _rows(values, last_day="2026-09-25"):
    end = dt.date.fromisoformat(last_day)
    days = []
    d = end
    while len(days) < len(values):
        if d.weekday() < 5:
            days.append(d)
        d -= dt.timedelta(days=1)
    days.reverse()
    return [{"t": f"{d.isoformat()}T04:00:00Z", "c": float(v)}
            for d, v in zip(days, values)]


def test_todays_live_price_is_todays_close():
    rows = {"SPY": _rows([10, 11, 12])}
    out = sl.closes_from_bars(rows, day="2026-09-28", live={"SPY": 9.0})
    assert list(out["SPY"]) == [10, 11, 12, 9], "appended: no bar for today yet"
    rows = {"SPY": _rows([10, 11, 12], last_day="2026-09-28")}
    out = sl.closes_from_bars(rows, day="2026-09-28", live={"SPY": 9.0})
    assert list(out["SPY"]) == [10, 11, 9], "today's forming bar is replaced"


def test_without_a_live_price_or_a_date_the_bars_stand():
    rows = {"SPY": _rows([10, 11, 12])}
    assert list(sl.closes_from_bars(rows, day="2026-09-28", live={})["SPY"]) == [10, 11, 12]
    assert list(sl.closes_from_bars(rows, day="2026-09-28",
                                    live={"SPY": 0.0})["SPY"]) == [10, 11, 12]
    assert list(sl.closes_from_bars(rows)["SPY"]) == [10, 11, 12]
    undated = {"SPY": [{"c": 10.0}, {"c": 11.0}]}
    assert list(sl.closes_from_bars(undated, day="2026-09-28",
                                    live={"SPY": 9.0})["SPY"]) == [10, 11]
    # A bar dated after "today" (a clock skew) is not overwritten.
    ahead = {"SPY": _rows([10, 11], last_day="2026-09-29")}
    assert list(sl.closes_from_bars(ahead, day="2026-09-28",
                                    live={"SPY": 9.0})["SPY"]) == [10, 11]


# -- in the session -------------------------------------------------------------------


class _Client:
    authenticated = True

    def __init__(self, closes):
        self.closes = closes

    async def bars(self, symbols, **kw):
        return {s: _rows(self.closes[s]) for s in symbols if s in self.closes}


def _session(equity=73.0):
    from imperium.execution.broker import PaperBroker
    from imperium.session import TradingSession
    from imperium.venues import registry

    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal(str(equity))
    session._attribution_loaded = True
    return session


def test_both_sleeves_on_leave_the_engine_half(monkeypatch):
    monkeypatch.setenv("GLOBAL_TREND_ENABLED", "true")
    session = _session()
    session.engine_equity(73.0)
    assert session.capital.share_for("mean_reversion") == pytest.approx(0.20)
    assert session.capital.share_for("engine") == pytest.approx(0.50)
    assert session.reserved_symbols()["SPY"] == "Mean reversion"


@pytest.mark.asyncio
async def test_todays_dip_is_bought_at_73_dollars_named_for_the_sleeve():
    """Yesterday's bars show only the uptrend; the fall is today's, on the
    live price. The sleeve buys on it, sends through the broker, and the
    journal names it."""
    session = _session(73.0)
    history = _uptrend()
    live = float(history[-1] * 0.96)
    session.client = _Client({"SPY": history, "QQQ": _uptrend(),
                              "IWM": _uptrend(), "DIA": _uptrend()})
    for s in mr.UNIVERSE:
        q = session.feed.quote(s)
        q.last = live if s == "SPY" else 150.0
        q.updated_at = time.time()
    sleeve = next(x for x in session.sleeves if x.name == "mean_reversion")
    await session._plan_sleeve(sleeve, "2026-09-28")
    assert not sleeve.last_error
    assert sleeve.last_targets == {"SPY": pytest.approx(0.5)}, sleeve.last_reasons
    session.client = None
    await session._tick()
    spy = session.broker.position("SPY")
    assert not spy.is_flat
    assert float(spy.quantity) * live == pytest.approx(73.0 * 0.20 * 0.5, rel=0.05)
    rows = session.trade_journal.read()
    assert rows and all(r.strategy == "mean_reversion" for r in rows)
    assert sleeve.memory["entered"] == {"SPY": "2026-09-28"}
    assert sleeve.last_run_day == "2026-09-28"


@pytest.mark.asyncio
async def test_without_a_fresh_price_the_decision_is_on_the_bars_alone():
    session = _session(73.0)
    session.client = _Client({"SPY": _uptrend()})
    q = session.feed.quote("SPY")
    q.last, q.updated_at = 140.0, time.time() - 600       # stale: not today's
    sleeve = next(x for x in session.sleeves if x.name == "mean_reversion")
    await session._plan_sleeve(sleeve, "2026-09-28")
    assert not sleeve.last_targets and not sleeve.pending
