"""The account's risk dial: volatility targeting and drawdown de-grossing.

The properties that matter: it only ever scales *down*; it never reaches
zero; a drawdown under the line changes nothing; the cut is a straight line
from the line to the floor; days away do not read as volatile days; and the
one number reaches every strategy -- the engine's decisions and the sleeves'.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal

import numpy as np
import pytest

from imperium.execution import risk_dial as rd


def _days(n, start="2026-08-01"):
    first = dt.date.fromisoformat(start)
    return [(first + dt.timedelta(days=i)).isoformat() for i in range(n)]


def _marks(returns, start=100.0):
    equity = start * np.exp(np.cumsum(np.concatenate([[0.0], returns])))
    return list(zip(_days(len(equity)), map(float, equity)))


def _alternating(daily_sd, n=21):
    """Returns of exactly +sd, -sd, ...: a known standard deviation."""
    return np.array([daily_sd if i % 2 else -daily_sd for i in range(n)])


# -- the measurements -------------------------------------------------------------------


def test_realised_volatility_is_annualised_on_calendar_days():
    sd = 0.01
    marks = _marks(_alternating(sd, rd.VOL_DAYS))
    returns = _alternating(sd, rd.VOL_DAYS)
    expected = float(np.std(returns, ddof=1)) * math.sqrt(365)
    assert rd.realised_vol(marks) == pytest.approx(expected)


def test_only_the_last_twenty_days_count():
    wild = _alternating(0.05, 30)
    calm = _alternating(0.002, rd.VOL_DAYS)
    marks = _marks(np.concatenate([wild, calm]))
    assert rd.realised_vol(marks) < 0.1


def test_days_away_are_not_volatile_days():
    # Ten days of 1% moves, taken as one mark every four days: a 4% move over
    # four days is a day's 2%, not a day's 4%.
    equity, marks = 100.0, []
    for i, day in enumerate(_days(60)[::4][:15]):
        equity *= math.exp(0.04 if i % 2 else -0.04)
        marks.append((day, equity))
    close_spaced = rd.realised_vol(marks)
    daily = _marks(_alternating(0.02, 14))
    assert close_spaced == pytest.approx(rd.realised_vol(daily), rel=1e-6)


def test_too_little_history_measures_nothing():
    assert rd.realised_vol(_marks(_alternating(0.01, rd.MIN_RETURNS - 1))) is None
    assert rd.realised_vol(_marks(_alternating(0.01, rd.MIN_RETURNS))) is not None
    assert rd.realised_vol([("garbage", 1.0), ("2026-01-01", 1.0)]) is None


@pytest.mark.parametrize("dd,scale", [
    (0.0, 1.0), (0.049, 1.0), (0.05, 1.0), (0.125, 0.625), (0.20, 0.25),
    (0.5, 0.25)])
def test_the_drawdown_cut_is_a_straight_line_to_the_floor(dd, scale):
    assert rd.drawdown_scale(dd) == pytest.approx(scale)


def test_a_degenerate_band_is_a_step():
    assert rd.drawdown_scale(0.04, 0.05, 0.05) == 1.0
    assert rd.drawdown_scale(0.05, 0.05, 0.05) == rd.FLOOR


# -- the dial ------------------------------------------------------------------------------


def test_no_history_is_full_size():
    assert rd.compute([], 0.0).value == 1.0
    assert rd.compute([], 73.0).value == 1.0


def test_volatility_over_the_target_scales_in_proportion():
    daily_sd = 0.30 / math.sqrt(365)            # 30% a year: twice the target
    marks = _marks(_alternating(daily_sd, rd.VOL_DAYS))
    d = rd.compute(marks, marks[-1][1])
    assert d.realised_vol == pytest.approx(0.30, rel=0.03)
    assert d.vol_scale == pytest.approx(0.15 / d.realised_vol)
    assert d.value == pytest.approx(d.vol_scale)
    assert "over the target 15%" in d.reason


def test_a_quiet_account_is_never_levered_up():
    marks = _marks(_alternating(0.001, rd.VOL_DAYS))
    d = rd.compute(marks, marks[-1][1])
    assert d.vol_scale == 1.0 and d.value == 1.0
    assert d.reason.startswith("full size")


def test_a_drawdown_de_grosses_and_names_the_high_water_mark():
    marks = [(day, 100.0) for day in _days(5)] + [("2026-08-06", 80.0)]
    d = rd.compute(marks, 87.5)                    # 12.5% below 100
    assert d.drawdown == pytest.approx(0.125) and d.high_water == 100.0
    assert d.dd_scale == pytest.approx(0.625) and d.value == pytest.approx(0.625)
    assert "12.5% below the high-water mark $100.00" in d.reason
    assert d.reason.startswith("every strategy at 63% of size")


def test_both_together_multiply_and_stop_at_the_floor():
    daily_sd = 0.60 / math.sqrt(365)
    marks = _marks(_alternating(daily_sd, rd.VOL_DAYS), start=100.0)
    high = max(e for _, e in marks)
    d = rd.compute(marks, high * 0.85)
    assert d.vol_scale < 0.5 and d.dd_scale < 1.0
    assert d.value == pytest.approx(max(rd.FLOOR, d.vol_scale * d.dd_scale))
    d = rd.compute(marks, high * 0.5)
    assert d.value == rd.FLOOR


def test_a_reset_high_water_mark_forgets_older_highs():
    marks = [("2026-08-01", 100.0), ("2026-08-02", 60.0), ("2026-08-03", 61.0)]
    assert rd.compute(marks, 61.0).value == rd.FLOOR
    d = rd.compute(marks, 61.0, since="2026-08-02")
    assert d.high_water == 61.0 and d.value == 1.0


def test_the_volatility_target_can_be_switched_off():
    marks = _marks(_alternating(0.05, rd.VOL_DAYS))
    assert rd.compute(marks, marks[-1][1], target_vol=0.0).vol_scale == 1.0


def test_settings_are_bounded(monkeypatch):
    monkeypatch.setenv("IMPERIUM_ACCOUNT_TARGET_VOL", "9")
    monkeypatch.setenv("IMPERIUM_DRAWDOWN_START", "0.3")
    monkeypatch.setenv("IMPERIUM_DRAWDOWN_FULL", "0.1")
    monkeypatch.setenv("IMPERIUM_RISK_DIAL", "off")
    cfg = rd.settings()
    assert cfg["target_vol"] == 2.0 and cfg["start"] == 0.3
    assert cfg["full"] == pytest.approx(0.31) and not cfg["enabled"]
    monkeypatch.setenv("IMPERIUM_ACCOUNT_TARGET_VOL", "nan")
    assert rd.settings()["target_vol"] == rd.TARGET_VOL


# -- where it acts ----------------------------------------------------------------------------


def _decision(engine, weight=0.10):
    from imperium.execution.engine import Decision
    d = Decision(symbol=engine.symbol)
    d.strategy = "trend"
    d.raw_weight = weight
    return d


def test_the_engine_scales_every_size_by_the_dial():
    from test_overnight import _engine

    engine = _engine("AAPL")
    engine.risk_dial = 0.5
    d = _decision(engine)
    engine._size_for_evidence(d)
    assert d.raw_weight == pytest.approx(0.05)
    assert d.capital_multiplier == pytest.approx(0.5)
    assert "account risk dial ×0.50" in d.capital_note


@pytest.mark.parametrize("bad", [0.0, -1.0, 2.0, float("nan")])
def test_a_nonsense_dial_is_full_size(bad):
    from test_overnight import _engine

    engine = _engine("AAPL")
    engine.risk_dial = bad
    d = _decision(engine)
    engine._size_for_evidence(d)
    assert d.raw_weight == pytest.approx(0.10)


def _session(equity):
    from imperium.execution.broker import PaperBroker
    from imperium.session import TradingSession
    from imperium.venues import registry

    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal(str(equity))
    session._attribution_loaded = True
    return session


def test_the_session_hands_the_dial_to_every_engine_and_says_so():
    session = _session(85.0)
    engine = session.engine("AAPL")
    history = session.equity_history()
    history.daily = [(day, 100.0) for day in _days(5)]
    session._revise_risk_dial(history, force=True)
    assert session.risk_dial.value == pytest.approx(rd.drawdown_scale(0.15))
    assert engine.risk_dial == session.risk_dial.value
    assert session.engine("LATER").risk_dial == session.risk_dial.value
    assert "Risk dial 100% →" in session._pending_notice
    assert any("risk dial" in e.get("message", "") for e in session.telemetry.events())
    session.client = None
    assert session.snapshot()["risk_dial"]["value"] == pytest.approx(session.risk_dial.value)

    # Throttled: the same minute does not recompute.
    session.broker.cash = Decimal("100")
    session._revise_risk_dial(history)
    assert session.risk_dial.value < 1.0
    session._revise_risk_dial(history, force=True)
    assert session.risk_dial.value == 1.0


def test_switched_off_it_is_full_size(monkeypatch):
    monkeypatch.setenv("IMPERIUM_RISK_DIAL", "false")
    session = _session(50.0)
    history = session.equity_history()
    history.daily = [(day, 100.0) for day in _days(5)]
    session._revise_risk_dial(history, force=True)
    assert session.risk_dial.value == 1.0 and "off" in session.risk_dial.reason


@pytest.mark.asyncio
async def test_the_sleeves_are_sized_through_the_dial(monkeypatch):
    import time

    from test_turn_of_month import _Client

    monkeypatch.setenv("TURN_OF_MONTH_ENABLED", "true")
    session = _session(73.0)
    session.risk_dial = rd.Dial(value=0.5)
    session.client = _Client()
    q = session.feed.quote("IVV")
    q.last, q.updated_at = 610.0, time.time()
    sleeve = next(x for x in session.sleeves if x.name == "turn_of_month")
    await session._plan_sleeve(sleeve, "2026-09-29")
    assert sleeve.pending[0].notional == pytest.approx(73.0 * 0.15 * 0.5)


def test_an_extreme_volatility_floors_the_volatility_scale_itself():
    marks = _marks(_alternating(0.2, rd.VOL_DAYS))
    assert rd.compute(marks, marks[-1][1], target_vol=0.15).vol_scale == rd.FLOOR


def test_the_two_scales_multiply_rather_than_the_stricter_winning():
    daily_sd = 0.20 / math.sqrt(365)            # a quarter over the target
    marks = _marks(_alternating(daily_sd, rd.VOL_DAYS))
    high = max(e for _, e in marks)
    d = rd.compute(marks, high * 0.875)           # 12.5% down: ×0.625
    assert d.vol_scale == pytest.approx(0.75, rel=0.05)
    assert d.value == pytest.approx(d.vol_scale * 0.625)
    assert d.value < min(d.vol_scale, d.dd_scale)


def test_the_reset_day_itself_counts():
    marks = [("2026-08-01", 100.0), ("2026-08-02", 70.0), ("2026-08-03", 61.0)]
    assert rd.compute(marks, 61.0, since="2026-08-02").high_water == 70.0


@pytest.mark.asyncio
async def test_every_tick_that_marks_the_account_revises_the_dial():
    session = _session(85.0)
    history = session.equity_history()
    history.daily = [(day, 100.0) for day in _days(5)]
    await session._tick()
    assert session.risk_dial.value < 1.0
