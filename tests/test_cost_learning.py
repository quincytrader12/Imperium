"""The cost gate, corrected by what real fills say crossing costs.

The property that matters most is the one that is easiest to get wrong: the
correction must not feed on itself. Measured against the corrected model, each
fill shows only the excess the correction has not yet absorbed, so the more it
corrects the less it learns -- simulated against a true excess of 6bp, that
version stalls at about 2.9bp and never gets closer. That is tested directly,
and mutation-checked.
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from imperium.execution import cost_learning as cl
from imperium.execution import costs
from imperium.execution.broker import MARKET_ON_CLOSE, Fill, Mode, PaperBroker
from imperium.session import TradingSession
from imperium.venues import registry


# -- the estimate -----------------------------------------------------------------


def test_nothing_is_corrected_on_a_handful_of_fills():
    cal = cl.CostCalibration()
    for _ in range(cl.MIN_FILLS - 1):
        cal.observe("us_equity", measured_bps=20.0, model_one_way_bps=2.0)
    assert cal.excess_bps("us_equity") == 0.0
    assert "4 of 5 real fills" in cal.describe("us_equity")


def test_the_correction_is_shrunk_by_how_much_evidence_there_is():
    """Five fills move the model a fifth of the way to what they say."""
    cal = cl.CostCalibration()
    for _ in range(5):
        cal.observe("us_equity", measured_bps=6.0, model_one_way_bps=2.0)
    assert cal.excess_bps("us_equity") == pytest.approx(4.0 * 5 / (5 + cl.CREDIBILITY))


def test_one_fill_in_a_fast_market_is_not_the_cost_of_crossing():
    """The median, not the mean."""
    cal = cl.CostCalibration()
    for _ in range(40):
        cal.observe("us_equity", measured_bps=3.0, model_one_way_bps=2.0)
    cal.observe("us_equity", measured_bps=400.0, model_one_way_bps=2.0)
    weight = 41 / (41 + cl.CREDIBILITY)
    assert cal.excess_bps("us_equity") == pytest.approx(1.0 * weight)


def test_an_implausible_fill_is_a_data_fault_not_a_crossing():
    cal = cl.CostCalibration()
    assert not cal.observe("us_equity", measured_bps=5_000.0, model_one_way_bps=2.0)
    assert cal.classes["us_equity"].rejected == 1
    assert cal.classes["us_equity"].samples == []


def test_the_correction_is_bounded_both_ways():
    cal = cl.CostCalibration()
    for _ in range(500):
        cal.observe("crypto", measured_bps=300.0, model_one_way_bps=1.0)
        cal.observe("us_equity", measured_bps=-300.0, model_one_way_bps=1.0)
    assert cal.excess_bps("crypto") == cl.MAX_CORRECTION_BPS
    assert cal.excess_bps("us_equity") == -cl.MAX_CORRECTION_BPS


def test_asset_classes_are_learned_separately():
    """An equity and a coin cross very different books."""
    cal = cl.CostCalibration()
    for _ in range(30):
        cal.observe("crypto", measured_bps=12.0, model_one_way_bps=2.0)
    assert cal.excess_bps("crypto") > 0
    assert cal.excess_bps("us_equity") == 0.0


def test_the_calibration_survives_a_restart():
    cal = cl.CostCalibration()
    for _ in range(30):
        cal.observe("crypto", measured_bps=8.0, model_one_way_bps=2.0)
    back = cl.CostCalibration.from_dict(cal.as_dict())
    assert back.excess_bps("crypto") == cal.excess_bps("crypto")


def test_corrupt_samples_on_disk_are_dropped():
    back = cl.CostCalibration.from_dict(
        {"crypto": {"samples": [1.0, "x", float("nan"), 9e9, 2.0]}})
    assert back.classes["crypto"].samples == [1.0, 2.0]


# -- applied in the one cost function ----------------------------------------------


def _taker(spread, correction=0.0):
    from imperium.venues.assets import spec_for, AssetClass

    return costs.round_trip_cost_bps(
        symbol="AAPL", fees=spec_for(AssetClass.US_EQUITY).cost_model,
        spread_bps=spread, style="taker", crossing_correction_bps=correction)


def test_a_correction_is_paid_on_both_crossings_of_a_round_trip():
    base = _taker(4.0)
    corrected = _taker(4.0, correction=3.0)
    assert corrected.round_trip_bps - base.round_trip_bps == pytest.approx(Decimal("6"))


def test_the_model_figure_is_reported_uncorrected():
    """The figure fills are measured against. If it moved with the correction,
    the correction would be learned away."""
    assert _taker(4.0, correction=3.0).model_one_way_bps == pytest.approx(Decimal("2"))
    assert _taker(4.0).model_one_way_bps == pytest.approx(Decimal("2"))


def test_a_cheaper_correction_never_makes_crossing_pay():
    est = _taker(4.0, correction=-40.0)
    assert est.crossing_bps == 0
    assert est.round_trip_bps >= 0


def test_a_passive_fill_is_not_corrected_by_what_market_orders_measured():
    from imperium.venues.assets import spec_for, AssetClass

    fees = spec_for(AssetClass.US_EQUITY).cost_model
    plain = costs.round_trip_cost_bps(symbol="AAPL", fees=fees, spread_bps=4.0,
                                      style="maker")
    corrected = costs.round_trip_cost_bps(symbol="AAPL", fees=fees,
                                          spread_bps=4.0, style="maker",
                                          crossing_correction_bps=5.0)
    assert corrected.round_trip_bps == plain.round_trip_bps


# -- the loop does not feed on itself ----------------------------------------------


def _engine_with_book(calibration):
    session = TradingSession()
    engine = session.engine("AAPL")
    engine.cost_calibration = calibration
    engine.set_book(99.98, 100.02)            # a 4bp spread: 2bp each way
    return engine


def test_the_correction_converges_rather_than_learning_itself_away():
    """True crossing is 6bp dearer than the model. Each fill is measured the
    way the session measures it -- against what the engine's own decision says
    the model priced -- and the correction must climb toward 6bp and stay.
    Measured against the corrected figure instead, it stalls at about half."""
    from imperium.execution.engine import Decision

    cal = cl.CostCalibration()
    engine = _engine_with_book(cal)
    history = []
    for _ in range(200):
        d = Decision(symbol="AAPL")
        engine._taker_cost(d)
        # What actually happened: the real half-spread plus 6bp of impact.
        measured = 2.0 + 6.0
        cal.observe("us_equity", measured, d.model_one_way_bps)
        history.append(cal.excess_bps("us_equity"))

    assert history[-1] == pytest.approx(6.0 * 200 / (200 + cl.CREDIBILITY), rel=1e-3)
    assert all(b >= a - 1e-12 for a, b in zip(history, history[1:])), (
        "the correction fell back while the evidence for it kept arriving")


def test_the_gate_carries_the_correction_through_the_engine():
    from imperium.execution.engine import Decision

    cal = cl.CostCalibration()
    engine = _engine_with_book(cal)
    before = engine._taker_cost(Decision(symbol="AAPL")).round_trip_bps
    for _ in range(40):
        cal.observe("us_equity", 8.0, 2.0)
    d = Decision(symbol="AAPL")
    after = engine._taker_cost(d).round_trip_bps
    assert after > before
    assert d.crossing_correction_bps == pytest.approx(cal.excess_bps("us_equity"))


# -- what the session learns from ---------------------------------------------------


class _VenueBroker(PaperBroker):
    """Fills like the venue's paper account: real, not charged by a model."""

    simulated = False


def _session_with_decision(home, *, spread_bps=4.0):
    session = TradingSession()
    session.broker = _VenueBroker(registry.get(registry.DEFAULT_VENUE))
    session._attribution_loaded = True
    engine = session.engine("AAPL")
    engine.decision.model_one_way_bps = spread_bps / 2
    return session


def _fill(*, price, ref, simulated=False, order=""):
    return Fill("AAPL", "BUY", Decimal("1"), Decimal(str(price)), time.time(),
                Mode.PAPER, "c", simulated, reference_price=Decimal(str(ref)),
                order=order)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    return tmp_path


def test_a_real_fill_teaches_the_gate(home):
    session = _session_with_decision(home)
    assert session._learn_crossing(_fill(price=100.08, ref=100.0))
    sample = session.cost_calibration.classes["us_equity"].samples[0]
    assert sample == pytest.approx(8.0 - 2.0)


def test_a_simulated_fill_teaches_nothing(home):
    """The local paper book charges the modelled cost by construction. Learning
    from it would be the model grading itself."""
    session = _session_with_decision(home)
    assert not session._learn_crossing(_fill(price=100.08, ref=100.0,
                                             simulated=True))
    assert "us_equity" not in session.cost_calibration.classes


def test_an_auction_fill_is_not_a_crossing(home):
    """Its distance from the last trade is the overnight move, not a spread."""
    session = _session_with_decision(home)
    assert not session._learn_crossing(_fill(price=100.5, ref=100.0,
                                             order=MARKET_ON_CLOSE))


@pytest.mark.asyncio
async def test_the_tick_learns_from_the_fills_it_books(home):
    """Through _attribute, which is what the tick calls, reading from the same
    cursor the attribution book uses -- so no fill is learned twice."""
    session = _session_with_decision(home)
    q = session.feed.quote("AAPL")
    q.last, q.updated_at = 100.0, time.time()
    await session.broker.apply_target("AAPL", 0.10, 100.0, 10_000.0,
                                      strategy="trend")
    session._attribute()
    session._attribute()
    rec = session.cost_calibration.classes.get("us_equity")
    assert rec is not None and rec.total == 1


def test_a_restart_brings_the_calibration_back_to_every_engine(home):
    first = _session_with_decision(home)
    for _ in range(30):
        first.cost_calibration.observe("us_equity", 8.0, 2.0)
    first._save_attribution()

    second = TradingSession()
    engine = second.engine("AAPL")
    second._load_attribution()
    assert engine.cost_calibration is second.cost_calibration
    assert second.cost_calibration.excess_bps("us_equity") > 0
    late = second.engine("MSFT")
    assert late.cost_calibration is second.cost_calibration


def test_a_long_record_is_believed_more_than_its_recent_window_alone():
    """The median follows the recent window; the credibility counts every fill
    ever measured. Weighted by the window alone the correction was capped at
    200/220 of what the fills said, however long the terminal ran."""
    cal = cl.CostCalibration()
    for _ in range(2_000):
        cal.observe("us_equity", 8.0, 2.0)
    assert len(cal.classes["us_equity"].samples) == cl.MAX_SAMPLES
    assert cal.excess_bps("us_equity") == pytest.approx(
        6.0 * 2_000 / (2_000 + cl.CREDIBILITY))
    assert cal.excess_bps("us_equity") > 5.9
