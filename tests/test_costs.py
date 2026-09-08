"""The cost gate. One implementation, and the arithmetic has to be right."""

from __future__ import annotations

from decimal import Decimal

import pytest

from imperium.execution import costs
from imperium.execution.costs import ADVERSE_SELECTION_FRACTION, gate, round_trip_cost_bps
from imperium.venues.registry import FeeSchedule, get

FREE = FeeSchedule(maker_bps=Decimal("0"), taker_bps=Decimal("0"), assumed=False,
                   source="test")
TEN = FeeSchedule(maker_bps=Decimal("2"), taker_bps=Decimal("10"), assumed=False,
                  source="test")


def test_a_round_trip_costs_one_full_spread_not_two():
    """Prevents: the 2x error at the heart of the cost gate. A round trip is two
    crossings; two crossings is two half-spreads, which is ONE full spread.
    Charging two full spreads refuses roughly twice as many symbols as it
    should, and the previous attempt's three implementations disagreed by 5.5x."""
    est = round_trip_cost_bps(symbol="X", fees=FREE, spread_bps="4.0", style="taker")
    assert est.crossing_bps == Decimal("4.0")      # one full spread, not 8
    assert est.round_trip_bps == Decimal("4.0")    # no fees in this fixture


def test_a_taker_pays_the_fee_on_both_legs():
    """Prevents: charging the fee once for a round trip. Both the entry and the
    exit pay it."""
    est = round_trip_cost_bps(symbol="X", fees=TEN, spread_bps="2.0", style="taker")
    assert est.round_trip_bps == Decimal("10") * 2 + Decimal("2.0")


def test_a_maker_pays_adverse_selection_not_the_spread():
    """Prevents: modelling a passive fill as free. A resting order fills
    precisely when the market is coming through it, so the fill is
    systematically on the wrong side of the next move."""
    est = round_trip_cost_bps(symbol="X", fees=TEN, spread_bps="4.0", style="maker")
    half = Decimal("2.0")
    assert est.crossing_bps == half * ADVERSE_SELECTION_FRACTION * 2
    assert est.round_trip_bps == Decimal("2") * 2 + est.crossing_bps
    assert est.crossing_bps < Decimal("4.0"), "a maker must not pay a full spread"


def test_a_sell_side_fee_is_added_once_not_twice():
    """Prevents: doubling a regulatory fee that applies only to the sell leg."""
    fees = FeeSchedule(maker_bps=Decimal("0"), taker_bps=Decimal("0"),
                       sell_side_bps=Decimal("3"), assumed=False, source="test")
    est = round_trip_cost_bps(symbol="X", fees=fees, spread_bps="0", style="taker")
    assert est.round_trip_bps == Decimal("3")


def test_an_assumed_fee_tier_is_reported_as_a_warning():
    """Prevents: an assumption becoming a fact by default. The fee tier decides
    the trade/no-trade line directly, and an operator who is never told it was
    guessed has no reason to check it."""
    est = round_trip_cost_bps(symbol="BTCUSDT", fees=get("binance_spot").fees,
                              spread_bps="2.0")
    assert est.fees_are_assumed
    assert any("assumed" in w for w in est.warnings)
    assert "ASSUMED tier" in est.explain()


def test_a_confirmed_fee_tier_produces_no_warning():
    """Prevents: warning fatigue. Once the account's real rates are read back,
    the warning must stop."""
    confirmed = get("binance_spot").fees.confirmed(
        Decimal("1"), Decimal("1"), "read from /api/v3/account/commission")
    est = round_trip_cost_bps(symbol="BTCUSDT", fees=confirmed, spread_bps="2.0")
    assert not est.fees_are_assumed
    assert est.warnings == ()


def test_an_unmeasured_spread_is_marked_assumed_rather_than_silently_defaulted():
    """Prevents: a default spread masquerading as a measurement. The operator
    must be able to tell a priced symbol from a guessed one."""
    est = costs.estimate_for_symbol("BTCUSDT", get("binance_spot"), bid=None, ask=None)
    assert est.spread_is_assumed
    measured = costs.estimate_for_symbol("BTCUSDT", get("binance_spot"),
                                         bid=100.0, ask=100.02)
    assert not measured.spread_is_assumed
    assert float(measured.spread_bps) == pytest.approx(2.0, rel=1e-3)


def test_a_crossed_book_is_refused_rather_than_producing_a_negative_spread():
    """Prevents: a momentarily crossed or stale book yielding a negative cost,
    which would admit every symbol it touched."""
    assert costs.spread_bps_from_book(100.0, 99.0) is None
    assert costs.spread_bps_from_book(0.0, 1.0) is None


def test_the_gate_requires_the_edge_to_beat_the_cost_by_a_margin():
    """Prevents: trading on an edge estimate that merely equals its cost. The
    edge is an out-of-sample forecast and the cost is close to an accounting
    fact; requiring the forecast to beat the fact is what stops a book that
    trades constantly for nothing."""
    est = round_trip_cost_bps(symbol="X", fees=FREE, spread_bps="10", style="taker")
    assert gate(expected_edge_bps="10", estimate=est).admitted is False
    assert gate(expected_edge_bps="16", estimate=est).admitted is True
    assert gate(expected_edge_bps="-5", estimate=est).admitted is False


def test_the_gate_reason_names_every_number_that_decided_it():
    """Prevents: 'not trading' with no way to see which number was binding."""
    est = round_trip_cost_bps(symbol="X", fees=TEN, spread_bps="2.0")
    reason = gate(expected_edge_bps="5", estimate=est).reason
    assert "5.0bp" in reason and "22.0bp" in reason and "1.5" in reason


def test_there_is_exactly_one_round_trip_cost_implementation():
    """Prevents: the failure this module exists to stop. The previous attempt had
    three implementations of this arithmetic that disagreed by 5.5x, so every
    caller must import this one rather than reimplement it."""
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "imperium"
    offenders = []
    for path in src.rglob("*.py"):
        if path.name == "costs.py":
            continue
        text = path.read_text(encoding="utf-8")
        # A local half-spread or round-trip computation outside costs.py.
        if re.search(r"spread\s*/\s*2|half_spread|round_trip_bps\s*=", text):
            offenders.append(str(path.relative_to(src)))
    assert offenders == [], (
        f"these modules compute round-trip cost themselves instead of importing "
        f"imperium.execution.costs: {offenders}"
    )
