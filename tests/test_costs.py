"""The cost gate, and why one gate cannot serve two asset classes."""

from __future__ import annotations

from decimal import Decimal

import pytest

from imperium.execution import costs
from imperium.execution.costs import ADVERSE_SELECTION_FRACTION, gate, round_trip_cost_bps
from imperium.venues.assets import AssetClass, CostModel, spec_for

FREE = CostModel(commission_bps=Decimal("0"), sell_side_bps=Decimal("0"),
                 default_spread_bps=Decimal("2"), assumed=False, source="test")
FEE = CostModel(commission_bps=Decimal("10"), sell_side_bps=Decimal("0"),
                default_spread_bps=Decimal("2"), assumed=False, source="test")


def test_a_round_trip_costs_one_full_spread_not_two():
    """Prevents the 2x error at the heart of the cost gate. A round trip is two
    crossings; two crossings is two half-spreads, which is ONE full spread."""
    est = round_trip_cost_bps(symbol="X", fees=FREE, spread_bps="4.0", style="taker")
    assert est.crossing_bps == Decimal("4.0")
    assert est.round_trip_bps == Decimal("4.0")


def test_commission_is_paid_on_both_legs():
    """Prevents charging the commission once for a round trip."""
    est = round_trip_cost_bps(symbol="X", fees=FEE, spread_bps="2.0", style="taker")
    assert est.round_trip_bps == Decimal("10") * 2 + Decimal("2.0")


def test_a_maker_pays_adverse_selection_not_the_spread():
    """Prevents modelling a passive fill as free. A resting order fills when the
    market is coming through it."""
    est = round_trip_cost_bps(symbol="X", fees=FEE, spread_bps="4.0", style="maker")
    assert est.crossing_bps == Decimal("2.0") * ADVERSE_SELECTION_FRACTION * 2
    assert est.crossing_bps < Decimal("4.0")


def test_a_sell_side_fee_is_added_once_not_twice():
    """Prevents doubling a regulatory fee that applies only to the sell leg.
    For US equities this is the whole of the non-spread cost."""
    fees = CostModel(commission_bps=Decimal("0"), sell_side_bps=Decimal("3"),
                     default_spread_bps=Decimal("0"), assumed=False, source="t")
    est = round_trip_cost_bps(symbol="X", fees=fees, spread_bps="0")
    assert est.round_trip_bps == Decimal("3")


def test_equities_and_crypto_are_priced_by_different_models():
    """Prevents the failure this split exists for: one cost model across both
    asset classes.

    A US equity at Alpaca pays no commission and roughly 1bp of regulatory fee
    on the sell leg, so its round trip is almost entirely spread. Crypto pays a
    commission on both legs that dwarfs the spread. Pricing an equity with the
    crypto model refuses nearly every equity the scanner sees; pricing crypto
    with the equity model admits trades that lose money on costs alone.
    """
    equity = costs.estimate_for_symbol("AAPL", bid=100.0, ask=100.03)
    crypto = costs.estimate_for_symbol("BTC/USD", bid=100.0, ask=100.03)

    assert equity.asset_class == "us_equity"
    assert crypto.asset_class == "crypto"
    assert equity.fee_bps == 0
    assert crypto.fee_bps > 0
    assert crypto.round_trip_bps > equity.round_trip_bps * 5, (
        f"equity {equity.round_trip_bps}bp vs crypto {crypto.round_trip_bps}bp"
    )

    # The same edge must decide differently for the two.
    assert gate(expected_edge_bps="20", estimate=equity).admitted is True
    assert gate(expected_edge_bps="20", estimate=crypto).admitted is False


def test_the_asset_class_is_inferred_from_the_symbol():
    """Prevents an equity being priced as crypto because nobody passed a class.
    Alpaca spells crypto pairs with a slash; that is the only signal there is."""
    assert costs.estimate_for_symbol("SPY").asset_class == "us_equity"
    assert costs.estimate_for_symbol("ETH/USD").asset_class == "crypto"


def test_an_assumed_fee_tier_is_reported_as_a_warning():
    """Prevents an assumption becoming a fact by default. Alpaca's equity
    regulatory fees are estimated, not read from the account."""
    est = costs.estimate_for_symbol("AAPL", bid=100.0, ask=100.03)
    assert est.fees_are_assumed
    assert any("assumed" in w for w in est.warnings)
    assert "ASSUMED tier" in est.explain()


def test_an_unmeasured_spread_is_marked_assumed():
    """Prevents a default spread masquerading as a measurement."""
    assert costs.estimate_for_symbol("AAPL").spread_is_assumed
    measured = costs.estimate_for_symbol("AAPL", bid=100.0, ask=100.02)
    assert not measured.spread_is_assumed
    assert float(measured.spread_bps) == pytest.approx(2.0, rel=1e-3)


def test_a_crossed_book_is_refused_rather_than_producing_a_negative_spread():
    """Prevents a stale or crossed book yielding a negative cost, which would
    admit every symbol it touched."""
    assert costs.spread_bps_from_book(100.0, 99.0) is None
    assert costs.spread_bps_from_book(0.0, 1.0) is None


def test_the_gate_requires_the_edge_to_beat_the_cost_by_a_margin():
    """Prevents trading on an edge estimate that merely equals its cost."""
    est = round_trip_cost_bps(symbol="X", fees=FREE, spread_bps="10")
    assert gate(expected_edge_bps="10", estimate=est).admitted is False
    assert gate(expected_edge_bps="16", estimate=est).admitted is True
    assert gate(expected_edge_bps="-5", estimate=est).admitted is False


def test_the_gate_reason_names_every_number_that_decided_it():
    """Prevents 'not trading' with no way to see which number was binding."""
    est = round_trip_cost_bps(symbol="X", fees=FEE, spread_bps="2.0")
    reason = gate(expected_edge_bps="5", estimate=est).reason
    assert "5.0bp" in reason and "22.0bp" in reason and "1.5" in reason


def test_there_is_exactly_one_round_trip_cost_implementation():
    """Prevents the failure this module exists to stop.

    A previous attempt had three implementations of this arithmetic that
    disagreed by 5.5x, so every caller must take the number from here rather
    than derive its own.

    What counts as an offence is *deriving* the cost, not naming it. A strategy
    that receives a round-trip cost as a parameter and reasons about it is doing
    the right thing; an earlier version of this test failed those too, which
    pushes callers toward worse parameter names to get green rather than toward
    the single implementation.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "imperium"
    #: Halving a spread by hand -- the exact arithmetic of the original bug.
    by_hand = re.compile(r"spread\s*/\s*2|half_spread")
    #: A second definition of the one function.
    redefined = re.compile(r"def\s+round_trip_cost_bps\b")
    #: Assigning a round-trip figure from the cost components themselves.
    derived = re.compile(
        r"round_trip\w*\s*=\s*[^\n=][^\n]*\b(spread|commission|fee)\w*\b",
        re.IGNORECASE)

    offenders = []
    for path in src.rglob("*.py"):
        if path.name == "costs.py":
            continue
        text = path.read_text(encoding="utf-8")
        if by_hand.search(text) or redefined.search(text) or derived.search(text):
            offenders.append(str(path.relative_to(src)))
    assert offenders == [], (
        f"these modules derive a round-trip cost themselves instead of taking "
        f"it from imperium.execution.costs: {offenders}")
