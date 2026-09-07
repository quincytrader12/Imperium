"""Symbol filters, rounding and decimal formatting."""

from __future__ import annotations

from decimal import Decimal

import pytest

from godalgo.venues.binance.filters import (
    floor_to_step, format_decimal, parse_symbol, round_price, to_decimal,
)
from mock_venue import EXCHANGE_INFO


@pytest.mark.parametrize("value,expected", [
    (0.00001, "0.00001"),
    (1e-8, "0.00000001"),
    (0.0000001, "0.0000001"),
    (1.5e3, "1500"),
    (100.0, "100"),
    (Decimal("0.00000100"), "0.000001"),
    (0, "0"),
])
def test_no_quantity_is_ever_rendered_in_scientific_notation(value, expected):
    """Prevents: str(0.00001) == '1e-05' going on the wire, which the venue
    rejects with -1100. Python's default float repr uses an exponent below
    1e-4, which is inside the range of ordinary crypto quantities."""
    rendered = format_decimal(value)
    assert rendered == expected
    assert "e" not in rendered.lower()


def test_quantities_round_down_never_up():
    """Prevents: a rounded quantity exceeding the size that was risk-checked.
    Rounding up can push an order past the cap that sized it, and can try to
    sell more of an asset than is held (-2010)."""
    assert floor_to_step("0.123456", "0.001") == Decimal("0.123")
    assert floor_to_step("1.9999999", "0.1") == Decimal("1.9")
    assert floor_to_step("0.0000099", "0.00001") == Decimal("0")


def test_rounding_uses_decimals_not_binary_floats():
    """Prevents: computing a quantity in floats, where 0.1+0.2 != 0.3, and
    landing one ULP above a step boundary -- which the venue rejects with -1111
    for a reason that is invisible in the printed value."""
    assert floor_to_step(0.1 + 0.2, "0.1") == Decimal("0.3")
    assert to_decimal(0.07) == Decimal("0.07")


def test_prices_round_toward_the_passive_side():
    """Prevents: a post-only buy rounded *up* through the spread, which turns a
    maker order into a taker or gets it rejected outright."""
    assert round_price("100.567", "0.01", "BUY") == Decimal("100.56")
    assert round_price("100.561", "0.01", "SELL") == Decimal("100.57")


def test_both_spellings_of_the_notional_filter_are_read():
    """Prevents: reading only MIN_NOTIONAL. Binance also uses NOTIONAL, and a
    symbol carrying only the other name silently yields a zero minimum, which
    disables the check that stops dust orders being sent."""
    by_symbol = {s["symbol"]: parse_symbol(s) for s in EXCHANGE_INFO["symbols"]}
    assert by_symbol["BTCUSDT"].min_notional == Decimal("5")   # NOTIONAL
    assert by_symbol["ETHUSDT"].min_notional == Decimal("10")  # MIN_NOTIONAL


def test_a_dust_order_is_refused_locally_with_the_reason_named():
    """Prevents: sending an order the venue will reject with a bare -1013 that
    does not say which filter failed. A local check costs nothing, spends no
    order-rate budget, and names the actual constraint."""
    btc = parse_symbol(EXCHANGE_INFO["symbols"][0])
    reason = btc.check_order(Decimal("0.00001"), Decimal("100"))
    assert reason is not None and "below the symbol minimum" in reason
    assert btc.check_order(Decimal("0.001"), Decimal("60000")) is None


def test_a_halted_symbol_is_refused_before_an_order_is_built():
    """Prevents: sizing and submitting into a symbol the venue has halted."""
    dead = parse_symbol(EXCHANGE_INFO["symbols"][2])
    assert dead.tradable is False
    assert "not trading" in dead.check_order(Decimal("10"), Decimal("10"))
