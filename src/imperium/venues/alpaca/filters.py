"""Quantity and price formatting, and per-asset rounding.

Two failure modes live here, and both are silent until the venue rejects an
order:

* **Scientific notation.** ``str(0.00001)`` is ``'1e-05'`` in Python, and the
  venue rejects it with ``-1100``. Every quantity and price that goes on the
  wire is formatted through :func:`format_decimal`, which never produces an
  exponent.
* **Binary float error.** ``0.1 + 0.2`` is not ``0.3``, and a quantity computed
  in floats and then rounded to a step size lands one ULP above the step often
  enough to matter. All rounding here is :class:`~decimal.Decimal`.

Quantities round **down** to the step size and prices round toward the passive
side, so rounding can never enlarge an order beyond what was sized.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_FLOOR, ROUND_CEILING, Decimal, InvalidOperation
from typing import Any, Iterable


def to_decimal(value: Any) -> Decimal:
    """Convert to Decimal without inheriting binary float error."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        # repr() of a float is the shortest string that round-trips, which is
        # the value the operator meant, rather than the full binary expansion.
        return Decimal(repr(value))
    return Decimal(str(value))


def format_decimal(value: Any, max_places: int = 20) -> str:
    """Render a number as a plain decimal string -- never an exponent.

    ``0.00001``, not ``1e-05``. Trailing zeros are stripped because some venue
    filters compare the string form.
    """
    d = to_decimal(value)
    if d == 0:
        return "0"
    # normalize() collapses 0.00001 to 1E-5, so quantize back into place value
    # form afterwards rather than trusting normalize's exponent.
    sign, digits, exponent = d.normalize().as_tuple()
    if isinstance(exponent, int) and exponent > 0:
        # e.g. Decimal('1E+2') -> we want '100'
        d = d.quantize(Decimal(1))
        text = f"{d:f}"
    else:
        text = f"{d.normalize():f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def floor_to_step(value: Any, step: Any) -> Decimal:
    """Round *down* to a multiple of ``step``.

    Rounding down is not a stylistic choice: rounding a quantity up can push an
    order past the position cap that sized it, and rounding a sell up can try to
    sell more of an asset than is held (``-2010``).
    """
    d = to_decimal(value)
    s = to_decimal(step)
    if s <= 0:
        return d
    return (d / s).to_integral_value(rounding=ROUND_FLOOR) * s


def round_price(value: Any, tick: Any, side: str) -> Decimal:
    """Round a price to the tick size, toward the passive side.

    A buy rounds down and a sell rounds up, so rounding never crosses the spread
    by accident and never turns a post-only order into a taker.
    """
    d = to_decimal(value)
    t = to_decimal(tick)
    if t <= 0:
        return d
    rounding = ROUND_FLOOR if side.upper() == "BUY" else ROUND_CEILING
    return (d / t).to_integral_value(rounding=rounding) * t


@dataclass(frozen=True)
class SymbolFilters:
    """The subset of the venue's asset record that constrains an order."""

    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    step_size: Decimal
    tick_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal
    #: Some symbols apply MIN_NOTIONAL to market orders too, some do not.
    apply_min_to_market: bool = True
    base_precision: int = 8
    quote_precision: int = 8
    permissions: tuple[str, ...] = ()

    @property
    def tradable(self) -> bool:
        return self.status == "TRADING"

    def quantize_qty(self, qty: Any) -> Decimal:
        return floor_to_step(qty, self.step_size)

    def quantize_price(self, price: Any, side: str) -> Decimal:
        return round_price(price, self.tick_size, side)

    def check_order(
        self, qty: Any, price: Any, *, is_market: bool = False
    ) -> str | None:
        """Return a human reason the order would be rejected, or None.

        Checking locally is not an optimisation. A rejected order costs a round
        trip, counts against the order-rate limit, and arrives as a bare
        ``-1013`` that does not say *which* filter failed.
        """
        q = to_decimal(qty)
        p = to_decimal(price)
        if not self.tradable:
            return f"{self.symbol} is not trading at the venue (status {self.status})"
        if q <= 0:
            return "quantity rounded to zero at this symbol's step size"
        if q < self.min_qty:
            return (f"quantity {format_decimal(q)} is below the symbol minimum "
                    f"{format_decimal(self.min_qty)}")
        if self.max_qty > 0 and q > self.max_qty:
            return (f"quantity {format_decimal(q)} exceeds the symbol maximum "
                    f"{format_decimal(self.max_qty)}")
        if is_market and not self.apply_min_to_market:
            return None
        notional = q * p
        if self.min_notional > 0 and notional < self.min_notional:
            return (f"notional {format_decimal(notional)} {self.quote_asset} is below "
                    f"the symbol minimum {format_decimal(self.min_notional)} "
                    f"{self.quote_asset}")
        return None


def _filter_value(filters: Iterable[dict[str, Any]], ftype: str, key: str,
                  default: str = "0") -> Decimal:
    for f in filters:
        if f.get("filterType") == ftype and key in f:
            return to_decimal(f[key])
    return to_decimal(default)


def parse_symbol(entry: dict[str, Any]) -> SymbolFilters:
    """Build :class:`SymbolFilters` from one exchangeInfo symbol entry.

    Venues have used two names for the notional filter -- ``MIN_NOTIONAL`` and
    the newer ``NOTIONAL`` -- and which one a symbol carries varies. Reading
    only one of them silently yields a zero minimum, which disables the check
    that stops dust orders being sent.
    """
    fs = entry.get("filters", [])
    min_notional = _filter_value(fs, "MIN_NOTIONAL", "minNotional")
    if min_notional == 0:
        min_notional = _filter_value(fs, "NOTIONAL", "minNotional")
    apply_to_market = True
    for f in fs:
        if f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
            if "applyToMarket" in f:
                apply_to_market = bool(f["applyToMarket"])
            elif "applyMinToMarket" in f:
                apply_to_market = bool(f["applyMinToMarket"])
    return SymbolFilters(
        symbol=entry["symbol"],
        base_asset=entry.get("baseAsset", ""),
        quote_asset=entry.get("quoteAsset", ""),
        status=entry.get("status", "UNKNOWN"),
        step_size=_filter_value(fs, "LOT_SIZE", "stepSize", "0.00000001"),
        tick_size=_filter_value(fs, "PRICE_FILTER", "tickSize", "0.00000001"),
        min_qty=_filter_value(fs, "LOT_SIZE", "minQty"),
        max_qty=_filter_value(fs, "LOT_SIZE", "maxQty"),
        min_notional=min_notional,
        apply_min_to_market=apply_to_market,
        base_precision=int(entry.get("baseAssetPrecision", 8)),
        quote_precision=int(entry.get("quoteAssetPrecision", 8)),
        permissions=tuple(entry.get("permissions", []) or ()),
    )
