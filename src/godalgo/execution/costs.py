"""The cost gate. One module, one implementation, imported by everything.

This decides whether anything trades at all, so it is written once and every
caller imports it. A previous attempt at this program had three implementations
that disagreed by 5.5x, which means two thirds of its symbols were admitted or
refused on arithmetic that was simply wrong.

The arithmetic, stated explicitly because this is where it goes wrong:

* **A round trip is two crossings.** Entering pays half the spread, exiting pays
  half the spread. Two half-spreads is **one full spread** -- not two. Charging
  two full spreads is the 2x error; charging two half-spreads *and* calling it
  two full spreads is the 4x one.
* **A taker** pays the taker fee plus half the spread, one way. Twice for a
  round trip: ``2 * taker_fee + spread``.
* **A maker** pays the maker fee plus a fraction of the half spread to
  **adverse selection** -- a passive fill happens precisely when the market is
  moving through the resting order, so the fill is systematically on the wrong
  side of the subsequent move. That fraction is an assumption, labelled as one:
  :data:`ADVERSE_SELECTION_FRACTION`.
* **Regulatory or sell-side fees** are added once, on the sell leg only.
* **An assumed fee tier is a warning, not information.** An assumption nobody is
  told about becomes a fact by default, and this one decides whether a symbol
  trades.

All rates are in basis points (1bp = 0.01%) throughout. There is no place in
this module where a fraction and a basis-point figure meet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from godalgo.venues.registry import FeeSchedule, VenueSpec

#: The share of the half-spread a passive fill gives up to adverse selection.
#: This is an assumption, not a measurement. It is stated here, once, so that it
#: can be argued with -- and it is reported alongside every admission decision
#: rather than buried.
ADVERSE_SELECTION_FRACTION = Decimal("0.35")

Style = Literal["maker", "taker"]


@dataclass(frozen=True)
class CostEstimate:
    """The full round-trip cost of trading one symbol, in basis points."""

    symbol: str
    style: Style
    fee_bps: Decimal
    spread_bps: Decimal
    #: What the crossing actually costs over a round trip, in bps.
    crossing_bps: Decimal
    sell_side_bps: Decimal
    round_trip_bps: Decimal
    spread_is_assumed: bool
    fees_are_assumed: bool
    warnings: tuple[str, ...] = ()

    def explain(self) -> str:
        """The line logged when a symbol is priced, per the brief.

        Every number that went into the decision, in one sentence, so that a
        surprising admission or refusal can be understood without a debugger.
        """
        fee_note = " (ASSUMED tier)" if self.fees_are_assumed else ""
        spread_note = " (ASSUMED)" if self.spread_is_assumed else ""
        return (
            f"{self.symbol}: {self.fee_bps}bp {self.style} fee{fee_note}, "
            f"{self.spread_bps}bp spread{spread_note} — a round trip must clear "
            f"about {self.round_trip_bps:.1f}bp before this symbol trades."
        )


def round_trip_cost_bps(
    *,
    symbol: str,
    fees: FeeSchedule,
    spread_bps: Decimal | float | str,
    style: Style = "taker",
    spread_is_assumed: bool = False,
) -> CostEstimate:
    """Cost of a full round trip in basis points.

    This is the only implementation of this calculation in the program.
    """
    spread = Decimal(str(spread_bps))
    if spread < 0:
        spread = Decimal("0")
    half_spread = spread / 2

    warnings: list[str] = []
    if fees.assumed:
        warnings.append(
            f"the fee tier for {symbol} is assumed, not confirmed against this "
            f"account ({fees.source}). A wrong tier moves the trade/no-trade "
            f"line directly."
        )
    if spread_is_assumed:
        warnings.append(
            f"the spread for {symbol} is an assumed default, not a measured "
            f"quote. Live book data replaces it as soon as it arrives."
        )

    if style == "taker":
        fee_bps = fees.taker_bps
        # Two crossings, each paying half the spread == one full spread.
        crossing = half_spread * 2
    else:
        fee_bps = fees.maker_bps
        # A passive fill pays no spread, but is selected against.
        crossing = half_spread * ADVERSE_SELECTION_FRACTION * 2

    # The fee is paid on both legs; the sell-side fee only on the sell leg.
    round_trip = fee_bps * 2 + crossing + fees.sell_side_bps

    return CostEstimate(
        symbol=symbol,
        style=style,
        fee_bps=fee_bps,
        spread_bps=spread,
        crossing_bps=crossing,
        sell_side_bps=fees.sell_side_bps,
        round_trip_bps=round_trip,
        spread_is_assumed=spread_is_assumed,
        fees_are_assumed=fees.assumed,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class GateResult:
    """Whether a symbol's expected edge survives its costs."""

    admitted: bool
    reason: str
    estimate: CostEstimate
    expected_edge_bps: Decimal
    required_bps: Decimal

    @property
    def margin_bps(self) -> Decimal:
        return self.expected_edge_bps - self.required_bps


def gate(
    *,
    expected_edge_bps: Decimal | float | str,
    estimate: CostEstimate,
    safety_multiple: Decimal | float | str = "1.5",
) -> GateResult:
    """Admit a symbol only if its expected edge clears its round-trip cost.

    ``safety_multiple`` exists because the edge estimate is the least reliable
    number in the system: it is an out-of-sample forecast, while the cost is
    close to an accounting fact. Requiring the forecast to beat the fact by a
    margin is what stops a book that trades constantly for nothing.
    """
    edge = Decimal(str(expected_edge_bps))
    required = estimate.round_trip_bps * Decimal(str(safety_multiple))

    if edge <= 0:
        reason = (f"no expected edge (round trip costs "
                  f"{estimate.round_trip_bps:.1f}bp)")
        return GateResult(False, reason, estimate, edge, required)
    if edge < required:
        reason = (f"edge {edge:.1f}bp does not clear {required:.1f}bp "
                  f"({estimate.round_trip_bps:.1f}bp round trip x "
                  f"{safety_multiple} safety)")
        return GateResult(False, reason, estimate, edge, required)
    reason = (f"edge {edge:.1f}bp clears {required:.1f}bp "
              f"({estimate.round_trip_bps:.1f}bp round trip x {safety_multiple})")
    return GateResult(True, reason, estimate, edge, required)


def spread_bps_from_book(bid: float, ask: float) -> Decimal | None:
    """Measured spread in bps from a live top-of-book, or None if unusable.

    Returning None rather than a default is deliberate: the caller must then
    mark the spread as assumed, and the operator sees that it was assumed.
    """
    if not (bid > 0 and ask > 0 and ask >= bid):
        return None
    mid = (Decimal(str(ask)) + Decimal(str(bid))) / 2
    if mid <= 0:
        return None
    return (Decimal(str(ask)) - Decimal(str(bid))) / mid * 10_000


def estimate_for_symbol(
    symbol: str,
    spec: VenueSpec,
    *,
    bid: float | None = None,
    ask: float | None = None,
    style: Style = "taker",
    default_spread_bps: Decimal | float | str = "2.0",
) -> CostEstimate:
    """Convenience wrapper: measured spread where available, assumed otherwise."""
    measured = spread_bps_from_book(bid, ask) if bid is not None and ask is not None else None
    if measured is None:
        return round_trip_cost_bps(
            symbol=symbol, fees=spec.fees, spread_bps=default_spread_bps,
            style=style, spread_is_assumed=True,
        )
    return round_trip_cost_bps(
        symbol=symbol, fees=spec.fees, spread_bps=measured, style=style,
        spread_is_assumed=False,
    )
