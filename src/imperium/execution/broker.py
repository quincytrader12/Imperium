"""Execution modes: dry run, paper, live.

Three brokers behind one interface, and the interface is deliberately narrow:
``target_weight`` in, fills out. Nothing above this layer knows about order
types.

The guards that matter:

* **Going live requires both a key marked tradeable and a typed confirmation
  phrase.** A one-click button that arms real money is a button that arms real
  money by accident.
* **Switching mode flattens first.** Positions opened in paper are not positions
  in the live account, and carrying the book's *belief* about them across the
  switch means the live broker starts by trying to sell things it does not own.
* **The UI never places an order.** It starts and stops sessions and switches
  mode. A UI that can trade is a second, untested path to the exchange.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from imperium.execution.costs import one_way_cost_bps
from imperium.venues.alpaca.client import AlpacaClient, VenueError
from imperium.venues.alpaca.filters import format_decimal, to_decimal
from imperium.venues.assets import AssetClass, classify_symbol, spec_for
from imperium.venues.registry import VenueSpec

log = logging.getLogger("imperium.broker")

#: The operator must type this exactly. Not a checkbox, not a click.
LIVE_CONFIRMATION_PHRASE = "GO LIVE"


class Mode(str, Enum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    LIVE = "live"


class ModeSwitchRefused(Exception):
    """A mode switch was refused, with the reason the operator needs."""


#: How an order is to reach the market. The overnight strategy is defined by
#: its fills happening at the closing and opening prints, so the order type is
#: part of the strategy rather than an execution detail: a market order sent
#: mid-session instead pays for intraday risk the strategy is not taking.
MARKET = ""
MARKET_ON_CLOSE = "market-on-close"
MARKET_ON_OPEN = "market-on-open"

#: Alpaca's time-in-force codes for the two auction orders. Both are equity
#: only, and both are rejected outright on a crypto symbol.
_AUCTION_TIF = {MARKET_ON_CLOSE: "cls", MARKET_ON_OPEN: "opg"}


@dataclass
class Position:
    symbol: str
    quantity: Decimal = Decimal("0")
    avg_price: Decimal = Decimal("0")

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


@dataclass
class Fill:
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    ts: float
    mode: Mode
    client_order_id: str
    simulated: bool
    note: str = ""
    #: The price the decision was taken at. Without it slippage is not
    #: measurable after the fact, and "did execution cost what the cost gate
    #: assumed" is the question that decides whether the gate is calibrated.
    reference_price: Decimal = Decimal("0")

    @property
    def slippage_bps(self) -> float:
        """Signed cost of crossing, in basis points. Positive is worse."""
        if self.reference_price <= 0:
            return 0.0
        delta = (self.price - self.reference_price) / self.reference_price
        signed = delta if self.side == "BUY" else -delta
        return float(signed) * 10_000

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "side": self.side,
                "quantity": format_decimal(self.quantity),
                "price": format_decimal(self.price), "ts": self.ts,
                "mode": self.mode.value, "simulated": self.simulated,
                "client_order_id": self.client_order_id, "note": self.note,
                "slippage_bps": round(self.slippage_bps, 2),
                "notional": float(self.notional)}


class Broker(Protocol):
    mode: Mode

    async def sync(self) -> None: ...
    async def apply_target(self, symbol: str, target_weight: float,
                           price: float, equity: float, *,
                           order: str = MARKET) -> Fill | None: ...
    async def flatten_all(self, prices: dict[str, float]) -> list[Fill]: ...


class _BaseBroker:
    """Shared position bookkeeping and target-to-order arithmetic."""

    mode: Mode = Mode.DRY_RUN
    simulated = True

    def __init__(self, spec: VenueSpec) -> None:
        self.spec = spec
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.cash: Decimal = Decimal("10000")
        self.realised_pnl: Decimal = Decimal("0")

    def position(self, symbol: str) -> Position:
        p = self.positions.get(symbol)
        if p is None:
            p = Position(symbol)
            self.positions[symbol] = p
        return p

    def equity(self, prices: dict[str, float]) -> Decimal:
        total = self.cash
        for sym, pos in self.positions.items():
            price = prices.get(sym)
            if price:
                total += pos.quantity * to_decimal(price)
        return total

    def weight_of(self, symbol: str, price: float, equity: float) -> float:
        if equity <= 0 or price <= 0:
            return 0.0
        pos = self.position(symbol)
        return float(pos.quantity * to_decimal(price)) / equity

    def _delta_quantity(self, symbol: str, target_weight: float, price: float,
                        equity: float) -> Decimal:
        if price <= 0 or equity <= 0:
            return Decimal("0")
        target_value = to_decimal(target_weight) * to_decimal(equity)
        target_qty = target_value / to_decimal(price)
        return target_qty - self.position(symbol).quantity

    #: Venue quote/base precision. Carrying more digits than this is noise: it
    #: is unreadable in the journal and finer than anything the venue accepts.
    _QUANTUM = Decimal("0.00000001")

    def _record(self, symbol: str, qty: Decimal, price: Decimal, coid: str,
                note: str = "", reference_price: Decimal | None = None) -> Fill:
        qty = qty.quantize(self._QUANTUM)
        price = price.quantize(self._QUANTUM)
        pos = self.position(symbol)
        side = "BUY" if qty > 0 else "SELL"
        if qty > 0:
            total_cost = pos.avg_price * pos.quantity + price * qty
            pos.quantity += qty
            pos.avg_price = (total_cost / pos.quantity) if pos.quantity else Decimal("0")
            self.cash -= price * qty
        else:
            closed = min(-qty, pos.quantity)
            self.realised_pnl += (price - pos.avg_price) * closed
            pos.quantity += qty
            self.cash -= price * qty
            if pos.quantity <= 0:
                pos.quantity = Decimal("0")
                pos.avg_price = Decimal("0")
        fill = Fill(symbol, side, abs(qty), price, time.time(), self.mode, coid,
                    self.simulated, note,
                    reference_price=(reference_price if reference_price is not None
                                     else price))
        self.fills.append(fill)
        return fill

    async def sync(self) -> None:
        return None

    async def flatten_all(self, prices: dict[str, float]) -> list[Fill]:
        out: list[Fill] = []
        for symbol, pos in list(self.positions.items()):
            if pos.is_flat:
                continue
            price = prices.get(symbol)
            if not price:
                log.warning(
                    "cannot flatten %s: no price available. The position is "
                    "still open at the venue.", symbol)
                continue
            fill = await self.apply_target(symbol, 0.0, price,
                                           float(self.equity(prices)))
            if fill:
                out.append(fill)
        return out


class DryRunBroker(_BaseBroker):
    """Evaluates everything and places nothing.

    Distinct from paper: paper maintains a simulated book, dry run maintains no
    book at all. It is the mode for watching the scanner reason without any
    position state to reconcile.
    """

    mode = Mode.DRY_RUN
    simulated = True

    async def apply_target(self, symbol: str, target_weight: float, price: float,
                           equity: float, *, order: str = MARKET) -> Fill | None:
        return None


class PaperBroker(_BaseBroker):
    """A simulated book filled at the touch, paying the modelled crossing cost.

    Fills are charged the taker cost from the cost gate rather than filling at
    the mid, because a paper book that fills at the mid shows an edge that does
    not exist and will not survive contact with the venue.
    """

    mode = Mode.PAPER
    simulated = True

    def __init__(self, spec: VenueSpec, starting_cash: Decimal = Decimal("10000")) -> None:
        super().__init__(spec)
        self.cash = starting_cash

    async def apply_target(self, symbol: str, target_weight: float, price: float,
                           equity: float, *, order: str = MARKET) -> Fill | None:
        delta = self._delta_quantity(symbol, target_weight, price, equity)
        if delta == 0:
            return None
        # Charged through the one cost module, using this symbol's own asset
        # class -- an equity fill charged crypto commission would show a paper
        # book far worse than the real one, and the reverse is worse still.
        model = spec_for(classify_symbol(symbol)).cost_model
        slip = one_way_cost_bps(
            fees=model, spread_bps=model.default_spread_bps, style="taker",
        ) / Decimal("10000")
        fill_price = to_decimal(price) * (1 + slip if delta > 0 else 1 - slip)
        coid = AlpacaClient.new_client_order_id("paper")
        note = "simulated fill, charged taker cost"
        if order:
            # An auction fill is simulated at the last price like any other,
            # which flatters it: the closing and opening prints are their own
            # auctions and neither is the last trade. Said here so the paper
            # book is not read as evidence the overnight strategy works.
            note += (f"; {order} simulated at the last trade, which is not the "
                     f"auction price")
        return self._record(symbol, delta, fill_price, coid, note=note,
                            reference_price=to_decimal(price))


class LiveBroker(_BaseBroker):
    """Real orders against a real Alpaca account.

    Constructing this class is not enough to trade -- :meth:`arm` must be called
    with the exact confirmation phrase, and the credential must itself be marked
    tradeable. Two independent gates, because either alone is a single mistake
    away from a real order.
    """

    mode = Mode.LIVE
    simulated = False

    def __init__(self, spec: VenueSpec, client: AlpacaClient,
                 credential_name: str) -> None:
        super().__init__(spec)
        self.client = client
        self.credential_name = credential_name
        self._armed = False

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self, phrase: str, credential_is_tradeable: bool) -> None:
        """Both gates, checked here so there is one place to audit."""
        if not credential_is_tradeable:
            raise ModeSwitchRefused(
                f"the credential {self.credential_name!r} is not marked tradeable. "
                "Enable it on that specific key first — storing a key does not "
                "authorise it to place orders."
            )
        if phrase.strip() != LIVE_CONFIRMATION_PHRASE:
            raise ModeSwitchRefused(
                f"type {LIVE_CONFIRMATION_PHRASE!r} exactly to go live. "
                "A one-click button that arms real money is a button that arms "
                "real money by accident."
            )
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    async def sync(self) -> None:
        """Read the real account so the book starts from it, not from zero."""
        account = await self.client.account()
        try:
            self.cash = to_decimal(account.get("cash", 0))
        except Exception:
            self.cash = Decimal("0")
        self.positions.clear()
        for pos in await self.client.positions():
            symbol = pos.get("symbol")
            if not symbol:
                continue
            qty = to_decimal(pos.get("qty", 0))
            avg = to_decimal(pos.get("avg_entry_price", 0))
            if qty != 0:
                self.positions[symbol] = Position(symbol, qty, avg)

    async def apply_target(self, symbol: str, target_weight: float, price: float,
                           equity: float, *, order: str = MARKET) -> Fill | None:
        if not self._armed:
            raise ModeSwitchRefused(
                "the live broker is not armed; no order will be sent")
        delta = self._delta_quantity(symbol, target_weight, price, equity)
        if delta == 0:
            return None

        asset_class = classify_symbol(symbol)
        try:
            asset = await self.client.asset(symbol)
        except VenueError:
            asset = None

        qty = abs(delta)
        if asset is not None:
            if not asset.tradable:
                log.info("not sending an order for %s: the venue lists it as "
                         "not tradable (status %s)", symbol, asset.status)
                return None
            if not asset.fractionable:
                # A fractional quantity on a non-fractionable name is rejected,
                # so it is floored here rather than discovered at the venue.
                qty = qty.to_integral_value(rounding="ROUND_DOWN")
            if asset.min_order_size and qty < asset.min_order_size:
                log.info("not sending an order for %s: %s is below the venue "
                         "minimum %s", symbol, qty, asset.min_order_size)
                return None
        if qty <= 0:
            return None

        side = "buy" if delta > 0 else "sell"
        if side == "sell":
            held = self.position(symbol).quantity
            if held > 0:
                # Never try to sell more than is held; that is a rejection, and
                # rounding is the usual cause.
                qty = min(qty, held)
            if qty <= 0:
                return None

        tif = _AUCTION_TIF.get(order)
        if tif is not None:
            if asset_class is not AssetClass.US_EQUITY:
                # Not a degraded fill -- a rejection. Refusing here keeps the
                # error where it can be read rather than in a venue response.
                log.info("not sending a %s order for %s: auction orders exist "
                         "only for US equities", order, symbol)
                return None
            if not qty == qty.to_integral_value():
                # Auction orders take whole shares only. Rounding down is the
                # only safe direction: rounding up buys more than was sized.
                qty = qty.to_integral_value(rounding="ROUND_DOWN")
                if qty <= 0:
                    log.info("not sending a %s order for %s: it sizes to less "
                             "than one whole share, and auctions do not take "
                             "fractions", order, symbol)
                    return None

        coid = self.client.new_client_order_id("imp")
        result = await self.client.place_order(
            symbol, side, qty=qty, order_type="market", client_order_id=coid,
            time_in_force=tif,
        )
        filled = to_decimal(result.get("filled_qty") or 0)
        avg = to_decimal(result.get("filled_avg_price") or 0)
        # A market order can be accepted but not yet filled; the fill price is
        # then unknown and the last trade is the best available estimate. An
        # auction order is *always* in that state when it is accepted -- the
        # auction has not happened yet -- so its recorded price is provisional
        # until the next account sync overwrites it.
        executed = filled if filled > 0 else qty
        fill_price = avg if avg > 0 else to_decimal(price)
        signed = executed if side == "buy" else -executed
        return self._record(symbol, signed, fill_price,
                            result.get("client_order_id", coid),
                            note=f"venue order {result.get('id')} "
                                 f"[{asset_class.value}"
                                 + (f", {order}]" if order else "]"),
                            reference_price=to_decimal(price))

    async def flatten_symbol(self, symbol: str) -> None:
        """Close a position at the venue rather than from our own quantity.

        Retirement and mode switches must not depend on the book's belief about
        what is held; the venue knows.
        """
        try:
            await self.client.close_position(symbol)
        except VenueError as exc:
            if exc.status != 404:
                raise


async def switch_mode(
    current: Broker,
    new_mode: Mode,
    prices: dict[str, float],
    *,
    make_broker,
    phrase: str = "",
    credential_is_tradeable: bool = False,
) -> tuple[Broker, list[Fill]]:
    """Flatten, then switch. In that order, always.

    Carrying a book across a mode switch means the new broker begins with a
    belief about positions the new venue does not share -- a paper book handed to
    the live broker starts by trying to sell coins the account never held.
    """
    flattened = await current.flatten_all(prices)
    broker = make_broker(new_mode)
    if isinstance(broker, LiveBroker):
        broker.arm(phrase, credential_is_tradeable)
        await broker.sync()
    return broker, flattened
