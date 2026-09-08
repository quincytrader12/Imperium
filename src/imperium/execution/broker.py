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
from imperium.venues.binance.client import BinanceSpotClient, VenueError
from imperium.venues.binance.filters import format_decimal, to_decimal
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
                           price: float, equity: float) -> Fill | None: ...
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
                           equity: float) -> Fill | None:
        return None


class PaperBroker(_BaseBroker):
    """A simulated book filled at the touch, paying the modelled crossing cost.

    Fills are charged the taker cost from the cost gate rather than filling at
    the mid, because a paper book that fills at the mid shows an edge that does
    not exist and will not survive contact with the venue.
    """

    mode = Mode.PAPER
    simulated = True

    #: Spread assumed for simulated fills when no live book is available.
    #: Labelled as an assumption, like every other assumed spread.
    assumed_spread_bps: Decimal = Decimal("2.0")

    def __init__(self, spec: VenueSpec, starting_cash: Decimal = Decimal("10000")) -> None:
        super().__init__(spec)
        self.cash = starting_cash

    async def apply_target(self, symbol: str, target_weight: float, price: float,
                           equity: float) -> Fill | None:
        delta = self._delta_quantity(symbol, target_weight, price, equity)
        if delta == 0:
            return None
        # Charged through the one cost module, never recomputed here.
        slip = one_way_cost_bps(
            fees=self.spec.fees, spread_bps=self.assumed_spread_bps, style="taker",
        ) / Decimal("10000")
        fill_price = to_decimal(price) * (1 + slip if delta > 0 else 1 - slip)
        coid = BinanceSpotClient.new_client_order_id("paper")
        return self._record(symbol, delta, fill_price, coid,
                            note="simulated fill, charged taker cost",
                            reference_price=to_decimal(price))


class LiveBroker(_BaseBroker):
    """Real orders against a real account.

    Constructing this class is not enough to trade -- :meth:`arm` must be called
    with the exact confirmation phrase, and the credential must itself be marked
    tradeable. Two independent gates, because either one alone is a single
    mistake away from a real order.
    """

    mode = Mode.LIVE
    simulated = False

    def __init__(self, spec: VenueSpec, client: BinanceSpotClient,
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
        """Read real balances so the book starts from the account, not from zero."""
        account = await self.client.account()
        self.positions.clear()
        for bal in account.get("balances", []):
            asset = bal["asset"]
            total = to_decimal(bal.get("free", 0)) + to_decimal(bal.get("locked", 0))
            if total <= 0:
                continue
            if asset in self.spec.quote_assets:
                self.cash = total
                continue
            for quote in self.spec.quote_assets:
                symbol = f"{asset}{quote}"
                self.positions[symbol] = Position(symbol, total, Decimal("0"))
                break

    async def apply_target(self, symbol: str, target_weight: float, price: float,
                           equity: float) -> Fill | None:
        if not self._armed:
            raise ModeSwitchRefused(
                "the live broker is not armed; no order will be sent"
            )
        delta = self._delta_quantity(symbol, target_weight, price, equity)
        if delta == 0:
            return None

        filters = await self.client.filters_for(symbol)
        qty = filters.quantize_qty(abs(delta))
        if qty <= 0:
            return None
        side = "BUY" if delta > 0 else "SELL"
        if side == "SELL":
            # Never try to sell more than is actually held: that is -2010, and
            # rounding is the usual cause.
            qty = min(qty, filters.quantize_qty(self.position(symbol).quantity))
            if qty <= 0:
                return None
        problem = filters.check_order(qty, to_decimal(price),
                                      is_market=True)
        if problem:
            log.info("not sending an order for %s: %s", symbol, problem)
            return None

        coid = self.client.new_client_order_id("gda")
        result = await self.client.place_order(
            symbol, side, quantity=qty, order_type="MARKET", client_order_id=coid,
        )
        executed = to_decimal(result.get("executedQty", qty))
        quote = to_decimal(result.get("cummulativeQuoteQty", 0))
        fill_price = (quote / executed) if executed > 0 else to_decimal(price)
        signed = executed if side == "BUY" else -executed
        return self._record(symbol, signed, fill_price,
                            result.get("clientOrderId", coid),
                            note=f"venue order {result.get('orderId')}",
                            reference_price=to_decimal(price))


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
