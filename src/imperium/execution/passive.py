"""Entries that rest at the mid first, and cross only if they have to.

Every entry this program made crossed the spread: a market order buys at the
ask and sells at the bid, so it pays half the spread on the way in, every
time. On a small account that is a large share of the whole edge. A limit
order at the midpoint pays nothing for the crossing *if it fills* -- and the
risk it takes to find out is the risk of not filling at all.

So an entry rests at the mid for a short window, and if the market has not
come to it by then, the remainder goes as an ordinary market order. **The
trade still happens.** The worst case is the market order that would have
been sent anyway, a few seconds later; the best is half a spread saved. What
this never does is let a decided trade quietly not happen because a limit
price was never touched.

WHAT RESTS
----------

Only an order that opens or adds to a position, sent as an ordinary market
order, with a fresh two-sided quote wide enough to have a midpoint on the tick
grid. Everything else is sent exactly as before:

* **exits and reductions** -- a stop, a give-back exit, a retirement, a
  flatten: the reason to leave is usually the reason not to wait;
* **auction orders**, which have their own price;
* a quote that is **stale, one-sided or crossed** -- a mid built from it would
  be a guess;
* a spread of **one tick** -- there is no price between bid and ask to rest
  at, and joining the bid is a different trade (a queue, not a midpoint).

HOW IT RUNS
-----------

Without waiting. The order is placed and the trading loop moves on; each tick
the broker asks the venue how it is doing. Filled: booked, at its own price.
Past its window: cancelled, whatever filled is booked, and the rest is sent at
market. A partial fill is booked exactly once, however the order ends.

The cost gate still prices every entry as a crossing. It is not told about
this -- deliberately: a fill at the mid is a saving the gate did not count on,
never one it relied on to let a trade through. Real fills still teach it what
execution actually cost, passive ones included, which is the measured truth
of the policy rather than the model's guess about it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

#: The order name recorded on a fill that rested and filled at its limit.
PASSIVE = "passive_mid"

#: Seconds an entry rests before the rest of it crosses. Long enough for a
#: liquid name's midpoint to trade through on an ordinary minute; short enough
#: that a signal on a multi-day horizon has not moved while it waited.
DEFAULT_SECONDS = 20.0
MIN_SECONDS, MAX_SECONDS = 3.0, 120.0

#: A quote older than this is not a quote to price a limit from.
MAX_QUOTE_AGE = 5.0

#: Beyond this spread the "mid" is a guess about where the market is, not a
#: price it trades at. The order crosses as before.
MAX_SPREAD_BPS = 150.0


def enabled() -> bool:
    """IMPERIUM_PASSIVE_ENTRIES, on unless it says otherwise."""
    raw = os.environ.get("IMPERIUM_PASSIVE_ENTRIES", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def window_seconds() -> float:
    try:
        value = float(os.environ.get("IMPERIUM_PASSIVE_SECONDS", DEFAULT_SECONDS))
    except ValueError:
        return DEFAULT_SECONDS
    if not math.isfinite(value):
        return DEFAULT_SECONDS
    return min(MAX_SECONDS, max(MIN_SECONDS, value))


def tick_for(price: float, increment: Decimal | None = None) -> Decimal:
    """The venue's price grid. Its own figure where it publishes one (crypto);
    for US equities, a cent at a dollar and above and a hundredth of a cent
    below, which is the sub-penny rule."""
    if increment is not None and increment > 0:
        return Decimal(increment)
    return Decimal("0.01") if price >= 1.0 else Decimal("0.0001")


def limit_price(side: str, bid: float, ask: float,
                tick: Decimal) -> Decimal | None:
    """The midpoint on the tick grid, rounded toward the passive side -- down
    for a buy, up for a sell -- so rounding can never make it the crossing
    price. None when the spread leaves no such price strictly inside it."""
    if not (bid > 0 and ask > 0 and ask > bid):
        return None
    b, a = Decimal(str(bid)), Decimal(str(ask))
    mid = (b + a) / 2
    rounding = ROUND_FLOOR if side == "buy" else ROUND_CEILING
    price = ((mid / tick).to_integral_value(rounding=rounding) * tick).quantize(tick)
    if not (b < price < a):
        return None
    return price


def why_not(*, side: str, held: Decimal, order: str, bid: float, ask: float,
            quote_age: float) -> str:
    """Why this order should cross rather than rest, or "" if it may rest."""
    if order:
        return "an auction order has its own price"
    if (side == "buy" and held < 0) or (side == "sell" and held > 0):
        return "an exit or a reduction goes at market"
    if not (bid > 0 and ask > 0):
        return "no two-sided quote"
    if ask <= bid:
        return "the quote is crossed"
    if not math.isfinite(quote_age) or quote_age > MAX_QUOTE_AGE:
        return "the quote is stale"
    mid = (bid + ask) / 2
    if (ask - bid) / mid * 10_000 > MAX_SPREAD_BPS:
        return "the spread is too wide for a midpoint to mean anything"
    return ""


@dataclass
class Working:
    """One entry resting at the venue."""

    symbol: str
    side: str
    quantity: Decimal
    limit: Decimal
    client_order_id: str
    order_id: str
    placed_at: float
    deadline: float
    reference_price: Decimal
    strategy: str
    bid: float
    ask: float
    fractionable: bool = True
    #: Quantity already booked from partial fills, and what it cost in all.
    booked: Decimal = Decimal("0")
    booked_value: Decimal = Decimal("0")
    #: A cancel has been sent; the remainder waits for the venue to confirm it.
    cancelling: bool = False

    @property
    def remaining(self) -> Decimal:
        return max(Decimal("0"), self.quantity - self.booked)

    @property
    def signed_remaining(self) -> Decimal:
        return self.remaining if self.side == "buy" else -self.remaining

    def touch_saving(self, filled: Decimal, price: Decimal) -> float:
        """What filling here saved against crossing at the touch it saw."""
        if self.side == "buy":
            return float((Decimal(str(self.ask)) - price) * filled)
        return float((price - Decimal(str(self.bid))) * filled)


@dataclass
class PassiveStats:
    """What resting has done, for the costs panel."""

    rested: int = 0
    filled: int = 0
    partial: int = 0
    crossed: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    saved: float = 0.0

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def as_dict(self, working: int = 0) -> dict[str, Any]:
        done = self.filled + self.partial + self.crossed
        return {"enabled": enabled(), "seconds": window_seconds(),
                "working": working, "rested": self.rested,
                "filled": self.filled, "partial": self.partial,
                "crossed": self.crossed,
                "fill_rate": (self.filled / done) if done else None,
                "saved": round(self.saved, 4),
                "skipped": dict(sorted(self.skipped.items(),
                                       key=lambda kv: -kv[1]))}
