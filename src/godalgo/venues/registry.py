"""The venue seam.

Everything that differs between venues lives here as *data*: the symbol
vocabulary, the fee schedule, the quote asset, which client speaks the API,
whether shorting exists. A second venue is a new entry in ``VENUES``, not a
rewrite.

The seam is deliberately thin. It is not an abstraction layer over venues -- it
is a table of facts about them, and the client classes it names are each written
directly against their own venue's documented API.

If a second venue is ever added it is a **separate pool of money**: its own
book, its own equity, its own risk limits. Two accounts cannot fund each other,
and a book that sizes a position against combined equity is proposing a trade
that cannot settle -- it will find that out at the venue rather than in the
allocator. :class:`VenueSpec.separate_book` records that this is intended, not
an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable


@dataclass(frozen=True)
class FeeSchedule:
    """Fee rates in basis points.

    ``assumed`` is the important field. A fee tier that was guessed rather than
    read from the account is reported to the operator as a *warning*, because an
    assumption nobody is told about becomes a fact by default -- and this
    particular assumption decides whether a symbol trades at all.
    """

    maker_bps: Decimal
    taker_bps: Decimal
    #: A regulatory or sell-side fee, applied once, on the sell leg only.
    sell_side_bps: Decimal = Decimal("0")
    assumed: bool = True
    source: str = "venue default schedule (not confirmed against this account)"

    def confirmed(self, maker_bps: Decimal, taker_bps: Decimal,
                  source: str) -> "FeeSchedule":
        return FeeSchedule(maker_bps=maker_bps, taker_bps=taker_bps,
                           sell_side_bps=self.sell_side_bps, assumed=False,
                           source=source)


@dataclass(frozen=True)
class VenueSpec:
    venue_id: str
    display_name: str
    base_url: str
    #: Public data host. Sometimes reachable where the main host is blocked,
    #: which helps tell "blocked" apart from "down".
    data_url: str
    ws_url: str
    quote_assets: tuple[str, ...]
    fees: FeeSchedule
    #: Binance Spot has nothing to borrow. A negative target is not a risky
    #: position, it is a rejected order every bar for as long as the signal
    #: points down -- so it is clamped to zero, not to a small long.
    allows_short: bool
    #: Each venue is its own pool of money.
    separate_book: bool = True
    #: A venue's own idea of how a symbol is spelled.
    symbol_style: str = "concat_upper"
    client_factory: str = "godalgo.venues.binance.client:BinanceSpotClient"
    default_universe: tuple[str, ...] = ()
    bar_seconds: int = 60
    #: The market never closes, so a year is every second of it. Using an
    #: equity calendar here would understate annualised volatility by 2.3x and
    #: size every position at 43% of target.
    seconds_per_year: int = 365 * 24 * 3600


BINANCE_SPOT = VenueSpec(
    venue_id="binance_spot",
    display_name="Binance Spot",
    base_url="https://api.binance.com",
    data_url="https://data-api.binance.vision",
    ws_url="wss://stream.binance.com:9443/stream",
    quote_assets=("USDT", "FDUSD", "USDC"),
    fees=FeeSchedule(
        # Binance's standard VIP-0 spot rate is 10bp both sides. This is the
        # published default and is explicitly marked assumed until the account's
        # own commission rates are read back from /api/v3/account/commission.
        maker_bps=Decimal("10"),
        taker_bps=Decimal("10"),
        sell_side_bps=Decimal("0"),
        assumed=True,
        source="Binance VIP-0 published spot schedule (not confirmed for this account)",
    ),
    allows_short=False,
    default_universe=(
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
        "MATICUSDT", "DOTUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
    ),
)

VENUES: dict[str, VenueSpec] = {BINANCE_SPOT.venue_id: BINANCE_SPOT}

DEFAULT_VENUE = BINANCE_SPOT.venue_id


def get(venue_id: str) -> VenueSpec:
    try:
        return VENUES[venue_id]
    except KeyError:
        known = ", ".join(sorted(VENUES))
        raise KeyError(f"unknown venue {venue_id!r}; known venues: {known}") from None


def load_client_factory(spec: VenueSpec) -> Callable[..., Any]:
    """Resolve the ``module:attr`` string into the client class."""
    module_name, _, attr = spec.client_factory.partition(":")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)
