"""The venue seam.

Everything that differs between venues lives here as *data*: the endpoints, the
environment, which client speaks the API, and the seed universe. Everything that
differs between **asset classes** -- the calendar, the cost model, whether
shorting exists, which calibration the regime classifier must use -- lives in
:mod:`imperium.venues.assets`, because those differences cut across venues
rather than along them.

That split is the point. A venue is a place; an asset class is a kind of
instrument. Alpaca serves both US equities and crypto, and they need different
arithmetic from the same connection.

If a second venue is ever added it is a **separate pool of money**: its own
book, its own equity, its own risk limits. Two accounts cannot fund each other,
and a book that sizes a position against combined equity is proposing a trade
that cannot settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from imperium.venues.assets import AssetClass, AssetClassSpec, spec_for


@dataclass(frozen=True)
class VenueSpec:
    venue_id: str
    display_name: str
    base_url: str
    #: Market data host. Separate from trading at Alpaca, and separately
    #: reachable -- which helps tell "blocked" apart from "down".
    data_url: str
    ws_url: str
    #: The trade-update stream, which is on the trading host rather than the
    #: data one.
    trade_ws_url: str
    client_factory: str
    #: Asset classes this venue can actually trade through this program.
    asset_classes: tuple[AssetClass, ...]
    #: A seed list, used before a live scan has run and as a fallback when the
    #: scan cannot reach the venue. The real universe is discovered.
    seed_universe: tuple[str, ...] = ()
    bar_seconds: int = 60
    #: Each venue is its own pool of money.
    separate_book: bool = True
    paper_base_url: str = ""
    live_base_url: str = ""
    #: Which market-data feed. 'iex' is free and partial; 'sip' is paid and
    #: consolidated. Which is in use changes what the prices mean.
    default_feed: str = "iex"

    def spec_for_class(self, asset_class: AssetClass | str) -> AssetClassSpec:
        return spec_for(asset_class)


ALPACA = VenueSpec(
    venue_id="alpaca",
    display_name="Alpaca",
    # Paper by default. Going live switches the host as well as arming the
    # broker, because a paper key against the live host returns 401 and looks
    # exactly like a bad key.
    base_url="https://paper-api.alpaca.markets",
    paper_base_url="https://paper-api.alpaca.markets",
    live_base_url="https://api.alpaca.markets",
    data_url="https://data.alpaca.markets",
    ws_url="wss://stream.data.alpaca.markets/v2",
    trade_ws_url="wss://paper-api.alpaca.markets/stream",
    client_factory="imperium.venues.alpaca.client:AlpacaClient",
    asset_classes=(AssetClass.US_EQUITY, AssetClass.CRYPTO),
    default_feed="iex",
    # Liquid, widely held names plus the two largest crypto pairs. This is only
    # a seed: the scanner replaces it with what the venue actually lists and
    # what is actually trading, ranked by turnover.
    seed_universe=(
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
        "TSLA", "AMD", "NFLX", "AVGO", "JPM", "XOM", "UNH", "COST",
        "IWM", "DIA", "PLTR", "SOFI", "INTC", "F", "BAC", "T",
        "BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD",
    ),
)

VENUES: dict[str, VenueSpec] = {ALPACA.venue_id: ALPACA}

DEFAULT_VENUE = ALPACA.venue_id


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
