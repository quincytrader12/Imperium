"""Configuration for the Sector Trend sleeve.

Every number the strategy uses lives here and comes from the environment, with
the paper's values as defaults. Nothing in the running program may change any
of them: a strategy that tunes itself has no out-of-sample period left, and the
backtest that justified it stops meaning anything the first time it adapts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: The liquid SPDR industry ETFs from the brief.
DEFAULT_UNIVERSE = (
    "XLF", "XLK", "XLE", "XLV", "XLI", "XBI", "XLU", "XLP", "XLY", "KRE",
    "XLB", "XLC", "XRT", "XOP", "XLRE", "XHB", "KBE", "XME", "KIE",
)

#: Alpaca's floor on a fractional buy, in dollars.
#:
#: Not a preference of this program -- the venue rejects a buy below it
#: outright. It is here because the sleeve's own arithmetic can produce target
#: positions under a dollar on a small account, and an order the venue will
#: refuse must be caught before it is sent rather than after.
MIN_FRACTIONAL_NOTIONAL = 1.0

#: The client order id prefix that marks an order as this sleeve's.
ORDER_PREFIX = "sectrend"


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _number(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class SectorTrendConfig:
    """What the sleeve is allowed to do, read once at startup."""

    #: Off until a backtest has been reviewed. The brief is explicit about this
    #: and it is the right default for any strategy that has not yet been run
    #: against real history on the machine that will trade it.
    enabled: bool = False
    #: Fraction of account equity this sleeve may use. It never sees the rest.
    allocation: float = 0.20
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    #: Daily volatility target for the whole sleeve.
    target_vol: float = 0.015
    #: 1.0 means no borrowing. The paper uses 2.0; it does not model the margin
    #: interest that Alpaca would charge for it, which is why raising this is a
    #: deliberate config change rather than a default.
    max_leverage: float = 1.0
    rebalance_threshold: float = 0.25
    #: "near_close" computes at 15:45 ET on the day's provisional close;
    #: "next_open" computes after the final close and trades the next morning.
    exec_mode: str = "near_close"
    run_time_et: str = "15:45"

    @property
    def universe_size(self) -> int:
        return len(self.universe)

    def sleeve_equity(self, account_equity: float) -> float:
        return max(0.0, float(account_equity)) * self.allocation

    def smallest_tradable_weight(self, account_equity: float) -> float:
        """The weight below which an order would be refused by the venue.

        Surfaced rather than buried in the order path so the panel can say
        "this account is too small for N of these symbols" before an operator
        watches a run place nothing.
        """
        sleeve = self.sleeve_equity(account_equity)
        if sleeve <= 0:
            return float("inf")
        return MIN_FRACTIONAL_NOTIONAL / sleeve


def from_environment() -> SectorTrendConfig:
    """Read the sleeve's configuration. Never raises on a bad value."""
    raw_universe = os.environ.get("SECTOR_TREND_UNIVERSE", "")
    symbols = tuple(
        s.strip().upper() for s in raw_universe.split(",") if s.strip()
    ) or DEFAULT_UNIVERSE

    mode = os.environ.get("SECTOR_TREND_EXEC_MODE", "near_close").strip().lower()
    if mode not in {"near_close", "next_open"}:
        mode = "near_close"

    return SectorTrendConfig(
        enabled=_flag("SECTOR_TREND_ENABLED", False),
        allocation=max(0.0, min(1.0, _number("SECTOR_TREND_ALLOCATION", 0.20))),
        universe=symbols,
        target_vol=max(0.0, _number("SECTOR_TREND_TARGET_VOL", 0.015)),
        max_leverage=max(0.0, _number("SECTOR_TREND_MAX_LEVERAGE", 1.0)),
        rebalance_threshold=max(
            0.0, _number("SECTOR_TREND_REBALANCE_THRESHOLD", 0.25)),
        exec_mode=mode,
        run_time_et=os.environ.get("SECTOR_TREND_RUN_TIME_ET", "15:45").strip(),
    )
