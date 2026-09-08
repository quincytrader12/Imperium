"""Risk limits.

``RiskLimits`` is frozen, and it appears in **no** strategy parameter search
space. That is enforced by :data:`OPTIMISABLE_PARAMETERS` and a test, not by
convention: an optimiser that can widen its own position cap does not have one,
and the failure is silent -- the log still says "capped" while the cap moves.

Every limit here **only ever reduces**. A cap that can block an exit is a cap
that traps you in a losing position, so every clamp is expressed as a minimum
against the incoming value and exits are explicitly exempt.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal


@dataclass(frozen=True)
class RiskLimits:
    """Hard bounds. Not tunable, not searchable, not reachable from a strategy."""

    #: Largest fraction of equity one symbol may hold.
    max_position_weight: float = 0.20
    #: Largest total exposure across the whole book.
    max_gross_exposure: float = 0.80
    #: Fraction of equity risked on a single trade at its stop.
    risk_per_trade: float = 0.005
    #: Annualised volatility the book is sized toward.
    target_volatility: float = 0.20
    #: Fraction of buying power never allocated. An account at 100% deployed
    #: cannot act on anything, including an exit.
    buying_power_reserve: float = 0.15
    #: Most symbols that may hold a position at once.
    max_concurrent_positions: int = 5
    #: Loss over one day that halts the book.
    daily_loss_halt: float = 0.04
    #: ATR multiple used for the stop distance.
    atr_stop_multiple: float = 2.5
    #: Equity below which US pattern-day-trader limits apply. Set by
    #: regulation, not by preference, which is why it is here rather than in a
    #: tunable: an account under this that makes a fourth day trade in five
    #: business days is restricted for ninety days.
    pdt_equity_floor: float = 25_000.0
    #: Day trades permitted in the rolling window below that floor. The rule
    #: allows three; stopping at two leaves room for the exit leg of a position
    #: opened earlier in the day, which would otherwise be the trade that trips
    #: it.
    pdt_max_day_trades: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0 < self.max_gross_exposure <= 1:
            raise ValueError("max_gross_exposure must be in (0, 1]")
        if self.max_concurrent_positions < 1:
            raise ValueError("max_concurrent_positions must be at least 1")
        if not 0 <= self.buying_power_reserve < 1:
            raise ValueError("buying_power_reserve must be in [0, 1)")
        if self.max_position_weight > self.max_gross_exposure:
            raise ValueError(
                "max_position_weight cannot exceed max_gross_exposure: a single "
                "position would be allowed to breach the whole-book ceiling"
            )


#: The complete set of names an optimiser or parameter search may vary.
#: RiskLimits fields are deliberately absent, and
#: ``tests/test_risk.py::test_no_risk_limit_is_optimisable`` asserts that by
#: comparing against the dataclass's actual fields, so adding a limit later
#: cannot quietly make it searchable.
OPTIMISABLE_PARAMETERS: frozenset[str] = frozenset({
    "fast_window", "slow_window", "zscore_window", "entry_z", "exit_z",
    "breakout_window", "atr_window", "min_edge_bps", "safety_multiple",
    "warmup_bars",
})


def risk_limit_field_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(RiskLimits))


def assert_search_space_is_safe(search_space: dict | frozenset | set) -> None:
    """Raise if a parameter search would be allowed to move a risk limit."""
    names = set(search_space)
    forbidden = names & risk_limit_field_names()
    if forbidden:
        raise ValueError(
            "a parameter search may not contain risk limits, but it contains "
            f"{sorted(forbidden)}. An optimiser that can widen its own position "
            "cap does not have one."
        )
    unknown = names - OPTIMISABLE_PARAMETERS
    if unknown:
        raise ValueError(
            f"unknown search parameters {sorted(unknown)}; add them to "
            "OPTIMISABLE_PARAMETERS deliberately if they really are safe to vary"
        )
