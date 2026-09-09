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

from dataclasses import dataclass, fields, replace
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


#: What one position has to be worth to be worth holding at all.
#:
#: Not a preference. Three venue facts set it:
#:
#: * Alpaca fills fractional equity orders down to $1 of notional, so anything
#:   under a few dollars cannot be trimmed or partially exited -- the exit is
#:   all-or-nothing at a size where a single tick is a large fraction of it.
#: * Market-on-close and market-on-open orders do **not** support fractional
#:   quantities. A position worth less than one share cannot be held overnight
#:   at all, and the median US listing trades in the tens of dollars.
#: * Non-fractionable symbols need whole shares for any order.
#:
#: Twenty-five dollars buys one share of a large part of the market, which is
#: what keeps the overnight strategy and non-fractionable names reachable.
VIABLE_POSITION_NOTIONAL = 25.0

#: The concurrency the base limits were written for. Used as the reference point
#: the scaling below measures against, so the two cannot drift apart.
REFERENCE_POSITIONS = 5

#: Ceilings on what scaling may reach, so a very small account cannot talk
#: itself into limits that are not risk management any more.
MAX_SCALED_RISK_PER_TRADE = 0.02
MAX_SCALED_DAILY_LOSS_HALT = 0.10

#: How far past the per-trade budget raising a position to the venue minimum
#: may go. Some overspend is unavoidable on a small account -- the floor is an
#: absolute amount and the budget is a fraction -- but it is bounded, so a name
#: whose stop is so wide that the smallest tradeable position would risk a
#: multiple of the budget is refused rather than quietly taken.
FLOOR_RISK_MULTIPLE = 2.0


@dataclass(frozen=True)
class AccountScale:
    """How the limits were adjusted for the size of this account, and why.

    A percentage-based risk framework quietly stops working at small balances,
    because the binding constraints stop being relative and start being
    absolute. On a $70 account the base limits allow five concurrent positions
    of $11 each; a 2%-ATR stock sizes to $7. Alpaca will accept that as a
    fractional order and it is not a trade -- it cannot be held overnight (no
    fractional auction orders), it cannot be taken at all in a non-fractionable
    name, and a single cent of spread is a meaningful fraction of it.

    So the account's size decides how many positions it can carry, and
    concentration follows from that rather than from a fixed percentage. As the
    balance grows the scaling converges on the base limits exactly -- above a
    few hundred dollars this changes nothing.

    Deliberately *not* scaled: ``atr_stop_multiple`` and ``target_volatility``.
    Both describe the market, not the wallet. How far a stock moves before a
    stop is a property of the stock; widening it because the account is small
    would size the market to the balance, and it also cuts the position for a
    given risk budget, which is the opposite of what a small account needs. The
    lever that makes positions viable is the risk budget itself, and that is
    scaled.
    """

    equity: float
    positions: int
    max_position_weight: float
    risk_per_trade: float
    daily_loss_halt: float
    #: What the smallest permitted position is worth, in currency.
    position_floor: float
    #: True when the account is large enough that nothing was adjusted.
    unscaled: bool
    note: str


def scale_for_equity(equity: float, base: RiskLimits | None = None) -> AccountScale:
    """Derive the limits this balance can actually trade under.

    Derivation, not optimisation: nothing here is searchable and nothing here
    can be widened by a strategy. It reads one input the strategy does not
    control -- the account balance -- and every adjustment tightens or
    concentrates in response to it.
    """
    base = base or RiskLimits()
    if equity <= 0:
        return AccountScale(
            equity=equity, positions=base.max_concurrent_positions,
            max_position_weight=base.max_position_weight,
            risk_per_trade=base.risk_per_trade,
            daily_loss_halt=base.daily_loss_halt,
            position_floor=VIABLE_POSITION_NOTIONAL, unscaled=True,
            note="the account balance is not known yet, so the base limits apply")

    deployable = equity * base.max_gross_exposure
    # How many positions of a size worth holding this balance can carry at once.
    positions = int(deployable // VIABLE_POSITION_NOTIONAL)
    positions = max(1, min(base.max_concurrent_positions, positions))

    if positions >= base.max_concurrent_positions:
        return AccountScale(
            equity=equity, positions=base.max_concurrent_positions,
            max_position_weight=base.max_position_weight,
            risk_per_trade=base.risk_per_trade,
            daily_loss_halt=base.daily_loss_halt,
            position_floor=VIABLE_POSITION_NOTIONAL, unscaled=True,
            note=(f"${equity:,.0f} carries the full "
                  f"{base.max_concurrent_positions} positions; base limits apply"))

    concentration = REFERENCE_POSITIONS / positions
    # Gross is unchanged: the book still risks the same fraction of itself in
    # total. It is spread over fewer names, so each may be larger.
    weight = min(1.0, base.max_gross_exposure / positions)
    # The per-trade budget follows the concentration, or it becomes the binding
    # constraint and hands back the size the position cap just allowed.
    risk = min(MAX_SCALED_RISK_PER_TRADE, base.risk_per_trade * concentration)
    # A concentrated book has larger single-name moves, so a daily band written
    # for five positions halts on ordinary noise when there are two.
    halt = min(MAX_SCALED_DAILY_LOSS_HALT, base.daily_loss_halt * concentration)

    return AccountScale(
        equity=equity, positions=positions, max_position_weight=weight,
        risk_per_trade=risk, daily_loss_halt=halt,
        position_floor=VIABLE_POSITION_NOTIONAL, unscaled=False,
        note=(f"${equity:,.2f} supports {positions} concurrent "
              f"{'position' if positions == 1 else 'positions'} of at least "
              f"${VIABLE_POSITION_NOTIONAL:,.0f}; concentrated to "
              f"{weight:.0%} per name with a {risk:.2%} per-trade budget"))


def limits_for_equity(equity: float, base: RiskLimits | None = None) -> RiskLimits:
    """The base limits, adjusted for what this balance can actually trade."""
    base = base or RiskLimits()
    scale = scale_for_equity(equity, base)
    if scale.unscaled:
        return base
    return replace(
        base,
        max_position_weight=scale.max_position_weight,
        max_concurrent_positions=scale.positions,
        risk_per_trade=scale.risk_per_trade,
        daily_loss_halt=scale.daily_loss_halt,
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
