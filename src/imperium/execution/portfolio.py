"""The portfolio layer.

Running N symbols is not running one symbol N times. N engines each obeying
their own per-symbol cap are collectively unbounded: five engines at a 20% cap
is a 100% book, and twelve is a 240% one.

The rules here, each of which exists because its absence is a specific failure:

* **A gross exposure ceiling** across the whole book.
* **Per-symbol budget = ceiling / max_concurrent, never / current count.**
  Dividing by the live count lets the first symbol admitted claim the entire
  book, and forces every later one to trade at a fraction of it -- allocation by
  arrival order rather than by merit. It also makes every existing engine's
  budget change whenever an unrelated symbol is admitted.
* **Buying power carries a reserve** never allocated. An account at 100%
  deployed cannot act on anything, including an exit.
* **Retiring a symbol means flattening it**, not just unsubscribing. A stopped
  engine still holding a position is a position with nothing managing its stop.
* **Every limit only ever reduces.** A clamp that can raise a target is not a
  limit, and a clamp that can block an exit traps you in a losing position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from imperium.execution.risk import RiskLimits

log = logging.getLogger("imperium.portfolio")


class Verdict(str, Enum):
    """Why a symbol is or is not trading. The middle two are different things.

    ``NOT_ADMITTED`` is not a fault of the symbol -- the concurrency limit is
    full. Showing it as ``REJECTED`` tells the operator the scanner dislikes a
    symbol it actually likes.
    """

    TRADING = "trading"
    NOT_ADMITTED = "not_admitted"
    REJECTED = "rejected"
    UNSCANNED = "unscanned"


@dataclass
class SymbolState:
    symbol: str
    verdict: Verdict = Verdict.UNSCANNED
    reason: str = "not yet scanned"
    target_weight: float = 0.0
    current_weight: float = 0.0
    turnover: float = 0.0
    admitted: bool = False
    score: float = 0.0


@dataclass(frozen=True)
class ClampResult:
    weight: float
    binding: str
    reason: str
    reduced: bool


class PortfolioAllocator:
    """Bounds the whole book, and hands each engine its own budget."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.states: dict[str, SymbolState] = {}
        self.equity: float = 0.0
        self.cash: float = 0.0
        self.halted: bool = False
        self.halt_reason: str = ""
        #: Day trades already used in the rolling five-business-day window, as
        #: the venue counts them. Believing the venue rather than counting
        #: locally is the only way to be right across restarts.
        self.day_trade_count: int = 0
        self.flagged_pattern_day_trader: bool = False
        #: Set when the market is closed, so the book stops taking exposure it
        #: cannot actually get filled on.
        self.market_open: bool = True
        self.market_note: str = ""

    # -- budgets ---------------------------------------------------------

    def pdt_blocked(self) -> str:
        """Why a new position would breach pattern-day-trader limits, if it would.

        A US margin account under $25,000 equity may make three day trades in
        five rolling business days; the fourth flags the account and restricts
        it for ninety days. An autonomous book will hit that within a morning if
        nothing stops it, and being restricted is far more expensive than any
        trade it would have made.

        This blocks *opening* exposure only. Closing is always allowed -- a
        limit that traps a position is worse than the limit it enforces.
        """
        if self.equity <= 0 or self.equity >= self.limits.pdt_equity_floor:
            return ""
        if self.day_trade_count >= self.limits.pdt_max_day_trades:
            return (f"pattern-day-trader limit: {self.day_trade_count} day trades "
                    f"used and equity is {self.equity:,.0f}, below the "
                    f"{self.limits.pdt_equity_floor:,.0f} floor")
        return ""

    @property
    def per_symbol_budget(self) -> float:
        """Ceiling divided by the *maximum* concurrency, never the current count.

        This is a constant for a given configuration. That is the point: an
        engine's budget must not change because some unrelated symbol was
        admitted or retired.
        """
        budget = self.limits.max_gross_exposure / self.limits.max_concurrent_positions
        return min(budget, self.limits.max_position_weight)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(s.current_weight) for s in self.states.values())

    @property
    def admitted_symbols(self) -> list[str]:
        return [s.symbol for s in self.states.values() if s.admitted]

    def buying_power(self) -> float:
        """Deployable cash, with the reserve withheld."""
        return max(0.0, self.cash * (1.0 - self.limits.buying_power_reserve))

    # -- admission -------------------------------------------------------

    def observe(self, symbol: str) -> SymbolState:
        state = self.states.get(symbol)
        if state is None:
            state = SymbolState(symbol=symbol)
            self.states[symbol] = state
        return state

    def set_scan(self, symbol: str, *, score: float, turnover: float,
                 tradeable: bool, reason: str) -> SymbolState:
        state = self.observe(symbol)
        state.score = score
        state.turnover = turnover
        if not tradeable:
            state.verdict = Verdict.REJECTED
            state.reason = reason
        return state

    def rebalance_admissions(self) -> tuple[list[str], list[str]]:
        """Choose which symbols hold the concurrency slots.

        Returns (admitted, retired). A symbol that already holds a position is
        never displaced by a higher-scoring newcomer: churning positions to chase
        a marginally better score pays a full round trip for the privilege.
        """
        eligible = [s for s in self.states.values() if s.verdict is not Verdict.REJECTED]
        holding = [s for s in eligible if abs(s.current_weight) > 0]
        idle = sorted((s for s in eligible if abs(s.current_weight) == 0),
                      key=lambda s: (-s.score, -s.turnover, s.symbol))

        slots = self.limits.max_concurrent_positions
        chosen: list[SymbolState] = holding[:slots]
        for state in idle:
            if len(chosen) >= slots:
                break
            chosen.append(state)

        chosen_names = {s.symbol for s in chosen}
        newly_admitted: list[str] = []
        retired: list[str] = []

        for state in self.states.values():
            if state.verdict is Verdict.REJECTED:
                if state.admitted:
                    state.admitted = False
                    retired.append(state.symbol)
                continue
            if state.symbol in chosen_names:
                if not state.admitted:
                    newly_admitted.append(state.symbol)
                state.admitted = True
                if state.verdict is not Verdict.TRADING:
                    state.verdict = Verdict.TRADING
            else:
                if state.admitted:
                    state.admitted = False
                    retired.append(state.symbol)
                state.verdict = Verdict.NOT_ADMITTED
                state.reason = (
                    f"selected, but the concurrency limit of "
                    f"{self.limits.max_concurrent_positions} is full — this is not "
                    f"a fault of {state.symbol}"
                )
        return newly_admitted, retired

    # -- clamping --------------------------------------------------------

    def clamp(self, symbol: str, desired_weight: float, *,
              overnight: bool = False) -> ClampResult:
        """Reduce a desired weight to what the book can afford.

        Never increases it, and never blocks a reduction. An exit is a reduction
        by definition, so it passes through every branch below untouched -- which
        is checked by a test, because a cap that can block an exit is a cap that
        traps you in a losing position.

        ``overnight`` marks a position that is entered on one session's close and
        exited on the next session's open. That is not a day trade under the
        pattern-day-trader rule, which counts a purchase and a sale of the same
        security *within one session*, so the PDT guard below does not apply to
        it. This is a genuine structural advantage of the overnight strategy on
        a small account rather than a loosened limit: an account under $25,000
        can hold overnight positions every night of the week without ever
        approaching the three-day-trade ceiling.
        """
        state = self.observe(symbol)
        current = state.current_weight

        if self.halted:
            # A halt stops new and increased exposure. It must never prevent a
            # reduction, or the halt itself becomes the risk.
            if abs(desired_weight) <= abs(current):
                return ClampResult(desired_weight, "halt", 
                                   f"book halted ({self.halt_reason}); reductions "
                                   "still pass", reduced=False)
            return ClampResult(current, "halt",
                               f"book halted ({self.halt_reason}); no new or "
                               "increased exposure", reduced=True)

        if abs(desired_weight) <= abs(current):
            return ClampResult(desired_weight, "none",
                               "a reduction is never clamped", reduced=False)

        # Beyond this point the request increases exposure, so the two
        # market-state guards apply. Both are deliberately placed after the
        # reduction check above: neither may ever block an exit.
        if not self.market_open:
            return ClampResult(current, "market closed",
                               self.market_note or "the market is closed, so no "
                               "new exposure is taken", reduced=True)

        pdt = "" if overnight else self.pdt_blocked()
        if pdt:
            return ClampResult(current, "pattern day trader", pdt, reduced=True)

        if not state.admitted:
            return ClampResult(
                min(desired_weight, current) if desired_weight >= 0 else current,
                "not admitted",
                f"{symbol} does not hold a concurrency slot", reduced=True,
            )

        allowed = desired_weight
        binding = "none"
        reason = "within every limit"

        budget = self.per_symbol_budget
        if allowed > budget:
            allowed, binding = budget, "per-symbol budget"
            reason = (f"per-symbol budget is "
                      f"{self.limits.max_gross_exposure:.0%} / "
                      f"{self.limits.max_concurrent_positions} = {budget:.1%}")

        # Room left under the whole-book ceiling, excluding this symbol's own
        # existing weight so a symbol is not made to compete with itself.
        others = self.gross_exposure - abs(current)
        headroom = max(0.0, self.limits.max_gross_exposure - others)
        if allowed > headroom:
            allowed, binding = headroom, "gross exposure ceiling"
            reason = (f"the book is at {others:.1%} of a "
                      f"{self.limits.max_gross_exposure:.0%} ceiling, leaving "
                      f"{headroom:.1%}")

        if self.equity > 0:
            affordable = self.buying_power() / self.equity + abs(current)
            if allowed > affordable:
                allowed, binding = max(0.0, affordable), "buying power"
                reason = (f"only {self.buying_power():,.0f} of buying power is "
                          f"deployable ({self.limits.buying_power_reserve:.0%} of "
                          f"cash is held in reserve)")

        allowed = min(allowed, desired_weight)          # only ever reduces
        return ClampResult(float(allowed), binding, reason,
                           reduced=allowed < desired_weight)

    # -- halt ------------------------------------------------------------

    def set_halt(self, halted: bool, reason: str = "") -> None:
        self.halted = halted
        self.halt_reason = reason
        if halted:
            log.warning("book halted: %s", reason)

    def check_daily_loss(self, day_start_equity: float) -> bool:
        """Halt the book if the day's loss breaches the limit."""
        if day_start_equity <= 0 or self.equity <= 0:
            return self.halted
        drawdown = (day_start_equity - self.equity) / day_start_equity
        if drawdown >= self.limits.daily_loss_halt:
            self.set_halt(True, f"daily loss {drawdown:.2%} reached the "
                                f"{self.limits.daily_loss_halt:.2%} limit")
        return self.halted
