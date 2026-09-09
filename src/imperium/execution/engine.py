"""The per-symbol decision engine.

One engine per symbol. It owns no money and places no orders; it produces a
:class:`Decision` -- a target weight and the reasoning behind it -- and the
session applies it.

**The allocator is a required constructor argument**, not an optional hook. The
brief's warning is specific: if the portfolio cap reaches the engine through an
optional attribute guarded by ``hasattr``, a rename makes the guard fail
silently forever, and the log will cheerfully say "capped" while applying
nothing. Here the engine cannot be constructed without an allocator, so there is
no configuration in which the clamp is absent, and
``tests/test_portfolio.py::test_the_real_engine_class_applies_the_portfolio_clamp``
asserts it against the real class rather than a stub.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np

from imperium.execution import costs
from imperium.execution.bars import Bar, BarSeries
from imperium.execution.portfolio import PortfolioAllocator, Verdict
from imperium.execution.risk import VIABLE_POSITION_NOTIONAL, RiskLimits
from imperium.execution.sizing import SizingResult, average_true_range, size_position
from imperium.strategy import regime as regime_mod
from imperium.strategy.regime import Regime, RegimeVerdict
from imperium.strategy import overnight as overnight_mod
from imperium.strategy.overnight import (
    OvernightSignal, PooledDrift, SessionPhase,
)
from imperium.strategy import trend as trend_mod
from imperium.strategy.trend import PooledTrend, TrendPhase, TrendSignal
from imperium.strategy.signals import (
    BlendedSignal, StrategyParams, blend, mean_reversion_signal, momentum_signal,
)
from imperium.telemetry.streams import TelemetryHub
from imperium.venues.assets import AssetClass, AssetClassSpec, classify_symbol, spec_for
from imperium.venues.registry import VenueSpec

log = logging.getLogger("imperium.engine")


@dataclass
class Decision:
    """What the engine concluded on its last bar, and why.

    This is the answer to "it says TRADING, so why is there no position".
    """

    symbol: str
    target_weight: float = 0.0
    raw_weight: float = 0.0
    verdict: Verdict = Verdict.UNSCANNED
    reason: str = "not yet scanned"
    regime: str = Regime.WARMING_UP.value
    regime_reason: str = ""
    conviction: float = 0.0
    bars_seen: int = 0
    warmup_bars: int = 0
    expected_edge_bps: float = 0.0
    round_trip_cost_bps: float = 0.0
    required_bps: float = 0.0
    spread_bps: float = 0.0
    #: True when the spread is a default rather than a live quote. Surfaced so
    #: the operator can tell a priced symbol from a guessed one.
    spread_assumed: bool = True
    clamp_binding: str = "none"
    clamp_reason: str = ""
    sizing_reason: str = ""
    cost_warnings: tuple[str, ...] = ()
    price: float = 0.0
    asset_class: str = ""
    #: Which strategy produced this decision. The overnight trade has a
    #: different holding period, a different risk profile and different order
    #: types from the intraday blend, so they are never merged into one number.
    strategy: str = "intraday"
    session_phase: str = ""
    overnight_bps: float = 0.0
    overnight_nights: int = 0
    #: Market-on-close / market-on-open, when the overnight trade is live.
    entry_order: str = ""
    #: True when the decision is to leave an existing position exactly as it
    #: is. Distinct from both "trade to this weight" and "rejected": a carried
    #: position asked to re-target itself pays a spread every evaluation on a
    #: delta of nearly nothing, which turns a low-turnover strategy into a
    #: high-turnover one without changing a single line of its reasoning.
    hold: bool = False
    #: The trend strategy's own state, when it owns this symbol.
    trend_score: float = 0.0
    trend_drift_bps: float = 0.0
    trend_min_hold_days: float = 0.0
    trend_days_held: float = 0.0
    trend_phase: str = ""

    @property
    def warming_up(self) -> bool:
        """Distinct from "seeing no opportunity" everywhere in this program.

        One resolves itself; the other is the bot deciding not to trade. Showing
        the first as the second is how a working bot gets mistaken for a broken
        one.
        """
        return self.bars_seen < self.warmup_bars

    @property
    def distance_to_trading(self) -> float:
        """How close this symbol is to trading, for ordering the reasoning panel.

        Smaller is closer. A warming-up symbol is ranked by how far through
        warmup it is, so the panel shows progress rather than a flat "waiting".
        """
        if self.verdict is Verdict.TRADING and self.target_weight > 0:
            return 0.0
        if self.warming_up:
            fraction = self.bars_seen / max(1, self.warmup_bars)
            return 1.0 + (1.0 - fraction)
        if self.required_bps > 0:
            shortfall = max(0.0, self.required_bps - self.expected_edge_bps)
            return 0.5 + min(0.5, shortfall / max(self.required_bps, 1e-9) * 0.5)
        return 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_weight": round(self.target_weight, 5),
            "raw_weight": round(self.raw_weight, 5),
            "verdict": self.verdict.value,
            "reason": self.reason,
            "regime": self.regime,
            "regime_reason": self.regime_reason,
            "conviction": round(self.conviction, 4),
            "bars_seen": self.bars_seen,
            "warmup_bars": self.warmup_bars,
            "warming_up": self.warming_up,
            "expected_edge_bps": round(self.expected_edge_bps, 2),
            "round_trip_cost_bps": round(self.round_trip_cost_bps, 2),
            "required_bps": round(self.required_bps, 2),
            "spread_bps": round(self.spread_bps, 3),
            "spread_assumed": self.spread_assumed,
            "clamp_binding": self.clamp_binding,
            "clamp_reason": self.clamp_reason,
            "sizing_reason": self.sizing_reason,
            "cost_warnings": list(self.cost_warnings),
            "price": self.price,
            "asset_class": self.asset_class,
            "strategy": self.strategy,
            "session_phase": self.session_phase,
            "overnight_bps": round(self.overnight_bps, 2),
            "overnight_nights": self.overnight_nights,
            "entry_order": self.entry_order,
            "hold": self.hold,
            "trend_score": round(self.trend_score, 3),
            "trend_drift_bps": round(self.trend_drift_bps, 3),
            "trend_min_hold_days": round(self.trend_min_hold_days, 1),
            "trend_days_held": round(self.trend_days_held, 1),
            "trend_phase": self.trend_phase,
            "distance": round(self.distance_to_trading, 4),
        }


class SymbolEngine:
    """Decides a target weight for one symbol."""

    def __init__(
        self,
        symbol: str,
        spec: VenueSpec,
        limits: RiskLimits,
        allocator: PortfolioAllocator,
        telemetry: TelemetryHub,
        params: StrategyParams | None = None,
        thresholds: dict | None = None,
    ) -> None:
        self.symbol = symbol
        self.spec = spec
        #: Everything that differs between asset classes: the calendar the
        #: sizer annualises over, whether the series has session seams, whether
        #: shorting exists, the cost model, and which calibration to use.
        self.asset: AssetClassSpec = spec_for(classify_symbol(symbol))
        self.limits = limits
        #: Required. See the module docstring: an optional clamp is not a clamp.
        self.allocator = allocator
        self.telemetry = telemetry
        self.params = params or StrategyParams()
        self.thresholds = thresholds
        self.series = BarSeries(symbol, bar_seconds=spec.bar_seconds)
        #: Set from the venue's own asset record when available: shortability
        #: and borrow are per-symbol facts, not class-wide ones.
        self.can_short: bool = self.asset.shortable
        self.tradable: bool = self.asset.tradeable
        self.decision = Decision(symbol=symbol, warmup_bars=self.params.warmup_bars)
        self.bid: float | None = None
        self.ask: float | None = None
        #: Daily bars, which is what the overnight decomposition needs: a
        #: bounded intraday ring holds only a few sessions, far too few to
        #: resolve a 3-5bp effect.
        self.daily_bars: list[Bar] = []
        #: The market-wide overnight estimate, set by the session. A symbol's
        #: own history cannot resolve this effect alone.
        self.pooled_drift: PooledDrift | None = None
        self.session_phase: SessionPhase = SessionPhase.CLOSED
        #: The market-wide trend premium, estimated across the universe. Held
        #: here rather than measured per symbol for the same reason as the
        #: overnight drift: one symbol's history cannot resolve it.
        self.pooled_trend: PooledTrend | None = None
        #: How long a trend position has been carried, in days, and whether one
        #: is open at all. Set by the session, which owns the book.
        self.trend_held: bool = False
        self.trend_days_held: float = 0.0

    # -- market data -----------------------------------------------------

    def seed(self, klines: list[list]) -> int:
        return self.series.ingest_klines(klines)

    def set_book(self, bid: float | None, ask: float | None) -> None:
        self.bid, self.ask = bid, ask

    # -- the decision ----------------------------------------------------

    def _thresholds(self) -> dict:
        if self.thresholds is not None:
            return self.thresholds
        return regime_mod.thresholds_for(self.asset.calibration_key)

    def evaluate(self) -> Decision:
        """Evaluate one closed bar. Emits a pulse for work actually done."""
        closes = self.series.closes()
        bars_seen = int(closes.size)
        d = Decision(symbol=self.symbol, bars_seen=bars_seen,
                     warmup_bars=self.params.warmup_bars,
                     asset_class=self.asset.asset_class.value)
        d.price = float(closes[-1]) if bars_seen else 0.0

        if not self.tradable:
            d.verdict = Verdict.REJECTED
            d.reason = (f"{self.asset.display_name} is not traded by this "
                        f"program: {self.asset.note or 'unsupported asset class'}")
            self.decision = d
            self.telemetry.pulse(self.symbol, "refused", d.reason, intensity=0.1)
            return d

        if bars_seen < self.params.warmup_bars:
            d.verdict = Verdict.REJECTED
            d.regime = Regime.WARMING_UP.value
            d.reason = (f"warming up: {bars_seen} of {self.params.warmup_bars} "
                        f"bars — this resolves itself, it is not a refusal to trade")
            self.decision = d
            self.telemetry.pulse(self.symbol, "warmup", d.reason,
                                 intensity=bars_seen / max(1, self.params.warmup_bars))
            return d

        # For an equity these drop the returns that span an overnight or
        # weekend seam. A close-to-open move is not a one-minute return, and
        # leaving it in inflates the volatility that sizes every position and
        # swamps the variance ratio that picks the strategy.
        rets = self.series.log_returns(
            exclude_session_gaps=self.asset.excludes_session_gaps)
        bar_vol = (float(np.std(rets[-self.params.zscore_window:], ddof=1))
                   if rets.size > 8 else float("nan"))

        d.session_phase = self.session_phase.value

        # One symbol, one strategy at a time. Three horizons share this book and
        # blending them would allocate the same capital twice, so the choice is
        # made once, explicitly, and recorded on the decision.
        #
        # A trend position already open owns its symbol until it closes: it is
        # the only thing that knows what the position cost and how long that
        # cost still needs to be carried, and handing it to another strategy
        # mid-life would pay the round trip and collect none of the edge.
        if self.trend_held:
            return self._decide_trend(d, closes)

        # The overnight trade is a different trade, not a variant of the
        # intraday one: a different holding period, no intraday stop, and its
        # own order types. It is evaluated in its own window and never blended
        # with the intraday signal, which would double-count the same capital.
        if (self.session_phase is SessionPhase.CLOSING
                and self.asset.asset_class is AssetClass.US_EQUITY):
            return self._decide_overnight(d, closes, rets)

        # An intraday strategy on an account that cannot close what it opens is
        # not constrained, it is prevented -- a position opened with no day
        # trade left to close it becomes an unplanned overnight hold with an
        # intraday stop behind it. Where the round trip is unavailable, the
        # multi-day horizon is not a fallback; it is the only horizon that
        # exists.
        if self.pdt_subject and not self.allocator.day_trades_available():
            return self._decide_trend(d, closes)

        verdict = regime_mod.classify(closes[-250:], self._thresholds(),
                                      returns=rets[-250:])
        d.regime = verdict.regime.value
        d.regime_reason = verdict.reason

        mom = momentum_signal(closes, self.params, bar_vol)
        rev = mean_reversion_signal(closes, self.params)
        signal: BlendedSignal = blend(mom, rev, verdict)
        d.conviction = signal.value

        estimate = costs.estimate_for_symbol(
            self.symbol, self.asset.asset_class, bid=self.bid, ask=self.ask,
            style="taker",
        )
        d.round_trip_cost_bps = float(estimate.round_trip_bps)
        d.spread_bps = float(estimate.spread_bps)
        d.spread_assumed = estimate.spread_is_assumed
        d.cost_warnings = estimate.warnings
        d.expected_edge_bps = signal.expected_edge_bps

        gate = costs.gate(
            expected_edge_bps=Decimal(str(round(signal.expected_edge_bps, 6))),
            estimate=estimate,
            safety_multiple=Decimal(str(self.params.safety_multiple)),
        )
        d.required_bps = float(gate.required_bps)

        if not gate.admitted:
            d.verdict = Verdict.REJECTED
            d.reason = gate.reason
            self.decision = d
            self.telemetry.pulse(self.symbol, "refused", gate.reason,
                                 intensity=0.25)
            return d

        atr = average_true_range(self.series.highs(), self.series.lows(),
                                 closes, self.params.atr_window)
        sized: SizingResult = size_position(
            signal=signal.value, returns=rets, price=d.price, atr=atr,
            limits=self.limits,
            # The class's own calendar. Annualising an equity over the crypto
            # figure overstates its volatility by 2.31x and sizes every
            # position at 43% of target.
            seconds_per_year=self.asset.seconds_per_year,
            bar_seconds=self.spec.bar_seconds,
            # Class permission AND the venue's per-symbol borrow. A name that
            # is shortable but hard to borrow accepts the order and then fails
            # to locate.
            allows_short=self.asset.shortable and self.can_short,
            # A weight is a fraction; on a small account a fraction can be an
            # amount no venue will trade. The sizer needs the balance to know.
            equity=self.allocator.equity,
        )
        d.raw_weight = sized.weight
        d.sizing_reason = sized.reason

        if sized.weight <= 0:
            d.verdict = Verdict.REJECTED
            d.reason = sized.reason
            self.decision = d
            self.telemetry.pulse(self.symbol, "refused", sized.reason, intensity=0.2)
            return d

        # The clamp. Not optional, not guarded by hasattr.
        clamped = self.allocator.clamp(self.symbol, sized.weight)
        d.target_weight = clamped.weight
        d.clamp_binding = clamped.binding
        d.clamp_reason = clamped.reason

        state = self.allocator.observe(self.symbol)
        if not state.admitted:
            d.verdict = Verdict.NOT_ADMITTED
            d.reason = state.reason or clamped.reason
            self.decision = d
            self.telemetry.pulse(self.symbol, "cap", d.reason, intensity=0.3)
            return d

        d.verdict = Verdict.TRADING
        if clamped.reduced:
            d.reason = (f"{signal.reason}; sized to {sized.weight:.1%} then "
                        f"reduced to {clamped.weight:.1%} by the "
                        f"{clamped.binding}")
            self.telemetry.pulse(self.symbol, "cap", d.reason,
                                 intensity=0.6)
        else:
            d.reason = f"{signal.reason}; {sized.reason}"
            self.telemetry.pulse(self.symbol, "decision", d.reason,
                                 intensity=min(1.0, 0.4 + abs(signal.value) * 0.6))
        self.decision = d
        return d

    @property
    def pdt_subject(self) -> bool:
        """Whether this symbol's round trips count as day trades.

        Equities and options do; crypto does not, because it is not a security
        under FINRA's rule. That single distinction is most of why a very small
        account can trade crypto continuously and equities only across sessions.
        """
        return self.asset.asset_class is not AssetClass.CRYPTO

    def _decide_trend(self, d: Decision, closes: np.ndarray) -> Decision:
        """Carry a multi-day trend, for at least as long as its costs require.

        The horizon is the point. A round trip is paid once per holding period,
        so the cost per day falls the longer the position is carried -- and a
        position held across a session close is not a day trade, which is what
        makes this the only strategy available to an account under the
        pattern-day-trader floor.

        Entering and staying are judged differently on purpose. Entering has to
        justify the whole round trip; staying only has to justify itself,
        because the entry cost is already spent and closing early throws it
        away without collecting the edge it bought.
        """
        d.strategy = "trend"
        d.regime = "trend"

        estimate = costs.estimate_for_symbol(
            self.symbol, self.asset.asset_class, bid=self.bid, ask=self.ask,
            style="taker")
        d.round_trip_cost_bps = float(estimate.round_trip_bps)
        d.spread_bps = float(estimate.spread_bps)
        d.spread_assumed = estimate.spread_is_assumed
        d.cost_warnings = estimate.warnings

        signal: TrendSignal = trend_mod.evaluate(
            self.daily_bars, asset_class=self.asset.asset_class,
            pooled=self.pooled_trend,
            round_trip_bps=float(estimate.round_trip_bps),
            safety_multiple=self.params.safety_multiple)

        d.trend_score = signal.score
        d.trend_drift_bps = signal.drift_bps_per_day
        d.trend_min_hold_days = (signal.min_hold_days
                                 if math.isfinite(signal.min_hold_days) else 0.0)
        d.trend_days_held = self.trend_days_held
        d.conviction = signal.value
        d.expected_edge_bps = signal.expected_edge_bps
        d.regime_reason = signal.reason

        phase = trend_mod.phase_for(
            held=self.trend_held, days_held=self.trend_days_held,
            min_hold_days=signal.min_hold_days, signal=signal)
        d.trend_phase = phase.value

        if phase is TrendPhase.EXIT:
            d.verdict = Verdict.TRADING
            d.target_weight = 0.0
            d.reason = (f"the trend that justified this position has gone "
                        f"(score {signal.score:+.2f}) after "
                        f"{self.trend_days_held:.0f} days — closing it")
            self.decision = d
            self.telemetry.pulse(self.symbol, "decision", d.reason, 0.8)
            return d

        if phase in (TrendPhase.HOLDING, TrendPhase.MATURE):
            # Already carried. The entry cost is spent, so the only question is
            # whether the reason to be here still holds -- and it does, or the
            # branch above would have taken it.
            held_weight = self.allocator.observe(self.symbol).current_weight
            d.verdict = Verdict.TRADING
            d.hold = True
            d.target_weight = held_weight
            d.raw_weight = held_weight
            d.reason = (
                f"carrying the trend: day {self.trend_days_held:.0f} of the "
                f"{signal.min_hold_days:.0f} its {signal.drift_bps_per_day:.2f}"
                f"bp/day needs to cover {estimate.round_trip_bps:.2f}bp of cost"
                if phase is TrendPhase.HOLDING else
                f"the trend has paid for its costs and is still intact "
                f"(score {signal.score:+.2f} after "
                f"{self.trend_days_held:.0f} days)")
            self.decision = d
            self.telemetry.pulse(self.symbol, "decision", d.reason, 0.4)
            return d

        if not signal.eligible:
            d.verdict = Verdict.REJECTED
            d.reason = signal.reason
            self.decision = d
            self.telemetry.pulse(self.symbol, "refused", signal.reason, 0.2)
            return d

        gate = costs.gate(
            expected_edge_bps=Decimal(str(round(signal.expected_edge_bps, 6))),
            estimate=estimate,
            safety_multiple=Decimal(str(self.params.safety_multiple)))
        d.required_bps = float(gate.required_bps)
        if not gate.admitted:
            d.verdict = Verdict.REJECTED
            d.reason = (f"the trend is worth {signal.expected_edge_bps:.1f}bp "
                        f"over {signal.min_hold_days:.0f} days and needs "
                        f"{gate.required_bps:.2f}bp to clear its costs")
            self.decision = d
            self.telemetry.pulse(self.symbol, "refused", d.reason, 0.25)
            return d

        # Sized on daily volatility over the class's own calendar, like every
        # other strategy here. The holding period is days, so the daily series
        # is the right one -- an intraday estimate would describe a risk this
        # position is not taking.
        daily_vol = signal.daily_vol_bps / 10_000.0
        if daily_vol <= 0:
            d.verdict = Verdict.REJECTED
            d.reason = "daily volatility is not estimable for this symbol"
            self.decision = d
            return d
        periods = (365.0 if self.asset.asset_class is AssetClass.CRYPTO else 252.0)
        annual_vol = daily_vol * math.sqrt(periods)
        weight = (self.limits.target_volatility / annual_vol) * signal.value
        weight = min(weight, self.limits.max_position_weight)
        # The loss that bounds a multi-day hold is a multi-day move, not a
        # single bar's ATR -- so the risk budget is measured against a 2-sigma
        # excursion over the planned holding period.
        horizon_sigma = daily_vol * math.sqrt(max(1.0, signal.min_hold_days))
        tail = 2.0 * horizon_sigma
        if tail > 0:
            weight = min(weight, self.limits.risk_per_trade / tail)
        weight = max(0.0, weight)

        equity = self.allocator.equity
        if equity > 0 and weight > 0:
            floor_weight = VIABLE_POSITION_NOTIONAL / equity
            if weight < floor_weight:
                if floor_weight > self.limits.max_position_weight:
                    d.verdict = Verdict.REJECTED
                    d.reason = (
                        f"a trend position here sizes to ${weight * equity:,.2f}, "
                        f"and the ${VIABLE_POSITION_NOTIONAL:,.0f} minimum would "
                        f"exceed the {self.limits.max_position_weight:.0%} "
                        f"per-symbol cap on a ${equity:,.2f} account")
                    self.decision = d
                    self.telemetry.pulse(self.symbol, "refused", d.reason, 0.2)
                    return d
                weight = floor_weight

        d.raw_weight = weight
        d.sizing_reason = (
            f"{self.limits.target_volatility:.0%} target against "
            f"{annual_vol:.0%} annualised daily volatility, capped by a "
            f"{self.limits.risk_per_trade:.2%} budget on a 2-sigma "
            f"{signal.min_hold_days:.0f}-day excursion ({tail * 100:.1f}%)")

        if d.raw_weight <= 0:
            d.verdict = Verdict.REJECTED
            d.reason = d.sizing_reason
            self.decision = d
            return d

        clamped = self.allocator.clamp(self.symbol, d.raw_weight, overnight=True)
        d.target_weight = clamped.weight
        d.clamp_binding = clamped.binding
        d.clamp_reason = clamped.reason

        state = self.allocator.observe(self.symbol)
        if not state.admitted:
            d.verdict = Verdict.NOT_ADMITTED
            d.reason = state.reason or clamped.reason
            self.decision = d
            self.telemetry.pulse(self.symbol, "cap", d.reason, 0.3)
            return d

        d.verdict = Verdict.TRADING
        d.reason = f"{signal.reason}; {d.sizing_reason}"
        self.telemetry.pulse(self.symbol, "decision", d.reason,
                             min(1.0, 0.4 + signal.value * 0.6))
        self.decision = d
        return d

    def _decide_overnight(self, d: Decision, closes: np.ndarray,
                          rets: np.ndarray) -> Decision:
        """Decide whether to carry this symbol through the close.

        Everything here differs from the intraday path, and each difference is
        a property of holding a position while the market is shut:

        * The edge is the pooled overnight drift, shrunk by this symbol's own
          history -- not a serial-correlation signal.
        * The cost gate is the same gate, which is the point: at 3-5bp the
          drift is the same order of magnitude as a round trip, and the
          published replications show costs erasing it. Most symbols are
          refused here, correctly.
        * Sizing uses **overnight** volatility, not the intraday estimate. The
          two are different distributions, and the overnight one has the fatter
          tail.
        * There is no ATR stop. A gap opens through a stop without touching it,
          so risk is bounded by size alone.
        """
        d.strategy = "overnight"
        # Whole shares, or nothing. The closing auction does not take
        # fractional quantities, so a name priced above everything this account
        # may put in one position is unreachable overnight however good the
        # drift is -- and on a small balance that is most of the market. Said
        # plainly, because "no trades overnight" with no reason given is
        # indistinguishable from a broken strategy.
        budget = self.limits.max_position_weight * self.allocator.equity
        if self.allocator.equity > 0 and d.price > budget:
            d.verdict = Verdict.REJECTED
            d.reason = (
                f"one share costs ${d.price:,.2f} and this account can put at "
                f"most ${budget:,.2f} into a single name; the closing auction "
                f"takes whole shares only, so this symbol cannot be held "
                f"overnight until the account is larger")
            self.decision = d
            self.telemetry.pulse(self.symbol, "refused", d.reason, 0.15)
            return d

        signal: OvernightSignal = overnight_mod.evaluate(
            self.daily_bars, pooled=self.pooled_drift)
        d.overnight_bps = signal.shrunk_bps or signal.mean_overnight_bps
        d.overnight_nights = signal.nights
        d.regime = "overnight_drift"
        d.regime_reason = signal.reason
        d.conviction = signal.value

        estimate = costs.estimate_for_symbol(
            self.symbol, self.asset.asset_class, bid=self.bid, ask=self.ask,
            style="taker")
        d.round_trip_cost_bps = float(estimate.round_trip_bps)
        d.spread_bps = float(estimate.spread_bps)
        d.spread_assumed = estimate.spread_is_assumed
        d.cost_warnings = estimate.warnings
        d.expected_edge_bps = signal.expected_edge_bps

        if not signal.eligible:
            d.verdict = Verdict.REJECTED
            d.reason = signal.reason
            self.decision = d
            self.telemetry.pulse(self.symbol, "refused", signal.reason,
                                 intensity=0.2)
            return d

        gate = costs.gate(
            expected_edge_bps=Decimal(str(round(signal.expected_edge_bps, 6))),
            estimate=estimate,
            safety_multiple=Decimal(str(self.params.safety_multiple)))
        d.required_bps = float(gate.required_bps)
        if not gate.admitted:
            d.verdict = Verdict.REJECTED
            # The most common and most important refusal in this strategy: the
            # drift is real and smaller than the cost of capturing it.
            d.reason = (f"overnight drift {signal.expected_edge_bps:.2f}bp does "
                        f"not clear {gate.required_bps:.2f}bp of cost — this is "
                        f"the reason the anomaly is hard to trade, not a fault")
            self.decision = d
            self.telemetry.pulse(self.symbol, "refused", d.reason, intensity=0.3)
            return d

        # Size on the overnight distribution, annualised over 252 nights.
        overnight_sd = signal.overnight_vol_bps / 10_000.0
        if overnight_sd <= 0:
            d.verdict = Verdict.REJECTED
            d.reason = "overnight volatility is not estimable for this symbol"
            self.decision = d
            return d
        annual_vol = overnight_sd * math.sqrt(252.0)
        weight = (self.limits.target_volatility / annual_vol) * signal.value
        weight = min(weight, self.limits.max_position_weight)
        # A gap cannot be stopped out of, so the loss that matters is the tail
        # of the overnight move rather than an ATR stop distance.
        tail = overnight_sd * 3.0
        if tail > 0:
            weight = min(weight, self.limits.risk_per_trade / tail)
        weight = max(0.0, weight)
        # The same venue floor as the intraday path, and it bites hardest here:
        # market-on-close orders take whole shares only, so an overnight
        # position below one share cannot be entered at all.
        equity = self.allocator.equity
        if equity > 0 and weight > 0:
            floor_weight = VIABLE_POSITION_NOTIONAL / equity
            if weight < floor_weight:
                if floor_weight > self.limits.max_position_weight:
                    d.verdict = Verdict.REJECTED
                    d.reason = (
                        f"an overnight position here sizes to "
                        f"${weight * equity:,.2f}; the closing auction takes "
                        f"whole shares only, and raising it to the "
                        f"${VIABLE_POSITION_NOTIONAL:,.0f} minimum would exceed "
                        f"the {self.limits.max_position_weight:.0%} per-symbol "
                        f"cap on a ${equity:,.2f} account")
                    self.decision = d
                    self.telemetry.pulse(self.symbol, "refused", d.reason, 0.2)
                    return d
                weight = floor_weight
        d.raw_weight = weight
        d.sizing_reason = (
            f"{self.limits.target_volatility:.0%} target against "
            f"{annual_vol:.0%} annualised overnight volatility, capped by a "
            f"{self.limits.risk_per_trade:.2%} risk budget on a 3-sigma gap "
            f"({tail * 100:.1f}%) — there is no stop behind an overnight hold")

        if d.raw_weight <= 0:
            d.verdict = Verdict.REJECTED
            d.reason = d.sizing_reason
            self.decision = d
            return d

        # Entered on this close and exited on the next open, which is not a day
        # trade. The PDT ceiling therefore does not apply to it -- see
        # PortfolioAllocator.clamp.
        clamped = self.allocator.clamp(self.symbol, d.raw_weight, overnight=True)
        d.target_weight = clamped.weight
        d.clamp_binding = clamped.binding
        d.clamp_reason = clamped.reason

        state = self.allocator.observe(self.symbol)
        if not state.admitted:
            d.verdict = Verdict.NOT_ADMITTED
            d.reason = state.reason or clamped.reason
            self.decision = d
            self.telemetry.pulse(self.symbol, "cap", d.reason, intensity=0.3)
            return d

        d.verdict = Verdict.TRADING
        # Entering on the close and exiting on the open is what the strategy
        # is; a mid-session market order takes intraday risk it is not paid for.
        d.entry_order = "market-on-close"
        d.reason = (f"{signal.reason}; {d.sizing_reason}")
        self.telemetry.pulse(self.symbol, "decision", d.reason,
                             intensity=min(1.0, 0.5 + signal.value * 0.5))
        self.decision = d
        return d

    def scan(self) -> Decision:
        """Evaluate and emit a scan pulse. Used by the universe sweep."""
        d = self.evaluate()
        self.telemetry.pulse(self.symbol, "scan",
                             f"{d.verdict.value}: {d.reason}", intensity=0.15)
        return d
