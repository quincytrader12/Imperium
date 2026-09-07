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

from godalgo.execution import costs
from godalgo.execution.bars import BarSeries
from godalgo.execution.portfolio import PortfolioAllocator, Verdict
from godalgo.execution.risk import RiskLimits
from godalgo.execution.sizing import SizingResult, average_true_range, size_position
from godalgo.strategy import regime as regime_mod
from godalgo.strategy.regime import Regime, RegimeVerdict
from godalgo.strategy.signals import (
    BlendedSignal, StrategyParams, blend, mean_reversion_signal, momentum_signal,
)
from godalgo.telemetry.streams import TelemetryHub
from godalgo.venues.registry import VenueSpec

log = logging.getLogger("godalgo.engine")


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
    clamp_binding: str = "none"
    clamp_reason: str = ""
    sizing_reason: str = ""
    cost_warnings: tuple[str, ...] = ()
    price: float = 0.0

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
            "clamp_binding": self.clamp_binding,
            "clamp_reason": self.clamp_reason,
            "sizing_reason": self.sizing_reason,
            "cost_warnings": list(self.cost_warnings),
            "price": self.price,
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
        self.limits = limits
        #: Required. See the module docstring: an optional clamp is not a clamp.
        self.allocator = allocator
        self.telemetry = telemetry
        self.params = params or StrategyParams()
        self.thresholds = thresholds
        self.series = BarSeries(symbol, bar_seconds=spec.bar_seconds)
        self.decision = Decision(symbol=symbol, warmup_bars=self.params.warmup_bars)
        self.bid: float | None = None
        self.ask: float | None = None

    # -- market data -----------------------------------------------------

    def seed(self, klines: list[list]) -> int:
        return self.series.ingest_klines(klines)

    def set_book(self, bid: float | None, ask: float | None) -> None:
        self.bid, self.ask = bid, ask

    # -- the decision ----------------------------------------------------

    def _thresholds(self) -> dict:
        if self.thresholds is not None:
            return self.thresholds
        return regime_mod.load_calibration()["thresholds"]

    def evaluate(self) -> Decision:
        """Evaluate one closed bar. Emits a pulse for work actually done."""
        closes = self.series.closes()
        bars_seen = int(closes.size)
        d = Decision(symbol=self.symbol, bars_seen=bars_seen,
                     warmup_bars=self.params.warmup_bars)
        d.price = float(closes[-1]) if bars_seen else 0.0

        if bars_seen < self.params.warmup_bars:
            d.verdict = Verdict.REJECTED
            d.regime = Regime.WARMING_UP.value
            d.reason = (f"warming up: {bars_seen} of {self.params.warmup_bars} "
                        f"bars — this resolves itself, it is not a refusal to trade")
            self.decision = d
            self.telemetry.pulse(self.symbol, "warmup", d.reason,
                                 intensity=bars_seen / max(1, self.params.warmup_bars))
            return d

        rets = np.diff(np.log(closes[closes > 0]))
        bar_vol = float(np.std(rets[-self.params.zscore_window:], ddof=1)) if rets.size > 8 else float("nan")

        verdict = regime_mod.classify(closes[-250:], self._thresholds())
        d.regime = verdict.regime.value
        d.regime_reason = verdict.reason

        mom = momentum_signal(closes, self.params, bar_vol)
        rev = mean_reversion_signal(closes, self.params)
        signal: BlendedSignal = blend(mom, rev, verdict)
        d.conviction = signal.value

        estimate = costs.estimate_for_symbol(
            self.symbol, self.spec, bid=self.bid, ask=self.ask, style="taker",
        )
        d.round_trip_cost_bps = float(estimate.round_trip_bps)
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
            limits=self.limits, seconds_per_year=self.spec.seconds_per_year,
            bar_seconds=self.spec.bar_seconds, allows_short=self.spec.allows_short,
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

    def scan(self) -> Decision:
        """Evaluate and emit a scan pulse. Used by the universe sweep."""
        d = self.evaluate()
        self.telemetry.pulse(self.symbol, "scan",
                             f"{d.verdict.value}: {d.reason}", intensity=0.15)
        return d
