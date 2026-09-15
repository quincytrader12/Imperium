"""The two strategies, and the blend the regime classifier decides.

Momentum and mean reversion are opposite bets on the sign of serial
correlation. Run at a fixed 50/50 they are close to self-annihilating: they take
opposing positions in the same symbol and pay the round-trip cost on both. The
regime classifier's tilt decides the split, and the *blend* is what reaches the
sizer.

Each strategy returns a signal in [-1, +1] **and** an expected edge in basis
points. The edge is what the cost gate consumes -- a signal alone cannot say
whether trading is worth it, because "strongly convinced of a 3bp move" is a
reason not to trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from imperium.strategy.regime import Regime, RegimeVerdict


@dataclass(frozen=True)
class StrategyParams:
    """Everything a parameter search is allowed to vary.

    Every name here appears in ``risk.OPTIMISABLE_PARAMETERS``. No risk limit
    does, and a test enforces that.
    """

    fast_window: int = 12
    slow_window: int = 48
    zscore_window: int = 96
    entry_z: float = 1.5
    exit_z: float = 0.3
    atr_window: int = 14
    warmup_bars: int = 128
    min_edge_bps: float = 1.0
    safety_multiple: float = 1.5


@dataclass(frozen=True)
class Signal:
    """One strategy's view on one symbol at one bar."""

    name: str
    value: float            # [-1, +1]
    expected_edge_bps: float
    reason: str

    @property
    def flat(self) -> bool:
        return self.value == 0.0


def _ema(values: np.ndarray, window: int) -> float:
    if values.size == 0:
        return float("nan")
    alpha = 2.0 / (window + 1.0)
    weights = (1 - alpha) ** np.arange(values.size - 1, -1, -1)
    return float(np.sum(weights * values) / np.sum(weights))


def momentum_signal(prices: np.ndarray, params: StrategyParams,
                    bar_vol: float) -> Signal:
    """Fast/slow EMA separation, scaled by volatility.

    The separation is expressed in units of per-bar volatility rather than in
    percent, so the same threshold means the same thing on a quiet symbol and a
    violent one.
    """
    p = np.asarray(prices, dtype=float)
    if p.size < params.slow_window + 2:
        return Signal("momentum", 0.0, 0.0,
                      f"warming up: {p.size}/{params.slow_window + 2} bars")
    if not math.isfinite(bar_vol) or bar_vol <= 0:
        return Signal("momentum", 0.0, 0.0, "volatility not estimable")

    fast = _ema(p[-params.fast_window * 4:], params.fast_window)
    slow = _ema(p[-params.slow_window * 4:], params.slow_window)
    if not (math.isfinite(fast) and math.isfinite(slow)) or slow <= 0:
        return Signal("momentum", 0.0, 0.0, "moving averages not estimable")

    separation = (fast - slow) / slow
    # In volatility units, so the threshold is comparable across symbols.
    normalised = separation / bar_vol
    value = float(np.clip(normalised / 3.0, -1.0, 1.0))

    # The expected edge is the *unrealised* part of the move: momentum's premise
    # is that a fraction of the current separation persists. Half is a stated
    # assumption, not a measurement, and it is deliberately conservative.
    edge_bps = abs(separation) * 0.5 * 10_000
    direction = "up" if separation > 0 else "down"
    return Signal(
        "momentum", value, edge_bps,
        f"fast EMA is {abs(normalised):.1f} bar-vols {direction} of slow EMA",
    )


def mean_reversion_signal(prices: np.ndarray, params: StrategyParams) -> Signal:
    """Z-score of price against its own rolling mean.

    The signal is the *negative* of the z-score: a price far above its mean is a
    reason to be short, which on a long-only venue means flat.
    """
    p = np.asarray(prices, dtype=float)
    if p.size < params.zscore_window + 2:
        return Signal("mean_reversion", 0.0, 0.0,
                      f"warming up: {p.size}/{params.zscore_window + 2} bars")

    window = p[-params.zscore_window:]
    mean = float(np.mean(window))
    sd = float(np.std(window, ddof=1))
    if sd <= 0 or mean <= 0:
        return Signal("mean_reversion", 0.0, 0.0, "price has not moved")

    z = (float(p[-1]) - mean) / sd
    if abs(z) < params.exit_z:
        return Signal("mean_reversion", 0.0, 0.0,
                      f"z {z:+.2f} is inside the exit band ±{params.exit_z}")
    if abs(z) < params.entry_z:
        return Signal("mean_reversion", 0.0, 0.0,
                      f"z {z:+.2f} has not reached the entry band ±{params.entry_z}")

    value = float(np.clip(-z / (params.entry_z * 2), -1.0, 1.0))
    # The edge is the distance back to the mean, which is what the trade is
    # betting on capturing.
    edge_bps = abs(float(p[-1]) - mean) / mean * 10_000
    side = "above" if z > 0 else "below"
    return Signal(
        "mean_reversion", value, edge_bps,
        f"price is {abs(z):.1f} sigma {side} its {params.zscore_window}-bar mean",
    )


@dataclass(frozen=True)
class BlendedSignal:
    """The combined view that reaches the sizer."""

    value: float
    expected_edge_bps: float
    reason: str
    momentum: Signal
    mean_reversion: Signal
    momentum_share: float


def blend(momentum: Signal, reversion: Signal, verdict: RegimeVerdict) -> BlendedSignal:
    """Combine the two strategies according to the regime's tilt.

    A contradicted or evidence-free regime yields a 50/50 share, and because the
    two strategies then usually disagree, the blend is near zero -- which is the
    correct behaviour. It is *not* special-cased to zero, because the two can
    legitimately agree (a symbol both above its mean and losing momentum), and
    that agreement is real information.
    """
    share = verdict.momentum_share
    value = share * momentum.value + (1.0 - share) * reversion.value
    value = float(np.clip(value, -1.0, 1.0))

    # Edge is blended on the same weights, then reduced when the two disagree:
    # a cancelled position does not earn the edge either component predicted.
    edge = share * momentum.expected_edge_bps + (1 - share) * reversion.expected_edge_bps
    disagree = (momentum.value * reversion.value) < 0
    if disagree:
        agreement_penalty = abs(value) / max(
            abs(momentum.value), abs(reversion.value), 1e-9
        )
        edge *= agreement_penalty

    if verdict.regime is Regime.WARMING_UP:
        reason = verdict.reason
    elif value == 0.0:
        parts = [s.reason for s in (momentum, reversion) if s.flat]
        reason = ("; ".join(parts) if parts
                  else "momentum and mean reversion cancelled exactly")
    else:
        lead = "momentum" if share >= 0.5 else "mean reversion"
        lead_signal = momentum if share >= 0.5 else reversion
        reason = (f"{lead} leads at {max(share, 1 - share):.0%} "
                  f"({verdict.regime.value}): {lead_signal.reason}")

    return BlendedSignal(value, float(max(edge, 0.0)), reason, momentum,
                         reversion, share)
