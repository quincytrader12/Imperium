"""Position sizing.

Two bounds that answer different questions, and both are applied:

* **Volatility targeting** bounds the *portfolio's* volatility. It asks: how
  large must this position be for the book to run at its target volatility?
* **The ATR risk cap** bounds what *one trade* can lose. It asks: how large can
  this position be so that being stopped out costs a fixed fraction of equity?

Neither subsumes the other -- vol targeting says nothing about a single bad
trade, and the risk cap says nothing about the book's aggregate volatility -- so
the binding one wins.

**Annualisation.** Crypto never closes, so a year is ``365 * 24 * 3600``
seconds. Using an equity calendar (``252 * 6.5 * 3600``) for a crypto pair
understates its annualised volatility by a factor of about 2.3, which sizes
every position at roughly 43% of target. The figure comes from the venue
registry, per venue, rather than being written here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from imperium.execution.risk import (
    FLOOR_RISK_MULTIPLE, VIABLE_POSITION_NOTIONAL, RiskLimits,
)


@dataclass(frozen=True)
class SizingResult:
    """A target weight, and which constraint produced it."""

    weight: float
    binding: str
    vol_target_weight: float
    risk_cap_weight: float
    annualised_vol: float
    reason: str

    @property
    def is_zero(self) -> bool:
        return self.weight == 0.0


def annualised_volatility(returns: np.ndarray, seconds_per_year: int,
                          bar_seconds: int) -> float:
    """Annualise per-bar volatility over the correct number of bars per year."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return float("nan")
    per_bar = float(np.std(r, ddof=1))
    if not math.isfinite(per_bar) or per_bar <= 0:
        return float("nan")
    bars_per_year = seconds_per_year / max(1, bar_seconds)
    return per_bar * math.sqrt(bars_per_year)


def average_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       window: int = 14) -> float:
    """Wilder's ATR. Falls back to the high-low range where no prior close exists."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = min(h.size, l.size, c.size)
    if n < 2:
        return float("nan")
    h, l, c = h[-n:], l[-n:], c[-n:]
    prev_close = c[:-1]
    tr = np.maximum.reduce([
        h[1:] - l[1:],
        np.abs(h[1:] - prev_close),
        np.abs(l[1:] - prev_close),
    ])
    tr = tr[np.isfinite(tr)]
    if tr.size == 0:
        return float("nan")
    w = min(window, tr.size)
    return float(np.mean(tr[-w:]))


def size_position(
    *,
    signal: float,
    returns: np.ndarray,
    price: float,
    atr: float,
    limits: RiskLimits,
    seconds_per_year: int,
    bar_seconds: int,
    allows_short: bool,
    equity: float = 0.0,
    position_floor: float = VIABLE_POSITION_NOTIONAL,
) -> SizingResult:
    """Return a target portfolio weight in [0, max_position_weight].

    ``signal`` is the strategy's conviction in [-1, +1].

    ``equity`` and ``position_floor`` add the bound that only exists on a small
    account: a weight is a fraction, and a fraction of a small balance can be
    an amount no venue will trade. On a $70 book a 2%-ATR name sizes to $17.50
    -- Alpaca accepts that as a fractional order, and it still cannot be held
    overnight, cannot be taken at all in a non-fractionable name, and cannot be
    trimmed. Below the floor the position is either raised to it, if that stays
    inside the per-symbol cap, or refused with the reason -- never quietly
    submitted at a size that cannot do what the strategy intends.
    """
    if not math.isfinite(signal) or signal == 0.0:
        return SizingResult(0.0, "no signal", 0.0, 0.0, float("nan"),
                            "the strategy has no view on this symbol")

    if signal < 0 and not allows_short:
        # Binance Spot has nothing to borrow. A short is not a risky position
        # here, it is a rejected order every bar for as long as the signal points
        # down -- so the target is clamped to zero, not to a small long.
        return SizingResult(
            0.0, "long-only venue", 0.0, 0.0, float("nan"),
            "signal is short and this venue is long-only, so the target is zero "
            "(clamped to flat, not to a small long)",
        )

    ann_vol = annualised_volatility(returns, seconds_per_year, bar_seconds)
    if not math.isfinite(ann_vol) or ann_vol <= 0:
        return SizingResult(0.0, "volatility unknown", 0.0, 0.0, ann_vol,
                            "not enough return history to estimate volatility")

    conviction = float(np.clip(abs(signal), 0.0, 1.0))

    # 1. Volatility targeting: bounds the book's volatility.
    vol_weight = (limits.target_volatility / ann_vol) * conviction

    # 2. ATR risk cap: bounds what one trade can lose.
    if math.isfinite(atr) and atr > 0 and price > 0:
        stop_fraction = (limits.atr_stop_multiple * atr) / price
        risk_weight = (limits.risk_per_trade / stop_fraction) if stop_fraction > 0 else 0.0
    else:
        # Without a usable stop distance there is no risk cap, so vol targeting
        # is the only bound. Reported rather than silently skipped.
        risk_weight = float("inf")

    weight = min(vol_weight, risk_weight, limits.max_position_weight)
    weight = max(0.0, weight)

    if weight == limits.max_position_weight:
        binding = "per-symbol cap"
        reason = (f"capped at the {limits.max_position_weight:.0%} per-symbol limit "
                  f"(vol target wanted {vol_weight:.1%})")
    elif risk_weight <= vol_weight:
        binding = "ATR risk cap"
        reason = (f"{limits.risk_per_trade:.2%} risk per trade at a "
                  f"{limits.atr_stop_multiple}x ATR stop allows {weight:.1%} "
                  f"(vol target wanted {vol_weight:.1%})")
    else:
        binding = "volatility target"
        reason = (f"{limits.target_volatility:.0%} target vol against "
                  f"{ann_vol:.0%} annualised allows {weight:.1%}")

    # The small-account bound. Applied last, because it is about what the venue
    # will do with the result rather than about how large the position should be.
    if equity > 0 and position_floor > 0 and weight > 0:
        floor_weight = position_floor / equity
        if weight < floor_weight:
            # What raising it to the floor would actually risk: the position is
            # still bounded by its stop, so the loss that matters is the weight
            # multiplied by the stop distance -- not the weight itself. A name
            # with a 5% stop can be raised a long way for very little; one with
            # a 60% stop cannot be raised at all.
            stop_fraction = ((limits.atr_stop_multiple * atr) / price
                             if math.isfinite(atr) and atr > 0 and price > 0
                             else float("nan"))
            floor_risk = (floor_weight * stop_fraction
                          if math.isfinite(stop_fraction) else float("nan"))
            risk_ceiling = limits.risk_per_trade * FLOOR_RISK_MULTIPLE

            too_concentrated = floor_weight > limits.max_position_weight
            too_risky = math.isfinite(floor_risk) and floor_risk > risk_ceiling

            if too_concentrated or too_risky:
                why = ("would take "
                       f"{floor_weight:.0%} of the account, past the "
                       f"{limits.max_position_weight:.0%} per-symbol cap"
                       if too_concentrated else
                       f"would risk {floor_risk:.1%} of the account at its stop, "
                       f"past the {risk_ceiling:.1%} this account will spend on "
                       f"one trade")
                return SizingResult(
                    0.0, "account too small", float(vol_weight),
                    float(risk_weight if math.isfinite(risk_weight) else -1.0),
                    float(ann_vol),
                    f"this symbol sizes to ${weight * equity:,.2f}, below the "
                    f"${position_floor:,.0f} a venue will treat as a position, "
                    f"and raising it {why}. A ${equity:,.2f} account cannot "
                    f"carry this name at this volatility — that is arithmetic, "
                    f"not a refusal to trade.")

            spent = (f"{floor_risk:.2%}" if math.isfinite(floor_risk)
                     else "an unmeasured amount")
            reason = (
                f"raised from {weight:.1%} (${weight * equity:,.2f}) to the "
                f"${position_floor:,.0f} smallest position this venue can "
                f"actually work with — {floor_weight:.1%} of a ${equity:,.2f} "
                f"account, risking {spent} at its stop against a "
                f"{limits.risk_per_trade:.2%} budget. Below this a position "
                f"cannot be held overnight, cannot be taken in a "
                f"non-fractionable name, and cannot be trimmed.")
            weight = floor_weight
            binding = "venue minimum"

    return SizingResult(
        weight=float(weight), binding=binding,
        vol_target_weight=float(vol_weight),
        risk_cap_weight=float(risk_weight if math.isfinite(risk_weight) else -1.0),
        annualised_vol=float(ann_vol), reason=reason,
    )
