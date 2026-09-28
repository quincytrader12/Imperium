"""The account's risk dial: one number every strategy's size passes through.

Everything else in the terminal sizes a strategy from its own evidence and a
symbol from its own risk. This looks at the one thing none of them can see:
the account as a whole, as it has actually behaved. Two rules, multiplied:

**Volatility targeting (step 5).** When the account's own realised volatility
runs above a target, every position is scaled down in proportion; when it is
at or under the target, nothing changes. Volatility clusters -- a wild month
is usually followed by another -- and Moreira and Muir (JF 2017) showed that
scaling exposure by the inverse of recent variance raised Sharpe ratios
across equity, momentum and currency strategies, because risk rises in those
spells and return does not. Only ever down: the dial never levers an account
up because a quiet month made its volatility look small.

**Drawdown de-grossing (step 6).** Below a drawdown of :data:`DD_START` from
the high-water mark, nothing; from there to :data:`DD_FULL` the exposure is
cut in a straight line to :data:`FLOOR`, and it stays there until the account
recovers. The point is survival, not prediction: a strategy set that has
stopped working shows it first as a drawdown, and cutting size in one buys
the time to find out whether it has -- the rule that keeps a bad stretch from
compounding into a lost account. (Grossman and Zhou 1993 derived the policy
of holding exposure in proportion to the distance from a drawdown limit.)

**Never to zero.** The floor is a quarter, not nothing. A strategy cut to
nothing produces no evidence, and then nothing can ever say it has started
working again -- the same reason the evidence allocator has a floor.

The account's equity includes deposits and withdrawals. A withdrawal looks
like a drawdown here and will de-gross until the new level becomes the
high-water mark; ``IMPERIUM_HIGH_WATER_SINCE=YYYY-MM-DD`` is how the operator
says that it was theirs and not the market's -- marks before that day no
longer count toward the high-water mark.
"""

from __future__ import annotations

import datetime as dt
import math
import os
from dataclasses import dataclass

import numpy as np

#: Annualised volatility the account is held to.
TARGET_VOL = 0.15
#: Days of the account's returns the realised volatility is measured over.
VOL_DAYS = 20
#: Returns needed before the volatility rule acts at all.
MIN_RETURNS = 10
#: Drawdown from the high-water mark at which de-grossing begins...
DD_START = 0.05
#: ...and at which it reaches the floor.
DD_FULL = 0.20
#: The dial is never lower than this.
FLOOR = 0.25

#: Calendar-day annualisation. The account is marked every day it runs,
#: weekends included when crypto is held; a day with no move contributes
#: nothing to the variance, so the sum over a calendar year is right.
ANNUALISE = math.sqrt(365.0)


@dataclass(frozen=True)
class Dial:
    value: float = 1.0
    vol_scale: float = 1.0
    dd_scale: float = 1.0
    realised_vol: float | None = None
    target_vol: float = TARGET_VOL
    drawdown: float = 0.0
    high_water: float = 0.0
    reason: str = "no account history yet: full size"

    def as_dict(self) -> dict:
        return {"value": round(self.value, 4), "vol_scale": round(self.vol_scale, 4),
                "dd_scale": round(self.dd_scale, 4),
                "realised_vol": (None if self.realised_vol is None
                                 else round(self.realised_vol, 4)),
                "target_vol": self.target_vol, "drawdown": round(self.drawdown, 4),
                "high_water": round(self.high_water, 2), "reason": self.reason,
                "dd_start": DD_START, "dd_full": DD_FULL, "floor": FLOOR}


def realised_vol(daily: list[tuple[str, float]], days: int = VOL_DAYS) -> float | None:
    """Annualised volatility of the account's daily returns over the last
    ``days`` points. A gap of several days between points (the terminal was
    off) is one return over that span, scaled to a day's worth so a weekend
    away does not read as a volatile day. None with too few returns."""
    points = [(d, e) for d, e in daily if e > 0 and math.isfinite(e)]
    points = points[-(days + 1):]
    returns = []
    for (d0, e0), (d1, e1) in zip(points, points[1:]):
        try:
            gap = (dt.date.fromisoformat(d1) - dt.date.fromisoformat(d0)).days
        except ValueError:
            gap = 1
        returns.append(math.log(e1 / e0) / math.sqrt(max(1, gap)))
    if len(returns) < MIN_RETURNS:
        return None
    return float(np.std(returns, ddof=1)) * ANNUALISE


def drawdown_scale(drawdown: float, start: float = DD_START,
                   full: float = DD_FULL, floor: float = FLOOR) -> float:
    if full <= start:
        return 1.0 if drawdown < start else floor
    depth = min(1.0, max(0.0, (drawdown - start) / (full - start)))
    return 1.0 - (1.0 - floor) * depth


def compute(daily: list[tuple[str, float]], equity: float, *,
            target_vol: float = TARGET_VOL, since: str = "",
            start: float = DD_START, full: float = DD_FULL) -> Dial:
    """The dial for an account whose daily marks are ``daily`` and whose
    equity is ``equity`` now. Marks before the day ``since`` do not count
    toward the high-water mark."""
    marks = [e for d, e in daily
             if e > 0 and math.isfinite(e) and (not since or d >= since)]
    if not marks and not (equity > 0 and math.isfinite(equity)):
        return Dial()
    now = equity if equity > 0 and math.isfinite(equity) else marks[-1]
    high = max(marks + [now])
    drawdown = max(0.0, 1.0 - now / high) if high > 0 else 0.0
    dd = drawdown_scale(drawdown, start, full)

    vol = realised_vol(daily)
    if target_vol > 0 and vol is not None and vol > target_vol:
        vs = max(FLOOR, target_vol / vol)
    else:
        vs = 1.0
    value = max(FLOOR, vs * dd)

    parts = []
    if vol is None:
        parts.append(f"volatility not yet measurable ({MIN_RETURNS} days needed)")
    elif vs < 1.0:
        parts.append(f"realised volatility {vol:.0%} over the target "
                     f"{target_vol:.0%}: ×{vs:.2f}")
    else:
        parts.append(f"realised volatility {vol:.0%}, within the "
                     f"{target_vol:.0%} target")
    if dd < 1.0:
        parts.append(f"{drawdown:.1%} below the high-water mark "
                     f"${high:,.2f}: ×{dd:.2f}")
    else:
        parts.append(f"drawdown {drawdown:.1%}, under the {start:.0%} line")
    head = ("full size" if value >= 1.0 else f"every strategy at {value:.0%} of size")
    return Dial(value=value, vol_scale=vs, dd_scale=dd, realised_vol=vol,
                target_vol=target_vol, drawdown=drawdown, high_water=high,
                reason=f"{head}: " + "; ".join(parts))


def _number(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return min(high, max(low, value)) if math.isfinite(value) else default


def settings() -> dict:
    enabled = os.environ.get("IMPERIUM_RISK_DIAL", "true").strip().lower() not in (
        "0", "false", "no", "off")
    start = _number("IMPERIUM_DRAWDOWN_START", DD_START, 0.0, 0.5)
    full = _number("IMPERIUM_DRAWDOWN_FULL", DD_FULL, 0.01, 0.9)
    return {"enabled": enabled,
            "target_vol": _number("IMPERIUM_ACCOUNT_TARGET_VOL", TARGET_VOL, 0.0, 2.0),
            "start": start, "full": max(full, start + 0.01),
            "since": os.environ.get("IMPERIUM_HIGH_WATER_SINCE", "").strip()}
