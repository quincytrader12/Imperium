"""Capital by what a strategy adds to the rest of the book, not only by what
it earns alone.

The evidence allocator asks of each strategy: does its own record say it
earns? This asks the other question a fund asks: when it earns, does it earn
on the days the others do? Two strategies with the same record are not worth
the same to an account if one makes its money on exactly the days the rest
of the book does -- that one adds size to the book's good days and bad days
alike -- and the other makes it when the rest is losing.

THE MEASUREMENT
---------------

For each strategy, the correlation of its daily contribution to the fund
with the *rest of the book's* combined contribution on the same days -- not
with each other strategy one pair at a time. What matters to the account is
the strategy against everything else it holds, and one number per strategy
keeps the reason sayable in a sentence.

The correlation is **shrunk toward zero** by n / (n + :data:`SHRINK_DAYS`):
thirty days of a correlation have a standard error near 0.18, and a tilt
built on the raw figure would chase noise. Below :data:`MIN_DAYS` shared
days nothing is measured at all, and the multiplier is exactly 1.

THE RULE
--------

multiplier = 1 - :data:`TILT` x shrunk correlation, bounded to
[:data:`LOW`, :data:`HIGH`]. A strategy moving with the rest of the book
(correlation +0.5) trades at three quarters; one moving against it (-0.5) at
a quarter more; one unrelated to it, unchanged. The move toward the target is
at most :data:`STEP` a day, as the evidence allocator's is.

It multiplies the evidence multiplier and the product is held inside the
evidence allocator's own bounds, so this can tilt capital between strategies
but can never take one past the limits every size is already held to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from imperium.execution import evidence as evidence_mod

MIN_DAYS = 30
SHRINK_DAYS = 30
TILT = 0.5
LOW = 0.75
HIGH = 1.25
STEP = 0.10

#: Not strategies, or not sized by this: the same list the evidence
#: allocator excludes.
EXCLUDED = frozenset({"unattributed", "sector"})


def returns_by_day(marks: list[tuple[str, float]],
                   equity: dict[str, float]) -> dict[str, float]:
    """A strategy's daily contribution to the fund, keyed by the day it
    ended: its change in profit over the fund's equity at the start. The same
    return the evidence allocator reads, keyed so strategies can be aligned."""
    out: dict[str, float] = {}
    for (prev_day, prev), (day, value) in zip(marks, marks[1:]):
        base = equity.get(prev_day, 0.0)
        if base > 0 and math.isfinite(value) and math.isfinite(prev):
            out[day] = (value - prev) / base
    return out


def correlation_to_rest(series: dict[str, dict[str, float]]
                        ) -> dict[str, tuple[float, int]]:
    """Each strategy's correlation with the sum of all the others, over the
    days it has a return; a strategy with no mark on a day contributed
    nothing that day. (correlation, days) for those with :data:`MIN_DAYS`
    and movement on both sides."""
    out: dict[str, tuple[float, int]] = {}
    for name, own in series.items():
        days = sorted(own)
        if len(days) < MIN_DAYS:
            continue
        mine = np.array([own[d] for d in days])
        rest = np.array([sum(other.get(d, 0.0) for n, other in series.items()
                             if n != name) for d in days])
        if np.std(mine) <= 0 or np.std(rest) <= 0:
            continue
        rho = float(np.corrcoef(mine, rest)[0, 1])
        if math.isfinite(rho):
            out[name] = (rho, len(days))
    return out


def target(rho: float, days: int) -> tuple[float, float]:
    """The (shrunk correlation, target multiplier) for a measured one."""
    shrunk = rho * days / (days + SHRINK_DAYS)
    return shrunk, min(HIGH, max(LOW, 1.0 - TILT * shrunk))


@dataclass
class Standing:
    correlation: float = 0.0
    shrunk: float = 0.0
    days: int = 0
    target: float = 1.0
    multiplier: float = 1.0
    reason: str = "not yet measured against the rest of the book"

    def as_dict(self) -> dict[str, Any]:
        return {"correlation": round(self.correlation, 4),
                "shrunk": round(self.shrunk, 4), "days": self.days,
                "target": round(self.target, 4),
                "multiplier": round(self.multiplier, 4), "reason": self.reason}


def combined_multiplier(evidence: Any, spread: Any, strategy: str) -> float:
    """The evidence multiplier times this one, held inside the evidence
    allocator's own bounds: a tilt between strategies, never a way past the
    limits every size is held to."""
    e = evidence.multiplier(strategy) if evidence is not None else 1.0
    s = spread.multiplier(strategy) if spread is not None else 1.0
    return min(evidence_mod.CAP, max(evidence_mod.FLOOR, e * s))


def _bounded(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(HIGH, max(LOW, value)) if math.isfinite(value) else 1.0


@dataclass
class Diversification:
    standings: dict[str, Standing] = field(default_factory=dict)
    revised_on: str = ""

    def multiplier(self, strategy: str) -> float:
        if not strategy or strategy in EXCLUDED:
            return 1.0
        found = self.standings.get(strategy)
        return _bounded(found.multiplier) if found is not None else 1.0

    def note(self, strategy: str) -> str:
        found = self.standings.get(strategy)
        return found.reason if found is not None else ""

    def revise(self, day: str, records: Iterable[Any],
               equity: dict[str, float]) -> list[tuple[str, float, float, str]]:
        """Re-measure every strategy and step each multiplier toward its
        target. Once a day. Returns (strategy, before, after, reason) for
        each that moved."""
        if day and day == self.revised_on:
            return []
        self.revised_on = day
        series = {r.name: returns_by_day(list(r.daily), equity)
                  for r in records if r.name not in EXCLUDED}
        measured = correlation_to_rest(series)
        moved = []
        for name in series:
            standing = self.standings.setdefault(name, Standing())
            before = standing.multiplier
            if name not in measured:
                have = len(series[name])
                standing.target = 1.0
                standing.reason = (f"{have} of {MIN_DAYS} days to measure it "
                                   f"against the rest of the book")
            else:
                rho, days = measured[name]
                shrunk, goal = target(rho, days)
                standing.correlation, standing.shrunk = rho, shrunk
                standing.days, standing.target = days, goal
                if goal > 1.0:
                    how = "moves against the rest of the book"
                elif goal < 1.0:
                    how = "moves with the rest of the book"
                else:
                    how = "is unrelated to the rest of the book"
                standing.reason = (f"{how}: correlation {rho:+.2f} over {days} "
                                   f"days ({shrunk:+.2f} shrunk), ×{goal:.2f}")
            step = max(-STEP, min(STEP, standing.target - before))
            standing.multiplier = _bounded(before + step)
            if abs(standing.multiplier - before) > 1e-9:
                moved.append((name, before, standing.multiplier, standing.reason))
        return moved

    def as_dict(self) -> dict[str, Any]:
        return {"revised_on": self.revised_on,
                "standings": {k: v.as_dict() for k, v in self.standings.items()}}

    @classmethod
    def from_dict(cls, payload: Any) -> "Diversification":
        out = cls()
        if not isinstance(payload, dict):
            return out
        out.revised_on = str(payload.get("revised_on") or "")
        for name, raw in (payload.get("standings") or {}).items():
            if not isinstance(raw, dict):
                continue
            try:
                out.standings[str(name)] = Standing(
                    correlation=float(raw.get("correlation", 0.0)),
                    shrunk=float(raw.get("shrunk", 0.0)),
                    days=int(raw.get("days", 0)),
                    target=_bounded(raw.get("target", 1.0)),
                    multiplier=_bounded(raw.get("multiplier", 1.0)),
                    reason=str(raw.get("reason") or ""))
            except (TypeError, ValueError):
                continue
        return out
