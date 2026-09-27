"""What crossing actually costs, learned from real fills.

The cost gate decides whether anything trades, and it prices the crossing
from a model: half the quoted spread each way, or a default spread when no
quote has arrived. A model is a claim. The fills are what happened. This is
the loop between them: every real fill says how much the crossing cost
against the price the decision was taken at, and the gate is corrected by
what the model got wrong.

WHAT IS MEASURED
----------------

The **excess** of the measured one-way crossing over what the model itself
priced one crossing at for that trade -- before any correction. Not the whole cost: the fee component is not in a fill's
price at all -- US equities are commission-free here, and crypto fees are
taken in the coin, not the price -- so a fill can only ever speak to the part
of the model that prices the spread, impact and latency. The fee tier keeps
its own "assumed" warning until it is confirmed, and this does not pretend to
settle it.

WHAT IS NOT COUNTED
-------------------

* **Simulated fills.** The local paper book charges each fill the modelled
  cost by construction; feeding that back would be the model grading itself.
  The venue's paper account is different: its fills come from the venue's own
  matching against real quotes, so they count.
* **Auction orders.** A market-on-close or market-on-open fill is priced by
  the auction, and its distance from the last trade is the overnight move, not
  a spread. Counted, it would charge the intraday gate for the overnight
  strategy's drift.
* **Fills with no reference price**, which have no crossing to measure.

HOW IT IS APPLIED
-----------------

Per asset class, because an equity and a coin cross very different books. The
correction is the *median* excess -- one fill during a fast market is not the
cost of crossing -- shrunk toward zero by credibility, ``n / (n + K)``: five
fills move the model a fifth of the way to what they say, eighty move it most
of the way. Nothing moves on fewer than :data:`MIN_FILLS`. Bounded both ways,
so a run of bad data cannot close the gate or throw it open; and it can make
crossing cheaper than the model as well as dearer, because a model that
overstates cost refuses trades that have real edge -- but never cheaper than
free.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

#: Fills in an asset class before its crossing is corrected at all.
MIN_FILLS = 5

#: The K in n / (n + K). The number of fills at which the correction is half
#: of what the median says.
CREDIBILITY = 20

#: The largest one-way correction, in basis points, either way. Wider than any
#: honest spread error on a liquid name; narrow enough that a bad feed cannot
#: turn the gate into a wall or a doorway.
MAX_CORRECTION_BPS = 50.0

#: Samples kept per asset class. Recent enough to follow a market whose
#: liquidity changes, long enough that the median is stable.
MAX_SAMPLES = 200

#: A single fill whose excess is beyond this is a data fault -- a stale
#: reference, a split mid-order -- not a crossing, and is not learned from.
IMPLAUSIBLE_BPS = 500.0


@dataclass
class ClassRecord:
    """Everything measured about crossing one asset class's books."""

    samples: list[float] = field(default_factory=list)
    #: Every fill ever learned from, not only those still in the window.
    total: int = 0
    rejected: int = 0

    def correction(self) -> float:
        n = len(self.samples)
        if n < MIN_FILLS:
            return 0.0
        median = statistics.median(self.samples)
        # Credibility from every fill ever measured, the median from the recent
        # window. They answer different questions: how much evidence is there,
        # and what is crossing costing *now*. Weighted by the window alone, the
        # correction was capped at 200/220 of what the fills said and never
        # got closer, however long the terminal ran.
        seen = max(n, self.total)
        weight = seen / (seen + CREDIBILITY)
        return max(-MAX_CORRECTION_BPS,
                   min(MAX_CORRECTION_BPS, median * weight))


class CostCalibration:
    """The measured correction to the gate's crossing cost, per asset class."""

    def __init__(self) -> None:
        self.classes: dict[str, ClassRecord] = {}

    def observe(self, asset_class: str, measured_bps: float,
                model_one_way_bps: float) -> bool:
        """Learn from one real fill. Returns whether it was used.

        ``model_one_way_bps`` must be the model's own figure, uncorrected --
        CostEstimate.model_one_way_bps. Against the corrected figure each fill
        would show only what the correction had not yet absorbed, and it would
        stall at about half the true cost.
        """
        rec = self.classes.setdefault(asset_class, ClassRecord())
        excess = float(measured_bps) - float(model_one_way_bps)
        if not math.isfinite(excess) or abs(excess) > IMPLAUSIBLE_BPS:
            rec.rejected += 1
            return False
        rec.samples.append(excess)
        if len(rec.samples) > MAX_SAMPLES:
            del rec.samples[:len(rec.samples) - MAX_SAMPLES]
        rec.total += 1
        return True

    def excess_bps(self, asset_class: str) -> float:
        """The one-way correction to add to the modelled half-spread."""
        rec = self.classes.get(asset_class)
        return rec.correction() if rec else 0.0

    def describe(self, asset_class: str) -> str:
        rec = self.classes.get(asset_class)
        n = len(rec.samples) if rec else 0
        if n < MIN_FILLS:
            return (f"crossing still modelled: {n} of {MIN_FILLS} real fills "
                    f"measured")
        correction = rec.correction()
        median = statistics.median(rec.samples)
        direction = "dearer" if correction > 0 else "cheaper"
        return (f"crossing measured {abs(median):.1f}bp {direction} than the "
                f"model over {n} real fills; the gate is corrected by "
                f"{correction:+.1f}bp each way")

    def rows(self) -> list[dict[str, Any]]:
        out = []
        for name, rec in sorted(self.classes.items()):
            n = len(rec.samples)
            out.append({
                "asset_class": name,
                "fills": n,
                "total": rec.total,
                "rejected": rec.rejected,
                "median_excess_bps": (round(statistics.median(rec.samples), 2)
                                      if n else None),
                "correction_bps": round(rec.correction(), 2),
                "note": self.describe(name),
            })
        return out

    # -- persistence -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {name: {"samples": list(rec.samples), "total": rec.total,
                       "rejected": rec.rejected}
                for name, rec in self.classes.items()}

    @classmethod
    def from_dict(cls, payload: Any) -> "CostCalibration":
        out = cls()
        if not isinstance(payload, dict):
            return out
        for name, raw in payload.items():
            if not isinstance(raw, dict):
                continue
            samples = []
            for value in raw.get("samples") or []:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and abs(value) <= IMPLAUSIBLE_BPS:
                    samples.append(value)
            try:
                total = int(raw.get("total") or len(samples))
                rejected = int(raw.get("rejected") or 0)
            except (TypeError, ValueError):
                total, rejected = len(samples), 0
            out.classes[str(name)] = ClassRecord(
                samples=samples[-MAX_SAMPLES:], total=total, rejected=rejected)
        return out
