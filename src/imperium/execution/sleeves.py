"""Sleeves: strategies that trade their own list of symbols, once a day.

The engine trades whatever the scanner finds, symbol by symbol, all day. A
sleeve is the other shape: a fixed universe, one decision a day near the
close, and a slice of the account of its own. The diversifiers are sleeves --
cross-asset trend, short-term mean reversion, the turn of the month -- because
what makes them diversifying is exactly what does not fit the engine: other
asset classes, other holding periods, other reasons to be in the market.

THE CONTRACT
------------

* **A pure decision.** Each sleeve is a function from daily closes (and its
  own small memory) to target weights, as fractions of the sleeve. No clock,
  no venue, no account. That is what makes each one testable against a
  history, and what lets the same function be backtested and run live.
* **Real orders, through the one order path.** Targets become orders through
  the broker every other strategy uses, named for the sleeve. So the mode
  switch, the GO LIVE gate, resting entries at the mid, the trade journal,
  per-strategy attribution and the evidence allocator all apply to a sleeve
  exactly as they do to the engine, with nothing re-implemented.
* **Its symbols are its own.** While a sleeve is on, no other part of the
  terminal may trade its universe: the engine does not enter it, the
  give-back ratchet does not exit it, the overnight flatten does not close it.
  One owner per symbol is what lets a sleeve size against the broker's
  position as its own; without it, two strategies' targets for the same
  symbol would net into one number and each would undo the other.
* **The venue's floor, before the order.** Alpaca refuses a fractional buy
  under a dollar. On a small account that is the binding constraint, so a buy
  below it is not sent -- it is reported, with the sleeve size at which it
  would be.
* **A share of the account, claimed.** A sleeve that is on claims its
  allocation from the capital plan, and the engine sizes against what is
  left. Its multiplier from the evidence allocator scales it like any other
  strategy.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

log = logging.getLogger("imperium.sleeves")

#: Alpaca's minimum notional for a fractional order.
MIN_ORDER = 1.0

#: Drift from target, as a fraction of it, before a held position is resized.
#: A daily decision that trades every small drift pays a spread a day for
#: nothing; a quarter is the same band the Sector Trend sleeve uses.
REBALANCE_BAND = 0.25

STATE_KEY = "sleeves"


@dataclass
class Targets:
    """What a sleeve's decision returns: weights as fractions of the sleeve."""

    weights: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class SleeveOrder:
    symbol: str
    side: str
    #: The position's target as a fraction of the whole account -- what the
    #: broker's apply_target takes.
    account_weight: float
    notional: float
    reason: str


@dataclass
class Sleeve:
    """One sleeve: its universe, its share, its decision, its memory."""

    name: str
    label: str
    universe: tuple[str, ...]
    allocation: float
    decide: Callable[[dict[str, np.ndarray], dict, str], Targets]
    enabled: bool = True
    #: Eastern time after which the day's decision is taken, on a trading
    #: day, while the market is open.
    run_after_et: str = "15:45"
    #: Calendar days of daily history the decision needs.
    history_days: int = 420
    summary: str = ""
    # -- persisted -------------------------------------------------------
    last_run_day: str = ""
    runs: int = 0
    memory: dict[str, Any] = field(default_factory=dict)
    last_targets: dict[str, float] = field(default_factory=dict)
    last_reasons: dict[str, str] = field(default_factory=dict)
    last_note: str = ""
    # -- not persisted ---------------------------------------------------
    last_error: str = ""
    last_too_small: dict[str, float] = field(default_factory=dict)
    pending: list[SleeveOrder] | None = None
    pending_day: str = ""
    #: The prices the pending orders were planned at, the fallback when no
    #: fresh quote is on the feed when they are sent.
    pending_prices: dict[str, float] = field(default_factory=dict)
    orders_sent: int = 0

    # -- when --------------------------------------------------------------

    def due(self, now_et: dt.datetime, market_open: bool) -> bool:
        """Once a trading day, after its run time, while the market is open."""
        if not self.enabled or not market_open:
            return False
        day = now_et.date().isoformat()
        if day == self.last_run_day or day == self.pending_day:
            return False
        hh, _, mm = self.run_after_et.partition(":")
        try:
            after = dt.time(int(hh), int(mm or 0))
        except ValueError:
            after = dt.time(15, 45)
        return now_et.time() >= after

    # -- what --------------------------------------------------------------

    def plan(self, closes: dict[str, np.ndarray], day: str, *,
             prices: dict[str, float], held: dict[str, float],
             account_equity: float, multiplier: float = 1.0
             ) -> list[SleeveOrder]:
        """Decide, and turn the decision into orders against what is held."""
        targets = self.decide(closes, self.memory, day)
        self.last_targets = dict(targets.weights)
        self.last_reasons = dict(targets.reasons)
        self.last_note = targets.note
        self.last_too_small = {}
        sleeve = max(0.0, self.allocation * account_equity * multiplier)
        orders: list[SleeveOrder] = []
        if account_equity <= 0:
            return orders
        for symbol in self.universe:
            price = float(prices.get(symbol) or 0.0)
            if price <= 0:
                continue
            quantity = float(held.get(symbol) or 0.0)
            weight = float(targets.weights.get(symbol, 0.0))
            target_value = weight * sleeve
            current = quantity * price
            reason = targets.reasons.get(symbol, "")
            if weight <= 0:
                if quantity > 0:
                    orders.append(SleeveOrder(symbol, "sell", 0.0, current,
                                              f"exit: {reason}" if reason else "exit"))
                continue
            delta = target_value - current
            if quantity <= 0:
                if target_value < MIN_ORDER:
                    self.last_too_small[symbol] = target_value
                    continue
                orders.append(SleeveOrder(symbol, "buy",
                                          target_value / account_equity,
                                          target_value, f"entry: {reason}"))
                continue
            if abs(delta) < max(MIN_ORDER, REBALANCE_BAND * target_value):
                continue
            orders.append(SleeveOrder(symbol, "buy" if delta > 0 else "sell",
                                      target_value / account_equity,
                                      abs(delta), f"rebalance: {reason}"))
        # Sells first: they fund the buys, and an exit never waits on one.
        orders.sort(key=lambda o: o.side != "sell")
        return orders

    def smallest_viable_equity(self) -> float:
        """The account balance at which every current target clears the floor."""
        live = [w for w in self.last_targets.values() if w > 0]
        if not live or self.allocation <= 0:
            return 0.0
        return MIN_ORDER / (min(live) * self.allocation)

    # -- what the terminal shows -------------------------------------------

    def panel(self, account_equity: float, held: dict[str, float],
              prices: dict[str, float]) -> dict[str, Any]:
        sleeve = self.allocation * max(0.0, account_equity)
        holdings = []
        for symbol in self.universe:
            quantity = float(held.get(symbol) or 0.0)
            if quantity <= 0:
                continue
            value = quantity * float(prices.get(symbol) or 0.0)
            holdings.append({"symbol": symbol, "value": round(value, 2),
                             "target": round(self.last_targets.get(symbol, 0.0), 4)})
        return {
            "name": self.name, "label": self.label, "enabled": self.enabled,
            "summary": self.summary, "allocation": self.allocation,
            "sleeve_equity": round(sleeve, 2), "universe": list(self.universe),
            "run_after_et": self.run_after_et, "last_run_day": self.last_run_day,
            "runs": self.runs, "targets": {k: round(v, 4) for k, v in self.last_targets.items()},
            "reasons": self.last_reasons, "note": self.last_note,
            "holdings": holdings, "too_small": {k: round(v, 2) for k, v in self.last_too_small.items()},
            "viable_from": round(self.smallest_viable_equity(), 0),
            "last_error": self.last_error, "orders_sent": self.orders_sent,
            "pending": len(self.pending or []),
        }

    # -- persistence -------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {"last_run_day": self.last_run_day, "runs": self.runs,
                "memory": self.memory, "last_targets": self.last_targets,
                "last_reasons": self.last_reasons, "last_note": self.last_note}

    def restore(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        self.last_run_day = str(payload.get("last_run_day") or "")
        try:
            self.runs = int(payload.get("runs") or 0)
        except (TypeError, ValueError):
            self.runs = 0
        memory = payload.get("memory")
        self.memory = dict(memory) if isinstance(memory, dict) else {}
        targets = payload.get("last_targets")
        if isinstance(targets, dict):
            self.last_targets = {}
            for k, v in targets.items():
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v) and v >= 0:
                    self.last_targets[str(k)] = v
        reasons = payload.get("last_reasons")
        self.last_reasons = ({str(k): str(v) for k, v in reasons.items()}
                             if isinstance(reasons, dict) else {})
        self.last_note = str(payload.get("last_note") or "")


# -- configuration --------------------------------------------------------------


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _share(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return min(0.6, max(0.0, value)) if math.isfinite(value) else default


def closes_from_bars(rows: dict[str, list[dict]] | None) -> dict[str, np.ndarray]:
    """Daily closes per symbol from the venue's bar rows, oldest first."""
    out: dict[str, np.ndarray] = {}
    for symbol, bars in (rows or {}).items():
        values = [float(b["c"]) for b in bars or []
                  if isinstance(b, dict) and isinstance(b.get("c"), (int, float))
                  and b["c"] > 0]
        if values:
            out[symbol] = np.asarray(values, dtype=float)
    return out


def build_all() -> list[Sleeve]:
    """Every sleeve this build knows, configured from the environment."""
    from imperium.strategy import global_trend

    def decide_global(closes, memory, day):
        t = global_trend.targets(closes)
        return Targets(weights=t.weights, reasons=t.reasons, note=t.note or (
            f"holding {len(t.weights)} of {len(global_trend.UNIVERSE)} asset "
            f"classes at {t.volatility:.0%} volatility"))

    return [
        Sleeve(
            name="global_trend", label="Global trend",
            universe=global_trend.UNIVERSE,
            allocation=_share("GLOBAL_TREND_ALLOCATION", 0.30),
            enabled=_flag("GLOBAL_TREND_ENABLED", True),
            run_after_et=os.environ.get("GLOBAL_TREND_RUN_TIME_ET", "15:45"),
            decide=decide_global,
            summary=("Bonds, gold, commodities, international and real estate, "
                     "held while trending up, sized to 10% volatility."),
        ),
    ]
