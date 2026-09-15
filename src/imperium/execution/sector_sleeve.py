"""The Sector Trend sleeve's daily run.

One function decides -- :func:`plan` -- and it is pure. It takes today's
prices, the sleeve's own ledger and the account equity, and returns a list of
intentions with a reason attached to each. A separate function sends them.

That split is deliberate and it is what makes this testable. A daily job that
interleaves "work out what to do" with "tell the venue" can only be tested by
mocking a venue, which tests the mock; and the interesting failures here are
all in the deciding -- an entry that should not have fired, a stop that moved
the wrong way, an order the venue will reject.

**What this sleeve may never do.** Read the account position for sizing or for
exits (see :mod:`imperium.execution.sleeve_ledger`). Trade twice in a day.
Place an order without the ``sectrend-`` prefix. Move a stop down. Trade at all
while ``SECTOR_TREND_ENABLED`` is false.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from imperium.execution.sleeve_ledger import SleeveLedger
from imperium.strategy import sector
from imperium.strategy.sector_config import (
    MIN_FRACTIONAL_NOTIONAL, ORDER_PREFIX, SectorTrendConfig,
)

log = logging.getLogger("imperium.sector")


@dataclass(frozen=True)
class Intention:
    """One thing the sleeve means to do today, and why."""

    symbol: str
    side: str                    # "buy" or "sell"
    #: Dollars, not shares. Alpaca's fractional orders are notional, and a
    #: share count computed here would be rounded again at the venue.
    notional: float
    reason: str                  # entry | exit_stop | rebalance
    detail: str = ""
    target_weight: float = 0.0

    @property
    def client_order_id_prefix(self) -> str:
        return ORDER_PREFIX


@dataclass
class Plan:
    """Everything today's run concluded, whether or not it trades."""

    intentions: list[Intention] = field(default_factory=list)
    stops: dict[str, float] = field(default_factory=dict)
    exits: list[str] = field(default_factory=list)
    entries: list[str] = field(default_factory=list)
    #: Symbols whose target was below the venue's minimum order size. Named
    #: rather than dropped quietly: on a small account this is most of them,
    #: and an operator watching a run place nothing deserves to know why.
    too_small: dict[str, float] = field(default_factory=dict)
    skipped_no_history: list[str] = field(default_factory=list)
    gross_weight: float = 0.0
    capped: bool = False
    buys_scaled: float = 1.0
    sleeve_equity: float = 0.0
    note: str = ""

    @property
    def trades(self) -> int:
        return len(self.intentions)


def plan(closes: dict[str, list[float]], ledger: SleeveLedger,
         account_equity: float, config: SectorTrendConfig) -> Plan:
    """Decide today's orders. Pure: no clock, no venue, no writes.

    ``closes`` is each symbol's adjusted daily closes, oldest first, with
    today's close (or the provisional one, in ``near_close`` mode) last.
    """
    import numpy as np

    out = Plan()
    out.sleeve_equity = config.sleeve_equity(account_equity)
    universe = [s for s in config.universe]

    series: dict[str, Any] = {}
    for symbol in universe:
        values = np.asarray(closes.get(symbol) or [], dtype=float)
        if values.size < sector.MIN_BARS:
            out.skipped_no_history.append(symbol)
            continue
        series[symbol] = values

    if not series:
        out.note = ("no symbol has the "
                    f"{sector.MIN_BARS} daily bars this strategy needs")
        return out

    band_of = {s: sector.bands(v) for s, v in series.items()}

    # 1. Exits first, on the stop carried in. Their cash funds the entries, and
    #    a stop must never wait behind a purchase.
    for symbol in ledger.longs():
        values = series.get(symbol)
        if values is None:
            continue
        t = len(values) - 1
        held = ledger.held(symbol)
        step = sector.step_position(float(values[t]), held.stop,
                                    float(band_of[symbol].lower[t]))
        if step.exited:
            out.exits.append(symbol)
            price = float(values[t])
            out.intentions.append(Intention(
                symbol=symbol, side="sell",
                notional=abs(held.quantity) * price, reason="exit_stop",
                detail=(f"closed at {price:,.2f}, below the "
                        f"{held.stop:,.2f} stop carried in from the previous "
                        f"session")))
        else:
            out.stops[symbol] = step.stop

    # 2. Entries: flat symbols clearing yesterday's upper band.
    for symbol, values in series.items():
        if symbol in ledger.longs() or symbol in out.exits:
            continue
        t = len(values) - 1
        if sector.entry_signal(values, band_of[symbol], t):
            out.entries.append(symbol)

    # 3. Size the whole long book.
    longs = sorted(
        (set(ledger.longs()) - set(out.exits)) | set(out.entries))
    sigmas = {s: sector.daily_sigma(series[s]) for s in longs if s in series}
    weights = sector.target_weights(
        sigmas, longs, universe_size=len(universe),
        target_vol=config.target_vol, max_leverage=config.max_leverage)
    out.gross_weight = weights.gross
    out.capped = weights.capped

    # 4. Turn weights into dollar deltas, respecting the rebalance threshold
    #    for symbols already held.
    buys: dict[str, float] = {}
    for symbol in longs:
        values = series.get(symbol)
        if values is None:
            continue
        price = float(values[-1])
        if price <= 0:
            continue
        weight = weights.weights.get(symbol, 0.0)
        target_notional = weight * out.sleeve_equity
        held_qty = ledger.held(symbol).quantity
        current_notional = held_qty * price
        is_entry = symbol in out.entries

        if not is_entry and not sector.needs_rebalance(
                held_qty, target_notional / price,
                config.rebalance_threshold):
            continue

        delta = target_notional - current_notional
        if abs(delta) < 1e-9:
            continue

        if delta > 0:
            # The venue's floor, checked before the order exists rather than
            # after it is refused. On a small account this is the difference
            # between a run that trades and a run that logs rejections.
            if delta < MIN_FRACTIONAL_NOTIONAL:
                out.too_small[symbol] = delta
                if is_entry:
                    out.entries.remove(symbol)
                continue
            buys[symbol] = delta
        else:
            out.intentions.append(Intention(
                symbol=symbol, side="sell", notional=abs(delta),
                reason="rebalance", target_weight=weight,
                detail=(f"trimming to {weight:.2%} of the sleeve")))

    # 5. Fit the buys to what the sleeve can actually spend.
    budget = out.sleeve_equity * config.max_leverage - sum(
        ledger.held(s).quantity * float(series[s][-1])
        for s in ledger.longs() if s in series and s not in out.exits)
    fitted, factor = sector.scale_buys_to_budget(buys, budget)
    out.buys_scaled = factor
    for symbol, notional in fitted.items():
        if notional < MIN_FRACTIONAL_NOTIONAL:
            # Scaling can push a buy under the floor that was over it before.
            out.too_small[symbol] = notional
            if symbol in out.entries:
                out.entries.remove(symbol)
            continue
        weight = weights.weights.get(symbol, 0.0)
        is_entry = symbol in out.entries
        values = series[symbol]
        t = len(values) - 1
        if is_entry:
            out.stops[symbol] = float(band_of[symbol].lower[t])
        out.intentions.append(Intention(
            symbol=symbol, side="buy", notional=notional,
            reason="entry" if is_entry else "rebalance",
            target_weight=weight,
            detail=(f"broke {float(band_of[symbol].upper[t - 1]):,.2f} on the "
                    f"previous session's upper band; stop at "
                    f"{out.stops.get(symbol, float('nan')):,.2f}"
                    if is_entry else
                    f"adding to reach {weight:.2%} of the sleeve")))

    if out.too_small:
        out.note = _too_small_note(out, weights)
    return out


def sleeve_equity_needed(weights: dict[str, float]) -> float:
    """The sleeve equity at which the *smallest* target clears the venue floor.

    The smallest weight is the binding one: a sleeve big enough for it is big
    enough for all of them. Separated out because the arithmetic is the whole
    answer to "can this account run this strategy", and it belongs somewhere a
    reader can check it rather than inside a format string.
    """
    live = [w for w in weights.values() if w > 0]
    if not live:
        return float("inf")
    return MIN_FRACTIONAL_NOTIONAL / min(live)


def _too_small_note(out: "Plan", weights: sector.SleeveWeights) -> str:
    count = len(out.too_small)
    smallest = min(out.too_small.values())
    needed = sleeve_equity_needed(weights.weights)
    sentence = (
        f"{count} symbol{'s' if count != 1 else ''} sized below Alpaca's "
        f"${MIN_FRACTIONAL_NOTIONAL:.0f} minimum order and {'were' if count != 1 else 'was'} "
        f"skipped (smallest ${smallest:.2f}). This sleeve holds "
        f"${out.sleeve_equity:,.2f}.")
    if math.isfinite(needed):
        sentence += (f" Every symbol in it clears that floor from about "
                     f"${needed:,.0f} of sleeve equity.")
    return sentence


def summarise(result: Plan, ledger: SleeveLedger) -> str:
    """The daily line for the notifier. Short enough to be read on a phone."""
    if result.note and not result.intentions:
        return f"Sector Trend: no orders — {result.note}"
    parts = [f"Sector Trend: {result.trades} order"
             f"{'s' if result.trades != 1 else ''}"]
    if result.exits:
        parts.append(f"out of {', '.join(result.exits)}")
    if result.entries:
        parts.append(f"into {', '.join(result.entries)}")
    parts.append(f"holding {len(ledger.longs())} at "
                 f"{result.gross_weight:.0%} of the sleeve")
    if result.capped:
        parts.append("(leverage cap binding)")
    if result.buys_scaled < 1.0:
        parts.append(f"(buys scaled to {result.buys_scaled:.0%} for cash)")
    return "; ".join(parts)


# -- the daily run --------------------------------------------------------

@dataclass
class SectorRunner:
    """Holds the sleeve's config and ledger, and runs it once a day.

    **Why this places orders through the client rather than through
    LiveBroker.** The brief asks that the existing order path be reused, and
    the client, its authentication, its retry rules and its
    never-retry-a-rejected-order discipline all are. What is deliberately not
    reused is the broker's *book*: ``LiveBroker`` keeps one position per
    symbol, and routing this sleeve through it would merge the sleeve's
    quantities into the same number every other strategy uses -- which is
    precisely the netting problem the ledger exists to solve. The order path is
    shared; the book is not.
    """

    config: SectorTrendConfig
    ledger: SleeveLedger = field(default_factory=SleeveLedger)
    last_plan: Plan | None = None
    last_error: str = ""
    orders_sent: int = 0
    orders_refused: int = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def due(self, day: str) -> bool:
        """Whether today's run still needs to happen."""
        return self.config.enabled and not self.ledger.already_ran(day)

    async def daily_closes(self, client: Any, *, lookback_days: int = 900,
                           ) -> dict[str, list[float]]:
        """Adjusted daily closes for the sleeve's universe.

        ``adjustment="all"`` is not optional here. An unadjusted series steps
        at every split and every dividend, and a breakout rule reads those
        steps as signals -- the sleeve would spend its life trading corporate
        actions.
        """
        import datetime as dt

        start = (dt.datetime.now(tz=dt.timezone.utc)
                 - dt.timedelta(days=lookback_days))
        rows = await client.bars(list(self.config.universe), timeframe="1Day",
                                 limit=10_000, start=start, adjustment="all")
        out: dict[str, list[float]] = {}
        for symbol, bars in (rows or {}).items():
            closes = [float(b["c"]) for b in bars
                      if isinstance(b.get("c"), (int, float)) and b["c"] > 0]
            if closes:
                out[symbol] = closes
        return out

    def simulate(self, result: Plan, prices: dict[str, float], day: str) -> None:
        """Apply a plan to the ledger without sending anything.

        Used in dry run and paper mode, and it is the *same* bookkeeping the
        live path performs after a fill -- so a paper sleeve and a live sleeve
        carry their positions and stops identically, which is the only way the
        paper run tells you anything about the live one.
        """
        for intention in result.intentions:
            price = float(prices.get(intention.symbol) or 0.0)
            if price <= 0:
                continue
            if intention.side == "sell" and intention.reason == "exit_stop":
                self.ledger.close_position(intention.symbol)
                continue
            delta = intention.notional / price
            held = self.ledger.held(intention.symbol)
            if intention.side == "sell":
                self.ledger.resize(intention.symbol,
                                   max(0.0, held.quantity - delta))
                continue
            if held.is_flat:
                self.ledger.open_position(
                    intention.symbol, delta, price,
                    result.stops.get(intention.symbol, float("nan")), day)
            else:
                self.ledger.resize(intention.symbol, held.quantity + delta)

        # Stops for everything still held, including symbols that placed no
        # order today. A stop that only moves on days the sleeve trades is a
        # stop that stops trailing exactly when the position is quiet.
        for symbol, stop in result.stops.items():
            position = self.ledger.positions.get(symbol)
            if position is not None and math.isfinite(stop):
                position.stop = max(position.stop, stop) \
                    if math.isfinite(position.stop) else stop

    def panel(self) -> dict[str, Any]:
        """What the terminal shows about this sleeve."""
        held = self.ledger.longs()
        result = self.last_plan
        return {
            "enabled": self.config.enabled,
            "allocation": self.config.allocation,
            "universe": len(self.config.universe),
            "holding": len(held),
            "symbols": held,
            "stops": {s: round(self.ledger.held(s).stop, 2)
                      for s in held if math.isfinite(self.ledger.held(s).stop)},
            "last_run_day": self.ledger.last_run_day,
            "runs": self.ledger.runs,
            "gross_weight": round(result.gross_weight, 4) if result else 0.0,
            "sleeve_equity": round(result.sleeve_equity, 2) if result else 0.0,
            "capped": bool(result.capped) if result else False,
            "too_small": len(result.too_small) if result else 0,
            "note": result.note if result else "",
            "orders_sent": self.orders_sent,
            "orders_refused": self.orders_refused,
            "last_error": self.last_error,
            "max_leverage": self.config.max_leverage,
            "exec_mode": self.config.exec_mode,
        }
