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
        """Whether the sleeve may trade: configured on, or armed by equity."""
        return bool(self.config.enabled or self.ledger.armed_on)

    def consider_arming(self, account_equity: float, day: str) -> str:
        """Arm the sleeve the first time equity reaches the threshold.

        Returns the announcement when it arms, and "" every other time.

        **One way only.** It arms and never disarms. A sleeve that switched
        itself off on a dip below the threshold would abandon whatever it was
        holding -- the positions would sit there with their stops no longer
        being trailed and nothing left to close them, which is worse than
        either state on its own. If the operator wants it off, that is a
        decision with a file to write it in.

        **Recorded, not re-derived.** The fact of arming is persisted, so a
        restart does not announce it again, and so the sleeve stays armed
        through a drawdown.
        """
        if self.config.enabled or self.ledger.armed_on:
            return ""
        threshold = float(self.config.arm_at_equity or 0.0)
        if threshold <= 0 or float(account_equity) < threshold:
            return ""

        self.ledger.armed_at_equity = float(account_equity)
        self.ledger.armed_on = day
        self.ledger.armed_by = "equity"
        self.ledger.save()
        sleeve = self.config.sleeve_equity(account_equity)
        return (f"Sector Trend has armed itself: equity reached "
                f"${account_equity:,.2f}, past the ${threshold:,.0f} "
                f"threshold. It now trades {self.config.universe_size} sector "
                f"ETFs on ${sleeve:,.2f} of its own capital, once a day. "
                f"The other strategies lose that slice. Disarm it in the "
                f"Sector trend panel while it is holding nothing.")

    def arm_by_hand(self, account_equity: float, day: str) -> str:
        """Arm it now, because the operator said so.

        Separate from ``consider_arming`` rather than a flag on it: one is the
        program deciding and the other is a person deciding, they are recorded
        differently, and only one of them is allowed to happen below the
        threshold.
        """
        if self.enabled:
            return ""
        self.ledger.armed_at_equity = float(account_equity)
        self.ledger.armed_on = day
        self.ledger.armed_by = "hand"
        self.ledger.save()
        sleeve = self.config.sleeve_equity(account_equity)
        return (f"Sector Trend armed by hand at ${account_equity:,.2f}. It "
                f"now trades {self.config.universe_size} sector ETFs on "
                f"${sleeve:,.2f} of its own capital, once a day, and the "
                f"other strategies lose that slice.")

    def disarm(self) -> str:
        """Switch it off. Refused while it is holding anything.

        The one-way rule was never about arming being sacred -- it was about
        what disarming does to open positions. A sleeve switched off mid-book
        leaves its ETFs sitting there with nobody trailing their stops and
        nothing left to close them, which is worse than either state on its
        own. Flat, there is nothing to abandon, so there is nothing to
        protect and the operator can have the switch.

        Returns "" on success, or the reason it was refused.
        """
        held = self.ledger.longs()
        if held:
            return (f"Sector Trend is holding {len(held)} position"
                    f"{'' if len(held) == 1 else 's'} "
                    f"({', '.join(held[:4])}"
                    f"{', and more' if len(held) > 4 else ''}). Disarming now "
                    f"would leave them with nobody trailing their stops. Wait "
                    f"for it to close them, or close them yourself first.")
        if self.config.enabled:
            return ("Sector Trend is switched on in settings.txt. Set "
                    "SECTOR_TREND_ENABLED=false there and restart; a button "
                    "cannot overrule a file the operator wrote.")
        self.ledger.armed_at_equity = 0.0
        self.ledger.armed_on = ""
        self.ledger.armed_by = ""
        self.ledger.save()
        return ""

    def arming_progress(self, account_equity: float) -> dict[str, Any]:
        """How close the account is to the threshold that arms this.

        Shown as a gauge rather than left implicit. The sleeve switching
        itself on is the most surprising thing this program does unattended,
        and a bar creeping toward a line is the difference between that being
        a surprise and being something the operator watched coming.
        """
        threshold = float(self.config.arm_at_equity or 0.0)
        equity = max(0.0, float(account_equity))
        return {
            "threshold": threshold,
            "equity": round(equity, 2),
            "fraction": round(min(1.0, equity / threshold), 4)
            if threshold > 0 else 0.0,
            "short_by": round(max(0.0, threshold - equity), 2)
            if threshold > 0 else 0.0,
            "watching": threshold > 0 and not self.enabled,
        }

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

    def panel(self, account_equity: float = 0.0) -> dict[str, Any]:
        """What the terminal shows about this sleeve.

        Takes the account balance because the arming gauge is about the
        account rather than about the sleeve: the sleeve has no capital at all
        until it arms, so it cannot answer "how close are we" from anything it
        owns.
        """
        held = self.ledger.longs()
        result = self.last_plan
        return {
            "enabled": self.enabled,
            "configured": self.config.enabled,
            "arm_at_equity": self.config.arm_at_equity,
            "armed_on": self.ledger.armed_on,
            "armed_at_equity": round(self.ledger.armed_at_equity, 2),
            "armed_by": self.ledger.armed_by,
            # Disarming is refused while it holds anything, so the button can
            # say why before it is pressed rather than after.
            "can_disarm": self.enabled and not self.ledger.longs()
                          and not self.config.enabled,
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
            "arming": self.arming_progress(account_equity),
            # The volatility ceiling this balance can reach. Below the
            # threshold the sleeve is too small for its most volatile names to
            # clear the venue's one-dollar minimum, and the operator should see
            # that before arming rather than discover it as a run that places
            # eight orders out of nineteen.
            "vol_ceiling": self.volatility_ceiling(account_equity),
        }

    def volatility_ceiling(self, account_equity: float) -> float:
        """The daily sigma above which a symbol is too small to order, as a
        fraction. Zero when the sleeve has no capital at all.

        Exact, not estimated. Sizing is ``w = (target_vol / N) / sigma``, so a
        position is worth ``sleeve * target_vol / (N * sigma)`` and clears the
        venue's one-dollar floor only while
        ``sigma <= sleeve * target_vol / N``.

        This is the number that makes a small sleeve dangerous rather than
        merely modest. Weight falls as volatility rises, so the symbols priced
        out first are the most volatile ones -- an undersized sleeve does not
        trade a smaller version of this strategy, it trades the calm half of
        it, which is a different strategy with different statistics and no
        backtest behind it.
        """
        sleeve = self.config.sleeve_equity(account_equity)
        universe = max(1, self.config.universe_size)
        if sleeve <= 0 or self.config.target_vol <= 0:
            return 0.0
        return sleeve * self.config.target_vol / (universe
                                                  * MIN_FRACTIONAL_NOTIONAL)
