"""The trading session: the object the server exposes and the UI renders.

Owns the venue client, the feed, the engines, the allocator and the broker, and
produces the one snapshot the websocket streams at a fixed 1Hz.

Everything here is failure-tolerant by construction. A venue that rejects a key,
a symbol that does not exist, a rate limit -- these are information the operator
needs on screen, and none of them may kill a background loop or reach the UI as
a traceback.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from imperium.execution.bars import Bar
from imperium.execution.broker import (
    Broker, DryRunBroker, Fill, LiveBroker, Mode, ModeSwitchRefused, PaperBroker,
)
from imperium.execution.engine import Decision, SymbolEngine
from imperium.execution.portfolio import PortfolioAllocator, Verdict
from imperium.execution.risk import RiskLimits
from imperium.security.credentials import Credential, CredentialStore
from imperium.execution.costs import ADVERSE_SELECTION_FRACTION
from imperium.strategy.regime import CalibrationMissing, Regime, load_calibration
from imperium.venues.assets import AssetClass, classify_symbol, spec_for
from imperium.strategy.signals import StrategyParams
from imperium.telemetry.streams import Level, TelemetryHub
from imperium.venues import registry
from imperium.venues.alpaca.client import AlpacaClient, MarketClock, VenueError
from imperium.venues.alpaca.feed import MarketFeed
from imperium.venues.alpaca.filters import format_decimal
from imperium.venues.registry import VenueSpec

log = logging.getLogger("imperium.session")


def _bar_ms(value: Any) -> int:
    """Alpaca timestamps are RFC-3339; bars are keyed by epoch milliseconds."""
    from imperium.venues.alpaca.feed import _ms

    return _ms(value)

#: A quote older than this is stale enough that acting on it is guessing.
STALE_AFTER_SECONDS = 20.0


@dataclass
class Lamps:
    """The header status lamps. Each is a fact, not an aspiration."""

    link: str = "off"       # websocket to the browser
    venue: str = "off"      # REST reachable
    data: str = "off"       # market data flowing
    key: str = "off"        # a credential is loaded and working
    session: str = "off"    # the trading loop is running

    def as_dict(self) -> dict[str, str]:
        return {"link": self.link, "venue": self.venue, "data": self.data,
                "key": self.key, "session": self.session}


class TradingSession:
    """One venue, one book, one pool of money."""

    def __init__(self, venue_id: str = registry.DEFAULT_VENUE,
                 limits: RiskLimits | None = None,
                 params: StrategyParams | None = None) -> None:
        self.spec: VenueSpec = registry.get(venue_id)
        self.limits = limits or RiskLimits()
        self.params = params or StrategyParams()
        self.telemetry = TelemetryHub()
        self.allocator = PortfolioAllocator(self.limits)
        self.engines: dict[str, SymbolEngine] = {}
        self.feed = MarketFeed(self.spec, self.telemetry, feed=self.spec.default_feed)
        self.feed.on_bar(self._on_bar)
        self.broker: Broker = DryRunBroker(self.spec)
        self.client: BinanceSpotClient | None = None
        self.credential: Credential | None = None
        self.lamps = Lamps()
        self.running = False
        self.started_at: float = 0.0
        self.day_start_equity: float = 0.0
        #: Seeded, then replaced by a live scan of what the venue actually
        #: lists and what is actually trading.
        self.universe: list[str] = list(self.spec.seed_universe)
        self.paper_endpoint: bool = True
        self.market_clock: MarketClock = MarketClock()
        self.universe_scanned_at: float = 0.0
        self.scan_note: str = "not yet scanned"
        self.status_message = "idle"
        self.venue_error: str = ""
        self.calibration_error: str = ""
        #: Why the credential store is unusable, if it is. Shown in the UI: a
        #: terminal that starts but silently has no credentials is worse than
        #: one that says why.
        self.store_error: str = ""
        self._loop_task: asyncio.Task | None = None
        self._pending_bars: asyncio.Queue[tuple[str, Bar]] = asyncio.Queue(maxsize=4096)
        self._thresholds: dict | None = None
        self._equity_curve: list[tuple[float, float]] = []
        self._account_checked_at: float = 0.0

    # -- setup -----------------------------------------------------------

    def thresholds(self) -> dict | None:
        """Kept only to surface a calibration failure early.

        Each engine now selects the measured thresholds for its own asset
        class, because equities and crypto are fitted separately.
        """
        try:
            load_calibration()
            self.calibration_error = ""
        except CalibrationMissing as exc:
            self.calibration_error = str(exc)
            self.telemetry.event(Level.ERROR, "strategy",
                                 "the regime classifier is not calibrated",
                                 detail=str(exc))
        return None

    def engine(self, symbol: str) -> SymbolEngine:
        e = self.engines.get(symbol)
        if e is None:
            e = SymbolEngine(symbol, self.spec, self.limits, self.allocator,
                             self.telemetry, self.params)
            self.engines[symbol] = e
            self.allocator.observe(symbol)
        return e

    async def attach_credential(self, store: CredentialStore,
                                name: str | None) -> None:
        """Attach a stored key, or run with none.

        Running with no key is a supported state, not an error: live prices and
        the whole scanner work without one, and requiring a credential before
        anything renders is how a first-run experience becomes unusable.
        """
        await self.detach_client()
        if name is None:
            self.credential = None
            self.lamps.key = "off"
            self.client = BinanceSpotClient(base_url=self.spec.base_url)
            return
        cred = store.require(name)
        self.credential = cred
        self.client = AlpacaClient(cred.api_key, cred.secret,
                                   paper=self.paper_endpoint,
                                   data_url=self.spec.data_url,
                                   feed=self.spec.default_feed)
        self.feed.set_credentials(cred.api_key, cred.secret)
        try:
            await self.client.account()
            self.market_clock = await self.client.get_clock()
        except VenueError as exc:
            self.lamps.key = "bad"
            self.venue_error = exc.operator_text()
            self.telemetry.event(Level.ERROR, "venue",
                                 f"the key {name!r} was rejected: {exc.message}",
                                 detail=exc.remedy)
            return
        self.lamps.key = "ok"
        self.lamps.venue = "ok"
        self.venue_error = ""
        self.telemetry.event(
            Level.GOOD, "venue",
            f"key {name!r} accepted by {self.spec.display_name} "
            f"({self.client.environment}) — {self.market_clock.describe()}")
        await self.scan_universe()

    async def refresh_clock(self) -> None:
        """Ask the venue whether the market is open.

        Believing the venue rather than computing a calendar locally is the only
        way to get early closes, holidays and unscheduled halts right, and each
        of those is a day a naive calendar trades into a closed market.
        """
        if self.client is None or not self.client.authenticated:
            return
        try:
            self.market_clock = await self.client.get_clock()
        except VenueError as exc:
            self.telemetry.event(Level.WARN, "venue",
                                 "could not read the market clock",
                                 detail=exc.message)
            return
        self.allocator.market_open = self.market_clock.is_open or self._crypto_only()
        self.allocator.market_note = (
            "" if self.allocator.market_open
            else f"{self.market_clock.describe()}; equities take no new exposure "
                 f"while closed")

    def _crypto_only(self) -> bool:
        """True when every admitted symbol trades around the clock.

        A closed equity market must not stop a crypto book, and vice versa.
        """
        admitted = self.allocator.admitted_symbols or self.universe
        return bool(admitted) and all(
            classify_symbol(s) is AssetClass.CRYPTO for s in admitted)

    async def scan_universe(self, limit: int = 40) -> None:
        """Discover what this account can actually trade, ranked by turnover.

        The seed list is a starting point, not the universe. This asks the venue
        what it lists, drops anything not tradable right now, and ranks what is
        left by dollar volume -- because a scanner whose job is to reject most of
        what it sees needs a real field to choose from, and because a symbol
        that is halted or delisted should never reach the sizer.
        """
        if self.client is None or not self.client.authenticated:
            self.scan_note = "no key attached, using the seed list"
            return
        try:
            assets = await self.client.assets()
        except VenueError as exc:
            self.scan_note = f"scan failed, using the seed list: {exc.message}"
            self.telemetry.event(Level.WARN, "universe", self.scan_note)
            return
        if not assets:
            self.scan_note = "the venue listed no assets; using the seed list"
            return

        candidates = [a for a in assets.values()
                      if a.tradable and spec_for(a.asset_class).tradeable]
        # Rank by traded value, which needs a quote. Snapshots are batched, so
        # ask about a bounded shortlist rather than every listed symbol.
        seeded = [s for s in self.spec.seed_universe if s in assets]
        others = [a.symbol for a in candidates if a.symbol not in set(seeded)]
        shortlist = seeded + others[:400]

        snaps = await self.client.snapshots(shortlist)
        ranked: list[tuple[float, str]] = []
        for symbol, snap in snaps.items():
            daily = snap.get("dailyBar") or snap.get("prevDailyBar") or {}
            try:
                close = float(daily.get("c", 0) or 0)
                volume = float(daily.get("v", 0) or 0)
            except (TypeError, ValueError):
                continue
            turnover = close * volume
            if turnover <= 0:
                continue
            q = self.feed.quote(symbol)
            q.quote_volume = turnover
            if not q.last:
                q.last = close
            open_px = float(daily.get("o", 0) or 0)
            if open_px > 0:
                q.change_pct = (close - open_px) / open_px * 100.0
            if not q.updated_at:
                q.updated_at = time.time()
            ranked.append((turnover, symbol))

        if not ranked:
            self.scan_note = ("the venue returned no traded volume; using the "
                              "seed list")
            return
        ranked.sort(reverse=True)
        chosen = [symbol for _, symbol in ranked[:limit]]
        # Keep anything currently held, whatever its rank: dropping a symbol
        # that holds a position leaves the position with nothing managing it.
        for symbol, pos in self.broker.positions.items():
            if not pos.is_flat and symbol not in chosen:
                chosen.append(symbol)

        self.universe = chosen
        self.universe_scanned_at = time.time()
        equities = sum(1 for s in chosen if classify_symbol(s) is AssetClass.US_EQUITY)
        crypto = len(chosen) - equities
        self.scan_note = (f"{len(chosen)} of {len(shortlist)} scanned "
                          f"({equities} equity, {crypto} crypto), ranked by "
                          f"traded value")
        self.telemetry.event(Level.INFO, "universe", f"scanned: {self.scan_note}")

    async def detach_client(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    # -- market data -----------------------------------------------------

    def _on_bar(self, symbol: str, bar: Bar) -> None:
        """Called from the feed. Cheap, and never raises into the socket loop."""
        try:
            self._pending_bars.put_nowait((symbol, bar))
        except asyncio.QueueFull:
            # Dropping a bar is better than stalling the socket. It is counted
            # so the health panel can show it rather than hiding it.
            self.telemetry.event(Level.WARN, "feed",
                                 "the bar queue is full; a bar was dropped")

    async def seed_history(self) -> None:
        """Seed each engine with recent bars, batched by asset class."""
        if not self.client:
            return
        try:
            batches = await self.client.bars(self.universe, timeframe="1Min",
                                             limit=1000)
        except VenueError as exc:
            self.telemetry.event(Level.WARN, "data",
                                 f"could not load history: {exc.message}",
                                 detail=exc.remedy)
            return
        for symbol in self.universe:
            rows = batches.get(symbol) or []
            if not rows:
                self.allocator.set_scan(
                    symbol, score=0.0, turnover=0.0, tradeable=False,
                    reason="no recent bars from the venue for this symbol")
                continue
            engine = self.engine(symbol)
            for index, row in enumerate(rows):
                try:
                    bar = Bar(
                        open_time=_bar_ms(row.get("t")),
                        open=float(row["o"]), high=float(row["h"]),
                        low=float(row["l"]), close=float(row["c"]),
                        volume=float(row.get("v", 0.0)),
                        # The final row is the bar still forming; treating it as
                        # closed makes every strategy act on a partial bar.
                        closed=index < len(rows) - 1,
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                engine.series.add(bar)

    async def refresh_universe(self) -> None:
        """Re-price and re-admit. Failures demote a symbol, never crash."""
        if not self.client:
            return
        await self.refresh_clock()
        try:
            snaps = await self.client.snapshots(self.universe)
            assets = await self.client.assets()
        except VenueError as exc:
            self.lamps.venue = "bad"
            self.venue_error = exc.operator_text()
            self.telemetry.event(Level.WARN, "venue",
                                 f"could not refresh the universe: {exc.message}",
                                 detail=exc.remedy)
            return
        self.lamps.venue = "ok"
        self.venue_error = ""

        for symbol in self.universe:
            snap = snaps.get(symbol) or {}
            daily = snap.get("dailyBar") or snap.get("prevDailyBar") or {}
            quote = snap.get("latestQuote") or {}
            trade = snap.get("latestTrade") or {}
            q = self.feed.quote(symbol)
            try:
                close = float(daily.get("c", 0) or 0)
                open_px = float(daily.get("o", 0) or 0)
                volume = float(daily.get("v", 0) or 0)
                last = float(trade.get("p", 0) or 0) or close
                bid = float(quote.get("bp", 0) or 0)
                ask = float(quote.get("ap", 0) or 0)
            except (TypeError, ValueError):
                continue
            if last > 0:
                q.last = last
            if bid > 0:
                q.bid = bid
            if ask > 0:
                q.ask = ask
            turnover = close * volume
            q.quote_volume = turnover
            if open_px > 0 and close > 0:
                q.change_pct = (close - open_px) / open_px * 100.0
            if q.last and not q.updated_at:
                q.updated_at = time.time()

            engine = self.engine(symbol)
            engine.set_book(q.bid or None, q.ask or None)
            # Shortability and borrow are per-symbol facts that change, so they
            # are refreshed rather than assumed once.
            asset = assets.get(symbol)
            if asset is not None:
                engine.can_short = asset.can_short
                engine.tradable = asset.tradable and spec_for(
                    asset.asset_class).tradeable

            reason = ""
            tradeable = True
            if asset is not None and not asset.tradable:
                tradeable, reason = False, (
                    f"the venue lists {symbol} as not tradable "
                    f"(status {asset.status})")
            elif turnover <= 0:
                tradeable, reason = False, "no traded value reported for this symbol"
            self.allocator.set_scan(
                symbol, score=abs(engine.decision.conviction), turnover=turnover,
                tradeable=tradeable, reason=reason)

        admitted, retired = self.allocator.rebalance_admissions()
        for symbol in admitted:
            self.telemetry.event(Level.INFO, "universe", f"{symbol} admitted")
        for symbol in retired:
            # Retiring means flattening. A stopped engine still holding a
            # position is a position with nothing managing its stop.
            self.telemetry.event(Level.WARN, "universe",
                                 f"{symbol} retired — flattening it")
            await self._flatten_symbol(symbol)

    async def _flatten_symbol(self, symbol: str) -> None:
        price = self.feed.quote(symbol).last
        if not price:
            self.telemetry.event(
                Level.ERROR, "risk",
                f"{symbol} was retired but has no price, so it could not be "
                f"flattened. The position is still open.")
            return
        try:
            fill = await self.broker.apply_target(symbol, 0.0, price, self.equity())
        except (VenueError, ModeSwitchRefused) as exc:
            self.telemetry.event(Level.ERROR, "risk",
                                 f"could not flatten {symbol}: {exc}")
            return
        if fill:
            self.telemetry.pulse(symbol, "order", f"flattened on retirement", 1.0)
            self.telemetry.event(Level.WARN, "order",
                                 f"flattened {symbol} on retirement")

    # -- the trading loop -------------------------------------------------

    def prices(self) -> dict[str, float]:
        return {s: q.last for s, q in self.feed.quotes.items() if q.last > 0}

    def equity(self) -> float:
        return float(self.broker.equity(self.prices()))

    async def _drain_bars(self) -> None:
        """Evaluate every bar that actually closed. One pulse per evaluation."""
        processed = 0
        while not self._pending_bars.empty() and processed < 512:
            symbol, bar = self._pending_bars.get_nowait()
            processed += 1
            engine = self.engine(symbol)
            newly_closed = engine.series.add(bar)
            q = self.feed.quote(symbol)
            engine.set_book(q.bid or None, q.ask or None)
            if not newly_closed:
                continue
            decision = engine.evaluate()
            await self._act_on(decision)

    async def _act_on(self, decision: Decision) -> None:
        if decision.verdict is not Verdict.TRADING:
            return
        price = self.feed.quote(decision.symbol).last
        if price <= 0:
            return
        quote_age = self.feed.quote(decision.symbol).age
        if quote_age > STALE_AFTER_SECONDS:
            self.telemetry.pulse(decision.symbol, "refused",
                                 f"quote is {quote_age:.0f}s stale", 0.4)
            return
        try:
            fill = await self.broker.apply_target(
                decision.symbol, decision.target_weight, price, self.equity())
        except ModeSwitchRefused as exc:
            self.telemetry.event(Level.ERROR, "order", str(exc))
            return
        except VenueError as exc:
            # A rejected order is information, not a crash.
            self.telemetry.event(Level.ERROR, "order",
                                 f"{decision.symbol}: {exc.message}",
                                 detail=exc.remedy)
            self.telemetry.pulse(decision.symbol, "refused", exc.message, 0.9)
            return
        if fill:
            self.allocator.observe(decision.symbol).current_weight = \
                self.broker.weight_of(decision.symbol, price, self.equity())
            qty_text = format_decimal(fill.quantity)
            px_text = format_decimal(fill.price)
            self.telemetry.pulse(decision.symbol, "order",
                                 f"{fill.side} {qty_text} at {px_text}", 1.0)
            self.telemetry.event(
                Level.INFO, "order",
                f"{fill.side} {qty_text} {decision.symbol} at {px_text}"
                + (" (simulated)" if fill.simulated else ""))

    async def _tick(self) -> None:
        await self._drain_bars()
        await self._refresh_account_limits()
        equity = self.equity()
        self.allocator.equity = equity
        self.allocator.cash = float(self.broker.cash)
        if self.day_start_equity <= 0:
            self.day_start_equity = equity
        if self.allocator.check_daily_loss(self.day_start_equity) and self.running:
            self.telemetry.pulse("BOOK", "halt", self.allocator.halt_reason, 1.0)
        self._equity_curve.append((time.time(), equity))
        if len(self._equity_curve) > 2000:
            self._equity_curve = self._equity_curve[-2000:]

    async def _refresh_account_limits(self) -> None:
        """Read the venue's own day-trade count and equity.

        Counting day trades locally cannot survive a restart or trades made
        elsewhere in the same account, and being wrong here means a
        ninety-day restriction rather than a missed trade.
        """
        if self.client is None or not self.client.authenticated:
            return
        if time.time() - self._account_checked_at < 30:
            return
        try:
            account = await self.client.account()
        except VenueError:
            return
        self._account_checked_at = time.time()
        try:
            self.allocator.day_trade_count = int(account.get("daytrade_count", 0) or 0)
            self.allocator.flagged_pattern_day_trader = bool(
                account.get("pattern_day_trader", False))
        except (TypeError, ValueError):
            pass

    async def _run(self) -> None:
        last_universe = 0.0
        while self.running:
            try:
                await self._tick()
                if time.time() - last_universe > 60:
                    await self.refresh_universe()
                    last_universe = time.time()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A failure in one tick must not stop the session. It is
                # recorded where the operator will see it.
                log.exception("the trading loop raised")
                self.telemetry.event(Level.ERROR, "session",
                                     f"the trading loop raised {type(exc).__name__}",
                                     detail=str(exc))
            await asyncio.sleep(1.0)

    # -- control ---------------------------------------------------------

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.started_at = time.time()
        self.lamps.session = "ok"
        self.status_message = f"running in {self.broker.mode.value}"
        self.telemetry.event(Level.GOOD, "session",
                             f"session started in {self.broker.mode.value}")
        await self.refresh_clock()
        await self.scan_universe()
        await self.seed_history()
        await self.refresh_universe()
        await self.feed.start(self.universe)
        self._loop_task = asyncio.create_task(self._run(), name="trading-loop")

    async def stop(self) -> None:
        self.running = False
        self.lamps.session = "off"
        self.status_message = "stopped"
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        await self.feed.stop()
        self.telemetry.event(Level.INFO, "session", "session stopped")

    async def set_mode(self, mode: Mode, phrase: str = "",
                       store: CredentialStore | None = None) -> None:
        """Switch mode, flattening first. Always in that order."""
        if mode is self.broker.mode:
            return
        prices = self.prices()
        flattened = await self.broker.flatten_all(prices)
        for fill in flattened:
            self.telemetry.event(Level.WARN, "order",
                                 f"flattened {fill.symbol} before switching mode")

        if mode is Mode.LIVE:
            if self.client is None or self.credential is None:
                raise ModeSwitchRefused(
                    "no credential is attached, so there is no account to trade")
            tradeable = self.credential.trade_enabled
            broker = LiveBroker(self.spec, self.client, self.credential.name)
            broker.arm(phrase, tradeable)
            await broker.sync()
            self.broker = broker
            self.telemetry.event(
                Level.WARN, "mode",
                f"LIVE trading armed on {self.credential.name!r} — real orders "
                "will be sent")
        elif mode is Mode.PAPER:
            self.broker = PaperBroker(self.spec)
            self.telemetry.event(Level.INFO, "mode", "switched to paper trading")
        else:
            self.broker = DryRunBroker(self.spec)
            self.telemetry.event(Level.INFO, "mode", "switched to dry run")

        for state in self.allocator.states.values():
            state.current_weight = 0.0
        self.day_start_equity = self.equity()
        self.status_message = (f"running in {self.broker.mode.value}"
                               if self.running else "stopped")

    # -- the snapshot the UI renders --------------------------------------

    def snapshot(self, pulse_window: int = 240) -> dict[str, Any]:
        prices = self.prices()
        equity = self.equity()
        age = self.feed.data_age
        self.lamps.data = ("ok" if age < STALE_AFTER_SECONDS
                           else ("stale" if math.isfinite(age) else "off"))

        rows = []
        for symbol in self.universe:
            state = self.allocator.observe(symbol)
            q = self.feed.quote(symbol)
            d = self.engines[symbol].decision if symbol in self.engines else Decision(symbol)
            rows.append({
                "symbol": symbol,
                "price": q.last,
                "change_pct": q.change_pct,
                "turnover": q.quote_volume,
                "verdict": state.verdict.value,
                "reason": (state.reason if state.verdict in
                           (Verdict.NOT_ADMITTED, Verdict.REJECTED) and state.reason
                           else d.reason),
                "weight": state.current_weight,
                "target": d.target_weight,
                "age": None if not math.isfinite(q.age) else round(q.age, 1),
                "decision": d.as_dict(),
            })

        health = self._health_score()
        positions = [
            {"symbol": s, "quantity": float(p.quantity),
             "avg_price": float(p.avg_price),
             "price": prices.get(s, 0.0),
             "value": float(p.quantity) * prices.get(s, 0.0),
             "unrealised": (prices.get(s, 0.0) - float(p.avg_price)) * float(p.quantity)
             if prices.get(s) else 0.0}
            for s, p in self.broker.positions.items() if not p.is_flat
        ]

        return {
            "ts": time.time(),
            "running": self.running,
            "uptime": (time.time() - self.started_at) if self.started_at else 0.0,
            "limits": self._limits_block(),
            "regime_census": self._regime_census(),
            "costs": self._cost_summary(),
            "venue_budget": self._venue_budget(),
            "counters": self.telemetry.kind_counts,
            "execution": self._execution_quality(),
            "drawdown": self._drawdown(),
            "mode": self.broker.mode.value,
            "simulated": getattr(self.broker, "simulated", True),
            "venue": self.spec.display_name,
            "environment": self.client.environment if self.client else "paper",
            "market": {
                "is_open": self.market_clock.is_open,
                "describe": self.market_clock.describe(),
                "next_open": (self.market_clock.next_open.isoformat()
                              if self.market_clock.next_open else None),
                "next_close": (self.market_clock.next_close.isoformat()
                               if self.market_clock.next_close else None),
                "crypto_only": self._crypto_only(),
                "feed": self.spec.default_feed,
            },
            "universe_scan": {
                "note": self.scan_note,
                "scanned_at": self.universe_scanned_at,
                "size": len(self.universe),
            },
            "status": self.status_message,
            "venue_error": self.venue_error,
            "calibration_error": self.calibration_error,
            "store_error": self.store_error,
            "lamps": self.lamps.as_dict(),
            "equity": equity,
            "cash": float(self.broker.cash),
            "realised_pnl": float(self.broker.realised_pnl),
            "unrealised_pnl": sum(p["unrealised"] for p in positions),
            "gross_exposure": self.allocator.gross_exposure,
            "gross_ceiling": self.limits.max_gross_exposure,
            "per_symbol_budget": self.allocator.per_symbol_budget,
            "halted": self.allocator.halted,
            "halt_reason": self.allocator.halt_reason,
            "credential": self.credential.name if self.credential else None,
            "trade_enabled": bool(self.credential and self.credential.trade_enabled),
            "watchlist": rows,
            "positions": positions,
            "fills": [f.as_dict() for f in self.broker.fills[-40:]][::-1],
            "events": self.telemetry.events(60),
            "pulses": self.telemetry.pulse_window(pulse_window),
            "pulse_seq": self.telemetry.latest_pulse_seq,
            "health": health,
            "feed": {
                "connected": self.feed.connected,
                "reconnects": self.feed.reconnects,
                "errors": self.feed.errors,
                "age": None if not math.isfinite(age) else round(age, 1),
                "last_error": self.feed.last_error,
            },
            "equity_curve": self._equity_curve[-240:],
        }

    def _limits_block(self) -> dict[str, Any]:
        """Every hard bound, and how much of each is currently spent.

        Published in full because an autonomous book that is not trading is
        usually being held by one specific limit, and guessing which one from a
        single "gross exposure" figure is exactly the diagnosis this panel is
        meant to remove.
        """
        lim = self.limits
        admitted = len(self.allocator.admitted_symbols)
        return {
            "max_gross_exposure": lim.max_gross_exposure,
            "max_position_weight": lim.max_position_weight,
            "max_concurrent_positions": lim.max_concurrent_positions,
            "buying_power_reserve": lim.buying_power_reserve,
            "daily_loss_halt": lim.daily_loss_halt,
            "risk_per_trade": lim.risk_per_trade,
            "target_volatility": lim.target_volatility,
            "atr_stop_multiple": lim.atr_stop_multiple,
            "slots_used": admitted,
            "day_trade_count": self.allocator.day_trade_count,
            "pdt_floor": lim.pdt_equity_floor,
            "pdt_max_day_trades": lim.pdt_max_day_trades,
            "pdt_blocked": self.allocator.pdt_blocked(),
            "flagged_pattern_day_trader": self.allocator.flagged_pattern_day_trader,
            "slots_max": lim.max_concurrent_positions,
            "buying_power": self.allocator.buying_power(),
            "reserved": float(self.broker.cash) * lim.buying_power_reserve,
        }

    def _drawdown(self) -> dict[str, Any]:
        """Loss against the day's opening equity, and how close that is to the halt."""
        start = self.day_start_equity
        equity = self.equity()
        if start <= 0:
            return {"pct": 0.0, "limit": self.limits.daily_loss_halt,
                    "used": 0.0, "day_start_equity": start}
        pct = max(0.0, (start - equity) / start)
        limit = self.limits.daily_loss_halt
        return {
            "pct": pct,
            "limit": limit,
            # Fraction of the halt budget consumed. This is the number that
            # matters: 3% of a 4% limit is 75% spent, not "only 3%".
            "used": min(1.0, pct / limit) if limit > 0 else 0.0,
            "day_start_equity": start,
        }

    def _regime_census(self) -> dict[str, int]:
        """What the book is currently seeing, counted by regime.

        A fleet-level answer to "why is nothing trading". Fifteen symbols all
        reading indeterminate is a different situation from fifteen still
        warming up, and both look like "no positions" without this.
        """
        census: dict[str, int] = {}
        for symbol in self.universe:
            engine = self.engines.get(symbol)
            regime = engine.decision.regime if engine else Regime.WARMING_UP.value
            census[regime] = census.get(regime, 0) + 1
        return census

    def _cost_summary(self) -> dict[str, Any]:
        """What the book is being priced at, per asset class.

        Reported per class rather than as one number, because the classes are
        not comparable: a US equity round trip is almost entirely spread, while
        a crypto round trip is dominated by commission. A single blended figure
        would describe neither.
        """
        decisions = [e.decision for e in self.engines.values()
                     if e.decision.round_trip_cost_bps > 0]
        by_class: dict[str, Any] = {}
        for asset_class in (AssetClass.US_EQUITY, AssetClass.CRYPTO):
            spec = spec_for(asset_class)
            members = [d for d in decisions if d.asset_class == asset_class.value]
            costs_bps = sorted(d.round_trip_cost_bps for d in members)
            by_class[asset_class.value] = {
                "display_name": spec.display_name,
                "commission_bps": float(spec.cost_model.commission_bps),
                "sell_side_bps": float(spec.cost_model.sell_side_bps),
                "assumed": spec.cost_model.assumed,
                "source": spec.cost_model.source,
                "symbols": len(members),
                "median_round_trip_bps": (costs_bps[len(costs_bps) // 2]
                                          if costs_bps else 0.0),
                "spreads_measured": sum(1 for d in members if not d.spread_assumed),
                "seconds_per_year": spec.seconds_per_year,
                "shortable": spec.shortable,
            }
        cheapest = min(decisions, key=lambda d: d.round_trip_cost_bps, default=None)
        dearest = max(decisions, key=lambda d: d.round_trip_cost_bps, default=None)
        all_costs = sorted(d.round_trip_cost_bps for d in decisions)
        return {
            "by_asset_class": by_class,
            "safety_multiple": self.params.safety_multiple,
            "adverse_selection_fraction": float(ADVERSE_SELECTION_FRACTION),
            "median_round_trip_bps": (all_costs[len(all_costs) // 2]
                                      if all_costs else 0.0),
            "fees_assumed": any(b["assumed"] for b in by_class.values()),
            "fee_source": "; ".join(
                f"{b['display_name']}: {b['source']}" for b in by_class.values()),
            "cheapest": ({"symbol": cheapest.symbol,
                          "bps": cheapest.round_trip_cost_bps} if cheapest else None),
            "dearest": ({"symbol": dearest.symbol,
                         "bps": dearest.round_trip_cost_bps} if dearest else None),
            "spreads_measured": sum(1 for d in decisions if not d.spread_assumed),
            "spreads_total": len(decisions),
            "maker_bps": 0.0,
            "taker_bps": float(spec_for(AssetClass.US_EQUITY).cost_model.commission_bps),
        }

    def _execution_quality(self) -> dict[str, Any]:
        """Did execution cost what the cost gate assumed?

        The gate admits a symbol on a *modelled* crossing cost. Measuring what
        crossing actually cost is the only way to find out the model is wrong
        before the P&L does -- and with two asset classes carrying very
        different models, one being wrong is easy to miss in a blended figure.
        """
        fills = self.broker.fills
        if not fills:
            return {"count": 0, "notional": 0.0, "buys": 0, "sells": 0,
                    "avg_slippage_bps": 0.0, "worst_slippage_bps": 0.0,
                    "modelled_bps": 0.0, "simulated": 0}
        slips = [f.slippage_bps for f in fills]
        modelled = [d.round_trip_cost_bps / 2 for d in
                    (e.decision for e in self.engines.values())
                    if d.round_trip_cost_bps > 0]
        return {
            "count": len(fills),
            "notional": float(sum(f.notional for f in fills)),
            "buys": sum(1 for f in fills if f.side == "BUY"),
            "sells": sum(1 for f in fills if f.side == "SELL"),
            "avg_slippage_bps": sum(slips) / len(slips),
            "worst_slippage_bps": max(slips),
            # One-way modelled cost, which is what a single fill should pay.
            "modelled_bps": (sum(modelled) / len(modelled)) if modelled else 0.0,
            "simulated": sum(1 for f in fills if f.simulated),
        }

    def _venue_budget(self) -> dict[str, Any]:
        """The venue's own view of request weight, plus measured clock drift.

        A repeated rate-limit ban lengthens each time, so utilisation is worth
        watching before it becomes a ban rather than after.
        """
        client = self.client
        if client is None:
            return {"used_weight": 0, "limit": 0, "utilisation": 0.0,
                    "clock_offset_ms": 0, "clock_measured": False,
                    "order_count_10s": 0}
        return {
            "used_weight": client.budget.used_weight,
            "limit": client.budget.limit,
            "utilisation": client.budget.utilisation,
            "order_count_10s": client.budget.order_count_10s,
            "clock_offset_ms": client.time_offset_ms,
            "clock_measured": client.time_offset_measured,
        }

    def _health_score(self) -> dict[str, Any]:
        """A composite 0..1 with the components named.

        A single number nobody can decompose is not diagnostic, so the parts are
        published alongside it.
        """
        age = self.feed.data_age
        data = 1.0 if age < 5 else (0.6 if age < STALE_AFTER_SECONDS else 0.0)
        link = 1.0 if self.feed.connected else 0.0
        venue = {"ok": 1.0, "off": 0.5, "bad": 0.0}.get(self.lamps.venue, 0.5)
        errors = max(0.0, 1.0 - min(1.0, self.feed.errors / 20.0))
        reconn = max(0.0, 1.0 - min(1.0, self.feed.reconnects / 10.0))
        score = 0.30 * data + 0.25 * link + 0.20 * venue + 0.15 * errors + 0.10 * reconn
        if self.allocator.halted:
            score = min(score, 0.35)
        return {
            "score": round(score, 3),
            "components": {"data": data, "link": link, "venue": venue,
                           "errors": errors, "reconnects": reconn},
            "data_age": None if not math.isfinite(age) else round(age, 1),
            "reconnects": self.feed.reconnects,
            "errors": self.feed.errors,
        }
