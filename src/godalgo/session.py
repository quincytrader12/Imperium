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

from godalgo.execution.bars import Bar
from godalgo.execution.broker import (
    Broker, DryRunBroker, Fill, LiveBroker, Mode, ModeSwitchRefused, PaperBroker,
)
from godalgo.execution.engine import Decision, SymbolEngine
from godalgo.execution.portfolio import PortfolioAllocator, Verdict
from godalgo.execution.risk import RiskLimits
from godalgo.security.credentials import Credential, CredentialStore
from godalgo.strategy.regime import CalibrationMissing, load_calibration
from godalgo.strategy.signals import StrategyParams
from godalgo.telemetry.streams import Level, TelemetryHub
from godalgo.venues import registry
from godalgo.venues.binance.client import BinanceSpotClient, VenueError
from godalgo.venues.binance.feed import MarketFeed
from godalgo.venues.binance.filters import format_decimal
from godalgo.venues.registry import VenueSpec

log = logging.getLogger("godalgo.session")

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
        self.feed = MarketFeed(self.spec.ws_url, self.telemetry)
        self.feed.on_bar(self._on_bar)
        self.broker: Broker = DryRunBroker(self.spec)
        self.client: BinanceSpotClient | None = None
        self.credential: Credential | None = None
        self.lamps = Lamps()
        self.running = False
        self.started_at: float = 0.0
        self.day_start_equity: float = 0.0
        self.universe: list[str] = list(self.spec.default_universe)
        self.status_message = "idle"
        self.venue_error: str = ""
        self.calibration_error: str = ""
        self._loop_task: asyncio.Task | None = None
        self._pending_bars: asyncio.Queue[tuple[str, Bar]] = asyncio.Queue(maxsize=4096)
        self._thresholds: dict | None = None
        self._equity_curve: list[tuple[float, float]] = []

    # -- setup -----------------------------------------------------------

    def thresholds(self) -> dict | None:
        if self._thresholds is None:
            try:
                self._thresholds = load_calibration()["thresholds"]
                self.calibration_error = ""
            except CalibrationMissing as exc:
                self.calibration_error = str(exc)
                self.telemetry.event(Level.ERROR, "strategy",
                                     "the regime classifier is not calibrated",
                                     detail=str(exc))
        return self._thresholds

    def engine(self, symbol: str) -> SymbolEngine:
        e = self.engines.get(symbol)
        if e is None:
            e = SymbolEngine(symbol, self.spec, self.limits, self.allocator,
                             self.telemetry, self.params, self.thresholds())
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
        self.client = BinanceSpotClient(cred.api_key, cred.secret,
                                        base_url=self.spec.base_url)
        try:
            await self.client.sync_time()
            await self.client.account()
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
        self.telemetry.event(Level.GOOD, "venue",
                             f"key {name!r} accepted by {self.spec.display_name}")
        await self._confirm_fee_tier()

    async def _confirm_fee_tier(self) -> None:
        """Replace the assumed fee schedule with the account's real rates.

        An assumed tier is reported as a warning precisely so that this can
        remove it. Until this succeeds, every cost estimate says ASSUMED.
        """
        if not self.client or not self.universe:
            return
        data = await self.client.account_commission(self.universe[0])
        if not data:
            return
        try:
            std = data["standardCommission"]
            maker = Decimal(str(std["maker"])) * 10_000
            taker = Decimal(str(std["taker"])) * 10_000
        except (KeyError, TypeError, ValueError):
            return
        confirmed = self.spec.fees.confirmed(
            maker, taker, "read from /api/v3/account/commission")
        object.__setattr__(self.spec, "fees", confirmed)
        self.telemetry.event(
            Level.GOOD, "costs",
            f"fee tier confirmed for this account: {maker}bp maker, {taker}bp taker",
        )

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
        if not self.client:
            return
        for symbol in self.universe:
            try:
                rows = await self.client.klines(symbol, "1m", limit=500)
            except VenueError as exc:
                self.telemetry.event(
                    Level.WARN, "data",
                    f"could not load history for {symbol}: {exc.message}",
                    detail=exc.remedy)
                self.allocator.set_scan(symbol, score=0.0, turnover=0.0,
                                        tradeable=False,
                                        reason=f"no price history: {exc.message}")
                continue
            self.engine(symbol).seed(rows)

    async def refresh_universe(self) -> None:
        """Score and admit symbols. Failures here demote a symbol, never crash."""
        if not self.client:
            return
        try:
            tickers = await self.client.ticker_24h(self.universe)
        except VenueError as exc:
            self.lamps.venue = "bad"
            self.venue_error = exc.operator_text()
            self.telemetry.event(Level.WARN, "venue",
                                 f"could not refresh the universe: {exc.message}",
                                 detail=exc.remedy)
            return
        self.lamps.venue = "ok"
        self.venue_error = ""
        for t in tickers:
            symbol = t.get("symbol", "")
            if not symbol:
                continue
            try:
                turnover = float(t.get("quoteVolume", 0.0))
                change = float(t.get("priceChangePercent", 0.0))
                last = float(t.get("lastPrice", 0.0))
            except (TypeError, ValueError):
                continue
            q = self.feed.quote(symbol)
            q.quote_volume = turnover
            q.change_pct = change
            if not q.last:
                q.last = last
            if not q.updated_at:
                q.updated_at = time.time()
            decision = self.engine(symbol).decision
            self.allocator.set_scan(
                symbol, score=abs(decision.conviction), turnover=turnover,
                tradeable=turnover > 0,
                reason="24h turnover is zero at the venue" if turnover <= 0 else "",
            )
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
            "mode": self.broker.mode.value,
            "simulated": getattr(self.broker, "simulated", True),
            "venue": self.spec.display_name,
            "status": self.status_message,
            "venue_error": self.venue_error,
            "calibration_error": self.calibration_error,
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
