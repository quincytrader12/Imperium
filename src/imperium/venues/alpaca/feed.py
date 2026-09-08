"""Live market data over Alpaca's websocket streams.

Alpaca splits the feed by asset class: equities stream from
``/v2/{feed}`` and crypto from ``/v1beta3/crypto/us``. They speak the same
protocol but are separate sockets with separate subscriptions, so a mixed
universe needs both -- running one and assuming it covers everything is how half
the book silently goes stale.

Authentication is an ``auth`` message rather than a header, and the server
answers before any data flows. That handshake is the only place a bad key shows
up on this transport, so it is reported specifically rather than as a generic
disconnect.

Failures here are events, not exceptions: a dropped socket must never kill the
loop or reach the UI as a traceback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import websockets

from imperium.execution.bars import Bar
from imperium.telemetry.streams import Level, TelemetryHub
from imperium.venues.assets import AssetClass, classify_symbol

log = logging.getLogger("imperium.feed")


@dataclass
class Quote:
    symbol: str
    last: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    change_pct: float = 0.0
    quote_volume: float = 0.0
    updated_at: float = 0.0

    @property
    def age(self) -> float:
        return time.time() - self.updated_at if self.updated_at else float("inf")


class MarketFeed:
    """Subscribes to bar and quote streams for a set of symbols.

    One connection per asset class, both managed here so the rest of the
    program sees a single feed with a single health story.
    """

    def __init__(self, spec, telemetry: TelemetryHub, feed: str = "iex") -> None:
        self.spec = spec
        self.telemetry = telemetry
        self.feed = feed
        self.api_key = ""
        self.secret = ""
        self.quotes: dict[str, Quote] = {}
        self.symbols: list[str] = []
        self.connected = False
        self.reconnects = 0
        self.errors = 0
        self.last_message_at: float = 0.0
        self.last_error: str = ""
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self._bar_handlers: list[Callable[[str, Bar], None]] = []
        self._live: dict[AssetClass, bool] = {}

    def set_credentials(self, api_key: str, secret: str) -> None:
        self.api_key, self.secret = api_key, secret

    def on_bar(self, handler: Callable[[str, Bar], None]) -> None:
        self._bar_handlers.append(handler)

    def quote(self, symbol: str) -> Quote:
        q = self.quotes.get(symbol)
        if q is None:
            q = Quote(symbol)
            self.quotes[symbol] = q
        return q

    @property
    def data_age(self) -> float:
        return time.time() - self.last_message_at if self.last_message_at else float("inf")

    def _url(self, asset_class: AssetClass) -> str:
        if asset_class is AssetClass.CRYPTO:
            return "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
        return f"wss://stream.data.alpaca.markets/v2/{self.feed}"

    # -- lifecycle -------------------------------------------------------

    async def start(self, symbols: list[str]) -> None:
        await self.stop()
        self.symbols = list(symbols)
        self._stop.clear()
        grouped: dict[AssetClass, list[str]] = {}
        for symbol in self.symbols:
            cls = classify_symbol(symbol)
            if cls is AssetClass.US_OPTION:
                continue
            grouped.setdefault(cls, []).append(symbol)
        for asset_class, group in grouped.items():
            self._tasks.append(asyncio.create_task(
                self._run(asset_class, group), name=f"feed-{asset_class.value}"))

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        self.connected = False
        self._live.clear()

    # -- the loop --------------------------------------------------------

    async def _run(self, asset_class: AssetClass, symbols: list[str]) -> None:
        backoff = 1.0
        url = self._url(asset_class)
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20,
                                              ping_timeout=20, close_timeout=5,
                                              max_queue=1024) as ws:
                    await self._handshake(ws, asset_class, symbols)
                    self._live[asset_class] = True
                    self.connected = any(self._live.values())
                    self.last_error = ""
                    backoff = 1.0
                    self.telemetry.event(
                        Level.GOOD, "feed",
                        f"{asset_class.value} data connected "
                        f"({len(symbols)} symbols)")
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        self._handle(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A dropped socket is an event, not an exception. It must never
                # kill this loop or reach the UI as a traceback.
                self._live[asset_class] = False
                self.connected = any(self._live.values())
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.telemetry.event(
                    Level.WARN, "feed",
                    f"{asset_class.value} data disconnected",
                    detail=self.last_error)
            if self._stop.is_set():
                break
            self._live[asset_class] = False
            self.connected = any(self._live.values())
            self.reconnects += 1
            # Jittered backoff: without it every stream retries in lockstep
            # after a venue blip and reproduces the overload.
            delay = min(30.0, backoff) * (0.5 + random.random())
            backoff = min(30.0, backoff * 2)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def _handshake(self, ws, asset_class: AssetClass,
                         symbols: list[str]) -> None:
        """Authenticate, then subscribe. A bad key shows up only here."""
        await ws.send(json.dumps({"action": "auth", "key": self.api_key,
                                  "secret": self.secret}))
        deadline = time.time() + 15
        while time.time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            messages = self._decode(raw)
            for msg in messages:
                msg_type = msg.get("T")
                if msg_type == "error":
                    raise RuntimeError(
                        f"the data feed rejected the connection: "
                        f"{msg.get('msg', 'unknown error')} "
                        f"(code {msg.get('code')}). A paper key works on the "
                        f"data feed, but the key must be valid and the plan "
                        f"must include the '{self.feed}' feed.")
                if msg_type == "success" and msg.get("msg") == "authenticated":
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "bars": symbols,
                        "quotes": symbols,
                    }))
                    return
        raise RuntimeError("the data feed did not answer the auth handshake")

    @staticmethod
    def _decode(raw: str | bytes) -> list[dict[str, Any]]:
        try:
            payload = json.loads(raw)
        except ValueError:
            return []
        if isinstance(payload, list):
            return [m for m in payload if isinstance(m, dict)]
        return [payload] if isinstance(payload, dict) else []

    def _handle(self, raw: str | bytes) -> None:
        messages = self._decode(raw)
        if not messages:
            self.errors += 1
            return
        self.last_message_at = time.time()
        for msg in messages:
            kind = msg.get("T")
            symbol = msg.get("S")
            if not symbol:
                continue
            if kind == "b":                       # a bar
                try:
                    bar = Bar(
                        open_time=_ms(msg.get("t")),
                        open=float(msg["o"]), high=float(msg["h"]),
                        low=float(msg["l"]), close=float(msg["c"]),
                        volume=float(msg.get("v", 0.0)), closed=True,
                    )
                except (KeyError, TypeError, ValueError):
                    self.errors += 1
                    continue
                q = self.quote(symbol)
                q.last = bar.close
                q.updated_at = self.last_message_at
                for handler in self._bar_handlers:
                    try:
                        handler(symbol, bar)
                    except Exception:
                        # One symbol's handler failing must not stop the feed
                        # for every other symbol.
                        log.exception("a bar handler failed for %s", symbol)
                        self.errors += 1
            elif kind == "q":                     # a quote
                q = self.quote(symbol)
                try:
                    bid = float(msg.get("bp", 0.0) or 0.0)
                    ask = float(msg.get("ap", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if bid > 0:
                    q.bid = bid
                if ask > 0:
                    q.ask = ask
                if bid > 0 and ask > 0:
                    q.last = q.last or (bid + ask) / 2
                q.updated_at = self.last_message_at
            elif kind == "error":
                self.errors += 1
                self.last_error = str(msg.get("msg", "feed error"))
                self.telemetry.event(Level.WARN, "feed", self.last_error)


def _ms(value: Any) -> int:
    """Alpaca timestamps are RFC-3339 strings; bars are keyed by epoch ms."""
    if isinstance(value, (int, float)):
        return int(value)
    if not value:
        return 0
    import datetime as dt

    text = str(value).replace("Z", "+00:00")
    # Trim nanoseconds, which fromisoformat cannot parse.
    if "." in text:
        head, _, tail = text.partition(".")
        digits = "".join(c for c in tail if c.isdigit())[:6]
        offset = tail[len(digits):] if len(tail) > len(digits) else ""
        for marker in ("+", "-"):
            if marker in tail:
                offset = tail[tail.index(marker):]
                break
        text = f"{head}.{digits or '0'}{offset}"
    try:
        return int(dt.datetime.fromisoformat(text).timestamp() * 1000)
    except ValueError:
        return 0
