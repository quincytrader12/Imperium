"""Live market data over the venue's websocket stream.

Two properties the UI depends on:

* **Data age is tracked and published.** A feed that silently stopped and a
  quiet market look identical from a price alone, and "the bot has stopped" is
  the exact question the terminal exists to answer.
* **Reconnects are counted, not hidden.** A feed that reconnects every 20
  seconds is working, technically, and is also broken.

Failures here are events, not exceptions: a dropped socket must never kill the
loop, and never propagate into the UI as a stack trace.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import websockets

from godalgo.execution.bars import Bar
from godalgo.telemetry.streams import Level, TelemetryHub

log = logging.getLogger("godalgo.feed")


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
    """Subscribes to kline and book-ticker streams for a set of symbols."""

    def __init__(self, ws_url: str, telemetry: TelemetryHub,
                 interval: str = "1m") -> None:
        self.ws_url = ws_url
        self.telemetry = telemetry
        self.interval = interval
        self.quotes: dict[str, Quote] = {}
        self.symbols: list[str] = []
        self.connected = False
        self.reconnects = 0
        self.errors = 0
        self.last_message_at: float = 0.0
        self.last_error: str = ""
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._bar_handlers: list[Callable[[str, Bar], None]] = []
        self._generation = 0

    # -- lifecycle -------------------------------------------------------

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

    async def start(self, symbols: list[str]) -> None:
        self.symbols = list(symbols)
        self._stop.clear()
        if self._task and not self._task.done():
            self._generation += 1        # force the running loop to resubscribe
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = asyncio.create_task(self._run(), name="market-feed")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.connected = False

    # -- the loop --------------------------------------------------------

    def _stream_url(self) -> str:
        parts: list[str] = []
        for s in self.symbols:
            low = s.lower()
            parts.append(f"{low}@kline_{self.interval}")
            parts.append(f"{low}@bookTicker")
        return f"{self.ws_url}?streams={'/'.join(parts)}"

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self._stream_url(), ping_interval=20, ping_timeout=20,
                    close_timeout=5, max_queue=512,
                ) as ws:
                    self.connected = True
                    self.last_error = ""
                    backoff = 1.0
                    self.telemetry.event(Level.GOOD, "feed",
                                         f"market data connected ({len(self.symbols)} symbols)")
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        self._handle(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A dropped socket is an event, not an exception. It must never
                # kill this loop, and never reach the UI as a traceback.
                self.connected = False
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.telemetry.event(
                    Level.WARN, "feed", "market data disconnected",
                    detail=self.last_error,
                )
            if self._stop.is_set():
                break
            self.connected = False
            self.reconnects += 1
            # Jittered backoff: without the jitter every symbol's feed retries in
            # lockstep after a venue blip and reproduces the overload.
            delay = min(30.0, backoff) * (0.5 + random.random())
            backoff = min(30.0, backoff * 2)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    def _handle(self, raw: str | bytes) -> None:
        try:
            payload = json.loads(raw)
        except ValueError:
            self.errors += 1
            return
        data = payload.get("data", payload)
        self.last_message_at = time.time()

        if "k" in data:
            k = data["k"]
            symbol = k["s"]
            bar = Bar(
                open_time=int(k["t"]), open=float(k["o"]), high=float(k["h"]),
                low=float(k["l"]), close=float(k["c"]), volume=float(k["v"]),
                closed=bool(k["x"]),
            )
            q = self.quote(symbol)
            q.last = bar.close
            q.updated_at = self.last_message_at
            for handler in self._bar_handlers:
                try:
                    handler(symbol, bar)
                except Exception:
                    # One symbol's handler blowing up must not stop the feed for
                    # every other symbol.
                    log.exception("a bar handler failed for %s", symbol)
                    self.errors += 1
        elif "b" in data and "a" in data:
            symbol = data["s"]
            q = self.quote(symbol)
            try:
                q.bid = float(data["b"])
                q.ask = float(data["a"])
            except (TypeError, ValueError):
                return
            q.updated_at = self.last_message_at
