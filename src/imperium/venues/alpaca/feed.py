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

#: Concurrent websocket subscriptions to start with.
#:
#: Alpaca's basic (free) plan caps them; thirty is the published figure for
#: crypto trade and quote channels and the commonly reported cap for the basic
#: stock stream. Treated as a starting point rather than a fact, because the
#: real limit depends on the account's data subscription and the program has no
#: endpoint that reports it -- see MarketFeed._handle_stream_error, which halves
#: this on a 405 until the venue accepts the request.
STREAM_SYMBOL_LIMIT = 30

#: Never negotiate below this. A stream of a handful of symbols is still worth
#: having, and halving without a floor would converge on zero.
MIN_STREAM_SYMBOLS = 5


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


#: What each of Alpaca's data-stream error codes actually means, and what an
#: operator can do about it.
#:
#: This table exists because the code it replaced printed *one guessed cause*
#: for every code -- "the key must be valid and the plan must include the iex
#: feed" -- regardless of what the venue had said. For the commonest rejection
#: of all, 406, that sentence is not merely unhelpful but wrong in both halves:
#: the key is fine and the plan is fine, and the operator is sent to check two
#: things that were never the problem while the real one (a second copy of this
#: program still holding the one connection the account is allowed) goes
#: unmentioned.
#:
#: One correction worth stating plainly, because the old message had it
#: backwards: IEX is the *free* feed, included with every Alpaca account, paper
#: and live alike. No plan needs to be bought to stream it. A subscription is
#: what SIP needs, and a plan complaint on an IEX connection means something
#: else is wrong.
STREAM_ERRORS: dict[int, tuple[str, str]] = {
    400: ("the subscribe request was malformed",
          "A bug in this program rather than anything on the account. Please "
          "report it with the code and this message."),
    401: ("the stream was used before it authenticated",
          "A bug in this program's handshake ordering. Please report it."),
    402: ("the key and secret were not accepted",
          "Copy both halves again from the Alpaca dashboard under Home → API "
          "Keys. A key stops working the moment it is regenerated, and the "
          "secret is shown only once when it is created. Paper keys stream "
          "market data perfectly well, so a paper key is not the problem."),
    403: ("this connection had already authenticated",
          "Usually harmless and usually transient. If it repeats, close every "
          "other copy of this program and start one."),
    404: ("the connection did not authenticate in time",
          "Normally a slow or intercepted network. Check whether a VPN, "
          "corporate proxy or security suite is inspecting websocket traffic."),
    405: ("more symbols were requested than the plan allows",
          "Handled automatically: the cap is halved and the subscription "
          "retried."),
    406: ("this Alpaca account already has a live market data connection",
          "Alpaca allows ONE at a time, and this is by far the commonest "
          "cause of a dark data lamp. Close any other copy of IMPERIUM "
          "(check Task Manager for a second IMPERIUM.exe), any notebook, "
          "script or third-party app using the same key, and any Alpaca "
          "dashboard page showing live prices. A connection from a run that "
          "has only just exited can take up to half a minute to be released, "
          "so restarting immediately will hit this too."),
    407: ("this program could not keep up with the stream and was dropped",
          "The machine is overloaded or the connection is slow. It will "
          "reconnect with fewer symbols if the venue keeps refusing."),
    408: ("this account is not enabled for v2 market data",
          "Open the Alpaca dashboard once and accept any outstanding market "
          "data agreement, then restart."),
    409: ("the plan does not include the feed that was asked for",
          "The 'sip' feed needs a paid Alpaca subscription. The 'iex' feed is "
          "free on every account and is what this program asks for by "
          "default, so seeing this on an IEX connection means the account is "
          "restricted in some other way -- check for an outstanding market "
          "data agreement on the dashboard."),
    500: ("the venue reported an internal error",
          "Nothing to do at this end. This program keeps retrying, and "
          "Alpaca's status page will say if it is widespread."),
}

#: How long to wait before retrying after error 406.
#:
#: Long, because retrying a connection limit cannot succeed: the limit is held
#: by something else and will not be released by asking again a second later.
#: A tight retry loop here produces a wall of identical errors that buries the
#: one line explaining what to close, and looks like a misbehaving client from
#: the venue's side.
CONNECTION_LIMIT_BACKOFF = 30.0


class FeedRejected(Exception):
    """The data stream refused the connection, with the venue's own reason.

    Carries the code so that the loop can treat a connection limit differently
    from a bad key, and so the operator can quote a number rather than a
    paraphrase.
    """

    def __init__(self, code: Any, venue_message: str) -> None:
        self.code = code if isinstance(code, int) else None
        self.venue_message = venue_message or "unknown error"
        cause, remedy = STREAM_ERRORS.get(
            self.code or -1,
            ("the data feed refused the connection", ""))
        self.cause = cause
        self.remedy = remedy
        super().__init__(self.operator_text())

    def operator_text(self) -> str:
        code = f" (code {self.code})" if self.code is not None else ""
        said = (f' Alpaca said "{self.venue_message}".'
                if self.venue_message.lower() not in self.cause.lower() else "")
        return f"{self.cause}{code}.{said}"


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
        #: What to do about ``last_error``, when the venue named a
        #: cause this program has a remedy for. Kept beside the
        #: error rather than folded into it so the panel can show
        #: the fault and the fix with different weight.
        self.last_remedy: str = ""
        #: How many symbols this plan will stream at once.
        #:
        #: Alpaca's basic plan caps concurrent subscriptions; the paid plans
        #: raise or remove the cap. The starting value is the basic figure, and
        #: it is *adapted* rather than trusted, because the real limit depends
        #: on a subscription this program cannot read. An over-limit request is
        #: rejected whole -- error 405, previous subscriptions untouched, which
        #: on a fresh connection means none at all.
        self.symbol_limit: int = STREAM_SYMBOL_LIMIT
        #: Symbols above the cap. They are still scanned, still priced by the
        #: snapshot sweep, and still tradeable on the daily-bar strategies --
        #: what they lose is the live minute stream.
        self.dropped: int = 0
        #: What each class's socket is actually subscribed to, per class
        #: rather than shared: equities and crypto are separate connections
        #: with separate subscriptions, and one list for both meant whichever
        #: handshake ran last overwrote the other's accounting.
        self._subscribed: dict[AssetClass, list[str]] = {}
        #: The live socket per class, so a subscription can be changed on it
        #: instead of by reconnecting.
        self._sockets: dict[AssetClass, Any] = {}
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        #: Set when the subscription must be renegotiated, so the socket loop
        #: reconnects rather than sitting on a subscription the plan refused.
        self._resubscribe = asyncio.Event()
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
        self.dropped = max(0, len(self.symbols) - self.symbol_limit)
        self._stop.clear()
        for asset_class in self._classes():
            self._tasks.append(asyncio.create_task(
                self._run(asset_class), name=f"feed-{asset_class.value}"))

    def _classes(self) -> list[AssetClass]:
        """The asset classes the current symbol set needs a socket for."""
        seen: list[AssetClass] = []
        for symbol in self.symbols:
            cls = classify_symbol(symbol)
            # Options do not stream on these endpoints.
            if cls is AssetClass.US_OPTION or cls in seen:
                continue
            seen.append(cls)
        return seen

    def _group(self, asset_class: AssetClass) -> list[str]:
        """This class's share of the current symbol set, in priority order.

        Read fresh on every connection attempt rather than captured when the
        task was created, so a socket that reconnects after the cohort has
        rotated subscribes to the symbols being reasoned about now -- not to
        the ones the session happened to start with.
        """
        return [s for s in self.symbols if classify_symbol(s) is asset_class]

    async def retarget(self, symbols: list[str]) -> int:
        """Point the live subscription at a new set of symbols.

        The cohort rotates every twenty seconds and, until this existed,
        nothing told the socket. ``start`` was called once at session start,
        so the stream went on feeding the symbols the universe held then --
        long since retired, with nothing looking at them -- while the symbols
        actually being reasoned about had no live price at all.

        The sweep meanwhile computes each engine's ``streamed`` flag from the
        *current* universe, so the terminal reported exactly those symbols as
        streamed. The one indicator that would have shown the problem stated
        the opposite of it.

        Changed on the open socket rather than by reconnecting: a reconnect
        every twenty seconds is a reconnect storm against a venue that allows
        one concurrent data connection per key, and it would also throw away
        the subscription cap negotiated on the way in.

        Returns how many subscriptions changed hands.
        """
        self.symbols = list(symbols)
        self.dropped = max(0, len(self.symbols) - self.symbol_limit)
        if not self._tasks:
            # Not started yet. ``start`` reads self.symbols, so there is
            # nothing to renegotiate.
            return 0

        changed = 0
        for asset_class in self._classes():
            wanted = self._group(asset_class)[:self.symbol_limit]
            current = self._subscribed.get(asset_class, [])
            added = [s for s in wanted if s not in set(current)]
            gone = [s for s in current if s not in set(wanted)]
            if not added and not gone:
                continue
            ws = self._sockets.get(asset_class)
            if ws is None:
                # No socket for this class yet -- a cohort that has just
                # brought in its first crypto name, say. Its own loop will
                # subscribe the current group when it connects.
                if not any(t.get_name() == f"feed-{asset_class.value}"
                           and not t.done() for t in self._tasks):
                    self._tasks.append(asyncio.create_task(
                        self._run(asset_class),
                        name=f"feed-{asset_class.value}"))
                continue
            try:
                if gone:
                    await ws.send(json.dumps({"action": "unsubscribe",
                                              "bars": gone, "quotes": gone}))
                if added:
                    await ws.send(json.dumps({"action": "subscribe",
                                              "bars": added, "quotes": added}))
            except Exception as exc:
                # A send that fails means the socket is on its way out. The
                # loop reconnects and subscribes the current group, so this
                # needs no recovery of its own -- only to not raise into the
                # caller, which is the trading loop.
                self.last_error = f"{type(exc).__name__}: {exc}"
                continue
            self._subscribed[asset_class] = wanted
            changed += len(added) + len(gone)
        return changed

    @property
    def streamed(self) -> int:
        """How many symbols are actually subscribed, after the plan's cap."""
        return min(len(self.symbols), self.symbol_limit)

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
        self._sockets.clear()
        self._subscribed.clear()

    # -- the loop --------------------------------------------------------

    async def _run(self, asset_class: AssetClass) -> None:
        backoff = 1.0
        url = self._url(asset_class)
        while not self._stop.is_set():
            # Read fresh, not captured when the task was created: after a
            # cohort rotation the symbols worth streaming are not the ones
            # this task started with.
            symbols = self._group(asset_class)
            try:
                async with websockets.connect(url, ping_interval=20,
                                              ping_timeout=20, close_timeout=5,
                                              max_queue=1024) as ws:
                    await self._handshake(ws, asset_class, symbols)
                    self._sockets[asset_class] = ws
                    self._live[asset_class] = True
                    self.connected = any(self._live.values())
                    self.last_error = ""
                    self.last_remedy = ""
                    backoff = 1.0
                    self.telemetry.event(
                        Level.GOOD, "feed",
                        f"{asset_class.value} data connected "
                        f"({len(self._subscribed.get(asset_class, []))} of "
                        f"{len(symbols)} symbols"
                        + (f", {len(symbols) - len(self._subscribed.get(asset_class, []))}"
                           f" above the plan's cap"
                           if len(self._subscribed.get(asset_class, [])) < len(symbols)
                           else "") + ")")
                    self._resubscribe.clear()
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        self._handle(raw)
                        if self._resubscribe.is_set():
                            # The plan refused this subscription. Reconnecting
                            # is the only way to renegotiate it, and sitting
                            # here would mean a connected socket delivering
                            # nothing.
                            self._resubscribe.clear()
                            break
                self._sockets.pop(asset_class, None)
            except asyncio.CancelledError:
                self._sockets.pop(asset_class, None)
                raise
            except FeedRejected as exc:
                # The venue answered and said no. That is a different thing
                # from a dropped socket and gets the venue's own reason and a
                # remedy, not a Python class name -- an operator reading
                # "RuntimeError" on a trading terminal learns nothing except
                # that something inside broke.
                self._sockets.pop(asset_class, None)
                self._live[asset_class] = False
                self.connected = any(self._live.values())
                self.errors += 1
                self.last_error = exc.operator_text()
                self.last_remedy = exc.remedy
                self.telemetry.event(
                    Level.ERROR, "feed",
                    f"{asset_class.value} data refused: {exc.cause}",
                    detail=exc.remedy or exc.operator_text())
                if exc.code == 406:
                    # Asking again in a second cannot work: the one connection
                    # this account is allowed is held by something else.
                    backoff = CONNECTION_LIMIT_BACKOFF
                    self._report_self_contention(asset_class)
            except Exception as exc:
                # A dropped socket is an event, not an exception. It must never
                # kill this loop or reach the UI as a traceback.
                self._sockets.pop(asset_class, None)
                self._live[asset_class] = False
                self.connected = any(self._live.values())
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.last_remedy = ""
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

    def _report_self_contention(self, refused: AssetClass) -> None:
        """Say so when the thing holding the connection is us.

        Alpaca serves equities and crypto from different endpoints, so this
        program opens a socket to each. Accounts are limited in how many
        market data connections they may hold at once, and where that limit is
        one, the second of our own two sockets is refused by the first.

        That case is worth separating from every other 406 because the remedy
        is completely different. "Close the other copy of IMPERIUM" is useless
        advice when there is no other copy -- the operator goes looking for a
        process that does not exist, which is the same wild goose chase the
        old guessed message sent them on, in a new costume. Detected with
        certainty rather than inferred: a stream of ours is live at the moment
        another of ours is refused.
        """
        holder = next((cls.value for cls, live in self._live.items()
                       if live and cls is not refused), "")
        if not holder:
            return
        self.last_error = (
            f"this Alpaca account allows one market data connection at a "
            f"time, and IMPERIUM's own {holder} stream is using it, so the "
            f"{refused.value} stream was refused")
        self.last_remedy = (
            f"Nothing else on this machine is at fault -- do not go looking "
            f"for a second copy. The {holder} stream is live and those "
            f"symbols are priced normally; {refused.value} symbols keep their "
            f"daily bars and the once-a-minute snapshot sweep, so the "
            f"daily-bar strategies still trade them and only the intraday "
            f"one cannot. To stream {refused.value} instead, scan only that "
            f"asset class -- or ask Alpaca about a plan with more than one "
            f"concurrent connection.")
        self.telemetry.event(
            Level.WARN, "feed",
            f"the {refused.value} stream lost the account's single data "
            f"connection to the {holder} stream",
            detail=self.last_remedy)

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
                    raise FeedRejected(msg.get("code"),
                                       str(msg.get("msg", "")))
                if msg_type == "success" and msg.get("msg") == "authenticated":
                    # Only as many as the plan allows. Alpaca rejects an
                    # over-limit subscription *whole* -- the response is error
                    # 405 and the previous subscriptions are left untouched,
                    # which for a fresh connection means no subscriptions at
                    # all. The terminal would then run with a connected socket
                    # and no data, which looks like a quiet market.
                    wanted = symbols[:self.symbol_limit]
                    self._subscribed[asset_class] = list(wanted)
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "bars": wanted,
                        "quotes": wanted,
                    }))
                    return
        raise FeedRejected(404, "no answer to the auth handshake")

    def _handle_stream_error(self, msg: dict[str, Any]) -> None:
        """Act on an error the stream sends after the subscription.

        These arrive with no symbol attached, and were previously dropped by
        the symbol filter -- so the one message that explains why no data is
        arriving was the one message thrown away.

        Code 405 is "symbol limit exceeded": the request asked for more symbols
        than the plan allows and was rejected in full. The cap is halved and
        the socket closed so the loop reconnects with a subscription the plan
        will accept. Halving rather than guessing, because the real limit
        depends on a subscription this program has no way to read.
        """
        code = msg.get("code")
        text = str(msg.get("msg", "unknown error"))
        self.errors += 1
        self.last_error = f"stream error {code}: {text}"

        if code == 405:
            previous = self.symbol_limit
            largest = max((len(v) for v in self._subscribed.values()),
                          default=self.symbol_limit)
            self.symbol_limit = max(MIN_STREAM_SYMBOLS, largest // 2)
            self.dropped = max(0, len(self.symbols) - self.symbol_limit)
            self.telemetry.event(
                Level.WARN, "feed",
                f"the data plan refused {previous} concurrent symbols; "
                f"retrying with {self.symbol_limit}",
                detail=(f"{text}. The symbols above the cap are still scanned "
                        f"and still priced by the snapshot sweep — they lose "
                        f"only the live minute stream, so the intraday "
                        f"strategy cannot trade them while the daily-bar "
                        f"strategies still can."))
            self._resubscribe.set()
            return

        rejection = FeedRejected(code, text)
        self.last_error = rejection.operator_text()
        self.last_remedy = rejection.remedy
        self.telemetry.event(Level.WARN, "feed",
                             f"the data stream reported an error: "
                             f"{rejection.cause}",
                             detail=rejection.remedy or f"code {code}")

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
            if kind == "error":
                self._handle_stream_error(msg)
                continue
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
