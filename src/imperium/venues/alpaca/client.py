"""A direct Alpaca REST client.

Deliberately not built on a multi-venue abstraction, for the same reason as
before: a library that covers a hundred venues flattens each venue's error
vocabulary into a common one, and that flattening discards exactly what the
operator needs.

Alpaca differs from a crypto exchange in ways that change the client, not just
its URLs:

* **No request signing.** Authentication is two headers. There is no HMAC, no
  timestamp and no recvWindow, so the entire class of signature failures simply
  does not exist here. What replaces it is an environment mismatch: a paper key
  against the live host, or the reverse, which returns 401 and looks exactly
  like a bad key.
* **The market is usually closed.** Equities trade 6.5 hours on weekdays. A bot
  that does not know this will queue orders into a void and read a stale last
  price as a live one, so the clock is a first-class part of this client.
* **Assets carry their own tradability.** Whether a symbol can be shorted, is
  easy to borrow, is fractionable, or is halted are per-asset facts the venue
  publishes; they are not global venue properties.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import random
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

import httpx

from imperium.venues.alpaca import errors
from imperium.venues.assets import AssetClass, classify_symbol

log = logging.getLogger("imperium.alpaca")

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

#: Alpaca's basic plan allows 200 requests a minute per key.
RATE_LIMIT_PER_MINUTE = 200


class VenueError(Exception):
    """A venue-level failure carrying an operator-facing remedy."""

    def __init__(self, message: str, *, code: int | None = None, remedy: str = "",
                 status: int | None = None, retryable: bool = False,
                 ambiguous: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.remedy = remedy
        self.status = status
        self.retryable = retryable
        #: True when we do not know whether the request took effect.
        self.ambiguous = ambiguous

    def operator_text(self) -> str:
        return " — ".join(p for p in (self.message, self.remedy) if p)

    def __str__(self) -> str:
        return self.operator_text()


def _looks_like_html(body: str) -> bool:
    """True if this reply never came from the venue.

    Alpaca answers in JSON. An HTML body means an edge proxy, a corporate
    filter or a captive portal answered instead, and printing a page of markup
    into a status panel as if it were a venue message wastes an hour.
    """
    head = body.lstrip()[:512].lower()
    return head.startswith(("<!doctype", "<html", "<?xml")) or "<title>" in head


@dataclass
class MarketClock:
    """The venue's own view of whether the market is open.

    Believing the venue rather than computing it locally is not laziness: it is
    the only way to get early closes, holidays and the occasional unscheduled
    halt right, and every one of those is a day a naive calendar trades into a
    closed market.
    """

    is_open: bool = False
    next_open: dt.datetime | None = None
    next_close: dt.datetime | None = None
    fetched_at: float = 0.0
    timestamp: dt.datetime | None = None

    @property
    def age(self) -> float:
        return time.time() - self.fetched_at if self.fetched_at else float("inf")

    def describe(self) -> str:
        if self.is_open:
            if self.next_close:
                return f"market open, closes {self.next_close:%H:%M UTC}"
            return "market open"
        if self.next_open:
            return f"market closed, opens {self.next_open:%a %H:%M UTC}"
        return "market closed"


@dataclass
class RateBudget:
    """Tracks the venue's own view of the request budget."""

    remaining: int = RATE_LIMIT_PER_MINUTE
    limit: int = RATE_LIMIT_PER_MINUTE
    reset_at: float = 0.0
    retry_after: float = 0.0

    def observe(self, headers: Mapping[str, str]) -> None:
        for key, value in headers.items():
            lower = key.lower()
            try:
                if lower == "x-ratelimit-remaining":
                    self.remaining = int(value)
                elif lower == "x-ratelimit-limit":
                    self.limit = int(value)
                elif lower == "x-ratelimit-reset":
                    self.reset_at = float(value)
                elif lower == "retry-after":
                    self.retry_after = time.monotonic() + float(value)
            except (TypeError, ValueError):
                continue

    @property
    def utilisation(self) -> float:
        if not self.limit:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.remaining / self.limit)))

    def pause_needed(self) -> float:
        now = time.monotonic()
        if self.retry_after > now:
            return self.retry_after - now
        if self.utilisation > 0.85:
            return min(2.0, (self.utilisation - 0.85) * 10.0)
        return 0.0


@dataclass(frozen=True)
class Asset:
    """One tradable instrument, as the venue describes it.

    Tradability is per-asset and changes: a symbol can be halted, become
    hard-to-borrow, or stop being fractionable. Caching this for a session and
    trusting it forever is how an order gets sent into a halt.
    """

    symbol: str
    name: str
    asset_class: AssetClass
    exchange: str
    tradable: bool
    shortable: bool
    easy_to_borrow: bool
    fractionable: bool
    status: str
    min_order_size: Decimal | None = None
    min_trade_increment: Decimal | None = None
    price_increment: Decimal | None = None

    @property
    def can_short(self) -> bool:
        """Shorting needs both permission and available borrow.

        ``shortable`` alone is not enough: a hard-to-borrow name will accept the
        order and then fail to locate, so both are required.
        """
        return bool(self.shortable and self.easy_to_borrow)


def parse_asset(payload: dict[str, Any]) -> Asset:
    def dec(key: str) -> Decimal | None:
        raw = payload.get(key)
        if raw in (None, ""):
            return None
        try:
            return Decimal(str(raw))
        except Exception:
            return None

    symbol = payload["symbol"]
    raw_class = payload.get("class") or payload.get("asset_class") or ""
    try:
        asset_class = AssetClass(raw_class)
    except ValueError:
        asset_class = classify_symbol(symbol)
    return Asset(
        symbol=symbol,
        name=payload.get("name", ""),
        asset_class=asset_class,
        exchange=payload.get("exchange", ""),
        tradable=bool(payload.get("tradable", False)),
        shortable=bool(payload.get("shortable", False)),
        easy_to_borrow=bool(payload.get("easy_to_borrow", False)),
        fractionable=bool(payload.get("fractionable", False)),
        status=payload.get("status", "unknown"),
        min_order_size=dec("min_order_size"),
        min_trade_increment=dec("min_trade_increment"),
        price_increment=dec("price_increment"),
    )


class AlpacaClient:
    """Thin, direct client for Alpaca's trading and market-data APIs."""

    venue_id = "alpaca"

    def __init__(
        self,
        api_key: str | None = None,
        secret: str | None = None,
        *,
        paper: bool = True,
        base_url: str | None = None,
        data_url: str = DATA_BASE_URL,
        timeout: float = 20.0,
        trust_env: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
        feed: str = "iex",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self._secret = (secret or "").strip()
        self.paper = paper
        self.base_url = (base_url or (PAPER_BASE_URL if paper else LIVE_BASE_URL)).rstrip("/")
        self.data_url = data_url.rstrip("/")
        #: 'iex' is the free feed and covers a fraction of consolidated volume;
        #: 'sip' is the paid full-market feed. Which one is in use changes what
        #: the prices mean, so it is reported rather than assumed.
        self.feed = feed
        self.max_retries = max_retries
        self.budget = RateBudget()
        self.clock = MarketClock()
        self._assets: dict[str, Asset] = {}
        self._assets_fetched_at = 0.0
        self._assets_lock = asyncio.Lock()

        headers = {
            "User-Agent": "imperium/0.1 (+direct-rest)",
            "Accept": "application/json",
        }
        if self.api_key and self._secret:
            headers["APCA-API-KEY-ID"] = self.api_key
            headers["APCA-API-SECRET-KEY"] = self._secret

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            # trust_env picks up HTTPS_PROXY. Without it this program simply
            # does not work on a corporate network.
            trust_env=trust_env,
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16,
                                keepalive_expiry=90.0),
            headers=headers,
            transport=transport,
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AlpacaClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    @property
    def authenticated(self) -> bool:
        return bool(self.api_key and self._secret)

    @property
    def environment(self) -> str:
        return "paper" if self.paper else "live"

    # -- transport -------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        data_api: bool = False,
        needs_auth: bool = True,
        idempotent: bool = True,
        retries: int | None = None,
    ) -> Any:
        """One venue request, with retries only where they are safe."""
        if needs_auth and not self.authenticated:
            raise VenueError(
                "this request needs an API key and none is configured",
                remedy="add an Alpaca key on the Connections panel, then retry",
            )
        attempts = self.max_retries if retries is None else retries
        root = self.data_url if data_api else self.base_url
        url = f"{root}{path}"
        last_error: VenueError | None = None

        for attempt in range(attempts + 1):
            pause = self.budget.pause_needed()
            if pause > 0:
                await asyncio.sleep(pause)
            try:
                response = await self._client.request(
                    method, url, params=dict(params or {}) or None, json=json_body,
                )
            except httpx.ProxyError:
                last_error = VenueError(
                    "the HTTP proxy refused this request",
                    remedy=("A proxy is configured via HTTPS_PROXY and it rejected "
                            "the connection. Run /diagnose to see which layer fails."),
                    retryable=False, ambiguous=not idempotent,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout):
                last_error = VenueError(
                    f"could not open a connection to {root}",
                    remedy=("Run /diagnose -- it separates DNS, TLS, proxy and "
                            "geo-block, which all produce this same symptom."),
                    retryable=True, ambiguous=False,
                )
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                last_error = VenueError(
                    f"the venue did not answer within the timeout "
                    f"({type(exc).__name__})",
                    remedy="The request may or may not have been executed.",
                    retryable=idempotent, ambiguous=not idempotent,
                )
            except httpx.HTTPError as exc:
                last_error = VenueError(
                    f"transport failure talking to the venue: {type(exc).__name__}",
                    remedy="Run /diagnose to identify which network layer fails.",
                    retryable=True, ambiguous=not idempotent,
                )
            else:
                self.budget.observe(response.headers)
                result = self._interpret(response, idempotent=idempotent)
                if isinstance(result, VenueError):
                    last_error = result
                    if not result.retryable:
                        raise result
                else:
                    return result

            if attempt >= attempts or (last_error and not last_error.retryable):
                break
            delay = min(8.0, 0.5 * (2 ** attempt)) * (0.5 + random.random())
            await asyncio.sleep(delay)

        assert last_error is not None
        raise last_error

    def _interpret(self, response: httpx.Response, *, idempotent: bool) -> Any:
        text = response.text

        if _looks_like_html(text):
            return VenueError(
                "the reply was an HTML page, so the request never reached the venue",
                remedy=("Alpaca always answers in JSON. An HTML body means an edge "
                        "proxy, a corporate web filter or a captive portal answered "
                        "instead. Run /diagnose."),
                status=response.status_code, retryable=False,
                ambiguous=not idempotent,
            )

        try:
            payload = response.json() if text.strip() else None
        except ValueError:
            return VenueError(
                f"the venue returned a body that is not JSON "
                f"(HTTP {response.status_code})",
                remedy="This is not a venue message; something in between answered.",
                status=response.status_code,
                retryable=response.status_code >= 500,
                ambiguous=not idempotent,
            )

        if 200 <= response.status_code < 300:
            return payload

        code = None
        message = ""
        if isinstance(payload, dict):
            code = payload.get("code")
            message = str(payload.get("message", "") or payload.get("msg", ""))

        remedy = errors.lookup(code, response.status_code)
        retryable = errors.is_retryable(code, response.status_code)

        if response.status_code == 401:
            # The most common real cause, and invisible from the message alone.
            hint = (f"This client is pointed at the {self.environment} endpoint "
                    f"({self.base_url}). A key from the other environment returns "
                    f"exactly this.")
            return VenueError(
                "the API key was not accepted",
                code=int(code) if code is not None else None,
                remedy=((remedy.remedy + " ") if remedy else "") + hint,
                status=401, retryable=False,
            )

        return VenueError(
            (remedy.meaning if remedy else (message or
                                            f"venue error {response.status_code}")),
            code=int(code) if code is not None else None,
            remedy=(remedy.remedy if remedy else message),
            status=response.status_code,
            retryable=retryable,
            ambiguous=False,      # the venue answered; nothing is ambiguous
        )

    # -- account and session ---------------------------------------------

    async def account(self) -> dict[str, Any]:
        return await self._request("GET", "/v2/account")

    async def get_clock(self) -> MarketClock:
        """Read the venue's market clock.

        Unauthenticated callers cannot reach this, so a session with no key
        keeps the last known clock rather than pretending the market is open.
        """
        payload = await self._request("GET", "/v2/clock")

        def when(key: str) -> dt.datetime | None:
            """Parse a venue timestamp, always as an aware UTC datetime.

            fromisoformat returns a *naive* datetime for a string carrying no
            offset. Alpaca documents an offset, but a naive value escaping this
            function is not a small inaccuracy: every consumer compares it
            against an aware now(), and that comparison raises TypeError rather
            than returning a wrong answer. Inside the trading loop the exception
            is caught and logged, the session phase never advances, and the
            overnight strategy silently never runs. A bare timestamp is read as
            UTC, which is what the venue serves.
            """
            raw = payload.get(key)
            if not raw:
                return None
            try:
                parsed = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)

        self.clock = MarketClock(
            is_open=bool(payload.get("is_open", False)),
            next_open=when("next_open"),
            next_close=when("next_close"),
            timestamp=when("timestamp"),
            fetched_at=time.time(),
        )
        return self.clock

    async def positions(self) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/v2/positions")
        return payload if isinstance(payload, list) else []

    # -- assets ----------------------------------------------------------

    async def assets(self, force: bool = False,
                     status: str = "active") -> dict[str, Asset]:
        """Fetch and cache the tradable universe.

        Cached for an hour: listings change rarely, and this is a large
        response. Halts are caught by the per-symbol check before an order,
        not by re-downloading the whole list.
        """
        async with self._assets_lock:
            fresh = time.monotonic() - self._assets_fetched_at < 3600
            if self._assets and fresh and not force:
                return self._assets
            out: dict[str, Asset] = {}
            for asset_class in ("us_equity", "crypto"):
                try:
                    payload = await self._request(
                        "GET", "/v2/assets",
                        params={"status": status, "asset_class": asset_class},
                    )
                except VenueError as exc:
                    log.warning("could not list %s assets: %s", asset_class,
                                exc.message)
                    continue
                for entry in payload or []:
                    try:
                        asset = parse_asset(entry)
                    except (KeyError, ValueError):
                        continue
                    out[asset.symbol] = asset
            if out:
                self._assets = out
                self._assets_fetched_at = time.monotonic()
            return self._assets

    async def asset(self, symbol: str) -> Asset:
        cache = await self.assets()
        found = cache.get(symbol)
        if found is None:
            raise VenueError(
                f"{symbol} is not a tradable asset on this account",
                status=404,
                remedy="Check the spelling for this venue: Alpaca writes crypto "
                       "pairs with a slash, such as BTC/USD.",
            )
        return found

    # -- market data ------------------------------------------------------

    def _data_path(self, asset_class: AssetClass, suffix: str) -> str:
        """Alpaca serves equities and crypto from different data namespaces."""
        if asset_class is AssetClass.CRYPTO:
            return f"/v1beta3/crypto/us{suffix}"
        return f"/v2/stocks{suffix}"

    async def bars(self, symbols: list[str], *, timeframe: str = "1Min",
                   limit: int = 1000,
                   start: dt.datetime | None = None) -> dict[str, list[dict[str, Any]]]:
        """Historical bars, batched by asset class.

        Equities and crypto come from different endpoints with different query
        shapes, so a mixed universe is split rather than sent as one request.
        """
        grouped: dict[AssetClass, list[str]] = {}
        for symbol in symbols:
            grouped.setdefault(classify_symbol(symbol), []).append(symbol)

        out: dict[str, list[dict[str, Any]]] = {}
        for asset_class, group in grouped.items():
            if asset_class is AssetClass.US_OPTION:
                continue
            params: dict[str, Any] = {
                "symbols": ",".join(group),
                "timeframe": timeframe,
                "limit": limit,
            }
            if asset_class is AssetClass.US_EQUITY:
                params["feed"] = self.feed
                # Free plans cannot read the most recent 15 minutes of SIP data;
                # asking for it returns an error rather than an empty result.
                params["adjustment"] = "raw"
            if start is not None:
                params["start"] = start.astimezone(dt.timezone.utc).isoformat()
            try:
                payload = await self._request(
                    "GET", self._data_path(asset_class, "/bars"),
                    params=params, data_api=True,
                )
            except VenueError as exc:
                log.warning("bars for %d %s symbols failed: %s", len(group),
                            asset_class.value, exc.message)
                continue
            for symbol, rows in (payload or {}).get("bars", {}).items():
                out[symbol] = rows or []
        return out

    async def snapshots(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Latest trade, quote and daily bar per symbol -- the scanner's input."""
        grouped: dict[AssetClass, list[str]] = {}
        for symbol in symbols:
            grouped.setdefault(classify_symbol(symbol), []).append(symbol)

        out: dict[str, dict[str, Any]] = {}
        for asset_class, group in grouped.items():
            if asset_class is AssetClass.US_OPTION:
                continue
            params: dict[str, Any] = {"symbols": ",".join(group)}
            if asset_class is AssetClass.US_EQUITY:
                params["feed"] = self.feed
            try:
                payload = await self._request(
                    "GET", self._data_path(asset_class, "/snapshots"),
                    params=params, data_api=True,
                )
            except VenueError as exc:
                log.warning("snapshots for %d %s symbols failed: %s", len(group),
                            asset_class.value, exc.message)
                continue
            # Equities return the map at the top level; crypto nests it under
            # "snapshots". Handling only one shape silently yields no prices.
            block = payload.get("snapshots", payload) if isinstance(payload, dict) else {}
            for symbol, snap in (block or {}).items():
                if isinstance(snap, dict):
                    out[symbol] = snap
        return out

    # -- orders -----------------------------------------------------------

    @staticmethod
    def new_client_order_id(tag: str = "imp") -> str:
        """A client-side order identity, generated before the request is sent.

        Without it there is no way to ask the venue whether a timed-out
        submission landed.
        """
        import uuid

        return f"{tag}-{uuid.uuid4().hex[:24]}"

    async def open_orders(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"status": "open", "limit": 500}
        if symbols:
            params["symbols"] = ",".join(symbols)
        payload = await self._request("GET", "/v2/orders", params=params)
        return payload if isinstance(payload, list) else []

    async def get_order_by_client_id(self, client_order_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", "/v2/orders:by_client_order_id",
            params={"client_order_id": client_order_id},
        )

    async def cancel_order(self, order_id: str) -> None:
        await self._request("DELETE", f"/v2/orders/{order_id}",
                            idempotent=False, retries=0)

    async def place_order(
        self,
        symbol: str,
        side: str,
        *,
        qty: Decimal | float | str | None = None,
        notional: Decimal | float | str | None = None,
        order_type: str = "market",
        time_in_force: str | None = None,
        limit_price: Decimal | float | str | None = None,
        client_order_id: str | None = None,
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        """Submit an order, resolving an ambiguous submission rather than guessing.

        Two rules are load-bearing, unchanged from the previous venue because
        they are properties of trading rather than of an API:

        * a rejected order is never retried -- it was wrong, and re-sending
          sends the same wrong order;
        * an ambiguous submission is resolved by looking the order up by the
          client order ID generated before sending, because reporting "failed"
          for an order that actually filled is how a bot ends up flat in its own
          records and long at the venue.
        """
        if (qty is None) == (notional is None):
            raise ValueError("exactly one of qty or notional is required")

        asset_class = classify_symbol(symbol)
        coid = client_order_id or self.new_client_order_id()
        if time_in_force is None:
            # Crypto is continuous and accepts gtc; a day order on a 24/7 market
            # expires at a boundary that means nothing there.
            time_in_force = "gtc" if asset_class is AssetClass.CRYPTO else "day"

        body: dict[str, Any] = {
            "symbol": symbol,
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": time_in_force,
            "client_order_id": coid,
        }
        if qty is not None:
            body["qty"] = str(qty)
        else:
            body["notional"] = str(notional)
        if limit_price is not None:
            body["limit_price"] = str(limit_price)
        if extended_hours:
            # Only a limit day order may run in extended hours; sending it on a
            # market order is rejected.
            body["extended_hours"] = True

        try:
            return await self._request("POST", "/v2/orders", json_body=body,
                                       idempotent=False, retries=0)
        except VenueError as exc:
            if not exc.ambiguous:
                raise
            log.warning("order submission for %s was ambiguous (%s); looking it "
                        "up by client order id", symbol, exc.message)
            resolved = await self._resolve_ambiguous(coid)
            if resolved is not None:
                log.warning("the ambiguous order %s did land, status %s", coid,
                            resolved.get("status"))
                return resolved
            exc.remedy = ((exc.remedy + " ") if exc.remedy else "") + (
                f"Checked: no order with client id {coid} exists at the venue, "
                "so the submission did not land.")
            raise

    async def _resolve_ambiguous(self, client_order_id: str,
                                 attempts: int = 3) -> dict[str, Any] | None:
        for attempt in range(attempts):
            try:
                return await self.get_order_by_client_id(client_order_id)
            except VenueError as exc:
                if exc.status == 404:
                    return None          # a definitive "no such order"
                if attempt == attempts - 1:
                    raise VenueError(
                        "an order submission could not be confirmed and the "
                        "follow-up lookup also failed",
                        remedy=(f"Check the venue manually for client order id "
                                f"{client_order_id} before trading this symbol "
                                f"again."),
                        ambiguous=True,
                    ) from exc
                await asyncio.sleep(0.5 * (attempt + 1))
        return None

    async def close_position(self, symbol: str) -> dict[str, Any]:
        """Flatten one symbol at the venue.

        Used on retirement and on a mode switch, where "reduce to zero" must not
        depend on the book's own idea of the quantity held.
        """
        return await self._request("DELETE", f"/v2/positions/{symbol}",
                                   idempotent=False, retries=0)
