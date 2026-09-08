"""A direct Binance Spot REST client.

Deliberately *not* built on a multi-venue abstraction layer. A library that
covers a hundred venues has to flatten each venue's error vocabulary into a
common one, and that flattening discards exactly the information an operator
needs: whether `-2015` was the IP allow-list or the trading permission. This
module is smaller than the wrapper such a library would need anyway.

The things that actually cost time, all in one place:

**Signing.** The HMAC is computed over the exact byte string that is sent. The
query string is built once, signed as a string, and the signature appended to
*that same string*. Passing a dict to the HTTP client after signing lets it
re-encode -- a different escaping of ``+`` or a different key order -- and the
result is `-1022` with no indication which of the two strings was wrong.

**Connection reuse.** One `AsyncClient` with keep-alive, so signed requests
share a source address. Some venues bind a key to an IP; a fresh connection per
request can egress from a different address on a multi-homed host or behind a
proxy pool, and the resulting failure is intermittent, which is the worst kind.

**trust_env=True.** So `HTTPS_PROXY` is honoured. On a corporate network this is
the difference between working and not working.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import random
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

import httpx

from imperium.venues.binance import errors
from imperium.venues.binance.filters import (
    SymbolFilters,
    format_decimal,
    parse_symbol,
    to_decimal,
)

log = logging.getLogger("imperium.binance")

DEFAULT_BASE_URL = "https://api.binance.com"
#: Public market data mirror. Serves unauthenticated endpoints and is sometimes
#: reachable where the main host is geo-blocked, which is useful for telling
#: "blocked" apart from "down".
DATA_BASE_URL = "https://data-api.binance.vision"

RECV_WINDOW_MS = 5000
#: Binance's spot REST weight budget per minute per IP.
WEIGHT_LIMIT_PER_MINUTE = 6000


class VenueError(Exception):
    """A venue-level failure, carrying an operator-facing remedy.

    Every failure this client can produce is one of these. Callers never see a
    raw ``httpx`` exception, because "ConnectError" is not something an operator
    can act on.
    """

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        remedy: str = "",
        status: int | None = None,
        retryable: bool = False,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.remedy = remedy
        self.status = status
        self.retryable = retryable
        #: True when we do not know whether the request took effect at the
        #: venue. Only ever set for state-changing requests.
        self.ambiguous = ambiguous

    def operator_text(self) -> str:
        parts = [self.message]
        if self.remedy:
            parts.append(self.remedy)
        return " — ".join(parts)

    def __str__(self) -> str:
        return self.operator_text()


def _looks_like_html(body: str) -> bool:
    """True if this response never came from the venue.

    Binance answers in JSON, always. An HTML body means something between here
    and the venue answered instead: an edge proxy, a corporate filter, a captive
    portal, or a regional block page. Printing a page of nginx markup into a
    status panel as if it were a venue message is how an operator spends an hour
    debugging the wrong machine.
    """
    head = body.lstrip()[:512].lower()
    return head.startswith(("<!doctype", "<html", "<?xml")) or "<title>" in head


@dataclass
class RateBudget:
    """Tracks the venue's own view of how much weight has been spent.

    Binance reports used weight in a response header. Believing that header is
    strictly better than counting locally, because it survives restarts, other
    processes on the same IP, and our own miscounting of an endpoint's weight.
    """

    used_weight: int = 0
    limit: int = WEIGHT_LIMIT_PER_MINUTE
    order_count_10s: int = 0
    updated_at: float = 0.0
    retry_after: float = 0.0

    def observe(self, headers: Mapping[str, str]) -> None:
        self.updated_at = time.monotonic()
        for key, value in headers.items():
            lower = key.lower()
            if lower.startswith("x-mbx-used-weight-"):
                try:
                    self.used_weight = int(value)
                except ValueError:
                    pass
            elif lower.startswith("x-mbx-order-count-10s"):
                try:
                    self.order_count_10s = int(value)
                except ValueError:
                    pass
            elif lower == "retry-after":
                try:
                    self.retry_after = time.monotonic() + float(value)
                except ValueError:
                    pass

    @property
    def utilisation(self) -> float:
        return min(1.0, self.used_weight / self.limit) if self.limit else 0.0

    def pause_needed(self) -> float:
        """Seconds to wait before the next request.

        Slowing down at 80% of the budget rather than at 100% matters because a
        `-1003` ban lengthens on repetition: the cost of being early is a few
        milliseconds, the cost of being late is minutes of being locked out.
        """
        now = time.monotonic()
        if self.retry_after > now:
            return self.retry_after - now
        if self.utilisation > 0.80:
            return min(2.0, (self.utilisation - 0.80) * 10.0)
        return 0.0


class BinanceSpotClient:
    """Thin, direct client for Binance Spot.

    Constructed without credentials it can still serve every public endpoint,
    which is what lets the watchlist show live prices before any key exists.
    """

    venue_id = "binance_spot"

    def __init__(
        self,
        api_key: str | None = None,
        secret: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 15.0,
        trust_env: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self._secret = (secret or "").strip().encode()
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.budget = RateBudget()
        #: Difference between the venue's clock and ours, in milliseconds.
        #: Measured, not assumed -- see :meth:`sync_time`.
        self.time_offset_ms = 0
        self.time_offset_measured = False
        self._filters: dict[str, SymbolFilters] = {}
        self._filters_fetched_at = 0.0
        self._filters_lock = asyncio.Lock()
        headers = {
            "User-Agent": "imperium/0.1 (+direct-rest)",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            # trust_env picks up HTTPS_PROXY. Without it this program simply does
            # not work on a corporate network, and the failure looks like the
            # venue being down.
            trust_env=trust_env,
            # Keep-alive pooling: signed requests share one source address.
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16,
                                keepalive_expiry=90.0),
            headers=headers,
            transport=transport,
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BinanceSpotClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    @property
    def authenticated(self) -> bool:
        return bool(self.api_key and self._secret)

    # -- signing ---------------------------------------------------------

    def _encode(self, params: Mapping[str, Any]) -> str:
        """Encode params to the exact query string that will be sent.

        ``quote_via=quote`` rather than the default ``quote_plus``: a literal
        ``+`` in a value must arrive as ``%2B``, and a space must not become
        ``+`` on one side of the signature and ``%20`` on the other.
        """
        items = [(k, v) for k, v in params.items() if v is not None]
        rendered = [
            (k, format_decimal(v) if isinstance(v, (Decimal, float)) else str(v))
            for k, v in items
        ]
        return urllib.parse.urlencode(rendered, quote_via=urllib.parse.quote,
                                      safe="")

    def _sign(self, query: str) -> str:
        """HMAC-SHA256 over the exact string that goes on the wire."""
        return hmac.new(self._secret, query.encode(), hashlib.sha256).hexdigest()

    def _signed_query(self, params: Mapping[str, Any]) -> str:
        if not self.authenticated:
            raise VenueError(
                "this request needs an API key and none is configured",
                remedy="add a key on the Connections panel, then retry",
            )
        merged = dict(params)
        merged.setdefault("recvWindow", RECV_WINDOW_MS)
        merged["timestamp"] = self._timestamp_ms()
        query = self._encode(merged)
        # Sign *this* string and append to *this* string. Re-encoding the params
        # after signing is the single most common cause of -1022.
        return f"{query}&signature={self._sign(query)}"

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000) + self.time_offset_ms

    # -- transport -------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        signed: bool = False,
        weight: int = 1,
        idempotent: bool = True,
        retries: int | None = None,
    ) -> Any:
        """Perform one venue request, with retries only where they are safe.

        ``idempotent=False`` marks a state-changing request. Those are never
        retried on a transport failure, because a timeout does not tell us
        whether the order landed -- see :meth:`place_order`, which resolves the
        ambiguity by looking the order up instead of guessing.
        """
        attempts = self.max_retries if retries is None else retries
        params = dict(params or {})
        last_error: VenueError | None = None

        for attempt in range(attempts + 1):
            pause = self.budget.pause_needed()
            if pause > 0:
                await asyncio.sleep(pause)

            if signed:
                query = self._signed_query(params)
            else:
                query = self._encode(params)
            url = f"{path}?{query}" if query else path

            try:
                response = await self._client.request(method, url)
            except httpx.ProxyError as exc:
                last_error = VenueError(
                    "the HTTP proxy refused this request",
                    remedy=("A proxy is configured via HTTPS_PROXY and it rejected "
                            "the connection to the venue. Run /diagnose to see which "
                            "layer fails."),
                    retryable=False, ambiguous=not idempotent,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = VenueError(
                    f"could not open a connection to {self.base_url}",
                    remedy=("Run /diagnose -- it separates DNS, TLS, proxy and "
                            "geo-block, which all produce this same symptom."),
                    retryable=True, ambiguous=False,
                )
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                # A read timeout on a state-changing request is the ambiguous
                # case: the request may well have been executed.
                last_error = VenueError(
                    f"the venue did not answer within the timeout ({type(exc).__name__})",
                    remedy="The request may or may not have been executed at the venue.",
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
            # Exponential backoff with jitter. Without the jitter, N engines that
            # failed on the same venue blip retry in lockstep and reproduce it.
            delay = min(8.0, 0.5 * (2 ** attempt)) * (0.5 + random.random())
            await asyncio.sleep(delay)

        assert last_error is not None
        raise last_error

    def _interpret(self, response: httpx.Response, *, idempotent: bool) -> Any:
        """Turn an HTTP response into data or a VenueError."""
        text = response.text

        if _looks_like_html(text):
            return VenueError(
                "the reply was an HTML page, so the request never reached the venue",
                remedy=("Binance always answers in JSON. An HTML body means an edge "
                        "proxy, a corporate web filter, or a regional block page "
                        "answered instead. Run /diagnose."),
                status=response.status_code, retryable=False,
                ambiguous=not idempotent,
            )

        try:
            payload = response.json() if text else None
        except ValueError:
            return VenueError(
                f"the venue returned a body that is not JSON (HTTP {response.status_code})",
                remedy="This is not a venue message; something in between answered.",
                status=response.status_code, retryable=response.status_code >= 500,
                ambiguous=not idempotent,
            )

        if response.status_code == 200:
            return payload

        code = None
        message = ""
        if isinstance(payload, dict):
            code = payload.get("code")
            message = str(payload.get("msg", ""))

        if response.status_code in (418, 429):
            # 418 is Binance's "you ignored a 429 and are now banned".
            self.budget.retry_after = time.monotonic() + 30.0
            return VenueError(
                "the venue is rate limiting this IP" if response.status_code == 429
                else "this IP has been temporarily banned by the venue for "
                     "ignoring rate limits",
                code=code, remedy=errors.describe(code, message) if code else
                "Back off and reduce request rate.",
                status=response.status_code, retryable=True,
            )

        remedy = errors.lookup(code)
        retryable = errors.is_retryable(code) if code is not None else (
            response.status_code >= 500
        )
        if code is not None:
            r = errors.lookup(code)
            return VenueError(
                (r.meaning if r else (message or f"venue error {code}")),
                code=int(code),
                remedy=(r.remedy if r else message),
                status=response.status_code,
                retryable=retryable,
                ambiguous=False,  # the venue answered; nothing is ambiguous
            )
        return VenueError(
            f"the venue returned HTTP {response.status_code}",
            remedy=message or "No venue error code accompanied this response.",
            status=response.status_code, retryable=retryable,
            ambiguous=not idempotent and response.status_code >= 500,
        )

    # -- public endpoints ------------------------------------------------

    async def ping(self) -> bool:
        await self._request("GET", "/api/v3/ping")
        return True

    async def server_time_ms(self) -> int:
        payload = await self._request("GET", "/api/v3/time")
        return int(payload["serverTime"])

    async def sync_time(self) -> int:
        """Measure the offset between this machine's clock and the venue's.

        Half the round trip is subtracted so that network latency is not counted
        as clock drift. The offset is *applied* to outgoing timestamps and also
        *reported*, because a machine that needs a 40-second correction has a
        broken clock that will break other things too.
        """
        t0 = time.time()
        payload = await self._request("GET", "/api/v3/time")
        t1 = time.time()
        server_ms = int(payload["serverTime"])
        local_mid_ms = (t0 + (t1 - t0) / 2) * 1000
        self.time_offset_ms = int(server_ms - local_mid_ms)
        self.time_offset_measured = True
        if abs(self.time_offset_ms) > 1000:
            log.warning(
                "this machine's clock differs from the venue by %d ms; "
                "signed requests are being corrected, but the clock should be synced",
                self.time_offset_ms,
            )
        return self.time_offset_ms

    async def exchange_info(self, force: bool = False) -> dict[str, SymbolFilters]:
        """Fetch and cache symbol filters.

        Cached for an hour: listings and filter changes are rare, and this is a
        heavy request (weight 20) that would otherwise be made per order.
        """
        async with self._filters_lock:
            fresh = time.monotonic() - self._filters_fetched_at < 3600
            if self._filters and fresh and not force:
                return self._filters
            payload = await self._request("GET", "/api/v3/exchangeInfo", weight=20)
            parsed: dict[str, SymbolFilters] = {}
            for entry in payload.get("symbols", []):
                try:
                    sf = parse_symbol(entry)
                except (KeyError, ValueError):
                    continue
                parsed[sf.symbol] = sf
            self._filters = parsed
            self._filters_fetched_at = time.monotonic()
            return parsed

    async def filters_for(self, symbol: str) -> SymbolFilters:
        info = await self.exchange_info()
        sf = info.get(symbol)
        if sf is None:
            raise VenueError(
                f"{symbol} is not listed on this venue",
                code=-1121,
                remedy=errors.describe(-1121),
            )
        return sf

    async def ticker_24h(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if symbols:
            # The array form must be JSON with no spaces, or the venue rejects it.
            params["symbols"] = "[" + ",".join(f'"{s}"' for s in symbols) + "]"
        payload = await self._request("GET", "/api/v3/ticker/24hr", params=params,
                                      weight=40 if not symbols else 2)
        return payload if isinstance(payload, list) else [payload]

    async def klines(self, symbol: str, interval: str = "1m",
                     limit: int = 500) -> list[list[Any]]:
        return await self._request(
            "GET", "/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            weight=2,
        )

    async def book_ticker(self, symbol: str | None = None) -> Any:
        params = {"symbol": symbol} if symbol else {}
        return await self._request("GET", "/api/v3/ticker/bookTicker", params=params)

    # -- signed endpoints ------------------------------------------------

    async def account(self) -> dict[str, Any]:
        return await self._request("GET", "/api/v3/account", signed=True, weight=20)

    async def balances(self, hide_dust: bool = True) -> list[dict[str, Any]]:
        payload = await self.account()
        out = []
        for b in payload.get("balances", []):
            free = to_decimal(b.get("free", 0))
            locked = to_decimal(b.get("locked", 0))
            if hide_dust and free + locked == 0:
                continue
            out.append({"asset": b["asset"], "free": free, "locked": locked,
                        "total": free + locked})
        out.sort(key=lambda r: r["asset"])
        return out

    async def account_commission(self, symbol: str) -> dict[str, Any] | None:
        """The account's *actual* fee rates for a symbol.

        Worth a request: assuming a fee tier is the difference between a symbol
        that clears the cost gate and one that does not, and an assumed tier is
        reported as a warning precisely because it might be wrong.
        """
        try:
            return await self._request(
                "GET", "/api/v3/account/commission",
                params={"symbol": symbol}, signed=True, weight=20,
            )
        except VenueError as exc:
            log.info("could not read live commission rates for %s: %s", symbol, exc.message)
            return None

    async def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else {}
        return await self._request("GET", "/api/v3/openOrders", params=params,
                                   signed=True, weight=6 if symbol else 80)

    async def get_order(self, symbol: str, *, order_id: int | None = None,
                        client_order_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        elif client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("one of order_id or client_order_id is required")
        return await self._request("GET", "/api/v3/order", params=params,
                                   signed=True, weight=4)

    async def cancel_order(self, symbol: str, *, order_id: int | None = None,
                           client_order_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        elif client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("one of order_id or client_order_id is required")
        return await self._request("DELETE", "/api/v3/order", params=params,
                                   signed=True, weight=1, idempotent=False,
                                   retries=0)

    @staticmethod
    def new_client_order_id(tag: str = "gda") -> str:
        """A client-side order identity, generated before the request is sent.

        This is what makes an ambiguous submission recoverable: without it there
        is no way to ask the venue "did the order I just tried to send land?".
        Binance allows up to 36 characters of [.A-Za-z0-9_-].
        """
        return f"{tag}-{uuid.uuid4().hex[:24]}"

    async def place_order(
        self,
        symbol: str,
        side: str,
        *,
        quantity: Any,
        order_type: str = "MARKET",
        price: Any = None,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
        quote_quantity: Any = None,
    ) -> dict[str, Any]:
        """Submit an order, resolving an ambiguous submission rather than guessing.

        Two rules are load-bearing here:

        * A rejected order is never retried. `-1013`, `-1100`, `-2010` and their
          neighbours mean the order was *wrong*; sending it again sends the same
          wrong order.
        * If the submission is ambiguous -- a timeout, a 5xx -- the order is
          looked up by its client order ID before any conclusion is drawn.
          Reporting "failed" for an order that actually filled is how a bot ends
          up flat in its own records and long at the venue.
        """
        side = side.upper()
        order_type = order_type.upper()
        coid = client_order_id or self.new_client_order_id()

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "newClientOrderId": coid,
            "newOrderRespType": "FULL",
        }
        if quote_quantity is not None:
            params["quoteOrderQty"] = format_decimal(quote_quantity)
        else:
            params["quantity"] = format_decimal(quantity)

        if order_type == "LIMIT_MAKER":
            # Post-only. Sending timeInForce alongside it is -1106; the field is
            # not merely redundant, it is rejected.
            if price is None:
                raise ValueError("LIMIT_MAKER requires a price")
            params["price"] = format_decimal(price)
        elif order_type == "LIMIT":
            if price is None:
                raise ValueError("LIMIT requires a price")
            params["price"] = format_decimal(price)
            params["timeInForce"] = time_in_force or "GTC"
        elif order_type != "MARKET":
            if price is not None:
                params["price"] = format_decimal(price)
            if time_in_force:
                params["timeInForce"] = time_in_force

        try:
            return await self._request(
                "POST", "/api/v3/order", params=params, signed=True,
                idempotent=False, retries=0,
            )
        except VenueError as exc:
            if not exc.ambiguous:
                raise
            log.warning(
                "order submission for %s was ambiguous (%s); looking it up by "
                "client order id before deciding", symbol, exc.message,
            )
            resolved = await self._resolve_ambiguous(symbol, coid)
            if resolved is not None:
                log.warning(
                    "the ambiguous order %s did land at the venue with status %s",
                    coid, resolved.get("status"),
                )
                return resolved
            exc.remedy = (
                (exc.remedy + " ") if exc.remedy else ""
            ) + (f"Checked: no order with client id {coid} exists at the venue, "
                 "so the submission did not land.")
            raise

    async def _resolve_ambiguous(self, symbol: str, client_order_id: str,
                                 attempts: int = 3) -> dict[str, Any] | None:
        """Ask the venue whether an order we could not confirm actually exists.

        Retried a few times because the lookup itself can fail on the same blip
        that made the submission ambiguous, and because an order accepted
        moments ago may not be queryable for a beat.
        """
        for attempt in range(attempts):
            try:
                return await self.get_order(symbol, client_order_id=client_order_id)
            except VenueError as exc:
                if exc.code in (-2013, -2011):
                    return None  # a definitive "no such order"
                if attempt == attempts - 1:
                    raise VenueError(
                        "an order submission could not be confirmed and the "
                        "follow-up lookup also failed",
                        remedy=(f"Check the venue manually for client order id "
                                f"{client_order_id} on {symbol} before trading "
                                "this symbol again."),
                        ambiguous=True,
                    ) from exc
                await asyncio.sleep(0.5 * (attempt + 1))
        return None
