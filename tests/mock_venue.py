"""A fake Alpaca that answers the way the real one does.

Alpaca has no request signing, so the thing the previous mock existed to verify
-- byte-exact HMAC -- is gone. What replaces it is subtler and easier to get
wrong: the API is split across two hosts and three data namespaces, and an
environment mismatch (a paper key against the live host) returns a bare 401 that
looks exactly like a bad key.

This mock therefore enforces:

* the two auth headers, and which *environment* the key belongs to;
* equities and crypto served from different paths with different response
  shapes -- the equity snapshot map is top level, the crypto one is nested;
* client_order_id lookup, so the ambiguous-submission path is testable.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from typing import Any, Callable

import httpx

KEY = "PKTEST" + "A" * 14
SECRET = "s3cr3t" + "z" * 34
LIVE_KEY = "AKLIVE" + "B" * 14

#: Where a real HTTP stack starts refusing. Servers and proxies vary; this is
#: the conservative end of the range, and the point is that *some* bound exists.
MAX_URL_BYTES = 8000

EQUITY_ASSETS = [
    {"symbol": "AAPL", "name": "Apple Inc", "class": "us_equity",
     "exchange": "NASDAQ", "tradable": True, "shortable": True,
     "easy_to_borrow": True, "fractionable": True, "status": "active"},
    {"symbol": "SPY", "name": "SPDR S&P 500", "class": "us_equity",
     "exchange": "ARCA", "tradable": True, "shortable": True,
     "easy_to_borrow": True, "fractionable": True, "status": "active"},
    {"symbol": "HARD", "name": "Hard To Borrow Co", "class": "us_equity",
     "exchange": "NASDAQ", "tradable": True, "shortable": True,
     "easy_to_borrow": False, "fractionable": False, "status": "active"},
    {"symbol": "HALTED", "name": "Halted Co", "class": "us_equity",
     "exchange": "NASDAQ", "tradable": False, "shortable": False,
     "easy_to_borrow": False, "fractionable": False, "status": "active"},
]

CRYPTO_ASSETS = [
    {"symbol": "BTC/USD", "name": "Bitcoin", "class": "crypto",
     "exchange": "CRYPTO", "tradable": True, "shortable": False,
     "easy_to_borrow": False, "fractionable": True, "status": "active",
     "min_order_size": "0.0001", "min_trade_increment": "0.0001"},
    {"symbol": "ETH/USD", "name": "Ethereum", "class": "crypto",
     "exchange": "CRYPTO", "tradable": True, "shortable": False,
     "easy_to_borrow": False, "fractionable": True, "status": "active",
     "min_order_size": "0.001", "min_trade_increment": "0.001"},
]


class MockVenue:
    """Configurable fake Alpaca.

    Set ``fail_next`` to make the next call fail in a specific way, and inspect
    ``requests`` to assert on what actually went on the wire.
    """

    def __init__(self, *, key: str = KEY, secret: str = SECRET,
                 paper: bool = True) -> None:
        self.key = key
        self.secret = secret
        self.paper = paper
        self.requests: list[httpx.Request] = []
        self.orders: dict[str, dict[str, Any]] = {}
        self.order_seq = 1000
        self.fail_next: list[Callable[[httpx.Request], httpx.Response] | Exception] = []
        self.market_open = True
        self.equity = 100_000.0
        self.daytrade_count = 0
        self.rate_remaining = 200
        self.auth_failures = 0
        #: Extra listed equities, so a test can ask for a market rather than a
        #: handful. A five-symbol listing cannot show a request that grew too
        #: long to send, which is the failure batching exists to prevent.
        self._filler: list[dict[str, Any]] = []
        #: Symbols per snapshot request, in order, so a test can assert the
        #: sweep was batched rather than sent as one enormous URL.
        self.snapshot_batches: list[int] = []

    def list_extra_equities(self, count: int) -> list[str]:
        """Add ``count`` plain tradable equities to what /v2/assets returns."""
        made = []
        for i in range(count):
            symbol = f"FILL{i:04d}"
            self._filler.append({
                "symbol": symbol, "name": f"Filler {i}", "class": "us_equity",
                "exchange": "NASDAQ", "tradable": True, "shortable": True,
                "easy_to_borrow": True, "fractionable": True, "status": "active",
            })
            made.append(symbol)
        return made

    # -- helpers ---------------------------------------------------------

    def _json(self, payload: Any, status: int = 200) -> httpx.Response:
        return httpx.Response(
            status, json=payload,
            headers={"x-ratelimit-remaining": str(self.rate_remaining),
                     "x-ratelimit-limit": "200",
                     "content-type": "application/json"})

    def _error(self, status: int, code: int, message: str) -> httpx.Response:
        return self._json({"code": code, "message": message}, status)

    def _check_auth(self, request: httpx.Request) -> httpx.Response | None:
        key = request.headers.get("APCA-API-KEY-ID")
        secret = request.headers.get("APCA-API-SECRET-KEY")
        if key != self.key or secret != self.secret:
            self.auth_failures += 1
            return self._error(401, 40110000, "access key verification failed")
        # A live key against the paper host, or the reverse, is the single most
        # common real-world failure and is indistinguishable from a bad key.
        looks_live = key.startswith("AK")
        if looks_live == self.paper:
            self.auth_failures += 1
            return self._error(401, 40110000, "access key verification failed")
        return None

    # -- transport -------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_next:
            nxt = self.fail_next.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt(request)

        path = request.url.path
        params = dict(request.url.params)

        bad = self._check_auth(request)
        if bad is not None:
            return bad

        if path == "/v2/clock":
            now = dt.datetime.now(dt.timezone.utc)
            return self._json({
                "timestamp": now.isoformat(),
                "is_open": self.market_open,
                "next_open": (now + dt.timedelta(hours=8)).isoformat(),
                "next_close": (now + dt.timedelta(hours=4)).isoformat(),
            })
        if path == "/v2/account":
            return self._json({
                "status": "ACTIVE", "currency": "USD",
                "cash": f"{self.equity:.2f}", "equity": f"{self.equity:.2f}",
                "buying_power": f"{self.equity * 2:.2f}",
                "daytrade_count": self.daytrade_count,
                "pattern_day_trader": False,
            })
        if path == "/v2/positions":
            return self._json([])
        if path == "/v2/assets":
            wanted = params.get("asset_class", "us_equity")
            if wanted == "us_equity":
                return self._json(EQUITY_ASSETS + self._filler)
            return self._json(CRYPTO_ASSETS)

        # -- market data --------------------------------------------------
        if path.endswith("/bars"):
            symbols = [s for s in params.get("symbols", "").split(",") if s]
            limit = int(params.get("limit", 100))
            timeframe = params.get("timeframe", "1Min")
            bars = {}
            for i, symbol in enumerate(symbols):
                bars[symbol] = self._bar_rows(i, limit, timeframe)
            return self._json({"bars": bars, "next_page_token": None})

        if path.endswith("/snapshots"):
            symbols = [s for s in params.get("symbols", "").split(",") if s]
            # The real venue rejects a request whose query string is too long.
            # A mock that accepts any length cannot show the bug that batching
            # exists to prevent, so it enforces the same bound.
            if len(str(request.url)) > MAX_URL_BYTES:
                return self._error(414, 41410000, "request URI too long")
            self.snapshot_batches.append(len(symbols))
            snaps = {}
            for i, symbol in enumerate(symbols):
                px = 100.0 * (i + 1)
                snaps[symbol] = {
                    "latestTrade": {"p": px, "s": 100},
                    "latestQuote": {"bp": px * 0.9999, "ap": px * 1.0001,
                                    "bs": 2, "as": 2},
                    "dailyBar": {"o": px * 0.99, "h": px * 1.02,
                                 "l": px * 0.98, "c": px, "v": 5_000_000},
                }
            # Equities return the map at the top level; crypto nests it. A
            # client that handles only one shape silently yields no prices.
            if "crypto" in path:
                return self._json({"snapshots": snaps})
            return self._json(snaps)

        # -- orders --------------------------------------------------------
        if path == "/v2/orders" and request.method == "POST":
            return self._place(json.loads(request.content or b"{}"))
        if path == "/v2/orders" and request.method == "GET":
            return self._json([o for o in self.orders.values()
                               if o["status"] in ("new", "accepted")])
        if path == "/v2/orders:by_client_order_id":
            coid = params.get("client_order_id", "")
            order = self.orders.get(coid)
            if order is None:
                return self._error(404, 40410000, "order not found")
            return self._json(order)
        if path.startswith("/v2/positions/") and request.method == "DELETE":
            return self._json({"symbol": path.rsplit("/", 1)[-1],
                               "status": "closed"})

        return self._error(404, 40410000, "endpoint not found")

    #: Overnight and intraday drift, in basis points, baked into the daily bars
    #: this mock serves. Separate values because a fixture where they are equal
    #: cannot catch a decomposition that reads one session's return as the
    #: other's -- which is the one error the overnight split must not make.
    daily_overnight_bps = 5.0
    daily_intraday_bps = -3.0

    def _bar_rows(self, index: int, limit: int,
                  timeframe: str) -> list[dict[str, Any]]:
        """Bars whose spacing and shape actually follow the requested timeframe.

        A mock that serves minute-spaced rows for a ``1Day`` request would let a
        caller that asks for daily history and silently receives minutes pass
        every test, and the overnight decomposition is exactly the caller that
        cannot survive that: it would read one-minute seams as overnight gaps.
        """
        base = 100.0 * (index + 1)
        rows: list[dict[str, Any]] = []
        start = dt.datetime(2026, 1, 5, 14, 30, tzinfo=dt.timezone.utc)

        if timeframe == "1Day":
            close = base
            for k in range(limit):
                open_px = close * (1 + self.daily_overnight_bps / 10_000)
                close = open_px * (1 + self.daily_intraday_bps / 10_000)
                rows.append({
                    "t": (start + dt.timedelta(days=k)).isoformat(),
                    "o": open_px, "h": max(open_px, close) * 1.004,
                    "l": min(open_px, close) * 0.996, "c": close,
                    "v": 5_000_000, "n": 40_000, "vw": (open_px + close) / 2})
            return rows

        for k in range(limit):
            px = base * (1 + 0.0004 * k)
            rows.append({"t": (start + dt.timedelta(minutes=k)).isoformat(),
                         "o": px, "h": px * 1.001, "l": px * 0.999,
                         "c": px, "v": 10_000, "n": 50, "vw": px})
        return rows

    def _place(self, body: dict[str, Any]) -> httpx.Response:
        symbol = body.get("symbol", "")
        known = {a["symbol"] for a in EQUITY_ASSETS + CRYPTO_ASSETS}
        if symbol not in known:
            return self._error(404, 40410000, f"asset {symbol} not found")
        asset = next(a for a in EQUITY_ASSETS + CRYPTO_ASSETS
                     if a["symbol"] == symbol)
        if not asset["tradable"]:
            return self._error(403, 40310000, f"{symbol} is not tradable")
        qty = body.get("qty")
        if qty is not None and not asset["fractionable"]:
            if float(qty) != int(float(qty)):
                return self._error(
                    422, 42210000,
                    f"{symbol} does not support fractional quantities")
        if body.get("time_in_force") in ("cls", "opg") and (
                asset.get("class") == "crypto" or "/" in symbol):
            return self._error(
                422, 42210000,
                "market-on-close and market-on-open orders are available for "
                "US equities only")
        if body.get("extended_hours") and body.get("type") != "limit":
            return self._error(422, 42210000,
                               "extended hours orders must be limit orders")
        self.order_seq += 1
        coid = body.get("client_order_id", f"auto{self.order_seq}")
        price = 100.0
        order = {
            "id": f"ord-{self.order_seq}",
            "client_order_id": coid,
            "symbol": symbol,
            "side": body.get("side"),
            "qty": str(qty) if qty is not None else None,
            "notional": body.get("notional"),
            "filled_qty": str(qty) if qty is not None else "0",
            "filled_avg_price": f"{price:.2f}",
            "type": body.get("type"),
            "time_in_force": body.get("time_in_force"),
            "status": "filled",
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self.orders[coid] = order
        return self._json(order)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def html_block_page(request: httpx.Request) -> httpx.Response:
    """What a corporate filter or edge proxy answers with."""
    return httpx.Response(
        403,
        text="<!DOCTYPE html>\n<html><head><title>Access Denied</title></head>"
             "<body><h1>Access Denied</h1><p>Blocked by policy.</p>"
             "<hr><center>nginx/1.24.0</center></body></html>",
        headers={"content-type": "text/html"})
