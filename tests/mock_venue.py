"""A fake Binance Spot that verifies signatures the way the real one does.

A mock that accepts any signature tests nothing about signing. This one
recomputes the HMAC over the exact query string it received and rejects with
`-1022` on a mismatch, which is what makes the byte-exactness tests in
``test_client_signing.py`` meaningful: if the client re-encodes params after
signing, this mock fails it, just as the venue would.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
from typing import Any, Callable

import httpx

API_KEY = "PK" + "A" * 62
SECRET = "S" * 64

EXCHANGE_INFO = {
    "symbols": [
        {
            "symbol": "BTCUSDT", "status": "TRADING",
            "baseAsset": "BTC", "quoteAsset": "USDT",
            "baseAssetPrecision": 8, "quoteAssetPrecision": 8,
            "permissions": ["SPOT"],
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                {"filterType": "LOT_SIZE", "stepSize": "0.00001000",
                 "minQty": "0.00001000", "maxQty": "9000.00000000"},
                {"filterType": "NOTIONAL", "minNotional": "5.00000000",
                 "applyMinToMarket": True},
            ],
        },
        {
            "symbol": "ETHUSDT", "status": "TRADING",
            "baseAsset": "ETH", "quoteAsset": "USDT",
            "baseAssetPrecision": 8, "quoteAssetPrecision": 8,
            "permissions": ["SPOT"],
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                {"filterType": "LOT_SIZE", "stepSize": "0.00010000",
                 "minQty": "0.00010000", "maxQty": "9000.00000000"},
                # The older filter name, so the parser is exercised on both.
                {"filterType": "MIN_NOTIONAL", "minNotional": "10.00000000",
                 "applyToMarket": False},
            ],
        },
        {
            "symbol": "DEADUSDT", "status": "BREAK",
            "baseAsset": "DEAD", "quoteAsset": "USDT",
            "permissions": ["SPOT"],
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "1",
                 "maxQty": "100"},
                {"filterType": "NOTIONAL", "minNotional": "5"},
            ],
        },
    ]
}


class MockVenue:
    """Configurable fake venue.

    Set ``fail_next`` to make the next call fail in a specific way, and inspect
    ``requests`` to assert on what actually went on the wire.
    """

    def __init__(self, *, api_key: str = API_KEY, secret: str = SECRET) -> None:
        self.api_key = api_key
        self.secret = secret.encode()
        self.requests: list[httpx.Request] = []
        self.orders: dict[str, dict[str, Any]] = {}
        self.order_seq = 1000
        #: A queue of canned failures; each entry is consumed by one request.
        self.fail_next: list[Callable[[httpx.Request], httpx.Response] | Exception] = []
        self.clock_skew_ms = 0
        self.used_weight = 1
        self.signature_failures = 0

    # -- helpers ---------------------------------------------------------

    def _json(self, payload: Any, status: int = 200) -> httpx.Response:
        return httpx.Response(
            status, json=payload,
            headers={"x-mbx-used-weight-1m": str(self.used_weight),
                     "content-type": "application/json"},
        )

    def _error(self, code: int, msg: str, status: int = 400) -> httpx.Response:
        return self._json({"code": code, "msg": msg}, status)

    def _verify(self, request: httpx.Request) -> httpx.Response | None:
        """Reject exactly as the venue would, on the exact received bytes."""
        raw_query = request.url.query.decode()
        if "signature=" not in raw_query:
            return self._error(-1102, "Mandatory parameter 'signature' was not sent.")
        if request.headers.get("X-MBX-APIKEY") != self.api_key:
            return self._error(-2015, "Invalid API-key, IP, or permissions for action.",
                               401)
        payload, _, signature = raw_query.rpartition("&signature=")
        expected = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            self.signature_failures += 1
            return self._error(-1022, "Signature for this request is not valid.")
        params = dict(urllib.parse.parse_qsl(payload, keep_blank_values=True))
        ts = int(params.get("timestamp", 0))
        recv = int(params.get("recvWindow", 5000))
        now = int(time.time() * 1000) + self.clock_skew_ms
        if abs(now - ts) > recv:
            return self._error(
                -1021,
                "Timestamp for this request is outside of the recvWindow.")
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
        params = dict(urllib.parse.parse_qsl(request.url.query.decode()))

        if path == "/api/v3/ping":
            return self._json({})
        if path == "/api/v3/time":
            return self._json({"serverTime": int(time.time() * 1000) + self.clock_skew_ms})
        if path == "/api/v3/exchangeInfo":
            return self._json(EXCHANGE_INFO)
        if path == "/api/v3/ticker/24hr":
            return self._json([
                {"symbol": "BTCUSDT", "lastPrice": "60000.00",
                 "priceChangePercent": "1.5", "quoteVolume": "900000000"},
                {"symbol": "ETHUSDT", "lastPrice": "3000.00",
                 "priceChangePercent": "-0.8", "quoteVolume": "400000000"},
            ])
        if path == "/api/v3/klines":
            n = int(params.get("limit", 500))
            out = []
            base = 60000.0
            t = int(time.time() * 1000) - n * 60_000
            for i in range(n):
                px = base + i * 0.5
                out.append([t + i * 60_000, f"{px:.2f}", f"{px + 5:.2f}",
                            f"{px - 5:.2f}", f"{px + 1:.2f}", "12.5",
                            t + i * 60_000 + 59_999, "750000.0", 400,
                            "6.0", "360000.0", "0"])
            return self._json(out)

        if path in ("/api/v3/account", "/api/v3/order", "/api/v3/openOrders",
                    "/api/v3/account/commission"):
            bad = self._verify(request)
            if bad is not None:
                return bad

        if path == "/api/v3/account":
            return self._json({
                "makerCommission": 10, "takerCommission": 10,
                "canTrade": True, "accountType": "SPOT",
                "balances": [
                    {"asset": "USDT", "free": "10000.00000000", "locked": "0.00000000"},
                    {"asset": "BTC", "free": "0.05000000", "locked": "0.00000000"},
                    {"asset": "XRP", "free": "0.00000000", "locked": "0.00000000"},
                ],
            })
        if path == "/api/v3/account/commission":
            return self._json({
                "symbol": params.get("symbol", "BTCUSDT"),
                "standardCommission": {"maker": "0.00100000", "taker": "0.00100000"},
            })
        if path == "/api/v3/openOrders":
            return self._json([o for o in self.orders.values()
                               if o["status"] == "NEW"])

        if path == "/api/v3/order" and request.method == "POST":
            return self._place(params)
        if path == "/api/v3/order" and request.method == "GET":
            coid = params.get("origClientOrderId")
            order = self.orders.get(coid) if coid else None
            if order is None:
                for o in self.orders.values():
                    if str(o["orderId"]) == params.get("orderId"):
                        order = o
                        break
            if order is None:
                return self._error(-2013, "Order does not exist.")
            return self._json(order)
        if path == "/api/v3/order" and request.method == "DELETE":
            coid = params.get("origClientOrderId")
            order = self.orders.get(coid) if coid else None
            if order is None:
                return self._error(-2011, "Unknown order sent.")
            order["status"] = "CANCELED"
            return self._json(order)

        return self._error(-1121, "Invalid symbol.", 400)

    def _place(self, params: dict[str, str]) -> httpx.Response:
        symbol = params.get("symbol", "")
        if symbol not in {s["symbol"] for s in EXCHANGE_INFO["symbols"]}:
            return self._error(-1121, "Invalid symbol.")
        if params.get("type") == "LIMIT_MAKER" and "timeInForce" in params:
            return self._error(-1106, "Parameter 'timeInForce' sent when not required.")
        qty_text = params.get("quantity", "0")
        if "e" in qty_text.lower():
            return self._error(-1100, "Illegal characters found in parameter 'quantity'.")
        self.order_seq += 1
        coid = params.get("newClientOrderId", f"auto{self.order_seq}")
        price = params.get("price") or "60000.00"
        order = {
            "symbol": symbol, "orderId": self.order_seq, "clientOrderId": coid,
            "transactTime": int(time.time() * 1000), "price": price,
            "origQty": qty_text, "executedQty": qty_text,
            "cummulativeQuoteQty": f"{float(qty_text) * float(price):.8f}",
            "status": "FILLED" if params.get("type") == "MARKET" else "NEW",
            "type": params.get("type"), "side": params.get("side"),
            "fills": ([{"price": price, "qty": qty_text, "commission": "0.001",
                        "commissionAsset": "USDT"}]
                      if params.get("type") == "MARKET" else []),
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
        headers={"content-type": "text/html"},
    )
