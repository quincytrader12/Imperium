"""Signing, transport and order-safety behaviour of the venue client.

The mock venue recomputes the HMAC over the exact query string it receives, so
every test here fails if the client re-encodes after signing.
"""

from __future__ import annotations

import urllib.parse

import httpx
import pytest

from godalgo.venues.binance.client import BinanceSpotClient, VenueError
from mock_venue import API_KEY, SECRET, MockVenue, html_block_page


async def test_signature_is_computed_over_the_exact_bytes_sent(client, venue):
    """Prevents: signing a rendered query string and then letting the HTTP client
    re-encode the params, which changes the bytes and yields -1022 with no
    indication which of the two strings was wrong.

    The mock verifies the HMAC against the raw received query, so a mismatch
    between signed and sent bytes fails this test the same way the venue would.
    """
    await client.account()
    assert venue.signature_failures == 0
    sent = venue.requests[-1].url.query.decode()
    payload, _, signature = sent.rpartition("&signature=")
    assert signature and "signature" not in payload
    # The signature must be last: anything appended after it is unsigned.
    assert sent.endswith(signature)


async def test_wrong_secret_is_reported_as_a_swapped_key_not_a_number(venue):
    """Prevents: surfacing '-1022' to the operator with no remedy. The single
    most common cause is the key and secret pasted into each other's fields, and
    the message has to say so."""
    client = BinanceSpotClient(API_KEY, "W" * 64, transport=venue.transport,
                               max_retries=0)
    with pytest.raises(VenueError) as excinfo:
        await client.account()
    assert excinfo.value.code == -1022
    assert "swapped" in excinfo.value.remedy or "pasted into each other" in excinfo.value.remedy
    await client.aclose()


async def test_recv_window_is_sent_on_signed_requests(client, venue):
    """Prevents: omitting recvWindow, which leaves the venue's 5000ms default
    implicit and makes a clock-drift failure harder to attribute."""
    await client.account()
    params = dict(urllib.parse.parse_qsl(venue.requests[-1].url.query.decode()))
    assert params["recvWindow"] == "5000"
    assert "timestamp" in params


async def test_clock_drift_is_reported_as_a_clock_problem(venue):
    """Prevents: reporting -1021 as 'timestamp outside recvWindow'. The operator
    cannot act on that; 'your machine's clock is wrong, sync it' ends the
    problem."""
    venue.clock_skew_ms = 60_000
    client = BinanceSpotClient(API_KEY, SECRET, transport=venue.transport,
                               max_retries=0)
    with pytest.raises(VenueError) as excinfo:
        await client.account()
    assert excinfo.value.code == -1021
    assert "clock" in excinfo.value.remedy.lower()
    await client.aclose()


async def test_measured_clock_offset_repairs_a_skewed_signed_request(venue):
    """Prevents: knowing the clock is wrong and still failing every signed
    request. sync_time measures the offset and outgoing timestamps carry it."""
    venue.clock_skew_ms = 60_000
    client = BinanceSpotClient(API_KEY, SECRET, transport=venue.transport,
                               max_retries=0)
    await client.sync_time()
    assert abs(client.time_offset_ms - 60_000) < 2000
    account = await client.account()          # would be -1021 without the offset
    assert account["accountType"] == "SPOT"
    await client.aclose()


async def test_html_body_is_reported_as_never_having_reached_the_venue(venue):
    """Prevents: printing a page of nginx markup into a status panel as if it
    were a venue message. Binance always answers JSON; an HTML body means an
    edge proxy, corporate filter or regional block answered instead."""
    venue.fail_next.append(html_block_page)
    client = BinanceSpotClient(API_KEY, SECRET, transport=venue.transport,
                               max_retries=0)
    with pytest.raises(VenueError) as excinfo:
        await client.account()
    assert "never reached the venue" in excinfo.value.message
    assert "<html" not in excinfo.value.operator_text().lower()
    await client.aclose()


async def test_a_rejected_order_is_never_retried(venue):
    """Prevents: retrying a -1013/-2010 style rejection. Those mean the order was
    wrong; re-sending it sends the same wrong order again, and on an order
    endpoint that is how a position gets opened twice.

    Verified by reverting: two independent guards enforce this -- ``retries=0``
    on the order POST and ``errors.is_retryable`` returning False for rejection
    codes -- and *either one alone* is sufficient, so this test only fails when
    both are reverted. It was checked that way rather than assumed.
    """
    client = BinanceSpotClient(API_KEY, SECRET, transport=venue.transport,
                               max_retries=5)
    before = len(venue.requests)
    with pytest.raises(VenueError):
        await client.place_order("NOPEUSDT", "BUY", quantity="1")
    posts = [r for r in venue.requests[before:]
             if r.method == "POST" and r.url.path == "/api/v3/order"]
    assert len(posts) == 1, "a rejected order must be sent exactly once"
    await client.aclose()


async def test_transport_failures_are_retried_but_orders_are_not(venue):
    """Prevents: treating 'retry on transport failure' as universal. A timed-out
    GET is safe to repeat; a timed-out order submission is not, because the
    order may already have landed."""
    venue.fail_next.append(httpx.ReadTimeout("boom"))
    client = BinanceSpotClient(API_KEY, SECRET, transport=venue.transport,
                               max_retries=2)
    assert await client.ping() is True          # retried, then succeeded

    venue.fail_next.append(httpx.ReadTimeout("boom"))
    before = len(venue.requests)
    with pytest.raises(VenueError):
        await client.place_order("BTCUSDT", "BUY", quantity="0.001")
    posts = [r for r in venue.requests[before:] if r.method == "POST"]
    assert len(posts) == 1, "an order must never be blind-retried after a timeout"
    await client.aclose()


async def test_ambiguous_submission_is_resolved_by_client_order_id(venue):
    """Prevents: reporting 'failed' for an order that actually filled, which is
    how a bot ends up flat in its own records and long at the venue.

    The POST times out *after* the venue accepted it. The client must look the
    order up by the client order ID it generated before sending, and return the
    real order."""
    real_handler = venue.handler

    def timeout_after_accepting(request: httpx.Request) -> httpx.Response:
        real_handler(request)          # the venue really does record the order
        raise httpx.ReadTimeout("connection dropped after the order landed")

    client = BinanceSpotClient(API_KEY, SECRET, transport=venue.transport,
                               max_retries=0)
    venue.fail_next.append(timeout_after_accepting)
    result = await client.place_order("BTCUSDT", "BUY", quantity="0.001",
                                      client_order_id="gda-known-id")
    assert result["clientOrderId"] == "gda-known-id"
    assert result["status"] == "FILLED"
    await client.aclose()


async def test_ambiguous_submission_that_did_not_land_says_so_definitively(venue):
    """Prevents: leaving an operator unable to tell 'the order did not send' from
    'we do not know'. When the lookup definitively returns no such order, the
    error says the submission did not land."""
    client = BinanceSpotClient(API_KEY, SECRET, transport=venue.transport,
                               max_retries=0)
    venue.fail_next.append(httpx.ReadTimeout("dropped before the venue saw it"))
    with pytest.raises(VenueError) as excinfo:
        await client.place_order("BTCUSDT", "BUY", quantity="0.001",
                                 client_order_id="gda-never-sent")
    assert "did not land" in excinfo.value.remedy
    await client.aclose()


async def test_post_only_sends_no_time_in_force(client, venue):
    """Prevents: sending LIMIT_MAKER with timeInForce, which is -1106. The field
    is not merely redundant on a post-only order, it is rejected."""
    await client.place_order("BTCUSDT", "BUY", quantity="0.001",
                             order_type="LIMIT_MAKER", price="59000.00")
    params = dict(urllib.parse.parse_qsl(venue.requests[-1].url.query.decode()))
    assert params["type"] == "LIMIT_MAKER"
    assert "timeInForce" not in params


async def test_quantities_go_on_the_wire_as_plain_decimals(client, venue):
    """Prevents: str(0.00001) == '1e-05' reaching the venue, which is -1100.
    The mock rejects scientific notation exactly as the venue does."""
    await client.place_order("BTCUSDT", "BUY", quantity=0.00001)
    params = dict(urllib.parse.parse_qsl(venue.requests[-1].url.query.decode()))
    assert params["quantity"] == "0.00001"
    assert "e" not in params["quantity"].lower()


async def test_every_order_carries_a_client_order_id(client, venue):
    """Prevents: submitting an order with no client-side identity. Without one
    there is no way to ask the venue whether a timed-out submission landed."""
    await client.place_order("BTCUSDT", "BUY", quantity="0.001")
    params = dict(urllib.parse.parse_qsl(venue.requests[-1].url.query.decode()))
    assert params["newClientOrderId"].startswith("gda-")


async def test_rate_limit_headers_are_believed_over_local_counting(client, venue):
    """Prevents: counting request weight locally. The venue's own header survives
    restarts and other processes sharing the IP; a local counter does not, and a
    repeated -1003 ban lengthens each time."""
    venue.used_weight = 5200
    await client.ping()
    assert client.budget.used_weight == 5200
    assert client.budget.pause_needed() > 0


async def test_unauthenticated_client_still_serves_public_data(venue):
    """Prevents: requiring a key before any price can be shown. The watchlist has
    to work with no credential stored at all."""
    client = BinanceSpotClient(transport=venue.transport, max_retries=0)
    assert await client.ping() is True
    assert len(await client.ticker_24h()) == 2
    with pytest.raises(VenueError) as excinfo:
        await client.account()
    assert "needs an API key" in excinfo.value.message
    await client.aclose()
