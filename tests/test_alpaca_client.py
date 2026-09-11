"""Transport, auth and order safety of the Alpaca client."""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from imperium.venues.alpaca.client import AlpacaClient, VenueError
from imperium.venues.assets import AssetClass
from mock_venue import KEY, LIVE_KEY, SECRET, MockVenue, html_block_page


async def test_both_auth_headers_are_sent_on_every_request(client, venue):
    """Prevents sending one header and getting a 401 that reads like a bad key.
    Alpaca authenticates with a key AND a secret header; there is no signature,
    so these two are the entire credential."""
    await client.account()
    sent = venue.requests[-1]
    assert sent.headers["APCA-API-KEY-ID"] == KEY
    assert sent.headers["APCA-API-SECRET-KEY"] == SECRET


async def test_a_key_from_the_wrong_environment_says_so(venue):
    """Prevents the single most common real Alpaca failure being reported as
    'bad key'.

    A live key against the paper host, or the reverse, returns exactly the same
    401 as a genuinely invalid key. The remedy has to name the environment,
    because the operator's key is fine and they will otherwise regenerate it.
    """
    client = AlpacaClient(LIVE_KEY, SECRET, paper=True,
                          transport=venue.transport, max_retries=0)
    with pytest.raises(VenueError) as excinfo:
        await client.account()
    assert excinfo.value.status == 401
    text = excinfo.value.operator_text()
    # The *actual host in use* must appear. A mutation test showed that
    # asserting only on the words "paper" and "endpoint" passed against the
    # static remedy text, so the dynamic part could be deleted unnoticed.
    assert client.base_url in text, text
    assert "paper" in text
    await client.aclose()


async def test_the_environment_is_reported_so_it_can_be_checked(venue):
    """Prevents an operator having no way to see which host is in use."""
    paper = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    live = AlpacaClient(KEY, SECRET, paper=False, transport=venue.transport)
    assert paper.environment == "paper" and "paper-api" in paper.base_url
    assert live.environment == "live" and live.base_url == "https://api.alpaca.markets"
    await paper.aclose()
    await live.aclose()


async def test_html_body_is_reported_as_never_having_reached_the_venue(venue):
    """Prevents printing a page of nginx markup into a status panel as if it
    were a venue message."""
    venue.fail_next.append(html_block_page)
    client = AlpacaClient(KEY, SECRET, transport=venue.transport, max_retries=0)
    with pytest.raises(VenueError) as excinfo:
        await client.account()
    assert "never reached the venue" in excinfo.value.message
    assert "<html" not in excinfo.value.operator_text().lower()
    await client.aclose()


async def test_a_rejected_order_is_never_retried(venue):
    """Prevents retrying a rejection. It means the order was wrong; re-sending
    sends the same wrong order, and on an order endpoint that is how a position
    gets opened twice."""
    client = AlpacaClient(KEY, SECRET, transport=venue.transport, max_retries=5)
    before = len(venue.requests)
    with pytest.raises(VenueError):
        await client.place_order("NOSUCH", "buy", qty=Decimal("1"))
    posts = [r for r in venue.requests[before:]
             if r.method == "POST" and r.url.path == "/v2/orders"]
    assert len(posts) == 1
    await client.aclose()


async def test_transport_failures_are_retried_but_orders_are_not(venue):
    """Prevents treating 'retry on transport failure' as universal. A timed-out
    read is safe to repeat; a timed-out order submission is not."""
    venue.fail_next.append(httpx.ReadTimeout("boom"))
    client = AlpacaClient(KEY, SECRET, transport=venue.transport, max_retries=2)
    assert (await client.get_clock()).is_open is True

    venue.fail_next.append(httpx.ReadTimeout("boom"))
    before = len(venue.requests)
    with pytest.raises(VenueError):
        await client.place_order("AAPL", "buy", qty=Decimal("1"))
    posts = [r for r in venue.requests[before:] if r.method == "POST"]
    assert len(posts) == 1
    await client.aclose()


async def test_ambiguous_submission_is_resolved_by_client_order_id(venue):
    """Prevents reporting 'failed' for an order that actually filled, which is
    how a bot ends up flat in its records and long at the venue."""
    real = venue.handler

    def timeout_after_accepting(request: httpx.Request) -> httpx.Response:
        real(request)
        raise httpx.ReadTimeout("dropped after the order landed")

    client = AlpacaClient(KEY, SECRET, transport=venue.transport, max_retries=0)
    venue.fail_next.append(timeout_after_accepting)
    result = await client.place_order("AAPL", "buy", qty=Decimal("1"),
                                      client_order_id="imp-known")
    assert result["client_order_id"] == "imp-known"
    assert result["status"] == "filled"
    await client.aclose()


async def test_ambiguous_submission_that_did_not_land_says_so_definitively(venue):
    """Prevents leaving an operator unable to tell 'did not send' from 'unknown'."""
    client = AlpacaClient(KEY, SECRET, transport=venue.transport, max_retries=0)
    venue.fail_next.append(httpx.ReadTimeout("dropped before the venue saw it"))
    with pytest.raises(VenueError) as excinfo:
        await client.place_order("AAPL", "buy", qty=Decimal("1"),
                                 client_order_id="imp-never")
    assert "did not land" in excinfo.value.remedy
    await client.aclose()


async def test_every_order_carries_a_client_order_id(client, venue):
    """Prevents submitting an order with no client-side identity. Without one
    there is no way to ask whether a timed-out submission landed."""
    await client.place_order("AAPL", "buy", qty=Decimal("1"))
    body = json.loads(venue.requests[-1].content)
    assert body["client_order_id"].startswith("imp-")


async def test_time_in_force_defaults_by_asset_class(client, venue):
    """Prevents a day order on a 24/7 market. 'day' expires at a boundary that
    does not exist for crypto, so the two classes need different defaults."""
    await client.place_order("AAPL", "buy", qty=Decimal("1"))
    assert json.loads(venue.requests[-1].content)["time_in_force"] == "day"
    await client.place_order("BTC/USD", "buy", qty=Decimal("0.01"))
    assert json.loads(venue.requests[-1].content)["time_in_force"] == "gtc"


async def test_qty_and_notional_are_mutually_exclusive(client):
    """Prevents sending both, which the venue rejects, or neither, which is a
    silently sizeless order."""
    with pytest.raises(ValueError):
        await client.place_order("AAPL", "buy")
    with pytest.raises(ValueError):
        await client.place_order("AAPL", "buy", qty=1, notional=100)


async def test_equity_and_crypto_data_come_from_different_namespaces(client, venue):
    """Prevents asking one endpoint for both. Alpaca serves equities from
    /v2/stocks and crypto from /v1beta3/crypto/us, and a client that uses one
    for both silently returns no prices for half the book."""
    await client.bars(["AAPL", "BTC/USD"], limit=10)
    paths = [r.url.path for r in venue.requests if "bars" in r.url.path]
    assert any("/v2/stocks/bars" in p for p in paths)
    assert any("/v1beta3/crypto/us/bars" in p for p in paths)


async def test_snapshots_handle_both_response_shapes(client):
    """Prevents reading only one shape. The equity snapshot map is top level and
    the crypto one is nested under 'snapshots'; handling one yields no crypto
    prices at all, which looks like an empty market rather than a bug."""
    snaps = await client.snapshots(["AAPL", "BTC/USD"])
    assert "AAPL" in snaps and "BTC/USD" in snaps
    assert snaps["AAPL"]["latestQuote"]["ap"] > 0
    assert snaps["BTC/USD"]["latestQuote"]["ap"] > 0


async def test_the_market_clock_is_read_from_the_venue(client, venue):
    """Prevents computing market hours locally. Only the venue knows about
    early closes, holidays and unscheduled halts, and each of those is a day a
    local calendar trades into a closed market."""
    venue.market_open = False
    clock = await client.get_clock()
    assert clock.is_open is False
    assert "closed" in clock.describe()
    assert clock.next_open is not None


async def test_shorting_needs_both_permission_and_borrow(client):
    """Prevents shorting a hard-to-borrow name. It accepts the order and then
    fails to locate, so both flags must be true."""
    assets = await client.assets()
    assert assets["AAPL"].can_short is True
    assert assets["HARD"].shortable is True
    assert assets["HARD"].can_short is False, "hard-to-borrow must not be shortable"


async def test_a_halted_asset_is_visible_as_untradable(client):
    """Prevents sizing into a halt. Tradability is a per-asset fact the venue
    publishes, not a venue-wide property."""
    assets = await client.assets()
    assert assets["HALTED"].tradable is False
    assert assets["AAPL"].tradable is True


async def test_crypto_assets_carry_their_minimum_order_size(client):
    """Prevents sending a dust crypto order that the venue rejects."""
    assets = await client.assets()
    assert assets["BTC/USD"].min_order_size == Decimal("0.0001")
    assert assets["BTC/USD"].asset_class is AssetClass.CRYPTO


async def test_rate_limit_headers_are_believed_over_local_counting(client, venue):
    """Prevents counting requests locally. The venue's own header survives
    restarts and other processes sharing the key."""
    venue.rate_remaining = 5
    await client.get_clock()
    assert client.budget.remaining == 5
    assert client.budget.pause_needed() > 0


async def test_an_unauthenticated_client_refuses_before_sending(venue):
    """Prevents a confusing 401 when the real problem is that no key is
    configured at all."""
    client = AlpacaClient(transport=venue.transport, max_retries=0)
    with pytest.raises(VenueError) as excinfo:
        await client.account()
    assert "needs an API key" in excinfo.value.message
    await client.aclose()
