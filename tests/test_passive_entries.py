"""Entries that rest at the mid, and cross only if they have to.

The one property above all: **a decided trade always happens.** A resting
order that never fills becomes a market order when its window closes; one
that partly fills sends the rest at market; exits never rest at all. And the
second, which is how resting orders go wrong in practice: nothing is bought
twice -- the market order for the remainder goes only after the venue has
confirmed the limit has stopped, and a new target while one rests sends
nothing more.
"""

from __future__ import annotations

import itertools
from decimal import Decimal

import pytest

from imperium.execution import passive as pm
from imperium.execution.broker import (
    MARKET_ON_CLOSE, LiveBroker, Mode, Position,
)
from imperium.venues import registry
from imperium.venues.alpaca.client import VenueError


class _Asset:
    tradable, fractionable, status = True, True, "active"
    min_order_size = None
    price_increment = None


class FakeClient:
    """Orders whose state the test decides."""

    def __init__(self):
        self.orders: dict[str, dict] = {}
        self.placed: list[dict] = []
        self.cancels: list[str] = []
        self.cancel_takes_effect = True
        self._ids = itertools.count(1)

    @staticmethod
    def new_client_order_id(tag="imp"):
        return f"{tag}-{next(FakeClient._seq)}"
    _seq = itertools.count(1)

    async def asset(self, symbol):
        return _Asset()

    async def positions(self):
        held: dict[str, Decimal] = {}
        for o in self.orders.values():
            q = Decimal(o["filled_qty"] or "0")
            held[o["symbol"]] = held.get(o["symbol"], Decimal("0")) + (
                q if o["side"] == "buy" else -q)
        return [{"symbol": k, "qty": str(v), "avg_entry_price": "100"}
                for k, v in held.items() if v]

    async def close_position(self, symbol):
        self.closed = symbol
        return {}

    async def account(self):
        return {"cash": "10000", "equity": "10000"}

    async def open_orders(self, symbols=None):
        return [o for o in self.orders.values() if o["status"] in ("new", "partially_filled")]

    async def place_order(self, symbol, side, *, qty=None, order_type="market",
                          limit_price=None, client_order_id=None,
                          time_in_force=None, **kw):
        oid = f"ord-{next(self._ids)}"
        row = {"id": oid, "client_order_id": client_order_id, "symbol": symbol,
               "side": side, "qty": str(qty), "type": order_type,
               "limit_price": None if limit_price is None else str(limit_price),
               "filled_qty": "0", "filled_avg_price": None, "status": "new"}
        if order_type == "market":
            row.update(filled_qty=str(qty), filled_avg_price="100.10",
                       status="filled")
        self.orders[client_order_id] = row
        self.placed.append(dict(row))
        return dict(row)

    async def get_order_by_client_id(self, coid):
        return dict(self.orders[coid])

    async def cancel_order(self, order_id):
        self.cancels.append(order_id)
        for row in self.orders.values():
            if row["id"] == order_id and self.cancel_takes_effect \
                    and row["status"] in ("new", "partially_filled"):
                row["status"] = "canceled"

    # test helpers
    def limit(self):
        return next(o for o in self.orders.values() if o["type"] == "limit")

    def fill_limit(self, qty, price, status="filled"):
        row = self.limit()
        row.update(filled_qty=str(qty), filled_avg_price=str(price), status=status)


def _broker(client, *, bid=99.90, ask=100.10, age=0.5, held="0"):
    b = LiveBroker(registry.get(registry.DEFAULT_VENUE), client, "k", mode=Mode.PAPER)
    b.arm_for_paper()
    b.quote_for = lambda s: (bid, ask, age)
    if held != "0":
        b.positions["AAPL"] = Position("AAPL", Decimal(held), Decimal("100"))
    return b


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("IMPERIUM_PASSIVE_ENTRIES", "true")
    monkeypatch.setenv("IMPERIUM_PASSIVE_SECONDS", "20")


# -- the price -------------------------------------------------------------------


def test_the_limit_is_the_mid_rounded_toward_the_passive_side():
    tick = Decimal("0.01")
    assert pm.limit_price("buy", 99.90, 100.10, tick) == Decimal("100.00")
    assert pm.limit_price("buy", 100.00, 100.03, tick) == Decimal("100.01")
    assert pm.limit_price("sell", 100.00, 100.03, tick) == Decimal("100.02")


def test_no_limit_when_there_is_no_price_inside_the_spread():
    tick = Decimal("0.01")
    assert pm.limit_price("buy", 100.00, 100.01, tick) is None
    assert pm.limit_price("buy", 100.00, 100.00, tick) is None
    assert pm.limit_price("buy", 0, 100.0, tick) is None


def test_the_tick_grid_follows_the_sub_penny_rule_and_the_venues_increment():
    assert pm.tick_for(12.0) == Decimal("0.01")
    assert pm.tick_for(0.52) == Decimal("0.0001")
    assert pm.tick_for(60_000.0, Decimal("0.5")) == Decimal("0.5")


@pytest.mark.parametrize("kw,reason", [
    (dict(order=MARKET_ON_CLOSE), "auction"),
    (dict(side="sell", held=Decimal("5")), "exit"),
    (dict(side="buy", held=Decimal("-5")), "exit"),
    (dict(bid=0.0), "two-sided"),
    (dict(bid=100.2, ask=100.1), "crossed"),
    (dict(quote_age=30.0), "stale"),
    (dict(bid=90.0, ask=110.0), "too wide"),
])
def test_what_never_rests(kw, reason):
    base = dict(side="buy", held=Decimal("0"), order="", bid=99.9, ask=100.1,
                quote_age=0.5)
    base.update(kw)
    assert reason in pm.why_not(**base)


def test_an_entry_or_an_addition_may_rest():
    for side, held in (("buy", Decimal("0")), ("buy", Decimal("3")),
                       ("sell", Decimal("0")), ("sell", Decimal("-3"))):
        assert pm.why_not(side=side, held=held, order="", bid=99.9, ask=100.1,
                          quote_age=0.5) == ""


def test_the_window_is_bounded_whatever_the_setting_says(monkeypatch):
    monkeypatch.setenv("IMPERIUM_PASSIVE_SECONDS", "0.01")
    assert pm.window_seconds() == pm.MIN_SECONDS
    monkeypatch.setenv("IMPERIUM_PASSIVE_SECONDS", "99999")
    assert pm.window_seconds() == pm.MAX_SECONDS
    monkeypatch.setenv("IMPERIUM_PASSIVE_SECONDS", "soon")
    assert pm.window_seconds() == pm.DEFAULT_SECONDS


# -- the broker ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_entry_rests_at_the_mid_and_books_nothing_until_it_fills():
    client = FakeClient()
    b = _broker(client)
    fill = await b.apply_target("AAPL", 0.10, 100.0, 10_000.0, strategy="trend")
    assert fill is None
    assert [o["type"] for o in client.placed] == ["limit"]
    assert client.placed[0]["limit_price"] == "100.00"
    assert b.position("AAPL").quantity == 0
    assert "AAPL" in b.working

    client.fill_limit("10", "100.00")
    fills = await b.service_passive()
    assert len(fills) == 1 and fills[0].order == pm.PASSIVE
    assert fills[0].price == Decimal("100.00") and fills[0].strategy == "trend"
    assert b.position("AAPL").quantity == Decimal("10")
    assert "AAPL" not in b.working
    assert b.passive_stats.filled == 1
    assert b.passive_stats.saved == pytest.approx(10 * 0.10)


@pytest.mark.asyncio
async def test_an_unfilled_entry_crosses_at_market_when_its_window_closes():
    """The trade still happens: the market order that would have gone
    anyway goes, for the whole quantity, once the limit is confirmed gone."""
    client = FakeClient()
    b = _broker(client)
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0, strategy="trend")
    assert await b.service_passive() == []            # still inside its window
    deadline = b.working["AAPL"].deadline
    fills = await b.service_passive(now=deadline + 1)
    assert client.cancels
    assert [o["type"] for o in client.placed] == ["limit", "market"]
    assert Decimal(client.placed[1]["qty"]) == 10
    assert len(fills) == 1 and fills[0].order == ""
    assert fills[0].strategy == "trend"
    assert b.position("AAPL").quantity == Decimal("10")
    assert b.passive_stats.crossed == 1


@pytest.mark.asyncio
async def test_a_partial_fill_is_booked_once_and_only_the_rest_crosses():
    client = FakeClient()
    b = _broker(client)
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0, strategy="trend")
    client.fill_limit("4", "100.00", status="partially_filled")
    first = await b.service_passive()
    assert [f.quantity for f in first] == [Decimal("4")]
    again = await b.service_passive()
    assert again == [], "the same partial fill was booked twice"
    fills = await b.service_passive(now=b.working["AAPL"].deadline + 1)
    assert [Decimal(o["qty"]) for o in client.placed if o["type"] == "market"] == [6]
    assert sum(f.quantity for f in first + fills) == Decimal("10")
    assert b.position("AAPL").quantity == Decimal("10")
    assert b.passive_stats.partial == 1


@pytest.mark.asyncio
async def test_pieces_are_priced_so_the_whole_averages_to_the_venues_figure():
    client = FakeClient()
    b = _broker(client)
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0)
    client.fill_limit("4", "100.00", status="partially_filled")
    a = await b.service_passive()
    client.fill_limit("10", "99.94")                    # the venue's average
    c = await b.service_passive()
    total = sum(f.price * f.quantity for f in a + c)
    assert total == pytest.approx(Decimal("999.40"))


@pytest.mark.asyncio
async def test_nothing_crosses_while_a_cancel_is_still_in_flight():
    """The double-buy: sending the remainder at market before the venue has
    stopped the limit means both can fill."""
    client = FakeClient()
    client.cancel_takes_effect = False
    b = _broker(client)
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0)
    late = b.working["AAPL"].deadline + 1
    assert await b.service_passive(now=late) == []
    assert await b.service_passive(now=late + 5) == []
    assert [o["type"] for o in client.placed] == ["limit"]
    assert len(client.cancels) == 1, "the cancel is sent once, not every tick"
    client.limit()["status"] = "canceled"
    fills = await b.service_passive(now=late + 10)
    assert [o["type"] for o in client.placed] == ["limit", "market"]
    assert fills and b.position("AAPL").quantity == Decimal("10")


@pytest.mark.asyncio
async def test_a_limit_that_fills_during_its_cancel_is_not_bought_again():
    client = FakeClient()
    b = _broker(client)
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0)
    client.fill_limit("10", "100.00")                   # filled as the window closed
    await b.service_passive(now=b.working["AAPL"].deadline + 1)
    assert [o["type"] for o in client.placed] == ["limit"]
    assert b.position("AAPL").quantity == Decimal("10")


@pytest.mark.asyncio
async def test_a_second_target_while_one_rests_sends_nothing_more():
    client = FakeClient()
    b = _broker(client)
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0)
    assert await b.apply_target("AAPL", 0.12, 100.0, 10_000.0) is None
    assert len(client.placed) == 1


@pytest.mark.asyncio
async def test_an_exit_withdraws_the_resting_entry_and_goes_at_market():
    client = FakeClient()
    b = _broker(client)
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0)
    client.fill_limit("3", "100.00", status="partially_filled")
    fill = await b.apply_target("AAPL", 0.0, 100.0, 10_000.0)
    assert client.cancels
    assert "AAPL" not in b.working
    assert client.placed[-1]["type"] == "market" and client.placed[-1]["side"] == "sell"
    assert Decimal(client.placed[-1]["qty"]) == 3
    assert fill is not None and b.position("AAPL").quantity == 0


@pytest.mark.asyncio
async def test_flattening_withdraws_a_resting_entry_first():
    """A retirement or mode switch closes at the venue; a resting entry left
    behind would reopen the position it just closed."""
    client = FakeClient()
    b = _broker(client)
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0)
    await b.flatten_symbol("AAPL")
    assert client.cancels and "AAPL" not in b.working
    assert client.closed == "AAPL"


@pytest.mark.asyncio
async def test_exits_and_reductions_never_rest():
    client = FakeClient()
    b = _broker(client, held="10")
    await b.apply_target("AAPL", 0.0, 100.0, 10_000.0)
    assert [o["type"] for o in client.placed] == ["market"]


@pytest.mark.asyncio
async def test_a_one_tick_spread_goes_straight_to_market_and_says_why():
    client = FakeClient()
    b = _broker(client, bid=100.00, ask=100.01)
    fill = await b.apply_target("AAPL", 0.10, 100.0, 10_000.0)
    assert [o["type"] for o in client.placed] == ["market"]
    assert fill is not None
    assert any("one tick" in k for k in b.passive_stats.skipped)


@pytest.mark.asyncio
async def test_switched_off_every_entry_crosses_as_before(monkeypatch):
    monkeypatch.setenv("IMPERIUM_PASSIVE_ENTRIES", "false")
    client = FakeClient()
    b = _broker(client)
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0)
    assert [o["type"] for o in client.placed] == ["market"]


@pytest.mark.asyncio
async def test_a_refused_limit_falls_back_to_market_at_once():
    client = FakeClient()
    real = client.place_order

    async def picky(symbol, side, **kw):
        if kw.get("order_type") == "limit":
            raise VenueError("limit orders not accepted for this asset")
        return await real(symbol, side, **kw)

    client.place_order = picky
    b = _broker(client)
    fill = await b.apply_target("AAPL", 0.10, 100.0, 10_000.0)
    assert fill is not None and fill.order == ""
    assert [o["type"] for o in client.placed] == ["market"]


# -- in the session --------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_session_books_a_resting_fill_through_the_tick_path():
    """Driven through the tick, not by calling the step: the loop must check
    resting orders by itself, before attribution, so a fill that rested is
    journalled and attributed to the strategy that decided it."""
    from imperium.session import TradingSession

    session = TradingSession()
    session._attribution_loaded = True
    client = FakeClient()
    b = _broker(client)
    b.quote_for = session._quote_for_resting
    q = session.feed.quote("AAPL")
    q.bid, q.ask, q.last = 99.90, 100.10, 100.0
    import time
    q.updated_at = time.time()
    session.broker = b
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0, strategy="trend")
    client.fill_limit("10", "100.00")
    await session._tick()
    rows = session.trade_journal.read()
    assert [(r.side, r.strategy, r.order) for r in rows] == [
        ("BUY", "trend", pm.PASSIVE)]
    assert session.snapshot()["passive"]["filled"] == 1


@pytest.mark.asyncio
async def test_stop_withdraws_anything_still_resting():
    from imperium.session import TradingSession

    session = TradingSession()
    client = FakeClient()
    b = _broker(client)
    session.broker = b
    await b.apply_target("AAPL", 0.10, 100.0, 10_000.0)
    await session.stop()
    assert client.cancels and not b.working
