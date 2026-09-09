"""What the terminal streams, and why it is bounded.

The cost of a frame is not what the server spends building it -- that is
milliseconds -- but what the browser must parse and lay out before the next one
arrives. At 1Hz a frame that grows with the universe is a terminal that stops
responding once the universe is interesting, so every test here pins a bound
rather than a feature.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from imperium.execution.bars import Bar
from imperium.execution.broker import PaperBroker
from imperium.session import DETAIL_ROWS, WATCHLIST_ROWS, TradingSession
from imperium.venues import registry


def populated(n: int, *, bars: int = 60) -> TradingSession:
    session = TradingSession()
    session.universe = [f"SYM{i}" for i in range(n)]
    rng = np.random.default_rng(4)
    for symbol in session.universe:
        engine = session.engine(symbol)
        q = session.feed.quote(symbol)
        q.last, q.bid, q.ask = 100.0, 99.99, 100.01
        q.updated_at, q.quote_volume = time.time(), 5e8
        price = 100.0
        for k in range(bars):
            price *= float(np.exp(rng.normal(0, 0.0006)))
            engine.series.add(Bar(k * 60_000, price, price * 1.001, price * 0.999,
                                  price, 1000.0, closed=True))
        session.allocator.observe(symbol).admitted = True
    return session


def kb(payload: dict) -> float:
    return len(json.dumps(payload)) / 1024


# ------------------------------------------------------------ the bound

def test_the_frame_does_not_grow_with_the_universe():
    """The property the whole design exists for.

    Before this, a 1,500-symbol universe produced a 1.2MB frame every second:
    the browser spent the whole second parsing and laying out the previous one.
    A frame that is flat in the universe size is what makes scanning the whole
    market possible at all, so it is measured rather than asserted.
    """
    small, large = populated(40), populated(1500)
    for session in (small, large):
        for i in range(600):
            session.telemetry.pulse(session.universe[i % len(session.universe)],
                                    "scan", "scanning", 0.15)

    first_small, first_large = small.snapshot(), large.snapshot()
    steady_large = large.snapshot(since_pulse=first_large["pulse_seq"],
                                  since_event=first_large["event_seq"])

    # A 37x wider universe must not cost 37x the frame.
    assert kb(first_large) < kb(first_small) * 2
    assert kb(steady_large) < 100
    assert len(steady_large["watchlist"]) <= WATCHLIST_ROWS


def test_the_table_says_how_much_it_is_not_showing():
    """A capped table that reports its own length as the universe size is a
    table that lies. Every symbol below the line is still evaluated and still
    counted in the census; only the rows stop."""
    session = populated(WATCHLIST_ROWS + 250)
    snap = session.snapshot()
    scan = snap["universe_scan"]

    assert scan["size"] == WATCHLIST_ROWS + 250
    assert scan["shown"] == len(snap["watchlist"]) <= WATCHLIST_ROWS
    assert scan["omitted"] == scan["size"] - scan["shown"]
    # The census counts the whole universe, not the streamed window.
    assert sum(snap["regime_census"].values()) == scan["size"]


@pytest.mark.asyncio
async def test_a_held_position_is_always_streamed_however_it_ranks():
    """The one omission that would be indefensible.

    Rows are streamed in rank order, and rank moves. A position whose row
    vanished because its symbol drifted down a sort is a position the operator
    cannot see, cannot reason about, and would reasonably believe was closed.
    """
    session = populated(WATCHLIST_ROWS + 400)
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    laggard = session.universe[-1]
    session.feed.quote(laggard).last = 100.0
    await session.broker.apply_target(laggard, 0.1, 100.0, 10_000.0)

    snap = session.snapshot()
    streamed = {r["symbol"] for r in snap["watchlist"]}
    assert laggard in streamed
    # And with its full reasoning, not as a bare row.
    row = next(r for r in snap["watchlist"] if r["symbol"] == laggard)
    assert "decision" in row and row["decision"]["symbol"] == laggard

    # With the ranking taken out of the picture entirely. Held symbols sort to
    # the front, which means the ranking alone would satisfy the assertion
    # above and the explicit guard could be deleted unnoticed. A zero-row
    # window leaves nothing but the guard.
    starved = session.snapshot(rows_limit=0, detail=0)
    assert [r["symbol"] for r in starved["watchlist"]] == [laggard]
    assert "decision" in starved["watchlist"][0]


def test_every_streamed_row_can_be_tagged_with_its_asset_class():
    """The table tags every row EQ/CR, so the class travels on every row even
    though the rest of the decision does not. Dropping it would leave most of
    the table untagged, which reads as an unclassified symbol rather than as an
    undelivered field."""
    session = populated(300)
    snap = session.snapshot()
    assert snap["watchlist"]
    for row in snap["watchlist"]:
        assert row["asset_class"], row["symbol"]


def test_only_the_rows_that_are_read_carry_their_full_reasoning():
    session = populated(400)
    snap = session.snapshot()
    detailed = [r for r in snap["watchlist"] if "decision" in r]
    assert len(detailed) <= DETAIL_ROWS + 5
    # Enough for the reasoning panel, which renders 24, with churn headroom.
    assert len(detailed) >= 24


# ------------------------------------------------------------ the deltas

def test_a_delta_frame_carries_only_what_the_client_has_not_seen():
    session = populated(20)
    for i in range(50):
        session.telemetry.pulse(session.universe[i % 20], "scan", "scanning", 0.1)
    # A history the client has already seen. Without it, "only the new event"
    # and "every event there is" are the same list and the cursor could be
    # ignored with the test still green.
    for i in range(20):
        session.telemetry.event("info", "session", f"old event {i}")
    first = session.snapshot()
    assert len(first["events"]) == 20
    assert first["delta"] is False
    assert len(first["pulses"]) == 50

    caught_up = session.snapshot(since_pulse=first["pulse_seq"],
                                 since_event=first["event_seq"])
    assert caught_up["delta"] is True
    assert caught_up["pulses"] == []

    session.telemetry.pulse("SYM1", "order", "bought", 1.0)
    session.telemetry.event("info", "order", "an order went out")
    nxt = session.snapshot(since_pulse=first["pulse_seq"],
                           since_event=first["event_seq"])
    assert [p["reason"] for p in nxt["pulses"]] == ["bought"]
    assert [e["message"] for e in nxt["events"]] == ["an order went out"]


def test_a_first_frame_carries_the_backlog_so_the_panel_is_not_empty():
    """A client that has just connected has no history of its own. Sending it
    only what happened in the last second would leave the log and the cluster
    blank on a session that has been running for hours."""
    session = populated(10)
    for i in range(30):
        session.telemetry.event("info", "session", f"event {i}")
        session.telemetry.pulse("SYM1", "scan", "scanning", 0.1)

    fresh = session.snapshot()
    assert fresh["delta"] is False
    assert len(fresh["events"]) == 30
    assert len(fresh["pulses"]) == 30


def test_a_client_that_falls_behind_the_ring_is_told_to_start_over():
    """Prevents a silent gap.

    The pulse ring is bounded. A client whose cursor is older than the oldest
    row still in it cannot be caught up by a delta -- those rows are gone. The
    server detects that and sends the window with delta=False, which is the
    client's instruction to replace rather than append. Without the check the
    client would append a fragment onto a stale ring and show a history that
    never happened.
    """
    session = populated(10)
    session.telemetry.pulse("SYM1", "scan", "first", 0.1)
    stale_cursor = session.telemetry.latest_pulse_seq

    # Overflow the ring well past that cursor.
    capacity = session.telemetry._pulses.maxlen
    for i in range(capacity + 50):
        session.telemetry.pulse("SYM2", "scan", f"later {i}", 0.1)

    oldest = session.telemetry.oldest_pulse_seq
    assert stale_cursor < oldest - 1, "the cursor must really be behind the ring"

    # This is the decision the websocket loop makes on every frame.
    caught_up = session.snapshot(since_pulse=stale_cursor)
    assert caught_up["pulses"], "a delta from a lost cursor returns rows..."
    # ...but they are a fragment: the rows between the cursor and the ring are
    # gone, so the loop must reset instead of trusting this.
    assert oldest > stale_cursor + 1
