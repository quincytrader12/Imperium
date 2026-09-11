"""Walking the whole market with a bounded number of engines.

The ranking covers every tradable listing. Only a cohort carries engines at
once, because a full bar ring measures about 286KB and an engine per listed
symbol would be gigabytes. The cursor walks the ranking so that every symbol is
eventually reasoned about rather than only the head of it -- and what is
retired is what has been *answered*, not what is unimportant.
"""

from __future__ import annotations

import contextlib
from decimal import Decimal

import pytest

from imperium.execution.broker import PaperBroker
from imperium.execution.portfolio import Verdict
from imperium.session import COHORT_SIZE, TradingSession
from imperium.venues import registry
from mock_venue import KEY, SECRET, MockVenue


@contextlib.asynccontextmanager
async def _session(extra_equities: int = 900):
    from imperium.venues.alpaca.client import AlpacaClient

    venue = MockVenue()
    venue.list_extra_equities(extra_equities)
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("70")
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    try:
        yield session, venue
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_the_whole_ranked_market_is_kept_but_only_a_cohort_is_resident():
    """The split the design rests on: rank everything, carry a cohort."""
    async with _session() as (session, _):
        await session.scan_universe()

        assert len(session.ranked_universe) >= 900, "the ranking is the market"
        assert len(session.universe) <= COHORT_SIZE, "the cohort is bounded"
        assert len(session.engines) <= COHORT_SIZE + 5


@pytest.mark.asyncio
async def test_a_rotation_retires_what_was_answered_and_brings_in_what_was_not():
    """The operator's actual request: a name that was scanned and had nothing
    to say makes way for one that has not been looked at yet."""
    async with _session() as (session, _):
        await session.scan_universe()
        first = list(session.universe)
        cursor = session._cohort_cursor

        session._cohort_rotated_at = 0.0
        await session.rotate_cohort()

        second = list(session.universe)
        assert session._cohort_cursor > cursor, "the cursor must advance"
        assert set(second) - set(first), "new symbols must arrive"
        assert set(first) - set(second), "answered symbols must leave"
        # And their engines go with them, which is the whole point.
        assert len(session.engines) <= COHORT_SIZE + 5


@pytest.mark.asyncio
async def test_the_cursor_walks_the_whole_ranking_and_comes_round():
    """Every symbol must eventually be reasoned about. A cursor that stalled
    would leave most of the market permanently unexamined -- which is the
    situation this replaces, only slower."""
    async with _session(extra_equities=400) as (session, _):
        await session.scan_universe()
        ranked = len(session.ranked_universe)

        seen: set[str] = set(session.universe)
        for _ in range(ranked // 10 + 20):
            session._cohort_rotated_at = 0.0
            await session.rotate_cohort()
            seen |= set(session.universe)
            if session.cohort_passes >= 1:
                break

        assert session.cohort_passes >= 1, "a full pass must complete"
        assert len(seen) > COHORT_SIZE * 2, (
            f"only {len(seen)} of {ranked} symbols were ever resident")


@pytest.mark.asyncio
async def test_a_position_is_never_rotated_out_from_under_itself():
    """The one retirement that would be indefensible.

    An engine is what manages a position -- its stop, its exit, its reasoning.
    Retiring it because the cursor moved on would leave a real position with
    nothing behind it and nothing on screen.
    """
    async with _session() as (session, _):
        await session.scan_universe()
        held = session.universe[0]
        session.feed.quote(held).last = 10.0
        await session.broker.apply_target(held, 0.2, 10.0, 70.0)
        assert not session.broker.positions[held].is_flat

        for _ in range(4):
            session._cohort_rotated_at = 0.0
            await session.rotate_cohort()

        assert held in session.universe
        assert held in session.engines


@pytest.mark.asyncio
async def test_a_symbol_worth_trading_survives_the_rotation():
    """A name that just cleared its cost gate must not be dropped because the
    cursor happened to move. It was answered "yes"."""
    async with _session() as (session, _):
        await session.scan_universe()
        winner = session.universe[3]
        session.engines[winner].decision.verdict = Verdict.TRADING

        session._cohort_rotated_at = 0.0
        await session.rotate_cohort()

        assert winner in session.universe


@pytest.mark.asyncio
async def test_an_overnight_or_trend_hold_survives_the_rotation():
    """Both carry an exit that has not happened yet. Losing the engine between
    the close and the next open is losing the trade."""
    async with _session() as (session, _):
        await session.scan_universe()
        night, carried = session.universe[1], session.universe[2]
        session.overnight_holdings[night] = 0.05
        session.trend_holdings[carried] = 0.0

        session._cohort_rotated_at = 0.0
        await session.rotate_cohort()

        assert night in session.universe and carried in session.universe


@pytest.mark.asyncio
async def test_a_rotation_is_rate_limited_rather_than_running_every_tick():
    """Each rotation costs a daily-bar request and a snapshot. Rotating on
    every tick would spend the whole request budget walking the ranking and
    leave none for the book it is holding."""
    async with _session() as (session, _):
        await session.scan_universe()
        cursor = session._cohort_cursor

        # Immediately after a rotation, nothing should move.
        for _ in range(5):
            assert await session.rotate_cohort() == 0
        assert session._cohort_cursor == cursor

        # Once the interval has passed, it advances.
        session._cohort_rotated_at = 0.0
        assert await session.rotate_cohort() > 0
        assert session._cohort_cursor > cursor


@pytest.mark.asyncio
async def test_the_pooled_estimate_spans_cohorts_rather_than_resetting():
    """The ranking is ordered by traded value, so a cohort is not a random
    sample: the first is mega-caps and the seventieth micro-caps. An estimate
    rebuilt from each in turn would swing between them and report the swing as
    a change in the market."""
    async with _session(extra_equities=400) as (session, _):
        await session.scan_universe()
        after_first = len(session._trend_samples)
        assert after_first > 0

        session._cohort_rotated_at = 0.0
        await session.rotate_cohort()

        assert len(session._trend_samples) > after_first, (
            "the second cohort's history must add to the first, not replace it")


@pytest.mark.asyncio
async def test_the_pooled_samples_stay_bounded():
    """Otherwise the accumulation becomes exactly the memory the cohort
    rotation exists to avoid.

    The ranking has to be wider than the cap for this to mean anything. With
    a market smaller than the bound the walk can never exceed it, the trim
    never runs, and the test passes with the trim deleted -- proving only
    that a small market is small.
    """
    from imperium.session import POOLED_SAMPLE_SYMBOLS

    async with _session(extra_equities=900) as (session, _):
        await session.scan_universe()
        seen: set[str] = set(session._trend_samples)
        for _ in range(12):
            session._cohort_rotated_at = 0.0
            await session.rotate_cohort()
            seen |= set(session._trend_samples)

        assert len(seen) > POOLED_SAMPLE_SYMBOLS, (
            f"only {len(seen)} symbols were ever sampled — the walk never "
            f"gave the bound anything to discard, so this proves nothing")
        assert len(session._trend_samples) <= POOLED_SAMPLE_SYMBOLS
        assert len(session._overnight_samples) <= POOLED_SAMPLE_SYMBOLS


@pytest.mark.asyncio
async def test_the_snapshot_reports_progress_through_the_market():
    """"Is it actually scanning everything" has to have a number, or the
    reasoning panel is just a list of verdicts with no sign of work."""
    async with _session(extra_equities=400) as (session, _):
        await session.scan_universe()
        scan = session.snapshot()["universe_scan"]

        assert scan["ranked"] >= 400
        assert scan["cohort_at"] > 0
        assert 0.0 < scan["cohort_progress"] <= 1.0
        assert scan["size"] <= COHORT_SIZE


@pytest.mark.asyncio
async def test_the_trading_loop_actually_rotates_the_cohort():
    """The call site, not the method.

    Every other test here calls ``rotate_cohort`` directly, so all of them
    still pass with the call removed from the trading loop -- a rotation that
    works perfectly and never happens. The terminal would then sit on the
    first cohort forever, which is the exact behaviour the rotation was
    written to replace, and nothing on screen would say so.

    So this drives the loop itself: start it, let one iteration run, stop it.
    """
    import asyncio

    async with _session() as (session, _):
        await session.scan_universe()
        first = list(session.universe)
        cursor = session._cohort_cursor
        # The loop is rate limited on purpose; without this the single
        # iteration below would correctly decline to rotate.
        session._cohort_rotated_at = 0.0

        session.running = True
        loop = asyncio.create_task(session._run())
        try:
            # One iteration, then the loop parks on its one-second sleep.
            for _ in range(200):
                await asyncio.sleep(0.01)
                if session._cohort_cursor > cursor:
                    break
        finally:
            session.running = False
            loop.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop

        assert session._cohort_cursor > cursor, (
            "the trading loop ran a full iteration and never rotated the "
            "cohort — the terminal would sit on the first 150 symbols "
            "forever")
        assert set(session.universe) - set(first), "new symbols must arrive"


@pytest.mark.asyncio
async def test_the_live_stream_follows_the_cohort_when_it_rotates(monkeypatch):
    """The regression the rotation introduced.

    ``feed.start`` is called once, at session start, with the universe as it
    was then. The cohort then replaces that universe every twenty seconds and
    nothing tells the socket. So the stream goes on feeding symbols that were
    retired -- nothing is looking at them -- while the symbols now being
    reasoned about have no live price at all.

    Worse, the sweep computes each engine's ``streamed`` flag from the
    *current* universe, so the terminal reports those symbols as streamed. The
    one indicator that would show the problem states the opposite.
    """
    import asyncio as _asyncio

    # The socket itself is not under test; the targeting is. Without this the
    # feed would spend the test trying to reach Alpaca.
    async def _no_connect(self, asset_class):
        await _asyncio.sleep(3600)

    monkeypatch.setattr("imperium.venues.alpaca.feed.MarketFeed._run",
                        _no_connect)

    async with _session() as (session, _):
        await session.scan_universe()
        await session.feed.start(session._stream_priority())
        streamed_before = set(session.feed.symbols[:session.feed.symbol_limit])

        session._cohort_rotated_at = 0.0
        await session.rotate_cohort()
        try:
            wanted = set(session._stream_priority()[:session.feed.symbol_limit])
            assert wanted - streamed_before, (
                "the fixture must actually change who most needs a stream")

            assert set(session.feed.symbols[:session.feed.symbol_limit]) == wanted, (
                "the cohort rotated and the socket is still subscribed to the "
                "symbols it was started with — the live stream is feeding "
                "names nothing is looking at any more")
        finally:
            await session.feed.stop()


@pytest.mark.asyncio
async def test_crypto_stays_resident_rather_than_being_walked_past():
    """The cursor must not retire the only market that is open.

    Crypto is the one class this balance can day-trade -- PDT counts equity
    round trips and exempts crypto -- and the only one trading while the
    equity market is shut. Rotating it out leaves the terminal with a cohort
    of closed-market equities, a stream with nothing to send, and a dark data
    lamp that is entirely correct and completely useless.
    """
    from imperium.venues.assets import AssetClass, classify_symbol

    async with _session(extra_equities=900) as (session, _):
        await session.scan_universe()
        crypto = [s for s in session.ranked_universe
                  if classify_symbol(s) is AssetClass.CRYPTO]
        assert crypto, "the fixture must list some crypto to pin"

        # Walk far enough that a cursor with no pin would have left them behind.
        for _ in range(6):
            session._cohort_rotated_at = 0.0
            await session.rotate_cohort()

        resident = set(session.universe)
        assert set(crypto) <= resident, (
            f"the cursor retired {sorted(set(crypto) - resident)} — the only "
            f"symbols that trade while the equity market is shut")


@pytest.mark.asyncio
async def test_the_stream_slots_go_to_crypto_before_arbitrary_equities():
    """The cap is thirty against a cohort of a hundred and fifty, so the order
    decides who gets a live price. Spending it on whichever equities the cursor
    stopped on gives each of them a stream for twenty seconds -- too short to
    warm a minute ring, and outside market hours not a stream at all.

    Built directly rather than driven through a rotation. The cohort is laid
    out as ``sorted(pinned) + fresh``, and real tickers put "BTC/USD" near the
    front of that sort, so crypto lands inside the cap by alphabet whatever
    the rule says. A fixture that cannot fail is not evidence: here the
    equities all sort *before* the crypto, which is the arrangement the rule
    exists for and the one where its absence shows.
    """
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.universe = [f"AAA{i:03d}" for i in range(120)] + ["BTC/USD", "ETH/USD"]
    limit = session.feed.symbol_limit
    assert len(session.universe) > limit * 2, "the cap must be contested"

    top = session._stream_priority()[:limit]

    assert {"BTC/USD", "ETH/USD"} <= set(top), (
        f"both crypto pairs lost their stream slot to equities that sort "
        f"ahead of them, so the feed is silent whenever the equity market "
        f"is: {top[:4]}")


@pytest.mark.asyncio
async def test_a_held_position_still_outranks_crypto_for_a_stream_slot():
    """The one ordering that must not change. A carried position is where a
    stale price means a stop that does not fire."""
    async with _session(extra_equities=900) as (session, _):
        await session.scan_universe()
        held = next(s for s in session.universe
                    if "/" not in s)          # an equity, so not crypto itself
        session.feed.quote(held).last = 10.0
        await session.broker.apply_target(held, 0.2, 10.0, 70.0)
        assert not session.broker.positions[held].is_flat

        assert session._stream_priority()[0] == held


@pytest.mark.asyncio
async def test_every_coin_stays_resident_however_it_ranks():
    """Reported as "the crypto tab says only one coin has daily data".

    The cohort is ranked by dollar turnover and a few dozen coins rank far
    below the large-cap equities, so one or two were resident at a time. A
    cross-section of one cannot be ranked, so the crypto panel read "1 coin
    has daily data, ranking needs 10" indefinitely and the strategy could
    never start — not because it was wrong, but because it was never shown the
    market it ranks within.

    Crypto is also the only class a small account can trade around the clock,
    being outside the pattern-day-trader rule, so it is the last thing that
    should be rotated out for an equity that ranks higher on turnover.
    """
    from imperium.venues.assets import AssetClass, classify_symbol

    async with _session(extra_equities=900) as (session, venue):
        venue.list_extra_crypto(16)
        await session.scan_universe()
        coins = {s for s in session.ranked_universe
                 if classify_symbol(s) is AssetClass.CRYPTO}
        assert coins, "the fixture must list some coins"
        # The fixture only proves anything if some coin ranks *below* the
        # cohort cutoff on turnover. A coin that would be resident anyway
        # cannot show that pinning does something.
        head = set(session.ranked_universe[:COHORT_SIZE])
        assert coins - head, (
            "every coin ranks inside the cohort on its own, so this fixture "
            "would pass with the pinning removed")

        for _ in range(6):
            session._cohort_rotated_at = 0.0
            await session.rotate_cohort()
            missing = coins - set(session.universe)
            assert not missing, (
                f"{len(missing)} of {len(coins)} coins were rotated out of the "
                f"cohort — the cross-section cannot rank a market it cannot see")


@pytest.mark.asyncio
async def test_pinning_the_coins_does_not_stop_the_equity_walk():
    """The guard on the fix. Coins are pinned, so they come out of the cohort's
    budget -- if that were unbounded the walk through the equity market would
    stall and most of the tape would never be reasoned about."""
    async with _session(extra_equities=900) as (session, venue):
        await session.scan_universe()
        cursor = session._cohort_cursor

        for _ in range(4):
            session._cohort_rotated_at = 0.0
            await session.rotate_cohort()

        assert session._cohort_cursor > cursor, (
            "the cursor stopped advancing once the coins were pinned")
        assert len(session.universe) <= COHORT_SIZE
