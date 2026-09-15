"""Every symbol gets reasoned about, whatever the feed is doing.

Until this existed the only path to a decision was a bar arriving on the
websocket. The data plan streams far fewer symbols than the scanner ranks, so
most symbols were never evaluated at all -- and with the market shut, none
were. The terminal showed an empty cluster, a reasoning panel full of symbols
nothing had looked at, and no decisions, which is indistinguishable from a
strategy that simply never fires.
"""

from __future__ import annotations

import datetime as dt
import time
from decimal import Decimal

import numpy as np
import pytest

from imperium.execution.bars import Bar
from imperium.execution.broker import PaperBroker
from imperium.session import EVAL_SLICE, TradingSession
from imperium.strategy import trend as trend_mod
from imperium.venues import registry
from imperium.venues.alpaca.client import MarketClock

UTC = dt.timezone.utc


async def _until_swept(session: TradingSession, passes: int = 1,
                       max_ticks: int = 200) -> int:
    """Tick until the sweep has completed ``passes`` full cycles.

    Bounded on purpose. An unbounded ``while session.sweeps < 1`` reads fine
    and hangs forever the moment the sweep stops running -- which is exactly
    the mutation these tests exist to catch. A test that hangs instead of
    failing is worse than no test: it stalls the suite and reports nothing.
    """
    ticks = 0
    while session.sweeps < passes:
        if ticks >= max_ticks:
            raise AssertionError(
                f"the sweep did not complete {passes} pass(es) in {max_ticks} "
                f"ticks — it is not running")
        await session._tick()
        ticks += 1
    return ticks


def _closed_session(n: int = 150, *, daily: bool = True) -> TradingSession:
    """A funded session, market shut, no websocket, n symbols in the universe."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("70")
    session.absorb_account({"equity": "70", "cash": "70", "currency": "USD"})
    session.market_clock = MarketClock(
        is_open=False, next_open=dt.datetime.now(UTC) + dt.timedelta(hours=14))
    rng = np.random.default_rng(3)
    session.universe = [f"EQ{i:03d}" for i in range(n)]
    scored = {}
    for i, symbol in enumerate(session.universe):
        engine = session.engine(symbol)
        price = 8.0 + (i % 15)
        q = session.feed.quote(symbol)
        q.last, q.bid, q.ask = price, price * 0.999, price * 1.001
        q.updated_at, q.quote_volume = time.time(), 5e8
        session.allocator.observe(symbol).admitted = True
        if not daily:
            continue
        p, bars, closes = price, [], []
        for k in range(320):
            arr = np.array(closes) if closes else np.array([p])
            score = (trend_mod.momentum_score(arr, 21) if arr.size > 23
                     else float("nan"))
            drift = (12.0 / 10_000.0) * score if np.isfinite(score) else 0.0
            p *= float(np.exp(drift + rng.normal(0, 0.02)))
            closes.append(p)
            bars.append(Bar(k * 86_400_000, p, p * 1.01, p * 0.99, p, 1e6,
                            closed=True))
        engine.daily_bars = bars
        pair = session._trend_observations(symbol, bars)
        if pair is not None:
            scored[symbol] = pair
    if scored:
        session.pooled_trend = trend_mod.pool(scored)
        for engine in session.engines.values():
            engine.pooled_trend = session.pooled_trend
    return session


@pytest.mark.asyncio
async def test_symbols_are_evaluated_with_no_stream_and_a_closed_market():
    """The exact situation the terminal was silent in.

    No websocket, no minute bars, market shut. Before the sweep this produced
    zero evaluations, zero pulses and an empty cluster -- which reads as a
    broken bot rather than a closed market.
    """
    session = _closed_session()
    assert session.snapshot()["counters"].get("scan", 0) == 0

    await _until_swept(session)

    counters = session.snapshot()["counters"]
    assert counters["scan"] == 150, "every symbol must have been looked at"
    assert counters["scan"] + counters.get("refused", 0) > 0
    assert session.snapshot()["pulses"], "the cluster needs pulses to draw orbs"


@pytest.mark.asyncio
async def test_the_sweep_is_divided_across_ticks_rather_than_done_at_once():
    """A full pass over 150 symbols costs about 140ms. Done every second that
    is a visible stall; divided, it is 23ms a tick and the whole universe is
    covered in six seconds."""
    session = _closed_session()

    await session._tick()

    scanned = session.snapshot()["counters"]["scan"]
    assert scanned == EVAL_SLICE, "one tick evaluates one slice, not everything"
    assert session.sweeps == 0

    ticks = 1 + await _until_swept(session)
    assert ticks == 150 // EVAL_SLICE


@pytest.mark.asyncio
async def test_the_cursor_wraps_so_no_symbol_is_starved():
    """A cursor that ran off the end would evaluate the first slice forever and
    leave everything after it permanently unlooked-at."""
    session = _closed_session(n=EVAL_SLICE + 7)

    seen: set[str] = set()
    for _ in range(6):
        before = {e.decision.reason for e in session.engines.values()}
        await session._tick()
        for symbol, engine in session.engines.items():
            if engine.decision.strategy:
                seen.add(symbol)
        del before

    assert seen == set(session.universe), "every symbol must come round"
    assert session.sweeps >= 2


@pytest.mark.asyncio
async def test_a_symbol_that_raises_does_not_stop_the_sweep():
    """One symbol's arithmetic must never strand the rest of the universe."""
    session = _closed_session(n=6)

    def _boom():
        raise ValueError("bad data")

    session.engines[session.universe[0]].scan = _boom      # type: ignore

    await session._tick()

    assert session.snapshot()["counters"]["scan"] >= 5
    assert any("could not evaluate" in e["message"]
               for e in session.telemetry.events(20))


@pytest.mark.asyncio
async def test_a_symbol_with_no_minute_bars_still_reaches_the_daily_strategy():
    """The gate that used to stop this.

    The warmup check reads the minute ring, and the multi-day strategy does not
    use it. Refusing an unstreamed symbol for want of minute bars judged it by
    the entry condition of a strategy it was never going to run.
    """
    session = _closed_session(n=EVAL_SLICE)
    engine = session.engines[session.universe[0]]
    assert len(engine.series) == 0, "no minute bars at all"
    assert engine.has_daily_history

    await session._tick()

    assert engine.decision.strategy == "trend"
    assert engine.decision.regime != "warming_up"


@pytest.mark.asyncio
async def test_a_symbol_with_neither_stream_nor_history_says_which_is_missing():
    """"Warming up" on a symbol that has no stream and never will is the wrong
    answer to the question being asked. The reason has to name the cap."""
    # Wider than the plan's cap, so symbols beyond it genuinely have no
    # stream. With a handful of symbols they all fit and the message would
    # correctly take the other branch.
    session = _closed_session(n=60, daily=False)
    limit = session.feed.symbol_limit
    beyond = session.universe[limit + 5]

    await _until_swept(session)

    engine = session.engines[beyond]
    assert not engine.streamed
    assert engine.decision.verdict.value == "rejected"
    assert "no live stream" in engine.decision.reason

    # And one inside the cap says the ordinary thing, because for it the ring
    # really is just filling up.
    inside = session.engines[session.universe[0]]
    assert inside.streamed
    assert "resolves itself" in inside.decision.reason


@pytest.mark.asyncio
async def test_the_snapshot_says_when_the_next_full_rescan_is_due():
    """A scan note that has not changed for fifteen minutes reads as a stall.
    Saying when the next one lands turns a hang into a schedule."""
    session = _closed_session(n=4)
    session.universe_scanned_at = time.time()

    scan = session.snapshot()["universe_scan"]
    assert scan["next_scan_in"] > 0
    assert scan["sweeps"] == session.sweeps


@pytest.mark.asyncio
async def test_the_sweep_acts_on_what_it_decides():
    """Prevents a sweep that reasons beautifully and never trades.

    Evaluating produces a pulse whether or not anything is acted on, so a
    sweep with its order path removed still lights the cluster, still fills the
    reasoning panel, and still looks entirely healthy. The only thing that
    distinguishes it is a fill.
    """
    session = _closed_session(n=EVAL_SLICE)
    await _until_swept(session)

    trading = [e for e in session.engines.values()
               if e.decision.verdict.value == "trading" and not e.decision.hold]
    assert trading, "the fixture must produce something worth trading"
    assert session.broker.fills, (
        "the sweep decided to trade and sent nothing — reasoning without an "
        "order path looks identical to reasoning with one")


@pytest.mark.asyncio
async def test_the_cursor_survives_the_universe_shrinking_under_it():
    """The guard the wrap-at-the-top exists for.

    The scanner re-ranks every fifteen minutes and prunes what falls out, so
    the universe can shrink while the cursor is past the new end. Without the
    check the next tick slices an empty range and evaluates nothing at all --
    a whole tick spent looking at no symbols.
    """
    session = _closed_session(n=EVAL_SLICE * 3)
    await session._tick()
    await session._tick()
    assert session._sweep_cursor >= EVAL_SLICE

    # The scan drops most of the universe out from under the cursor.
    session.universe = session.universe[:3]

    before = session.snapshot()["counters"]["scan"]
    await session._tick()
    after = session.snapshot()["counters"]["scan"]

    assert after > before, (
        "the tick after a shrink evaluated nothing — the cursor was left past "
        "the end of the universe")


@pytest.mark.asyncio
async def test_the_overnight_measurement_is_reported_in_words():
    """Prevents the terminal reporting a statistic instead of a sentence.

    "market overnight drift +4.12bp/night (t=+6.50) from 3,560 symbol-nights"
    is precise and tells an operator nothing about what the program will do.
    The event has to say that; the raw figures belong in the detail.
    """
    from imperium.strategy.overnight import PooledDrift

    from imperium.venues.alpaca.client import AlpacaClient
    from mock_venue import KEY, SECRET, MockVenue

    # Driven through the code that emits it, not built by hand: an event
    # composed in the test proves the wording exists, not that the terminal
    # uses it.
    venue = MockVenue()
    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    session.universe = ["AAPL", "SPY"]
    try:
        await session.refresh_daily_history(force=True)
    finally:
        await session.detach_client()

    overnight = [e for e in session.telemetry.events(20)
                 if e["source"] == "overnight"]
    assert overnight, "the measurement must be reported at all"
    event = overnight[0]

    assert "basis points a night" in event["message"] or \
        "Not enough overnight history" in event["message"], (
            f"the event is still a statistic rather than a sentence: "
            f"{event['message']!r}")
    assert "symbol-nights" in event["detail"], "the figures stay available"
    # And an ordinary measurement must not be dressed as a fault.
    assert event["level"] != "warn"

    # The credible wording, checked against the type that produces it.
    credible = PooledDrift(4.12, 6.50, 3560, 40, 82.0, intraday_bps=-1.8).explain()
    assert "basis points a night" in credible
    assert "beats what the trade costs" in credible


def test_an_unmeasurable_drift_says_what_is_missing_not_just_that_it_failed():
    from imperium.strategy.overnight import PooledDrift

    explanation = PooledDrift(9.9, 0.41, 960, 4, 82.0).explain()
    assert "4 of 5 symbols" in explanation
    assert "2.0 is the bar" in explanation
    assert "until it is measurable" in explanation
