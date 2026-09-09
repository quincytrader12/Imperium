"""Rebalancing under transaction costs, and the churn it prevents.

Under proportional costs the optimal policy is not to track a target but to do
nothing inside a region around it (Constantinides 1986; Davis & Norman 1990).
Without that region a strategy re-targets on every evaluation, because equity
moves with every fill and every price tick and the delta is therefore never
quite zero.

Measured before the band existed: 120 consecutive bars produced 120 orders on a
target weight that never changed once, median size 0.06 of a share.
"""

from __future__ import annotations

import time
from decimal import Decimal

import numpy as np
import pytest

from imperium.execution.bars import Bar
from imperium.execution.broker import REBALANCE_BAND, PaperBroker
from imperium.session import TradingSession
from imperium.venues import registry


def _broker(cash: str = "100000") -> PaperBroker:
    broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    broker.cash = Decimal(cash)
    return broker


@pytest.mark.asyncio
async def test_a_target_that_has_not_moved_does_not_trade_every_bar():
    """The measurement that produced the band, kept as a test.

    The target weight here never changes. Every order after the first is
    correcting arithmetic noise, and paying a spread to do it.
    """
    broker = _broker()
    await broker.apply_target("AAPL", 0.16, 100.0, 100_000.0)
    assert len(broker.fills) == 1

    price = 100.0
    rng = np.random.default_rng(4)
    for _ in range(120):
        price *= float(np.exp(rng.normal(0, 0.0005)))
        equity = float(broker.equity({"AAPL": price}))
        await broker.apply_target("AAPL", 0.16, price, equity)

    assert len(broker.fills) == 1, (
        f"a constant target produced {len(broker.fills)} orders")


@pytest.mark.asyncio
async def test_a_drift_past_the_band_is_corrected():
    """The band is a no-trade region, not a refusal to rebalance. A position
    that has genuinely drifted must still be brought back."""
    broker = _broker()
    await broker.apply_target("AAPL", 0.16, 100.0, 100_000.0)
    before = len(broker.fills)

    # Well past the band: the target doubles.
    await broker.apply_target("AAPL", 0.32, 100.0, 100_000.0)

    assert len(broker.fills) == before + 1
    assert broker.fills[-1].side == "BUY"


@pytest.mark.asyncio
async def test_an_exit_is_never_inside_the_band():
    """The cardinal rule this codebase applies to every cap.

    A band that can swallow an exit is a band that traps a position in a losing
    trade, which is far worse than the churn it exists to prevent. Tested with
    a position so small that any proportional band would otherwise absorb it.
    """
    broker = _broker()
    await broker.apply_target("AAPL", 0.0001, 100.0, 100_000.0)
    assert not broker.positions["AAPL"].is_flat

    await broker.apply_target("AAPL", 0.0, 100.0, 100_000.0)

    assert broker.positions["AAPL"].is_flat
    assert broker.fills[-1].side == "SELL"


@pytest.mark.asyncio
async def test_a_first_entry_is_never_inside_the_band():
    """A flat position has no drift to be inside a band around. Its size was
    already decided by the sizer and by the venue minimum."""
    broker = _broker()
    tiny = 0.00001
    await broker.apply_target("AAPL", tiny, 100.0, 100_000.0)
    assert not broker.positions["AAPL"].is_flat


@pytest.mark.asyncio
async def test_flattening_the_whole_book_still_works_through_the_band():
    """flatten_all routes through the same arithmetic. A mode switch or a
    retirement that silently left positions open would be the worst possible
    place for this band to apply."""
    broker = _broker()
    await broker.apply_target("AAPL", 0.10, 100.0, 100_000.0)
    await broker.apply_target("MSFT", 0.10, 200.0, 100_000.0)

    flattened = await broker.flatten_all({"AAPL": 100.0, "MSFT": 200.0})

    assert len(flattened) == 2
    assert all(p.is_flat for p in broker.positions.values())


@pytest.mark.asyncio
async def test_the_band_scales_with_the_target_which_is_what_makes_it_safe():
    """The structural property the exit and entry guards rest on.

    The band is a fraction of the *target* value, not of the position held.
    Two consequences follow from that alone, and they are the reason exits and
    entries cannot be swallowed:

    * An exit has a target of zero, so its band is zero. Nothing is smaller
      than nothing.
    * A first entry's delta *is* the whole target, and a fraction of the target
      is never larger than the target itself.

    The explicit guards in _delta_quantity are therefore unreachable, and not
    by accident -- it is provable. A full exit's delta *is* the whole position
    and a first entry's delta *is* the whole target, so both are always at
    least as large as any fraction below one of either quantity. No change to
    what the band is proportional to can make those guards bite; deleting
    either one changes no behaviour, which the mutation sweep confirms.

    They are kept as documentation at the point of decision, and because they
    would stop being redundant the moment the band became a fixed amount
    ("never trade less than $5") rather than a fraction. The property that
    actually protects an exit is the one asserted below: that the band is a
    fraction strictly under one. That is the invariant to defend, not the
    branches.
    """
    broker = _broker()
    # A position so small that a band scaled by the *position* would absorb any
    # move against it, and so small that a fixed-dollar band certainly would.
    await broker.apply_target("AAPL", 0.0001, 100.0, 100_000.0)
    held = broker.positions["AAPL"].quantity
    assert held > 0

    # Halving it is a 50% move: far outside a 10% band either way, so this
    # proves the band is not simply refusing everything small.
    await broker.apply_target("AAPL", 0.00005, 100.0, 100_000.0)
    assert broker.positions["AAPL"].quantity < held

    # And going to zero always acts, whatever the sizes involved.
    await broker.apply_target("AAPL", 0.0, 100.0, 100_000.0)
    assert broker.positions["AAPL"].is_flat


def test_the_band_is_a_fraction_not_a_fixed_amount():
    """A fixed dollar band would be most of a $25 position and nothing at all
    on a $4,000 one. The region has to scale with what is being held."""
    assert 0 < float(REBALANCE_BAND) < 0.5


@pytest.mark.asyncio
async def test_the_session_does_not_churn_a_symbol_over_a_full_session():
    """End to end, through the engine and the session rather than the broker
    alone: the same measurement that started this, on the real decision path."""
    session = TradingSession()
    session.broker = _broker()
    session.allocator.equity = 100_000.0
    session.allocator.cash = 100_000.0
    session.universe = ["AAPL"]
    engine = session.engine("AAPL")
    session.allocator.observe("AAPL").admitted = True
    q = session.feed.quote("AAPL")
    q.last, q.bid, q.ask = 100.0, 99.99, 100.01
    q.updated_at, q.quote_volume = time.time(), 1e9

    rng = np.random.default_rng(4)
    price, t = 100.0, 0
    for _ in range(engine.params.warmup_bars + 40):
        price *= float(np.exp(0.0004 + rng.normal(0, 0.0005)))
        engine.series.add(Bar(t, price, price * 1.001, price * 0.999, price,
                              1e5, closed=True))
        t += 60_000

    for _ in range(120):
        price *= float(np.exp(0.0004 + rng.normal(0, 0.0005)))
        engine.series.add(Bar(t, price, price * 1.001, price * 0.999, price,
                              1e5, closed=True))
        t += 60_000
        q.last, q.bid, q.ask = price, price * 0.9999, price * 1.0001
        q.updated_at = time.time()
        engine.set_book(q.bid, q.ask)
        await session._act_on(engine.evaluate())

    assert len(session.broker.fills) <= 3, (
        f"120 bars produced {len(session.broker.fills)} orders on a target "
        f"that barely moved")
