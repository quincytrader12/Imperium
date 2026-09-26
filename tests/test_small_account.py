"""A $70 account is not a small version of a $70,000 one.

A percentage-based risk framework stops working quietly at a small balance,
because the binding constraints stop being relative and start being absolute.
Five positions of $11 is not a diversified book; it is five orders no venue
will treat as positions. Every test here pins one place where the arithmetic
has to change rather than scale.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from imperium.execution.broker import PaperBroker
from imperium.execution.risk import (
    MAX_SCALED_POSITION_WEIGHT, REFERENCE_POSITIONS, VIABLE_POSITION_NOTIONAL,
    RiskLimits, limits_for_equity,
    scale_for_equity,
)
from imperium.execution.sizing import size_position
from imperium.session import TradingSession
from imperium.venues import registry
from imperium.venues.assets import AssetClass, spec_for


#: An account small enough that the concentration rule still fires.
#:
#: This was $70 for most of this module's life, because the book-level floor
#: was $25 and $70 of 80%-deployable equity carries two of those. The floor is
#: $5 now -- $25 was an auction-order constraint being charged to every
#: strategy, which priced a small account out of ordinary fractional trading
#: altogether -- so $70 carries the full five positions and concentrates no
#: more than a large account does.
#:
#: Concentration begins below ``$25 / 0.80`` = $31.25. The tests that are
#: about concentration use this; the tests that are about $70 specifically
#: still say $70, and now assert the un-concentrated limits.
CONCENTRATES = 20.0


def _returns(sigma: float = 0.006, n: int = 300) -> np.ndarray:
    return np.random.default_rng(1).normal(0, sigma, n)


# --------------------------------------------------------- the scaling

def test_a_seventy_dollar_account_concentrates_instead_of_diversifying():
    """The base limits allow five positions of $11 on a $70 account.

    $11 is not a position. It cannot be held overnight (auction orders take
    whole shares), cannot be taken at all in a non-fractionable name, and
    cannot be trimmed. Diversification is a luxury that costs more than it
    returns at this size, so the account carries fewer, larger names.
    """
    base = RiskLimits()
    assert base.max_gross_exposure / base.max_concurrent_positions * 70 < 12

    scale = scale_for_equity(CONCENTRATES)
    limits = limits_for_equity(CONCENTRATES)

    assert scale.positions == 3
    assert not scale.unscaled
    assert limits.max_concurrent_positions == 3

    # But concentration has a floor under it now. This assertion used to read
    # ``max_position_weight * 70 >= VIABLE_POSITION_NOTIONAL`` -- the cap was
    # required to be wide enough to fit the $25 equity floor, which at two
    # positions meant 40% of the account in one name. A live crypto position
    # took 35% of a $70 book on exactly that arithmetic and the book was down
    # 34% by the time the daily halt measured it.
    #
    # Concentrating is still right at this size. Concentrating without a
    # ceiling is not a risk framework, it is a single bet.
    assert limits.max_position_weight <= MAX_SCALED_POSITION_WEIGHT
    assert limits.max_position_weight * CONCENTRATES == pytest.approx(5.0)


def test_the_scaling_disappears_once_the_account_can_afford_the_base_limits():
    """Convergence, tested at the boundary rather than asserted.

    This must change nothing for an account of any ordinary size, or it is not
    a small-account adjustment, it is a different risk framework.
    """
    base = RiskLimits()
    needed = (VIABLE_POSITION_NOTIONAL * base.max_concurrent_positions
              / base.max_gross_exposure)

    assert scale_for_equity(needed * 1.05).unscaled
    assert limits_for_equity(needed * 1.05) == base
    assert limits_for_equity(10_000.0) == base
    assert limits_for_equity(1_000_000.0) == base


def test_the_per_trade_budget_follows_the_concentration():
    """Otherwise it becomes the binding constraint and hands back exactly the
    size the position cap just allowed. A 0.5% budget written for five
    positions permits a $7 position on a $70 account at ordinary volatility."""
    base = RiskLimits()
    scaled = limits_for_equity(CONCENTRATES)
    scale = scale_for_equity(CONCENTRATES)

    assert scaled.risk_per_trade > base.risk_per_trade
    assert scaled.risk_per_trade == pytest.approx(
        base.risk_per_trade * REFERENCE_POSITIONS / scale.positions)
    # And it is bounded: a tiny account must not talk itself into risking
    # arbitrarily much per trade.
    assert limits_for_equity(1.0).risk_per_trade <= 0.02


def test_a_concentrated_book_gets_a_wider_daily_band():
    """A 4% daily halt written for five positions halts on ordinary noise when
    there are two: one name is 40% of the book, so a routine 10% move against
    it is the whole day's budget."""
    assert (limits_for_equity(CONCENTRATES).daily_loss_halt
            > RiskLimits().daily_loss_halt)
    assert limits_for_equity(CONCENTRATES).daily_loss_halt <= 0.10


def test_the_market_facing_parameters_are_not_scaled_by_the_wallet():
    """Deliberate, and the one place this could go wrong quietly.

    How far a stock moves before a stop is a property of the stock. Widening it
    because the account is small would size the market to the balance -- and it
    would *cut* the position for a given risk budget, which is the opposite of
    what a small account needs.
    """
    base, small = RiskLimits(), limits_for_equity(CONCENTRATES)
    assert small.atr_stop_multiple == base.atr_stop_multiple
    assert small.target_volatility == base.target_volatility
    assert small.max_gross_exposure == base.max_gross_exposure


def test_scaling_is_derivation_and_still_not_optimisable():
    """The limits remain unreachable from a strategy. Scaling reads one input
    the strategy does not control -- the balance -- and only ever tightens or
    concentrates in response to it."""
    from imperium.execution.risk import (
        OPTIMISABLE_PARAMETERS, risk_limit_field_names,
    )

    assert risk_limit_field_names() & OPTIMISABLE_PARAMETERS == frozenset()
    assert "equity" not in OPTIMISABLE_PARAMETERS


# ----------------------------------------------------- the venue floor

def test_a_position_too_small_for_the_venue_is_raised_not_submitted():
    """The bound that only exists on a small account.

    A weight is a fraction, and a fraction of a small balance can be an amount
    the venue will take and the strategy cannot use: too small to hold
    overnight, to take in a non-fractionable name, or to trim. Below the floor
    the position is raised to it -- but only as far as the per-symbol cap
    allows.
    """
    limits = limits_for_equity(40.0)
    result = size_position(
        signal=0.8, returns=_returns(), price=100.0, atr=2.0, limits=limits,
        seconds_per_year=5_896_800, bar_seconds=60, allows_short=False,
        equity=40.0)

    assert result.weight * 40.0 == pytest.approx(VIABLE_POSITION_NOTIONAL)
    assert result.binding == "venue minimum"
    # And it says what that costs, rather than quietly overspending the budget.
    assert "budget" in result.reason


def test_raising_to_the_floor_stops_at_the_per_symbol_cap():
    """The case that cost real money, now the other way round.

    An auction position on a $70 account needs a whole share -- call it $25 --
    which is 36% of the book. That used to be allowed, because the cap at two
    positions was 40%, so a correctly sized position was raised to the floor
    and one name carried a third of the account.

    The refusal is the honest outcome. An account that can only hold this name
    by putting a third of itself in it cannot hold this name, and saying so is
    worth more than a position sized to satisfy a minimum.
    """
    limits = limits_for_equity(70.0)
    result = size_position(
        signal=0.8, returns=_returns(), price=100.0, atr=2.0, limits=limits,
        seconds_per_year=5_896_800, bar_seconds=60, allows_short=False,
        equity=70.0,
        position_floor=float(
            spec_for(AssetClass.US_EQUITY).auction_position_notional))

    assert result.weight == 0.0
    assert result.binding == "account too small"
    assert "per-symbol cap" in result.reason


def test_the_auction_floor_is_charged_only_to_auction_orders():
    """Where the 35% position actually came from.

    The $25 is a **market-on-close** fact: the closing auction will not take a
    fractional quantity, so an overnight position must be whole shares, and
    $25 buys one share of a median US listing. It was being charged to every
    strategy, including the ones that send ordinary fractional orders and
    never go near an auction. On a $70 book that turned a correct $10.70 into
    $25 -- 35% of the account -- and refused everything it could not inflate.

    So the two floors are asserted against each other on the same account: the
    ordinary one leaves the position where the sizer put it, the auction one
    cannot be reached at all and says so.
    """
    limits = limits_for_equity(70.0)
    spec = spec_for(AssetClass.US_EQUITY)
    common = dict(
        signal=0.8, returns=_returns(), price=100.0, atr=2.0, limits=limits,
        seconds_per_year=5_896_800, bar_seconds=60, allows_short=False,
        equity=70.0)

    ordinary = size_position(
        **common, position_floor=float(spec.viable_position_notional))
    auction = size_position(
        **common, position_floor=float(spec.auction_position_notional))

    assert ordinary.weight > 0, "an ordinary fractional order was refused"
    assert ordinary.weight <= limits.max_position_weight
    assert ordinary.weight * 70.0 < float(spec.auction_position_notional), (
        "the position is still being raised to the auction floor")
    assert auction.weight == 0.0, (
        "a whole share of a $25 name is 36% of a $70 book and must be refused")


def test_a_symbol_the_account_cannot_reach_is_refused_with_the_arithmetic():
    """When even the floor would breach the per-symbol cap there is no size
    that works. Saying so beats a silent zero, which reads as "no opportunity"
    when it is really "not with this much money"."""
    limits = limits_for_equity(70.0)
    result = size_position(
        signal=0.8, returns=_returns(sigma=0.05), price=100.0, atr=25.0,
        limits=limits, seconds_per_year=5_896_800, bar_seconds=60,
        allows_short=False, equity=70.0)

    assert result.weight == 0.0
    assert result.binding == "account too small"
    assert "cannot carry this name" in result.reason
    assert "not a refusal to trade" in result.reason


def test_a_large_account_is_never_pushed_up_to_the_floor():
    """The floor is a small-account bound. On any ordinary balance the sizer's
    own answer is far above it and must be left exactly alone."""
    result = size_position(
        signal=0.8, returns=_returns(), price=100.0, atr=2.0,
        limits=RiskLimits(), seconds_per_year=5_896_800, bar_seconds=60,
        allows_short=False, equity=100_000.0)
    assert result.binding != "venue minimum"
    assert result.weight * 100_000 > VIABLE_POSITION_NOTIONAL * 10


# ------------------------------------------------------ the session

@pytest.mark.asyncio
async def test_the_session_rescales_when_the_balance_changes():
    """Re-derived rather than set once, because the whole point is that the
    balance grows. An account that reaches a few hundred dollars should spread
    back out on its own; one that draws down should concentrate again rather
    than keep sizing for money it no longer has."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.absorb_account(
        {"equity": str(CONCENTRATES), "cash": str(CONCENTRATES),
         "currency": "USD"})

    assert session.limits.max_concurrent_positions == 3
    assert session.allocator.limits is session.limits

    engine = session.engine("AAPL")
    assert engine.limits is session.limits

    session.absorb_account({"equity": "5000", "cash": "5000", "currency": "USD"})

    assert session.limits == RiskLimits(), "it must spread back out as it grows"
    assert session.allocator.limits is session.limits
    # The engine that already existed must not be left sizing for $70.
    assert engine.limits is session.limits


@pytest.mark.asyncio
async def test_a_drawdown_concentrates_the_book_again():
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.absorb_account({"equity": "5000", "cash": "5000", "currency": "USD"})
    assert session.limits.max_concurrent_positions == 5

    session.absorb_account({"equity": "20", "cash": "20", "currency": "USD"})
    assert session.limits.max_concurrent_positions < 5


@pytest.mark.asyncio
async def test_the_scale_is_published_with_the_numbers_it_implies():
    """"Why is it not trading" on a small account is almost always this, and
    the answer is arithmetic rather than a fault."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("70")
    session.absorb_account({"equity": "70", "cash": "70", "currency": "USD"})

    block = session.snapshot()["account_scale"]
    assert block["known"]
    # $70 carries the full five positions now. It used to carry two, because
    # the book-level floor was an auction constraint charged to every
    # strategy; five positions of $14 are ordinary fractional orders and are
    # entirely tradeable.
    assert block["positions"] == 5
    assert block["max_position_value"] == pytest.approx(14.0)
    assert block["position_floor"] == VIABLE_POSITION_NOTIONAL
    # The share-price reach limit: auction orders take whole shares, so this
    # is the dearest share the overnight strategy can reach here -- and the
    # floor it must clear is published beside it, because "why is it not
    # trading overnight" is a different question with a different answer.
    assert block["overnight_max_share_price"] == pytest.approx(14.0)
    assert block["auction_floor"] == pytest.approx(25.0)
    assert block["auction_floor"] > block["overnight_max_share_price"], (
        "at this balance no whole share is reachable overnight")


def test_the_per_symbol_cap_refuses_the_floor_on_its_own():
    """The two refusal branches, separated.

    A wide-stopped name is refused because raising it would risk too much; a
    genuinely tiny account refuses even a calm name because the smallest
    tradeable position is more of the book than any one name may be. Tested
    apart, because a case that trips the risk branch first would let the cap
    check be deleted with everything still green.
    """
    equity = 15.0
    limits = limits_for_equity(equity)
    floor_weight = VIABLE_POSITION_NOTIONAL / equity
    assert floor_weight > limits.max_position_weight, "the cap must be what bites"

    result = size_position(
        signal=0.8, returns=_returns(), price=100.0, atr=1.0, limits=limits,
        seconds_per_year=5_896_800, bar_seconds=60, allows_short=False,
        equity=equity)

    assert result.weight == 0.0
    assert result.binding == "account too small"
    assert "per-symbol cap" in result.reason


@pytest.mark.asyncio
async def test_the_limits_follow_the_book_even_with_no_account_attached():
    """Rescaling has to happen on the tick, not only when a balance arrives.

    An account is read once and then the book moves on its own -- that is what
    trading is. A session that only rescales on absorb_account would size a
    doubled or halved book against the balance it started with, and in dry run
    there is no account to absorb at all.
    """
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal(str(CONCENTRATES))

    await session._tick()
    assert session.limits.max_concurrent_positions == 3

    # The book grows on its own, with nothing absorbed.
    session.broker.cash = Decimal("5000")
    await session._tick()

    assert session.limits == RiskLimits()
    assert session.allocator.limits is session.limits


@pytest.mark.asyncio
async def test_a_share_the_account_cannot_buy_whole_is_refused_overnight():
    """Auction orders take whole shares, so this is a hard reach limit.

    A $200 share cannot be held overnight on a $70 account at any weight, and
    on a small balance that excludes most of the market. Saying so beats a
    silent absence of overnight trades, which is indistinguishable from a
    strategy that does not work.
    """
    import datetime as dt

    from imperium.execution.engine import SymbolEngine
    from imperium.execution.portfolio import PortfolioAllocator, Verdict
    from imperium.strategy.overnight import PooledDrift, SessionPhase
    from imperium.telemetry.streams import TelemetryHub

    limits = limits_for_equity(70.0)
    allocator = PortfolioAllocator(limits)
    allocator.equity, allocator.cash = 70.0, 70.0
    engine = SymbolEngine("AAPL", registry.get(registry.DEFAULT_VENUE), limits,
                          allocator, TelemetryHub())
    allocator.observe("AAPL").admitted = True
    engine.session_phase = SessionPhase.CLOSING
    engine.pooled_drift = PooledDrift(45.0, 8.0, 3560, 40, 80.0)

    from imperium.execution.bars import Bar
    price = 200.0
    for k in range(engine.params.warmup_bars + 5):
        engine.series.add(Bar(k * 60_000, price, price * 1.001, price * 0.999,
                              price, 1000.0, closed=True))

    d = engine.evaluate()

    assert d.strategy == "overnight"
    assert d.verdict is Verdict.REJECTED
    assert "whole shares" in d.reason
    assert "$200.00" in d.reason


@pytest.mark.asyncio
async def test_admissions_follow_the_rescaled_concurrency():
    """The scaling has to reach the thing that hands out slots.

    Deriving a two-position limit and then admitting five symbols to trade
    would leave the account exactly where it started: five orders too small to
    be positions. This is the join between the arithmetic and the behaviour.
    """
    from imperium.execution.portfolio import Verdict

    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal(str(CONCENTRATES))
    session.absorb_account(
        {"equity": str(CONCENTRATES), "cash": str(CONCENTRATES),
         "currency": "USD"})

    for i, symbol in enumerate(["A", "B", "C", "D", "E", "F"]):
        session.allocator.set_scan(symbol, score=1.0 - i * 0.1,
                                   turnover=1e6, tradeable=True, reason="")
    admitted, _ = session.allocator.rebalance_admissions()

    assert len(admitted) == 3, "three slots means three symbols, not six"
    assert len(session.allocator.admitted_symbols) == 3
    # And the ones that missed out say why rather than looking rejected.
    for symbol in ["D", "E", "F"]:
        assert session.allocator.observe(symbol).verdict is not Verdict.TRADING


@pytest.mark.asyncio
async def test_growing_the_account_hands_back_the_slots():
    """The point of the whole exercise: it is supposed to grow out of this.

    Measured on the *book*, deliberately. A paper run never moves the venue's
    own balance -- no order is sent to it -- so scaling on the account record
    would freeze a paper book at its opening limits however well it did, and
    the growth this whole adjustment exists to enable would never register.
    """
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal(str(CONCENTRATES))
    session.absorb_account(
        {"equity": str(CONCENTRATES), "cash": str(CONCENTRATES),
         "currency": "USD"})
    for i, symbol in enumerate(["A", "B", "C", "D", "E", "F"]):
        session.allocator.set_scan(symbol, score=1.0 - i * 0.1,
                                   turnover=1e6, tradeable=True, reason="")
    session.allocator.rebalance_admissions()
    assert len(session.allocator.admitted_symbols) == 3

    session.broker.cash = Decimal("5000")
    await session._tick()
    session.allocator.rebalance_admissions()

    assert len(session.allocator.admitted_symbols) == 5
