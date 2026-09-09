"""The overnight drift strategy: what it claims, and what it refuses.

The published effect is 3-5bp a night on US equities (Cooper, Cliff and Gulen
measured 2.82-4.76bp on S&P 500 constituents against day returns of -2.85 to
+0.22bp). That is the same order of magnitude as one round trip, which is why
practitioner replications find it evaporates on costs and why two ETFs built to
harvest it -- NSPY and NIWM, launched 2022 -- closed within a year.

So the tests that matter here are mostly tests that the strategy declines. A
version of this module that traded more would be worse, not better, and each
test below pins one specific way that could happen by accident.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from imperium.execution.bars import Bar
from imperium.execution.broker import (
    MARKET_ON_CLOSE, MARKET_ON_OPEN, LiveBroker, PaperBroker,
)
from imperium.execution.engine import SymbolEngine
from imperium.execution.portfolio import PortfolioAllocator, Verdict
from imperium.execution.risk import RiskLimits
from imperium.strategy import overnight
from imperium.strategy.overnight import (
    MIN_NIGHTS, MOC_CUTOFF_MINUTES, MOO_CUTOFF_MINUTES, PooledDrift,
    SessionPhase, evaluate, phase_from_clock, pool, split_daily,
)
from imperium.telemetry.streams import TelemetryHub
from imperium.venues import registry


UTC = dt.timezone.utc


def daily_bars(n: int, *, overnight_bps: float, intraday_bps: float,
               sigma_bps: float = 80.0, seed: int = 7) -> list[Bar]:
    """Daily bars carrying a known overnight drift and a known intraday drift.

    Built from the two session returns rather than from a price path, so the
    thing the tests assert on is the thing the generator put in.
    """
    rng = np.random.default_rng(seed)
    bars: list[Bar] = []
    close = 100.0
    t = 0
    for _ in range(n):
        gap = overnight_bps / 10_000 + rng.normal(0, sigma_bps / 10_000)
        open_px = close * math.exp(gap)
        day = intraday_bps / 10_000 + rng.normal(0, sigma_bps / 10_000)
        close = open_px * math.exp(day)
        bars.append(Bar(t, open_px, max(open_px, close) * 1.001,
                        min(open_px, close) * 0.999, close, 1e6, closed=True))
        t += 86_400_000
    return bars


# ---------------------------------------------------------------- the split

def test_the_split_separates_the_two_sessions_it_claims_to():
    """Prevents the one error this module must not make: putting an intraday
    return into the overnight bucket.

    Overnight is close-to-open *across* bars; intraday is open-to-close within
    one. Getting them the wrong way round would report the intraday drift as
    the overnight one and the strategy would trade on the wrong number without
    anything looking broken.
    """
    # Quiet on purpose: this asserts the split is wired to the right ends of
    # the right bars, so the noise is kept well under the effect.
    bars = daily_bars(400, overnight_bps=6.0, intraday_bps=-4.0,
                      sigma_bps=10.0, seed=1)
    split = split_daily(bars)

    assert split.nights == len(bars) - 1        # n bars give n-1 seams
    assert float(np.mean(split.overnight)) * 10_000 == pytest.approx(6.0, abs=1.5)
    assert float(np.mean(split.intraday)) * 10_000 == pytest.approx(-4.0, abs=1.5)


def test_the_split_needs_two_bars_to_have_a_seam_at_all():
    assert split_daily([]).nights == 0
    assert split_daily(daily_bars(1, overnight_bps=5, intraday_bps=0)).nights == 0


# ------------------------------------------------- why pooling is required

def test_one_symbol_cannot_resolve_the_effect_but_the_universe_can():
    """The measurement that forced the pooled design, kept as a test.

    A 4bp effect against an 80bp nightly standard deviation gives a standard
    error of 80/sqrt(90) = 8.4bp on one symbol's own year of history. The
    t-statistic is under 1: the symbol cannot tell 4bp from zero, and any
    symbol that *does* look significant on its own history is being selected on
    noise. Pooling forty symbols multiplies the sample by forty and sqrt(40)
    shrinks the standard error to where the effect is visible.

    If this ever fails because the per-symbol t crosses 2, the strategy has
    started trading single-symbol noise.
    """
    splits = {}
    per_symbol_t = []
    for i in range(40):
        bars = daily_bars(90, overnight_bps=4.0, intraday_bps=0.0, seed=100 + i)
        split = split_daily(bars)
        splits[f"S{i}"] = split
        per_symbol_t.append(abs(overnight._t_stat(split.overnight)))

    # The typical symbol cannot see the effect at all...
    assert float(np.median(per_symbol_t)) < 1.0

    # ...and yet somebody always looks significant. That is the hazard, not an
    # accident of this seed: across forty independent symbols the largest
    # t-statistic crosses 2 most of the time even though every symbol was drawn
    # from the same 4bp distribution. Picking that symbol to trade is selecting
    # on noise, and it is why nothing here trades on a symbol's own history.
    assert max(per_symbol_t) > 2.0

    pooled = pool(splits)
    assert pooled.observations == 40 * 89
    assert pooled.symbols == 40
    assert pooled.mean_bps == pytest.approx(4.0, abs=1.0)
    assert pooled.t_stat > 2.0
    assert pooled.credible


def test_a_pooled_estimate_from_too_little_data_is_not_credible():
    """Prevents the strategy acting on a number it has no right to trust.

    ``credible`` is what gates trading. Three symbols with a handful of nights
    each can produce a large t-statistic by chance, so the gate is on sample
    size and breadth, not on the statistic alone.
    """
    thin = {f"S{i}": split_daily(daily_bars(8, overnight_bps=40.0,
                                            intraday_bps=0.0, seed=i))
            for i in range(3)}
    pooled = pool(thin)
    assert pooled.observations < 200
    assert not pooled.credible

    engine_signal = evaluate(daily_bars(200, overnight_bps=40.0, intraday_bps=0.0),
                             pooled=pooled)
    assert not engine_signal.eligible
    assert "cannot resolve" in engine_signal.reason


def test_shrinkage_pulls_a_lucky_symbol_back_toward_the_market():
    """Prevents sizing on a symbol's own good luck.

    Inverse-variance weighting: a symbol with 90 noisy nights carries far less
    information than 3,560 pooled symbol-nights, so its estimate barely moves
    the prior. A symbol that measured +13.6bp on its own history is traded as
    roughly the market's 4bp, not as 13.6bp.
    """
    prior = PooledDrift(mean_bps=4.0, t_stat=6.5, observations=3560, symbols=40,
                        vol_bps=80.0)
    prior_se = prior.vol_bps / math.sqrt(prior.observations)
    shrunk = overnight._shrink(13.6, 80.0 / math.sqrt(90), 4.0, prior_se)

    assert 4.0 <= shrunk < 5.0, "the symbol's own history should barely move it"
    assert shrunk < 13.6 / 2


# -------------------------------------------------------- what it refuses

def test_it_refuses_until_it_has_enough_nights():
    signal = evaluate(daily_bars(MIN_NIGHTS - 5, overnight_bps=50.0,
                                 intraday_bps=0.0))
    assert not signal.eligible
    assert "warming up" in signal.reason


def test_a_large_recent_gap_is_treated_as_an_event_not_as_drift():
    """Prevents an earnings gap being harvested as though it were drift.

    Alpaca's basic plan publishes no earnings calendar, so this is a
    statistical stand-in: a last gap beyond three sigma of the symbol's own
    overnight volatility is far likelier to be news than premium, and holding
    into the next one is taking event risk the drift does not pay for.
    """
    bars = daily_bars(120, overnight_bps=4.0, intraday_bps=0.0, seed=3)
    last = bars[-1]
    # A 12% gap up, which is many sigma of an 80bp nightly distribution.
    bars[-1] = Bar(last.open_time, last.open * 1.12, last.high * 1.12,
                   last.low * 1.12, last.close * 1.12, last.volume, closed=True)

    prior = PooledDrift(4.0, 6.5, 3560, 40, 80.0)
    signal = evaluate(bars, pooled=prior)
    assert signal.event_risk
    assert not signal.eligible
    assert "event" in signal.reason


def test_a_negative_market_drift_is_not_traded_from_the_short_side():
    """Prevents inverting the anomaly.

    The published effect is a positive overnight premium. A period where the
    pooled estimate is negative is a period where the premium is absent, not an
    invitation to short the close -- the reversal literature attributes the
    negative side to retail attention in specific names, which is not what this
    pooled estimate measures.
    """
    prior = PooledDrift(-5.0, -6.5, 3560, 40, 80.0)
    signal = evaluate(daily_bars(200, overnight_bps=-5.0, intraday_bps=0.0),
                      pooled=prior)
    assert signal.value == 0.0
    assert not signal.eligible
    assert "no premium to harvest" in signal.reason


def test_the_edge_is_capped_however_good_the_history_looks():
    """Prevents a runaway estimate sizing a position on a fluke.

    An uncapped edge feeds straight into the cost gate and then into the sizer.
    A 500bp "overnight drift" is a data error, not an opportunity.
    """
    prior = PooledDrift(600.0, 12.0, 3560, 40, 80.0)
    # The symbol's own history is ordinary; only the pooled estimate is absurd.
    # A fixture whose own last gap was 600bp would trip the event-risk branch
    # and return before the cap was ever reached, so the test would pass
    # without the cap existing.
    signal = evaluate(daily_bars(200, overnight_bps=5.0, intraday_bps=0.0),
                      pooled=prior, max_edge_bps=60.0)
    assert not signal.event_risk
    assert signal.eligible
    assert signal.shrunk_bps > 400, "the prior dominates, as it should"
    assert signal.expected_edge_bps == 60.0


# ------------------------------------------------------- the clock windows

def _clock(minutes_to_close):
    now = dt.datetime(2026, 3, 4, 19, 0, tzinfo=UTC)
    return now, now + dt.timedelta(minutes=minutes_to_close)


def test_the_entry_window_closes_before_the_venue_stops_taking_the_order():
    """Prevents lodging a market-on-close order the venue will reject.

    Alpaca refuses an MOC inside the last ten minutes of the session. An entry
    window that ran to the bell would spend its last ten minutes sending orders
    that are rejected rather than filled -- and the failure mode is silent from
    the strategy's side: it believes it is holding overnight and is flat.
    """
    now, close = _clock(5)                     # inside the venue's cutoff
    assert phase_from_clock(now, close, True) is SessionPhase.INTRADAY

    now, close = _clock(MOC_CUTOFF_MINUTES)    # the first minute that works
    assert phase_from_clock(now, close, True) is SessionPhase.CLOSING

    now, close = _clock(20)
    assert phase_from_clock(now, close, True) is SessionPhase.CLOSING

    now, close = _clock(120)                   # mid-session
    assert phase_from_clock(now, close, True) is SessionPhase.INTRADAY


def test_the_exit_window_is_before_the_open_not_after_it():
    """Prevents giving back the drift the strategy just earned.

    The trade is paid the close-to-open move. A market order sent after the
    bell has already missed it, so the exit is a market-on-open order lodged
    while the market is still shut -- and Alpaca stops accepting those two
    minutes before the open.
    """
    now = dt.datetime(2026, 3, 4, 13, 0, tzinfo=UTC)

    def phase_when_open_is_in(minutes):
        return phase_from_clock(now, None, False,
                                next_open=now + dt.timedelta(minutes=minutes))

    assert phase_when_open_is_in(30) is SessionPhase.PREOPEN

    # Written as a literal rather than as MOO_CUTOFF_MINUTES - 1: a bound
    # expressed in terms of the constant under test moves with it, and the
    # test then passes with the cutoff set to zero.
    assert phase_when_open_is_in(1) is SessionPhase.CLOSED
    assert MOO_CUTOFF_MINUTES > 1, "the venue's own cutoff is two minutes"

    assert phase_when_open_is_in(20 * 60) is SessionPhase.CLOSED


def test_an_unknown_clock_never_produces_a_trading_window():
    """A missing clock must fail closed. Guessing the phase from a local
    calendar is how a program trades into an early close."""
    now = dt.datetime(2026, 3, 4, 19, 0, tzinfo=UTC)
    assert phase_from_clock(now, None, True) is SessionPhase.INTRADAY
    assert phase_from_clock(now, None, False) is SessionPhase.CLOSED


# --------------------------------------------------- the engine's decision

def _engine(symbol="AAPL", *, limits=None):
    limits = limits or RiskLimits()
    allocator = PortfolioAllocator(limits)
    allocator.equity = 100_000.0
    allocator.cash = 100_000.0
    engine = SymbolEngine(symbol, registry.get(registry.DEFAULT_VENUE), limits,
                          allocator, TelemetryHub())
    allocator.observe(symbol).admitted = True
    return engine


def _warm(engine, price=100.0):
    """Enough minute bars to clear warmup, so the overnight branch is reached."""
    t = 0
    for _ in range(engine.params.warmup_bars + 5):
        engine.series.add(Bar(t, price, price * 1.001, price * 0.999, price,
                              1000.0, closed=True))
        t += 60_000
    return engine


def test_the_overnight_branch_is_only_taken_for_equities_in_the_closing_window():
    """Prevents the overnight trade firing on crypto or mid-session.

    Crypto has no overnight session to decompose -- it never closes, so there is
    no close-to-open move and nothing to harvest. Mid-session the intraday
    blend owns the book, and running both would allocate the same capital
    twice.
    """
    crypto = _warm(_engine("BTC/USD"))
    crypto.session_phase = SessionPhase.CLOSING
    crypto.pooled_drift = PooledDrift(4.0, 6.5, 3560, 40, 80.0)
    assert crypto.evaluate().strategy == "intraday"

    equity = _warm(_engine("AAPL"))
    equity.session_phase = SessionPhase.INTRADAY
    assert equity.evaluate().strategy == "intraday"

    equity.session_phase = SessionPhase.CLOSING
    assert equity.evaluate().strategy == "overnight"


def test_the_overnight_trade_pays_the_same_cost_gate_as_everything_else():
    """The single most important property of this strategy.

    The drift is a few basis points; a US equity round trip is a few basis
    points. Exempting the overnight trade from the cost gate would produce a
    strategy that trades every night and loses slowly, which is exactly the
    outcome the published replications describe. The refusal is the correct
    answer.
    """
    engine = _warm(_engine("AAPL"))
    engine.session_phase = SessionPhase.CLOSING
    engine.daily_bars = daily_bars(200, overnight_bps=4.0, intraday_bps=0.0)
    engine.pooled_drift = PooledDrift(4.0, 6.5, 3560, 40, 80.0)
    engine.set_book(99.99, 100.01)

    d = engine.evaluate()
    assert d.strategy == "overnight"
    assert d.verdict is Verdict.REJECTED
    assert d.expected_edge_bps < d.required_bps
    assert "does not clear" in d.reason
    assert d.target_weight == 0.0


def test_a_drift_that_clears_its_cost_is_entered_on_the_closing_auction():
    """The other side of the same gate: when the edge is genuinely larger than
    the cost, the trade is taken -- and taken as a market-on-close order, which
    is what makes it an overnight trade rather than a late intraday one."""
    engine = _warm(_engine("AAPL"))
    engine.session_phase = SessionPhase.CLOSING
    engine.daily_bars = daily_bars(200, overnight_bps=45.0, intraday_bps=0.0)
    engine.pooled_drift = PooledDrift(45.0, 8.0, 3560, 40, 80.0)
    engine.set_book(99.995, 100.005)

    d = engine.evaluate()
    assert d.verdict is Verdict.TRADING, d.reason
    assert d.expected_edge_bps > d.required_bps
    assert d.target_weight > 0
    assert d.entry_order == MARKET_ON_CLOSE


def test_the_overnight_position_is_sized_on_gap_risk_not_on_an_atr_stop():
    """Prevents sizing a stopless position as though it had a stop.

    A gap opens *through* a stop without ever touching it, so the loss that
    bounds this position is the tail of the overnight distribution, and the
    sizer must react to that distribution widening. An ATR-based sizer would
    not: ATR is measured inside the session, and the whole risk here is the
    part of the move that happens while the session is shut.

    Measured on the raw sized weight rather than the target, because the
    portfolio clamp caps a quiet symbol at the per-symbol budget and would hide
    the sizer's own arithmetic behind a flat ceiling.
    """
    def sized(sigma_bps):
        engine = _warm(_engine("AAPL"))
        engine.session_phase = SessionPhase.CLOSING
        engine.daily_bars = daily_bars(300, overnight_bps=45.0, intraday_bps=0.0,
                                       sigma_bps=sigma_bps, seed=11)
        engine.pooled_drift = PooledDrift(45.0, 8.0, 3560, 40, 80.0)
        engine.set_book(99.995, 100.005)
        d = engine.evaluate()
        assert d.verdict is Verdict.TRADING, d.reason
        return d

    calm, rough = sized(120.0), sized(240.0)

    # Twice the overnight volatility, half the position.
    assert rough.raw_weight == pytest.approx(calm.raw_weight / 2, rel=0.05)
    # And it says so, naming the bound that actually applied.
    assert "3-sigma gap" in calm.sizing_reason
    assert "no stop behind an overnight hold" in calm.sizing_reason


def test_an_overnight_entry_is_not_blocked_by_the_pattern_day_trader_ceiling():
    """A real structural advantage, and the reason it is worth encoding.

    The PDT rule counts a purchase and a sale of the same security within one
    session. Entering on one close and exiting on the next open is not that, so
    a sub-$25,000 account can run this every night without approaching the
    ceiling -- while the intraday strategy on the same book stays blocked.
    """
    limits = RiskLimits()
    allocator = PortfolioAllocator(limits)
    allocator.equity = 10_000.0                      # under the $25k floor
    allocator.cash = 10_000.0
    allocator.day_trade_count = limits.pdt_max_day_trades
    allocator.observe("AAPL").admitted = True
    assert allocator.pdt_blocked()

    intraday = allocator.clamp("AAPL", 0.05)
    assert intraday.binding == "pattern day trader"
    assert intraday.weight == 0.0

    overnight_clamp = allocator.clamp("AAPL", 0.05, overnight=True)
    assert overnight_clamp.binding != "pattern day trader"
    assert overnight_clamp.weight > 0


# ------------------------------------------------------------ the orders

@pytest.mark.asyncio
async def test_an_auction_order_never_reaches_a_crypto_symbol():
    """Prevents a guaranteed venue rejection.

    ``cls`` and ``opg`` are US-equity time-in-force codes. Sent on a crypto
    symbol Alpaca rejects the order outright, and the strategy would believe it
    held a position it does not hold.
    """
    sent = []

    class _Client:
        @staticmethod
        def new_client_order_id(prefix):
            return prefix + "-1"

        async def asset(self, symbol):
            return None

        async def place_order(self, symbol, side, **kw):
            sent.append((symbol, kw.get("time_in_force")))
            return {"id": "x", "client_order_id": "c", "filled_qty": "1",
                    "filled_avg_price": "100"}

    broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), _Client(), "k")
    broker.arm("GO LIVE", True)

    assert await broker.apply_target("BTC/USD", 0.1, 100.0, 100_000.0,
                                     order=MARKET_ON_CLOSE) is None
    assert sent == []

    await broker.apply_target("AAPL", 0.1, 100.0, 100_000.0,
                              order=MARKET_ON_CLOSE)
    assert sent == [("AAPL", "cls")]

    await broker.apply_target("AAPL", 0.0, 100.0, 100_000.0,
                              order=MARKET_ON_OPEN)
    assert sent[-1] == ("AAPL", "opg")


@pytest.mark.asyncio
async def test_an_auction_order_is_rounded_down_to_whole_shares():
    """Auctions take whole shares only, and rounding up would buy more than
    the sizer allowed."""
    sent = []

    class _Client:
        @staticmethod
        def new_client_order_id(prefix):
            return prefix + "-1"

        async def asset(self, symbol):
            return None

        async def place_order(self, symbol, side, *, qty, **kw):
            sent.append(qty)
            return {"id": "x", "client_order_id": "c", "filled_qty": str(qty),
                    "filled_avg_price": "100"}

    broker = LiveBroker(registry.get(registry.DEFAULT_VENUE), _Client(), "k")
    broker.arm("GO LIVE", True)
    # 0.107 of a $100,000 book at $100 is 107.0 shares; make it fractional.
    await broker.apply_target("AAPL", 0.10707, 100.0, 100_000.0,
                              order=MARKET_ON_CLOSE)
    assert sent and sent[0] == sent[0].to_integral_value()
    assert float(sent[0]) == 107.0


# ----------------------------------------------------- the session wiring

"""The strategy is only real if the session actually feeds it.

Everything above tests a module that could be perfectly correct and never
run. These test the wiring: that daily history is fetched as daily history,
that one pooled estimate reaches every engine, that the phase is read from the
venue's clock rather than a local calendar, and that a position carried through
a close has an exit lodged before the next open.
"""

import contextlib
import time

from imperium.session import TradingSession
from imperium.venues.alpaca.client import MarketClock
from mock_venue import KEY, SECRET, MockVenue


@contextlib.asynccontextmanager
async def _session(venue: MockVenue):
    from imperium.venues.alpaca.client import AlpacaClient

    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    session.universe = ["AAPL", "SPY", "BTC/USD"]
    try:
        yield session
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_the_session_fetches_daily_history_as_daily_history():
    """Prevents the failure that produced 14 nights out of 80.

    The first version of this read the minute ring, which holds a few sessions,
    and reported a confident-looking estimate from a fraction of the history it
    believed it had. The fix is to ask the venue for ``1Day`` bars, so that is
    what is asserted -- on the wire, not on the result.
    """
    venue = MockVenue()
    async with _session(venue) as session:
        await session.refresh_daily_history(force=True)

    timeframes = [dict(r.url.params).get("timeframe")
                  for r in venue.requests if r.url.path.endswith("/bars")]
    assert timeframes and all(tf == "1Day" for tf in timeframes)

    # Crypto daily bars *are* fetched -- the trend strategy runs on them, and
    # crypto is the one market a small account can trade continuously. What
    # crypto must never do is enter the overnight decomposition: a market that
    # does not close has no close-to-open seam to measure.
    asked = ",".join(dict(r.url.params).get("symbols", "")
                     for r in venue.requests if r.url.path.endswith("/bars"))
    assert "AAPL" in asked and "BTC/USD" in asked


@pytest.mark.asyncio
async def test_every_engine_reads_the_same_pooled_estimate():
    """Prevents a per-symbol prior creeping back in.

    There is one market and one measurement of it. An engine holding its own
    estimate would be an engine trading its own noise, which the power
    measurement above shows it cannot resolve.
    """
    venue = MockVenue()
    async with _session(venue) as session:
        await session.refresh_daily_history(force=True)

        assert session.pooled_drift is not None
        # Two equities. BTC/USD is in the universe and has daily bars, but a
        # market that never closes contributes no overnight observations.
        assert session.pooled_drift.symbols == 2          # AAPL and SPY
        # The mock bakes a known drift into its daily bars; recovering it
        # proves the decomposition survived the round trip through the wire
        # format, not just the unit test's own generator.
        assert session.pooled_drift.mean_bps == pytest.approx(
            venue.daily_overnight_bps, abs=0.2)
        assert session.pooled_drift.intraday_bps == pytest.approx(
            venue.daily_intraday_bps, abs=0.2)

        for symbol in ("AAPL", "SPY"):
            assert session.engines[symbol].pooled_drift is session.pooled_drift
            assert session.engines[symbol].daily_bars


@pytest.mark.asyncio
async def test_the_session_phase_comes_from_the_venue_clock():
    """Prevents a local calendar deciding when to lodge an auction order.

    An early close or a holiday computed locally is a day the strategy sends a
    market-on-close order into a market that has already closed. The venue's
    clock is the only thing that knows.
    """
    venue = MockVenue()
    async with _session(venue) as session:
        now = dt.datetime.now(UTC)

        session.market_clock = MarketClock(
            is_open=True, next_close=now + dt.timedelta(minutes=18))
        assert session._update_session_phase() is SessionPhase.CLOSING

        # The same wall-clock instant, but the venue says it closes in four
        # hours. Nothing local changed; the decision did.
        session.market_clock = MarketClock(
            is_open=True, next_close=now + dt.timedelta(hours=4))
        assert session._update_session_phase() is SessionPhase.INTRADAY

        session.market_clock = MarketClock(
            is_open=False, next_open=now + dt.timedelta(minutes=25))
        assert session._update_session_phase() is SessionPhase.PREOPEN

        # And it reaches the engines, which is where it is acted on.
        session.engine("AAPL")
        session._update_session_phase()
        assert session.engines["AAPL"].session_phase is SessionPhase.PREOPEN


@pytest.mark.asyncio
async def test_an_overnight_hold_has_its_exit_lodged_before_the_open():
    """Prevents the position with nothing behind it.

    An entry recorded and an exit never sent is a position held indefinitely by
    a strategy that believes it is flat. The exit goes in during the pre-open
    window as a market-on-open order, because the drift is paid at the opening
    print and a market order sent after the bell has already missed it.
    """
    venue = MockVenue()
    async with _session(venue) as session:
        session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
        session.feed.quote("AAPL").last = 100.0
        await session.broker.apply_target("AAPL", 0.1, 100.0, 10_000.0)
        session.overnight_holdings["AAPL"] = 0.1
        assert not session.broker.positions["AAPL"].is_flat

        session.market_clock = MarketClock(
            is_open=False, next_open=dt.datetime.now(UTC) + dt.timedelta(minutes=25))
        await session._tick()

        assert session.broker.positions["AAPL"].is_flat
        assert "AAPL" not in session.overnight_holdings
        assert session.broker.fills[-1].side == "SELL"
        assert "market-on-open" in session.broker.fills[-1].note


@pytest.mark.asyncio
async def test_nothing_is_exited_at_the_open_that_was_not_entered_for_the_night():
    """Prevents adopting an unrelated position.

    "Holds an equity while the market is shut" is not the same fact as "was
    entered on last night's close". An intraday position that failed to flatten
    is a problem to report, not a position for this strategy to quietly close
    into the opening auction as though it had planned it.
    """
    venue = MockVenue()
    async with _session(venue) as session:
        session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
        session.feed.quote("SPY").last = 200.0
        await session.broker.apply_target("SPY", 0.1, 200.0, 10_000.0)
        assert session.overnight_holdings == {}

        session.market_clock = MarketClock(
            is_open=False, next_open=dt.datetime.now(UTC) + dt.timedelta(minutes=25))
        await session._tick()

        assert not session.broker.positions["SPY"].is_flat


@pytest.mark.asyncio
async def test_the_snapshot_shows_the_drift_next_to_what_it_costs():
    """The panel must not be able to show an edge without its cost.

    A few basis points a night reads as free money on its own and as what it is
    beside a round trip of the same order. The two travel together in the
    snapshot so the UI cannot render one without the other.
    """
    venue = MockVenue()
    async with _session(venue) as session:
        await session.refresh_daily_history(force=True)
        snap = session.snapshot()

    o = snap["overnight"]
    assert o["measured"]
    assert o["mean_bps"] == pytest.approx(venue.daily_overnight_bps, abs=0.2)
    assert o["entry_order"] == MARKET_ON_CLOSE
    assert o["exit_order"] == MARKET_ON_OPEN
    assert o["exempt_from_pdt"] is True
    assert snap["costs"]["by_asset_class"]["us_equity"]["commission_bps"] is not None


@pytest.mark.asyncio
async def test_the_session_carries_the_auction_order_from_decision_to_broker():
    """Prevents the two ways the handoff can silently degrade the trade.

    The engine decides "market-on-close" and the broker is what sends it, so
    everything between them has to carry the order type. Drop it and the
    session places an ordinary market order in the last quarter hour of the
    session -- a real position, a real cost, and none of the strategy: the
    fill happens at whatever the market is doing at 15:45 rather than at the
    closing print the drift is measured against.

    And the hold has to be recorded on submission. An entry the session does
    not remember is an entry with no exit behind it: nothing lodges the
    market-on-open order, and the position simply stays on the book.
    """
    venue = MockVenue()
    async with _session(venue) as session:
        session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
        q = session.feed.quote("AAPL")
        q.last, q.updated_at = 100.0, time.time()

        engine = session.engine("AAPL")
        decision = engine.decision
        decision.symbol = "AAPL"
        decision.verdict = Verdict.TRADING
        decision.target_weight = 0.1
        decision.strategy = "overnight"
        decision.entry_order = MARKET_ON_CLOSE

        await session._act_on(decision)

        assert session.overnight_holdings.get("AAPL") == 0.1
        fill = session.broker.fills[-1]
        assert fill.side == "BUY"
        assert MARKET_ON_CLOSE in fill.note


@pytest.mark.asyncio
async def test_an_overnight_book_survives_closing_the_application(tmp_path,
                                                                  monkeypatch):
    """Prevents the failure that a desktop application makes routine.

    Entering at the close and exiting at the next open means the application is
    normally *shut* in between -- the operator closes the laptop and reopens it
    before the bell. An in-memory record of which positions are being carried
    would be gone by then, no opening exit would be lodged, and a real position
    would be left with nothing managing it.
    """
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    venue = MockVenue()

    # Driven through the real entry path rather than by calling the save
    # method: an earlier version of this test saved and loaded by hand, so
    # deleting either call site left it passing while the feature was gone.
    async with _session(venue) as first:
        first.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
        q = first.feed.quote("AAPL")
        q.last, q.updated_at = 100.0, time.time()
        d = first.engine("AAPL").decision
        d.symbol, d.verdict, d.target_weight = "AAPL", Verdict.TRADING, 0.08
        d.entry_order = MARKET_ON_CLOSE
        await first._act_on(d)

    assert (tmp_path / "state.json").exists()

    # A completely new session object, as a restarted process builds, brought
    # up through start() so the recovery has to be wired into it.
    async with _session(venue) as second:
        assert second.overnight_holdings == {}
        await second.start()
        try:
            assert second.overnight_holdings == {"AAPL": 0.08}
        finally:
            await second.stop()


@pytest.mark.asyncio
async def test_a_corrupt_state_file_does_not_stop_the_session_starting(
        tmp_path, monkeypatch):
    """A state file is a convenience. Refusing to start because one is
    unreadable turns a lost note into an outage."""
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "state.json").write_text("{not json at all", encoding="utf-8")

    async with _session(MockVenue()) as session:
        session._load_overnight_state()          # must not raise
        assert session.overnight_holdings == {}


@pytest.mark.asyncio
async def test_an_equity_held_through_the_close_by_nothing_is_reported(tmp_path,
                                                                      monkeypatch):
    """Prevents a position with nothing managing it going unmentioned.

    An equity held while the market is shut that this strategy did not enter is
    either an orphaned overnight hold or an intraday position that failed to
    flatten. Both are unmanaged, both are the operator's call, and silence is
    the one wrong answer. It is said once per window, not once per tick.
    """
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    venue = MockVenue()
    async with _session(venue) as session:
        session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
        session.feed.quote("SPY").last = 200.0
        await session.broker.apply_target("SPY", 0.1, 200.0, 10_000.0)

        session.market_clock = MarketClock(
            is_open=False,
            next_open=dt.datetime.now(UTC) + dt.timedelta(minutes=25))
        await session._tick()
        await session._tick()
        await session._tick()

        warnings = [e for e in session.telemetry.events(200)
                    if e["source"] == "overnight" and "SPY" in e["message"]]
        assert len(warnings) == 1, "said once per window, not once per tick"
        assert "no opening exit will be lodged" in warnings[0]["message"]
        # And it is still not exited on the strategy's own authority.
        assert not session.broker.positions["SPY"].is_flat


@pytest.mark.asyncio
async def test_the_unmanaged_warning_returns_on_the_next_morning(tmp_path,
                                                                 monkeypatch):
    """Said once per window, but said again the next window.

    Deduplicating without ever clearing would report an unmanaged position on
    the first morning and stay silent every morning after, which is worse than
    not deduplicating at all: the operator would read the silence as the
    position having been dealt with.
    """
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    async with _session(MockVenue()) as session:
        session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
        session.feed.quote("SPY").last = 200.0
        await session.broker.apply_target("SPY", 0.1, 200.0, 10_000.0)

        def warnings():
            return [e for e in session.telemetry.events(200)
                    if e["source"] == "overnight" and "SPY" in e["message"]]

        preopen = MarketClock(
            is_open=False, next_open=dt.datetime.now(UTC) + dt.timedelta(minutes=25))
        session.market_clock = preopen
        await session._tick()
        await session._tick()
        assert len(warnings()) == 1

        # The session opens, runs the day, and closes again.
        session.market_clock = MarketClock(
            is_open=True, next_close=dt.datetime.now(UTC) + dt.timedelta(hours=4))
        await session._tick()

        session.market_clock = preopen
        await session._tick()
        assert len(warnings()) == 2, "a new morning is a new warning"


@pytest.mark.asyncio
async def test_a_symbol_the_scanner_admits_later_is_not_stranded(tmp_path,
                                                                 monkeypatch):
    """Prevents a refusal that describes the wiring rather than the market.

    The universe is scanned continuously, so an equity can be admitted hours
    after the last daily-history pull. The refresh interval exists to stop this
    spending request budget to learn nothing -- daily bars change once a day --
    but a symbol with no history at all is not "nothing to learn". Stranded, it
    would carry no overnight sample and refuse every night for up to six hours
    with "warming up", which is a statement about this method, not about the
    symbol.
    """
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    venue = MockVenue()
    async with _session(venue) as session:
        await session.refresh_daily_history(force=True)
        first_pull = len([r for r in venue.requests if r.url.path.endswith("/bars")])
        assert session.engines["AAPL"].daily_bars

        # Nothing changed: the interval holds and no request is made.
        await session.refresh_daily_history()
        assert len([r for r in venue.requests
                    if r.url.path.endswith("/bars")]) == first_pull

        # The scanner admits a new equity. That one has no history at all.
        session.universe = ["AAPL", "SPY", "BTC/USD", "HARD"]
        await session.refresh_daily_history()

        assert len([r for r in venue.requests
                    if r.url.path.endswith("/bars")]) > first_pull
        assert session.engines["HARD"].daily_bars
        assert session.engines["HARD"].pooled_drift is session.pooled_drift


@pytest.mark.asyncio
async def test_an_engine_built_after_a_refresh_starts_from_the_same_estimate():
    """Prevents a lazily created engine trading in the dark.

    Engines are built on demand. One created after the last refresh would hold
    no market estimate and no session phase, so it would refuse the overnight
    trade for a reason that has nothing to do with the market -- and, worse,
    would run the intraday path during the closing window because its phase
    still said CLOSED.
    """
    venue = MockVenue()
    async with _session(venue) as session:
        await session.refresh_daily_history(force=True)
        session.market_clock = MarketClock(
            is_open=True,
            next_close=dt.datetime.now(UTC) + dt.timedelta(minutes=18))
        assert session._update_session_phase() is SessionPhase.CLOSING

        fresh = session.engine("NVDA")           # never seen before now
        assert fresh.pooled_drift is session.pooled_drift
        assert fresh.session_phase is SessionPhase.CLOSING


async def _closing_session(venue, symbol="AAPL"):
    """A session in the closing window holding a position in ``symbol``."""
    from imperium.venues.alpaca.client import AlpacaClient

    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    session.universe = [symbol, "BTC/USD"]
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    q = session.feed.quote(symbol)
    q.last, q.bid, q.ask, q.updated_at = 100.0, 99.99, 100.01, time.time()
    await session.broker.apply_target(symbol, 0.1, 100.0, 10_000.0)
    session.market_clock = MarketClock(
        is_open=True, next_close=dt.datetime.now(UTC) + dt.timedelta(minutes=18))
    return session


@pytest.mark.asyncio
async def test_a_position_the_night_does_not_want_is_closed_not_carried(
        tmp_path, monkeypatch):
    """Prevents an intraday position becoming an accidental overnight one.

    _act_on returns early on any verdict that is not TRADING, so a refusal
    never reduces a position. During the session that is right -- "no new
    exposure" is not "sell what you have". At the close it is wrong: the
    position is carried through the night, sized against an intraday
    distribution and stopped by an ATR stop, and a gap goes through both.

    The overnight strategy has just answered exactly this question and said no.
    Acting on the no is the point of asking it.
    """
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    venue = MockVenue()
    session = await _closing_session(venue)
    try:
        engine = session.engine("AAPL")
        engine.daily_bars = daily_bars(200, overnight_bps=4.0, intraday_bps=0.0)
        engine.pooled_drift = PooledDrift(4.0, 6.5, 3560, 40, 80.0)
        engine.session_phase = SessionPhase.CLOSING
        _warm(engine)
        engine.set_book(99.99, 100.01)

        d = engine.evaluate()
        assert d.strategy == "overnight" and d.verdict is Verdict.REJECTED
        assert not session.broker.positions["AAPL"].is_flat

        await session._tick()

        assert session.broker.positions["AAPL"].is_flat
        assert session.broker.fills[-1].side == "SELL"
        assert MARKET_ON_CLOSE in session.broker.fills[-1].note
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_a_position_the_night_does_want_is_left_alone(tmp_path, monkeypatch):
    """The other side: a symbol the strategy is deliberately holding must not
    be closed by the same sweep that closes the ones it declined."""
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    venue = MockVenue()
    session = await _closing_session(venue)
    try:
        session.overnight_holdings["AAPL"] = 0.1
        engine = session.engine("AAPL")
        engine.session_phase = SessionPhase.CLOSING
        engine.decision.strategy = "overnight"
        engine.decision.verdict = Verdict.REJECTED   # a later, weaker read

        await session._tick()

        assert not session.broker.positions["AAPL"].is_flat
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_a_stale_intraday_verdict_is_not_treated_as_an_overnight_answer(
        tmp_path, monkeypatch):
    """Prevents flattening on a verdict that answered a different question.

    An engine that has not yet evaluated inside the closing window still holds
    an intraday decision. "The intraday blend sees no edge right now" is not
    "this is not worth holding overnight", and closing a position on it would
    be acting on an answer to a question nobody asked.
    """
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    venue = MockVenue()
    session = await _closing_session(venue)
    try:
        engine = session.engine("AAPL")
        engine.decision.strategy = "intraday"
        engine.decision.verdict = Verdict.REJECTED

        await session._tick()

        assert not session.broker.positions["AAPL"].is_flat
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_crypto_is_never_closed_for_a_session_that_does_not_end(
        tmp_path, monkeypatch):
    """Crypto has no close to flatten into. Applying an equity's session
    boundary to it would liquidate the book once a day for no reason."""
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    venue = MockVenue()
    session = await _closing_session(venue, symbol="BTC/USD")
    try:
        engine = session.engine("BTC/USD")
        engine.decision.strategy = "overnight"      # cannot happen, but pin it
        engine.decision.verdict = Verdict.REJECTED

        await session._tick()

        assert not session.broker.positions["BTC/USD"].is_flat
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_a_symbol_the_strategy_wants_is_not_closed_by_the_same_sweep(
        tmp_path, monkeypatch):
    """The TRADING guard, tested on its own.

    A symbol can want to be held without yet appearing in the holdings map --
    the decision is taken on a closed bar and recorded when the order goes out.
    In that gap the only thing standing between an intended overnight hold and
    the sweep that closes unwanted ones is the verdict check, so it is tested
    with the holdings map deliberately empty. An earlier version of this file
    asserted the same property through a symbol that was *also* in the holdings
    map, which meant the verdict check could be deleted with everything green.
    """
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    session = await _closing_session(MockVenue())
    try:
        assert session.overnight_holdings == {}
        engine = session.engine("AAPL")
        engine.session_phase = SessionPhase.CLOSING
        engine.decision.strategy = "overnight"
        engine.decision.verdict = Verdict.TRADING
        engine.decision.target_weight = 0.1

        await session._tick()

        assert not session.broker.positions["AAPL"].is_flat
    finally:
        await session.detach_client()


@pytest.mark.asyncio
async def test_a_venue_timestamp_without_an_offset_still_compares(monkeypatch):
    """Prevents the whole strategy failing silently on a parsing detail.

    datetime.fromisoformat returns a *naive* datetime for a string with no UTC
    offset. Alpaca documents an offset, but a naive value reaching the phase
    calculation is not a small inaccuracy: it is compared against an aware
    now(), which raises TypeError instead of returning a wrong answer. Inside
    the trading loop that exception is caught and logged once a second, the
    session phase never advances past whatever it was, and the overnight
    strategy never fires -- with nothing on screen saying why.
    """
    from imperium.venues.alpaca.client import AlpacaClient

    venue = MockVenue()

    def naive_clock(request):
        now = dt.datetime.now(UTC)
        return venue._json({
            # No offset, and no trailing Z.
            "timestamp": now.replace(tzinfo=None).isoformat(),
            "is_open": True,
            "next_open": (now + dt.timedelta(hours=8)).replace(tzinfo=None).isoformat(),
            "next_close": (now + dt.timedelta(minutes=18)).replace(tzinfo=None).isoformat(),
        })

    venue.fail_next.append(naive_clock)
    client = AlpacaClient(KEY, SECRET, paper=True, transport=venue.transport)
    try:
        clock = await client.get_clock()
    finally:
        await client.aclose()

    assert clock.next_close is not None
    assert clock.next_close.tzinfo is not None, "a naive value poisons every comparison"

    session = TradingSession()
    session.market_clock = clock
    assert session._update_session_phase() is SessionPhase.CLOSING
