"""Multi-day time-series momentum: the horizon a small account can trade.

The argument for this strategy is regulatory before it is statistical. Under
FINRA's pattern-day-trader rule a margin account below $25,000 may make three
day trades in five business days. An intraday strategy on such an account is
not constrained, it is prevented: a position it cannot close the same day is
not an intraday position at all. A trade carried across a session close is not
a day trade, so the multi-day horizon removes the binding constraint rather
than working around it.

The second argument is arithmetic. A round trip is paid once per holding
period, so the cost per unit of time falls as the period grows. Every test here
pins one half of that: that the horizon is real, or that the edge is measured
honestly enough to justify committing to it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imperium.execution.bars import Bar
from imperium.strategy import trend
from imperium.strategy.trend import (
    CRYPTO_TREND, EQUITY_TREND, MAX_DAILY_DRIFT_BPS, MIN_HOLD_DAYS,
    MIN_POOLED_SYMBOLS, PooledTrend, TrendPhase, evaluate, minimum_holding_days,
    momentum_score, phase_for, pool, spec_for,
)
from imperium.venues.assets import AssetClass


def daily(n: int, *, drift_per_day: float = 0.0, sigma: float = 0.02,
          seed: int = 3, start: float = 50.0) -> list[Bar]:
    rng = np.random.default_rng(seed)
    price, bars = start, []
    for k in range(n):
        price *= float(np.exp(drift_per_day + rng.normal(0, sigma)))
        bars.append(Bar(k * 86_400_000, price, price * 1.01, price * 0.99,
                        price, 1e6, closed=True))
    return bars


def trending(n: int, beta_bps: float, lookback: int = 21, *, seed: int = 1,
             sigma: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """A price path whose next-day return really does depend on its own trend."""
    rng = np.random.default_rng(seed)
    closes, scores, forward = [100.0], [], []
    for _ in range(n):
        arr = np.array(closes)
        score = momentum_score(arr, lookback) if arr.size > lookback + 2 else np.nan
        drift = (beta_bps / 10_000.0) * score if math.isfinite(score) else 0.0
        step = drift + rng.normal(0, sigma)
        closes.append(closes[-1] * float(np.exp(step)))
        if math.isfinite(score):
            scores.append(score)
            forward.append(step)
    return np.array(scores), np.array(forward)


# ------------------------------------------------------ the holding period

def test_the_holding_period_is_what_makes_the_cost_affordable():
    """The whole argument for this strategy, as arithmetic.

    A round trip is paid once per holding period. Crypto pays around 50bp and
    a US equity 2-4bp, so at the same drift the two need very different
    commitments -- and a strategy that does not know that will open a crypto
    position it cannot afford to close.
    """
    # 20bp a day against a 50bp crypto round trip, 1.5x safety.
    assert minimum_holding_days(50.0, 20.0, 1.5) == pytest.approx(3.75)
    # The same drift against a 4bp equity round trip.
    assert minimum_holding_days(4.0, 20.0, 1.5) == pytest.approx(0.3)
    # Halving the drift doubles the commitment.
    assert minimum_holding_days(50.0, 10.0, 1.5) == pytest.approx(7.5)


def test_no_holding_period_redeems_a_trade_with_no_edge():
    assert minimum_holding_days(50.0, 0.0, 1.5) == float("inf")
    assert minimum_holding_days(50.0, -5.0, 1.5) == float("inf")


def test_the_plan_is_never_shorter_than_one_session():
    """Prevents the strategy quietly becoming the thing it exists to avoid.

    A strong trend against a cheap round trip solves to a fraction of a day.
    Acting on that would open and close inside one session -- a day trade, on
    an account whose inability to make day trades is the entire reason this
    strategy was selected.
    """
    pooled = PooledTrend(beta_bps=30.0, t_stat=8.0, observations=5000, symbols=40)
    signal = evaluate(daily(200, drift_per_day=0.002),
                      asset_class=AssetClass.US_EQUITY, pooled=pooled,
                      round_trip_bps=2.0, safety_multiple=1.5)

    assert signal.eligible
    # The raw arithmetic really is shorter than a day...
    raw = minimum_holding_days(2.0, signal.drift_bps_per_day, 1.5)
    assert raw < 1.0
    # ...and the plan is floored anyway.
    assert signal.min_hold_days == MIN_HOLD_DAYS
    # The edge collected is a full day's drift, not a fraction of one.
    assert signal.expected_edge_bps == pytest.approx(signal.drift_bps_per_day)


def test_a_trend_too_slow_to_pay_for_itself_is_refused():
    """A real trend that needs sixty days to cover a crypto round trip is not
    an opportunity at a horizon this estimate is good for."""
    pooled = PooledTrend(beta_bps=1.0, t_stat=3.0, observations=5000, symbols=40)
    signal = evaluate(daily(200, drift_per_day=0.0005, sigma=0.03),
                      asset_class=AssetClass.CRYPTO, pooled=pooled,
                      round_trip_bps=58.0, safety_multiple=1.5,
                      planned_hold_days=40)

    assert not signal.eligible
    assert "too slow to pay for itself" in signal.reason


# ------------------------------------------------------- the pooled premium

def test_the_pooled_estimate_recovers_a_premium_one_symbol_cannot_see():
    """The same power problem the overnight module measured, and the same fix.

    A single symbol's own history cannot separate a few basis points a day from
    noise. Worse, across forty symbols several will cross a t of 2 by chance
    even when the true premium is exactly zero -- so a strategy that picked its
    best-looking symbol would be selecting on noise every time.
    """
    scored = {f"S{i}": trending(300, 10.0, seed=100 + i) for i in range(40)}
    pooled = pool(scored)

    assert pooled.symbols == 40
    assert pooled.observations > 5_000
    assert pooled.beta_bps > 0
    assert pooled.t_stat > 3.0
    assert pooled.credible

    per_symbol = []
    for scores, forward in scored.values():
        one = pool({"one": (scores, forward)})
        per_symbol.append(abs(one.t_stat))
    assert float(np.median(per_symbol)) < 2.0, "the typical symbol sees nothing"


def test_a_premium_that_is_not_there_is_not_found():
    """The test that matters most. An estimator that reports an edge on random
    data would have this book trading noise with real money."""
    scored = {f"S{i}": trending(300, 0.0, seed=500 + i) for i in range(40)}
    pooled = pool(scored)

    assert abs(pooled.t_stat) < 2.5
    assert not pooled.credible


def test_an_uncredible_premium_is_never_traded():
    thin = {f"S{i}": trending(120, 40.0, seed=i) for i in range(3)}
    pooled = pool(thin)
    assert pooled.symbols < MIN_POOLED_SYMBOLS
    assert not pooled.credible

    signal = evaluate(daily(200, drift_per_day=0.002),
                      asset_class=AssetClass.US_EQUITY, pooled=pooled,
                      round_trip_bps=4.0)
    assert not signal.eligible
    assert "not measurable yet" in signal.reason


def test_no_pooled_estimate_at_all_means_no_trade():
    signal = evaluate(daily(200, drift_per_day=0.002),
                      asset_class=AssetClass.US_EQUITY, pooled=None,
                      round_trip_bps=4.0)
    assert not signal.eligible


def test_the_estimated_drift_is_capped():
    """An uncapped premium feeds straight into the cost gate and the sizer. A
    200bp-a-day trend is a data error, not an opportunity."""
    pooled = PooledTrend(beta_bps=500.0, t_stat=9.0, observations=9000, symbols=40)
    signal = evaluate(daily(200, drift_per_day=0.003),
                      asset_class=AssetClass.US_EQUITY, pooled=pooled,
                      round_trip_bps=4.0)
    assert signal.drift_bps_per_day <= MAX_DAILY_DRIFT_BPS


# --------------------------------------------------------------- the signal

def test_the_lookbacks_differ_by_asset_class_because_the_evidence_does():
    """Not a tuned parameter. Moskowitz, Ooi and Pedersen measure
    equity-like instruments at one to twelve months; Liu and Tsyvinski find
    crypto momentum concentrated at one to four weeks. A market with a
    different clientele has a different memory."""
    assert spec_for(AssetClass.US_EQUITY) is EQUITY_TREND
    assert spec_for(AssetClass.CRYPTO) is CRYPTO_TREND
    # Crypto: one to four weeks. Equity: one to six months. They overlap
    # around a month, which is real -- the claim is that crypto's memory is
    # shorter throughout, not that the windows are disjoint.
    assert max(CRYPTO_TREND.lookbacks) <= 28
    assert min(EQUITY_TREND.lookbacks) >= 21
    assert max(CRYPTO_TREND.lookbacks) < max(EQUITY_TREND.lookbacks)
    assert min(CRYPTO_TREND.lookbacks) < min(EQUITY_TREND.lookbacks)
    assert "Moskowitz" in EQUITY_TREND.source
    assert "Liu" in CRYPTO_TREND.source


def test_the_score_is_in_standard_deviations_not_raw_return():
    """Prevents comparing a quiet instrument's small move with a violent one's
    large move as though they meant the same thing. Normalising is also what
    makes one pooled premium applicable across a universe of unlike symbols."""
    calm = np.array([100.0 * (1.002 ** k) for k in range(60)])
    wild = calm.copy()
    rng = np.random.default_rng(0)
    wild = np.array([100.0 * float(np.exp(0.002 * k + rng.normal(0, 0.05)))
                     for k in range(60)])

    calm_score = momentum_score(calm, 21)
    wild_score = momentum_score(wild, 21)
    # Same underlying drift, far more noise: the normalised score must be
    # smaller for the wild one, not the same.
    assert calm_score > wild_score


def test_a_down_trend_is_flat_not_short():
    """This book is long-only here: crypto cannot be shorted at Alpaca at all,
    and an equity short needs a locate. A negative score is a reason to hold
    nothing, not a reason to sell."""
    pooled = PooledTrend(beta_bps=15.0, t_stat=8.0, observations=9000, symbols=40)
    signal = evaluate(daily(200, drift_per_day=-0.004),
                      asset_class=AssetClass.US_EQUITY, pooled=pooled,
                      round_trip_bps=4.0)
    assert signal.value == 0.0
    assert not signal.eligible
    assert "long-only" in signal.reason


def test_it_warms_up_rather_than_refusing():
    signal = evaluate(daily(10), asset_class=AssetClass.US_EQUITY,
                      pooled=PooledTrend(15.0, 8.0, 9000, 40), round_trip_bps=4.0)
    assert not signal.eligible
    assert "warming up" in signal.reason
    assert "resolves itself" in signal.reason


# ---------------------------------------------------------------- the phase

def _live_signal(score: float = 1.0) -> trend.TrendSignal:
    return trend.TrendSignal(value=0.5, score=score, eligible=True,
                             drift_bps_per_day=10.0, min_hold_days=4.0)


def test_holding_and_entering_are_judged_differently():
    """The asymmetry is the point. Entering must justify the whole round trip;
    staying only has to justify itself, because the entry cost is already spent
    and closing early throws it away without collecting the edge it bought."""
    assert phase_for(held=False, days_held=0, min_hold_days=4.0,
                     signal=_live_signal()) is TrendPhase.ENTER
    assert phase_for(held=True, days_held=1.0, min_hold_days=4.0,
                     signal=_live_signal()) is TrendPhase.HOLDING
    assert phase_for(held=True, days_held=9.0, min_hold_days=4.0,
                     signal=_live_signal()) is TrendPhase.MATURE


def test_a_position_is_closed_when_its_reason_goes_not_when_it_matures():
    """Maturity is permission to leave, not an instruction to. The trend ending
    is the instruction."""
    gone = trend.TrendSignal(value=0.0, score=-0.4, eligible=False)
    assert phase_for(held=True, days_held=1.0, min_hold_days=9.0,
                     signal=gone) is TrendPhase.EXIT
    assert phase_for(held=True, days_held=99.0, min_hold_days=4.0,
                     signal=_live_signal()) is TrendPhase.MATURE


def test_nothing_to_hold_and_nothing_to_enter_is_flat():
    dead = trend.TrendSignal(eligible=False, score=0.2)
    assert phase_for(held=False, days_held=0, min_hold_days=1.0,
                     signal=dead) is TrendPhase.FLAT


# ------------------------------------------------- which strategy gets picked

from decimal import Decimal

from imperium.execution.broker import PaperBroker
from imperium.execution.engine import SymbolEngine
from imperium.execution.portfolio import PortfolioAllocator, Verdict
from imperium.execution.risk import limits_for_equity
from imperium.session import TradingSession
from imperium.telemetry.streams import TelemetryHub
from imperium.venues import registry


def _engine(symbol: str, *, equity: float = 70.0, day_trades: int = 0):
    limits = limits_for_equity(equity)
    allocator = PortfolioAllocator(limits)
    allocator.equity, allocator.cash = equity, equity
    allocator.day_trade_count = day_trades
    engine = SymbolEngine(symbol, registry.get(registry.DEFAULT_VENUE), limits,
                          allocator, TelemetryHub())
    allocator.observe(symbol).admitted = True
    price = 20.0
    for k in range(engine.params.warmup_bars + 5):
        engine.series.add(Bar(k * 60_000, price, price * 1.001, price * 0.999,
                              price, 1000.0, closed=True))
    engine.set_book(19.995, 20.005)
    return engine


def test_an_account_with_no_day_trades_left_uses_the_multi_day_horizon():
    """The answer to "what can $70 actually run".

    Two day trades used out of two, on an account far below the $25,000 floor.
    An intraday strategy here cannot close what it opens, so the engine stops
    offering it one -- the multi-day horizon is not a fallback, it is the only
    horizon that exists.
    """
    engine = _engine("PLTR", equity=70.0, day_trades=2)
    assert not engine.allocator.day_trades_available()

    assert engine.evaluate().strategy == "trend"


def test_an_account_with_day_trades_left_still_trades_intraday():
    engine = _engine("PLTR", equity=70.0, day_trades=0)
    assert engine.allocator.day_trades_available()
    assert engine.evaluate().strategy == "intraday"


def test_crypto_is_never_pushed_onto_the_multi_day_horizon_by_the_pdt_rule():
    """Crypto is not a security under FINRA's rule, so no number of day trades
    exhausts anything. Treating it as PDT-subject would push a 24/7 market onto
    a horizon it never needed."""
    engine = _engine("BTC/USD", equity=70.0, day_trades=99)
    assert not engine.pdt_subject
    assert engine.evaluate().strategy == "intraday"


def test_a_large_account_is_unaffected():
    engine = _engine("PLTR", equity=100_000.0, day_trades=99)
    assert engine.allocator.day_trades_available(), "the floor does not apply"
    assert engine.evaluate().strategy == "intraday"


def test_an_open_trend_position_keeps_its_symbol():
    """A position mid-life belongs to the strategy that opened it. It is the
    only thing that knows what the position cost and how much of that cost the
    elapsed holding period has paid for; handing it to another strategy would
    spend a round trip and collect none of the edge."""
    engine = _engine("PLTR", equity=70.0, day_trades=0)
    assert engine.evaluate().strategy == "intraday"

    engine.trend_held = True
    engine.trend_days_held = 2.0
    assert engine.evaluate().strategy == "trend"


@pytest.mark.asyncio
async def test_the_session_only_claims_positions_this_strategy_opened():
    """Prevents adopting somebody else's position.

    A position the intraday blend or the overnight strategy opened is not a
    trend position, and treating it as one would apply a holding-period rule to
    a trade that never agreed to it.
    """
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("10000")
    session.feed.quote("AAPL").last = 100.0
    await session.broker.apply_target("AAPL", 0.1, 100.0, 10_000.0)

    engine = session.engine("AAPL")
    engine.decision.strategy = "intraday"
    session._sync_trend_holdings()

    assert "AAPL" not in session.trend_holdings
    assert not engine.trend_held

    # One the trend strategy did open is claimed, and its clock starts.
    engine.decision.strategy = "trend"
    session._sync_trend_holdings()
    assert "AAPL" in session.trend_holdings
    assert engine.trend_held


@pytest.mark.asyncio
async def test_a_closed_position_stops_being_carried():
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("10000")
    session.feed.quote("AAPL").last = 100.0
    await session.broker.apply_target("AAPL", 0.1, 100.0, 10_000.0)
    engine = session.engine("AAPL")
    engine.decision.strategy = "trend"
    session._sync_trend_holdings()
    assert engine.trend_held

    await session.broker.apply_target("AAPL", 0.0, 100.0, 10_000.0)
    session._sync_trend_holdings()

    assert "AAPL" not in session.trend_holdings
    assert not engine.trend_held
    assert engine.trend_days_held == 0.0


@pytest.mark.asyncio
async def test_a_trend_position_survives_a_restart_with_its_clock_intact():
    """The holding period is the strategy. A restart that reset it to zero
    would re-derive an entry every time the process bounced and pay the round
    trip again."""
    import time as _t

    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("10000")
    session.feed.quote("AAPL").last = 100.0
    await session.broker.apply_target("AAPL", 0.1, 100.0, 10_000.0)
    session.trend_holdings["AAPL"] = _t.time() - 3 * 86_400

    engine = session.engine("AAPL")
    engine.decision.strategy = "trend"
    session._sync_trend_holdings()

    assert engine.trend_days_held == pytest.approx(3.0, abs=0.01)


def test_the_pooled_fit_never_sees_the_day_it_is_predicting():
    """The most dangerous bug this file can have, tested directly.

    A score is built from a trailing window, so the most recent return is
    *inside* it. Pair a score with that return instead of the next one and the
    correlation is mechanical: the estimator reports a large, highly
    significant premium on pure noise, the credibility gate passes, and the
    book trades a number that does not exist.

    So: random walks in, nothing out.
    """
    session = TradingSession()
    rng = np.random.default_rng(2024)
    scored = {}
    for i in range(30):
        price, bars = 50.0, []
        for k in range(320):
            price *= float(np.exp(rng.normal(0, 0.02)))
            bars.append(Bar(k * 86_400_000, price, price * 1.01, price * 0.99,
                            price, 1e6, closed=True))
        pair = session._trend_observations(f"EQ{i}", bars)
        if pair is not None:
            scored[f"EQ{i}"] = pair

    assert len(scored) >= 20, "the fixture must actually produce observations"
    pooled = pool(scored)
    assert pooled.observations > 3_000
    assert abs(pooled.t_stat) < 3.0, (
        f"a premium of t={pooled.t_stat:.1f} on random walks means the score "
        f"is being paired with a return it already contains")
    assert not pooled.credible


def test_the_standard_error_is_robust_to_heteroskedasticity():
    """Daily returns are not homoskedastic by any stretch, and the classical
    standard error assumes they are.

    It matters in the direction that costs money: where the noise is largest,
    the classical error understates it, the t-statistic is inflated, and the
    credibility gate lets through an estimate that has not earned it. Checked
    against both formulas rather than against a threshold, so the two cannot be
    confused.
    """
    rng = np.random.default_rng(9)
    x = rng.normal(0, 1, 4000)
    # Noise whose size grows with the regressor -- exactly the case the two
    # formulas disagree about.
    y = 0.0 * x + rng.normal(0, 1, 4000) * (0.2 + 2.0 * np.abs(x))

    pooled = pool({"S": (x, y)})

    xc = x - x.mean()
    denom = float(np.sum(xc ** 2))
    beta = float(np.sum(xc * (y - y.mean())) / denom)
    resid = y - (y.mean() + beta * xc)
    robust = math.sqrt(float(np.sum((xc ** 2) * (resid ** 2)) / denom ** 2))
    classical = math.sqrt(float(np.sum(resid ** 2) / (x.size - 2) / denom))

    assert robust > classical * 1.5, "the fixture must separate the two"
    assert abs(pooled.t_stat) == pytest.approx(abs(beta / robust), rel=1e-6)
    assert abs(pooled.t_stat) != pytest.approx(abs(beta / classical), rel=1e-3)
