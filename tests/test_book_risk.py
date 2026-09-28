"""The book as one position.

Every sizing rule used to look at one symbol at a time, so five positions
that fall together could hold the book far above its volatility target while
every individual check passed. These cover the measurement -- checked against
independent implementations -- and the one rule it enforces.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from imperium.execution import book_risk as br

# -- helpers ------------------------------------------------------------------


def _bars(returns, start_day=0, step=1):
    price, out = 100.0, [SimpleNamespace(open_time=start_day * br.DAY_MS,
                                         close=100.0, closed=True)]
    for i, r in enumerate(returns, start=1):
        price *= math.exp(r)
        out.append(SimpleNamespace(open_time=(start_day + i * step) * br.DAY_MS,
                                   close=price, closed=True))
    return out


def _book(target=0.20, *, n=200, factor_sd=0.03, noise_sd=0.006, names="ABCDE",
          seed=0):
    """Holdings that share one factor: the correlated case this exists for."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n) * factor_sd
    risk = br.BookRisk(target_volatility=target)
    for name in names:
        risk.set_series(name, _bars(factor + rng.standard_normal(n) * noise_sd))
    return risk, factor


# -- the estimator, against independent implementations -------------------------


# scikit-learn 1.9.1's ledoit_wolf on this exact input, rescaled by n/(n-1).
# Recorded rather than imported: sklearn is not a dependency of this program,
# and the terminal must not grow one to carry a check.
SKLEARN_SHRINKAGE = 0.12725975589447627
SKLEARN_COV = np.array([
    [8.779591039007132e-05, 4.5316875740236395e-05, 2.3442191220514144e-05],
    [4.5316875740236395e-05, 0.00012164563389418658, 6.687820729138861e-05],
    [2.3442191220514144e-05, 6.687820729138861e-05, 0.00014090148051079006],
])


def test_the_shrunk_covariance_matches_scikit_learn():
    rng = np.random.default_rng(12345)
    x = rng.standard_normal((60, 3)) @ np.array(
        [[1.0, 0.6, 0.2], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]) * 0.01
    cov, intensity = br.shrunk_covariance(x)
    assert intensity == pytest.approx(SKLEARN_SHRINKAGE, rel=1e-12)
    np.testing.assert_allclose(cov, SKLEARN_COV, rtol=1e-12)


def test_more_data_means_less_shrinkage():
    """Shrinkage is how much the data cannot support. More data supports more."""
    rng = np.random.default_rng(3)
    mix = rng.standard_normal((4, 4))
    few, _ = rng.standard_normal((40, 4)) @ mix, None
    many = rng.standard_normal((4000, 4)) @ mix
    _, k_few = br.shrunk_covariance(few)
    _, k_many = br.shrunk_covariance(many)
    assert 0.0 <= k_many < k_few <= 1.0


# -- aligning series ----------------------------------------------------------


def test_an_equity_and_a_coin_are_compared_on_the_days_they_share():
    """Five trading days a week against seven. Aligned by position, a Monday
    would be compared with a Saturday."""
    weekdays = [d for d in range(1, 60) if d % 7 not in (5, 6)]
    equity = {d: 0.01 for d in weekdays}
    coin = {d: 0.02 for d in range(1, 60)}
    # Both orders. With the equity first, taking only its days happens to be
    # right; the coin first is what shows whether the days are really shared.
    for order in (["E", "C"], ["C", "E"]):
        days, matrix = br.align({"E": equity, "C": coin}, order)
        assert days == weekdays, order
        assert matrix.shape == (len(weekdays), 2)


def test_an_unclosed_bar_is_not_a_return():
    bars = _bars([0.01, 0.02])
    bars[-1].closed = False
    assert len(br.returns_by_day(bars)) == 1


# -- what it measures ---------------------------------------------------------


def test_five_positions_that_move_together_are_one_bet():
    risk, _ = _book(noise_sd=0.0005)
    a = risk.update({s: 0.12 for s in "ABCDE"})
    assert a.effective_bets == pytest.approx(1.0, abs=0.1)
    assert a.clusters, "five near-identical holdings were not reported"


def test_five_independent_positions_are_about_five_bets():
    rng = np.random.default_rng(9)
    risk = br.BookRisk(0.20)
    for s in "ABCDE":
        risk.set_series(s, _bars(rng.standard_normal(250) * 0.02))
    a = risk.update({s: 0.12 for s in "ABCDE"})
    assert a.effective_bets == pytest.approx(5.0, abs=0.6)
    assert not a.clusters


def test_risk_contributions_account_for_the_whole_book():
    risk, _ = _book()
    a = risk.update({"A": 0.10, "B": 0.05, "C": 0.15})
    assert sum(a.contribution.values()) == pytest.approx(1.0)
    assert a.contribution["C"] > a.contribution["B"]


def test_risk_is_attributed_to_the_strategy_that_holds_it():
    risk, _ = _book()
    a = risk.update({"A": 0.10, "B": 0.10, "C": 0.10},
                    owners={"A": "trend", "B": "trend", "C": "overnight"})
    assert sum(a.by_strategy.values()) == pytest.approx(1.0)
    assert a.by_strategy["trend"] == pytest.approx(
        a.contribution["A"] + a.contribution["B"])


def test_beta_is_measured_against_the_market():
    rng = np.random.default_rng(4)
    market = rng.standard_normal(200) * 0.01
    risk = br.BookRisk(0.20)
    risk.set_market(_bars(market))
    risk.set_series("A", _bars(1.5 * market + rng.standard_normal(200) * 0.002))
    a = risk.update({"A": 0.5})
    assert a.beta == pytest.approx(0.5 * 1.5, abs=0.05)


def test_no_market_history_means_no_beta_rather_than_zero():
    risk, _ = _book()
    assert risk.update({"A": 0.1}).beta is None


def test_a_symbol_without_enough_history_is_named_not_guessed():
    risk, _ = _book()
    risk.set_series("NEW", _bars([0.01] * 5))
    a = risk.update({"A": 0.1, "NEW": 0.1})
    assert "NEW" in a.unmeasured
    assert a.measured == 1


# -- the rule -------------------------------------------------------------------


def _brute(risk, holdings, symbol, desired):
    names = sorted(holdings) + [symbol]
    _, m = br.align(risk.series, names)
    cov, _ = br.shrunk_covariance(m)
    cov *= br.ANNUALISATION
    base = [holdings[s] for s in sorted(holdings)]
    best = 0.0
    for x in np.linspace(0, desired, 40001):
        w = np.array(base + [x])
        if math.sqrt(w @ cov @ w) <= risk.target + 1e-12:
            best = x
    return best


def test_the_cap_is_solved_exactly():
    risk, _ = _book()
    risk.update({"A": 0.16, "B": 0.16})
    allowed, why = risk.max_weight("C", 0.16, 0.0)
    assert allowed < 0.16, "the correlated addition was not reduced"
    assert allowed == pytest.approx(_brute(risk, {"A": 0.16, "B": 0.16}, "C", 0.16),
                                    abs=1e-5)
    assert "volatility target" in why


def test_a_trade_that_fits_is_left_alone():
    rng = np.random.default_rng(9)
    risk = br.BookRisk(0.20)
    for s in "AB":
        risk.set_series(s, _bars(rng.standard_normal(250) * 0.01))
    risk.update({"A": 0.10})
    assert risk.max_weight("B", 0.10, 0.0) == (0.10, "")


def test_an_exit_or_a_reduction_is_never_touched():
    """A cap that can block an exit is a cap that traps you in a loss."""
    risk, _ = _book(target=0.01)          # a target nothing could meet
    risk.update({"A": 0.16, "B": 0.16})
    assert risk.max_weight("A", 0.0, 0.16) == (0.0, "")
    assert risk.max_weight("A", 0.08, 0.16) == (0.08, "")


def test_a_book_already_over_target_keeps_what_it_holds_and_adds_nothing():
    risk, _ = _book(target=0.05)
    risk.update({"A": 0.16, "B": 0.16})
    allowed, why = risk.max_weight("C", 0.16, 0.04)
    assert allowed == pytest.approx(0.04)
    # "already at", not "already": the ordinary sizing message says a symbol
    # "moves with what it already holds", and a looser check passed against a
    # version with this branch removed.
    assert "already at" in why, why


def test_a_symbol_it_cannot_measure_is_left_to_the_per_position_rules():
    risk, _ = _book(target=0.05)
    risk.update({"A": 0.16})
    risk.set_series("NEW", _bars([0.01] * 5))
    assert risk.max_weight("NEW", 0.16, 0.0) == (0.16, "")


def test_it_never_increases_a_trade():
    risk, _ = _book()
    risk.update({"A": 0.05})
    for desired in (0.01, 0.05, 0.10, 0.20):
        allowed, _ = risk.max_weight("B", desired, 0.0)
        assert allowed <= desired


# -- through the real engine -----------------------------------------------------


def _engine_holding_its_twin():
    """A real engine for PLTR, and a book already holding a series that is
    PLTR's own history -- the most correlated addition there can be."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_trend import _engine, daily

    from imperium.strategy.trend import PooledTrend

    engine = _engine("PLTR", equity=10_000.0, day_trades=2)
    engine.pooled_trend = PooledTrend(15.0, 8.0, 9000, 40)
    engine.daily_bars = daily(200, drift_per_day=0.004, seed=5)
    return engine


def _twin_book(engine, room):
    """A book holding PLTR's own history, sized to leave ``room`` of weight
    before it reaches the target. Computed from the series' measured
    volatility, so the scenario is as tight as the test says it is rather than
    as tight as a guessed weight happens to make it."""
    risk = br.BookRisk(target_volatility=0.20)
    risk.set_series("TWIN", engine.daily_bars)
    # Over the same window the engine measures, or "room" means something
    # different to this test than to the code it is testing.
    series = risk.series["TWIN"]
    r = np.array([series[d] for d in sorted(series)[-br.WINDOW:]])
    sigma = math.sqrt(np.var(r, ddof=1) * br.ANNUALISATION)
    risk.update({"TWIN": risk.target / sigma - room})
    engine.book_risk = risk
    return risk


def test_the_engine_holds_a_correlated_entry_to_the_book_target():
    free = _engine_holding_its_twin().evaluate()
    assert free.strategy == "trend" and free.target_weight > 0.05

    engine = _engine_holding_its_twin()
    risk = _twin_book(engine, room=0.05)
    held = engine.evaluate()

    # Against brute force over the same shrunk covariance -- not against the
    # 0.05 the room was built from. Two identical series are a rank-one
    # sample covariance, and the shrinkage treats part of that perfect
    # correlation as estimation noise, so it leaves slightly more room than
    # arithmetic without it would. That is the estimator doing its job.
    expected = _brute(risk, risk.holdings, "PLTR", free.raw_weight)
    assert expected < free.raw_weight
    assert held.target_weight == pytest.approx(expected, abs=2e-5), (
        f"a correlated entry was not held to the room the book had left: "
        f"{held.target_weight:.4f} against {expected:.4f}")
    assert held.target_weight == pytest.approx(0.05, abs=0.005)
    assert "volatility target" in held.book_risk_note


def test_what_is_left_under_the_fee_floor_is_not_opened():
    """The tilt and the evidence multiplier lift a trade to the fee floor
    because they are preferences. This is a limit: if what the book can take
    is not worth the fees, the position is not opened, and the decision says
    so."""
    engine = _engine_holding_its_twin()
    floor = engine.position_floor / engine.allocator.equity
    _twin_book(engine, room=floor / 3)
    held = engine.evaluate()
    assert held.target_weight == 0.0
    assert "fee floor" in held.book_risk_note


# -- in the session --------------------------------------------------------------


def _paper():
    from imperium.execution.broker import PaperBroker
    from imperium.session import TradingSession
    from imperium.venues import registry

    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    return session


def _daily_rows(returns, start=100.0):
    rows, price = [], start
    for i, r in enumerate(returns):
        price *= math.exp(r)
        rows.append({"t": (i + 1) * br.DAY_MS, "o": price, "h": price,
                     "l": price, "c": price, "v": 1.0})
    return rows


class _BarsClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    async def bars(self, symbols, **kw):
        self.calls += 1
        return {s: self.rows for s in symbols}


@pytest.mark.asyncio
async def test_the_session_measures_what_the_book_holds():
    import time as _t

    session = _paper()
    rng = np.random.default_rng(0)
    factor = rng.standard_normal(200) * 0.02
    for symbol in ("AAPL", "MSFT"):
        engine = session.engine(symbol)
        engine.daily_bars = _bars(factor + rng.standard_normal(200) * 0.002)
        q = session.feed.quote(symbol)
        q.last, q.updated_at = 100.0, _t.time()
        await session.broker.apply_target(symbol, 0.10, 100.0, 10_000.0,
                                          strategy="trend")
        session.allocator.observe(symbol).current_weight = 0.10

    session._attribute()
    session._measure_book()
    a = session.snapshot()["book_risk"]
    assert a["positions"] == 2 and a["measured"] == 2
    assert a["effective_bets"] < 1.3, "two near-identical holdings read as two bets"
    assert a["by_strategy"] == {"trend": pytest.approx(1.0)}


def test_a_fault_in_the_measurement_does_not_stop_the_tick(monkeypatch):
    session = _paper()

    def boom(*a, **kw):
        raise RuntimeError("broken")

    monkeypatch.setattr(session.book_risk, "update", boom)
    session._measure_book()                      # must not raise


def test_every_engine_shares_the_one_book_view():
    session = _paper()
    assert session.engine("AAPL").book_risk is session.book_risk
    assert session.engine("LATER").book_risk is session.book_risk


@pytest.mark.asyncio
async def test_the_benchmark_is_fetched_but_never_becomes_something_to_trade():
    """Riding in the universe's batch it would have been given an engine like
    any other symbol -- and an engine is something that can place an order."""
    session = _paper()
    session.client = _BarsClient(_daily_rows([0.001] * 120))
    await session.refresh_benchmark(force=True)
    assert len(session._benchmark_bars) == 120
    assert br.BENCHMARK not in session.engines


@pytest.mark.asyncio
async def test_the_benchmark_is_not_re_fetched_every_minute():
    session = _paper()
    client = _BarsClient(_daily_rows([0.001] * 120))
    session.client = client
    await session.refresh_benchmark(force=True)
    for _ in range(10):
        await session.refresh_benchmark()
    assert client.calls == 1


@pytest.mark.asyncio
async def test_a_failed_benchmark_fetch_keeps_the_last_good_series():
    from imperium.venues.alpaca.client import VenueError

    session = _paper()
    session.client = _BarsClient(_daily_rows([0.001] * 120))
    await session.refresh_benchmark(force=True)

    class _Down:
        async def bars(self, *a, **kw):
            raise VenueError("down")

    session.client = _Down()
    await session.refresh_benchmark(force=True)
    assert len(session._benchmark_bars) == 120, (
        "a failed fetch threw away the benchmark, and beta with it")


@pytest.mark.asyncio
async def test_beta_is_measured_once_the_benchmark_is_in():
    import time as _t

    session = _paper()
    rng = np.random.default_rng(1)
    market = rng.standard_normal(200) * 0.01
    session.client = _BarsClient(_daily_rows(market))
    await session.refresh_benchmark(force=True)

    engine = session.engine("AAPL")
    engine.daily_bars = [b for b in session._benchmark_bars]   # beta of 1
    q = session.feed.quote("AAPL")
    q.last, q.updated_at = 100.0, _t.time()
    await session.broker.apply_target("AAPL", 0.10, 100.0, 10_000.0,
                                      strategy="trend")
    session.allocator.observe("AAPL").current_weight = 0.10
    session._measure_book()
    assert session.book_risk.assessment.beta == pytest.approx(0.10, abs=0.01)


@pytest.mark.asyncio
async def test_the_trading_tick_measures_the_book_by_itself():
    """Through the tick, with nothing calling the measurement directly."""
    import time as _t

    session = _paper()
    rng = np.random.default_rng(2)
    engine = session.engine("AAPL")
    engine.daily_bars = _bars(rng.standard_normal(200) * 0.02)
    q = session.feed.quote("AAPL")
    q.last, q.updated_at = 100.0, _t.time()
    await session.broker.apply_target("AAPL", 0.10, 100.0, 10_000.0,
                                      strategy="trend")
    session.allocator.observe("AAPL").current_weight = 0.10
    await session._tick()
    assert session.book_risk.assessment.positions == 1
    assert session.book_risk.assessment.measured == 1
