"""Capital follows evidence.

The statistics are cross-checked against scipy -- which is a dev dependency
only, so the terminal itself uses the standard library's NormalDist -- and the
deflation bar against a simulation of the thing it approximates. The rules are
tested for the properties that make them safe to run unattended: nothing moves
on a short record, nothing moves by more than a step a day, nothing reaches
zero, and a strategy that stopped working is cut even if its history is good.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from imperium.execution import evidence as ev
from imperium.execution.attribution import StrategyRecord

EQUITY = 1_000.0


def _record(name: str, returns: list[float], round_trips: int = 20) -> StrategyRecord:
    """A strategy whose daily contribution to the fund is ``returns``."""
    rec = StrategyRecord(name=name, round_trips=round_trips)
    value = 0.0
    rec.daily.append(("d0000", 0.0))
    for i, r in enumerate(returns, start=1):
        value += r * EQUITY
        rec.daily.append((f"d{i:04d}", value))
    return rec


def _equity(records) -> dict[str, float]:
    days = {d for rec in records for d, _ in rec.daily}
    return {d: EQUITY for d in days}


def _series(mean: float, sd: float, n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


def _settle(weights, records, *, days: int = 20) -> None:
    """Revise on successive days until the multipliers stop moving."""
    for k in range(days):
        weights.revise(f"rev{k:03d}", records, _equity(records))


# -- the statistics against independent implementations ---------------------


def test_moments_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    r = _series(0.002, 0.01, 80, seed=4)
    m = ev.moments(r)
    assert m.sharpe == pytest.approx(np.mean(r) / np.std(r, ddof=1))
    assert m.skew == pytest.approx(stats.skew(r))
    assert m.kurtosis == pytest.approx(stats.kurtosis(r, fisher=False))


def test_probabilistic_sharpe_matches_the_published_formula():
    stats = pytest.importorskip("scipy.stats")
    m = ev.moments(_series(0.002, 0.01, 80, seed=4))
    sr, t, g3, g4 = m.sharpe, m.count, m.skew, m.kurtosis
    expected = stats.norm.cdf(
        (sr - 0.05) * math.sqrt(t - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    assert ev.probabilistic_sharpe(m, 0.05) == pytest.approx(expected)


@pytest.mark.parametrize("trials", [5, 10])
def test_the_deflation_bar_is_the_expected_best_of_n_by_chance(trials):
    """Checked against what it approximates: the mean of the largest of N
    standard normal draws. The published formula is asymptotic, so the
    tolerance is loose -- but it errs on the side of a higher bar here, which
    is the safe direction."""
    rng = np.random.default_rng(1)
    simulated = rng.standard_normal((200_000, trials)).max(axis=1).mean()
    formula = ev.deflated_benchmark(trials, 1.0)
    assert formula == pytest.approx(simulated, abs=0.06)
    assert formula >= simulated - 0.01


def test_one_strategy_has_no_selection_to_correct_for():
    assert ev.deflated_benchmark(1, 1.0) == 0.0


def test_a_series_that_never_moved_has_no_sharpe_rather_than_zero():
    """"Earns nothing" and "has not been measured" are different facts."""
    assert ev.moments([0.0] * 30) is None
    assert ev.moments([0.01, 0.02]) is None


def test_returns_are_contributions_to_the_fund_not_dollars():
    marks = [("a", 0.0), ("b", 10.0), ("c", 5.0)]
    assert ev.daily_returns(marks, {"a": 1000.0, "b": 500.0}) == [
        pytest.approx(0.01), pytest.approx(-0.01)]


def test_a_day_with_no_known_equity_is_skipped_not_guessed():
    marks = [("a", 0.0), ("b", 10.0), ("c", 5.0)]
    assert ev.daily_returns(marks, {"b": 500.0}) == [pytest.approx(-0.01)]


# -- the rules ----------------------------------------------------------------


def test_nothing_moves_on_a_short_record():
    """Ten great days is a lucky fortnight, not evidence."""
    weights = ev.CapitalWeights()
    rec = _record("trend", [0.01] * 5 + [0.012] * 5, round_trips=30)
    _settle(weights, [rec])
    assert weights.multiplier("trend") == 1.0
    assert "gathering evidence: 10 of 20 days" in weights.note("trend")


def test_many_days_but_few_trades_is_still_not_enough():
    """A month of marks on one position is one decision, not thirty."""
    weights = ev.CapitalWeights()
    rec = _record("trend", _series(0.004, 0.005, 60, seed=1), round_trips=1)
    _settle(weights, [rec])
    assert weights.multiplier("trend") == 1.0
    assert "1 of 8 closed trades" in weights.note("trend")


def test_a_strong_consistent_earner_is_given_more_capital():
    weights = ev.CapitalWeights()
    rec = _record("trend", _series(0.004, 0.005, 60, seed=1))
    _settle(weights, [rec])
    assert weights.multiplier("trend") == pytest.approx(ev.CAP)
    assert "strong evidence it earns" in weights.note("trend")


def test_capital_moves_by_at_most_a_step_a_day():
    weights = ev.CapitalWeights()
    rec = _record("trend", _series(0.004, 0.005, 60, seed=1))
    moved = weights.revise("day1", [rec], _equity([rec]))
    assert moved and moved[0][2] - moved[0][1] == pytest.approx(ev.STEP)


def test_a_second_revision_on_the_same_day_does_nothing():
    weights = ev.CapitalWeights()
    rec = _record("trend", _series(0.004, 0.005, 60, seed=1))
    weights.revise("day1", [rec], _equity([rec]))
    assert weights.revise("day1", [rec], _equity([rec])) == []
    assert weights.multiplier("trend") == pytest.approx(1.0 + ev.STEP)


def test_a_strong_loser_is_cut_to_the_floor_and_never_to_zero():
    """A strategy cut to nothing can never produce the evidence that would
    restore it."""
    weights = ev.CapitalWeights()
    rec = _record("overnight", _series(-0.004, 0.005, 60, seed=2))
    _settle(weights, [rec])
    assert weights.multiplier("overnight") == pytest.approx(ev.FLOOR)
    assert ev.FLOOR > 0


def test_a_worthless_strategy_is_rarely_given_more_capital():
    """Measured as a rate, because it is one. Under the null the probability
    is close to uniform, so each threshold is a false-positive rate -- and the
    first version of these bands, boosting at 80%, handed one worthless
    strategy in five more money. A single-seed test could not have shown that;
    it only showed one unlucky draw."""
    boosted = floored = 0
    trials = 400
    for seed in range(trials):
        weights = ev.CapitalWeights()
        rec = _record("intraday", _series(0.0, 0.01, 60, seed=1000 + seed))
        _settle(weights, [rec], days=6)
        m = weights.multiplier("intraday")
        boosted += m > 1.0
        floored += m <= ev.FLOOR
    assert boosted / trials <= 0.08, (
        f"{boosted / trials:.0%} of strategies with no edge were given more "
        f"capital")
    assert floored / trials <= 0.08, (
        f"{floored / trials:.0%} of strategies with no edge were cut to the "
        f"floor")


def test_a_strategy_that_stopped_working_is_cut_whatever_its_history():
    """The decay rule. A good long record and a bad recent one is exactly the
    shape of an edge that has gone, and a full-history statistic is the last
    thing to notice."""
    weights = ev.CapitalWeights()
    good_then_bad = (_series(0.004, 0.004, 80, seed=5)
                     + _series(-0.006, 0.003, 20, seed=6))
    rec = _record("trend", good_then_bad)
    weights.revise("r0", [rec], _equity([rec]))
    standing = weights.standings["trend"]
    assert standing.probability > 0.8, "the whole record should still look good"
    assert standing.target == pytest.approx(ev.FLOOR)
    assert "decaying" in standing.reason


def test_more_strategies_raise_the_bar_each_one_has_to_clear():
    """The best of several worthless strategies looks good by chance alone.
    A record that clears the bar on its own must find it harder with company."""
    modest = _record("trend", _series(0.0012, 0.006, 60, seed=7))

    alone = ev.CapitalWeights()
    alone.revise("r0", [modest], _equity([modest]))

    crowd = [modest] + [_record(f"noise{i}", _series(0.0, 0.006, 60, seed=20 + i))
                        for i in range(6)]
    together = ev.CapitalWeights()
    together.revise("r0", crowd, _equity(crowd))

    assert (together.standings["trend"].probability
            < alone.standings["trend"].probability)


def test_the_unattributed_row_and_the_sleeve_are_never_sized():
    weights = ev.CapitalWeights()
    bad = _series(-0.004, 0.005, 60, seed=2)
    _settle(weights, [_record("unattributed", bad), _record("sector", bad)])
    assert weights.multiplier("unattributed") == 1.0
    assert weights.multiplier("sector") == 1.0


# -- safety at the edges ------------------------------------------------------


def test_a_corrupt_state_file_cannot_size_outside_the_bounds():
    weights = ev.CapitalWeights.from_dict({"standings": {
        "trend": {"multiplier": 50.0, "target": 50.0},
        "overnight": {"multiplier": -3.0, "target": 0.0},
        "intraday": {"multiplier": "nan"},
        "cross_section": "rubbish"}})
    assert weights.multiplier("trend") == ev.CAP
    assert weights.multiplier("overnight") == ev.FLOOR
    assert weights.multiplier("intraday") == 1.0
    assert weights.multiplier("cross_section") == 1.0


def test_the_weights_survive_a_restart():
    weights = ev.CapitalWeights()
    rec = _record("trend", _series(0.004, 0.005, 60, seed=1))
    _settle(weights, [rec])
    back = ev.CapitalWeights.from_dict(weights.as_dict())
    assert back.multiplier("trend") == weights.multiplier("trend")
    assert back.revised_on == weights.revised_on


# -- applied in the engine, before the clamp ----------------------------------


def _engine_with(multiplier: float, equity: float = 10_000.0):
    from imperium.execution.engine import Decision
    from imperium.session import TradingSession

    session = TradingSession()
    session.allocator.equity = equity
    engine = session.engine("AAPL")
    weights = ev.CapitalWeights()
    weights.standings["trend"] = ev.Standing("trend", multiplier=multiplier,
                                             target=multiplier, reason="test")
    engine.capital = weights
    d = Decision(symbol="AAPL", strategy="trend")
    return engine, d


def test_a_cut_shrinks_the_trade_before_the_clamp_sees_it():
    engine, d = _engine_with(0.5)
    d.raw_weight = 0.10
    engine._size_for_evidence(d)
    assert d.raw_weight == pytest.approx(0.05)
    assert d.capital_multiplier == 0.5


def test_a_cut_never_shrinks_a_trade_below_what_the_fees_allow():
    engine, d = _engine_with(ev.FLOOR, equity=100.0)
    floor = engine.position_floor / 100.0
    d.raw_weight = floor * 1.2
    engine._size_for_evidence(d)
    assert d.raw_weight == pytest.approx(floor), (
        "cut under the fee floor, so it cannot trade and cannot earn its way "
        "back")


def test_more_capital_cannot_breach_the_per_symbol_cap():
    engine, d = _engine_with(ev.CAP)
    cap = engine.limits.max_position_weight
    d.raw_weight = cap * 0.9
    engine._size_for_evidence(d)
    assert d.raw_weight == pytest.approx(cap)


def test_an_exit_is_never_resized():
    engine, d = _engine_with(ev.CAP)
    d.raw_weight = 0.0
    engine._size_for_evidence(d)
    assert d.raw_weight == 0.0


def test_every_strategy_path_goes_through_the_same_hook():
    """Four strategy paths each call the pre-clamp hook and then the clamp. A
    fifth that called the news tilt directly would silently skip the
    allocator."""
    import inspect

    from imperium.execution import engine as engine_mod

    source = inspect.getsource(engine_mod)
    assert source.count("self._before_clamp(d)") == 4
    assert source.count("self._tilt_for_news(d)") == 1, (
        "the news tilt is called from somewhere other than the shared hook")


def test_the_multiplier_is_bounded_even_if_a_standing_is_not():
    """The second line of defence, on its own: a standing that reached memory
    without going through from_dict -- a future code path, a bug -- still
    cannot size a position outside the promised range."""
    weights = ev.CapitalWeights()
    weights.standings["trend"] = ev.Standing("trend", multiplier=50.0)
    weights.standings["overnight"] = ev.Standing("overnight", multiplier=-1.0)
    weights.standings["intraday"] = ev.Standing("intraday",
                                                multiplier=float("nan"))
    assert weights.multiplier("trend") == ev.CAP
    assert weights.multiplier("overnight") == ev.FLOOR
    assert weights.multiplier("intraday") == 1.0


# -- through the whole engine ---------------------------------------------------


def _real_trend_decision(multiplier):
    """A genuine multi-day trend entry, evaluated end to end by the engine."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_trend import _engine, daily

    from imperium.strategy.trend import PooledTrend

    engine = _engine("PLTR", equity=10_000.0, day_trades=2)
    engine.pooled_trend = PooledTrend(15.0, 8.0, 9000, 40)
    engine.daily_bars = daily(200, drift_per_day=0.004, seed=5)
    if multiplier is not None:
        weights = ev.CapitalWeights()
        weights.standings["trend"] = ev.Standing(
            "trend", multiplier=multiplier, target=multiplier, reason="test")
        engine.capital = weights
    return engine.evaluate()


def test_a_cut_halves_a_real_trend_entry():
    base = _real_trend_decision(None)
    cut = _real_trend_decision(0.5)
    assert base.strategy == cut.strategy == "trend"
    assert base.target_weight > 0
    assert cut.target_weight == pytest.approx(base.target_weight * 0.5)


def test_more_capital_is_still_held_to_every_limit():
    """The proof the multiplier is applied before the clamp and not after: the
    raw weight grows by half, and the portfolio clamp cuts it back to the
    per-symbol budget. Applied after the clamp, it would have gone straight
    past the limit."""
    base = _real_trend_decision(None)
    boosted = _real_trend_decision(1.5)
    assert boosted.raw_weight == pytest.approx(base.raw_weight * 1.5)
    assert boosted.target_weight < boosted.raw_weight, (
        "a boosted trade was not clamped")
    assert boosted.target_weight <= base.target_weight * 1.5
