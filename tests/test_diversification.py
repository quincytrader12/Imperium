"""Capital by what a strategy adds to the rest of the book.

What must hold: a strategy that moves with the rest of the book gets less,
one that moves against it gets more, one with a short record is untouched;
the measured correlation is shrunk on a short record; the multiplier walks
toward its target rather than jumping; and the product with the evidence
multiplier stays inside the bounds every size is already held to.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from imperium.execution import diversification as dv
from imperium.execution import evidence as ev
from imperium.execution.attribution import StrategyRecord


def _days(n):
    first = dt.date(2026, 6, 1)
    return [(first + dt.timedelta(days=i)).isoformat() for i in range(n)]


def _record(name, returns, equity=100.0):
    """A strategy whose daily contribution is ``returns`` of a flat fund."""
    days = _days(len(returns) + 1)
    pnl = np.concatenate([[0.0], np.cumsum(returns) * equity])
    return StrategyRecord(name, daily=list(zip(days, map(float, pnl))))


def _equity(n, value=100.0):
    return {d: value for d in _days(n + 1)}


RNG = np.random.default_rng(4)
BASE = RNG.normal(0, 0.01, 90)
NOISE = RNG.normal(0, 0.002, 90)


# -- the measurement --------------------------------------------------------------------


def test_returns_are_keyed_by_the_day_they_end():
    r = _record("trend", [0.01, -0.02])
    out = dv.returns_by_day(r.daily, _equity(2))
    assert out == {"2026-06-02": pytest.approx(0.01), "2026-06-03": pytest.approx(-0.02)}
    assert dv.returns_by_day(r.daily, {}) == {}, "no denominator, no return"


def test_with_the_book_is_positive_against_it_is_negative():
    series = {"a": dict(zip(_days(90), BASE)),
              "b": dict(zip(_days(90), BASE + NOISE)),
              "hedge": dict(zip(_days(90), -2 * BASE + NOISE))}
    rho = dv.correlation_to_rest(series)
    # a against b + hedge = -BASE + 2*NOISE: against the rest.
    assert rho["hedge"][0] < -0.9
    assert rho["a"][0] < 0
    two = {"a": series["a"], "b": series["b"]}
    assert dv.correlation_to_rest(two)["a"][0] > 0.9


def test_a_day_another_strategy_did_not_mark_counts_as_nothing_from_it():
    days = _days(40)
    series = {"a": dict(zip(days, BASE[:40])),
              "late": dict(zip(days[20:], BASE[20:40]))}
    rho, n = dv.correlation_to_rest(series)["a"]
    assert n == 40 and 0 < rho < 1


def test_too_short_or_flat_is_not_measured():
    short = {"a": dict(zip(_days(dv.MIN_DAYS - 1), BASE)),
             "b": dict(zip(_days(dv.MIN_DAYS - 1), BASE))}
    assert dv.correlation_to_rest(short) == {}
    enough = {"a": dict(zip(_days(dv.MIN_DAYS), BASE)),
              "b": dict(zip(_days(dv.MIN_DAYS), BASE))}
    assert set(dv.correlation_to_rest(enough)) == {"a", "b"}
    flat = {"a": dict(zip(_days(60), [0.0] * 60)),
            "b": dict(zip(_days(60), BASE[:60]))}
    assert dv.correlation_to_rest(flat) == {}
    alone = {"a": dict(zip(_days(60), BASE[:60]))}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no division by a flat rest
        assert dv.correlation_to_rest(alone) == {}, "no rest of the book to move with"


@pytest.mark.parametrize("rho,days,shrunk,goal", [
    (0.5, 30, 0.25, 0.875),
    (-0.5, 30, -0.25, 1.125),
    (0.0, 90, 0.0, 1.0),
    (1.0, 10_000, 0.997, dv.LOW),
    (-1.0, 10_000, -0.997, dv.HIGH),
])
def test_the_target_shrinks_then_tilts_within_bounds(rho, days, shrunk, goal):
    s, g = dv.target(rho, days)
    assert s == pytest.approx(shrunk, abs=1e-3) and g == pytest.approx(goal)


# -- the allocator -------------------------------------------------------------------------


def _book(n=90):
    return [_record("trend", BASE[:n]), _record("overnight", BASE[:n] + NOISE[:n]),
            _record("global_trend", -BASE[:n] + NOISE[:n]),
            _record("unattributed", BASE[:n])]


def test_it_walks_toward_the_target_a_step_a_day():
    twins = [_record("a", BASE), _record("b", BASE + NOISE)]
    d = dv.Diversification()
    d.revise("2026-09-01", twins, _equity(90))
    a = d.standings["a"]
    assert a.target == dv.LOW
    assert a.multiplier == pytest.approx(1.0 - dv.STEP)
    assert d.revise("2026-09-01", twins, _equity(90)) == [], "once a day"
    d.revise("2026-09-02", twins, _equity(90))
    assert a.multiplier == pytest.approx(1.0 - 2 * dv.STEP)
    d.revise("2026-09-03", twins, _equity(90))
    assert a.multiplier == pytest.approx(dv.LOW), "and stops at the target"
    assert d.revise("2026-09-04", twins, _equity(90)) == []


def test_the_hedge_gains_and_the_twin_loses():
    d = dv.Diversification()
    moved = d.revise("2026-09-01", _book(), _equity(90))
    names = {m[0] for m in moved}
    assert "global_trend" in names and "unattributed" not in names
    assert d.multiplier("global_trend") > 1.0
    assert d.multiplier("overnight") < 1.0
    assert "moves against the rest of the book" in d.note("global_trend")
    assert "moves with the rest of the book" in d.note("overnight")
    assert d.multiplier("unattributed") == 1.0 and d.multiplier("") == 1.0
    d.standings["sector"] = dv.Standing(multiplier=0.8)
    assert d.multiplier("sector") == 1.0, "the sector sleeve sizes itself"


def test_a_short_record_says_how_far_it_has_to_go():
    d = dv.Diversification()
    d.revise("2026-09-01", _book(12), _equity(12))
    assert d.multiplier("trend") == 1.0
    assert d.note("trend") == "12 of 30 days to measure it against the rest of the book"


def test_the_product_with_the_evidence_stays_inside_its_bounds():
    class Fixed:
        def __init__(self, m):
            self.m = m

        def multiplier(self, _):
            return self.m

    assert dv.combined_multiplier(Fixed(ev.CAP), Fixed(dv.HIGH), "x") == ev.CAP
    assert dv.combined_multiplier(Fixed(ev.FLOOR), Fixed(dv.LOW), "x") == ev.FLOOR
    assert dv.combined_multiplier(Fixed(1.2), Fixed(0.8), "x") == pytest.approx(0.96)
    assert dv.combined_multiplier(None, None, "x") == 1.0


def test_it_survives_a_round_trip_and_a_corrupt_file():
    d = dv.Diversification()
    d.revise("2026-09-01", _book(), _equity(90))
    back = dv.Diversification.from_dict(d.as_dict())
    assert back.revised_on == "2026-09-01"
    assert back.multiplier("global_trend") == pytest.approx(d.multiplier("global_trend"))
    bad = dv.Diversification.from_dict({"standings": {
        "a": {"multiplier": "nan"}, "b": {"multiplier": 9}, "c": "junk",
        "d": {"days": "many"}}})
    assert bad.multiplier("a") == 1.0 and bad.multiplier("b") == dv.HIGH
    assert "c" not in bad.standings and "d" not in bad.standings
    assert dv.Diversification.from_dict(None).standings == {}


# -- where it acts --------------------------------------------------------------------------


def test_the_engine_sizes_by_evidence_times_diversification():
    from test_overnight import _engine

    from imperium.execution.engine import Decision

    engine = _engine("AAPL")
    spread = dv.Diversification()
    spread.standings["trend"] = dv.Standing(multiplier=0.8, reason="moves with it")
    engine.diversification = spread
    d = Decision(symbol="AAPL")
    d.strategy, d.raw_weight = "trend", 0.10
    engine._size_for_evidence(d)
    assert d.raw_weight == pytest.approx(0.08)
    assert "moves with it" in d.capital_note


def test_the_session_revises_it_with_the_days_mark_and_shows_it(monkeypatch):
    from decimal import Decimal

    from imperium.execution.broker import PaperBroker
    from imperium.session import TradingSession
    from imperium.venues import registry

    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("73")
    session._attribution_loaded = True
    book = session.attribution.book_for(session.broker.mode.value)
    for r in _book():
        book.records[r.name] = r
    book.equity_marks.update(_equity(90))
    session._mark_strategies("2026-09-01")
    assert session.diversification.multiplier("global_trend") > 1.0
    assert session.engine("AAPL").diversification is session.diversification
    rows = {r["strategy"]: r for r in session._strategies_block()["rows"]}
    row = rows["global_trend"]
    assert row["multiplier"] == pytest.approx(dv.combined_multiplier(
        session.capital_weights, session.diversification, "global_trend"))
    assert "Diversification: moves against" in row["capital_reason"]

    # Saved with the attribution, and read back on the next start.
    session._save_attribution()
    fresh = TradingSession()
    early = fresh.engine("AAPL")              # built before the state is read
    fresh._load_attribution()
    assert fresh.diversification.multiplier("global_trend") == pytest.approx(
        session.diversification.multiplier("global_trend"))
    assert early.diversification is fresh.diversification
