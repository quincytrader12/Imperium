"""The nightly research desk: is each strategy's edge still there?

What makes it safe to act on: the daily premium is the textbook slope, a
sample is only ever filed under its own day, an edge that has not changed is
rarely called fading and almost never reversed, one that has gone usually is,
and a verdict can hold capital down but never lift it. The reason for measuring
day by day rather than pooled is tested directly: with symbols that move
together, the pooled t-statistic finds an edge in pure noise and this does not.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import numpy as np
import pytest

from imperium.execution import evidence as ev
from imperium.execution import research as rs
from imperium.execution.attribution import StrategyRecord
from imperium.strategy import trend as trend_mod


# -- the daily premium -----------------------------------------------------------


def test_each_days_premium_is_the_ordinary_least_squares_slope():
    rng = np.random.default_rng(1)
    rows, expected = [], {}
    for day in range(10):
        x = rng.normal(size=30)
        y = 0.002 * x + rng.normal(scale=0.01, size=30)
        rows += [(day, a, b) for a, b in zip(x, y)]
        expected[day] = np.polyfit(x, y, 1)[0]
    got = dict(rs.slopes_by_day(rows))
    assert got.keys() == expected.keys()
    for day, slope in expected.items():
        assert got[day] == pytest.approx(slope)


def test_a_day_too_thin_or_with_no_spread_in_its_scores_is_left_out():
    rows = [(1, float(i), 0.01 * i) for i in range(rs.MIN_NAMES - 1)]
    rows += [(2, 1.0, 0.01 * i) for i in range(10)]
    rows += [(3, float(i), 0.01 * i) for i in range(rs.MIN_NAMES)]
    assert [d for d, _ in rs.slopes_by_day(rows)] == [3]
    assert [d for d, _ in rs.means_by_day(
        [(1, 0.1)] * (rs.MIN_NAMES - 1) + [(2, 0.1)] * rs.MIN_NAMES)] == [2]


def test_a_sample_is_only_ever_filed_under_its_own_day():
    """A symbol whose dates do not line up with its samples is dropped, not
    trimmed: there is no way to tell which end is off."""
    samples = {"A": (np.array([1.0, 2.0]), np.array([0.1, 0.2])),
               "B": (np.array([3.0, 4.0, 5.0]), np.array([0.3, 0.4, 0.5]))}
    days = {"A": np.array([10, 11]), "B": np.array([10, 11])}
    assert list(rs.dated(samples, days)) == [(10, 1.0, 0.1), (11, 2.0, 0.2)]


# -- the verdict -----------------------------------------------------------------


def _premia(values):
    return list(enumerate(values))


def _series(mean, sd, n, seed):
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


def _exact(t, sd, n, seed):
    """A series whose sample t-statistic is exactly ``t``: noise, rescaled
    to the stated deviation and shifted to the mean that gives that t."""
    raw = np.asarray(_series(0.0, 1.0, n, seed))
    raw = (raw - raw.mean()) / raw.std(ddof=1) * sd
    return list(raw + t * sd / math.sqrt(n))


def test_nothing_is_judged_on_a_short_history():
    f = rs.judge("trend", _premia(_series(0.001, 0.001, 119, 1)))
    assert f.verdict == rs.UNMEASURED and f.cap is None
    assert "119 of 120 days" in f.reason


def test_an_edge_that_has_not_changed_is_holding():
    f = rs.judge("trend", _premia(_series(0.0004, 0.001, 250, 2)))
    assert f.verdict == rs.HOLDING and f.cap is None
    assert f.earlier_t > rs.T_BAR


def test_an_edge_that_has_gone_is_fading():
    values = _series(0.0004, 0.001, 190, 3) + _series(0.0, 0.001, 60, 4)
    f = rs.judge("trend", _premia(values))
    assert f.verdict == rs.FADING
    assert f.cap == rs.FADING_CAP
    assert f.dropped >= rs.DROP_BAR and f.recent_t < rs.T_BAR
    assert "edge fading" in f.reason and "bp/day" in f.reason


def test_an_edge_now_pointing_the_wrong_way_is_reversed():
    values = _series(0.0004, 0.001, 190, 5) + _series(-0.0006, 0.001, 60, 6)
    f = rs.judge("trend", _premia(values))
    assert f.verdict == rs.REVERSED and f.cap == rs.REVERSED_CAP


def test_there_is_nothing_to_fade_from_without_a_real_long_run_edge():
    """Lower recently than before, when "before" was noise, is not decay."""
    values = _exact(1.8, 0.002, 190, 7) + _exact(-1.5, 0.002, 60, 8)
    f = rs.judge("trend", _premia(values))
    assert f.earlier_t == pytest.approx(1.8)
    assert f.recent_t == pytest.approx(-1.5)
    assert f.dropped >= rs.DROP_BAR, "the fixture does not show a drop"
    assert f.verdict == rs.HOLDING


def test_an_edge_still_significant_on_its_own_is_not_fading_for_being_smaller():
    values = _series(0.0012, 0.001, 190, 9) + _series(0.0006, 0.001, 60, 10)
    f = rs.judge("trend", _premia(values))
    assert f.dropped >= rs.DROP_BAR and f.recent_t >= rs.T_BAR
    assert f.verdict == rs.HOLDING


def test_an_unchanged_edge_is_rarely_called_fading_and_almost_never_reversed():
    """Measured, not assumed. An edge with a long-run t of about 4 that has
    not changed at all, judged on 500 independent histories."""
    fading = reversed_ = 0
    for seed in range(500):
        f = rs.judge("trend", _premia(_series(0.0003, 0.001, 250, 100 + seed)))
        fading += f.verdict == rs.FADING
        reversed_ += f.verdict == rs.REVERSED
    assert fading / 500 < 0.08
    assert reversed_ / 500 < 0.01


def test_an_edge_that_has_gone_is_usually_caught():
    found = 0
    for seed in range(300):
        values = (_series(0.0003, 0.001, 190, 1000 + seed)
                  + _series(0.0, 0.001, 60, 5000 + seed))
        found += rs.judge("trend", _premia(values)).verdict in (rs.FADING,
                                                                rs.REVERSED)
    assert found / 300 > 0.6


def test_day_by_day_is_honest_where_pooling_finds_an_edge_in_noise():
    """Why the desk measures one premium a day instead of re-running the
    pooled regression on a window. Symbols share a market move, and a score
    that happens to line up with market sensitivity picks it up every day.
    There is no edge at all. Pooled, the t-statistic treats 200 symbols on a
    day as 200 independent observations and 'finds' one in about 60% of
    histories; one slope a day finds one in about 5%, the rate a 2.0 bar
    should. Bounded well short of both."""
    rng = np.random.default_rng(3)
    pooled_hits = daily_hits = 0
    trials = 100
    for _ in range(trials):
        n_sym, n_days = 200, 60
        beta = rng.normal(1.0, 0.5, n_sym)
        scores = beta[:, None] + rng.normal(scale=0.3, size=(n_sym, n_days))
        market = rng.normal(scale=0.02, size=n_days)
        forward = beta[:, None] * market[None, :] + rng.normal(
            scale=0.01, size=(n_sym, n_days))
        pooled = trend_mod.pool({f"S{i}": (scores[i], forward[i])
                                 for i in range(n_sym)})
        pooled_hits += abs(pooled.t_stat) >= 2
        premia = rs.slopes_by_day(
            (d, scores[i, d], forward[i, d])
            for i in range(n_sym) for d in range(n_days))
        values = [p for _, p in premia]
        mean, se, t = rs._mean_t(values)
        daily_hits += abs(t) >= 2
    assert pooled_hits / trials > 0.4
    assert daily_hits / trials < 0.12


# -- the desk --------------------------------------------------------------------


def _fading_premia():
    return _premia(_series(0.0004, 0.001, 190, 3) + _series(0.0, 0.001, 60, 4))


def test_a_change_is_reported_once_and_a_steady_verdict_not_at_all():
    desk = rs.ResearchDesk()
    steady = {"trend": (_premia(_series(0.0004, 0.001, 250, 2)), "bp")}
    assert [(n, old, f.verdict) for n, old, f in desk.run("d1", steady)] == [
        ("trend", rs.UNMEASURED, rs.HOLDING)]
    assert desk.run("d1", {"trend": (_fading_premia(), "bp")}) == [], \
        "a second run on the same day ran"
    assert desk.run("d2", steady) == []
    moved = desk.run("d3", {"trend": (_fading_premia(), "bp")})
    assert [(n, old, f.verdict) for n, old, f in moved] == [
        ("trend", rs.HOLDING, rs.FADING)]


def test_a_strategy_without_enough_history_is_not_announced():
    desk = rs.ResearchDesk()
    assert desk.run("d1", {"overnight": (_premia([0.001] * 30), "bp")}) == []
    assert desk.findings["overnight"].verdict == rs.UNMEASURED
    assert desk.caps() == {}


def test_the_findings_survive_a_restart_and_a_corrupt_file():
    desk = rs.ResearchDesk()
    desk.run("d1", {"trend": (_fading_premia(), "bp")})
    back = rs.ResearchDesk.from_dict(desk.as_dict())
    assert back.ran_on == "d1"
    assert back.caps() == desk.caps()
    assert back.run("d1", {"trend": (_fading_premia(), "bp")}) == []
    payload = desk.as_dict()
    payload["findings"]["trend"]["verdict"] = "sell everything"
    corrupt = rs.ResearchDesk.from_dict(payload)
    assert corrupt.caps() == {}
    assert "trend" not in corrupt.findings


# -- applied by the allocator ----------------------------------------------------

EQUITY = 1_000.0


def _record(name, returns, round_trips=20):
    rec = StrategyRecord(name=name, round_trips=round_trips)
    value = 0.0
    rec.daily.append(("d0000", 0.0))
    for i, r in enumerate(returns, start=1):
        value += r * EQUITY
        rec.daily.append((f"d{i:04d}", value))
    return rec


def _equity(records):
    return {d: EQUITY for rec in records for d, _ in rec.daily}


def test_a_fading_edge_gets_no_more_capital_whatever_its_record():
    rec = _record("trend", [0.004 + 0.001 * math.sin(i) for i in range(60)])
    w = ev.CapitalWeights()
    for k in range(6):
        w.revise(f"r{k}", [rec], _equity([rec]))
    assert w.multiplier("trend") > 1.0, "the fixture does not earn a boost"
    capped = ev.CapitalWeights()
    caps = {"trend": (rs.FADING_CAP, "edge fading: numbers")}
    for k in range(6):
        capped.revise(f"r{k}", [rec], _equity([rec]), caps=caps)
    assert capped.multiplier("trend") == 1.0
    assert "edge fading" in capped.note("trend")


def test_a_reversed_edge_is_cut_even_before_its_own_record_can_be_judged():
    rec = _record("overnight", [0.0] * 3, round_trips=1)
    w = ev.CapitalWeights()
    caps = {"overnight": (rs.REVERSED_CAP, "edge reversed")}
    for k in range(4):
        w.revise(f"r{k}", [rec], _equity([rec]), caps=caps)
    assert w.multiplier("overnight") == rs.REVERSED_CAP
    assert "gathering evidence" in w.note("overnight")


def test_research_can_hold_capital_down_but_never_lift_it():
    rec = _record("trend", [-0.004 - 0.001 * math.sin(i) for i in range(60)])
    w = ev.CapitalWeights()
    caps = {"trend": (rs.FADING_CAP, "edge fading")}
    for k in range(8):
        w.revise(f"r{k}", [rec], _equity([rec]), caps=caps)
    assert w.multiplier("trend") == ev.FLOOR


def test_a_cap_on_a_sleeve_the_allocator_does_not_size_is_ignored():
    rec = _record("sector", [0.0] * 3)
    w = ev.CapitalWeights()
    w.revise("r0", [rec], _equity([rec]), caps={"sector": (0.5, "x")})
    assert w.multiplier("sector") == 1.0


# -- in the session --------------------------------------------------------------


def test_split_daily_dates_each_night_by_the_morning_it_ended():
    from imperium.execution.bars import Bar
    from imperium.strategy import overnight as om

    day = 86_400_000
    bars = [Bar(open_time=(20_000 + i) * day, open=100.0 + i, high=101 + i,
                low=99.0 + i, close=100.5 + i, volume=1.0, closed=True)
            for i in range(5)]
    split = om.split_daily(bars)
    assert list(split.days) == [20_001, 20_002, 20_003, 20_004]
    assert split.overnight[0] == pytest.approx(math.log(101.0 / 100.5))


def test_a_trend_sample_is_dated_by_the_day_it_predicts(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    from imperium.execution.bars import Bar
    from imperium.session import TradingSession

    day = 86_400_000
    rng = random.Random(1)
    price, bars = 100.0, []
    for i in range(400):
        price *= math.exp(rng.gauss(0.0005, 0.01))
        bars.append(Bar(open_time=(19_000 + i) * day, open=price, high=price,
                        low=price, close=price, volume=1.0, closed=True))
    session = TradingSession()
    scores, forward = session._trend_observations("AAPL", bars)
    days = session._trend_days["AAPL"]
    assert len(days) == len(scores) == len(forward)
    by_day = {b.open_time // day: b.close for b in bars}
    for d, f in zip(days[:5], forward[:5]):
        assert f == pytest.approx(math.log(by_day[d] / by_day[d - 1]))


@pytest.mark.asyncio
async def test_the_desk_measures_every_strategy_from_the_venues_history(
        tmp_path, monkeypatch):
    """End to end through the real pull: the same samples the pooled
    estimates are fitted on, dated, one premium a day per strategy."""
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    from decimal import Decimal

    from imperium.execution.broker import PaperBroker
    from imperium.session import TradingSession
    from imperium.venues import registry
    from imperium.venues.alpaca.client import AlpacaClient
    from mock_venue import KEY, SECRET, MockVenue

    venue = MockVenue()
    venue.list_extra_crypto(16)
    venue.list_extra_equities(8)
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("100000")
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    try:
        await session.scan_universe()
        await session.refresh_daily_history(force=True)
    finally:
        await session.detach_client()

    premia = session.research_premia()
    assert set(premia) == {"trend", "cross_section", "overnight"}
    for name, (series, _) in premia.items():
        assert series, f"no daily premium measured for {name}"
        days = [d for d, _ in series]
        assert days == sorted(days) and len(set(days)) == len(days)
        assert all(math.isfinite(p) for _, p in series)
    # Coins trade at weekends; the ranking's days include them.
    weekdays = {dt.date.fromtimestamp(d * 86_400).weekday()
                for d, _ in premia["cross_section"][0]}
    assert weekdays & {5, 6}

    session._attribution_loaded = True
    session._run_research("2026-09-23")
    assert session.research.ran_on == "2026-09-23"
    assert set(session.research.findings) == set(premia)
    assert session.snapshot()["research"]


@pytest.mark.asyncio
async def test_the_brief_runs_the_desk_before_the_allocator_revises(
        tmp_path, monkeypatch):
    """Tonight's findings must bind tonight's revision, not tomorrow's."""
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    sys_path = __import__("sys").path
    sys_path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_daily_brief import _evening_session

    session, notifier = _evening_session()
    order = []
    monkeypatch.setattr(session, "_run_research",
                        lambda day: order.append(("research", day)))
    monkeypatch.setattr(session, "_mark_strategies",
                        lambda day: order.append(("mark", day)))
    await session._maybe_send_daily_brief()
    assert [step for step, _ in order] == ["research", "mark"]
    assert order[0][1] == order[1][1] == "2026-09-23"


def test_a_change_of_verdict_reaches_the_operator_once(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    from imperium.session import TradingSession

    session = TradingSession()
    session._attribution_loaded = True
    monkeypatch.setattr(session, "research_premia",
                        lambda: {"trend": (_fading_premia(), "bp/day")})
    session._run_research("2026-09-23")
    assert "RESEARCH" in session._pending_notice
    assert "Multi-day trend: edge fading" in session._pending_notice
    said = [e for e in session.telemetry.events() if e.get("source") == "research"]
    assert len(said) == 1
    session._pending_notice = ""
    session.research.ran_on = ""
    session._run_research("2026-09-24")
    assert session._pending_notice == ""

    row_block = session._strategies_block()
    assert row_block["rows"] == [] or all(
        "research" in r for r in row_block["rows"])


def test_the_daily_mark_holds_capital_to_what_research_found(tmp_path,
                                                            monkeypatch):
    """Through _mark_strategies, which the brief calls: a finding that never
    reaches the allocator is a report, not a control."""
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    from imperium.session import TradingSession

    session = TradingSession()
    session._attribution_loaded = True
    monkeypatch.setattr(session, "research_premia", lambda: {
        "trend": (_premia(_series(0.0004, 0.001, 190, 5)
                          + _series(-0.0006, 0.001, 60, 6)), "bp/day")})
    session._run_research("2026-09-23")
    book = session.attribution.book_for(session.broker.mode.value)
    book.records["trend"] = _record("trend", [0.0] * 3, round_trips=1)
    monkeypatch.setattr(book, "mark_day", lambda *a, **k: None)
    session._mark_strategies("2026-09-23")
    standing = session.capital_weights.standings["trend"]
    assert standing.target == rs.REVERSED_CAP
    assert "edge reversed" in standing.reason


def test_a_restart_brings_the_findings_back(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    from imperium.session import TradingSession

    first = TradingSession()
    first._attribution_loaded = True
    monkeypatch.setattr(first, "research_premia",
                        lambda: {"trend": (_fading_premia(), "bp/day")})
    first._run_research("2026-09-23")
    second = TradingSession()
    second._load_attribution()
    assert second.research.findings["trend"].verdict == rs.FADING
    assert second.research.caps()["trend"][0] == rs.FADING_CAP


def test_the_brief_says_what_the_desk_found():
    from imperium.notify import daily

    brief = daily.Brief(day="Mon", equity=100.0, day_start_equity=100.0,
                        cash=10.0, research=(
                            ("Multi-day trend", "holding", "edge holding: x"),
                            ("Crypto ranking", "fading", "edge fading: y")))
    text = daily.build(brief)
    assert "✅ Multi-day trend: holding" in text
    assert "⚠️ Crypto ranking: edge fading: y" in text
