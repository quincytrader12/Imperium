"""Selecting overnight stocks by their own record (Lou, Polk & Skouras).

The pooled estimate says whether there is an overnight premium at all; the
ranking says which stocks it lives in. What must hold: the rank uses a year
of each stock's own nights and nothing older, a stock too new to rank is
not carried on the strength of other stocks' history, a cross-section too
small to rank is not filtered at all, and every refusal says why.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imperium.execution.bars import Bar
from imperium.strategy import overnight as ov

DAY_MS = 86_400_000


def _split(means_bps, sd=0.004, seed=0):
    """A split whose overnight returns are the given means plus noise."""
    rng = np.random.default_rng(seed)
    values = np.asarray(means_bps, dtype=float) / 10_000 + rng.normal(0, sd, len(means_bps))
    return ov.SessionSplit(values, np.zeros(len(values)), len(values))


def _bars(drift_bps, nights=260, seed=0):
    """Daily bars whose overnight returns average ``drift_bps`` exactly, with
    a last night that is ordinary (not an event gap)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.004, nights)
    noise -= noise.mean()
    noise[-1] = 0.0
    gaps = drift_bps / 10_000 + noise
    bars, close = [], 100.0
    for i, gap in enumerate(np.concatenate([[0.0], gaps])):
        open_ = close * math.exp(gap)
        close = open_ * math.exp(-gap)                    # flat intraday
        bars.append(Bar(open_time=1_600_000_000_000 + i * DAY_MS, open=open_,
                        high=max(open_, close), low=min(open_, close),
                        close=close, volume=1e6, closed=True))
    return bars


POOLED = ov.PooledDrift(mean_bps=4.0, t_stat=5.0, observations=5000,
                        symbols=20, vol_bps=40.0)


# -- the ranking -----------------------------------------------------------------------


def test_the_rank_orders_by_each_stocks_own_mean():
    splits = {s: _split([m] * 200, sd=0.0) for s, m in
              zip("ABCDE", (1.0, 5.0, 3.0, -2.0, 4.0))}
    r = ov.rank(splits)
    assert r.percentiles == {"D": 0.0, "A": 0.25, "C": 0.5, "E": 0.75, "B": 1.0}
    assert r.means_bps["B"] == pytest.approx(5.0)


def test_ties_share_their_place():
    splits = {s: _split([m] * 200, sd=0.0) for s, m in zip("ABC", (1.0, 1.0, 9.0))}
    r = ov.rank(splits)
    assert r.percentiles["A"] == r.percentiles["B"] == pytest.approx(0.25)
    assert r.percentiles["C"] == 1.0


def test_only_the_last_year_counts():
    # Strong two years ago, weak for the last year: ranked on the last year.
    old_star = _split([50.0] * 200 + [-1.0] * ov.RANK_NIGHTS, sd=0.0)
    steady = _split([2.0] * (200 + ov.RANK_NIGHTS), sd=0.0)
    r = ov.rank({"OLD": old_star, "NOW": steady})
    assert r.percentiles == {"OLD": 0.0, "NOW": 1.0}
    assert r.means_bps["OLD"] == pytest.approx(-1.0)


def test_too_little_own_history_is_not_ranked():
    r = ov.rank({"NEW": _split([9.0] * (ov.MIN_RANK_NIGHTS - 1)),
                 "OLD": _split([1.0] * ov.MIN_RANK_NIGHTS)})
    assert set(r.percentiles) == {"OLD"}
    assert ov.rank({}).peers == 0


def test_a_small_cross_section_is_not_filtered():
    few = {f"S{i}": _split([float(i)] * 200, sd=0.0)
           for i in range(ov.MIN_RANK_PEERS - 1)}
    assert not ov.rank(few).active
    assert "carrying without the ranking" in ov.rank(few).describe()
    few["LAST"] = _split([0.0] * 200, sd=0.0)
    assert ov.rank(few).active
    assert not ov.rank(few, min_rank=0.0).active
    assert ov.rank(few).describe() == ("carrying only the top 50% of 10 symbols "
                                       "by their own year of overnight returns")


# -- the decision -----------------------------------------------------------------------


def _ranking(**percentiles):
    names = dict(percentiles)
    names.update({f"P{i}": 0.5 for i in range(ov.MIN_RANK_PEERS)})
    return ov.Ranking(percentiles=names,
                      means_bps={k: 1.0 for k in names}, min_rank=0.5)


def test_a_stock_whose_returns_do_not_come_overnight_is_not_carried():
    s = ov.evaluate(_bars(4.0), pooled=POOLED,
                    ranking=_ranking(WEAK=0.2), symbol="WEAK")
    assert not s.eligible
    assert "ranks 20% of 11, under the 50% line" in s.reason


def test_a_stock_in_the_upper_half_is_carried_and_says_where_it_ranks():
    s = ov.evaluate(_bars(4.0), pooled=POOLED,
                    ranking=_ranking(GOOD=0.8), symbol="GOOD")
    assert s.eligible and "ranks 80% of 11" in s.reason
    # The line itself is in.
    assert ov.evaluate(_bars(4.0), pooled=POOLED, ranking=_ranking(EDGE=0.5),
                       symbol="EDGE").eligible


def test_a_stock_too_new_to_rank_is_not_carried_on_others_history():
    s = ov.evaluate(_bars(4.0), pooled=POOLED, ranking=_ranking(), symbol="NEW")
    assert not s.eligible and f"needs {ov.MIN_RANK_NIGHTS}" in s.reason


def test_without_an_active_ranking_the_decision_is_as_before():
    before = ov.evaluate(_bars(4.0), pooled=POOLED)
    inactive = ov.Ranking(percentiles={"X": 0.0}, means_bps={"X": 0.0})
    after = ov.evaluate(_bars(4.0), pooled=POOLED, ranking=inactive, symbol="X")
    assert before.eligible and after.eligible
    assert after.expected_edge_bps == before.expected_edge_bps


def test_the_market_reasons_come_before_the_ranking():
    """No premium at all is the reason to give, not the stock's rank."""
    flat = ov.PooledDrift(mean_bps=-1.0, t_stat=-3.0, observations=5000,
                          symbols=20, vol_bps=40.0)
    s = ov.evaluate(_bars(4.0), pooled=flat, ranking=_ranking(WEAK=0.1),
                    symbol="WEAK")
    assert "no premium to harvest" in s.reason


# -- in the session -----------------------------------------------------------------------


class _Client:
    def __init__(self, drifts):
        self.drifts = drifts

    async def bars(self, symbols, **kw):
        out = {}
        for i, s in enumerate(symbols):
            out[s] = [{"t": b.open_time, "o": b.open, "h": b.high, "l": b.low,
                       "c": b.close, "v": b.volume}
                      for b in _bars(self.drifts[s], seed=i)]
        return out


@pytest.mark.asyncio
async def test_the_session_ranks_the_universe_and_hands_every_engine_the_ranking(monkeypatch):
    from imperium.session import TradingSession

    drifts = {f"S{i:02d}": float(i) for i in range(12)}
    session = TradingSession()
    session.client = _Client(drifts)
    session.universe = list(drifts)
    await session.refresh_daily_history(force=True)
    ranking = session.overnight_ranking
    assert ranking is not None and ranking.active and ranking.peers == 12
    assert ranking.percentiles["S11"] == 1.0 and ranking.percentiles["S00"] == 0.0
    assert all(e.overnight_ranking is ranking for e in session.engines.values())
    assert session.engine("LATER").overnight_ranking is ranking
    assert "top 50% of 12" in session.overnight_note
    client, session.client = session.client, None
    assert "top 50%" in session.snapshot()["overnight"]["ranking"]
    session.client = client

    monkeypatch.setenv("IMPERIUM_OVERNIGHT_MIN_RANK", "0")
    await session.refresh_daily_history(force=True)
    assert session.overnight_ranking is None
    monkeypatch.setenv("IMPERIUM_OVERNIGHT_MIN_RANK", "0.7")
    await session.refresh_daily_history(force=True)
    assert session.overnight_ranking.min_rank == pytest.approx(0.7)
    monkeypatch.setenv("IMPERIUM_OVERNIGHT_MIN_RANK", "nonsense")
    await session.refresh_daily_history(force=True)
    assert session.overnight_ranking.min_rank == ov.MIN_RANK


def test_the_engine_decides_with_the_ranking_it_was_handed():
    from test_overnight import _engine, _warm

    from imperium.strategy.overnight import SessionPhase

    engine = _warm(_engine("WEAK"))
    engine.session_phase = SessionPhase.CLOSING
    engine.pooled_drift = POOLED
    engine.daily_bars = _bars(4.0)
    engine.overnight_ranking = _ranking(WEAK=0.1)
    d = engine.evaluate()
    assert d.strategy == "overnight" and "under the 50% line" in d.reason


@pytest.mark.parametrize("raw,expected", [("0.95", 0.9), ("-1", 0.0), ("inf", 0.5),
                                          ("0.6", 0.6)])
def test_the_setting_is_bounded(monkeypatch, raw, expected):
    from imperium.session import _overnight_min_rank

    monkeypatch.setenv("IMPERIUM_OVERNIGHT_MIN_RANK", raw)
    assert _overnight_min_rank() == pytest.approx(expected)
