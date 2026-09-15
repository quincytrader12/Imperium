"""The backtester, the ledger, and the properties that make them trustworthy.

The single most valuable test here is the lookahead one. Every other number a
backtest produces is worthless if a decision on day t was informed by day t+1,
and the failure is silent: the equity curve simply looks wonderful. So it is
tested structurally -- truncating the history must not change any decision made
before the truncation point -- rather than by inspecting results and judging
them plausible.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from imperium.execution.sleeve_ledger import (
    SleeveLedger, SleevePosition, trading_day,
)
from imperium.strategy import sector_backtest as bt
from imperium.strategy import sector_config as sc


def _series(n: int = 900, symbols: int = 5, seed: int = 11):
    """A synthetic history with genuinely distinct, increasing dates.

    The dates matter more than they look. An earlier version built them from
    ``i % 12`` and ``i % 28``, which repeats every few weeks -- and a test that
    keys results by date then silently collapses different days onto one
    another and compares numbers from unrelated points in the run. It reported
    a lookahead bug that did not exist.
    """
    import datetime as dt

    rng = np.random.default_rng(seed)
    dates: list[str] = []
    day = dt.date(2010, 1, 4)
    while len(dates) < n:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += dt.timedelta(days=1)
    assert len(set(dates)) == n, "the fixture produced duplicate dates"

    closes = {}
    for k in range(symbols):
        rets = rng.normal(0.0003 + 0.0001 * k, 0.012, n)
        closes[chr(ord("A") + k)] = 100 * np.exp(np.cumsum(rets))
    return closes, dates


# -- the lookahead test ---------------------------------------------------

def test_the_backtest_cannot_see_past_the_day_it_is_on():
    """THE test in this file, and the second attempt at it.

    The first version truncated the history once and compared a prefix well
    short of the cut. That cannot detect anything: a one-bar lookahead at day t
    reads day t+1, so only decisions within a bar of the boundary change, and
    comparing a window a hundred days earlier was comparing two runs that had
    seen identical data. Three real lookahead mutations survived it.

    This version replaces the future with a completely different one at many
    points through the series, and requires the equity curve to be identical up
    to and including the last day before the change. It is repeated across
    cut points because the rebalance threshold means most days place no trades
    at all -- on such a day a corrupted future changes nothing even when the
    code is reading it, so one sample proves little and twenty prove a lot.
    """
    closes, dates = _series(900)
    base = bt.run(closes, dates)
    base_by_date = dict(zip(base.dates, base.equity))

    checked = 0
    for cut in range(300, 880, 30):
        tampered = {s: c.copy() for s, c in closes.items()}
        for series in tampered.values():
            series[cut:] *= 2.5          # an entirely different future
        after = bt.run(tampered, dates)
        after_by_date = dict(zip(after.dates, after.equity))

        # Everything strictly before the change must be untouched. Day cut-1
        # is included deliberately: that is the day a one-bar lookahead reads
        # across, and excluding it is what made the first version blind.
        for day in dates[:cut]:
            if day in base_by_date and day in after_by_date:
                checked += 1
                assert base_by_date[day] == pytest.approx(
                    after_by_date[day], rel=1e-12, abs=1e-9), (
                    f"changing the future from {dates[cut]} moved the equity "
                    f"on {day} — the backtest is reading ahead")
    assert checked > 1000, "the comparison covered too little to prove anything"


def test_the_exit_is_judged_on_the_stop_carried_in_from_yesterday():
    """Prevents an ordering bug an equity curve cannot show.

    The exit must be measured against the stop as it stood at the end of
    yesterday. Trailing the stop up first and then testing would measure the
    position against a level that moved today, on the strength of the very
    band the fall is dragging around.

    Checked by replaying the stop independently with the pure functions, from
    each trade's entry to the day before its exit, and requiring the stop the
    backtester actually used to equal that number. An equality against an
    independently derived value, rather than an inequality that happens to
    hold: the carried stop is the running maximum of the lower band and is
    therefore always at or above today's band, so no inequality between the
    two can distinguish the orderings at all. A first attempt at this test
    tried exactly that and asserted something impossible.
    """
    closes, dates = _series(900)
    result = bt.run(closes, dates)
    exits = [t for t in result.trades if t.reason == "exit_stop"]
    assert exits, "no stop exits occurred, so the ordering proves nothing"

    from imperium.strategy import sector

    checked = 0
    for trade in exits:
        band = sector.bands(closes[trade.symbol])
        # The stop as of the close of the day before the exit: seeded on the
        # entry day's lower band, then trailed to the day before the exit.
        stop = float(band.lower[trade.opened])
        for day in range(trade.opened + 1, trade.closed):
            stop = sector.trail_stop(stop, float(band.lower[day]))
        assert trade.stop_used == pytest.approx(stop, rel=1e-9), (
            f"{trade.symbol} exited on a stop of {trade.stop_used}, but the "
            f"stop carried in from the previous day was {stop}")
        checked += 1
    assert checked >= 5


# -- costs ----------------------------------------------------------------

def test_slippage_makes_the_result_worse_and_is_actually_charged():
    """Prevents a backtest filling at the untouched close, which shows an edge
    that does not exist and will not survive contact with a venue."""
    closes, dates = _series(700)
    free = bt.run(closes, dates, config=bt.BacktestConfig(slippage_bps=0.0))
    costly = bt.run(closes, dates, config=bt.BacktestConfig(slippage_bps=50.0))
    assert costly.costs_paid > free.costs_paid == 0.0
    assert costly.equity[-1] < free.equity[-1]


def test_margin_interest_is_charged_only_above_full_investment():
    """At a 1.0 cap the sleeve never borrows, so it must never pay interest.
    Prevents a cost model that quietly taxes an unlevered book."""
    closes, dates = _series(700)
    unlevered = bt.run(closes, dates,
                       config=bt.BacktestConfig(max_leverage=1.0,
                                                margin_rate=0.20))
    assert unlevered.interest_paid == pytest.approx(0.0, abs=1e-9)


def test_borrowing_costs_money_at_a_two_times_cap():
    closes, dates = _series(700)
    levered = bt.run(closes, dates,
                     config=bt.BacktestConfig(max_leverage=2.0,
                                              margin_rate=0.20))
    assert levered.interest_paid > 0.0


def test_the_leverage_cap_bounds_the_exposure_actually_carried():
    """Prevents a cap that is computed and then not applied."""
    closes, dates = _series(700)
    for cap in (1.0, 2.0):
        result = bt.run(closes, dates,
                        config=bt.BacktestConfig(max_leverage=cap,
                                                 margin_rate=0.0))
        assert max(result.exposure) <= cap + 0.05, (
            f"exposure reached {max(result.exposure):.2f} under a {cap} cap")


# -- metrics --------------------------------------------------------------

def test_max_drawdown_is_measured_peak_to_trough():
    assert bt.max_drawdown(np.array([100.0, 120.0, 60.0, 90.0])) == \
        pytest.approx(0.5)
    assert bt.max_drawdown(np.array([100.0, 110.0, 120.0])) == \
        pytest.approx(0.0)


def test_beta_against_itself_is_one():
    """A sanity anchor for the regression: a series regressed on itself has a
    beta of exactly one and no alpha."""
    closes, dates = _series(600)
    result = bt.run(closes, dates)
    curve = np.asarray(result.equity, dtype=float)
    metrics = bt.measure(result, benchmark=curve)
    assert metrics.beta == pytest.approx(1.0, abs=1e-9)
    assert metrics.alpha == pytest.approx(0.0, abs=1e-9)


def test_the_sanity_check_flags_a_result_that_is_too_good():
    """The point of the check: a spectacular backtest of this strategy is
    evidence of a bug, not of an edge."""
    notes = bt.sanity_check(bt.Metrics(sharpe=2.4, max_drawdown=0.04,
                                       cagr=0.45, trades=10))
    joined = " ".join(notes).lower()
    assert "lookahead" in joined
    assert len(notes) >= 3


def test_the_sanity_check_is_quiet_on_a_plausible_result():
    """Prevents a warning that fires on everything and is therefore ignored."""
    assert bt.sanity_check(bt.Metrics(sharpe=0.6, max_drawdown=0.24,
                                      cagr=0.077, trades=300)) == []


# -- the sleeve ledger ----------------------------------------------------

def test_a_second_run_on_the_same_day_is_refused():
    """The brief's idempotency requirement. Without it a restart mid-session
    recomputes the same targets and trades the difference a second time."""
    ledger = SleeveLedger()
    assert ledger.already_ran("2026-09-15") is False
    ledger.complete_run("2026-09-15")
    assert ledger.already_ran("2026-09-15") is True
    assert ledger.already_ran("2026-09-16") is False


def test_another_strategy_holding_the_symbol_does_not_move_this_sleeve():
    """The brief's ledger requirement, and the reason the file exists.

    Alpaca nets positions by symbol across the account. If this sleeve sized or
    exited from the account position it would sell another strategy's shares to
    close its own, and both books would then be wrong with no way to tell which.
    """
    ledger = SleeveLedger()
    ledger.open_position("XLK", 2.0, 100.0, 90.0, "2026-09-15")

    account_position = 7.0      # this sleeve's 2 plus somebody else's 5
    assert ledger.held("XLK").quantity == 2.0
    assert ledger.held("XLK").quantity != account_position

    ledger.close_position("XLK")
    assert ledger.held("XLK").quantity == 0.0
    assert ledger.longs() == []


def test_closing_a_position_clears_its_stop_too():
    """Prevents a stop left behind on a flat symbol firing against the next
    entry on the day it opens, at a level from a position that no longer
    exists."""
    ledger = SleeveLedger()
    ledger.open_position("XLE", 3.0, 80.0, 70.0, "2026-09-15")
    ledger.close_position("XLE")
    assert math.isnan(ledger.held("XLE").stop)


def test_the_stop_survives_a_restart():
    """A forgotten stop is an unbounded position."""
    ledger = SleeveLedger()
    ledger.open_position("XLV", 1.5, 140.0, 130.0, "2026-09-15")
    ledger.raise_stop("XLV", 135.0)
    ledger.save()

    back = SleeveLedger.load()
    assert back.held("XLV").stop == pytest.approx(135.0)
    assert back.held("XLV").quantity == pytest.approx(1.5)


def test_saving_the_sleeve_does_not_clobber_the_rest_of_the_book():
    """The same read-modify-write rule the rest of this state file follows.
    Replacing the file to store one section would lose the overnight holdings,
    the trend holdings and the attached credential -- on a restart, the book."""
    from imperium import config

    config.ensure_home()
    config.state_path().write_text(
        json.dumps({"overnight": {"AAPL": {"weight": 0.2}},
                    "attached": "alpaca-paper"}),
        encoding=config.TEXT_ENCODING)

    ledger = SleeveLedger()
    ledger.open_position("XLI", 4.0, 120.0, 110.0, "2026-09-15")
    ledger.save()

    saved = json.loads(config.state_path().read_text(
        encoding=config.TEXT_ENCODING))
    assert saved["overnight"] == {"AAPL": {"weight": 0.2}}
    assert saved["attached"] == "alpaca-paper"
    assert saved["sector_trend"]["positions"]["XLI"]["quantity"] == 4.0


def test_a_missing_stop_reloads_as_missing_rather_than_as_a_number():
    """NaN is not JSON. Storing it as null and reading it back as NaN keeps a
    "no stop yet" position from acquiring a stop of zero, which would never
    fire."""
    position = SleevePosition(symbol="X", quantity=1.0)
    assert position.as_dict()["stop"] is None
    assert math.isnan(SleevePosition.from_dict(position.as_dict()).stop)


def test_the_trading_day_is_eastern_not_utc():
    """A UTC day boundary puts a 15:45 ET run on one date in summer and
    another in winter, and the idempotency guard would then let a second run
    through on exactly the days it mattered."""
    import datetime as dt
    late = dt.datetime(2026, 9, 16, 1, 30, tzinfo=dt.timezone.utc)
    assert trading_day(late) == "2026-09-15"


# -- config ---------------------------------------------------------------

def test_the_sleeve_is_off_until_somebody_turns_it_on():
    assert sc.from_environment().enabled is False


def test_the_smallest_tradable_weight_reflects_the_venues_dollar_floor():
    """The arithmetic that decides whether this strategy can run on a given
    account at all. On $70 at a 20% allocation the sleeve is $14, so a weight
    under about 7% is an order Alpaca refuses."""
    cfg = sc.SectorTrendConfig(allocation=0.20)
    assert cfg.sleeve_equity(70.0) == pytest.approx(14.0)
    assert cfg.smallest_tradable_weight(70.0) == pytest.approx(1.0 / 14.0)
    assert cfg.smallest_tradable_weight(0.0) == float("inf")


def test_a_broken_environment_value_falls_back_rather_than_crashing(monkeypatch):
    """A typo in an env var must not stop the terminal starting."""
    monkeypatch.setenv("SECTOR_TREND_ALLOCATION", "not-a-number")
    monkeypatch.setenv("SECTOR_TREND_EXEC_MODE", "whenever")
    cfg = sc.from_environment()
    assert cfg.allocation == pytest.approx(0.20)
    assert cfg.exec_mode == "near_close"
