"""Sector Trend: the indicators, against numbers worked out by hand.

Every expected value in the indicator tests below was computed independently
of the code under test. That is the only kind of test worth writing here --
asserting that ``ema()`` equals ``ema()`` would pass against any
implementation, including a wrong one.

The rest of the file guards the four properties that decide whether a backtest
of this strategy means anything: entries look at yesterday's band, the stop
never falls, the leverage cap actually binds, and a held position is not
churned for a rounding error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imperium.strategy import sector


# -- indicators, against hand-computed values ----------------------------

def test_ema_matches_a_hand_computed_series():
    """closes [1,2,3,4,5], span 3.

    Seed = mean(1,2,3) = 2 at index 2. alpha = 2/(3+1) = 0.5.
    index 3 = 0.5*4 + 0.5*2 = 3.  index 4 = 0.5*5 + 0.5*3 = 4.
    """
    out = sector.ema(np.array([1, 2, 3, 4, 5], float), 3)
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.0)


def test_mean_absolute_change_matches_a_hand_computed_series():
    """closes [10,12,11,15], window 2.

    |ΔP| = [2, 1, 4].
    t=2 averages the two changes ending at t: (2+1)/2 = 1.5.
    t=3 averages (1+4)/2 = 2.5.
    """
    out = sector.mean_absolute_change(np.array([10, 12, 11, 15], float), 2)
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2] == pytest.approx(1.5)
    assert out[3] == pytest.approx(2.5)


def test_rolling_max_and_min_match_hand_computed_windows():
    """closes [5,3,4,7,2], window 3."""
    closes = np.array([5, 3, 4, 7, 2], float)
    up = sector.rolling_max(closes, 3)
    down = sector.rolling_min(closes, 3)
    assert math.isnan(up[0]) and math.isnan(up[1])
    assert [up[2], up[3], up[4]] == [5.0, 7.0, 7.0]
    assert [down[2], down[3], down[4]] == [3.0, 3.0, 2.0]


def test_the_rolling_window_includes_today():
    """Prevents an off-by-one that would make the bands describe yesterday.

    DonchianUp20_t is the highest close of the last twenty *including* day t.
    A window that stopped at t-1 would be a different indicator, and entries
    against it would fire a day late for the whole backtest.
    """
    closes = np.array([1, 1, 1, 9], float)
    assert sector.rolling_max(closes, 2)[3] == 9.0


def test_the_upper_band_is_the_lower_of_the_two_entry_triggers():
    """min for the upper, max for the lower: both mean "whichever binds
    sooner". Getting either backwards makes the strategy trade a different
    rule that still looks plausible on a chart."""
    closes = np.asarray(
        [100 + 10 * math.sin(i / 3.0) + i * 0.4 for i in range(120)], float)
    band = sector.bands(closes)
    at = -1
    assert band.upper[at] == pytest.approx(
        min(band.donchian_up[at], band.keltner_up[at]))
    assert band.lower[at] == pytest.approx(
        max(band.donchian_down[at], band.keltner_down[at]))


def test_bands_are_nan_until_both_legs_have_a_full_window():
    """Prevents a band computed from one leg while the other has no history.

    ``np.fmax`` ignores a NaN operand and returns the other, which would have
    published a Donchian-only lower band for the twenty bars before the
    Keltner leg existed -- presented as though both legs had agreed. This test
    found exactly that in the first version of ``bands()``. The difference is
    invisible on a chart and changes every early trade in a backtest.
    """
    closes = np.asarray([100 + math.sin(i / 4.0) for i in range(120)], float)
    band = sector.bands(closes)

    # Neither band exists before its longest input does.
    assert math.isnan(band.lower[10])
    assert math.isnan(band.upper[10])
    # The Donchian leg alone is ready before the Keltner leg. The published
    # band must still be absent at that point.
    ready_donchian = sector.DONCHIAN_DOWN_DAYS - 1
    assert math.isfinite(band.donchian_down[ready_donchian])
    assert math.isnan(band.keltner_down[ready_donchian])
    assert math.isnan(band.lower[ready_donchian]), (
        "the lower band was published from the Donchian leg alone")
    assert band.usable_from >= sector.DONCHIAN_DOWN_DAYS


# -- the lookahead property ----------------------------------------------

def test_an_entry_reads_yesterdays_band_and_not_todays():
    """THE test in this file.

    The bands include the current bar by construction, so an entry rule that
    tested today's close against today's band would be comparing a number with
    a maximum that already contains it. That is the single easiest way to
    produce a backtest that looks wonderful and is worthless.

    Asserted on which index the function reads, rather than on a constructed
    price path. A path only demonstrates the bug where the two answers happen
    to differ -- and on the Donchian leg they never do, because
    ``max(P_t..P_{t-19}) >= P_t`` is equivalent to
    ``max(P_{t-1}..P_{t-20}) >= P_t``. Poisoning one index and then the other
    proves which one decides, on every leg, always.
    """
    closes = np.asarray([100 + math.sin(i / 5.0) for i in range(120)], float)
    band = sector.bands(closes)
    t = len(closes) - 1

    # Yesterday's band unreachable, today's trivially clear: a correct rule
    # refuses, a lookahead rule enters.
    band.upper[t - 1] = 1e9
    band.upper[t] = 0.0
    assert sector.entry_signal(closes, band, t) is False, (
        "the entry read today's band -- this is a lookahead bug")

    # And the reverse, so the test cannot pass by always refusing.
    band.upper[t - 1] = 0.0
    band.upper[t] = 1e9
    assert sector.entry_signal(closes, band, t) is True


def test_a_breakout_above_a_settled_band_is_an_entry():
    """The ordinary case, so the lookahead guard above cannot be satisfied by
    a rule that never trades at all."""
    closes = np.array([10.0] * 60 + [50.0])
    band = sector.bands(closes)
    t = len(closes) - 1
    assert band.upper[t - 1] < 50.0
    assert sector.entry_signal(closes, band, t) is True


def test_no_entry_before_there_is_a_band_to_clear():
    closes = np.arange(1, 20, dtype=float)
    band = sector.bands(closes)
    assert sector.entry_signal(closes, band, 5) is False


# -- the stop -------------------------------------------------------------

@pytest.mark.parametrize("previous,today,expected", [
    (10.0, 12.0, 12.0),     # the band rose: the stop follows it up
    (12.0, 9.0, 12.0),      # the band fell: the stop stays put
    (12.0, 12.0, 12.0),
    (float("nan"), 8.0, 8.0),   # first day of a position
    (11.0, float("nan"), 11.0),  # no band today: keep what we had
])
def test_the_trailing_stop_never_moves_down(previous, today, expected):
    """A stop that can fall is not a stop. This one line is what bounds the
    loss on any single trade in this sleeve."""
    assert sector.trail_stop(previous, today) == pytest.approx(expected)


def test_the_stop_is_monotone_over_a_whole_falling_series():
    """The property, not one case: over a series that rises then falls hard,
    the stop must be non-decreasing at every step."""
    closes = np.asarray([100 + i for i in range(80)]
                        + [180 - 4 * i for i in range(40)], float)
    band = sector.bands(closes)
    stop = float("nan")
    seen: list[float] = []
    for t in range(band.usable_from, len(closes)):
        stop = sector.trail_stop(stop, band.lower[t])
        seen.append(stop)
    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:])), "the stop fell"


def test_an_exit_uses_the_stop_carried_in_from_yesterday():
    """Prevents a position surviving on a stop that only moved because of the
    very fall that should have closed it."""
    assert sector.exit_signal(99.0, 100.0) is True
    assert sector.exit_signal(100.0, 100.0) is False   # strictly below
    assert sector.exit_signal(101.0, 100.0) is False
    assert sector.exit_signal(50.0, float("nan")) is False


# -- sizing ---------------------------------------------------------------

def test_weights_divide_by_the_universe_not_by_the_number_of_positions():
    """Prevents the sleeve growing each position precisely when few symbols
    qualify -- which is when the market is least hospitable. The paper divides
    the volatility budget by the size of the active universe, so a thin signal
    stays a small book."""
    sigmas = {"A": 0.01, "B": 0.01}
    one = sector.target_weights(sigmas, ["A"], universe_size=19,
                                target_vol=0.015, max_leverage=1.0)
    two = sector.target_weights(sigmas, ["A", "B"], universe_size=19,
                                target_vol=0.015, max_leverage=1.0)
    assert one.weights["A"] == pytest.approx(two.weights["A"])
    assert one.gross == pytest.approx(two.gross / 2)


def test_a_hand_computed_weight():
    """(0.015 / 19) / 0.012 = 0.06578947..."""
    out = sector.target_weights({"XLK": 0.012}, ["XLK"], universe_size=19,
                                target_vol=0.015, max_leverage=1.0)
    assert out.weights["XLK"] == pytest.approx(0.015 / 19 / 0.012)
    assert out.capped is False


def test_the_leverage_cap_scales_every_weight_proportionally():
    """Prevents a cap that truncates the largest position instead of scaling
    the book, which would silently change the strategy's mix."""
    sigmas = {s: 0.002 for s in "ABCDEFGH"}
    out = sector.target_weights(sigmas, list("ABCDEFGH"), universe_size=8,
                                target_vol=0.015, max_leverage=1.0)
    assert out.capped is True
    assert out.gross == pytest.approx(1.0)
    # Equal sigmas must still give equal weights after the cap.
    assert len(set(round(w, 9) for w in out.weights.values())) == 1
    assert out.gross_before_cap > 1.0


def test_a_symbol_with_no_volatility_estimate_gets_no_weight():
    """Prevents an invented sigma becoming an invented position size."""
    out = sector.target_weights({"A": float("nan"), "B": 0.01}, ["A", "B"],
                                universe_size=19, target_vol=0.015,
                                max_leverage=1.0)
    assert "A" not in out.weights
    assert "B" in out.weights


def test_two_times_leverage_is_reachable_only_by_raising_the_cap():
    sigmas = {s: 0.002 for s in "ABCDEFGH"}
    at_one = sector.target_weights(sigmas, list("ABCDEFGH"), universe_size=8,
                                   target_vol=0.015, max_leverage=1.0)
    at_two = sector.target_weights(sigmas, list("ABCDEFGH"), universe_size=8,
                                   target_vol=0.015, max_leverage=2.0)
    assert at_one.gross == pytest.approx(1.0)
    assert at_two.gross == pytest.approx(2.0)


# -- rebalance threshold --------------------------------------------------

@pytest.mark.parametrize("current,target,expected", [
    (100.0, 100.0, False),
    (100.0, 110.0, False),   # 10% drift, under the 25% threshold
    (100.0, 124.0, False),
    (100.0, 125.0, True),    # exactly at it
    (100.0, 200.0, True),
    (100.0, 70.0, True),     # 30% down
])
def test_the_rebalance_threshold_skips_small_changes(current, target, expected):
    """Prevents paying a spread every day to correct a weight by a percent,
    which turns a low-turnover strategy into a high-turnover one without
    changing a single signal."""
    assert sector.needs_rebalance(current, target, 0.25) is expected


def test_opening_a_new_position_is_never_suppressed_by_the_threshold():
    """Entries and exits always execute; only a held symbol is throttled."""
    assert sector.needs_rebalance(0.0, 5.0, 0.25) is True


# -- the order of the exit test and the stop trail ------------------------

def test_the_stop_is_tested_before_it_is_trailed():
    """The discriminating case, and the reason this is a function.

    Carried stop 100, today's close 105, today's lower band risen to 110.
    Testing first: no exit, and the stop becomes 110 tomorrow. Trailing first:
    the stop is 110 before the test, 105 is below it, and the position is
    closed on a level that did not exist when the day began.

    The forty-day low rises whenever an old low rolls out of the window, so
    this is rare rather than impossible -- which is exactly the kind of bug
    that survives a random backtest and shows up on a real account.
    """
    step = sector.step_position(close_today=105.0, stop_carried_in=100.0,
                                lower_band_today=110.0)
    assert step.exited is False
    assert step.stop == pytest.approx(110.0)


def test_a_real_break_of_the_carried_stop_still_exits():
    """So the test above cannot be satisfied by a rule that never exits."""
    step = sector.step_position(close_today=95.0, stop_carried_in=100.0,
                                lower_band_today=110.0)
    assert step.exited is True
    assert math.isnan(step.stop)


def test_a_surviving_position_keeps_its_stop_when_the_band_falls():
    step = sector.step_position(close_today=105.0, stop_carried_in=100.0,
                                lower_band_today=80.0)
    assert step.exited is False
    assert step.stop == pytest.approx(100.0)
