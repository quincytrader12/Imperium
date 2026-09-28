"""The give-back ratchet.

Written from a live week. A crypto position ran to roughly +43%, the account
marked $82.52, and by the evening the book was $78.78 with the position still
open — and still, by every rule the program had, perfectly fine. Every exit in
the main engine is a *signal* exit, and a signal exit says nothing about what
the position is worth.

The risk in a rule like this is the opposite failure: closing winners early,
which is the one thing a book that pays for many small losers with a few large
gains cannot afford. So most of these tests are about what it must **not** do.
"""

from __future__ import annotations

import pytest

from imperium.execution import protect


# -- what it must not do ----------------------------------------------------


def test_a_gain_smaller_than_the_round_trip_is_not_protected():
    """Arming here would close a position whose gain does not cover the cost
    of having opened and closed it."""
    verdict = protect.assess(gain=0.002, peak=0.004,
                             round_trip_bps=58, sigma=0.01)
    assert not verdict.armed
    assert not verdict.exit_now
    assert "round trip" in verdict.reason


def test_a_large_winner_is_not_closed_by_a_small_pullback():
    """The band is a fraction of the peak, so it widens as the position wins.

    A fixed band would stop a +40% position out on a 2% wiggle, which is how
    a trailing rule turns into a profit target by accident.
    """
    verdict = protect.assess(gain=0.38, peak=0.40,
                             round_trip_bps=58, sigma=0.01)
    assert verdict.armed
    assert not verdict.exit_now


def test_ordinary_volatility_does_not_close_a_position():
    """A coin whose daily range is 5% must not be closed for moving 4%.

    Without the noise floor the band at a 9% peak is 3%, and this position
    would be shut on a day it did nothing unusual.
    """
    quiet = protect.assess(gain=0.05, peak=0.09, round_trip_bps=58, sigma=0.05)
    assert not quiet.exit_now, (
        f"closed inside one bar of noise: band {quiet.band:.2%}")
    assert quiet.band >= protect.NOISE_SIGMAS * 0.05


def test_it_is_a_ratchet_and_never_lowers_the_mark():
    peak = protect.PositionPeak()
    for gain in (0.05, 0.12, 0.08, 0.30, 0.11):
        peak.observe(gain)
    assert peak.peak == pytest.approx(0.30)


def test_a_position_that_never_went_up_is_left_alone():
    """Losers are the stop's business, not this rule's. A position underwater
    from the first tick has no high-water mark to protect."""
    verdict = protect.assess(gain=-0.20, peak=0.0,
                             round_trip_bps=58, sigma=0.03)
    assert not verdict.armed
    assert not verdict.exit_now


def test_an_unmeasurable_gain_does_nothing():
    assert not protect.assess(gain=float("nan"), peak=0.4,
                              round_trip_bps=58, sigma=0.02).exit_now


# -- what it must do --------------------------------------------------------


def test_the_live_position_that_prompted_this_would_have_closed():
    """The actual numbers off the venue dashboard.

    FIL: 30.709 bought at $0.812054, marked $1.04 — up 28.1%. The book's high
    was $3.74 above where it ended, essentially all of it this position, which
    puts the peak near +43%. Handing back 15 points of a 43-point gain is the
    case this rule exists for.
    """
    entry, now, qty = 0.812054, 1.04, 30.709164661
    gain = (now - entry) / entry
    peak = ((now + 3.74 / qty) - entry) / entry

    verdict = protect.assess(gain=gain, peak=peak,
                             round_trip_bps=58, sigma=0.06)
    assert verdict.exit_now, (
        f"gave back {peak - gain:.1%} of a {peak:+.1%} best and held on")
    assert "gave back" in verdict.reason


def test_the_band_is_reported_so_the_exit_can_be_argued_with():
    verdict = protect.assess(gain=0.20, peak=0.40,
                             round_trip_bps=58, sigma=0.02)
    assert verdict.exit_now
    assert verdict.band == pytest.approx(0.40 * protect.GIVE_BACK_FRACTION)
    assert f"{verdict.band:.2%}" in verdict.reason


def test_a_missing_cost_estimate_still_arms_at_something_sane():
    """A venue reporting no costs would otherwise arm the ratchet on the first
    tick in profit and close every position at its first pullback."""
    assert protect.arm_at(0.0) == protect.ARM_MIN_GAIN
    assert protect.arm_at(float("nan")) == protect.ARM_MIN_GAIN
    assert protect.arm_at(58) > protect.ARM_MIN_GAIN
