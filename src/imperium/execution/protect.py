"""Protecting an unrealised gain.

The gap this fills, from a live week: a crypto position ran to roughly +24%,
the account marked $82.52 at its high, and nothing closed it. The book was
$78.78 by the evening. The position was never wrong -- it is still green --
but a quarter of what it had made was handed back, because the only exits the
main engine has are *signal* exits. The trend sleeve trails a stop; the engine
that took this trade has no concept of a high-water mark at all, so a position
can round-trip its entire gain and every rule in the program will agree that
nothing happened.

This is the missing rule, and it is deliberately not a profit target. A target
caps the winners, which is the one thing a book that pays for its losers with
a few large gains cannot afford. What this does is ratchet: it lets a position
run as far as it likes, and only ever asks how much of its own best it is
allowed to give back.

Three numbers, each bounded by something real rather than chosen:

* **Nothing is protected until the gain is worth more than the round trip.**
  Arming below that would exit positions whose gain does not cover the cost of
  having taken them.
* **The give-back band is a fraction of the peak**, so it widens as the
  position wins and a large winner is never stopped out by a small pullback.
* **The band is never narrower than ordinary noise.** A name that moves 5% on
  a quiet day must not be closed by a 2% wiggle, and a fraction-only rule does
  exactly that once the peak is small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: How many round trips of gain a position must show before anything is
#: protected. Below this there is nothing to protect: the position has not yet
#: made back what it cost to open and close.
ARM_ROUND_TRIPS = 3.0

#: An absolute floor under the arming threshold, for the case where the cost
#: estimate is missing or implausibly small. Without it a venue reporting zero
#: costs would arm the ratchet on the first tick in profit and close every
#: position at its first pullback.
ARM_MIN_GAIN = 0.01

#: How much of its best a position may give back before it is closed.
#:
#: A third, which is the loosest of the usual trailing choices and is loose on
#: purpose. This rule runs on top of the signal exits rather than instead of
#: them, so it should only fire when a position has given back enough that the
#: move it was holding for is plainly over.
GIVE_BACK_FRACTION = 1.0 / 3.0

#: The narrowest the band may ever be, in units of the bar's own volatility.
#:
#: Without this the rule is self-defeating on a volatile name: at a 9% peak the
#: band is 3%, and a coin whose ordinary daily range is 5% trips it on noise,
#: closing a position for moving the way it always moves.
NOISE_SIGMAS = 1.5


@dataclass
class PositionPeak:
    """The high-water mark of one position's unrealised gain.

    A ratchet: ``peak`` only ever rises while the position is open, and the
    whole record is dropped when it closes. Keeping it across a close would
    protect the *next* position in that symbol against a high-water mark it
    never reached.
    """

    peak: float = 0.0

    def observe(self, gain: float) -> float:
        if math.isfinite(gain):
            self.peak = max(self.peak, gain)
        return self.peak


@dataclass(frozen=True)
class Protection:
    """What the ratchet says about one position, and why."""

    exit_now: bool
    armed: bool
    gain: float
    peak: float
    band: float
    reason: str


def arm_at(round_trip_bps: float) -> float:
    """The gain at which protecting this position starts to mean something."""
    if not math.isfinite(round_trip_bps) or round_trip_bps <= 0:
        return ARM_MIN_GAIN
    return max(ARM_MIN_GAIN, ARM_ROUND_TRIPS * round_trip_bps / 10_000.0)


def assess(*, gain: float, peak: float, round_trip_bps: float,
           sigma: float) -> Protection:
    """Whether a position has given back enough of its best to be closed.

    ``gain`` and ``peak`` are fractions of the entry price; ``sigma`` is the
    per-bar return standard deviation, used only to stop the band closing
    inside ordinary noise.
    """
    if not math.isfinite(gain) or not math.isfinite(peak):
        return Protection(False, False, gain, peak, 0.0,
                          "the position's gain is not measurable")

    threshold = arm_at(round_trip_bps)
    if peak < threshold:
        return Protection(
            False, False, gain, peak, 0.0,
            f"best {peak:+.2%} has not reached the {threshold:.2%} worth "
            f"protecting — below that the gain does not cover the round trip")

    noise = NOISE_SIGMAS * sigma if math.isfinite(sigma) and sigma > 0 else 0.0
    band = max(GIVE_BACK_FRACTION * peak, noise)
    trigger = peak - band

    if gain > trigger:
        return Protection(
            False, True, gain, peak, band,
            f"holding: {gain:+.2%} against a best of {peak:+.2%}, and the "
            f"{band:.2%} it may give back is not spent")

    return Protection(
        True, True, gain, peak, band,
        f"closing: gave back {peak - gain:.2%} of a {peak:+.2%} best, past "
        f"the {band:.2%} this position may return. Holding for the signal "
        f"alone would risk the rest of it.")
