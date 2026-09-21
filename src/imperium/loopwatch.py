"""Measure how late the event loop is running.

Written to answer one question that kept being argued from theory: when the
terminal's link lamp goes red with close code 1006, did the *server* stall, or
did something below it drop the connection?

1006 means the socket died without a close frame from either side. On a
loopback connection there is no network between the two ends, so the list of
things that can do that is short -- and one of them is this process being too
busy or too blocked to answer. uvicorn sends a WebSocket ping on a timer and
drops the connection when the pong does not come back in time; an event loop
that is stuck cannot send the ping *or* read the pong, so a stall shows up as
a dropped socket rather than as anything that looks like a stall.

The measurement is the oldest one there is: ask to sleep for a known interval
and see how much longer it actually took. Nothing else on the loop can be
running while this coroutine is resumed, so the difference is time the loop
spent unable to get to it -- a blocking call inside an async function, a long
synchronous computation, or the whole process being descheduled by the
operating system.

Deliberately not a fix for anything. It is evidence, and it is cheap enough to
leave running for weeks: one wakeup a second and two floating point numbers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: How often to take the measurement. One second is short enough to catch a
#: stall while it is still the thing the operator is looking at, and long
#: enough that the watch itself is invisible in a CPU graph.
INTERVAL = 1.0

#: A lag worth writing down. Scheduling jitter on a loaded Windows machine is
#: routinely tens of milliseconds and occasionally a few hundred; a full five
#: seconds is not jitter, it is something holding the loop.
STALL_SECONDS = 5.0

#: Anything above this is reported as a stall that could itself have closed a
#: websocket, because it exceeds uvicorn's ping timeout.
PING_TIMEOUT_SECONDS = 20.0


@dataclass
class LoopWatch:
    """Running record of how far behind schedule the loop is.

    Keeps the worst case rather than an average on purpose: a mean over hours
    hides exactly the event being looked for, which is rare, brief and total.
    """

    interval: float = INTERVAL
    stall_after: float = STALL_SECONDS
    #: Most recent measurement, in seconds of lateness.
    last: float = 0.0
    #: Worst single measurement since the process started.
    worst: float = 0.0
    #: When that worst one happened, as a unix timestamp.
    worst_at: float = 0.0
    #: How many measurements exceeded :attr:`stall_after`.
    stalls: int = 0
    #: How many of those were long enough to have dropped a websocket.
    socket_killers: int = 0
    samples: int = 0
    _now: "callable" = field(default=time.time, repr=False)

    def observe(self, slept: float) -> float:
        """Record one sleep that asked for ``interval`` and took ``slept``.

        Returns the lateness, so the caller can decide whether to say anything
        about it. Negative lateness -- a sleep that returned early, which some
        platforms allow by a hair -- is recorded as zero rather than as a
        negative number that would make ``worst`` meaningless.
        """
        lag = slept - self.interval
        if lag < 0:
            lag = 0.0
        self.last = lag
        self.samples += 1
        if lag > self.worst:
            self.worst = lag
            self.worst_at = self._now()
        if lag >= self.stall_after:
            self.stalls += 1
            if lag >= PING_TIMEOUT_SECONDS:
                self.socket_killers += 1
        return lag

    def report(self) -> dict:
        """What the terminal shows and what a log line quotes."""
        return {
            "last_ms": round(self.last * 1000, 1),
            "worst_ms": round(self.worst * 1000, 1),
            "worst_at": self.worst_at or None,
            "stalls": self.stalls,
            "socket_killers": self.socket_killers,
            "samples": self.samples,
        }

    def verdict(self) -> str:
        """One sentence, in the operator's terms.

        This exists so the answer to "is the terminal stalling?" is on the
        screen rather than in a number that has to be interpreted.
        """
        if self.samples == 0:
            return "not measured yet"
        if self.socket_killers:
            return (f"the loop has stalled past {PING_TIMEOUT_SECONDS:.0f}s "
                    f"{self.socket_killers} time(s), long enough to drop the "
                    f"link by itself")
        if self.stalls:
            return (f"{self.stalls} stall(s) over {self.stall_after:.0f}s, "
                    f"worst {self.worst:.1f}s")
        return f"no stall; worst lateness {self.worst * 1000:.0f}ms"


async def watch(record: LoopWatch, *, on_stall=None) -> None:
    """Run the measurement until cancelled.

    ``on_stall`` is called with the lateness in seconds whenever one exceeds
    the threshold, so the stall can reach the activity log the operator is
    already reading and not only a file.
    """
    while True:
        started = time.perf_counter()
        await asyncio.sleep(record.interval)
        lag = record.observe(time.perf_counter() - started)
        if lag >= record.stall_after:
            log.warning("the event loop was %.1fs late; something held it", lag)
            if on_stall is not None:
                try:
                    on_stall(lag)
                except Exception:       # noqa: BLE001 - reporting must not kill the watch
                    log.exception("the stall handler failed")
