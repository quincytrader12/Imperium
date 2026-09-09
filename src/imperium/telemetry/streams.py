"""Two streams, deliberately not merged.

* An **event log**: a bounded ring of things a human should read. A key was
  rejected, the universe rotated, an order filled. Tens per hour.
* A **pulse stream**: every bar evaluated, every symbol scanned, every refusal
  to trade. Several per second.

Folding pulses into the event log evicts every readable event within a minute --
the high-frequency stream drowns the low-frequency signal, and the panel a human
actually reads becomes unreadable exactly when the bot gets busy. They are
separate rings with separate capacities.

**A pulse is a fact, never a prediction.** Nothing emits one for work it is
about to do. An orb on screen must correspond to a bar that was actually
evaluated, or a busy display is a decorative one.

Every pulse carries a monotonic sequence number. The client dedupes on it, so
overlapping snapshot windows spawn each orb exactly once and a dropped frame
costs nothing.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal

PulseKind = Literal["scan", "decision", "refused", "cap", "order", "warmup", "halt"]

PULSE_KINDS: frozenset[str] = frozenset(
    {"scan", "decision", "refused", "cap", "order", "warmup", "halt"}
)


class Level(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    GOOD = "good"


@dataclass(frozen=True)
class Event:
    """Something a human should read."""

    seq: int
    ts: float
    level: Level
    source: str
    message: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "level": self.level.value,
                "source": self.source, "message": self.message,
                "detail": self.detail}


@dataclass(frozen=True)
class Pulse:
    """One unit of work that actually happened."""

    seq: int
    ts: float
    symbol: str
    kind: str
    reason: str
    intensity: float

    def as_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": round(self.ts, 3), "symbol": self.symbol,
                "kind": self.kind, "reason": self.reason,
                "intensity": round(self.intensity, 3)}


class TelemetryHub:
    """Holds both rings. Thread-safe, because the engine loop and the websocket
    serialiser run on different tasks and, under uvicorn, different threads."""

    def __init__(self, event_capacity: int = 400, pulse_capacity: int = 4096) -> None:
        self._events: deque[Event] = deque(maxlen=event_capacity)
        self._pulses: deque[Pulse] = deque(maxlen=pulse_capacity)
        self._event_seq = itertools.count(1)
        self._pulse_seq = itertools.count(1)
        self._lock = threading.Lock()
        #: Lifetime count per pulse kind. Distinct from the ring, which is a
        #: bounded window: "how many bars have been evaluated since start" is a
        #: different question from "what happened recently", and answering the
        #: first from the ring would silently under-report once it wraps.
        self._kind_counts: dict[str, int] = {k: 0 for k in sorted(PULSE_KINDS)}

    # -- writing ---------------------------------------------------------

    def event(self, level: Level | str, source: str, message: str,
              detail: str = "") -> Event:
        ev = Event(next(self._event_seq), time.time(), Level(level), source,
                   message, detail)
        with self._lock:
            self._events.append(ev)
        return ev

    def pulse(self, symbol: str, kind: str, reason: str,
              intensity: float = 0.5) -> Pulse:
        """Record work that has already been done.

        ``kind`` is validated against the known set: a typo would otherwise
        produce orbs the front end cannot colour and silently drops.
        """
        if kind not in PULSE_KINDS:
            raise ValueError(
                f"unknown pulse kind {kind!r}; known kinds: {sorted(PULSE_KINDS)}"
            )
        p = Pulse(next(self._pulse_seq), time.time(), symbol, kind, reason,
                  float(max(0.0, min(1.0, intensity))))
        with self._lock:
            self._pulses.append(p)
            self._kind_counts[kind] = self._kind_counts.get(kind, 0) + 1
        return p

    # -- reading ---------------------------------------------------------

    def events(self, limit: int = 120, since: int = 0) -> list[dict[str, Any]]:
        """The most recent events, newest first.

        ``since`` returns only what the caller has not already seen. A stream
        that re-sends its whole window every second spends nearly all of its
        bandwidth on rows the client already has, and at 1Hz that is the
        difference between a terminal that keeps up and one that does not.
        """
        with self._lock:
            items = [e for e in self._events if e.seq > since][-limit:]
        return [e.as_dict() for e in reversed(items)]

    def pulse_window(self, limit: int = 240, since: int = 0) -> list[dict[str, Any]]:
        """The most recent pulses, oldest first.

        With ``since`` this is a delta. Without it -- a first frame, or a
        reconnect -- it is the whole window, because a client that has just
        connected needs the backlog to draw anything at all.
        """
        with self._lock:
            items = [p for p in self._pulses if p.seq > since][-limit:]
        return [p.as_dict() for p in items]

    @property
    def latest_event_seq(self) -> int:
        with self._lock:
            return self._events[-1].seq if self._events else 0

    @property
    def oldest_pulse_seq(self) -> int:
        """The oldest pulse still in the ring.

        A client whose cursor is older than this has fallen behind the ring and
        cannot be caught up by a delta -- it needs the window again. Saying so
        is the difference between a visible gap and a silent one.
        """
        with self._lock:
            return self._pulses[0].seq if self._pulses else 0

    @property
    def kind_counts(self) -> dict[str, int]:
        """Lifetime totals per pulse kind, unaffected by ring eviction."""
        with self._lock:
            return dict(self._kind_counts)

    @property
    def latest_pulse_seq(self) -> int:
        with self._lock:
            return self._pulses[-1].seq if self._pulses else 0

    @property
    def pulse_count(self) -> int:
        with self._lock:
            return len(self._pulses)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._pulses.clear()
            self._kind_counts = {k: 0 for k in sorted(PULSE_KINDS)}
