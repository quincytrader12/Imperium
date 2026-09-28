"""The nightly research desk: is each strategy's edge still there?

The live allocator judges a strategy by what it earned, which is the right
test and a slow one: a strategy trades a handful of positions, so its own
record takes months to say anything, and by the time a full-history statistic
notices that an edge has gone, the money has already found out. The market
knows sooner. Every strategy here trades a premium estimated across the whole
universe -- trend, the crypto ranking, the overnight drift -- and the same
history that estimates it can be asked, every night, whether the premium of
the last few months still looks like the premium of the year before.

THE MEASUREMENT
---------------

One number per trading day: what the strategy's signal was worth *that day*
across every symbol that had one. For the trend and the ranking, the
cross-sectional slope of the next day's return on the score; for the drift,
the mean overnight return. This is the Fama-MacBeth construction, and it is
chosen over re-running the pooled regression on a shorter window for one
reason: symbols move together. Six hundred symbols on one day are not six
hundred independent observations, and a pooled t-statistic that treats them
as such is too large by a factor that is largest on exactly the days that
matter. A series of daily premia has one observation per day, and its
t-statistic means what it says.

THE VERDICTS
------------

The last :data:`RECENT_DAYS` against everything before them:

* **reversed** -- the recent premium is significantly *negative* on its own
  (t at or below -:data:`T_BAR`). The signal is currently pointing the wrong
  way. Its capital is capped at :data:`REVERSED_CAP`.
* **fading** -- the long-run premium was real (t at least :data:`T_BAR`), the
  recent one is significantly lower (:data:`DROP_BAR` one-sided), and it is no
  longer significant by itself. No more capital than x1.00 until it recovers:
  a fading edge is not one to be given more money on the strength of a record
  it is no longer making.
* **holding** -- neither.

A research verdict can hold a strategy's capital down; it can never raise it.
The live record is still what earns a strategy more, and the pooled estimate
is still what decides whether it trades at all. Evaluated nightly on a rolling
window, a strategy whose edge has not changed will read as fading on some
nights -- measured, 4.8% of them -- which costs it no more than a pause in any
boost. Reversed is far stricter and is the only verdict that cuts: it was
never once returned for an unchanged edge in 3,000 histories. An edge with a
long-run t of about 4 that disappears entirely is caught within its sixty-day
window in 65% of histories, and the rest as the window fills with it.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np

DAY_MS = 86_400_000

#: The window judged against the rest. About three months of trading.
RECENT_DAYS = 60
#: The history before it needed for a long-run premium worth comparing to.
MIN_EARLIER_DAYS = 60
#: Symbols on a day before that day's premium is measured at all.
MIN_NAMES = 5

T_BAR = 2.0
DROP_BAR = 0.95

FADING_CAP = 1.0
REVERSED_CAP = 0.5

HOLDING = "holding"
FADING = "fading"
REVERSED = "reversed"
UNMEASURED = "unmeasured"

_NORMAL = NormalDist()


# -- the daily premium ---------------------------------------------------------


def slopes_by_day(rows: Iterable[tuple[int, float, float]]) -> list[tuple[int, float]]:
    """Each day's cross-sectional slope of return on score.

    ``rows`` are (day, score, return) for every symbol on every day. A day
    with fewer than :data:`MIN_NAMES` symbols, or no spread in the scores, is
    left out rather than given a slope it cannot support.
    """
    by_day: dict[int, tuple[list[float], list[float]]] = {}
    for day, x, y in rows:
        if math.isfinite(x) and math.isfinite(y):
            xs, ys = by_day.setdefault(int(day), ([], []))
            xs.append(float(x))
            ys.append(float(y))
    out = []
    for day in sorted(by_day):
        xs, ys = by_day[day]
        if len(xs) < MIN_NAMES:
            continue
        x = np.asarray(xs) - np.mean(xs)
        denominator = float(np.sum(x * x))
        if denominator <= 1e-18:
            continue
        out.append((day, float(np.sum(x * (np.asarray(ys) - np.mean(ys)))
                               / denominator)))
    return out


def means_by_day(rows: Iterable[tuple[int, float]]) -> list[tuple[int, float]]:
    """Each day's mean return across the symbols that had one."""
    by_day: dict[int, list[float]] = {}
    for day, y in rows:
        if math.isfinite(y):
            by_day.setdefault(int(day), []).append(float(y))
    return [(day, statistics.fmean(v)) for day, v in sorted(by_day.items())
            if len(v) >= MIN_NAMES]


def dated(samples: dict[str, tuple[np.ndarray, ...]],
          days: dict[str, np.ndarray]) -> Iterable[tuple]:
    """Flatten per-symbol samples into (day, *values) rows.

    A symbol whose dates and samples do not line up is left out entirely: a
    sample filed under the wrong day is a premium measured against the wrong
    market, and there is no way to tell which one is off.
    """
    for symbol, arrays in samples.items():
        when = days.get(symbol)
        if when is None or any(len(a) != len(when) for a in arrays):
            continue
        yield from zip(when, *arrays)


# -- the verdict ---------------------------------------------------------------


@dataclass
class Finding:
    strategy: str
    verdict: str = UNMEASURED
    reason: str = "not yet researched"
    days: int = 0
    recent_bps: float | None = None
    recent_t: float | None = None
    earlier_bps: float | None = None
    earlier_t: float | None = None
    #: P(recent premium < earlier premium).
    dropped: float | None = None

    @property
    def cap(self) -> float | None:
        return {FADING: FADING_CAP, REVERSED: REVERSED_CAP}.get(self.verdict)

    def as_dict(self) -> dict[str, Any]:
        def r(v, n=3):
            return None if v is None else round(v, n)
        return {"strategy": self.strategy, "verdict": self.verdict,
                "reason": self.reason, "days": self.days, "cap": self.cap,
                "recent_bps": r(self.recent_bps), "recent_t": r(self.recent_t),
                "earlier_bps": r(self.earlier_bps),
                "earlier_t": r(self.earlier_t), "dropped": r(self.dropped, 4)}


def _mean_t(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("inf")
    t = mean / se if se > 0 and math.isfinite(se) else 0.0
    return mean, se, t


def judge(strategy: str, premia: list[tuple[int, float]], *,
          unit: str = "bp/day per unit of score") -> Finding:
    """Compare the recent premium with the one before it."""
    values = [p for _, p in premia]
    need = RECENT_DAYS + MIN_EARLIER_DAYS
    if len(values) < need:
        return Finding(strategy, UNMEASURED,
                       f"{len(values)} of {need} days of history measured",
                       days=len(values))
    recent, earlier = values[-RECENT_DAYS:], values[:-RECENT_DAYS]
    r_mean, r_se, r_t = _mean_t(recent)
    e_mean, e_se, e_t = _mean_t(earlier)
    spread = math.sqrt(r_se ** 2 + e_se ** 2)
    dropped = (_NORMAL.cdf((e_mean - r_mean) / spread)
               if spread > 0 and math.isfinite(spread) else 0.5)
    f = Finding(strategy, HOLDING, "", len(values), r_mean * 1e4, r_t,
                e_mean * 1e4, e_t, dropped)
    now = (f"{r_mean * 1e4:+.2f}{unit} over the last {RECENT_DAYS} days "
           f"(t {r_t:+.1f}) against {e_mean * 1e4:+.2f} over the "
           f"{len(earlier)} before (t {e_t:+.1f})")
    if r_t <= -T_BAR:
        f.verdict = REVERSED
        f.reason = f"edge reversed: {now}; capital capped at x{REVERSED_CAP:.2f}"
    elif e_t >= T_BAR and dropped >= DROP_BAR and r_t < T_BAR:
        f.verdict = FADING
        f.reason = (f"edge fading: {now}, {dropped:.0%} likely lower; no more "
                    f"than x{FADING_CAP:.2f} until it recovers")
    else:
        f.reason = f"edge holding: {now}"
    return f


# -- the desk ------------------------------------------------------------------


@dataclass
class ResearchDesk:
    """Last night's findings, kept across restarts so a change is said once."""

    findings: dict[str, Finding] = field(default_factory=dict)
    ran_on: str = ""

    def run(self, day: str, premia: dict[str, tuple[list[tuple[int, float]], str]]
            ) -> list[tuple[str, str, Finding]]:
        """Judge every strategy; return (strategy, old verdict, finding) for
        each whose verdict changed. Once a day."""
        if day and day == self.ran_on:
            return []
        changed = []
        for strategy, (series, unit) in premia.items():
            found = judge(strategy, series, unit=unit)
            old = self.findings.get(strategy)
            before = old.verdict if old else UNMEASURED
            self.findings[strategy] = found
            if found.verdict != before and found.verdict != UNMEASURED:
                changed.append((strategy, before, found))
        self.ran_on = day
        return changed

    def caps(self) -> dict[str, tuple[float, str]]:
        return {name: (f.cap, f.reason) for name, f in self.findings.items()
                if f.cap is not None}

    def rows(self) -> list[dict[str, Any]]:
        return [f.as_dict() for _, f in sorted(self.findings.items())]

    def as_dict(self) -> dict[str, Any]:
        return {"ran_on": self.ran_on,
                "findings": {k: v.as_dict() for k, v in self.findings.items()}}

    @classmethod
    def from_dict(cls, payload: Any) -> "ResearchDesk":
        out = cls()
        if not isinstance(payload, dict):
            return out
        out.ran_on = str(payload.get("ran_on") or "")
        for name, raw in (payload.get("findings") or {}).items():
            if not isinstance(raw, dict):
                continue
            verdict = raw.get("verdict")
            if verdict not in (HOLDING, FADING, REVERSED, UNMEASURED):
                continue
            out.findings[str(name)] = Finding(
                strategy=str(name), verdict=verdict,
                reason=str(raw.get("reason") or ""),
                days=int(raw.get("days") or 0),
                recent_bps=raw.get("recent_bps"), recent_t=raw.get("recent_t"),
                earlier_bps=raw.get("earlier_bps"),
                earlier_t=raw.get("earlier_t"), dropped=raw.get("dropped"))
        return out
