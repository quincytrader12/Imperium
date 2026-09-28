"""Capital follows evidence.

Each strategy's positions are scaled by how strongly its own live record says
it earns. Not by how good it looked in a backtest, and not by how it did last
week: by a statistic built to be hard to fool with a short, lucky record.

THE STATISTIC
-------------

The probabilistic Sharpe ratio (Bailey and Lopez de Prado, 2012): the
probability that a strategy's true Sharpe ratio is above a benchmark, given
the one it measured, how long it measured it for, and how lopsided and
fat-tailed its returns were. A Sharpe of 0.3 a day over twenty days is not the
same evidence as the same figure over two hundred, and a record made of many
small gains and one large loss is not the same as a symmetric one. The plain
Sharpe ratio treats all of those alike.

The benchmark is *deflated* (Bailey and Lopez de Prado, 2014). With several
strategies running, the best of them will look good by chance alone -- the
maximum of five noisy estimates is biased upward even if every one of them is
worthless. So the bar a strategy has to clear is the Sharpe ratio the best of
N worthless strategies would be expected to show, not zero. With one strategy
the bar is zero; with five it is materially higher; and a strategy has to beat
it before it gets more capital.

Returns are each strategy's daily *contribution* to the fund: its change in
profit over a day, divided by the fund's equity at the start of that day. That
is what the fund actually experiences, and it is scale-free, so a deposit does
not read as a good day.

THE RULE
--------

* Below a minimum track record -- :data:`MIN_DAYS` marked days and
  :data:`MIN_ROUND_TRIPS` closed trades -- the multiplier is exactly 1. No
  statistic at all is computed on less; a short record's Sharpe ratio is
  mostly noise, and acting on noise is how an allocator ends up chasing luck.
* Beyond it, the probability maps to one of five bands, from
  :data:`FLOOR` to :data:`CAP`. Bands rather than a curve, so the terminal can
  say in one sentence why a strategy is where it is.
* A strategy whose *recent* record is strongly negative is cut to the floor
  even if its whole history is fine. That is the decay rule: a strategy that
  worked and stopped working has a good long record and a bad recent one, and
  a full-history statistic is the last thing to notice.
* The applied multiplier moves toward its target by at most :data:`STEP` a
  day. One good week cannot double a strategy overnight, and one bad day
  cannot gut it.
* **The floor is not zero.** A strategy cut to nothing can never produce the
  evidence that would restore it, so it keeps trading at the smallest size
  the fees allow and can earn its way back.

Everything here is applied before the portfolio clamp, so every hard limit --
the per-symbol cap, the gross ceiling, buying power, a halt -- still binds on
top. A multiplier above 1 is permission to use more of the room those limits
leave, never a way past them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Iterable

#: Marked days before any statistic is computed. A month of trading. Fewer
#: and the Sharpe estimate's own standard error is larger than any edge worth
#: allocating to.
MIN_DAYS = 20

#: Closed round trips before any statistic is computed. Days alone are not
#: enough: a strategy that held one position for a month has twenty marks and
#: one decision behind them.
MIN_ROUND_TRIPS = 8

#: The window the decay rule reads, in days, and the history needed before it
#: is consulted -- a recent window that is most of the record is not recent.
RECENT_DAYS = 20
DECAY_NEEDS = 2 * RECENT_DAYS

#: Bounds on the multiplier. Above 1 is more of the room the limits leave;
#: below 1 is less; neither end can breach a limit.
CAP = 1.5
FLOOR = 0.25

#: Largest change to the applied multiplier in one day.
STEP = 0.25

#: The bands, highest first: (lowest probability, multiplier, meaning).
#:
#: Asymmetric on purpose. Under the null -- a strategy with no edge -- the
#: probability is close to uniform, so a threshold *is* a false-positive rate:
#: a boost at 80% would hand one worthless strategy in five more capital. The
#: two mistakes do not cost the same. A false boost puts more money into
#: nothing. A false cut trims exposure to nothing, and a real edge cut by a bad
#: stretch keeps trading at the floor and recovers as its record lengthens. So
#: gaining capital takes 95% and the full step takes 99%; losing it takes a
#: tenth, and the floor a fortieth.
BANDS: tuple[tuple[float, float, str], ...] = (
    (0.99, CAP, "strong evidence it earns"),
    (0.95, 1.25, "evidence it earns"),
    (0.10, 1.0, "no evidence either way"),
    (0.025, 0.5, "evidence it loses"),
    (0.0, FLOOR, "strong evidence it loses"),
)

#: How likely the recent window must be to be losing before the decay rule
#: cuts a strategy regardless of its history. The same bar as the floor band,
#: so the rule cannot cut noise more often than the bands already would.
DECAY_BAR = 0.025

#: Strategies this does not size. Unattributed is not a strategy, and the
#: Sector Trend sleeve sizes itself against its own share in its own ledger.
EXCLUDED = frozenset({"unattributed", "sector"})

_EULER = 0.5772156649015329
_NORMAL = NormalDist()


# -- the statistics ---------------------------------------------------------


@dataclass(frozen=True)
class Moments:
    """What the probabilistic Sharpe ratio needs of a return series."""

    count: int
    sharpe: float
    skew: float
    #: Not excess: a normal distribution has 3.
    kurtosis: float


def moments(returns: list[float]) -> Moments | None:
    """Sharpe ratio, skewness and kurtosis of a return series.

    None for fewer than three returns, or no variation at all: a series that
    never moved has no Sharpe ratio, and a zero there would read as "earns
    nothing" when the truth is "has not been measured".
    """
    n = len(returns)
    if n < 3:
        return None
    mean = sum(returns) / n
    dev = [r - mean for r in returns]
    var = sum(d * d for d in dev) / (n - 1)
    if var <= 0 or not math.isfinite(var):
        return None
    sd = math.sqrt(var)
    # Population moments for the shape; the sample variance for the scale,
    # which is what the Sharpe estimate itself uses.
    m2 = sum(d * d for d in dev) / n
    m3 = sum(d ** 3 for d in dev) / n
    m4 = sum(d ** 4 for d in dev) / n
    skew = m3 / m2 ** 1.5 if m2 > 0 else 0.0
    kurt = m4 / m2 ** 2 if m2 > 0 else 3.0
    return Moments(count=n, sharpe=mean / sd, skew=skew, kurtosis=kurt)


def probabilistic_sharpe(m: Moments, benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), from Bailey and Lopez de Prado (2012).

    PSR = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    The denominator is the estimator's standard error, widened for negative
    skew and fat tails -- the two properties that make a good-looking record
    least trustworthy.
    """
    spread = 1.0 - m.skew * m.sharpe + (m.kurtosis - 1.0) / 4.0 * m.sharpe ** 2
    if spread <= 0 or m.count < 2:
        return 0.5
    z = (m.sharpe - benchmark) * math.sqrt(m.count - 1) / math.sqrt(spread)
    return _NORMAL.cdf(z)


def deflated_benchmark(trials: int, sharpe_variance: float) -> float:
    """The Sharpe ratio the best of ``trials`` worthless strategies would show.

    Bailey and Lopez de Prado (2014): the expected maximum of N draws, scaled
    by the spread of the estimates,

        SR* = sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))

    with g the Euler-Mascheroni constant. Zero for a single trial -- with one
    strategy there is no selection to correct for.
    """
    if trials <= 1 or sharpe_variance <= 0:
        return 0.0
    a = _NORMAL.inv_cdf(1.0 - 1.0 / trials)
    b = _NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return math.sqrt(sharpe_variance) * ((1.0 - _EULER) * a + _EULER * b)


def daily_returns(marks: list[tuple[str, float]],
                  equity: dict[str, float]) -> list[float]:
    """Each day's change in a strategy's profit, over the fund's equity.

    A day whose previous equity is unknown or not positive is skipped rather
    than guessed: dividing by a made-up denominator puts a made-up return in
    the statistic.
    """
    out: list[float] = []
    for (prev_day, prev), (_, value) in zip(marks, marks[1:]):
        base = equity.get(prev_day, 0.0)
        if base > 0:
            out.append((value - prev) / base)
    return out


# -- the allocator ----------------------------------------------------------


def _bounded(value: Any) -> float:
    try:
        value = float(value if value is not None else 1.0)
    except (TypeError, ValueError):
        return 1.0
    return min(CAP, max(FLOOR, value)) if math.isfinite(value) else 1.0


@dataclass
class Standing:
    """Where one strategy stands, and why."""

    strategy: str
    multiplier: float = 1.0
    target: float = 1.0
    probability: float | None = None
    recent_probability: float | None = None
    days: int = 0
    round_trips: int = 0
    reason: str = "no record yet"
    #: The market-regime factor applied to the evidence target, and why.
    regime_factor: float = 1.0
    regime_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "multiplier": round(self.multiplier, 4),
                "target": round(self.target, 4),
                "regime_factor": round(self.regime_factor, 4),
                "regime_reason": self.regime_reason,
                "probability": (None if self.probability is None
                                else round(self.probability, 4)),
                "recent_probability": (None if self.recent_probability is None
                                       else round(self.recent_probability, 4)),
                "days": self.days, "round_trips": self.round_trips,
                "reason": self.reason}


def band(probability: float) -> tuple[float, str]:
    for lowest, multiplier, meaning in BANDS:
        if probability >= lowest:
            return multiplier, meaning
    return FLOOR, BANDS[-1][2]


@dataclass
class CapitalWeights:
    """The applied multiplier per strategy, and the reasoning behind each."""

    standings: dict[str, Standing] = field(default_factory=dict)
    #: The last day the weights were revised. Revised at most once a day, so
    #: the daily step really is a daily step.
    revised_on: str = ""

    def multiplier(self, strategy: str) -> float:
        if not strategy or strategy in EXCLUDED:
            return 1.0
        found = self.standings.get(strategy)
        if found is None:
            return 1.0
        m = found.multiplier
        # Defensive: a corrupt state file must not be able to size a position
        # outside the bounds this module promises.
        if not math.isfinite(m):
            return 1.0
        return min(CAP, max(FLOOR, m))

    def note(self, strategy: str) -> str:
        found = self.standings.get(strategy)
        return found.reason if found else "no record yet"

    def revise(self, day: str, records: Iterable[Any],
               equity: dict[str, float],
               states: dict[str, Any] | None = None,
               caps: dict[str, tuple[float, str]] | None = None,
               ) -> list[tuple[str, float, float, str]]:
        """Recompute every standing and step the multipliers toward target.

        ``records`` are the attribution book's StrategyRecord objects;
        ``equity`` is the fund's equity at each marked day; ``states`` is the
        market's state by day, from market_regime.states_by_day, and without
        it no strategy is tilted by regime; ``caps`` is the research desk's
        ceiling per strategy, with its reason, which can hold a target down
        and never raise it. Returns
        (strategy, old multiplier, new multiplier, reason) for every one that
        moved, so the caller can say so -- once, when it happens.
        """
        if day and day == self.revised_on:
            return []
        # Imported here: market_regime is built on this module's statistics.
        from imperium.execution import market_regime

        now = market_regime.state_on(states, day) if states else None
        eligible: list[tuple[Any, list[float], Moments]] = []
        for rec in records:
            if rec.name in EXCLUDED:
                continue
            standing = self.standings.setdefault(rec.name, Standing(rec.name))
            returns = daily_returns(rec.daily, equity)
            # Days of *returns*, not marks: the first mark has nothing before
            # it, and the statistic's sample size is what the minimum guards.
            standing.days = len(returns)
            standing.round_trips = rec.round_trips
            m = moments(returns)
            if (standing.days < MIN_DAYS or standing.round_trips < MIN_ROUND_TRIPS
                    or m is None):
                standing.target = 1.0
                standing.regime_factor, standing.regime_reason = 1.0, ""
                standing.probability = standing.recent_probability = None
                standing.reason = (
                    f"gathering evidence: {min(standing.days, MIN_DAYS)} of "
                    f"{MIN_DAYS} days, {min(standing.round_trips, MIN_ROUND_TRIPS)}"
                    f" of {MIN_ROUND_TRIPS} closed trades")
                continue
            eligible.append((rec, returns, m))

        # The deflation bar is set by everything being compared at once. The
        # spread of the estimates is floored at the sampling variance a
        # worthless strategy's Sharpe would have over the shortest record, so
        # a handful of strategies that happen to agree cannot shrink the bar
        # to nothing.
        trials = len(eligible)
        sharpes = [m.sharpe for _, _, m in eligible]
        spread = 0.0
        if trials >= 2:
            mean = sum(sharpes) / trials
            spread = sum((s - mean) ** 2 for s in sharpes) / (trials - 1)
        if eligible:
            shortest = min(m.count for _, _, m in eligible)
            spread = max(spread, 1.0 / max(1, shortest - 1))
        bar = deflated_benchmark(trials, spread)

        for rec, returns, m in eligible:
            standing = self.standings[rec.name]
            p = probabilistic_sharpe(m, bar)
            standing.probability = p
            target, meaning = band(p)
            reason = (f"{meaning}: {p:.0%} probability its Sharpe beats "
                      f"{'the best of ' + str(trials) + ' by chance' if trials > 1 else 'zero'}"
                      f" over {m.count} days")
            standing.recent_probability = None
            decaying = False
            if len(returns) >= DECAY_NEEDS:
                recent = moments(returns[-RECENT_DAYS:])
                if recent is not None:
                    rp = probabilistic_sharpe(recent, 0.0)
                    standing.recent_probability = rp
                    if rp <= DECAY_BAR and target > FLOOR:
                        decaying = True
                        target = FLOOR
                        reason = (f"decaying: its last {RECENT_DAYS} days are "
                                  f"{1 - rp:.0%} likely to be losing, whatever "
                                  f"the longer record says")
            # The regime tilts the evidence target; it does not replace it,
            # and it never lifts a strategy the decay rule has cut.
            standing.regime_factor, standing.regime_reason = 1.0, ""
            if states and not decaying:
                tilt = market_regime.tilt(
                    market_regime.filed_returns(rec.daily, equity, states), now)
                standing.regime_reason = tilt.reason
                if tilt.factor != 1.0:
                    standing.regime_factor = tilt.factor
                    target = min(CAP, max(FLOOR, target * tilt.factor))
                    reason = f"{reason}; {tilt.reason}"
            standing.target = target
            standing.reason = reason

        # Last, so nothing above can lift a strategy past what the research
        # says its edge still supports -- whether or not its own record is
        # long enough to have been judged yet.
        for name, (cap, why) in (caps or {}).items():
            standing = self.standings.get(name)
            if standing is None or name in EXCLUDED:
                continue
            if standing.target > cap:
                standing.target = cap
                standing.reason = f"{standing.reason}; {why}"

        moved: list[tuple[str, float, float, str]] = []
        for standing in self.standings.values():
            old = standing.multiplier
            gap = standing.target - old
            new = old + max(-STEP, min(STEP, gap))
            new = min(CAP, max(FLOOR, new))
            if abs(new - old) > 1e-9:
                standing.multiplier = new
                moved.append((standing.strategy, old, new, standing.reason))
        self.revised_on = day
        return moved

    def rows(self) -> list[dict[str, Any]]:
        return [s.as_dict() for s in sorted(self.standings.values(),
                                            key=lambda s: s.strategy)]

    # -- persistence -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {"revised_on": self.revised_on,
                "standings": {k: v.as_dict() for k, v in self.standings.items()}}

    @classmethod
    def from_dict(cls, payload: Any) -> "CapitalWeights":
        out = cls()
        if not isinstance(payload, dict):
            return out
        out.revised_on = str(payload.get("revised_on") or "")
        for name, raw in (payload.get("standings") or {}).items():
            if not isinstance(raw, dict):
                continue
            try:
                multiplier = float(raw.get("multiplier", 1.0))
                target = float(raw.get("target", 1.0))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(multiplier) and math.isfinite(target)):
                continue
            out.standings[str(name)] = Standing(
                strategy=str(name),
                multiplier=min(CAP, max(FLOOR, multiplier)),
                target=min(CAP, max(FLOOR, target)),
                probability=raw.get("probability"),
                recent_probability=raw.get("recent_probability"),
                days=int(raw.get("days") or 0),
                round_trips=int(raw.get("round_trips") or 0),
                reason=str(raw.get("reason") or ""),
                regime_factor=_bounded(raw.get("regime_factor")),
                regime_reason=str(raw.get("regime_reason") or ""))
        return out
