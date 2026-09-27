"""The book as one position.

Every sizing rule in this program looked at one symbol at a time. Each
position targeted its own volatility, each respected its own cap -- and
nothing ever asked what they added up to. Five positions that fall together
are one position five times the size, and a book of them can sit far above
the volatility it is "sized toward" while every individual check passes.
``RiskLimits.target_volatility`` said 20%; nothing enforced it.

This measures the book as a whole and enforces that limit.

WHAT IT MEASURES
----------------

* **Book volatility**, annualised, from a covariance matrix of the holdings'
  daily returns.
* **Effective number of bets**: the square of the diversification ratio, the
  sum of each position's own volatility over the book's. Five independent
  equal positions are five bets; five perfectly correlated ones are one.
* **Market beta**, against the S&P 500 ETF: how much of the book is just a bet
  on the market going up.
* **Risk contribution** per position and per strategy -- the share of the
  book's volatility each one carries, which sums to the whole.
* **Clusters**: pairs of holdings whose returns move together closely enough
  that holding both is mostly holding one twice.

HOW THE COVARIANCE IS ESTIMATED
-------------------------------

Shrunk toward a scaled identity (Ledoit and Wolf, 2004). A sample covariance
from a few months of returns on a handful of names overstates how extreme the
correlations are -- the estimation noise goes straight into the off-diagonal
terms -- and a limit enforced on noise refuses trades for correlations that
are not there. Shrinkage pulls the noise toward "uncorrelated, average
volatility" by exactly as much as the data cannot support, which the formula
estimates from the data itself.

WHAT IT ENFORCES
----------------

One thing, and only on the way in: a new or larger position may not take the
book's volatility above the target. The largest weight that keeps it there is
solved exactly -- book variance along one position's weight is a quadratic --
rather than searched for. It never blocks an exit or a reduction, never
increases a trade, and never acts on a symbol it has no history for: a
position this cannot measure is left to the per-position rules that always
applied to it, rather than refused for a risk nobody computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

DAY_MS = 86_400_000

#: Trading days of returns the covariance is estimated from. Six months:
#: recent enough to reflect the correlations the book is holding now, long
#: enough that the estimate is not one week's mood.
WINDOW = 120

#: Common days two series need before their covariance is used at all.
MIN_OVERLAP = 40

ANNUALISATION = 252

#: Correlation above which two holdings are reported as moving together.
CLUSTER_CORRELATION = 0.7

BENCHMARK = "SPY"


def returns_by_day(bars: Iterable[Any]) -> dict[int, float]:
    """Close-to-close log returns, keyed by the day each return ends on.

    Keyed by calendar day rather than by position so that an equity (five
    days a week) and a coin (seven) can be aligned on the days they share.
    """
    out: dict[int, float] = {}
    prev: tuple[int, float] | None = None
    for bar in bars:
        close = float(getattr(bar, "close", 0.0) or 0.0)
        if close <= 0 or not getattr(bar, "closed", True):
            continue
        day = int(getattr(bar, "open_time", 0)) // DAY_MS
        if prev is not None and day > prev[0]:
            out[day] = math.log(close / prev[1])
        prev = (day, close)
    return out


def align(series: dict[str, dict[int, float]], symbols: list[str],
          window: int = WINDOW) -> tuple[list[int], np.ndarray]:
    """The last ``window`` days every one of ``symbols`` has a return for."""
    if not symbols:
        return [], np.empty((0, 0))
    common = set(series[symbols[0]])
    for s in symbols[1:]:
        common &= set(series[s])
    days = sorted(common)[-window:]
    matrix = np.array([[series[s][d] for s in symbols] for d in days],
                      dtype=float).reshape(len(days), len(symbols))
    return days, matrix


def shrunk_covariance(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) covariance toward a scaled identity.

    Returns the covariance and the shrinkage intensity used, in [0, 1].

    With ``S`` the sample covariance, ``mu`` its average variance and ``F =
    mu I`` the target, the intensity is ``min(b2, d2) / d2`` where ``d2 =
    ||S - F||^2`` measures how far the sample is from the target and ``b2`` --
    the average squared distance of each day's outer product from ``S`` -- how
    much of that distance is noise.
    """
    n, p = returns.shape
    if n < 2 or p == 0:
        return np.zeros((p, p)), 1.0
    x = returns - returns.mean(axis=0)
    s = x.T @ x / n
    mu = float(np.trace(s)) / p
    target = mu * np.eye(p)
    d2 = float(np.sum((s - target) ** 2))
    if d2 <= 0:
        return s * n / (n - 1), 0.0
    b2 = 0.0
    for k in range(n):
        outer = np.outer(x[k], x[k])
        b2 += float(np.sum((outer - s) ** 2))
    b2 /= n * n
    intensity = min(b2, d2) / d2
    shrunk = intensity * target + (1.0 - intensity) * s
    # Rescaled to the unbiased sample variance, so a series' own volatility is
    # reported as the usual n - 1 estimate would report it.
    return shrunk * n / (n - 1), intensity


@dataclass
class Assessment:
    """The book, measured as a whole."""

    volatility: float = 0.0
    target: float = 0.0
    effective_bets: float = 0.0
    positions: int = 0
    measured: int = 0
    beta: float | None = None
    shrinkage: float = 0.0
    days: int = 0
    #: symbol -> share of book volatility, summing to 1 over what is measured.
    contribution: dict[str, float] = field(default_factory=dict)
    #: strategy -> share of book volatility.
    by_strategy: dict[str, float] = field(default_factory=dict)
    clusters: list[tuple[str, str, float]] = field(default_factory=list)
    unmeasured: list[str] = field(default_factory=list)
    note: str = "the book holds nothing"

    def as_dict(self) -> dict[str, Any]:
        return {
            "volatility": round(self.volatility, 4),
            "target": round(self.target, 4),
            "effective_bets": round(self.effective_bets, 2),
            "positions": self.positions,
            "measured": self.measured,
            "beta": None if self.beta is None else round(self.beta, 3),
            "shrinkage": round(self.shrinkage, 3),
            "days": self.days,
            "contribution": {k: round(v, 4) for k, v in self.contribution.items()},
            "by_strategy": {k: round(v, 4) for k, v in self.by_strategy.items()},
            "clusters": [[a, b, round(c, 3)] for a, b, c in self.clusters],
            "unmeasured": list(self.unmeasured),
            "note": self.note,
        }


class BookRisk:
    """The session's view of the book's risk, and the gate on new exposure."""

    def __init__(self, target_volatility: float = 0.20) -> None:
        self.target = float(target_volatility)
        #: Weights of what the book holds, as fractions of equity.
        self.holdings: dict[str, float] = {}
        #: Daily returns of every symbol that might be measured: the holdings,
        #: and whatever a candidate engine hands in.
        self.series: dict[str, dict[int, float]] = {}
        self.owners: dict[str, str] = {}
        self.market: dict[int, float] = {}
        self.assessment = Assessment(target=self.target)
        self._keys: dict[str, tuple[int, int]] = {}

    # -- keeping it current ----------------------------------------------

    def set_series(self, symbol: str, bars: Iterable[Any]) -> None:
        self.series[symbol] = returns_by_day(bars)

    def ensure_series(self, symbol: str, bars: list[Any] | None) -> None:
        """Measure a candidate's history, once per new bar.

        Called on every evaluation of every symbol, so it recomputes only when
        the daily series has actually changed -- its length or its last bar.
        """
        if not bars:
            return
        key = (len(bars), int(getattr(bars[-1], "open_time", 0)))
        if self._keys.get(symbol) == key:
            return
        self._keys[symbol] = key
        self.series[symbol] = returns_by_day(bars)

    def set_market(self, bars: Iterable[Any]) -> None:
        self.market = returns_by_day(bars)

    def update(self, holdings: dict[str, float],
               owners: dict[str, str] | None = None) -> Assessment:
        self.holdings = {s: float(w) for s, w in holdings.items()
                         if abs(float(w)) > 1e-9}
        self.owners = dict(owners or {})
        self.assessment = self.assess(self.holdings)
        return self.assessment

    # -- measuring -------------------------------------------------------

    def _measurable(self, symbols: Iterable[str]) -> tuple[list[str], list[str]]:
        have = [s for s in symbols if len(self.series.get(s, {})) >= MIN_OVERLAP]
        missing = [s for s in symbols if s not in have]
        return have, missing

    def _covariance(self, symbols: list[str]) -> tuple[np.ndarray, float, int] | None:
        if not symbols:
            return None
        days, matrix = align(self.series, symbols)
        if len(days) < MIN_OVERLAP:
            return None
        cov, intensity = shrunk_covariance(matrix)
        return cov * ANNUALISATION, intensity, len(days)

    def assess(self, holdings: dict[str, float]) -> Assessment:
        out = Assessment(target=self.target, positions=len(holdings))
        if not holdings:
            return out
        symbols, missing = self._measurable(sorted(holdings))
        out.unmeasured = missing
        found = self._covariance(symbols)
        if found is None:
            out.note = ("not enough shared daily history to measure the book "
                        "as a whole yet")
            out.unmeasured = sorted(holdings)
            return out
        cov, intensity, days = found
        w = np.array([holdings[s] for s in symbols])
        variance = float(w @ cov @ w)
        vol = math.sqrt(max(variance, 0.0))
        out.volatility, out.shrinkage, out.days = vol, intensity, days
        out.measured = len(symbols)

        sigmas = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if vol > 0:
            marginal = cov @ w
            shares = w * marginal / variance
            out.contribution = {s: float(v) for s, v in zip(symbols, shares)}
            ratio = float(np.abs(w) @ sigmas) / vol
            out.effective_bets = ratio * ratio
            for s, share in out.contribution.items():
                owner = self.owners.get(s, "unattributed")
                out.by_strategy[owner] = out.by_strategy.get(owner, 0.0) + share

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                denom = sigmas[i] * sigmas[j]
                if denom <= 0:
                    continue
                rho = float(cov[i, j] / denom)
                if rho >= CLUSTER_CORRELATION:
                    out.clusters.append((symbols[i], symbols[j], rho))
        out.clusters.sort(key=lambda c: -c[2])

        out.beta = self._beta(symbols, w)
        out.note = self._describe(out)
        return out

    def _beta(self, symbols: list[str], w: np.ndarray) -> float | None:
        if len(self.market) < MIN_OVERLAP:
            return None
        total, used = 0.0, 0.0
        for symbol, weight in zip(symbols, w):
            own = self.series.get(symbol, {})
            common = sorted(set(own) & set(self.market))[-WINDOW:]
            if len(common) < MIN_OVERLAP:
                continue
            r = np.array([own[d] for d in common])
            m = np.array([self.market[d] for d in common])
            var_m = float(np.var(m, ddof=1))
            if var_m <= 0:
                continue
            beta = float(np.cov(r, m, ddof=1)[0, 1] / var_m)
            total += weight * beta
            used += abs(weight)
        return total if used > 0 else None

    def _describe(self, a: Assessment) -> str:
        parts = [f"book volatility {a.volatility:.0%} of a {a.target:.0%} target",
                 f"{a.effective_bets:.1f} independent bet"
                 f"{'' if abs(a.effective_bets - 1) < 0.05 else 's'} across "
                 f"{a.measured} position{'' if a.measured == 1 else 's'}"]
        if a.beta is not None:
            parts.append(f"market beta {a.beta:.2f}")
        if a.clusters:
            first = a.clusters[0]
            parts.append(f"{first[0]} and {first[1]} move together "
                         f"(correlation {first[2]:.2f})")
        return "; ".join(parts)

    # -- the gate ----------------------------------------------------------

    def max_weight(self, symbol: str, desired: float, current: float) -> tuple[float, str]:
        """The largest weight in [current, desired] that keeps book vol on target.

        Long-only increases only. Everything else passes untouched: a
        reduction, an exit, a short, and any symbol without enough history to
        measure -- the per-position rules still bound those, as they always
        have, and refusing a trade for a risk that was never computed would be
        worse than not checking.
        """
        if desired <= current or desired <= 0 or self.target <= 0:
            return desired, ""
        others = {s: w for s, w in self.holdings.items() if s != symbol}
        symbols, _ = self._measurable(sorted(others))
        if len(self.series.get(symbol, {})) < MIN_OVERLAP:
            return desired, ""
        names = symbols + [symbol]
        found = self._covariance(names)
        if found is None:
            return desired, ""
        cov, _, _ = found
        w_others = np.array([others[s] for s in symbols])
        a = float(cov[-1, -1])
        b = float(cov[-1, :-1] @ w_others) if symbols else 0.0
        c = float(w_others @ cov[:-1, :-1] @ w_others) if symbols else 0.0
        limit = self.target ** 2

        def variance(x: float) -> float:
            return a * x * x + 2 * b * x + c

        if variance(desired) <= limit:
            return desired, ""
        if a <= 0:
            return desired, ""
        # The book is already over target without this symbol: it may keep
        # what it holds, and add nothing.
        if variance(current) >= limit:
            return current, (f"the book is already at {math.sqrt(max(c, 0)):.0%} "
                             f"volatility against a {self.target:.0%} target, so "
                             f"{symbol} is not increased")
        # a x^2 + 2 b x + (c - T^2) = 0, the larger root.
        disc = b * b - a * (c - limit)
        root = (-b + math.sqrt(max(disc, 0.0))) / a
        allowed = max(current, min(desired, root))
        return allowed, (f"sized to {allowed:.1%} from {desired:.1%} so the "
                         f"book stays at its {self.target:.0%} volatility "
                         f"target; {symbol} moves with what it already holds")
