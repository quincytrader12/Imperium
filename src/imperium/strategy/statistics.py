"""Estimators for the regime classifier.

Three tests on a rolling window. None of their thresholds are written down here:
every bound this module's callers use comes from
``imperium/strategy/null_calibration.json``, which is *generated* by
``imperium.strategy.calibration`` from simulated nulls. That indirection is the
point. Textbook values are wrong at these sample sizes -- the R/S Hurst
estimator in particular is nowhere near centred on 0.5 on 250 bars -- and a
threshold that fires on half of all pure noise is worse than no test.

Implemented with numpy only. scipy is a test-time dependency, not a runtime one,
because a one-file Windows build should not carry it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Below this many returns the estimators are too noisy to mean anything, and
#: reporting a regime from them is worse than reporting "warming up".
MIN_SAMPLES = 64


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns, with non-positive prices dropped rather than producing nan."""
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size < 2:
        return np.zeros(0)
    return np.diff(np.log(p))


def variance_ratio_horizon(n: int) -> int:
    """Derive the aggregation horizon q from the window length.

    ``clip(n // 12, 4, 64)`` rather than a fixed constant: q must grow with the
    window or the test is measuring a shorter and shorter fraction of it, and
    the variance of the estimator depends on the ratio q/n, not on q alone.
    """
    return int(np.clip(n // 12, 4, 64))


@dataclass(frozen=True)
class VarianceRatio:
    """Lo--MacKinlay (1988) variance ratio.

    ``vr > 1`` means positively autocorrelated increments -- trending.
    ``vr < 1`` means mean reversion. The z-statistic is the
    heteroskedasticity-robust one (their statistic z*), not the homoskedastic
    form: financial returns have volatility clustering, and the homoskedastic
    statistic rejects the random walk on volatility clustering alone, which is
    not the thing being tested for.
    """

    vr: float
    z: float
    q: int
    n: int
    valid: bool

    @property
    def trending(self) -> bool:
        return self.vr > 1.0


def variance_ratio(returns: np.ndarray, q: int | None = None) -> VarianceRatio:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if q is None:
        q = variance_ratio_horizon(n)
    if n < max(MIN_SAMPLES, 2 * q):
        return VarianceRatio(1.0, 0.0, q, n, valid=False)

    mu = float(r.mean())
    e = r - mu
    # Unbiased 1-period variance.
    var_1 = float(e @ e) / (n - 1)
    if var_1 <= 0:
        return VarianceRatio(1.0, 0.0, q, n, valid=False)

    # q-period overlapping sums, with the Lo--MacKinlay bias correction m.
    cum = np.concatenate(([0.0], np.cumsum(r)))
    q_sums = cum[q:] - cum[:-q]                 # length n-q+1
    m = q * (n - q + 1) * (1.0 - q / n)
    if m <= 0:
        return VarianceRatio(1.0, 0.0, q, n, valid=False)
    dev = q_sums - q * mu
    var_q = float(dev @ dev) / m

    vr = var_q / var_1

    # Heteroskedasticity-robust variance of the ratio (their theta).
    e2 = e * e
    denom = float(e2.sum()) ** 2
    if denom <= 0:
        return VarianceRatio(vr, 0.0, q, n, valid=False)
    theta = 0.0
    for j in range(1, q):
        delta_j = float(e2[j:] @ e2[:-j]) / denom
        weight = 2.0 * (q - j) / q
        theta += weight * weight * delta_j
    if theta <= 0:
        return VarianceRatio(vr, 0.0, q, n, valid=False)

    z = (vr - 1.0) / math.sqrt(theta)
    if not math.isfinite(z):
        return VarianceRatio(vr, 0.0, q, n, valid=False)
    return VarianceRatio(vr, z, q, n, valid=True)


def hurst_rs(returns: np.ndarray, min_chunk: int = 8) -> float:
    """Rescaled-range (R/S) Hurst exponent.

    **This estimator is not centred on 0.5 in small samples.** On 250-bar random
    walks its median sits well below 0.5 -- the measured value is recorded in
    ``null_calibration.json`` and is what the classifier compares against. Any
    threshold written as "below 0.45 means mean-reverting" fires on more than
    half of pure noise.

    Returned as a raw number here; :mod:`imperium.strategy.regime` converts it to
    a percentile against the measured null before using it.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < max(MIN_SAMPLES, min_chunk * 4):
        return float("nan")

    # Chunk sizes spaced geometrically, so the regression is not dominated by
    # the many small chunks.
    sizes: list[int] = []
    size = min_chunk
    while size <= n // 2:
        sizes.append(size)
        size = int(math.ceil(size * 1.6))
    if len(sizes) < 3:
        return float("nan")

    logs_n: list[float] = []
    logs_rs: list[float] = []
    for size in sizes:
        count = n // size
        if count < 1:
            continue
        chunks = r[: count * size].reshape(count, size)
        means = chunks.mean(axis=1, keepdims=True)
        dev = np.cumsum(chunks - means, axis=1)
        ranges = dev.max(axis=1) - dev.min(axis=1)
        stds = chunks.std(axis=1, ddof=1)
        good = stds > 0
        if not np.any(good):
            continue
        rs = float(np.mean(ranges[good] / stds[good]))
        if rs > 0:
            logs_n.append(math.log(size))
            logs_rs.append(math.log(rs))

    if len(logs_n) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.asarray(logs_n), np.asarray(logs_rs), 1)
    return float(slope)


@dataclass(frozen=True)
class ADFResult:
    """Augmented Dickey--Fuller test on the *price level*.

    Rejecting the unit root means the series is stationary, i.e. mean-reverting.
    The critical value used by the classifier is the measured one from the null
    calibration, not a MacKinnon table lookup, so that it matches this
    implementation and this window length exactly.
    """

    stat: float
    lags: int
    n: int
    valid: bool


def adf(series: np.ndarray, max_lags: int | None = None) -> ADFResult:
    """ADF with a constant, no trend, lag order by Schwert's rule."""
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < MIN_SAMPLES:
        return ADFResult(0.0, 0, n, valid=False)

    if max_lags is None:
        max_lags = int(np.floor(12 * (n / 100.0) ** 0.25))
    lags = int(min(max_lags, max(1, n // 10)))

    dy = np.diff(y)
    rows = n - lags - 1
    if rows <= lags + 3:
        return ADFResult(0.0, lags, n, valid=False)

    # dy_t = a + b*y_{t-1} + sum_i c_i * dy_{t-i} + e
    y_lag = y[lags:-1]
    target = dy[lags:]
    cols = [np.ones(rows), y_lag]
    for i in range(1, lags + 1):
        cols.append(dy[lags - i: -i] if i > 0 else dy[lags:])
    X = np.column_stack(cols)
    if X.shape[0] != target.shape[0]:
        return ADFResult(0.0, lags, n, valid=False)

    try:
        beta, residuals, rank, _ = np.linalg.lstsq(X, target, rcond=None)
    except np.linalg.LinAlgError:
        return ADFResult(0.0, lags, n, valid=False)
    if rank < X.shape[1]:
        return ADFResult(0.0, lags, n, valid=False)

    resid = target - X @ beta
    dof = X.shape[0] - X.shape[1]
    if dof <= 0:
        return ADFResult(0.0, lags, n, valid=False)
    s2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return ADFResult(0.0, lags, n, valid=False)
    se = math.sqrt(max(s2 * xtx_inv[1, 1], 1e-300))
    stat = float(beta[1] / se)
    if not math.isfinite(stat):
        return ADFResult(0.0, lags, n, valid=False)
    return ADFResult(stat, lags, n, valid=True)


# -- generators used for calibration and for tests ------------------------
#
# Kept beside the estimators deliberately: a calibration is only meaningful
# against a stated generator, and a test that asserts "power against mean
# reversion" needs the same OU process the calibration measured.

def random_walk(n: int, rng: np.random.Generator, sigma: float = 0.01,
                s0: float = 100.0) -> np.ndarray:
    """The null: a driftless geometric random walk."""
    return s0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))


def ou_process(n: int, rng: np.random.Generator, theta: float = 0.06,
               sigma: float = 0.01, s0: float = 100.0) -> np.ndarray:
    """Mean-reverting: an Ornstein--Uhlenbeck process in log price."""
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] - theta * x[t - 1] + rng.normal(0.0, sigma)
    return s0 * np.exp(x)


def trend_process(n: int, rng: np.random.Generator, drift: float = 0.0012,
                  sigma: float = 0.01, s0: float = 100.0) -> np.ndarray:
    """Trending: a random walk with drift."""
    return s0 * np.exp(np.cumsum(rng.normal(drift, sigma, n)))


def momentum_process(n: int, rng: np.random.Generator, phi: float = 0.18,
                     sigma: float = 0.01, s0: float = 100.0) -> np.ndarray:
    """Trending via positive serial correlation rather than drift.

    A drifting series and a positively autocorrelated one are both "trending",
    but only this one is trending in the sense the variance ratio measures. A
    classifier validated only on drift has not been validated.
    """
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0.0, sigma)
    return s0 * np.exp(np.cumsum(r))


def garch_process(n: int, rng: np.random.Generator, omega: float = 1e-6,
                  alpha: float = 0.08, beta: float = 0.90,
                  s0: float = 100.0) -> np.ndarray:
    """Held-out null: a martingale with volatility clustering.

    Real crypto returns are heteroskedastic. A classifier that only sees
    constant-variance nulls will have its false-positive rate measured on a
    world that does not exist.
    """
    r = np.zeros(n)
    var = omega / max(1e-12, (1 - alpha - beta))
    for t in range(n):
        e = rng.normal(0.0, math.sqrt(var))
        r[t] = e
        var = omega + alpha * e * e + beta * var
    return s0 * np.exp(np.cumsum(r))


def fat_tail_process(n: int, rng: np.random.Generator, df: int = 4,
                     sigma: float = 0.01, s0: float = 100.0) -> np.ndarray:
    """Held-out null: a martingale with t(4) innovations."""
    raw = rng.standard_t(df, n)
    raw = raw / math.sqrt(df / (df - 2))     # unit variance
    return s0 * np.exp(np.cumsum(raw * sigma))
