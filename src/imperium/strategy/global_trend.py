"""Trend across asset classes: bonds, gold, commodities, the world, property.

Every other strategy in this terminal trades US equities or crypto, and all
but one of them is some form of momentum in them. They win together and they
lose together, and a sharp equity sell-off is the day they all lose at once.
This sleeve is the diversifier: the same idea -- own what has been going up --
applied to asset classes that do not move with US stocks, and that often move
against them on exactly the days that matter.

**The evidence.** Time-series momentum is documented across asset classes, not
only within equities: Moskowitz, Ooi and Pedersen (JFE 2012) found it in every
one of 58 futures markets, and the diversification across classes is where
most of its risk-adjusted return comes from. Faber (2007) showed the ETF form
of it -- hold an asset class while it is above its long-run trend, step aside
when it is not -- roughly halved the drawdown of a buy-and-hold mix for a
similar return. What an account this size can hold is seven liquid ETFs, one
per class, and this trades those.

THE UNIVERSE
------------

One ETF per asset class, chosen for liquidity and a long history:

  IEF  7-10 year Treasuries        TLT  20+ year Treasuries
  GLD  gold                        DBC  a broad commodity basket
  EFA  developed markets ex-US     EEM  emerging markets
  VNQ  US real estate

US equities are deliberately absent: the engine already trades them, and a
sleeve that owned SPY would be the engine's bet a second time.

THE RULE
--------

* **Signal.** The trend score is the mean sign of the 1, 3, 6 and 12-month
  returns, from -1 (falling on every horizon) to +1 (rising on every one). An
  asset is held while its score is positive -- rising on more horizons than it
  is falling -- and in proportion to it. Four horizons rather than one so no
  single lookback's luck decides.
* **Size.** Inverse volatility, then the whole sleeve scaled to a target
  volatility measured from the assets' own covariance, not assumed. Bonds are
  calm and get more weight; emerging markets are not and get less. Scaled on
  the real covariance because these assets are diversifying precisely because
  their correlations are low, and a sum of volatilities would overstate the
  risk and under-invest.
* **Bounds.** No leverage (gross at most 1), no asset above 40% of the sleeve.
  What the target does not use is held as cash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

UNIVERSE: tuple[str, ...] = ("IEF", "TLT", "GLD", "DBC", "EFA", "EEM", "VNQ")

#: Horizons, in trading days: a month, a quarter, half a year, a year.
LOOKBACKS: tuple[int, ...] = (21, 63, 126, 252)

#: Days of returns the volatility and covariance are measured over.
VOL_DAYS = 60

#: Annualised volatility the sleeve is scaled to.
TARGET_VOL = 0.10

MAX_WEIGHT = 0.40
MAX_GROSS = 1.0

#: History before an asset can be scored at all: the longest lookback plus one.
MIN_BARS = max(LOOKBACKS) + 1

ANNUALISE = math.sqrt(252.0)


def trend_score(closes: np.ndarray) -> float:
    """Mean sign of the return over each lookback, in [-1, 1]; NaN when the
    history is too short to measure the longest one."""
    closes = np.asarray(closes, dtype=float)
    if closes.size < MIN_BARS or closes[-1] <= 0:
        return float("nan")
    signs = []
    for days in LOOKBACKS:
        past = closes[-1 - days]
        if past <= 0:
            return float("nan")
        signs.append(np.sign(closes[-1] / past - 1.0))
    return float(np.mean(signs))


def _returns(closes: np.ndarray, days: int) -> np.ndarray:
    tail = np.asarray(closes, dtype=float)[-(days + 1):]
    return np.diff(np.log(tail))


@dataclass
class Targets:
    """What the sleeve should hold, and why, as fractions of the sleeve."""

    weights: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    #: Annualised volatility of the target book, from the measured covariance.
    volatility: float = 0.0
    gross: float = 0.0
    note: str = ""


def targets(closes: dict[str, np.ndarray], *, target_vol: float = TARGET_VOL,
            max_weight: float = MAX_WEIGHT, max_gross: float = MAX_GROSS
            ) -> Targets:
    """Today's target weights. Pure: history in, weights out."""
    out = Targets()
    rising: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    for symbol, series in closes.items():
        score = trend_score(series)
        if not math.isfinite(score):
            out.reasons[symbol] = f"only {len(series)} of {MIN_BARS} days of history"
            continue
        out.scores[symbol] = score
        rets = _returns(series, VOL_DAYS)
        sigma = float(np.std(rets, ddof=1)) if rets.size > 2 else 0.0
        if not (sigma > 0 and math.isfinite(sigma)):
            out.reasons[symbol] = "no measurable volatility"
            continue
        sigmas[symbol] = sigma
        if score > 0:
            rising[symbol] = score
        else:
            up = int(round((score + 1) / 2 * len(LOOKBACKS)))
            out.reasons[symbol] = (f"trend down: rising on {up} of "
                                   f"{len(LOOKBACKS)} horizons")
    if not rising:
        out.note = "no asset class is trending up; the sleeve is in cash"
        return out

    names = sorted(rising)
    raw = np.array([rising[s] / sigmas[s] for s in names])
    raw = raw / raw.sum()

    # Scale to the target on the measured covariance of the assets held.
    length = min(VOL_DAYS, min(len(closes[s]) - 1 for s in names))
    matrix = np.vstack([_returns(closes[s], length) for s in names])
    cov = np.atleast_2d(np.cov(matrix))
    daily = float(math.sqrt(max(0.0, raw @ cov @ raw)))
    scale = (target_vol / ANNUALISE) / daily if daily > 0 else 0.0
    weights = raw * scale

    # Bounds: each asset, then the whole sleeve. Capping an asset frees no
    # weight for the others -- redistributing it would concentrate the book
    # in whatever was left, which is the opposite of the point.
    weights = np.minimum(weights, max_weight)
    gross = float(weights.sum())
    if gross > max_gross:
        weights *= max_gross / gross
    out.weights = {s: float(w) for s, w in zip(names, weights) if w > 1e-6}
    out.gross = float(sum(out.weights.values()))
    out.volatility = float(math.sqrt(max(0.0, weights @ cov @ weights))) * ANNUALISE
    for s in names:
        up = int(round((rising[s] + 1) / 2 * len(LOOKBACKS)))
        out.reasons[s] = (f"trend up on {up} of {len(LOOKBACKS)} horizons; "
                          f"{out.weights.get(s, 0.0):.0%} of the sleeve at "
                          f"{sigmas[s] * ANNUALISE:.0%} volatility")
    return out
