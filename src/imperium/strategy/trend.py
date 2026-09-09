"""Time-series momentum at the multi-day horizon.

This is the strategy a very small account can actually run, and the reason is
regulatory before it is statistical.

**Why a small account needs this one.** Under FINRA's pattern-day-trader rule a
margin account below $25,000 may make three day trades in five business days;
the fourth restricts it for ninety days. An intraday strategy is therefore not
merely constrained on a $70 account -- it is unusable, because a position it
cannot close the same day is not an intraday position at all. A trade held
*through* a session close is not a day trade under the rule, so a multi-day
horizon removes the binding constraint rather than working around it. Crypto is
outside the rule entirely.

(FINRA has since approved removing the $25,000 threshold, effective June 2026,
with firms given until October 2027 to implement it. Nothing here assumes
either state: the day-trade count and the pattern-day-trader flag are read from
the venue on every tick, so this program follows whatever the broker actually
enforces.)

**Why a longer horizon also survives its own costs.** A round trip is paid once
per holding period, so the cost per unit of time falls as the holding period
grows. With an expected drift of ``mu`` basis points a day and a round trip of
``C`` basis points, a position must be held

    H* = k * C / mu    days

before the edge has covered the cost, where ``k`` is the same safety multiple
every other strategy here is judged against. That is the whole design: this
module computes H*, refuses to open a position it cannot justify holding that
long, and refuses to close one before H* unless the signal itself has gone.
Crypto pays roughly 50bp a round trip, so at a 20bp/day drift it needs about
four days; a US equity pays 2-4bp and needs about one.

**The evidence.** Moskowitz, Ooi and Pedersen (JFE 2012) found positive
time-series momentum in *every one* of 58 futures contracts across equity
indices, currencies, commodities and bonds, with 52 of the 58 significant at
5%. Hurst, Ooi and Pedersen extended the result over a century of data. In
crypto, Liu and Tsyvinski (RFS 2021) found strong time-series momentum at one-
to four-week horizons: a one-standard-deviation increase in a week's return
predicts a 3.16% higher Bitcoin return the following week.

**The part of that evidence a small account does not get.** The headline Sharpe
of roughly 1.0 in Moskowitz et al. is a *diversified* portfolio of 58 markets.
Single-instrument time-series momentum is far weaker -- the diversification is
where most of the risk-adjusted return comes from, and an account carrying two
positions receives almost none of it. This module therefore estimates the trend
premium **pooled across the whole universe** rather than per symbol, for the
same reason the overnight module does: a single instrument cannot resolve an
effect this size from its own history, and any instrument that appears to is
being selected on noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from imperium.execution.bars import Bar
from imperium.venues.assets import AssetClass

#: Daily observations before a symbol's own trend estimate means anything.
MIN_DAYS = 40

#: Pooled observations and breadth before the premium is credible. The same
#: gate as the overnight estimate, for the same reason: a large t-statistic on
#: a handful of symbol-days is a large t-statistic on noise.
MIN_POOLED_OBSERVATIONS = 400
MIN_POOLED_SYMBOLS = 5

#: How far the estimated premium may be trusted, in basis points a day. A
#: pooled regression that reports more than this has found a data problem, not
#: an opportunity.
MAX_DAILY_DRIFT_BPS = 60.0

#: The shortest holding period this strategy will plan for, in days.
#:
#: Not a preference and not a rounding convenience. A position opened and
#: closed inside one session **is** a day trade, and avoiding that is the whole
#: reason this strategy exists for a small account. When the cost arithmetic
#: says a strong trend has covered its round trip in a quarter of a day, acting
#: on that would spend a day trade the account does not have -- so the position
#: is held across the close regardless, and the edge collected is a full day's
#: worth rather than a fraction.
MIN_HOLD_DAYS = 1.0

#: Longest holding period worth planning around. Beyond this the estimate that
#: justified the trade has decayed and the position is being held on faith.
MAX_HOLD_DAYS = 40


class TrendPhase(str, Enum):
    """Where a symbol is in the life of a trend position."""

    #: No position, and the signal does not justify one.
    FLAT = "flat"
    #: No position, and the signal does justify one.
    ENTER = "enter"
    #: Held, and inside the minimum holding period the cost implies.
    HOLDING = "holding"
    #: Held, past the minimum holding period, signal intact.
    MATURE = "mature"
    #: Held, and the reason for holding has gone.
    EXIT = "exit"


@dataclass(frozen=True)
class TrendSpec:
    """Lookbacks for one asset class, and why they differ.

    The horizons are not a free parameter -- they come from where each
    literature finds the effect. Moskowitz et al. measure equity-like
    instruments at one to twelve months; Liu and Tsyvinski find crypto momentum
    concentrated at one to four weeks, which is a different market with a
    different clientele and a much shorter memory.
    """

    lookbacks: tuple[int, ...]
    display_name: str
    source: str

    @property
    def required_days(self) -> int:
        return max(self.lookbacks) + 5


EQUITY_TREND = TrendSpec(
    lookbacks=(21, 63, 126),
    display_name="US equity",
    source="Moskowitz, Ooi & Pedersen (JFE 2012): one to twelve months",
)

CRYPTO_TREND = TrendSpec(
    lookbacks=(7, 14, 28),
    display_name="crypto",
    source="Liu & Tsyvinski (RFS 2021): one to four weeks",
)


def spec_for(asset_class: AssetClass) -> TrendSpec:
    return CRYPTO_TREND if asset_class is AssetClass.CRYPTO else EQUITY_TREND


# --------------------------------------------------------------- the signal

def daily_closes(bars: list[Bar]) -> np.ndarray:
    return np.array([b.close for b in bars if b.close > 0], dtype=float)


def momentum_score(closes: np.ndarray, lookback: int) -> float:
    """Volatility-normalised trailing return over ``lookback`` days.

    Normalised because a raw return compares a quiet instrument's small move
    with a violent one's large move as though they meant the same thing. It is
    the *number of standard deviations* the instrument has travelled that
    carries the signal, which is also what makes one pooled estimate applicable
    across a universe of very different symbols.
    """
    if closes.size < lookback + 2:
        return float("nan")
    window = closes[-(lookback + 1):]
    total = math.log(window[-1] / window[0])
    steps = np.diff(np.log(window))
    sd = float(np.std(steps, ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        return float("nan")
    # Scaled to a per-day figure so lookbacks of different lengths are
    # comparable and can be averaged.
    return float(total / (sd * math.sqrt(lookback)))


def blended_score(closes: np.ndarray, spec: TrendSpec) -> tuple[float, dict[int, float]]:
    """Average the lookbacks that have enough history to speak.

    Averaging rather than choosing: picking the best-performing horizon per
    symbol is how a backtest is fitted to its own sample. Every horizon in the
    spec carries equal weight, and one that lacks history is simply absent
    rather than silently zero -- a zero would drag the blend toward flat and
    read as "no trend" when it means "not measured".
    """
    scores: dict[int, float] = {}
    for lookback in spec.lookbacks:
        value = momentum_score(closes, lookback)
        if math.isfinite(value):
            scores[lookback] = value
    if not scores:
        return float("nan"), {}
    return float(np.mean(list(scores.values()))), scores


# ------------------------------------------------------- the pooled premium

@dataclass(frozen=True)
class PooledTrend:
    """How much next-day return one unit of trend score has been worth.

    Estimated across the whole universe at once. A single symbol's history
    cannot separate a few basis points a day from noise -- the same power
    problem the overnight module measured -- and the symbol that looks
    strongest on its own data is the one that got lucky.
    """

    beta_bps: float
    t_stat: float
    observations: int
    symbols: int
    #: Daily return standard deviation across the pool, in basis points. Used
    #: to express the premium as a drift for a given symbol.
    residual_bps: float = 0.0

    @property
    def credible(self) -> bool:
        return (self.observations >= MIN_POOLED_OBSERVATIONS
                and self.symbols >= MIN_POOLED_SYMBOLS
                and abs(self.t_stat) >= 2.0)

    def describe(self) -> str:
        if not self.observations:
            return "no trend premium measured yet"
        return (f"trend premium {self.beta_bps:+.2f}bp/day per unit of score "
                f"(t={self.t_stat:+.2f}) from {self.observations:,} symbol-days "
                f"across {self.symbols} symbols")


def pool(scored: dict[str, tuple[np.ndarray, np.ndarray]]) -> PooledTrend:
    """Regress next-day return on the trend score, pooled across symbols.

    ``scored`` maps a symbol to (scores, next-day returns) of equal length.
    A single pooled slope with a heteroskedasticity-robust standard error: the
    question is whether the *market* pays for trend, not whether any particular
    symbol did.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    symbols = 0
    for scores, forward in scored.values():
        mask = np.isfinite(scores) & np.isfinite(forward)
        if mask.sum() < 2:
            continue
        xs.append(scores[mask])
        ys.append(forward[mask])
        symbols += 1
    if not xs:
        return PooledTrend(0.0, 0.0, 0, 0)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    n = x.size
    if n < 2 or float(np.std(x)) <= 0:
        return PooledTrend(0.0, 0.0, n, symbols)

    x_centred = x - x.mean()
    denominator = float(np.sum(x_centred ** 2))
    if denominator <= 0:
        return PooledTrend(0.0, 0.0, n, symbols)
    beta = float(np.sum(x_centred * (y - y.mean())) / denominator)
    residuals = y - (y.mean() + beta * x_centred)
    # White's heteroskedasticity-robust standard error. Daily returns are not
    # homoskedastic by any stretch, and the classical error would overstate the
    # significance of exactly the periods that matter.
    variance = float(np.sum((x_centred ** 2) * (residuals ** 2)) / denominator ** 2)
    se = math.sqrt(variance) if variance > 0 else float("nan")
    t_stat = beta / se if math.isfinite(se) and se > 0 else 0.0

    return PooledTrend(
        beta_bps=beta * 10_000.0,
        t_stat=float(t_stat),
        observations=int(n),
        symbols=symbols,
        residual_bps=float(np.std(residuals)) * 10_000.0,
    )


# ------------------------------------------------------- holding-period math

def minimum_holding_days(round_trip_bps: float, drift_bps_per_day: float,
                         safety_multiple: float) -> float:
    """How long a position must be held before its edge has paid for its costs.

    The entire argument for this strategy on a small account, in one line. A
    round trip is paid once per holding period, so the cost per day falls as
    the holding period grows: ``H* = k * C / mu``. Infinite when the drift is
    not positive, which is the correct answer -- no holding period redeems a
    trade with no edge.
    """
    if drift_bps_per_day <= 0 or round_trip_bps <= 0:
        return float("inf")
    return safety_multiple * round_trip_bps / drift_bps_per_day


@dataclass(frozen=True)
class TrendSignal:
    """One symbol's trend view, and what acting on it would require."""

    value: float = 0.0
    score: float = 0.0
    scores: dict[int, float] = field(default_factory=dict)
    expected_edge_bps: float = 0.0
    drift_bps_per_day: float = 0.0
    min_hold_days: float = float("inf")
    daily_vol_bps: float = 0.0
    days: int = 0
    eligible: bool = False
    reason: str = ""


def evaluate(daily_bars: list[Bar], *, asset_class: AssetClass,
             pooled: PooledTrend | None = None,
             round_trip_bps: float = 0.0,
             safety_multiple: float = 1.5,
             planned_hold_days: int = MAX_HOLD_DAYS) -> TrendSignal:
    """Decide whether this symbol's trend is worth carrying, and for how long.

    The edge is the pooled premium applied to this symbol's own score and its
    own volatility -- not this symbol's own historical performance, which it
    does not have enough history to measure.
    """
    spec = spec_for(asset_class)
    closes = daily_closes(daily_bars)
    days = int(closes.size)

    if days < MIN_DAYS:
        return TrendSignal(days=days, reason=(
            f"warming up: {days} of {MIN_DAYS} daily bars — this resolves "
            f"itself, it is not a refusal to trade"))

    score, scores = blended_score(closes, spec)
    if not math.isfinite(score):
        return TrendSignal(days=days, reason=(
            f"not enough history for any {spec.display_name} lookback "
            f"({', '.join(str(x) for x in spec.lookbacks)} days)"))

    steps = np.diff(np.log(closes))
    daily_vol = float(np.std(steps[-max(spec.lookbacks):], ddof=1))
    daily_vol_bps = daily_vol * 10_000.0

    if pooled is None or not pooled.credible:
        return TrendSignal(
            score=score, scores=scores, days=days, daily_vol_bps=daily_vol_bps,
            reason=("the market-wide trend premium is not measurable yet, and a "
                    "single symbol's own history cannot resolve it — nothing is "
                    "traded on an unmeasured edge"))

    # Long only. A negative score means the trend is down; on a venue where
    # equities need a locate and crypto cannot be shorted at all, that is a
    # reason to hold nothing rather than a reason to sell short.
    if score <= 0 or pooled.beta_bps <= 0:
        return TrendSignal(
            score=score, scores=scores, days=days, daily_vol_bps=daily_vol_bps,
            reason=("the trend is flat or down and this book is long-only here, "
                    "so there is nothing to carry"))

    # The premium is per unit of score; the score is already in units of the
    # symbol's own daily standard deviations, so no further scaling by vol is
    # applied -- doing that would count the symbol's volatility twice.
    drift = min(MAX_DAILY_DRIFT_BPS, pooled.beta_bps * score)
    # Floored at one session. A holding period shorter than a day is a day
    # trade, which is the one thing this strategy exists not to be.
    hold = max(MIN_HOLD_DAYS,
               minimum_holding_days(round_trip_bps, drift, safety_multiple))

    if hold > planned_hold_days:
        return TrendSignal(
            score=score, scores=scores, days=days, daily_vol_bps=daily_vol_bps,
            drift_bps_per_day=drift, min_hold_days=hold,
            reason=(f"a {drift:.2f}bp/day drift needs {hold:.0f} days to cover "
                    f"{round_trip_bps:.2f}bp of round trip, past the "
                    f"{planned_hold_days}-day horizon this estimate is good "
                    f"for — the trend is real and too slow to pay for itself"))

    # Conviction saturates: a score of two standard deviations is a strong
    # trend, and anything beyond it is not proportionally more reliable.
    value = float(np.clip(score / 2.0, 0.0, 1.0))
    edge = drift * hold

    return TrendSignal(
        value=value, score=score, scores=scores,
        expected_edge_bps=edge, drift_bps_per_day=drift,
        min_hold_days=hold, daily_vol_bps=daily_vol_bps, days=days,
        eligible=True,
        reason=(f"{spec.display_name} trend {score:+.2f}σ over "
                f"{'/'.join(str(x) for x in sorted(scores))} days at "
                f"{pooled.beta_bps:+.2f}bp per unit gives {drift:.2f}bp/day; "
                f"held {hold:.0f} days that is {edge:.1f}bp against "
                f"{round_trip_bps:.2f}bp of cost"))


def phase_for(*, held: bool, days_held: float, min_hold_days: float,
              signal: TrendSignal) -> TrendPhase:
    """Where this symbol sits in the life of a trend position.

    The asymmetry is deliberate. Entering requires the edge to justify the
    whole round trip; *staying* requires only that the reason to be there still
    holds, because the entry cost is already spent. Exiting early throws away
    the cost without collecting the edge it bought.
    """
    if not held:
        return TrendPhase.ENTER if signal.eligible else TrendPhase.FLAT
    if signal.score <= 0:
        return TrendPhase.EXIT
    if days_held < min_hold_days:
        return TrendPhase.HOLDING
    return TrendPhase.MATURE
