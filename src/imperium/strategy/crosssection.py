"""Cross-sectional momentum, for the crypto book.

The time-series strategy in :mod:`imperium.strategy.trend` asks one question of
each symbol: *is this thing trending up?* That is the right question for a
market of eleven thousand listings where the pooled premium is estimated across
the whole tape. It is the wrong question for a venue that lists thirty-nine
coins, for two reasons.

The first is statistical. Time-series momentum needs the *market-wide* premium
to be credible before any single symbol may be traded on it, and that estimate
is built from symbol-days. Thirty-nine coins reach four hundred symbol-days in
under a fortnight, but the estimate they produce is dominated by whatever the
crypto market as a whole did in that fortnight -- it is close to a single
observation of one market, dressed as four hundred.

The second is economic, and it is the one the literature actually addresses.
Liu, Tsyvinski and Wu (*Common Risk Factors in Cryptocurrency*, Journal of
Finance 77(2), 2022, pp. 1133-1177) find that three factors -- market, size and
momentum -- capture the cross-section of expected cryptocurrency returns, and
that ten characteristics form long-short strategies with sizable, significant
excess returns which the three-factor model then explains. The momentum result
there is *cross-sectional*: it is about which coins outperform which, not about
whether the asset class is going up. Ranking thirty-nine coins against each
other estimates that directly, and a rank is well-conditioned in a way a level
is not -- it is unaffected by the market-wide move that contaminates the
time-series estimate.

**What this strategy does not claim.** It is long-only, because the venue does
not lend coins to short. A long-only slice of a long-short factor is the factor
*plus* the market: buying the strongest coins in a falling market still loses
money. Two things follow, and both are implemented rather than noted:

* the coin's own trend must also be positive, so the cross-sectional rank
  decides *which* coin and the time-series signal decides *whether to be in at
  all*; and
* the book stands down in a panic state.

**The panic state** is not a hunch. Daniel and Moskowitz (*Momentum Crashes*,
Journal of Financial Economics 122(2), 2016, pp. 221-247) show that momentum
strategies suffer infrequent, persistent strings of losses which are *partly
forecastable*: they occur following market declines and when market volatility
is high, and coincide with market rebounds. Grobys, Kolari, Sandretto, Shahzad
and Äijö (*Cryptocurrency momentum has (not) its moments*, Financial Markets
and Portfolio Management, 2025) find the same in crypto specifically, and
worse: power-law tests indicate the variance of the tail of crypto momentum
returns is *undefined*, the effect is concentrated in large caps, and
volatility management mitigates the crashes without changing the tail risk.

That last finding is why sizing here is capped rather than merely
volatility-scaled. A position sized from a two-sigma excursion assumes a second
moment exists to be estimated. If it does not, the sizing rule is not
conservative -- it is meaningless -- so there is a hard ceiling underneath it
that does not depend on the estimate at all.

**What is estimated here rather than assumed.** No figure from any of those
papers is hard-coded as an expected return. The literature chooses the *shape*:
which horizon to look at (one to four weeks), that the signal is a rank rather
than a level, that a panic state exists and must be avoided, and that the tail
is not to be trusted. The *magnitude* -- how many basis points a day a unit of
cross-sectional rank is worth -- is measured from the account's own history,
pooled across coins with a heteroskedasticity-robust standard error, and is
refused until it is statistically credible. A published premium is evidence
that the effect existed in someone else's sample. It is not a licence to trade
this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Lookbacks for the cross-sectional rank, in days.
#:
#: One to four weeks, following Liu and Tsyvinski's finding that crypto
#: momentum lives at that horizon rather than the one-to-twelve-month window
#: that works in equities. Blended across the three so a single week's noise
#: cannot decide the ranking on its own.
LOOKBACKS: tuple[int, ...] = (7, 14, 28)

#: The cross-section has to be wide enough to rank within.
#:
#: Ranking five coins is not a cross-section, it is a short list: the top
#: "decile" is one coin, and its score says more about which coins the venue
#: happens to list than about momentum. Alpaca lists enough to clear this
#: comfortably; the gate exists for the day it does not.
MIN_CROSS_SECTION = 10

#: Pooled observations and breadth before the premium may be traded. The same
#: gate as the other two strategies, for the same reason: a large t-statistic
#: on a handful of symbol-days is a large t-statistic on noise.
MIN_POOLED_OBSERVATIONS = 400
MIN_POOLED_SYMBOLS = 5
MIN_ABS_T_STAT = 2.0

#: The most a day's drift may be believed, in basis points.
#:
#: A pooled regression reporting more than this on daily crypto data has found
#: a data problem -- a split, a re-listing, a stale quote -- not an
#: opportunity. Deliberately higher than the equity ceiling because crypto
#: genuinely does move more, and still a ceiling.
MAX_DAILY_DRIFT_BPS = 120.0

#: A rank this far from the middle is as much conviction as is available.
#:
#: Conviction saturates: the top-ranked coin is not twice as reliable as the
#: second. Clipping at one standard deviation of the cross-section keeps the
#: single strongest name from taking the whole book.
SCORE_SATURATION = 1.0

#: The hard ceiling on position weight, independent of any volatility estimate.
#:
#: Grobys et al. find the tail variance of crypto momentum returns is undefined
#: under power-law tests. A two-sigma sizing rule assumes a second moment
#: exists; if it does not, that rule is not conservative, it is meaningless.
#: This ceiling does not depend on the estimate and is therefore the only part
#: of the sizing that still means something if the tail is as fat as they say.
MAX_CRYPTO_WEIGHT = 0.25

#: The least cross-sectional dispersion that counts as a ranking at all.
#:
#: Not a tidiness check -- a correctness one. Dividing by the cross-sectional
#: standard deviation amplifies whatever is in the numerator, and when the
#: coins have all moved by nearly the same amount what is left in the numerator
#: is floating-point rounding error. Twelve coins each up exactly 40% produce a
#: standard deviation of 5.5e-17 rather than zero, so a bare ``> 0`` guard
#: passes and every coin scores a full -1.0: maximum conviction, manufactured
#: from the last bit of a double.
#:
#: That is not a contrived case. A market-wide move with little dispersion is
#: the *common* case in crypto, which is the one condition under which this
#: strategy has nothing to say and must say so.
#:
#: One basis point of dispersion in a one-to-four-week return. Real crypto
#: cross-sections run percent-scale, so this only ever fires on a market that
#: has genuinely moved as one.
MIN_CROSS_SECTION_SPREAD = 1e-4

#: Trailing window for the market state, in days.
MARKET_WINDOW = 28

#: Volatility ratio above which the market counts as agitated.
#:
#: Daniel and Moskowitz's panic state is a market that has fallen *and* is
#: volatile. Recent realised volatility against its own longer run is the
#: cheapest honest measure of the second half.
PANIC_VOL_RATIO = 1.5


def daily_returns(closes: np.ndarray) -> np.ndarray:
    """Log returns from a close series, oldest first."""
    if closes.size < 2:
        return np.array([])
    return np.diff(np.log(closes))


def trailing_return(closes: np.ndarray, days: int) -> float:
    """Log return over the last ``days`` closes. NaN when there is not enough."""
    if closes.size < days + 1:
        return float("nan")
    return float(np.log(closes[-1] / closes[-1 - days]))


def blended_return(closes: np.ndarray,
                   lookbacks: tuple[int, ...] = LOOKBACKS) -> float:
    """Average trailing return across the lookbacks that are available.

    Averaged rather than taken at the longest available window, so a coin with
    a short history is not silently compared against a different horizon from
    its peers -- which would rank listing date rather than momentum.
    """
    values = [trailing_return(closes, d) for d in lookbacks]
    usable = [v for v in values if math.isfinite(v)]
    if not usable:
        return float("nan")
    return float(np.mean(usable))


def cross_sectional_scores(raw: dict[str, float]) -> dict[str, float]:
    """Turn each coin's trailing return into a score against its peers.

    The score is a z-score of the cross-section, not of the coin's own history.
    That is the whole point: subtracting the cross-sectional mean removes the
    market-wide move, which is the component a long-only book gets from simply
    holding the asset class and does not need a strategy to find.

    Returns an empty mapping when the cross-section is too narrow to rank
    within, or when every coin moved identically and there is nothing to rank.
    """
    usable = {s: v for s, v in raw.items() if math.isfinite(v)}
    if len(usable) < MIN_CROSS_SECTION:
        return {}
    values = np.array(list(usable.values()), dtype=float)
    spread = float(np.std(values))
    if spread < MIN_CROSS_SECTION_SPREAD:
        # Everything moved together, so nothing outperformed. Returning a
        # ranking here would divide rounding error by rounding error and hand
        # back full conviction. See MIN_CROSS_SECTION_SPREAD.
        return {}
    mean = float(np.mean(values))
    return {s: (v - mean) / spread for s, v in usable.items()}


def observations(
        closes: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pair each day's ranking with the return that *follows* it.

    Strictly forward, and that is the entire job of this function. The score at
    day ``t`` is built from closes up to and including ``t``, so it already
    contains the return that ended at ``t``. Pairing it with that same return
    rather than the next one regresses a number on itself: the fit is close to
    perfect, the t-statistic enormous, and the strategy unimplementable because
    the signal is not known until after the move it claims to predict.

    A lookahead is the one error here that makes everything look better. Every
    other bug in this file shows up as a strategy that will not trade.
    """
    symbols = sorted(closes)
    if len(symbols) < MIN_CROSS_SECTION:
        return {}
    length = min(int(closes[s].size) for s in symbols)
    longest = max(LOOKBACKS)
    if length < longest + 2:
        return {}

    scores: dict[str, list[float]] = {s: [] for s in symbols}
    forwards: dict[str, list[float]] = {s: [] for s in symbols}
    # ``end`` is the last day the score may see. The paired return runs from
    # ``end`` to ``end + 1``, so the loop stops one short of the series.
    for end in range(longest, length - 1):
        raw = {s: blended_return(closes[s][:end + 1]) for s in symbols}
        ranked = cross_sectional_scores(raw)
        if not ranked:
            continue
        for symbol, score in ranked.items():
            after = closes[symbol][end + 1]
            before = closes[symbol][end]
            if not (after > 0 and before > 0):
                continue
            forward = float(np.log(after / before))
            if not math.isfinite(forward):
                continue
            scores[symbol].append(score)
            forwards[symbol].append(forward)

    return {s: (np.array(scores[s]), np.array(forwards[s]))
            for s in symbols if scores[s]}


@dataclass(frozen=True)
class PooledCrossSection:
    """What a unit of cross-sectional rank has been worth, per day."""

    beta_bps: float = 0.0
    t_stat: float = 0.0
    observations: int = 0
    symbols: int = 0
    residual_bps: float = 0.0

    @property
    def credible(self) -> bool:
        return (self.observations >= MIN_POOLED_OBSERVATIONS
                and self.symbols >= MIN_POOLED_SYMBOLS
                and abs(self.t_stat) >= MIN_ABS_T_STAT)

    def describe(self) -> str:
        if not self.observations:
            return "no cross-sectional premium measured yet"
        return (f"cross-sectional premium {self.beta_bps:+.2f}bp/day per unit "
                f"of rank (t={self.t_stat:+.2f}) from {self.observations:,} "
                f"coin-days across {self.symbols} coins")

    def explain(self) -> str:
        """Why the premium is or is not being traded, in a sentence."""
        if not self.observations:
            return ("No cross-sectional history has been measured yet, so no "
                    "coin is ranked against its peers.")
        if self.observations < MIN_POOLED_OBSERVATIONS:
            return (f"Only {self.observations:,} coin-days measured of the "
                    f"{MIN_POOLED_OBSERVATIONS:,} needed — a premium on this "
                    f"little history is noise with a decimal point.")
        if self.symbols < MIN_POOLED_SYMBOLS:
            return (f"Only {self.symbols} of {MIN_POOLED_SYMBOLS} coins "
                    f"contributed, so this measures a few coins rather than "
                    f"the cross-section.")
        if abs(self.t_stat) < MIN_ABS_T_STAT:
            return (f"The measured premium is {self.beta_bps:+.2f} basis "
                    f"points a day per unit of rank, but t={self.t_stat:+.2f} "
                    f"and {MIN_ABS_T_STAT:.1f} is the bar — it cannot be told "
                    f"from zero, so nothing is traded on it.")
        return (f"Ranking coins against each other is worth "
                f"{self.beta_bps:+.2f} basis points a day per unit of rank "
                f"(t={self.t_stat:+.2f}), measured across {self.symbols} coins "
                f"over {self.observations:,} coin-days.")


def pool(scored: dict[str, tuple[np.ndarray, np.ndarray]]) -> PooledCrossSection:
    """Regress next-day return on the cross-sectional score, pooled.

    ``scored`` maps a coin to (scores, next-day returns) of equal length. One
    pooled slope with a heteroskedasticity-robust standard error, because the
    question is whether *the ranking* pays -- not whether any one coin did.
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
        return PooledCrossSection()

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    n = int(x.size)
    if n < 2 or float(np.std(x)) <= 0:
        return PooledCrossSection(observations=n, symbols=symbols)

    x_centred = x - x.mean()
    denominator = float(np.sum(x_centred ** 2))
    if denominator <= 0:
        return PooledCrossSection(observations=n, symbols=symbols)
    beta = float(np.sum(x_centred * (y - y.mean())) / denominator)
    residuals = y - (y.mean() + beta * x_centred)
    # White's heteroskedasticity-robust standard error. Crypto daily returns
    # are wildly heteroskedastic, and the classical error would overstate
    # significance in exactly the volatile stretches that decide the outcome.
    variance = float(np.sum((x_centred ** 2) * (residuals ** 2)) / denominator ** 2)
    se = math.sqrt(variance) if variance > 0 else float("nan")
    t_stat = beta / se if math.isfinite(se) and se > 0 else 0.0

    return PooledCrossSection(
        beta_bps=beta * 10_000.0,
        t_stat=float(t_stat),
        observations=n,
        symbols=symbols,
        residual_bps=float(np.std(residuals)) * 10_000.0,
    )


@dataclass(frozen=True)
class MarketState:
    """The crypto market's own condition, which gates every coin at once."""

    trailing_return: float = 0.0
    recent_vol: float = 0.0
    baseline_vol: float = 0.0
    panic: bool = False
    reason: str = ""


def market_state(market_closes: np.ndarray) -> MarketState:
    """Is the market in the state where momentum crashes?

    Daniel and Moskowitz's panic state, applied to crypto: the market has
    fallen *and* is more volatile than its own recent past. Both halves are
    required. A falling calm market is an ordinary drawdown that momentum
    survives; a volatile rising market is a bull run. It is the combination
    that precedes the rebound in which past losers violently outperform and a
    long-momentum book takes its worst losses.
    """
    if market_closes.size < MARKET_WINDOW + 2:
        return MarketState(reason="not enough market history to judge the state")

    trailing = trailing_return(market_closes, MARKET_WINDOW)
    steps = daily_returns(market_closes)
    recent = float(np.std(steps[-MARKET_WINDOW:]))
    baseline = float(np.std(steps))
    if not math.isfinite(trailing) or baseline <= 0:
        return MarketState(reason="the market state is not estimable")

    agitated = recent > baseline * PANIC_VOL_RATIO
    panic = trailing < 0 and agitated
    if panic:
        reason = (f"the crypto market is down {abs(trailing) * 100:.1f}% over "
                  f"{MARKET_WINDOW} days with volatility {recent / baseline:.1f}x "
                  f"its own baseline — momentum crashes in exactly this state, "
                  f"so nothing new is opened")
    elif trailing < 0:
        reason = (f"the market is down {abs(trailing) * 100:.1f}% over "
                  f"{MARKET_WINDOW} days but calm, which momentum survives")
    else:
        reason = (f"the market is up {trailing * 100:.1f}% over "
                  f"{MARKET_WINDOW} days")
    return MarketState(trailing_return=trailing, recent_vol=recent,
                       baseline_vol=baseline, panic=panic, reason=reason)


@dataclass(frozen=True)
class CrossSectionSignal:
    """One coin's cross-sectional view, and what acting on it would require."""

    value: float = 0.0
    score: float = 0.0
    rank: int = 0
    cohort: int = 0
    expected_edge_bps: float = 0.0
    drift_bps_per_day: float = 0.0
    min_hold_days: float = 0.0
    daily_vol_bps: float = 0.0
    eligible: bool = False
    reason: str = ""
    lookbacks: dict[int, float] = field(default_factory=dict)


def minimum_holding_days(round_trip_bps: float, drift_bps_per_day: float,
                         safety_multiple: float) -> float:
    """How long the position must be held for the edge to cover the round trip.

    ``H* = k * C / mu``. Identical in form to the time-series strategy, and it
    matters more here: crypto costs 25 basis points a side at Alpaca's first
    tier, so a round trip is fifty before the spread, against equities that
    are commission-free.
    """
    if drift_bps_per_day <= 0 or round_trip_bps <= 0:
        return float("inf")
    return safety_multiple * round_trip_bps / drift_bps_per_day


def signal(symbol: str, score: float, rank: int, cohort: int,
           closes: np.ndarray, pooled: PooledCrossSection | None,
           state: MarketState, *, round_trip_bps: float = 0.0,
           safety_multiple: float = 1.5,
           planned_hold_days: float = 28.0,
           own_trend_positive: bool = True) -> CrossSectionSignal:
    """Decide whether this coin's rank is worth buying, and for how long."""
    vol_bps = 0.0
    steps = daily_returns(closes)
    if steps.size >= LOOKBACKS[-1]:
        vol_bps = float(np.std(steps[-LOOKBACKS[-1]:])) * 10_000.0

    base = CrossSectionSignal(score=score, rank=rank, cohort=cohort,
                              daily_vol_bps=vol_bps)

    if cohort < MIN_CROSS_SECTION:
        return CrossSectionSignal(
            **{**base.__dict__,
               "reason": (f"only {cohort} coins priced, and ranking needs at "
                          f"least {MIN_CROSS_SECTION} — a top pick out of "
                          f"{cohort} is a short list, not a cross-section")})

    if pooled is None or not pooled.credible:
        return CrossSectionSignal(
            **{**base.__dict__,
               "reason": ((pooled.explain() if pooled else
                           "no cross-sectional premium measured yet")
                          + " Nothing is traded on an unmeasured edge.")})

    if state.panic:
        return CrossSectionSignal(**{**base.__dict__, "reason": state.reason})

    # A negative premium is not the same fact as a low rank, and reporting
    # one as the other sends an operator looking at the wrong thing. When the
    # measured premium is negative the ranking predicts *reversal*: today's
    # winners underperform tomorrow. The tradeable side of that is short, and
    # this book cannot short, so the whole strategy stands down rather than
    # any particular coin being unattractive.
    if pooled.beta_bps <= 0:
        return CrossSectionSignal(
            **{**base.__dict__,
               "reason": (f"the ranking currently predicts reversal, not "
                          f"momentum ({pooled.beta_bps:+.2f}bp/day per unit of "
                          f"rank, t={pooled.t_stat:+.2f}) — the tradeable side "
                          f"of that is short and this book cannot short, so "
                          f"the crypto ranking is standing down entirely")})

    # Long-only: a below-average coin is one to avoid, not one to sell short.
    if score <= 0:
        return CrossSectionSignal(
            **{**base.__dict__,
               "reason": (f"ranks {rank} of {cohort} on one-to-four-week "
                          f"momentum, below the cross-sectional average, and "
                          f"this book cannot short — there is nothing to buy "
                          f"here")})

    # The cross-section says which coin; the coin's own trend says whether to
    # be in the asset class at all. A long-only slice of a long-short factor is
    # the factor plus the market, and this is the part that declines the market.
    if not own_trend_positive:
        return CrossSectionSignal(
            **{**base.__dict__,
               "reason": (f"ranks {rank} of {cohort} against its peers but is "
                          f"falling in its own right — relative strength in a "
                          f"falling coin is still a losing long")})

    conviction = float(np.clip(score / SCORE_SATURATION, 0.0, 1.0))
    drift = min(MAX_DAILY_DRIFT_BPS, pooled.beta_bps * score)
    hold = minimum_holding_days(round_trip_bps, drift, safety_multiple)
    hold = max(1.0, hold)

    if hold > planned_hold_days:
        return CrossSectionSignal(
            **{**base.__dict__,
               "drift_bps_per_day": drift, "min_hold_days": hold,
               "reason": (f"a {drift:.2f}bp/day edge needs {hold:.0f} days to "
                          f"cover {round_trip_bps:.2f}bp of round trip, past "
                          f"the {planned_hold_days:.0f}-day horizon this "
                          f"estimate is good for — the ranking is real and too "
                          f"slow to pay for itself at these fees")})

    edge = drift * hold
    return CrossSectionSignal(
        value=conviction, score=score, rank=rank, cohort=cohort,
        expected_edge_bps=edge, drift_bps_per_day=drift, min_hold_days=hold,
        daily_vol_bps=vol_bps, eligible=True,
        reason=(f"ranks {rank} of {cohort} on one-to-four-week momentum "
                f"({score:+.2f}σ above the cross-section) at "
                f"{pooled.beta_bps:+.2f}bp per unit of rank, giving "
                f"{drift:.2f}bp/day; held {hold:.0f} days that is {edge:.1f}bp "
                f"against {round_trip_bps:.2f}bp of cost"))
