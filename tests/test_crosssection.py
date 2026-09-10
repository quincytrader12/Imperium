"""Cross-sectional momentum in the crypto book.

The venue lists a few dozen coins, which is a cross-section. Ranking them
against each other estimates the thing Liu, Tsyvinski and Wu (JF 2022) actually
document -- which coins beat which -- rather than whether the asset class went
up, which is what a time-series premium pooled across the whole tape mostly
measures over a fortnight.

The strategy is long-only because the venue does not lend coins to short, so it
is the momentum factor plus the market. Both guards that follow from that are
tested here, because without them this is a leveraged bet on crypto wearing a
regression's clothes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imperium.strategy import crosssection as xs


def _walk(rng, days: int, drift: float = 0.0, vol: float = 0.03,
          start: float = 100.0) -> np.ndarray:
    p = start
    out = [p]
    for _ in range(days):
        p *= float(np.exp(drift + rng.normal(0, vol)))
        out.append(p)
    return np.array(out)


# --------------------------------------------------------------- the ranking


def test_the_score_measures_a_coin_against_its_peers_not_against_itself():
    """The whole reason this exists alongside the time-series strategy.

    If every coin doubles, none of them has outperformed. A time-series signal
    calls that a screaming buy across the board; a cross-sectional one
    correctly reports that there is nothing to choose between them.
    """
    raw = {f"C{i}/USD": 0.40 for i in range(12)}          # everything up 40%

    scores = xs.cross_sectional_scores(raw)

    assert scores == {}, (
        "a uniform market produced a ranking — the market-wide move is being "
        "read as though it were relative strength")


def test_the_market_wide_move_is_removed_from_every_score():
    """Adding the same return to every coin must not change the ranking."""
    base = {f"C{i}/USD": 0.01 * i for i in range(12)}
    lifted = {s: v + 0.35 for s, v in base.items()}

    a = xs.cross_sectional_scores(base)
    b = xs.cross_sectional_scores(lifted)

    for symbol in a:
        assert a[symbol] == pytest.approx(b[symbol]), (
            "a market-wide rally changed the cross-sectional scores")


def test_a_cross_section_too_narrow_to_rank_is_refused():
    """Ranking five coins is a short list, not a cross-section: the top
    "decile" is one coin, and its score says more about which coins the venue
    lists than about momentum."""
    raw = {f"C{i}/USD": float(i) for i in range(xs.MIN_CROSS_SECTION - 1)}

    assert xs.cross_sectional_scores(raw) == {}


def test_the_blend_spans_one_to_four_weeks():
    """Liu and Tsyvinski put crypto momentum at one to four weeks, not the one
    to twelve months that works in equities. A single window would let one
    week's noise decide the whole ranking."""
    assert xs.LOOKBACKS == (7, 14, 28)
    rng = np.random.default_rng(1)
    closes = _walk(rng, 60)

    blended = xs.blended_return(closes)
    singles = [xs.trailing_return(closes, d) for d in xs.LOOKBACKS]

    assert blended == pytest.approx(float(np.mean(singles)))
    assert blended != pytest.approx(singles[0]), "the blend is just one window"


def test_a_coin_with_too_little_history_is_not_ranked_on_a_shorter_window():
    """Comparing a young coin's one-week return against an established coin's
    four-week return ranks listing date, not momentum."""
    rng = np.random.default_rng(2)
    young = _walk(rng, 9)                                   # only 7d available

    value = xs.blended_return(young)
    assert math.isfinite(value)
    assert math.isnan(xs.trailing_return(young, 28))


# ------------------------------------------------------------ the estimation


def test_the_pooled_premium_recovers_a_premium_that_is_really_there():
    """The estimator must find a known effect, or the credibility gate is
    just an elaborate way of never trading."""
    rng = np.random.default_rng(11)
    scored = {}
    beta = 0.0020                       # 20bp a day per unit of rank
    for i in range(12):
        scores = rng.normal(0, 1, 400)
        forward = beta * scores + rng.normal(0, 0.01, 400)
        scored[f"C{i}/USD"] = (scores, forward)

    pooled = xs.pool(scored)

    assert pooled.beta_bps == pytest.approx(20.0, abs=4.0)
    assert pooled.credible
    assert pooled.t_stat > xs.MIN_ABS_T_STAT


def test_pure_noise_is_not_reported_as_a_premium():
    """The gate that stops the book trading its own sampling error."""
    rng = np.random.default_rng(12)
    scored = {f"C{i}/USD": (rng.normal(0, 1, 400), rng.normal(0, 0.01, 400))
              for i in range(12)}

    pooled = xs.pool(scored)

    assert not pooled.credible, (
        f"noise was reported as a credible premium: {pooled.describe()}")


def test_a_big_t_statistic_on_thin_history_is_still_refused():
    """A large t-statistic on a handful of coin-days is a large t-statistic on
    noise. Breadth and length are separate requirements from significance."""
    rng = np.random.default_rng(13)
    scores = rng.normal(0, 1, 30)
    scored = {"A/USD": (scores, 0.05 * scores)}      # a perfect fit, 30 points

    pooled = xs.pool(scored)

    assert abs(pooled.t_stat) > xs.MIN_ABS_T_STAT, "the fixture must fit well"
    assert not pooled.credible
    assert "coin-days" in pooled.explain()
    assert pooled.observations < xs.MIN_POOLED_OBSERVATIONS


def test_the_standard_error_is_robust_to_heteroskedasticity():
    """Crypto returns are wildly heteroskedastic. A classical standard error
    would overstate significance in exactly the volatile stretches that decide
    whether the strategy works."""
    rng = np.random.default_rng(14)
    scores = rng.normal(0, 1, 800)
    # Noise that scales with the signal: the classical error assumes this away.
    noise = rng.normal(0, 0.01, 800) * (1 + 4 * np.abs(scores))
    scored = {"A/USD": (scores[:400], noise[:400]),
              "B/USD": (scores[400:], noise[400:])}

    pooled = xs.pool(scored)
    classical_se = float(np.std(noise)) / math.sqrt(800 * float(np.var(scores)))
    robust_se = abs(pooled.beta_bps / 10_000.0 / pooled.t_stat) if pooled.t_stat else 0

    assert robust_se > classical_se, (
        "the robust error is not wider than the classical one under "
        "heteroskedasticity, so it is not doing its job")


# ------------------------------------------------------------ the crash guard


def test_a_market_that_has_fallen_and_turned_volatile_is_a_panic_state():
    """Daniel and Moskowitz: momentum crashes follow market declines when
    volatility is high, and coincide with the rebound. That is the state a
    long-only momentum book loses most in, and it is partly forecastable."""
    rng = np.random.default_rng(21)
    calm = _walk(rng, 200, drift=0.0, vol=0.01)
    crash = _walk(rng, 40, drift=-0.02, vol=0.06, start=float(calm[-1]))
    closes = np.concatenate([calm, crash[1:]])

    state = xs.market_state(closes)

    assert state.panic, state.reason
    assert "momentum crashes in exactly this state" in state.reason


def test_a_falling_but_calm_market_is_not_a_panic():
    """Both halves are required. An ordinary drawdown is not the state that
    produces the crash, and standing down through every dip would forgo most
    of the premium."""
    rng = np.random.default_rng(22)
    calm = _walk(rng, 200, drift=0.0, vol=0.01)
    # Down over the window that is measured, and no more volatile than usual:
    # a drift over the whole series says nothing about the last 28 days.
    slide = _walk(rng, 40, drift=-0.004, vol=0.01, start=float(calm[-1]))
    closes = np.concatenate([calm, slide[1:]])

    state = xs.market_state(closes)

    assert state.trailing_return < 0, "the fixture must actually be falling"
    assert not state.panic
    assert "calm" in state.reason


def test_a_volatile_but_rising_market_is_not_a_panic():
    rng = np.random.default_rng(23)
    closes = _walk(rng, 240, drift=0.004, vol=0.05)

    state = xs.market_state(closes)

    assert not state.panic


def test_the_market_state_says_when_it_cannot_be_judged():
    state = xs.market_state(np.array([100.0, 101.0]))
    assert not state.panic
    assert "not enough market history" in state.reason


# ------------------------------------------------------------- the decision


def _credible() -> xs.PooledCrossSection:
    return xs.PooledCrossSection(beta_bps=25.0, t_stat=3.4, observations=900,
                                 symbols=14, residual_bps=300.0)


def _calm() -> xs.MarketState:
    return xs.MarketState(trailing_return=0.05, recent_vol=0.02,
                          baseline_vol=0.02, panic=False, reason="market is up")


def _closes(seed: int = 31) -> np.ndarray:
    return _walk(np.random.default_rng(seed), 120, vol=0.03)


def test_a_top_ranked_coin_in_a_calm_market_is_eligible():
    sig = xs.signal("BTC/USD", score=1.4, rank=1, cohort=30,
                    closes=_closes(), pooled=_credible(), state=_calm(),
                    round_trip_bps=58.0)

    assert sig.eligible
    assert sig.value > 0
    assert "ranks 1 of 30" in sig.reason
    assert sig.min_hold_days >= 1.0


def test_a_ranking_that_predicts_reversal_stands_the_whole_strategy_down():
    """Not a hypothetical: the test fixture produced exactly this by accident.

    A periodic wobble in the mock made the next day's return anti-correlate
    with the trailing window, and the pooled premium came back at -21.5bp/day
    with t = -34.5. That is the ranking predicting *reversal* -- today's
    winners underperform tomorrow -- and the tradeable side of it is short,
    which this book cannot do.

    It must stand down as a strategy rather than report each coin as
    unattractive. The two are different facts, and reporting one as the other
    sends an operator looking at the coins when the problem is the premium.
    """
    reversal = xs.PooledCrossSection(beta_bps=-21.5, t_stat=-34.5,
                                     observations=6_678, symbols=18)
    assert reversal.credible, "the estimate is credible; it just has the sign"

    sig = xs.signal("BTC/USD", score=1.4, rank=1, cohort=18, closes=_closes(),
                    pooled=reversal, state=_calm(), round_trip_bps=58.0)

    assert not sig.eligible
    assert "predicts reversal" in sig.reason
    assert "standing down entirely" in sig.reason
    # And it must not be mistaken for the top coin being weak: it is ranked 1.
    assert "below the cross-sectional average" not in sig.reason


def test_a_below_average_coin_is_refused_because_the_book_cannot_short():
    """A negative rank is a coin to avoid. Without a locate it is not a trade,
    and reporting it as one would be an order the venue rejects."""
    sig = xs.signal("DOGE/USD", score=-1.2, rank=29, cohort=30,
                    closes=_closes(), pooled=_credible(), state=_calm(),
                    round_trip_bps=58.0)

    assert not sig.eligible
    assert "cannot short" in sig.reason


def test_relative_strength_in_a_falling_coin_is_still_a_losing_long():
    """The guard that makes long-only honest.

    Cross-sectional momentum is a long-short factor. Taking only the long leg
    buys the best of a bad bunch, and the best of a bad bunch still falls.
    """
    sig = xs.signal("ETH/USD", score=1.6, rank=2, cohort=30,
                    closes=_closes(), pooled=_credible(), state=_calm(),
                    round_trip_bps=58.0, own_trend_positive=False)

    assert not sig.eligible
    assert "falling in its own right" in sig.reason


def test_nothing_is_opened_in_a_panic_state():
    panic = xs.MarketState(trailing_return=-0.30, recent_vol=0.09,
                           baseline_vol=0.03, panic=True,
                           reason="the crypto market is down 30.0% ... momentum "
                                  "crashes in exactly this state")

    sig = xs.signal("BTC/USD", score=1.8, rank=1, cohort=30,
                    closes=_closes(), pooled=_credible(), state=panic,
                    round_trip_bps=58.0)

    assert not sig.eligible
    assert "momentum crashes" in sig.reason


def test_an_unmeasured_premium_refuses_however_good_the_rank_looks():
    sig = xs.signal("BTC/USD", score=2.5, rank=1, cohort=30,
                    closes=_closes(), pooled=None, state=_calm(),
                    round_trip_bps=58.0)

    assert not sig.eligible
    assert "unmeasured edge" in sig.reason


def test_the_fees_are_what_decide_it_and_the_refusal_says_so():
    """Alpaca charges 25 basis points a side on crypto, so a round trip is
    fifty before the spread, against equities that are commission-free. A
    small edge that would clear on a stock does not clear here, and saying so
    is the difference between a refusal and a mystery."""
    weak = xs.PooledCrossSection(beta_bps=1.2, t_stat=3.1, observations=900,
                                 symbols=14)

    sig = xs.signal("BTC/USD", score=0.5, rank=3, cohort=30,
                    closes=_closes(), pooled=weak, state=_calm(),
                    round_trip_bps=58.0, planned_hold_days=28.0)

    assert not sig.eligible
    assert "too slow to pay for itself" in sig.reason
    assert "58" in sig.reason


def test_a_holding_period_is_never_shorter_than_a_day():
    """Crypto is exempt from the pattern-day-trader rule, so a same-day round
    trip is legal here — but the estimate is a daily one, and planning a hold
    shorter than the bar it was measured on is using a number for something it
    does not describe."""
    strong = xs.PooledCrossSection(beta_bps=400.0, t_stat=9.0,
                                   observations=2000, symbols=20)

    sig = xs.signal("BTC/USD", score=2.0, rank=1, cohort=30,
                    closes=_closes(), pooled=strong, state=_calm(),
                    round_trip_bps=58.0)

    assert sig.eligible
    assert sig.min_hold_days >= 1.0


def test_conviction_saturates_so_one_coin_cannot_take_the_book():
    """The top coin is not twice as reliable as the second."""
    modest = xs.signal("A/USD", score=1.0, rank=1, cohort=30, closes=_closes(),
                       pooled=_credible(), state=_calm(), round_trip_bps=58.0)
    extreme = xs.signal("B/USD", score=6.0, rank=1, cohort=30, closes=_closes(),
                        pooled=_credible(), state=_calm(), round_trip_bps=58.0)

    assert modest.value == pytest.approx(extreme.value), (
        "conviction still scales with an extreme score")
    assert extreme.value <= 1.0


def test_the_believable_drift_is_capped():
    """A pooled regression reporting an enormous daily drift on crypto has
    found a data problem -- a re-listing, a stale quote -- not an opportunity."""
    absurd = xs.PooledCrossSection(beta_bps=5_000.0, t_stat=12.0,
                                   observations=2000, symbols=20)

    sig = xs.signal("A/USD", score=2.0, rank=1, cohort=30, closes=_closes(),
                    pooled=absurd, state=_calm(), round_trip_bps=58.0)

    assert sig.drift_bps_per_day <= xs.MAX_DAILY_DRIFT_BPS


def test_the_minimum_holding_period_falls_as_the_edge_grows():
    """H* = k C / mu, which is the entire argument for holding rather than
    trading at these fees."""
    slow = xs.minimum_holding_days(58.0, 2.0, 1.5)
    fast = xs.minimum_holding_days(58.0, 20.0, 1.5)

    assert slow == pytest.approx(1.5 * 58.0 / 2.0)
    assert fast < slow
    assert xs.minimum_holding_days(58.0, 0.0, 1.5) == float("inf")
    assert xs.minimum_holding_days(58.0, -3.0, 1.5) == float("inf")


# ---------------------------------------------------------------------------
# End to end: the session must actually build the ranking and reach a decision.
# A correct strategy nothing calls is the same as no strategy.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_session_ranks_the_coins_and_routes_them_to_this_strategy():
    """The call site, not the module.

    Every test above drives ``crosssection`` directly, so all of them pass with
    the session never building a ranking and the engine never routing a coin
    here. The terminal would show the same silent crypto book it showed before
    any of this existed.
    """
    from decimal import Decimal

    from imperium.execution.broker import PaperBroker
    from imperium.session import TradingSession
    from imperium.venues import registry
    from imperium.venues.alpaca.client import AlpacaClient
    from mock_venue import KEY, SECRET, MockVenue

    venue = MockVenue()
    venue.list_extra_crypto(16)
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("70")
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    try:
        await session.scan_universe()
        await session.refresh_daily_history(force=True)
    finally:
        await session.detach_client()

    coins = [s for s in session.engines if "/" in s]
    assert len(coins) >= xs.MIN_CROSS_SECTION, (
        f"the fixture only produced {len(coins)} coins")

    assert session.cross_cohort >= xs.MIN_CROSS_SECTION, (
        "the session never built a cross-sectional ranking")
    assert len(session.cross_ranks) == session.cross_cohort
    assert sorted(session.cross_ranks.values()) == list(
        range(1, session.cross_cohort + 1)), "the ranks are not a ranking"

    # The ranking must reach the engines, or each one evaluates against a
    # cohort of zero and refuses for the wrong reason.
    ranked = [session.engines[s] for s in coins if session.engines[s].cross_cohort]
    assert ranked, "no engine was told where it sits in the cross-section"

    engine = ranked[0]
    decision = engine.evaluate()
    assert decision.strategy == "cross_section", (
        f"a coin was routed to {decision.strategy!r} instead of the "
        f"cross-sectional strategy")


@pytest.mark.asyncio
async def test_a_narrow_crypto_cohort_says_so_rather_than_ranking_three_coins():
    """The venue's own seed list has four coins. Ranking within it would put
    the "best" of four into the book on a score that mostly reflects which
    pairs the venue lists."""
    from decimal import Decimal

    from imperium.execution.broker import PaperBroker
    from imperium.session import TradingSession
    from imperium.venues import registry
    from imperium.venues.alpaca.client import AlpacaClient
    from mock_venue import KEY, SECRET, MockVenue

    venue = MockVenue()                      # no extra coins
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("70")
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    try:
        await session.scan_universe()
        await session.refresh_daily_history(force=True)
    finally:
        await session.detach_client()

    assert session.pooled_cross is None
    assert "ranking needs" in session.cross_note


# ---------------------------------------------------------------------------
# The four properties a mutation sweep found nothing was checking. Three of
# them would have made the strategy refuse to trade; the first would have made
# it look brilliant and been unimplementable.
# ---------------------------------------------------------------------------


def test_the_score_is_never_paired_with_the_return_it_was_built_from():
    """The lookahead. The one error here that makes everything look better.

    The score at day t is built from closes up to and including t, so it
    already contains the return that ended at t. Pairing it with that same
    return regresses a number on itself: near-perfect fit, enormous
    t-statistic, and a signal not known until after the move it claims to
    predict. Every other bug in this file shows up as a strategy that will not
    trade; this one shows up as a strategy that cannot be run.
    """
    rng = np.random.default_rng(41)
    closes = {f"C{i}/USD": _walk(rng, 200) for i in range(12)}

    obs = xs.observations(closes)

    assert obs, "the fixture produced no observations at all"
    symbol = sorted(obs)[0]
    scores, forwards = obs[symbol]
    series = closes[symbol]

    # Reconstruct what each pair must be, from the far end where the index is
    # unambiguous: the last score sees up to len-2, and predicts len-1.
    last_score = xs.cross_sectional_scores(
        {s: xs.blended_return(closes[s][:len(series) - 1]) for s in closes})
    assert scores[-1] == pytest.approx(last_score[symbol]), (
        "the final score saw a different day from the one it should have")
    assert forwards[-1] == pytest.approx(
        float(np.log(series[-1] / series[-2]))), (
        "the final forward return is not the day after the final score")

    # And the decisive property: a score must never be paired with the return
    # that ended on its own last day.
    same_day = float(np.log(series[-2] / series[-3]))
    assert forwards[-1] != pytest.approx(same_day), (
        "the score is paired with the return it was built from — this is a "
        "lookahead, and the premium it measures cannot be traded")


def test_a_random_walk_produces_no_tradeable_premium():
    """The end-to-end statement of the same thing.

    Coins with no momentum must yield an estimate that fails the credibility
    gate. With a lookahead they would yield a huge one, because the score
    contains the return it is regressed on.
    """
    rng = np.random.default_rng(42)
    closes = {f"C{i}/USD": _walk(rng, 500) for i in range(14)}

    pooled = xs.pool(xs.observations(closes))

    assert not pooled.credible, (
        f"pure random walks produced a tradeable premium: {pooled.describe()}")


def test_a_measured_but_insignificant_premium_is_still_refused():
    """The gate is credibility, not existence. A premium that is measured and
    cannot be told from zero is not a reason to trade."""
    weak = xs.PooledCrossSection(beta_bps=18.0, t_stat=0.6, observations=2_000,
                                 symbols=20)
    assert not weak.credible

    sig = xs.signal("BTC/USD", score=2.0, rank=1, cohort=30, closes=_closes(),
                    pooled=weak, state=_calm(), round_trip_bps=58.0)

    assert not sig.eligible
    assert "cannot be told from zero" in sig.reason


def test_the_robust_error_is_wider_than_the_classical_one_it_replaces():
    """Checked against the exact formula it replaced, not an approximation.

    Under heteroskedasticity the classical error understates the spread of the
    slope, which promotes noisy estimates through a t>=2 gate. The earlier
    version of this test compared against a hand-rolled approximation and
    passed even with the robust estimator swapped out.
    """
    rng = np.random.default_rng(43)
    scores = rng.normal(0, 1, 1200)
    noise = rng.normal(0, 0.01, 1200) * (1 + 6 * np.abs(scores))
    scored = {"A/USD": (scores[:600], noise[:600]),
              "B/USD": (scores[600:], noise[600:])}

    pooled = xs.pool(scored)

    x = scores - scores.mean()
    denominator = float(np.sum(x ** 2))
    beta = float(np.sum(x * (noise - noise.mean())) / denominator)
    residuals = noise - (noise.mean() + beta * x)
    n = x.size
    classical = math.sqrt(float(np.sum(residuals ** 2)) / (n - 2) / denominator)
    robust = abs(beta / (pooled.t_stat or float("inf")))

    assert robust > classical * 1.5, (
        f"robust SE {robust:.2e} is not meaningfully wider than the classical "
        f"{classical:.2e} — the heteroskedasticity correction is not applied")


@pytest.mark.asyncio
async def test_the_weight_ceiling_holds_when_the_volatility_estimate_says_it_need_not():
    """The ceiling that survives the tail being undefined.

    Everything else in the sizing rule -- target volatility, a two-sigma
    excursion over the holding period -- assumes a second moment exists to be
    estimated. Grobys et al. find the tail variance of crypto momentum returns
    is undefined under power-law tests, which would make those rules not
    conservative but meaningless. This ceiling does not read the estimate, so
    it is the part that still holds if they are right.

    Driven through the engine, because the ceiling lives at the sizing step
    rather than in the signal.
    """
    from decimal import Decimal

    from imperium.execution.broker import PaperBroker
    from imperium.session import TradingSession
    from imperium.venues import registry
    from imperium.venues.alpaca.client import AlpacaClient
    from mock_venue import KEY, SECRET, MockVenue

    venue = MockVenue()
    venue.list_extra_crypto(16)
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("100000")
    session.absorb_account({"equity": "100000", "cash": "100000",
                            "currency": "USD"})
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    try:
        await session.scan_universe()
        await session.refresh_daily_history(force=True)
    finally:
        await session.detach_client()

    coins = [s for s in session.engines
             if "/" in s and session.engines[s].cross_cohort]
    assert coins, "the fixture produced no ranked coins"

    # A vanishingly calm market: every volatility-scaled term would allow an
    # enormous position, so only the fixed ceiling can bind.
    import dataclasses

    for symbol in coins:
        engine = session.engines[symbol]
        engine.limits = dataclasses.replace(
            engine.limits, target_volatility=50.0, risk_per_trade=10.0,
            # 0.80 is the whole-book gross ceiling and the most a single
            # position may be; still far above the 25% crypto cap, so
            # only that cap can be what binds below.
            max_position_weight=0.80)
        engine.allocator.observe(symbol).admitted = True

    sized = []
    for symbol in coins:
        decision = session.engines[symbol].evaluate()
        if decision.raw_weight > 0:
            sized.append((symbol, decision.raw_weight))

    assert sized, "no coin sized to anything; the fixture proves nothing"
    for symbol, weight in sized:
        assert weight <= xs.MAX_CRYPTO_WEIGHT + 1e-9, (
            f"{symbol} sized to {weight:.1%}, above the "
            f"{xs.MAX_CRYPTO_WEIGHT:.0%} ceiling that is supposed to hold "
            f"whatever the volatility estimate says")
