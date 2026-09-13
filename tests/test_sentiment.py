"""News sentiment: what it reads, and the limits on what it may do with it.

The load-bearing tests in this file are not the scoring ones -- a lexicon can
always be argued with. They are the ones that hold the factor inside its box:
it may scale a position the strategy already decided to take, by a bounded
amount, and it may do nothing else. If those pass and the lexicon is mediocre,
the worst case is a position sized a few percent wrong. If those fail, a
word list decides what this program trades.
"""

from __future__ import annotations

import math

import pytest

from imperium.execution.bars import Bar
from imperium.execution.engine import SymbolEngine
from imperium.execution.newsdesk import NewsDesk, to_articles
from imperium.execution.portfolio import PortfolioAllocator, Verdict
from imperium.execution.risk import limits_for_equity
from imperium.strategy import sentiment as st
from imperium.strategy.sentiment import Article, Sentiment
from imperium.telemetry.streams import TelemetryHub
from imperium.venues import registry


def _engine(symbol: str = "PLTR", *, equity: float = 70.0, day_trades: int = 0):
    limits = limits_for_equity(equity)
    allocator = PortfolioAllocator(limits)
    allocator.equity, allocator.cash = equity, equity
    allocator.day_trade_count = day_trades
    engine = SymbolEngine(symbol, registry.get(registry.DEFAULT_VENUE), limits,
                          allocator, TelemetryHub())
    allocator.observe(symbol).admitted = True
    price = 20.0
    for k in range(engine.params.warmup_bars + 5):
        engine.series.add(Bar(k * 60_000, price, price * 1.001, price * 0.999,
                              price, 1000.0, closed=True))
    engine.set_book(19.995, 20.005)
    return engine


def _news(score: float, *, articles: int = 5) -> Sentiment:
    return Sentiment(symbol="PLTR", score=score, articles=articles,
                     freshest_hours=1.0, headline="a headline", covered=True)


# -- the box the factor lives in ----------------------------------------

def test_sentiment_can_never_admit_or_reject_a_symbol():
    """THE test in this file. The factor is applied after the cost gate, so a
    glowing headline cannot manufacture the edge that pays for a spread and a
    grim one cannot veto a trade the numbers already justified. Run across the
    full range of scores against a fixed market: the verdict must not move."""
    verdicts = set()
    for score in (-1.0, -0.5, 0.0, 0.5, 1.0):
        engine = _engine(day_trades=2)
        engine.news = _news(score)
        verdicts.add(engine.evaluate().verdict)
    assert len(verdicts) == 1, (
        f"news changed the verdict: {[v.value for v in verdicts]}")


def _trending_engine(score: float, *, articles: int = 6):
    """An engine with enough daily history to reach the cost gate.

    The plain ``_engine`` above stops at "warming up: 0 of 40 daily bars" and
    never consults the gate at all, which makes it useless for asking what the
    gate was told.
    """
    import math
    from imperium.strategy.trend import PooledTrend

    engine = _engine(day_trades=2)
    price = 10.0
    for k in range(120):
        price *= 1.004 + 0.004 * math.sin(k * 1.7)
        engine.daily_bars.append(Bar(k * 86_400_000, price * 0.999, price * 1.01,
                                     price * 0.99, price, 1e6, closed=True))
    engine.pooled_trend = PooledTrend(15.0, 8.0, 9000, 40)
    engine.news = _news(score, articles=articles) if articles else Sentiment("PLTR")
    return engine


def test_the_number_the_cost_gate_sees_never_depends_on_the_news(monkeypatch):
    """The structural form of "it is not a gate", and the stronger one.

    Checking verdicts can only catch a leak where the market happens to sit on
    the gate's boundary -- and on the multi-day path it never does, because
    that strategy derives its holding period so the drift exactly covers the
    round trip. A mutation feeding the sentiment score into the gate therefore
    survives every verdict-based test. So this watches the gate itself:
    whatever the headlines say, the expected edge weighed against the spread
    must be the same number to the last decimal place.

    This is the single claim the whole design rests on. If it fails, news is
    deciding what gets traded, not how much of it.
    """
    from imperium.execution import costs as costs_mod
    from imperium.execution import engine as engine_mod

    real_gate = costs_mod.gate
    seen: dict[float, list] = {}

    for score in (-1.0, -0.5, 0.0, 0.5, 1.0):
        captured: list = []

        def spy(*args, expected_edge_bps=None, **kwargs):
            captured.append(expected_edge_bps)
            return real_gate(*args, expected_edge_bps=expected_edge_bps,
                             **kwargs)

        monkeypatch.setattr(engine_mod.costs, "gate", spy)
        decision = _trending_engine(score).evaluate()
        assert decision.verdict is Verdict.TRADING, decision.reason
        seen[score] = captured

    assert all(len(v) == 1 for v in seen.values()), (
        f"the gate was not consulted exactly once per decision: {seen}")
    values = {v[0] for v in seen.values()}
    assert len(values) == 1, (
        f"the news changed what the cost gate was asked to admit: {seen}")


def test_sentiment_never_flips_the_direction_of_a_position():
    """Prevents: the worst possible bug here -- a word list reversing a
    position that a measured signal chose the direction of."""
    for weight in (0.25, -0.25):
        for score in (-1.0, -0.4, 0.4, 1.0):
            tilted = _news(score).tilt(weight)
            assert math.copysign(1.0, tilted) == math.copysign(1.0, weight)
            assert tilted != 0.0


@pytest.mark.parametrize("score", [-1.0, -0.3, 0.0, 0.3, 1.0])
def test_the_tilt_is_bounded_at_the_declared_cap(score):
    """Prevents: the cap drifting upward until sentiment is the strategy. The
    literature this factor rests on -- Tetlock (2007) -- measures a net
    multi-day effect that is not distinguishable from zero, so the bound is
    the honest part of the design."""
    tilted = _news(score).tilt(0.30)
    assert abs(tilted - 0.30) <= 0.30 * st.MAX_TILT + 1e-12


def test_good_news_makes_a_long_bigger_and_a_short_smaller():
    assert _news(1.0).tilt(0.20) > 0.20
    assert abs(_news(1.0).tilt(-0.20)) < 0.20


def test_bad_news_makes_a_long_smaller_and_a_short_bigger():
    assert _news(-1.0).tilt(0.20) < 0.20
    assert abs(_news(-1.0).tilt(-0.20)) > 0.20


def test_no_coverage_changes_nothing_at_all():
    """Prevents: a symbol with no news being quietly penalised. Most of the
    crypto pairs and most small equities have no coverage on any given day,
    and "nobody wrote about it" is not information about its return."""
    quiet = Sentiment(symbol="XYZ", covered=False)
    assert quiet.tilt(0.25) == 0.25
    assert quiet.explain() == ""


def test_a_tilt_down_never_shrinks_a_position_below_the_viable_notional():
    """Prevents: sentiment producing a position too small to be worth its
    fees -- which would be the factor causing a bad trade rather than
    preventing one."""
    engine = _engine(day_trades=2)
    engine.news = _news(-1.0)
    grim = engine.evaluate()
    engine2 = _engine(day_trades=2)
    engine2.news = _news(0.0)
    plain = engine2.evaluate()
    if plain.verdict is Verdict.TRADING and plain.raw_weight > 0:
        from imperium.execution.risk import VIABLE_POSITION_NOTIONAL
        floor = VIABLE_POSITION_NOTIONAL / engine.allocator.equity
        assert grim.raw_weight >= min(plain.raw_weight, floor) - 1e-12


def test_every_decision_reports_what_the_news_did_even_when_it_did_nothing():
    """Prevents: an invisible input. An operator must be able to read the
    factor's contribution off the decision, including when it was zero."""
    engine = _engine(day_trades=2)
    payload = engine.evaluate().as_dict()
    for key in ("news_score", "news_label", "news_articles", "news_tilt_pct"):
        assert key in payload
    assert payload["news_label"] == "no news"
    assert payload["news_tilt_pct"] == 0.0


# -- the scoring ---------------------------------------------------------

def test_a_negator_still_counts_as_its_own_tone_word():
    """Prevents the bug this module shipped with once: returning on the
    negator, so "fails to beat" scored as a headline that said nothing."""
    reading = st.score_text("Acme fails to beat estimates")
    assert reading.matched >= 2
    assert reading.score < 0


def test_negation_flips_a_positive_word():
    assert st.score_text("the division is not profitable").score < 0


def test_accounting_vocabulary_is_not_read_as_negative():
    """Loughran and McDonald (2011) built their lists because general-purpose
    dictionaries call this sentence deeply negative. In financial text it says
    nothing at all."""
    reading = st.score_text(
        "Quarterly tax liability, depreciation and cost of capital reported")
    assert reading.matched == 0
    assert reading.score == 0.0


def test_a_plainly_good_headline_scores_positive():
    assert st.score_text("Shares surge after record earnings beat").score > 0.5


def test_a_plainly_bad_headline_scores_negative():
    assert st.score_text("Stock plunges as regulator opens fraud probe").score < -0.5


def test_the_score_is_a_proportion_so_length_does_not_decide_it():
    """Prevents: a long article outweighing a short one for no reason but its
    length -- the reason both Loughran-McDonald and Tetlock use proportions."""
    short = st.score_text("surge")
    long = st.score_text("surge " * 40)
    assert short.score == pytest.approx(long.score)


# -- weighting -----------------------------------------------------------

def test_one_headline_is_shrunk_harder_than_ten():
    """Prevents: a single anecdote moving a position as much as a consensus."""
    one = st.evaluate("A", [Article("shares surge on record beat", 0.5)])
    many = st.evaluate("A", [Article("shares surge on record beat", 0.5)] * 10)
    assert 0 < one.score < many.score


def test_an_old_headline_counts_for_less_than_a_new_one():
    """Prevents: week-old news sizing today's position. Tetlock,
    Saar-Tsechansky and Macskassy measure the effect at a one-day horizon."""
    fresh = st.evaluate("A", [Article("shares surge on record beat", 0.5)])
    stale = st.evaluate("A", [Article("shares surge on record beat", 120.0)])
    assert fresh.score > stale.score


def test_headlines_past_the_age_limit_are_not_read_at_all():
    old = st.evaluate("A", [Article("shares surge on record beat",
                                    st.MAX_AGE_HOURS + 1)])
    assert not old.covered
    assert old.tilt(0.2) == 0.2


def test_headlines_with_no_tone_words_do_not_dilute_the_ones_that_have_them():
    """Prevents: a symbol with one real story and nine wire notices scoring as
    nine-tenths neutral."""
    mixed = st.evaluate("A", [
        Article("shares surge on record beat", 0.5),
        *[Article("company schedules quarterly conference call", 0.5)] * 9,
    ])
    alone = st.evaluate("A", [Article("shares surge on record beat", 0.5)])
    assert mixed.score == pytest.approx(alone.score)


# -- the desk ------------------------------------------------------------

class _FakeClient:
    def __init__(self, payload, *, raises: Exception | None = None):
        self.payload, self.raises, self.calls = payload, raises, 0

    async def news(self, symbols, **kwargs):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.payload


@pytest.mark.asyncio
async def test_the_desk_scores_what_the_venue_returns():
    import datetime as dt
    now = dt.datetime.now(tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    desk = NewsDesk()
    await desk.refresh(_FakeClient({
        "AAPL": [{"headline": "Apple shares surge on record beat",
                  "created_at": now, "summary": "", "source": "benzinga"}],
    }), ["AAPL", "MSFT"])
    assert desk.sentiment("AAPL").score > 0
    assert desk.sentiment("AAPL").covered
    assert not desk.sentiment("MSFT").covered


@pytest.mark.asyncio
async def test_a_news_outage_is_absence_and_never_an_exception():
    """Prevents: a text API taking the trading loop down with it. The factor is
    secondary; its failure mode must be that positions are sized exactly as
    they were before it existed."""
    desk = NewsDesk()
    await desk.refresh(_FakeClient(None, raises=RuntimeError("boom")), ["AAPL"])
    assert desk.last_error
    assert desk.sentiment("AAPL").tilt(0.25) == 0.25


@pytest.mark.asyncio
async def test_switching_the_factor_off_stops_it_asking_and_stops_it_tilting():
    desk = NewsDesk()
    desk.enabled = False
    client = _FakeClient({})
    await desk.refresh(client, ["AAPL"])
    assert client.calls == 0
    assert desk.sentiment("AAPL").tilt(0.25) == 0.25


def test_the_desk_does_not_go_back_to_the_venue_on_every_tick():
    """Prevents: the news factor spending the request budget the scanner needs
    to price the book it is holding."""
    desk = NewsDesk()
    assert desk.due()
    desk.refreshed_at = 1_000.0
    assert not desk.due(now=1_000.0 + desk.refresh_seconds - 1)
    assert desk.due(now=1_000.0 + desk.refresh_seconds + 1)


def test_a_story_filed_in_the_future_does_not_get_more_than_full_weight():
    """Prevents: a clock skew of a few seconds between the venue and this
    machine turning the decay curve upward."""
    import datetime as dt
    ahead = (dt.datetime.now(tz=dt.timezone.utc)
             + dt.timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    articles = to_articles([{"headline": "x", "created_at": ahead}])
    assert articles[0].age_hours == 0.0


def test_an_unreadable_timestamp_is_treated_as_infinitely_old():
    """Prevents: a malformed date defaulting to "just now" and giving a story
    of unknown vintage full weight."""
    articles = to_articles([{"headline": "x", "created_at": "not a date"}])
    assert math.isinf(articles[0].age_hours)
    assert not st.evaluate("A", articles).covered


def test_a_held_position_still_reports_the_news_it_is_not_being_resized_by():
    """Prevents: the panel saying "no news" about a symbol that has news,
    purely because the hold path does not resize and so never ran the tilt.
    Omitting an input is tolerable; misreporting it is not."""
    engine = _engine(day_trades=2)
    engine.news = _news(0.8)
    engine.trend_held = True
    engine.trend_days_held = 3.0
    decision = engine.evaluate()
    assert decision.news_label == "positive"
    assert decision.news_articles == 5
    assert decision.news_score == pytest.approx(0.8)


def test_a_decision_with_no_position_does_not_claim_the_news_changed_its_size():
    """Prevents: "news reads positive, which changed the size" appended to a
    decision whose size is zero."""
    engine = _engine(day_trades=2)
    engine.news = _news(0.9)
    engine.tradable = False
    decision = engine.evaluate()
    assert "changed the size" not in decision.reason


def test_a_tilt_the_limits_swallow_says_so_rather_than_looking_broken():
    """Prevents: an operator reading "news reads positive" beside a position
    that did not change and concluding the factor does not work. When a limit
    above the factor takes the whole tilt, the reason has to name the limit."""
    import dataclasses
    engine = _engine(day_trades=2)
    engine.news = _news(0.9)
    # Squeeze the per-symbol cap so a tilt up has nowhere to go. The sizing
    # already runs into the cap, so the tilt is cancelled by it entirely.
    engine.limits = dataclasses.replace(engine.limits,
                                        max_position_weight=0.0001)
    decision = engine.evaluate()
    if decision.verdict is Verdict.TRADING and decision.news_note:
        assert ("absorbed" in decision.reason
                or "% larger" in decision.reason
                or "% smaller" in decision.reason)


@pytest.mark.parametrize("score", [-1.0, -0.6, 0.6, 1.0])
def test_the_engine_applies_the_tilt_once_and_within_the_cap(score):
    """Prevents: the tilt being applied twice.

    ``Sentiment.tilt`` is bounded, but that bounds one call. An engine that
    calls it on two paths through the same decision compounds it -- two
    applications of a 20% cap is a 44% swing, and nothing that tests the
    function in isolation would notice. Measured end to end against the same
    market with no news at all."""
    quiet = _trending_engine(0.0, articles=0).evaluate()
    loud = _trending_engine(score).evaluate()
    assert quiet.verdict is Verdict.TRADING
    assert loud.verdict is Verdict.TRADING
    assert quiet.raw_weight > 0

    ratio = loud.raw_weight / quiet.raw_weight
    assert 1.0 - st.MAX_TILT - 1e-9 <= ratio <= 1.0 + st.MAX_TILT + 1e-9, (
        f"a score of {score} moved the weight by {(ratio - 1) * 100:.1f}%, "
        f"past the {st.MAX_TILT:.0%} cap")
    # And the decision's own report of the tilt must match what it did.
    assert loud.news_tilt_pct == pytest.approx((ratio - 1.0) * 100, abs=0.6)


#: Headlines in the shape Alpaca's newswire actually files them, with the
#: direction a reader would assign. Not a benchmark -- fifteen cases prove
#: nothing about accuracy. It is a regression fence: the word lists get edited,
#: and an edit that fixes one ticker while breaking "Fails To Beat" should be
#: visible before it ships rather than after.
HEADLINES = [
    ("Nvidia Q3 Earnings Beat Estimates, Raises Guidance", 1),
    ("Tesla Shares Plunge After Q2 Delivery Miss", -1),
    ("Apple Announces $110 Billion Buyback, Boosts Dividend", 1),
    ("SEC Opens Fraud Investigation Into XYZ Corp", -1),
    ("Analyst Upgrades Palantir To Buy On Strong Backlog", 1),
    ("Goldman Downgrades Ford, Cuts Price Target", -1),
    ("Boeing CEO Resigns Amid Safety Probe", -1),
    ("Bitcoin Rallies Past $70,000 As ETF Inflows Surge", 1),
    ("Coinbase Q4 Revenue Tops Estimates", 1),
    ("Company To Present At Investor Conference On Tuesday", 0),
    ("XYZ Announces Quarterly Dividend Of $0.25 Per Share", 1),
    ("Firm Fails To Beat Consensus, Warns On Full-Year Outlook", -1),
    ("Solana Slides As Network Outage Halts Transactions", -1),
    ("Ethereum Climbs After Successful Upgrade", 1),
    ("Q3 tax liability and depreciation expense reported", 0),
]


@pytest.mark.parametrize("headline,want", HEADLINES)
def test_the_lexicon_reads_real_newswire_headlines_the_way_a_person_would(
        headline, want):
    reading = st.score_text(headline)
    got = 1 if reading.score > 0.15 else (-1 if reading.score < -0.15 else 0)
    assert got == want, (
        f"read {reading.score:+.2f} "
        f"({reading.positive} positive / {reading.negative} negative)")
