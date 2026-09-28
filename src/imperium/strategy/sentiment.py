"""News sentiment, as a factor rather than a gate.

**What the evidence actually supports.** Tetlock, Saar-Tsechansky and
Macskassy (*Journal of Finance* 63(3), 2008, 1437-1467) find that the fraction
of negative words in firm-specific news forecasts low earnings and low
next-day returns -- prices "briefly underreact" and then correct. Tetlock
(*Journal of Finance* 62(3), 2007, 1139-1168) measures the aggregate version
and reports the shape plainly: a one-standard-deviation rise in media pessimism
costs 8.1bp the next day and gives 6.8bp of it back over the following four,
leaving a net of -1.3bp that is *not* significantly different from zero.

Read together those two results say something this module is built around: news
tone carries information over roughly a day, and almost none over a week. Every
position this program opens is held for days, because a small account cannot
round-trip equities intraday without tripping the pattern-day-trader rule. The
horizon where sentiment pays is therefore mostly *shorter* than the horizon this
program trades on.

**So it is a factor, and it is bounded.** Sentiment here can do exactly one
thing: scale the size of a position that some other strategy has already decided
to take, by at most :data:`MAX_TILT`. It cannot admit a symbol, it cannot reject
one, it cannot flip a long into a short, and it is applied *after* the cost gate
so that it can never manufacture the edge that pays for a trade. That ordering
is the whole design, and
``tests/test_sentiment.py::test_sentiment_can_never_admit_or_reject_a_symbol``
holds it in place.

**The cap is a prior, not an estimate.** This program has not measured what
headline tone is worth on its own fills, and until it has, the honest thing is
to bound the factor rather than fit it. Everything else here is estimated from
data and reported with a standard error; this number is not, and says so.

**The word lists.** Loughran and McDonald (*Journal of Finance* 66(1), 2011,
35-65) showed that general-purpose sentiment dictionaries are wrong for
financial text -- nearly three-quarters of the words the Harvard-IV list calls
negative are neutral accounting vocabulary ("liability", "tax", "cost",
"depreciation"). The lists below are drawn from their financial word lists and
extended with the vocabulary that is specific to newswire *headlines*, which
are a different genre again from the 10-K filings that dictionary was built on:
a filing never says "plunges", "beats" or "downgrades", and a headline says
little else. It is a working subset, not the full dictionary, and
:func:`score_text` reports how many words it actually matched so a caller can
tell a measured reading from a shrug.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

#: The most a headline can move a position's size, as a fraction.
#:
#: A cap, not a coefficient: sentiment scales an already-approved weight by at
#: most +/-20%, so the strategy that opened the position always keeps at least
#: four-fifths of the say. Chosen against the measured magnitudes above -- a
#: factor whose net multi-day effect is statistically indistinguishable from
#: zero has no business being a controlling input to a multi-day position.
MAX_TILT = 0.20

#: How fast a headline stops counting, in hours.
#:
#: Tetlock, Saar-Tsechansky and Macskassy find the return effect at a one-day
#: horizon; Tetlock finds the aggregate version substantially reversed by day
#: five. A 24-hour half-life puts a day-old headline at half weight and a
#: week-old one at under one percent, which is about what those two results
#: jointly imply.
HALF_LIFE_HOURS = 24.0

#: Headlines older than this are not read at all.
MAX_AGE_HOURS = 168.0

#: Shrinkage constant for thin coverage.
#:
#: One headline is an anecdote. The score is pulled toward zero by
#: ``w / (w + SHRINKAGE)``, where ``w`` is the sum of the articles' recency
#: weights rather than their raw count -- so a single fresh article carries a
#: quarter of its face value, three carry half, ten carry three-quarters, and a
#: week-old one carries almost nothing however loudly it was written. This is
#: the standard shrink-toward-the-prior-mean move, with the prior mean set to
#: "no view" -- and like :data:`MAX_TILT` the constant is a prior rather than a
#: fitted number.
SHRINKAGE = 3.0

#: How far after a negator its effect reaches, in words.
#:
#: Loughran and McDonald note that negation matters mainly for positive words
#: -- "not profitable" is the common construction, "not a loss" is rare. Three
#: words is their own window and it is kept here.
NEGATION_WINDOW = 3

_NEGATORS = frozenset({
    "no", "not", "none", "neither", "never", "nobody", "nothing", "nowhere",
    "cannot", "cant", "wont", "without", "fails", "fail", "failed", "failing",
    "lacks", "lack", "lacked", "denies", "denied", "deny", "halts", "halted",
    "misses", "miss", "missed",
})

#: Negative vocabulary: Loughran-McDonald financial negatives, plus the words
#: a newswire headline uses to say a price went down or a company is in
#: trouble.
_NEGATIVE = frozenset({
    # Loughran-McDonald financial negatives.
    "adverse", "adversely", "against", "bankruptcy", "bankrupt", "breach",
    "burden", "closure", "concern", "concerns", "complaint", "complaints",
    "criticism", "damages", "decline", "declined", "declines",
    "declining", "default", "defaults", "deficiency", "deficit", "delay",
    "delays", "delinquent", "deteriorate", "deteriorating", "diminish",
    "diminished", "discontinued", "dispute", "disputes", "disruption",
    "downgrade", "downgraded", "downgrades", "downturn", "doubt", "doubts",
    "erosion", "fail", "failed", "failing", "fails", "failure",
    "failures", "fraud", "fraudulent", "harm", "hurt",
    "impair", "impaired", "impairment", "inadequate", "insolvency",
    "insolvent", "investigation", "investigations", "lawsuit", "lawsuits",
    "layoff", "layoffs", "liquidation", "litigation", "loss", "losses",
    "denied", "denies", "lacked", "lacks", "misconduct", "negative",
    "penalty", "penalties", "poor", "probe",
    "probes", "problem", "problems", "recall", "recalls", "recession",
    "restructuring", "sanction", "sanctions", "sceptical",
    "shortfall", "shutdown", "suspend", "suspended", "suspension",
    "terminate", "terminated", "termination", "unfavorable", "unfavourable",
    "violation", "violations", "warn", "warned", "warning", "weak",
    "weakened", "weakness", "worse", "worsening", "writedown", "writeoff",
    # Headline-specific: how a newswire says the price fell or the news is bad.
    "bearish", "cautious", "crash", "crashes", "crashed", "cut", "cuts",
    "dip", "dips", "drop", "drops", "dropped", "falls", "fell", "halt",
    "lawsuit", "lower", "miss", "missed", "misses", "plummet", "plummets",
    "plunge", "plunges", "plunged", "pressure", "rejected", "rejection",
    "resign", "resigned", "resignation", "selloff", "sink", "sinks", "sank",
    "slash", "slashes", "slashed", "slide", "slides", "slip", "slips",
    "slump", "slumps", "stumble", "stumbles", "sue", "sued", "sues",
    "tumble", "tumbles", "tumbled", "underperform", "underperforms",
    "underwhelming", "skepticism", "slowdown",
})

#: Positive vocabulary, on the same basis.
_POSITIVE = frozenset({
    # Loughran-McDonald financial positives.
    "achieve", "achieved", "achievement", "advantage", "advantages", "beneficial",
    "benefit", "benefits", "best", "better", "boost", "boosted", "breakthrough",
    "efficiency", "efficient", "enhance", "enhanced", "enhancement",
    "excellent", "exceptional", "favorable", "favourable", "gain", "gained",
    "gains", "good", "great", "greater", "growth", "highest", "improve",
    "improved", "improvement", "improves", "innovation", "innovative",
    "leading", "opportunity", "opportunities", "outperform", "outperformed",
    "outperforms", "pleased", "positive", "profitable", "profitability",
    "progress", "record", "resolve", "resolved", "reward", "stability",
    "stable", "strength", "strengthen", "strengthened", "strong", "stronger",
    "succeed", "success", "successful", "superior", "upgrade", "upgraded",
    "upgrades", "win", "wins", "won",
    # Headline-specific.
    "acquire", "acquired", "acquisition", "approval", "approved", "approves",
    "beat", "beats", "bullish", "buy", "climb", "climbs", "climbed", "deal",
    "expansion", "expands", "higher", "jump", "jumps", "jumped", "launch",
    "launches", "rally", "rallies", "rallied", "raise", "raises", "raised",
    "rebound", "rebounds", "rise", "rises", "rose", "soar", "soars",
    "soared", "spike", "spikes", "surge", "surges", "surged", "tops",
    "topped", "upbeat", "upside", "buyback", "dividend", "partnership",
})

_WORD = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Article:
    """One headline, with the age that decides how much it counts."""

    headline: str
    age_hours: float
    summary: str = ""
    source: str = ""

    @property
    def text(self) -> str:
        # The headline is weighted the same as the summary rather than more.
        # A headline is written to be clicked and overstates; the summary is
        # the sober version of the same story, and averaging the two is the
        # cheapest correction available for that bias.
        return f"{self.headline} {self.summary}".strip()


@dataclass(frozen=True)
class TextScore:
    """A raw reading of one piece of text, before any weighting."""

    score: float
    positive: int
    negative: int

    @property
    def matched(self) -> int:
        return self.positive + self.negative


@dataclass
class Sentiment:
    """What the news says about one symbol, and how much of it there was.

    ``score`` is on [-1, 1] and is already shrunk and decayed, so a caller can
    use it directly. ``articles`` is kept beside it because a score of +0.6
    from one headline and the same score from twenty are not the same claim,
    and an operator reading the panel must be able to tell them apart.
    """

    symbol: str = ""
    score: float = 0.0
    articles: int = 0
    positive_words: int = 0
    negative_words: int = 0
    freshest_hours: float = math.inf
    headline: str = ""
    #: False when nothing was found, which is the normal case for most crypto
    #: pairs and for the long tail of small equities.
    covered: bool = False
    #: Set when the desk could not ask, as opposed to asking and finding
    #: nothing. The two look identical on a panel unless they are separated.
    unavailable: str = ""

    @property
    def label(self) -> str:
        if not self.covered:
            return "no news"
        if self.score >= 0.25:
            return "positive"
        if self.score <= -0.25:
            return "negative"
        return "mixed"

    def tilt(self, weight: float) -> float:
        """Scale an already-approved weight. Never changes its sign.

        Multiplicative and signed by the position's own direction: good news
        makes a long larger and a short smaller, bad news does the reverse.
        Bounded by :data:`MAX_TILT` in both directions, so the strategy that
        opened the position always keeps the majority of the say.
        """
        if not self.covered or weight == 0.0:
            return weight
        direction = 1.0 if weight > 0 else -1.0
        factor = 1.0 + MAX_TILT * _clamp(self.score) * direction
        return weight * factor

    def explain(self) -> str:
        """One clause for the reasoning panel, or "" when there is nothing."""
        if self.unavailable:
            return ""
        if not self.covered:
            return ""
        count = f"{self.articles} headline{'s' if self.articles != 1 else ''}"
        when = _age_text(self.freshest_hours)
        return (f"news reads {self.label} ({self.score:+.2f} from {count}, "
                f"freshest {when})")

    def as_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 3),
            "label": self.label,
            "articles": self.articles,
            "covered": self.covered,
            "headline": self.headline,
            "freshest_hours": (round(self.freshest_hours, 2)
                               if math.isfinite(self.freshest_hours) else None),
            "unavailable": self.unavailable,
            "tilt_pct": round(MAX_TILT * _clamp(self.score) * 100, 1),
        }


def _age_text(hours: float) -> str:
    if not math.isfinite(hours):
        return "unknown"
    if hours < 1.0:
        minutes = max(1, int(round(hours * 60)))
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if hours < 48.0:
        whole = int(round(hours))
        return f"{whole} hour{'s' if whole != 1 else ''} ago"
    days = int(round(hours / 24.0))
    return f"{days} day{'s' if days != 1 else ''} ago"


def _clamp(value: float, limit: float = 1.0) -> float:
    return max(-limit, min(limit, value))


def score_text(text: str) -> TextScore:
    """Count tone words in one piece of text, honouring negation.

    The score is ``(positive - negative) / matched`` -- a *proportion*, the
    form Loughran-McDonald and Tetlock both use, rather than a raw count. A
    raw count would make a long article outweigh a short one for no reason
    other than its length.
    """
    words = _WORD.findall((text or "").lower())
    positive = negative = 0
    negated_until = -1
    for index, word in enumerate(words):
        # Whether *this* word was negated is decided before it opens a window
        # of its own, and a negator still counts as a tone word in its own
        # right. "Fails to beat" has to score twice: once for the failure and
        # once for the beat it flips. An earlier version returned on the
        # negator and scored that headline as having said nothing at all.
        flipped = index <= negated_until
        if word in _NEGATORS:
            negated_until = index + NEGATION_WINDOW
        if word in _POSITIVE:
            # "not profitable" is negative; "not a loss" is rarely written, so
            # only positives are flipped. That asymmetry is theirs, not an
            # oversight here.
            negative += 1 if flipped else 0
            positive += 0 if flipped else 1
        elif word in _NEGATIVE:
            negative += 1
    matched = positive + negative
    if not matched:
        return TextScore(0.0, 0, 0)
    return TextScore((positive - negative) / matched, positive, negative)


def evaluate(symbol: str, articles: list[Article]) -> Sentiment:
    """Turn a symbol's recent headlines into a bounded, shrunk score."""
    fresh = [a for a in articles if a.age_hours <= MAX_AGE_HOURS]
    if not fresh:
        return Sentiment(symbol=symbol, covered=False)

    weighted_sum = 0.0
    weight_total = 0.0
    positive = negative = 0
    counted = 0
    loudest: tuple[float, str] = (0.0, "")
    for article in fresh:
        reading = score_text(article.text)
        if not reading.matched:
            # Read, but with nothing in it this module recognises. Counted in
            # neither direction rather than as a neutral vote, which would
            # dilute the symbols that do say something.
            continue
        decay = 0.5 ** (max(0.0, article.age_hours) / HALF_LIFE_HOURS)
        weighted_sum += reading.score * decay
        weight_total += decay
        positive += reading.positive
        negative += reading.negative
        counted += 1
        strength = abs(reading.score) * decay
        if strength > loudest[0]:
            loudest = (strength, article.headline)

    if not counted or weight_total <= 0.0:
        return Sentiment(symbol=symbol, covered=False,
                         freshest_hours=min(a.age_hours for a in fresh))

    raw = weighted_sum / weight_total
    # Shrunk on the *decayed* weight rather than the raw article count.
    #
    # The weighted average above already decides how much each article says
    # relative to the others -- but on its own it divides the decay straight
    # back out, so a single week-old headline scored exactly as loudly as a
    # single one filed this morning. Using the sum of the decay weights as the
    # effective sample size fixes that: age now costs an article both its share
    # of the average and its contribution to how much the average is trusted,
    # which is what a 24-hour half-life was meant to mean.
    shrunk = raw * (weight_total / (weight_total + SHRINKAGE))
    return Sentiment(
        symbol=symbol,
        score=_clamp(shrunk),
        articles=counted,
        positive_words=positive,
        negative_words=negative,
        freshest_hours=min(a.age_hours for a in fresh),
        headline=loudest[1],
        covered=True,
    )
