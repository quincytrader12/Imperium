"""Fetching headlines and keeping a scored view of them per symbol.

Separate from :mod:`imperium.strategy.sentiment` on purpose: that module is
pure -- text in, a bounded number out -- and can be reasoned about and tested
without a venue. This one owns the parts that can fail: the network, the clock,
the cache, and the request budget.

**Why it is cheap.** News is a secondary factor. It is not allowed to cost the
scanner anything it would otherwise have spent on prices, so this refreshes on
its own slow interval, asks only about symbols that are actually candidates,
and treats every failure as absence rather than as an error. A symbol with no
coverage and a symbol the desk could not ask about both end up with a
:class:`~imperium.strategy.sentiment.Sentiment` that tilts nothing -- but they
report differently, because "there is no news" and "I could not look" are
different things to an operator.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any, Iterable, Protocol

from imperium.venues import yahoo
from imperium.strategy.sentiment import (
    MAX_TILT, Article, Sentiment, evaluate,
)

log = logging.getLogger("imperium.news")

#: How often the desk goes back to the venue, in seconds.
#:
#: Ten minutes. Headlines arrive in minutes, not seconds, and the factor they
#: feed has a 24-hour half-life -- refreshing faster would spend request budget
#: to sharpen a number whose own decay curve cannot tell the difference.
REFRESH_SECONDS = 600.0

#: How far back each refresh reads, in hours.
#:
#: Matches ``sentiment.MAX_AGE_HOURS``: reading further back would fetch
#: articles that score zero weight anyway.
LOOKBACK_HOURS = 168.0

#: The most symbols the desk will ask about in one refresh.
#:
#: A cap on the request budget rather than on what can be scored. Candidates
#: are asked about in priority order, so the symbols closest to being traded
#: are the ones that get covered when the universe is wider than this.
#:
#: Sixty rather than the venue path's natural few hundred, because RSS gives
#: no per-item symbol attribution and so costs one request per symbol. At
#: eight in flight that is a refresh of a few seconds every ten minutes; three
#: hundred would be three hundred requests to a public feed, which is not a
#: reasonable thing to do to one.
MAX_SYMBOLS = 60


def _age_hours(created_at: Any, *, now: float) -> float:
    """Hours since an Alpaca timestamp, or infinity if it cannot be read."""
    text = str(created_at or "").strip()
    if not text:
        return float("inf")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        when = dt.datetime.fromisoformat(text)
    except ValueError:
        return float("inf")
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    # Negative ages happen: the venue's clock and this machine's disagree by
    # seconds, and a story filed "in the future" would otherwise decay to more
    # than full weight. Clamped at zero rather than discarded.
    return max(0.0, (now - when.timestamp()) / 3600.0)


def to_articles(rows: Iterable[dict[str, Any]], *,
                now: float | None = None) -> list[Article]:
    """Alpaca news rows as scoreable articles."""
    at = time.time() if now is None else now
    out: list[Article] = []
    for row in rows:
        headline = str(row.get("headline") or "").strip()
        if not headline:
            continue
        # Two shapes reach here. The RSS source has already turned an RFC 822
        # date into an age, because only it knows that its dates are RFC 822;
        # the venue source carries an ISO timestamp. Whichever is present is
        # used, and a row carrying neither is infinitely old rather than new.
        if row.get("age_hours") is not None:
            try:
                age = max(0.0, float(row["age_hours"]))
            except (TypeError, ValueError):
                age = float("inf")
        else:
            age = _age_hours(row.get("created_at"), now=at)
        out.append(Article(
            headline=headline,
            age_hours=age,
            summary=str(row.get("summary") or "").strip(),
            source=str(row.get("source") or "").strip(),
        ))
    return out


class NewsSource(Protocol):
    """Anything that can answer "what has been written about these symbols".

    A protocol rather than a base class so that a test can pass a plain
    function, and so that the desk is not coupled to either implementation.
    """

    label: str

    async def __call__(self, symbols: list[str], *, now: float
                       ) -> dict[str, list[dict[str, Any]]]:
        ...


def yahoo_source(*, transport: Any = None) -> NewsSource:
    """The default. Yahoo Finance's public per-symbol RSS."""

    async def fetch(symbols: list[str], *, now: float
                    ) -> dict[str, list[dict[str, Any]]]:
        return await yahoo.headlines(symbols, now=now, transport=transport)

    fetch.label = "yahoo"                                   # type: ignore[attr-defined]
    return fetch                                            # type: ignore[return-value]


def alpaca_source(client: Any) -> NewsSource:
    """The venue's own news endpoint, for anyone who would rather use it.

    Kept working and tested rather than deleted: it batches fifty symbols into
    one request where RSS needs fifty, which is the right trade for an account
    whose data plan includes it and whose rate limit has room.
    """

    async def fetch(symbols: list[str], *, now: float
                    ) -> dict[str, list[dict[str, Any]]]:
        if client is None:
            return {}
        start = (dt.datetime.now(tz=dt.timezone.utc)
                 - dt.timedelta(hours=LOOKBACK_HOURS))
        return await client.news(symbols, start=start)

    fetch.label = "alpaca"                                  # type: ignore[attr-defined]
    return fetch                                            # type: ignore[return-value]


class NewsDesk:
    """The scored news view, refreshed on a slow interval.

    Never raises into the trading loop. The worst outcome of a broken news
    desk is that every symbol reads "no news", which is the same state the
    program is in before its first refresh and trades perfectly well in.
    """

    def __init__(self, *, refresh_seconds: float = REFRESH_SECONDS,
                 source: "NewsSource | None" = None) -> None:
        self.refresh_seconds = refresh_seconds
        #: Where headlines come from. Yahoo's public RSS by default -- it needs
        #: no key, so news works before a credential is attached, and it does
        #: not spend the venue rate limit the scanner needs for prices. The
        #: venue's own endpoint is still available through
        #: :func:`alpaca_source` for anyone who would rather use it.
        self.source: NewsSource = source or yahoo_source()
        self.source_name = getattr(self.source, "label", "yahoo")
        self.scored: dict[str, Sentiment] = {}
        self.refreshed_at = 0.0
        self.last_error = ""
        self.articles_seen = 0
        self.symbols_covered = 0
        #: Consecutive refreshes that asked about symbols and got nothing back
        #: at all. See :meth:`feed_looks_broken`.
        self.silent_refreshes = 0
        self.symbols_asked = 0
        #: Set when the operator has switched the factor off. Kept here rather
        #: than read from a config on every decision so that turning it off
        #: takes effect at once and cannot half-apply.
        self.enabled = True

    def due(self, *, now: float | None = None) -> bool:
        at = time.time() if now is None else now
        return at - self.refreshed_at >= self.refresh_seconds

    def sentiment(self, symbol: str) -> Sentiment:
        """The current reading for a symbol. Always answers."""
        if not self.enabled:
            return Sentiment(symbol=symbol, covered=False,
                             unavailable="the news factor is switched off")
        found = self.scored.get(symbol)
        if found is not None:
            return found
        if not self.refreshed_at:
            return Sentiment(symbol=symbol, covered=False,
                             unavailable="no news has been read yet")
        return Sentiment(symbol=symbol, covered=False)

    async def refresh(self, symbols: list[str], *,
                      now: float | None = None) -> None:
        """Pull and score headlines for up to :data:`MAX_SYMBOLS` symbols.

        Takes no venue client. The default source is a public feed that needs
        no credential, which is the point: news is readable on first launch,
        before a key has been attached, and it never competes with the scanner
        for the venue's rate limit.
        """
        at = time.time() if now is None else now
        if not symbols or not self.enabled:
            self.refreshed_at = at
            return
        wanted = list(dict.fromkeys(symbols))[:MAX_SYMBOLS]
        try:
            grouped = await self.source(wanted, now=at)
        except Exception as exc:                            # noqa: BLE001
            # Deliberately broad. This runs inside the trading loop, and there
            # is no failure of a news fetch that is worth stopping a scan for.
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("news refresh failed: %s", self.last_error)
            self.refreshed_at = at
            return

        scored: dict[str, Sentiment] = {}
        articles = 0
        for symbol in wanted:
            rows = grouped.get(symbol) or []
            articles += len(rows)
            scored[symbol] = evaluate(symbol, to_articles(rows, now=at))
        self.scored = scored
        self.articles_seen = articles
        self.symbols_asked = len(wanted)
        self.symbols_covered = sum(1 for s in scored.values() if s.covered)
        # A refresh that asked and heard nothing whatsoever is recorded, not
        # shrugged off. See feed_looks_broken for why that matters here more
        # than it would for most sources.
        self.silent_refreshes = 0 if articles else self.silent_refreshes + 1
        self.refreshed_at = at
        self.last_error = ""

    #: How many symbols must be asked before silence means anything.
    #:
    #: Ten, against a seven-day lookback, holding the highest-turnover names
    #: the scanner has. A week in which not one of them produced a single
    #: headline does not happen.
    SILENCE_NEEDS_SYMBOLS = 10

    #: How many consecutive silent refreshes before saying so.
    #:
    #: Two, which is twenty minutes. One would let a transient network fault
    #: accuse the feed.
    SILENCE_NEEDS_REFRESHES = 2

    def feed_looks_broken(self) -> str:
        """Whether the source has stopped answering, in as many words.

        This exists because the default source is a public RSS feed rather
        than a documented API: it can change or disappear without notice, and
        when it does the symptom is *no coverage* -- which is indistinguishable
        from a quiet news week unless something counts. Nothing else in this
        program needs a check like this, and this source does.

        The cost of being wrong is low in both directions: a false alarm is a
        line of text, and a missed one costs a factor that only ever moved
        position size by a fifth.
        """
        if not self.enabled or self.last_error:
            return ""
        if (self.symbols_asked >= self.SILENCE_NEEDS_SYMBOLS
                and self.silent_refreshes >= self.SILENCE_NEEDS_REFRESHES):
            return (f"{self.source_name} returned no headlines at all for "
                    f"{self.symbols_asked} symbols, {self.silent_refreshes} "
                    f"refreshes running. The feed has most likely changed or "
                    f"is blocking this machine. Positions are being sized "
                    f"exactly as they would be with the factor switched off.")
        return ""

    def panel(self) -> dict[str, Any]:
        """What the UI shows about the desk itself, not about one symbol."""
        leaders = sorted((s for s in self.scored.values() if s.covered),
                         key=lambda s: -abs(s.score))[:6]
        return {
            "enabled": self.enabled,
            "source": self.source_name,
            # Sent rather than hard-coded in the page: the cap is a property of
            # the factor, and a UI that states its own number would go on
            # stating it after the factor's changed.
            "max_tilt_pct": round(MAX_TILT * 100),
            "age": (time.time() - self.refreshed_at) if self.refreshed_at else None,
            "articles": self.articles_seen,
            "covered": self.symbols_covered,
            "scored": len(self.scored),
            "last_error": self.last_error,
            "broken": self.feed_looks_broken(),
            "leaders": [{"symbol": s.symbol, **s.as_dict()} for s in leaders],
        }
