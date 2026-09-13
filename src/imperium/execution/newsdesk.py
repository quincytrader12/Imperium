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
from typing import Any, Iterable

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
MAX_SYMBOLS = 150


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
        out.append(Article(
            headline=headline,
            age_hours=_age_hours(row.get("created_at"), now=at),
            summary=str(row.get("summary") or "").strip(),
            source=str(row.get("source") or "").strip(),
        ))
    return out


class NewsDesk:
    """The scored news view, refreshed on a slow interval.

    Never raises into the trading loop. The worst outcome of a broken news
    desk is that every symbol reads "no news", which is the same state the
    program is in before its first refresh and trades perfectly well in.
    """

    def __init__(self, *, refresh_seconds: float = REFRESH_SECONDS) -> None:
        self.refresh_seconds = refresh_seconds
        self.scored: dict[str, Sentiment] = {}
        self.refreshed_at = 0.0
        self.last_error = ""
        self.articles_seen = 0
        self.symbols_covered = 0
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

    async def refresh(self, client: Any, symbols: list[str], *,
                      now: float | None = None) -> None:
        """Pull and score headlines for up to :data:`MAX_SYMBOLS` symbols."""
        at = time.time() if now is None else now
        if client is None or not symbols or not self.enabled:
            self.refreshed_at = at
            return
        wanted = list(dict.fromkeys(symbols))[:MAX_SYMBOLS]
        start = (dt.datetime.now(tz=dt.timezone.utc)
                 - dt.timedelta(hours=LOOKBACK_HOURS))
        try:
            grouped = await client.news(wanted, start=start)
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
        self.symbols_covered = sum(1 for s in scored.values() if s.covered)
        self.refreshed_at = at
        self.last_error = ""

    def panel(self) -> dict[str, Any]:
        """What the UI shows about the desk itself, not about one symbol."""
        leaders = sorted((s for s in self.scored.values() if s.covered),
                         key=lambda s: -abs(s.score))[:6]
        return {
            "enabled": self.enabled,
            # Sent rather than hard-coded in the page: the cap is a property of
            # the factor, and a UI that states its own number would go on
            # stating it after the factor's changed.
            "max_tilt_pct": round(MAX_TILT * 100),
            "age": (time.time() - self.refreshed_at) if self.refreshed_at else None,
            "articles": self.articles_seen,
            "covered": self.symbols_covered,
            "scored": len(self.scored),
            "last_error": self.last_error,
            "leaders": [{"symbol": s.symbol, **s.as_dict()} for s in leaders],
        }
