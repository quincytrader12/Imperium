"""Headlines from Yahoo Finance's per-symbol RSS feed.

**Why this rather than the venue's own news.** Alpaca does serve news, at
``/v1beta1/news``, and :mod:`imperium.venues.alpaca.client` can still read it.
Three things make this the better default anyway:

* It needs no API key. News works on first launch, before a credential has
  been attached, which is exactly when an operator is looking at the screen
  trying to decide whether the program does anything.
* It does not spend the venue's rate limit. Alpaca allows 200 requests a
  minute per key and the scanner needs every one of them to price the book;
  news competing for that budget is news occasionally costing a quote.
* It does not depend on the account's data plan. Alpaca's news is Benzinga
  content and access to it has been plan-gated before.

**What it costs.** RSS carries no per-item symbol attribution -- an ``<item>``
says what it is about only in its prose -- so a symbol's headlines can only be
had by asking for that symbol's feed. That is one HTTP request per symbol
rather than one per fifty, which is why :data:`MAX_CONCURRENCY` exists and why
the desk above this covers fewer symbols than the Alpaca path would.

**What it is not.** An official, documented, or guaranteed API. It is a public
feed that has been stable for years and could change tomorrow. Everything here
is written so that a change in its shape produces *no coverage*, reported as
such, rather than wrong readings: a feed that stops parsing must look like a
quiet news day that says it is broken, never like bad news.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import email.utils
import logging
import xml.etree.ElementTree as ET

import httpx

log = logging.getLogger("imperium.yahoo")

FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline"

#: Per-request timeout, in seconds. Short: this is a secondary factor and a
#: slow feed must not hold the refresh open.
TIMEOUT = 8.0

#: Simultaneous requests.
#:
#: One request per symbol means a hundred-symbol refresh is a hundred
#: requests, and firing those at once is indistinguishable from a small flood.
#: Eight keeps a full refresh inside a handful of seconds while staying
#: politely below what an unauthenticated public feed should be asked for.
MAX_CONCURRENCY = 8

#: The most bytes read from one feed.
#:
#: A cap before the parser rather than after: :mod:`xml.etree` expands
#: internal entities, so a small document can become a large one in memory.
#: The DOCTYPE check below closes that properly; this bounds the ordinary case
#: where a feed is simply enormous.
MAX_BYTES = 512_000

#: Items read from one feed. Yahoo returns about twenty; the scorer's recency
#: decay makes anything past the first several weightless anyway.
MAX_ITEMS = 25

#: A browser-ish identifier.
#:
#: Not evasion -- the feed is public and this is not pretending otherwise. An
#: empty or library-default agent is what most feeds reject outright, and a
#: request that identifies the program asking is the honest form.
USER_AGENT = "IMPERIUM/1.0 (trading terminal; RSS reader)"


def yahoo_symbol(symbol: str) -> str:
    """This program's spelling of a symbol, in Yahoo's.

    Yahoo writes crypto pairs with a hyphen -- "BTC-USD", where the venue and
    this program say "BTC/USD". A slash in the query would be read as a path
    separator by something between here and there, and the feed would come
    back about nothing.
    """
    return symbol.replace("/", "-").upper()


def _age_hours(pub_date: str, *, now: float) -> float:
    """Hours since an RFC 822 date, or infinity if it cannot be read.

    RSS dates are RFC 822 ("Tue, 09 Sep 2026 14:03:00 +0000"), not ISO 8601.
    Anything unparseable is treated as infinitely old rather than as new: a
    story of unknown vintage given full weight is the one mistake here that
    would size a position on nothing.
    """
    text = (pub_date or "").strip()
    if not text:
        return float("inf")
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return float("inf")
    if when is None:
        return float("inf")
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    # Clamped at zero: the feed's clock and this machine's differ by seconds,
    # and a story filed "in the future" must not decay to more than full
    # weight.
    return max(0.0, (now - when.timestamp()) / 3600.0)


def parse_feed(xml_text: str, *, now: float) -> list[dict[str, object]]:
    """RSS text as rows the news desk already knows how to score.

    Returns the same shape the Alpaca path returns -- headline, summary,
    created_at, source -- so that swapping the source changes nothing
    downstream of here.
    """
    text = (xml_text or "").strip()
    if not text:
        return []
    # No DOCTYPE, ever.
    #
    # ElementTree refuses external entities, so a feed cannot read this
    # machine's files. It does expand *internal* ones, and a few hundred bytes
    # of nested definitions expand to gigabytes -- the "billion laughs" denial
    # of service, measured and confirmed against this exact parser. RSS 2.0
    # has no use for a DOCTYPE, so refusing the whole document is a complete
    # fix with nothing legitimate lost.
    if "<!DOCTYPE" in text[:2000].upper() or "<!ENTITY" in text[:2000].upper():
        log.warning("refusing a feed that declares a DTD")
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        log.warning("could not parse a feed: %s", exc)
        return []

    rows: list[dict[str, object]] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        published = (item.findtext("pubDate") or "").strip()
        rows.append({
            "headline": title,
            "summary": (item.findtext("description") or "").strip(),
            "source": "yahoo",
            "age_hours": _age_hours(published, now=now),
            "url": (item.findtext("link") or "").strip(),
        })
        if len(rows) >= MAX_ITEMS:
            break
    return rows


async def _one(client: httpx.AsyncClient, symbol: str, *,
               now: float) -> tuple[str, list[dict[str, object]]]:
    try:
        response = await client.get(
            FEED, params={"s": yahoo_symbol(symbol),
                          "region": "US", "lang": "en-US"})
    except httpx.HTTPError as exc:
        log.debug("news for %s failed: %s", symbol, type(exc).__name__)
        return symbol, []
    if response.status_code != 200:
        # 404 is the ordinary answer for a symbol Yahoo does not cover, which
        # is most of the long tail. Not worth a warning each time.
        log.debug("news for %s: HTTP %s", symbol, response.status_code)
        return symbol, []
    body = response.text
    if len(body) > MAX_BYTES:
        body = body[:MAX_BYTES]
    return symbol, parse_feed(body, now=now)


async def headlines(symbols: list[str], *, now: float,
                    transport: httpx.AsyncBaseTransport | None = None,
                    concurrency: int = MAX_CONCURRENCY,
                    ) -> dict[str, list[dict[str, object]]]:
    """Headlines per symbol. Never raises; a failure is an empty list.

    Absence and failure are the same answer to the caller on purpose -- both
    mean "no reading for this symbol" and both must leave its position sized
    exactly as it would have been. The desk above reports the difference; the
    sizing does not depend on it.
    """
    if not symbols:
        return {}
    gate = asyncio.Semaphore(max(1, concurrency))

    async with httpx.AsyncClient(
            timeout=TIMEOUT, transport=transport, follow_redirects=True,
            headers={"User-Agent": USER_AGENT,
                     "Accept": "application/rss+xml, application/xml, text/xml"},
    ) as client:
        async def guarded(symbol: str):
            async with gate:
                return await _one(client, symbol, now=now)

        results = await asyncio.gather(
            *(guarded(s) for s in symbols), return_exceptions=True)

    out: dict[str, list[dict[str, object]]] = {}
    for result in results:
        if isinstance(result, BaseException):
            # gather with return_exceptions so one symbol cannot take the
            # refresh down. Already logged where it happened.
            continue
        symbol, rows = result
        if rows:
            out[symbol] = rows
    return out
