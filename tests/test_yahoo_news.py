"""The Yahoo RSS news source: its request, its parser, and its refusals.

This source cannot be verified against the live feed from the machine it was
written on, so everything it assumes about the feed's shape is stated as a
test here. If Yahoo changes the feed, these are the assertions that will still
pass while production quietly reads nothing -- so the parser is written to
produce *no coverage* on anything it does not recognise, and the desk above
reports that, rather than inventing a reading.
"""

from __future__ import annotations

import math
import time

import httpx
import pytest

from imperium.execution.newsdesk import NewsDesk, to_articles, yahoo_source
from imperium.venues import yahoo

NOW = time.time()

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Yahoo! Finance: AAPL News</title>
  <item>
    <title>Apple Surges On Record Earnings Beat</title>
    <description>The company also raised full-year guidance.</description>
    <link>https://finance.yahoo.com/news/one</link>
    <pubDate>Sun, 13 Sep 2026 12:00:00 +0000</pubDate>
  </item>
  <item>
    <title>Analyst Downgrades Apple, Cuts Price Target</title>
    <description>Citing slowing demand.</description>
    <link>https://finance.yahoo.com/news/two</link>
    <pubDate>Sat, 12 Sep 2026 09:30:00 GMT</pubDate>
  </item>
</channel></rss>"""


def _at(hours_after: str = "Sun, 13 Sep 2026 14:00:00 +0000") -> float:
    import email.utils
    return email.utils.parsedate_to_datetime(hours_after).timestamp()


# -- the parser ----------------------------------------------------------

def test_the_parser_reads_the_fields_the_scorer_needs():
    rows = yahoo.parse_feed(FEED, now=_at())
    assert len(rows) == 2
    assert rows[0]["headline"] == "Apple Surges On Record Earnings Beat"
    assert "guidance" in str(rows[0]["summary"])
    assert rows[0]["source"] == "yahoo"
    assert rows[0]["age_hours"] == pytest.approx(2.0, abs=0.01)


def test_rfc_822_dates_are_read_rather_than_iso_ones():
    """Prevents the mistake that makes every headline weightless. RSS dates
    are RFC 822; the venue's API uses ISO 8601. Parsing one as the other
    yields no date, every article is infinitely old, and the desk reports no
    coverage for a feed that answered perfectly."""
    rows = yahoo.parse_feed(FEED, now=_at())
    assert all(math.isfinite(float(r["age_hours"])) for r in rows)
    # The second item uses "GMT" rather than "+0000" -- both occur in the wild.
    assert rows[1]["age_hours"] == pytest.approx(28.5, abs=0.01)


def test_an_unreadable_date_is_infinitely_old_and_never_brand_new():
    feed = FEED.replace("Sun, 13 Sep 2026 12:00:00 +0000", "sometime recently")
    rows = yahoo.parse_feed(feed, now=_at())
    assert math.isinf(float(rows[0]["age_hours"]))


def test_a_story_filed_in_the_future_does_not_exceed_full_weight():
    rows = yahoo.parse_feed(FEED, now=_at("Sun, 13 Sep 2026 06:00:00 +0000"))
    assert rows[0]["age_hours"] == 0.0


@pytest.mark.parametrize("body", [
    "", "   ", "not xml at all", "<rss><channel>", "<html><body>404</body></html>",
])
def test_anything_unparseable_is_no_coverage_rather_than_an_exception(body):
    """Prevents: a feed outage or a redirect to an HTML error page reaching
    the trading loop as a traceback."""
    assert yahoo.parse_feed(body, now=NOW) == []


def test_an_item_with_no_headline_is_skipped():
    feed = FEED.replace("<title>Apple Surges On Record Earnings Beat</title>",
                        "<title></title>")
    rows = yahoo.parse_feed(feed, now=_at())
    assert len(rows) == 1


def test_the_parser_refuses_a_feed_that_declares_a_dtd():
    """Prevents a denial of service that was measured, not assumed.

    ElementTree refuses *external* entities, so a hostile feed cannot read
    this machine's files. It does expand *internal* ones: a few hundred bytes
    of nested definitions become gigabytes in memory. RSS 2.0 has no use for a
    DOCTYPE, so the whole document is refused and nothing legitimate is lost.
    """
    bomb = ('<?xml version="1.0"?>\n<!DOCTYPE lolz [\n'
            ' <!ENTITY lol "lol">\n'
            ' <!ENTITY lol2 "' + "&lol;" * 10 + '">\n'
            ' <!ENTITY lol3 "' + "&lol2;" * 10 + '">\n'
            ' <!ENTITY lol4 "' + "&lol3;" * 10 + '">\n'
            ']>\n<rss><channel><item><title>&lol4;</title>'
            '<pubDate>Sun, 13 Sep 2026 12:00:00 +0000</pubDate>'
            '</item></channel></rss>')
    assert yahoo.parse_feed(bomb, now=NOW) == []


def test_the_parser_caps_how_many_items_it_keeps():
    item = ("<item><title>Shares surge</title>"
            "<pubDate>Sun, 13 Sep 2026 12:00:00 +0000</pubDate></item>")
    feed = "<rss><channel>" + item * 200 + "</channel></rss>"
    assert len(yahoo.parse_feed(feed, now=_at())) == yahoo.MAX_ITEMS


# -- the symbol spelling -------------------------------------------------

@pytest.mark.parametrize("ours,theirs", [
    ("AAPL", "AAPL"),
    ("BTC/USD", "BTC-USD"),
    ("ETH/USD", "ETH-USD"),
    ("brk.b", "BRK.B"),
])
def test_crypto_pairs_are_spelled_the_way_the_feed_spells_them(ours, theirs):
    """Prevents the coins -- the one asset class a small account can trade
    around the clock -- being the ones that never get a reading. A slash in
    the query is a path separator to something between here and there."""
    assert yahoo.yahoo_symbol(ours) == theirs


# -- the request ---------------------------------------------------------

@pytest.mark.asyncio
async def test_the_request_asks_the_feed_for_the_right_symbol():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=FEED,
                              headers={"content-type": "application/xml"})

    out = await yahoo.headlines(["AAPL", "BTC/USD"], now=_at(),
                                transport=httpx.MockTransport(handler))
    assert set(out) == {"AAPL", "BTC/USD"}
    asked = sorted(dict(r.url.params)["s"] for r in seen)
    assert asked == ["AAPL", "BTC-USD"]
    assert seen[0].url.path.endswith("/rss/2.0/headline")
    assert "IMPERIUM" in seen[0].headers["user-agent"]


@pytest.mark.asyncio
async def test_a_symbol_the_feed_does_not_cover_is_absent_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found")

    assert await yahoo.headlines(["NOPE"], now=NOW,
                                 transport=httpx.MockTransport(handler)) == {}


@pytest.mark.asyncio
async def test_one_failing_symbol_does_not_lose_the_others():
    """Prevents: a single bad symbol emptying a whole refresh."""
    def handler(request: httpx.Request) -> httpx.Response:
        if dict(request.url.params)["s"] == "BAD":
            raise httpx.ConnectError("nope")
        return httpx.Response(200, text=FEED)

    out = await yahoo.headlines(["AAPL", "BAD", "MSFT"], now=_at(),
                                transport=httpx.MockTransport(handler))
    assert set(out) == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_the_source_never_opens_more_connections_than_its_limit():
    """Prevents: a sixty-symbol refresh arriving at a public feed as sixty
    simultaneous requests, which is not a reasonable thing to do to one."""
    import asyncio
    live = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return httpx.Response(200, text=FEED)

    await yahoo.headlines([f"S{i}" for i in range(40)], now=_at(),
                          transport=httpx.MockTransport(handler),
                          concurrency=4)
    assert peak <= 4, f"{peak} requests were in flight at once"


# -- end to end through the desk -----------------------------------------

@pytest.mark.asyncio
async def test_the_desk_scores_a_real_shaped_feed_end_to_end():
    """The whole path: RSS text in, a bounded tilt out."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=FEED)

    desk = NewsDesk(source=yahoo_source(transport=httpx.MockTransport(handler)))
    await desk.refresh(["AAPL"], now=_at())
    reading = desk.sentiment("AAPL")
    assert reading.covered
    assert reading.articles == 2
    # One story up, one down, the fresher one positive: a small positive net.
    assert 0.0 < reading.score < 0.5
    tilted = reading.tilt(0.20)
    assert 0.20 < tilted <= 0.20 * 1.2 + 1e-12


# -- telling a broken feed from a quiet week -----------------------------

@pytest.mark.asyncio
async def test_a_silent_feed_is_eventually_called_out_rather_than_shrugged_off():
    """Prevents the specific risk of depending on an undocumented feed.

    If Yahoo changes or blocks this, the symptom is no coverage -- which is
    indistinguishable from a quiet news week unless something counts. Asking
    about dozens of the highest-turnover names in the market and hearing
    nothing for twenty minutes is not a quiet week.
    """
    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<rss><channel></channel></rss>")

    desk = NewsDesk(source=yahoo_source(transport=httpx.MockTransport(empty)))
    symbols = [f"S{i}" for i in range(20)]

    await desk.refresh(symbols, now=NOW)
    assert desk.feed_looks_broken() == "", "one silent refresh is not proof"

    await desk.refresh(symbols, now=NOW)
    said = desk.feed_looks_broken()
    assert said
    assert "yahoo" in said
    assert "sized exactly as they would be" in said


@pytest.mark.asyncio
async def test_a_few_quiet_symbols_are_not_called_a_broken_feed():
    """Prevents the false alarm. Most symbols have no news most days, and a
    handful of quiet ones says nothing about the feed."""
    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<rss><channel></channel></rss>")

    desk = NewsDesk(source=yahoo_source(transport=httpx.MockTransport(empty)))
    for _ in range(5):
        await desk.refresh(["AAPL", "MSFT"], now=NOW)
    assert desk.feed_looks_broken() == ""


@pytest.mark.asyncio
async def test_one_answering_symbol_clears_the_silence():
    def handler(request: httpx.Request) -> httpx.Response:
        if dict(request.url.params)["s"] == "S0":
            return httpx.Response(200, text=FEED)
        return httpx.Response(200, text="<rss><channel></channel></rss>")

    desk = NewsDesk(source=yahoo_source(transport=httpx.MockTransport(handler)))
    symbols = [f"S{i}" for i in range(20)]
    for _ in range(3):
        await desk.refresh(symbols, now=_at())
    assert desk.silent_refreshes == 0
    assert desk.feed_looks_broken() == ""


# -- the news refresh must never stall the book --------------------------

@pytest.mark.asyncio
async def test_the_trading_loop_does_not_wait_for_the_news_feed():
    """Prevents a stranger's latency sitting on the book's critical path.

    A refresh is up to sixty requests to a third party this program does not
    control. The first version of this awaited them inside the trading loop,
    and the symptom was the loop failing to rotate its cohort within two
    seconds -- intermittently, because it depended on how slow the feed was
    that minute. The same stall would have delayed stops and exits on a real
    book, and would have been almost impossible to attribute.

    Driven through the session's own kick, against a source that never
    answers: if the kick awaits, this test times out.
    """
    import asyncio
    from imperium.session import TradingSession

    started = asyncio.Event()

    async def never_answers(symbols, *, now):
        started.set()
        await asyncio.sleep(30)
        return {}

    never_answers.label = "hung"

    session = TradingSession()
    session.newsdesk = NewsDesk(source=never_answers)

    # The kick itself must return immediately, not in thirty seconds.
    session.kick_news()
    await asyncio.wait_for(started.wait(), timeout=2.0)
    assert session._news_task is not None and not session._news_task.done()

    # And a second kick must not pile a duplicate refresh on top of the first.
    first = session._news_task
    session.kick_news()
    assert session._news_task is first

    session._news_task.cancel()
    with pytest.raises((asyncio.CancelledError, Exception)):
        await session._news_task


@pytest.mark.asyncio
async def test_stopping_the_session_does_not_leave_a_refresh_running():
    """Prevents: a detached task outliving the book it belongs to and going on
    requesting headlines for a session nobody is trading."""
    import asyncio
    from imperium.session import TradingSession

    async def slow(symbols, *, now):
        await asyncio.sleep(30)
        return {}

    slow.label = "slow"

    session = TradingSession()
    session.newsdesk = NewsDesk(source=slow)
    # Not session.start(): that dials the live market-data websocket, which a
    # test has no business doing. The two things stop() needs to see are a
    # running session and a refresh in flight, and both are set here directly.
    session.keep_awake.acquire()
    session.running = True
    session.kick_news()
    assert session._news_task is not None
    await session.stop()
    assert session._news_task is None
