"""The brief's picture, and the Sunday summary.

The image is drawn without Pillow -- the Windows build excludes it -- so the
PNG writer is checked against Pillow here, where it is available as a test
dependency. The rest: the chart shows what it claims, the font covers every
character the chart prints, the photo goes as a real multipart upload, and
the weekly summary goes once and cannot take the daily brief down with it.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import time

import httpx
import numpy as np
import pytest

from imperium.notify import chart_png as cp
from imperium.notify import daily
from imperium.notify.telegram import Notifier


def _points(n=31, drift=0.003, seed=2):
    import random

    rng = random.Random(seed)
    v, out, now = 1000.0, [], time.time()
    for i in range(n):
        v *= math.exp(rng.gauss(drift, 0.01))
        out.append((now - 86_400 * (n - 1 - i), v))
    return out


def _decode(png: bytes):
    PIL = pytest.importorskip("PIL.Image")
    return np.asarray(PIL.open(io.BytesIO(png)).convert("RGB"))


# -- the image ------------------------------------------------------------------


def test_the_png_writer_produces_a_file_pillow_reads_back_exactly():
    rgb = (np.arange(4 * 6 * 3) % 256).astype(np.uint8).reshape(4, 6, 3)
    assert (_decode(cp.png(rgb)) == rgb).all()


def test_the_chart_is_drawn_at_its_stated_size():
    img = _decode(cp.render_equity(_points(), title="EQUITY 30 DAYS",
                                   width=960, height=520))
    assert img.shape == (520, 960, 3)


def test_nothing_is_drawn_from_fewer_than_two_good_readings():
    now = time.time()
    assert cp.render_equity([], title="X") is None
    assert cp.render_equity([(now, 100.0)], title="X") is None
    assert cp.render_equity([(now, 100.0), (now + 1, float("nan")),
                             (now + 2, -5.0)], title="X") is None


def test_the_change_is_coloured_by_its_direction():
    """The header's change reads green on a rising month and red on a
    falling one -- the one place the picture uses colour for meaning."""
    def share(img, color):
        region = img[40:100, 480:, :].astype(int)
        close = np.abs(region - np.array(color)).sum(axis=2) < 60
        return close.mean()

    up = _decode(cp.render_equity(_points(drift=0.01), title="UP"))
    down = _decode(cp.render_equity(_points(drift=-0.01), title="DOWN"))
    assert share(up, cp.GOOD) > 0.005 and share(up, cp.BAD) == 0
    assert share(down, cp.BAD) > 0.005 and share(down, cp.GOOD) == 0


def test_the_line_ends_at_the_last_reading():
    """The live end is marked with a dot at the right edge of the plot, at
    the height of the last value: bright pixels exist there."""
    pts = _points()
    img = _decode(cp.render_equity(pts, title="X", width=960, height=520))
    right = 960 - 128
    column = img[100:390, right - 6:right + 6].astype(int).sum(axis=2)
    assert column.max() > 600, "no line end at the right edge of the plot"


def test_the_font_has_every_character_the_chart_prints():
    printed = set()
    for v in (0.5, 12.34, -1234.5, 99999.0, 1_234_567.0):
        printed |= set(cp._money(v, True)) | set(cp._axis(v))
    for month in range(1, 13):
        printed |= set(cp._date(dt.datetime(2026, month, 28,
                                            tzinfo=dt.timezone.utc).timestamp()))
    printed |= set("IMPERIUM · EQUITY 30 DAYS TODAY THIS WEEK DRAWDOWN")
    printed |= set(f"{-0.0346:+.2%}{-0.078:.1%}()")
    missing = {c for c in printed if c.upper() not in cp._GLYPHS}
    assert not missing, f"no glyph for {sorted(missing)}"


def test_a_weeks_worth_of_five_minute_points_draws_quickly():
    now = time.time()
    pts = [(now - 7 * 86_400 + i * 300, 1000 + math.sin(i / 50) * 5)
           for i in range(2016)]
    started = time.perf_counter()
    assert cp.render_equity(pts, title="EQUITY THIS WEEK")
    assert time.perf_counter() - started < 3.0


# -- the upload -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_photo_goes_as_a_multipart_upload_to_the_linked_chat():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["type"] = request.headers["content-type"]
        seen["body"] = request.content
        return httpx.Response(200, json={"ok": True, "result": {}})

    n = Notifier("123:ABC", "42", transport=httpx.MockTransport(handler))
    image = cp.render_equity(_points(), title="X")
    assert await n.send_photo(image, "caption here")
    assert seen["path"].endswith("/sendPhoto")
    assert seen["type"].startswith("multipart/form-data")
    assert b'name="chat_id"' in seen["body"] and b"42" in seen["body"]
    assert b'name="photo"; filename="equity.png"' in seen["body"]
    assert image[:64] in seen["body"]
    assert b"caption here" in seen["body"]


@pytest.mark.asyncio
async def test_a_refused_photo_is_recorded_not_raised():
    def handler(request):
        return httpx.Response(400, json={"ok": False, "description": "bad photo"})

    n = Notifier("123:ABC", "42", transport=httpx.MockTransport(handler))
    assert not await n.send_photo(b"\x89PNG", "c")
    assert n.failed == 1 and "bad photo" in n.last_error
    assert not await Notifier().send_photo(b"x")          # not linked


# -- the weekly summary ----------------------------------------------------------


def _week(**kw):
    base = dict(label="week to Sun 27 Sep", start_equity=1000.0,
                end_equity=1042.5, max_drawdown=-0.031, fills=9,
                strategies=(("trend", 30.0), ("overnight", -4.0)),
                closes=(daily.WeekTrade("NVDA", "trend", 18.2, 1_790_000_000),
                        daily.WeekTrade("AAPL", "trend", 7.5, 1_790_100_000),
                        daily.WeekTrade("SPY", "overnight", -3.1, 1_790_200_000),
                        daily.WeekTrade("BTC/USD", "cross_section", -0.9,
                                        1_790_300_000)))
    base.update(kw)
    return daily.Week(**base)


def test_the_weekly_summary_names_the_result_the_strategies_and_the_trades():
    text = daily.build_weekly(_week())
    assert text.startswith("📅 WEEKLY SUMMARY — week to Sun 27 Sep")
    assert "Week +$42.50  (+4.25%)" in text
    assert "Deepest fall from a high: -3.1%" in text
    assert "9 fills · 4 closed · 50% won" in text
    assert "Multi-day trend" in text and "Overnight drift" in text
    best = text.index("Best trades")
    worst = text.index("Worst trades")
    assert text.index("NVDA") > best and text.index("NVDA") < worst
    assert text.index("SPY") > worst
    assert text.index("SPY") < text.index("BTC/USD"), "worst first"


def test_a_quiet_week_says_so_rather_than_printing_empty_sections():
    text = daily.build_weekly(_week(closes=(), strategies=(), fills=0,
                                    start_equity=0.0))
    assert "no equity history for the whole week yet" in text
    assert "Nothing closed this week." in text
    assert "Best trades" not in text and "By strategy" not in text


# -- in the session ---------------------------------------------------------------


class _Notifier:
    linked = True

    def __init__(self, photo_fails=False):
        self.sent, self.photos, self.photo_fails = [], [], photo_fails

    async def send(self, text):
        self.sent.append(text)
        return True

    async def send_photo(self, image, caption=""):
        if self.photo_fails:
            raise RuntimeError("the picture broke")
        self.photos.append((image, caption))
        return True


def _evening(weekday_date):
    from imperium.session import TradingSession

    session = TradingSession()
    session.running = True
    session.notifier = _Notifier()
    y, m, d = weekday_date
    session.market_clock.timestamp = dt.datetime(y, m, d, 21, 30,
                                                 tzinfo=dt.timezone.utc)
    history = session.equity_history()
    now = time.time()
    for i, (t, v) in enumerate(_points(n=10)):
        history.record(now - 86_400 * (9 - i), v, f"2026-09-{14 + i:02d}")
    return session


@pytest.mark.asyncio
async def test_the_brief_is_followed_by_its_chart():
    session = _evening((2026, 9, 23))                    # a Wednesday
    await session._maybe_send_daily_brief()
    notifier = session.notifier
    assert len(notifier.sent) == 1 and "DAILY BRIEF" in notifier.sent[0]
    assert len(notifier.photos) == 1
    image, caption = notifier.photos[0]
    assert image.startswith(b"\x89PNG") and caption.startswith("Equity 30 Days")


@pytest.mark.asyncio
async def test_sunday_adds_the_weekly_summary_once():
    session = _evening((2026, 9, 27))                    # a Sunday
    await session._maybe_send_daily_brief()
    notifier = session.notifier
    assert len(notifier.sent) == 2
    assert "DAILY BRIEF" in notifier.sent[0]
    assert notifier.sent[1].startswith("📅 WEEKLY SUMMARY")
    assert len(notifier.photos) == 2
    # A restart the same evening sends neither again.
    session._brief_sent_day = ""
    assert not await session._send_weekly("2026-09-27")
    assert session._weekly_sent_day == "2026-09-27"


@pytest.mark.asyncio
async def test_a_broken_picture_never_costs_the_brief():
    session = _evening((2026, 9, 27))
    session.notifier = _Notifier(photo_fails=True)
    await session._maybe_send_daily_brief()               # must not raise
    assert "DAILY BRIEF" in session.notifier.sent[0]


def test_the_week_is_read_from_the_journal_and_the_history(monkeypatch):
    from decimal import Decimal

    from imperium.execution import journal as jr
    from imperium.execution.broker import Fill, Mode
    from imperium.session import TradingSession

    session = TradingSession()
    now = time.time()
    mode = Mode(session.broker.mode.value)
    old = Fill("OLD", "SELL", Decimal("1"), Decimal("10"), now - 9 * 86_400,
               mode, "o", True, strategy="trend", realised=99.0)
    new = Fill("NVDA", "SELL", Decimal("1"), Decimal("10"), now - 3600,
               mode, "n", True, strategy="trend", realised=4.0)
    entry = Fill("AAPL", "BUY", Decimal("1"), Decimal("10"), now - 1800,
                 mode, "e", True, strategy="trend")
    session.trade_journal.append(jr.row_from_fill(f) for f in (old, new, entry))
    history = session.equity_history()
    history.record(now - 5 * 86_400, 1000.0, "2026-09-22")
    history.record(now, 1010.0, "2026-09-27")
    week = session._week("2026-09-27", now)
    assert week.fills == 2
    assert [t.symbol for t in week.closes] == ["NVDA"]
    assert (week.start_equity, week.end_equity) == (1000.0, 1010.0)
