"""Attribution wired into the running session.

The accounting is tested on its own in test_attribution.py. These cover the
seams: that the one path that names a strategy does name it, that an exit made
after a restart still finds the strategy that opened the position, that an
empty book can never be written over a real one, and that the daily brief says
which strategy earned the day.
"""

from __future__ import annotations

import datetime as dt
import time

import pytest

from imperium.execution.attribution import UNATTRIBUTED
from imperium.execution.broker import PaperBroker
from imperium.execution.portfolio import Verdict
from imperium.session import TradingSession
from imperium.venues import registry


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    return tmp_path


def _paper_session() -> TradingSession:
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session._attribution_loaded = True
    return session


def _quote(session, symbol, price):
    q = session.feed.quote(symbol)
    q.last, q.updated_at = price, time.time()


async def _enter(session, symbol, weight, price, strategy):
    _quote(session, symbol, price)
    decision = session.engine(symbol).decision
    decision.symbol = symbol
    decision.verdict = Verdict.TRADING
    decision.target_weight = weight
    decision.strategy = strategy
    decision.hold = False
    decision.entry_order = ""
    await session._act_on(decision)
    session._attribute()


async def _exit_unnamed(session, symbol, price):
    """What the ratchet, the overnight exit and a retirement flatten do."""
    _quote(session, symbol, price)
    await session.broker.apply_target(symbol, 0.0, price,
                                      session.engine_equity())
    session._attribute()


def _row(session, name):
    rows = session._strategies_block()["rows"]
    return next(r for r in rows if r["strategy"] == name)


@pytest.mark.asyncio
async def test_an_entry_through_the_session_is_booked_to_its_strategy(home):
    session = _paper_session()
    await _enter(session, "AAPL", 0.10, 100.0, "trend")

    rows = session._strategies_block()["rows"]
    assert [r["strategy"] for r in rows] == ["trend"], (
        "the entry path did not name the strategy that decided it")
    assert session.broker.fills[-1].strategy == "trend"


@pytest.mark.asyncio
async def test_an_unnamed_exit_is_credited_to_the_strategy_that_opened_it(home):
    session = _paper_session()
    await _enter(session, "AAPL", 0.10, 100.0, "trend")
    await _exit_unnamed(session, "AAPL", 110.0)

    row = _row(session, "trend")
    assert row["round_trips"] == 1
    assert row["realised"] > 0, "the gain went somewhere other than trend"
    assert session.broker.fills[-1].strategy == "trend", (
        "the journal does not say whose exit it was")


@pytest.mark.asyncio
async def test_an_exit_after_a_restart_still_finds_its_strategy(home):
    """The terminal is restarted by a batch file whenever it exits. An owner
    kept only in memory would book every exit after a restart to nobody."""
    first = _paper_session()
    await _enter(first, "AAPL", 0.10, 100.0, "trend")

    second = _paper_session()
    second._load_attribution()
    # The broker is rebuilt on restart; the position comes back from it.
    second.broker = first.broker
    second.attribution._broker_id = id(first.broker)
    second.attribution._seen = first.broker.fills_total

    await _exit_unnamed(second, "AAPL", 110.0)
    assert _row(second, "trend")["round_trips"] == 1


@pytest.mark.asyncio
async def test_an_empty_book_is_never_written_over_a_real_one(home):
    """The first tick after start books and saves. A session that had not yet
    read the book from disk would erase every strategy's history with it."""
    real = _paper_session()
    await _enter(real, "AAPL", 0.10, 100.0, "trend")

    fresh = TradingSession()           # not loaded
    fresh._save_attribution()

    reread = _paper_session()
    reread._load_attribution()
    assert "trend" in reread.attribution.book_for("paper").records


@pytest.mark.asyncio
async def test_a_position_held_at_start_is_shown_as_unattributed(home):
    session = _paper_session()
    _quote(session, "MSFT", 300.0)
    # Present in the book before this program opened anything.
    await session.broker.apply_target("MSFT", 0.10, 300.0, 10_000.0)
    session.attribution._broker_id = id(session.broker)
    session.attribution._seen = session.broker.fills_total   # not seen as a fill
    session._attribute()

    assert _row(session, UNATTRIBUTED)["open"] == ["MSFT"]


def test_a_fault_in_attribution_does_not_stop_the_tick(home, monkeypatch):
    session = _paper_session()

    def boom(broker):
        raise RuntimeError("the ledger is broken")

    monkeypatch.setattr(session.attribution, "consume", boom)
    session._attribute()               # must not raise


@pytest.mark.asyncio
async def test_the_brief_says_which_strategy_earned_the_day(home):
    session = _paper_session()
    session.running = True
    sent: list[str] = []

    class _Notifier:
        linked = True

        async def send(self, text):
            sent.append(text)
            return True

    session.notifier = _Notifier()
    await _enter(session, "AAPL", 0.10, 100.0, "trend")
    await _exit_unnamed(session, "AAPL", 110.0)
    await _enter(session, "BTC/USD", 0.05, 50_000.0, "cross_section")
    await _exit_unnamed(session, "BTC/USD", 49_000.0)

    session.market_clock.timestamp = dt.datetime(2026, 9, 23, 21, 30,
                                                 tzinfo=dt.timezone.utc)
    await session._maybe_send_daily_brief()

    briefs = [t for t in sent if "DAILY BRIEF" in t]
    assert briefs, "no brief was sent"
    text = briefs[0]
    assert "By strategy" in text, text
    trend_line = next(l for l in text.splitlines() if "Multi-day trend" in l)
    crypto_line = next(l for l in text.splitlines() if "Crypto ranking" in l)
    assert trend_line.startswith("🟢"), trend_line
    assert crypto_line.startswith("🔴"), crypto_line
    assert text.index("Multi-day trend") < text.index("Crypto ranking"), (
        "the best strategy should be listed first")


@pytest.mark.asyncio
async def test_each_day_is_marked_once_for_the_allocator(home):
    session = _paper_session()
    session.running = True

    class _Notifier:
        linked = True

        async def send(self, text):
            return True

    session.notifier = _Notifier()
    await _enter(session, "AAPL", 0.10, 100.0, "trend")
    session.market_clock.timestamp = dt.datetime(2026, 9, 23, 21, 30,
                                                 tzinfo=dt.timezone.utc)
    for _ in range(3):
        await session._maybe_send_daily_brief()

    daily = session.attribution.book_for("paper").records["trend"].daily
    assert [d for d, _ in daily] == ["2026-09-23"]


@pytest.mark.asyncio
async def test_the_trading_tick_books_fills_by_itself(home):
    """Driven through the tick rather than by calling the booking directly:
    the property is that the loop does it, with nothing asking it to."""
    session = _paper_session()
    _quote(session, "AAPL", 100.0)
    await session.broker.apply_target("AAPL", 0.10, 100.0, 10_000.0,
                                      strategy="trend")
    await session._tick()

    assert "trend" in session.attribution.book_for("paper").records, (
        "a full trading tick ran and the fill it followed was never booked")


@pytest.mark.asyncio
async def test_starting_the_session_reads_the_book_back(home, monkeypatch):
    """Everything else about restarts depends on this: a start that did not
    load the book would begin each run with no owners, and every exit after a
    restart would be booked to nobody."""
    first = _paper_session()
    await _enter(first, "AAPL", 0.10, 100.0, "trend")

    second = TradingSession()

    async def nothing(*a, **kw):
        return None

    # The venue-facing parts of start, which have nothing to reach here.
    for name in ("refresh_clock", "scan_universe", "seed_history",
                 "refresh_daily_history", "refresh_universe"):
        monkeypatch.setattr(second, name, nothing)
    monkeypatch.setattr(second.feed, "start", nothing)
    try:
        await second.start()
    finally:
        await second.stop()

    assert second._attribution_loaded
    assert "trend" in second.attribution.book_for("paper").records
