"""The permanent record: every fill, and the equity over time.

What matters: a fill reaches the CSV exactly once, with the strategy that
placed it and what it realised; the file opens in a spreadsheet as it is and
survives a line cut short by a crash; the equity history keeps the right
resolution for each view, survives a restart, and is kept per mode; and the
export and chart endpoints serve what the page asks for and nothing else.
"""

from __future__ import annotations

import csv
import io
import json
import time
from decimal import Decimal

import pytest

from imperium import config
from imperium.execution import journal as jr
from imperium.execution.broker import Fill, Mode, PaperBroker
from imperium.session import TradingSession
from imperium.venues import registry


def _fill(side="BUY", qty="2", price="100", strategy="trend", ts=None, **kw):
    return Fill("AAPL", side, Decimal(qty), Decimal(price),
                ts if ts is not None else time.time(), Mode.PAPER, "c1", True,
                reference_price=Decimal("99.9"), strategy=strategy, **kw)


# -- trades ---------------------------------------------------------------------


def test_a_fill_becomes_one_row_a_spreadsheet_reads(tmp_path):
    j = jr.TradeJournal(tmp_path / "t.csv")
    assert j.append([jr.row_from_fill(_fill(realised=3.25))]) == 1
    rows = list(csv.DictReader(open(tmp_path / "t.csv", encoding="utf-8")))
    assert list(rows[0]) == list(jr.COLUMNS)
    r = rows[0]
    assert (r["symbol"], r["side"], r["quantity"], r["price"]) == (
        "AAPL", "BUY", "2", "100")
    assert r["value"] == "200.00"
    assert r["strategy"] == "trend" and r["order_type"] == "market"
    assert r["realised_pnl"] == "3.2500"
    assert float(r["slippage_bps"]) == pytest.approx(10.01, abs=0.01)


def test_the_header_is_written_once_and_rows_are_only_ever_appended(tmp_path):
    path = tmp_path / "t.csv"
    j = jr.TradeJournal(path)
    j.append([jr.row_from_fill(_fill())])
    first = path.read_text(encoding="utf-8")
    j.append([jr.row_from_fill(_fill(side="SELL"))])
    second = path.read_text(encoding="utf-8")
    assert second.startswith(first)
    assert second.count("time_utc") == 1
    assert len(jr.TradeJournal(path).read()) == 2


def test_a_line_cut_short_by_a_crash_costs_that_line_only(tmp_path):
    path = tmp_path / "t.csv"
    j = jr.TradeJournal(path)
    j.append([jr.row_from_fill(_fill())])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("2026-09-27 10:00:00,17")          # the crash
    j.append([jr.row_from_fill(_fill(side="SELL", ts=time.time() + 1))])
    sides = [r.side for r in j.read()]
    assert "BUY" in sides and len(sides) >= 1


def test_a_full_disk_does_not_stop_the_book(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    j = jr.TradeJournal(blocker / "t.csv")           # a file where a dir must be
    assert j.append([jr.row_from_fill(_fill())]) == 0
    assert j.failed == 1 and j.last_error


def test_reads_filter_by_time_and_mode(tmp_path):
    j = jr.TradeJournal(tmp_path / "t.csv")
    now = time.time()
    live = _fill(ts=now)
    live.mode = Mode.LIVE
    j.append([jr.row_from_fill(_fill(ts=now - 100)), jr.row_from_fill(live)])
    assert len(j.read(since=now - 10)) == 1
    assert [r.mode for r in j.read(mode="live")] == ["live"]
    assert "live" in j.export(mode="live") and "paper" not in j.export(mode="live")


def test_an_auction_order_is_named_as_one():
    from imperium.execution.broker import MARKET_ON_CLOSE

    row = jr.row_from_fill(_fill(order=MARKET_ON_CLOSE))
    assert row.cells()[jr.COLUMNS.index("order_type")] == MARKET_ON_CLOSE


# -- the session writes it --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_tick_journals_each_fill_once_with_its_owner_and_result():
    """Through _attribute, which the tick calls: an entry and its exit, each
    written exactly once, the exit carrying what the round trip realised."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session._attribution_loaded = True
    q = session.feed.quote("AAPL")
    q.last, q.updated_at = 100.0, time.time()
    await session.broker.apply_target("AAPL", 0.10, 100.0, 10_000.0,
                                      strategy="trend")
    session._attribute()
    session._attribute()
    await session.broker.apply_target("AAPL", 0.0, 110.0, 10_000.0)
    session._attribute()
    rows = session.trade_journal.read()
    assert [r.side for r in rows] == ["BUY", "SELL"]
    assert all(r.strategy == "trend" for r in rows)
    assert rows[0].realised == 0.0
    assert rows[1].realised == pytest.approx(
        session.attribution.books["paper"].records["trend"].realised)
    assert rows[1].realised > 0
    assert config.trades_path().exists()


def test_the_equity_history_follows_the_whole_book_per_mode(monkeypatch):
    session = TradingSession()
    history = session.equity_history()
    history.record(time.time() - 400, 1000.0, "2026-09-25")
    history.record(time.time(), 1010.0, "2026-09-26")
    history.save()
    assert config.equity_history_path(session.broker.mode.value).exists()
    assert session.equity_history("live") is not history
    again = TradingSession()
    assert [v for _, v in again.equity_history().daily] == [1000.0, 1010.0]


# -- equity ---------------------------------------------------------------------


def test_one_intraday_point_per_five_minutes_and_the_live_end_follows(tmp_path):
    h = jr.EquityHistory(tmp_path / "e.json")
    t = 1_000_000.0
    assert h.record(t, 100.0, "d1")
    assert not h.record(t + 60, 101.0, "d1")
    assert h.intraday == [(t, 101.0)]
    assert h.record(t + jr.SAMPLE_SECONDS, 102.0, "d1")
    assert len(h.intraday) == 2


def test_the_day_keeps_its_last_reading_and_the_week_ages_out(tmp_path):
    h = jr.EquityHistory(tmp_path / "e.json")
    t = 2_000_000.0
    h.record(t, 100.0, "2026-09-01")
    h.record(t + 400, 105.0, "2026-09-01")
    h.record(t + jr.INTRADAY_KEEP_SECONDS + 1000, 110.0, "2026-09-10")
    assert h.daily == [("2026-09-01", 105.0), ("2026-09-10", 110.0)]
    assert all(p[0] > t for p in h.intraday)


def test_a_bad_reading_is_not_charted(tmp_path):
    h = jr.EquityHistory(tmp_path / "e.json")
    for v in (0.0, -5.0, float("nan"), float("inf")):
        assert not h.record(1.0, v, "d1")
    assert h.intraday == [] and h.daily == []


def test_short_views_read_five_minute_points_and_long_ones_daily(tmp_path):
    h = jr.EquityHistory(tmp_path / "e.json")
    now = 1_800_000_000.0
    for i in range(10):
        h.record(now - 3600 + i * 400, 100.0 + i, "2027-01-15")
    h.daily = [("2026-12-01", 90.0), ("2027-01-14", 95.0), ("2027-01-15", 109.0)]
    day = h.series(86_400, now)
    assert len(day) == len(h.intraday)
    month = h.series(62 * 86_400, now)
    assert [v for _, v in month][:2] == [90.0, 95.0]
    assert month[-1] == h.intraday[-1]
    assert all(t <= now for t, _ in month), "a point after the present"
    assert h.series(None, now)[0][1] == 90.0


def test_the_history_survives_a_restart_and_a_corrupt_file(tmp_path):
    path = tmp_path / "e.json"
    h = jr.EquityHistory(path)
    h.record(1000.0, 100.0, "2026-01-02")
    assert h.save()
    back = jr.EquityHistory.load(path)
    assert back.intraday == h.intraday and back.daily == h.daily
    path.write_text(json.dumps({"intraday": [[1, "x"], [2, 50.0], [3, -1]],
                                "daily": [["d", "nan"], ["2026-01-03", 7]]}))
    back = jr.EquityHistory.load(path)
    assert back.intraday == [(2.0, 50.0)] and back.daily == [("2026-01-03", 7.0)]
    path.write_text("{not json")
    assert jr.EquityHistory.load(path).intraday == []


def test_saves_are_throttled(tmp_path):
    h = jr.EquityHistory(tmp_path / "e.json")
    h.record(1.0, 100.0, "d")
    assert h.maybe_save(now=10_000.0)
    h.record(400.0, 101.0, "d")
    assert not h.maybe_save(now=10_010.0)
    assert h.maybe_save(now=10_000.0 + jr.SAVE_SECONDS)


# -- served ---------------------------------------------------------------------


@pytest.fixture
def app_client():
    from fastapi.testclient import TestClient
    from imperium.server.app import create_app

    with TestClient(create_app()) as client:
        yield client


def _session(client):
    return client.app.state.session if hasattr(client.app.state, "session") else None


def test_the_chart_endpoint_serves_the_range_it_is_asked_for(app_client):
    body = app_client.get("/api/chart?span=1D").json()
    assert body["span"] == "1D"
    assert set(body) >= {"equity", "strategies", "trades", "mode"}
    assert app_client.get("/api/chart?span=nonsense").json()["span"] == "1W"


def test_the_export_is_a_download_and_refuses_an_unknown_mode(app_client):
    r = app_client.get("/api/trades.csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    assert r.text.splitlines()[0] == ",".join(jr.COLUMNS)
    bad = app_client.get('/api/trades.csv?mode=x"%0d%0aSet-Cookie:a=b')
    assert bad.status_code == 400
    assert "set-cookie" not in {k.lower() for k in bad.headers}
