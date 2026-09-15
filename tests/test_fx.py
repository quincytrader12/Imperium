"""A second currency beside the dollar balance.

One rule runs through this file: a converted figure is never presented without
the rate that produced it and how old that rate is. The rate moves, the last
fetch may have failed hours ago, and an operator reading a stale conversion as
current is worse off than one reading dollars.

The second rule is that none of it ever touches a decision. Every number this
program reasons about stays in dollars; this is a label.
"""

from __future__ import annotations

import time

import httpx
import pytest

from imperium.venues import fx
from imperium.venues.fx import FxDesk, Rate


def _ok(rate: float = 18.42):
    return httpx.MockTransport(lambda r: httpx.Response(
        200, json={"amount": 1, "base": "USD", "date": "2026-09-15",
                   "rates": {"ZAR": rate}}))


@pytest.mark.asyncio
async def test_a_fetched_rate_converts_the_balance():
    desk = FxDesk("ZAR", transport=_ok())
    await desk.refresh()
    assert desk.rate.known
    assert desk.rate.convert(73.40) == pytest.approx(73.40 * 18.42)


@pytest.mark.asyncio
async def test_the_rate_and_its_age_travel_with_the_number():
    """Prevents a figure in rands appearing on screen with nothing behind it."""
    desk = FxDesk("ZAR", transport=_ok())
    await desk.refresh()
    panel = desk.panel()
    assert panel["rate"] == pytest.approx(18.42)
    assert panel["age"] is not None
    assert panel["source"]


def test_an_unfetched_rate_is_not_presented_as_a_number():
    """Before the first fetch there is no conversion to show, and the panel
    must say so rather than showing a zero."""
    panel = FxDesk("ZAR").panel()
    assert panel["known"] is False
    assert panel["rate"] is None


def test_a_rate_that_has_aged_out_is_marked_stale_rather_than_hidden():
    """An old rate is still useful if you know it is old. Hiding it would
    replace a slightly wrong number with no number, which is worse."""
    desk = FxDesk("ZAR")
    desk.rate = Rate(quote="ZAR", rate=18.0,
                     fetched_at=time.time() - fx.STALE_AFTER - 60)
    assert desk.rate.known is True
    assert desk.rate.stale is True
    assert desk.panel()["stale"] is True


def test_a_fresh_rate_is_not_marked_stale():
    desk = FxDesk("ZAR")
    desk.rate = Rate(quote="ZAR", rate=18.0, fetched_at=time.time())
    assert desk.rate.stale is False


@pytest.mark.asyncio
async def test_a_failed_fetch_keeps_the_last_rate_and_records_why():
    """The previous rate ages into "stale" on its own rather than vanishing
    the moment a request fails."""
    desk = FxDesk("ZAR", transport=_ok(18.0))
    await desk.refresh()
    assert desk.rate.rate == pytest.approx(18.0)

    desk.transport = httpx.MockTransport(lambda r: httpx.Response(503))
    desk._checked_at = 0.0
    await desk.refresh()
    assert desk.rate.rate == pytest.approx(18.0), "the last rate was discarded"
    assert desk.rate.error


@pytest.mark.asyncio
async def test_a_currency_outage_never_reaches_the_trading_loop():
    """A display conversion taking the terminal down would be an absurd
    trade."""
    def explode(request):
        raise httpx.ConnectError("no route")

    desk = FxDesk("ZAR", transport=httpx.MockTransport(explode))
    await desk.refresh()          # must not raise
    assert desk.rate.known is False
    assert desk.rate.error


@pytest.mark.asyncio
async def test_a_hand_set_rate_is_never_overwritten_by_a_fetch():
    """Someone who typed a rate meant it."""
    desk = FxDesk("ZAR", manual_rate=19.5, transport=_ok(18.42))
    assert desk.due() is False
    await desk.refresh()
    assert desk.rate.rate == pytest.approx(19.5)
    assert "hand" in desk.rate.source


def test_a_blank_currency_disables_the_whole_thing():
    desk = FxDesk("")
    assert desk.enabled is False
    assert desk.due() is False
    assert desk.panel()["enabled"] is False


def test_it_does_not_ask_on_every_tick():
    desk = FxDesk("ZAR")
    assert desk.due()
    desk._checked_at = 1_000.0
    assert desk.due(now=1_000.0 + fx.REFRESH_SECONDS - 1) is False
    assert desk.due(now=1_000.0 + fx.REFRESH_SECONDS + 1) is True


# -- it must never influence a decision ----------------------------------

def test_no_strategy_or_sizing_module_imports_the_currency_desk():
    """THE constraint. Every decision this program makes stays in dollars, and
    the cheapest way to keep that true is for the code that decides to have no
    access to the conversion at all."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "imperium"
    offenders = []
    for folder in ("strategy", "execution"):
        for path in (root / folder).rglob("*.py"):
            if "fx" in path.read_text(encoding="utf-8"):
                text = path.read_text(encoding="utf-8")
                if "venues.fx" in text or "import fx" in text:
                    offenders.append(str(path.relative_to(root)))
    assert not offenders, (
        f"the currency conversion reached decision code: {offenders}")


def test_the_ui_shows_the_rate_beside_the_converted_balance():
    """The display half of the same rule."""
    from pathlib import Path

    static = (Path(__file__).resolve().parents[1] / "src" / "imperium"
              / "server" / "static")
    app = (static / "app.js").read_text(encoding="utf-8")
    index = (static / "index.html").read_text(encoding="utf-8")

    assert 'id="h-equity-alt"' in index
    assert "h-equity-alt" in app
    assert "1 USD = " in app, "the rate is never shown next to the number"
    assert "(stale)" in app, "a stale rate is never marked as such"
    assert "Display only" in app
