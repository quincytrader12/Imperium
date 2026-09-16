"""Arming the Sector Trend sleeve on equity.

A strategy that switches itself on is the most surprising thing this program
can do unattended, so the rules around it are narrow and each is tested: it
arms once, it never disarms, it survives a restart without re-announcing, and
the moment it happens the capital split moves with it.
"""

from __future__ import annotations

import datetime as dt
import decimal

import pytest

from imperium.execution.broker import PaperBroker
from imperium.venues import registry
from imperium.venues.alpaca.client import MarketClock

from imperium.execution.capital import ENGINE
from imperium.execution.sector_sleeve import SectorRunner
from imperium.execution.sleeve_ledger import SleeveLedger
from imperium.session import TradingSession
from imperium.strategy.sector_config import SectorTrendConfig


def _runner(threshold: float = 200.0, **kwargs) -> SectorRunner:
    return SectorRunner(
        config=SectorTrendConfig(arm_at_equity=threshold, **kwargs),
        ledger=SleeveLedger())


def test_it_stays_off_below_the_threshold():
    runner = _runner()
    assert runner.consider_arming(199.99, "2026-09-15") == ""
    assert runner.enabled is False


def test_it_arms_when_equity_reaches_the_threshold():
    runner = _runner()
    message = runner.consider_arming(200.0, "2026-09-15")
    assert message
    assert runner.enabled is True
    assert runner.ledger.armed_on == "2026-09-15"


def test_the_announcement_says_what_changed_and_how_to_stop_it():
    """An unattended strategy switching itself on must explain itself, and the
    explanation has to include the off switch."""
    message = _runner().consider_arming(250.0, "2026-09-15")
    assert "$250.00" in message and "$200" in message
    assert "own capital" in message
    assert "other strategies lose that slice" in message
    # It used to say "set SECTOR_TREND_ARM_AT_EQUITY=0 and restart".
    # There is a Disarm button in that panel now, so pointing at a
    # file would be sending the operator the long way round.
    assert "Disarm" in message and "panel" in message


def test_it_announces_once_and_not_on_every_tick():
    """Prevents a notification every second for the rest of the session."""
    runner = _runner()
    assert runner.consider_arming(300.0, "2026-09-15")
    assert runner.consider_arming(300.0, "2026-09-15") == ""
    assert runner.consider_arming(900.0, "2026-09-16") == ""


def test_it_never_disarms_on_a_drawdown():
    """THE rule here.

    A sleeve that switched itself off below the threshold would abandon
    whatever it was holding: the positions would sit there with their stops no
    longer trailed and nothing left to close them, which is worse than either
    state on its own. Turning it off is a decision with a file to write it in.
    """
    runner = _runner()
    runner.consider_arming(220.0, "2026-09-15")
    runner.ledger.open_position("XLK", 1.0, 200.0, 190.0, "2026-09-15")

    assert runner.consider_arming(40.0, "2026-09-20") == ""
    assert runner.enabled is True, "the sleeve disarmed and stranded a position"
    assert runner.ledger.longs() == ["XLK"]


def test_arming_survives_a_restart_without_announcing_again():
    runner = _runner()
    assert runner.consider_arming(500.0, "2026-09-15")

    fresh = SectorRunner(config=SectorTrendConfig(arm_at_equity=200.0),
                         ledger=SleeveLedger.load())
    assert fresh.enabled is True
    assert fresh.consider_arming(500.0, "2026-09-16") == ""


def test_a_threshold_of_zero_never_arms():
    """The off switch the announcement points at."""
    runner = _runner(threshold=0.0)
    assert runner.consider_arming(1_000_000.0, "2026-09-15") == ""
    assert runner.enabled is False


def test_a_sleeve_already_switched_on_by_hand_does_not_announce():
    runner = _runner(enabled=True)
    assert runner.consider_arming(500.0, "2026-09-15") == ""
    assert runner.enabled is True


def test_the_default_threshold_is_where_every_etf_clears_the_venue_floor():
    """Not an arbitrary round number. Alpaca refuses a fractional buy under a
    dollar; at the default allocation a $200 account gives the sleeve $40,
    which is enough for the most volatile ETF in the universe. Below about
    $130 the sleeve would quietly trade only the calm half of its universe --
    a different strategy from the one that was backtested."""
    cfg = SectorTrendConfig()
    assert cfg.arm_at_equity == pytest.approx(200.0)
    sleeve = cfg.sleeve_equity(cfg.arm_at_equity)
    assert sleeve == pytest.approx(40.0)
    # The most volatile ETF here sits near 2% daily; its weight must still be
    # worth more than a dollar.
    weight = (cfg.target_vol / cfg.universe_size) / 0.020
    assert weight * sleeve >= 1.0


# -- through the session --------------------------------------------------

@pytest.mark.asyncio
async def test_arming_moves_the_capital_split_on_the_same_tick():
    """Prevents the engine sizing against capital the sleeve has just taken.

    Arming changes the split. If the engine were told its new share a tick
    later it would place one round of orders against money that is no longer
    its own.
    """
    session = TradingSession()
    session.sector.config = SectorTrendConfig(arm_at_equity=200.0)

    assert session.engine_equity(100.0) == pytest.approx(100.0)

    message = session.sector.consider_arming(400.0, "2026-09-15")
    assert message
    assert session.engine_equity(400.0) == pytest.approx(320.0)
    assert session.capital.share_for(ENGINE) == pytest.approx(0.80)
    assert session.capital.share_for("sector_trend") == pytest.approx(0.20)


@pytest.mark.asyncio
async def test_arming_does_not_halt_the_book_on_a_loss_it_never_took():
    """The trap arming walks straight into if nothing moves the reference.

    ``day_start_equity`` is the engine's share at the start of the day and
    ``allocator.equity`` is the engine's share now. Arming takes the engine
    from the whole account to four fifths of it in a single tick. The account
    has lost nothing, but one side of that comparison is measured in fifths of
    five and the other in fifths of four, so the check reads a 20% loss against
    a 4% limit and halts the book -- on its first tick past the threshold,
    having traded nothing, for a reason no position could explain.
    """
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.market_clock = MarketClock(
        is_open=True,
        timestamp=dt.datetime(2026, 3, 4, 15, tzinfo=dt.timezone.utc),
        next_close=dt.datetime(2026, 3, 4, 21, tzinfo=dt.timezone.utc))
    session.sector.config = SectorTrendConfig(arm_at_equity=10_000.0)
    # A dollar under the threshold, so that crossing it is the only thing that
    # happens between the two ticks. A larger jump would hide the defect: a
    # gain big enough to swamp the re-basing leaves the check reading a profit
    # either way, and the test would pass with the bug still in place.
    session.broker.cash = decimal.Decimal("9999")
    # The venue's balance as well as the book: arming reads the account,
    # and the daily-loss reference is measured on the book.
    session.account_equity = 9999.0

    await session._tick()
    assert session.sector.enabled is False
    assert session.day_start_equity == pytest.approx(9_999.0)

    # One dollar up, and over the line. The account gained; nothing was lost.
    session.broker.cash = decimal.Decimal("10000")
    session.account_equity = 10000.0
    await session._tick()

    assert session.sector.enabled is True
    assert session.capital.share_for(ENGINE) == pytest.approx(0.80)
    assert not session.allocator.halted, session.allocator.halt_reason
    # The same start of day, re-expressed in the share the engine now holds.
    assert session.day_start_equity == pytest.approx(7_999.2)


@pytest.mark.asyncio
async def test_a_real_loss_across_an_arming_still_halts_the_book():
    """The other half of the same rule. Re-basing the reference must not
    become a way for a loss to go unnoticed: what changes is the base, not
    the limit."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.market_clock = MarketClock(
        is_open=True,
        timestamp=dt.datetime(2026, 3, 4, 15, tzinfo=dt.timezone.utc),
        next_close=dt.datetime(2026, 3, 4, 21, tzinfo=dt.timezone.utc))
    session.sector.config = SectorTrendConfig(arm_at_equity=10_000.0)
    session.broker.cash = decimal.Decimal("10000")
    session.account_equity = 10000.0

    await session._tick()
    assert session.sector.enabled is True
    assert session.day_start_equity == pytest.approx(8_000.0)
    assert not session.allocator.halted

    # Now a genuine 5% fall, on the far side of the arming.
    session.broker.cash = decimal.Decimal("9500")
    await session._tick()

    assert session.allocator.halted
    assert session.allocator.halt_source == "daily_loss"


@pytest.mark.asyncio
async def test_it_does_not_arm_on_the_simulated_book():
    """Caught by looking at a running terminal, not by a test.

    A dry-run book opens at a default ten thousand dollars. Arming read the
    book rather than the account, so a terminal with no key attached at all
    armed the sleeve on its first tick -- against money that does not exist,
    persisted to the state file, and never disarming. An operator starting at
    fifty dollars would have found a strategy running that their balance had
    never reached the threshold for.
    """
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = decimal.Decimal("10000")   # the simulation
    session.account_equity = 0.0                     # nothing read yet
    session.market_clock = MarketClock(
        is_open=True,
        timestamp=dt.datetime(2026, 3, 4, 15, tzinfo=dt.timezone.utc),
        next_close=dt.datetime(2026, 3, 4, 21, tzinfo=dt.timezone.utc))
    session.sector.config = SectorTrendConfig(arm_at_equity=200.0)

    await session._tick()

    assert session.sector.enabled is False
    assert session.sector.ledger.armed_on == ""
    assert session.capital.share_for(ENGINE) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_it_arms_once_the_venue_reports_a_balance_past_the_threshold():
    """The other half: a real balance does arm it, on the tick it is read."""
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = decimal.Decimal("10000")
    session.market_clock = MarketClock(
        is_open=True,
        timestamp=dt.datetime(2026, 3, 4, 15, tzinfo=dt.timezone.utc),
        next_close=dt.datetime(2026, 3, 4, 21, tzinfo=dt.timezone.utc))
    session.sector.config = SectorTrendConfig(arm_at_equity=200.0)

    session.account_equity = 199.0
    await session._tick()
    assert session.sector.enabled is False

    session.account_equity = 214.80
    await session._tick()
    assert session.sector.enabled is True
    assert session.sector.ledger.armed_at_equity == pytest.approx(214.80)


def test_an_unread_balance_is_not_treated_as_zero_growth():
    """arming_equity returns zero while the account is unknown, and zero never
    reaches a threshold -- so "not read yet" and "too small" produce the same
    inaction, which is the safe one."""
    session = TradingSession()
    session.account_equity = 0.0
    assert session.arming_equity() == 0.0
    session.account_equity = 214.80
    assert session.arming_equity() == pytest.approx(214.80)


def test_the_panel_distinguishes_configured_from_self_armed():
    """"On because I said so" and "on because it decided to" are different
    facts, and an operator seeing the sleeve trading deserves to know which."""
    runner = _runner()
    runner.consider_arming(300.0, "2026-09-15")
    panel = runner.panel()
    assert panel["enabled"] is True
    assert panel["configured"] is False
    assert panel["armed_on"] == "2026-09-15"
    assert panel["armed_at_equity"] == pytest.approx(300.0)


# -- arming from the panel ---------------------------------------------------


def test_arming_by_hand_records_that_a_person_did_it():
    """"It armed itself" and "somebody armed it" are different facts.

    A sleeve that switched itself on is the program acting unattended; one
    armed by hand is a decision a person made and may not remember making. The
    panel says which, so neither has to be guessed at later.
    """
    runner = _runner()
    assert runner.arm_by_hand(143.20, "2026-09-16")
    assert runner.enabled is True
    assert runner.ledger.armed_by == "hand"
    assert runner.ledger.armed_at_equity == pytest.approx(143.20)


def test_arming_by_hand_below_the_threshold_is_allowed():
    """It is the operator's money. The warning is the server's job, not a
    refusal."""
    runner = _runner(threshold=200.0)
    assert runner.arm_by_hand(50.0, "2026-09-16")
    assert runner.enabled is True


def test_arming_twice_by_hand_does_nothing_the_second_time():
    runner = _runner()
    first = runner.arm_by_hand(143.20, "2026-09-16")
    second = runner.arm_by_hand(999.0, "2026-09-17")
    assert first and not second
    assert runner.ledger.armed_at_equity == pytest.approx(143.20)


def test_disarming_is_refused_while_it_holds_anything():
    """The whole reason arming was one-way.

    A sleeve switched off mid-book leaves its ETFs sitting there with nobody
    trailing their stops and nothing left to close them, which is worse than
    either state on its own.
    """
    runner = _runner()
    runner.arm_by_hand(400.0, "2026-09-16")
    runner.ledger.open_position("XLF", quantity=2.0, price=40.0,
                                stop=38.0, day="2026-09-16")

    refusal = runner.disarm()
    assert refusal
    assert "XLF" in refusal
    assert "trailing their stops" in refusal
    assert runner.enabled is True, "it disarmed anyway"


def test_disarming_works_once_it_is_flat():
    runner = _runner()
    runner.arm_by_hand(400.0, "2026-09-16")
    assert runner.disarm() == ""
    assert runner.enabled is False
    assert runner.ledger.armed_by == ""


def test_a_button_cannot_overrule_the_settings_file():
    """Somebody who wrote SECTOR_TREND_ENABLED=true meant it, and a control in
    a web page that silently undid a file they edited would be the program
    disagreeing with its own configuration."""
    runner = SectorRunner(config=SectorTrendConfig(enabled=True),
                          ledger=SleeveLedger())
    refusal = runner.disarm()
    assert "settings.txt" in refusal
    assert runner.enabled is True


def test_the_arming_gauge_measures_the_account_not_the_sleeve():
    """The sleeve has no capital until it arms, so it cannot answer "how close
    are we" from anything it owns."""
    runner = _runner(threshold=200.0)
    progress = runner.arming_progress(143.20)
    assert progress["threshold"] == 200.0
    assert progress["equity"] == pytest.approx(143.20)
    assert progress["fraction"] == pytest.approx(0.716)
    assert progress["short_by"] == pytest.approx(56.80)
    assert progress["watching"] is True

    # Past the line it reads full rather than over-full.
    assert runner.arming_progress(400.0)["fraction"] == 1.0
    assert runner.arming_progress(400.0)["short_by"] == 0.0


def test_the_gauge_stops_watching_once_it_is_armed():
    runner = _runner(threshold=200.0)
    runner.arm_by_hand(400.0, "2026-09-16")
    assert runner.arming_progress(400.0)["watching"] is False


def test_the_volatility_ceiling_is_the_exact_sizing_arithmetic():
    """Not an estimate.

    Sizing is w = (target_vol / N) / sigma, so a position is worth
    sleeve * target_vol / (N * sigma) and clears the venue's one-dollar floor
    only while sigma <= sleeve * target_vol / N. This is the number that makes
    an undersized sleeve a *different* strategy rather than a smaller one:
    weight falls as volatility rises, so the names priced out first are the
    volatile ones the returns come from.
    """
    runner = _runner()
    config = runner.config
    for equity in (50.0, 143.20, 200.0, 1000.0):
        sleeve = equity * config.allocation
        expected = sleeve * config.target_vol / config.universe_size
        assert runner.volatility_ceiling(equity) == pytest.approx(expected)

    # At the default threshold every SPDR sector ETF clears it; the most
    # volatile of them run around 2.5% a day.
    assert runner.volatility_ceiling(200.0) > 0.025
    # Well under it, they do not.
    assert runner.volatility_ceiling(100.0) < 0.020


def test_the_volatility_ceiling_is_zero_with_no_capital():
    assert _runner().volatility_ceiling(0.0) == 0.0


# -- the endpoints the button calls ------------------------------------------


def _client(equity: float = 0.0):
    from fastapi.testclient import TestClient

    from imperium.server.app import create_app

    session = TradingSession()
    session.sector.config = SectorTrendConfig(arm_at_equity=200.0)
    session.account_equity = equity
    return TestClient(create_app(session)), session


def test_the_arm_endpoint_refuses_without_a_known_balance():
    """There is nothing to size a sleeve against, so there is no honest
    answer to "arm it"."""
    client, _ = _client(equity=0.0)
    with client:
        reply = client.post("/api/sector/arm", json={})
        assert reply.status_code == 409
        assert "balance is not known" in reply.json()["detail"]


def test_arming_early_asks_once_and_says_exactly_what_it_costs():
    """The refusal has to carry the reason, not just say no.

    The cost of an early arm is not "it will be small" -- it is that the names
    priced out are the volatile ones, so it would be trading the calm half of
    the universe, which has no backtest behind it.
    """
    client, _ = _client(equity=143.20)
    with client:
        reply = client.post("/api/sector/arm", json={})
        assert reply.status_code == 409
        detail = reply.json()["detail"]
        assert "$143.20" in detail and "$200" in detail
        assert "2.26% a day" in detail, detail
        assert "most volatile" in detail
        assert "Arm anyway" in detail, (
            "the refusal does not tell the operator how to proceed")


def test_acknowledging_arms_it():
    client, session = _client(equity=143.20)
    with client:
        reply = client.post("/api/sector/arm", json={"acknowledged": True})
        assert reply.status_code == 200
        assert reply.json()["armed"] is True
        assert session.sector.enabled is True
        assert session.sector.ledger.armed_by == "hand"


def test_arming_above_the_threshold_needs_no_acknowledgement():
    client, session = _client(equity=400.0)
    with client:
        assert client.post("/api/sector/arm", json={}).status_code == 200
        assert session.sector.enabled is True


def test_the_disarm_endpoint_refuses_while_it_holds_something():
    client, session = _client(equity=400.0)
    with client:
        client.post("/api/sector/arm", json={})
        session.sector.ledger.open_position("XLK", quantity=1.0, price=200.0,
                                            stop=190.0, day="2026-09-16")
        reply = client.post("/api/sector/disarm")
        assert reply.status_code == 409
        assert "XLK" in reply.json()["detail"]
        assert session.sector.enabled is True


def test_disarming_works_from_the_endpoint_when_flat():
    client, session = _client(equity=400.0)
    with client:
        client.post("/api/sector/arm", json={})
        assert client.post("/api/sector/disarm").status_code == 200
        assert session.sector.enabled is False


def test_the_panel_carries_the_gauge_the_button_needs():
    client, _ = _client(equity=143.20)
    with client:
        sector = client.get("/api/snapshot").json()["sector"]
        assert sector["arming"]["threshold"] == 200.0
        assert sector["arming"]["equity"] == pytest.approx(143.20)
        assert sector["arming"]["watching"] is True
        assert sector["vol_ceiling"] > 0
        assert sector["can_disarm"] is False
        assert sector["armed_by"] == ""
