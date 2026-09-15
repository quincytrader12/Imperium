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
    assert "SECTOR_TREND_ARM_AT_EQUITY=0" in message
    assert "settings.txt" in message


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
