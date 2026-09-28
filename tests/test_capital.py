"""Dividing one account between strategies that do not know about each other.

The bug this file is about: every strategy sized against the *whole* account.
The engine's allocator took total equity as its base and the Sector Trend
sleeve took a fifth of total equity on top, so the two together intended 120%
of the money. Neither was wrong about its own share; neither knew the other
existed. Nothing noticed, because each was internally consistent.

The second test here guards a trap the fix itself created, which is the more
dangerous of the two.
"""

from __future__ import annotations

import pytest

from imperium.execution.capital import ENGINE, Claim, divide
from imperium.execution.portfolio import PortfolioAllocator
from imperium.execution.risk import limits_for_equity
from imperium.session import TradingSession
from imperium.strategy import sector_config as sc


# -- the split ------------------------------------------------------------

def test_the_shares_never_add_up_to_more_than_the_account():
    """THE test. Before this, engine 1.0 plus sleeve 0.2 was a real state."""
    plan = divide(70.0, [Claim("sector_trend", 0.20)])
    assert plan.claimed == pytest.approx(1.0)
    assert plan.share_for("sector_trend") == pytest.approx(0.20)
    assert plan.share_for(ENGINE) == pytest.approx(0.80)
    assert plan.equity_for("sector_trend") + plan.equity_for(ENGINE) == \
        pytest.approx(70.0)


def test_a_disabled_sleeve_reserves_nothing():
    """Capital held by something switched off is capital doing nothing while
    the strategies that could use it are told they may not."""
    plan = divide(70.0, [Claim("sector_trend", 0.20, enabled=False)])
    assert plan.share_for(ENGINE) == pytest.approx(1.0)
    assert "sector_trend" not in plan.shares


def test_turning_a_sleeve_on_takes_the_capital_from_the_engine():
    """And not from thin air, which is what happened before."""
    off = divide(70.0, [Claim("s", 0.20, enabled=False)])
    on = divide(70.0, [Claim("s", 0.20, enabled=True)])
    assert off.share_for(ENGINE) - on.share_for(ENGINE) == pytest.approx(0.20)


def test_an_oversubscribed_account_refuses_the_last_claim_and_says_so():
    """Predictable rather than arbitrary: claims are honoured in order, so it
    is the last one that is cut, not whichever a dictionary yielded first. A
    strategy given nothing must be named -- switched on and starved looks
    exactly like switched on and finding no trades."""
    plan = divide(100.0, [Claim("a", 0.70), Claim("b", 0.60)])
    assert plan.share_for("a") == pytest.approx(0.70)
    assert plan.share_for("b") == pytest.approx(0.30)
    assert plan.refused["b"] == pytest.approx(0.60)
    assert plan.claimed <= 1.0 + 1e-12


def test_a_sleeve_that_can_have_less_than_it_asked_still_trades():
    """A sleeve that wanted a fifth and can have a tenth should trade the
    tenth rather than nothing."""
    plan = divide(100.0, [Claim("a", 0.95), Claim("b", 0.20)])
    assert plan.share_for("b") == pytest.approx(0.05)


def test_the_engine_cannot_claim_a_share_of_its_own():
    """It is the residual by construction. Accepting a claim for it would let
    the total exceed one."""
    plan = divide(100.0, [Claim(ENGINE, 0.90), Claim("s", 0.20)])
    assert plan.share_for(ENGINE) == pytest.approx(0.80)
    assert plan.claimed == pytest.approx(1.0)


@pytest.mark.parametrize("equity", [0.0, -5.0])
def test_a_worthless_account_divides_to_nothing_rather_than_raising(equity):
    plan = divide(equity, [Claim("s", 0.20)])
    assert plan.equity_for(ENGINE) == 0.0
    assert plan.equity_for("s") == 0.0


# -- the trap the fix created --------------------------------------------

def test_the_split_does_not_halt_the_book_with_an_imaginary_loss():
    """The dangerous one, and the reason every equity goes through one method.

    ``check_daily_loss`` measures ``day_start_equity`` against
    ``allocator.equity``. Set the allocator to the engine's 80% share while
    the day's reference is still the account's 100% and the check reads a 20%
    loss -- against a 4% limit. The book would halt on its first tick, having
    traded nothing, and the reason on screen would be a loss that never
    happened.
    """
    limits = limits_for_equity(70.0)
    allocator = PortfolioAllocator(limits)

    account_equity = 70.0
    engine_share = 0.80

    # The bug: two different bases.
    allocator.equity = account_equity * engine_share
    assert allocator.check_daily_loss(account_equity) is True, (
        "this is the failure mode being guarded against")

    # The fix: one basis, consistently.
    fixed = PortfolioAllocator(limits_for_equity(70.0))
    fixed.equity = account_equity * engine_share
    assert fixed.check_daily_loss(account_equity * engine_share) is False


def test_the_session_uses_one_basis_for_both_sides_of_that_check(monkeypatch):
    """The same property through the real session rather than a stub."""
    monkeypatch.setenv("SECTOR_TREND_ENABLED", "true")
    session = TradingSession()
    session.sector.config = sc.from_environment()
    assert session.sector.config.enabled

    engine = session.engine_equity(70.0)
    assert engine == pytest.approx(56.0)

    session.allocator.equity = engine
    session.day_start_equity = engine
    assert session.allocator.check_daily_loss(session.day_start_equity) is False
    assert session.allocator.halted is False


def test_the_engine_sizes_against_its_share_and_not_the_account(monkeypatch):
    """Prevents the original double count coming back through the sizing
    path, which is where it would actually cost money."""
    monkeypatch.setenv("SECTOR_TREND_ENABLED", "true")
    session = TradingSession()
    session.sector.config = sc.from_environment()
    assert session.engine_equity(1_000.0) == pytest.approx(800.0)

    monkeypatch.setenv("SECTOR_TREND_ENABLED", "false")
    session.sector.config = sc.from_environment()
    assert session.engine_equity(1_000.0) == pytest.approx(1_000.0)


def test_the_sleeve_and_the_divider_agree_on_the_sleeves_size(monkeypatch):
    """Two places compute the sleeve's equity -- the divider, and the sleeve's
    own config. They must not drift apart, or the engine would be told to
    stand aside for capital the sleeve never used."""
    monkeypatch.setenv("SECTOR_TREND_ENABLED", "true")
    monkeypatch.setenv("SECTOR_TREND_ALLOCATION", "0.35")
    session = TradingSession()
    session.sector.config = sc.from_environment()
    session.engine_equity(1_000.0)

    from_divider = session.capital.equity_for("sector_trend")
    from_sleeve = session.sector.config.sleeve_equity(1_000.0)
    assert from_divider == pytest.approx(from_sleeve)
    assert from_divider == pytest.approx(350.0)


def test_the_shares_are_fixed_for_the_life_of_a_session():
    """A share that moved mid-session would look to the daily-loss check like
    a sudden loss. The config is read once at construction, so it cannot."""
    session = TradingSession()
    first = session.engine_equity(1_000.0)
    second = session.engine_equity(1_000.0)
    assert first == pytest.approx(second)


def test_the_snapshot_shows_where_the_money_went(monkeypatch):
    monkeypatch.setenv("SECTOR_TREND_ENABLED", "true")
    session = TradingSession()
    session.sector.config = sc.from_environment()
    session.engine_equity(70.0)
    block = session.snapshot()["capital"]
    assert block["shares"][ENGINE] == pytest.approx(0.80)
    assert block["allocated"]["sector_trend"] == pytest.approx(14.0)
