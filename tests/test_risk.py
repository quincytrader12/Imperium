"""Risk limits, sizing, and the boundary an optimiser may not cross."""

from __future__ import annotations

import math

import numpy as np
import pytest

from imperium.execution.risk import (
    OPTIMISABLE_PARAMETERS, RiskLimits, assert_search_space_is_safe,
    risk_limit_field_names,
)
from imperium.execution.sizing import annualised_volatility, size_position
from imperium.strategy.signals import StrategyParams

CRYPTO_YEAR = 365 * 24 * 3600
EQUITY_YEAR = int(252 * 6.5 * 3600)


def test_no_risk_limit_is_optimisable():
    """Prevents: an optimiser widening its own position cap. Compared against the
    dataclass's real fields, so adding a limit later cannot quietly make it
    searchable by being forgotten."""
    assert risk_limit_field_names() & OPTIMISABLE_PARAMETERS == frozenset()
    with pytest.raises(ValueError, match="may not contain risk limits"):
        assert_search_space_is_safe({"fast_window", "max_position_weight"})


def test_every_strategy_parameter_is_declared_optimisable():
    """Prevents: the reverse mistake -- a tunable parameter that no search space
    knows about, so it silently never gets tuned."""
    from dataclasses import fields

    names = {f.name for f in fields(StrategyParams)}
    assert names <= OPTIMISABLE_PARAMETERS, names - OPTIMISABLE_PARAMETERS


def test_a_search_space_cannot_contain_an_unrecognised_name():
    """Prevents: a typo in a search space silently tuning nothing."""
    with pytest.raises(ValueError, match="unknown search parameters"):
        assert_search_space_is_safe({"fastwindow"})


def test_a_single_position_cannot_be_allowed_to_breach_the_book_ceiling():
    """Prevents: a configuration in which one symbol may hold more than the whole
    book is permitted to hold, which makes the gross ceiling decorative."""
    with pytest.raises(ValueError, match="cannot exceed"):
        RiskLimits(max_position_weight=0.9, max_gross_exposure=0.5)


def test_crypto_volatility_is_annualised_over_a_market_that_never_closes():
    """Prevents: using an equity calendar for a crypto pair. It understates
    annualised volatility by about 2.3x, and since vol targeting divides by it,
    every position ends up at roughly 43% of target -- a bot that looks like it
    is working while running at less than half its intended risk."""
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, 500)
    crypto = annualised_volatility(r, CRYPTO_YEAR, 60)
    equity = annualised_volatility(r, EQUITY_YEAR, 60)
    assert crypto / equity == pytest.approx(2.31, abs=0.05)
    assert crypto == pytest.approx(0.01 * math.sqrt(CRYPTO_YEAR / 60), rel=0.1)


def test_each_asset_class_declares_its_own_trading_calendar():
    """Prevents one calendar being applied to both classes.

    A mutation test found the earlier version of this gap: the annualisation
    test below uses its own constants, so it kept passing when the shipped
    calendar changed. The value that actually reaches the sizer is the one on
    the asset-class spec, so that is what this asserts -- and it asserts that
    the two classes differ, because a single figure cannot be right for a
    market that closes and one that does not.
    """
    from imperium.venues.assets import AssetClass, spec_for

    crypto = spec_for(AssetClass.CRYPTO)
    equity = spec_for(AssetClass.US_EQUITY)

    assert crypto.seconds_per_year == CRYPTO_YEAR, (
        "crypto never closes, so a year is every second of it"
    )
    assert equity.seconds_per_year == EQUITY_YEAR, (
        "equities trade 252 days of 6.5 hours"
    )
    assert crypto.seconds_per_year > equity.seconds_per_year * 5


def test_a_short_on_a_long_only_venue_is_clamped_to_flat_not_to_a_small_long():
    """Prevents: turning a short signal into a small long. Binance Spot has
    nothing to borrow, so a short is not a risky position -- it is a rejected
    order every bar for as long as the signal points down. And a small long is
    the opposite of what the strategy asked for."""
    rng = np.random.default_rng(2)
    result = size_position(
        signal=-0.9, returns=rng.normal(0, 0.01, 300), price=100.0, atr=1.0,
        limits=RiskLimits(), seconds_per_year=CRYPTO_YEAR, bar_seconds=60,
        allows_short=False,
    )
    assert result.weight == 0.0
    assert "long-only" in result.reason


def test_the_binding_constraint_wins_and_is_named():
    """Prevents: applying only one of the two bounds. Volatility targeting bounds
    the portfolio's volatility; the ATR cap bounds what one trade can lose.
    Neither subsumes the other, so both apply and the tighter one wins."""
    rng = np.random.default_rng(3)
    quiet = rng.normal(0, 0.0005, 400)   # low vol -> vol target wants a lot
    tight = size_position(
        signal=1.0, returns=quiet, price=100.0, atr=8.0, limits=RiskLimits(),
        seconds_per_year=CRYPTO_YEAR, bar_seconds=60, allows_short=False,
    )
    # An 8.0 ATR on a 100 price is a 20% stop; 0.5% risk allows only 2.5%.
    assert tight.binding == "ATR risk cap"
    assert tight.weight < tight.vol_target_weight

    loud = rng.normal(0, 0.05, 400)
    volbound = size_position(
        signal=1.0, returns=loud, price=100.0, atr=0.01, limits=RiskLimits(),
        seconds_per_year=CRYPTO_YEAR, bar_seconds=60, allows_short=False,
    )
    assert volbound.binding == "volatility target"


def test_a_position_never_exceeds_the_per_symbol_cap():
    """Prevents: a very low volatility symbol being sized without bound. Vol
    targeting divides by volatility, so as volatility approaches zero the
    requested weight approaches infinity."""
    result = size_position(
        signal=1.0, returns=np.full(300, 1e-9), price=100.0, atr=0.001,
        limits=RiskLimits(max_position_weight=0.2), seconds_per_year=CRYPTO_YEAR,
        bar_seconds=60, allows_short=False,
    )
    assert result.weight <= 0.2
