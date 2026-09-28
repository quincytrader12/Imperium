"""Why crypto, equities and options cannot share one strategy.

Every test here defends a difference that changes the arithmetic, not the
labelling. If these collapse, the program runs one strategy over two markets
and is wrong in at least one of them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imperium.execution.bars import Bar, BarSeries
from imperium.strategy.regime import CalibrationMissing, thresholds_for
from imperium.venues.assets import (
    AssetClass, CRYPTO_SECONDS_PER_YEAR, EQUITY_SECONDS_PER_YEAR,
    classify_symbol, spec_for,
)


def test_the_two_classes_annualise_over_different_years():
    """Prevents the error the brief singles out: using the crypto calendar for
    an equity.

    Crypto never closes, so a year is every second of it. Equities trade 6.5
    hours on 252 days. Annualising an equity over the crypto figure overstates
    its volatility by 2.31x, and since volatility is the denominator of the
    volatility-target sizer, every position ends up at roughly 43% of target --
    a book that looks like it is working while running at under half its
    intended risk.
    """
    ratio = math.sqrt(CRYPTO_SECONDS_PER_YEAR / EQUITY_SECONDS_PER_YEAR)
    assert ratio == pytest.approx(2.31, abs=0.02)
    assert 1.0 / ratio == pytest.approx(0.43, abs=0.02)

    assert spec_for(AssetClass.CRYPTO).seconds_per_year == CRYPTO_SECONDS_PER_YEAR
    assert spec_for(AssetClass.US_EQUITY).seconds_per_year == EQUITY_SECONDS_PER_YEAR


def test_an_overnight_gap_is_not_a_one_minute_return():
    """Prevents an equity's close-to-open move being read as a one-minute
    return.

    A gap is many times a one-minute move. Left in, it inflates the volatility
    estimate that sizes every position, and a handful of huge pseudo-returns
    dominate the variance ratio that chooses the strategy. Crypto has no such
    seam, which is exactly why this is a per-class decision.
    """
    series = BarSeries("AAPL", bar_seconds=60)
    rng = np.random.default_rng(0)
    price, t = 100.0, 0
    for session in range(3):
        for _ in range(100):
            price *= float(np.exp(rng.normal(0, 0.0006)))
            series.add(Bar(t, price, price * 1.001, price * 0.999, price,
                           1000.0, closed=True))
            t += 60_000
        price *= 1.02                       # the overnight gap
        t += int(17.5 * 3600 * 1000)

    naive = series.log_returns()
    clean = series.log_returns(exclude_session_gaps=True)

    assert clean.size < naive.size, "the gap-spanning returns must be dropped"
    assert abs(naive).max() > 5 * abs(clean).max(), (
        "the gap should dominate the naive series"
    )
    assert naive.std(ddof=1) > 1.3 * clean.std(ddof=1), (
        f"gaps inflate volatility {naive.std(ddof=1) / clean.std(ddof=1):.2f}x"
    )


def test_only_the_equity_class_asks_for_gap_exclusion():
    """Prevents dropping returns from a 24/7 series, where a large interval
    between bars means missing data rather than a session boundary."""
    assert spec_for(AssetClass.US_EQUITY).excludes_session_gaps is True
    assert spec_for(AssetClass.CRYPTO).excludes_session_gaps is False


def test_crypto_is_long_only_and_equities_are_not():
    """Prevents clamping an equity short to flat, and prevents proposing a
    crypto short that has nothing to borrow behind it."""
    assert spec_for(AssetClass.CRYPTO).shortable is False
    assert spec_for(AssetClass.US_EQUITY).shortable is True


def test_the_cost_models_differ_in_shape_not_only_in_rate():
    """Prevents one cost model across both classes.

    A US equity pays no commission and a small regulatory fee on the sell leg
    only, so its round trip is almost all spread. Crypto pays commission on
    both legs that dwarfs the spread. These are different shapes, and a single
    model misprices whichever class it was not built for.
    """
    equity = spec_for(AssetClass.US_EQUITY).cost_model
    crypto = spec_for(AssetClass.CRYPTO).cost_model

    assert equity.commission_bps == 0
    assert equity.sell_side_bps > 0
    assert crypto.commission_bps > 0
    assert crypto.sell_side_bps == 0
    assert crypto.commission_bps > equity.sell_side_bps * 10


def test_each_class_has_its_own_measured_thresholds():
    """Prevents one calibration serving both.

    Equity minute bars have an opening auction, a U-shaped intraday volatility
    profile and overnight seams; crypto has none of them. Each class is fitted
    against a null that reflects how it actually behaves, and the fitted bars
    differ -- so handing one class the other's thresholds changes its error
    rate directly.
    """
    crypto = thresholds_for("crypto")
    equity = thresholds_for("us_equity")

    assert crypto["vr_z"]["reject_trend_above"] != equity["vr_z"]["reject_trend_above"]
    assert crypto["adf"]["stationary_below"] != equity["adf"]["stationary_below"]
    for th in (crypto, equity):
        assert th["vr_z"]["reject_revert_below"] < 0 < th["vr_z"]["reject_trend_above"]
        assert abs(th["hurst"]["null_median"] - 0.5) > 0.05, (
            "the Hurst null is not centred on 0.5 for either class"
        )


def test_an_unknown_asset_class_has_no_thresholds_to_borrow():
    """Prevents silently falling back to another class's calibration. Options
    are recognised but not calibrated, and borrowing the equity thresholds for
    them would be a fabricated measurement."""
    with pytest.raises(CalibrationMissing):
        thresholds_for("us_option")


def test_options_are_recognised_but_not_traded():
    """Prevents an option being sized as though it were its underlying.

    An option's return is a non-linear function of the underlying's, so the
    variance-ratio regime test and the volatility-target sizer -- both of which
    assume returns are the thing being forecast -- do not carry over. The class
    is recognised so that a position in one is never mistaken for equity, and
    marked untradeable so nothing sizes it.
    """
    option = spec_for(AssetClass.US_OPTION)
    assert option.tradeable is False
    assert "non-linear" in option.note
    assert spec_for(AssetClass.US_EQUITY).tradeable is True
    assert spec_for(AssetClass.CRYPTO).tradeable is True


@pytest.mark.parametrize("symbol,expected", [
    ("AAPL", AssetClass.US_EQUITY),
    ("SPY", AssetClass.US_EQUITY),
    ("BRK.B", AssetClass.US_EQUITY),
    ("BTC/USD", AssetClass.CRYPTO),
    ("ETH/USD", AssetClass.CRYPTO),
    ("AAPL241220C00150000", AssetClass.US_OPTION),
    ("SPY250117P00400000", AssetClass.US_OPTION),
])
def test_the_class_is_read_from_the_venue_symbol_vocabulary(symbol, expected):
    """Prevents guessing the class wrong, which picks the wrong calendar, the
    wrong costs and the wrong calibration all at once. Alpaca writes crypto
    pairs with a slash and options in the OCC form."""
    assert classify_symbol(symbol) is expected
