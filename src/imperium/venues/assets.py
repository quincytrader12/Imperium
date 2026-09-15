"""Asset classes, and everything that differs between them.

Crypto and equities are not the same instrument wearing different tickers, and
running one strategy over both is how a book loses money in two markets at
once. The differences that actually change the arithmetic:

* **The calendar.** Crypto never closes, so a year is every second of it.
  Equities trade 6.5 hours a day, 252 days a year -- ``252 * 6.5 * 3600``
  seconds. Annualising an equity's per-bar volatility over the crypto figure
  overstates it by ``sqrt(8760 / 1638) = 2.31x``, which then sizes every
  position at 43% of target. This is measured in the tests, not asserted.

* **Overnight gaps.** An equity's close-to-open move is not a one-minute
  return, and treating it as one inflates the volatility estimate and corrupts
  the variance ratio -- the gap dominates every statistic computed over the
  window. Crypto has no such seam. See ``excludes_session_gaps``.

* **Costs.** US equities are commission-free at Alpaca but carry regulatory
  fees on the **sell leg only** (SEC Section 31 and FINRA TAF). Crypto pays a
  spread-based commission on both legs. These are different enough that a
  single cost gate would admit the wrong symbols in one class or the other.

* **Shorting.** An equity may be shortable, hard-to-borrow, or not shortable at
  all, and the venue says which per asset. Alpaca crypto cannot be shorted at
  all, so a negative target is clamped to flat rather than to a small long.

* **The regime null.** Equity minute bars have an opening auction, a
  U-shaped intraday volume profile and a closing cross. Their null distribution
  is not crypto's, so each class carries its own measured calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class AssetClass(str, Enum):
    US_EQUITY = "us_equity"
    CRYPTO = "crypto"
    #: Recognised so that an options position is never silently treated as
    #: equity, but not tradeable -- see OPTION_SPEC.
    US_OPTION = "us_option"


#: Seconds of *trading* in a year, per class. The denominator of every
#: annualisation in the program.
CRYPTO_SECONDS_PER_YEAR = 365 * 24 * 3600           # 31,536,000
EQUITY_SECONDS_PER_YEAR = int(252 * 6.5 * 3600)     #  5,896,800


@dataclass(frozen=True)
class CostModel:
    """What a round trip costs, in basis points, for one asset class."""

    #: Commission per leg. Zero for US equities at Alpaca.
    commission_bps: Decimal
    #: Regulatory and exchange fees charged once, on the sell leg only.
    #: For US equities this is SEC Section 31 plus FINRA TAF.
    sell_side_bps: Decimal
    #: A default spread used only until a live quote arrives, and always
    #: reported as assumed when it is in play.
    default_spread_bps: Decimal
    assumed: bool = True
    source: str = "published schedule, not confirmed against this account"


@dataclass(frozen=True)
class AssetClassSpec:
    """Everything that differs between asset classes, as data."""

    asset_class: AssetClass
    display_name: str
    seconds_per_year: int
    #: True where a bar series contains overnight and weekend seams that must
    #: not be read as returns.
    excludes_session_gaps: bool
    #: Whether the class can be shorted at all. Per-symbol borrow availability
    #: is a separate, stricter check.
    shortable: bool
    trades_continuously: bool
    cost_model: CostModel
    #: Which measured calibration the regime classifier must use.
    calibration_key: str
    #: Smallest price increment above $1. Sub-dollar US equities quote in
    #: $0.0001, which is handled in the rounding rather than here.
    tick_size: Decimal
    #: Whether fractional quantities are accepted.
    fractional: bool
    tradeable: bool = True
    note: str = ""


EQUITY_SPEC = AssetClassSpec(
    asset_class=AssetClass.US_EQUITY,
    display_name="US equity",
    seconds_per_year=EQUITY_SECONDS_PER_YEAR,
    excludes_session_gaps=True,
    shortable=True,
    trades_continuously=False,
    cost_model=CostModel(
        # Alpaca charges no commission on US equities. The regulatory fees are
        # real and are charged on sells only; at published rates they come to
        # roughly 1bp of the sell notional, which is small but not nothing
        # against an intraday edge.
        commission_bps=Decimal("0"),
        sell_side_bps=Decimal("1.0"),
        default_spread_bps=Decimal("3.0"),
        assumed=True,
        source="Alpaca commission-free equities; SEC Section 31 and FINRA TAF "
               "estimated on the sell leg, not confirmed against this account",
    ),
    calibration_key="us_equity",
    tick_size=Decimal("0.01"),
    fractional=True,
)

CRYPTO_SPEC = AssetClassSpec(
    asset_class=AssetClass.CRYPTO,
    display_name="Crypto",
    seconds_per_year=CRYPTO_SECONDS_PER_YEAR,
    excludes_session_gaps=False,
    # Alpaca crypto is long-only: there is nothing to borrow, so a short is not
    # a risky position, it is a rejected order every bar.
    shortable=False,
    trades_continuously=True,
    cost_model=CostModel(
        commission_bps=Decimal("25"),
        sell_side_bps=Decimal("0"),
        default_spread_bps=Decimal("8.0"),
        assumed=True,
        source="Alpaca crypto published taker tier, not confirmed against this "
               "account",
    ),
    calibration_key="crypto",
    tick_size=Decimal("0.01"),
    fractional=True,
)

#: An option contract is a hundred shares. This is the single most important
#: number for a small account: it makes the position size *quantised*, so the
#: smallest possible trade is a hundred times the quoted premium and cannot be
#: reduced any further.
OPTION_CONTRACT_MULTIPLIER = 100

#: Per-contract regulatory pass-through, round trip, in dollars.
#:
#: Alpaca charges **no commission** on options -- an earlier version of this
#: program asserted a $0.65 per-contract commission and concluded on that basis
#: that options were unusable. The premise was wrong. What remains is small:
#: OCC clearing around $0.025 and the Options Regulatory Fee around $0.02 on
#: each leg, plus FINRA's TAF of about $0.0033 on the sell. Assumed from
#: published schedules, not confirmed against a live account.
#:
#: Note what shape this cost is: a **flat fee per contract**, not a rate. Nine
#: cents is 18bp of a $50 contract and 1.8bp of a $500 one, so the cheapest
#: contracts -- the only ones a small account can reach -- are the ones where
#: it bites hardest.
OPTION_FEES_PER_CONTRACT_ROUND_TRIP = Decimal("0.093")

#: Account equity below which no option position is worth opening.
#:
#: Derived, not chosen. One contract must fit inside the per-symbol cap, so at
#: a 20% cap an account of E can afford a premium of E * 0.20 / 100. Below
#: roughly $1,500 that only reaches contracts under $3, where the quoted spread
#: is a large fraction of the premium: a $0.50 contract quoted 0.48/0.52 must
#: gain over 12% before the round trip breaks even, and a $0.20 contract over
#: 23%. Those are not positions, they are lottery tickets with a house edge.
#: ``scripts/option_affordability.py`` computes the whole table.
MIN_OPTION_ACCOUNT_EQUITY = Decimal("1500")


OPTION_SPEC = AssetClassSpec(
    asset_class=AssetClass.US_OPTION,
    display_name="US option",
    seconds_per_year=EQUITY_SECONDS_PER_YEAR,
    excludes_session_gaps=True,
    shortable=False,
    trades_continuously=False,
    cost_model=CostModel(
        # Commission-free at Alpaca. The real cost is the spread, and it is
        # quoted in cents on a premium of a few dollars, so as a *rate* it
        # depends entirely on which contract is bought -- roughly 200bp round
        # trip on a liquid at-the-money contract and 4,000bp on a far
        # out-of-the-money one. The default here is the liquid end; the gate
        # will refuse anything cheaper on its own arithmetic.
        commission_bps=Decimal("0"),
        sell_side_bps=Decimal("0"),
        default_spread_bps=Decimal("200.0"),
        assumed=True,
        source="Alpaca is commission-free on options; per-contract regulatory "
               "fees are assumed from published schedules and the spread is "
               "not calibrated",
    ),
    calibration_key="us_equity",
    tick_size=Decimal("0.01"),
    fractional=False,
    tradeable=False,
    note=(
        "Options are recognised so that an option position is never silently "
        "sized as though it were its underlying, but they are NOT traded, for "
        "three reasons in order of how binding they are.\n\n"
        "First, affordability. A contract is a hundred shares, so position "
        "size is quantised and the smallest possible trade is a hundred times "
        "the premium. Under roughly $1,500 of equity an account can only reach "
        "contracts cheap enough that the quoted spread is a large fraction of "
        "the premium -- a $0.50 contract must gain over 12% simply to break "
        "even on the round trip.\n\n"
        "Second, the model. An option's return is a non-linear function of the "
        "underlying's, so the variance-ratio regime test and the "
        "volatility-target sizer -- both of which assume returns are the thing "
        "being forecast -- do not carry over.\n\n"
        "Third, theta. The strategies here forecast drift in the underlying, "
        "and an option pays for calendar time whether or not the drift "
        "arrives. Trading options needs a model of implied volatility, not a "
        "different threshold on this one."
    ),
)

SPECS: dict[AssetClass, AssetClassSpec] = {
    AssetClass.US_EQUITY: EQUITY_SPEC,
    AssetClass.CRYPTO: CRYPTO_SPEC,
    AssetClass.US_OPTION: OPTION_SPEC,
}


def spec_for(asset_class: AssetClass | str) -> AssetClassSpec:
    key = AssetClass(asset_class)
    return SPECS[key]


def classify_symbol(symbol: str) -> AssetClass:
    """Infer the asset class from the venue's own symbol vocabulary.

    Alpaca spells crypto pairs with a slash (``BTC/USD``) and equities as a
    bare ticker (``AAPL``); option contracts use the OCC 21-character form
    (``AAPL241220C00150000``). Guessing wrong picks the wrong calendar, the
    wrong costs and the wrong calibration, so this is deliberately narrow and
    falls back to equity only for things that look like plain tickers.
    """
    s = symbol.strip().upper()
    if "/" in s:
        return AssetClass.CRYPTO
    # OCC option symbols: root, 6-digit date, C/P, 8-digit strike.
    if len(s) >= 15 and s[-9] in ("C", "P") and s[-8:].isdigit() and s[-15:-9].isdigit():
        return AssetClass.US_OPTION
    return AssetClass.US_EQUITY
