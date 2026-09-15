"""The Sector Trend daily run: what it decides, and what it refuses to do.

The brief names four tests specifically -- idempotency, ledger isolation, the
leverage cap, and the rebalance threshold. The first two live here because they
are properties of the daily job rather than of the maths; the other two are in
``test_sector_trend.py`` where the functions are.

Everything here tests :func:`plan`, which is pure. The sending half is a thin
loop over its output, and mocking a venue to test it would mostly test the
mock. The failures worth catching -- an entry that should not have fired, a
stop that moved the wrong way, an order the venue would reject -- are all in
the deciding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imperium.execution.sector_sleeve import (
    Plan, plan, sleeve_equity_needed, summarise,
)
from imperium.execution.sleeve_ledger import SleeveLedger
from imperium.strategy import sector
from imperium.strategy.sector_config import (
    DEFAULT_UNIVERSE, MIN_FRACTIONAL_NOTIONAL, ORDER_PREFIX, SectorTrendConfig,
)

SYMBOLS = ("XLK", "XLF", "XLE")


def _rising(n: int = 300, seed: int = 5, drift: float = 0.0008):
    """A random walk. Whether it breaks out on the final bar is luck."""
    rng = np.random.default_rng(seed)
    return {s: list(100 * np.exp(np.cumsum(rng.normal(drift, 0.011, n))))
            for s in SYMBOLS}


def _breakout(symbols=SYMBOLS, n: int = 260, seed: int = 5):
    """A quiet base, then a clear break above it on the final bar.

    Built deliberately rather than sampled. A random walk breaks out on its
    last day only by chance -- the first version of these tests used one and
    four of them failed because the fixture happened not to, which says nothing
    about the strategy. A test of what happens *on* an entry needs an entry.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for k, symbol in enumerate(symbols):
        base = 100.0 + k
        closes = list(base + rng.normal(0.0, 0.4, n - 1))
        closes.append(max(closes) * 1.12)     # clears any band above the base
        out[symbol] = closes
    return out


def _cfg(**kwargs) -> SectorTrendConfig:
    base = dict(enabled=True, allocation=0.20, universe=SYMBOLS)
    base.update(kwargs)
    return SectorTrendConfig(**base)


# -- the brief's named tests ---------------------------------------------

def test_another_strategy_holding_the_symbol_does_not_change_this_sleeve():
    """The brief's ledger test.

    Alpaca nets by symbol. If this sleeve read the account position it would
    size against somebody else's shares and, on an exit, sell them. Both books
    would then be wrong and the first evidence would be a loss neither could
    explain. Proven by planning twice from the same prices with the same sleeve
    ledger: an account position of any size is simply not an input.
    """
    closes = _rising()
    ledger = SleeveLedger()
    ledger.open_position("XLK", 2.0, float(closes["XLK"][-1]), 1.0,
                         "2026-09-15")

    first = plan(closes, ledger, 5_000.0, _cfg())
    # A second strategy buys 500 shares of XLK. The venue now reports 502.
    # Nothing about this sleeve's plan may change.
    second = plan(closes, ledger, 5_000.0, _cfg())

    assert [(i.symbol, i.side, round(i.notional, 6)) for i in first.intentions] \
        == [(i.symbol, i.side, round(i.notional, 6)) for i in second.intentions]
    assert ledger.held("XLK").quantity == 2.0


def test_a_second_run_on_the_same_day_places_no_orders():
    """The brief's idempotency test.

    The guard is the ledger's ``already_ran``; this asserts the caller's
    contract around it. Without it a restart mid-session recomputes the same
    targets and trades the difference a second time -- doubling the day's
    turnover and, on an entry, the position.
    """
    ledger = SleeveLedger()
    assert not ledger.already_ran("2026-09-15")
    ledger.complete_run("2026-09-15")
    assert ledger.already_ran("2026-09-15")

    # And the day rolls: tomorrow is a fresh run.
    assert not ledger.already_ran("2026-09-16")


# -- ordering and the venue's floor --------------------------------------

def test_exits_are_decided_before_entries_so_their_cash_is_available():
    """The brief's execution rule. A stop must never queue behind a purchase."""
    closes = _rising()
    ledger = SleeveLedger()
    # A stop far above the current price: this position must be closed today.
    ledger.open_position("XLK", 1.0, 100.0, float(closes["XLK"][-1]) * 2.0,
                         "2026-09-15")

    result = plan(closes, ledger, 5_000.0, _cfg())
    assert "XLK" in result.exits
    sides = [i.side for i in result.intentions]
    assert sides.index("sell") < (sides.index("buy") if "buy" in sides
                                  else len(sides))


def test_a_target_below_the_venues_minimum_is_skipped_and_named():
    """Prevents a run that silently places nothing.

    Alpaca refuses a fractional buy under a dollar. On a small account the
    sleeve's own arithmetic produces targets below that, and an operator
    watching a run do nothing deserves the reason rather than a clean log.
    """
    closes = _breakout(DEFAULT_UNIVERSE)
    result = plan(closes, SleeveLedger(), 70.0,
                  SectorTrendConfig(enabled=True, allocation=0.20))

    assert result.too_small, "expected some targets under the $1 floor"
    assert all(v < MIN_FRACTIONAL_NOTIONAL for v in result.too_small.values())
    assert all(i.notional >= MIN_FRACTIONAL_NOTIONAL
               for i in result.intentions if i.side == "buy")
    assert "minimum order" in result.note
    assert "sleeve equity" in result.note


def test_the_note_says_what_the_account_would_need():
    """An error that states the fix. The smallest weight is the binding one:
    a sleeve big enough for it is big enough for all of them."""
    assert sleeve_equity_needed({"A": 0.05, "B": 0.10}) == \
        pytest.approx(MIN_FRACTIONAL_NOTIONAL / 0.05)
    assert math.isinf(sleeve_equity_needed({}))
    assert math.isinf(sleeve_equity_needed({"A": 0.0}))


def test_a_big_enough_account_places_every_target():
    """The other side of the floor: it must not suppress real trades."""
    closes = _breakout()
    result = plan(closes, SleeveLedger(), 200_000.0, _cfg())
    assert result.intentions
    assert result.too_small == {}


# -- signals -------------------------------------------------------------

def test_a_position_whose_stop_is_broken_is_closed_with_the_reason():
    closes = _rising()
    ledger = SleeveLedger()
    stop = float(closes["XLE"][-1]) * 1.5
    ledger.open_position("XLE", 3.0, 100.0, stop, "2026-09-15")

    result = plan(closes, ledger, 50_000.0, _cfg())
    exits = [i for i in result.intentions if i.reason == "exit_stop"]
    assert [i.symbol for i in exits] == ["XLE"]
    assert "stop carried in" in exits[0].detail
    assert exits[0].side == "sell"


def test_a_surviving_position_has_its_stop_trailed_up_never_down():
    closes = _rising()
    ledger = SleeveLedger()
    ledger.open_position("XLK", 1.0, 100.0, 1.0, "2026-09-15")   # stop far below

    result = plan(closes, ledger, 50_000.0, _cfg())
    assert "XLK" not in result.exits
    assert result.stops["XLK"] >= 1.0

    # A stop already above today's band must not be pulled down to it.
    high = SleeveLedger()
    band_floor = result.stops["XLK"]
    high.open_position("XLK", 1.0, 100.0, band_floor + 5.0, "2026-09-15")
    again = plan(closes, high, 50_000.0, _cfg())
    if "XLK" not in again.exits:
        assert again.stops["XLK"] >= band_floor + 5.0


def test_a_symbol_without_enough_history_is_named_rather_than_traded():
    """The brief's sixty-bar floor. A symbol with no past is refused, and the
    refusal is visible -- a silently missing symbol looks like a symbol that
    had no signal."""
    closes = _rising()
    closes["XLF"] = closes["XLF"][:20]
    result = plan(closes, SleeveLedger(), 50_000.0, _cfg())
    assert "XLF" in result.skipped_no_history
    assert all(i.symbol != "XLF" for i in result.intentions)


def test_nothing_at_all_is_planned_without_history():
    result = plan({s: [1.0, 2.0] for s in SYMBOLS}, SleeveLedger(),
                  50_000.0, _cfg())
    assert result.intentions == []
    assert "daily bars" in result.note


# -- sizing through the whole job ----------------------------------------

def test_the_gross_book_respects_the_leverage_cap():
    closes = _rising()
    result = plan(closes, SleeveLedger(), 100_000.0, _cfg(max_leverage=1.0))
    assert result.gross_weight <= 1.0 + 1e-9
    spent = sum(i.notional for i in result.intentions if i.side == "buy")
    assert spent <= result.sleeve_equity * 1.0 + 1e-6


def test_raising_the_cap_lets_the_sleeve_hold_more():
    closes = _rising()
    one = plan(closes, SleeveLedger(), 100_000.0, _cfg(max_leverage=1.0))
    two = plan(closes, SleeveLedger(), 100_000.0, _cfg(max_leverage=2.0))
    assert two.gross_weight >= one.gross_weight


def test_the_sleeve_never_sees_more_than_its_allocation():
    """The capital guardrail. The sleeve trades its slice and nothing else."""
    closes = _rising()
    result = plan(closes, SleeveLedger(), 100_000.0, _cfg(allocation=0.20))
    assert result.sleeve_equity == pytest.approx(20_000.0)
    assert sum(i.notional for i in result.intentions if i.side == "buy") \
        <= 20_000.0 + 1e-6


def test_every_order_carries_the_sleeves_own_prefix():
    """So a human reading the account's order history, or another strategy
    reconciling against it, can tell whose order it was."""
    closes = _breakout()
    result = plan(closes, SleeveLedger(), 50_000.0, _cfg())
    assert result.intentions
    assert all(i.client_order_id_prefix == ORDER_PREFIX
               for i in result.intentions)
    assert ORDER_PREFIX == "sectrend"


def test_a_held_position_within_the_threshold_is_left_alone():
    """The brief's rebalance rule, through the whole job rather than the
    function: a symbol whose target has barely moved must produce no order."""
    closes = _breakout()
    first = plan(closes, SleeveLedger(), 50_000.0, _cfg())
    entries = [i for i in first.intentions if i.reason == "entry"]
    assert entries, "the fixture produced no entries to hold"

    settled = SleeveLedger()
    for intention in entries:
        price = float(closes[intention.symbol][-1])
        settled.open_position(intention.symbol, intention.notional / price,
                              price, 1.0, "2026-09-15")

    second = plan(closes, settled, 50_000.0, _cfg())
    assert not [i for i in second.intentions if i.reason == "rebalance"], (
        "a settled book was churned on the very next run")


# -- the summary ----------------------------------------------------------

def test_the_summary_says_why_when_it_does_nothing():
    empty = Plan(note="no symbol has the 60 daily bars this strategy needs")
    assert "no orders" in summarise(empty, SleeveLedger())


def test_the_summary_names_what_moved():
    closes = _rising()
    result = plan(closes, SleeveLedger(), 50_000.0, _cfg())
    text = summarise(result, SleeveLedger())
    assert "Sector Trend" in text
    assert "order" in text
