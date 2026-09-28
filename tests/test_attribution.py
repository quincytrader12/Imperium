"""Which strategy earned what.

Every expected figure below is worked out by hand in the test, so a change to
the accounting has to be argued with arithmetic rather than accepted because
the new number looks plausible.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from imperium.execution.attribution import (
    UNATTRIBUTED, Attribution, StrategyBook,
)


def _buy(book, symbol, qty, price, strategy="", ref=0.0, ts=1.0):
    return book.book(symbol=symbol, side="BUY", quantity=qty, price=price,
                     ts=ts, strategy=strategy, reference_price=ref)


def _sell(book, symbol, qty, price, strategy="", ref=0.0, ts=2.0):
    return book.book(symbol=symbol, side="SELL", quantity=qty, price=price,
                     ts=ts, strategy=strategy, reference_price=ref)


# -- the rule: exits belong to whoever took the risk -----------------------


def test_an_unnamed_exit_is_booked_to_the_strategy_that_opened_it():
    """The give-back ratchet, the overnight exit and a retirement flatten all
    sell without naming a strategy. Booking them to a row of their own would
    credit whatever closed a trade with the profit of whoever found it."""
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="trend")
    owner = _sell(book, "AAPL", 10, 110.0)          # the ratchet closing it

    assert owner == "trend"
    assert book.records["trend"].realised == pytest.approx(100.0)
    assert set(book.records) == {"trend"}, (
        "an exit created a row of its own and took the credit")


def test_two_strategies_on_different_symbols_are_kept_apart():
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="trend")
    _buy(book, "BTC/USD", 0.01, 50_000.0, strategy="cross_section")
    _sell(book, "AAPL", 10, 90.0)                   # trend loses 100
    _sell(book, "BTC/USD", 0.01, 52_000.0)          # cross-section makes 20

    assert book.records["trend"].realised == pytest.approx(-100.0)
    assert book.records["cross_section"].realised == pytest.approx(20.0)


# -- average cost -----------------------------------------------------------


def test_adding_to_a_position_averages_its_cost():
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="trend")
    _buy(book, "AAPL", 10, 110.0, strategy="trend")
    lot = book.records["trend"].lots["AAPL"]
    assert lot.quantity == pytest.approx(20)
    assert lot.avg_price == pytest.approx(105.0)      # (1000 + 1100) / 20


def test_a_partial_exit_realises_only_what_it_closes():
    book = StrategyBook()
    _buy(book, "AAPL", 20, 105.0, strategy="trend")
    _sell(book, "AAPL", 5, 125.0)                     # (125 - 105) * 5 = 100

    rec = book.records["trend"]
    assert rec.realised == pytest.approx(100.0)
    assert rec.lots["AAPL"].quantity == pytest.approx(15)
    assert rec.lots["AAPL"].avg_price == pytest.approx(105.0), (
        "selling part of a position changed the cost of what is left")
    assert rec.round_trips == 0, "a partial exit was counted as a round trip"


def test_unrealised_is_marked_against_average_cost():
    book = StrategyBook()
    _buy(book, "AAPL", 15, 105.0, strategy="trend")
    assert book.records["trend"].unrealised({"AAPL": 110.0}) == pytest.approx(75.0)


def test_a_symbol_with_no_price_is_not_marked_at_zero():
    """Marking a position with no quote at zero would report its whole cost as
    a loss the moment a feed dropped."""
    book = StrategyBook()
    _buy(book, "AAPL", 15, 105.0, strategy="trend")
    assert book.records["trend"].unrealised({}) == 0.0


# -- round trips, wins and losses ------------------------------------------


def test_a_round_trip_is_judged_as_a_whole_when_it_closes():
    """Two partial exits, a large loss and then a smaller gain, are one losing
    trade. Judging it by its last exit would call it a win -- the order the
    exits happened in would decide the hit rate."""
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="trend")
    _sell(book, "AAPL", 5, 80.0)       # -100
    _sell(book, "AAPL", 5, 110.0)      # +50, but the trade lost 50

    rec = book.records["trend"]
    assert rec.round_trips == 1
    assert (rec.wins, rec.losses) == (0, 1)
    assert rec.gross_loss == pytest.approx(50.0)


def test_hit_rate_and_profit_factor():
    book = StrategyBook()
    for price in (110.0, 120.0, 90.0):                 # +10, +20, -10 per share
        _buy(book, "AAPL", 1, 100.0, strategy="trend")
        _sell(book, "AAPL", 1, price)

    row = book.summary({})[0]
    assert row["hit_rate"] == pytest.approx(2 / 3)
    assert row["profit_factor"] == pytest.approx(30.0 / 10.0)


def test_no_closed_trades_is_not_reported_as_a_zero_hit_rate():
    """"Has not traded yet" and "never wins" are different facts, and the
    allocator will treat them very differently."""
    book = StrategyBook()
    _buy(book, "AAPL", 1, 100.0, strategy="trend")
    row = book.summary({})[0]
    assert row["hit_rate"] is None
    assert row["profit_factor"] is None


def test_selling_through_flat_opens_the_other_side_at_the_fill_price():
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="intraday")
    _sell(book, "AAPL", 15, 110.0, strategy="intraday")   # closes 10, shorts 5

    rec = book.records["intraday"]
    assert rec.realised == pytest.approx(100.0)
    assert rec.round_trips == 1
    lot = rec.lots["AAPL"]
    assert lot.quantity == pytest.approx(-5)
    assert lot.avg_price == pytest.approx(110.0)


def test_a_short_realises_when_bought_back_lower():
    book = StrategyBook()
    _sell(book, "XYZ", 10, 50.0, strategy="intraday")
    _buy(book, "XYZ", 10, 45.0)                       # (50 - 45) * 10
    assert book.records["intraday"].realised == pytest.approx(50.0)


# -- slippage ---------------------------------------------------------------


def test_paying_above_the_decision_price_is_a_cost_on_a_buy():
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.10, strategy="trend", ref=100.0)
    assert book.records["trend"].slippage == pytest.approx(1.0)


def test_selling_below_the_decision_price_is_a_cost_on_a_sell():
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="trend", ref=100.0)
    _sell(book, "AAPL", 10, 109.90, ref=110.0)
    assert book.records["trend"].slippage == pytest.approx(1.0)


def test_slippage_is_not_charged_twice():
    """It is already in the fill price, so already in realised P&L. Adding it
    again would make every strategy look worse by exactly its crossing cost."""
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.10, strategy="trend", ref=100.0)
    _sell(book, "AAPL", 10, 110.0, ref=110.0)
    rec = book.records["trend"]
    assert rec.realised == pytest.approx((110.0 - 100.10) * 10)


# -- the venue is the truth -------------------------------------------------


def test_a_position_found_at_start_is_unattributed_not_guessed():
    book = StrategyBook()
    corrected = book.sync({"AAPL": (5.0, 100.0)})
    assert corrected == ["AAPL"]
    assert book.owner("AAPL") == UNATTRIBUTED
    assert book.records[UNATTRIBUTED].lots["AAPL"].quantity == pytest.approx(5)


def test_a_quantity_the_fills_did_not_explain_is_resized_and_counted():
    """A partial auction fill, a crypto fee taken in the asset, a trade made by
    hand. The price of what happened is unknown, so it is not booked as profit
    or loss -- but it is counted, so it cannot accumulate quietly."""
    book = StrategyBook()
    _buy(book, "BTC/USD", 0.010, 50_000.0, strategy="cross_section")
    book.sync({"BTC/USD": (0.00998, 50_000.0)})         # fee taken in coin

    rec = book.records["cross_section"]
    assert rec.lots["BTC/USD"].quantity == pytest.approx(0.00998)
    assert rec.realised == 0.0, "a correction was booked as P&L"
    assert book.corrections == 1


def test_a_position_the_venue_no_longer_holds_is_closed_without_inventing_pnl():
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="trend")
    book.sync({})
    assert book.records["trend"].lots["AAPL"].is_flat
    assert book.records["trend"].realised == 0.0
    assert book.owner("AAPL") == UNATTRIBUTED


def test_agreement_with_the_venue_is_not_a_correction():
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="trend")
    assert book.sync({"AAPL": (10.0, 100.0)}) == []
    assert book.corrections == 0


# -- daily marks --------------------------------------------------------------


def test_marking_the_same_day_twice_replaces_rather_than_adds():
    """A restart that marks again must not put two points on one day. It reads
    as a day with no change and flatters every volatility figure built on it."""
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="trend")
    book.mark_day("2026-09-25", {"AAPL": 101.0})
    book.mark_day("2026-09-25", {"AAPL": 102.0})
    assert book.records["trend"].daily == [("2026-09-25", pytest.approx(20.0))]


def test_drawdown_is_the_largest_fall_from_a_high():
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="trend")
    for day, price in (("d1", 110.0), ("d2", 104.0), ("d3", 112.0), ("d4", 101.0)):
        book.mark_day(day, {"AAPL": price})
    # peak 120 at d3, then 10 at d4 -> 110
    assert book.records["trend"].max_drawdown() == pytest.approx(110.0)


# -- persistence --------------------------------------------------------------


def test_the_book_survives_a_restart_intact():
    book = StrategyBook()
    _buy(book, "AAPL", 10, 100.0, strategy="trend", ref=99.9)
    _sell(book, "AAPL", 4, 110.0)
    book.mark_day("2026-09-25", {"AAPL": 105.0})

    back = StrategyBook.from_dict(book.as_dict())
    assert back.summary({"AAPL": 105.0}) == book.summary({"AAPL": 105.0})
    assert back.owner("AAPL") == "trend", (
        "after a restart an exit would be booked to nobody")


def test_a_corrupt_record_is_dropped_not_guessed_at():
    back = StrategyBook.from_dict({"records": {"trend": "nonsense",
                                               "overnight": {"realised": "x"}}})
    assert back.records == {}


# -- reading fills off the broker --------------------------------------------


def _fill(symbol, side, qty, price, strategy="", ref=0):
    return SimpleNamespace(symbol=symbol, side=side, quantity=Decimal(str(qty)),
                           price=Decimal(str(price)), ts=1.0, strategy=strategy,
                           reference_price=Decimal(str(ref)))


class _Broker:
    def __init__(self, mode="paper"):
        self.fills: list = []
        self.fills_total = 0
        self.mode = SimpleNamespace(value=mode)
        self.positions: dict = {}

    def add(self, fill):
        self.fills.append(fill)
        self.fills_total += 1


def test_every_fill_is_booked_once_whatever_placed_it():
    attr = Attribution()
    broker = _Broker()
    broker.add(_fill("AAPL", "BUY", 10, 100, strategy="trend"))
    assert attr.consume(broker) == 1
    assert attr.consume(broker) == 0, "a fill was booked twice"
    broker.add(_fill("AAPL", "SELL", 10, 110))
    assert attr.consume(broker) == 1
    assert attr.book_for("paper").records["trend"].realised == pytest.approx(100.0)


def test_the_exit_is_labelled_on_the_fill_for_the_journal():
    attr = Attribution()
    broker = _Broker()
    broker.add(_fill("AAPL", "BUY", 10, 100, strategy="trend"))
    exit_fill = _fill("AAPL", "SELL", 10, 110)
    broker.add(exit_fill)
    attr.consume(broker)
    assert exit_fill.strategy == "trend"


def test_a_new_broker_is_read_from_its_first_fill():
    """A mode switch builds a fresh broker whose count starts again at zero.
    Carrying the old cursor over would skip its first fills as already seen."""
    attr = Attribution()
    old = _Broker("paper")
    for _ in range(3):
        old.add(_fill("AAPL", "BUY", 1, 100, strategy="trend"))
    attr.consume(old)

    new = _Broker("paper")
    new.add(_fill("MSFT", "BUY", 1, 300, strategy="intraday"))
    assert attr.consume(new) == 1


def test_fills_that_left_the_ring_unread_are_counted():
    attr = Attribution()
    broker = _Broker()
    broker.add(_fill("AAPL", "BUY", 1, 100, strategy="trend"))
    broker.fills_total += 5          # five more happened and fell off the ring
    attr.consume(broker)
    assert attr.missed == 5


def test_modes_are_kept_in_separate_books():
    """A dry run's simulated fills are not evidence of the same quality as the
    venue's paper account, and one record would let the easier flatter the
    harder."""
    attr = Attribution()
    dry, paper = _Broker("dry_run"), _Broker("paper")
    dry.add(_fill("AAPL", "BUY", 1, 100, strategy="trend"))
    paper.add(_fill("MSFT", "BUY", 1, 300, strategy="intraday"))
    attr.consume(dry)
    attr.consume(paper)
    assert set(attr.book_for("dry_run").records) == {"trend"}
    assert set(attr.book_for("paper").records) == {"intraday"}
