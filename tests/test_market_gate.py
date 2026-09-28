"""A closed equity market must not close the crypto book.

Reported as "I've run the terminal for close to four hours and it hasn't made
a single decision". The market-open guard was one global flag applied to every
symbol, set from ``is_open or _crypto_only()`` -- and ``_crypto_only`` asks
whether *everything* resident is crypto. Crypto is pinned resident alongside
equities, so it is almost never everything, and the flag was therefore False
every night. Every increase in exposure was clamped, on a market that never
closes, for as long as the equity market was shut.

The file header for the venue layer has said from the start that a closed
equity market must not stop a crypto book. It stopped it.
"""

from __future__ import annotations

from imperium.execution.portfolio import PortfolioAllocator
from imperium.execution.risk import RiskLimits


def _allocator(equity: float = 70.0) -> PortfolioAllocator:
    alloc = PortfolioAllocator(RiskLimits())
    alloc.equity = equity
    alloc.cash = equity
    for symbol in ("AAPL", "BTC/USD"):
        alloc.observe(symbol).admitted = True
    return alloc


def test_crypto_can_still_open_while_the_equity_market_is_shut():
    """The bug, stated as the behaviour it cost.

    Overnight is most of the day. A book that cannot open a crypto position
    between the close and the next open is a book that trades one asset class
    on one venue for six and a half hours, which is not what it was built to
    be -- and from the terminal it looks identical to a strategy that simply
    never fires.
    """
    alloc = _allocator()
    alloc.set_market_state(equities_open=False)

    result = alloc.clamp("BTC/USD", 0.10)

    assert result.weight == 0.10, (
        f"crypto was clamped to {result.weight} because the equity market is "
        f"shut — {result.reason}")
    assert not result.reduced


def test_equities_are_still_held_back_while_their_market_is_shut():
    """The guard this must not weaken. An equity order lodged at 2am does not
    reach a venue that is closed; it sits until the open and fills at a price
    nothing in the decision anticipated."""
    alloc = _allocator()
    alloc.set_market_state(equities_open=False)

    result = alloc.clamp("AAPL", 0.10)

    assert result.weight == 0.0
    assert result.reduced
    assert "closed" in result.reason


def test_an_exit_is_never_blocked_by_a_closed_market_either():
    """A cap that can block an exit traps you in a losing position. The market
    guard sits after the reduction check for this reason, and must stay there
    however the asset class is decided."""
    alloc = _allocator()
    alloc.set_market_state(equities_open=False)
    alloc.observe("AAPL").weight = 0.30

    result = alloc.clamp("AAPL", 0.0)

    assert result.weight == 0.0
    assert not result.reduced


def test_both_classes_open_when_the_equity_market_is():
    alloc = _allocator()
    alloc.set_market_state(equities_open=True)

    assert alloc.clamp("AAPL", 0.10).weight == 0.10
    assert alloc.clamp("BTC/USD", 0.10).weight == 0.10


# ---------------------------------------------------------------------------
# "Why is nothing trading" — answered for the cohort, not one symbol at a time.
# ---------------------------------------------------------------------------


import pytest

from imperium.session import TradingSession
from imperium.execution.portfolio import Verdict
from imperium.venues import registry
from imperium.execution.broker import PaperBroker
from decimal import Decimal


def _session_with(blockers: dict[str, int]) -> TradingSession:
    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal("70")
    i = 0
    for label, count in blockers.items():
        for _ in range(count):
            engine = session.engine(f"SYM{i:03d}")
            engine.decision.verdict = Verdict.REJECTED
            engine.decision.blocker = label
            engine.decision.strategy = "trend"
            i += 1
    return session


def test_the_dominant_blocker_is_named_with_its_share():
    """Four silent hours and the terminal could not say why. Every reason was
    in the panel, one symbol at a time, which is the one shape that cannot
    answer a question about all of them."""
    session = _session_with({"costs": 120, "warming up": 20, "sizing": 10})

    block = session._blockers()

    assert block["trading"] == 0
    assert "nothing is trading" in block["summary"]
    assert "costs" in block["summary"]
    assert "80%" in block["summary"], block["summary"]
    assert block["counts"][0] == {"blocker": "costs", "symbols": 120}


def test_a_book_that_is_trading_says_so_instead():
    session = _session_with({"costs": 5})
    engine = session.engine("WINNER")
    engine.decision.verdict = Verdict.TRADING
    engine.decision.strategy = "trend"

    block = session._blockers()

    assert block["trading"] == 1
    assert "worth trading" in block["summary"]
    assert "nothing is trading" not in block["summary"]


def test_a_carried_hold_is_not_counted_as_a_fresh_trade():
    """A position being held is not a decision to open one. Counting it as
    trading would report a working book on a night nothing fired."""
    session = _session_with({"costs": 3})
    engine = session.engine("HELD")
    engine.decision.verdict = Verdict.TRADING
    engine.decision.hold = True
    engine.decision.strategy = "trend"

    assert session._blockers()["trading"] == 0


def test_unscanned_symbols_are_distinguished_from_refused_ones():
    """"Not looked at yet" and "looked at and refused" are different answers,
    and collapsing them would hide a sweep that is not running."""
    session = TradingSession()
    session.engine("FRESH")          # never evaluated: no strategy, no blocker

    block = session._blockers()

    assert block["counts"][0]["blocker"] == "not yet scanned"


def test_every_refusal_in_the_engine_carries_a_label():
    """A refusal with no label lands in "other", and a cohort held up by
    "other" is exactly the dead end this replaces."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src/imperium/execution/engine.py"
    text = src.read_text(encoding="utf-8")
    refusals = len(re.findall(r"d\.verdict = Verdict\.(REJECTED|NOT_ADMITTED)", text))
    labelled = len(re.findall(r"d\.blocker = ", text))
    assert labelled == refusals, (
        f"{refusals} refusal sites but only {labelled} carry a blocker label — "
        f"the unlabelled ones will be reported as \"other\"")


@pytest.mark.asyncio
async def test_reading_the_clock_actually_sets_the_per_class_gate():
    """The call site, not the method.

    ``set_market_state`` can be perfectly correct and never reached: the tests
    above all call it directly, so every one of them still passes with the
    session left assigning the old single flag. The allocator would then run
    all night on whatever the flag last held, which is the bug this whole file
    exists for. So this drives the real path -- read the venue's clock, then
    ask the allocator what it believes.
    """
    from imperium.venues.alpaca.client import AlpacaClient
    from mock_venue import KEY, SECRET, MockVenue

    venue = MockVenue()
    venue.market_open = False
    session = TradingSession()
    session.client = AlpacaClient(KEY, SECRET, paper=True,
                                  transport=venue.transport)
    session.universe = ["AAPL", "BTC/USD"]
    for symbol in session.universe:
        session.allocator.observe(symbol).admitted = True
    session.allocator.equity = session.allocator.cash = 70.0
    try:
        await session.refresh_clock()
    finally:
        await session.detach_client()

    assert session.allocator.market_is_open("BTC/USD") is True, (
        "the equity market is shut and the crypto book was shut with it")
    assert session.allocator.market_is_open("AAPL") is False
    assert session.allocator.clamp("BTC/USD", 0.10).weight == 0.10
    assert session.allocator.clamp("AAPL", 0.10).reduced
