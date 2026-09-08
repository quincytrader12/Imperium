"""The portfolio layer: where a fleet of engines goes unbounded."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from imperium.execution.engine import SymbolEngine
from imperium.execution.portfolio import PortfolioAllocator, Verdict
from imperium.execution.risk import RiskLimits
from imperium.strategy.signals import StrategyParams
from imperium.telemetry.streams import TelemetryHub
from imperium.venues.registry import get


def make_allocator(**kw) -> PortfolioAllocator:
    a = PortfolioAllocator(RiskLimits(**kw))
    a.equity = 100_000.0
    a.cash = 100_000.0
    return a


def test_n_engines_inside_their_own_caps_are_collectively_bounded():
    """Prevents: the failure this layer exists for. Five engines each obeying a
    20% per-symbol cap is a 100% book, and twelve is a 240% one. Each engine is
    individually correct and the book is unbounded."""
    a = make_allocator(max_gross_exposure=0.80, max_concurrent_positions=5,
                       max_position_weight=0.20)
    symbols = [f"S{i}USDT" for i in range(12)]
    for i, s in enumerate(symbols):
        a.set_scan(s, score=1.0 - i * 0.01, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()
    for s in symbols:
        result = a.clamp(s, 0.20)
        a.observe(s).current_weight = result.weight
    assert a.gross_exposure <= 0.80 + 1e-9, a.gross_exposure


def test_per_symbol_budget_divides_by_max_concurrency_not_the_live_count():
    """Prevents: allocation by arrival order. Dividing by the *current* count
    lets the first symbol admitted claim the entire book and forces every later
    one to trade at a fraction of it. It also silently changes every existing
    engine's budget whenever an unrelated symbol is admitted."""
    a = make_allocator(max_gross_exposure=0.80, max_concurrent_positions=5,
                       max_position_weight=0.50)
    a.set_scan("BTCUSDT", score=1.0, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()
    budget_alone = a.per_symbol_budget
    first = a.clamp("BTCUSDT", 0.80)
    assert first.weight == pytest.approx(0.16), "one symbol must not claim the book"

    for i in range(4):
        a.set_scan(f"X{i}USDT", score=0.5, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()
    assert a.per_symbol_budget == budget_alone, (
        "an existing engine's budget changed because an unrelated symbol was "
        "admitted"
    )


def test_a_buying_power_reserve_is_never_allocated():
    """Prevents: an account at 100% deployed, which cannot act on anything —
    including an exit."""
    a = make_allocator(buying_power_reserve=0.15, max_gross_exposure=1.0,
                       max_position_weight=1.0, max_concurrent_positions=1)
    a.set_scan("BTCUSDT", score=1.0, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()
    assert a.buying_power() == pytest.approx(85_000.0)
    assert a.clamp("BTCUSDT", 1.0).weight <= 0.85 + 1e-9


def test_every_limit_only_ever_reduces():
    """Prevents: a 'clamp' that raises a target. A limit that can increase
    exposure is not a limit."""
    a = make_allocator()
    a.set_scan("BTCUSDT", score=1.0, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()
    for desired in (0.0, 0.01, 0.05, 0.16, 0.5, 1.0):
        assert a.clamp("BTCUSDT", desired).weight <= desired + 1e-12


def test_no_limit_can_block_an_exit():
    """Prevents: a cap that traps you in a losing position. Every constraint here
    is a bound on *taking* exposure; a reduction must pass through all of them,
    including a halt, a full book, and exhausted buying power."""
    a = make_allocator()
    a.set_scan("BTCUSDT", score=1.0, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()
    a.observe("BTCUSDT").current_weight = 0.16
    a.cash = 0.0                      # no buying power at all
    a.set_halt(True, "daily loss limit")
    for other in range(4):            # book completely full
        a.observe(f"F{other}").current_weight = 0.16
    assert a.clamp("BTCUSDT", 0.0).weight == 0.0
    assert a.clamp("BTCUSDT", 0.08).weight == pytest.approx(0.08)


def test_a_halt_still_blocks_new_exposure():
    """Prevents: the exit exemption above swallowing the halt entirely."""
    a = make_allocator()
    a.set_scan("BTCUSDT", score=1.0, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()
    a.observe("BTCUSDT").current_weight = 0.05
    a.set_halt(True, "daily loss limit")
    assert a.clamp("BTCUSDT", 0.16).weight == pytest.approx(0.05)


def test_not_admitted_is_distinguished_from_rejected():
    """Prevents: telling the operator the scanner dislikes a symbol it actually
    likes. 'The concurrency limit is full' is not a fault of the symbol, and
    collapsing the two verdicts hides which one it is."""
    a = make_allocator(max_concurrent_positions=2)
    a.set_scan("GOODUSDT", score=0.9, turnover=1e9, tradeable=True, reason="ok")
    a.set_scan("ALSOUSDT", score=0.8, turnover=1e9, tradeable=True, reason="ok")
    a.set_scan("QUEUEUSDT", score=0.7, turnover=1e9, tradeable=True, reason="ok")
    a.set_scan("BADUSDT", score=0.0, turnover=1.0, tradeable=False,
               reason="24h turnover below the floor")
    a.rebalance_admissions()
    assert a.states["QUEUEUSDT"].verdict is Verdict.NOT_ADMITTED
    assert "not a fault of QUEUEUSDT" in a.states["QUEUEUSDT"].reason
    assert a.states["BADUSDT"].verdict is Verdict.REJECTED
    assert "turnover" in a.states["BADUSDT"].reason


def test_a_symbol_holding_a_position_is_not_displaced_by_a_better_score():
    """Prevents: churning positions to chase a marginally better score, which
    pays a full round trip for the privilege — and momentarily leaves the
    displaced position with nothing managing it."""
    a = make_allocator(max_concurrent_positions=1)
    a.set_scan("HELDUSDT", score=0.1, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()
    a.observe("HELDUSDT").current_weight = 0.16
    a.set_scan("BETTERUSDT", score=0.99, turnover=1e9, tradeable=True, reason="ok")
    admitted, retired = a.rebalance_admissions()
    assert "HELDUSDT" not in retired
    assert a.states["BETTERUSDT"].verdict is Verdict.NOT_ADMITTED


def test_retiring_a_symbol_is_reported_so_it_can_be_flattened():
    """Prevents: a stopped engine still holding a position — a position with
    nothing managing its stop. Retirement has to be announced, not silent, or
    the session cannot know to flatten it."""
    a = make_allocator(max_concurrent_positions=2)
    for s in ("AUSDT", "BUSDT"):
        a.set_scan(s, score=0.9, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()
    a.set_scan("AUSDT", score=0.0, turnover=1.0, tradeable=False,
               reason="delisted at the venue")
    _, retired = a.rebalance_admissions()
    assert "AUSDT" in retired


def test_a_symbol_displaced_from_a_slot_is_also_reported_for_flattening():
    """Prevents: only announcing retirement for *rejected* symbols.

    A mutation test found this gap: removing the retirement report from the
    concurrency-displacement branch left the suite green, because the only test
    covering retirement used a rejected symbol and took a different branch. A
    symbol displaced from its slot while holding a position is exactly as
    dangerous -- a stopped engine still holding a position is a position with
    nothing managing its stop.
    """
    a = make_allocator(max_concurrent_positions=2)
    for s in ("AUSDT", "BUSDT"):
        a.set_scan(s, score=0.9, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()
    a.observe("AUSDT").current_weight = 0.10
    a.observe("BUSDT").current_weight = 0.10
    assert set(a.admitted_symbols) == {"AUSDT", "BUSDT"}

    # The operator tightens concurrency while both symbols hold positions.
    a.limits = RiskLimits(max_concurrent_positions=1)
    _, retired = a.rebalance_admissions()

    assert len(retired) == 1, "a displaced holder must be reported once"
    assert retired[0] in {"AUSDT", "BUSDT"}
    assert a.states[retired[0]].verdict is Verdict.NOT_ADMITTED
    assert not a.states[retired[0]].admitted


# -- the wiring test the brief specifically asks for ---------------------

def test_the_real_engine_class_requires_an_allocator():
    """Prevents: the portfolio cap reaching the engine through an optional hook.
    A hasattr guard around a method that does not exist fails silently forever,
    and the log will cheerfully say 'capped' while applying nothing.

    Asserted against the real class, not a stub — a stub with the attribute
    proves nothing about the object that actually runs."""
    params = inspect.signature(SymbolEngine.__init__).parameters
    assert "allocator" in params
    assert params["allocator"].default is inspect.Parameter.empty, (
        "allocator must be required; an optional clamp is not a clamp"
    )
    with pytest.raises(TypeError):
        SymbolEngine("BTCUSDT", get("binance_spot"), RiskLimits())  # type: ignore[call-arg]


def test_the_real_engine_class_applies_the_portfolio_clamp():
    """Prevents: the clamp being computed and then not applied. This drives the
    *real* SymbolEngine through a real evaluation and asserts the emitted target
    weight actually respects the allocator's budget."""
    # The allocator's budget (0.80 / 16 = 5%) must be *tighter* than what sizing
    # wants (~7.5% here) and than the per-symbol sizing cap (50%). Otherwise
    # sizing clips the weight first, the clamp has nothing to do, and this test
    # passes even with the clamp deleted -- which is exactly what happened on
    # the first attempt, and is why the fixture asserts its own premise below.
    a = make_allocator(max_gross_exposure=0.80, max_concurrent_positions=16,
                       max_position_weight=0.50)
    hub = TelemetryHub()
    spec = get("binance_spot")
    params = StrategyParams(warmup_bars=150)
    engine = SymbolEngine("BTCUSDT", spec, a.limits, a, hub, params)

    # A strongly trending, low-volatility series: sizing wants far more than the
    # allocator will allow, so the clamp must be the binding constraint.
    rng = np.random.default_rng(11)
    price = 100.0
    rows = []
    t = 0
    for _ in range(400):
        price *= math.exp(0.0015 + rng.normal(0, 0.0008))
        rows.append([t, price, price * 1.001, price * 0.999, price, 10.0])
        t += 60_000
    engine.seed([[r[0], f"{r[1]}", f"{r[2]}", f"{r[3]}", f"{r[4]}", f"{r[5]}"]
                 for r in rows] + [[t, f"{price}", f"{price}", f"{price}",
                                    f"{price}", "10.0"]])
    engine.set_book(price * 0.99995, price * 1.00005)
    a.set_scan("BTCUSDT", score=1.0, turnover=1e9, tradeable=True, reason="ok")
    a.rebalance_admissions()

    decision = engine.evaluate()
    assert a.per_symbol_budget == pytest.approx(0.05)
    assert decision.raw_weight > a.per_symbol_budget, (
        "fixture is wrong: sizing must want more than the allocator allows, or "
        "this test cannot detect an unapplied clamp"
    )
    assert decision.target_weight <= a.per_symbol_budget + 1e-9, (
        f"the engine emitted {decision.target_weight}, above the allocator's "
        f"budget of {a.per_symbol_budget}"
    )
    assert decision.clamp_binding == "per-symbol budget"
