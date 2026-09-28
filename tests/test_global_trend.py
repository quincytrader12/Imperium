"""The Global Trend sleeve, and the sleeve framework it runs on.

What makes it safe to run on a $73 account: it holds only what is trending
up, sized by inverse volatility to a target measured on the real covariance;
it never orders below Alpaca's $1 floor and says so; nothing else in the
terminal can trade its symbols; and its orders go through the one order path,
named for the sleeve, so every guard and every record applies to it.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from decimal import Decimal

import numpy as np
import pytest

from imperium.execution import sleeves as sl
from imperium.strategy import global_trend as gt


@pytest.fixture(autouse=True)
def _sleeve_on(monkeypatch):
    monkeypatch.setenv("GLOBAL_TREND_ENABLED", "true")


def _walk(n=300, drift=0.0005, sd=0.008, seed=1, start=100.0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(drift, sd, n)))


def _universe(up=("IEF", "GLD", "EFA"), seed=3):
    out = {}
    for i, s in enumerate(gt.UNIVERSE):
        drift = 0.003 if s in up else -0.003
        sd = {"IEF": 0.003, "TLT": 0.009, "GLD": 0.008, "DBC": 0.011,
              "EFA": 0.010, "EEM": 0.013, "VNQ": 0.012}[s]
        out[s] = _walk(drift=drift, sd=sd, seed=seed + i)
    return out


# -- the signal -----------------------------------------------------------------


def test_the_trend_score_counts_the_horizons_rising():
    assert gt.trend_score(np.linspace(50, 100, 300)) == 1.0
    assert gt.trend_score(np.linspace(100, 50, 300)) == -1.0
    assert math.isnan(gt.trend_score(np.linspace(50, 100, gt.MIN_BARS - 1)))
    # Up over a year, down over the last month: three of four.
    series = np.concatenate([np.linspace(50, 120, 280), np.linspace(120, 110, 20)])
    assert gt.trend_score(series) == pytest.approx(0.5)


def test_only_what_is_trending_up_is_held():
    t = gt.targets(_universe(up=("IEF", "GLD", "EFA")))
    assert set(t.weights) == {"IEF", "GLD", "EFA"}
    assert "trend down" in t.reasons["EEM"]


def test_the_calmer_asset_gets_more_weight():
    t = gt.targets(_universe(up=("IEF", "EEM")))
    assert t.weights["IEF"] > t.weights["EEM"]


def test_the_book_is_scaled_to_its_target_on_the_measured_covariance():
    closes = _universe(up=("GLD", "EFA", "VNQ", "DBC"))
    t = gt.targets(closes, max_weight=1.0, max_gross=10.0)
    assert t.volatility == pytest.approx(gt.TARGET_VOL, rel=1e-6)


def test_correlated_assets_are_given_less_than_uncorrelated_ones():
    """The point of measuring the covariance rather than summing volatilities:
    two assets that move together are one bet and are sized as one."""
    rng = np.random.default_rng(5)
    common = rng.normal(0.004, 0.01, 300)
    independent = rng.normal(0.004, 0.01, 300)
    together = {"GLD": 100 * np.exp(np.cumsum(common)),
                "EFA": 100 * np.exp(np.cumsum(common + rng.normal(0, 0.001, 300)))}
    apart = {"GLD": 100 * np.exp(np.cumsum(common)),
             "EFA": 100 * np.exp(np.cumsum(independent))}
    kw = dict(max_weight=1.0, max_gross=10.0)
    assert gt.targets(apart, **kw).gross > gt.targets(together, **kw).gross * 1.2


def test_no_leverage_and_no_asset_above_its_cap():
    calm = {s: _walk(drift=0.002, sd=0.001, seed=i) for i, s in enumerate(gt.UNIVERSE)}
    t = gt.targets(calm)
    assert t.gross <= gt.MAX_GROSS + 1e-9
    assert max(t.weights.values()) <= gt.MAX_WEIGHT + 1e-9


def test_one_calm_asset_cannot_take_the_whole_sleeve():
    """Inverse volatility gives a very calm bond fund almost everything; the
    per-asset cap is what keeps the sleeve diversified."""
    closes = {"IEF": _walk(drift=0.001, sd=0.0005, seed=1),
              "GLD": _walk(drift=0.004, sd=0.01, seed=2)}
    t = gt.targets(closes)
    assert t.weights["IEF"] == pytest.approx(gt.MAX_WEIGHT)
    assert t.gross <= gt.MAX_GROSS + 1e-9


def test_nothing_rising_means_cash_and_says_so():
    t = gt.targets(_universe(up=()))
    assert t.weights == {} and "in cash" in t.note


# -- turning targets into orders -----------------------------------------------------


def _sleeve(allocation=0.30, weights=None, reasons=None):
    weights = weights or {"IEF": 0.4, "GLD": 0.2, "EEM": 0.02}

    def decide(closes, memory, day):
        return sl.Targets(weights=dict(weights), reasons=reasons or {})
    return sl.Sleeve(name="global_trend", label="Global trend",
                     universe=("IEF", "GLD", "EEM", "TLT"), allocation=allocation,
                     decide=decide)


PRICES = {"IEF": 95.0, "GLD": 180.0, "EEM": 40.0, "TLT": 90.0}


def test_at_73_dollars_the_sleeve_buys_what_clears_the_floor_and_names_the_rest():
    s = _sleeve()
    orders = s.plan({}, "d", prices=PRICES, held={}, account_equity=73.0)
    sleeve = 0.30 * 73.0
    by = {o.symbol: o for o in orders}
    assert by["IEF"].notional == pytest.approx(0.4 * sleeve)
    assert by["IEF"].account_weight == pytest.approx(0.4 * sleeve / 73.0)
    assert "EEM" not in by                        # 2% of $21.90 is 44 cents
    assert s.last_too_small["EEM"] == pytest.approx(0.02 * sleeve)
    assert all(o.notional >= sl.MIN_ORDER for o in orders if o.side == "buy")
    assert s.smallest_viable_equity() == pytest.approx(1.0 / (0.02 * 0.30))


def test_an_asset_no_longer_wanted_is_sold_whatever_its_size():
    s = _sleeve(weights={"IEF": 0.4})
    orders = s.plan({}, "d", prices=PRICES, held={"TLT": 0.005},
                    account_equity=73.0)
    exit_ = next(o for o in orders if o.symbol == "TLT")
    assert exit_.side == "sell" and exit_.account_weight == 0.0
    assert orders[0].side == "sell", "sells go first, to fund the buys"


def test_a_small_drift_is_left_alone_and_a_large_one_is_rebalanced():
    # At $1,000 so a 10% drift is $12 -- past the dollar floor, inside the band.
    s = _sleeve(weights={"IEF": 0.4})
    target_qty = 0.4 * 0.30 * 1000.0 / 95.0
    near = s.plan({}, "d", prices=PRICES, held={"IEF": target_qty * 0.9},
                  account_equity=1000.0)
    assert [o.symbol for o in near] == []
    far = s.plan({}, "d", prices=PRICES, held={"IEF": target_qty * 0.5},
                 account_equity=1000.0)
    assert [(o.symbol, o.side) for o in far] == [("IEF", "buy")]


def test_the_evidence_multiplier_scales_the_whole_sleeve():
    s = _sleeve(weights={"IEF": 0.4})
    full = s.plan({}, "d", prices=PRICES, held={}, account_equity=73.0)[0]
    half = s.plan({}, "d", prices=PRICES, held={}, account_equity=73.0,
                  multiplier=0.5)[0]
    assert half.notional == pytest.approx(full.notional / 2)


def test_it_decides_once_a_day_after_its_time_while_the_market_is_open():
    s = _sleeve()
    before = dt.datetime(2026, 9, 28, 15, 30)
    after = dt.datetime(2026, 9, 28, 15, 50)
    assert not s.due(before, True)
    assert not s.due(after, False)
    assert s.due(after, True)
    s.pending_day = "2026-09-28"
    assert not s.due(after, True)
    s.pending_day, s.last_run_day = "", "2026-09-28"
    assert not s.due(after, True)
    s.enabled = False
    assert not s.due(dt.datetime(2026, 9, 29, 15, 50), True)


def test_its_memory_survives_a_restart_and_a_corrupt_file():
    s = _sleeve()
    s.plan({}, "d", prices=PRICES, held={}, account_equity=73.0)
    s.last_run_day, s.runs, s.memory = "2026-09-28", 4, {"x": 1}
    back = _sleeve()
    back.restore(s.as_dict())
    assert (back.last_run_day, back.runs, back.memory) == ("2026-09-28", 4, {"x": 1})
    assert back.last_targets == s.last_targets
    back.restore({"last_targets": {"IEF": "nan", "GLD": -1, "EFA": 0.2},
                  "runs": "many"})
    assert back.last_targets == {"EFA": 0.2} and back.runs == 0


def test_configuration_is_read_from_the_environment_and_bounded(monkeypatch):
    monkeypatch.setenv("GLOBAL_TREND_ALLOCATION", "5")
    monkeypatch.setenv("GLOBAL_TREND_ENABLED", "false")
    s = next(x for x in sl.build_all() if x.name == "global_trend")
    assert s.allocation == 0.6 and not s.enabled


# -- in the session ---------------------------------------------------------------


class _Client:
    """Daily bars for the sleeve's universe, in the venue's row shape."""

    authenticated = True

    def __init__(self, closes):
        self.closes = closes
        self.asked = []

    async def bars(self, symbols, **kw):
        self.asked.append((tuple(symbols), kw.get("adjustment")))
        return {s: [{"c": float(c)} for c in self.closes[s]] for s in symbols
                if s in self.closes}


def _session(equity=73.0):
    from imperium.execution.broker import PaperBroker
    from imperium.session import TradingSession
    from imperium.venues import registry

    session = TradingSession()
    session.broker = PaperBroker(registry.get(registry.DEFAULT_VENUE))
    session.broker.cash = Decimal(str(equity))
    session._attribution_loaded = True
    return session


def test_the_sleeve_claims_its_share_and_the_engine_sizes_against_the_rest():
    session = _session()
    session.engine_equity(73.0)
    assert session.capital.share_for("global_trend") == pytest.approx(0.30)
    assert session.capital.share_for("engine") == pytest.approx(0.70)


@pytest.mark.asyncio
async def test_no_other_path_may_trade_a_sleeves_symbols():
    """The engine does not enter them, the give-back ratchet does not exit
    them, the close does not flatten them."""
    from imperium.execution.engine import Decision
    from imperium.execution.portfolio import Verdict

    session = _session()
    assert "GLD" in session.reserved_symbols()
    d = Decision(symbol="GLD")
    d.verdict = Verdict.TRADING
    d.target_weight = 0.2
    q = session.feed.quote("GLD")
    q.last, q.updated_at = 180.0, time.time()
    await session._act_on(d)
    assert session.broker.position("GLD").is_flat

    from imperium.execution.broker import Position
    session.broker.positions["GLD"] = Position("GLD", Decimal("0.05"), Decimal("100"))
    q.last = 300.0
    await session._protect_positions()        # up 200%: a give-back candidate
    q.last = 200.0
    await session._protect_positions()
    assert session.broker.position("GLD").quantity == Decimal("0.05")


@pytest.mark.asyncio
async def test_the_close_the_retirement_sweep_and_the_unmanaged_check_leave_it_alone():
    from imperium.execution.broker import Position
    from imperium.execution.portfolio import Verdict

    session = _session()
    session.broker.positions["GLD"] = Position("GLD", Decimal("0.05"), Decimal("180"))
    q = session.feed.quote("GLD")
    q.last, q.updated_at = 180.0, time.time()
    engine = session.engine("GLD")
    engine.decision.strategy = "overnight"
    engine.decision.verdict = Verdict.REJECTED
    await session._flatten_unwanted_before_the_close()
    await session._retire(["GLD"])
    session._report_unmanaged_equity()
    assert session.broker.position("GLD").quantity == Decimal("0.05")
    assert not [e for e in session.telemetry.events()
                if "GLD" in e.get("message", "") and "unmanaged" in e.get("message", "")
                or "GLD is held through the close" in e.get("message", "")]
    # A symbol that is not a sleeve's is still retired as before.
    session.broker.positions["XYZ"] = Position("XYZ", Decimal("1"), Decimal("10"))
    qx = session.feed.quote("XYZ")
    qx.last, qx.updated_at = 10.0, time.time()
    await session._retire(["XYZ"])
    assert session.broker.position("XYZ").is_flat


@pytest.mark.asyncio
async def test_a_days_decision_becomes_real_orders_named_for_the_sleeve():
    """Through the loop: the background decision leaves orders, the tick
    sends them through the broker, and they are journalled and attributed
    to the sleeve -- at $73."""
    session = _session(73.0)
    closes = _universe(up=("IEF", "GLD", "EFA"))
    session.client = _Client(closes)
    for s, v in closes.items():
        q = session.feed.quote(s)
        q.last, q.updated_at = float(v[-1]), time.time()
    sleeve = next(x for x in session.sleeves if x.name == "global_trend")
    await session._plan_sleeve(sleeve, "2026-09-28")
    assert session.client.asked[0][1] == "all", "adjusted closes, always"
    assert sleeve.pending
    # The decision needed the venue; sending goes through the broker alone.
    session.client = None
    await session._tick()
    held = {s: p for s, p in session.broker.positions.items() if not p.is_flat}
    assert set(held) <= {"IEF", "GLD", "EFA"} and held
    rows = session.trade_journal.read()
    assert rows and all(r.strategy == "global_trend" for r in rows)
    assert all(r.value >= sl.MIN_ORDER - 1e-6 for r in rows)
    assert sleeve.last_run_day == "2026-09-28" and sleeve.pending is None
    panel = session.snapshot()["sleeves"][0]
    assert panel["label"] == "Global trend" and panel["holdings"]


@pytest.mark.asyncio
async def test_a_sleeve_that_cannot_read_history_tries_again_later():
    session = _session()

    class Broken:
        async def bars(self, *a, **k):
            raise RuntimeError("no data plan")

    session.client = Broken()
    sleeve = session.sleeves[0]
    await session._plan_sleeve(sleeve, "2026-09-28")
    assert sleeve.pending is None and "no data plan" in sleeve.last_error
    assert sleeve.due(dt.datetime(2026, 9, 28, 15, 50), True)
