"""Capital follows the market's regime, where a strategy's own record says so.

The properties that make it safe to run unattended: a state never peeks at
the future, a return is filed under the state the allocator knew when the day
began, nothing tilts on a thin record, a worthless strategy is rarely tilted
(measured, across every state), a real regime dependence usually is, and the
tilt can neither break the allocator's bounds nor lift a decaying strategy.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest

from imperium.execution import evidence as ev
from imperium.execution import market_regime as mr
from imperium.execution.attribution import StrategyRecord
from imperium.execution.bars import Bar

EQUITY = 1_000.0
START = dt.date(2024, 1, 1)


# -- building a market ---------------------------------------------------------


def _weekdays(n: int, start: dt.date = START) -> list[dt.date]:
    out, day = [], start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def _bars(closes: list[float], start: dt.date = START, hour: int = 4
          ) -> list[Bar]:
    """Daily bars, stamped by default the way the venue stamps them in
    summer: midnight Eastern, 04:00 UTC."""
    out = []
    for day, close in zip(_weekdays(len(closes), start), closes):
        stamp = dt.datetime(day.year, day.month, day.day, hour,
                            tzinfo=dt.timezone.utc)
        out.append(Bar(open_time=int(stamp.timestamp() * 1000), open=close,
                       high=close, low=close, close=close, volume=1.0,
                       closed=True))
    return out


def _walk(n: int, drift: float, sd: float, seed: int, start: float = 100.0):
    rng = random.Random(seed)
    closes, price = [], start
    for _ in range(n):
        price *= math.exp(rng.gauss(drift, sd))
        closes.append(price)
    return closes


# -- the state of the market ---------------------------------------------------


def test_a_rising_market_reads_as_rising_and_a_falling_one_as_falling():
    up = mr.states_by_day(_bars(_walk(400, 0.002, 0.005, seed=1)))
    down = mr.states_by_day(_bars(_walk(400, -0.002, 0.005, seed=1)))
    assert mr.current(up).state.startswith("rising")
    assert mr.current(down).state.startswith("falling")
    assert mr.current(up).trend > 0 > mr.current(down).trend


def test_a_turbulent_month_reads_as_volatile_for_this_market():
    calm = _walk(380, 0.001, 0.005, seed=2)
    storm = _walk(20, 0.001, 0.03, seed=3, start=calm[-1])
    states = mr.states_by_day(_bars(calm + storm))
    assert mr.current(states).state == mr.RISING_VOLATILE
    assert mr.current(states).vol_ratio > 2
    quiet = mr.states_by_day(_bars(calm))
    assert mr.current(quiet).state == mr.RISING_CALM


def test_volatile_means_volatile_for_this_market_about_half_the_time():
    """Split at its own median, so in a market whose volatility neither rises
    nor falls about half the days read volatile -- not the few above a line
    someone drew."""
    rng = random.Random(9)
    closes, price, sd = [], 100.0, 0.01
    for _ in range(900):
        # Volatility that clusters and reverts, like a real market's.
        sd = max(0.003, min(0.03, sd * math.exp(rng.gauss(0, 0.1))
                            + 0.05 * (0.01 - sd)))
        price *= math.exp(rng.gauss(0.0, sd))
        closes.append(price)
    states = mr.states_by_day(_bars(closes))
    volatile = sum(s.state.endswith("volatile") for s in states.values())
    assert 0.3 < volatile / len(states) < 0.7


def test_nothing_is_read_before_there_is_history_to_read_it_from():
    assert mr.states_by_day(_bars(_walk(mr.TREND_DAYS - 1, 0.001, 0.01, 4))) == {}
    first = min(mr.states_by_day(_bars(_walk(400, 0.001, 0.01, 4))))
    assert first == _weekdays(mr.TREND_DAYS)[-1].isoformat()


def test_a_state_never_peeks_at_the_future():
    """Replayed over history, a state that saw later closes would file returns
    under a state the allocator could not have known. Changing everything after
    a day must leave every state up to that day exactly as it was."""
    closes = _walk(420, 0.0005, 0.01, seed=5)
    cut = 330
    altered = closes[:cut] + _walk(90, -0.01, 0.04, seed=6, start=closes[cut - 1])
    a, b = mr.states_by_day(_bars(closes)), mr.states_by_day(_bars(altered))
    last_shared = _weekdays(cut)[-1].isoformat()
    shared = [d for d in a if d <= last_shared]
    assert shared and all(a[d] == b[d] for d in shared)
    assert any(a[d] != b[d] for d in a if d > last_shared)


@pytest.mark.parametrize("hour", [0, 4, 5])
def test_a_bar_is_filed_under_its_own_session_date(hour):
    """Midnight Eastern in summer and in winter, and midnight UTC: every
    convention a daily bar is stamped with lands on the session's date. Read
    in Eastern, the 04:00 UTC stamp of a winter bar is the evening before."""
    bars = _bars(_walk(300, 0.001, 0.01, 7), hour=hour)
    days = mr.states_by_day(bars)
    assert all(dt.date.fromisoformat(d).weekday() < 5 for d in days)
    assert max(days) == _weekdays(300)[-1].isoformat()


def test_a_weekend_mark_is_filed_under_fridays_state():
    states = mr.states_by_day(_bars(_walk(300, 0.001, 0.01, 8)))
    friday = max(d for d in states
                 if dt.date.fromisoformat(d).weekday() == 4)
    saturday = (dt.date.fromisoformat(friday) + dt.timedelta(days=1)).isoformat()
    assert mr.state_on(states, saturday) == states[friday].state
    assert mr.state_on(states, "2000-01-01") is None


# -- filing a strategy's returns -----------------------------------------------


def _state(day: str, state: str) -> mr.MarketState:
    return mr.MarketState(day, state, 0.01, 1.0)


def test_a_return_is_filed_under_the_state_known_when_its_day_began():
    """The allocator sets tomorrow's size from today's close. The day that
    follows is what that decision earned, so it is filed under today's state
    -- not the state the market had moved into by the time it was over."""
    states = {"2024-03-04": _state("2024-03-04", mr.RISING_CALM),
              "2024-03-05": _state("2024-03-05", mr.FALLING_VOLATILE)}
    marks = [("2024-03-04", 0.0), ("2024-03-05", 10.0), ("2024-03-06", 5.0)]
    equity = {d: EQUITY for d, _ in marks}
    assert mr.filed_returns(marks, equity, states) == [
        (mr.RISING_CALM, 0.01), (mr.FALLING_VOLATILE, -0.005)]


def test_a_day_with_no_state_or_no_equity_is_left_out_not_guessed():
    states = {"2024-03-05": _state("2024-03-05", mr.RISING_CALM)}
    marks = [("2024-03-04", 0.0), ("2024-03-05", 10.0), ("2024-03-06", 5.0),
             ("2024-03-07", 6.0)]
    equity = {"2024-03-04": EQUITY, "2024-03-05": EQUITY}
    assert mr.filed_returns(marks, equity, states) == [(mr.RISING_CALM, -0.005)]


# -- the tilt ------------------------------------------------------------------


def _filed(here: list[float], rest: list[float], state=mr.FALLING_VOLATILE,
           other=mr.RISING_CALM):
    return [(state, r) for r in here] + [(other, r) for r in rest]


def _gauss(mean, sd, n, seed):
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


def test_nothing_tilts_on_a_thin_record_on_either_side():
    losing = _gauss(-0.01, 0.002, 200, 1)
    earning = _gauss(0.01, 0.002, 200, 2)
    few_here = _filed(losing[:mr.MIN_STATE_DAYS - 1], earning)
    few_rest = _filed(losing, earning[:mr.MIN_STATE_DAYS - 1])
    for filed in (few_here, few_rest):
        t = mr.tilt(filed, mr.FALLING_VOLATILE)
        assert t.factor == 1.0 and t.different is None
    assert "19 of 20 days" in mr.tilt(few_here, mr.FALLING_VOLATILE).reason
    ok = mr.tilt(_filed(losing[:mr.MIN_STATE_DAYS], earning[:mr.MIN_STATE_DAYS]),
                 mr.FALLING_VOLATILE)
    assert ok.factor == mr.LESS


def test_a_strategy_that_loses_in_this_market_is_given_less_here():
    t = mr.tilt(_filed(_gauss(-0.003, 0.01, 60, 3), _gauss(0.004, 0.01, 200, 4)),
                mr.FALLING_VOLATILE)
    assert t.factor == mr.LESS
    assert "loses when the market is falling and volatile" in t.reason


def test_a_strategy_that_earns_more_in_this_market_is_given_more_here():
    t = mr.tilt(_filed(_gauss(0.006, 0.01, 60, 5), _gauss(0.0, 0.01, 200, 6)),
                mr.FALLING_VOLATILE)
    assert t.factor == mr.MORE
    assert t.different >= mr.DIFFERENT_BAR and t.earns >= mr.EARNS_BAR


def test_worse_here_but_still_earning_is_not_cut():
    """Significantly worse here than elsewhere, and still earning here. A
    strategy that earns everywhere is not cut for earning a little less."""
    here = _gauss(0.004, 0.004, 80, 7)
    rest = _gauss(0.010, 0.004, 200, 8)
    t = mr.tilt(_filed(here, rest), mr.FALLING_VOLATILE)
    assert t.different < 1 - mr.DIFFERENT_BAR
    assert t.earns > 0.5
    assert t.factor == 1.0


def test_better_here_but_not_earning_here_is_not_boosted():
    here = _gauss(-0.0002, 0.01, 60, 9)
    rest = _gauss(-0.008, 0.01, 200, 10)
    t = mr.tilt(_filed(here, rest), mr.FALLING_VOLATILE)
    assert t.different > mr.DIFFERENT_BAR
    assert t.factor == 1.0


def test_a_worthless_strategy_is_rarely_tilted_in_any_state():
    """The false-positive rate, measured. A strategy with no edge and no
    regime dependence, across a market that moves between four persistent
    states, checked in every one of them -- the way the allocator will check
    it over time. The bars are 1% each way per state, and the earn/lose
    condition barely thins them -- a mean far above the rest is usually a
    Sharpe well above zero too -- so the rate observed is about 1.8%, half of
    it more capital and half less. The bound is set above that rather than
    fitted to it."""
    rng = random.Random(11)
    order = [mr.RISING_CALM, mr.RISING_VOLATILE, mr.FALLING_CALM,
             mr.FALLING_VOLATILE]
    tilted = checked = 0
    for _ in range(600):
        state, filed = rng.choice(order), []
        for _ in range(240):
            if rng.random() < 0.05:                   # regimes persist
                state = rng.choice(order)
            filed.append((state, rng.gauss(0.0, 0.01)))
        for s in order:
            t = mr.tilt(filed, s)
            if t.different is None:
                continue
            checked += 1
            tilted += t.factor != 1.0
    assert checked > 1500
    assert tilted / checked < 0.025


def test_a_real_regime_dependence_is_usually_found():
    """Power, so the bars are not so strict that nothing ever moves: a
    strategy that earns in calm markets and loses in volatile ones, at a
    Sharpe of about 0.3 a day either way, over a year of trading."""
    rng = random.Random(12)
    found = 0
    for trial in range(200):
        filed = []
        for i in range(250):
            state = mr.RISING_CALM if (i // 25) % 2 == 0 else mr.RISING_VOLATILE
            mean = 0.003 if state == mr.RISING_CALM else -0.003
            filed.append((state, rng.gauss(mean, 0.01)))
        found += mr.tilt(filed, mr.RISING_VOLATILE).factor == mr.LESS
    assert found / 200 > 0.8


def test_an_unknown_market_state_tilts_nothing():
    t = mr.tilt(_filed(_gauss(-0.01, 0.01, 60, 1), _gauss(0.01, 0.01, 60, 2)),
                None)
    assert t.factor == 1.0 and "not yet known" in t.reason


# -- applied by the allocator --------------------------------------------------


def _market_and_record(here_mean: float, rest_mean: float, *, n: int = 240,
                       seed: int = 13, tail: str | None = None):
    """A record over ``n`` days alternating between two states in 20-day
    blocks, earning ``here_mean`` in falling-volatile and ``rest_mean``
    otherwise; the market ends in ``tail`` (falling-volatile by default)."""
    rng = random.Random(seed)
    days = [d.isoformat() for d in _weekdays(n + 1)]
    states: dict[str, mr.MarketState] = {}
    rec = StrategyRecord(name="trend", round_trips=40)
    value = 0.0
    rec.daily.append((days[0], 0.0))
    for i, day in enumerate(days[:-1]):
        state = mr.FALLING_VOLATILE if (i // 20) % 2 else mr.RISING_CALM
        states[day] = _state(day, state)
        mean = here_mean if state == mr.FALLING_VOLATILE else rest_mean
        value += rng.gauss(mean, 0.01) * EQUITY
        rec.daily.append((days[i + 1], value))
    last = days[-1]
    states[last] = _state(last, tail or mr.FALLING_VOLATILE)
    equity = {d: EQUITY for d, _ in rec.daily}
    return rec, equity, states, last


def _revise_until_settled(weights, rec, equity, states, last):
    for k in range(8):
        weights.revise(f"{last}#{k}", [rec], equity, states=states)


def test_the_allocator_sizes_a_strategy_down_in_the_market_it_loses_in():
    rec, equity, states, last = _market_and_record(-0.003, 0.004)
    weights = ev.CapitalWeights()
    weights.revise(last, [rec], equity, states=states)
    standing = weights.standings["trend"]
    assert standing.regime_factor == mr.LESS
    assert "loses when the market is falling and volatile" in standing.reason
    plain = ev.CapitalWeights()
    plain.revise(last, [rec], equity)
    assert standing.target == pytest.approx(
        max(ev.FLOOR, plain.standings["trend"].target * mr.LESS))


def test_the_same_record_is_sized_normally_in_the_market_it_earns_in():
    rec, equity, states, last = _market_and_record(-0.003, 0.004,
                                                   tail=mr.RISING_CALM)
    weights = ev.CapitalWeights()
    weights.revise(last, [rec], equity, states=states)
    assert weights.standings["trend"].regime_factor != mr.LESS


def test_without_a_market_state_nothing_is_tilted():
    rec, equity, _, last = _market_and_record(-0.003, 0.004)
    weights = ev.CapitalWeights()
    weights.revise(last, [rec], equity, states=None)
    assert weights.standings["trend"].regime_factor == 1.0
    assert weights.standings["trend"].regime_reason == ""


def test_the_regime_still_moves_capital_a_step_a_day():
    rec, equity, states, last = _market_and_record(-0.003, 0.004)
    weights = ev.CapitalWeights()
    weights.revise(last, [rec], equity, states=states)
    assert weights.standings["trend"].multiplier >= 1.0 - ev.STEP - 1e-9


def test_the_regime_never_lifts_a_strategy_the_decay_rule_cut():
    """Earned in this market for most of its record, and has stopped: its
    last twenty days are strongly negative. The decay rule's cut stands."""
    rec, equity, states, last = _market_and_record(0.012, -0.002, n=240)
    rng = random.Random(14)
    value = rec.daily[-1][1]
    days = [d.isoformat() for d in _weekdays(
        30, dt.date.fromisoformat(last) + dt.timedelta(days=1))]
    for day in days:
        states[day] = _state(day, mr.FALLING_VOLATILE)
        value += rng.gauss(-0.01, 0.003) * EQUITY
        rec.daily.append((day, value))
        equity[day] = EQUITY
    # Judged on the market alone it would be given more here -- which is
    # exactly what must not happen.
    alone = mr.tilt(mr.filed_returns(rec.daily, equity, states),
                    mr.FALLING_VOLATILE)
    assert alone.factor == mr.MORE
    weights = ev.CapitalWeights()
    weights.revise(days[-1], [rec], equity, states=states)
    standing = weights.standings["trend"]
    assert standing.target == ev.FLOOR
    assert standing.regime_factor == 1.0
    assert "decaying" in standing.reason


def test_the_tilt_is_held_inside_the_allocators_bounds():
    rec, equity, states, last = _market_and_record(-0.02, 0.004)
    weights = ev.CapitalWeights()
    _revise_until_settled(weights, rec, equity, states, last)
    assert weights.standings["trend"].target >= ev.FLOOR
    assert ev.FLOOR <= weights.multiplier("trend") <= ev.CAP


def test_the_regime_verdict_survives_a_restart_and_a_corrupt_file():
    rec, equity, states, last = _market_and_record(-0.003, 0.004)
    weights = ev.CapitalWeights()
    weights.revise(last, [rec], equity, states=states)
    back = ev.CapitalWeights.from_dict(weights.as_dict())
    assert back.standings["trend"].regime_factor == mr.LESS
    assert back.standings["trend"].regime_reason == \
        weights.standings["trend"].regime_reason
    payload = weights.as_dict()
    payload["standings"]["trend"]["regime_factor"] = "nan"
    assert ev.CapitalWeights.from_dict(payload).standings[
        "trend"].regime_factor == 1.0
    payload["standings"]["trend"]["regime_factor"] = 99
    assert ev.CapitalWeights.from_dict(payload).standings[
        "trend"].regime_factor == ev.CAP


# -- in the session ------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPERIUM_HOME", str(tmp_path))
    return tmp_path


def test_the_session_reads_the_market_from_the_benchmark_it_already_holds(home):
    from imperium.session import TradingSession

    session = TradingSession()
    assert session.snapshot()["market_regime"] is None
    session._benchmark_bars = _bars(_walk(400, -0.002, 0.005, seed=15))
    snap = session.snapshot()["market_regime"]
    assert snap["state"].startswith("falling")
    assert "below its 200-day average" in snap["reason"]
    assert session._market_line().startswith("falling")


def test_the_daily_mark_tilts_capital_by_the_market_state(home, monkeypatch):
    """Through _mark_strategies, which the brief calls: the states it passes
    must be the ones read from the benchmark, or nothing is ever tilted."""
    from imperium.session import TradingSession

    session = TradingSession()
    session._attribution_loaded = True
    rec, equity, states, last = _market_and_record(-0.003, 0.004)
    book = session.attribution.book_for(session.broker.mode.value)
    book.records["trend"] = rec
    book.equity_marks.update(equity)
    monkeypatch.setattr(session, "market_states", lambda: states)
    monkeypatch.setattr(book, "mark_day", lambda *a, **k: None)
    session._mark_strategies(last)
    assert session.capital_weights.standings["trend"].regime_factor == mr.LESS
    row = next(r for r in session._strategies_block()["rows"]
               if r["strategy"] == "trend")
    assert row["regime_factor"] == mr.LESS
    assert "falling and volatile" in row["regime_reason"]


def test_the_market_state_is_read_once_per_new_bar(home, monkeypatch):
    from imperium.session import TradingSession

    session = TradingSession()
    session._benchmark_bars = _bars(_walk(300, 0.001, 0.01, seed=16))
    calls = []
    real = mr.states_by_day
    monkeypatch.setattr(mr, "states_by_day",
                        lambda bars: calls.append(1) or real(bars))
    for _ in range(5):
        session.market_states()
    assert len(calls) == 1
    session._benchmark_bars = session._benchmark_bars + _bars(
        [session._benchmark_bars[-1].close],
        start=_weekdays(301)[-1])
    session.market_states()
    assert len(calls) == 2


def test_the_brief_names_the_market():
    from imperium.notify import daily

    brief = daily.Brief(day="Mon 01 Jan", equity=100.0, day_start_equity=100.0,
                        cash=10.0, market="rising and calm — the benchmark is "
                        "2.0% above its 200-day average")
    assert "Market: rising and calm" in daily.build(brief)
