"""A daily event loop for the Sector Trend sleeve.

Imports the live signal functions from :mod:`imperium.strategy.sector` rather
than reimplementing them. That is the whole discipline of this file: a
backtester that carries its own copy of the rules measures a strategy nobody
will trade, and the divergence is usually a single index.

**What it charges.** A configurable slippage per side, and margin interest on
whatever exposure sits above 100% of sleeve equity. The paper runs at a 200%
cap and does not model the interest that borrowing would cost, which is the
main reason its headline number should not be taken as a net result.

**What it cannot tell you.** Whether the strategy works. It is one path through
one history, and the honest use of it is as a *bug detector*: the paper reports
roughly 7.7% CAGR at a Sharpe near 0.6 with a 24% drawdown over 2005-2024 on a
wider universe, and a result far outside that neighbourhood -- a Sharpe above
1.2, say, or a drawdown under 10% -- is very much more likely to be lookahead
than alpha. :func:`sanity_check` says so in as many words rather than leaving
it to a reader who wants the number to be good.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from imperium.strategy import sector

#: Trading days in a year, for annualising.
TRADING_DAYS = 252

#: The paper's own results over 2005-2024, 31 ETFs, a 200% cap. Used only to
#: flag a result that is too good to be true.
PAPER_CAGR = 0.077
PAPER_SHARPE = 0.6
PAPER_MAX_DRAWDOWN = 0.24
PAPER_BETA = 0.4


@dataclass(frozen=True)
class BacktestConfig:
    target_vol: float = 0.015
    max_leverage: float = 1.0
    rebalance_threshold: float = 0.25
    #: Charged on every side of every trade, in basis points of notional.
    slippage_bps: float = 5.0
    #: Annual rate on exposure above 100% of sleeve equity.
    margin_rate: float = 0.07
    #: "near_close" fills on the signal day's close; "next_open" on the
    #: following bar's open.
    exec_mode: str = "near_close"
    starting_equity: float = 100_000.0


@dataclass
class Trade:
    symbol: str
    opened: int
    closed: int = -1
    entry_price: float = 0.0
    exit_price: float = 0.0
    reason: str = ""
    #: The stop that fired, and the lower band on the same day. Recorded so a
    #: test can prove the exit was judged on the stop *carried in* rather than
    #: on one trailed up earlier in the same day -- an ordering that cannot be
    #: seen in an equity curve but changes which day a position closes.
    stop_used: float = float("nan")
    lower_band_at_exit: float = float("nan")

    @property
    def days_held(self) -> int:
        return max(0, self.closed - self.opened) if self.closed >= 0 else 0


@dataclass
class BacktestResult:
    dates: list[str] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    exposure: list[float] = field(default_factory=list)
    costs_paid: float = 0.0
    interest_paid: float = 0.0
    #: Days on which new buys had to be shrunk to fit the leverage cap or the
    #: cash available. Counted rather than hidden: a sleeve that is constantly
    #: trimming its entries is not running the strategy that was backtested.
    buys_scaled: int = 0

    def returns(self) -> np.ndarray:
        curve = np.asarray(self.equity, dtype=float)
        if curve.size < 2:
            return np.zeros(0)
        return np.diff(curve) / curve[:-1]


def _fill_price(price: float, side: int, slippage_bps: float) -> float:
    """A fill that pays the spread, in the direction that hurts.

    A backtest that fills at the untouched close shows an edge that does not
    exist. Buys lift, sells hit.
    """
    return float(price) * (1.0 + side * slippage_bps / 10_000.0)


def run(closes: dict[str, np.ndarray], dates: list[str], *,
        config: BacktestConfig | None = None,
        opens: dict[str, np.ndarray] | None = None) -> BacktestResult:
    """Walk the history one day at a time.

    ``closes`` must be split- and dividend-adjusted. Unadjusted prices put a
    gap at every split and every dividend, and a breakout rule reads those as
    signals -- the strategy would spend the backtest trading corporate actions.
    """
    cfg = config or BacktestConfig()
    symbols = sorted(closes)
    if not symbols or not dates:
        return BacktestResult()

    length = len(dates)
    band_by_symbol = {s: sector.bands(closes[s]) for s in symbols}

    cash = cfg.starting_equity
    quantities: dict[str, float] = {}
    stops: dict[str, float] = {}
    open_trades: dict[str, Trade] = {}

    result = BacktestResult()
    first = max((band_by_symbol[s].usable_from for s in symbols), default=0)
    first = max(first, sector.SIGMA_DAYS + 1)

    for t in range(first, length):
        # 1. Mark the book at today's close.
        def price_of(sym: str, index: int = t) -> float:
            value = float(closes[sym][index])
            return value if math.isfinite(value) and value > 0 else 0.0

        equity = cash + sum(quantities.get(s, 0.0) * price_of(s)
                            for s in quantities)

        # 2. Exits, judged on the stop carried in from yesterday. Before
        #    entries, so the cash they free is available.
        for symbol in list(quantities):
            carried = stops.get(symbol, float("nan"))
            step = sector.step_position(price_of(symbol), carried,
                                        float(band_by_symbol[symbol].lower[t]))
            if step.exited:
                fill = _fill_price(price_of(symbol), -1, cfg.slippage_bps)
                qty = quantities.pop(symbol)
                cash += qty * fill
                result.costs_paid += abs(qty) * price_of(symbol) * \
                    cfg.slippage_bps / 10_000.0
                stops.pop(symbol, None)
                trade = open_trades.pop(symbol, None)
                if trade is not None:
                    trade.closed, trade.exit_price = t, fill
                    trade.reason = "exit_stop"
                    trade.stop_used = float(carried)
                    trade.lower_band_at_exit = float(
                        band_by_symbol[symbol].lower[t])
                    result.trades.append(trade)

        # 3. Trail the surviving stops up. sector.step_position already
        #    decided both halves in the right order; this applies the half that
        #    survives, and the ordering itself is enforced and tested there
        #    rather than reproduced here.
        for symbol in list(quantities):
            band = band_by_symbol[symbol]
            stops[symbol] = sector.step_position(
                price_of(symbol), stops.get(symbol, float("nan")),
                float(band.lower[t])).stop

        # 4. Entries: flat symbols whose close clears yesterday's upper band.
        entering: list[str] = []
        for symbol in symbols:
            if symbol in quantities:
                continue
            if sector.entry_signal(closes[symbol], band_by_symbol[symbol], t):
                entering.append(symbol)

        # 5. Size the whole long book, held and entering alike.
        longs = sorted(set(quantities) | set(entering))
        sigmas = {s: sector.daily_sigma(closes[s][:t + 1]) for s in longs}
        targets = sector.target_weights(
            sigmas, longs, universe_size=len(symbols),
            target_vol=cfg.target_vol, max_leverage=cfg.max_leverage)

        equity = cash + sum(quantities.get(s, 0.0) * price_of(s)
                            for s in quantities)
        if equity <= 0:
            break

        # Decide every trade first, then fit the buys to the money available.
        # Deciding and executing in one pass would let whichever symbol the
        # loop reached first spend the budget and the rest go unfilled, which
        # is a different portfolio from the one the strategy chose.
        deltas: dict[str, float] = {}
        for symbol in longs:
            weight = targets.weights.get(symbol, 0.0)
            price = price_of(symbol)
            if price <= 0:
                continue
            target_qty = (weight * equity) / price
            current_qty = quantities.get(symbol, 0.0)

            is_entry = symbol in entering
            if not is_entry and not sector.needs_rebalance(
                    current_qty, target_qty, cfg.rebalance_threshold):
                continue

            delta = target_qty - current_qty
            if abs(delta) > 1e-12:
                deltas[symbol] = delta

        # Sells first: they free the cash the buys will spend.
        for symbol, delta in deltas.items():
            if delta >= 0:
                continue
            price = price_of(symbol)
            fill = _fill_price(price, -1, cfg.slippage_bps)
            cash -= delta * fill
            result.costs_paid += abs(delta) * price * cfg.slippage_bps / 10_000.0
            quantities[symbol] = quantities.get(symbol, 0.0) + delta

        # The buys, scaled to what the leverage cap leaves. The floor on cash
        # is what the cap means in dollars: at 1.0 the sleeve may not go below
        # zero cash, at 2.0 it may borrow up to its own equity.
        buy_cost = {s: deltas[s] * _fill_price(price_of(s), 1, cfg.slippage_bps)
                    for s in deltas if deltas[s] > 0}
        floor = equity * (1.0 - cfg.max_leverage)
        fitted, factor = sector.scale_buys_to_budget(buy_cost, cash - floor)
        for symbol, cost in fitted.items():
            price = price_of(symbol)
            fill = _fill_price(price, 1, cfg.slippage_bps)
            if fill <= 0:
                continue
            bought = cost / fill
            cash -= cost
            result.costs_paid += abs(bought) * price * cfg.slippage_bps / 10_000.0
            quantities[symbol] = quantities.get(symbol, 0.0) + bought
            if symbol in entering and bought > 0:
                stops[symbol] = float(band_by_symbol[symbol].lower[t])
                open_trades[symbol] = Trade(symbol=symbol, opened=t,
                                            entry_price=fill, reason="entry")
        if factor < 1.0:
            result.buys_scaled += 1

        # 6. Margin interest on exposure above the sleeve's own equity.
        gross = sum(abs(quantities.get(s, 0.0)) * price_of(s)
                    for s in quantities)
        equity = cash + sum(quantities.get(s, 0.0) * price_of(s)
                            for s in quantities)
        borrowed = max(0.0, gross - equity)
        if borrowed > 0 and cfg.margin_rate > 0:
            charge = borrowed * cfg.margin_rate / TRADING_DAYS
            cash -= charge
            result.interest_paid += charge
            equity -= charge

        result.dates.append(dates[t])
        result.equity.append(equity)
        result.exposure.append(gross / equity if equity > 0 else 0.0)

    for symbol, trade in open_trades.items():
        trade.closed = length - 1
        trade.reason = "open_at_end"
        result.trades.append(trade)
    return result


# -- reporting ------------------------------------------------------------

@dataclass
class Metrics:
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    beta: float = float("nan")
    alpha: float = float("nan")
    trades: int = 0
    average_hold_days: float = 0.0
    average_exposure: float = 0.0
    years: float = 0.0
    final_equity: float = 0.0
    yearly: dict[str, float] = field(default_factory=dict)


def max_drawdown(curve: np.ndarray) -> float:
    """Deepest peak-to-trough fall, as a positive fraction."""
    curve = np.asarray(curve, dtype=float)
    if curve.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(curve)
    return float(np.max((peaks - curve) / peaks)) if np.all(peaks > 0) else 0.0


def measure(result: BacktestResult, *,
            benchmark: np.ndarray | None = None) -> Metrics:
    """Turn a run into the numbers the brief asks for.

    The risk-free rate is zero throughout, which is stated rather than assumed:
    over 2005-2024 that flatters Sharpe in the high-rate years and is why the
    figure is comparable to the paper's only loosely.
    """
    out = Metrics()
    curve = np.asarray(result.equity, dtype=float)
    if curve.size < 2:
        return out

    returns = result.returns()
    out.final_equity = float(curve[-1])
    out.years = curve.size / TRADING_DAYS
    if out.years > 0 and curve[0] > 0:
        out.cagr = float((curve[-1] / curve[0]) ** (1.0 / out.years) - 1.0)
    out.volatility = float(np.std(returns, ddof=1) * math.sqrt(TRADING_DAYS))
    if out.volatility > 0:
        out.sharpe = float(np.mean(returns) / np.std(returns, ddof=1)
                           * math.sqrt(TRADING_DAYS))
    downside = returns[returns < 0]
    if downside.size > 1:
        spread = float(np.std(downside, ddof=1))
        if spread > 0:
            out.sortino = float(np.mean(returns) / spread
                                * math.sqrt(TRADING_DAYS))
    out.max_drawdown = max_drawdown(curve)
    out.trades = len([t for t in result.trades if t.reason != "open_at_end"])
    holds = [t.days_held for t in result.trades if t.days_held > 0]
    out.average_hold_days = float(np.mean(holds)) if holds else 0.0
    if result.exposure:
        out.average_exposure = float(np.mean(result.exposure))

    if benchmark is not None and len(benchmark) == returns.size + 1:
        bench = np.diff(benchmark) / benchmark[:-1]
        if np.std(bench, ddof=1) > 0:
            beta = float(np.cov(returns, bench, ddof=1)[0, 1]
                         / np.var(bench, ddof=1))
            out.beta = beta
            out.alpha = float((np.mean(returns) - beta * np.mean(bench))
                              * TRADING_DAYS)

    by_year: dict[str, list[float]] = {}
    for day, ret in zip(result.dates[1:], returns):
        by_year.setdefault(str(day)[:4], []).append(float(ret))
    for year, rets in by_year.items():
        out.yearly[year] = float(np.prod([1 + r for r in rets]) - 1)
    return out


def sanity_check(metrics: Metrics) -> list[str]:
    """Warnings for a result that is too good to be true.

    The paper's neighbourhood is the reference. A backtest of this code that
    lands far outside it has almost certainly found a bug and not an edge, and
    the most valuable thing this function can do is say so before anybody
    enables the strategy on the strength of the number.
    """
    notes: list[str] = []
    if metrics.sharpe > 1.2:
        notes.append(
            f"Sharpe {metrics.sharpe:.2f} is far above the paper's ~{PAPER_SHARPE}. "
            f"Suspect lookahead before believing it.")
    if 0 < metrics.max_drawdown < 0.10:
        notes.append(
            f"a {metrics.max_drawdown:.1%} maximum drawdown is implausibly "
            f"shallow for a long-only equity sleeve (the paper's is "
            f"~{PAPER_MAX_DRAWDOWN:.0%}).")
    if metrics.max_drawdown > 0.40:
        notes.append(
            f"a {metrics.max_drawdown:.1%} drawdown is far deeper than the "
            f"paper's ~{PAPER_MAX_DRAWDOWN:.0%}; check the stop is being "
            f"applied at all.")
    if metrics.cagr > 0.25:
        notes.append(
            f"a {metrics.cagr:.1%} CAGR is several times the paper's "
            f"~{PAPER_CAGR:.1%} on a wider universe.")
    if metrics.trades == 0:
        notes.append("no trades were taken at all — check the universe has "
                     "enough history and the bands are finite.")
    return notes
