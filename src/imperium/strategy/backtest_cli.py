"""The Sector Trend backtest, as a command.

This lives in the package rather than in ``scripts/`` for one reason: the
packaged Windows build is the only copy of this program most of its operators
will ever have, and a backtest they cannot run is a backtest that does not
exist for them. PyInstaller bundles the ``imperium`` package and nothing else,
so a module here ships and a script beside the repository does not.

``scripts/backtest_sector_trend.py`` is a four-line wrapper around this, and
``IMPERIUM.exe --backtest`` is the same thing with no checkout required.

Needs an Alpaca key, which it reads from the same credential file the terminal
uses -- it will not ask you to paste one anywhere else. Bars are requested
split- and dividend-adjusted; an unadjusted series steps at every split and
every dividend, and a breakout rule reads those steps as signals.

**Read the warnings at the bottom of the report before believing the numbers.**
The paper this implements reports roughly 7.7% CAGR at a Sharpe near 0.6 with a
24% drawdown over 2005-2024, on a wider universe than the nineteen ETFs here. A
result far better than that is much more likely to be a bug than an edge, and
the report says so rather than leaving it to a reader who wants it to be good.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np

from imperium.security.credentials import CredentialStore
from imperium.strategy import sector_backtest as bt
from imperium.strategy.sector_config import from_environment
from imperium.venues import registry
from imperium.venues.alpaca.client import AlpacaClient

BENCHMARK = "SPY"

#: The two things the report is swept over.
#:
#: Both are questions about the same strategy rather than variants of it:
#: whether the fill is assumed at today's close or tomorrow's open changes the
#: result by more than most parameter choices do, and a reader who sees only
#: the favourable one has been shown a number, not a measurement.
EXEC_MODES = ("near_close", "next_open")
LEVERAGE_CAPS = (1.0, 2.0)


async def fetch(symbols: list[str], start: dt.datetime,
                ) -> tuple[dict[str, np.ndarray], list[str]]:
    """Adjusted daily closes from Alpaca, aligned on a common date index."""
    store = CredentialStore()
    store.load()
    tradeable = [c for c in store if c.venue == "alpaca"]
    if not tradeable:
        raise SystemExit(
            "No Alpaca credential is stored. Open the terminal's Connections "
            "panel and add one, then run this again.")
    cred = tradeable[0]
    spec = registry.get(registry.DEFAULT_VENUE)
    client = AlpacaClient(cred.api_key, cred.secret, paper=True,
                          data_url=spec.data_url, feed=spec.default_feed)
    try:
        rows: dict[str, list[dict]] = {}
        # One symbol at a time: twenty years of daily bars is well past a
        # single page, and the client's batching is built for breadth rather
        # than for depth.
        for symbol in symbols:
            got = await client.bars([symbol], timeframe="1Day", limit=10_000,
                                    start=start, adjustment="all")
            rows[symbol] = got.get(symbol) or []
            print(f"  {symbol:6} {len(rows[symbol]):>6} bars", flush=True)
    finally:
        await client.aclose()

    by_symbol: dict[str, dict[str, float]] = {}
    for symbol, bars in rows.items():
        series = {}
        for bar in bars:
            stamp = str(bar.get("t") or "")[:10]
            close = bar.get("c")
            if stamp and isinstance(close, (int, float)) and close > 0:
                series[stamp] = float(close)
        if series:
            by_symbol[symbol] = series

    if not by_symbol:
        raise SystemExit("No bars came back. Check the key and the date range.")
    return _align(by_symbol)


def from_csv(folder: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Closes from ``<folder>/<SYMBOL>.csv`` with ``date,close`` columns."""
    by_symbol: dict[str, dict[str, float]] = {}
    for path in sorted(folder.glob("*.csv")):
        series: dict[str, float] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                stamp = (row.get("date") or row.get("Date") or "")[:10]
                raw = row.get("close") or row.get("Close")
                try:
                    value = float(raw) if raw is not None else 0.0
                except ValueError:
                    continue
                if stamp and value > 0:
                    series[stamp] = value
        if series:
            by_symbol[path.stem.upper()] = series
    if not by_symbol:
        raise SystemExit(f"No usable CSVs in {folder}")
    return _align(by_symbol)


def _align(by_symbol: dict[str, dict[str, float]],
           ) -> tuple[dict[str, np.ndarray], list[str]]:
    """Put every symbol on one date index.

    Intersected rather than forward-filled. A filled bar is an invented price,
    and a breakout rule would trade the invention -- a symbol that was halted
    for a week comes back with a flat line that reads as a tightening range and
    then a breakout, none of which happened.
    """
    common = sorted(set.intersection(*(set(v) for v in by_symbol.values())))
    closes = {s: np.asarray([by_symbol[s][d] for d in common], dtype=float)
              for s in by_symbol}
    return closes, common


def report(name: str, metrics: bt.Metrics, result: bt.BacktestResult,
           benchmark: bt.Metrics | None) -> None:
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    rows = [
        ("CAGR", f"{metrics.cagr:>9.2%}"),
        ("Volatility", f"{metrics.volatility:>9.2%}"),
        ("Sharpe (rf=0)", f"{metrics.sharpe:>9.2f}"),
        ("Sortino", f"{metrics.sortino:>9.2f}"),
        ("Max drawdown", f"{metrics.max_drawdown:>9.2%}"),
        ("Beta vs SPY", f"{metrics.beta:>9.2f}"),
        ("Alpha vs SPY", f"{metrics.alpha:>9.2%}"),
        ("Trades", f"{metrics.trades:>9d}"),
        ("Avg holding days", f"{metrics.average_hold_days:>9.1f}"),
        ("Avg gross exposure", f"{metrics.average_exposure:>9.2f}"),
        ("Slippage paid", f"{result.costs_paid:>9,.0f}"),
        ("Margin interest", f"{result.interest_paid:>9,.0f}"),
        ("Days buys were scaled", f"{result.buys_scaled:>9d}"),
        ("Years", f"{metrics.years:>9.1f}"),
        ("Final equity", f"{metrics.final_equity:>9,.0f}"),
    ]
    for label, value in rows:
        print(f"  {label:<24}{value}")

    if benchmark is not None:
        print(f"\n  {'SPY buy and hold':<24}"
              f"CAGR {benchmark.cagr:.2%}   Sharpe {benchmark.sharpe:.2f}   "
              f"MDD {benchmark.max_drawdown:.2%}")

    if metrics.yearly:
        print("\n  Yearly")
        for year in sorted(metrics.yearly):
            print(f"    {year}  {metrics.yearly[year]:>8.2%}")

    warnings = bt.sanity_check(metrics)
    if warnings:
        print("\n  !! CHECK BEFORE BELIEVING THIS")
        for note in warnings:
            print(f"     - {note}")
    else:
        print("\n  No sanity warnings: the result sits in the paper's "
              "neighbourhood.")


def build_parser(prog: str = "backtest_sector_trend") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Backtest the Sector Trend sleeve and print the report.",
        epilog=("Bars come from the Alpaca key already stored in the terminal, "
                "split- and dividend-adjusted. Nothing is ordered and nothing "
                "is armed: this only reads."),
    )
    parser.add_argument("--start", default="2005-01-01",
                        help="first bar to request (default 2005-01-01)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="read bars from a folder of <SYMBOL>.csv files "
                             "with date,close columns instead of the venue")
    parser.add_argument("--slippage-bps", type=float, default=5.0,
                        help="one-way slippage charged on every fill "
                             "(default 5)")
    parser.add_argument("--margin-rate", type=float, default=0.07,
                        help="annual rate charged on borrowed cash "
                             "(default 0.07)")
    return parser


def main(argv: list[str] | None = None,
         prog: str = "backtest_sector_trend") -> int:
    args = build_parser(prog).parse_args(
        sys.argv[1:] if argv is None else argv)

    cfg = from_environment()
    wanted = list(cfg.universe) + [BENCHMARK]

    if args.csv is not None:
        closes, dates = from_csv(args.csv)
    else:
        start = dt.datetime.fromisoformat(args.start).replace(
            tzinfo=dt.timezone.utc)
        print(f"Fetching adjusted daily bars from {args.start}…")
        closes, dates = asyncio.run(fetch(wanted, start))

    spy = closes.pop(BENCHMARK, None)
    if not closes:
        raise SystemExit("No sector ETFs had data.")
    print(f"\n{len(closes)} symbols, {len(dates)} common trading days "
          f"({dates[0]} to {dates[-1]})")

    bench_metrics = None
    for mode in EXEC_MODES:
        for cap in LEVERAGE_CAPS:
            config = bt.BacktestConfig(
                target_vol=cfg.target_vol, max_leverage=cap,
                rebalance_threshold=cfg.rebalance_threshold,
                slippage_bps=args.slippage_bps, margin_rate=args.margin_rate,
                exec_mode=mode)
            result = bt.run(closes, dates, config=config)
            bench = None
            if spy is not None and len(result.equity) >= 2:
                offset = len(dates) - len(result.equity)
                bench = spy[offset:offset + len(result.equity)]
                if bench_metrics is None:
                    holder = bt.BacktestResult(
                        dates=result.dates, equity=list(bench))
                    bench_metrics = bt.measure(holder)
            metrics = bt.measure(result, benchmark=bench, benchmark_dates=dates)
            report(f"{mode}, leverage cap {cap:.1f}x", metrics, result,
                   bench_metrics)

    print("\nThe paper's own 2005-2024 figures, on 31 ETFs at a 200% cap: "
          f"CAGR ~{bt.PAPER_CAGR:.1%}, Sharpe ~{bt.PAPER_SHARPE}, "
          f"MDD ~{bt.PAPER_MAX_DRAWDOWN:.0%}, beta ~{bt.PAPER_BETA}.")
    print("This universe is smaller, so differences are expected. Results far "
          "outside that range are a bug until proven otherwise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
