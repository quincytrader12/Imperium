"""Run the Sector Trend backtest from a source checkout.

    uv run python scripts/backtest_sector_trend.py
    uv run python scripts/backtest_sector_trend.py --start 2005-01-01
    uv run python scripts/backtest_sector_trend.py --csv ./bars

The backtest itself lives in ``imperium.strategy.backtest_cli``, because the
packaged build ships the package and not this folder -- someone running the
.exe gets the same command as ``IMPERIUM.exe --backtest``. All this file does
is put ``src`` on the path first, which a checkout needs and the build does
not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperium.strategy.backtest_cli import main   # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(prog="scripts/backtest_sector_trend.py"))
