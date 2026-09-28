"""The permanent record: every fill, and the account's equity over time.

The broker keeps a ring of recent fills and the session a ring of recent
equity readings, both for the screen and both gone on a restart. That is
fine for a live view and useless for the three things that need the whole
history:

* the **trade export**, which is for the operator's own analysis and for
  tax, and must hold every fill ever made rather than the last few hundred;
* the **equity chart**, whose one-month and all-time views are the ones
  worth looking at;
* the **weekly summary**, which names the best and worst trades of the week.

TRADES
------

One CSV file, appended one row per fill, never rewritten. A spreadsheet opens
it as it is, and a crash can at worst lose the row being written -- never the
rows before it, which a rewritten file can. Every row carries the strategy
that placed it and the profit or loss that fill realised, so the export
answers "what did each strategy make" without the terminal running.

No key material is in it, by construction: a fill has a symbol, a side, a
price and a quantity, and nothing else here reads the credential store.

EQUITY
------

Two resolutions, because the two views want different things: a point every
five minutes for the last eight days, for the one-day and one-week views, and
one point per trading day for as long as the terminal has run, for the month
and all-time views. Kept in its own file, not the state file: the state file
is rewritten on every attribution save, and ten thousand points in it would
make each of those writes slower for no reason.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

TRADES_FILE = "trades.csv"
EQUITY_FILE = "equity_history.json"

#: The columns of the export, in order. Changing this order breaks every
#: spreadsheet an operator has built on the file, so new columns go last.
COLUMNS = ("time_utc", "timestamp", "mode", "symbol", "side", "quantity",
           "price", "value", "decision_price", "slippage_bps", "strategy",
           "order_type", "simulated", "realised_pnl")

#: Seconds between intraday equity points, and how long they are kept.
SAMPLE_SECONDS = 300
INTRADAY_KEEP_SECONDS = 8 * 86_400
#: Trading days kept at daily resolution. Ten years.
DAILY_KEEP = 2_600
#: How often the equity file is written. A reading lost to a crash is five
#: minutes of a line, not a fact anyone relies on.
SAVE_SECONDS = 300


@dataclass(frozen=True)
class TradeRow:
    ts: float
    mode: str
    symbol: str
    side: str
    quantity: float
    price: float
    reference: float
    slippage_bps: float
    strategy: str
    order: str
    simulated: bool
    realised: float

    @property
    def value(self) -> float:
        return self.quantity * self.price

    def cells(self) -> list[str]:
        when = dt.datetime.fromtimestamp(self.ts, tz=dt.timezone.utc)
        return [when.strftime("%Y-%m-%d %H:%M:%S"), f"{self.ts:.3f}",
                self.mode, self.symbol, self.side, _num(self.quantity),
                _num(self.price), f"{self.value:.2f}", _num(self.reference),
                f"{self.slippage_bps:.2f}", self.strategy,
                self.order or "market", "yes" if self.simulated else "no",
                f"{self.realised:.4f}"]

    def as_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "mode": self.mode, "symbol": self.symbol,
                "side": self.side, "quantity": self.quantity,
                "price": self.price, "strategy": self.strategy,
                "order": self.order, "realised": round(self.realised, 4),
                "simulated": self.simulated}


def _num(value: float) -> str:
    """A plain decimal a spreadsheet reads as a number: no exponent."""
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text or "0"


def row_from_fill(fill: Any) -> TradeRow:
    """A journal row from a broker Fill."""
    mode = getattr(getattr(fill, "mode", None), "value", "") or ""
    return TradeRow(
        ts=float(fill.ts), mode=str(mode), symbol=str(fill.symbol),
        side=str(fill.side).upper(), quantity=abs(float(fill.quantity)),
        price=float(fill.price),
        reference=float(getattr(fill, "reference_price", 0) or 0),
        slippage_bps=float(getattr(fill, "slippage_bps", 0.0) or 0.0),
        strategy=str(getattr(fill, "strategy", "") or ""),
        order=str(getattr(fill, "order", "") or ""),
        simulated=bool(getattr(fill, "simulated", True)),
        realised=float(getattr(fill, "realised", 0.0) or 0.0))


class TradeJournal:
    """Every fill, appended to one CSV file that is never rewritten."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.written = 0
        self.failed = 0
        self.last_error = ""

    def append(self, rows: Iterable[TradeRow]) -> int:
        """Append rows; returns how many were written. Never raises: a disk
        that is full must not stop the book the journal records."""
        rows = list(rows)
        if not rows:
            return 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fresh = not self.path.exists() or self.path.stat().st_size == 0
            with open(self.path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if fresh:
                    writer.writerow(COLUMNS)
                for row in rows:
                    writer.writerow(row.cells())
        except OSError as exc:
            self.failed += len(rows)
            self.last_error = str(exc)
            log.warning("could not write the trade journal: %s", exc)
            return 0
        self.written += len(rows)
        return len(rows)

    def read(self, *, since: float = 0.0, mode: str | None = None
             ) -> list[TradeRow]:
        """Rows from ``since`` on, optionally for one mode. A row that does
        not parse -- a line cut short by a crash -- is skipped."""
        if not self.path.exists():
            return []
        out: list[TradeRow] = []
        try:
            with open(self.path, newline="", encoding="utf-8") as fh:
                for raw in csv.DictReader(fh):
                    try:
                        ts = float(raw["timestamp"])
                        if ts < since or (mode and raw["mode"] != mode):
                            continue
                        out.append(TradeRow(
                            ts=ts, mode=raw["mode"], symbol=raw["symbol"],
                            side=raw["side"], quantity=float(raw["quantity"]),
                            price=float(raw["price"]),
                            reference=float(raw["decision_price"] or 0),
                            slippage_bps=float(raw["slippage_bps"] or 0),
                            strategy=raw["strategy"],
                            order="" if raw["order_type"] == "market"
                            else raw["order_type"],
                            simulated=raw["simulated"] == "yes",
                            realised=float(raw["realised_pnl"] or 0)))
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError as exc:
            log.warning("could not read the trade journal: %s", exc)
        return out

    def export(self, *, mode: str | None = None) -> str:
        """The whole journal as CSV text, optionally for one mode."""
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(COLUMNS)
        for row in self.read(mode=mode):
            writer.writerow(row.cells())
        return buffer.getvalue()


class EquityHistory:
    """The account's equity: five-minute points for a week, daily for good."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        #: (unix seconds, equity), oldest first.
        self.intraday: list[tuple[float, float]] = []
        #: (trading day, equity at the last reading that day), oldest first.
        self.daily: list[tuple[str, float]] = []
        self._saved_at = 0.0
        self._dirty = False

    def record(self, ts: float, equity: float, day: str) -> bool:
        """Take a reading. Keeps one intraday point per five minutes and
        always moves the day's close. Returns whether a point was added."""
        if not (math.isfinite(equity) and equity > 0):
            return False
        equity = float(equity)
        if self.daily and self.daily[-1][0] == day:
            self.daily[-1] = (day, equity)
        elif not self.daily or day > self.daily[-1][0]:
            self.daily.append((day, equity))
            if len(self.daily) > DAILY_KEEP:
                del self.daily[:len(self.daily) - DAILY_KEEP]
        self._dirty = True
        added = False
        if not self.intraday or ts - self.intraday[-1][0] >= SAMPLE_SECONDS:
            self.intraday.append((float(ts), equity))
            added = True
        else:
            # The latest point follows the account, so the line's live end is
            # never five minutes stale.
            self.intraday[-1] = (self.intraday[-1][0], equity)
        cutoff = ts - INTRADAY_KEEP_SECONDS
        if self.intraday and self.intraday[0][0] < cutoff:
            self.intraday = [p for p in self.intraday if p[0] >= cutoff]
        return added

    def series(self, seconds: float | None, now: float | None = None
               ) -> list[tuple[float, float]]:
        """Points for a view ``seconds`` long, or all of it for None.

        Up to a week is read from the five-minute points; anything longer
        from the daily closes, stamped at 21:00 UTC -- after the US close in
        either half of the year -- so both resolutions share one time axis.
        """
        now = time.time() if now is None else now
        if seconds is not None and seconds <= INTRADAY_KEEP_SECONDS:
            start = now - seconds
            return [p for p in self.intraday if p[0] >= start]
        out = []
        for day, value in self.daily:
            try:
                stamp = dt.datetime.fromisoformat(day).replace(
                    hour=21, tzinfo=dt.timezone.utc).timestamp()
            except ValueError:
                continue
            # Not after now: today's close is stamped at 21:00 UTC, which in
            # the morning is hours ahead, and the line would run past the
            # present. Today comes from the live reading below instead.
            if stamp > now:
                continue
            if seconds is None or stamp >= now - seconds:
                out.append((stamp, value))
        # The live reading as the last point, so the long views end today.
        if self.intraday and (not out or self.intraday[-1][0] > out[-1][0]):
            out.append(self.intraday[-1])
        return out

    # -- persistence -----------------------------------------------------

    def maybe_save(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if not self._dirty or now - self._saved_at < SAVE_SECONDS:
            return False
        return self.save(now)

    def save(self, now: float | None = None) -> bool:
        payload = {"intraday": [[round(t, 1), round(v, 4)]
                                for t, v in self.intraday],
                   "daily": [[d, round(v, 4)] for d, v in self.daily]}
        tmp = self.path.with_suffix(".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, separators=(",", ":")),
                           encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("could not save the equity history: %s", exc)
            return False
        self._saved_at = time.time() if now is None else now
        self._dirty = False
        return True

    @classmethod
    def load(cls, path: Path) -> "EquityHistory":
        out = cls(path)
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return out
        for item in payload.get("intraday") or []:
            try:
                t, v = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(t) and math.isfinite(v) and v > 0:
                out.intraday.append((t, v))
        for item in payload.get("daily") or []:
            try:
                d, v = str(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(v) and v > 0:
                out.daily.append((d, v))
        out.intraday.sort()
        out.daily.sort()
        out._saved_at = time.time()
        return out
