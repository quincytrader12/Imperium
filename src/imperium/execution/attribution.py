"""Which strategy earned what.

The book trades several strategies at once -- the intraday blend, the
multi-day trend, the overnight drift, the crypto cross-section -- and until
this existed nothing could say which of them made the money and which lost
it. A ``Fill`` did not even record who placed it. An autonomous fund that
cannot answer that question cannot move capital toward what works, which is
the only thing that makes it a fund rather than a set of fixed rules.

THE ACCOUNTING RULE
-------------------

An entry is booked to the strategy that decided it. Every exit -- the
give-back ratchet closing a winner, the overnight exit on the opening
auction, a flatten on retirement -- is booked to the strategy that *opened*
the position. A round trip's profit belongs to whoever took the risk, not to
whatever happened to close it; booking the ratchet's exits to a "protect"
row would credit it with every winner the other strategies found.

Average cost, per strategy and symbol. Realised P&L is booked as a position
is reduced; a round trip is counted when a lot returns to flat, and it is a
win or a loss by the sum of everything realised on the way down.

WHAT THIS DELIBERATELY DOES NOT CLAIM
------------------------------------

* **Fees are not in it.** US equities trade commission-free here; crypto fees
  are taken in the asset, which shows up as a quantity the venue holds that
  differs from the fills -- and that is exactly what :meth:`sync` records as
  a correction, rather than inventing a fee figure.
* **Slippage is not added to P&L.** It is already inside it: the fill price
  is what was paid. It is tracked separately as a diagnostic -- what the
  crossing cost against the price the decision was taken at -- because that
  is the number the cost gate needs to be checked against.
* **A position found at start with no known owner is not guessed at.** It is
  booked to :data:`UNATTRIBUTED`, visibly, until it closes.

One book per mode. A dry run's simulated fills are not evidence of the same
quality as the venue's paper account, and mixing them into one record would
let the easier one flatter the harder one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

#: Owner of a position nobody here opened: held at start, or traded outside
#: this program. Kept visible rather than folded into a strategy, because
#: crediting a strategy with a position it never took would be a lie in the
#: one table meant to be trusted.
UNATTRIBUTED = "unattributed"

#: Daily marks kept per strategy. A year and a half of trading days is enough
#: for any statistic the allocator will ask of it, and bounded so a terminal
#: that runs for years does not grow its state file without end.
MAX_DAILY_POINTS = 400

#: Quantities below this are flat. Fractional shares and crypto units arrive
#: as floats with representation noise; a lot left at 1e-12 is not a position.
FLAT = 1e-9


@dataclass
class Lot:
    """One strategy's position in one symbol."""

    quantity: float = 0.0
    avg_price: float = 0.0
    #: Realised since this lot last opened, so a round trip can be judged as
    #: a whole when it closes rather than one partial exit at a time.
    realised: float = 0.0
    opened_at: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < FLAT


@dataclass
class StrategyRecord:
    """Everything one strategy has done, in one mode."""

    name: str
    realised: float = 0.0
    #: Dollars paid against the price the decision was taken at. Positive is a
    #: cost. Already inside ``realised`` -- this is the diagnostic, not a
    #: second charge.
    slippage: float = 0.0
    traded: float = 0.0
    fills: int = 0
    round_trips: int = 0
    wins: int = 0
    losses: int = 0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    first_fill_at: float = 0.0
    last_fill_at: float = 0.0
    lots: dict[str, Lot] = field(default_factory=dict)
    #: (trading day, realised + unrealised at that day's mark), oldest first.
    #: The raw material for the allocator: daily changes are daily returns.
    daily: list[tuple[str, float]] = field(default_factory=list)

    def unrealised(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, lot in self.lots.items():
            if lot.is_flat:
                continue
            price = prices.get(symbol) or 0.0
            if price > 0:
                total += (price - lot.avg_price) * lot.quantity
        return total

    def open_symbols(self) -> list[str]:
        return sorted(s for s, lot in self.lots.items() if not lot.is_flat)

    def max_drawdown(self) -> float:
        """Largest fall from a daily high, in dollars. Zero with no history."""
        peak = -math.inf
        worst = 0.0
        for _, value in self.daily:
            peak = max(peak, value)
            worst = max(worst, peak - value)
        return worst


class StrategyBook:
    """The per-strategy record for one mode."""

    def __init__(self) -> None:
        self.records: dict[str, StrategyRecord] = {}
        #: symbol -> the strategy that most recently opened it. The owner an
        #: unnamed exit is booked to.
        self.owners: dict[str, str] = {}
        #: How many times the venue's position disagreed with what the fills
        #: said. Each is a fill this book never saw -- a partial auction fill,
        #: a crypto fee taken in the asset, a trade made by hand -- and is
        #: counted rather than absorbed so it cannot quietly accumulate.
        self.corrections: int = 0
        #: The fund's equity at each day's mark. The denominator that turns a
        #: strategy's daily change in profit into its contribution to the
        #: fund's return, which is what the allocator measures.
        self.equity_marks: dict[str, float] = {}

    # -- recording -------------------------------------------------------

    def record(self, name: str) -> StrategyRecord:
        found = self.records.get(name)
        if found is None:
            found = StrategyRecord(name=name)
            self.records[name] = found
        return found

    def owner(self, symbol: str) -> str:
        return self.owners.get(symbol, UNATTRIBUTED)

    def book(self, *, symbol: str, side: str, quantity: float, price: float,
             ts: float, strategy: str = "",
             reference_price: float = 0.0) -> str:
        """Book one fill. Returns the strategy it was booked to.

        ``strategy`` is the name of whoever decided it, or empty for an exit
        that belongs to whoever holds the position.
        """
        quantity = abs(float(quantity))
        price = float(price)
        if quantity < FLAT or price <= 0:
            return strategy or self.owner(symbol)
        signed = quantity if side.upper() == "BUY" else -quantity

        # Which lot this changes. A named strategy trades its own lot. An
        # unnamed fill reduces the owner's -- and if the owner's lot is not on
        # the other side of it, this is not an exit at all, and the owner is
        # still the right place for it.
        name = strategy or self.owner(symbol)
        rec = self.record(name)
        lot = rec.lots.setdefault(symbol, Lot())

        # An unnamed fill that the owner cannot absorb spills into whoever
        # else holds the symbol. Rare -- the engine gives one symbol to one
        # strategy at a time -- but a handover can leave two lots behind.
        if not strategy and self._reduces(lot, signed) is False:
            other = self._other_holder(symbol, against=signed, skip=name)
            if other is not None:
                name, rec = other, self.record(other)
                lot = rec.lots[symbol]

        rec.fills += 1
        rec.traded += quantity * price
        rec.first_fill_at = rec.first_fill_at or ts
        rec.last_fill_at = ts
        if reference_price and reference_price > 0:
            # Buying above the decision price costs; selling below it costs.
            rec.slippage += (price - reference_price) * signed

        self._apply(rec, lot, signed, price, ts)
        if strategy and not lot.is_flat:
            self.owners[symbol] = strategy
        elif lot.is_flat and self.owners.get(symbol) == name:
            # Handed to whoever else still holds it, or released.
            holder = self._other_holder(symbol, against=0.0, skip=name)
            if holder is None:
                self.owners.pop(symbol, None)
            else:
                self.owners[symbol] = holder
        return name

    @staticmethod
    def _reduces(lot: Lot, signed: float) -> bool:
        return (not lot.is_flat) and (lot.quantity > 0) != (signed > 0)

    def _other_holder(self, symbol: str, *, against: float,
                      skip: str) -> str | None:
        """Another strategy with an open lot in ``symbol``.

        With ``against`` non-zero, only one whose lot that fill would reduce.
        Largest first, so the choice is deterministic.
        """
        best: tuple[float, str] | None = None
        for name, rec in self.records.items():
            if name == skip:
                continue
            lot = rec.lots.get(symbol)
            if lot is None or lot.is_flat:
                continue
            if against and not self._reduces(lot, against):
                continue
            size = abs(lot.quantity)
            if best is None or size > best[0]:
                best = (size, name)
        return best[1] if best else None

    @staticmethod
    def _apply(rec: StrategyRecord, lot: Lot, signed: float, price: float,
               ts: float) -> None:
        """Average-cost update, realising whatever the fill closes."""
        if lot.is_flat:
            lot.quantity, lot.avg_price = signed, price
            lot.realised, lot.opened_at = 0.0, ts
            return
        if (lot.quantity > 0) == (signed > 0):
            total = lot.quantity + signed
            lot.avg_price = ((lot.avg_price * lot.quantity + price * signed)
                             / total)
            lot.quantity = total
            return

        closing = min(abs(signed), abs(lot.quantity))
        direction = 1.0 if lot.quantity > 0 else -1.0
        pnl = (price - lot.avg_price) * closing * direction
        rec.realised += pnl
        lot.realised += pnl
        remainder = abs(signed) - closing
        lot.quantity += closing * (-direction)

        if lot.is_flat:
            lot.quantity = 0.0
            rec.round_trips += 1
            if lot.realised > 0:
                rec.wins += 1
                rec.gross_win += lot.realised
            elif lot.realised < 0:
                rec.losses += 1
                rec.gross_loss += -lot.realised
            lot.avg_price, lot.realised = 0.0, 0.0
            if remainder > FLAT:
                # Went through flat and out the other side: a new lot, at this
                # price, in the fill's direction.
                lot.quantity = remainder * (1.0 if signed > 0 else -1.0)
                lot.avg_price, lot.opened_at = price, ts

    # -- the venue is the truth ------------------------------------------

    def sync(self, positions: dict[str, tuple[float, float]]) -> list[str]:
        """Make the lots agree with what the venue actually holds.

        ``positions`` maps symbol to (quantity, average price) from the
        broker's book, which reconcile keeps true to the venue. Returns the
        symbols corrected.

        A difference is resized, never booked as profit or loss: the fill that
        caused it was never seen, so its price is unknown, and inventing one
        would put a number in this table that nothing measured. The count of
        corrections is kept and shown instead.
        """
        corrected: list[str] = []
        symbols = set(positions) | {s for rec in self.records.values()
                                    for s, lot in rec.lots.items()
                                    if not lot.is_flat}
        for symbol in sorted(symbols):
            venue_qty, venue_avg = positions.get(symbol, (0.0, 0.0))
            held = self._held(symbol)
            if abs(held - venue_qty) < max(FLAT, abs(venue_qty) * 1e-6):
                continue
            corrected.append(symbol)
            self.corrections += 1
            if abs(venue_qty) < FLAT:
                for rec in self.records.values():
                    lot = rec.lots.get(symbol)
                    if lot is not None:
                        lot.quantity, lot.avg_price, lot.realised = 0.0, 0.0, 0.0
                self.owners.pop(symbol, None)
                continue
            name = self.owners.get(symbol)
            if name is None:
                name = UNATTRIBUTED
                self.owners[symbol] = name
            lot = self.record(name).lots.setdefault(symbol, Lot())
            others = held - (0.0 if lot.is_flat else lot.quantity)
            lot.quantity = venue_qty - others
            if lot.avg_price <= 0:
                lot.avg_price = float(venue_avg)
            if lot.is_flat:
                lot.quantity = 0.0
        return corrected

    def _held(self, symbol: str) -> float:
        return sum(lot.quantity for rec in self.records.values()
                   for s, lot in rec.lots.items() if s == symbol)

    # -- marks -----------------------------------------------------------

    def mark_day(self, day: str, prices: dict[str, float],
                 equity: float | None = None) -> None:
        """Record each strategy's total P&L at this day's mark.

        Idempotent per day: marking twice replaces the first, so a restart
        that marks again does not put two points on one day -- which would
        read as a day with no change and flatter every volatility figure.
        """
        if equity is not None and equity > 0 and math.isfinite(equity):
            self.equity_marks[day] = float(equity)
            if len(self.equity_marks) > MAX_DAILY_POINTS:
                for old in sorted(self.equity_marks)[:len(self.equity_marks)
                                                     - MAX_DAILY_POINTS]:
                    del self.equity_marks[old]
        for rec in self.records.values():
            value = rec.realised + rec.unrealised(prices)
            if rec.daily and rec.daily[-1][0] == day:
                rec.daily[-1] = (day, value)
            else:
                rec.daily.append((day, value))
            if len(rec.daily) > MAX_DAILY_POINTS:
                del rec.daily[:len(rec.daily) - MAX_DAILY_POINTS]

    def summary(self, prices: dict[str, float]) -> list[dict[str, Any]]:
        """One row per strategy, best total first. What the terminal shows."""
        rows = []
        for rec in self.records.values():
            unrealised = rec.unrealised(prices)
            closed = rec.wins + rec.losses
            rows.append({
                "strategy": rec.name,
                "total": round(rec.realised + unrealised, 4),
                "realised": round(rec.realised, 4),
                "unrealised": round(unrealised, 4),
                "fills": rec.fills,
                "round_trips": rec.round_trips,
                "wins": rec.wins,
                "losses": rec.losses,
                # None rather than 0 with nothing closed: "no trades yet" and
                # "never wins" are different facts.
                "hit_rate": (rec.wins / closed) if closed else None,
                "profit_factor": ((rec.gross_win / rec.gross_loss)
                                  if rec.gross_loss > 0 else None),
                "slippage": round(rec.slippage, 4),
                "traded": round(rec.traded, 2),
                "open": rec.open_symbols(),
                "max_drawdown": round(rec.max_drawdown(), 4),
                "days": len(rec.daily),
            })
        rows.sort(key=lambda r: r["total"], reverse=True)
        return rows

    # -- persistence -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "owners": dict(self.owners),
            "corrections": self.corrections,
            "equity_marks": dict(self.equity_marks),
            "records": {
                name: {
                    "realised": rec.realised, "slippage": rec.slippage,
                    "traded": rec.traded, "fills": rec.fills,
                    "round_trips": rec.round_trips, "wins": rec.wins,
                    "losses": rec.losses, "gross_win": rec.gross_win,
                    "gross_loss": rec.gross_loss,
                    "first_fill_at": rec.first_fill_at,
                    "last_fill_at": rec.last_fill_at,
                    "lots": {s: {"quantity": lot.quantity,
                                 "avg_price": lot.avg_price,
                                 "realised": lot.realised,
                                 "opened_at": lot.opened_at}
                             for s, lot in rec.lots.items() if not lot.is_flat},
                    "daily": [list(p) for p in rec.daily],
                }
                for name, rec in self.records.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "StrategyBook":
        """Rebuild from disk. Anything malformed is dropped, not guessed at."""
        book = cls()
        if not isinstance(payload, dict):
            return book
        owners = payload.get("owners")
        if isinstance(owners, dict):
            book.owners = {str(k): str(v) for k, v in owners.items()}
        try:
            book.corrections = int(payload.get("corrections") or 0)
        except (TypeError, ValueError):
            book.corrections = 0
        marks = payload.get("equity_marks")
        if isinstance(marks, dict):
            for day, value in marks.items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if value > 0 and math.isfinite(value):
                    book.equity_marks[str(day)] = value
        records = payload.get("records")
        if not isinstance(records, dict):
            return book
        for name, raw in records.items():
            if not isinstance(raw, dict):
                continue
            try:
                rec = StrategyRecord(
                    name=str(name),
                    realised=float(raw.get("realised") or 0.0),
                    slippage=float(raw.get("slippage") or 0.0),
                    traded=float(raw.get("traded") or 0.0),
                    fills=int(raw.get("fills") or 0),
                    round_trips=int(raw.get("round_trips") or 0),
                    wins=int(raw.get("wins") or 0),
                    losses=int(raw.get("losses") or 0),
                    gross_win=float(raw.get("gross_win") or 0.0),
                    gross_loss=float(raw.get("gross_loss") or 0.0),
                    first_fill_at=float(raw.get("first_fill_at") or 0.0),
                    last_fill_at=float(raw.get("last_fill_at") or 0.0),
                )
                for symbol, lot in (raw.get("lots") or {}).items():
                    rec.lots[str(symbol)] = Lot(
                        quantity=float(lot.get("quantity") or 0.0),
                        avg_price=float(lot.get("avg_price") or 0.0),
                        realised=float(lot.get("realised") or 0.0),
                        opened_at=float(lot.get("opened_at") or 0.0))
                rec.daily = [(str(d), float(v))
                             for d, v in (raw.get("daily") or [])]
            except (AttributeError, TypeError, ValueError):
                continue
            book.records[rec.name] = rec
        return book


class Attribution:
    """The books for every mode, and the cursor into the broker's fills.

    Reads fills from the broker's own ring rather than being handed them at
    each call site. There are six places that place orders; a ledger fed by
    each of them is one forgotten call away from silently missing a strategy,
    and reading the ring means every fill is booked exactly once whatever
    path produced it.
    """

    def __init__(self) -> None:
        self.books: dict[str, StrategyBook] = {}
        self._broker_id: int | None = None
        self._seen: int = 0
        #: Fills that left the ring before they were read. Should stay zero;
        #: shown if it does not, because it means the table is incomplete.
        self.missed: int = 0

    def book_for(self, mode: str) -> StrategyBook:
        found = self.books.get(mode)
        if found is None:
            found = StrategyBook()
            self.books[mode] = found
        return found

    def consume(self, broker: Any) -> int:
        """Book every fill the broker has made since the last call."""
        return len(self.consume_fills(broker))

    def consume_fills(self, broker: Any) -> list[Any]:
        """Book every new fill and return them, oldest first.

        Returned so that whatever else learns from fills -- the crossing-cost
        calibration -- reads the same ones, exactly once, from the same
        cursor, instead of keeping a second one that could disagree.

        A replaced broker -- a mode switch builds a new one -- starts its own
        count at zero, so the cursor restarts with it rather than skipping the
        new broker's first fills as already seen.
        """
        if id(broker) != self._broker_id:
            self._broker_id = id(broker)
            self._seen = 0
        total = int(getattr(broker, "fills_total", 0) or 0)
        fresh = total - self._seen
        if fresh <= 0:
            return []
        ring = list(getattr(broker, "fills", []) or [])
        if fresh > len(ring):
            self.missed += fresh - len(ring)
            fresh = len(ring)
        mode = getattr(getattr(broker, "mode", None), "value", "unknown")
        book = self.book_for(str(mode))
        batch = ring[len(ring) - fresh:]
        for fill in batch:
            before = sum(r.realised for r in book.records.values())
            owner = book.book(
                symbol=fill.symbol, side=fill.side,
                quantity=float(fill.quantity), price=float(fill.price),
                ts=float(fill.ts), strategy=getattr(fill, "strategy", "") or "",
                reference_price=float(getattr(fill, "reference_price", 0) or 0))
            # Written back, so the journal on screen shows who placed it and
            # the permanent one what it made.
            realised = sum(r.realised for r in book.records.values()) - before
            try:
                fill.strategy = owner
                fill.realised = realised
            except AttributeError:
                pass
        self._seen = total
        return batch

    def as_dict(self) -> dict[str, Any]:
        return {"books": {m: b.as_dict() for m, b in self.books.items()},
                "missed": self.missed}

    @classmethod
    def from_dict(cls, payload: Any) -> "Attribution":
        out = cls()
        if isinstance(payload, dict):
            books = payload.get("books")
            if isinstance(books, dict):
                out.books = {str(m): StrategyBook.from_dict(b)
                             for m, b in books.items()}
            try:
                out.missed = int(payload.get("missed") or 0)
            except (TypeError, ValueError):
                out.missed = 0
        return out


def positions_of(broker: Any) -> dict[str, tuple[float, float]]:
    """(quantity, average price) per symbol, from any broker's book."""
    out: dict[str, tuple[float, float]] = {}
    for symbol, pos in (getattr(broker, "positions", {}) or {}).items():
        if pos is None or pos.is_flat:
            continue
        out[symbol] = (float(pos.quantity), float(pos.avg_price))
    return out


def names(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [r["strategy"] for r in rows]
