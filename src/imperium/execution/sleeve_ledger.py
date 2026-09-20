"""This sleeve's own book, kept apart from the account's.

**Why this file has to exist.** Alpaca nets positions by symbol across the
whole account. If the multi-day trend strategy is long 3 shares of XLK and this
sleeve enters 2 more, the venue reports one position of 5 and nothing in that
number says which strategy owns what. A sleeve that sized or closed from the
account position would, on that example, sell another strategy's shares to
"exit" its own -- and the other strategy would then reconcile against a
position it never changed. Both books would be wrong, and the first evidence
would be a loss neither could explain.

So the sleeve keeps its own quantities, and every decision it makes is against
*those*. The account position is never read for sizing or for exits. It is read
for exactly one purpose: to notice that someone else is also holding the symbol
and say so.

**Where it lives.** ``~/.imperium/state.json``, under its own key, alongside
the overnight and trend holdings the terminal already persists there. The
brief asks for a table; this repo has no SQL and inventing a database for one
strategy would be a second way of storing state rather than a better one.
Writes are read-modify-write so the sleeve cannot clobber the rest of the book,
which is the same rule the credential-name write follows.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from imperium import config

log = logging.getLogger("imperium.sleeve")

#: The key this sleeve owns inside the shared state file.
STATE_KEY = "sector_trend"


@dataclass
class SleevePosition:
    """One symbol this sleeve holds, and the stop protecting it."""

    symbol: str
    quantity: float = 0.0
    #: The trailing stop, carried across days. This is the number the exit
    #: rule reads, and it is persisted because a terminal that restarts must
    #: not forget where its stops were -- a forgotten stop is an unbounded
    #: position.
    stop: float = float("nan")
    entry_price: float = 0.0
    entered_on: str = ""
    days_held: int = 0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-9

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        # NaN is not JSON. A missing stop is stored as null and read back as
        # NaN, rather than as a number that would be treated as a real stop.
        out["stop"] = None if not math.isfinite(self.stop) else self.stop
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SleevePosition":
        stop = payload.get("stop")
        return cls(
            symbol=str(payload.get("symbol") or ""),
            quantity=float(payload.get("quantity") or 0.0),
            stop=float(stop) if isinstance(stop, (int, float)) else float("nan"),
            entry_price=float(payload.get("entry_price") or 0.0),
            entered_on=str(payload.get("entered_on") or ""),
            days_held=int(payload.get("days_held") or 0),
        )


@dataclass
class SleeveLedger:
    """The sleeve's positions, stops, and the last day it completed a run."""

    positions: dict[str, SleevePosition] = field(default_factory=dict)
    #: The trading day whose run finished. The idempotency guard reads this:
    #: a second run on the same day must place no orders, because the first one
    #: already moved the book to its target and the second would compute the
    #: same target and trade the difference twice.
    last_run_day: str = ""
    runs: int = 0
    #: Set once, when equity first reached the arming threshold. Persisted so
    #: a restart does not re-announce it, and so the sleeve stays armed after
    #: a drawdown that takes equity back below the threshold -- see
    #: SectorRunner.consider_arming for why disarming is the wrong move.
    armed_at_equity: float = 0.0
    armed_on: str = ""
    #: How it came to be armed: "equity" when the threshold was reached,
    #: "hand" when the operator pressed the button. Worth recording because
    #: the two answer different questions later -- a sleeve that armed itself
    #: is the program doing something unattended, and a sleeve armed by hand
    #: is a decision somebody made and may not remember making.
    armed_by: str = ""

    # -- reading ---------------------------------------------------------

    def held(self, symbol: str) -> SleevePosition:
        """This sleeve's position in a symbol. Never the account's."""
        found = self.positions.get(symbol)
        return found if found is not None else SleevePosition(symbol=symbol)

    def longs(self) -> list[str]:
        return sorted(s for s, p in self.positions.items() if not p.is_flat)

    def already_ran(self, day: str) -> bool:
        return bool(day) and self.last_run_day == day

    # -- writing ---------------------------------------------------------

    def open_position(self, symbol: str, quantity: float, price: float,
                      stop: float, day: str) -> SleevePosition:
        position = SleevePosition(symbol=symbol, quantity=float(quantity),
                                  stop=float(stop), entry_price=float(price),
                                  entered_on=day, days_held=0)
        self.positions[symbol] = position
        return position

    def resize(self, symbol: str, quantity: float) -> None:
        position = self.positions.get(symbol)
        if position is not None:
            position.quantity = float(quantity)

    def raise_stop(self, symbol: str, lower_band: float) -> float:
        """Trail the stop up, never down. Returns the stop now in force."""
        from imperium.strategy.sector import trail_stop

        position = self.positions.get(symbol)
        if position is None:
            return float("nan")
        position.stop = trail_stop(position.stop, lower_band)
        return position.stop

    def close_position(self, symbol: str) -> None:
        """Remove the symbol and its stop together.

        Both at once, deliberately. A stop left behind on a flat symbol is a
        stop that will fire against the next entry on the day it is opened,
        using a level from a position that no longer exists.
        """
        self.positions.pop(symbol, None)

    def age_positions(self) -> None:
        for position in self.positions.values():
            position.days_held += 1

    def complete_run(self, day: str) -> None:
        self.last_run_day = day
        self.runs += 1

    # -- persistence -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "positions": {s: p.as_dict() for s, p in self.positions.items()},
            "last_run_day": self.last_run_day,
            "runs": self.runs,
            "armed_at_equity": self.armed_at_equity,
            "armed_on": self.armed_on,
            "armed_by": self.armed_by,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SleeveLedger":
        rows = payload.get("positions") or {}
        positions: dict[str, SleevePosition] = {}
        if isinstance(rows, dict):
            for symbol, row in rows.items():
                if isinstance(row, dict):
                    position = SleevePosition.from_dict({**row,
                                                         "symbol": symbol})
                    positions[symbol] = position
        return cls(
            positions=positions,
            last_run_day=str(payload.get("last_run_day") or ""),
            runs=int(payload.get("runs") or 0),
            armed_at_equity=float(payload.get("armed_at_equity") or 0.0),
            armed_on=str(payload.get("armed_on") or ""),
            armed_by=str(payload.get("armed_by") or ""),
        )

    @classmethod
    def load(cls) -> "SleeveLedger":
        """Read the sleeve's book. A missing or broken file is an empty book."""
        try:
            raw = config.state_path().read_text(encoding=config.TEXT_ENCODING)
            payload = json.loads(raw)
        except (OSError, ValueError):
            return cls()
        if not isinstance(payload, dict):
            return cls()
        section = payload.get(STATE_KEY)
        return cls.from_dict(section) if isinstance(section, dict) else cls()

    def save(self) -> None:
        """Merge into the shared state file, never overwrite it.

        The same file carries the overnight holdings, the trend holdings and
        the name of the attached credential. Writing this sleeve's section by
        replacing the file would lose all of them, which on a restart is the
        whole book.
        """
        try:
            config.ensure_home()
            path = config.state_path()
            payload: dict[str, Any] = {}
            try:
                loaded = json.loads(path.read_text(encoding=config.TEXT_ENCODING))
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, ValueError):
                payload = {}
            payload[STATE_KEY] = self.as_dict()
            path.write_text(json.dumps(payload, indent=2),
                            encoding=config.TEXT_ENCODING)
            try:
                path.chmod(0o600)
            except (OSError, NotImplementedError):
                pass
        except OSError as exc:
            # Best effort, like every other write to this file -- but unlike
            # the others this one loses stops, so it is logged loudly rather
            # than passed over.
            log.error("could not persist the sector sleeve ledger: %s", exc)


def _nth_sunday(year: int, month: int, nth: int) -> int:
    """Day of the month of the nth Sunday. ``nth`` is 1-based."""
    first = dt.date(year, month, 1)
    # weekday(): Monday is 0, Sunday is 6.
    return 1 + (6 - first.weekday()) % 7 + 7 * (nth - 1)


def eastern_offset(moment: dt.datetime) -> dt.timedelta:
    """The UTC offset for US Eastern, from the statutory rule.

    Since the Energy Policy Act of 2005 took effect in 2007, US daylight time
    runs from 02:00 local on the second Sunday in March to 02:00 local on the
    first Sunday in November. Those two instants are 07:00 and 06:00 UTC, and
    the offset is -4 between them and -5 outside.

    Computed rather than looked up because the alternative is shipping the
    IANA database -- 605 files, for one zone -- inside a Windows executable
    whose first launch is already scanned file by file by Defender. That is
    what this program did for exactly one build, and that build did not start
    inside the twenty seconds its own launcher waits.

    The liability is that this is law, and law changes: a US move to permanent
    daylight time would make it wrong, silently, by an hour. That is why the
    IANA database is still preferred wherever it exists -- see to_eastern --
    and why this is the fallback rather than the rule.
    """
    year = moment.year
    begins = dt.datetime(year, 3, _nth_sunday(year, 3, 2), 7,
                         tzinfo=dt.timezone.utc)
    ends = dt.datetime(year, 11, _nth_sunday(year, 11, 1), 6,
                       tzinfo=dt.timezone.utc)
    return dt.timedelta(hours=-4 if begins <= moment < ends else -5)


def to_eastern(now: dt.datetime | None = None) -> dt.datetime:
    """A moment in US Eastern terms, on a machine that may not know what that is.

    Windows ships no IANA time zone database, so
    ``ZoneInfo("America/New_York")`` raises there unless ``tzdata`` is
    installed -- and this program's whole reason for existing is to run
    unattended on a Windows desktop.

    Bundling tzdata solved that and caused a worse problem: 605 files for one
    zone, in an executable Defender scans file by file on first launch, and a
    build that did not answer within the twenty seconds its own launcher
    waits before giving up on opening a browser.

    So the database is used where it already exists, which is every developer
    machine and every CI runner, and the statutory rule stands in where it
    does not. The fallback is now *correct* rather than an hour wrong, which
    is what makes it safe to rely on.
    """
    moment = now or dt.datetime.now(tz=dt.timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        return moment.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        return moment.astimezone(dt.timezone(eastern_offset(moment)))


def trading_day(now: dt.datetime | None = None) -> str:
    """The calendar day a run belongs to, in US Eastern terms.

    Eastern rather than UTC because the run is scheduled against the US equity
    session. A UTC day boundary would put a 15:45 ET run in New York on one
    date and the same run in December on another, and the idempotency guard
    would let a second run through on exactly the days it mattered.
    """
    return to_eastern(now).date().isoformat()
