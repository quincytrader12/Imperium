"""A second currency beside the account's own.

The account is denominated in dollars and every number this program reasons
about stays in dollars -- positions, limits, the cost gate, the risk budget.
This converts *for display only*, because "my balance is $73.40" is not what an
operator in Johannesburg thinks in, and a terminal they have to do arithmetic
against is one they read less often.

**The rule this module exists to keep.** A converted figure is never shown
without the rate that produced it and how old that rate is. A number in rands
with no rate beside it is a claim the program cannot support: the rate moves,
the last fetch may have failed hours ago, and an operator reading a stale
conversion as current is worse off than one reading dollars.

Nothing here is ever used for sizing, risk, or any decision. It is a label.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger("imperium.fx")

#: The European Central Bank's reference rates, served without a key.
#:
#: Chosen because it needs no account and no credential: a display convenience
#: should not require the operator to sign up for anything, and should not put
#: another key on the machine.
API = "https://api.frankfurter.app/latest"

TIMEOUT = 8.0

#: How often to re-ask, in seconds. The ECB publishes once a working day, so
#: an hour is already far more often than the number changes.
REFRESH_SECONDS = 3600.0

#: Past this age the rate is shown as stale rather than quietly presented as
#: current. Six hours: long enough to survive a transient outage, short enough
#: that a rate from yesterday is never passed off as today's.
STALE_AFTER = 6 * 3600.0


@dataclass
class Rate:
    """One conversion, with everything needed to judge it."""

    base: str = "USD"
    quote: str = ""
    rate: float = 0.0
    fetched_at: float = 0.0
    source: str = ""
    error: str = ""

    @property
    def known(self) -> bool:
        return self.rate > 0

    @property
    def age(self) -> float:
        return time.time() - self.fetched_at if self.fetched_at else float("inf")

    @property
    def stale(self) -> bool:
        return self.known and self.age > STALE_AFTER

    def convert(self, amount: float) -> float:
        return float(amount) * self.rate

    def as_dict(self) -> dict[str, object]:
        return {
            "quote": self.quote,
            "rate": round(self.rate, 4) if self.known else None,
            "known": self.known,
            "stale": self.stale,
            "age": round(self.age) if self.fetched_at else None,
            "source": self.source,
            "error": self.error,
        }


class FxDesk:
    """Holds the current rate and refreshes it on a slow interval.

    Never raises into the caller. A currency label that takes the terminal
    down would be an absurd trade, so every failure is absence plus a reason.
    """

    def __init__(self, quote: str = "", *, manual_rate: float = 0.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.quote = (quote or "").strip().upper()
        self.transport = transport
        self.rate = Rate(quote=self.quote)
        if manual_rate > 0:
            # An operator-supplied rate is treated as known and never stale:
            # they typed it, they own it, and overriding it with a fetch would
            # ignore the instruction.
            self.rate = Rate(quote=self.quote, rate=float(manual_rate),
                             fetched_at=time.time(), source="set by hand")
        self.manual = manual_rate > 0
        self._checked_at = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.quote)

    def due(self, *, now: float | None = None) -> bool:
        if not self.enabled or self.manual:
            return False
        at = time.time() if now is None else now
        return at - self._checked_at >= REFRESH_SECONDS

    async def refresh(self) -> None:
        """Ask for the rate. Absence on failure, never an exception."""
        if not self.enabled or self.manual:
            return
        self._checked_at = time.time()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT,
                                         transport=self.transport) as client:
                response = await client.get(
                    API, params={"from": "USD", "to": self.quote})
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.json()
            value = float((payload.get("rates") or {}).get(self.quote) or 0.0)
            if value <= 0:
                raise RuntimeError(f"no {self.quote} rate in the reply")
        except Exception as exc:                            # noqa: BLE001
            # Deliberately broad: this runs on the trading loop's schedule and
            # no failure of a display conversion is worth interrupting it. The
            # previous rate is kept and ages into "stale" on its own.
            self.rate.error = f"{type(exc).__name__}: {exc}"
            log.debug("fx refresh failed: %s", self.rate.error)
            return
        self.rate = Rate(base="USD", quote=self.quote, rate=value,
                         fetched_at=time.time(), source="ECB via frankfurter")

    def panel(self) -> dict[str, object]:
        return {"enabled": self.enabled, **self.rate.as_dict()}
