"""The trading session: the object the server exposes and the UI renders.

Owns the venue client, the feed, the engines, the allocator and the broker, and
produces the one snapshot the websocket streams at a fixed 1Hz.

Everything here is failure-tolerant by construction. A venue that rejects a key,
a symbol that does not exist, a rate limit -- these are information the operator
needs on screen, and none of them may kill a background loop or reach the UI as
a traceback.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import math
import time

import numpy as np
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from imperium import config
from imperium.execution.bars import Bar
from imperium.keepalive import KeepAwake
from imperium.execution.broker import (
    MARKET_ON_CLOSE, MARKET_ON_OPEN, Broker, DryRunBroker, Fill, LiveBroker,
    Mode, ModeSwitchRefused, PaperBroker,
)
from imperium.execution.engine import Decision, SymbolEngine
from imperium.execution.portfolio import PortfolioAllocator, Verdict
from imperium.execution import risk as risk_mod
from imperium.execution.risk import RiskLimits
from imperium.security.credentials import Credential, CredentialStore
from imperium.execution.costs import ADVERSE_SELECTION_FRACTION
from imperium.strategy import overnight as overnight_mod
from imperium.strategy import trend as trend_mod
from imperium.strategy.overnight import PooledDrift, SessionPhase
from imperium.strategy.trend import PooledTrend
from imperium.strategy.regime import CalibrationMissing, Regime, load_calibration
from imperium.venues import assets as assets_mod
from imperium.venues.assets import AssetClass, classify_symbol, spec_for
from imperium.strategy.signals import StrategyParams
from imperium.telemetry.streams import Level, TelemetryHub
from imperium.venues import registry
from imperium.venues.alpaca.client import AlpacaClient, MarketClock, VenueError
from imperium.venues.alpaca.feed import MarketFeed
from imperium.venues.alpaca.filters import format_decimal, to_decimal
from imperium.venues.registry import VenueSpec

log = logging.getLogger("imperium.session")


def _bar_ms(value: Any) -> int:
    """Alpaca timestamps are RFC-3339; bars are keyed by epoch milliseconds."""
    from imperium.venues.alpaca.feed import _ms

    return _ms(value)

#: A quote older than this is stale enough that acting on it is guessing.
STALE_AFTER_SECONDS = 20.0

#: Calendar days of daily bars pulled for the overnight decomposition. Roughly
#: a year of sessions. The effect being measured is 3-5bp against an overnight
#: standard deviation two orders of magnitude larger, so the estimate lives or
#: dies on observation count -- see PooledDrift for the measurement that showed
#: a single symbol's history cannot resolve it at all.
OVERNIGHT_HISTORY_DAYS = 400

#: How often the daily history is re-pulled. Daily bars change once a day, so
#: anything faster spends request budget to learn nothing.
OVERNIGHT_REFRESH_SECONDS = 6 * 3600

#: How often the account is re-read. It is one cheap request against a budget
#: of two hundred a minute, and it is the number the operator watches.
ACCOUNT_REFRESH_SECONDS = 15

#: How often a live book is checked against the venue's own positions. Often
#: enough that a rejected or partly filled order is caught within a minute,
#: rarely enough that it costs two requests a minute against a budget of two
#: hundred.
RECONCILE_SECONDS = 30

#: Symbols reasoned about per tick.
#:
#: Measured: one evaluation costs about 0.9ms, so a full pass over 150 symbols
#: is roughly 140ms -- a visible stall if done every second. Twenty-five costs
#: about 23ms a tick and covers the whole universe every six seconds, which is
#: far faster than any of the strategies here can act on anyway.
EVAL_SLICE = 25



#: How many symbols carry an engine and a bar ring -- the set actually reasoned
#: about bar by bar. The scan ranks the entire tradable listing; this bounds
#: what is kept from it. The bound is memory, measured rather than guessed: a
#: full ring costs about 286KB, so this is roughly 45MB of bar history, which a
#: laptop running the terminal alongside a browser can carry all day. Raising it
#: costs that much again per hundred and 1,500 symbols would be 430MB.
TRADED_UNIVERSE = 150

#: Symbols carrying an engine at once. The ranking covers the whole market; this
#: bounds what is resident, because a full bar ring measures about 286KB and an
#: engine per listed symbol would be gigabytes.
COHORT_SIZE = TRADED_UNIVERSE
#: Crypto pairs kept resident regardless of where the cursor is.
#:
#: Crypto is pinned rather than walked past for three reasons that all point
#: the same way on a small account: PDT counts equity round trips and exempts
#: crypto, so it is the only class this balance can day-trade; it trades
#: around the clock, so it is the only thing that can keep a live stream --
#: and a lit data lamp -- while the equity market is shut; and the listing is
#: tens of pairs, so residency is bounded by the market rather than by a guess.
CRYPTO_RESIDENT = 30

#: How often the cohort rotates. One rotation costs a daily-bar request and a
#: snapshot -- two against a budget of two hundred a minute -- so a twenty
#: second cycle walks eleven thousand symbols in about twenty-five minutes
#: while spending three requests a minute.
COHORT_SECONDS = 20

#: How many symbols' worth of history the pooled estimates keep. Several
#: cohorts, so the market-wide figures do not swing as the cursor walks from
#: mega-caps to micro-caps, and bounded so they do not become the memory the
#: rotation exists to avoid.
POOLED_SAMPLE_SYMBOLS = 600

#: How often the whole listing is re-ranked. A full sweep is one request per
#: two hundred symbols, so re-ranking the US equity market costs around fifty --
#: affordable on a quarter-hour cycle against a budget of two hundred a minute,
#: and pointless faster: turnover rankings do not move minute to minute.
FULL_SCAN_SECONDS = 15 * 60

#: A loop that has not begun an iteration in this long is not running. The loop
#: sleeps a second between ticks, so this is generous by two orders of
#: magnitude and will not fire on a slow scan.
LOOP_STALL_SECONDS = 90

#: How many symbols carry their full reasoning in one frame. The reasoning
#: panel renders 24; this leaves headroom for the sort to move between frames
#: without a row blinking out, and bounds the frame no matter how wide the
#: universe gets.
DETAIL_ROWS = 48

#: How many symbols reach the watchlist table at all. The engine evaluates the
#: whole universe; this bounds only what is *streamed*. A table cannot usefully
#: show a thousand rows -- the DOM alone would be tens of thousands of nodes
#: rewritten every second -- and the regime census and counters already
#: summarise every symbol, including the ones below this line. The panel says
#: how many it is not showing rather than pretending the universe is this size.
WATCHLIST_ROWS = 120


@dataclass
class Lamps:
    """The header status lamps. Each is a fact, not an aspiration."""

    link: str = "off"       # websocket to the browser
    venue: str = "off"      # REST reachable
    data: str = "off"       # market data flowing
    key: str = "off"        # a credential is loaded and working
    session: str = "off"    # the trading loop is running

    def as_dict(self) -> dict[str, str]:
        return {"link": self.link, "venue": self.venue, "data": self.data,
                "key": self.key, "session": self.session}


class TradingSession:
    """One venue, one book, one pool of money."""

    def __init__(self, venue_id: str = registry.DEFAULT_VENUE,
                 limits: RiskLimits | None = None,
                 params: StrategyParams | None = None) -> None:
        self.spec: VenueSpec = registry.get(venue_id)
        #: The limits as configured, before any account-size adjustment. Kept
        #: so re-scaling always derives from the original rather than
        #: compounding on its own previous output.
        self.base_limits = limits or RiskLimits()
        self.limits = self.base_limits
        #: How those limits were adjusted for this balance, and why.
        self.account_scale: risk_mod.AccountScale | None = None
        self.params = params or StrategyParams()
        self.telemetry = TelemetryHub()
        self.allocator = PortfolioAllocator(self.limits)
        self.engines: dict[str, SymbolEngine] = {}
        self.feed = MarketFeed(self.spec, self.telemetry, feed=self.spec.default_feed)
        self.feed.on_bar(self._on_bar)
        self.broker: Broker = DryRunBroker(self.spec)
        self.client: AlpacaClient | None = None
        self.credential: Credential | None = None
        self.lamps = Lamps()
        self.running = False
        self.started_at: float = 0.0
        self.day_start_equity: float = 0.0
        #: Seeded, then replaced by a live scan of what the venue actually
        #: lists and what is actually trading.
        self.universe: list[str] = list(self.spec.seed_universe)
        self.paper_endpoint: bool = True
        self.market_clock: MarketClock = MarketClock()
        self.universe_scanned_at: float = 0.0
        #: How wide the last sweep actually looked, and how much of it carried
        #: a price. Published so "scanning everything" is a number rather than
        #: a claim.
        self.universe_considered: int = 0
        self.universe_priced: int = 0
        #: The whole ranked market, as names. Cheap to hold; what costs memory
        #: is an engine and its bar ring, and only a cohort carries those.
        self.ranked_universe: list[str] = []
        self._cohort_cursor: int = 0
        self._cohort_rotated_at: float = 0.0
        #: Complete passes over the ranked market.
        self.cohort_passes: int = 0
        #: Pooled-estimate inputs, carried across cohort rotations and bounded.
        self._overnight_samples: dict[str, Any] = {}
        self._trend_samples: dict[str, Any] = {}
        self.scan_note: str = "not yet scanned"
        self.status_message = "idle"
        self.venue_error: str = ""
        self.calibration_error: str = ""
        #: Why the credential store is unusable, if it is. Shown in the UI: a
        #: terminal that starts but silently has no credentials is worse than
        #: one that says why.
        self.store_error: str = ""
        self._loop_task: asyncio.Task | None = None
        self._pending_bars: asyncio.Queue[tuple[str, Bar]] = asyncio.Queue(maxsize=4096)
        self._thresholds: dict | None = None
        #: The market-wide overnight drift, estimated across the whole universe
        #: at once. Held on the session rather than per engine because it is one
        #: measurement of one market, and every engine reads the same one.
        self.pooled_drift: PooledDrift | None = None
        #: The market-wide trend premium, and the positions carried on it.
        self.pooled_trend: PooledTrend | None = None
        self.trend_note: str = "not yet measured"
        #: symbol -> unix time the trend position was opened. Persisted, because
        #: a multi-day hold outlives the process by design.
        self.trend_holdings: dict[str, float] = {}
        self.session_phase: SessionPhase = SessionPhase.CLOSED
        self.overnight_note: str = "not yet measured"
        #: Symbols this strategy carried through a close, and the night it
        #: entered them. Tracked explicitly rather than inferred from "holds an
        #: equity while shut", so an intraday position that failed to flatten is
        #: never silently adopted and exited as though it were planned.
        #:
        #: Written to disk on every change. This program is a desktop
        #: application: closing the laptop after the close and reopening it
        #: before the bell is the *normal* way to use it, and an in-memory-only
        #: record would lose the one fact that says which positions still need
        #: an opening exit -- leaving a real position with nothing managing it.
        self.overnight_holdings: dict[str, float] = {}
        #: Symbols already reported as unmanaged in this pre-open window, so the
        #: warning is said once rather than once per tick.
        self._unmanaged_reported: set[str] = set()
        self._daily_loaded_at: float = 0.0
        self._equity_curve: list[tuple[float, float]] = []
        self._account_checked_at: float = 0.0
        #: The account as the venue reports it, refreshed on a timer. Held
        #: separately from the book because in dry run and paper the book is
        #: simulated and these are not: conflating them is how a terminal ends
        #: up showing an invented balance.
        self.account_equity: float = 0.0
        self.account_cash: float = 0.0
        self.account_buying_power: float = 0.0
        self.account_last_equity: float = 0.0
        self.account_currency: str = "USD"
        self.account_status: str = ""
        self.account_number: str = ""
        self.account_updated_at: float = 0.0
        self.account_error: str = ""
        self._seeded_from_account: bool = False
        #: The venue's trading date, as an ISO string. Empty until the first
        #: tick so that startup is not treated as a rollover.
        self._trading_day: str = ""
        #: When the trading loop last began an iteration. A loop that stops
        #: ticking is invisible from every other indicator.
        self._loop_beat: float = 0.0
        #: How many times the supervisor has had to restart the loop. Published
        #: rather than hidden: a terminal that quietly restarts itself all night
        #: is a terminal with a problem worth seeing.
        self.restarts: int = 0
        #: How many times the venue disagreed with this book. Published rather
        #: than hidden: a book that keeps needing correction is a book whose
        #: orders are not doing what it thinks.
        self.reconciliations: int = 0
        self._reconciled_at: float = 0.0
        #: Where the evaluation sweep has reached, and how many full passes it
        #: has completed. Published so "is it actually looking at anything" has
        #: a number rather than an impression.
        self._sweep_cursor: int = 0
        self.sweeps: int = 0
        #: Asked for while a session runs, handed back when it stops. A book
        #: holding an overnight position through a suspended laptop is a book
        #: whose opening exit never gets lodged.
        self.keep_awake = KeepAwake()

    # -- setup -----------------------------------------------------------

    def thresholds(self) -> dict | None:
        """Kept only to surface a calibration failure early.

        Each engine now selects the measured thresholds for its own asset
        class, because equities and crypto are fitted separately.
        """
        try:
            load_calibration()
            self.calibration_error = ""
        except CalibrationMissing as exc:
            self.calibration_error = str(exc)
            self.telemetry.event(Level.ERROR, "strategy",
                                 "the regime classifier is not calibrated",
                                 detail=str(exc))
        return None

    def engine(self, symbol: str) -> SymbolEngine:
        e = self.engines.get(symbol)
        if e is None:
            e = SymbolEngine(symbol, self.spec, self.limits, self.allocator,
                             self.telemetry, self.params)
            # Engines are created lazily, so one built after the last refresh
            # would otherwise start with no market estimate and no idea what
            # time it is -- and would then refuse the overnight trade with a
            # reason that describes the wiring rather than the market.
            e.pooled_drift = self.pooled_drift
            e.pooled_trend = self.pooled_trend
            e.session_phase = self.session_phase
            self.engines[symbol] = e
            self.allocator.observe(symbol)
        return e

    async def attach_credential(self, store: CredentialStore,
                                name: str | None) -> None:
        """Attach a stored key, or run with none.

        Running with no key is a supported state, not an error: live prices and
        the whole scanner work without one, and requiring a credential before
        anything renders is how a first-run experience becomes unusable.
        """
        await self.detach_client()
        if name is None:
            self.credential = None
            self.lamps.key = "off"
            # No key: an unauthenticated client still serves the clock and the
            # public data endpoints, which is what the scanner renders from.
            self.client = AlpacaClient("", "", paper=self.paper_endpoint,
                                       data_url=self.spec.data_url,
                                       feed=self.spec.default_feed)
            return
        cred = store.require(name)
        self.credential = cred
        self.client = AlpacaClient(cred.api_key, cred.secret,
                                   paper=self.paper_endpoint,
                                   data_url=self.spec.data_url,
                                   feed=self.spec.default_feed)
        self.feed.set_credentials(cred.api_key, cred.secret)
        try:
            # The key check already costs this request, so the balance arrives
            # with it rather than up to a refresh interval later. An operator
            # who has just attached a key expects to see their money now.
            self.absorb_account(await self.client.account())
            self._account_checked_at = time.time()
            self.market_clock = await self.client.get_clock()
        except VenueError as exc:
            self.lamps.key = "bad"
            self.venue_error = exc.operator_text()
            self.telemetry.event(Level.ERROR, "venue",
                                 f"the key {name!r} was rejected: {exc.message}",
                                 detail=exc.remedy)
            return
        self.lamps.key = "ok"
        self.lamps.venue = "ok"
        self.venue_error = ""
        self.telemetry.event(
            Level.GOOD, "venue",
            f"key {name!r} accepted by {self.spec.display_name} "
            f"({self.client.environment}) — {self.market_clock.describe()}")
        await self.scan_universe()

    async def refresh_clock(self) -> None:
        """Ask the venue whether the market is open.

        Believing the venue rather than computing a calendar locally is the only
        way to get early closes, holidays and unscheduled halts right, and each
        of those is a day a naive calendar trades into a closed market.
        """
        if self.client is None or not self.client.authenticated:
            return
        try:
            self.market_clock = await self.client.get_clock()
        except VenueError as exc:
            self.telemetry.event(Level.WARN, "venue",
                                 "could not read the market clock",
                                 detail=exc.message)
            return
        self.allocator.market_open = self.market_clock.is_open or self._crypto_only()
        self.allocator.market_note = (
            "" if self.allocator.market_open
            else f"{self.market_clock.describe()}; equities take no new exposure "
                 f"while closed")

    def _crypto_only(self) -> bool:
        """True when every admitted symbol trades around the clock.

        A closed equity market must not stop a crypto book, and vice versa.
        """
        admitted = self.allocator.admitted_symbols or self.universe
        return bool(admitted) and all(
            classify_symbol(s) is AssetClass.CRYPTO for s in admitted)

    async def scan_universe(self, limit: int = TRADED_UNIVERSE) -> None:
        """Discover what this account can actually trade, ranked by turnover.

        The seed list is a starting point, not the universe. This asks the venue
        what it lists, drops anything not tradable right now, and ranks what is
        left by dollar volume -- because a scanner whose job is to reject most of
        what it sees needs a real field to choose from, and because a symbol
        that is halted or delisted should never reach the sizer.
        """
        if self.client is None or not self.client.authenticated:
            self.scan_note = "no key attached, using the seed list"
            return
        try:
            assets = await self.client.assets()
        except VenueError as exc:
            self.scan_note = f"scan failed, using the seed list: {exc.message}"
            self.telemetry.event(Level.WARN, "universe", self.scan_note)
            return
        if not assets:
            self.scan_note = "the venue listed no assets; using the seed list"
            return

        candidates = [a for a in assets.values()
                      if a.tradable and spec_for(a.asset_class).tradeable]
        # Every tradable listing, not a shortlist of the first four hundred.
        # Snapshots are batched by the client, so the whole listing costs one
        # request per two hundred symbols -- around fifty for the US equity
        # market -- and that is spent on a slow cycle rather than every minute.
        seeded = [s for s in self.spec.seed_universe if s in assets]
        shortlist = seeded + [a.symbol for a in candidates
                              if a.symbol not in set(seeded)]

        snaps = await self.client.snapshots(shortlist)
        ranked: list[tuple[float, str]] = []
        for symbol, snap in snaps.items():
            daily = snap.get("dailyBar") or snap.get("prevDailyBar") or {}
            try:
                close = float(daily.get("c", 0) or 0)
                volume = float(daily.get("v", 0) or 0)
            except (TypeError, ValueError):
                continue
            turnover = close * volume
            if turnover <= 0:
                continue
            q = self.feed.quote(symbol)
            q.quote_volume = turnover
            if not q.last:
                q.last = close
            open_px = float(daily.get("o", 0) or 0)
            if open_px > 0:
                q.change_pct = (close - open_px) / open_px * 100.0
            if not q.updated_at:
                q.updated_at = time.time()
            ranked.append((turnover, symbol))

        if not ranked:
            self.scan_note = ("the venue returned no traded volume; using the "
                              "seed list")
            return
        ranked.sort(reverse=True)
        # The whole ranked market is kept, not just the head of it. Symbols are
        # cheap to hold as a list of names; what costs memory is an engine and
        # its bar ring, and those belong to the cohort currently being
        # evaluated rather than to the ranking.
        self.ranked_universe = [symbol for _, symbol in ranked]
        self.universe_scanned_at = time.time()
        self.universe_considered = len(shortlist)
        self.universe_priced = len(ranked)
        self._cohort_cursor = 0
        self.cohort_passes = 0
        await self.rotate_cohort(force=True)

        equities = sum(1 for s in self.ranked_universe
                       if classify_symbol(s) is AssetClass.US_EQUITY)
        crypto = len(self.ranked_universe) - equities
        self.scan_note = (f"{len(ranked):,} priced from {len(shortlist):,} "
                          f"listed ({equities:,} equity, {crypto} crypto), "
                          f"ranked by traded value")
        self.telemetry.event(Level.INFO, "universe", f"scanned: {self.scan_note}")

    async def rotate_cohort(self, *, force: bool = False) -> int:
        """Retire the symbols that did not make the cut, and bring in the next.

        The ranking covers the whole market; this is what walks it. Only a
        cohort carries engines at once -- a full bar ring costs around 286KB,
        so an engine per listed symbol would be gigabytes -- and the cursor
        advances through the ranking so that every symbol is eventually
        evaluated rather than only the head of it.

        What survives a rotation is what has earned it: anything holding a
        position, anything the strategies are carrying overnight or across
        days, and anything whose last decision said it was worth trading. The
        rest is retired and its memory released. That is the difference between
        a watchlist and a leaderboard -- a name that was scanned and refused
        has been answered, and holding it forever would crowd out the names
        that have not been looked at yet.
        """
        if not self.ranked_universe:
            return 0
        if not force and time.time() - self._cohort_rotated_at < COHORT_SECONDS:
            return 0

        # Everything that has earned its place, whatever it ranks.
        keep = {s for s, pos in self.broker.positions.items() if not pos.is_flat}
        keep |= set(self.overnight_holdings) | set(self.trend_holdings)
        keep |= {s for s, e in self.engines.items()
                 if e.decision.verdict is Verdict.TRADING}
        keep &= set(self.ranked_universe) | keep      # held names stay regardless
        # Pinned, not walked past -- see CRYPTO_RESIDENT. Without this the
        # cursor retires the crypto pairs on its next pass and the only feed
        # that reports anything outside market hours goes with them.
        keep |= set(self.crypto_universe()[:CRYPTO_RESIDENT])

        size = max(1, COHORT_SIZE - len(keep))
        if self._cohort_cursor >= len(self.ranked_universe):
            self._cohort_cursor = 0
            self.cohort_passes += 1
        fresh = self.ranked_universe[self._cohort_cursor:self._cohort_cursor + size]
        self._cohort_cursor += len(fresh)
        if self._cohort_cursor >= len(self.ranked_universe):
            self._cohort_cursor = 0
            self.cohort_passes += 1

        retired = [s for s in self.universe if s not in keep and s not in fresh]
        self.universe = sorted(keep) + [s for s in fresh if s not in keep]
        self._cohort_rotated_at = time.time()
        self._sweep_cursor = 0
        self._prune(self.universe)

        # The socket has to follow the cohort. Without this the stream stays on
        # whatever the universe held when the session started -- retired names
        # nothing is looking at -- while the symbols now being reasoned about
        # have no live price, and the sweep still marks them streamed because
        # it reads that flag from the universe rather than the subscription.
        await self.feed.retarget(self._stream_priority())

        # A cohort with no daily bars cannot be reasoned about by the multi-day
        # strategy, which is the only one that works on a symbol with no live
        # stream -- so the bars come with the rotation rather than up to six
        # hours later.
        if fresh:
            await self.refresh_daily_history(force=True)

        if retired:
            self.telemetry.event(
                Level.INFO, "universe",
                f"{len(retired)} symbols scanned and retired, {len(fresh)} "
                f"brought in — {self.cohort_progress:.0%} through the market",
                detail="Retired means answered, not rejected forever: it comes "
                       "round again on the next pass. Anything holding a "
                       "position or worth trading stays.")
        return len(fresh)

    @property
    def cohort_progress(self) -> float:
        """How far through the ranked market this pass has reached."""
        if not self.ranked_universe:
            return 0.0
        return min(1.0, self._cohort_cursor / len(self.ranked_universe))

    def _prune(self, keep: list[str]) -> None:
        """Drop state for symbols that are no longer traded.

        The scan ranks the whole listing, so over a long run this session sees
        far more symbols than it trades, and each one that keeps an engine keeps
        a bar ring with it -- measured at around 286KB once full. A terminal
        meant to run for days cannot accumulate those for every symbol that was
        briefly interesting on a Tuesday.

        A symbol holding a position is never pruned, whatever it ranks: its
        engine is the thing managing that position.
        """
        held = {sym for sym, pos in self.broker.positions.items() if not pos.is_flat}
        wanted = set(keep) | held | set(self.overnight_holdings)
        for symbol in [s for s in self.engines if s not in wanted]:
            self.engines.pop(symbol, None)
        # Quotes are small but there is one per symbol ever seen, and the scan
        # now sees the whole market.
        for symbol in [s for s in self.feed.quotes if s not in wanted]:
            self.feed.quotes.pop(symbol, None)
        self.allocator.forget(set(self.allocator.states) - wanted)

    async def detach_client(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    # -- market data -----------------------------------------------------

    def _on_bar(self, symbol: str, bar: Bar) -> None:
        """Called from the feed. Cheap, and never raises into the socket loop."""
        try:
            self._pending_bars.put_nowait((symbol, bar))
        except asyncio.QueueFull:
            # Dropping a bar is better than stalling the socket. It is counted
            # so the health panel can show it rather than hiding it.
            self.telemetry.event(Level.WARN, "feed",
                                 "the bar queue is full; a bar was dropped")

    async def seed_history(self) -> None:
        """Seed each engine with recent bars, batched by asset class."""
        if not self.client:
            return
        try:
            batches = await self.client.bars(self.universe, timeframe="1Min",
                                             limit=1000)
        except VenueError as exc:
            self.telemetry.event(Level.WARN, "data",
                                 f"could not load history: {exc.message}",
                                 detail=exc.remedy)
            return
        for symbol in self.universe:
            rows = batches.get(symbol) or []
            if not rows:
                self.allocator.set_scan(
                    symbol, score=0.0, turnover=0.0, tradeable=False,
                    reason="no recent bars from the venue for this symbol")
                continue
            engine = self.engine(symbol)
            for index, row in enumerate(rows):
                try:
                    bar = Bar(
                        open_time=_bar_ms(row.get("t")),
                        open=float(row["o"]), high=float(row["h"]),
                        low=float(row["l"]), close=float(row["c"]),
                        volume=float(row.get("v", 0.0)),
                        # The final row is the bar still forming; treating it as
                        # closed makes every strategy act on a partial bar.
                        closed=index < len(rows) - 1,
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                engine.series.add(bar)

    async def refresh_daily_history(self, *, force: bool = False) -> None:
        """Pull daily bars and re-estimate the market-wide overnight drift.

        Daily bars, not the minute ring: the ring holds a few sessions, and the
        first attempt at this measured 14 nights where it should have seen 80.
        One daily row carries exactly the open and close the decomposition
        needs, so a year of nights costs a few hundred rows per symbol.

        Equities only. Crypto never closes, so it has no overnight session to
        decompose, and options are not traded by this program at all -- the
        drift is a few basis points and the cheapest option time decay that
        could carry it costs an order of magnitude more (see
        ``scripts/overnight_option_arithmetic.py``).
        """
        if self.client is None:
            return
        # Both classes: the overnight decomposition is equity-only, but the
        # trend strategy runs on daily bars for crypto too -- and crypto is
        # where a small account can trade continuously, because it sits outside
        # the pattern-day-trader rule entirely.
        wanted = [s for s in self.universe
                  if classify_symbol(s) is not AssetClass.US_OPTION]
        equities = [s for s in wanted
                    if classify_symbol(s) is AssetClass.US_EQUITY]
        # Daily bars change once a day, so the interval exists to stop this
        # spending request budget to learn nothing. It must not strand a symbol
        # the scanner admitted since the last pull: that symbol would carry no
        # history for up to six hours and refuse every night in that window
        # with "warming up", which describes this method rather than the market.
        missing = any(not self.engines.get(s) or not self.engines[s].daily_bars
                      for s in wanted)
        if (not force and not missing
                and time.time() - self._daily_loaded_at < OVERNIGHT_REFRESH_SECONDS):
            return
        if not wanted:
            self.overnight_note = ("nothing in the universe has a daily series "
                                   "to measure")
            self.pooled_drift = None
            self.pooled_trend = None
            return

        start = (dt.datetime.now(tz=dt.timezone.utc)
                 - dt.timedelta(days=OVERNIGHT_HISTORY_DAYS))
        try:
            batches = await self.client.bars(wanted, timeframe="1Day",
                                             limit=OVERNIGHT_HISTORY_DAYS,
                                             start=start)
        except VenueError as exc:
            self.overnight_note = f"daily history unavailable: {exc.message}"
            self.telemetry.event(Level.WARN, "overnight", self.overnight_note,
                                 detail=exc.remedy)
            return
        self._daily_loaded_at = time.time()

        splits: dict[str, Any] = {}
        scored: dict[str, tuple[Any, Any]] = {}
        for symbol in wanted:
            rows = batches.get(symbol) or []
            bars: list[Bar] = []
            for row in rows:
                try:
                    bars.append(Bar(
                        open_time=_bar_ms(row.get("t")),
                        open=float(row["o"]), high=float(row["h"]),
                        low=float(row["l"]), close=float(row["c"]),
                        volume=float(row.get("v", 0.0)), closed=True))
                except (KeyError, TypeError, ValueError):
                    continue
            engine = self.engine(symbol)
            engine.daily_bars = bars
            if classify_symbol(symbol) is AssetClass.US_EQUITY:
                split = overnight_mod.split_daily(bars)
                if split.nights:
                    splits[symbol] = split
            pair = self._trend_observations(symbol, bars)
            if pair is not None:
                scored[symbol] = pair

        # Accumulated across cohorts, not recomputed from the current one.
        # The ranking is ordered by traded value, so a cohort is not a random
        # sample of the market -- the first is mega-caps and the seventieth is
        # micro-caps, and an estimate rebuilt from each in turn would swing
        # between them and call the swing a change in the market. Bounded, so
        # this does not become the memory the cohort exists to avoid.
        self._overnight_samples.update(splits)
        self._trend_samples.update(scored)
        self._forget_oldest_samples()
        splits = dict(self._overnight_samples)
        scored = dict(self._trend_samples)

        pooled = overnight_mod.pool(splits) if splits else None
        self.pooled_drift = pooled
        pooled_trend = trend_mod.pool(scored) if scored else None
        self.pooled_trend = pooled_trend
        self.trend_note = (pooled_trend.describe() if pooled_trend
                           else "no trend premium measured yet")
        # Every engine reads the same market estimate. Assigned rather than
        # looked up so an engine created later in the session cannot quietly
        # run against no prior and refuse everything for the wrong reason.
        for engine in self.engines.values():
            engine.pooled_drift = pooled
            engine.pooled_trend = pooled_trend

        if pooled is None:
            self.overnight_note = "no daily history returned for any equity"
            self.telemetry.event(
                Level.WARN, "overnight",
                "No daily history came back, so the overnight drift cannot be "
                "measured and nothing will be held overnight.",
                detail="This usually means the data plan did not return daily "
                       "bars for the equities in the universe.")
        else:
            self.overnight_note = pooled.describe()
            # Said in words, not in statistics. An operator reading this at
            # seven in the morning needs to know what the program will do, and
            # a level that does not make an ordinary measurement look like a
            # fault -- "not enough history yet" is information, not a warning.
            self.telemetry.event(
                Level.GOOD if pooled.credible else Level.INFO,
                "overnight", pooled.explain(), detail=pooled.describe())

    def _save_overnight_state(self) -> None:
        """Persist which symbols are being carried overnight.

        Best effort by design: a state file that cannot be written must not
        stop the session, and a session that cannot read one starts with an
        empty map and reports the positions it cannot account for rather than
        guessing at them.
        """
        try:
            config.ensure_home()
            path = config.state_path()
            path.write_text(json.dumps(
                {"overnight_holdings": self.overnight_holdings,
                 "trend_holdings": self.trend_holdings,
                 "saved_at": time.time()}, indent=2), encoding="utf-8")
            try:
                path.chmod(0o600)
            except (OSError, NotImplementedError):
                # Windows ignores POSIX modes. The file holds no secrets --
                # only symbols and weights -- so this is tidiness, not a gate.
                pass
        except OSError as exc:
            self.telemetry.event(
                Level.WARN, "overnight",
                "could not save which positions are held overnight; a restart "
                "before the open would not know to exit them",
                detail=str(exc))

    def _load_overnight_state(self) -> None:
        """Recover the overnight book across a restart."""
        try:
            raw = config.state_path().read_text(encoding="utf-8")
        except (OSError, ValueError):
            return
        try:
            payload = json.loads(raw)
            holdings = payload.get("overnight_holdings")
            if not isinstance(holdings, dict):
                return
            recovered = {str(k): float(v) for k, v in holdings.items()}
        except (AttributeError, TypeError, ValueError) as exc:
            # A corrupt state file is not a reason to refuse to start. It is a
            # reason to say the overnight book is unknown.
            self.telemetry.event(
                Level.WARN, "overnight",
                "the saved overnight state could not be read; any position "
                "held overnight will be reported as unmanaged rather than "
                "exited automatically", detail=str(exc))
            return
        try:
            carried = payload.get("trend_holdings")
            if isinstance(carried, dict):
                self.trend_holdings = {str(k): float(v) for k, v in carried.items()}
        except (AttributeError, TypeError, ValueError):
            # A trend position whose start time is unreadable is treated as
            # opened now: it will simply be held longer than it needs to be,
            # which costs nothing and is the safe direction. Closing it early
            # would throw away a round trip already paid for.
            self.trend_holdings = {
                symbol: time.time()
                for symbol, pos in self.broker.positions.items() if not pos.is_flat}
        if recovered:
            self.overnight_holdings = recovered
            self.telemetry.event(
                Level.INFO, "overnight",
                f"recovered {len(recovered)} overnight "
                f"{'hold' if len(recovered) == 1 else 'holds'} across a "
                f"restart: {', '.join(sorted(recovered))}")

    def _sync_trend_holdings(self) -> None:
        """Tell each engine how long its trend position has been carried.

        The holding period is not bookkeeping here, it is the strategy: the
        decision to keep a position is made against how much of its round-trip
        cost the elapsed time has already paid for. An engine that does not know
        how long it has held would re-derive an entry every day and pay the
        round trip each time.
        """
        now = time.time()
        for symbol, engine in self.engines.items():
            position = self.broker.positions.get(symbol)
            held = position is not None and not position.is_flat
            if held and symbol not in self.trend_holdings:
                # Only claim a position this strategy actually opened.
                engine.trend_held = engine.decision.strategy == "trend"
                if engine.trend_held:
                    self.trend_holdings[symbol] = now
                    self._save_overnight_state()
            elif not held and symbol in self.trend_holdings:
                self.trend_holdings.pop(symbol, None)
                self._save_overnight_state()
                engine.trend_held = False
            else:
                engine.trend_held = held and symbol in self.trend_holdings
            opened = self.trend_holdings.get(symbol)
            engine.trend_days_held = ((now - opened) / 86_400.0) if opened else 0.0

    def _report_unmanaged_equity(self) -> None:
        """Name any equity held while shut that this strategy did not enter.

        It is not exited automatically, because "holds an equity while the
        market is closed" is not the same fact as "was entered on last night's
        close" -- it is equally the signature of an intraday position that
        failed to flatten. Both are positions with nothing managing them, and
        both are the operator's call. Silence would be the one wrong answer.
        """
        for symbol, position in self.broker.positions.items():
            if position.is_flat or symbol in self.overnight_holdings:
                continue
            if classify_symbol(symbol) is not AssetClass.US_EQUITY:
                continue
            if symbol in self._unmanaged_reported:
                continue
            self._unmanaged_reported.add(symbol)
            self.telemetry.event(
                Level.WARN, "overnight",
                f"{symbol} is held through the close but was not entered by the "
                f"overnight strategy, so no opening exit will be lodged for it",
                detail="flatten it by hand, or halt and let the retirement "
                       "sweep close it")

    def crypto_universe(self) -> list[str]:
        """The ranked crypto pairs, best first."""
        return [s for s in self.ranked_universe
                if classify_symbol(s) is AssetClass.CRYPTO]

    def _stream_priority(self) -> list[str]:
        """The universe, ordered by who most needs a live stream.

        The plan caps concurrent subscriptions well below the number of symbols
        this scans, so the order decides who gets one. A position being carried
        goes first without exception: it is the one symbol where a stale price
        means a stop that does not fire and an exit sized on a number from
        several minutes ago.

        Crypto comes next, because the slot is worth most where it can be
        used. The intraday strategy needs a warmed minute ring, and on this
        balance PDT forbids the equity round trip it would open anyway --
        while crypto is exempt from PDT and trades around the clock. Handing
        the cap to whichever equities the cursor happened to stop on gives
        thirty symbols a stream for twenty seconds each, which is not long
        enough for any of them to warm up and, outside market hours, is not a
        stream at all.

        Everything below the cap is still scanned, still priced by the
        snapshot sweep every minute, and still tradeable by the daily-bar
        strategies. What it loses is the intraday path, which cannot work on a
        minute-old price anyway.
        """
        held = [s for s, pos in self.broker.positions.items() if not pos.is_flat]
        ordered = [s for s in held if s in self.universe]
        seen = set(ordered)
        crypto = [s for s in self.universe
                  if classify_symbol(s) is AssetClass.CRYPTO and s not in seen]
        ordered += crypto
        seen |= set(crypto)
        ordered += [s for s in self.universe if s not in seen]
        return ordered

    def _trend_observations(self, symbol: str, bars: list[Bar]):
        """One symbol's (trend score, next-day return) pairs for the pooled fit.

        Every score is computed from bars strictly *before* the return it is
        paired with. That is the whole discipline of this function: a score
        that peeked at the day it is predicting would produce a spectacular
        premium and an unplaceable trade.
        """
        closes = trend_mod.daily_closes(bars)
        # Classified from the symbol: a Bar carries no identity of its own, and
        # defaulting to the equity lookbacks would have measured crypto -- a
        # market whose momentum lives at one to four weeks -- on a
        # one-to-six-month window.
        spec = trend_mod.spec_for(classify_symbol(symbol))
        longest = max(spec.lookbacks)
        if closes.size < longest + trend_mod.MIN_DAYS:
            return None

        scores: list[float] = []
        forward: list[float] = []
        for i in range(longest + 2, closes.size - 1):
            score, _ = trend_mod.blended_score(closes[: i + 1], spec)
            if not math.isfinite(score):
                continue
            # closes[i] is the last bar the score saw; closes[i + 1] is the day
            # it is being asked to predict.
            scores.append(score)
            forward.append(math.log(closes[i + 1] / closes[i]))
        if len(scores) < 10:
            return None
        return np.asarray(scores), np.asarray(forward)

    def _forget_oldest_samples(self) -> None:
        """Bound the pooled samples to a few cohorts' worth.

        Insertion-ordered, so this drops what was measured longest ago. The
        point is an estimate that spans several cohorts rather than one, not an
        estimate that remembers the whole market -- that would be exactly the
        memory the rotation exists to avoid.
        """
        for store in (self._overnight_samples, self._trend_samples):
            while len(store) > POOLED_SAMPLE_SYMBOLS:
                store.pop(next(iter(store)))

    def _update_session_phase(self) -> SessionPhase:
        """Where the clock is, relative to the two auction windows.

        Read on every tick rather than latched, because a phase held across a
        halt or an early close is a phase that lodges an auction order the venue
        has already stopped accepting.
        """
        now = dt.datetime.now(tz=dt.timezone.utc)
        self.session_phase = overnight_mod.phase_from_clock(
            now, self.market_clock.next_close, self.market_clock.is_open,
            next_open=self.market_clock.next_open)
        for engine in self.engines.values():
            engine.session_phase = self.session_phase
        if self.session_phase is not SessionPhase.PREOPEN:
            self._unmanaged_reported.clear()
        return self.session_phase

    async def _flatten_unwanted_before_the_close(self) -> None:
        """Close any equity the overnight strategy has just declined to hold.

        The engine refuses a target by returning a verdict that is not TRADING,
        and :meth:`_act_on` returns early on those -- so a refusal never reduces
        a position. That is right during the session: "no new exposure" is not
        "sell what you have". It is wrong at the close, because the position
        does not simply sit there until tomorrow. It is carried through the
        night, and the intraday strategy that opened it sized it against an
        intraday distribution and put an ATR stop behind it, neither of which
        survives a gap.

        So the closing window is where that decision gets made explicitly. The
        overnight strategy has just evaluated this exact question -- is this
        symbol worth holding through the night -- and said no. Acting on the no
        is the whole point of asking.

        Only symbols the overnight strategy actually evaluated in this window
        are considered: a stale intraday verdict is not an answer to the
        question being asked, and skipping is the safe direction.
        """
        for symbol, position in list(self.broker.positions.items()):
            if position.is_flat or symbol in self.overnight_holdings:
                continue
            if classify_symbol(symbol) is not AssetClass.US_EQUITY:
                continue
            engine = self.engines.get(symbol)
            if engine is None or engine.decision.strategy != "overnight":
                continue
            if engine.decision.verdict is Verdict.TRADING:
                continue
            price = self.feed.quote(symbol).last
            if price <= 0:
                self.telemetry.event(
                    Level.ERROR, "overnight",
                    f"{symbol} should not be carried overnight but has no "
                    f"price, so no closing order could be sized. It will be "
                    f"held through the night.")
                continue
            try:
                fill = await self.broker.apply_target(
                    symbol, 0.0, price, self.equity(), order=MARKET_ON_CLOSE)
            except (VenueError, ModeSwitchRefused) as exc:
                self.telemetry.event(
                    Level.ERROR, "overnight",
                    f"could not lodge the closing exit for {symbol}: {exc}")
                continue
            self.allocator.observe(symbol).current_weight = 0.0
            if fill:
                self.telemetry.pulse(symbol, "order",
                                     "closed rather than carried overnight", 1.0)
            self.telemetry.event(
                Level.INFO, "overnight",
                f"{symbol}: closing on the auction rather than carrying it "
                f"overnight — {engine.decision.reason}")

    async def _exit_overnight_holdings(self) -> None:
        """Sell every overnight hold on the opening auction.

        The strategy is paid the close-to-open move and nothing after it, so the
        exit is a market-on-open order lodged before the bell. Waiting for the
        open and then sending a market order gives back the part of the drift
        that has already printed, which at 3-5bp is most of it.
        """
        for symbol in list(self.overnight_holdings):
            position = self.broker.positions.get(symbol)
            if position is None or position.is_flat:
                self.overnight_holdings.pop(symbol, None)
                self._save_overnight_state()
                continue
            price = self.feed.quote(symbol).last or float(position.avg_price)
            if price <= 0:
                self.telemetry.event(
                    Level.ERROR, "overnight",
                    f"{symbol} is held overnight but has no price, so no exit "
                    f"order could be sized. The position is still open.")
                continue
            try:
                fill = await self.broker.apply_target(
                    symbol, 0.0, price, self.equity(), order=MARKET_ON_OPEN)
            except (VenueError, ModeSwitchRefused) as exc:
                self.telemetry.event(
                    Level.ERROR, "overnight",
                    f"could not lodge the opening exit for {symbol}: {exc}")
                continue
            self.overnight_holdings.pop(symbol, None)
            self._save_overnight_state()
            self.allocator.observe(symbol).current_weight = 0.0
            if fill:
                self.telemetry.pulse(symbol, "order",
                                     "overnight exit on the opening auction", 1.0)
                self.telemetry.event(
                    Level.INFO, "overnight",
                    f"{symbol}: market-on-open exit lodged, closing the "
                    f"overnight hold")

    async def refresh_universe(self) -> None:
        """Re-price and re-admit. Failures demote a symbol, never crash."""
        if not self.client:
            return
        await self.refresh_clock()
        try:
            snaps = await self.client.snapshots(self.universe)
            assets = await self.client.assets()
        except VenueError as exc:
            self.lamps.venue = "bad"
            self.venue_error = exc.operator_text()
            self.telemetry.event(Level.WARN, "venue",
                                 f"could not refresh the universe: {exc.message}",
                                 detail=exc.remedy)
            return
        self.lamps.venue = "ok"
        self.venue_error = ""

        for symbol in self.universe:
            snap = snaps.get(symbol) or {}
            daily = snap.get("dailyBar") or snap.get("prevDailyBar") or {}
            quote = snap.get("latestQuote") or {}
            trade = snap.get("latestTrade") or {}
            q = self.feed.quote(symbol)
            try:
                close = float(daily.get("c", 0) or 0)
                open_px = float(daily.get("o", 0) or 0)
                volume = float(daily.get("v", 0) or 0)
                last = float(trade.get("p", 0) or 0) or close
                bid = float(quote.get("bp", 0) or 0)
                ask = float(quote.get("ap", 0) or 0)
            except (TypeError, ValueError):
                continue
            if last > 0:
                q.last = last
            if bid > 0:
                q.bid = bid
            if ask > 0:
                q.ask = ask
            turnover = close * volume
            q.quote_volume = turnover
            if open_px > 0 and close > 0:
                q.change_pct = (close - open_px) / open_px * 100.0
            if q.last and not q.updated_at:
                q.updated_at = time.time()

            engine = self.engine(symbol)
            engine.set_book(q.bid or None, q.ask or None)
            # Shortability and borrow are per-symbol facts that change, so they
            # are refreshed rather than assumed once.
            asset = assets.get(symbol)
            if asset is not None:
                engine.can_short = asset.can_short
                engine.tradable = asset.tradable and spec_for(
                    asset.asset_class).tradeable

            reason = ""
            tradeable = True
            if asset is not None and not asset.tradable:
                tradeable, reason = False, (
                    f"the venue lists {symbol} as not tradable "
                    f"(status {asset.status})")
            elif turnover <= 0:
                tradeable, reason = False, "no traded value reported for this symbol"
            self.allocator.set_scan(
                symbol, score=abs(engine.decision.conviction), turnover=turnover,
                tradeable=tradeable, reason=reason)

        admitted, retired = self.allocator.rebalance_admissions()
        for symbol in admitted:
            self.telemetry.event(Level.INFO, "universe", f"{symbol} admitted")
        for symbol in retired:
            # Retiring means flattening. A stopped engine still holding a
            # position is a position with nothing managing its stop.
            self.telemetry.event(Level.WARN, "universe",
                                 f"{symbol} retired — flattening it")
            await self._flatten_symbol(symbol)

    async def _flatten_symbol(self, symbol: str) -> None:
        price = self.feed.quote(symbol).last
        if not price:
            self.telemetry.event(
                Level.ERROR, "risk",
                f"{symbol} was retired but has no price, so it could not be "
                f"flattened. The position is still open.")
            return
        try:
            fill = await self.broker.apply_target(symbol, 0.0, price, self.equity())
        except (VenueError, ModeSwitchRefused) as exc:
            self.telemetry.event(Level.ERROR, "risk",
                                 f"could not flatten {symbol}: {exc}")
            return
        if fill:
            self.telemetry.pulse(symbol, "order", f"flattened on retirement", 1.0)
            self.telemetry.event(Level.WARN, "order",
                                 f"flattened {symbol} on retirement")

    # -- the trading loop -------------------------------------------------

    def prices(self) -> dict[str, float]:
        return {s: q.last for s, q in self.feed.quotes.items() if q.last > 0}

    def equity(self) -> float:
        return float(self.broker.equity(self.prices()))

    async def _drain_bars(self) -> None:
        """Evaluate every bar that actually closed. One pulse per evaluation."""
        processed = 0
        while not self._pending_bars.empty() and processed < 512:
            symbol, bar = self._pending_bars.get_nowait()
            processed += 1
            engine = self.engine(symbol)
            newly_closed = engine.series.add(bar)
            q = self.feed.quote(symbol)
            engine.set_book(q.bid or None, q.ask or None)
            if not newly_closed:
                continue
            decision = engine.evaluate()
            await self._act_on(decision)

    async def _sweep(self) -> int:
        """Reason about a slice of the universe, then move the cursor on.

        The scanner ranks far more symbols than the data plan will stream, and
        until this existed a symbol without a stream was never evaluated at
        all: the only path to a decision was a bar arriving on the websocket.
        With 150 symbols ranked and 30 streamed, 120 of them sat permanently
        "unscanned" -- and with the market shut, all 150 did, which is why the
        cluster showed no orbs and the reasoning panel filled with symbols
        nothing had ever looked at.

        Divided across ticks rather than done in one: a full pass costs about
        140ms at 150 symbols, which would be a visible stall once a second. A
        slice of :data:`EVAL_SLICE` costs about 23ms and covers the whole
        universe every six seconds, so the work is spread instead of spiked.

        Streamed symbols still evaluate the moment their bar closes, in
        _drain_bars. This is the floor under that, not a replacement: it
        guarantees every symbol is reasoned about on a bounded cycle whatever
        the feed is doing.
        """
        universe = list(self.universe)
        if not universe:
            return 0
        if self._sweep_cursor >= len(universe):
            self._sweep_cursor = 0
        slice_ = universe[self._sweep_cursor:self._sweep_cursor + EVAL_SLICE]
        self._sweep_cursor += len(slice_)
        if self._sweep_cursor >= len(universe):
            self._sweep_cursor = 0
            self.sweeps += 1

        streamed = set(self._stream_priority()[:self.feed.symbol_limit])
        for symbol in slice_:
            engine = self.engine(symbol)
            engine.streamed = symbol in streamed
            quote = self.feed.quote(symbol)
            engine.last_price = quote.last
            engine.set_book(quote.bid or None, quote.ask or None)
            try:
                decision = engine.scan()
            except Exception as exc:
                # One symbol's arithmetic must never stop the sweep; the rest
                # of the universe is still waiting to be looked at.
                log.exception("evaluating %s raised", symbol)
                self.telemetry.event(Level.WARN, "strategy",
                                     f"could not evaluate {symbol}",
                                     detail=f"{type(exc).__name__}: {exc}")
                continue
            await self._act_on(decision)
        return len(slice_)

    async def _act_on(self, decision: Decision) -> None:
        if decision.verdict is not Verdict.TRADING:
            return
        if decision.hold:
            # Leave it exactly as it is. Re-targeting a carried position to the
            # weight it already holds looks like a no-op and is not: equity
            # moves with every fill and every price tick, so the delta is never
            # quite zero, and each evaluation pays a spread to trade a fraction
            # of a share. Over a session that is a low-turnover strategy
            # quietly becoming a high-turnover one.
            return
        price = self.feed.quote(decision.symbol).last
        if price <= 0:
            return
        quote_age = self.feed.quote(decision.symbol).age
        if quote_age > STALE_AFTER_SECONDS:
            self.telemetry.pulse(decision.symbol, "refused",
                                 f"quote is {quote_age:.0f}s stale", 0.4)
            return
        try:
            fill = await self.broker.apply_target(
                decision.symbol, decision.target_weight, price, self.equity(),
                order=decision.entry_order)
        except ModeSwitchRefused as exc:
            self.telemetry.event(Level.ERROR, "order", str(exc))
            return
        except VenueError as exc:
            # A rejected order is information, not a crash.
            self.telemetry.event(Level.ERROR, "order",
                                 f"{decision.symbol}: {exc.message}",
                                 detail=exc.remedy)
            self.telemetry.pulse(decision.symbol, "refused", exc.message, 0.9)
            return
        if decision.entry_order == MARKET_ON_CLOSE and decision.target_weight > 0:
            # Recorded on submission, not on fill: the closing auction has not
            # happened yet, and a hold that is forgotten because the fill was
            # still pending is a hold with no exit order behind it.
            self.overnight_holdings[decision.symbol] = decision.target_weight
            self._save_overnight_state()
        if fill:
            self.allocator.observe(decision.symbol).current_weight = \
                self.broker.weight_of(decision.symbol, price, self.equity())
            qty_text = format_decimal(fill.quantity)
            px_text = format_decimal(fill.price)
            self.telemetry.pulse(decision.symbol, "order",
                                 f"{fill.side} {qty_text} at {px_text}", 1.0)
            self.telemetry.event(
                Level.INFO, "order",
                f"{fill.side} {qty_text} {decision.symbol} at {px_text}"
                + (" (simulated)" if fill.simulated else ""))

    async def _tick(self) -> None:
        phase = self._update_session_phase()
        self._sync_trend_holdings()
        await self._drain_bars()
        # Every symbol gets looked at on a bounded cycle, whatever the feed is
        # doing. Without this, only the symbols the plan streams were ever
        # evaluated -- and with the market shut, none of them were.
        await self._sweep()
        if phase is SessionPhase.CLOSING:
            # After the drain, so a position entered on this tick is already
            # recorded as an intentional overnight hold and is not closed again.
            await self._flatten_unwanted_before_the_close()
        elif phase is SessionPhase.PREOPEN:
            await self._exit_overnight_holdings()
            self._report_unmanaged_equity()
        await self._refresh_account_limits()
        await self._reconcile_book()
        equity = self.equity()
        self.allocator.equity = equity
        self.allocator.cash = float(self.broker.cash)
        # Scaled on the book that is actually being traded, not on the account
        # record. In live mode they are the same number. In paper they are not:
        # the venue's own paper balance never moves, because no order is sent to
        # it, so scaling on the account would freeze a paper book at its opening
        # limits however much it grew -- and growing out of those limits is the
        # entire point. The book starts from the real balance because
        # absorb_account seeds it.
        self.apply_account_scale(equity)
        self._roll_trading_day(equity)
        if self.day_start_equity <= 0:
            self.day_start_equity = equity
        if self.allocator.check_daily_loss(self.day_start_equity) and self.running:
            self.telemetry.pulse("BOOK", "halt", self.allocator.halt_reason, 1.0)
        self._equity_curve.append((time.time(), equity))
        if len(self._equity_curve) > 2000:
            self._equity_curve = self._equity_curve[-2000:]

    def apply_account_scale(self, equity: float) -> bool:
        """Re-derive the limits for what this balance can actually trade.

        A percentage framework stops working quietly at a small balance: the
        base limits allow five positions of $11 on a $70 account, and $11 is
        not a position -- it cannot be held overnight, cannot be taken in a
        non-fractionable name, and cannot be trimmed. The account's size decides
        how many positions it can carry, and concentration follows from that.

        Re-derived rather than set once, because the balance moves and the whole
        point is that it grows: an account that reaches a few hundred dollars
        should spread back out on its own, and one that draws down should
        concentrate again rather than keep sizing for money it no longer has.

        The engines and the allocator are handed the new limits explicitly.
        They hold their own reference, so replacing only this one would leave
        every existing engine sizing against the limits of an account that no
        longer exists.
        """
        scale = risk_mod.scale_for_equity(equity)
        if (self.account_scale is not None
                and scale.positions == self.account_scale.positions
                and abs(scale.risk_per_trade - self.account_scale.risk_per_trade) < 1e-9):
            self.account_scale = scale
            return False

        previous = self.account_scale
        self.account_scale = scale
        self.limits = risk_mod.limits_for_equity(equity, self.base_limits)
        self.allocator.limits = self.limits
        for engine in self.engines.values():
            engine.limits = self.limits
        if previous is not None:
            self.telemetry.event(
                Level.INFO, "risk",
                f"limits re-scaled for a ${equity:,.2f} account: "
                f"{scale.positions} concurrent "
                f"{'position' if scale.positions == 1 else 'positions'} at up to "
                f"{scale.max_position_weight:.0%} each",
                detail=scale.note)
        return True

    def _roll_trading_day(self, equity: float) -> None:
        """Start a new day when the venue's calendar does.

        This is the single thing most likely to break a terminal left running
        for a week, and it breaks quietly. The daily-loss reference was taken
        once at startup and never moved, so by Thursday the "daily" loss was
        measured against Monday's equity -- and once the halt tripped, it was
        permanent: nothing cleared it, so the book stopped trading after its
        first bad afternoon and never started again.

        The day boundary comes from the venue's own clock rather than from
        local midnight, because the trading day this rule is about is the
        venue's, and a laptop in another timezone would otherwise roll the book
        in the middle of a session.
        """
        stamp = self.market_clock.timestamp or dt.datetime.now(tz=dt.timezone.utc)
        today = stamp.astimezone(dt.timezone.utc).date().isoformat()
        if today == self._trading_day:
            return
        first = self._trading_day == ""
        self._trading_day = today
        self.day_start_equity = equity if equity > 0 else self.day_start_equity
        self._seeded_from_account = self._seeded_from_account and not first
        if first:
            return

        released = self.allocator.roll_session()
        self.telemetry.event(
            Level.INFO, "session",
            f"new trading day {today}: the daily loss budget resets from "
            f"{equity:,.2f}" + (" and the daily-loss halt is released"
                                if released else ""))
        if released:
            self.telemetry.pulse("BOOK", "decision",
                                 "daily-loss halt released with the new day", 0.8)

    async def _reconcile_book(self) -> None:
        """Check the live book against the venue, and report any difference.

        Only in live mode: a simulated book has no venue to disagree with. A
        difference is not tidied away quietly -- it means an order did not do
        what this program was told it did, and that is the single most
        important thing an operator can be shown.
        """
        broker = self.broker
        if getattr(broker, "simulated", True) or not hasattr(broker, "reconcile"):
            return
        if time.time() - self._reconciled_at < RECONCILE_SECONDS:
            return
        try:
            drift = await broker.reconcile()
        except VenueError as exc:
            self.telemetry.event(Level.WARN, "order",
                                 "could not reconcile the book with the venue",
                                 detail=exc.message)
            return
        self._reconciled_at = time.time()
        for symbol, local, actual in drift:
            self.reconciliations += 1
            self.allocator.observe(symbol).current_weight = self.broker.weight_of(
                symbol, self.feed.quote(symbol).last, self.equity())
            self.telemetry.event(
                Level.ERROR, "order",
                f"{symbol}: this book held {format_decimal(local)} and the "
                f"venue holds {format_decimal(actual)} — corrected to the "
                f"venue",
                detail=("An accepted order can still be rejected, partly "
                        "filled, or filled at an auction hours later. Every "
                        "decision taken between then and now was sized against "
                        "the wrong number."))
            self.telemetry.pulse(symbol, "refused",
                                 "book corrected against the venue", 0.9)

    async def _refresh_account_limits(self) -> None:
        """Read the account the venue actually holds.

        The day-trade count is read rather than counted locally because a local
        count cannot survive a restart or trades made elsewhere in the same
        account, and being wrong means a ninety-day restriction rather than a
        missed trade.

        Equity and buying power are read for a plainer reason: they are the
        operator's money, and a terminal that displays a simulated default in
        the place where the balance goes is a terminal reporting a number it
        made up.
        """
        if self.client is None or not self.client.authenticated:
            return
        if time.time() - self._account_checked_at < ACCOUNT_REFRESH_SECONDS:
            return
        try:
            account = await self.client.account()
        except VenueError as exc:
            self.account_error = exc.message
            return
        self._account_checked_at = time.time()
        self.account_error = ""
        self.absorb_account(account)

    def absorb_account(self, account: dict[str, Any]) -> None:
        """Take the venue's numbers, field by field, tolerating any of them.

        Alpaca sends these as strings, and a single unparseable one must not
        discard the rest: an account whose buying power came back malformed
        still knows its own equity.
        """
        def number(key: str) -> float | None:
            raw = account.get(key)
            if raw in (None, ""):
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        for key, attr in (("equity", "account_equity"),
                          ("cash", "account_cash"),
                          ("buying_power", "account_buying_power"),
                          ("last_equity", "account_last_equity")):
            value = number(key)
            if value is not None:
                setattr(self, attr, value)

        self.account_currency = str(account.get("currency") or "USD")
        self.account_status = str(account.get("status") or "")
        self.account_number = str(account.get("account_number") or "")
        self.account_updated_at = time.time()

        try:
            self.allocator.day_trade_count = int(account.get("daytrade_count", 0) or 0)
            self.allocator.flagged_pattern_day_trader = bool(
                account.get("pattern_day_trader", False))
        except (TypeError, ValueError):
            pass

        # A simulated book starts from the real balance rather than from a
        # round number. Sizing against $10,000 when the account holds $700 does
        # not produce a smaller version of the same decisions -- it produces
        # different ones, because every limit here is a fraction of equity.
        # Only before the book has traded: re-seeding a book that already holds
        # positions would silently rewrite its P&L.
        if (getattr(self.broker, "simulated", False)
                and not self._seeded_from_account
                and self.account_cash > 0
                and not self.broker.fills):
            self.broker.cash = to_decimal(self.account_cash)
            self._seeded_from_account = True
            self.day_start_equity = self.account_equity or self.account_cash
            self.telemetry.event(
                Level.INFO, "account",
                f"simulated book seeded from the real account: "
                f"{self.account_cash:,.2f} {self.account_currency}")
        # The allocator's view of cash is otherwise only refreshed inside the
        # trading loop, so a terminal that has read an account but not yet been
        # started reported zero buying power against a funded balance.
        self.allocator.cash = float(self.broker.cash)
        self.allocator.equity = self.account_equity or self.allocator.equity
        if self.account_equity > 0:
            self.apply_account_scale(self.account_equity)

    async def _run(self) -> None:
        last_universe = 0.0
        last_full_scan = time.time()
        while self.running:
            self._loop_beat = time.time()
            try:
                await self._tick()
                if time.time() - last_universe > 60:
                    # The fast loop re-prices only what is traded: one request.
                    await self.refresh_universe()
                    # Cheap: it returns immediately unless a day has passed.
                    await self.refresh_daily_history()
                    last_universe = time.time()
                # Walk the ranking: retire what has been answered, bring in
                # what has not been looked at yet.
                await self.rotate_cohort()
                if time.time() - last_full_scan > FULL_SCAN_SECONDS:
                    # The slow loop re-ranks the whole market. Kept off the
                    # per-minute path because it is fifty requests, and a
                    # scanner that spends its budget ranking cannot re-price
                    # the book it is holding.
                    last_full_scan = time.time()
                    await self.scan_universe()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A failure in one tick must not stop the session. It is
                # recorded where the operator will see it.
                log.exception("the trading loop raised")
                self.telemetry.event(Level.ERROR, "session",
                                     f"the trading loop raised {type(exc).__name__}",
                                     detail=str(exc))
            await asyncio.sleep(1.0)

    # -- control ---------------------------------------------------------

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.started_at = time.time()
        self.lamps.session = "ok"
        self.keep_awake.acquire()
        self.status_message = f"running in {self.broker.mode.value}"
        self.telemetry.event(Level.GOOD, "session",
                             f"session started in {self.broker.mode.value}")
        self._load_overnight_state()
        await self.refresh_clock()
        await self.scan_universe()
        await self.seed_history()
        await self.refresh_daily_history(force=True)
        await self.refresh_universe()
        await self.feed.start(self._stream_priority())
        self._loop_task = asyncio.create_task(self._run(), name="trading-loop")

    async def supervise(self) -> None:
        """Restart the trading loop if it has stopped without being asked to.

        The loop catches everything inside its body, so the way it dies is not
        an exception -- it is the task itself ending: a cancellation from
        somewhere unexpected, or a failure in the ``await`` between iterations.
        The session then reports ``running`` with a live websocket, a green
        health score, and nothing evaluating a single bar. That is the worst
        failure mode this program has, because every indicator says it is fine.

        Called from the server's own heartbeat rather than from inside the loop,
        for the obvious reason that a dead loop cannot restart itself.
        """
        if not self.running:
            return
        task = self._loop_task
        alive = task is not None and not task.done()
        stalled = (self._loop_beat > 0
                   and time.time() - self._loop_beat > LOOP_STALL_SECONDS)
        if alive and not stalled:
            return

        if task is not None and task.done():
            # Surface why, if it left a reason behind.
            detail = ""
            with contextlib.suppress(Exception):
                exc = task.exception()
                if exc is not None:
                    detail = f"{type(exc).__name__}: {exc}"
            self.telemetry.event(
                Level.ERROR, "session",
                "the trading loop had stopped and was restarted",
                detail=detail or "the task ended without raising")
        elif stalled:
            self.telemetry.event(
                Level.ERROR, "session",
                f"the trading loop has not ticked for "
                f"{time.time() - self._loop_beat:.0f}s; restarting it")
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        self.restarts += 1
        self._loop_beat = time.time()
        self._loop_task = asyncio.create_task(self._run(), name="trading-loop")

    async def stop(self) -> None:
        self.running = False
        self.lamps.session = "off"
        # Handed back promptly: a stopped session has no claim on the machine.
        self.keep_awake.release()
        self.status_message = "stopped"
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        await self.feed.stop()
        self.telemetry.event(Level.INFO, "session", "session stopped")

    async def set_mode(self, mode: Mode, phrase: str = "",
                       store: CredentialStore | None = None) -> None:
        """Switch mode, flattening first. Always in that order."""
        if mode is self.broker.mode:
            return
        prices = self.prices()
        flattened = await self.broker.flatten_all(prices)
        for fill in flattened:
            self.telemetry.event(Level.WARN, "order",
                                 f"flattened {fill.symbol} before switching mode")

        if mode is Mode.LIVE:
            if self.client is None or self.credential is None:
                raise ModeSwitchRefused(
                    "no credential is attached, so there is no account to trade")
            tradeable = self.credential.trade_enabled
            broker = LiveBroker(self.spec, self.client, self.credential.name)
            broker.arm(phrase, tradeable)
            await broker.sync()
            self.broker = broker
            self.telemetry.event(
                Level.WARN, "mode",
                f"LIVE trading armed on {self.credential.name!r} — real orders "
                "will be sent")
        elif mode is Mode.PAPER:
            self.broker = PaperBroker(self.spec)
            self.telemetry.event(Level.INFO, "mode", "switched to paper trading")
        else:
            self.broker = DryRunBroker(self.spec)
            self.telemetry.event(Level.INFO, "mode", "switched to dry run")

        for state in self.allocator.states.values():
            state.current_weight = 0.0
        self.day_start_equity = self.equity()
        self.status_message = (f"running in {self.broker.mode.value}"
                               if self.running else "stopped")

    # -- the snapshot the UI renders --------------------------------------

    def snapshot(self, pulse_window: int = 240, *, since_pulse: int = 0,
                 since_event: int = 0, detail: int = DETAIL_ROWS,
                 rows_limit: int = WATCHLIST_ROWS) -> dict[str, Any]:
        """The one frame the websocket streams.

        Two things here exist purely to keep this affordable at 1Hz over a wide
        universe, because the cost of a frame is not what it takes to build --
        that is milliseconds -- but what the browser must parse and lay out
        before the next one arrives.

        ``since_pulse``/``since_event`` turn the telemetry rings into deltas.
        Re-sending a 240-pulse window every second spends almost all of its
        bandwidth on rows the client already has.

        ``detail`` bounds how many symbols carry their full reasoning. The
        decision dictionary is the largest thing per row and the reasoning panel
        reads only a couple of dozen of them, so it is sent for the symbols
        closest to trading, plus everything actually holding a position -- a
        position whose reasoning vanished because its symbol fell down a sort
        order is the one case that would be indefensible.
        """
        prices = self.prices()
        equity = self.equity()
        age = self.feed.data_age
        self.lamps.data = ("ok" if age < STALE_AFTER_SECONDS
                           else ("stale" if math.isfinite(age) else "off"))

        held = {sym for sym, pos in self.broker.positions.items() if not pos.is_flat}
        ranked = sorted(
            self.universe,
            key=lambda sym: (
                0.0 if sym in held
                else (self.engines[sym].decision.distance_to_trading
                      if sym in self.engines else 9.0)))
        detailed = set(ranked[:max(0, detail)]) | held
        # Held symbols are always streamed, wherever they rank. A position that
        # disappeared from the table because its symbol drifted down a sort
        # order is the one omission that would be indefensible.
        shown = ranked[:max(0, rows_limit)]
        shown_set = set(shown) | held
        omitted = len(self.universe) - len(shown_set)

        rows = []
        for symbol in self.universe:
            if symbol not in shown_set:
                continue
            state = self.allocator.observe(symbol)
            q = self.feed.quote(symbol)
            d = self.engines[symbol].decision if symbol in self.engines else Decision(symbol)
            row = {
                "symbol": symbol,
                "price": q.last,
                "change_pct": q.change_pct,
                "turnover": q.quote_volume,
                "verdict": state.verdict.value,
                "weight": state.current_weight,
                "target": d.target_weight,
                "age": None if not math.isfinite(q.age) else round(q.age, 1),
                # Classified from the symbol rather than read off the
                # decision, so the tag is right on the first frame. A decision
                # that has never been evaluated carries no class, which left
                # every row untagged at startup -- and an untagged row reads as
                # an unclassified symbol rather than as a field that has not
                # arrived yet.
                "asset_class": classify_symbol(symbol).value,
                "strategy": d.strategy,
            }
            if symbol in detailed:
                row["decision"] = d.as_dict()
                # The table shows this as a hover tooltip only, so it travels
                # with the reasoning rather than on every row: a sentence per
                # symbol per second is most of the frame at a wide universe.
                row["reason"] = (
                    state.reason if state.verdict in
                    (Verdict.NOT_ADMITTED, Verdict.REJECTED) and state.reason
                    else d.reason)
            rows.append(row)

        health = self._health_score()
        positions = [
            {"symbol": s, "quantity": float(p.quantity),
             "avg_price": float(p.avg_price),
             "price": prices.get(s, 0.0),
             "value": float(p.quantity) * prices.get(s, 0.0),
             "unrealised": (prices.get(s, 0.0) - float(p.avg_price)) * float(p.quantity)
             if prices.get(s) else 0.0}
            for s, p in self.broker.positions.items() if not p.is_flat
        ]

        return {
            "ts": time.time(),
            "running": self.running,
            "uptime": (time.time() - self.started_at) if self.started_at else 0.0,
            "restarts": self.restarts,
            "reconciliations": self.reconciliations,
            "keep_awake": self.keep_awake.as_dict(),
            "loop_age": (time.time() - self._loop_beat) if self._loop_beat else None,
            "limits": self._limits_block(),
            "account_scale": self._scale_block(),
            "regime_census": self._regime_census(),
            "costs": self._cost_summary(),
            "venue_budget": self._venue_budget(),
            "counters": self.telemetry.kind_counts,
            "execution": self._execution_quality(),
            "drawdown": self._drawdown(),
            "mode": self.broker.mode.value,
            "simulated": getattr(self.broker, "simulated", True),
            "venue": self.spec.display_name,
            "environment": self.client.environment if self.client else "paper",
            "market": {
                "is_open": self.market_clock.is_open,
                "describe": self.market_clock.describe(),
                "next_open": (self.market_clock.next_open.isoformat()
                              if self.market_clock.next_open else None),
                "next_close": (self.market_clock.next_close.isoformat()
                               if self.market_clock.next_close else None),
                "crypto_only": self._crypto_only(),
                "feed": self.spec.default_feed,
            },
            "overnight": self._overnight_block(),
            "trend": self._trend_block(),
            "universe_scan": {
                "note": self.scan_note,
                "scanned_at": self.universe_scanned_at,
                # When the next full re-rank is due. Without it a note that has
                # not changed for fifteen minutes reads as a stall.
                "next_scan_in": max(0.0, FULL_SCAN_SECONDS - (
                    time.time() - self.universe_scanned_at))
                if self.universe_scanned_at else 0.0,
                "sweeps": self.sweeps,
                "sweep_at": self._sweep_cursor,
                # The cohort walking the ranked market. "150 of 11,005" is the
                # answer to "is it actually scanning everything".
                "ranked": len(self.ranked_universe),
                "cohort_at": self._cohort_cursor,
                "cohort_passes": self.cohort_passes,
                "cohort_progress": self.cohort_progress,
                "pooled_symbols": len(self._trend_samples),
                "size": len(self.universe),
                # Streamed vs evaluated. Every symbol below the line is still
                # being scanned and still counted in the census; it is only the
                # table that stops at the line.
                "shown": len(rows),
                "omitted": max(0, omitted),
                "considered": self.universe_considered,
                "priced": self.universe_priced,
            },
            "status": self.status_message,
            "venue_error": self.venue_error,
            "calibration_error": self.calibration_error,
            "store_error": self.store_error,
            "lamps": self.lamps.as_dict(),
            "equity": equity,
            "account": self._account_block(),
            "cash": float(self.broker.cash),
            "realised_pnl": float(self.broker.realised_pnl),
            "unrealised_pnl": sum(p["unrealised"] for p in positions),
            "gross_exposure": self.allocator.gross_exposure,
            "gross_ceiling": self.limits.max_gross_exposure,
            "per_symbol_budget": self.allocator.per_symbol_budget,
            "halted": self.allocator.halted,
            "halt_reason": self.allocator.halt_reason,
            "credential": self.credential.name if self.credential else None,
            "trade_enabled": bool(self.credential and self.credential.trade_enabled),
            "watchlist": rows,
            "positions": positions,
            # A deque does not slice; the journal shows the newest first.
            "fills": [f.as_dict() for f in list(self.broker.fills)[-40:]][::-1],
            "events": self.telemetry.events(60, since=since_event),
            "pulses": self.telemetry.pulse_window(pulse_window, since=since_pulse),
            "pulse_seq": self.telemetry.latest_pulse_seq,
            "event_seq": self.telemetry.latest_event_seq,
            # True when this frame carries only what changed. The client keeps
            # its own rings in that case instead of replacing them, and a frame
            # that says False is its instruction to start over -- which is what
            # a reconnect, or falling behind the ring, needs.
            "delta": bool(since_pulse or since_event),
            "detailed": sorted(detailed),
            "health": health,
            "feed": {
                "connected": self.feed.connected,
                # Streamed against scanned. The gap is the plan's subscription
                # cap, not a fault, and the symbols above it are still scanned.
                "streamed": self.feed.streamed,
                "symbol_limit": self.feed.symbol_limit,
                "dropped": self.feed.dropped,
                "reconnects": self.feed.reconnects,
                "errors": self.feed.errors,
                "age": None if not math.isfinite(age) else round(age, 1),
                "last_error": self.feed.last_error,
                # The dark-lamp explanation. Computed here because the reason
                # depends on the market clock and the session state, neither of
                # which the feed knows about.
                "reason": self._feed_reason(),
            },
            "equity_curve": self._equity_curve[-240:],
        }

    def _scale_block(self) -> dict[str, Any]:
        """How the limits were adjusted for this balance, and what that costs.

        Published in full because "why is it not trading" on a small account is
        almost always this, and the answer is arithmetic rather than a fault: a
        position has to be worth something before a venue will treat it as one.
        """
        scale = self.account_scale
        equity = self.account_equity or self.equity()
        floor = risk_mod.VIABLE_POSITION_NOTIONAL
        return {
            "known": scale is not None,
            "equity": equity,
            "positions": scale.positions if scale else self.limits.max_concurrent_positions,
            "max_position_weight": self.limits.max_position_weight,
            "max_position_value": self.limits.max_position_weight * equity,
            "risk_per_trade": self.limits.risk_per_trade,
            "risk_per_trade_value": self.limits.risk_per_trade * equity,
            "daily_loss_halt": self.limits.daily_loss_halt,
            "daily_loss_value": self.limits.daily_loss_halt * equity,
            "position_floor": floor,
            "scaled": bool(scale and not scale.unscaled),
            "note": scale.note if scale else "the account balance is not known yet",
            #: The most expensive share this account can hold overnight. Auction
            #: orders take whole shares, so this is a hard reach limit rather
            #: than a preference, and on a small balance it excludes most of the
            #: market.
            "overnight_max_share_price": self.limits.max_position_weight * equity,
        }

    def _account_block(self) -> dict[str, Any]:
        """The real account, kept distinct from the simulated book.

        Both are published because in dry run and paper they are different
        numbers and the difference is the point: the account is the operator's
        actual money, the book is what this program has done to a copy of it.
        Showing one in the other's place is how a terminal comes to report a
        balance nobody has.
        """
        known = self.account_updated_at > 0
        book = self.equity()
        return {
            "known": known,
            "equity": self.account_equity,
            "cash": self.account_cash,
            "buying_power": self.account_buying_power,
            "last_equity": self.account_last_equity,
            "currency": self.account_currency,
            "status": self.account_status,
            "updated_at": self.account_updated_at,
            "age": (time.time() - self.account_updated_at) if known else None,
            "error": self.account_error,
            "seeded": self._seeded_from_account,
            "book_equity": book,
            # In live mode the book *is* the account, so a gap between them is
            # a reconciliation failure rather than simulated P&L.
            "simulated": bool(getattr(self.broker, "simulated", True)),
            #: Day P&L against the venue's own previous close, which is the
            #: figure the broker's own app shows.
            "day_pnl": (self.account_equity - self.account_last_equity
                        if known and self.account_last_equity else 0.0),
        }

    def _trend_block(self) -> dict[str, Any]:
        """The trend strategy's state, and the constraint that selects it.

        The day-trade budget is published alongside, because on a small account
        that is *why* this strategy is the one running: an intraday round trip
        it cannot close is not a trade, and a multi-day hold is not a day trade.
        """
        pooled = self.pooled_trend
        decisions = [e.decision for e in self.engines.values()
                     if e.decision.strategy == "trend"]
        carried = [
            {"symbol": symbol,
             "days": round((time.time() - opened) / 86_400.0, 2),
             "min_days": round(
                 self.engines[symbol].decision.trend_min_hold_days, 1)
             if symbol in self.engines else 0.0}
            for symbol, opened in sorted(self.trend_holdings.items())
        ]
        return {
            "measured": pooled is not None,
            "credible": bool(pooled and pooled.credible),
            "beta_bps": pooled.beta_bps if pooled else 0.0,
            "t_stat": pooled.t_stat if pooled else 0.0,
            "observations": pooled.observations if pooled else 0,
            "symbols": pooled.symbols if pooled else 0,
            "note": self.trend_note,
            "candidates": len(decisions),
            "eligible": sum(1 for d in decisions if d.verdict is Verdict.TRADING),
            "holdings": carried,
            "day_trades_available": self.allocator.day_trades_available(),
            "day_trades_left": max(
                0, self.limits.pdt_max_day_trades - self.allocator.day_trade_count),
            #: When options become reachable, and why they are not yet. The
            #: question comes up on every small account, and the answer is
            #: arithmetic rather than a policy.
            "options": self._options_block(),
            # The reason this strategy exists, in one line.
            "why": ("a position held across a session close is not a day trade, "
                    "so this is the horizon an account under the "
                    f"${self.limits.pdt_equity_floor:,.0f} floor can actually "
                    "trade"),
        }

    def _options_block(self) -> dict[str, Any]:
        """Why options are not traded, expressed as a number rather than a rule.

        A contract is a hundred shares, so the smallest possible option
        position is a hundred times the quoted premium and cannot be reduced.
        Below the threshold an account can only reach contracts so cheap that
        the quoted spread is a large fraction of the premium.
        """
        equity = self.account_equity or self.equity()
        budget = self.limits.max_position_weight * equity
        reachable = budget / float(assets_mod.OPTION_CONTRACT_MULTIPLIER)
        threshold = float(assets_mod.MIN_OPTION_ACCOUNT_EQUITY)
        return {
            "tradeable": False,
            "contract_multiplier": assets_mod.OPTION_CONTRACT_MULTIPLIER,
            "max_premium_reachable": reachable,
            "min_equity": threshold,
            "affordable": equity >= threshold,
            "fees_per_contract_round_trip": float(
                assets_mod.OPTION_FEES_PER_CONTRACT_ROUND_TRIP),
            "note": (
                f"one contract is 100 shares, so this account can reach a "
                f"premium of ${reachable:,.2f} — "
                + ("enough for liquid contracts, but options still need an "
                   "implied-volatility model this program does not have"
                   if equity >= threshold else
                   f"only contracts cheap enough that the spread is most of "
                   f"the premium. Options need about ${threshold:,.0f}.")),
        }

    def _overnight_block(self) -> dict[str, Any]:
        """The overnight drift strategy's whole state, published in full.

        The headline number is deliberately shown next to the cost that has to
        be cleared, because the honest summary of this anomaly is that it is
        real and roughly the size of a round trip. A panel that showed only the
        drift would read as free money; showing both shows why the strategy
        refuses most nights, which is the correct behaviour rather than a fault.
        """
        pooled = self.pooled_drift
        candidates = [e.decision for e in self.engines.values()
                      if e.decision.strategy == "overnight"]
        eligible = [d for d in candidates if d.verdict is Verdict.TRADING]
        held = [{"symbol": s, "weight": w}
                for s, w in sorted(self.overnight_holdings.items())]
        return {
            "phase": self.session_phase.value,
            "note": self.overnight_note,
            "measured": pooled is not None,
            "credible": bool(pooled and pooled.credible),
            "mean_bps": pooled.mean_bps if pooled else 0.0,
            "intraday_bps": pooled.intraday_bps if pooled else 0.0,
            "t_stat": pooled.t_stat if pooled else 0.0,
            "observations": pooled.observations if pooled else 0,
            "symbols": pooled.symbols if pooled else 0,
            "vol_bps": pooled.vol_bps if pooled else 0.0,
            "candidates": len(candidates),
            "eligible": len(eligible),
            "holdings": held,
            "entry_order": MARKET_ON_CLOSE,
            "exit_order": MARKET_ON_OPEN,
            # The single most useful fact about this strategy on a small
            # account, and the one nothing else on screen would tell you.
            "exempt_from_pdt": True,
            "options_note": (
                "options are not used for this: an at-the-money option needs "
                "roughly 53bp of overnight drift at one day to expiry, and 10bp "
                "at thirty, against a measured drift of a few bp"),
        }

    def _limits_block(self) -> dict[str, Any]:
        """Every hard bound, and how much of each is currently spent.

        Published in full because an autonomous book that is not trading is
        usually being held by one specific limit, and guessing which one from a
        single "gross exposure" figure is exactly the diagnosis this panel is
        meant to remove.
        """
        lim = self.limits
        admitted = len(self.allocator.admitted_symbols)
        return {
            "max_gross_exposure": lim.max_gross_exposure,
            "max_position_weight": lim.max_position_weight,
            "max_concurrent_positions": lim.max_concurrent_positions,
            "buying_power_reserve": lim.buying_power_reserve,
            "daily_loss_halt": lim.daily_loss_halt,
            "risk_per_trade": lim.risk_per_trade,
            "target_volatility": lim.target_volatility,
            "atr_stop_multiple": lim.atr_stop_multiple,
            "slots_used": admitted,
            "day_trade_count": self.allocator.day_trade_count,
            "pdt_floor": lim.pdt_equity_floor,
            "pdt_max_day_trades": lim.pdt_max_day_trades,
            "pdt_blocked": self.allocator.pdt_blocked(),
            "flagged_pattern_day_trader": self.allocator.flagged_pattern_day_trader,
            "slots_max": lim.max_concurrent_positions,
            "buying_power": self.allocator.buying_power(),
            "reserved": float(self.broker.cash) * lim.buying_power_reserve,
        }

    def _drawdown(self) -> dict[str, Any]:
        """Loss against the day's opening equity, and how close that is to the halt."""
        start = self.day_start_equity
        equity = self.equity()
        if start <= 0:
            return {"pct": 0.0, "limit": self.limits.daily_loss_halt,
                    "used": 0.0, "day_start_equity": start}
        pct = max(0.0, (start - equity) / start)
        limit = self.limits.daily_loss_halt
        return {
            "pct": pct,
            "limit": limit,
            # Fraction of the halt budget consumed. This is the number that
            # matters: 3% of a 4% limit is 75% spent, not "only 3%".
            "used": min(1.0, pct / limit) if limit > 0 else 0.0,
            "day_start_equity": start,
        }

    def _regime_census(self) -> dict[str, int]:
        """What the book is currently seeing, counted by regime.

        A fleet-level answer to "why is nothing trading". Fifteen symbols all
        reading indeterminate is a different situation from fifteen still
        warming up, and both look like "no positions" without this.
        """
        census: dict[str, int] = {}
        for symbol in self.universe:
            engine = self.engines.get(symbol)
            regime = engine.decision.regime if engine else Regime.WARMING_UP.value
            census[regime] = census.get(regime, 0) + 1
        return census

    def _cost_summary(self) -> dict[str, Any]:
        """What the book is being priced at, per asset class.

        Reported per class rather than as one number, because the classes are
        not comparable: a US equity round trip is almost entirely spread, while
        a crypto round trip is dominated by commission. A single blended figure
        would describe neither.
        """
        decisions = [e.decision for e in self.engines.values()
                     if e.decision.round_trip_cost_bps > 0]
        by_class: dict[str, Any] = {}
        for asset_class in (AssetClass.US_EQUITY, AssetClass.CRYPTO):
            spec = spec_for(asset_class)
            members = [d for d in decisions if d.asset_class == asset_class.value]
            costs_bps = sorted(d.round_trip_cost_bps for d in members)
            by_class[asset_class.value] = {
                "display_name": spec.display_name,
                "commission_bps": float(spec.cost_model.commission_bps),
                "sell_side_bps": float(spec.cost_model.sell_side_bps),
                "assumed": spec.cost_model.assumed,
                "source": spec.cost_model.source,
                "symbols": len(members),
                "median_round_trip_bps": (costs_bps[len(costs_bps) // 2]
                                          if costs_bps else 0.0),
                "spreads_measured": sum(1 for d in members if not d.spread_assumed),
                "seconds_per_year": spec.seconds_per_year,
                "shortable": spec.shortable,
            }
        cheapest = min(decisions, key=lambda d: d.round_trip_cost_bps, default=None)
        dearest = max(decisions, key=lambda d: d.round_trip_cost_bps, default=None)
        all_costs = sorted(d.round_trip_cost_bps for d in decisions)
        return {
            "by_asset_class": by_class,
            "safety_multiple": self.params.safety_multiple,
            "adverse_selection_fraction": float(ADVERSE_SELECTION_FRACTION),
            "median_round_trip_bps": (all_costs[len(all_costs) // 2]
                                      if all_costs else 0.0),
            "fees_assumed": any(b["assumed"] for b in by_class.values()),
            "fee_source": "; ".join(
                f"{b['display_name']}: {b['source']}" for b in by_class.values()),
            "cheapest": ({"symbol": cheapest.symbol,
                          "bps": cheapest.round_trip_cost_bps} if cheapest else None),
            "dearest": ({"symbol": dearest.symbol,
                         "bps": dearest.round_trip_cost_bps} if dearest else None),
            "spreads_measured": sum(1 for d in decisions if not d.spread_assumed),
            "spreads_total": len(decisions),
            "maker_bps": 0.0,
            "taker_bps": float(spec_for(AssetClass.US_EQUITY).cost_model.commission_bps),
        }

    def _execution_quality(self) -> dict[str, Any]:
        """Did execution cost what the cost gate assumed?

        The gate admits a symbol on a *modelled* crossing cost. Measuring what
        crossing actually cost is the only way to find out the model is wrong
        before the P&L does -- and with two asset classes carrying very
        different models, one being wrong is easy to miss in a blended figure.
        """
        fills = self.broker.fills
        if not fills:
            return {"count": 0, "notional": 0.0, "buys": 0, "sells": 0,
                    "avg_slippage_bps": 0.0, "worst_slippage_bps": 0.0,
                    "modelled_bps": 0.0, "simulated": 0}
        slips = [f.slippage_bps for f in fills]
        modelled = [d.round_trip_cost_bps / 2 for d in
                    (e.decision for e in self.engines.values())
                    if d.round_trip_cost_bps > 0]
        return {
            # Lifetime, not the length of the ring. The ring is bounded so a
            # week of trading is not a leak; reporting its length as the fill
            # count would make the session's own history appear to reset.
            "count": getattr(self.broker, "fills_total", len(fills)),
            "window": len(fills),
            "notional": float(getattr(self.broker, "notional_total", 0)
                              or sum(f.notional for f in fills)),
            "buys": sum(1 for f in fills if f.side == "BUY"),
            "sells": sum(1 for f in fills if f.side == "SELL"),
            "avg_slippage_bps": sum(slips) / len(slips),
            "worst_slippage_bps": max(slips),
            # One-way modelled cost, which is what a single fill should pay.
            "modelled_bps": (sum(modelled) / len(modelled)) if modelled else 0.0,
            "simulated": sum(1 for f in fills if f.simulated),
        }

    def _venue_budget(self) -> dict[str, Any]:
        """The venue's own view of the request budget, as it reports it.

        Alpaca publishes a remaining-requests count per minute in response
        headers, so this is read rather than counted locally: a local count
        cannot see requests the same key made elsewhere, and being wrong here
        means a 429 in the middle of an exit.
        """
        client = self.client
        if client is None:
            return {"remaining": 0, "limit": 0, "utilisation": 0.0,
                    "retry_after": 0.0, "throttled": False}
        budget = client.budget
        pause = budget.pause_needed()
        return {
            "remaining": budget.remaining,
            "limit": budget.limit,
            "utilisation": budget.utilisation,
            "retry_after": round(pause, 2),
            "throttled": pause > 0,
        }

    def _feed_reason(self) -> str:
        """Why the data lamp reads the way it does, in a sentence.

        A dark lamp covers five different situations -- not started, still
        connecting, refused by the venue, connected with the market shut, or
        connected with the plan silently refusing the subscription -- and an
        operator cannot act on a lamp that will not say which. Every fact
        needed to tell them apart was already being sent to the browser and
        thrown away there, so the lamp was the only signal and it was
        ambiguous.
        """
        if not self.running:
            return "the session is not started, so nothing is subscribed"
        if not self.feed.connected:
            if self.feed.last_error:
                return f"the data socket is not connected — {self.feed.last_error}"
            return "the data socket is connecting"

        streamed = self.feed.streamed
        age = self.feed.data_age
        # A subscription of nothing on a connected socket is the failure that
        # looks most like a quiet market: the plan rejected the request whole
        # and the socket is sitting there delivering silence.
        if streamed == 0:
            return "connected, but nothing is subscribed"
        if math.isfinite(age) and age < STALE_AFTER_SECONDS:
            return f"{streamed} symbols streaming live"

        shut = not self.market_clock.is_open and not self._crypto_only()
        when = ""
        if shut and self.market_clock.next_open:
            when = f", which opens {self.market_clock.describe()}"
        if not math.isfinite(age):
            if shut:
                return (f"connected and subscribed to {streamed} symbols; the "
                        f"equity market is shut{when}, so there are no bars to "
                        f"send yet — this is not a fault")
            return (f"connected and subscribed to {streamed} symbols, none of "
                    f"which has sent anything yet")
        if shut:
            return (f"last price {age:.0f}s ago; the equity market is "
                    f"shut{when}, so the stream is idle by design")
        return (f"connected to {streamed} symbols but nothing has arrived for "
                f"{age:.0f}s — the market is open, so this is worth watching")

    def _health_score(self) -> dict[str, Any]:
        """A composite 0..1 with the components named.

        A single number nobody can decompose is not diagnostic, so the parts are
        published alongside it.
        """
        age = self.feed.data_age
        data = 1.0 if age < 5 else (0.6 if age < STALE_AFTER_SECONDS else 0.0)
        link = 1.0 if self.feed.connected else 0.0
        venue = {"ok": 1.0, "off": 0.5, "bad": 0.0}.get(self.lamps.venue, 0.5)
        errors = max(0.0, 1.0 - min(1.0, self.feed.errors / 20.0))
        reconn = max(0.0, 1.0 - min(1.0, self.feed.reconnects / 10.0))
        score = 0.30 * data + 0.25 * link + 0.20 * venue + 0.15 * errors + 0.10 * reconn
        if self.allocator.halted:
            score = min(score, 0.35)
        return {
            "score": round(score, 3),
            "components": {"data": data, "link": link, "venue": venue,
                           "errors": errors, "reconnects": reconn},
            "data_age": None if not math.isfinite(age) else round(age, 1),
            "reconnects": self.feed.reconnects,
            "errors": self.feed.errors,
        }
