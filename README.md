# IMPERIUM

A self-contained algorithmic trading terminal for **Alpaca** — US equities and
crypto — with a
live instrument-panel UI, packaged as a double-clickable Windows executable.

Python 3.11+, `uv` for dependencies, FastAPI and a websocket for the UI, plain
HTML/CSS/canvas for the front end. No framework, no build step.

---

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"

# 1. Can this machine even reach the venue? Six layers, first failure wins.
uv run imperium diagnose

# 2. Store a key. It cannot trade until you separately enable it.
uv run imperium keys add --name main          # prompts without echo
uv run imperium balance                       # the Phase-1 deliverable

# 3. Measure the regime classifier's thresholds. Nothing is hardcoded.
uv run imperium calibrate

# 4. Run the terminal.
uv run imperium serve
```

The terminal is at <http://127.0.0.1:8787/> and the diagnostics at
<http://127.0.0.1:8787/diagnose>.

---

## Design decisions worth knowing about

### No ccxt, and no multi-venue abstraction

`src/imperium/venues/alpaca/` is a direct REST client against the documented
API. A library covering a hundred venues must flatten each venue's error
vocabulary into a common one, and that flattening discards what an operator
needs. Alpaca's most common real failure is a good example: a **paper key
against the live host** (or the reverse) returns a bare `401` that is
indistinguishable from an invalid key. The client names the environment in the
remedy, so you don't regenerate a key that was fine.

There is no request signing here — authentication is two headers — so the entire
class of signature failures does not exist. What replaces it is the market
clock: equities trade 6.5 hours a day, so the venue's own clock is a first-class
part of the client rather than a local calendar guess.

### Asset classes are where the strategy actually splits

Crypto and equities are not the same instrument wearing different tickers.
`venues/assets.py` holds every difference that changes the arithmetic:

| | US equity | Crypto |
|---|---|---|
| trading seconds/year | `252 × 6.5 × 3600` = 5,896,800 | `365 × 24 × 3600` = 31,536,000 |
| session gaps in the series | yes — dropped before any statistic | none |
| shorting | if shortable **and** easy to borrow | not at all |
| commission | none | ~25bp per leg |
| regulatory fee | ~1bp, **sell leg only** | none |
| measured round trip | ~2bp (almost all spread) | ~51bp (almost all commission) |
| regime calibration | fitted on session-and-gap nulls | fitted on unbroken walks |

Three of these are load-bearing:

- **The calendar.** Annualising an equity over the crypto figure overstates its
  volatility by **2.31×**, and since volatility is the denominator of the
  volatility-target sizer, every position lands at ~43% of target.
- **Overnight gaps.** A close-to-open move is not a one-minute return. Measured
  on a three-session series, leaving gaps in inflates per-bar volatility by
  **1.57×** and lets a handful of pseudo-returns dominate the variance ratio.
- **Cost shape.** An equity round trip is almost entirely spread; a crypto one
  is almost entirely commission. The same 20bp edge is *admitted* for AAPL and
  *refused* for BTC/USD. One cost model would get one of those wrong.

**Options are recognised but not traded.** An option's return is a non-linear
function of the underlying's, so the variance-ratio regime test and the
volatility-target sizer — both of which assume returns are the thing being
forecast — do not carry over. The class exists so an option position is never
silently sized as though it were its underlying. Trading them needs a different
model, not a different threshold.

### The overnight drift strategy

US equities have historically earned most of their return while the market is
shut. The literature is consistent about the effect and equally consistent
about how hard it is to keep:

| source | finding |
|---|---|
| Cooper, Cliff & Gulen (2008) | S&P 500 constituents 1993–2006: **night 2.82–4.76bp**, day **−2.85 to +0.22bp** |
| Lou, Polk & Skouras (JFE 2019) | a "tug of war" — for large stocks momentum accrues overnight, for small stocks intraday |
| Berkman et al. (JFQA 2012) | the reversal concentrates in high-retail-attention, hard-to-value names; selling into an inflated open is the favourable side |
| practitioner replications | at **$0.01/share** the strategy's Sharpe falls to ~**0.31** |
| NSPY / NIWM ETFs | launched 2022 specifically to harvest it; **both closed within a year** |

The honest summary is that the drift is real, is a few basis points a night,
and is the same order of magnitude as one round trip. So the strategy is built
around that fact rather than around the headline:

- **Same cost gate as everything else.** A US equity round trip is ~2–4bp
  against a drift of ~4bp. Most nights it does not clear, and the refusal says
  so in those words — that is the correct answer, not a fault. Exempting this
  trade from the gate would produce a strategy that trades every night and
  loses slowly, which is precisely what the replications describe.
- **Pooled estimation, not per-symbol.** A 4bp effect against an ~80bp nightly
  standard deviation gives a standard error of ~8.4bp on one symbol's year of
  history: t ≈ 0.1, invisible. Pooled across 40 symbols × 89 nights = 3,560
  symbol-nights, t ≈ 6.5. Across forty symbols *somebody* always looks
  significant on their own data — that is selection on noise, and
  `tests/test_overnight.py` pins both halves of it.
- **Shrinkage toward the market.** A symbol that measured +13.6bp on its own
  history is traded as ~+4.6bp, by inverse-variance weighting. Its own data
  carries almost no information at this effect size.
- **Auction orders, not market orders.** Entry is **market-on-close** (`cls`),
  exit is **market-on-open** (`opg`). The trade is defined by being paid the
  close-to-open move; a market order at 15:45 takes intraday risk it is not
  paid for, and a market order after the bell has already missed the print.
  Alpaca refuses an MOC inside the last 10 minutes and an MOO inside the last
  2 before the open, so the entry and exit windows close before the venue's do.
- **Sized on gap risk, with no stop.** A gap opens *through* a stop without
  touching it, so there is no ATR stop behind this position. Size is bounded by
  the risk budget against a 3-sigma overnight move, using the **overnight**
  volatility — a different, fatter-tailed distribution than the intraday one.
- **Event risk is refused.** Alpaca's basic plan publishes no earnings
  calendar, so a last gap beyond 3 sigma of the symbol's own overnight
  volatility is treated as news rather than premium. This is a statistical
  stand-in and is labelled as one.
- **A position the night does not want is closed, not carried.** A refusal
  never reduces a position during the session — "no new exposure" is not "sell
  what you have" — but at the close that would silently turn an intraday
  position into an overnight one, sized against an intraday distribution and
  stopped by an ATR stop that a gap goes straight through. The closing window
  is where that gets decided explicitly, on the answer the overnight strategy
  just gave.
- **It is not a day trade.** Entering on one close and exiting on the next open
  does not touch the PDT counter. This is a genuine structural advantage on a
  sub-$25,000 account: the intraday strategy stops at two day trades, the
  overnight one can run every night. The exemption is in
  `PortfolioAllocator.clamp` and is the only thing that bypasses that ceiling.

#### Why options do not carry this

Options are not used for the overnight trade, and the reason is arithmetic
rather than caution. `scripts/overnight_option_arithmetic.py` computes the
overnight drift an at-the-money option needs simply to break even on one
night's time decay:

| days to expiry | 1 | 2 | 7 | 30 | 90 | 365 |
|---|---|---|---|---|---|---|
| breakeven overnight drift | 52.7bp | 37.4bp | 20.2bp | 10.0bp | 5.9bp | 3.0bp |

Against a measured drift of ~4bp, every tenor that has enough leverage to be
worth the spread loses to theta by an order of magnitude, and the only tenors
that break even are so long-dated that delta is small and the position is a
worse-priced stock substitute. Deep-in-the-money options are stock substitutes
with a wider spread. There is no version of this that works, so the program
does not pretend otherwise.

### A small account is not a scaled-down large one

The limits are percentages, and a percentage of a small balance can be an
amount no venue will trade. On a $70 account the base limits allow five
positions of $11 each, and a 2%-ATR name sizes to **$7.00**. Alpaca accepts
that as a fractional order, which is exactly the problem — it looks like it
worked. A $7 position cannot be held overnight (auction orders take whole
shares), cannot be taken at all in a non-fractionable name, and cannot be
trimmed.

So the balance decides how many positions the book can carry, and concentration
follows from that:

| equity | positions | max position | risk/trade | daily halt |
|---|---|---|---|---|
| $70 | 2 | $28.00 (40%) | 1.25% | 10% |
| $120 | 3 | $32.00 (27%) | 0.83% | 6.7% |
| $200+ | 5 | 20% | 0.50% | 4% |

It converges exactly on the base limits above ~$160 and changes nothing for an
account of any ordinary size. It is re-derived on every tick from the book
being traded, so an account that grows spreads back out on its own and one that
draws down concentrates again.

**The floor.** $25 is the smallest position this program will place, set by
three venue facts rather than by preference: fractional fills stop at $1 of
notional so anything smaller cannot be trimmed; auction orders take whole
shares; and non-fractionable symbols need whole shares for any order. $25 buys
one share of a large part of the market. A position sized below it is raised to
it when that stays inside the per-symbol cap *and* risks no more than 2× the
per-trade budget at its stop — and refused, with the arithmetic, when it does
not. At $70 that means calm names trade at $25 and volatile ones are declined:

| ATR | stop | size | risk at stop | |
|---|---|---|---|---|
| 1% | 2.5% | $25.00 | 0.89% | raised to the floor |
| 2% | 5.0% | $25.00 | 1.79% | raised to the floor |
| 4% | 10.0% | — | — | refused: too much risk for the smallest tradeable size |

**Deliberately not scaled:** `atr_stop_multiple` and `target_volatility`. Both
describe the market, not the wallet. How far a stock moves before a stop is a
property of the stock, and widening it because the account is small would
*cut* the position for a given risk budget — the opposite of what a small
account needs. The lever that makes positions viable is the risk budget, and
that is scaled.

**What a $70 account cannot do.** Hold any share priced above $28 overnight, at
any weight, because the closing auction is whole-share only. That excludes most
of the market, and the terminal says so per symbol rather than showing an
unexplained absence of overnight trades. This is also why the overnight
strategy matters at this size: it is not a day trade, so it is the one strategy
a sub-$25,000 account can run every night without touching the PDT ceiling.

### Pattern-day-trader limits

A US margin account under $25,000 may make three day trades in five rolling
business days; the fourth restricts it for ninety days. An autonomous book hits
that in a morning. The day-trade count is read from the venue on each tick —
counting locally cannot survive a restart or trades made elsewhere — and new
exposure stops at two. Exits always pass.

The overnight drift trade is the one exception, and it is not a loosened limit:
buying on one session's close and selling on the next session's open is not a
day trade under the rule at all.

### Running continuously

The terminal is built to be left on. That takes more than not crashing:

- **The trading day rolls over.** The daily-loss reference used to be taken
  once at startup and never moved, so by Thursday the "daily" loss was measured
  against Monday's equity — and the halt it applied was permanent. A terminal
  left running stopped trading after its first bad afternoon and never started
  again, while still showing a running session, a live feed and a green health
  score. The day boundary now comes from the venue's clock, and only the
  daily-loss halt clears with it. A halt a human applied survives every
  rollover.
- **A dead trading loop is restarted.** The loop catches everything inside its
  body, so it does not die by raising — it dies by the task ending, or by
  wedging on a request that never returns. Both look healthy from every other
  indicator. A heartbeat distinguishes running from merely existing, and
  restarts are counted and shown rather than hidden.
- **Memory is bounded.** Engines, quotes and allocator states are pruned for
  symbols that fall out of the universe — the sweep sees the whole market every
  quarter hour, so without this a symbol that was briefly interesting keeps a
  ~286KB bar ring forever. The fill journal is a bounded ring with lifetime
  totals kept separately. Nothing holding a position is ever pruned.
- **The machine is asked to stay awake** while a session runs (Windows only,
  system but not display). A book holding an overnight position through a
  suspended laptop never lodges its opening exit. Closing the lid still
  suspends; nothing here overrides that.
- **`Run-IMPERIUM-247.bat`** runs it and restarts it if it exits, backing off
  after repeated immediate failures rather than spinning on a broken binary.
  It does not survive a reboot or closing its window.

### The frame the UI renders is bounded

The cost of a frame is not what the server spends building it — that is
milliseconds — but what the browser must parse and lay out before the next one
arrives. Measured at 900 symbols, the original design sent ~730KB every second
and the browser spent most of the second on it.

| | before | after |
|---|---|---|
| frame at 40 symbols | 62KB | 30KB |
| frame at 900 symbols | ~730KB | 57KB |
| frame at 1,500 symbols | ~1.2MB | 63KB |
| frame → paint | — | 15ms median |

Three bounds: the telemetry rings are deltas against a per-connection cursor;
full reasoning travels only for the symbols closest to trading plus everything
holding a position; and the watchlist streams a ranked window rather than the
universe, saying how many rows it is not showing. Every symbol below the line
is still evaluated and still counted in the census.

### The universe is scanned, not hardcoded

The seed list is a starting point. On start the session asks the venue what it
lists, drops anything not tradable *right now*, and ranks the rest by traded
value. Anything currently held is kept regardless of rank, because dropping a
symbol that holds a position leaves the position with nothing managing it.

The sweep covers the **whole tradable listing**, not a shortlist — batched at
200 symbols per request, because the symbol list travels in the query string
and one request for the whole market is a URL that is refused before it reaches
the venue. That refusal would have surfaced as "the scanner found nothing",
which reads as a quiet market rather than a broken request.

Scanning everything and *trading* everything are different things. The top 150
by turnover carry an engine and a bar ring; the rest are ranked and dropped.
The bound is memory, measured rather than guessed: a full ring costs ~286KB, so
150 symbols is ~45MB of bar history and 1,500 would be 430MB on a laptop that
is also running a browser. The scan runs on a quarter-hour cycle (~50 requests
against a budget of 200/minute); the per-minute loop re-prices only what is
traded, which is one request.

### Every threshold is measured

`imperium calibrate` generates nulls, measures the estimators against them, and
writes `src/imperium/strategy/null_calibration.json`. That file is the only
source of thresholds; the classifier raises rather than falling back to textbook
constants, because a silent fallback would produce something that fires on a
large share of pure noise while appearing to work.

What the measurement found at a 250-bar window (seed 20240517, 1500 trials):

| | measured | the textbook value |
|---|---|---|
| variance-ratio z, 2.5–97.5% | `-1.68` … `+2.47` | `±1.96` (symmetric) |
| R/S Hurst null median | `0.605` | `0.5` |
| ADF critical value | `-2.82` | table lookup for a different specification |

Held-out generators that were used to fit nothing — GARCH(1,1), t(4) tails, high
and low volatility walks — give false-regime rates of **4.2%–5.3%** against a
nominal 5%.

The classifier is deliberately blind to pure drift: a random walk with drift has
independent increments, so its variance ratio is 1 by construction. Detecting it
at the false-positive rate is the correct result, and the calibration output says
so rather than tuning it away.

### One cost gate

`execution/costs.py` is the only implementation of round-trip cost, and
`tests/test_costs.py` greps the source tree for a second one. A round trip is two
crossings, so two half-spreads — which is **one full spread**, not two. A maker
pays adverse selection instead. An assumed fee tier is reported as a *warning*,
because an assumption nobody is told about becomes a fact by default.

That grep test has already earned its place: it caught the paper broker growing
its own half-spread constant during development.

### Security, encoded rather than documented

- The server **validates** its bind address; `0.0.0.0` raises. This process holds
  API keys and has no authentication.
- Credentials live in `~/.imperium/credentials.json`, owner-only (`0600`, or an
  icacls ACL on Windows), and the permissions are **verified on load** — the
  interesting case is a file created correctly and later copied or restored.
- A credential never leaves the process. The connections endpoint returns masked
  views, and the secret is not a field on them.
- Redaction is mechanical: a logging filter scrubs every record, including
  third-party loggers and exception arguments.
- A stored key is `trade_enabled=False`. Going live needs **both** a key marked
  tradeable and the phrase `GO LIVE` typed exactly.
- **The UI never places an order.** A test greps every route to keep it that way.

### The UI answers one question

A bot that is working and a bot that has silently stopped look identical, and
"nothing trading" is the normal case — a scanner refusing everything it sees is
behaving correctly. So:

- **Two telemetry streams, never merged.** Tens of events per hour a human should
  read; several pulses per second of work actually done. A shared ring would lose
  every readable event within a minute.
- **"Warming up" is never shown as "seeing no opportunity."** One resolves
  itself; the other is a decision.
- **Four watchlist verdicts**, and the middle two differ: `not admitted` means
  the concurrency limit is full, which is not a fault of the symbol.
- The reasoning panel gives, per symbol, expected edge against round-trip cost —
  the answer to *"it says TRADING, so why is there no position"*.

---

## Layout

```
src/imperium/
  cli.py                    keys / balance / diagnose / serve / calibrate
  config.py                 paths and invariants
  logging_setup.py          redaction that cannot be bypassed
  session.py                the object the server exposes
  security/credentials.py   the credential store
  venues/
    registry.py             what differs between venues, as data
    binance/                client, errors, filters, feed
  diagnostics/layers.py     six layers, first failure wins
  strategy/
    statistics.py           variance ratio, Hurst, ADF
    calibration.py          measures the thresholds
    regime.py               the classifier
    signals.py              momentum, mean reversion, the blend
  execution/
    costs.py                the one cost gate
    sizing.py  risk.py  portfolio.py  engine.py  broker.py  bars.py
  telemetry/streams.py      the two rings
  server/app.py + static/   the terminal
```

---

## Testing

```bash
uv run pytest tests/ -q
```

Every test states in its docstring **which specific failure it prevents**, and
each was verified by reverting the fix it covers and confirming it fails. Two
were found to be passing for the wrong reason during development and were fixed:
the portfolio-clamp test (sizing's own cap was already tighter than the budget,
so the clamp had nothing to do) and the no-retry-on-reject test (two independent
guards, either sufficient alone).

The front-end invariants that a screenshot cannot show — orb dedupe, bounded
replay, a bloom-measured budget, jittered speed and depth — run under Node from
`tests/test_cluster_js.py`.

## Running the Windows build

**Download `IMPERIUM-windows.zip`** from the
[latest release](https://github.com/quincytrader12/Imperium/releases/tag/windows-latest-build).

1. Right-click the downloaded zip → **Properties** → tick **Unblock** → OK.
   Windows marks anything downloaded from the internet, and that mark is what
   makes SmartScreen refuse to run what is inside.
2. **Extract it** anywhere — the Desktop is fine. Do not run it from inside the
   zip; Windows runs that from a temporary folder and it will not work properly.
3. Double-click **Start-IMPERIUM.bat**.

Your browser opens at <http://127.0.0.1:8787/>. The console window shows the URL
and stays open; close it or press Ctrl+C to stop.

### If it does not start

The folder contains **`imperium-startup.log`** after any attempt, successful or
not. That file says what happened.

By far the most common cause is **Microsoft Defender deleting the file** —
unsigned PyInstaller executables are a frequent false positive. Check Windows
Security → Virus & threat protection → **Protection history**. If it is there,
restore it and add the folder as an exclusion.

Use `Start-IMPERIUM.bat` rather than the `.exe` directly: if the program exits
for any reason, the batch file keeps the window open so you can read why. A
double-clicked `.exe` closes its own console and takes the message with it.

### Why a folder rather than one file

A single-file build unpacks its whole payload into `%TEMP%` on every launch.
Defender scans that unpack each time, which is slow and is the usual reason the
file gets quarantined — "extract a pile of DLLs to a temp folder and run them"
is also what malware does. The folder build runs in place and starts
immediately. `IMPERIUM.exe` is still published for anyone who wants one file.

## Options

`IMPERIUM.exe` is a single self-contained file — no Python install, no
dependencies, nothing to unpack. Double-click it, or run it from a terminal:

```
IMPERIUM.exe                  start on http://127.0.0.1:8787/ and open a browser
IMPERIUM.exe --port 9000      serve on port 9000 instead
IMPERIUM.exe --no-browser     start without opening a browser
IMPERIUM.exe --version
IMPERIUM.exe --help
```

It prints the URL it is serving on:

```
==============================================================
  IMPERIUM — built by Quincy Gininda
==============================================================
  Open:        http://127.0.0.1:8787/
  Diagnostics: http://127.0.0.1:8787/diagnose
  Bound to 127.0.0.1 only — not reachable from your network.
  Press Ctrl+C to stop.
==============================================================
```

If the port is already taken — a second copy, or something else on 8787 — it
uses the next free one and says so, rather than dying with a bare
`WinError 10048` before printing anything.

**It binds 127.0.0.1 only, and there is no flag to change that.** The process
holds API keys and has no authentication of its own, so the bind address is a
rule rather than an option. Reaching it from another machine would mean putting
your own authenticated proxy in front of it deliberately.

Nothing needs to be configured before the first run: open it, and the watchlist
and scanner work with no API key at all. Add a key only when you want balances
or trading, and note that a stored key cannot trade until you separately enable
it and type the confirmation phrase.

On first run Windows SmartScreen will warn about an unsigned executable — the
binary is not code-signed. "More info" → "Run anyway", or check the SHA-256
against the one published with the build.

## Packaging

PyInstaller does not cross-compile, so the Windows `.exe` is built on a Windows
runner (`.github/workflows/build.yml`). The workflow runs the full suite on both
Linux and Windows first, then builds, then runs `packaging/verify_build.py`
against the produced binary — which launches it and requires that it serves its
own page, its assets, its API and its diagnostics. A build that silently loses
`--add-data` starts, serves the API, and 404s its own page; PyInstaller calls
that a success, so the binary itself is checked.

## Known limits

- **The live network path is unverified from the development sandbox.** Alpaca
  is blocked there by a TLS-intercepting proxy, which `imperium diagnose`
  correctly identifies. Every authenticated path is tested against a mock venue
  that recomputes the HMAC over the exact query string it receives, so signing is
  genuinely exercised — but no order has been sent to the real venue.
- The adverse-selection fraction (0.35 of the half spread) is a stated
  assumption, not a measurement.
- Momentum's expected edge assumes half the current EMA separation persists.
  Also an assumption, and deliberately conservative.
