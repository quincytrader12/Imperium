# GODALGO

A self-contained algorithmic crypto trading terminal for **Binance Spot**, with a
live instrument-panel UI, packaged as a double-clickable Windows executable.

Python 3.11+, `uv` for dependencies, FastAPI and a websocket for the UI, plain
HTML/CSS/canvas for the front end. No framework, no build step.

---

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"

# 1. Can this machine even reach the venue? Six layers, first failure wins.
uv run godalgo diagnose

# 2. Store a key. It cannot trade until you separately enable it.
uv run godalgo keys add --name main          # prompts without echo
uv run godalgo balance                       # the Phase-1 deliverable

# 3. Measure the regime classifier's thresholds. Nothing is hardcoded.
uv run godalgo calibrate

# 4. Run the terminal.
uv run godalgo serve
```

The terminal is at <http://127.0.0.1:8787/> and the diagnostics at
<http://127.0.0.1:8787/diagnose>.

---

## Design decisions worth knowing about

### No ccxt, and no multi-venue abstraction

`src/godalgo/venues/binance/` is a direct REST client against the documented
HTTP API. A library covering a hundred venues must flatten each venue's error
vocabulary into a common one, and that flattening discards precisely what an
operator needs: whether `-2015` was the key's IP allow-list, its trading
permission, or the key itself. Every documented code is translated into a
remedy in `errors.py`.

The seam for a second venue is `venues/registry.py`, which holds what differs
between venues **as data**. A second venue would be a new entry there — and a
separate pool of money, with its own book, equity and risk limits, because two
accounts cannot fund each other.

### Every threshold is measured

`godalgo calibrate` generates nulls, measures the estimators against them, and
writes `src/godalgo/strategy/null_calibration.json`. That file is the only
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
- Credentials live in `~/.godalgo/credentials.json`, owner-only (`0600`, or an
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
src/godalgo/
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

`GODALGO.exe` is a single self-contained file — no Python install, no
dependencies, nothing to unpack. Double-click it, or run it from a terminal:

```
GODALGO.exe                  start on http://127.0.0.1:8787/ and open a browser
GODALGO.exe --port 9000      serve on port 9000 instead
GODALGO.exe --no-browser     start without opening a browser
GODALGO.exe --version
GODALGO.exe --help
```

It prints the URL it is serving on:

```
==============================================================
  GODALGO — built by Quincy Gininda
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

- **The live network path is unverified from the development sandbox.** Binance
  is blocked there by a TLS-intercepting proxy, which `godalgo diagnose`
  correctly identifies. Every authenticated path is tested against a mock venue
  that recomputes the HMAC over the exact query string it receives, so signing is
  genuinely exercised — but no order has been sent to the real venue.
- The adverse-selection fraction (0.35 of the half spread) is a stated
  assumption, not a measurement.
- Momentum's expected edge assumes half the current EMA separation persists.
  Also an assumption, and deliberately conservative.
