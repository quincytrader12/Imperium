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

### The live book is reconciled against the venue

Everything the live broker records is optimistic. An order is booked when the
venue *accepts* it, because that is the only moment a market order yields a
number — and several things happen afterwards that the book never hears about:

- an accepted order rejected later (buying power, locate failure, halt,
  wash-trade block);
- a partial fill whose remainder is cancelled at the close;
- **every** market-on-close and market-on-open order, accepted now and filled
  at an auction hours later at a price nobody yet knows;
- a trade made by hand, or by something else, in the same account.

`sync()` used to run exactly once, on the switch into live. After that the book
drifted from reality with nothing to correct it, and every decision was sized
against a fiction while the terminal reported a position it believed in
completely.

The venue's positions are now read every 30 seconds and the book corrected to
them. A difference is reported at error level rather than quietly absorbed — it
means an order did not do what this program was told it did, which is the most
important thing an operator can be shown. Symbols with an order still open are
skipped: a market-on-close order holds no position until the auction, and
"correcting" that would flatten the book and immediately re-submit it.

### The data plan's subscription cap

The scan ranks the whole market and 150 symbols carry an engine, but the
websocket cannot subscribe to nearly that many. Alpaca's basic plan limits
concurrent subscriptions — 30 is the published figure for crypto trade and
quote channels and the commonly reported cap for the basic stock stream.

Two properties make that dangerous rather than merely restrictive:

- an over-limit request is rejected **whole** (error 405, previous
  subscriptions untouched), which on a fresh connection means *no*
  subscriptions at all;
- the rejection arrives as a message with no symbol attached, and the message
  handler filtered on symbols — so the one message explaining the silence was
  the one message discarded.

Together: a connected socket delivering nothing, reading as a quiet market. The
cap is now applied before subscribing, negotiated down by halving on a 405
(the true limit depends on a data subscription this program cannot read), and
floored so it never converges on zero.

**Held positions are first in the queue for a stream.** A carried position is
the one symbol where a stale price means a stop that does not fire and an exit
sized on a number from minutes ago. Symbols below the cap are still scanned,
still priced by the snapshot sweep every minute, and still tradeable by the
daily-bar strategies — what they lose is the intraday path, which cannot work
on a minute-old price anyway.

### The no-trade band

A strategy that re-targets exactly will trade on every evaluation, because
equity moves with every fill and every price tick and the delta is therefore
never quite zero. Measured before this existed: **120 consecutive bars produced
120 orders** on a target weight that never changed once, median size 0.06 of a
share. Across a 150-symbol universe that is 150 orders a minute into a venue
that rate-limits them.

Under proportional transaction costs the optimal policy is not to track a
target but to do nothing inside a region around it (Constantinides 1986; Davis
& Norman 1990). A position is corrected only once it has drifted more than 10%
from its target — after which the same 120 bars produce **one** order.

The band is a fraction of the target, strictly under one, and that is what
makes it safe rather than the guards that read as if they do. A full exit's
delta *is* the whole position and a first entry's delta *is* the whole target,
so neither can ever be smaller than a fraction of itself. A cap that can trap a
position would be worse than the churn it prevents, so this is the invariant
the tests defend.

### Three horizons, and which one a small account can actually use

| strategy | horizon | order type | day trade? | where it works |
|---|---|---|---|---|
| intraday blend | minutes | market | **yes** | any account with day trades left |
| multi-day trend | days–weeks | market (fractional) | no | **any account, and the only one under the PDT floor** |
| overnight drift | one night | MOC / MOO (whole shares) | no | equities priced under the position cap |

One symbol is owned by exactly one strategy at a time. Blending them would
allocate the same capital twice, so the choice is made once, explicitly, and
shown on the decision.

**Why a $70 account needs the middle row.** Under FINRA's pattern-day-trader
rule a margin account below $25,000 may make three day trades in five business
days. An intraday strategy there is not constrained, it is *prevented*: a
position it cannot close the same day is not an intraday position, it is an
accidental overnight hold with an intraday stop behind it. A trade carried
across a session close is not a day trade at all, so the multi-day horizon
removes the binding constraint rather than working around it. Crypto sits
outside the rule entirely, which is why it keeps the intraday path at any
balance.

(FINRA has approved removing the $25,000 threshold, effective June 2026, with
firms given until October 2027 to implement. Nothing here assumes either state
— the day-trade count and the PDT flag are read from the venue every tick, so
the program follows whatever the broker actually enforces.)

**The arithmetic.** A round trip is paid once per holding period, so cost per
unit of time falls as the period grows. With drift `μ` bp/day and round trip
`C` bp, a position must be held

    H* = k · C / μ    days

before the edge has covered the cost, `k` being the same safety multiple every
other strategy is judged against. Crypto pays ~50bp and needs ~4 days at a
20bp/day drift; a US equity pays 2–4bp and needs about one. **H\* is floored at
one session** — a shorter plan would be a day trade, which is the one thing
this strategy exists not to be.

Entering and staying are judged differently on purpose: entering must justify
the whole round trip, staying only has to justify itself, because the entry
cost is already spent and closing early throws it away without collecting the
edge it bought.

**The evidence.** Moskowitz, Ooi & Pedersen (JFE 2012) found positive
time-series momentum in *every one* of 58 futures contracts, 52 significant at
5%. Liu & Tsyvinski (RFS 2021) found strong crypto time-series momentum at one-
to four-week horizons — a one-SD week predicting **+3.16%** the next week for
Bitcoin. The lookbacks differ by asset class for that reason: 21/63/126 days
for equities, 7/14/28 for crypto.

**What that evidence does not give a small account.** The headline Sharpe near
1.0 in Moskowitz et al. is a *diversified* portfolio of 58 markets. Single-
instrument time-series momentum is far weaker, and an account carrying two
positions receives almost none of that diversification. The premium is
therefore estimated **pooled across the universe**, never per symbol — measured
here, 4 of 40 symbols cross |t| > 2 on data with a true premium of exactly
zero, so a strategy that picked its best-looking symbol would be selecting on
noise every time.

### Options: affordable long before they are sensible

An earlier version of this program asserted a $0.65 per-contract commission and
concluded options were unusable. **That premise was wrong.** Alpaca is
commission-free on options; what remains is regulatory pass-through of about
**$0.09 per contract round trip** — OCC clearing, ORF, and TAF on the sell.
Fees are not the obstacle.

Two other things are, and both are structural:

1. **A contract is 100 shares**, so position size is quantised. The smallest
   possible trade is 100× the premium and cannot be reduced.
2. **The spread is the whole cost**, and it is worst exactly where a small
   account is forced to shop.

| account | premium reachable | round-trip spread | breakeven move |
|---|---|---|---|
| $70 | $0.28 | ~15% | **23%** |
| $250 | $0.50 | ~8% | 12% |
| $500 | $1.00 | ~5% | 8% |
| $1,500+ | $3.00 | ~2–3% | 4% |

So options become *affordable* around $1,500 — and are still not traded, for a
reason that does not go away with money: the strategies here forecast drift in
the underlying, and an option pays theta for calendar time whether or not that
drift arrives. Trading them needs a model of implied volatility, not a
different threshold on this one. `scripts/option_affordability.py` computes the
table; the terminal shows the reachable premium against the threshold.

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

### The process orb

The centre panel is one object that shows what the terminal is doing. It
replaced a 2D pulse field, and it had to keep everything that field carried,
because an operator glancing at it was reading three things at once:

| What you read | How the orb says it |
| --- | --- |
| what kind of work is happening | the core's colour zones, one per process kind |
| which symbol just fired | a ripple from that symbol's own point on the surface |
| how hard it is working | the morph state, the heartbeat rate, the bloom |

The third is why it is a body and not a chart. A number telling you the scan
rate is something you have to read; a thing that breathes faster is something
you notice while looking at something else, and this panel is looked at out of
the corner of an eye for hours.

**Three layers over one shared displacement field** (`orb.shaders.js`): a
glossy liquid core, two frosted membranes that refract it, and a point-cloud
skin just outside. They deform together because they evaluate the same
function, not because their animations are kept in step.

**Four morph states**, blended over 1.5s and never snapped: `idle` is nearly a
sphere, breathing; `active` wobbles; `intense` grows lumpy, higher-frequency
lobes; `alert` melts, the lower hemisphere drawn down with a gravity falloff.
The state is chosen from total activity with hysteresis so a rate sitting on a
boundary cannot flicker between two bodies, and `setMode` overrides it.

**Nothing here is a hardcoded colour.** Every process colour comes from
`palette.js`, which is also what paints the legend and what the server
validates its pulse kinds against. A test asserts the three agree.

**Three.js is vendored, not loaded from a CDN.** This ships as an offline
Windows executable; a CDN tag would leave the panel dead on a machine with no
internet. The vendored copy lives in `static/vendor/` and is bundled with the
rest of `static/`. A bare `three` specifier resolves through an import map in
the page, which keeps the project's no-build-step rule.

**It measures itself.** Four quality tiers, from a 64-detail icosahedron with
14k skin points down to a 20-detail body with none. Sustained slow frames drop
a tier, a long clean run earns one back — quick to cut, slow to relax, because
a panel that recovers eagerly oscillates and the flicker is worse than simply
running lower. The tier is named in the panel caption when it is not the top
one, so a body that has quietly gone simpler says so.

#### Testing it without live data

`?debug=orb` in the URL, or Ctrl+Shift+O, opens a panel with a slider per
process, a mode selector, and a button that fires ten ripples. It loads
lil-gui on demand, so an operator who never opens it never downloads it.

#### Adding a process type

Three edits, and a test will tell you if you miss one:

1. **`imperium/telemetry/streams.py`** — add the name to `PULSE_KINDS`. The
   server refuses to emit a pulse of an unknown kind, so this is what makes it
   exist at all.
2. **`static/palette.js`** — add it to `KIND_COLOR` with an RGB triplet, to
   `KIND_LABEL` with what it should be called out loud, and to `KIND_ORDER`
   where it belongs on the scale from "background hum" to "something is
   wrong". The legend and the orb both read this; there is no second list.
3. **`static/orb.math.js`** — add it to `FULL_RATE`: how many of these per
   second counts as an activity of 1.0. This matters more than it looks.
   Scans run at tens per second and orders at a handful per day, so a shared
   denominator would make everything except scanning invisible.

If the new kind means something has gone wrong, add it to `ALERT_KINDS` in
`static/orb.boot.js` and it will put the orb into its melting state.

`MAX_PROCESSES` in `orb.js` is a GLSL array bound, compiled into the shaders
as a `#define`. It is 8; past that, raise it there and nowhere else.

### Pressing Start

It greets you by name and reads a trading quote — out loud if a voice is
connected, and on screen either way, because the voice is optional and this
should not be. The line fades after fourteen seconds: a quote still sitting on
the header an hour later is furniture.

The mode is named as part of it. "Trading live" and "Dry run, no orders will
be placed" are one glance apart in the header and a completely different fact
about the next hour, and Start is the moment that distinction matters most.

**The quotes are attributed, and correctly.** Trading quotations circulate in a
state of near-total attribution collapse — the most famous line in the file is
given to Keynes almost everywhere and he never wrote it; it is A. Gary
Shilling's, and the terminal says so when it reads it. A program whose whole
claim is that it reports what it measured cannot open by passing on something
it did not check. A test asserts every quote names who said it.

It never says the same quote twice in a row. A quote coming round again next
week is a rotation; the same one twice running is a program that is not really
choosing.

The voice is given the name as written, which was checked against a speech
synthesiser rather than assumed: *Gininda* comes out as /dʒɪ.ˈnɪn.də/, which is
right. Not every name is so lucky, so `IMPERIUM_OPERATOR_SPOKEN` hands the
engine a different spelling while the screen keeps the real one — and if you
ever need it, keep each syllable sayable. A consonant run an engine cannot
pronounce makes it give up on the word and read the letters instead; *ndhha*
came back as "EN-DEE-AITCH-AITCH-AY", which is worse than any mispronunciation.
A phonetic respelling rather than SSML, because a `<phoneme>` tag is honoured
by some ElevenLabs models and ignored silently by others.

Set `IMPERIUM_OPERATOR` in `settings.txt` to change the name.

### Asking it out loud

Press **Ask**, say a question, and it answers in the same voice as the
briefing. What it heard and what it said appear as text under the header too,
because a misheard question is otherwise invisible — you would hear a confident
answer to a question you did not ask with no way to tell.

Questions it knows:

| Ask | It tells you |
| --- | --- |
| *what's looking good* | What is admitted, or the closest miss and what stopped it |
| *what have you found so far* | Scans, decisions, refusals and orders, counted |
| *why aren't you trading* | The blockers panel, spoken |
| *what do I own* | Open positions and what they are worth together |
| *how much money do I have* | The balance, in dollars and in your second currency |
| *how am I doing today* | The day, against the day's loss budget |
| *what's the news saying* | Stories scored, and which symbols lead |
| *what about the ETFs* | The Sector Trend sleeve, or what would switch it on |
| *are you healthy* | The health score and what is dragging it down |
| *are you live or paper* | The question with the most expensive wrong answer |
| *what are your limits* | The bounds it trades inside |
| *what is it costing to trade* | The cost gate everything has to clear |
| *why aren't you trading SPY* | That symbol's actual decision |
| *what can I ask* | This list, spoken |

**There is no language model in this, and that is the design.** Every answer is
assembled from fields in the same snapshot the panels render, so a spoken
answer cannot say anything the screen does not. An answer generated from a
prompt would be a fluent sentence with no measurement behind it, and the least
acceptable place for that is a confident voice telling you how your money is
doing. When the snapshot does not contain the answer, the answer is that it
does not — and a question it did not understand is refused rather than guessed
at, because a wrong answer and a right one sound identical.

**Your words do not leave the machine.** The question is answered locally and
never sent anywhere; only the sentence it wrote is passed to the speech
service, and only if you have connected one. Recognition itself is the
browser's — on Chrome and Edge that means the browser sends the audio to its
own service to transcribe, which is Chrome's behaviour and not this program's.
The **Ask** button is absent in browsers without speech recognition, rather
than present and broken.

---

## Sector Trend (a sleeve, off by default)

Donchian/Keltner breakouts on nineteen liquid SPDR industry ETFs, after
Zarattini and Antonacci, *A Century of Profitable Industry Trends*.

**What it does.** Each day it looks at adjusted daily closes. A flat symbol
whose close clears *yesterday's* upper band is bought; a held symbol whose
close falls below the stop it carried in from yesterday is sold. Positions are
sized by volatility — `w = (target_vol / N) / sigma` — where `N` is the size of
the whole universe, not the number of positions, so a thin signal stays a small
book. The trailing stop never moves down.

**Why it is a sleeve rather than a strategy.** It trades a fixed slice of
equity and keeps its own ledger of quantities and stops, separate from the
broker's book. Alpaca nets positions by symbol across the account: if the
multi-day trend strategy is long 3 XLK and this sleeve enters 2 more, the venue
reports 5 and nothing in that number says who owns what. A sleeve that sized or
exited from the account position would sell another strategy's shares to close
its own. The ledger is at `~/.imperium/state.json` under `sector_trend`, and
every order carries a `sectrend-` client order id.

**It is off until you turn it on.** `SECTOR_TREND_ENABLED` defaults to `false`.
Run the backtest first.

### Where to change any of this

`%USERPROFILE%\.imperium\settings.txt` on Windows, `~/.imperium/settings.txt`
elsewhere — the same folder as your API keys. IMPERIUM creates it on first run
with every option listed and commented out, so it is already there when you go
looking. Uncomment a line, save, restart.

```ini
# SECTOR_TREND_ENABLED=false     <- delete the # and set it to true
```

A real environment variable, if you set one, wins over the file. The file can
only set the options below: a line naming anything else is ignored and logged,
so a typo costs a setting rather than being mistaken for one that worked.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SECTOR_TREND_ENABLED` | `false` | The kill switch. Nothing runs while this is false. |
| `SECTOR_TREND_ALLOCATION` | `0.20` | Fraction of account equity the sleeve may use. |
| `SECTOR_TREND_UNIVERSE` | 19 SPDR ETFs | Comma separated. |
| `SECTOR_TREND_TARGET_VOL` | `0.015` | Daily volatility target for the sleeve. |
| `SECTOR_TREND_MAX_LEVERAGE` | `1.0` | `2.0` is the paper's figure; it does not model the margin interest Alpaca would charge. |
| `SECTOR_TREND_REBALANCE_THRESHOLD` | `0.25` | Held positions are only resized past this drift. Entries and exits always execute. |
| `SECTOR_TREND_EXEC_MODE` | `near_close` | Or `next_open`. |
| `SECTOR_TREND_RUN_TIME_ET` | `15:45` | |
| `SECTOR_TREND_ARM_AT_EQUITY` | `200` | Equity at which the sleeve switches itself on. `0` never. |
| `IMPERIUM_SECONDARY_CURRENCY` | `ZAR` | A second currency beside the dollar balance. Display only. |
| `IMPERIUM_FX_RATE` | *(fetched)* | Pin the rate by hand instead of fetching it. |
| `IMPERIUM_OPERATOR` | `Mr Gininda` | What it calls you when you press Start. |
| `IMPERIUM_OPERATOR_SPOKEN` | *(the name)* | Only if a voice mispronounces it. |

### Arming it

The Sector trend panel carries a gauge and a button. The gauge reads in the
account's own money — *$143.20 of $200, $56.80 to go* — because "72%" is a
number you have to do arithmetic on and a balance is one you can act on. The
button arms it now; once armed, the same place says how it armed (by hand, by
equity, or from `settings.txt`) and offers **Disarm**.

**Disarm is refused while it holds anything**, and that is the whole reason
arming used to be one-way. A sleeve switched off mid-book leaves its ETFs
sitting there with nobody trailing their stops and nothing left to close them,
which is worse than either state. Flat, there is nothing to abandon, so the
switch is yours. A button also cannot overrule `SECTOR_TREND_ENABLED=true` —
somebody who wrote that in a file meant it.

**Arming below the threshold asks once.** It is allowed — your money, your
call — but the refusal carries the arithmetic rather than a vague warning.
Sizing is `w = (target_vol / N) / sigma`, so a position is worth
`sleeve × target_vol / (N × sigma)` and clears Alpaca's $1 floor only while
`sigma ≤ sleeve × target_vol / N`. At $143 that ceiling is 2.26% a day; at
$200 it is 3.16%, which clears every ETF in the universe. The names priced out
first are the most volatile — which is where the strategy's return comes from
— so an undersized sleeve is not a smaller version of this strategy, it is the
calm half of it, with no backtest behind it.

### Arming itself

With `SECTOR_TREND_ARM_AT_EQUITY` set, the sleeve switches itself on the first
time account equity reaches that figure, announces it everywhere it can reach,
and takes its slice from the engine's share on the same tick.

$200 is not a round number chosen for looks. At the default 0.20 allocation it
gives the sleeve $40, which is where *every* ETF in the universe clears
Alpaca's $1 minimum order — including the most volatile. Below about $130 the
sleeve would quietly trade only the calm half of its universe, which is a
different strategy from the one the paper describes.

It arms once and **never disarms**. A sleeve that switched itself off on a dip
would abandon whatever it was holding: those positions would sit there with
their stops no longer being trailed and nothing left to close them, which is
worse than either state on its own. Turning it off is a decision, and
`settings.txt` is where you make it.

### Running the backtest

From the Windows build, where most people have one — double-click
**Backtest-Sector-Trend.bat**, or from a command prompt in that folder:

```
IMPERIUM.exe --backtest                        from the stored Alpaca key
IMPERIUM.exe --backtest --start 2010-01-01     a shorter history
IMPERIUM.exe --backtest --csv .\bars           date,close CSVs instead
```

From a checkout:

```bash
uv run python scripts/backtest_sector_trend.py                 # from Alpaca
uv run python scripts/backtest_sector_trend.py --start 2005-01-01
uv run python scripts/backtest_sector_trend.py --csv ./bars    # date,close CSVs
```

Both run the same code — `imperium.strategy.backtest_cli` — so the two cannot
drift apart and give different answers. It places no orders and arms nothing,
and it reads `settings.txt`, so it measures the universe and volatility target
the sleeve would actually trade rather than the defaults.

It runs both execution modes at both leverage caps, charges 5bp of slippage per
side and 7% annual margin interest on exposure above 100%, and reports CAGR,
volatility, Sharpe, Sortino, max drawdown, beta and alpha against SPY, trade
count, average holding days and yearly returns.

**Read the warnings at the bottom.** The paper reports roughly 7.7% CAGR at a
Sharpe near 0.6 with a 24% drawdown over 2005–2024 on a wider universe. A
result far better than that is much more likely to be a bug — lookahead, an
unadjusted price series — than an edge, and the report says so rather than
leaving it to a reader who wants the number to be good.

### Enabling it on paper

```bash
export SECTOR_TREND_ENABLED=true
```

Then start the terminal as usual. The **Sector trend** panel in the left rail
shows the sleeve's equity, how many ETFs it holds, its gross weight, and each
position's stop.

### One thing to check before you enable it

Alpaca refuses a fractional buy below **$1.00 notional**. The sleeve's own
arithmetic can produce targets under that on a small account:

| Account | Allocation | Sleeve | Result |
| --- | --- | --- | --- |
| $70 | 0.20 | $14.00 | about half the universe sizes under $1 and is skipped |
| $70 | 0.36 | $25.20 | every symbol clears the floor |
| $127 | 0.20 | $25.40 | every symbol clears the floor |

The sleeve checks this before sending anything and names the shortfall in the
panel rather than letting you watch a run place nothing. The exact figure
depends on the least volatile ETF in the universe on the day, because the
smallest weight is the binding one.

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

The front-end invariants that a screenshot cannot show — frame-rate independent
damping, mode hysteresis, the particle distribution, symbol placement, pulse
dedupe and bounded replay — run under Node from `tests/test_orb_js.py`. The ones
that need a GPU — that every shader compiles, that each mode actually deforms
the mesh, that disposal releases the context — render in a real browser from
`tests/test_orb_browser.py`, and are skipped where no browser is installed.
That file takes about three and a half minutes; `-k "not orb_browser"` skips it.

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
