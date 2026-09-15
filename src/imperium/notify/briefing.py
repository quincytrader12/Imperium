"""What IMPERIUM says out loud.

This turns a snapshot into a script meant to be *heard*, which is a different
job from the panels it is built from. Three rules follow from that and shape
almost every line here.

**A listener cannot skim.** The screen can afford a hundred and fifty rows
because the eye jumps to the one that matters. Speech is linear and at the
listener's mercy, so this says the few things that change what an operator
would do -- is it running, is it armed, what does it hold, what is stopping it
-- and stops. A spoken read-out of the whole watchlist is not thorough, it is
unlistenable.

**Symbols do not survive being spoken.** "$70.00" read literally is "dollar
seventy point zero zero"; "4.12bp" is "four point one two bee pee"; "AAPL" is
"aapple" to some voices and "A A P L" to others. Everything numeric is written
out as words in the order a person would say them, and tickers are spaced so
they are spelled rather than guessed at.

**Silence is a sentence.** A book that is holding nothing and refusing
everything is the normal state of this program most of the time, and the
briefing has to say so plainly rather than trailing off. "Nothing is being
traded" followed by the reason is the most useful thing it can say, and it is
what it will say most days.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

#: How many admitted symbols are named before the rest are counted.
#:
#: Five is about as many tickers as a listener retains from one sentence. The
#: rest become "and four others", which is the honest shape: the count matters,
#: the eleventh ticker does not.
MAX_NAMED = 5

#: How many distinct blockers are named.
MAX_BLOCKERS = 3


def greeting(now: dt.datetime | None = None) -> str:
    """Good morning, afternoon or evening, by the operator's own clock."""
    now = now or dt.datetime.now()
    hour = now.hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def say_money(amount: float, currency: str = "USD") -> str:
    """Money as a person would say it.

    "$70.00" read aloud is "dollar seventy point zero zero". Whole amounts drop
    the cents entirely, because "seventy dollars" is what anyone would say and
    "seventy dollars and zero cents" is what only a machine would.
    """
    unit = "dollars" if currency.upper() == "USD" else currency.upper()
    if abs(amount - round(amount)) < 0.005:
        whole = int(round(amount))
        return f"{whole:,} {unit[:-1] if whole == 1 and unit == 'dollars' else unit}"
    return f"{amount:,.2f} {unit}"


def say_ticker(symbol: str) -> str:
    """A ticker a voice will pronounce rather than guess at.

    Letters are spaced so they are spelled out: "AAPL" unspaced is read as a
    word by most voices, and the word is not the company. A crypto pair's
    slash becomes "against", which is how the pair is actually spoken.
    """
    if "/" in symbol:
        base, _, quote = symbol.partition("/")
        return f"{_spell(base)} against {_spell(quote)}"
    return _spell(symbol)


def _spell(text: str) -> str:
    """Letters spaced, digit runs kept whole.

    "AAPL" spaced is spelled out, which is right. "CO007" spaced becomes
    "C O zero zero seven", which is not how anyone says a number -- so runs of
    digits stay together and are read as a figure.
    """
    out: list[str] = []
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
            continue
        if digits:
            out.append(digits)
            digits = ""
        out.append(char)
    if digits:
        out.append(digits)
    return " ".join(out)


def say_list(items: list[str], limit: int = MAX_NAMED) -> str:
    """A spoken list, with the tail counted rather than read."""
    if not items:
        return ""
    named = items[:limit]
    rest = len(items) - len(named)
    if len(named) == 1:
        head = named[0]
    elif len(named) == 2:
        # "X, and Y" is how a list of three or more ends. For exactly two it is
        # a stumble, and this is read aloud.
        head = f"{named[0]} and {named[1]}"
    else:
        head = ", ".join(named[:-1]) + f", and {named[-1]}"
    if rest > 0:
        return f"{head}, and {rest} other{'s' if rest != 1 else ''}"
    return head


def _percent(value: float) -> str:
    return f"{value * 100:.1f} percent"


def admissions(snapshot: dict[str, Any]) -> str:
    """Which symbols hold a concurrency slot, and why they hold it.

    Admission is not a verdict on a symbol's quality -- it is the allocation of
    a fixed number of slots. A symbol already holding a position keeps its slot
    and is never displaced by a higher-scoring newcomer, because churning a
    position to chase a marginally better score pays a full round trip for the
    privilege. The rest go to the best-scoring idle candidates. Saying which of
    those two reasons applies is the whole point of this paragraph.
    """
    watchlist = snapshot.get("watchlist") or []
    limits = snapshot.get("limits") or {}
    slots = limits.get("max_concurrent_positions") or 0

    admitted = [w for w in watchlist if w.get("verdict") == "trading"]
    held = {p.get("symbol") for p in (snapshot.get("positions") or [])
            if p.get("symbol")}

    if not admitted:
        if slots:
            return (f"Nothing holds one of the {slots} position slots right "
                    f"now.")
        return "Nothing has been admitted."

    holding = [say_ticker(w["symbol"]) for w in admitted
               if w.get("symbol") in held]
    fresh = [say_ticker(w["symbol"]) for w in admitted
             if w.get("symbol") not in held]

    parts: list[str] = []
    total = len(admitted)
    parts.append(f"{total} symbol{'s' if total != 1 else ''} "
                 f"{'hold' if total != 1 else 'holds'} a position slot"
                 + (f", out of {slots}" if slots else "") + ".")
    if holding:
        parts.append(f"{say_list(holding)} "
                     f"{'keep' if len(holding) != 1 else 'keeps'} a slot "
                     f"because {'they are' if len(holding) != 1 else 'it is'} "
                     f"already holding a position. An open position is "
                     f"never displaced to chase a better score.")
    if fresh:
        parts.append(f"{say_list(fresh)} "
                     f"{'were' if len(fresh) != 1 else 'was'} admitted on "
                     f"score, as the best of what was eligible and idle.")
    return " ".join(parts)


def say_blocker(label: str) -> str:
    """A blocker label as a clause rather than a column heading.

    The labels are written to be counted in a table -- "costs", "warming up",
    "concurrency slot" -- and read aloud in a list they land as fragments.
    This turns the ones that occur into something that finishes the sentence
    "nothing is being traded, N because ...".
    """
    spoken = {
        "costs": "the edge does not cover the cost of trading",
        "warming up": "they are still warming up",
        "trend not eligible": "the trend does not qualify them",
        "cross-section not eligible": "the crypto ranking does not qualify them",
        "overnight not eligible": "the overnight drift does not qualify them",
        "concurrency slot": "the position slots are full",
        "account too small": "the account is too small to size them",
        "share costs more than the budget": "one share costs more than the "
                                            "per-position budget",
        "no volatility estimate": "their volatility cannot be estimated",
        "not tradable": "the venue does not trade them",
        "not yet scanned": "they have not been reached yet",
        "no signal": "there is no signal in them",
        "sizing": "they size to nothing",
    }.get(label)
    return spoken or label


def _sentence(text: str) -> str:
    """Capitalised and full-stopped, so it can be spoken on its own."""
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def _split_by_class(rows: list[dict[str, Any]]) -> tuple[list, list]:
    """Equities and crypto, told apart by the asset class the snapshot carries.

    By the field rather than by a slash in the symbol: the venue's own
    classification is the one the rest of the program sizes and gates on, and a
    briefing that disagreed with it would describe a different book.
    """
    equities, crypto = [], []
    for row in rows:
        target = crypto if str(row.get("asset_class") or "") == "crypto" \
            else equities
        target.append(row)
    return equities, crypto


def _positions_line(label: str, held: list[dict[str, Any]],
                    currency: str) -> str:
    if not held:
        return f"No {label} positions are open."
    names = say_list([say_ticker(p["symbol"]) for p in held if p.get("symbol")])
    value = sum(abs(float(p.get("value") or 0.0)) for p in held)
    return (f"{len(held)} {label} position{'s' if len(held) != 1 else ''}: "
            f"{names}, worth {say_money(value, currency)}.")


def build(snapshot: dict[str, Any], *, now: dt.datetime | None = None) -> str:
    """The whole spoken briefing.

    The order is the operator's, and it is a running order rather than a
    ranking: is the machine working, what is the news, then each book in turn
    -- equities, crypto, the ETF sleeve -- and finally what the scanner is
    doing right now. It ends on the live activity because that is the part that
    will have changed by the time they ask again.
    """
    lines: list[str] = []
    say = lines.append

    mode = str(snapshot.get("mode") or "dry run").replace("_", " ")
    running = bool(snapshot.get("running"))
    simulated = bool(snapshot.get("simulated", True))
    currency = str((snapshot.get("account") or {}).get("currency") or "USD")
    positions = snapshot.get("positions") or []
    equities, crypto = _split_by_class(positions)

    say(f"{greeting(now)}. This is IMPERIUM.")

    # 1. Terminal health. Everything below is worthless if this part is wrong,
    #    so it is said first and in plain terms.
    if not running:
        say("The session is stopped, so nothing is being scanned or traded.")
    else:
        detail = ("Orders are simulated in this process and never reach the "
                  "venue." if simulated else
                  "Orders are really sent to the venue.")
        say(f"Running in {mode} mode. {detail}")

    feed = snapshot.get("feed") or {}
    if running and feed.get("connected") is False:
        reason = str(feed.get("reason") or "").strip()
        say(_sentence(reason) if reason
            else "The market data feed is not connected.")
    elif running:
        streamed = feed.get("streamed") or 0
        if streamed:
            say(f"The data feed is live on {streamed} symbols.")

    equity = float(snapshot.get("equity") or 0.0)
    if equity:
        say(f"Equity is {say_money(equity, currency)}.")
    unrealised = float(snapshot.get("unrealised_pnl") or 0.0)
    realised = float(snapshot.get("realised_pnl") or 0.0)
    if abs(unrealised) >= 0.01:
        way = "up" if unrealised > 0 else "down"
        say(f"Unrealised profit and loss is {way} "
            f"{say_money(abs(unrealised), currency)}.")
    if abs(realised) >= 0.01:
        way = "made" if realised > 0 else "lost"
        say(f"Realised, this session has {way} "
            f"{say_money(abs(realised), currency)}.")
    if snapshot.get("halted"):
        say(f"The book is halted. {snapshot.get('halt_reason') or ''} "
            f"Exits still pass, but no new exposure is opened.")

    # 2. News sentiment.
    news = snapshot.get("news") or {}
    if not news.get("enabled"):
        say("The news factor is switched off.")
    elif news.get("broken") or news.get("last_error"):
        say("Headlines could not be read, so nothing is being sized on them.")
    elif news.get("covered"):
        leaders = [l for l in (news.get("leaders") or [])
                   if abs(float(l.get("score") or 0)) >= 0.2]
        if leaders:
            top = leaders[0]
            way = "positive" if float(top["score"]) > 0 else "negative"
            say(f"News: {news.get('covered')} symbols have headlines, and "
                f"{say_ticker(str(top['symbol']))} reads most strongly {way}. "
                f"Headlines adjust position size by up to "
                f"{news.get('max_tilt_pct', 0)} percent and never decide "
                f"whether something is traded.")
        else:
            say(f"News: {news.get('covered')} symbols have headlines, none "
                f"strongly one way or the other.")
    else:
        say("There is no news on anything being watched.")

    # 3. Equities.
    market = snapshot.get("market") or {}
    say("Equities. " + ("The market is open."
                        if market.get("is_open") else "The market is closed."))
    say(_positions_line("equity", equities, currency))

    # 4. Crypto, which trades around the clock and is unaffected by the above.
    say("Crypto, which trades around the clock.")
    say(_positions_line("crypto", crypto, currency))
    cross = snapshot.get("cross_section") or {}
    if cross.get("credible"):
        leaders = [say_ticker(l["symbol"]) for l in (cross.get("leaders") or [])]
        if leaders:
            say(f"On the crypto ranking, the strongest names are "
                f"{say_list(leaders, 3)}.")
    elif cross.get("cohort"):
        say(f"The crypto ranking is not measurable yet. "
            f"{cross.get('explain') or ''}")

    # 5. The ETF sleeve, which is a separate book on its own capital.
    sector = snapshot.get("sector") or {}
    if not sector.get("enabled"):
        say("The sector ETF sleeve is switched off.")
    else:
        holding = sector.get("holding") or 0
        say(f"Sector ETFs: holding {holding} of "
            f"{sector.get('universe', 0)}, on "
            f"{say_money(float(sector.get('sleeve_equity') or 0.0), currency)} "
            f"of its own capital.")
        if sector.get("symbols"):
            say(f"Those are {say_list([say_ticker(x) for x in sector['symbols']])}.")
        if sector.get("note"):
            say(_sentence(str(sector["note"])))

    # 6. What is scanning and firing, last because it is the part that will
    #    have changed by the time the operator asks again.
    scan = snapshot.get("universe_scan") or {}
    ranked, cohort = scan.get("ranked"), scan.get("size")
    if ranked and cohort:
        passes = scan.get("cohort_passes") or 0
        progress = (f", and has completed {passes} full pass"
                    f"{'es' if passes != 1 else ''}" if passes else
                    ", and has not finished its first full pass yet")
        say(f"The scanner is walking {ranked:,} ranked symbols, {cohort} at a "
            f"time{progress}.")

    say(admissions(snapshot))

    blockers = snapshot.get("blockers") or {}
    trading = int(blockers.get("trading") or 0)
    counts = blockers.get("counts") or []
    if trading:
        say(f"{trading} symbol{'s' if trading != 1 else ''} "
            f"{'are' if trading != 1 else 'is'} currently worth trading.")
    elif counts:
        top = counts[:MAX_BLOCKERS]
        spoken = ", ".join(
            f"{c['symbols']} because {say_blocker(str(c['blocker']))}"
            for c in top)
        say(f"Nothing is being traded. {spoken}.")

    return " ".join(part.strip() for part in lines if part and part.strip())
