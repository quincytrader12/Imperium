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


def build(snapshot: dict[str, Any], *, now: dt.datetime | None = None) -> str:
    """The whole spoken briefing, in the order a person would want it."""
    lines: list[str] = []
    say = lines.append

    # 1. Who and what state.
    mode = str(snapshot.get("mode") or "dry run").replace("_", " ")
    running = bool(snapshot.get("running"))
    simulated = bool(snapshot.get("simulated", True))
    say(f"{greeting(now)}. This is IMPERIUM.")
    if not running:
        say("The session is stopped, so nothing is being scanned or traded.")
    else:
        detail = ("Orders are simulated in this process and never reach the "
                  "venue." if simulated else
                  "Orders are really sent to the venue.")
        say(f"Running in {mode} mode. {detail}")

    # 2. The book, which is what an operator checks first.
    equity = float(snapshot.get("equity") or 0.0)
    currency = str((snapshot.get("account") or {}).get("currency") or "USD")
    unrealised = float(snapshot.get("unrealised_pnl") or 0.0)
    realised = float(snapshot.get("realised_pnl") or 0.0)
    positions = snapshot.get("positions") or []
    if equity:
        say(f"Equity is {say_money(equity, currency)}.")
    if positions:
        names = say_list([say_ticker(p["symbol"]) for p in positions
                          if p.get("symbol")])
        say(f"{len(positions)} open position"
            f"{'s' if len(positions) != 1 else ''}: {names}.")
        if abs(unrealised) >= 0.01:
            way = "up" if unrealised > 0 else "down"
            say(f"Unrealised profit and loss is {way} "
                f"{say_money(abs(unrealised), currency)}.")
    else:
        say("The book is flat, with no open positions.")
    if abs(realised) >= 0.01:
        way = "made" if realised > 0 else "lost"
        say(f"Realised, this session has {way} "
            f"{say_money(abs(realised), currency)}.")

    # 3. The halt, if there is one. Said early: nothing below it matters while
    #    the book is stopped.
    if snapshot.get("halted"):
        say(f"The book is halted. {snapshot.get('halt_reason') or ''} "
            f"Exits still pass, but no new exposure is opened.")

    # 4. The market, and whether anything can trade at all right now.
    market = snapshot.get("market") or {}
    if market.get("is_open"):
        say("The equity market is open.")
    else:
        say("The equity market is closed. Crypto trades around the clock and "
            "is unaffected.")

    # 5. What it is looking at.
    scan = snapshot.get("universe_scan") or {}
    ranked, cohort = scan.get("ranked"), scan.get("size")
    if ranked and cohort:
        passes = scan.get("cohort_passes") or 0
        progress = (f", and has completed {passes} full pass"
                    f"{'es' if passes != 1 else ''}" if passes else
                    ", and has not finished its first full pass yet")
        say(f"It is walking {ranked:,} ranked symbols, {cohort} at a time"
            f"{progress}.")

    # 6. Admissions — asked for by name.
    say(admissions(snapshot))

    # 7. Why nothing is trading, which on most days is the substance of it.
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

    # 8. The crypto ranking, when it has something to say.
    cross = snapshot.get("cross_section") or {}
    if cross.get("credible"):
        leaders = [say_ticker(l["symbol"]) for l in (cross.get("leaders") or [])]
        if leaders:
            say(f"On the crypto ranking, the strongest names are "
                f"{say_list(leaders, 3)}.")
    elif cross.get("cohort"):
        say(f"The crypto ranking is not measurable yet. "
            f"{cross.get('explain') or ''}")

    # 8b. What the headlines contributed. Said as a factor, because that is
    #     what it is -- a briefing that announces "sentiment is positive" on a
    #     trading terminal invites the listener to think it decided something.
    news = snapshot.get("news") or {}
    if news.get("enabled") and news.get("covered"):
        leaders = news.get("leaders") or []
        loud = [l for l in leaders if abs(float(l.get("score") or 0)) >= 0.2]
        if loud:
            top = loud[0]
            way = "positive" if float(top["score"]) > 0 else "negative"
            say(f"Of the symbols with news, {say_ticker(str(top['symbol']))} "
                f"reads most strongly {way}. Headlines adjust position size by "
                f"up to {news.get('max_tilt_pct', 0)} percent and never decide "
                f"whether a symbol is traded.")

    # 9. Anything actually wrong. Last, because a health line read before the
    #    book is a health line nobody waits for.
    feed = snapshot.get("feed") or {}
    if running and feed.get("connected") is False:
        reason = str(feed.get("reason") or "").strip()
        # The reason is written for a panel and starts lowercase; spoken after
        # a full stop that reads as a stumble. It also already says "not
        # connected" in most of its forms, so it replaces the sentence rather
        # than following it.
        say(_sentence(reason) if reason
            else "The market data feed is not connected.")

    return " ".join(part.strip() for part in lines if part and part.strip())
