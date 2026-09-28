"""Asking the terminal a question out loud, and getting an honest answer.

The screen already holds every fact this program knows. What it does not hold
is a way to ask it something without reading a hundred and fifty rows to find
out. This turns a spoken question into a spoken answer, built from the same
snapshot the panels render.

**Every answer is read, never inferred.** There is no language model here and
no call to one, and that is the design rather than a limitation. This program's
entire claim is that it reports what it measured; an answer generated from a
prompt would be a fluent sentence with no measurement behind it, and the one
place that is least acceptable is the place an operator is most likely to trust
-- a confident voice telling them how their money is doing. So every sentence
below is assembled from fields in the snapshot, and when the snapshot does not
contain the answer, the answer is that it does not.

**A question it did not understand is not an answer it should invent.** The
matcher is deliberately literal: it scores a question against a table of
intents and, if nothing scores, it says so and lists what it *can* answer. A
near-miss that confidently answers the wrong question is worse than a miss,
because the operator has no way to tell the difference by ear.

**Spoken, not printed.** Answers go through the same say_* helpers as the
briefing, because "$214.80" read aloud is "dollar two hundred and fourteen
point eight zero" and "XLF" is either "eks ell eff" or a word, depending on the
voice. A listener also cannot skim, so answers are two or three sentences and
stop, even where the panel behind them has forty rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from imperium.notify import briefing as brief

#: How many items an answer names before it starts counting instead.
#:
#: Three, not the briefing's five. A briefing is listened to deliberately; an
#: answer to a question is listened to while doing something else, and the
#: fourth ticker in a spoken list is already gone.
MAX_NAMED = 3

#: The score a question must reach before it is treated as understood.
#:
#: One whole keyword. Below this the matcher is guessing from a stray word --
#: "how" appears in half of these phrasings -- and guessing is the failure this
#: module is most concerned with avoiding.
MIN_SCORE = 1.0


def _clean(question: str) -> str:
    """Lowercase words only. Speech recognition supplies its own punctuation."""
    return " " + re.sub(r"[^a-z0-9 ]+", " ", question.lower()).strip() + " "


def _has(text: str, phrase: str) -> bool:
    """Whole-word containment, so "pnl" does not match inside another word."""
    return f" {phrase} " in text


# ---------------------------------------------------------------------------
# the answers
#
# Each takes the snapshot and returns what to say. They are ordinary functions
# rather than templates because most of them have a branch in them: the honest
# answer to "what looks good" when nothing does is a different sentence, not
# the same sentence with an empty list in it.
# ---------------------------------------------------------------------------


def _admitted(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = snapshot.get("watchlist") or []
    return [r for r in rows if (r.get("verdict") or "") == "trading"]


def _candidates(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Symbols that cleared the regime test but not the cost gate.

    The interesting near-misses: something the engine liked enough to price,
    which is what "what are you looking at" actually means.
    """
    rows = []
    for row in snapshot.get("watchlist") or []:
        decision = row.get("decision") or {}
        if (row.get("verdict") or "") == "trading":
            continue
        if float(decision.get("expected_edge_bps") or 0.0) <= 0:
            continue
        rows.append(row)
    rows.sort(key=lambda r: float((r.get("decision") or {})
                                  .get("expected_edge_bps") or 0.0),
              reverse=True)
    return rows


def answer_good(snapshot: dict[str, Any]) -> str:
    """"What's looking good?" -- the question the operator actually asks."""
    held = _admitted(snapshot)
    if held:
        names = [brief.say_ticker(r["symbol"]) for r in held]
        return (f"{brief.say_list(names, MAX_NAMED)} "
                f"{'is' if len(held) == 1 else 'are'} admitted and being "
                f"traded. {_edge_line(held[0])}")

    near = _candidates(snapshot)
    if near:
        best = near[0]
        decision = best.get("decision") or {}
        return (f"Nothing is admitted. The closest is "
                f"{brief.say_ticker(best['symbol'])}, "
                f"{_edge_line(best)} "
                f"{_why_not(decision)}")

    scan = snapshot.get("universe_scan") or {}
    if not int(scan.get("considered") or 0):
        return ("Nothing looks good yet, because nothing has been measured "
                "yet. " + _scan_progress(snapshot))
    return ("Nothing looks good right now. Every symbol scanned has either "
            "failed its regime test or costs more to trade than it is "
            "expected to make.")


def _edge_line(row: dict[str, Any]) -> str:
    decision = row.get("decision") or {}
    edge = float(decision.get("expected_edge_bps") or 0.0)
    cost = float(decision.get("round_trip_cost_bps") or 0.0)
    if edge <= 0 and cost <= 0:
        return "No edge has been measured on it yet."
    return (f"It is expected to make {brief.say_basis_points(edge)} "
            f"against {brief.say_basis_points(cost)} of round trip cost.")


def _why_not(decision: dict[str, Any]) -> str:
    reason = (decision.get("blocker") or decision.get("clamp_reason")
              or decision.get("reason") or "")
    return f"It is not admitted because {reason}." if reason else ""


def answer_found(snapshot: dict[str, Any]) -> str:
    """"What have you found so far?" -- the work done, counted."""
    counters = snapshot.get("counters") or {}
    scans = int(counters.get("scan") or 0)
    decisions = int(counters.get("decision") or 0)
    refused = int(counters.get("refused") or 0)
    orders = int(counters.get("order") or 0)
    if not scans:
        return "Nothing yet. " + _scan_progress(snapshot)
    parts = [f"{brief.say_count(scans, 'scan')}, "
             f"{brief.say_count(decisions, 'decision')}, "
             f"{brief.say_count(refused, 'refusal')} on cost, and "
             f"{brief.say_count(orders, 'order')} sent."]
    held = _admitted(snapshot)
    if held:
        names = [brief.say_ticker(r["symbol"]) for r in held]
        parts.append(f"Currently admitted: {brief.say_list(names, MAX_NAMED)}.")
    else:
        parts.append("Nothing is admitted at the moment.")
    return " ".join(parts)


def _scan_progress(snapshot: dict[str, Any]) -> str:
    scan = snapshot.get("universe_scan") or {}
    considered = int(scan.get("considered") or 0)
    size = int(scan.get("size") or 0)
    if not size:
        return "The universe has not been scanned yet."
    return (f"{brief.say_number(considered)} of "
            f"{brief.say_number(size)} symbols have been looked at so far.")


def answer_blocked(snapshot: dict[str, Any]) -> str:
    """"Why aren't you trading?" -- straight off the blockers panel."""
    blockers = snapshot.get("blockers") or {}
    if not snapshot.get("running"):
        return "The session is not started, so nothing is being scanned."
    counts = blockers.get("counts") or []
    if not counts:
        if _admitted(snapshot):
            return "Nothing is blocking it. Positions are admitted and open."
        return ("Nothing is blocking it in particular. Nothing has cleared "
                "the regime test yet.")
    named = [f"{brief.say_blocker(c.get('label', ''))} on "
             f"{brief.say_count(int(c.get('count') or 0), 'symbol')}"
             for c in counts[:MAX_NAMED]]
    return "The main reasons are " + brief.say_list(named, MAX_NAMED) + "."


def answer_positions(snapshot: dict[str, Any]) -> str:
    """"What do I own?" -- the book, not the watchlist."""
    rows = snapshot.get("positions") or []
    if not rows:
        return "The book is flat. There are no open positions."
    named = [f"{brief.say_ticker(r.get('symbol', ''))} worth "
             f"{brief.say_money(abs(float(r.get('value') or 0.0)))}"
             for r in rows[:MAX_NAMED]]
    line = (f"{brief.say_count(len(rows), 'open position')}: "
            + brief.say_list(named, MAX_NAMED) + ".")
    pnl = sum(float(r.get("pnl") or 0.0) for r in rows)
    if pnl:
        way = "up" if pnl > 0 else "down"
        line += f" Together they are {way} {brief.say_money(abs(pnl))}."
    return line


def answer_money(snapshot: dict[str, Any]) -> str:
    """"How much have I got?" -- in dollars, and in rands if a rate is known."""
    account = snapshot.get("account") or {}
    if not account.get("known"):
        return ("The account balance is not known yet. "
                + (account.get("error") or "No key is attached."))
    equity = float(account.get("equity") or 0.0)
    line = f"The account holds {brief.say_money(equity)}"
    fx = snapshot.get("fx") or {}
    if fx.get("enabled") and fx.get("known"):
        converted = equity * float(fx.get("rate") or 0.0)
        stale = " on a rate that is now stale" if fx.get("stale") else ""
        line += (f", which is about {brief.say_number(round(converted))} "
                 f"{fx.get('quote', '')}{stale}")
    cash = float(account.get("cash") or 0.0)
    return line + f". {brief.say_money(cash)} of that is uninvested cash."


def answer_pnl(snapshot: dict[str, Any]) -> str:
    """"How am I doing today?" -- the day, against the day's budget."""
    account = snapshot.get("account") or {}
    drawdown = snapshot.get("drawdown") or {}
    day = float(account.get("day_pnl") or 0.0)
    if not account.get("known"):
        return "The account balance is not known yet, so neither is the day."
    if abs(day) < 0.005:
        head = "The account is flat on the day."
    else:
        head = (f"The account is {'up' if day > 0 else 'down'} "
                f"{brief.say_money(abs(day))} on the day.")
    used = float(drawdown.get("used") or 0.0)
    limit = float(drawdown.get("limit") or 0.0)
    if used > 0:
        return (f"{head} That is {brief.say_percent(used)} of the "
                f"{brief.say_percent(limit)} daily loss budget.")
    return f"{head} None of the daily loss budget has been used."


def answer_news(snapshot: dict[str, Any]) -> str:
    """"What's the news saying?" -- the sentiment factor, and its coverage."""
    news = snapshot.get("news") or {}
    if not news.get("enabled"):
        return "News sentiment is switched off."
    if news.get("broken"):
        return f"The news feed is not working: {news['broken']}."
    scored = int(news.get("scored") or 0)
    if not scored:
        return ("No news has been scored yet. It tilts position size by at "
                f"most {brief.say_percent(float(news.get('max_tilt_pct') or 0) / 100)}, "
                "so nothing is being tilted right now.")
    leaders = news.get("leaders") or []
    if not leaders:
        return (f"{brief.say_count(scored, 'story')} scored, but no symbol "
                f"has a strong enough tilt to name.")
    named = []
    for row in leaders[:MAX_NAMED]:
        score = float(row.get("score") or 0.0)
        way = "positive" if score > 0 else "negative"
        named.append(f"{brief.say_ticker(row.get('symbol', ''))} {way}")
    return (f"{brief.say_count(scored, 'story')} scored. "
            + brief.say_list(named, MAX_NAMED) + ".")


def answer_sector(snapshot: dict[str, Any]) -> str:
    """"What about the ETFs?" -- the Sector Trend sleeve."""
    sector = snapshot.get("sector") or {}
    if not sector.get("enabled"):
        threshold = float(sector.get("arm_at_equity") or 0.0)
        if threshold > 0:
            return (f"The sector trend sleeve is off. It switches itself on "
                    f"when the account reaches "
                    f"{brief.say_money(threshold)}.")
        return "The sector trend sleeve is off."
    holding = int(sector.get("holding") or 0)
    universe = int(sector.get("universe") or 0)
    how = ("It armed itself" if sector.get("armed_on")
           and not sector.get("configured") else "It is switched on")
    if not holding:
        return (f"{how} and is watching {brief.say_number(universe)} sector "
                f"ETFs, holding none of them. Nothing has broken out.")
    names = [brief.say_ticker(s) for s in (sector.get("symbols") or [])]
    return (f"{how}, watching {brief.say_number(universe)} sector ETFs and "
            f"holding {brief.say_number(holding)}: "
            + brief.say_list(names, MAX_NAMED) + ".")


def answer_health(snapshot: dict[str, Any]) -> str:
    """"Are you healthy?" -- the score, and what is dragging it down."""
    health = snapshot.get("health") or {}
    score = float(health.get("score") or 0.0)
    line = f"Health is {brief.say_percent(score)}."
    parts = health.get("components") or {}
    weak = sorted((v, k) for k, v in parts.items() if float(v) < 1.0)
    if weak:
        worst = weak[0][1]
        line += f" The {worst} check is what is holding it down."
    errors = int(health.get("errors") or 0)
    reconnects = int(health.get("reconnects") or 0)
    if errors or reconnects:
        line += (f" {brief.say_count(errors, 'error')} and "
                 f"{brief.say_count(reconnects, 'reconnect')} so far.")
    return line


def answer_mode(snapshot: dict[str, Any]) -> str:
    """"Are you live?" -- the question with the most expensive wrong answer."""
    mode = str(snapshot.get("mode") or "")
    venue = snapshot.get("venue") or "the venue"
    market = (snapshot.get("market") or {}).get("describe") or ""
    spoken = {"live": "placing real orders",
              "paper": "on the venue's paper account",
              "dry_run": "in dry run, placing no orders at all"}.get(
                  mode, mode.replace("_", " "))
    running = "Running" if snapshot.get("running") else "Stopped"
    where = f"{market[0].upper()}{market[1:]}." if market else ""
    return f"{running} on {venue}, {spoken}. {where}".strip()


def answer_risk(snapshot: dict[str, Any]) -> str:
    """"What are your limits?" -- the bounds it is trading inside."""
    limits = snapshot.get("limits") or {}
    return (f"At most {brief.say_percent(float(limits.get('max_gross_exposure') or 0))} "
            f"gross exposure, "
            f"{brief.say_number(int(limits.get('max_concurrent_positions') or 0))} "
            f"positions at once, no more than "
            f"{brief.say_percent(float(limits.get('max_position_weight') or 0))} "
            f"in any one, and the book halts for the day at a "
            f"{brief.say_percent(float(limits.get('daily_loss_halt') or 0))} loss.")


def answer_costs(snapshot: dict[str, Any]) -> str:
    """"What is it costing?" -- the gate everything has to clear."""
    costs = snapshot.get("costs") or {}
    median = float(costs.get("median_round_trip_bps") or 0.0)
    if not median:
        assumed = "assumed, not measured" if costs.get("fees_assumed") else ""
        return ("No round trip has been measured yet, so costs are "
                f"{assumed or 'not yet known'}. Nothing trades until its "
                "expected edge clears its modelled cost.")
    return (f"The median round trip costs {brief.say_basis_points(median)}, "
            f"and an idea has to beat that by "
            f"{costs.get('safety_multiple', 1)} times before it is admitted.")


def answer_symbol(snapshot: dict[str, Any], symbol: str) -> str:
    """"Why aren't you buying Apple?" -- that symbol's actual decision."""
    wanted = symbol.upper()
    for row in snapshot.get("watchlist") or []:
        if (row.get("symbol") or "").upper() != wanted:
            continue
        decision = row.get("decision") or {}
        verdict = row.get("verdict") or ""
        spoken = brief.say_ticker(wanted)
        if verdict == "trading":
            return f"{spoken} is admitted. {_edge_line(row)}"
        reason = decision.get("reason") or "no reason was recorded"
        return f"{spoken} is not trading: {reason}."
    return (f"{brief.say_ticker(wanted)} is not on the watchlist, so nothing "
            f"has been measured about it.")


# ---------------------------------------------------------------------------
# the matcher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Intent:
    """One thing that can be asked, and the words people ask it with.

    Phrases are scored rather than matched in order, so "how much am I down
    today" reaches the P&L answer and not the money answer, even though both
    contain "how much".
    """

    name: str
    answer: Callable[[dict[str, Any]], str]
    #: Phrases that identify this intent. Multi-word phrases score higher,
    #: because "looking good" is far more specific than "good".
    phrases: tuple[str, ...] = ()
    #: One example, spoken back when the operator asks what they can ask.
    example: str = ""
    #: Phrases that rule this intent *out*. "how much" means money, unless the
    #: question also says "today", in which case it means the day's P&L.
    against: tuple[str, ...] = field(default_factory=tuple)


INTENTS: tuple[Intent, ...] = (
    Intent("good", answer_good, example="what's looking good",
           phrases=("looking good", "look good", "looks good", "best idea",
                    "best trade", "anything good", "what should i buy",
                    "any good trades", "promising", "strongest",
                    "what do you like", "opportunities")),
    Intent("found", answer_found, example="what have you found so far",
           phrases=("found so far", "what have you found", "found anything",
                    "what are you doing", "how far", "progress",
                    "how many scans", "what have you seen", "scanned")),
    Intent("blocked", answer_blocked, example="why aren't you trading",
           phrases=("why are you not trading", "why aren t you trading",
                    "why no trades", "what is blocking", "what s blocking",
                    "why nothing", "blocked", "blockers", "why not trading",
                    "why isn t it trading", "stopping you")),
    Intent("positions", answer_positions, example="what do I own",
           phrases=("what do i own", "positions", "holdings", "what am i in",
                    "what do you hold", "am i holding", "open positions",
                    "what are we holding")),
    Intent("money", answer_money, example="how much money do I have",
           against=("today", "day", "p n l", "pnl", "profit", "loss"),
           phrases=("how much money", "balance", "equity", "how much do i have",
                    "how much cash", "account value", "buying power",
                    "how much is in", "in rands", "in rand")),
    Intent("pnl", answer_pnl, example="how am I doing today",
           # Every way a person asks this out loud, and there are a lot of
           # them: "made" and "make" and "making" are three different words to
           # a whole-word matcher, and all three are the same question.
           phrases=("how am i doing", "p n l", "pnl", "profit", "made today",
                    "lost today", "up or down", "how are we doing",
                    "how much today", "down today", "up today", "made money",
                    "did i make", "did i lose", "make today", "making today",
                    "lose today", "losing today", "how did we do",
                    "how did i do", "am i up", "am i down", "doing today")),
    Intent("news", answer_news, example="what's the news saying",
           phrases=("news", "sentiment", "headlines", "what are they saying",
                    "stories", "articles")),
    Intent("sector", answer_sector, example="what about the ETFs",
           phrases=("etf", "etfs", "sector", "sectors", "sleeve",
                    "sector trend", "the etf strategy")),
    Intent("health", answer_health, example="are you healthy",
           phrases=("healthy", "health", "are you ok", "are you okay",
                    "status", "everything ok", "any errors", "all good")),
    Intent("mode", answer_mode, example="are you live or paper",
           phrases=("live or paper", "are you live", "what mode", "real money",
                    "dry run", "paper", "is it running", "are you running")),
    Intent("risk", answer_risk, example="what are your limits",
           phrases=("limits", "risk", "how much can you lose", "exposure",
                    "how big", "position size", "max position")),
    Intent("costs", answer_costs, example="what is it costing to trade",
           phrases=("cost", "costs", "costing", "fees", "spread", "slippage",
                    "commission", "expensive", "how much does it cost")),
)

#: Intents that a named symbol narrows to that symbol.
#:
#: Not all of them. "How much money do I have" is about the account however
#: many tickers the sentence happens to contain, and answering it with one
#: symbol's decision would be answering a question nobody asked.
SYMBOL_AWARE = frozenset({"good", "blocked", "found", "positions"})

#: Words that mean "list what you can answer" rather than any one intent.
HELP_PHRASES = ("what can i ask", "what can you tell me", "help",
                "what do you know", "options", "commands", "what can you do")

#: Short words that are also real tickers.
#:
#: IT, ON, ALL, SO, ARE, BE, GO and a dozen more are listed US equities, and
#: once the universe is the whole venue rather than a seed list they are all on
#: the watchlist. Without this, "why are you not trading" finds ARE, and "how
#: much is in it" finds IT, and the terminal confidently answers a question
#: about a company nobody mentioned. A symbol has to survive this list before
#: it is treated as one.
SPOKEN_STOPWORDS = frozenset("""
a all am an and any are as at be been but by can did do does for from go
had has have he her him his how i if in is it its me my no not now of off
on one or our out say see she so some that the their them then there they
this to up us was we were what when where which who why will with you your
""".split())


def _score(text: str, intent: Intent) -> float:
    """How well a question matches one intent.

    Longer phrases score higher because they are more specific: matching
    "looking good" should beat matching "good", or every complimentary
    question lands on the same answer.
    """
    if any(_has(text, phrase) for phrase in intent.against):
        return 0.0
    best = 0.0
    for phrase in intent.phrases:
        if _has(text, phrase):
            best = max(best, 1.0 + 0.5 * phrase.count(" "))
    return best


def help_text() -> str:
    """What this can be asked, said out loud."""
    examples = [i.example for i in INTENTS if i.example]
    return ("You can ask me " + ", ".join(examples[:-1])
            + ", or " + examples[-1]
            + ". You can also ask about any symbol by name, "
              "for example, why aren't you trading S P Y.")


def find_symbol(text: str, snapshot: dict[str, Any]) -> str:
    """A ticker in the question, but only one the terminal actually watches.

    Checked against the watchlist rather than against a pattern. Almost every
    short English word looks like a ticker -- "is", "it", "up", "so" -- and a
    matcher that trusted the shape of the word would answer "why aren't you
    trading IT" to half the questions asked of it.
    """
    known = {(r.get("symbol") or "").upper()
             for r in (snapshot.get("watchlist") or [])}
    known |= {s.upper() for s in ((snapshot.get("sector") or {})
                                  .get("symbols") or [])}
    if not known:
        return ""
    # Spoken letter-by-letter: "s p y" should find SPY.
    squashed = re.sub(r"\b([a-z])\s+(?=[a-z]\b)", r"\1", text)
    for candidate in sorted(known, key=len, reverse=True):
        token = candidate.lower()
        if token in SPOKEN_STOPWORDS:
            continue
        if _has(text, token) or _has(squashed, token):
            return candidate
    return ""


@dataclass(frozen=True)
class Answer:
    """What was asked, what was said, and which intent produced it."""

    question: str
    text: str
    intent: str
    understood: bool = True

    def as_dict(self) -> dict[str, object]:
        return {"question": self.question, "answer": self.text,
                "intent": self.intent, "understood": self.understood}


def respond(question: str, snapshot: dict[str, Any]) -> Answer:
    """Answer one question from the snapshot. Never raises, never invents.

    The order matters. Help first, because "what can you tell me about the
    news" should be about the news and "what can you tell me" should be the
    list -- so help only wins when no intent scores. Then the best-scoring
    intent. Then a symbol, because "SPY" alone is a question about SPY. Then
    the honest miss.
    """
    text = _clean(question or "")
    if not text.strip():
        return Answer(question, "I did not catch that. " + help_text(),
                      "unknown", understood=False)

    scored = sorted(((_score(text, i), i) for i in INTENTS),
                    key=lambda pair: pair[0], reverse=True)
    top, intent = scored[0]

    if top < MIN_SCORE and any(_has(text, p) for p in HELP_PHRASES):
        return Answer(question, help_text(), "help")

    # A named symbol makes the question about that symbol, even when a global
    # intent also matched: "why aren't you trading" is about the book, and
    # "why aren't you trading S P Y" is about S P Y, and they share every word
    # but the last one.
    symbol = find_symbol(text, snapshot)
    if symbol and (top < MIN_SCORE or intent.name in SYMBOL_AWARE):
        return Answer(question, answer_symbol(snapshot, symbol), "symbol")

    if top >= MIN_SCORE:
        try:
            return Answer(question, intent.answer(snapshot), intent.name)
        except Exception as exc:                            # noqa: BLE001
            # A malformed snapshot must not take the voice down with it. Said
            # plainly rather than swallowed: "I could not read that" is a fact
            # the operator can act on, silence is not.
            return Answer(question,
                          f"I could not read the {intent.name} panel just "
                          f"now. {type(exc).__name__}.",
                          intent.name, understood=False)

    if any(_has(text, p) for p in HELP_PHRASES):
        return Answer(question, help_text(), "help")

    return Answer(question,
                  "I did not understand that. " + help_text(),
                  "unknown", understood=False)
