"""Asking the terminal questions.

Two things are being protected here, and they pull in opposite directions.

The first is that an answer must come from the snapshot. Every test below
asserts on a figure that is in the fixture, because the failure this module
exists to prevent is a fluent sentence with nothing behind it -- and that
failure is invisible unless a test pins the number.

The second is that a question it did not understand must say so. A matcher
that stretches to answer everything is worse than one that admits a miss,
because a spoken wrong answer and a spoken right answer sound identical.
"""

from __future__ import annotations

import pytest

from imperium.notify import ask
from imperium.notify import briefing as brief


def _snapshot(**overrides):
    """A snapshot shaped like the real one, with nothing interesting in it.

    Built from the same keys ``TradingSession.snapshot`` emits. Tests override
    only the panel they are about, so a test about news cannot accidentally
    depend on the shape of the positions block.
    """
    base = {
        "running": True,
        "mode": "paper",
        "venue": "Alpaca",
        "market": {"describe": "market open"},
        "watchlist": [],
        "positions": [],
        "counters": {"scan": 0, "decision": 0, "refused": 0, "order": 0},
        "universe_scan": {"considered": 0, "size": 28},
        "blockers": {"summary": "", "trading": 0, "counts": []},
        "account": {"known": False, "equity": 0.0, "cash": 0.0,
                    "day_pnl": 0.0, "error": ""},
        "drawdown": {"used": 0.0, "limit": 0.04},
        "news": {"enabled": True, "scored": 0, "leaders": [],
                 "max_tilt_pct": 20, "broken": ""},
        "sector": {"enabled": False, "arm_at_equity": 200.0, "universe": 19,
                   "holding": 0, "symbols": [], "armed_on": "",
                   "configured": False},
        "fx": {"enabled": True, "known": False, "quote": "ZAR", "rate": None,
               "stale": False},
        "health": {"score": 1.0, "components": {}, "errors": 0,
                   "reconnects": 0},
        "limits": {"max_gross_exposure": 0.8, "max_concurrent_positions": 5,
                   "max_position_weight": 0.2, "daily_loss_halt": 0.04},
        "costs": {"median_round_trip_bps": 0.0, "fees_assumed": True,
                  "safety_multiple": 1.5},
    }
    base.update(overrides)
    return base


def _row(symbol, verdict="unscanned", **decision):
    payload = {"symbol": symbol, "verdict": verdict, "reason": "not yet scanned",
               "expected_edge_bps": 0.0, "round_trip_cost_bps": 0.0,
               "blocker": "", "clamp_reason": ""}
    payload.update(decision)
    return {"symbol": symbol, "verdict": verdict, "decision": payload}


# -- the matcher ------------------------------------------------------------


def test_every_intent_answers_its_own_example():
    """The test that caught the costs intent.

    Its example was "what is it costing to trade" and its phrases were "cost"
    and "costs" -- neither of which is a whole word in "costing". So the one
    question the terminal *offers* to answer, out loud, when asked what it can
    be asked, was a question it then failed to understand. An example that does
    not reach its own intent is a promise the program breaks the first time
    anybody takes it up.
    """
    snapshot = _snapshot()
    for intent in ask.INTENTS:
        assert intent.example, f"{intent.name} has no example to offer"
        answer = ask.respond(intent.example, snapshot)
        assert answer.intent == intent.name, (
            f"{intent.name!r} offers the example {intent.example!r}, which "
            f"the matcher routes to {answer.intent!r}")


def test_every_intent_is_reachable_and_distinct():
    """No two intents may claim the same phrase.

    A duplicated phrase makes one of them unreachable, and which one wins
    depends on the order of a tuple -- so the bug appears when somebody
    reorders the table for readability, long after the phrase was added.
    """
    seen: dict[str, str] = {}
    for intent in ask.INTENTS:
        for phrase in intent.phrases:
            assert phrase not in seen, (
                f"{phrase!r} is claimed by both {seen[phrase]!r} and "
                f"{intent.name!r}")
            seen[phrase] = intent.name


def test_a_question_it_does_not_understand_says_so():
    answer = ask.respond("make me a sandwich", _snapshot())
    assert answer.understood is False
    assert answer.intent == "unknown"
    # And it is useful about it, rather than only apologetic.
    assert "looking good" in answer.text


def test_an_empty_question_is_a_miss_not_a_crash():
    for question in ("", "   ", "..."):
        answer = ask.respond(question, _snapshot())
        assert answer.understood is False


def test_a_longer_phrase_beats_a_shorter_one():
    """"looking good" must beat "good", or every complimentary question lands
    on the same answer."""
    assert ask._score(" what is looking good ",
                      next(i for i in ask.INTENTS if i.name == "good")) > 1.0


def test_money_and_the_day_are_told_apart():
    """Both are asked with "how much". Only one is about today."""
    snapshot = _snapshot(account={"known": True, "equity": 214.80, "cash": 12.0,
                                  "day_pnl": -3.0, "error": ""})
    assert ask.respond("how much money do I have", snapshot).intent == "money"
    assert ask.respond("how much did I make today", snapshot).intent == "pnl"


def test_a_malformed_panel_does_not_take_the_voice_down():
    """A snapshot missing a block it expects is answered, not raised."""
    answer = ask.respond("what do I own", {"positions": "not a list"})
    assert answer.understood is False
    assert "could not read" in answer.text


# -- naming a symbol --------------------------------------------------------


def test_naming_a_symbol_narrows_the_question_to_it():
    snapshot = _snapshot(watchlist=[_row("SPY", reason="spread too wide")])
    answer = ask.respond("why aren't you trading SPY", snapshot)
    assert answer.intent == "symbol"
    assert "spread too wide" in answer.text


def test_the_same_question_without_a_symbol_is_about_the_book():
    snapshot = _snapshot(watchlist=[_row("SPY")])
    assert ask.respond("why aren't you trading", snapshot).intent == "blocked"


def test_a_symbol_spoken_letter_by_letter_is_found():
    """Recognition renders a ticker as separate letters about as often as not.
    "S P Y" and "SPY" are the same question."""
    snapshot = _snapshot(watchlist=[_row("SPY", reason="no edge")])
    assert ask.find_symbol(" why not s p y ", snapshot) == "SPY"


def test_a_common_word_that_is_also_a_ticker_is_not_treated_as_one():
    """IT, ARE, ALL and ON are all listed US equities.

    Once the universe is the whole venue rather than a seed list, they are all
    on the watchlist -- and without a guard "why are you not trading" finds
    ARE, and the terminal answers a question about a company nobody named.
    """
    snapshot = _snapshot(watchlist=[_row("IT"), _row("ARE"), _row("ALL"),
                                    _row("ON"), _row("SPY")])
    assert ask.find_symbol(" why are you not trading ", snapshot) == ""
    assert ask.find_symbol(" what is in it for all of you ", snapshot) == ""
    # The real ticker in the same sentence still is one.
    assert ask.find_symbol(" are you not trading spy ", snapshot) == "SPY"


def test_a_symbol_never_hijacks_a_question_about_the_account():
    """"How much money do I have" is about the account however many tickers
    the sentence happens to contain."""
    snapshot = _snapshot(watchlist=[_row("SPY")],
                         account={"known": True, "equity": 214.80,
                                  "cash": 12.0, "day_pnl": 0.0, "error": ""})
    answer = ask.respond("how much money do I have in SPY terms", snapshot)
    assert answer.intent == "money"


def test_an_unknown_symbol_says_it_is_not_watched():
    snapshot = _snapshot(watchlist=[_row("SPY")])
    assert "not on the watchlist" in ask.answer_symbol(snapshot, "TSLA")


# -- the answers themselves -------------------------------------------------


def test_what_looks_good_names_what_is_admitted():
    snapshot = _snapshot(watchlist=[
        _row("SPY", verdict="trading", expected_edge_bps=12.0,
             round_trip_cost_bps=4.0),
        _row("QQQ")])
    text = ask.respond("what's looking good", snapshot).text
    assert "S P Y" in text
    assert "12 basis points" in text
    assert "4 basis points" in text


def test_what_looks_good_names_the_closest_miss_when_nothing_is_admitted():
    """The useful answer when the book is flat, which is most of the time.

    "Nothing" is true but unhelpful: what the operator wants to know is what
    came closest and what stopped it.
    """
    snapshot = _snapshot(watchlist=[
        _row("QQQ", expected_edge_bps=3.0, round_trip_cost_bps=5.0,
             blocker="the expected edge does not clear the cost"),
        _row("SPY", expected_edge_bps=9.0, round_trip_cost_bps=10.0,
             blocker="the expected edge does not clear the cost")],
        universe_scan={"considered": 28, "size": 28})
    text = ask.respond("what's looking good", snapshot).text
    # The best near-miss, not the first in the list.
    assert "S P Y" in text
    assert "does not clear the cost" in text


def test_what_looks_good_admits_when_nothing_has_been_measured():
    text = ask.respond("what's looking good", _snapshot()).text
    assert "nothing has been measured" in text
    assert "0 of 28" in text


def test_what_have_you_found_counts_the_work():
    snapshot = _snapshot(counters={"scan": 1420, "decision": 31,
                                   "refused": 29, "order": 2})
    text = ask.respond("what have you found so far", snapshot).text
    assert "1,420 scans" in text
    assert "31 decisions" in text
    assert "29 refusals" in text
    assert "2 orders" in text


def test_a_count_of_one_is_not_said_in_the_plural():
    snapshot = _snapshot(counters={"scan": 1, "decision": 1, "refused": 0,
                                   "order": 1})
    text = ask.respond("what have you found so far", snapshot).text
    assert "one scan," in text
    assert "no refusals" in text


def test_why_not_trading_reads_the_blockers_panel():
    snapshot = _snapshot(blockers={"counts": [
        {"label": "cost", "count": 18}, {"label": "regime", "count": 6}]})
    text = ask.respond("why aren't you trading", snapshot).text
    assert "18 symbols" in text
    assert "6 symbols" in text


def test_why_not_trading_says_when_the_session_is_simply_off():
    text = ask.respond("why aren't you trading",
                       _snapshot(running=False)).text
    assert "not started" in text


def test_positions_are_named_with_their_value_and_total_pnl():
    snapshot = _snapshot(positions=[
        {"symbol": "XLF", "value": 22.5, "pnl": 1.25},
        {"symbol": "XLK", "value": 19.0, "pnl": -0.5}])
    text = ask.respond("what do I own", snapshot).text
    assert "2 open positions" in text
    assert "X L F" in text
    assert "up" in text and "0.75" in text


def test_a_flat_book_says_so_plainly():
    assert "flat" in ask.respond("what do I own", _snapshot()).text


def test_money_is_given_in_the_second_currency_when_a_rate_is_known():
    snapshot = _snapshot(
        account={"known": True, "equity": 214.80, "cash": 12.30,
                 "day_pnl": 0.0, "error": ""},
        fx={"enabled": True, "known": True, "quote": "ZAR", "rate": 18.42,
            "stale": False})
    text = ask.respond("how much money do I have", snapshot).text
    assert "214.80 dollars" in text
    assert "ZAR" in text
    assert "3,957" in text          # 214.80 * 18.42, rounded


def test_a_stale_rate_is_said_to_be_stale():
    snapshot = _snapshot(
        account={"known": True, "equity": 100.0, "cash": 0.0, "day_pnl": 0.0,
                 "error": ""},
        fx={"enabled": True, "known": True, "quote": "ZAR", "rate": 18.0,
            "stale": True})
    assert "stale" in ask.respond("what's my balance", snapshot).text


def test_an_unknown_balance_is_not_guessed_at():
    text = ask.respond("how much money do I have",
                       _snapshot(account={"known": False, "error": "no key",
                                          "equity": 0.0, "cash": 0.0,
                                          "day_pnl": 0.0})).text
    assert "not known yet" in text
    assert "no key" in text


def test_the_day_is_measured_against_the_days_budget():
    snapshot = _snapshot(
        account={"known": True, "equity": 210.0, "cash": 0.0,
                 "day_pnl": -4.80, "error": ""},
        drawdown={"used": 0.55, "limit": 0.04})
    text = ask.respond("how am I doing today", snapshot).text
    assert "down" in text
    assert "4.80 dollars" in text
    assert "55 percent" in text
    assert "4 percent" in text


def test_the_sector_sleeve_says_what_would_switch_it_on():
    text = ask.respond("what about the ETFs", _snapshot()).text
    assert "200 dollars" in text


def test_a_self_armed_sleeve_says_it_armed_itself():
    snapshot = _snapshot(sector={
        "enabled": True, "configured": False, "armed_on": "2026-09-15",
        "arm_at_equity": 200.0, "universe": 19, "holding": 2,
        "symbols": ["XLF", "XLK"]})
    text = ask.respond("what about the ETFs", snapshot).text
    assert "armed itself" in text
    assert "X L F" in text


def test_the_mode_answer_is_unambiguous_about_real_money():
    live = ask.respond("are you live or paper", _snapshot(mode="live")).text
    assert "real orders" in live
    dry = ask.respond("are you live or paper", _snapshot(mode="dry_run")).text
    assert "no orders at all" in dry


def test_health_names_what_is_dragging_the_score_down():
    snapshot = _snapshot(health={"score": 0.62,
                                 "components": {"venue": 0.5, "errors": 1.0},
                                 "errors": 3, "reconnects": 1})
    text = ask.respond("are you healthy", snapshot).text
    assert "62 percent" in text
    assert "venue" in text
    assert "3 errors" in text
    assert "one reconnect" in text


def test_news_says_what_it_scored_and_who_leads():
    snapshot = _snapshot(news={"enabled": True, "scored": 14, "max_tilt_pct": 20,
                               "broken": "",
                               "leaders": [{"symbol": "AAPL", "score": 0.4},
                                           {"symbol": "XOM", "score": -0.3}]})
    text = ask.respond("what's the news saying", snapshot).text
    assert "14 stories" in text
    assert "A A P L positive" in text
    assert "X O M negative" in text


def test_a_broken_news_feed_is_reported_not_papered_over():
    snapshot = _snapshot(news={"enabled": True, "scored": 0, "leaders": [],
                               "max_tilt_pct": 20, "broken": "HTTP 503"})
    assert "HTTP 503" in ask.respond("what's the news saying", snapshot).text


def test_the_help_answer_lists_every_intent():
    text = ask.help_text()
    for intent in ask.INTENTS:
        assert intent.example in text


def test_help_only_wins_when_nothing_else_does():
    """"What can you tell me about the news" is about the news."""
    snapshot = _snapshot()
    assert ask.respond("what can I ask", snapshot).intent == "help"
    assert ask.respond("what can you tell me about the news",
                       snapshot).intent == "news"


# -- the spoken helpers these rely on ---------------------------------------


@pytest.mark.parametrize("value,expected", [
    (0.80, "80 percent"), (0.04, "4 percent"), (0.045, "4.5 percent"),
    (0.778, "77.8 percent"), (0.0, "0 percent")])
def test_percentages_lose_a_pointless_decimal(value, expected):
    assert brief.say_percent(value) == expected


@pytest.mark.parametrize("value,expected", [
    (4.0, "4 basis points"), (1.0, "1 basis point"), (4.12, "4.1 basis points")])
def test_basis_points_are_spelled_out(value, expected):
    assert brief.say_basis_points(value) == expected


@pytest.mark.parametrize("number,expected", [
    (0, "no orders"), (1, "one order"), (2, "2 orders"), (1420, "1,420 orders")])
def test_zero_is_said_as_no_rather_than_as_a_digit(number, expected):
    assert brief.say_count(number, "order") == expected


def test_no_answer_contains_a_character_a_voice_would_mispronounce():
    """The whole point of the say_ helpers.

    A "$" or a "%" or a "bp" reaching the synthesiser is read literally, and
    the answer stops being English. This sweeps every intent against a
    populated snapshot and checks the output is speakable.
    """
    snapshot = _snapshot(
        running=True,
        watchlist=[_row("SPY", verdict="trading", expected_edge_bps=12.0,
                        round_trip_cost_bps=4.0)],
        positions=[{"symbol": "SPY", "value": 40.0, "pnl": 1.0}],
        counters={"scan": 100, "decision": 4, "refused": 3, "order": 1},
        account={"known": True, "equity": 214.80, "cash": 12.0,
                 "day_pnl": -2.0, "error": ""},
        drawdown={"used": 0.2, "limit": 0.04},
        fx={"enabled": True, "known": True, "quote": "ZAR", "rate": 18.42,
            "stale": False},
        news={"enabled": True, "scored": 9, "max_tilt_pct": 20, "broken": "",
              "leaders": [{"symbol": "AAPL", "score": 0.5}]},
        sector={"enabled": True, "configured": True, "armed_on": "",
                "arm_at_equity": 200.0, "universe": 19, "holding": 1,
                "symbols": ["XLF"]},
        blockers={"counts": [{"label": "cost", "count": 4}]},
        costs={"median_round_trip_bps": 6.2, "fees_assumed": False,
               "safety_multiple": 1.5})
    for intent in ask.INTENTS:
        text = ask.respond(intent.example, snapshot).text
        assert text, f"{intent.name} said nothing"
        for glyph in ("$", "%", "bp", "  "):
            assert glyph not in text, (
                f"{intent.name} said {glyph!r}, which a voice reads literally: "
                f"{text!r}")


# -- the endpoints ----------------------------------------------------------


def _client():
    from fastapi.testclient import TestClient

    from imperium.server.app import create_app

    return TestClient(create_app())


def test_the_endpoint_answers_from_the_live_session():
    with _client() as client:
        reply = client.post("/api/ask", json={"question": "are you live or paper"})
        assert reply.status_code == 200
        body = reply.json()
        assert body["intent"] == "mode"
        assert body["understood"] is True
        assert "dry run" in body["answer"]


def test_the_endpoint_bounds_the_question():
    """A question is echoed back and, spoken, billed per character."""
    with _client() as client:
        assert client.post("/api/ask", json={"question": ""}).status_code == 422
        assert client.post("/api/ask",
                           json={"question": "x" * 400}).status_code == 422


def test_speaking_an_answer_needs_a_voice_and_says_so():
    with _client() as client:
        reply = client.post("/api/ask/speak", json={"question": "are you live"})
        assert reply.status_code == 400
        assert "ElevenLabs" in reply.json()["detail"]


def test_only_the_answer_is_ever_synthesised_never_the_question(monkeypatch):
    """The operator's own words do not leave this machine.

    Worth pinning rather than assuming. The obvious way to build this feature
    is to post the question to a language model and read back what it says,
    and that sends someone's spoken words about their own account to a third
    party. Here the only text handed to the speech service is the sentence
    this program wrote from the local snapshot.
    """
    from fastapi.testclient import TestClient

    from imperium.server.app import create_app
    from imperium.session import TradingSession

    sent: list[str] = []
    session = TradingSession()

    async def capture(text):
        sent.append(text)
        return b"audio"

    # The real speaker asks a paid service; the point here is what is handed
    # to it, not that it works.
    session.speaker.speak = capture
    # Through monkeypatch, not by assigning to the class. "enabled" is a
    # property on Speaker, so setting it and then deleting it does not restore
    # it -- it removes the real one for every test that runs afterwards, which
    # is how this test first took five unrelated ones down with it.
    monkeypatch.setattr(type(session.speaker), "enabled",
                        property(lambda self: True))

    with TestClient(create_app(session)) as client:
        question = "how much money do I have"
        written = client.post("/api/ask",
                              json={"question": question}).json()["answer"]
        reply = client.post("/api/ask/speak", json={"question": question})
        assert reply.status_code == 200, reply.text
        assert reply.headers["X-Imperium-Intent"] == "money"

    assert len(sent) == 1
    # The question itself was never sent.
    assert question not in sent[0]
    # And what was spoken is exactly what the screen shows.
    assert sent[0] == written


def test_the_answer_is_the_same_whether_it_is_read_or_spoken():
    """One description of the state, two ways of delivering it -- the same
    rule the briefing follows. A spoken answer that differed from the written
    one would make the strip useless for checking what was heard."""
    with _client() as client:
        first = client.post("/api/ask",
                            json={"question": "what about the etfs"}).json()
        second = client.post("/api/ask",
                             json={"question": "what about the etfs"}).json()
        assert first["answer"] == second["answer"]
