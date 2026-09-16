"""What the terminal says when the operator presses Start.

A greeting by name and one quote worth hearing. It is a small thing and it is
deliberate: pressing Start is the moment real money starts being risked, and a
half-second that says *the machine is awake, and here is something a better
trader than you learned the hard way* is a better use of that moment than
silence.

**The quotes are attributed, and correctly.** Trading quotations are passed
around the internet in a state of near-total attribution collapse -- the most
famous one in this file is attached to Keynes almost everywhere and he did not
say it. A terminal whose entire claim is that it reports what it measured
cannot open by telling the operator something false, however charming. Where
an attribution is commonly wrong, the line below says so.

Read aloud, so the text is written to be heard: no em dashes the voice reads
as pauses in the wrong place, no numerals, and the attribution follows the
line the way a person would say it.
"""

from __future__ import annotations

import datetime as dt
import os
import random
from dataclasses import dataclass

from imperium.notify import briefing as brief

#: Who the terminal is talking to, as written.
#:
#: Overridable, because this program is packaged and someone else may end up
#: running it, and being greeted by another person's name is a small thing
#: that makes software feel like it was not written for you.
DEFAULT_OPERATOR = "Mr Gininda"

#: The same name, spelled for a voice rather than for a reader.
#:
#: English text-to-speech guesses at "Gininda" and guesses wrong. A phonetic
#: respelling is the fix that works on every engine: SSML <phoneme> tags are
#: supported by some ElevenLabs models and silently ignored by others, and a
#: greeting that depends on which model the account happens to be using is a
#: greeting that mispronounces its owner's name half the time.
#:
#: This is why the spoken and written forms were separate from the start. The
#: screen shows the name; the voice gets the sounds.
#:
#: The operator's own respelling was "Gee-neen-ndhha", and the last syllable
#: is changed here for a measured reason: "ndhha" is not a pronounceable
#: English cluster, and an engine that cannot say a cluster falls back to
#: spelling it. Run through a speech synthesiser it came out as
#: "Gee-neen-EN-DEE-AITCH-AITCH-AY" -- the name read out as letters, which is
#: a worse failure than the mispronunciation this is fixing. "dah" gives
#: /dʒiː.niːn.dɑː/ cleanly, and the first two syllables are untouched.
#:
#: The true sound is a prenasalised, breathy d of the kind Nguni languages
#: have and English orthography cannot carry, so this is the closest an
#: English respelling gets rather than an exact rendering.
DEFAULT_OPERATOR_SPOKEN = "Mr Gee-neen-dah"


def operator_name() -> str:
    """The name as it appears on screen."""
    return (os.environ.get("IMPERIUM_OPERATOR") or DEFAULT_OPERATOR).strip() \
        or DEFAULT_OPERATOR


def operator_spoken() -> str:
    """The name as the voice should say it.

    The respelling belongs to the name, not to the slot. Somebody who sets
    IMPERIUM_OPERATOR to their own name and nothing else must not have this
    program read out a phonetic spelling of a stranger's surname at them -- so
    a custom written name falls back to itself, and the default respelling is
    used only for the default name.
    """
    explicit = (os.environ.get("IMPERIUM_OPERATOR_SPOKEN") or "").strip()
    if explicit:
        return explicit
    written = operator_name()
    if written == DEFAULT_OPERATOR:
        return DEFAULT_OPERATOR_SPOKEN
    return written


@dataclass(frozen=True)
class Quote:
    """One line, and who actually said it."""

    text: str
    who: str
    #: Set where the usual attribution is wrong, and said out loud. Getting
    #: this right costs one clause and is the difference between a terminal
    #: that repeats what it heard and one that checked.
    note: str = ""

    def spoken(self) -> str:
        tail = f", {self.who}" if self.who else ""
        if self.note:
            tail += f". {self.note}"
        return f"{self.text}{tail}"


#: The rotation.
#:
#: Chosen for what they tell an operator watching an automated book rather than
#: for how they look on a poster: most of these are about losing money, waiting,
#: and not being clever, which is what this program spends its day doing.
QUOTES: tuple[Quote, ...] = (
    Quote("The market can stay irrational longer than you can stay solvent",
          "A. Gary Shilling",
          note="Almost everyone gives that one to Keynes. He never wrote it"),
    Quote("It was never my thinking that made the big money for me. "
          "It was always my sitting",
          "Jesse Livermore"),
    Quote("The four most dangerous words in investing are, "
          "this time it is different",
          "Sir John Templeton"),
    Quote("Be fearful when others are greedy, and greedy when others are "
          "fearful",
          "Warren Buffett"),
    Quote("Risk comes from not knowing what you are doing",
          "Warren Buffett"),
    Quote("The stock market is a device for transferring money from the "
          "impatient to the patient",
          "Warren Buffett"),
    Quote("Losers average losers",
          "Paul Tudor Jones",
          note="He kept it taped above his desk"),
    Quote("I am always thinking about losing money, as opposed to making "
          "money",
          "Paul Tudor Jones"),
    Quote("Amateurs think about how much money they can make. Professionals "
          "think about how much money they could lose",
          "Jack Schwager"),
    Quote("The big money is not in the buying and the selling, but in the "
          "waiting",
          "Charlie Munger"),
    Quote("In investing, what is comfortable is rarely profitable",
          "Robert Arnott"),
    Quote("The essence of investment management is the management of risks, "
          "not the management of returns",
          "Benjamin Graham"),
    Quote("Know what you own, and know why you own it",
          "Peter Lynch"),
    Quote("There is a time to go long, a time to go short, and a time to go "
          "fishing",
          "Jesse Livermore"),
    Quote("Do more of what works, and less of what does not",
          "Steve Clark"),
    Quote("Price is what you pay. Value is what you get",
          "Warren Buffett"),
    Quote("The goal of a successful trader is to make the best trades. "
          "Money is secondary",
          "Alexander Elder"),
    Quote("Markets are never wrong. Opinions often are",
          "Jesse Livermore"),
    Quote("An investor without a plan is a speculator with extra steps",
          "Benjamin Graham",
          note="Paraphrased from his rule that an investment operation "
               "promises safety of principal first"),
    Quote("Rule number one, never lose money. Rule number two, never forget "
          "rule number one",
          "Warren Buffett"),
)


def pick(previous: int = -1, rng: random.Random | None = None) -> int:
    """An index into QUOTES, never the one just used.

    Not shuffled through the whole list before repeating: an operator who
    starts the terminal twice in a morning should not be able to predict the
    second line, and one who starts it forty times should not have to wait out
    a cycle for a favourite. Only the immediate repeat is worth avoiding,
    because that is the one that reads as a bug.
    """
    chooser = rng or random
    if len(QUOTES) < 2:
        return 0
    while True:
        index = chooser.randrange(len(QUOTES))
        if index != previous:
            return index


@dataclass(frozen=True)
class Opening:
    """One greeting, in both forms, from one draw of the quote.

    Both forms in one object rather than two functions that each pick, because
    two picks is two different quotes: the terminal would say one thing and
    print another, which is a small bug that reads as the program not knowing
    what it is doing.
    """

    spoken: str
    hello: str
    quote: str
    who: str
    note: str
    index: int

    def as_dict(self) -> dict[str, object]:
        return {"spoken": self.spoken, "hello": self.hello,
                "quote": self.quote, "who": self.who, "note": self.note}


def opening(*, mode: str = "", previous: int = -1,
            now: dt.datetime | None = None,
            rng: random.Random | None = None) -> Opening:
    """The greeting, spoken and written, from a single draw.

    The mode is named because this is the one moment it matters most: "running
    live" and "running in dry run" are the same sentence to a glance at the
    header and a completely different fact about the next hour.

    The spoken and written forms are not the same text. The voice needs "and"
    where the screen wants an em dash, and the screen can put the attribution
    on its own line where speech has to run it on.
    """
    index = pick(previous, rng)
    quote = QUOTES[index]
    opener = brief.greeting(now)
    hello = f"{opener}, {operator_name()}."
    # The one difference between the two forms: the voice is handed a phonetic
    # respelling of the name, the screen is handed the name.
    heard = f"{opener}, {operator_spoken()}."
    where = {"live": " Trading live.",
             "paper": " Trading on paper.",
             "dry_run": " Dry run. No orders will be placed."}.get(mode, "")
    return Opening(
        spoken=f"{heard}{where} {quote.spoken()}.",
        hello=hello,
        quote=quote.text + ".",
        who=quote.who,
        note=quote.note,
        index=index,
    )
