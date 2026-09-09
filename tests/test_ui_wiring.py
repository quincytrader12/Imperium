"""The terminal must actually render what the server sends it.

This is the class of bug that produced the operator's report. The snapshot
already carried everything needed to explain a dark data lamp -- whether the
socket was connected, what the venue's last error was, how many symbols were
subscribed -- and the browser read none of it. The server was right, the
payload was right, and the screen said nothing.

Nothing caught it because no test connected the three: the payload the session
builds, the ids the script writes to, and the elements the page actually has.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "imperium" / "server" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")

#: Ids the script builds at runtime from a prefix plus a key. They are covered
#: by the lamp assertion below instead.
DYNAMIC = re.compile(r"\$\('[^']*'\s*\+")


def _referenced_ids() -> set[str]:
    """Every element id the script asks for by a literal name."""
    return set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", APP_JS))


def _declared_ids() -> set[str]:
    """Ids that exist by the time the script asks for them.

    The page is one source; the script is the other. A few panels build their
    own markup on first render -- the collapsible fee notice injects its body
    into ``cost-warn`` -- so those ids are declared in a JavaScript string
    rather than in the HTML, and are no less real for it.
    """
    return (set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', INDEX))
            | set(re.findall(r"\bid=\\?'([A-Za-z0-9_-]+)\\?'", APP_JS))
            | set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', APP_JS)))


def test_every_element_the_script_writes_to_exists_in_the_page():
    """A renderer writing into an element that is not there fails silently.

    ``$('hz-feed')`` returns null, the assignment throws, and the rest of that
    render function never runs -- so a typo in one id can blank several panels
    at once with nothing on screen or in the console the operator will see.
    """
    missing = sorted(_referenced_ids() - _declared_ids())
    assert not missing, (
        f"the script writes to elements nothing ever creates: {missing}")


def test_the_page_declares_the_ids_the_script_expects_it_to_own():
    """The check above passes if the script creates an id itself, so this
    pins the ones that must come from the page rather than from a renderer
    that may never run."""
    page = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', INDEX))
    for required in ("hz-feed", "hz-age", "hz-errors", "reasoning", "banner"):
        assert required in page, f"{required} must exist before any render"


def test_the_lamp_ids_the_script_builds_are_all_declared():
    """The one place ids are composed rather than written out."""
    assert DYNAMIC.search(APP_JS), "the composed-id pattern moved; update this"
    declared = _declared_ids()
    for key in ("link", "venue", "data", "key", "session"):
        assert f"lamp-{key}" in declared, f"lamp-{key} is missing from the page"


def test_the_reason_the_data_lamp_is_dark_is_put_on_screen():
    """The operator's report, as an assertion.

    The session computes the sentence and puts it in the payload. If nothing
    writes it into an element the sentence exists, is transmitted, and is
    invisible -- precisely the state this was written to end.

    This pins the assignment rather than any mention of ``reason``. The first
    version of this test searched the whole file, and passed happily with the
    visible line blanked, because the lamp's tooltip mentions ``reason`` too:
    it proved the word was present, not that anything was displayed.

    Structural rather than executed: the client is one closure with no exports
    and a websocket opened at load, so calling the renderer would mean stubbing
    most of a browser. What it does check is the exact statement that failed.
    """
    assert "hz-feed" in INDEX, "the page has nowhere to show it"
    block = re.search(r"\$\('hz-feed'\)(.*?)\n    \}", APP_JS, re.S)
    assert block, "the health renderer no longer resolves hz-feed"
    assert re.search(r"\.textContent\s*=\s*[^;]*\breason\b", block.group(1)), (
        "hz-feed is resolved and never given the feed's explanation — the "
        "sentence is computed, sent, and shown nowhere")


def test_the_venue_error_behind_a_dead_socket_is_reachable():
    """``last_error`` is the only thing that separates a rejected key from a
    blocked port, and it is sent on every frame."""
    assert "last_error" in APP_JS, (
        "the venue's own words about why the socket died are sent and never "
        "shown")
