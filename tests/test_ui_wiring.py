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
STYLES = (STATIC / "styles.css").read_text(encoding="utf-8")

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


def test_the_cohort_wide_reason_for_not_trading_is_rendered():
    """After four silent hours the only question worth asking is what is
    holding up *all* of the symbols. The session counts it; if nothing reads
    the count, the answer is computed, transmitted and invisible -- which is
    the state the per-symbol panel was already in."""
    assert "blockers" in APP_JS, (
        "nothing reads the cohort-wide blocker tally, so 'why is nothing "
        "trading' is still unanswerable on screen")
    assert "b.summary" in APP_JS, "the summary sentence itself is not rendered"
    assert 'id="blockers"' in INDEX, "the page has nowhere to show it"


# ---------------------------------------------------------------------------
# Script dependencies. app.js reads globals another file defines, and a missing
# script is not a degraded page — it throws on the first line that touches the
# global and the whole terminal comes up blank.
# ---------------------------------------------------------------------------


def _script_order() -> list[str]:
    return re.findall(r'<script[^>]+src="/static/([^"]+)"', INDEX)


def test_every_global_the_script_depends_on_is_loaded_by_the_page():
    """A missing <script> is not a missing feature.

    ``app.js`` reads ``window.Fmt`` at the top of its body, so if fmt.js is not
    loaded the reference throws immediately and every panel stays empty. The
    page would be blank, with one line in a console nobody has open.
    """
    scripts = _script_order()
    for global_name, provider in (("window.Fmt", "fmt.js"),
                                  ("window.Cluster", "cluster.js")):
        if global_name in APP_JS:
            assert provider in scripts, (
                f"app.js reads {global_name} and the page never loads "
                f"{provider} — the terminal would come up blank")


def test_dependencies_are_loaded_before_the_script_that_reads_them():
    """Order is the other half of it: a provider loaded after its consumer is
    the same failure with a longer explanation."""
    scripts = _script_order()
    assert "app.js" in scripts
    app_at = scripts.index("app.js")
    for provider in ("fmt.js", "cluster.js"):
        if provider in scripts:
            assert scripts.index(provider) < app_at, (
                f"{provider} is loaded after app.js, which reads it")


def test_the_build_verifies_every_asset_the_page_asks_for():
    """The packaging check derives its list from the page for the same reason.

    Kept by hand it goes stale silently: a new script ships unverified, and a
    build that dropped it still reports success.
    """
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "packaging"))
    from verify_build import page_assets

    found = page_assets(INDEX)
    for script in _script_order():
        assert f"/static/{script}" in found, (
            f"the build check would not verify {script} shipped")
    assert "/static/styles.css" in found, "the stylesheet is not checked either"


def test_the_build_check_actually_fails_when_an_asset_is_missing():
    """The check that guards the shipped artefact, checked itself.

    A version of this loop that verifies nothing is indistinguishable from one
    that verifies everything: both print a build that passed. That is the exact
    shape of failure this whole file exists for, one level up — so the loop is
    driven here against a bundle with a file deliberately missing.
    """
    import sys
    import urllib.error
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "packaging"))
    from verify_build import check_assets

    body = INDEX

    def fetch_all_present(path):
        return 200, "x" * 5000

    def fetch_missing_fmt(path):
        if path.endswith("fmt.js"):
            raise urllib.error.HTTPError(path, 404, "Not Found", None, None)
        return 200, "x" * 5000

    def fetch_truncated(path):
        return 200, "x" * 10          # served, but empty enough to be broken

    ok: list[str] = []
    check_assets(fetch_all_present, body, ok)
    assert ok == [], f"a complete bundle was reported broken: {ok}"

    missing: list[str] = []
    check_assets(fetch_missing_fmt, body, missing)
    assert any("fmt.js" in f for f in missing), (
        "a build that dropped fmt.js was reported as passing — the terminal "
        "would come up blank")
    assert any("add-data" in f for f in missing), "the failure must name the cause"

    truncated: list[str] = []
    check_assets(fetch_truncated, body, truncated)
    assert len(truncated) >= 3, (
        "files served as near-empty stubs were accepted")

    empty_page: list[str] = []
    check_assets(fetch_all_present, "<html><body>nothing</body></html>", empty_page)
    assert any("no static files" in f for f in empty_page), (
        "a page referencing nothing passed, so the loop can be satisfied by "
        "checking zero files")


# ---------------------------------------------------------------------------
# The watchlist is the scanner. Its columns must line up with its headings.
# ---------------------------------------------------------------------------


def _watchlist_headers() -> list[str]:
    """Just the watchlist's headings. The page has four tables, and matching
    <th> across all of them counts the fills journal's columns too."""
    table = re.search(r'<table id="watchlist".*?</thead>', INDEX, re.S)
    assert table, "the watchlist table moved; update this test"
    return re.findall(r"<th[^>]*>(.*?)</th>", table.group(0), re.S)


def test_the_watchlist_body_has_a_cell_for_every_heading():
    """The defect: five cells under four headings.

    HTML sizes a table by the widest row, so the extra cell shifted every
    heading one column left -- "Price" sat over the asset-class tag, "Verdict"
    over the 24h change, and the verdict itself, the one column the scanner
    exists to show, had no heading at all. Nothing failed; it just quietly
    read wrong.
    """
    headers = _watchlist_headers()
    built = re.search(r"\[('sym'.*?)\]\.forEach", APP_JS, re.S)
    assert built, "the row builder moved; update this test"
    cells = re.findall(r"'([a-z]+)'", built.group(1))
    assert len(cells) == len(headers), (
        f"the body builds {len(cells)} cells {cells} under {len(headers)} "
        f"headings {headers} — every heading after the extra cell labels the "
        f"wrong column")


def test_every_watchlist_column_has_a_declared_width():
    """Automatic layout lets the widest cell in any row decide the column, so
    one long verdict or a five-figure price squeezes the verdict into an
    ellipsis. A half-rendered verdict is worse than no verdict."""
    assert "#watchlist { table-layout: fixed; }" in STYLES, (
        "the table still sizes itself from its content")
    widths = re.findall(r"#watchlist th:nth-child\((\d)\)", STYLES)
    headers = _watchlist_headers()
    assert len(set(widths)) == len(headers), (
        f"{len(set(widths))} columns have a declared width but the table has "
        f"{len(headers)}")


def test_the_verdict_chip_cannot_be_clipped_silently():
    """"trading" and "warming up" must both be readable without the operator
    scrolling the panel sideways to find out which one it is."""
    assert "#watchlist .v-chip" in STYLES
    assert "text-overflow: ellipsis" in STYLES
