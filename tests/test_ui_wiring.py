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
                                  ("window.Palette", "palette.js")):
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
    for provider in ("fmt.js", "palette.js"):
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


def test_the_news_panel_is_rendered_and_states_that_it_is_not_a_gate():
    """Prevents two separate failures at once.

    The first is the one this file exists for: a panel whose ids the server
    fills and the page never shows. The second is specific to this factor --
    an operator who sees "News sentiment" on a trading terminal will assume it
    decides what gets traded, because on most trading terminals something
    called that does. It does not here, and the panel has to say so on screen
    rather than only in the source.
    """
    assert "renderNews(s)" in APP_JS, "the news panel is never rendered"
    for element in ("news-note", "news-grid", "news-leaders", "news-why"):
        assert 'id="' + element + '"' in INDEX, f"{element} is not on the page"
    assert "never admit a symbol" in APP_JS
    assert "after the cost gate" in APP_JS


def test_the_news_panel_says_when_its_feed_has_gone_silent():
    """Prevents an undocumented feed failing invisibly. Yahoo's RSS can change
    without notice, and the symptom is no coverage -- which looks exactly like
    a quiet news week on a panel that does not say otherwise."""
    assert "n.broken" in APP_JS, "the silent-feed warning is never shown"
    assert "n.source" in APP_JS, "the panel does not say which source it used"


def test_a_refused_data_socket_shows_its_remedy_on_screen():
    """Prevents the fix for a dark data lamp living only in a tooltip.

    A refused data socket is the one fault that stops this terminal being
    useful at all, and the operator report that prompted this had them
    checking their API key and their Alpaca plan — neither of which was the
    problem. The thing to actually do belongs on the screen.
    """
    assert 'id="hz-feed-remedy"' in INDEX, "the remedy element is not on the page"
    assert "hz-feed-remedy" in APP_JS, "the remedy is never written to"
    assert "f.remedy" in APP_JS, "the remedy is never read from the snapshot"
    assert ".feed-remedy" in STYLES, "the remedy has no styling"


def test_the_sector_sleeve_panel_is_rendered_and_says_when_it_is_off():
    """Prevents the commonest question about a disabled strategy.

    A sleeve that is switched off looks exactly like a sleeve that is on and
    finding nothing, and the difference matters most to the person wondering
    why nothing has traded. The panel has to say which, and say how to arm it.
    """
    assert "renderSector(s)" in APP_JS, "the sector panel is never rendered"
    for element in ("st-note", "st-grid", "st-holdings", "st-why"):
        assert 'id="' + element + '"' in INDEX, f"{element} is not on the page"
    # It used to tell the operator to edit SECTOR_TREND_ENABLED. There is an
    # arming control in the panel now, so the page must offer that instead of
    # sending them to a file and a restart.
    assert "SECTOR_TREND_ENABLED" not in APP_JS, (
        "the panel still sends the operator to a settings file it no longer "
        "needs them to edit")
    assert 'id="st-arm"' in INDEX, "the arming control is not on the page"
    assert "/api/sector/arm" in APP_JS and "/api/sector/disarm" in APP_JS
    assert "keeps its own book" in APP_JS.lower() or "its own book" in APP_JS


def test_the_capital_panel_shows_how_the_account_is_divided():
    """Prevents the split existing only in the source. Every strategy's
    position size is a fraction of its share rather than of the account, and
    before this panel there was nowhere on screen that said what any share
    was -- or that one had been refused for want of room."""
    assert "renderCapital(s)" in APP_JS, "the capital panel is never rendered"
    for element in ("cap-note", "cap-grid", "cap-why"):
        assert 'id="' + element + '"' in INDEX, f"{element} is not on the page"
    assert "oversubscribed" in APP_JS
    assert "c.refused" in APP_JS, "a refused claim is never surfaced"


def _cell_sub_labels(source: str) -> list[str]:
    """The fourth argument of every cell(...) call that passes a literal.

    Parsed by matching parentheses rather than by regex: the arguments contain
    nested calls and their own brackets, and a regex that tried to span them
    either stopped at the first ")" or ran past the end of the call. Arguments
    that are not plain string literals -- a ternary, a variable -- are skipped;
    this measures the wording that is written down.
    """
    labels: list[str] = []
    start = 0
    while True:
        at = source.find("cell(", start)
        if at < 0:
            return labels
        # A "cell(" that is part of a longer identifier is not a cell call.
        if at and (source[at - 1].isalnum() or source[at - 1] in "_$."):
            start = at + 5
            continue
        depth, i, args, current = 0, at + 4, [], []
        quote = ""
        while i < len(source):
            ch = source[i]
            if quote:
                if ch == "\\":
                    current.append(source[i:i + 2])
                    i += 2
                    continue
                if ch == quote:
                    quote = ""
                current.append(ch)
            elif ch in "\"'":
                quote = ch
                current.append(ch)
            elif ch in "([{":
                depth += 1
                if depth > 1:
                    current.append(ch)
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    args.append("".join(current).strip())
                    break
                current.append(ch)
            elif ch == "," and depth == 1:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
            i += 1
        start = i + 1
        if len(args) < 4:
            continue
        sub = args[3]
        literal = _string_literal(sub)
        if literal is not None:
            labels.append(literal)


def _string_literal(text: str) -> str | None:
    """The value of `text` if it is one whole string literal, else None.

    Starting and ending with a quote is not enough: `'a ' + x + ' b'` does
    too, and reading it as a literal measured the JavaScript rather than the
    words on screen. The closing quote has to be the last character.
    """
    if len(text) < 2 or text[0] not in "\"'" or text[-1] != text[0]:
        return None
    quote, i = text[0], 1
    while i < len(text) - 1:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return None
        i += 1
    return text[1:-1]


def test_the_small_panel_sub_labels_fit_the_rail_they_share():
    """Prevents the defect that read "pehisttoryof rank" on screen.

    The left rail is 300px and its grids put three cells across it, so each
    sub-label has roughly a third of that. Longer wording does not wrap, it
    overlaps its neighbour — and an overlapping label is worse than a terse
    one, because it is unreadable rather than merely brief.

    The budget: 300px of rail, less the panel's padding, over three cells is
    about 90px each; at the 10px the sub-label is set in that is roughly 18
    characters. Measured, not guessed — the three labels that overlapped were
    20, 22 and 26 characters and the ones beside them at 17 did not.
    """
    labels = _cell_sub_labels(APP_JS)
    assert len(labels) > 10, (
        f"the parser found only {len(labels)} sub-labels; it has stopped "
        f"reading the file it is meant to be checking")
    offenders = [label for label in labels if len(label) > 18]
    assert not offenders, (
        f"sub-labels too long for the 300px rail: {offenders}")


def test_the_theme_states_its_contrast_reasoning():
    """The colours here were solved against a measured background rather than
    picked, and the file has to say so — otherwise the next person to adjust
    them by eye undoes it without knowing."""
    assert "4.5:1" in STYLES or "4.5" in STYLES
    assert "--dimmer" in STYLES


def test_the_orb_is_loaded_as_a_module_with_its_import_map_first():
    """An import map has to be in the document before the module that needs it.

    The vendored three.js addons import from a bare "three" specifier. The
    browser resolves that through the import map, and a map declared after the
    module that triggers the resolution is ignored -- so the orb would fail to
    load with a specifier error and the centre panel would simply be empty,
    with one line in a console nobody has open.
    """
    map_at = INDEX.find('type="importmap"')
    orb_at = INDEX.find('src="/static/orb.boot.js"')
    assert map_at != -1, "no import map; the bare \"three\" specifier cannot resolve"
    assert orb_at != -1, "the page never loads the orb"
    assert map_at < orb_at, "the import map is declared after the module that needs it"
    assert 'type="module"' in INDEX[max(0, orb_at - 120):orb_at], (
        "orb.boot.js is loaded as a classic script; its imports would throw")


def test_the_page_still_carries_what_just_fired_as_text():
    """The orb replaced a 2D field that named the symbol that fired.

    A three-dimensional body cannot spell a ticker, so the names moved into
    the DOM beside it. This is also the entire panel on a machine with no
    working WebGL, which is why it is ordinary markup and not drawn into the
    canvas.
    """
    assert 'id="orb-ticker"' in INDEX
    assert "orb-tick" in APP_JS, "nothing ever populates the ticker strip"
    assert "window.__orbError" in APP_JS, (
        "the page never reports a failed orb; a blank panel with no "
        "explanation is the failure this terminal exists to prevent")
