"""A new build must not render as the old one.

The symptom is indistinguishable from a failed install: the operator extracts
a new zip, starts it, refreshes, and the terminal is unchanged. What actually
happened is that ``/`` was regenerated from disk -- so the markup was current
-- while every asset it named was an unversioned URL served with no
Cache-Control at all. With no cache directive a browser may apply heuristic
freshness, and an ordinary refresh does not revalidate subresources; only a
hard reload does. New HTML, old JavaScript.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from imperium.server import app as app_mod
from imperium.server.app import create_app, version_assets
from imperium.session import TradingSession


def _client() -> TestClient:
    return TestClient(create_app(TradingSession()))


def test_every_asset_the_page_names_is_versioned():
    with _client() as client:
        body = client.get("/").text
    unversioned = [u for u in re.findall(r'["\'](/static/[^"\']+)["\']', body)
                   if "?v=" not in u]
    assert not unversioned, (
        f"these would be served from cache after an upgrade: {unversioned}")


def test_the_import_map_is_versioned_too():
    """It names files no tag points at.

    A hand-maintained list of assets to stamp would go stale exactly when a
    new module was added, which is the moment it matters.
    """
    with _client() as client:
        body = client.get("/").text
    three = re.search(r'"three":\s*"([^"]+)"', body)
    assert three and "?v=" in three.group(1), (
        "the vendored three.js would be pinned to whatever the browser "
        "already had")


def test_the_page_itself_is_never_cached():
    """It carries the token. A stale copy of the page pins the browser to a
    stale build however new the files on disk are."""
    with _client() as client:
        assert client.get("/").headers["cache-control"] == "no-store"


def test_a_versioned_asset_may_be_cached_forever():
    with _client() as client:
        body = client.get("/").text
        token = re.search(r"\?v=([0-9a-f]+)", body).group(1)
        headers = client.get(f"/static/app.js?v={token}").headers
    assert "immutable" in headers["cache-control"]


def test_an_unversioned_asset_must_be_revalidated():
    """The hole the markup rewrite cannot reach.

    The orb is ES modules, and ``import { ProcessOrb } from './orb.js'``
    resolves to a URL no rewrite of the page can touch. Left to heuristic
    caching, a new build renders an old orb.
    """
    with _client() as client:
        assert client.get("/static/orb.js").headers["cache-control"] == "no-cache"


def test_the_token_changes_when_a_file_changes(tmp_path, monkeypatch):
    """The whole point. A token that does not move is a cache that never
    clears, and it fails silently on exactly the build that needed it."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "app.js").write_text("console.log(1);", encoding="utf-8")
    monkeypatch.setattr(app_mod, "static_dir", lambda: static)

    monkeypatch.setattr(app_mod, "_ASSET_VERSION", "")
    before = app_mod.asset_version()

    (static / "app.js").write_text("console.log(2);", encoding="utf-8")
    monkeypatch.setattr(app_mod, "_ASSET_VERSION", "")
    after = app_mod.asset_version()

    assert before != after, "the same token for different bytes"


def test_the_token_is_stable_for_unchanged_files(tmp_path, monkeypatch):
    """Or every restart would re-download the whole bundle."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "app.js").write_text("console.log(1);", encoding="utf-8")
    monkeypatch.setattr(app_mod, "static_dir", lambda: static)

    monkeypatch.setattr(app_mod, "_ASSET_VERSION", "")
    first = app_mod.asset_version()
    monkeypatch.setattr(app_mod, "_ASSET_VERSION", "")
    assert app_mod.asset_version() == first


def test_index_html_is_not_part_of_the_token():
    """It is rewritten on the way out and never served from disk as-is, so
    including it would change the token for no reason a browser can see."""
    markup = '<script src="/static/app.js"></script>'
    assert version_assets(markup, "abc") == \
        '<script src="/static/app.js?v=abc"></script>'


def test_a_url_that_already_has_a_query_is_left_alone():
    markup = '<img src="/static/x.png?size=2">'
    assert version_assets(markup, "abc") == markup


# -- what the bundle may cost to start ---------------------------------------


def test_the_time_zone_database_is_not_bundled():
    """605 files for one time zone, and it broke the build that shipped it.

    Windows Defender scans every newly extracted file on a PyInstaller
    bundle's first launch. Adding tzdata took the file count from 132 to 782
    and first-launch startup from 0.7s to 6s here -- on Linux, with no
    antivirus at all. On the operator's machine it passed the twenty seconds
    the launcher waits, and they saw "server did not answer within 20s, not
    opening browser" on a terminal that never opened.

    US Eastern is computed from the statutory rule instead. That rule is
    checked hour by hour against the real database in test_daily_brief.py.
    """
    from pathlib import Path

    spec = (Path(__file__).resolve().parents[1]
            / "packaging" / "imperium-folder.spec").read_text(encoding="utf-8")
    assert '"tzdata"' not in spec.split("excludes=")[0], (
        "tzdata is a hidden import again; the bundle will grow six-fold")
    assert "tzdata" in spec.split("excludes=")[1].split("]")[0], (
        "nothing stops a dependency pulling tzdata back into the bundle")


def test_the_build_check_enforces_a_startup_budget():
    """The property no unit test can see.

    A build that is correct in every way and takes half a minute to answer is
    a build whose own launcher gives up on it. The budget is read from the
    launcher rather than chosen, so the two cannot drift apart.
    """
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "packaging"))
    from verify_build import STARTUP_BUDGET

    from imperium.server.app import _open_when_ready

    waits = _open_when_ready.__defaults__[0]
    assert STARTUP_BUDGET <= waits, (
        f"the build check allows {STARTUP_BUDGET}s but the launcher only "
        f"waits {waits}s, so a build can pass and still never open a window")
