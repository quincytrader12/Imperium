"""Verify a packaged build actually works, before it is published.

A build that lost its static files starts, serves the API, and 404s its own
page. `pyinstaller` reports success for that build, so CI has to check the thing
itself: launch the executable, wait for it to answer, and require that the page,
the API, the diagnostics endpoint and the shipped calibration are all really
there.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 8787
BASE = f"http://127.0.0.1:{PORT}"


def fetch(path: str, timeout: float = 5.0) -> tuple[int, str]:
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "verify"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def page_assets(body: str) -> list[str]:
    """Every static file the page itself asks the browser to load.

    Read out of the markup rather than kept as a list here. A hand-kept list is
    a second copy of the same fact, and this is the copy that goes stale: a new
    script the page needs would ship unverified, and because one missing file
    throws and stops the whole bundle, the terminal would come up blank on a
    build that reported success.
    """
    assets = re.findall(
        r'<(?:script|link)[^>]+(?:src|href)="(/static/[^"]+)"', body)

    # An import map names files no tag points at. The browser fetches them the
    # moment a module imports the bare specifier, so a build that lost the
    # vendored three.js would pass a tag-only check and then serve a terminal
    # whose centre panel is empty -- with one line in a console nobody has
    # open, which is the exact failure this script exists to catch.
    assets += re.findall(r'"(/static/vendor/[^"]+)"', body)

    # Deduped, order preserved: the same file can be named by a tag and by the
    # map, and reporting it twice makes a clean build look suspicious.
    seen: set[str] = set()
    unique = []
    for asset in assets:
        if asset not in seen:
            seen.add(asset)
            unique.append(asset)
    return unique


def check_assets(fetch, body: str, failures: list[str]) -> list[str]:
    """Fetch every static file the page asks for, recording what is missing.

    Separated from the run so it can be exercised directly. Left inline it was
    the one part of this script nothing could test: a version of this loop that
    checks nothing looks identical from the outside to one that checks
    everything -- both print a build that passed.
    """
    notes: list[str] = []
    assets = list(page_assets(body))
    for required in ("/static/app.js", "/static/styles.css",
                     "/static/orb.boot.js", "/static/palette.js"):
        if required not in assets:
            failures.append(f"{required} is not referenced by the page at all")
    if not assets:
        failures.append("the page references no static files at all")
    # Walked as a graph, not as a list.
    #
    # Half of this page's JavaScript is now ES modules, which pull their own
    # dependencies in with relative imports that appear in no tag. The orb's
    # shaders, its arithmetic and the three.js postprocessing addons are all
    # reached that way, so a check that only fetched what the markup names
    # would pass a build that had lost every one of them.
    queued = list(assets)
    seen = set(queued)
    while queued:
        asset = queued.pop(0)
        try:
            status, content = fetch(asset)
        except urllib.error.HTTPError as exc:
            failures.append(f"{asset} returned HTTP {exc.code} — the build "
                            f"lost its static files (--add-data)")
            continue
        if status != 200 or len(content) < 200:
            failures.append(f"{asset} returned HTTP {status}, {len(content)} bytes")
            continue
        notes.append(f"  served {asset} ({len(content)} bytes)")
        if asset.endswith(".js"):
            for imported in module_imports(asset, content):
                if imported not in seen:
                    seen.add(imported)
                    queued.append(imported)
    return notes


def module_imports(asset: str, source: str) -> list[str]:
    """The relative imports of one module, as absolute /static paths.

    Bare specifiers such as "three" are skipped: those resolve through the
    page's import map, whose targets are already collected from the markup.
    """
    base = posixpath.dirname(asset)
    found = re.findall(r"""(?:from|import)\s*\(?\s*['"](\.[^'"]+)['"]""",
                       source)
    return [posixpath.normpath(posixpath.join(base, spec)) for spec in found]


def wait_for_server(proc: subprocess.Popen, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out = (proc.stdout.read() if proc.stdout else "") or ""
            raise SystemExit(
                f"the executable exited with code {proc.returncode} before "
                f"serving anything:\n{out[-4000:]}")
        try:
            status, _ = fetch("/api/health", timeout=2)
            if status == 200:
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise SystemExit(f"the executable did not answer within {timeout:.0f}s")


def main(exe: str) -> int:
    path = Path(exe)
    if not path.exists():
        raise SystemExit(f"no executable at {path}")
    print(f"launching {path} ({path.stat().st_size / 1e6:.1f} MB)")

    proc = subprocess.Popen(
        [str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env={**__import__("os").environ, "IMPERIUM_NO_BROWSER": "1",
             "IMPERIUM_NO_PAUSE": "1"},
    )
    failures: list[str] = []
    try:
        wait_for_server(proc)
        print("  server answered /api/health")

        status, body = fetch("/")
        if status != 200:
            failures.append(f"the page returned HTTP {status}")
        elif "IMPERIUM" not in body:
            failures.append("the page did not contain the expected markup")
        else:
            print("  served its own page")

        # The exact failure this script exists for: --add-data silently dropped.
        for line in check_assets(fetch, body, failures):
            print(line)

        status, body = fetch("/api/snapshot")
        if status != 200 or '"watchlist"' not in body:
            failures.append("the snapshot endpoint did not return a watchlist")
        else:
            print("  snapshot endpoint answered")

        # The calibration must be bundled, or the classifier refuses to run.
        if '"calibration_error": ""' not in body.replace(", ", ", "):
            import json

            snap = json.loads(body)
            if snap.get("calibration_error"):
                failures.append(
                    "the shipped build has no null_calibration.json, so the "
                    "regime classifier will refuse to run: "
                    + snap["calibration_error"][:200])
            else:
                print("  calibration is bundled")
        else:
            print("  calibration is bundled")

        try:
            status, body = fetch("/diagnose", timeout=60)
            if status != 200 or "VERDICT:" not in body:
                failures.append("the diagnostics endpoint did not produce a verdict")
            else:
                print("  diagnostics produced a verdict")
        except Exception as exc:
            failures.append(f"the diagnostics endpoint failed: {exc}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    if failures:
        print("\nBUILD VERIFICATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nbuild verified: it serves its own page, its assets, its API and "
          "its diagnostics.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "dist/IMPERIUM.exe"))
