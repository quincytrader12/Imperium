"""Verify a packaged build actually works, before it is published.

A build that lost its static files starts, serves the API, and 404s its own
page. `pyinstaller` reports success for that build, so CI has to check the thing
itself: launch the executable, wait for it to answer, and require that the page,
the API, the diagnostics endpoint and the shipped calibration are all really
there.
"""

from __future__ import annotations

import math
import posixpath
import re
import subprocess
import sys
import tempfile
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
    # Compared without the cache-busting token, fetched with it.
    #
    # The page stamps every asset URL with a content hash, so the markup says
    # "/static/app.js?v=e241e4cddcb5" and an exact membership test against
    # "/static/app.js" fails on a build where every one of those files was
    # served perfectly. Which is what happened: four "is not referenced by the
    # page at all" failures on a run whose own log showed each file fetched.
    #
    # The token still has to be exercised, so the graph walk below keeps the
    # full URL. Only this presence check drops it.
    named = {without_token(a) for a in assets}
    for required in ("/static/app.js", "/static/styles.css",
                     "/static/orb.boot.js", "/static/palette.js"):
        if required not in named:
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


def without_token(url: str) -> str:
    """An asset URL with its cache-busting query removed."""
    return url.split("?", 1)[0]


def module_imports(asset: str, source: str) -> list[str]:
    """The relative imports of one module, as absolute /static paths.

    Bare specifiers such as "three" are skipped: those resolve through the
    page's import map, whose targets are already collected from the markup.
    """
    base = posixpath.dirname(without_token(asset))
    found = re.findall(r"""(?:from|import)\s*\(?\s*['"](\.[^'"]+)['"]""",
                       source)
    return [posixpath.normpath(posixpath.join(base, spec)) for spec in found]


def synthetic_bars(folder: Path, symbols=("AAA", "BBB", "CCC"),
                   days: int = 420) -> None:
    """Bars with trends in them, written where the backtest can read them.

    Synthetic rather than real for two reasons. A build check must not need a
    credential or a network, and it must not depend on what the market did --
    a check whose result moves with SPY is not a check. These are generated
    from a fixed formula, so every run of this script measures the same thing.

    Trending on purpose: a flat series takes no trades, and a backtest that
    takes no trades exercises almost none of the code this is verifying. Each
    symbol gets a different period and phase so they break out at different
    times, which is what makes the sizing and the leverage cap do any work.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for offset, symbol in enumerate(symbols):
        period = 70 + 23 * offset
        price = 100.0 + 10.0 * offset
        lines = ["date,close"]
        for day in range(days):
            # A slow cycle for the trend, a fast one for the noise the bands
            # have to see through. Deterministic, so a failure reproduces.
            price *= (1.0 + 0.004 * math.sin(2 * math.pi * day / period)
                      + 0.0015 * math.sin(day * 1.7 + offset))
            stamp = f"{2015 + day // 252:04d}-{(day % 252) // 21 + 1:02d}-" \
                    f"{(day % 21) + 1:02d}"
            lines.append(f"{stamp},{price:.4f}")
        (folder / f"{symbol}.csv").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")


def check_backtest(exe: Path, failures: list[str]) -> None:
    """The backtest must run from the build, not only from a checkout.

    It used to live in scripts/, which PyInstaller does not bundle, so the
    packaged terminal shipped a strategy with no way to measure it -- and
    "arm the sleeve and find out" is not an acceptable substitute. This runs
    the command the shipped .bat runs, on bars generated here, and requires a
    finished report.

    Worth its own check rather than folding into the import graph: numpy is
    the largest thing in the bundle and the only place it is reached from is
    this command, so an `excludes` entry or a hook change could drop it and
    every other check here would still pass.
    """
    import os

    with tempfile.TemporaryDirectory() as tmp:
        bars = Path(tmp) / "bars"
        synthetic_bars(bars)
        try:
            done = subprocess.run(
                [str(exe), "--backtest", "--csv", str(bars)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", timeout=600,
                env={**os.environ, "IMPERIUM_NO_PAUSE": "1"})
        except subprocess.TimeoutExpired:
            failures.append("the backtest did not finish within 10 minutes")
            return

    tail = (done.stdout or "")[-3000:]
    if done.returncode != 0:
        failures.append(
            f"--backtest exited {done.returncode}:\n{tail}")
        return
    for expected in ("Final equity", "Max drawdown", "near_close",
                     "next_open"):
        if expected not in done.stdout:
            failures.append(
                f"the backtest report is missing {expected!r}:\n{tail}")
            return
    print("  --backtest ran and produced a report")


#: The longest a build may take to answer, in seconds.
#:
#: Derived from the program, not chosen: ``_open_when_ready`` gives the server
#: this long to respond before it gives up and does not open a browser, so a
#: build slower than this is one whose own launcher has already declared it
#: broken. Measured here because it is not a property any unit test can see.
#:
#: One build spent six seconds of its startup opening 605 bundled time zone
#: files. On a Windows machine, where Defender scans each newly extracted file
#: on first launch, that was enough to pass the limit -- and the operator saw
#: "server did not answer within 20s, not opening browser".
STARTUP_BUDGET = 20.0


def wait_for_server(proc: subprocess.Popen, timeout: float = 90.0) -> float:
    """Block until the build answers, and return how long that took."""
    started = time.time()
    deadline = started + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out = (proc.stdout.read() if proc.stdout else "") or ""
            raise SystemExit(
                f"the executable exited with code {proc.returncode} before "
                f"serving anything:\n{out[-4000:]}")
        try:
            status, _ = fetch("/api/health", timeout=2)
            if status == 200:
                return time.time() - started
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
        took = wait_for_server(proc)
        print(f"  server answered /api/health in {took:.1f}s")
        if took > STARTUP_BUDGET:
            failures.append(
                f"the build took {took:.1f}s to answer, past the "
                f"{STARTUP_BUDGET:.0f}s its own launcher waits before giving "
                f"up on opening a browser. An operator would see 'server did "
                f"not answer' and a window that never opens. This is usually "
                f"the bundle having grown: every file in it is scanned on "
                f"first launch.")

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

        check_backtest(path, failures)
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
          "its diagnostics, and it can backtest.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "dist/IMPERIUM.exe"))
