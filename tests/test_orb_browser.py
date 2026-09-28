"""The orb, checked in a real WebGL context.

Everything here is a property the source can claim and the GPU can still
refuse. A shader that fails to compile does not raise -- three logs an error
and carries on rendering the layers that did compile, which looks exactly like
the layer having never been added. A morph state that sets its uniform but
deforms nothing looks fine in code and identical on screen. So these measure
rendered pixels.

Skipped where Playwright or a browser is missing, rather than failed: a
developer without one should still be able to run the suite, and the
arithmetic is covered under Node in test_orb_js.py either way.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHROMIUM = "/opt/pw-browsers/chromium"

pytestmark = pytest.mark.skipif(
    not Path(CHROMIUM).exists(),
    reason="no browser available to render the orb")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def _terminal():
    """A real server, because the orb is loaded as a module from /static.

    A file:// page cannot use an import map to resolve the bare "three"
    specifier the vendored addons import, so there is no way to test this
    without serving it the way it actually ships.
    """
    port = _free_port()
    env = dict(os.environ, IMPERIUM_NO_BROWSER="1")
    proc = subprocess.Popen(
        [sys.executable, "-m", "imperium.cli", "serve",
         "--no-browser", "--port", str(port)],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError("the terminal exited before serving")
                time.sleep(0.25)
        else:
            raise RuntimeError("the terminal did not start in time")
        yield port
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


@pytest.fixture(scope="module")
def probe():
    pytest.importorskip("playwright.async_api")
    sys.path.insert(0, str(Path(__file__).parent / "browser"))
    from orb_browser import probe as run_probe        # noqa: E402

    with _terminal() as port:
        normal = asyncio.run(run_probe(port))
        reduced = asyncio.run(run_probe(port, reduced=True))
    return {"normal": normal, "reduced": reduced}


def test_every_layer_renders_and_no_shader_fails_to_compile(probe):
    """A failed shader is not an exception, it is a missing layer.

    Three logs the compile error and renders whatever else built, so the only
    way to know the glass membrane is there is to look for it. Two shader bugs
    got this far during development -- an undeclared array bound, and defines
    dropped by assigning over three's own -- and both presented as "the shell
    seems to have no effect".
    """
    normal = probe["normal"]
    assert normal["errors"] == [], f"console errors: {normal['errors']}"
    assert normal["webgl"] is True
    layers = normal["layers"]
    assert layers["core"], "the liquid core did not render"
    assert layers["shells"] >= 1, "no glass membrane rendered"
    assert layers["skin"], "the particle skin did not render"
    assert layers["shadow"], "the contact shadow did not render"
    assert layers["bloom"], "bloom is not in the pipeline"


def test_the_four_modes_produce_four_different_bodies(probe):
    """Each mode must actually deform the mesh, not merely set a uniform."""
    heights = {name: shape["top"] - shape["bottom"]
               for name, shape in probe["normal"]["shapes"].items()}
    assert len(set(heights.values())) == 4, (
        f"two modes render an identical silhouette: {heights}")
    assert heights["intense"] > heights["idle"], (
        f"intense should push lobes outward: {heights}")


def test_intense_is_lumpier_than_idle(probe):
    assert probe["normal"]["intenseIsLumpier"], (
        f"intense did not widen the body: {probe['normal']['shapes']}")


def test_alert_melts_downward_rather_than_shrinking(probe):
    """The bug this caught, kept as a test.

    The alert state was written as a negative term in the shared displacement
    field -- but that field scales the surface along its own radius, so a
    negative value on the lower hemisphere pulls it *inward*. The orb got
    smaller when it was meant to be dripping. Gravity does not act along a
    surface normal, so the melt is a vector offset now.
    """
    shapes = probe["normal"]["shapes"]
    assert probe["normal"]["meltsDownward"], (
        f"alert did not reach lower than idle: idle bottom "
        f"{shapes['idle']['bottom']}, alert bottom {shapes['alert']['bottom']}")
    assert (shapes["alert"]["top"] - shapes["alert"]["bottom"]) > \
           (shapes["idle"]["top"] - shapes["idle"]["bottom"]), \
           "alert should stretch the body, not compress it"


def test_it_never_looks_frozen(probe):
    """Two frames a second and a half apart must differ.

    The one thing this panel must never do is look stopped, because a stopped
    animation and a stopped trading loop are indistinguishable at a glance --
    which is the exact confusion the whole terminal is built to prevent.
    """
    assert probe["normal"]["movesWhenIdle"], (
        "the orb rendered identical frames while idle")


def test_a_process_event_raises_a_ripple_and_the_slots_are_bounded(probe):
    normal = probe["normal"]
    assert normal["ripplesFired"] == 3, (
        f"three events should raise three ripples, got {normal['ripplesFired']}")
    assert normal["rippleSlotsBounded"], (
        "a burst overran the ripple slots; GLSL array bounds are fixed and "
        "writing past them is undefined")


def test_the_dominant_colour_takes_over_the_body(probe):
    """The takeover has to actually complete, not merely start."""
    tint = probe["normal"]["dominantTint"]
    target = probe["normal"]["uColor0"]
    for channel, (got, want) in enumerate(zip(tint, target)):
        assert abs(got - want) < 0.06, (
            f"channel {channel} reached {got:.3f}, wanted {want:.3f}; the "
            f"dominant colour did not finish spreading through the core")


def test_a_weak_process_stays_visible_while_it_is_running(probe):
    """A process that is alive and invisible is a lie about what is running."""
    assert probe["normal"]["weakProcessWeight"] > 0, (
        "a running process was given no weight at all")


def test_the_pixel_ratio_is_capped(probe):
    assert probe["normal"]["pixelRatio"] <= 2, (
        f"devicePixelRatio was not capped: {probe['normal']['pixelRatio']}")


def test_disposal_releases_the_gpu_context(probe):
    """A panel that is torn down and rebuilt must not leak a context.

    Browsers allow a small, fixed number of live WebGL contexts and drop the
    oldest when that is exceeded -- so a leak here does not show up as a leak,
    it shows up as some *other* canvas on the page going blank.
    """
    disposed = probe["normal"]["dispose"]
    assert disposed["before"]["core"], "nothing was there to dispose"
    assert disposed["disposedFlag"] is True
    assert disposed["contextLost"] is True, "the WebGL context was not released"
    assert disposed["frameAfterDispose"] == "ignored", (
        "a disposed orb threw when the host's animation loop called it again; "
        "that loop runs whether or not this panel still does")


def test_reduced_motion_keeps_the_colour_and_cuts_the_movement(probe):
    """The brief's rule, and the right one: someone who asked for less motion
    asked for less motion, not for less information."""
    reduced = probe["reduced"]
    assert reduced["reducedMotion"]["reduced"] is True
    assert reduced["reducedMotion"]["amp"] < 0.5, (
        "reduced motion did not cut the morph amplitude")
    assert reduced["ripplesFired"] == 0, (
        "surface ripples still fire under reduced motion")

    # The colour is information, so it must survive untouched.
    tint = reduced["dominantTint"]
    target = reduced["uColor0"]
    for got, want in zip(tint, target):
        assert abs(got - want) < 0.06, (
            "reduced motion lost the dominant colour, which is information "
            "rather than movement")

    # And the body still moves, just far less.
    heights = {k: v["top"] - v["bottom"] for k, v in reduced["shapes"].items()}
    spread_reduced = max(heights.values()) - min(heights.values())
    full = {k: v["top"] - v["bottom"] for k, v in probe["normal"]["shapes"].items()}
    spread_full = max(full.values()) - min(full.values())
    assert spread_reduced < spread_full, (
        f"reduced motion morphed as much as full motion: "
        f"{spread_reduced} vs {spread_full}")
