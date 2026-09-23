"""The orb's front-end checks, run from the Python suite.

Two layers, because they need different things.

``orb_checks.mjs`` covers the arithmetic under Node: damping, mode hysteresis,
the particle distribution, symbol placement, activity scaling, the heartbeat.
None of it needs a GPU and all of it is invisible in a screenshot -- damping
that is frame-rate dependent looks correct at 60fps and wrong at 144.

The rest -- that the shaders compile, that the modes actually deform the body,
that disposal releases the context -- needs a real WebGL context, so it lives
in ``tests/browser/orb_browser.py`` and runs only where Playwright and a
browser are installed. It is skipped rather than failed elsewhere: a developer
without a browser should still be able to run the suite.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

CHECKS = Path(__file__).parent / "js" / "orb_checks.mjs"
STATIC = Path(__file__).resolve().parents[1] / "src" / "imperium" / "server" / "static"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not available to run the front-end checks")
def test_the_orb_arithmetic_holds():
    """Every property that decides whether the orb reads as alive.

    Prevents, in one run: frame-rate dependent damping (the orb moving at a
    different speed on a different monitor); mode bands with no overlap (a
    reading on the boundary flipping the body between two morph states several
    times a second); a clumped particle skin (which reads as a defect in the
    mesh rather than as texture); a symbol whose ripple position is assigned
    on arrival rather than derived from its name (so the spatial information
    the 2D field carried is silently lost); every process kind normalised
    against one denominator (which makes one order a day indistinguishable
    from none); and a single-sine heartbeat (a pulsing light, not a chest).
    """
    proc = subprocess.run(["node", str(CHECKS)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert proc.stdout.strip(), f"no output from the checks: {proc.stderr[-1200:]}"
    results = json.loads(proc.stdout)
    failed = [r for r in results if not r["pass"]]
    assert not failed, "\n".join(f"{r['name']}: {r['detail']}" for r in failed)
    assert len(results) >= 25, f"only {len(results)} checks ran"


def test_the_shader_defines_bound_every_glsl_loop():
    """A GLSL loop needs a constant bound, so the array limits are #defines.

    If a shader indexes an array whose size is a #define the component does not
    also declare, it fails to compile at runtime with "undeclared identifier"
    and three renders the other layers -- which looks exactly like the layer
    was never added rather than like an error. That happened twice while this
    was being built, once for MAX_RIPPLES and once because assigning to
    `material.defines` dropped the ones three.js had put there itself.
    """
    shaders = (STATIC / "orb.shaders.js").read_text(encoding="utf-8")
    component = (STATIC / "orb.js").read_text(encoding="utf-8")

    used = set()
    for name in ("MAX_PROCESSES", "MAX_RIPPLES"):
        if name in shaders:
            used.add(name)
    assert used, "the shaders declare no bounded arrays at all"

    for name in sorted(used):
        assert f"{name}: {name}" in component or f"{name}," in component, (
            f"{name} is used in GLSL but never passed as a define")
        assert f"export const {name}" in component, (
            f"{name} has no single definition in the component")


def test_the_physical_material_defines_are_merged_not_replaced():
    """Guards a fix that is one character from being undone.

    Assigning to `material.defines` on a three.js material silently drops the
    flags three put there itself. On a MeshPhysicalMaterial that removed
    PHYSICAL, which compiles a physical material without its physical branch,
    and its own transmission code then failed to build against a struct that
    no longer had the field it wanted.

    The shell is a plain ShaderMaterial now and owns its defines outright, so
    this only has to hold for anything that reaches for a stock material.
    """
    component = (STATIC / "orb.js").read_text(encoding="utf-8")
    for line in component.splitlines():
        stripped = line.strip()
        if ".defines =" in stripped and not stripped.startswith("//"):
            pytest.fail(
                f"assigning to .defines drops three's own flags: {stripped!r}; "
                f"use Object.assign, or pass defines to the constructor")


def test_three_is_vendored_rather_than_fetched_from_a_cdn():
    """This ships as an offline Windows executable.

    A <script src="https://cdn..."> would leave the orb dead on any machine
    without internet, and the rest of the terminal is built to work without
    one -- the whole diagnostics layer exists to say so when the network is
    the problem. The vendored copy is inside static/, which the PyInstaller
    spec bundles wholesale, so it ships automatically.
    """
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "cdn.jsdelivr" not in page and "unpkg.com" not in page, (
        "the page loads a script from a CDN; the packaged build has no network")
    assert (STATIC / "vendor" / "three" / "three.module.js").exists()
    # The bare "three" specifier the vendored addons use has to resolve.
    assert 'type="importmap"' in page
    assert '"three": "/static/vendor/three/three.module.js"' in page


def test_the_orb_reads_the_terminals_own_palette():
    """One palette, not two.

    The orb's colours have to be the same object the legend paints from and
    the same set the server validates, or a colour changed in one place goes
    stale in another and the legend starts explaining a field it disagrees
    with.
    """
    boot = (STATIC / "orb.boot.js").read_text(encoding="utf-8")
    assert "window.Palette" in boot, "the bridge does not read the shared palette"
    orb = (STATIC / "orb.js").read_text(encoding="utf-8")
    # The component takes colours as arguments; it must not name any itself.
    for kind in ("#39ff8c", "#be3cff", "#ff9628", "#ff5c6c"):
        assert kind not in orb, (
            f"{kind} is hardcoded in the component; colours come from the "
            f"palette through setProcesses")


def test_the_pulse_kinds_and_the_palette_agree():
    """A kind the server can emit and the palette has never heard of renders
    untinted, and nothing says so."""
    from imperium.telemetry.streams import PULSE_KINDS

    palette = (STATIC / "palette.js").read_text(encoding="utf-8")
    for kind in sorted(PULSE_KINDS):
        assert f"{kind}:" in palette, (
            f"the server can emit a {kind!r} pulse but the palette has no "
            f"colour for it")

    import re
    listed = set(re.findall(r"^\s{4}(\w+):\s*\[", palette, re.M))
    assert listed == set(PULSE_KINDS), (
        f"palette has {sorted(listed - set(PULSE_KINDS))} the server cannot "
        f"emit, and is missing {sorted(set(PULSE_KINDS) - listed)}")


# -- can the kinds be told apart? -----------------------------------------
#
# From the operator, unprompted: "it is difficult to tell between scan and
# refused, and warmup and order look almost similar." Both were true and both
# were measurable. Converted to CIELAB, the palette that reading came from had
# its closest pair at dE 25.8 (order against warmup) and its second at 42.6
# (scan against refused) -- the two named, in the order they were named.
#
# A colour difference is not a matter of taste that has to be re-argued every
# time someone changes a hex value. It is a distance, so it is asserted like
# one.


def _lab(rgb):
    """sRGB 0-255 to CIELAB, D65."""
    import math

    def linear(u):
        u = u / 255
        return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(v) for v in rgb)
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    def f(t):
        return math.pow(t, 1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x / 0.95047), f(y / 1.0), f(z / 1.08883)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def _palette_colours() -> dict[str, list[int]]:
    import re

    text = (STATIC / "palette.js").read_text(encoding="utf-8")
    body = text.split("var KIND_COLOR = {", 1)[1].split("};", 1)[0]
    out = {}
    for name, triple in re.findall(r"(\w+):\s*\[([^\]]+)\]", body):
        out[name] = [int(v) for v in triple.split(",")]
    return out


def _floor() -> float:
    import re

    text = (STATIC / "palette.js").read_text(encoding="utf-8")
    found = re.search(r"var MIN_SEPARATION = ([\d.]+);", text)
    assert found, "the palette no longer states the floor it was built to"
    return float(found.group(1))


def test_no_two_process_colours_are_hard_to_tell_apart():
    import itertools
    import math

    colours = _palette_colours()
    floor = _floor()
    assert len(colours) >= 7, colours

    worst = sorted(
        (math.dist(_lab(colours[a]), _lab(colours[b])), a, b)
        for a, b in itertools.combinations(colours, 2)
    )
    distance, one, two = worst[0]
    assert distance >= floor, (
        f"{one} and {two} are {distance:.1f} apart in CIELAB, under the "
        f"palette's own floor of {floor}. That is the fault the operator "
        f"reported: two kinds that read as the same colour on the orb.\n"
        + "\n".join(f"  {d:6.1f}  {x} / {y}" for d, x, y in worst[:4]))


def test_scan_stays_the_quietest_thing_on_the_orb():
    """Scan is by far the commonest pulse -- tens a second on a full sweep --
    so whatever colour it is, that is the colour of the terminal. It has to
    stay the least saturated, or the display is never monochrome and nothing
    that happens can stand out from it."""
    colours = _palette_colours()

    def chroma(rgb):
        _, a, b = _lab(rgb)
        return (a * a + b * b) ** 0.5

    scan = chroma(colours["scan"])
    for kind, rgb in colours.items():
        if kind == "scan":
            continue
        assert chroma(rgb) > scan, (
            f"{kind} is no more colourful than the background hum "
            f"({chroma(rgb):.1f} against scan's {scan:.1f})")


def test_the_legend_dot_is_lit_by_its_own_colour():
    """A flat 6px dot on a dark panel is a shade. The same dot with its colour
    bleeding out of it is a light, and a light is what the eye picks a kind out
    by at a glance."""
    text = (STATIC / "palette.js").read_text(encoding="utf-8")
    assert "boxShadow" in text, "the legend dots are unlit"
    assert "kind === 'scan'" in text, (
        "the hum glows as brightly as the things that matter, which makes the "
        "legend as loud as the orb")
