# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the one-file Windows build.

The static files are the whole point of the --add-data below. A build that
silently loses them starts, serves the API, and 404s its own page -- which looks
like a broken program rather than a broken build. `godalgo.server.app.static_dir`
raises a specific error in that case, and `verify_build.py` catches it in CI
before a release is published.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"

datas = [
    # (source, destination inside the bundle)
    (str(SRC / "godalgo" / "server" / "static"), "godalgo/server/static"),
    # The measured thresholds. Without this the classifier refuses to run, which
    # is the correct behaviour but a terrible first-run experience.
    (str(SRC / "godalgo" / "strategy" / "null_calibration.json"), "godalgo/strategy"),
]

hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("websockets")
    + ["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
       "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on"]
)

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pytest", "PIL"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="GODALGO",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
