# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the one-FOLDER Windows build.

This is the build people should actually run, and it exists because the
one-file build is hostile to Windows in three specific ways:

* A one-file executable unpacks its whole ~27MB payload into %TEMP% on **every**
  launch. Defender scans that unpack each time, which is slow at best and a
  quarantine at worst -- PyInstaller one-file binaries are a well-known source
  of antivirus false positives precisely because "extract a pile of DLLs to temp
  and execute them" is also what malware does.
* A bare .exe download is frequently blocked outright by the browser before the
  user ever sees a file.
* When it is blocked or killed, nothing is left behind to explain why.

A folder build executes its DLLs from where they were extracted once, by the
user, and ships inside a .zip, which browsers do not treat as dangerous. It also
starts noticeably faster because there is no per-launch unpack.

The one-file spec is kept alongside for people who want a single file and are
willing to allow it through Defender.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"

datas = [
    # A build that silently loses these starts, serves the API, and 404s its own
    # page. verify_build.py catches that before a release is published.
    (str(SRC / "godalgo" / "server" / "static"), "godalgo/server/static"),
    # The measured thresholds. Without them the classifier refuses to run, which
    # is correct behaviour and a terrible first-run experience.
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
    [],
    exclude_binaries=True,          # the difference: binaries live beside it
    name="GODALGO",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GODALGO",
)
