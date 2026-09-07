"""Entry point for the packaged executable.

Double-clicking this must either open a working terminal or print a reason.
Anything that fails here fails before a browser is opened, so the operator never
sees a connection error for a program that is about to work -- or a browser
pointed at nothing for a program that is not.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback


def _pause() -> None:
    """Hold the console open so the operator can read the error.

    Only when there is a real terminal. On a CI runner stdin is closed, and
    input() then raises EOFError -- which buries the actual failure under an
    unrelated traceback and makes a clear packaging error look like a crash.
    """
    if not sys.stdin or not sys.stdin.isatty():
        return
    try:
        input("\nPress Enter to close...")
    except (EOFError, KeyboardInterrupt):
        pass


def main() -> int:
    # Required before anything else on Windows: without it a frozen one-file
    # build re-executes itself for every child process, which looks like the
    # program launching itself in an infinite loop.
    multiprocessing.freeze_support()

    try:
        from godalgo import config, logging_setup
        from godalgo.server.app import run_server
    except Exception:
        traceback.print_exc()
        print("\nGODALGO could not start: its own modules failed to import.")
        print("This is a packaging fault, not a configuration problem.")
        _pause()
        return 1

    logging_setup.configure()
    try:
        config.ensure_home()
        # CI launches this to verify the build; opening a browser on a headless
        # runner is at best noise and at worst a hang.
        open_browser = not os.environ.get("GODALGO_NO_BROWSER")
        return run_server(host="127.0.0.1", port=config.DEFAULT_PORT,
                          open_browser=open_browser)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(f"\nGODALGO stopped: {exc}")
        _pause()
        return 1


if __name__ == "__main__":
    sys.exit(main())
