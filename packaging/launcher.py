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


def parse_args() -> "argparse.Namespace":
    """Arguments for the double-clickable build.

    Deliberately small. There is no --host: the bind address is a rule, not an
    option, because this process holds API keys and has no authentication.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="GODALGO",
        description=("GODALGO trading terminal — built by Quincy Gininda. "
                     "Serves a local web UI on 127.0.0.1."),
        epilog=("Examples:\n"
                "  GODALGO.exe                  start on the default port and "
                "open a browser\n"
                "  GODALGO.exe --port 9000      start on port 9000\n"
                "  GODALGO.exe --no-browser     start without opening a browser\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("GODALGO_PORT", 0)) or None,
        help=f"port to serve on (default {config_default_port()}; if it is busy, "
             f"the next free port above it is used)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser on start")
    parser.add_argument("--version", action="store_true",
                        help="print the version and exit")
    args = parser.parse_args()
    if args.port is None:
        args.port = config_default_port()
    if not (1 <= args.port <= 65535):
        parser.error(f"--port must be between 1 and 65535, not {args.port}")
    if args.version:
        import godalgo

        print(f"GODALGO {godalgo.__version__} — built by Quincy Gininda")
        raise SystemExit(0)
    return args


def config_default_port() -> int:
    from godalgo import config

    return config.DEFAULT_PORT


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

    args = parse_args()

    logging_setup.configure()
    try:
        config.ensure_home()
        # CI launches this to verify the build; opening a browser on a headless
        # runner is at best noise and at worst a hang.
        open_browser = not (args.no_browser or os.environ.get("GODALGO_NO_BROWSER"))
        return run_server(host="127.0.0.1", port=args.port,
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
