"""Entry point for the packaged executable.

Double-clicking this must either open a working terminal or leave a readable
reason behind. The second half matters more than it sounds: a console window
that appears and vanishes takes the error with it, and "it's not running" is
then unanswerable by anyone, including the person who wrote it.

So, before anything else can fail, this writes everything it prints to a log
file on disk. Whatever happens, there is evidence afterwards.
"""

from __future__ import annotations

import datetime
import multiprocessing
import os
import sys
import traceback

LOG_NAME = "imperium-startup.log"


def _log_path() -> "os.PathLike[str] | str":
    """Somewhere writable to leave evidence, chosen without importing anything.

    Beside the executable first, because that is where someone will look. If
    that is read-only -- Program Files, a network share, a mounted image -- fall
    back to the user profile rather than failing to log at all.
    """
    import pathlib

    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(pathlib.Path(sys.executable).parent)
    candidates.append(pathlib.Path.home() / ".imperium")
    candidates.append(pathlib.Path(os.environ.get("TEMP", ".")))

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / LOG_NAME
            with open(probe, "a", encoding="utf-8"):
                pass
            return probe
        except Exception:
            continue
    return LOG_NAME


class _Tee:
    """Write to the console and to the log file at once.

    Deliberately forgiving: a frozen GUI-less process can have a broken or
    absent stdout, and losing the log because the console went away would defeat
    the point.
    """

    def __init__(self, stream, handle) -> None:
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        try:
            if self._stream is not None:
                self._stream.write(text)
        except Exception:
            pass
        try:
            self._handle.write(text)
            self._handle.flush()
        except Exception:
            pass
        return len(text)

    def flush(self) -> None:
        for target in (self._stream, self._handle):
            try:
                if target is not None:
                    target.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        try:
            return bool(self._stream is not None and self._stream.isatty())
        except Exception:
            return False


def _start_logging():
    try:
        handle = open(_log_path(), "a", encoding="utf-8", errors="replace")
    except Exception:
        return None
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    handle.write(f"\n{'=' * 70}\n{stamp}  IMPERIUM starting\n")
    handle.write(f"  executable: {sys.executable}\n")
    handle.write(f"  frozen:     {getattr(sys, 'frozen', False)}\n")
    handle.write(f"  argv:       {sys.argv}\n")
    handle.write(f"  python:     {sys.version.split()[0]}\n")
    handle.flush()
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    return handle


def parse_args():
    """Arguments for the packaged build.

    Deliberately small. There is no --host: the bind address is a rule, not an
    option, because this process holds API keys and has no authentication.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="IMPERIUM",
        description=("IMPERIUM trading terminal — built by Quincy Gininda. "
                     "Serves a local web UI on 127.0.0.1."),
        epilog=("Examples:\n"
                "  IMPERIUM.exe                  start on the default port and "
                "open a browser\n"
                "  IMPERIUM.exe --port 9000      start on port 9000\n"
                "  IMPERIUM.exe --no-browser     start without opening a browser\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("IMPERIUM_PORT", 0)) or None,
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
        import imperium

        print(f"IMPERIUM {imperium.__version__} — built by Quincy Gininda")
        raise SystemExit(0)
    return args


def config_default_port() -> int:
    from imperium import config

    return config.DEFAULT_PORT


def _pause() -> None:
    """Hold the console open so the operator can read the error.

    Skipped where there is no terminal -- on a CI runner stdin is closed and
    input() raises EOFError, which buries the actual failure under an unrelated
    traceback.
    """
    if os.environ.get("IMPERIUM_NO_PAUSE"):
        return
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return
    except Exception:
        return
    try:
        input("\nPress Enter to close...")
    except (EOFError, KeyboardInterrupt):
        pass


def _report(message: str, log_handle) -> None:
    print("")
    print("-" * 70)
    print(message)
    if log_handle is not None:
        try:
            print(f"A full log was written to: {_log_path()}")
        except Exception:
            pass
    print("-" * 70)


def main() -> int:
    # Required before anything else on Windows: without it a frozen one-file
    # build re-executes itself for every child process, which looks like the
    # program launching itself in an infinite loop.
    multiprocessing.freeze_support()

    log_handle = _start_logging()

    try:
        from imperium import config, logging_setup
        from imperium.server.app import run_server
    except Exception:
        traceback.print_exc()
        _report("IMPERIUM could not start: its own modules failed to import.\n"
                "This is a packaging fault, not a configuration problem.",
                log_handle)
        _pause()
        return 1

    try:
        args = parse_args()
    except SystemExit as exc:          # --help, --version, or a bad argument
        return int(exc.code or 0)

    logging_setup.configure()
    try:
        config.ensure_home()
        # CI launches this to verify the build; opening a browser on a headless
        # runner is at best noise and at worst a hang.
        open_browser = not (args.no_browser or os.environ.get("IMPERIUM_NO_BROWSER"))
        return run_server(host="127.0.0.1", port=args.port,
                          open_browser=open_browser)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0
    except Exception as exc:
        traceback.print_exc()
        _report(f"IMPERIUM stopped: {exc}", log_handle)
        _pause()
        return 1
    finally:
        if log_handle is not None:
            try:
                log_handle.flush()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
