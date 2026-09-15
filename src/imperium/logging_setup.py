"""Logging that cannot leak credentials.

The brief says "no key material in any log at any level". Enforcing that by
convention means one careless f-string undoes it, so it is enforced
mechanically instead: every secret handed to the credential store is registered
here, and a filter scrubs every record -- message, args and exception text --
before any handler sees it.

Two independent layers, because either alone has a hole:

* the *registry* catches known secrets even when they are embedded in a larger
  string (a signed URL, a repr of a config object);
* the *patterns* catch secrets that were never registered, which is the case
  that matters -- a key the operator pasted into the wrong form and which we
  are about to echo back in an error.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from typing import Any, Iterable

_REDACTED = "[redacted]"

_lock = threading.Lock()
_secrets: set[str] = set()

#: Patterns for secret-shaped text that was never registered.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # A proxy or any URL carrying userinfo. A proxy URL embeds credentials and
    # this output gets pasted into bug reports.
    (re.compile(r"(?P<scheme>\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"),
     r"\g<scheme>" + _REDACTED + "@"),
    # Query-string secrets, wherever they turn up.
    (re.compile(r"(?i)\b(signature|secret|secretKey|apiKey|api_key|token|password)=[^&\s\"']+"),
     r"\1=" + _REDACTED),
    # The two headers Alpaca authenticates with. There is no request signing --
    # authentication *is* these headers -- so a logged request header is the
    # whole credential rather than a derived signature.
    (re.compile(r"(?i)(APCA-API-(?:KEY-ID|SECRET-KEY)['\"]?\s*[:=]\s*)\S+"),
     r"\1" + _REDACTED),
    # An Alpaca key id: twenty uppercase characters beginning PK (paper) or AK
    # (live). Short enough that the length rule below would never catch it,
    # which is exactly the gap this closes.
    (re.compile(r"\b[AP]K[A-Z0-9]{18}\b"), _REDACTED),
    # Anything else long enough to be a secret rather than prose. Forty
    # characters is set by Alpaca's secret length; the cost of the lower bound
    # is that a bare 40-character git SHA would also be redacted, which this
    # program never logs. Client order ids are unaffected: "imp-" breaks the
    # run, leaving 24 characters.
    (re.compile(r"\b[A-Za-z0-9]{40,}\b"), _REDACTED),
)

#: Below this length a "secret" is too short to scrub without mangling
#: unrelated log text (a two-character secret would redact half the alphabet).
_MIN_SCRUB_LEN = 8


def register_secret(secret: str | None) -> None:
    """Register a value that must never appear in a log record."""
    if not secret or len(secret) < _MIN_SCRUB_LEN:
        return
    with _lock:
        _secrets.add(secret)


def forget_secrets() -> None:
    """Drop the registry. Used by tests; never needed in normal operation."""
    with _lock:
        _secrets.clear()


def scrub(text: str) -> str:
    """Remove every known and secret-shaped value from ``text``."""
    if not text:
        return text
    with _lock:
        known = sorted(_secrets, key=len, reverse=True)
    for secret in known:
        if secret in text:
            text = text.replace(secret, _REDACTED)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def mask(value: str | None, keep_head: int = 4, keep_tail: int = 4) -> str:
    """Render a credential as a masked view: ``PK12…3456``.

    This is the *only* representation of a credential that may leave the
    process, over HTTP or into a log.

    ``keep_tail=0`` must reveal nothing. Slicing with ``value[-keep_tail:]``
    would return the *entire* string when ``keep_tail`` is 0, which is how a
    "masked" secret ends up published verbatim; the tail is therefore sliced
    only when it is non-empty.
    """
    if not value:
        return "(unset)"
    keep_head = max(0, keep_head)
    keep_tail = max(0, keep_tail)
    if keep_head + keep_tail == 0:
        return "•" * 8
    if len(value) <= keep_head + keep_tail:
        return "*" * len(value)
    head = value[:keep_head] if keep_head else ""
    tail = value[-keep_tail:] if keep_tail else ""
    return f"{head}…{tail}"


def describe_secret(value: str | None) -> str:
    """Describe a secret without revealing any of it.

    A secret has no useful masked form: the operator only needs to know that one
    is stored and how long it is, which is enough to spot a truncated paste.
    """
    if not value:
        return "(unset)"
    return f"•••••••• ({len(value)} chars)"


class RedactingFilter(logging.Filter):
    """Scrub every record. Installed on the root logger, so it covers
    third-party libraries (httpx logs URLs) as well as our own code.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Render first: scrubbing the format string alone misses a secret
            # that arrives through an argument.
            record.msg = scrub(record.getMessage())
            record.args = ()
        except Exception:  # pragma: no cover - never break logging
            record.msg = "[log record could not be rendered safely]"
            record.args = ()
        if record.exc_info:
            # An exception carrying a signed URL in its message would otherwise
            # bypass the filter entirely via the traceback.
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_value is not None:
                scrubbed_args = tuple(
                    scrub(a) if isinstance(a, str) else a for a in exc_value.args
                )
                try:
                    exc_value.args = scrubbed_args
                except Exception:  # pragma: no cover - immutable exception args
                    pass
        return True


def safe_env_names(names: Iterable[str]) -> list[str]:
    """Return proxy-ish environment variable *names* that are set.

    Names only, never values: a proxy URL embeds credentials, and this output is
    designed to be pasted into a bug report.
    """
    import os

    return sorted(n for n in names if os.environ.get(n))


_configured = False


def configure(level: int = logging.INFO, stream: Any = None) -> None:
    """Install the redacting filter on the root logger. Idempotent."""
    global _configured
    root = logging.getLogger()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
                          datefmt="%H:%M:%S")
    )
    handler.addFilter(RedactingFilter())
    if _configured:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
    # Also filter at the logger level, so a handler added later by an embedding
    # application (uvicorn adds its own) still receives scrubbed records.
    root.addFilter(RedactingFilter())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _configured = True
