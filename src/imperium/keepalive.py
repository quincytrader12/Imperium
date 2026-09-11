"""Keeping the machine awake while the book is live.

A trading loop cannot run through a suspend. On Windows the default power plan
sleeps an idle laptop within minutes, and "idle" is measured by input, not by
whether a process is doing something -- so a terminal left running overnight to
hold an overnight position is exactly the case the operating system is most
confident is idle.

This asks Windows to keep the *system* awake without keeping the *display*
awake, so the screen still turns off and the battery is not spent on a panel
nobody is looking at. It is a request, not a guarantee: closing a laptop lid
still suspends on most configurations, and nothing here overrides that.

Everywhere other than Windows this is a no-op that reports itself as one,
rather than a silent failure that would leave the operator believing the
machine was being held awake.
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger("imperium.keepalive")

#: Windows SetThreadExecutionState flags.
_ES_CONTINUOUS = 0x80000000        # the state persists until changed again
_ES_SYSTEM_REQUIRED = 0x00000001   # do not sleep the system


class KeepAwake:
    """Requests that the system not sleep while this is held."""

    def __init__(self) -> None:
        self.active = False
        self.supported = sys.platform == "win32"
        self.note = ("" if self.supported else
                     f"not supported on {sys.platform}; the operating system's "
                     f"own power settings decide whether this machine sleeps")

    def _set(self, flags: int) -> bool:
        import ctypes

        try:
            kernel32 = ctypes.windll.kernel32           # type: ignore[attr-defined]
            # A zero return means the request was refused.
            return bool(kernel32.SetThreadExecutionState(ctypes.c_uint(flags)))
        except Exception as exc:                         # pragma: no cover
            self.note = f"the sleep request was refused: {exc}"
            log.warning("could not set the execution state: %s", exc)
            return False

    def acquire(self) -> bool:
        """Ask the system to stay awake. Returns whether the request took."""
        if not self.supported or self.active:
            return self.active
        if self._set(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED):
            self.active = True
            self.note = ("holding the system awake; the display may still "
                         "sleep, and closing the lid still suspends")
            log.info("system sleep suppressed while the session runs")
        return self.active

    def release(self) -> None:
        """Hand power management back. Safe to call when not held."""
        if not self.supported or not self.active:
            return
        self._set(_ES_CONTINUOUS)
        self.active = False
        self.note = "power management handed back to the system"
        log.info("system sleep suppression released")

    def as_dict(self) -> dict[str, object]:
        return {"supported": self.supported, "active": self.active,
                "note": self.note}
