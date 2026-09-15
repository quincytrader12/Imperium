"""Process-wide paths and invariants.

Everything that another module might otherwise hardcode lives here, so that a
test can point the whole application at a temporary directory by setting one
environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "imperium"

#: The only address the server is ever permitted to bind. This is not a default
#: that can be overridden -- see imperium.server.app.validate_bind_host, which
#: raises on anything else. The process holds API keys and has no authentication.
ALLOWED_BIND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

DEFAULT_PORT = 8787

#: Text files are always read with this encoding. The platform default is cp1252
#: on Windows, and a single box-drawing character in a source comment is enough
#: to fail a Windows build that passed everywhere it was tested.
TEXT_ENCODING = "utf-8"


def home_dir() -> Path:
    """Return the configuration directory, honouring IMPERIUM_HOME for tests."""
    override = os.environ.get("IMPERIUM_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / f".{APP_NAME}"


def credentials_path() -> Path:
    return home_dir() / "credentials.json"


def journal_path() -> Path:
    return home_dir() / "journal.sqlite3"


def state_path() -> Path:
    return home_dir() / "state.json"


def ensure_home() -> Path:
    """Create the configuration directory owner-only if it does not exist."""
    d = home_dir()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d

def settings_path() -> Path:
    return home_dir() / "settings.txt"


#: What the settings file says when it is created.
#:
#: Written out in full, with every option commented and its default shown,
#: because the alternative is an empty file that tells an operator nothing.
#: This program is shipped as a Windows executable launched from a batch file:
#: there is no shell to export a variable in, and "set SECTOR_TREND_ENABLED=1"
#: is not an instruction anybody can act on when the thing they double-clicked
#: is an icon. A file next to their API keys is somewhere they can find.
SETTINGS_TEMPLATE = """\
# IMPERIUM settings
#
# One SETTING=value per line. Lines starting with # are ignored.
# Save the file and restart IMPERIUM for a change to take effect.
#
# This file lives beside your credentials and is read at startup. A real
# environment variable, if you set one, always wins over what is written here.

# ---------------------------------------------------------------- Sector Trend
# A separate sleeve that trades 19 liquid sector ETFs on Donchian/Keltner
# breakouts, once a day, using its own slice of the account. Off by default:
# run scripts/backtest_sector_trend.py and read the result before arming it.
#
# Note on small accounts: Alpaca refuses a fractional buy under $1.00. At the
# default 0.20 allocation, an account under about $130 will have the sleeve
# skip its more volatile ETFs -- the panel says which and why.

# SECTOR_TREND_ENABLED=false
# SECTOR_TREND_ALLOCATION=0.20
# SECTOR_TREND_MAX_LEVERAGE=1.0
# SECTOR_TREND_TARGET_VOL=0.015
# SECTOR_TREND_REBALANCE_THRESHOLD=0.25
# SECTOR_TREND_EXEC_MODE=near_close
# SECTOR_TREND_RUN_TIME_ET=15:45
# SECTOR_TREND_UNIVERSE=XLF,XLK,XLE,XLV,XLI,XBI,XLU,XLP,XLY,KRE,XLB,XLC,XRT,XOP,XLRE,XHB,KBE,XME,KIE

# Equity at which the sleeve switches itself on, or 0 to never. It arms once
# and never disarms -- switching off a sleeve that holds positions would leave
# them with nobody trailing their stops. $200 is where every ETF in the
# universe clears Alpaca's $1 minimum order at the default allocation.
# SECTOR_TREND_ARM_AT_EQUITY=200

# ------------------------------------------------------------ second currency
# Show the account balance in a second currency beside the dollar figure.
# Display only: every decision this program makes stays in dollars. Blank
# disables it. The rate comes from the ECB and is shown with its age; set
# IMPERIUM_FX_RATE to pin it by hand instead of fetching.

# IMPERIUM_SECONDARY_CURRENCY=ZAR
# IMPERIUM_FX_RATE=
"""

#: Settings this file is allowed to define.
#:
#: An allow-list rather than "anything that looks like a variable". This file
#: is read into the process environment, and a typo'd or hostile line should
#: not be able to set PATH, a proxy, or anything else the rest of the program
#: trusts the environment for.
SETTABLE = frozenset({
    "SECTOR_TREND_ENABLED",
    "SECTOR_TREND_ALLOCATION",
    "SECTOR_TREND_UNIVERSE",
    "SECTOR_TREND_TARGET_VOL",
    "SECTOR_TREND_MAX_LEVERAGE",
    "SECTOR_TREND_REBALANCE_THRESHOLD",
    "SECTOR_TREND_EXEC_MODE",
    "SECTOR_TREND_RUN_TIME_ET",
    "SECTOR_TREND_ARM_AT_EQUITY",
    "IMPERIUM_SECONDARY_CURRENCY",
    "IMPERIUM_FX_RATE",
})


def parse_settings(text: str) -> tuple[dict[str, str], list[str]]:
    """Settings from the file's text, and the names it refused.

    Pure, so the parsing can be tested without a filesystem. Returns only names
    in :data:`SETTABLE`; everything else is reported as refused rather than
    silently dropped, because a setting that does nothing and says nothing is
    the worst of the three possible outcomes.
    """
    found: dict[str, str] = {}
    refused: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().upper()
        value = value.strip().strip('"').strip("'")
        if name in SETTABLE:
            found[name] = value
        elif name:
            refused.append(name)
    return found, refused


def load_settings() -> tuple[list[str], list[str]]:
    """Apply the settings file to the environment. Returns (applied, refused).

    Names only, never values, because this returns straight into a log line.

    A real environment variable wins: someone who has gone to the trouble of
    setting one in a shell means it, and a file quietly overriding them would
    be the kind of surprise that costs an hour to find.

    Creates the file with everything commented out if it does not exist, so
    that the answer to "where do I set this" is a file that already exists and
    explains itself.
    """
    path = settings_path()
    try:
        if not path.exists():
            ensure_home()
            path.write_text(SETTINGS_TEMPLATE, encoding=TEXT_ENCODING)
            try:
                path.chmod(0o600)
            except (OSError, NotImplementedError):
                pass
            return [], []
        text = path.read_text(encoding=TEXT_ENCODING)
    except OSError:
        # A settings file that cannot be read must not stop the terminal
        # starting. Defaults are a working configuration.
        return [], []

    found, refused = parse_settings(text)
    applied = []
    for name, value in found.items():
        if name in os.environ:
            continue
        os.environ[name] = value
        applied.append(name)
    return sorted(applied), sorted(set(refused))
