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


def trades_path() -> Path:
    return home_dir() / "trades.csv"


def equity_history_path(mode: str = "") -> Path:
    suffix = f"_{mode}" if mode else ""
    return home_dir() / f"equity_history{suffix}.json"


def state_path() -> Path:
    return home_dir() / "state.json"


def ensure_home() -> Path:
    """Create the configuration directory owner-only if it does not exist."""
    d = home_dir()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d


def read_state() -> dict:
    """Everything in the shared state file, or an empty dict.

    Never raises. A missing, unreadable or corrupt file reads as empty, which
    every caller already treats as "start with nothing and say so".
    """
    import json

    try:
        loaded = json.loads(state_path().read_text(encoding=TEXT_ENCODING))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def update_state(**sections) -> None:
    """Replace the named top-level sections of the state file, keeping the rest.

    The only way anything in this program writes that file, and it exists
    because three writers each did their own read-modify-write and one of them
    did not. The overnight save wrote its five keys as a fresh file, so every
    overnight save deleted the Sector Trend sleeve's section: its positions,
    its trailing stops and the record of it arming. Its own docstring warned
    that "a forgotten stop is an unbounded position", and it was forgetting
    them several times a day.

    Atomic as well as merging. Written to a temporary file beside the real one
    and moved over it, so a crash or a full disk mid-write leaves the previous
    file intact rather than a truncated one -- which every reader would treat
    as empty, losing the whole book at once.

    Raises OSError on failure. Callers decide how loud a lost write is; for
    the sleeve it is loud, because stops are in it.
    """
    import json

    ensure_home()
    path = state_path()
    payload = read_state()
    payload.update(sections)
    scratch = path.with_name(path.name + ".tmp")
    scratch.write_text(json.dumps(payload, indent=2), encoding=TEXT_ENCODING)
    try:
        scratch.chmod(0o600)
    except (OSError, NotImplementedError):
        # Windows ignores POSIX modes. Nothing secret is in this file -- names,
        # symbols and weights -- so this is tidiness, not a gate.
        pass
    os.replace(scratch, path)

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

# ---------------------------------------------------------------- Global Trend
# A sleeve that holds bonds, gold, commodities, international stocks and real
# estate (IEF TLT GLD DBC EFA EEM VNQ) while each is trending up, sized to 10%
# volatility, decided once a day after the time below. It diversifies away
# from US stocks, which every other strategy here trades. Real orders in
# paper and live mode; buys under Alpaca's $1 minimum are skipped and named.

# GLOBAL_TREND_ENABLED=true
# GLOBAL_TREND_ALLOCATION=0.30
# GLOBAL_TREND_RUN_TIME_ET=15:45

# ------------------------------------------------------------ mean reversion
# A sleeve that buys an index ETF (SPY QQQ IWM DIA) after two sharp down days
# -- a two-day RSI under 10 -- but only while it is above its 200-day average,
# and sells on the first close above its 5-day average or after ten trading
# days. At most two at once. In cash most of the time; it earns in the choppy
# markets where the trend strategies bleed.

# MEAN_REVERSION_ENABLED=true
# MEAN_REVERSION_ALLOCATION=0.20
# MEAN_REVERSION_RUN_TIME_ET=15:45

# --------------------------------------------------------- turn of the month
# A sleeve that holds the S&P 500 (IVV) from the close of the second-to-last
# trading day of each month to the close of the third trading day of the
# next -- the few days where, historically, most of the market's return has
# arrived -- and is in cash the rest of the month.

# TURN_OF_MONTH_ENABLED=true
# TURN_OF_MONTH_ALLOCATION=0.15
# TURN_OF_MONTH_RUN_TIME_ET=15:45

# ------------------------------------------------ overnight stock selection
# The overnight strategy carries only stocks whose own last year of overnight
# returns ranks at or above this percentile among the ones it measures (Lou,
# Polk & Skouras: a stock's overnight tendency persists for years). 0.5 is
# the top half; 0 turns the selection off.

# IMPERIUM_OVERNIGHT_MIN_RANK=0.5

# ----------------------------------------------------------------- risk dial
# One scale on every strategy's size, from how the whole account behaves:
# when its realised volatility (last 20 days) runs over the target, sizes
# shrink in proportion; from DRAWDOWN_START below the high-water mark to
# DRAWDOWN_FULL they are cut in a straight line to a quarter. Never to zero,
# never above full size. A withdrawal reads as a drawdown: set
# IMPERIUM_HIGH_WATER_SINCE to the day after it so older highs stop counting.

# IMPERIUM_RISK_DIAL=true
# IMPERIUM_ACCOUNT_TARGET_VOL=0.15
# IMPERIUM_DRAWDOWN_START=0.05
# IMPERIUM_DRAWDOWN_FULL=0.20
# IMPERIUM_HIGH_WATER_SINCE=

# ----------------------------------------------------------- passive entries
# An entry first rests as a limit order at the midpoint between bid and ask.
# If it has not filled after IMPERIUM_PASSIVE_SECONDS, whatever is left is
# sent as an ordinary market order -- the trade happens either way; resting is
# only a chance not to pay the spread. Exits, stops and auction orders always
# go at market. Applies to paper and live, which send real orders; dry run
# fills nothing. Set to false to send every entry at market as before.

# IMPERIUM_PASSIVE_ENTRIES=true
# IMPERIUM_PASSIVE_SECONDS=20

# ------------------------------------------------------------ second currency
# Show the account balance in a second currency beside the dollar figure.
# Display only: every decision this program makes stays in dollars. Blank
# disables it. The rate comes from the ECB and is shown with its age; set
# IMPERIUM_FX_RATE to pin it by hand instead of fetching.

# IMPERIUM_SECONDARY_CURRENCY=ZAR
# IMPERIUM_FX_RATE=

# ------------------------------------------------------------------ greeting
# What the terminal calls you when you press Start. It says this and then a
# trading quote, out loud if a voice is connected and on screen either way.

# IMPERIUM_OPERATOR=Mr Gininda

# Only set this if a voice mispronounces the name above. It is handed to the
# speech engine instead, spelled however it needs to be to sound right; the
# screen always shows IMPERIUM_OPERATOR. Keep each syllable sayable -- a
# consonant run an engine cannot pronounce makes it spell the letters out
# instead, which is worse than any mispronunciation.
# IMPERIUM_OPERATOR_SPOKEN=
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
    "IMPERIUM_OPERATOR",
    "IMPERIUM_OPERATOR_SPOKEN",
    "GLOBAL_TREND_ENABLED",
    "GLOBAL_TREND_ALLOCATION",
    "GLOBAL_TREND_RUN_TIME_ET",
    "MEAN_REVERSION_ENABLED",
    "MEAN_REVERSION_ALLOCATION",
    "MEAN_REVERSION_RUN_TIME_ET",
    "TURN_OF_MONTH_ENABLED",
    "TURN_OF_MONTH_ALLOCATION",
    "TURN_OF_MONTH_RUN_TIME_ET",
    "IMPERIUM_OVERNIGHT_MIN_RANK",
    "IMPERIUM_RISK_DIAL",
    "IMPERIUM_ACCOUNT_TARGET_VOL",
    "IMPERIUM_DRAWDOWN_START",
    "IMPERIUM_DRAWDOWN_FULL",
    "IMPERIUM_HIGH_WATER_SINCE",
    "IMPERIUM_PASSIVE_ENTRIES",
    "IMPERIUM_PASSIVE_SECONDS",
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
