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
