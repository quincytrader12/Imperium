"""The credential store.

One file, `~/.godalgo/credentials.json`, owner-only. The rules encoded here
rather than documented:

* a credential is never returned over HTTP -- `Credential.masked()` is the only
  view that leaves the process, and the secret is not a field on it;
* a stored key is `trade_enabled=False` until the operator ticks a box on that
  specific key, so pasting a key into a form cannot by itself authorise orders;
* permissions are *verified* on load, not merely set on write, because the
  interesting case is a file that was created correctly and then copied,
  restored from a backup, or synced to a shared drive.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

from godalgo import config
from godalgo.logging_setup import describe_secret, mask, register_secret

IS_WINDOWS = sys.platform.startswith("win")


class CredentialError(Exception):
    """A credential could not be stored or loaded. Carries an operator remedy."""

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.remedy = remedy


@dataclass(frozen=True)
class Credential:
    """One venue API key.

    ``secret`` is deliberately excluded from ``masked()``. There is no code path
    that serialises this dataclass wholesale to a response body; the server
    handlers call ``masked()``.
    """

    name: str
    venue: str
    api_key: str
    secret: str
    trade_enabled: bool = False
    note: str = ""

    def masked(self) -> dict[str, Any]:
        """The only representation permitted to leave the process."""
        return {
            "name": self.name,
            "venue": self.venue,
            "api_key_masked": mask(self.api_key),
            "secret_masked": describe_secret(self.secret),
            "trade_enabled": self.trade_enabled,
            "note": self.note,
        }

    def __repr__(self) -> str:  # pragma: no cover - defensive
        # A bare repr of this object must not print the secret, because an
        # unhandled exception in a handler will print its frame locals.
        return (
            f"Credential(name={self.name!r}, venue={self.venue!r}, "
            f"api_key={mask(self.api_key)!r}, secret=[redacted], "
            f"trade_enabled={self.trade_enabled!r})"
        )


@dataclass
class PermissionReport:
    """What the store found when it checked the file's permissions."""

    path: Path
    ok: bool
    detail: str
    remedy: str = ""


def _chmod_owner_only(path: Path) -> None:
    os.chmod(path, 0o600)


def _icacls_owner_only(path: Path) -> None:
    """Restrict a file to the current user on Windows.

    ``chmod`` is close to a no-op on Windows -- it toggles the read-only bit and
    nothing else -- so the ACL has to be rewritten explicitly. Inheritance is
    removed first, otherwise the directory's inherited "Users: read" ACE
    survives and the file stays world-readable despite the grant below.
    """
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    domain = os.environ.get("USERDOMAIN", "")
    principal = f"{domain}\\{user}" if domain and user else (user or "%USERNAME%")
    commands = [
        ["icacls", str(path), "/inheritance:r"],
        ["icacls", str(path), "/grant:r", f"{principal}:(R,W)"],
    ]
    for cmd in commands:
        subprocess.run(
            cmd, check=True, capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


def _check_permissions(path: Path) -> PermissionReport:
    """Verify the file is owner-only, and say precisely how to fix it if not."""
    if not path.exists():
        return PermissionReport(path, True, "file does not exist yet")
    if IS_WINDOWS:
        try:
            out = subprocess.run(
                ["icacls", str(path)], capture_output=True, text=True,
                encoding=config.TEXT_ENCODING, errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
        except OSError as exc:
            return PermissionReport(
                path, True, f"could not run icacls to verify permissions: {exc}"
            )
        risky = [g for g in ("Everyone", "BUILTIN\\Users", "Authenticated Users")
                 if g in out]
        if risky:
            return PermissionReport(
                path, False,
                f"the credentials file grants access to {', '.join(risky)}",
                remedy=(f'run: icacls "{path}" /inheritance:r '
                        f'/grant:r "%USERNAME%:(R,W)"'),
            )
        return PermissionReport(path, True, "owner-only ACL")

    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        return PermissionReport(
            path, False,
            f"the credentials file is mode {mode:04o}; group or others can read it",
            remedy=f"run: chmod 600 {path}",
        )
    return PermissionReport(path, True, f"mode {mode:04o}")


def _describe_json(value: Any) -> str:
    """Name a JSON value's type in words an operator can act on."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a true/false value"
    if isinstance(value, str):
        return f"the text {value[:24]!r}" if len(value) <= 24 else "a piece of text"
    if isinstance(value, (int, float)):
        return f"the number {value}"
    if isinstance(value, list):
        return f"a list of {len(value)} item(s)"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__


def _atomic_write(path: Path, payload: str) -> None:
    """Write owner-only, atomically.

    The temporary file is created with the restrictive mode *before* any bytes
    are written to it -- writing first and chmod-ing after leaves a window in
    which the secret exists world-readable on disk.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".cred-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        os.close(fd)
        if IS_WINDOWS:
            _icacls_owner_only(tmp)
        else:
            _chmod_owner_only(tmp)
        tmp.write_text(payload, encoding=config.TEXT_ENCODING)
        os.replace(tmp, path)
        if IS_WINDOWS:
            _icacls_owner_only(path)
        else:
            _chmod_owner_only(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover
                pass


class CredentialStore:
    """Load, save and mutate the credential file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.credentials_path()
        self._creds: dict[str, Credential] = {}
        self.permission_report: PermissionReport = PermissionReport(
            self.path, True, "not loaded"
        )
        self.load()

    # -- persistence -----------------------------------------------------

    def load(self) -> PermissionReport:
        self.permission_report = _check_permissions(self.path)
        self._creds = {}
        if not self.path.exists():
            return self.permission_report
        repair = f"repair or delete {self.path} and re-add the key"
        try:
            raw = json.loads(self.path.read_text(encoding=config.TEXT_ENCODING))
        except json.JSONDecodeError as exc:
            raise CredentialError(
                f"the credentials file is not valid JSON ({exc.msg} at line {exc.lineno})",
                remedy=repair,
            ) from exc
        except OSError as exc:
            raise CredentialError(
                f"the credentials file could not be read: {exc.strerror or exc}",
                remedy=f"check that {self.path} exists and is readable by you",
            ) from exc

        # Everything below validates *shape* before touching it. Valid JSON says
        # nothing about structure, and a file that parses but is shaped wrongly
        # used to raise a bare TypeError out of this method -- which took the
        # whole application down, because callers reasonably only expect
        # CredentialError from here.
        if not isinstance(raw, dict):
            raise CredentialError(
                f"the credentials file should contain a JSON object, but it "
                f"contains {_describe_json(raw)}",
                remedy=repair,
            )
        entries = raw.get("credentials", [])
        if entries is None:
            entries = []
        if isinstance(entries, dict):
            # A plausible hand-edit: {"credentials": {"main": {...}}}. Accept it
            # rather than refuse, since the intent is unambiguous.
            entries = list(entries.values())
        if not isinstance(entries, list):
            raise CredentialError(
                f"the 'credentials' field should be a list, but it is "
                f"{_describe_json(entries)}",
                remedy=repair,
            )

        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise CredentialError(
                    f"credential #{index + 1} should be a JSON object, but it is "
                    f"{_describe_json(entry)}",
                    remedy=repair,
                )
            missing = [f for f in ("name", "venue", "api_key", "secret")
                       if f not in entry]
            if missing:
                raise CredentialError(
                    f"credential #{index + 1} is missing "
                    f"{', '.join(repr(m) for m in missing)}",
                    remedy=repair,
                )
            wrong = [f for f in ("name", "venue", "api_key", "secret")
                     if not isinstance(entry[f], str)]
            if wrong:
                raise CredentialError(
                    f"credential #{index + 1} has non-text "
                    f"{', '.join(repr(w) for w in wrong)}",
                    remedy=repair,
                )
            note = entry.get("note", "")
            cred = Credential(
                name=entry["name"],
                venue=entry["venue"],
                api_key=entry["api_key"],
                secret=entry["secret"],
                trade_enabled=bool(entry.get("trade_enabled", False)),
                note=note if isinstance(note, str) else str(note),
            )
            self._creds[cred.name] = cred
            register_secret(cred.secret)
            register_secret(cred.api_key)
        return self.permission_report

    def save(self) -> None:
        payload = json.dumps(
            {
                "version": 1,
                "credentials": [
                    {
                        "name": c.name,
                        "venue": c.venue,
                        "api_key": c.api_key,
                        "secret": c.secret,
                        "trade_enabled": c.trade_enabled,
                        "note": c.note,
                    }
                    for c in self._creds.values()
                ],
            },
            indent=2,
        )
        _atomic_write(self.path, payload)
        self.permission_report = _check_permissions(self.path)

    # -- mutation --------------------------------------------------------

    def add(
        self,
        name: str,
        venue: str,
        api_key: str,
        secret: str,
        note: str = "",
    ) -> Credential:
        """Store a key. It is not tradeable; that is a separate, later decision."""
        name = name.strip()
        if not name:
            raise CredentialError("a credential needs a name",
                                  remedy="give the key a label you will recognise")
        api_key = api_key.strip()
        secret = secret.strip()
        if not api_key or not secret:
            raise CredentialError(
                "both an API key and a secret are required",
                remedy="paste both fields from the venue's API management page",
            )
        cred = Credential(
            name=name, venue=venue, api_key=api_key, secret=secret,
            trade_enabled=False, note=note,
        )
        self._creds[name] = cred
        register_secret(secret)
        register_secret(api_key)
        self.save()
        return cred

    def set_trade_enabled(self, name: str, enabled: bool) -> Credential:
        cred = self.require(name)
        updated = replace(cred, trade_enabled=bool(enabled))
        self._creds[name] = updated
        self.save()
        return updated

    def remove(self, name: str) -> None:
        self._creds.pop(name, None)
        self.save()

    # -- access ----------------------------------------------------------

    def get(self, name: str) -> Credential | None:
        return self._creds.get(name)

    def require(self, name: str) -> Credential:
        cred = self._creds.get(name)
        if cred is None:
            known = ", ".join(sorted(self._creds)) or "(none stored)"
            raise CredentialError(
                f"no stored credential named {name!r}",
                remedy=f"stored credentials: {known}",
            )
        return cred

    def __iter__(self) -> Iterator[Credential]:
        return iter(self._creds.values())

    def __len__(self) -> int:
        return len(self._creds)

    def masked_list(self) -> list[dict[str, Any]]:
        """The connections endpoint's entire response. Nothing else."""
        return [c.masked() for c in self._creds.values()]

    def tradeable(self, venue: str | None = None) -> list[Credential]:
        return [
            c for c in self._creds.values()
            if c.trade_enabled and (venue is None or c.venue == venue)
        ]
