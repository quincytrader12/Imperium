"""The credential store's security properties."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from imperium import config
from imperium.logging_setup import describe_secret, mask
from imperium.security.credentials import CredentialError, CredentialStore

SECRET = "S3cr3t" + "x" * 58
KEY = "PK" + "A" * 62


def test_a_stored_key_cannot_trade_until_separately_enabled():
    """Prevents: a form paste arming real money. Storing a key and authorising it
    to place orders must be two decisions, on that specific key."""
    store = CredentialStore()
    cred = store.add("main", "binance_spot", KEY, SECRET)
    assert cred.trade_enabled is False
    assert store.tradeable() == []
    store.set_trade_enabled("main", True)
    assert [c.name for c in store.tradeable()] == ["main"]


def test_no_credential_material_appears_in_the_masked_view():
    """Prevents: the connections endpoint leaking a secret.

    This caught a real bug: mask(value, keep_tail=0) returned the *entire*
    string, because value[-0:] is the whole string in Python. A 'masked' secret
    was being rendered verbatim.
    """
    store = CredentialStore()
    store.add("main", "binance_spot", KEY, SECRET)
    blob = json.dumps(store.masked_list())
    assert SECRET not in blob
    assert SECRET[:12] not in blob
    assert KEY not in blob
    assert "secret" not in json.loads(blob)[0]  # the field itself is absent


def test_mask_with_no_tail_reveals_nothing():
    """Prevents: the value[-0:] slicing bug directly, at the unit that had it.

    ``value[-0:]`` is the *whole string* in Python, so a "masked" value with
    keep_tail=0 was being rendered verbatim. Both shapes are covered:

    * ``keep_head=0, keep_tail=0`` takes an early return;
    * ``keep_head=4, keep_tail=0`` reaches the slicing itself, which is where
      the bug actually was. A mutation test found that only the first was
      covered, so reverting the real fix left the suite green.
    """
    secret = "SECRETVALUE" + "z" * 29
    assert "z" not in mask(secret, keep_head=0, keep_tail=0)
    assert "SECRET" not in mask(secret, keep_head=0, keep_tail=0)

    head_only = mask(secret, keep_head=4, keep_tail=0)
    assert head_only == "SECR…", head_only
    assert "z" not in head_only, "the tail leaked through value[-0:]"

    tail_only = mask(secret, keep_head=0, keep_tail=4)
    assert tail_only == "…zzzz", tail_only
    assert "SECRET" not in tail_only

    assert describe_secret("S" * 40) == "•••••••• (40 chars)"
    assert describe_secret("") == "(unset)"


def test_repr_of_a_credential_does_not_print_the_secret():
    """Prevents: an unhandled exception printing frame locals and dumping the
    secret into a traceback that gets pasted into a bug report."""
    store = CredentialStore()
    cred = store.add("main", "binance_spot", KEY, SECRET)
    assert SECRET not in repr(cred)
    assert "[redacted]" in repr(cred)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX modes")
def test_credentials_file_is_written_owner_only():
    """Prevents: a world-readable credentials file. The file holds live API
    secrets and the process has no authentication of its own."""
    store = CredentialStore()
    store.add("main", "binance_spot", KEY, SECRET)
    mode = stat.S_IMODE(os.stat(store.path).st_mode)
    assert mode == 0o600, f"credentials file is {mode:04o}"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX modes")
def test_loose_permissions_are_detected_on_load_with_a_fix():
    """Prevents: setting permissions on write and never checking them again. The
    interesting case is a file created correctly and then copied, restored from
    a backup, or synced to a shared drive."""
    store = CredentialStore()
    store.add("main", "binance_spot", KEY, SECRET)
    os.chmod(store.path, 0o644)
    report = CredentialStore().permission_report
    assert report.ok is False
    assert "chmod 600" in report.remedy


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX modes")
def test_the_secret_is_never_world_readable_even_momentarily():
    """Prevents: writing the file and chmod-ing afterwards, which leaves a window
    in which the secret exists on disk readable by anyone."""
    store = CredentialStore()
    store.add("main", "binance_spot", KEY, SECRET)
    # No leftover temp files, which would carry the secret at default umask.
    leftovers = list(store.path.parent.glob(".cred-*"))
    assert leftovers == []


def test_a_missing_credential_lists_what_is_actually_stored():
    """Prevents: 'credential not found' with no way to discover the right name."""
    store = CredentialStore()
    store.add("alpha", "binance_spot", KEY, SECRET)
    with pytest.raises(CredentialError) as excinfo:
        store.require("beta")
    assert "alpha" in excinfo.value.remedy


def test_a_corrupt_credentials_file_names_the_file_to_repair():
    """Prevents: a bare JSONDecodeError traceback on startup, which tells the
    operator nothing about which file to fix."""
    config.ensure_home()
    config.credentials_path().write_text("{not json", encoding="utf-8")
    with pytest.raises(CredentialError) as excinfo:
        CredentialStore()
    assert str(config.credentials_path()) in excinfo.value.remedy


@pytest.mark.parametrize("content,label", [
    ('{"credentials": "main"}', "the credentials field is a string"),
    ('{"credentials": ["main"]}', "a list of strings instead of objects"),
    ('[{"name": "x"}]', "the whole file is a list"),
    ('"hello"', "the whole file is a string"),
    ('42', "the whole file is a number"),
    ('true', "the whole file is a boolean"),
    ('{"credentials": [{"name": 1, "venue": "v", "api_key": "k", "secret": "s"}]}',
     "a field of the wrong type"),
    ('{"credentials": [{"name": "n"}]}', "an entry missing fields"),
    ('', "the file is empty"),
])
def test_a_wrongly_shaped_file_raises_a_credential_error_not_a_bare_type_error(
        content, label):
    """Prevents the failure that took the whole application down on Windows.

    The loader anticipated exactly two problems -- invalid JSON and a missing
    field -- and let every other shape raise whatever it happened to raise. A
    file containing {"credentials": "main"} parsed fine, then raised
    ``TypeError: string indices must be integers`` from inside the loop. Callers
    only catch CredentialError, so uvicorn aborted startup and the terminal
    never opened.

    Valid JSON says nothing about structure. Every shape must produce a
    CredentialError carrying a remedy.
    """
    config.ensure_home()
    config.credentials_path().write_text(content, encoding="utf-8")
    with pytest.raises(CredentialError) as excinfo:
        CredentialStore()
    assert excinfo.value.remedy, f"no remedy offered for {label}"
    assert str(config.credentials_path()) in excinfo.value.remedy


def test_a_hand_written_mapping_of_credentials_is_accepted():
    """Prevents refusing a file whose intent is unambiguous. Someone editing by
    hand may key the credentials by name rather than listing them; that is not
    ambiguous, so it is read rather than rejected."""
    config.ensure_home()
    config.credentials_path().write_text(
        '{"credentials": {"main": {"name": "main", "venue": "binance_spot",'
        ' "api_key": "PK1234567890", "secret": "SECRETVALUE0000"}}}',
        encoding="utf-8")
    store = CredentialStore()
    assert [c.name for c in store] == ["main"]


def test_a_null_credentials_field_loads_as_empty():
    """Prevents a file with no credentials in it being treated as corrupt."""
    config.ensure_home()
    config.credentials_path().write_text('{"credentials": null}', encoding="utf-8")
    assert len(CredentialStore()) == 0


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX modes")
def test_quarantining_does_not_leave_a_stale_permission_warning():
    """Prevents a loud "INSECURE CREDENTIAL FILE" alarm about a file that no
    longer exists.

    The permission report is taken at the start of the load, against the file
    that is then moved aside. Reporting it afterwards describes a file that is
    gone -- and a false alarm about credential security is worse than none,
    because it teaches the operator to ignore the real one.
    """
    config.ensure_home()
    path = config.credentials_path()
    path.write_text('{"credentials": "main"}', encoding="utf-8")
    os.chmod(path, 0o644)

    store = CredentialStore(quarantine_corrupt=True)
    assert store.quarantined_to is not None
    assert store.permission_report.ok is True, (
        f"stale warning about a moved file: {store.permission_report.detail}"
    )
    assert "does not exist" in store.permission_report.detail
