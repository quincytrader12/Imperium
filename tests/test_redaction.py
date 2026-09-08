"""The logger must not be able to emit credential material."""

from __future__ import annotations

import io
import logging

from imperium import logging_setup
from imperium.logging_setup import RedactingFilter, forget_secrets, register_secret, scrub

SECRET = "S3cr3t" + "x" * 58


def test_a_registered_secret_cannot_be_logged_even_via_an_argument():
    """Prevents: scrubbing the format string but not the interpolated arguments,
    which lets logger.info('key=%s', secret) through untouched."""
    forget_secrets()
    register_secret(SECRET)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger("imperium.test.redaction")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.info("connecting with key=%s", SECRET)
    logger.debug("full url https://api.example/x?signature=%s", SECRET)
    out = stream.getvalue()
    assert SECRET not in out
    assert "[redacted]" in out
    forget_secrets()


def test_the_registry_catches_a_secret_no_pattern_would_match():
    """Prevents: relying entirely on the secret-shaped patterns. A mutation test
    found this gap -- disabling the registry entirely left every other redaction
    test passing, because their 64-character secrets also match the key-shaped
    pattern. This one uses a secret with punctuation and ordinary length, which
    no pattern matches, so only the registry can catch it."""
    forget_secrets()
    odd_secret = "corr-horse-battery"      # matches none of the patterns
    assert odd_secret in scrub(f"key={odd_secret}")   # not caught before
    register_secret(odd_secret)
    assert odd_secret not in scrub(f"key={odd_secret}")
    forget_secrets()


def test_a_proxy_url_never_has_its_credentials_printed():
    """Prevents: reporting proxy configuration by value. A proxy URL embeds a
    username and password, and this output is designed to be pasted into a bug
    report."""
    assert "hunter2" not in scrub("HTTPS_PROXY=http://alice:hunter2@proxy.corp:3128")
    assert "alice" not in scrub("http://alice:hunter2@proxy.corp:3128")


def test_an_unregistered_key_shaped_string_is_still_scrubbed():
    """Prevents: relying only on the registry. The case that matters is a key the
    operator pasted into the wrong field, which was never registered and which
    we are about to echo back in an error message."""
    forget_secrets()
    never_registered = "Zx" + "9" * 62
    assert never_registered not in scrub(f"rejected key {never_registered}")


def test_an_exception_carrying_a_signed_url_is_scrubbed(caplog):
    """Prevents: a traceback bypassing the filter. An exception message rendered
    by the logging machinery is not the record's message."""
    forget_secrets()
    register_secret(SECRET)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger("imperium.test.exc")
    logger.handlers = [handler]
    logger.propagate = False
    try:
        raise RuntimeError(f"failed for signature={SECRET}")
    except RuntimeError:
        logger.exception("request failed")
    assert SECRET not in stream.getvalue()
    forget_secrets()


def test_proxy_environment_is_reported_by_name_only():
    """Prevents: printing proxy variable values in diagnostics."""
    import os

    os.environ["HTTPS_PROXY"] = "http://bob:swordfish@proxy:8080"
    try:
        names = logging_setup.safe_env_names(["HTTPS_PROXY", "NOT_SET_AT_ALL"])
        assert names == ["HTTPS_PROXY"]
        assert not any("swordfish" in n for n in names)
    finally:
        os.environ.pop("HTTPS_PROXY", None)
