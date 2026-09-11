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


def test_an_alpaca_key_is_scrubbed_even_though_it_is_short():
    """Prevents the gap the venue switch opened.

    Binance keys are 64 characters, so a single length rule covered them. An
    Alpaca key id is twenty, and Alpaca does not sign requests -- authentication
    *is* the header -- so a key echoed into an error message is the whole
    credential, not a derived signature. The length rule alone would let it
    straight through.
    """
    forget_secrets()
    paper = "PK" + "7ABCDEFGHIJKLMNOPQ"
    live = "AK" + "9ZYXWVUTSRQPONMLKJ"
    assert len(paper) == 20 and len(live) == 20

    for key in (paper, live):
        out = scrub(f"the venue rejected the key {key} against the paper host")
        assert key not in out
        assert "[redacted]" in out
        # The surrounding text has to survive, or the message stops being
        # diagnostic and the operator learns nothing.
        assert "the venue rejected" in out and "paper host" in out


def test_an_alpaca_auth_header_never_reaches_a_handler():
    """Each redaction layer is tested on a value only that layer can catch.

    The value here is deliberately short and ordinary-looking: not long enough
    for the length rule, not shaped like a key id. If the header rule were
    deleted, nothing else would save it -- which is the point, and is what an
    earlier version of this test missed by using a 40-character value that the
    length rule caught regardless.
    """
    forget_secrets()
    short = "notlongenough12"
    assert len(short) < 40 and not short.startswith(("PK", "AK"))
    for header in ("APCA-API-KEY-ID", "APCA-API-SECRET-KEY"):
        out = scrub(f"{header}: {short}")
        assert short not in out
        assert header in out, "the header name is diagnostic; only its value is not"


def test_a_secret_shaped_string_is_scrubbed_on_length_alone():
    """The length rule, on a value no other rule covers.

    Forty characters is Alpaca's secret length. Not in a header, not shaped
    like a key id: if the bound drifts back up, this is the test that goes red.
    """
    forget_secrets()
    secret = "Zq" + "3wRt7yUi0pLkJhGfDsAaQwErTyUiOpZxCv18"
    assert len(secret) == 38
    secret += "Bn"
    assert len(secret) == 40 and not secret.startswith(("PK", "AK"))
    out = scrub(f"the store rejected {secret} as malformed")
    assert secret not in out
    assert "the store rejected" in out


def test_a_client_order_id_is_still_readable_after_scrubbing():
    """The length rule was lowered to forty characters to cover an Alpaca
    secret. A client order id must not be collateral damage: it is the only
    handle on an ambiguous submission, and an unreadable journal is how a
    duplicate order gets sent."""
    from imperium.venues.alpaca.client import AlpacaClient

    forget_secrets()
    coid = AlpacaClient.new_client_order_id("imp")
    assert coid in scrub(f"venue accepted {coid}")
