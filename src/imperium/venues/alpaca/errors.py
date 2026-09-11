"""Alpaca error codes translated into remedies.

Same principle as the venue client itself: a code is a fact about the
operator's situation, not a number to print. Alpaca's vocabulary is entirely
different from a crypto exchange's -- there is no signature to get wrong, but
there are half a dozen ways an account can be permitted to hold an asset and
still not be permitted to trade it right now.

The codes below are the ones that actually stop an autonomous book. Alpaca
returns them as ``{"code": N, "message": "..."}`` alongside an HTTP status.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Remedy:
    code: int
    meaning: str
    remedy: str
    #: True when the venue rejected the *content* of the request. Re-sending an
    #: identical request sends the same wrong request again.
    terminal: bool = True


_TABLE: dict[int, Remedy] = {
    40110000: Remedy(
        40110000,
        "the API key was not accepted",
        "Check that the key and secret are pasted into the right fields and "
        "belong to the environment you are pointing at. A paper key does not "
        "work against the live endpoint, and a live key does not work against "
        "the paper one -- that mismatch is the single most common cause.",
    ),
    40310000: Remedy(
        40310000,
        "the account is not permitted to do this",
        "Usually insufficient buying power, or an attempt to sell more than is "
        "held. It also covers an account that is restricted, or an asset the "
        "account is not approved to trade.",
    ),
    40310010: Remedy(
        40310010,
        "the position is not held, or is smaller than the order",
        "Selling requires holding the shares. Check the position before "
        "reducing it.",
    ),
    40410000: Remedy(
        40410000,
        "not found",
        "The order, position or asset does not exist under this account.",
    ),
    42210000: Remedy(
        42210000,
        "the order was rejected as invalid",
        "The order's fields do not describe a valid order for this asset: a "
        "quantity below the minimum, a price off the tick, or a time-in-force "
        "the asset does not accept.",
    ),
    40010001: Remedy(
        40010001,
        "a request parameter was invalid",
        "This is a bug in this program rather than a problem with the account.",
    ),
    40310100: Remedy(
        40310100,
        "buying power or shares are insufficient",
        "The order needs more buying power than the account has. Reduce the "
        "size, or free up cash.",
    ),
    42910000: Remedy(
        42910000,
        "too many requests",
        "The API rate limit was exceeded. Back off; the budget refills each "
        "minute.",
        terminal=False,
    ),
    50010000: Remedy(
        50010000,
        "an internal error at the venue",
        "The venue failed, not this program. Retry with backoff.",
        terminal=False,
    ),
}

#: HTTP statuses that carry their own meaning when no code accompanies them.
_STATUS_TABLE: dict[int, Remedy] = {
    401: Remedy(
        401,
        "the API key was not accepted (HTTP 401)",
        "Check the key and secret, and that they match the environment: paper "
        "keys only work against the paper endpoint and live keys only against "
        "the live one.",
    ),
    403: Remedy(
        403,
        "forbidden (HTTP 403)",
        "The credentials were understood but the account may not do this -- "
        "commonly insufficient buying power, or an unapproved asset class.",
    ),
    404: Remedy(404, "not found (HTTP 404)",
                "The order, position or asset does not exist under this account."),
    422: Remedy(422, "the order was rejected as invalid (HTTP 422)",
                "A field on the order is not valid for this asset."),
    429: Remedy(429, "rate limited (HTTP 429)",
                "Back off; the request budget refills each minute.",
                terminal=False),
}

#: Codes meaning "the order was wrong". Re-sending is guaranteed to fail the
#: same way, and on an order endpoint a blind retry after an ambiguous timeout
#: is how a position gets opened twice.
NON_RETRYABLE_ORDER_CODES = frozenset(
    {40310000, 40310010, 40310100, 42210000, 40010001, 40410000}
)


def lookup(code: int | None, status: int | None = None) -> Remedy | None:
    if code is not None and int(code) in _TABLE:
        return _TABLE[int(code)]
    if status is not None and int(status) in _STATUS_TABLE:
        return _STATUS_TABLE[int(status)]
    return None


def describe(code: int | None, message: str = "",
             status: int | None = None) -> str:
    remedy = lookup(code, status)
    if remedy is None:
        base = f"venue error {code}" if code is not None else "venue error"
        return f"{base}: {message}" if message else base
    return f"{remedy.meaning} — {remedy.remedy}"


def is_retryable(code: int | None, status: int | None = None) -> bool:
    """Only transport failures and venue-side faults are retryable."""
    remedy = lookup(code, status)
    if remedy is None:
        # An unknown failure is treated as terminal. Guessing that an
        # unrecognised rejection is transient is how a rejected order gets sent
        # in a loop.
        return bool(status is not None and status >= 500)
    return not remedy.terminal
