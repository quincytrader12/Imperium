"""Binance error codes translated into remedies.

A venue error code is a fact about *the operator's situation*, not a number to
print. `-1021` is not "timestamp for this request was outside the recvWindow";
it is "this machine's clock is wrong, sync it", and the second sentence is the
one that ends the support conversation.

This is also why the client speaks to the venue directly rather than through a
multi-venue library: a library normalises `-2015` into a generic
`AuthenticationError`, discarding precisely the distinction between "wrong key",
"wrong IP" and "Spot trading not enabled on this key" that the operator needs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Remedy:
    """What a venue error means and what the operator should do about it."""

    code: int
    meaning: str
    remedy: str
    #: True when the venue rejected the *content* of the request. Re-sending an
    #: identical request sends the same wrong request again, so these are never
    #: retried.
    terminal: bool = True


_TABLE: dict[int, Remedy] = {
    -2015: Remedy(
        -2015,
        "invalid API key, IP address, or permissions",
        "Three different causes share this code. Check, in order: the key's IP "
        "allow-list on the venue (add this machine's public address, or set the "
        "key to unrestricted); that 'Enable Spot & Margin Trading' is ticked on "
        "the key; and that the key has not been deleted or has not expired.",
    ),
    -2014: Remedy(
        -2014,
        "malformed API key",
        "The key itself is the wrong shape, which almost always means the paste "
        "was truncated or carries a leading/trailing space or newline. Re-copy "
        "the full API key.",
    ),
    -1022: Remedy(
        -1022,
        "signature for this request is not valid",
        "The secret is wrong, or the API key and secret have been pasted into "
        "each other's fields. Both are the same length and shape, so swapping "
        "them is easy and produces exactly this error.",
    ),
    -1021: Remedy(
        -1021,
        "timestamp outside recvWindow",
        "This machine's clock is wrong. Sync it: on Windows run "
        "'w32tm /resync' as administrator; on Linux/macOS enable NTP. A drift of "
        "more than about 5 seconds is enough to produce this.",
    ),
    -1121: Remedy(
        -1121,
        "invalid symbol",
        "That pair is not listed on this venue. Check the spelling for *this* "
        "venue specifically -- symbol vocabularies differ, and a pair that "
        "exists elsewhere may not exist here.",
    ),
    -1100: Remedy(
        -1100,
        "illegal characters in a parameter",
        "A parameter contains a character the venue rejects -- most often a "
        "symbol with a slash or a dash in it, or a quantity formatted in "
        "scientific notation.",
    ),
    -1013: Remedy(
        -1013,
        "filter failure: the order violates a symbol filter",
        "The order is below the symbol's minimum notional, or the quantity is "
        "not a multiple of its step size, or the price is not a multiple of its "
        "tick size. Raise the order size or re-round against exchangeInfo.",
    ),
    -1111: Remedy(
        -1111,
        "precision is over the maximum defined for this asset",
        "The quantity or price has more decimal places than the symbol allows. "
        "Round down to the step size and tick size from exchangeInfo.",
    ),
    -1106: Remedy(
        -1106,
        "a parameter was sent that was not required",
        "Most commonly: a LIMIT_MAKER (post-only) order was sent with a "
        "timeInForce field. LIMIT_MAKER takes no timeInForce.",
    ),
    -2010: Remedy(
        -2010,
        "new order rejected",
        "Usually insufficient balance for this order, or an attempt to sell more "
        "of an asset than is held. Binance Spot has nothing to borrow, so a "
        "short is not a risky position -- it is a rejected order.",
    ),
    -2011: Remedy(
        -2011,
        "cancel rejected: no such order",
        "The order is already filled, already cancelled, or was never accepted. "
        "Look it up by client order ID before concluding it is lost.",
    ),
    -2013: Remedy(
        -2013,
        "order does not exist",
        "The order is not on the book. If a submission timed out, query it by "
        "origClientOrderId -- an order that filled and an order that never "
        "landed are indistinguishable from the timeout alone.",
    ),
    -1003: Remedy(
        -1003,
        "too many requests -- rate limit exceeded",
        "Back off. If this persists the request weight budget is being spent too "
        "fast; a repeated ban raises the ban duration each time.",
        terminal=False,
    ),
    -1015: Remedy(
        -1015,
        "too many new orders",
        "The order-rate limit was hit. Slow down order submission.",
        terminal=False,
    ),
    -1001: Remedy(
        -1001,
        "internal error at the venue",
        "The venue failed, not this program. Retry with backoff.",
        terminal=False,
    ),
    -1016: Remedy(
        -1016,
        "this service is no longer available",
        "The endpoint has been retired by the venue.",
    ),
    -1020: Remedy(
        -1020,
        "unsupported operation",
        "The venue does not support this operation for this symbol or account "
        "type.",
    ),
    -1102: Remedy(
        -1102,
        "a mandatory parameter was empty or missing",
        "A required field was not sent. This is a bug in this program, not a "
        "problem with the account.",
    ),
    -3045: Remedy(
        -3045,
        "the system does not have enough asset now",
        "The venue cannot fill this from its own liquidity pool at present.",
    ),
}

#: Codes that mean "the order was wrong". Re-sending is guaranteed to fail the
#: same way, and on an order endpoint a blind retry after an ambiguous timeout
#: is how a position gets opened twice.
NON_RETRYABLE_ORDER_CODES = frozenset({-1013, -1100, -1111, -1121, -2010, -2011, -2013})


def lookup(code: int | None) -> Remedy | None:
    if code is None:
        return None
    return _TABLE.get(int(code))


def describe(code: int | None, venue_message: str = "") -> str:
    """Render a one-line operator-facing explanation for a venue error."""
    remedy = lookup(code)
    if remedy is None:
        base = f"venue error {code}" if code is not None else "venue error"
        return f"{base}: {venue_message}" if venue_message else base
    return f"{remedy.meaning} ({remedy.code}) — {remedy.remedy}"


def is_retryable(code: int | None) -> bool:
    """Only transport failures and venue-side faults are retryable."""
    remedy = lookup(code)
    if remedy is None:
        # An unknown code is treated as terminal. Guessing that an unrecognised
        # rejection is transient is how a rejected order gets sent in a loop.
        return False
    return not remedy.terminal
