"""Dividing one account between several strategies that do not know about
each other.

**The bug this exists to fix.** Every strategy in this terminal sized itself
against the *whole* account. The engine's portfolio allocator took total equity
as its base, and the Sector Trend sleeve took its own slice of total equity on
top. With the sleeve enabled at a fifth of the account, the two together
intended a hundred and twenty percent of the money -- not because either was
wrong about its own share, but because neither knew the other existed. Nothing
in the program noticed, because each was internally consistent. The first
evidence would have been an order rejected for buying power, on a day when
several strategies happened to want capital at once.

**What a share means here.** A fraction of account equity that one strategy may
size against and no other may. It is not a cash reservation and not a margin
allocation: cash remains the account's real number and remains a hard ceiling
underneath all of this, because a strategy cannot spend money that is not
there. The share governs how big a position is *allowed* to be, which is the
number every sizing rule in this program reads.

**Why a disabled strategy gets nothing.** A share held by something that is
switched off is capital doing nothing while the strategies that could use it
are told they may not. So shares are claimed by the strategies actually
running, and whatever is left goes to the engine -- which is why turning the
sector sleeve on automatically takes the engine from the whole account to four
fifths of it, with no configuration change and no double count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The name the per-symbol engine strategies trade under.
#:
#: They share one book and one concurrency limit, so they share one share.
#: Splitting the intraday, overnight, trend and cross-section strategies
#: further would be inventing a precision this program does not have: they
#: already contend for the same position slots, and the allocator resolves that
#: contention.
ENGINE = "engine"


@dataclass(frozen=True)
class Claim:
    """One strategy's request for part of the account."""

    name: str
    share: float
    enabled: bool = True
    note: str = ""


@dataclass
class CapitalPlan:
    """Who may size against what, and what is left over."""

    equity: float = 0.0
    shares: dict[str, float] = field(default_factory=dict)
    #: Claims that were refused because the account was already fully spoken
    #: for. Named rather than silently zeroed: a strategy that is switched on
    #: and given nothing looks exactly like a strategy that is finding no
    #: trades, and the two want completely different responses.
    refused: dict[str, float] = field(default_factory=dict)
    idle_share: float = 0.0

    def share_for(self, name: str) -> float:
        return float(self.shares.get(name, 0.0))

    def equity_for(self, name: str) -> float:
        """The equity a strategy may size against."""
        return max(0.0, self.equity) * self.share_for(name)

    @property
    def claimed(self) -> float:
        return float(sum(self.shares.values()))

    def as_dict(self) -> dict[str, object]:
        return {
            "equity": round(self.equity, 2),
            "shares": {k: round(v, 4) for k, v in self.shares.items()},
            "allocated": {k: round(self.equity_for(k), 2) for k in self.shares},
            "idle_share": round(self.idle_share, 4),
            "idle": round(max(0.0, self.equity) * self.idle_share, 2),
            "refused": {k: round(v, 4) for k, v in self.refused.items()},
        }


def divide(equity: float, claims: list[Claim]) -> CapitalPlan:
    """Split the account, giving the engine whatever the sleeves do not take.

    Claims are honoured in the order given, which makes the outcome
    predictable rather than dependent on a dictionary's iteration: if the
    account is oversubscribed, it is the *last* claim that is refused, not an
    arbitrary one.

    The engine is deliberately not a claim. It is the residual, so that capital
    a sleeve is not using is always available to something rather than sitting
    idle because nobody declared a share for it.
    """
    plan = CapitalPlan(equity=max(0.0, float(equity)))
    remaining = 1.0

    for claim in claims:
        if claim.name == ENGINE:
            # The engine cannot claim: it takes what is left. Accepting a claim
            # for it would let the total exceed one by construction.
            continue
        wanted = max(0.0, min(1.0, float(claim.share)))
        if not claim.enabled or wanted <= 0.0:
            continue
        if wanted > remaining + 1e-12:
            plan.refused[claim.name] = wanted
            # Give it what is actually left rather than nothing: a sleeve that
            # asked for a fifth and can have a tenth should trade the tenth.
            wanted = max(0.0, remaining)
            if wanted <= 0.0:
                continue
        plan.shares[claim.name] = wanted
        remaining -= wanted

    plan.shares[ENGINE] = max(0.0, remaining)
    plan.idle_share = 0.0
    return plan
