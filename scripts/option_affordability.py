"""When does an options position become affordable, and what does it cost?

My earlier arithmetic for options assumed a $0.65 per-contract commission and
concluded they were unusable. That premise was wrong. Alpaca charges **no
commission** on options; what remains is regulatory pass-through, and it is
small:

    OCC clearing fee        $0.025 per contract, both legs
    Options Regulatory Fee  $0.015-0.023 per contract, both legs
    FINRA TAF               $0.00329 per contract, sell leg only

    round trip              ~$0.09 per contract

(Figures from Alpaca's published schedules as summarised by search; the primary
pages are unreachable from this build sandbox, so they are treated as assumed
and the terminal says so, exactly as it does for equity and crypto fees.)

So fees are not the obstacle. Two other things are, and they are structural:

1. **A contract is 100 shares.** The position cannot be sized; it is quantised
   at one contract. The cheapest thing an account can buy is one contract, and
   that costs 100x the quoted premium.
2. **The bid-ask spread is the whole cost**, and it is worst exactly where a
   small account is forced to shop -- the cheapest contracts.

This script computes what an account of a given size can actually reach, and
what crossing the spread on it costs.
"""

from __future__ import annotations

CONTRACT_MULTIPLIER = 100

#: Per contract, per round trip. Assumed, not confirmed against a live account.
OCC_CLEARING = 0.025
ORF = 0.020
TAF_SELL = 0.00329
ROUND_TRIP_FEES = 2 * (OCC_CLEARING + ORF) + TAF_SELL

#: Typical quoted spreads by premium, from published market-quality studies and
#: exchange data. Cheap contracts are quoted in penny increments too, but a
#: penny on a $0.20 option is 5% -- the *proportional* spread explodes as the
#: premium falls, which is the entire problem for a small account.
SPREAD_BY_PREMIUM = [
    # (premium, typical bid-ask width in dollars, note)
    (0.05, 0.02, "far out of the money, effectively a lottery ticket"),
    (0.20, 0.03, "cheap weekly, wide in proportional terms"),
    (0.50, 0.04, "short-dated near the money"),
    (1.00, 0.05, "liquid near-dated"),
    (2.50, 0.07, "liquid at the money"),
    (5.00, 0.10, "liquid at the money, longer dated"),
]


def line(premium: float, width: float, note: str) -> str:
    cost = premium * CONTRACT_MULTIPLIER
    spread_cost = width * CONTRACT_MULTIPLIER           # one full crossing
    total = spread_cost + ROUND_TRIP_FEES
    bps = 10_000 * total / cost
    return (f"  ${premium:>5.2f}  ${cost:>8.2f}  ${width:>5.2f}  "
            f"${total:>7.2f}  {bps:>8.0f}bp   {note}")


def main() -> None:
    print("One contract is 100 shares, so the smallest position is 100x the "
          "quoted premium.\n")
    print(f"Regulatory fees per contract, round trip: ${ROUND_TRIP_FEES:.4f} "
          f"— negligible beside the spread.\n")
    print("  prem   1 contract  spread   RT cost   as bps of premium")
    for premium, width, note in SPREAD_BY_PREMIUM:
        print(line(premium, width, note))

    print("\nWhat each account size can actually reach, at a 20% per-symbol cap")
    print("(and at the concentrated cap a small account runs under):\n")
    print(f"  {'equity':>9}  {'cap':>6}  {'max premium':>12}  what that buys")
    for equity, cap in ((70, 0.40), (250, 0.20), (500, 0.20), (1_500, 0.20),
                        (5_000, 0.20), (25_000, 0.20)):
        budget = equity * cap
        max_premium = budget / CONTRACT_MULTIPLIER
        if max_premium < 0.10:
            what = "nothing — one contract costs more than the whole position"
        elif max_premium < 0.35:
            what = "only far-OTM lottery tickets, 10-30% round-trip spread"
        elif max_premium < 1.00:
            what = "cheap short-dated, ~8% round-trip spread"
        elif max_premium < 2.50:
            what = "liquid near-dated, ~5% round-trip spread"
        else:
            what = "liquid at-the-money, ~2-4% round-trip spread"
        print(f"  ${equity:>8,}  {cap:>5.0%}  ${max_premium:>11.2f}  {what}")

    print("\nThe breakeven move, as a fraction of premium, before an option "
          "position\nis worth opening at all (spread + fees, times the 1.5x "
          "safety multiple\nthis program applies to every strategy):\n")
    for premium, width, _ in SPREAD_BY_PREMIUM:
        cost = premium * CONTRACT_MULTIPLIER
        total = width * CONTRACT_MULTIPLIER + ROUND_TRIP_FEES
        print(f"  ${premium:>5.2f} premium: the option must gain "
              f"{1.5 * total / cost:>5.1%} before the trade breaks even")


if __name__ == "__main__":
    main()
