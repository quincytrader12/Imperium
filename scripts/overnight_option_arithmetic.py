"""Is the overnight drift harvestable through options?

The drift is a property of the UNDERLYING: 3-5 bps of close-to-open move on the
S&P 500 (Cooper, Cliff & Gulen). An option gives delta exposure to that move but
pays theta for the privilege of holding overnight, and theta accrues on calendar
days rather than trading days.

So the question is arithmetic: for a position with the same delta-equivalent
exposure, does the drift captured exceed the theta paid?
"""
import math

def norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def norm_pdf(x): return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_call(S, K, T, r, sigma):
    if T <= 0: return max(0.0, S - K), 1.0 if S > K else 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    delta = norm_cdf(d1)
    theta = (-(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * norm_cdf(d2))     # per year
    return price, delta, theta

S, r, sigma = 100.0, 0.04, 0.25
DRIFT_BPS = 4.0                      # midpoint of the measured 2.82-4.76 bps
NIGHT = 1.0 / 365.0                  # one calendar night of theta

print(f"Underlying ${S:.0f}, IV {sigma:.0%}, overnight drift {DRIFT_BPS:.1f}bp")
print(f"Drift on $100 of underlying exposure: ${S * DRIFT_BPS / 10000:.4f}\n")
print(f"{'DTE':>5} {'strike':>7} {'delta':>7} {'premium':>9} {'delta-equiv':>12} "
      f"{'drift $':>9} {'theta $':>9} {'net $':>9} {'net bp*':>9}")
print("-" * 86)

for dte in (1, 7, 30, 90, 365):
    T = dte / 365.0
    for moneyness, label in ((1.00, "ATM"), (1.05, "5% OTM"), (0.90, "10% ITM")):
        K = S * moneyness
        price, delta, theta_yr = bs_call(S, K, T, r, sigma)
        if price <= 0.01:
            continue
        # Hold enough contracts to carry $100 of delta-equivalent exposure, so
        # the comparison is like-for-like with holding the stock itself.
        notional = 100.0
        contracts = notional / (delta * S) if delta > 0 else 0
        drift_gain = notional * DRIFT_BPS / 10000.0
        theta_cost = -theta_yr * NIGHT * contracts
        net = drift_gain - theta_cost
        # Net in bps of the *capital deployed* (the premium), which is what the
        # position actually costs.
        capital = price * contracts
        net_bp = (net / capital * 10000.0) if capital > 0 else 0.0
        print(f"{dte:>5} {K:>7.0f} {delta:>7.3f} {price:>9.2f} {contracts:>12.3f} "
              f"{drift_gain:>9.4f} {theta_cost:>9.4f} {net:>9.4f} {net_bp:>9.1f}")

print("\n* net in bps of premium deployed. Negative means theta exceeds the drift.")
print("\nBreakeven: how much overnight drift would be needed to cover one night")
print("of theta, per DTE, at the same delta-equivalent exposure:")
print(f"{'DTE':>5} {'ATM breakeven drift (bp)':>28}")
for dte in (1, 2, 7, 30, 90, 365):
    T = dte / 365.0
    price, delta, theta_yr = bs_call(S, S, T, r, sigma)
    contracts = 100.0 / (delta * S)
    theta_cost = -theta_yr * NIGHT * contracts
    need_bp = theta_cost / 100.0 * 10000.0
    print(f"{dte:>5} {need_bp:>28.1f}")
