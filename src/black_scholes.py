"""
Black-Scholes option pricing and Greeks.
Used for delta estimation when selecting strikes (Polygon has no historical greeks).
"""
import math
import numpy as np


# ── Pure-Python normal distribution (no scipy dependency) ────────────────────

class _Norm:
    """Standard normal distribution CDF/PDF using math.erf."""

    @staticmethod
    def cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def pdf(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

norm = _Norm()


def d1(S, K, T, r, sigma):
    """Compute d1 term of Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S, K, T, r, sigma):
    """Compute d2 term of Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


# ── Pricing ──────────────────────────────────────────────────────────────────

def call_price(S, K, T, r, sigma):
    """Black-Scholes call option price.
    S: underlying price
    K: strike price
    T: time to expiry in years
    r: risk-free rate (annualized)
    sigma: implied volatility (annualized)
    """
    if T <= 0:
        return max(S - K, 0.0)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(_d1) - K * math.exp(-r * T) * norm.cdf(_d2)


def put_price(S, K, T, r, sigma):
    """Black-Scholes put option price."""
    if T <= 0:
        return max(K - S, 0.0)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)


# ── Greeks ───────────────────────────────────────────────────────────────────

def call_delta(S, K, T, r, sigma):
    """Delta of a call option (0 to 1)."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    return norm.cdf(d1(S, K, T, r, sigma))


def put_delta(S, K, T, r, sigma):
    """Delta of a put option (-1 to 0)."""
    if T <= 0:
        return -1.0 if S < K else 0.0
    return norm.cdf(d1(S, K, T, r, sigma)) - 1.0


def gamma(S, K, T, r, sigma):
    """Gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    return norm.pdf(_d1) / (S * sigma * math.sqrt(T))


def call_theta(S, K, T, r, sigma):
    """Theta of a call (per year, divide by 252 for daily)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    t1 = -(S * norm.pdf(_d1) * sigma) / (2 * math.sqrt(T))
    t2 = -r * K * math.exp(-r * T) * norm.cdf(_d2)
    return t1 + t2


def vega(S, K, T, r, sigma):
    """Vega (same for calls and puts). Per 1.0 change in vol."""
    if T <= 0 or sigma <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(_d1) * math.sqrt(T)


# ── Implied Volatility ──────────────────────────────────────────────────────

def implied_vol(option_price, S, K, T, r, is_call=True, tol=1e-6, max_iter=100):
    """
    Solve for implied volatility using Newton-Raphson.
    Returns IV (annualized) or None if convergence fails.
    """
    if T <= 0:
        return None

    # Intrinsic value check
    intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
    if option_price <= intrinsic + tol:
        return 0.001  # Near-zero time value

    # Initial guess
    sigma = 0.25

    price_fn = call_price if is_call else put_price

    for _ in range(max_iter):
        price = price_fn(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)

        if v < 1e-10:
            # Vega too small, can't converge
            break

        diff = price - option_price
        sigma -= diff / v

        if sigma <= 0.001:
            sigma = 0.001
        if sigma > 5.0:
            sigma = 5.0

        if abs(diff) < tol:
            return sigma

    return sigma  # Return best estimate even if not fully converged


# ── Utility ──────────────────────────────────────────────────────────────────

def find_strike_for_delta(S, strikes, target_delta, T, r, sigma, is_call=True):
    """
    Find the strike from a list that gives the closest delta to target_delta.

    Parameters
    ----------
    S : float - underlying price
    strikes : list[float] - available strike prices
    target_delta : float - desired absolute delta (e.g., 0.30, 0.50)
    T : float - time to expiry in years
    r : float - risk-free rate
    sigma : float - implied volatility estimate
    is_call : bool - True for calls, False for puts

    Returns
    -------
    (best_strike, actual_delta)
    """
    delta_fn = call_delta if is_call else put_delta
    target = target_delta if is_call else -target_delta  # puts have negative delta

    best_strike = None
    best_diff = float("inf")
    best_delta = 0.0

    for K in strikes:
        d = delta_fn(S, K, T, r, sigma)
        diff = abs(d - target)
        if diff < best_diff:
            best_diff = diff
            best_strike = K
            best_delta = d

    return best_strike, abs(best_delta)
