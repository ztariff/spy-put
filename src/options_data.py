"""
Options data utilities: expiry selection, IV estimation, strike selection.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

from src.black_scholes import call_delta, put_delta, find_strike_for_delta


# ── Expiry Selection ─────────────────────────────────────────────────────────

# SPY: daily expirations started ~May 2022 (Mon/Wed/Fri from late 2020, then daily)
# QQQ: daily expirations started ~2023 (Mon/Wed/Fri from mid-2022)
# Before daily expirations: fall back to nearest available expiry

def select_expiry(available_expiries: list, trade_date: date) -> date:
    """
    Select the best expiration date for a trade.
    Prefers 0DTE (same-day expiry). Falls back to nearest future expiry.

    Parameters
    ----------
    available_expiries : list[date] - sorted available expiration dates
    trade_date : date - the trade entry date

    Returns
    -------
    date - selected expiry
    """
    if not available_expiries:
        return None

    # Best case: 0DTE
    if trade_date in available_expiries:
        return trade_date

    # Fall back to nearest future expiry
    future = [e for e in available_expiries if e >= trade_date]
    if future:
        return future[0]

    # Shouldn't happen, but return the latest available
    return available_expiries[-1]


def get_next_friday(d: date) -> date:
    """Get the next Friday on or after date d."""
    days_ahead = 4 - d.weekday()  # Friday = 4
    if days_ahead < 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)


# ── IV Estimation ────────────────────────────────────────────────────────────

def estimate_realized_vol(daily_bars: pd.DataFrame, trade_date: date,
                           window: int = 20) -> float:
    """
    Estimate annualized volatility from recent daily close-to-close returns.
    Used as a proxy for implied volatility when historical IV isn't available.

    Parameters
    ----------
    daily_bars : DataFrame with 'close' column, DatetimeIndex
    trade_date : date
    window : int - lookback days

    Returns
    -------
    float - annualized volatility
    """
    # Normalize index for matching
    idx = daily_bars.index
    if hasattr(idx, 'date'):
        dates = idx.date if hasattr(idx, 'tz') else idx.date
    else:
        dates = pd.to_datetime(idx).date

    # Get bars up to trade_date
    mask = dates <= trade_date
    recent = daily_bars[mask].tail(window + 1)

    if len(recent) < 5:
        return 0.20  # Default fallback

    returns = np.log(recent["close"] / recent["close"].shift(1)).dropna()
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)

    # Clamp to reasonable range
    return max(0.08, min(annual_vol, 1.5))


# ── Time to Expiry ───────────────────────────────────────────────────────────

def time_to_expiry_years(entry_time: pd.Timestamp, expiry_date: date) -> float:
    """
    Compute time to expiry in years.
    For 0DTE, this is the fraction of the trading day remaining.

    Parameters
    ----------
    entry_time : pd.Timestamp (timezone-aware, ET)
    expiry_date : date

    Returns
    -------
    float - time to expiry in years (trading year = 252 days)
    """
    if entry_time.date() == expiry_date:
        # 0DTE: calculate hours remaining until 4:00 PM ET close
        market_close_hour = 16.0
        current_hour = entry_time.hour + entry_time.minute / 60.0
        hours_remaining = max(market_close_hour - current_hour, 0.01)
        # Convert to years: hours / (6.5 hours per day * 252 trading days)
        return hours_remaining / (6.5 * 252)
    else:
        # Multi-day: calendar days to expiry
        days_to_expiry = (expiry_date - entry_time.date()).days
        # Convert to trading days (rough: 5/7 ratio)
        trading_days = days_to_expiry * 5 / 7
        return max(trading_days / 252, 0.001)


# ── Strike Selection ─────────────────────────────────────────────────────────

def select_strike(underlying_price: float, available_strikes: list,
                  target_delta: float, T: float, r: float, sigma: float,
                  is_call: bool = True) -> tuple:
    """
    Select the strike closest to a target delta using Black-Scholes.

    Parameters
    ----------
    underlying_price : float
    available_strikes : list[float]
    target_delta : float - absolute delta (0.30, 0.50, etc.)
    T : float - time to expiry in years
    r : float - risk-free rate
    sigma : float - estimated IV
    is_call : bool

    Returns
    -------
    (strike, actual_delta) or (None, None) if no strikes available
    """
    if not available_strikes:
        return None, None

    return find_strike_for_delta(
        S=underlying_price,
        strikes=available_strikes,
        target_delta=target_delta,
        T=T,
        r=r,
        sigma=sigma,
        is_call=is_call,
    )


def get_strikes_for_type(contracts: list, contract_type: str) -> list:
    """Extract sorted unique strikes for a given contract type (CALL/PUT)."""
    strikes = sorted(set(
        c["strike"] for c in contracts
        if c["type"].upper() == contract_type.upper() and c["strike"] is not None
    ))
    return strikes
