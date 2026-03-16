"""
Intraday Condition Scanner (Phase 2)
======================================
Defines all observable market conditions that can be computed at any bar close.
Each condition function takes an intraday bar DataFrame and returns a boolean
Series (mask) indicating where the condition is true.

Conditions are organized by category matching the research plan.
"""
import numpy as np
import pandas as pd


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Compute cumulative VWAP within each session."""
    dates = bars.index.date
    cum_vol_price = (bars["close"] * bars["volume"]).groupby(dates).cumsum()
    cum_vol = bars["volume"].groupby(dates).cumsum()
    return cum_vol_price / cum_vol.replace(0, np.nan)


def _session_open(bars: pd.DataFrame) -> pd.Series:
    """Get the session open price for each bar."""
    dates = bars.index.date
    first_open = bars.groupby(dates)["open"].transform("first")
    return first_open


def _session_high(bars: pd.DataFrame) -> pd.Series:
    """Running session high at each bar."""
    dates = bars.index.date
    return bars.groupby(dates)["high"].cummax()


def _session_low(bars: pd.DataFrame) -> pd.Series:
    """Running session low at each bar."""
    dates = bars.index.date
    return bars.groupby(dates)["low"].cummin()


def _prior_day_range(bars: pd.DataFrame, daily: pd.DataFrame) -> tuple:
    """Get prior day's high and low for each intraday bar."""
    # Strip timezone from daily index for date matching
    daily_clean = daily.copy()
    if daily_clean.index.tz is not None:
        daily_clean.index = daily_clean.index.tz_localize(None)
    daily_clean.index = daily_clean.index.normalize()

    prev_high = daily_clean["high"].shift(1)
    prev_low = daily_clean["low"].shift(1)

    # Map each intraday bar to its date, then look up prior day values
    bar_dates = pd.DatetimeIndex(bars.index.date)
    ph = prev_high.reindex(bar_dates, method="ffill")
    pl = prev_low.reindex(bar_dates, method="ffill")
    ph.index = bars.index
    pl.index = bars.index
    return ph, pl


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY A: Price vs HTF Range Levels
# ═════════════════════════════════════════════════════════════════════════════

def cond_breakout_above_prior_day_high(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Bar where price first crosses above prior day's high."""
    ph, _ = _prior_day_range(bars, daily)
    above = bars["close"] > ph
    above_prev = bars["close"].shift(1) <= ph
    return above & above_prev


def cond_breakout_below_prior_day_low(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Bar where price first crosses below prior day's low."""
    _, pl = _prior_day_range(bars, daily)
    below = bars["close"] < pl
    below_prev = bars["close"].shift(1) >= pl
    return below & below_prev


def cond_above_prior_day_high(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Price is above prior day's high."""
    ph, _ = _prior_day_range(bars, daily)
    return bars["close"] > ph


def cond_below_prior_day_low(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Price is below prior day's low."""
    _, pl = _prior_day_range(bars, daily)
    return bars["close"] < pl


def _clean_daily_for_reindex(daily: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone and normalize daily index for date-based reindexing."""
    d = daily.copy()
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    d.index = d.index.normalize()
    return d


def cond_breakout_above_prior_week_high(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Bar where price first crosses above prior week's high (rolling 5-day)."""
    d = _clean_daily_for_reindex(daily)
    wh = d["high"].rolling(5).max().shift(1)
    bar_dates = pd.DatetimeIndex(bars.index.date)
    wh_intra = wh.reindex(bar_dates, method="ffill")
    wh_intra.index = bars.index
    above = bars["close"] > wh_intra
    above_prev = bars["close"].shift(1) <= wh_intra
    return above & above_prev


def cond_breakout_below_prior_week_low(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Bar where price first crosses below prior week's low (rolling 5-day)."""
    d = _clean_daily_for_reindex(daily)
    wl = d["low"].rolling(5).min().shift(1)
    bar_dates = pd.DatetimeIndex(bars.index.date)
    wl_intra = wl.reindex(bar_dates, method="ffill")
    wl_intra.index = bars.index
    below = bars["close"] < wl_intra
    below_prev = bars["close"].shift(1) >= wl_intra
    return below & below_prev


def cond_breakout_above_prior_month_high(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Bar where price first crosses above prior month's high (rolling 21-day)."""
    d = _clean_daily_for_reindex(daily)
    mh = d["high"].rolling(21).max().shift(1)
    bar_dates = pd.DatetimeIndex(bars.index.date)
    mh_intra = mh.reindex(bar_dates, method="ffill")
    mh_intra.index = bars.index
    above = bars["close"] > mh_intra
    above_prev = bars["close"].shift(1) <= mh_intra
    return above & above_prev


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY B: Price-Based Conditions
# ═════════════════════════════════════════════════════════════════════════════

def cond_above_vwap(bars: pd.DataFrame) -> pd.Series:
    """Price above session VWAP."""
    vwap = _session_vwap(bars)
    return bars["close"] > vwap


def cond_cross_above_vwap(bars: pd.DataFrame) -> pd.Series:
    """Bar where price crosses above VWAP."""
    vwap = _session_vwap(bars)
    above = bars["close"] > vwap
    below_prev = bars["close"].shift(1) <= vwap
    return above & below_prev


def cond_above_ema(bars: pd.DataFrame, span: int = 20) -> pd.Series:
    """Price above N-period EMA."""
    ema = _ema(bars["close"], span)
    return bars["close"] > ema


def cond_new_n_bar_high(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Price making the highest close of the last N bars."""
    return bars["close"] == bars["close"].rolling(n).max()


def cond_new_n_bar_low(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Price making the lowest close of the last N bars."""
    return bars["close"] == bars["close"].rolling(n).min()


def cond_above_opening_range_high(bars: pd.DataFrame, n_bars: int = 6) -> pd.Series:
    """Price above the opening range high (first N bars of session)."""
    dates = bars.index.date
    # Compute opening range high per day
    or_high = bars.groupby(dates)["high"].transform(
        lambda x: x.iloc[:n_bars].max() if len(x) >= n_bars else np.nan
    )
    # Count bar number within session
    bar_num = bars.groupby(dates).cumcount()
    return (bars["close"] > or_high) & (bar_num >= n_bars)


def cond_consecutive_up_bars(bars: pd.DataFrame, n: int = 3) -> pd.Series:
    """N consecutive bars closing higher than previous bar."""
    up = bars["close"] > bars["close"].shift(1)
    result = up.copy()
    for i in range(1, n):
        result = result & up.shift(i)
    return result.fillna(False)


def cond_consecutive_down_bars(bars: pd.DataFrame, n: int = 3) -> pd.Series:
    """N consecutive bars closing lower than previous bar."""
    down = bars["close"] < bars["close"].shift(1)
    result = down.copy()
    for i in range(1, n):
        result = result & down.shift(i)
    return result.fillna(False)


def cond_near_session_high(bars: pd.DataFrame, pct: float = 0.002) -> pd.Series:
    """Price within X% of the session high."""
    sess_high = _session_high(bars)
    return (sess_high - bars["close"]) / sess_high < pct


def cond_near_session_low(bars: pd.DataFrame, pct: float = 0.002) -> pd.Series:
    """Price within X% of the session low."""
    sess_low = _session_low(bars)
    return (bars["close"] - sess_low) / bars["close"] < pct


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY C: Prior Bar Character
# ═════════════════════════════════════════════════════════════════════════════

def cond_prior_bar_wide_range(bars: pd.DataFrame, threshold: float = 1.5) -> pd.Series:
    """Prior bar's range > threshold × 20-bar average range."""
    bar_range = bars["high"] - bars["low"]
    avg_range = bar_range.rolling(20).mean()
    return (bar_range.shift(1) > threshold * avg_range.shift(1))


def cond_prior_bar_narrow_range(bars: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Prior bar's range < threshold × 20-bar average range."""
    bar_range = bars["high"] - bars["low"]
    avg_range = bar_range.rolling(20).mean()
    return (bar_range.shift(1) < threshold * avg_range.shift(1))


def cond_prior_bar_close_near_high(bars: pd.DataFrame) -> pd.Series:
    """Prior bar closed in the top 25% of its range."""
    br = (bars["high"].shift(1) - bars["low"].shift(1)).replace(0, np.nan)
    loc = (bars["close"].shift(1) - bars["low"].shift(1)) / br
    return loc > 0.75


def cond_prior_bar_close_near_low(bars: pd.DataFrame) -> pd.Series:
    """Prior bar closed in the bottom 25% of its range."""
    br = (bars["high"].shift(1) - bars["low"].shift(1)).replace(0, np.nan)
    loc = (bars["close"].shift(1) - bars["low"].shift(1)) / br
    return loc < 0.25


def cond_prior_bar_high_volume(bars: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    """Prior bar volume > threshold × 20-bar average volume."""
    avg_vol = bars["volume"].rolling(20).mean()
    return bars["volume"].shift(1) > threshold * avg_vol.shift(1)


def cond_prior_bar_conviction(bars: pd.DataFrame) -> pd.Series:
    """Prior bar: wide range + high volume + close near high (bullish conviction)."""
    wide = cond_prior_bar_wide_range(bars, 1.5)
    high_vol = cond_prior_bar_high_volume(bars, 1.5)
    close_high = cond_prior_bar_close_near_high(bars)
    return wide & high_vol & close_high


def cond_prior_bar_exhaustion(bars: pd.DataFrame) -> pd.Series:
    """Prior bar: wide range + high volume + close near low (bearish exhaustion / selling climax)."""
    wide = cond_prior_bar_wide_range(bars, 1.5)
    high_vol = cond_prior_bar_high_volume(bars, 1.5)
    close_low = cond_prior_bar_close_near_low(bars)
    return wide & high_vol & close_low


def cond_range_contraction(bars: pd.DataFrame, n: int = 3) -> pd.Series:
    """N consecutive bars with decreasing range (coiling)."""
    bar_range = bars["high"] - bars["low"]
    shrinking = bar_range < bar_range.shift(1)
    result = shrinking.copy()
    for i in range(1, n):
        result = result & shrinking.shift(i)
    return result.fillna(False)


def cond_range_expansion_after_contraction(bars: pd.DataFrame, contraction_bars: int = 3) -> pd.Series:
    """Current bar range > prior bar range, after N bars of contraction."""
    bar_range = bars["high"] - bars["low"]
    contracted = cond_range_contraction(bars, contraction_bars).shift(1)
    expanding = bar_range > bar_range.shift(1) * 1.5
    return contracted & expanding


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY D: Volume-Based Conditions
# ═════════════════════════════════════════════════════════════════════════════

def cond_high_relative_volume(bars: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    """Current bar volume > threshold × 20-bar average."""
    avg_vol = bars["volume"].rolling(20).mean()
    return bars["volume"] > threshold * avg_vol


def cond_volume_surge_bullish(bars: pd.DataFrame, vol_threshold: float = 2.0) -> pd.Series:
    """High volume bar with close in top 25% of range (bullish thrust)."""
    high_vol = cond_high_relative_volume(bars, vol_threshold)
    br = (bars["high"] - bars["low"]).replace(0, np.nan)
    close_loc = (bars["close"] - bars["low"]) / br
    return high_vol & (close_loc > 0.75)


def cond_volume_surge_bearish(bars: pd.DataFrame, vol_threshold: float = 2.0) -> pd.Series:
    """High volume bar with close in bottom 25% of range (bearish thrust)."""
    high_vol = cond_high_relative_volume(bars, vol_threshold)
    br = (bars["high"] - bars["low"]).replace(0, np.nan)
    close_loc = (bars["close"] - bars["low"]) / br
    return high_vol & (close_loc < 0.25)


def cond_volume_dryup(bars: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Current bar volume < threshold × 20-bar average (drying up)."""
    avg_vol = bars["volume"].rolling(20).mean()
    return bars["volume"] < threshold * avg_vol


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY E: Volatility/Range Conditions
# ═════════════════════════════════════════════════════════════════════════════

def cond_atr_squeeze(bars: pd.DataFrame) -> pd.Series:
    """Short-term ATR < 0.7 × longer-term ATR (volatility compression)."""
    br = bars["high"] - bars["low"]
    short_atr = br.rolling(5).mean()
    long_atr = br.rolling(50).mean()
    return short_atr < 0.7 * long_atr


def cond_atr_expansion(bars: pd.DataFrame) -> pd.Series:
    """Short-term ATR > 1.5 × longer-term ATR (volatility expansion)."""
    br = bars["high"] - bars["low"]
    short_atr = br.rolling(5).mean()
    long_atr = br.rolling(50).mean()
    return short_atr > 1.5 * long_atr


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY F: Time-of-Day Conditions
# ═════════════════════════════════════════════════════════════════════════════

def cond_first_30_min(bars: pd.DataFrame) -> pd.Series:
    """Bar is within the first 30 minutes of the session."""
    t = bars.index.time
    from datetime import time as dt_time
    return pd.Series([(dt_time(9, 30) <= x <= dt_time(10, 0)) for x in t], index=bars.index)


def cond_mid_morning(bars: pd.DataFrame) -> pd.Series:
    """Bar is in mid-morning (10:00 - 11:30)."""
    t = bars.index.time
    from datetime import time as dt_time
    return pd.Series([(dt_time(10, 0) < x <= dt_time(11, 30)) for x in t], index=bars.index)


def cond_lunch_hour(bars: pd.DataFrame) -> pd.Series:
    """Bar is during lunch (11:30 - 13:30)."""
    t = bars.index.time
    from datetime import time as dt_time
    return pd.Series([(dt_time(11, 30) < x <= dt_time(13, 30)) for x in t], index=bars.index)


def cond_afternoon(bars: pd.DataFrame) -> pd.Series:
    """Bar is in afternoon session (13:30 - 15:30)."""
    t = bars.index.time
    from datetime import time as dt_time
    return pd.Series([(dt_time(13, 30) < x <= dt_time(15, 30)) for x in t], index=bars.index)


def cond_last_30_min(bars: pd.DataFrame) -> pd.Series:
    """Bar is in the last 30 minutes (15:30 - 16:00)."""
    t = bars.index.time
    from datetime import time as dt_time
    return pd.Series([(dt_time(15, 30) < x <= dt_time(16, 0)) for x in t], index=bars.index)


# ═════════════════════════════════════════════════════════════════════════════
# CONDITION REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

# Conditions that only need intraday bars
INTRADAY_CONDITIONS = {
    # Price-based
    "above_vwap": cond_above_vwap,
    "cross_above_vwap": cond_cross_above_vwap,
    "above_ema_9": lambda bars: cond_above_ema(bars, 9),
    "above_ema_20": lambda bars: cond_above_ema(bars, 20),
    "new_20bar_high": lambda bars: cond_new_n_bar_high(bars, 20),
    "new_20bar_low": lambda bars: cond_new_n_bar_low(bars, 20),
    "new_50bar_high": lambda bars: cond_new_n_bar_high(bars, 50),
    "above_or_high_30m": lambda bars: cond_above_opening_range_high(bars, 6),  # 6 bars × 5m = 30min
    "3_consecutive_up": lambda bars: cond_consecutive_up_bars(bars, 3),
    "3_consecutive_down": lambda bars: cond_consecutive_down_bars(bars, 3),
    "5_consecutive_up": lambda bars: cond_consecutive_up_bars(bars, 5),
    "near_session_high": cond_near_session_high,
    "near_session_low": cond_near_session_low,

    # Prior bar character
    "prior_bar_wide_range": cond_prior_bar_wide_range,
    "prior_bar_narrow_range": cond_prior_bar_narrow_range,
    "prior_bar_close_near_high": cond_prior_bar_close_near_high,
    "prior_bar_close_near_low": cond_prior_bar_close_near_low,
    "prior_bar_high_volume": cond_prior_bar_high_volume,
    "prior_bar_conviction_bullish": cond_prior_bar_conviction,
    "prior_bar_exhaustion_bearish": cond_prior_bar_exhaustion,
    "range_contraction_3bar": lambda bars: cond_range_contraction(bars, 3),
    "range_expansion_after_contraction": cond_range_expansion_after_contraction,

    # Volume
    "high_relative_volume": cond_high_relative_volume,
    "volume_surge_bullish": cond_volume_surge_bullish,
    "volume_surge_bearish": cond_volume_surge_bearish,
    "volume_dryup": cond_volume_dryup,

    # Volatility
    "atr_squeeze": cond_atr_squeeze,
    "atr_expansion": cond_atr_expansion,

    # Time of day
    "first_30_min": cond_first_30_min,
    "mid_morning": cond_mid_morning,
    "lunch_hour": cond_lunch_hour,
    "afternoon": cond_afternoon,
    "last_30_min": cond_last_30_min,
}

# Conditions that need both intraday bars AND daily bars
HTF_BRIDGE_CONDITIONS = {
    "breakout_above_prior_day_high": cond_breakout_above_prior_day_high,
    "breakout_below_prior_day_low": cond_breakout_below_prior_day_low,
    "above_prior_day_high": cond_above_prior_day_high,
    "below_prior_day_low": cond_below_prior_day_low,
    "breakout_above_prior_week_high": cond_breakout_above_prior_week_high,
    "breakout_below_prior_week_low": cond_breakout_below_prior_week_low,
    "breakout_above_prior_month_high": cond_breakout_above_prior_month_high,
}
