"""
Higher-Timeframe Context Computation (Phase 1)
================================================
Computes all HTF factors from daily/weekly bars and produces
a per-day context tag DataFrame that gets attached to every
intraday occurrence in the P&L snapshot engine.
"""
import numpy as np
import pandas as pd


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = compute_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period).mean()


def compute_all_htf_factors(daily: pd.DataFrame, weekly: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute all Phase 1 HTF factors from daily (and optionally weekly) bars.

    Returns a DataFrame indexed by date with one column per factor.
    Each row represents the HTF context for that trading day,
    computed using ONLY prior data (no lookahead).

    Parameters
    ----------
    daily : DataFrame with columns: open, high, low, close, volume
    weekly : DataFrame (optional) with same columns
    """
    df = daily.copy()
    ctx = pd.DataFrame(index=df.index)

    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    # ── A. Trend & Regime ────────────────────────────────────────────────

    # 1. Trend direction (price vs EMAs) — use prior day's close vs prior day's EMA
    for span in [10, 20, 50, 200]:
        ema = compute_ema(c, span)
        ctx[f"above_ema_{span}"] = (c.shift(1) > ema.shift(1)).astype(int)

    # 2. Trend strength (ADX)
    adx = compute_adx(h, l, c, 14)
    ctx["adx_14"] = adx.shift(1)
    ctx["adx_trending"] = (adx.shift(1) > 25).astype(int)

    # 3. Volatility regime
    atr = compute_atr(h, l, c, 14)
    atr_pct = atr / c  # ATR as % of price
    ctx["atr_14_pct"] = atr_pct.shift(1)
    ctx["atr_percentile"] = atr_pct.shift(1).rolling(50).rank(pct=True)

    # ── B. Prior Period Range Context ────────────────────────────────────

    # 5. Position vs prior day range
    prev_high = h.shift(1)
    prev_low = l.shift(1)
    ctx["above_prior_day_high"] = (o > prev_high).astype(int)  # opens above yesterday's high
    ctx["below_prior_day_low"] = (o < prev_low).astype(int)    # opens below yesterday's low
    ctx["inside_prior_day"] = ((o <= prev_high) & (o >= prev_low)).astype(int)

    # 6. Position vs prior week range
    if weekly is not None and not weekly.empty:
        # Map each daily bar to its prior week's range
        weekly_high = weekly["high"].shift(1)
        weekly_low = weekly["low"].shift(1)
        # Reindex to daily
        wh = weekly_high.reindex(df.index, method="ffill")
        wl = weekly_low.reindex(df.index, method="ffill")
        ctx["above_prior_week_high"] = (c.shift(1) > wh).astype(int)
        ctx["below_prior_week_low"] = (c.shift(1) < wl).astype(int)
    else:
        # Approximate weekly range from daily data (rolling 5-day high/low)
        wh = h.rolling(5).max().shift(1)
        wl = l.rolling(5).min().shift(1)
        ctx["above_prior_week_high"] = (c.shift(1) > wh).astype(int)
        ctx["below_prior_week_low"] = (c.shift(1) < wl).astype(int)

    # 7. Position vs prior month range (rolling 21-day)
    mh = h.rolling(21).max().shift(1)
    ml = l.rolling(21).min().shift(1)
    ctx["above_prior_month_high"] = (c.shift(1) > mh).astype(int)
    ctx["below_prior_month_low"] = (c.shift(1) < ml).astype(int)

    # 8. Multi-period range alignment
    ctx["aligned_daily_weekly_up"] = (ctx.get("above_prior_day_high", 0) &
                                       ctx.get("above_prior_week_high", 0)).astype(int)
    ctx["aligned_daily_weekly_down"] = (ctx.get("below_prior_day_low", 0) &
                                         ctx.get("below_prior_week_low", 0)).astype(int)

    # 9. Distance from prior range edge (as % of prior day's range)
    prev_range = prev_high - prev_low
    prev_range = prev_range.replace(0, np.nan)
    ctx["dist_from_prior_day_high_pct"] = (o - prev_high) / prev_range
    ctx["dist_from_prior_day_low_pct"] = (prev_low - o) / prev_range

    # ── C. Prior Day Character ───────────────────────────────────────────

    # 11. Prior day range width (% of close)
    prior_range_pct = (prev_high - prev_low) / c.shift(1)
    ctx["prior_day_range_pct"] = prior_range_pct
    avg_range = prior_range_pct.rolling(20).mean()
    ctx["prior_day_range_vs_avg"] = prior_range_pct / avg_range.replace(0, np.nan)

    # 12. Prior day range percentile (NR/WR classification)
    range_abs = h - l
    ctx["prior_day_range_pctile"] = range_abs.shift(1).rolling(50).rank(pct=True)
    ctx["is_nr4"] = (range_abs.shift(1) == range_abs.shift(1).rolling(4).min()).astype(int)
    ctx["is_nr7"] = (range_abs.shift(1) == range_abs.shift(1).rolling(7).min()).astype(int)

    # 13. Prior day volume vs average
    vol_avg = v.rolling(20).mean()
    ctx["prior_day_rvol"] = v.shift(1) / vol_avg.shift(1).replace(0, np.nan)

    # 14. Prior day volume × range interaction
    ctx["prior_day_high_vol_wide_range"] = (
        (ctx["prior_day_rvol"] > 1.5) & (ctx["prior_day_range_vs_avg"] > 1.5)
    ).astype(int)
    ctx["prior_day_low_vol_wide_range"] = (
        (ctx["prior_day_rvol"] < 0.7) & (ctx["prior_day_range_vs_avg"] > 1.5)
    ).astype(int)

    # 15. Prior day close location
    prev_close = c.shift(1)
    close_loc = (prev_close - prev_low) / (prev_high - prev_low).replace(0, np.nan)
    ctx["prior_day_close_location"] = close_loc
    ctx["prior_day_close_near_high"] = (close_loc > 0.75).astype(int)
    ctx["prior_day_close_near_low"] = (close_loc < 0.25).astype(int)

    # 16. Prior day body vs wick ratio
    body = (c.shift(1) - o.shift(1)).abs()
    full_range = (h.shift(1) - l.shift(1)).replace(0, np.nan)
    ctx["prior_day_body_ratio"] = body / full_range

    # 17. Consecutive day direction
    daily_dir = (c > c.shift(1)).astype(int)  # 1 = up day, 0 = down day
    # Count consecutive same-direction days
    streaks = []
    current_streak = 0
    prev_dir = None
    for d in daily_dir.shift(1):  # shift to avoid lookahead
        if pd.isna(d):
            streaks.append(0)
            continue
        if d == prev_dir:
            current_streak += 1
        else:
            current_streak = 1
        prev_dir = d
        streaks.append(current_streak)
    ctx["consecutive_day_streak"] = streaks

    # 18. Inside day / outside day
    ctx["is_inside_day"] = ((h.shift(1) <= h.shift(2)) & (l.shift(1) >= l.shift(2))).astype(int)
    ctx["is_outside_day"] = ((h.shift(1) > h.shift(2)) & (l.shift(1) < l.shift(2))).astype(int)

    # ── D. Gap & Opening Conditions ──────────────────────────────────────

    # 19. Gap classification
    gap_pct = (o - c.shift(1)) / c.shift(1)
    ctx["gap_pct"] = gap_pct
    ctx["gap_up_small"] = ((gap_pct > 0.001) & (gap_pct <= 0.005)).astype(int)
    ctx["gap_up_medium"] = ((gap_pct > 0.005) & (gap_pct <= 0.01)).astype(int)
    ctx["gap_up_large"] = ((gap_pct > 0.01) & (gap_pct <= 0.02)).astype(int)
    ctx["gap_up_huge"] = (gap_pct > 0.02).astype(int)
    ctx["gap_down_small"] = ((gap_pct < -0.001) & (gap_pct >= -0.005)).astype(int)
    ctx["gap_down_medium"] = ((gap_pct < -0.005) & (gap_pct >= -0.01)).astype(int)
    ctx["gap_down_large"] = ((gap_pct < -0.01) & (gap_pct >= -0.02)).astype(int)
    ctx["gap_down_huge"] = (gap_pct < -0.02).astype(int)

    # 20. Gap vs trend alignment
    trend_up = ctx["above_ema_20"]
    ctx["gap_aligned_with_trend"] = (
        ((gap_pct > 0) & (trend_up == 1)) |
        ((gap_pct < 0) & (trend_up == 0))
    ).astype(int)

    # 21. Gap vs prior range
    ctx["gap_above_prior_day_range"] = (o > prev_high).astype(int)
    ctx["gap_below_prior_day_range"] = (o < prev_low).astype(int)

    # ── E. Key Levels ────────────────────────────────────────────────────

    # 23. Distance from 52-week high/low
    high_52w = h.rolling(252).max()
    low_52w = l.rolling(252).min()
    ctx["dist_from_52w_high"] = (c.shift(1) - high_52w.shift(1)) / high_52w.shift(1)
    ctx["dist_from_52w_low"] = (c.shift(1) - low_52w.shift(1)) / low_52w.shift(1)
    ctx["near_52w_high"] = (ctx["dist_from_52w_high"] > -0.05).astype(int)

    # 24. Round number proximity
    round_10 = (c.shift(1) / 10).round() * 10
    ctx["dist_from_round_10"] = (c.shift(1) - round_10).abs() / c.shift(1)

    return ctx


def get_htf_tags_for_date(htf_context: pd.DataFrame, date) -> pd.Series:
    """
    Get the HTF context tags for a specific date.
    Used to attach HTF tags to intraday P&L snapshots.
    """
    date = pd.Timestamp(date).normalize()
    if date in htf_context.index:
        return htf_context.loc[date]

    # Find the most recent prior date
    prior = htf_context.index[htf_context.index <= date]
    if len(prior) > 0:
        return htf_context.loc[prior[-1]]

    return pd.Series(dtype=float)
