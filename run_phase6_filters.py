#!/usr/bin/env python3
"""
Phase 6: Intraday Confirmation Filter Scanner
================================================
Tests whether adding intraday confirmation filters at the moment of entry
improves the edge on our validated rules.

For each rule × filter combination, we compare:
  - Baseline: rule as-is (no intraday filter)
  - Filtered: rule + intraday filter must be true at entry

Filters tested:
  A. RVOL filters (is volume above average at entry?)
  B. Opening range direction (did first 30 min close green?)
  C. VWAP position (is price above session VWAP?)
  D. Prior day range position (where is price vs yesterday's high/low?)
  E. Intraday range width (is the opening range wide or narrow?)
  F. Price momentum confirmation (is price trending up intraday?)
  G. Prior week / month range position

Usage:
    python run_phase6_filters.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import time as dt_time
from dataclasses import dataclass

from src.config import OUTPUT_DIR
from src.downloader import load_data
from src.htf_context import compute_all_htf_factors
from src.conditions import INTRADAY_CONDITIONS

# ── Rules to test (our validated drift rules) ──
@dataclass
class RuleSpec:
    name: str
    condition_name: str
    htf_factor: str
    ticker: str
    max_hold_bars: int

RULES = [
    RuleSpec("HighVolWR_30min_SPY", "first_30_min", "prior_day_high_vol_wide_range", "SPY", 55),
    RuleSpec("PriorDayWeak_30min_SPY", "first_30_min", "prior_day_close_near_low", "SPY", 55),
    RuleSpec("PriorDayWeak_30min_QQQ", "first_30_min", "prior_day_close_near_low", "QQQ", 55),
    RuleSpec("PriorDayWeak_50Hi_SPY", "new_50bar_high", "prior_day_close_near_low", "SPY", 50),
]

SLIPPAGE = 0.0001


def compute_intraday_features(intraday: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all intraday features at each bar that can be used as entry filters.
    Everything uses only data available at the bar close (no lookahead).
    """
    feat = pd.DataFrame(index=intraday.index)
    dates = intraday.index.date

    c = intraday["close"]
    h = intraday["high"]
    l = intraday["low"]
    o = intraday["open"]
    v = intraday["volume"]

    # ── A. RVOL (Relative Volume) ──────────────────────────────────────
    # Intraday RVOL: current bar volume vs 20-bar avg
    avg_vol_20 = v.rolling(20).mean()
    feat["bar_rvol"] = v / avg_vol_20.replace(0, np.nan)
    feat["bar_rvol_gt_1"] = (feat["bar_rvol"] > 1.0).astype(int)
    feat["bar_rvol_gt_1_5"] = (feat["bar_rvol"] > 1.5).astype(int)
    feat["bar_rvol_gt_2"] = (feat["bar_rvol"] > 2.0).astype(int)

    # Session cumulative RVOL: total volume so far today vs same-bar avg
    cum_vol = v.groupby(dates).cumsum()
    bar_num = v.groupby(dates).cumcount() + 1
    # Compare to rolling 20-day avg for same bar position
    avg_daily_vol_20 = v.groupby(dates).sum().rolling(20).mean()
    bar_dates = pd.Series(dates, index=intraday.index)
    total_bars_per_day = v.groupby(dates).count()

    # Simple session RVOL: cumulative volume / expected at this point
    session_vol = cum_vol
    expected_vol = avg_vol_20 * bar_num  # rough approximation
    feat["session_rvol"] = session_vol / expected_vol.replace(0, np.nan)
    feat["session_rvol_gt_1"] = (feat["session_rvol"] > 1.0).astype(int)
    feat["session_rvol_gt_1_5"] = (feat["session_rvol"] > 1.5).astype(int)

    # ── B. Opening Range Direction ─────────────────────────────────────
    # Session open price
    session_open = o.groupby(dates).transform("first")
    feat["above_session_open"] = (c > session_open).astype(int)  # price > open = green so far

    # First bar direction (was the first 5-min bar green?)
    first_close = c.groupby(dates).transform("first")
    first_open = o.groupby(dates).transform("first")
    feat["first_bar_green"] = (first_close > first_open).astype(int)

    # Opening range (first 6 bars = 30 min on 5m)
    or_high = h.groupby(dates).transform(lambda x: x.iloc[:6].max() if len(x) >= 6 else np.nan)
    or_low = l.groupby(dates).transform(lambda x: x.iloc[:6].min() if len(x) >= 6 else np.nan)
    or_mid = (or_high + or_low) / 2
    feat["above_or_mid"] = (c > or_mid).astype(int)
    feat["above_or_high"] = (c > or_high).astype(int)
    feat["below_or_low"] = (c < or_low).astype(int)

    # Opening range width as % of price
    or_width_pct = (or_high - or_low) / session_open
    feat["or_width_pct"] = or_width_pct
    # Compare to 20-day avg opening range width
    daily_or_width = or_width_pct.groupby(dates).first()
    avg_or_width = daily_or_width.rolling(20).mean()
    or_width_ratio = daily_or_width / avg_or_width.replace(0, np.nan)
    # Map back to intraday
    or_ratio_daily = or_width_ratio.reindex(pd.DatetimeIndex(dates), method="ffill")
    or_ratio_daily.index = intraday.index
    feat["or_width_vs_avg"] = or_ratio_daily
    feat["or_narrow"] = (or_ratio_daily < 0.7).astype(int)
    feat["or_wide"] = (or_ratio_daily > 1.5).astype(int)

    # ── C. VWAP Position ───────────────────────────────────────────────
    cum_vol_price = (c * v).groupby(dates).cumsum()
    cum_vol_clean = v.groupby(dates).cumsum()
    vwap = cum_vol_price / cum_vol_clean.replace(0, np.nan)
    feat["above_vwap"] = (c > vwap).astype(int)
    feat["pct_from_vwap"] = (c - vwap) / vwap

    # ── D. Prior Day Range Position ────────────────────────────────────
    # Strip tz from daily
    daily_clean = daily.copy()
    if daily_clean.index.tz is not None:
        daily_clean.index = daily_clean.index.tz_localize(None)
    daily_clean.index = daily_clean.index.normalize()

    prev_high = daily_clean["high"].shift(1)
    prev_low = daily_clean["low"].shift(1)
    prev_close = daily_clean["close"].shift(1)

    # Map to intraday
    bar_dates_idx = pd.DatetimeIndex(dates)
    ph = prev_high.reindex(bar_dates_idx, method="ffill")
    pl = prev_low.reindex(bar_dates_idx, method="ffill")
    pc = prev_close.reindex(bar_dates_idx, method="ffill")
    ph.index = intraday.index
    pl.index = intraday.index
    pc.index = intraday.index

    feat["above_prior_day_high"] = (c > ph).astype(int)
    feat["below_prior_day_low"] = (c < pl).astype(int)
    feat["inside_prior_day_range"] = ((c <= ph) & (c >= pl)).astype(int)
    feat["above_prior_close"] = (c > pc).astype(int)

    # Position within prior day range (0 = at low, 1 = at high)
    prior_range = (ph - pl).replace(0, np.nan)
    feat["pos_in_prior_range"] = (c - pl) / prior_range
    feat["in_upper_half_prior_range"] = (feat["pos_in_prior_range"] > 0.5).astype(int)
    feat["in_lower_half_prior_range"] = (feat["pos_in_prior_range"] < 0.5).astype(int)

    # ── E. Prior Week Range Position ───────────────────────────────────
    pw_high = daily_clean["high"].rolling(5).max().shift(1)
    pw_low = daily_clean["low"].rolling(5).min().shift(1)
    pwh = pw_high.reindex(bar_dates_idx, method="ffill")
    pwl = pw_low.reindex(bar_dates_idx, method="ffill")
    pwh.index = intraday.index
    pwl.index = intraday.index
    feat["above_prior_week_high"] = (c > pwh).astype(int)
    feat["below_prior_week_low"] = (c < pwl).astype(int)
    feat["inside_prior_week"] = ((c <= pwh) & (c >= pwl)).astype(int)

    # ── F. Prior Month Range Position ──────────────────────────────────
    pm_high = daily_clean["high"].rolling(21).max().shift(1)
    pm_low = daily_clean["low"].rolling(21).min().shift(1)
    pmh = pm_high.reindex(bar_dates_idx, method="ffill")
    pml = pm_low.reindex(bar_dates_idx, method="ffill")
    pmh.index = intraday.index
    pml.index = intraday.index
    feat["above_prior_month_high"] = (c > pmh).astype(int)
    feat["below_prior_month_low"] = (c < pml).astype(int)

    # ── G. Intraday Momentum ──────────────────────────────────────────
    # EMA crossover
    ema_fast = c.ewm(span=9, adjust=False).mean()
    ema_slow = c.ewm(span=20, adjust=False).mean()
    feat["ema9_above_ema20"] = (ema_fast > ema_slow).astype(int)

    # Session return so far (from open)
    session_return = (c - session_open) / session_open
    feat["session_return_positive"] = (session_return > 0).astype(int)
    feat["session_return_gt_0_1pct"] = (session_return > 0.001).astype(int)
    feat["session_return_gt_0_2pct"] = (session_return > 0.002).astype(int)

    # Running session high — are we near it?
    sess_high = h.groupby(dates).cummax()
    feat["at_session_high"] = ((sess_high - c) / c < 0.001).astype(int)

    # Bar-level momentum: last 3 bars all up?
    up_bar = (c > c.shift(1)).fillna(False).astype(bool)
    feat["last_3_bars_up"] = (up_bar & up_bar.shift(1).fillna(False) & up_bar.shift(2).fillna(False)).astype(int)

    return feat


def run_filter_test():
    """Test each filter on each rule and compare to baseline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Select filters to test (column names from compute_intraday_features)
    FILTERS_TO_TEST = [
        # RVOL
        ("bar_rvol_gt_1", "Bar RVOL > 1.0"),
        ("bar_rvol_gt_1_5", "Bar RVOL > 1.5"),
        ("bar_rvol_gt_2", "Bar RVOL > 2.0"),
        ("session_rvol_gt_1", "Session RVOL > 1.0"),
        ("session_rvol_gt_1_5", "Session RVOL > 1.5"),
        # Opening range
        ("above_session_open", "Price > Session Open"),
        ("first_bar_green", "First Bar Green"),
        ("above_or_mid", "Above OR Midpoint"),
        ("above_or_high", "Above OR High"),
        ("or_narrow", "Narrow Opening Range"),
        ("or_wide", "Wide Opening Range"),
        # VWAP
        ("above_vwap", "Above VWAP"),
        # Prior day range
        ("above_prior_day_high", "Above Prior Day High"),
        ("inside_prior_day_range", "Inside Prior Day Range"),
        ("above_prior_close", "Above Prior Close"),
        ("in_upper_half_prior_range", "Upper Half Prior Range"),
        ("in_lower_half_prior_range", "Lower Half Prior Range"),
        # Prior week/month
        ("above_prior_week_high", "Above Prior Week High"),
        ("below_prior_week_low", "Below Prior Week Low"),
        ("above_prior_month_high", "Above Prior Month High"),
        ("below_prior_month_low", "Below Prior Month Low"),
        ("inside_prior_week", "Inside Prior Week"),
        # Momentum
        ("ema9_above_ema20", "EMA9 > EMA20"),
        ("session_return_positive", "Session Return > 0"),
        ("session_return_gt_0_1pct", "Session Return > 0.1%"),
        ("session_return_gt_0_2pct", "Session Return > 0.2%"),
        ("at_session_high", "At Session High"),
        ("last_3_bars_up", "Last 3 Bars Up"),
    ]

    all_results = []
    data_cache = {}

    print("=" * 80)
    print("PHASE 6: INTRADAY CONFIRMATION FILTER SCANNER")
    print(f"  Testing {len(FILTERS_TO_TEST)} filters × {len(RULES)} rules = {len(FILTERS_TO_TEST)*len(RULES)} combinations")
    print("=" * 80)

    for rule in RULES:
        print(f"\n{'━'*60}")
        print(f"  Rule: {rule.name}")
        print(f"{'━'*60}")

        # Load data
        cache_key = rule.ticker
        if cache_key not in data_cache:
            intraday = load_data(rule.ticker, "5m")
            daily = load_data(rule.ticker, "1D")
            weekly = load_data(rule.ticker, "1W")

            daily_clean = daily.copy()
            if daily_clean.index.tz is not None:
                daily_clean.index = daily_clean.index.tz_localize(None)
            weekly_clean = weekly.copy()
            if weekly_clean.index.tz is not None:
                weekly_clean.index = weekly_clean.index.tz_localize(None)

            # Filter regular hours
            if hasattr(intraday.index, 'time'):
                time_mask = [(dt_time(9, 30) <= t <= dt_time(16, 0)) for t in intraday.index.time]
                intraday = intraday[time_mask]

            htf = compute_all_htf_factors(daily_clean, weekly_clean)
            features = compute_intraday_features(intraday, daily)

            data_cache[cache_key] = (intraday, daily, htf, features)

        intraday, daily, htf, features = data_cache[cache_key]

        # Get condition mask
        cond_fn = INTRADAY_CONDITIONS[rule.condition_name]
        try:
            mask = cond_fn(intraday)
            if isinstance(mask, pd.Series):
                mask = mask.fillna(False).astype(bool)
        except Exception as e:
            print(f"  Condition error: {e}")
            continue

        # Get HTF factor
        htf_series = htf.get(rule.htf_factor)
        if htf_series is None:
            continue

        # Build entry signal (condition + HTF + first per day)
        entries = []
        dates = intraday.index.date
        traded_dates = set()
        for i in range(len(intraday)):
            if not (mask.iloc[i] if isinstance(mask, pd.Series) else mask[i]):
                entries.append(False)
                continue
            bar_date = dates[i]
            bar_date_ts = pd.Timestamp(bar_date)
            if bar_date_ts in htf_series.index:
                htf_val = htf_series.loc[bar_date_ts]
            else:
                prior = htf_series.index[htf_series.index <= bar_date_ts]
                htf_val = htf_series.loc[prior[-1]] if len(prior) > 0 else 0
            if htf_val != 1 or bar_date in traded_dates:
                entries.append(False)
                continue
            entries.append(True)
            traded_dates.add(bar_date)

        entry_mask = pd.Series(entries, index=intraday.index)
        entry_indices = intraday.index[entry_mask]

        if len(entry_indices) == 0:
            print("  No entries")
            continue

        # Compute forward returns for each entry
        close = intraday["close"].values
        idx_map = {ts: i for i, ts in enumerate(intraday.index)}

        forward_pnl = []
        for ts in entry_indices:
            i = idx_map[ts]
            entry_price = close[i] * (1 + SLIPPAGE)
            # Find exit: time exit or EOD
            exit_bar = min(i + rule.max_hold_bars, len(close) - 1)
            # Check for EOD before time exit
            for j in range(i + 1, exit_bar + 1):
                if j >= len(close):
                    break
                if dates[j] != dates[i]:
                    exit_bar = j - 1
                    break
            exit_price = close[exit_bar] * (1 - SLIPPAGE)
            pnl_pct = (exit_price - entry_price) / entry_price
            forward_pnl.append(pnl_pct)

        pnl_series = pd.Series(forward_pnl, index=entry_indices)

        # Baseline stats
        n_base = len(pnl_series)
        mean_base = pnl_series.mean()
        wr_base = (pnl_series > 0).mean()
        t_base = mean_base / (pnl_series.std() / np.sqrt(n_base)) if pnl_series.std() > 0 else 0
        sharpe_base = mean_base / pnl_series.std() * np.sqrt(252) if pnl_series.std() > 0 else 0

        all_results.append({
            "rule": rule.name, "filter": "BASELINE (no filter)",
            "filter_col": "", "n_trades": n_base,
            "mean_pnl_pct": mean_base, "win_rate": wr_base,
            "t_stat": t_base, "sharpe": sharpe_base,
            "improvement": 0,
        })

        print(f"  BASELINE: {n_base} trades, mean {mean_base*100:+.4f}%, WR {wr_base:.1%}, t={t_base:.2f}")

        # Test each filter
        for filter_col, filter_name in FILTERS_TO_TEST:
            if filter_col not in features.columns:
                continue

            filter_vals = features[filter_col].reindex(entry_indices)
            filtered_pnl = pnl_series[filter_vals == 1]

            n_filt = len(filtered_pnl)
            if n_filt < 20:  # need minimum sample
                continue

            mean_filt = filtered_pnl.mean()
            wr_filt = (filtered_pnl > 0).mean()
            t_filt = mean_filt / (filtered_pnl.std() / np.sqrt(n_filt)) if filtered_pnl.std() > 0 else 0
            sharpe_filt = mean_filt / filtered_pnl.std() * np.sqrt(252) if filtered_pnl.std() > 0 else 0

            improvement = mean_filt - mean_base

            all_results.append({
                "rule": rule.name, "filter": filter_name,
                "filter_col": filter_col, "n_trades": n_filt,
                "mean_pnl_pct": mean_filt, "win_rate": wr_filt,
                "t_stat": t_filt, "sharpe": sharpe_filt,
                "improvement": improvement,
            })

            # Highlight significant improvements
            if improvement > 0 and t_filt > t_base:
                marker = "★" if improvement > mean_base * 0.5 else "+"
                print(f"  {marker} {filter_name:<30} {n_filt:>4} trades, "
                      f"mean {mean_filt*100:+.4f}%, WR {wr_filt:.1%}, "
                      f"t={t_filt:.2f}  (Δ={improvement*100:+.4f}%)")
            elif improvement < -mean_base * 0.3:
                print(f"  ✗ {filter_name:<30} {n_filt:>4} trades, "
                      f"mean {mean_filt*100:+.4f}%  (HURTS: Δ={improvement*100:+.4f}%)")

    # ── Save results ──
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_DIR, "phase6_filter_results.csv"), index=False)

    # ── Summary: best filters across all rules ──
    print(f"\n\n{'='*80}")
    print("TOP FILTERS BY IMPROVEMENT (across all rules)")
    print(f"{'='*80}")

    # Average improvement by filter across rules
    filter_agg = df[df["filter"] != "BASELINE (no filter)"].groupby("filter").agg(
        avg_improvement=("improvement", "mean"),
        avg_t_stat=("t_stat", "mean"),
        avg_win_rate=("win_rate", "mean"),
        avg_mean_pnl=("mean_pnl_pct", "mean"),
        rules_improved=("improvement", lambda x: (x > 0).sum()),
        total_rules=("improvement", "count"),
        avg_trades=("n_trades", "mean"),
    ).sort_values("avg_improvement", ascending=False)

    print(f"\n{'Filter':<35} {'Avg Δ PnL%':>10} {'Avg t':>7} {'Avg WR':>7} {'Improved':>10} {'Avg N':>7}")
    print("-" * 80)
    for filt, row in filter_agg.iterrows():
        improved_str = f"{int(row['rules_improved'])}/{int(row['total_rules'])}"
        color = "\033[92m" if row["avg_improvement"] > 0 else "\033[91m"
        reset = "\033[0m"
        print(f"{filt:<35} {color}{row['avg_improvement']*100:>+9.4f}%{reset} "
              f"{row['avg_t_stat']:>6.2f} {row['avg_win_rate']:>6.1%} "
              f"{improved_str:>10} {row['avg_trades']:>6.0f}")

    # ── Heatmap: filter × rule improvement ──
    pivot = df[df["filter"] != "BASELINE (no filter)"].pivot_table(
        index="filter", columns="rule", values="improvement", aggfunc="first"
    )
    # Sort by avg improvement
    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg", ascending=False)
    pivot = pivot.drop(columns="avg")

    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.35)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    data = pivot.values * 100  # convert to percentage
    vmax = max(0.05, np.nanmax(np.abs(data)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in pivot.columns], color="white", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color="white", fontsize=8)

    for yi in range(len(pivot.index)):
        for xi in range(len(pivot.columns)):
            val = data[yi, xi]
            if not np.isnan(val):
                color_t = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(xi, yi, f"{val:+.3f}%", ha="center", va="center",
                        fontsize=7, color=color_t)

    ax.set_title("Phase 6: Filter Improvement over Baseline (% PnL change per trade)",
                 fontsize=12, fontweight="bold", color="white", pad=15)
    plt.colorbar(im, ax=ax, label="Improvement (%)")
    fig.savefig(os.path.join(OUTPUT_DIR, "phase6_filter_heatmap.png"),
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)

    print(f"\n  Results saved → {OUTPUT_DIR}/")
    print(f"  • phase6_filter_results.csv")
    print(f"  • phase6_filter_heatmap.png")
    print("=" * 80)


if __name__ == "__main__":
    run_filter_test()
