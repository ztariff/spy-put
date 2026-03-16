#!/usr/bin/env python3
"""
Phase 3: Multi-Timeframe Combination Discovery
================================================
Takes the best intraday conditions from Phase 2 and slices them by
the best HTF context factors from Phase 1 to find the highest-edge
condition × context pairings.

Run download_data.py, run_phase1.py, and run_phase2.py first.

Usage:
    python run_phase3.py
"""
import os
import numpy as np
import pandas as pd

from src.config import ACTIVE_TICKERS, OUTPUT_DIR, MIN_COMBO_OCCURRENCES
from src.downloader import load_data
from src.htf_context import compute_all_htf_factors
from src.snapshot_engine import build_pnl_matrix
from src.conditions import INTRADAY_CONDITIONS, HTF_BRIDGE_CONDITIONS
from src.analysis import (
    plot_pnl_curves_comparison, plot_pnl_curve,
    plot_htf_comparison, plot_mfe_mae, plot_win_rate_evolution,
)


# ── Top conditions from Phase 2 (by t-stat, excluding end-of-day artifacts) ──
# Selected: t-stat > 2.0, peak_bar < 70 (5m) or < 23 (15m), AND meaningful P&L
TOP_CONDITIONS_5M = [
    "cross_above_vwap",        # QQQ t=3.36
    "5_consecutive_up",        # SPY t=4.71
    "prior_bar_wide_range",    # SPY t=3.00
    "new_20bar_high",          # QQQ t=3.21, SPY t=3.29
    "new_50bar_high",          # SPY t=4.12
    "volume_surge_bullish",    # QQQ t=2.18
    "prior_bar_conviction_bullish",  # SPY t=1.17 on 5m but strong on 15m
    "prior_bar_exhaustion_bearish",  # SPY t=2.85
    "above_or_high_30m",       # SPY t=3.94
    "first_30_min",            # SPY t=3.33
    "volume_dryup",            # QQQ t=3.93
    "breakout_above_prior_day_high",  # SPY t=1.62 (bridge)
    "breakout_below_prior_week_low",  # SPY t=2.07 (bridge)
]

TOP_CONDITIONS_15M = [
    "prior_bar_conviction_bullish",  # SPY t=2.93
    "range_expansion_after_contraction",  # SPY t=2.31
    "new_20bar_high",          # QQQ t=3.24, SPY t=3.14
    "new_50bar_high",          # SPY t=2.55
    "volume_surge_bullish",    # SPY t=1.83, QQQ t=2.08
    "prior_bar_exhaustion_bearish",  # SPY t=2.26
    "cross_above_vwap",        # QQQ t=2.35
    "above_or_high_30m",       # QQQ t=1.91
    "prior_bar_close_near_high",  # QQQ t=2.22
    "volume_dryup",            # SPY t=1.95
    "atr_expansion",           # SPY t=2.32
    "breakout_above_prior_day_high",  # SPY t=2.88 (bridge)
    "breakout_below_prior_week_low",  # SPY t=1.43 (bridge)
    "breakout_below_prior_day_low",   # bridge
]

# ── Top HTF factors from Phase 1 ──
# Binary factors that showed the strongest edge splits
HTF_FACTORS_TO_TEST = [
    "gap_up_large",
    "gap_up_medium",
    "gap_down_large",
    "gap_down_medium",
    "gap_above_prior_day_range",
    "gap_below_prior_day_range",
    "above_prior_day_high",
    "below_prior_day_low",
    "above_prior_week_high",
    "below_prior_week_low",
    "above_prior_month_high",
    "below_prior_month_low",
    "aligned_daily_weekly_up",
    "aligned_daily_weekly_down",
    "is_inside_day",
    "is_outside_day",
    "is_nr7",
    "prior_day_close_near_high",
    "prior_day_close_near_low",
    "prior_day_high_vol_wide_range",
    "gap_aligned_with_trend",
    "near_52w_high",
    "adx_trending",
]


def run_phase3(tickers=None, timeframes=None):
    """
    For each ticker × timeframe × top condition:
    1. Rebuild P&L matrix with HTF tags
    2. Slice by each HTF factor (value=1 vs value=0)
    3. Compare sliced stats
    4. Rank all combinations
    """
    if tickers is None:
        tickers = ACTIVE_TICKERS
    if timeframes is None:
        timeframes = ["5m", "15m"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_combos = []  # list of dicts for ranking

    for ticker in tickers:
        for tf in timeframes:
            print(f"\n{'='*70}")
            print(f"PHASE 3: {ticker} — {tf}")
            print(f"{'='*70}")

            # Load data
            try:
                intraday = load_data(ticker, tf)
                daily = load_data(ticker, "1D")
                weekly = load_data(ticker, "1W")
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

            # Strip timezone from daily/weekly
            if daily.index.tz is not None:
                daily = daily.copy()
                daily.index = daily.index.tz_localize(None)
            if weekly.index.tz is not None:
                weekly = weekly.copy()
                weekly.index = weekly.index.tz_localize(None)

            # Compute HTF context
            htf = compute_all_htf_factors(daily, weekly)

            # Filter regular hours
            if hasattr(intraday.index, 'time'):
                from datetime import time as dt_time
                time_mask = [(dt_time(9, 30) <= t <= dt_time(16, 0))
                             for t in intraday.index.time]
                intraday = intraday[time_mask]

            print(f"  Bars: {len(intraday):,}  |  HTF context days: {len(htf):,}")

            session_dates = pd.Series(intraday.index.date, index=intraday.index)

            # Choose condition set for this timeframe
            cond_names = TOP_CONDITIONS_5M if tf == "5m" else TOP_CONDITIONS_15M

            for cond_name in cond_names:
                # Look up condition function
                cond_fn = None
                is_bridge = False
                if cond_name in INTRADAY_CONDITIONS:
                    cond_fn = INTRADAY_CONDITIONS[cond_name]
                elif cond_name in HTF_BRIDGE_CONDITIONS:
                    cond_fn = HTF_BRIDGE_CONDITIONS[cond_name]
                    is_bridge = True
                else:
                    continue

                try:
                    # Evaluate condition
                    if is_bridge:
                        mask = cond_fn(intraday, daily)
                        if isinstance(mask, pd.Series):
                            mask = mask.reindex(intraday.index).fillna(False).astype(bool)
                    else:
                        mask = cond_fn(intraday)
                        if isinstance(mask, pd.Series):
                            mask = mask.fillna(False).astype(bool)

                    n = mask.sum()
                    if n < MIN_COMBO_OCCURRENCES:
                        continue

                    # Build P&L matrix
                    result = build_pnl_matrix(
                        bars=intraday,
                        condition_mask=mask,
                        session_dates=session_dates,
                    )
                    result.condition_name = cond_name
                    result.timeframe = tf
                    result.ticker = ticker

                    # Attach HTF tags
                    entry_dates = [t.date() if hasattr(t, 'date') else t
                                   for t in result.entry_times]
                    htf_rows = []
                    for d in entry_dates:
                        d_ts = pd.Timestamp(d)
                        if d_ts in htf.index:
                            htf_rows.append(htf.loc[d_ts])
                        else:
                            prior = htf.index[htf.index <= d_ts]
                            if len(prior) > 0:
                                htf_rows.append(htf.loc[prior[-1]])
                            else:
                                htf_rows.append(pd.Series(dtype=float))

                    if htf_rows:
                        result.htf_tags = pd.DataFrame(htf_rows).reset_index(drop=True)

                    # Base stats (unsliced)
                    base_stats = result.summary_stats()
                    if base_stats.get("skip"):
                        continue
                    base_t = base_stats.get("t_stat", 0)
                    base_pnl = base_stats.get("mean_pnl_at_peak", 0)

                    print(f"\n  {cond_name} ({ticker}/{tf})  n={n:,}  "
                          f"base_t={base_t:.2f}  base_pnl={base_pnl*100:+.4f}%")

                    # ── Slice by each HTF factor ──
                    for htf_factor in HTF_FACTORS_TO_TEST:
                        if result.htf_tags is None or htf_factor not in result.htf_tags.columns:
                            continue

                        try:
                            # Slice for factor = 1 (condition present)
                            sliced_1 = result.slice_by_htf(htf_factor, 1)
                            # Slice for factor = 0 (condition absent)
                            sliced_0 = result.slice_by_htf(htf_factor, 0)

                            if sliced_1.n_occurrences < MIN_COMBO_OCCURRENCES // 2:
                                continue

                            stats_1 = sliced_1.summary_stats()
                            stats_0 = sliced_0.summary_stats()

                            if stats_1.get("skip") or stats_0.get("skip"):
                                continue

                            t1 = stats_1.get("t_stat", 0)
                            t0 = stats_0.get("t_stat", 0)
                            pnl1 = stats_1.get("mean_pnl_at_peak", 0)
                            pnl0 = stats_0.get("mean_pnl_at_peak", 0)
                            wr1 = stats_1.get("win_rate_at_peak", 0)
                            wr0 = stats_0.get("win_rate_at_peak", 0)
                            n1 = sliced_1.n_occurrences
                            n0 = sliced_0.n_occurrences

                            # Compute edge delta (how much the HTF factor improves the condition)
                            pnl_delta = pnl1 - pnl0
                            t_delta = t1 - t0

                            # Store combo
                            combo = {
                                "condition": cond_name,
                                "htf_factor": htf_factor,
                                "ticker": ticker,
                                "timeframe": tf,
                                "n_base": n,
                                "n_htf1": n1,
                                "n_htf0": n0,
                                "base_t": base_t,
                                "base_pnl": base_pnl,
                                "htf1_t": t1,
                                "htf1_pnl": pnl1,
                                "htf1_wr": wr1,
                                "htf1_peak_bar": stats_1.get("peak_bar", 0),
                                "htf0_t": t0,
                                "htf0_pnl": pnl0,
                                "htf0_wr": wr0,
                                "pnl_delta": pnl_delta,
                                "t_delta": t_delta,
                                "htf1_mfe": stats_1.get("avg_mfe", 0),
                                "htf1_mae": stats_1.get("avg_mae", 0),
                            }
                            all_combos.append(combo)

                            # Log significant combos
                            if abs(t1) > 2.0 and abs(pnl_delta) > 0.0005:
                                marker = "★★★" if abs(t1) > 3.0 else "★★" if abs(t1) > 2.5 else "★"
                                print(f"    {htf_factor}=1:  n={n1:>5,}  "
                                      f"pnl={pnl1*100:+.4f}%  t={t1:.2f}  "
                                      f"wr={wr1*100:.1f}%  "
                                      f"Δpnl={pnl_delta*100:+.4f}%  {marker}")

                        except (ValueError, KeyError) as e:
                            continue

                except Exception as e:
                    print(f"    ERROR: {cond_name} — {e}")

    # ── Global Ranking of Combinations ────────────────────────────────────
    if not all_combos:
        print("\nNo valid combinations found.")
        return

    combo_df = pd.DataFrame(all_combos)
    combo_df = combo_df.sort_values("htf1_t", ascending=False, key=abs)

    # Save full ranking
    combo_df.to_csv(os.path.join(OUTPUT_DIR, "phase3_combinations.csv"), index=False)

    print(f"\n{'='*70}")
    print(f"PHASE 3: TOP COMBINATIONS")
    print(f"{'='*70}")

    # Filter: htf1 needs at least MIN_COMBO_OCCURRENCES/2 observations
    # and |htf1_t| > 1.5
    strong = combo_df[
        (combo_df["n_htf1"] >= MIN_COMBO_OCCURRENCES // 2) &
        (abs(combo_df["htf1_t"]) > 1.5)
    ].copy()

    # Rank by absolute t-stat of the HTF=1 slice
    strong = strong.sort_values("htf1_t", ascending=False, key=abs).head(40)

    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")

    display_cols = [
        "condition", "htf_factor", "ticker", "timeframe",
        "n_htf1", "htf1_pnl", "htf1_t", "htf1_wr", "htf1_peak_bar",
        "n_htf0", "htf0_pnl", "htf0_t",
        "pnl_delta",
    ]
    print(strong[display_cols].to_string(index=False))

    strong.to_csv(os.path.join(OUTPUT_DIR, "phase3_top_combos.csv"), index=False)

    # ── Generate comparison plots for top 15 combos ──
    print(f"\n  Generating combo split plots...")

    # We need to re-build the results for plotting
    # Group combos by (condition, ticker, timeframe) to avoid redundant rebuilds
    plot_combos = strong.head(15)
    plotted = set()

    for _, row in plot_combos.iterrows():
        key = (row["condition"], row["htf_factor"], row["ticker"], row["timeframe"])
        if key in plotted:
            continue
        plotted.add(key)

        cond_name = row["condition"]
        htf_factor = row["htf_factor"]
        tkr = row["ticker"]
        tf_val = row["timeframe"]

        try:
            # Reload and rebuild
            intra = load_data(tkr, tf_val)
            daily_d = load_data(tkr, "1D")
            weekly_d = load_data(tkr, "1W")

            if daily_d.index.tz is not None:
                daily_d = daily_d.copy()
                daily_d.index = daily_d.index.tz_localize(None)
            if weekly_d.index.tz is not None:
                weekly_d = weekly_d.copy()
                weekly_d.index = weekly_d.index.tz_localize(None)

            htf_ctx = compute_all_htf_factors(daily_d, weekly_d)

            from datetime import time as dt_time
            if hasattr(intra.index, 'time'):
                time_mask = [(dt_time(9, 30) <= t <= dt_time(16, 0))
                             for t in intra.index.time]
                intra = intra[time_mask]

            sess = pd.Series(intra.index.date, index=intra.index)

            # Get condition
            is_bridge = False
            if cond_name in INTRADAY_CONDITIONS:
                cond_fn = INTRADAY_CONDITIONS[cond_name]
            elif cond_name in HTF_BRIDGE_CONDITIONS:
                cond_fn = HTF_BRIDGE_CONDITIONS[cond_name]
                is_bridge = True
            else:
                continue

            if is_bridge:
                mask = cond_fn(intra, daily_d)
                if isinstance(mask, pd.Series):
                    mask = mask.reindex(intra.index).fillna(False).astype(bool)
            else:
                mask = cond_fn(intra)
                if isinstance(mask, pd.Series):
                    mask = mask.fillna(False).astype(bool)

            result = build_pnl_matrix(bars=intra, condition_mask=mask, session_dates=sess)
            result.condition_name = cond_name
            result.timeframe = tf_val
            result.ticker = tkr

            # Attach HTF tags
            entry_dates = [t.date() if hasattr(t, 'date') else t for t in result.entry_times]
            htf_rows = []
            for d in entry_dates:
                d_ts = pd.Timestamp(d)
                if d_ts in htf_ctx.index:
                    htf_rows.append(htf_ctx.loc[d_ts])
                else:
                    prior = htf_ctx.index[htf_ctx.index <= d_ts]
                    if len(prior) > 0:
                        htf_rows.append(htf_ctx.loc[prior[-1]])
                    else:
                        htf_rows.append(pd.Series(dtype=float))
            if htf_rows:
                result.htf_tags = pd.DataFrame(htf_rows).reset_index(drop=True)

            # Plot HTF comparison
            safe_cond = cond_name.replace(" ", "_")
            safe_htf = htf_factor.replace(" ", "_")
            fname = f"phase3_{safe_cond}_by_{safe_htf}_{tkr}_{tf_val}.png"
            plot_htf_comparison(result, htf_factor, [1, 0], filename=fname)

            # Also plot the P&L curve for the HTF=1 slice
            sliced = result.slice_by_htf(htf_factor, 1)
            if sliced.n_occurrences >= 30 and not np.all(np.isnan(sliced.pnl_matrix)):
                fname2 = f"phase3_detail_{safe_cond}_{safe_htf}_1_{tkr}_{tf_val}.png"
                plot_pnl_curve(sliced, filename=fname2)

            print(f"    ✓ {cond_name} × {htf_factor} ({tkr}/{tf_val})")

        except Exception as e:
            print(f"    ✗ {cond_name} × {htf_factor}: {e}")

    # ── Summary comparison plot: top 5 combo P&L curves ──
    print(f"\n  Generating top 5 combo comparison plot...")
    top5_results = []
    for _, row in plot_combos.head(5).iterrows():
        cond_name = row["condition"]
        htf_factor = row["htf_factor"]
        tkr = row["ticker"]
        tf_val = row["timeframe"]

        try:
            intra = load_data(tkr, tf_val)
            daily_d = load_data(tkr, "1D")
            weekly_d = load_data(tkr, "1W")

            if daily_d.index.tz is not None:
                daily_d = daily_d.copy()
                daily_d.index = daily_d.index.tz_localize(None)
            if weekly_d.index.tz is not None:
                weekly_d = weekly_d.copy()
                weekly_d.index = weekly_d.index.tz_localize(None)

            htf_ctx = compute_all_htf_factors(daily_d, weekly_d)

            from datetime import time as dt_time
            if hasattr(intra.index, 'time'):
                time_mask = [(dt_time(9, 30) <= t <= dt_time(16, 0))
                             for t in intra.index.time]
                intra = intra[time_mask]

            sess = pd.Series(intra.index.date, index=intra.index)

            is_bridge = False
            if cond_name in INTRADAY_CONDITIONS:
                cond_fn = INTRADAY_CONDITIONS[cond_name]
            elif cond_name in HTF_BRIDGE_CONDITIONS:
                cond_fn = HTF_BRIDGE_CONDITIONS[cond_name]
                is_bridge = True
            else:
                continue

            if is_bridge:
                mask = cond_fn(intra, daily_d)
                if isinstance(mask, pd.Series):
                    mask = mask.reindex(intra.index).fillna(False).astype(bool)
            else:
                mask = cond_fn(intra)
                if isinstance(mask, pd.Series):
                    mask = mask.fillna(False).astype(bool)

            result = build_pnl_matrix(bars=intra, condition_mask=mask, session_dates=sess)
            result.condition_name = cond_name
            result.timeframe = tf_val
            result.ticker = tkr

            entry_dates = [t.date() if hasattr(t, 'date') else t for t in result.entry_times]
            htf_rows = []
            for d in entry_dates:
                d_ts = pd.Timestamp(d)
                if d_ts in htf_ctx.index:
                    htf_rows.append(htf_ctx.loc[d_ts])
                else:
                    prior = htf_ctx.index[htf_ctx.index <= d_ts]
                    if len(prior) > 0:
                        htf_rows.append(htf_ctx.loc[prior[-1]])
                    else:
                        htf_rows.append(pd.Series(dtype=float))
            if htf_rows:
                result.htf_tags = pd.DataFrame(htf_rows).reset_index(drop=True)

            sliced = result.slice_by_htf(htf_factor, 1)
            sliced.condition_name = f"{cond_name} | {htf_factor}"
            top5_results.append(sliced)

        except Exception as e:
            print(f"    ✗ Top5 build error: {e}")

    if len(top5_results) >= 2:
        plot_pnl_curves_comparison(
            top5_results,
            title="Top 5 Condition × HTF Combinations — Mean P&L Curves",
            filename="phase3_top5_combos_comparison.png",
        )

    print(f"\n  All Phase 3 outputs saved → {OUTPUT_DIR}/")
    print(f"  Total combinations evaluated: {len(all_combos):,}")
    print(f"  Statistically significant (|t|>1.5): {len(strong):,}")
    print("=" * 70)


if __name__ == "__main__":
    run_phase3()
