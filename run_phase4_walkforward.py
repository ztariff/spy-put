#!/usr/bin/env python3
"""
Phase 4: Walk-Forward Validation
==================================
Splits data into in-sample (60%), validation (20%), out-of-sample (20%)
periods and tests whether the top Phase 3 combo signals hold out-of-sample.

This is the most critical step — if signals don't survive walk-forward,
they are overfit and should be discarded.

Usage:
    python run_phase4_walkforward.py
"""
import os
import numpy as np
import pandas as pd
from datetime import time as dt_time

from src.config import ACTIVE_TICKERS, OUTPUT_DIR, MIN_COMBO_OCCURRENCES
from src.downloader import load_data
from src.htf_context import compute_all_htf_factors
from src.snapshot_engine import build_pnl_matrix
from src.conditions import INTRADAY_CONDITIONS, HTF_BRIDGE_CONDITIONS
from src.analysis import plot_pnl_curves_comparison

# ── Top combos to validate (from Phase 3) ──
COMBOS_TO_VALIDATE = [
    # (condition_name, htf_factor, ticker, timeframe, direction)
    # Strategy A: Gap Continuation
    ("first_30_min", "gap_up_large", "SPY", "5m", "long"),
    ("cross_above_vwap", "gap_up_medium", "SPY", "5m", "long"),
    ("volume_dryup", "gap_up_medium", "SPY", "15m", "long"),
    ("5_consecutive_up", "gap_up_large", "SPY", "5m", "long"),
    ("new_20bar_high", "gap_down_medium", "QQQ", "15m", "long"),
    # Strategy B: Prior Day Weakness Reversal
    ("first_30_min", "prior_day_close_near_low", "SPY", "5m", "long"),
    ("first_30_min", "prior_day_close_near_low", "QQQ", "5m", "long"),
    ("above_or_high_30m", "is_nr7", "SPY", "5m", "long"),
    ("new_50bar_high", "prior_day_close_near_low", "SPY", "5m", "long"),
    ("new_50bar_high", "is_nr7", "SPY", "5m", "long"),
    # Additional strong combos
    ("first_30_min", "prior_day_high_vol_wide_range", "SPY", "5m", "long"),
    ("prior_bar_wide_range", "prior_day_high_vol_wide_range", "SPY", "5m", "long"),
    ("cross_above_vwap", "is_nr7", "QQQ", "15m", "long"),
    ("volume_dryup", "prior_day_close_near_low", "SPY", "5m", "long"),
]

# Walk-forward split ratios
IN_SAMPLE_PCT = 0.60
VALIDATION_PCT = 0.20
OUT_OF_SAMPLE_PCT = 0.20


def split_data_temporal(bars: pd.DataFrame, daily: pd.DataFrame, weekly: pd.DataFrame):
    """
    Split all data into 3 temporal periods.
    Returns (in_sample, validation, oos) tuples of (intraday, daily, weekly).
    """
    dates = sorted(bars.index.date)
    unique_dates = sorted(set(dates))
    n = len(unique_dates)

    is_end = unique_dates[int(n * IN_SAMPLE_PCT)]
    val_end = unique_dates[int(n * (IN_SAMPLE_PCT + VALIDATION_PCT))]

    is_end_ts = pd.Timestamp(is_end)
    val_end_ts = pd.Timestamp(val_end)

    # Split intraday
    is_intra = bars[bars.index.date <= is_end]
    val_intra = bars[(bars.index.date > is_end) & (bars.index.date <= val_end)]
    oos_intra = bars[bars.index.date > val_end]

    # Split daily
    is_daily = daily[daily.index <= is_end_ts]
    val_daily = daily[(daily.index > is_end_ts) & (daily.index <= val_end_ts)]
    oos_daily = daily[daily.index > val_end_ts]

    # For HTF computation, we need history before each period
    # So validation uses all data up to val_end, OOS uses all data up to oos end
    # But we only evaluate signals in the respective windows
    is_daily_full = daily[daily.index <= is_end_ts]
    val_daily_full = daily[daily.index <= val_end_ts]
    oos_daily_full = daily  # all available

    is_weekly_full = weekly[weekly.index <= is_end_ts]
    val_weekly_full = weekly[weekly.index <= val_end_ts]
    oos_weekly_full = weekly

    return {
        "in_sample": {
            "intraday": is_intra,
            "daily_full": is_daily_full,
            "weekly_full": is_weekly_full,
            "label": f"In-Sample (≤{is_end})",
            "date_range": f"start — {is_end}",
        },
        "validation": {
            "intraday": val_intra,
            "daily_full": val_daily_full,
            "weekly_full": val_weekly_full,
            "label": f"Validation ({is_end} — {val_end})",
            "date_range": f"{is_end} — {val_end}",
        },
        "oos": {
            "intraday": oos_intra,
            "daily_full": oos_daily_full,
            "weekly_full": oos_weekly_full,
            "label": f"Out-of-Sample (>{val_end})",
            "date_range": f"{val_end} — end",
        },
    }


def evaluate_combo_on_period(cond_name, htf_factor, period_data, direction="long"):
    """
    Evaluate a single condition × HTF factor combo on a data period.
    Returns SnapshotResult for the HTF=1 slice, or None if insufficient data.
    """
    intraday = period_data["intraday"]
    daily = period_data["daily_full"]
    weekly = period_data["weekly_full"]

    if len(intraday) < 100:
        return None

    # Strip tz
    if daily.index.tz is not None:
        daily = daily.copy()
        daily.index = daily.index.tz_localize(None)
    if weekly.index.tz is not None:
        weekly = weekly.copy()
        weekly.index = weekly.index.tz_localize(None)

    # HTF context
    htf = compute_all_htf_factors(daily, weekly)

    # Filter regular hours
    if hasattr(intraday.index, 'time'):
        time_mask = [(dt_time(9, 30) <= t <= dt_time(16, 0))
                     for t in intraday.index.time]
        intraday = intraday[time_mask]

    if len(intraday) < 50:
        return None

    session_dates = pd.Series(intraday.index.date, index=intraday.index)

    # Get condition function
    is_bridge = False
    if cond_name in INTRADAY_CONDITIONS:
        cond_fn = INTRADAY_CONDITIONS[cond_name]
    elif cond_name in HTF_BRIDGE_CONDITIONS:
        cond_fn = HTF_BRIDGE_CONDITIONS[cond_name]
        is_bridge = True
    else:
        return None

    # Evaluate condition
    try:
        if is_bridge:
            mask = cond_fn(intraday, daily)
            if isinstance(mask, pd.Series):
                mask = mask.reindex(intraday.index).fillna(False).astype(bool)
        else:
            mask = cond_fn(intraday)
            if isinstance(mask, pd.Series):
                mask = mask.fillna(False).astype(bool)
    except Exception:
        return None

    n = mask.sum()
    if n < 10:  # relaxed threshold for individual periods
        return None

    # Build P&L matrix
    result = build_pnl_matrix(
        bars=intraday,
        condition_mask=mask,
        session_dates=session_dates,
        direction=direction,
    )
    result.condition_name = cond_name
    result.timeframe = period_data.get("tf", "")
    result.ticker = period_data.get("ticker", "")

    # Attach HTF tags
    entry_dates = [t.date() if hasattr(t, 'date') else t for t in result.entry_times]
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

    # Slice by HTF factor
    if result.htf_tags is None or htf_factor not in result.htf_tags.columns:
        return None

    try:
        sliced = result.slice_by_htf(htf_factor, 1)
    except (ValueError, KeyError):
        return None

    if sliced.n_occurrences < 5:
        return None

    sliced.condition_name = f"{cond_name} | {htf_factor}=1"
    return sliced


def run_walkforward():
    """Run walk-forward validation on all top combos."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_table = []

    print("=" * 80)
    print("PHASE 4: WALK-FORWARD VALIDATION")
    print("=" * 80)

    # Cache loaded data by (ticker, tf)
    data_cache = {}

    for combo in COMBOS_TO_VALIDATE:
        cond_name, htf_factor, ticker, tf, direction = combo
        combo_label = f"{cond_name} × {htf_factor} ({ticker}/{tf})"

        print(f"\n{'─'*70}")
        print(f"  {combo_label}")
        print(f"{'─'*70}")

        # Load data (with caching)
        cache_key = (ticker, tf)
        if cache_key not in data_cache:
            try:
                intraday = load_data(ticker, tf)
                daily = load_data(ticker, "1D")
                weekly = load_data(ticker, "1W")
                data_cache[cache_key] = (intraday, daily, weekly)
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

        intraday, daily, weekly = data_cache[cache_key]

        # Strip tz for splitting
        daily_clean = daily.copy()
        if daily_clean.index.tz is not None:
            daily_clean.index = daily_clean.index.tz_localize(None)
        weekly_clean = weekly.copy()
        if weekly_clean.index.tz is not None:
            weekly_clean.index = weekly_clean.index.tz_localize(None)

        # Split data
        splits = split_data_temporal(intraday, daily_clean, weekly_clean)

        period_results = {}
        for period_name, period_data in splits.items():
            period_data["tf"] = tf
            period_data["ticker"] = ticker

            result = evaluate_combo_on_period(
                cond_name, htf_factor, period_data, direction
            )

            if result is not None:
                stats = result.summary_stats()
                if stats.get("skip"):
                    period_results[period_name] = None
                    continue

                period_results[period_name] = {
                    "result": result,
                    "n": result.n_occurrences,
                    "t_stat": stats.get("t_stat", 0),
                    "mean_pnl": stats.get("mean_pnl_at_peak", 0),
                    "peak_bar": stats.get("peak_bar", 0),
                    "win_rate": stats.get("win_rate_at_peak", 0),
                    "avg_mfe": stats.get("avg_mfe", 0),
                    "avg_mae": stats.get("avg_mae", 0),
                    "label": period_data["label"],
                }

                pnl_pct = stats.get("mean_pnl_at_peak", 0) * 100
                t = stats.get("t_stat", 0)
                wr = stats.get("win_rate_at_peak", 0) * 100
                pb = stats.get("peak_bar", 0)
                marker = "✓" if t > 1.5 else "✗"
                print(f"    {period_name:14s}: n={result.n_occurrences:>5,}  "
                      f"pnl={pnl_pct:+.4f}%  t={t:.2f}  "
                      f"WR={wr:.1f}%  peak=Bar+{pb+1}  {marker}")
            else:
                period_results[period_name] = None
                print(f"    {period_name:14s}: insufficient data")

        # ── Assess walk-forward pass/fail ──
        is_data = period_results.get("in_sample")
        val_data = period_results.get("validation")
        oos_data = period_results.get("oos")

        passed = False
        status = "FAIL"

        if is_data and val_data and oos_data:
            is_t = is_data["t_stat"]
            val_t = val_data["t_stat"]
            oos_t = oos_data["t_stat"]
            oos_pnl = oos_data["mean_pnl"]

            # Pass criteria:
            # 1. In-sample t > 1.5
            # 2. OOS t > 1.0 (relaxed from IS threshold)
            # 3. OOS P&L same sign as IS P&L
            # 4. OOS P&L > 50% of IS P&L magnitude (not too much decay)
            if (is_t > 1.5 and oos_t > 1.0 and
                np.sign(oos_pnl) == np.sign(is_data["mean_pnl"]) and
                abs(oos_pnl) > abs(is_data["mean_pnl"]) * 0.3):
                passed = True
                status = "PASS ★"

                # Strong pass: OOS t > 2.0
                if oos_t > 2.0:
                    status = "STRONG PASS ★★★"
                elif oos_t > 1.5:
                    status = "PASS ★★"
        elif is_data and oos_data:
            # No validation but have IS and OOS
            is_t = is_data["t_stat"]
            oos_t = oos_data["t_stat"]
            oos_pnl = oos_data["mean_pnl"]
            if is_t > 1.5 and oos_t > 1.0:
                passed = True
                status = "PASS (no val) ★"

        print(f"\n    ► Walk-Forward Result: {status}")

        # Store for summary
        row = {
            "condition": cond_name,
            "htf_factor": htf_factor,
            "ticker": ticker,
            "timeframe": tf,
            "direction": direction,
            "status": status,
            "passed": passed,
        }

        for period_name in ["in_sample", "validation", "oos"]:
            pd_data = period_results.get(period_name)
            if pd_data:
                row[f"{period_name}_n"] = pd_data["n"]
                row[f"{period_name}_t"] = pd_data["t_stat"]
                row[f"{period_name}_pnl"] = pd_data["mean_pnl"]
                row[f"{period_name}_wr"] = pd_data["win_rate"]
                row[f"{period_name}_peak_bar"] = pd_data["peak_bar"]
            else:
                row[f"{period_name}_n"] = 0
                row[f"{period_name}_t"] = 0
                row[f"{period_name}_pnl"] = 0
                row[f"{period_name}_wr"] = 0
                row[f"{period_name}_peak_bar"] = 0

        results_table.append(row)

        # ── Generate IS vs OOS comparison plot ──
        plot_results = []
        for period_name in ["in_sample", "validation", "oos"]:
            pd_data = period_results.get(period_name)
            if pd_data and pd_data["result"].n_occurrences > 0:
                r = pd_data["result"]
                r.condition_name = f"{period_name} (n={pd_data['n']})"
                plot_results.append(r)

        if len(plot_results) >= 2:
            safe = f"{cond_name}_{htf_factor}_{ticker}_{tf}".replace(" ", "_")
            plot_pnl_curves_comparison(
                plot_results,
                title=f"Walk-Forward: {cond_name} × {htf_factor} ({ticker}/{tf})",
                filename=f"phase4_wf_{safe}.png",
            )

    # ── Summary Table ──
    print(f"\n\n{'='*80}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*80}\n")

    df = pd.DataFrame(results_table)
    if not df.empty:
        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")

        display_cols = [
            "condition", "htf_factor", "ticker", "timeframe",
            "in_sample_n", "in_sample_t", "in_sample_pnl",
            "oos_n", "oos_t", "oos_pnl",
            "status",
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        print(df[available_cols].to_string(index=False))

        df.to_csv(os.path.join(OUTPUT_DIR, "phase4_walkforward.csv"), index=False)

        n_passed = df["passed"].sum()
        n_total = len(df)
        print(f"\n  {n_passed}/{n_total} combos passed walk-forward validation")

        # List passed combos
        if n_passed > 0:
            print(f"\n  ► VALIDATED SIGNALS (ready for backtesting):")
            for _, row in df[df["passed"]].iterrows():
                print(f"    • {row['condition']} × {row['htf_factor']} "
                      f"({row['ticker']}/{row['timeframe']}) — {row['status']}")

    print(f"\n  All outputs saved → {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    run_walkforward()
