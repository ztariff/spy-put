#!/usr/bin/env python3
"""
Phase 2: Intraday Condition Scanning via P&L Snapshots
=======================================================
Scans all observable conditions on intraday bars, builds forward P&L
matrices, and ranks conditions by the quality of their P&L curves.

Run download_data.py and run_phase1.py first.

Usage:
    python run_phase2.py
"""
import os
import numpy as np
import pandas as pd

from src.config import ACTIVE_TICKERS, OUTPUT_DIR, MIN_OCCURRENCES
from src.downloader import load_data
from src.htf_context import compute_all_htf_factors
from src.snapshot_engine import build_pnl_matrix
from src.conditions import INTRADAY_CONDITIONS, HTF_BRIDGE_CONDITIONS
from src.analysis import (
    build_ranking_table, plot_pnl_curve, plot_pnl_curves_comparison,
    plot_peak_bar_distribution, plot_mfe_mae, plot_win_rate_evolution,
)


def run_phase2(tickers: list[str] = None,
               timeframes: list[str] = None):
    """
    For each ticker × timeframe:
    1. Load intraday and daily bars
    2. Compute HTF context (to tag each occurrence)
    3. Evaluate every condition from the registry
    4. Build P&L snapshot matrix for each condition
    5. Rank and visualize
    """
    if tickers is None:
        tickers = ACTIVE_TICKERS
    if timeframes is None:
        timeframes = ["5m", "15m"]  # Start with these, expand later

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for ticker in tickers:
        for tf in timeframes:
            print(f"\n{'='*70}")
            print(f"PHASE 2: {ticker} — {tf}")
            print(f"{'='*70}")

            # Load data
            try:
                intraday = load_data(ticker, tf)
                daily = load_data(ticker, "1D")
                weekly = load_data(ticker, "1W")
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

            print(f"  Intraday bars: {len(intraday):,}")

            # Strip timezone from daily/weekly for HTF computation
            if daily.index.tz is not None:
                daily = daily.copy()
                daily.index = daily.index.tz_localize(None)
            if weekly.index.tz is not None:
                weekly = weekly.copy()
                weekly.index = weekly.index.tz_localize(None)

            # Compute HTF context for tagging
            htf = compute_all_htf_factors(daily, weekly)

            # Filter to regular hours only
            if hasattr(intraday.index, 'time'):
                from datetime import time as dt_time
                time_mask = [(dt_time(9, 30) <= t <= dt_time(16, 0))
                             for t in intraday.index.time]
                intraday = intraday[time_mask]
                print(f"  Regular hours bars: {len(intraday):,}")

            # Precompute session dates
            session_dates = pd.Series(intraday.index.date, index=intraday.index)

            # ── Evaluate intraday-only conditions ────────────────────────
            print(f"\n  Scanning {len(INTRADAY_CONDITIONS)} intraday conditions...")

            for cond_name, cond_fn in INTRADAY_CONDITIONS.items():
                try:
                    mask = cond_fn(intraday)
                    if isinstance(mask, pd.Series):
                        mask = mask.fillna(False).astype(bool)
                    n = mask.sum()

                    if n < MIN_OCCURRENCES:
                        continue

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

                    all_results.append(result)

                    # Quick summary
                    stats = result.summary_stats()
                    peak = stats.get("peak_bar", 0)
                    peak_pnl = stats.get("mean_pnl_at_peak", 0) * 100
                    wr = stats.get("win_rate_at_peak", 0) * 100
                    t_stat = stats.get("t_stat", 0)
                    marker = ""
                    if abs(t_stat) > 2.0:
                        marker = " ★" if peak_pnl > 0 else " ▼"
                    print(f"    {cond_name:40s}  n={n:>6,}  "
                          f"peak={peak_pnl:+.4f}% @Bar+{peak+1:>2}  "
                          f"WR={wr:.1f}%  t={t_stat:.2f}{marker}")

                except Exception as e:
                    print(f"    {cond_name:40s}  ERROR: {e}")

            # ── Evaluate HTF bridge conditions ───────────────────────────
            print(f"\n  Scanning {len(HTF_BRIDGE_CONDITIONS)} HTF bridge conditions...")

            for cond_name, cond_fn in HTF_BRIDGE_CONDITIONS.items():
                try:
                    mask = cond_fn(intraday, daily)
                    if isinstance(mask, pd.Series):
                        mask = mask.reindex(intraday.index).fillna(False).astype(bool)
                    n = mask.sum()

                    if n < MIN_OCCURRENCES // 2:  # lower threshold for rarer events
                        continue

                    result = build_pnl_matrix(
                        bars=intraday,
                        condition_mask=mask,
                        session_dates=session_dates,
                    )

                    result.condition_name = cond_name
                    result.timeframe = tf
                    result.ticker = ticker

                    all_results.append(result)

                    stats = result.summary_stats()
                    peak = stats.get("peak_bar", 0)
                    peak_pnl = stats.get("mean_pnl_at_peak", 0) * 100
                    t_stat = stats.get("t_stat", 0)
                    marker = " ★" if abs(t_stat) > 2.0 and peak_pnl > 0 else ""
                    print(f"    {cond_name:40s}  n={n:>6,}  "
                          f"peak={peak_pnl:+.4f}% @Bar+{peak+1:>2}  "
                          f"t={t_stat:.2f}{marker}")

                except Exception as e:
                    print(f"    {cond_name:40s}  ERROR: {e}")

    # ── Global Ranking ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHASE 2: GLOBAL RANKING")
    print(f"{'='*70}")

    ranking = build_ranking_table(all_results, min_occurrences=MIN_OCCURRENCES // 2)
    if not ranking.empty:
        pd.set_option("display.max_rows", 100)
        pd.set_option("display.width", 180)
        pd.set_option("display.float_format", lambda x: f"{x:.6f}")
        print(ranking.to_string())
        ranking.to_csv(os.path.join(OUTPUT_DIR, "phase2_ranking.csv"), index=False)

    # ── Top 10 Detailed Plots ────────────────────────────────────────────
    print(f"\n  Generating detailed plots for top conditions...")

    # Sort results by absolute peak P&L
    scored = [(r, abs(r.summary_stats().get("mean_pnl_at_peak", 0)))
              for r in all_results if r.n_occurrences >= MIN_OCCURRENCES // 2]
    scored.sort(key=lambda x: x[1], reverse=True)

    for rank, (r, _) in enumerate(scored[:10]):
        print(f"    #{rank+1}: {r.condition_name} ({r.ticker}/{r.timeframe})")
        try:
            # Skip results with all-NaN P&L matrices
            if np.all(np.isnan(r.pnl_matrix)):
                print(f"      (skipped — all-NaN P&L matrix)")
                continue
            plot_pnl_curve(r)
            plot_peak_bar_distribution(r)
            plot_mfe_mae(r)
            plot_win_rate_evolution(r)
        except Exception as e:
            print(f"      (plot error: {e})")

    # Comparison plot of top 5
    if len(scored) >= 5:
        top5 = [s[0] for s in scored[:5]]
        plot_pnl_curves_comparison(
            top5,
            title="Top 5 Conditions — Mean P&L Curves",
            filename="phase2_top5_comparison.png",
        )

    print(f"\n  All outputs saved → {OUTPUT_DIR}/")
    print("="*70)


if __name__ == "__main__":
    run_phase2()
