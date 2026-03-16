#!/usr/bin/env python3
"""
Phase 1: Higher-Timeframe Context Discovery
=============================================
Computes all HTF factors from daily data, then uses the P&L snapshot engine
to measure how each HTF regime affects intraday P&L paths.

Run download_data.py first.

Usage:
    python run_phase1.py
"""
import os
import numpy as np
import pandas as pd

from src.config import ACTIVE_TICKERS, OUTPUT_DIR, MIN_OCCURRENCES
from src.downloader import load_data
from src.htf_context import compute_all_htf_factors
from src.snapshot_engine import build_daily_pnl_matrix
from src.analysis import (
    build_ranking_table, plot_pnl_curves_comparison, plot_pnl_curve,
)


def run_phase1(tickers: list[str] = None,
               intraday_tf: str = "5m"):
    """
    For each ticker:
    1. Load daily bars, compute HTF context factors
    2. Load intraday bars
    3. For each HTF factor, split days into bins
    4. Build P&L snapshot matrix for each bin (enter at open, track intraday)
    5. Compare P&L curves across bins
    """
    if tickers is None:
        tickers = ACTIVE_TICKERS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for ticker in tickers:
        print(f"\n{'='*70}")
        print(f"PHASE 1: {ticker}")
        print(f"{'='*70}")

        # Load data
        try:
            daily = load_data(ticker, "1D")
            weekly = load_data(ticker, "1W")
            intraday = load_data(ticker, intraday_tf)
        except FileNotFoundError as e:
            print(f"  Skipping {ticker}: {e}")
            continue

        print(f"  Daily bars:    {len(daily):,} ({daily.index[0].date()} → {daily.index[-1].date()})")
        print(f"  Intraday bars: {len(intraday):,} ({intraday.index[0]} → {intraday.index[-1]})")

        # Compute HTF context
        print(f"  Computing HTF factors...")
        htf = compute_all_htf_factors(daily, weekly)

        # ── Test each binary factor ──────────────────────────────────────
        binary_factors = [col for col in htf.columns
                          if htf[col].dropna().isin([0, 1]).all()]

        print(f"  Testing {len(binary_factors)} binary factors...")

        for factor in binary_factors:
            mask_true = htf[factor] == 1
            mask_false = htf[factor] == 0

            n_true = mask_true.sum()
            n_false = mask_false.sum()

            if n_true < 50 or n_false < 50:
                continue

            # Build P&L matrix for each bin
            result_true = build_daily_pnl_matrix(daily, intraday, mask_true)
            result_false = build_daily_pnl_matrix(daily, intraday, mask_false)

            result_true.condition_name = f"{factor}=1"
            result_true.timeframe = intraday_tf
            result_true.ticker = ticker

            result_false.condition_name = f"{factor}=0"
            result_false.timeframe = intraday_tf
            result_false.ticker = ticker

            all_results.append(result_true)
            all_results.append(result_false)

            # Quick summary
            if result_true.n_occurrences > 0 and result_false.n_occurrences > 0:
                peak_t = result_true.peak_bar_mean
                peak_f = result_false.peak_bar_mean
                pnl_t = result_true.mean_curve[peak_t] * 100 if len(result_true.mean_curve) > peak_t else 0
                pnl_f = result_false.mean_curve[peak_f] * 100 if len(result_false.mean_curve) > peak_f else 0
                diff = pnl_t - pnl_f
                marker = " ◀ EDGE" if abs(diff) > 0.05 else ""
                print(f"    {factor:40s}  "
                      f"=1: {pnl_t:+.4f}% (n={result_true.n_occurrences:,})  "
                      f"=0: {pnl_f:+.4f}% (n={result_false.n_occurrences:,})  "
                      f"Δ={diff:+.4f}%{marker}")

        # ── Plot top factors ─────────────────────────────────────────────
        print(f"\n  Generating comparison plots...")

        # Find factors with biggest divergence between =1 and =0
        factor_diffs = []
        for i in range(0, len(all_results) - 1, 2):
            r1 = all_results[i]    # =1
            r0 = all_results[i+1]  # =0
            if r1.ticker != ticker:
                continue
            if r1.n_occurrences > 0 and r0.n_occurrences > 0:
                p1 = r1.mean_curve[r1.peak_bar_mean] if len(r1.mean_curve) > 0 else 0
                p0 = r0.mean_curve[r0.peak_bar_mean] if len(r0.mean_curve) > 0 else 0
                factor_diffs.append((abs(p1 - p0), r1, r0))

        factor_diffs.sort(key=lambda x: x[0], reverse=True)

        # Plot top 10 factors
        for rank, (diff, r1, r0) in enumerate(factor_diffs[:10]):
            fname = r1.condition_name.replace("=1", "").strip()
            plot_pnl_curves_comparison(
                [r1, r0],
                title=f"{ticker} — {fname} (=1 vs =0)",
                filename=f"phase1_{ticker}_{fname}.png",
            )

    # ── Global ranking ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHASE 1: RANKING TABLE")
    print(f"{'='*70}")

    ranking = build_ranking_table(all_results, min_occurrences=50)
    if not ranking.empty:
        pd.set_option("display.max_rows", 100)
        pd.set_option("display.width", 160)
        pd.set_option("display.float_format", lambda x: f"{x:.6f}")
        print(ranking.to_string())
        ranking.to_csv(os.path.join(OUTPUT_DIR, "phase1_ranking.csv"), index=False)
        print(f"\n  Saved → {os.path.join(OUTPUT_DIR, 'phase1_ranking.csv')}")

    print(f"\n  Charts saved → {OUTPUT_DIR}/")
    print("="*70)


if __name__ == "__main__":
    run_phase1()
