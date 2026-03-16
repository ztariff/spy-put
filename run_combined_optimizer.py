#!/usr/bin/env python3
"""
Combined Optimizer: Delta-Scaled Sizing + Per-Rule Optimal Exit Times
=====================================================================
Re-simulates each optimal trade using:
  1. Delta-scaled premium budgets ($10K for Δ0.10, $50K for Δ0.50, $75K for Δ0.70)
  2. Per-rule optimal exit times (from exit_optimization.csv analysis)

Reads cached 5-min option bars (no API calls needed).

Usage:
    python run_combined_optimizer.py
"""
import os
import sys
import csv
import json
import base64
import re
import argparse
import numpy as np
import pandas as pd
from datetime import time, datetime
from collections import defaultdict

from src.config import OUTPUT_DIR, DATA_DIR
from src.options_client import OptionsClient


# ── Configuration ────────────────────────────────────────────────────────────

SLIPPAGE_PCT = 0.01  # 1% per side (same as backtest)

# Per-rule optimal exit times (from exit_optimization.csv)
OPTIMAL_EXITS = {
    "GapLarge_First30min_SPY":            time(14, 50),
    "HighVolWR_30min_SPY_filtered":       time(14, 45),
    "PriorDayStrong_AboveOR_QQQ_short":   time(10, 0),
    "PriorDayStrong_AboveOR_SPY_short":   time(10, 10),
    "PriorDayWeak_30min_QQQ":             time(15, 20),
    "PriorDayWeak_30min_SPY_filtered":    time(15, 5),
    "PriorDayWeak_50Hi_SPY_filtered":     time(15, 5),
}

# Delta-scaled premium budgets (same as main backtest)
PREMIUM_BY_DELTA = {
    0.10: 10_000,
    0.20: 20_000,
    0.30: 30_000,
    0.40: 40_000,
    0.50: 50_000,
    0.60: 60_000,
    0.70: 75_000,
    0.80: 80_000,
    0.90: 90_000,
}

SHORT_NAMES = {
    'GapLarge_First30min_SPY': 'GapLarge',
    'HighVolWR_30min_SPY_filtered': 'HighVolWR',
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
    'PriorDayWeak_30min_QQQ': 'QQQ Weak',
    'PriorDayWeak_30min_SPY_filtered': 'SPY Weak',
    'PriorDayWeak_50Hi_SPY_filtered': '50Hi Weak',
}


# ── Core Logic ───────────────────────────────────────────────────────────────

def simulate_with_optimal_exit(bars_df, entry_time, optimal_exit_time, entry_price,
                                target_delta, num_contracts_orig, premium_paid_orig):
    """
    Re-simulate a trade using the optimal exit time and delta-scaled sizing.

    Returns dict with new P&L or None if bars unavailable.
    """
    trade_date = entry_time.date()

    # Build target exit timestamp
    exit_dt = pd.Timestamp(
        year=trade_date.year, month=trade_date.month, day=trade_date.day,
        hour=optimal_exit_time.hour, minute=optimal_exit_time.minute,
        tz='America/New_York'
    )

    # Can't exit before entry
    if exit_dt <= entry_time:
        return None

    # Parse bar timestamps
    if 'timestamp' in bars_df.columns:
        bars_df = bars_df.copy()
        bars_df['ts'] = pd.to_datetime(bars_df['timestamp'], utc=True).dt.tz_convert('America/New_York')
    elif hasattr(bars_df.index, 'tz'):
        bars_df = bars_df.copy()
        ts = bars_df.index
        bars_df['ts'] = ts.tz_convert('America/New_York') if ts.tz else ts.tz_localize('America/New_York')
    else:
        bars_df = bars_df.copy()
        bars_df['ts'] = pd.to_datetime(bars_df.index, utc=True).tz_convert('America/New_York')

    # Filter to same day, at or before exit time
    day_bars = bars_df[bars_df['ts'].dt.date == trade_date]
    valid = day_bars[day_bars['ts'] <= exit_dt]

    if valid.empty:
        return None

    exit_bar = valid.iloc[-1]
    exit_price = float(exit_bar['close'])
    exit_bar_time = str(exit_bar['ts'])

    # Delta-scaled sizing
    budget = PREMIUM_BY_DELTA.get(target_delta, 10_000)
    cost_per_contract = entry_price * 100
    if cost_per_contract <= 0:
        return None
    num_contracts = max(1, min(500, int(budget / cost_per_contract)))
    premium_paid = num_contracts * cost_per_contract

    # P&L with slippage
    entry_adj = entry_price * (1 + SLIPPAGE_PCT)
    exit_adj = exit_price * (1 - SLIPPAGE_PCT)
    exit_adj = max(exit_adj, 0)
    pnl_per_share = exit_adj - entry_adj
    pnl = pnl_per_share * num_contracts * 100
    pnl = max(pnl, -premium_paid)
    pnl_pct = pnl / premium_paid if premium_paid > 0 else 0

    return {
        'exit_price': exit_price,
        'exit_time': exit_bar_time,
        'num_contracts': num_contracts,
        'premium_paid': premium_paid,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
    }


def run_combined_optimization():
    """Main: replay all optimal trades with delta-scaled sizing + optimal exits."""

    # Load optimal trades
    trades_path = os.path.join(OUTPUT_DIR, "options_backtest_trades_optimal.csv")
    print(f"Loading trades from {trades_path}")
    opt = pd.read_csv(trades_path)
    ok = opt[opt['status'] == 'ok'].copy()
    print(f"Processing {len(ok)} OK trades")

    print(f"\nPer-rule optimal exit times:")
    for rule, exit_t in sorted(OPTIMAL_EXITS.items()):
        sn = SHORT_NAMES.get(rule, rule)
        print(f"  {sn:<18} → {exit_t.strftime('%-I:%M %p')}")

    print(f"\nDelta-scaled budgets:")
    for delta, budget in sorted(PREMIUM_BY_DELTA.items()):
        print(f"  Δ{delta:.2f} → ${budget:,}")

    client = OptionsClient()

    # Results tracking
    new_trades = []  # Full trade records for CSV/calendar
    trade_count = 0
    skipped = 0

    for idx, (_, trade) in enumerate(ok.iterrows()):
        rule = trade['rule']
        option_ticker = trade['option_ticker']
        entry_price = float(trade['option_entry_price'])
        target_delta = float(trade['target_delta'])
        entry_dt = pd.to_datetime(trade['entry_time'], utc=True).tz_convert('America/New_York')
        trade_date = entry_dt.date()

        optimal_exit = OPTIMAL_EXITS.get(rule, time(14, 5))  # fallback to 2:05 PM

        # Load cached bars
        try:
            bars = client.get_options_bars(option_ticker, str(trade_date), str(trade_date))
        except Exception:
            skipped += 1
            # Keep original trade data as fallback
            new_trades.append({
                'rule': rule, 'ticker': trade['ticker'], 'direction': trade['direction'],
                'entry_time': trade['entry_time'], 'exit_time': trade['exit_time'],
                'option_ticker': option_ticker, 'strike': trade['strike'],
                'expiry_date': trade['expiry_date'], 'option_type': trade['option_type'],
                'target_delta': target_delta, 'actual_delta': trade['actual_delta'],
                'option_entry_price': entry_price,
                'option_exit_price': trade['option_exit_price'],
                'num_contracts': int(trade['num_contracts']),
                'premium_paid': trade['premium_paid'],
                'pnl': trade['pnl'], 'pnl_pct': trade['pnl_pct'],
                'exit_reason': 'original_fallback', 'status': 'ok',
            })
            continue

        if bars is None or bars.empty:
            skipped += 1
            new_trades.append({
                'rule': rule, 'ticker': trade['ticker'], 'direction': trade['direction'],
                'entry_time': trade['entry_time'], 'exit_time': trade['exit_time'],
                'option_ticker': option_ticker, 'strike': trade['strike'],
                'expiry_date': trade['expiry_date'], 'option_type': trade['option_type'],
                'target_delta': target_delta, 'actual_delta': trade['actual_delta'],
                'option_entry_price': entry_price,
                'option_exit_price': trade['option_exit_price'],
                'num_contracts': int(trade['num_contracts']),
                'premium_paid': trade['premium_paid'],
                'pnl': trade['pnl'], 'pnl_pct': trade['pnl_pct'],
                'exit_reason': 'original_fallback', 'status': 'ok',
            })
            continue

        result = simulate_with_optimal_exit(
            bars, entry_dt, optimal_exit, entry_price, target_delta,
            int(trade['num_contracts']), float(trade['premium_paid'])
        )

        if result is None:
            # Exit before entry (e.g., short exits at 10:00 but entry is 9:31 — should be fine,
            # but some edge cases). Use original.
            new_trades.append({
                'rule': rule, 'ticker': trade['ticker'], 'direction': trade['direction'],
                'entry_time': trade['entry_time'], 'exit_time': trade['exit_time'],
                'option_ticker': option_ticker, 'strike': trade['strike'],
                'expiry_date': trade['expiry_date'], 'option_type': trade['option_type'],
                'target_delta': target_delta, 'actual_delta': trade['actual_delta'],
                'option_entry_price': entry_price,
                'option_exit_price': trade['option_exit_price'],
                'num_contracts': int(trade['num_contracts']),
                'premium_paid': trade['premium_paid'],
                'pnl': trade['pnl'], 'pnl_pct': trade['pnl_pct'],
                'exit_reason': 'original_fallback', 'status': 'ok',
            })
            skipped += 1
            continue

        trade_count += 1
        new_trades.append({
            'rule': rule, 'ticker': trade['ticker'], 'direction': trade['direction'],
            'entry_time': trade['entry_time'], 'exit_time': result['exit_time'],
            'option_ticker': option_ticker, 'strike': trade['strike'],
            'expiry_date': trade['expiry_date'], 'option_type': trade['option_type'],
            'target_delta': target_delta, 'actual_delta': trade['actual_delta'],
            'option_entry_price': entry_price,
            'option_exit_price': result['exit_price'],
            'num_contracts': result['num_contracts'],
            'premium_paid': result['premium_paid'],
            'pnl': result['pnl'], 'pnl_pct': result['pnl_pct'],
            'exit_reason': f"optimal_exit_{optimal_exit.strftime('%H%M')}",
            'status': 'ok',
        })

        if (idx + 1) % 50 == 0 or idx == len(ok) - 1:
            print(f"  [{idx+1}/{len(ok)}] {trade_count} optimized, {skipped} fallback")

    print(f"\nDone: {trade_count} optimized, {skipped} used fallback")

    # Save trades CSV
    df = pd.DataFrame(new_trades)
    out_csv = os.path.join(OUTPUT_DIR, "options_combined_optimal.csv")
    df.to_csv(out_csv, index=False)
    print(f"Trades saved to {out_csv}")

    # ── Summary ──────────────────────────────────────────────────────────────
    ok_df = df[df['status'] == 'ok']
    rules = sorted(ok_df['rule'].unique())

    print(f"\n{'='*110}")
    print(f"COMBINED OPTIMIZATION: Delta-Scaled Sizing + Optimal Exit Times")
    print(f"{'='*110}")

    total_pnl = 0
    total_trades_n = 0
    total_premium = 0

    for rule in rules:
        r = ok_df[ok_df['rule'] == rule]
        rpnl = r['pnl'].sum()
        rprem = r['premium_paid'].sum()
        rwr = (r['pnl'] > 0).mean() * 100
        rroi = rpnl / rprem * 100 if rprem > 0 else 0
        delta = r['target_delta'].iloc[0]
        exit_t = OPTIMAL_EXITS.get(rule, time(14, 5))
        sn = SHORT_NAMES.get(rule, rule)
        total_pnl += rpnl
        total_trades_n += len(r)
        total_premium += rprem
        print(f"  {sn:<18} Δ{delta:.1f}  Exit {exit_t.strftime('%-I:%M %p'):<10} {len(r):>4} trades  ${rpnl:>+11,.0f}  {rwr:.0f}% WR  {rroi:+.1f}% ROI")

    wr = (ok_df['pnl'] > 0).mean() * 100
    roi = total_pnl / total_premium * 100 if total_premium > 0 else 0
    print(f"  {'─'*100}")
    print(f"  {'TOTAL':<18}                       {total_trades_n:>4} trades  ${total_pnl:>+11,.0f}  {wr:.0f}% WR  {roi:+.1f}% ROI")

    # Compare to previous
    prev_path = os.path.join(OUTPUT_DIR, "options_backtest_trades_optimal.csv")
    prev = pd.read_csv(prev_path)
    prev_ok = prev[prev['status'] == 'ok']
    prev_pnl = prev_ok['pnl'].sum()
    improvement = total_pnl - prev_pnl
    print(f"\n  Previous (delta-scaled, default exits): ${prev_pnl:>+11,.0f}")
    print(f"  Combined (delta-scaled + optimal exits): ${total_pnl:>+11,.0f}")
    print(f"  Improvement:                             ${improvement:>+11,.0f}  ({improvement/abs(prev_pnl)*100:+.1f}%)")

    # By year
    ok_df2 = ok_df.copy()
    ok_df2['year'] = pd.to_datetime(ok_df2['entry_time']).dt.year
    print(f"\n  BY YEAR:")
    for yr in sorted(ok_df2['year'].unique()):
        yr_df = ok_df2[ok_df2['year'] == yr]
        yr_pnl = yr_df['pnl'].sum()
        print(f"    {yr}: ${yr_pnl:>+11,.0f}  ({len(yr_df)} trades)")


if __name__ == "__main__":
    run_combined_optimization()
