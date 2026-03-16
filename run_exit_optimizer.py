#!/usr/bin/env python3
"""
Exit Time Optimizer for Options Strategies
==========================================
Uses cached 5-min option bars to test every possible exit time
from 10:00 AM to 3:55 PM in 5-min increments.

For each trade, we already have the option ticker and entry price.
This script reads the full day's bars and computes P&L at each exit time.

Usage:
    python run_exit_optimizer.py              # Full analysis
    python run_exit_optimizer.py --rule HighVolWR_30min_SPY_filtered  # Single rule
    python run_exit_optimizer.py --interval 30  # 30-min intervals (default: 5)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import time

from src.config import OUTPUT_DIR, DATA_DIR
from src.options_client import OptionsClient


# ── Configuration ────────────────────────────────────────────────────────────

SLIPPAGE_PCT = 0.01  # 1% per side

# Exit times to test (NY time) — every 5 minutes from 10:00 to 15:55
ALL_EXIT_TIMES = []
for h in range(10, 16):
    for m in range(0, 60, 5):
        if h == 15 and m > 55:
            break
        ALL_EXIT_TIMES.append(time(h, m))


# ── Core Logic ───────────────────────────────────────────────────────────────

def compute_pnl_at_exit(bars_df, entry_time, exit_time_target, entry_price, num_contracts, premium_paid):
    """
    Given an option's 5-min bars, compute PnL if we exit at exit_time_target.

    Parameters
    ----------
    bars_df : DataFrame with 5-min bars (index or column = timestamp, 'close' column)
    entry_time : pd.Timestamp (NY tz)
    exit_time_target : datetime.time (e.g. time(14, 0))
    entry_price : float (option entry price per share)
    num_contracts : int
    premium_paid : float

    Returns
    -------
    dict with exit_price, pnl, pnl_pct, or None if no bar found
    """
    trade_date = entry_time.date()

    # Build target exit timestamp
    exit_dt = pd.Timestamp(
        year=trade_date.year, month=trade_date.month, day=trade_date.day,
        hour=exit_time_target.hour, minute=exit_time_target.minute,
        tz='America/New_York'
    )

    # Can't exit before entry
    if exit_dt <= entry_time:
        return None

    # Find the closest bar at or before the target exit time
    if 'timestamp' in bars_df.columns:
        bars_df = bars_df.copy()
        bars_df['ts'] = pd.to_datetime(bars_df['timestamp'], utc=True).dt.tz_convert('America/New_York')
    elif hasattr(bars_df.index, 'tz'):
        bars_df = bars_df.copy()
        bars_df['ts'] = bars_df.index.tz_convert('America/New_York') if bars_df.index.tz else bars_df.index.tz_localize('America/New_York')
    else:
        bars_df = bars_df.copy()
        bars_df['ts'] = pd.to_datetime(bars_df.index, utc=True).tz_convert('America/New_York')

    # Filter to same day and at or before exit time
    day_bars = bars_df[bars_df['ts'].dt.date == trade_date]
    valid = day_bars[day_bars['ts'] <= exit_dt]

    if valid.empty:
        return None

    # Use the last available bar's close
    exit_bar = valid.iloc[-1]
    exit_price = float(exit_bar['close'])

    # P&L with slippage
    entry_adj = entry_price * (1 + SLIPPAGE_PCT)
    exit_adj = exit_price * (1 - SLIPPAGE_PCT)
    exit_adj = max(exit_adj, 0)

    pnl_per_share = exit_adj - entry_adj
    pnl = pnl_per_share * num_contracts * 100
    pnl = max(pnl, -premium_paid)  # Cap loss at premium
    pnl_pct = pnl / premium_paid if premium_paid > 0 else 0

    return {
        'exit_price': exit_price,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
    }


def run_exit_optimization(rule_filter=None, interval=5):
    """
    Main function: test all exit times for each optimal trade.
    """
    # Load optimal trades
    trades_path = os.path.join(OUTPUT_DIR, "options_backtest_trades_optimal.csv")
    print(f"Loading trades from {trades_path}")
    opt = pd.read_csv(trades_path)
    ok = opt[opt['status'] == 'ok'].copy()

    if rule_filter:
        ok = ok[ok['rule'] == rule_filter]
        print(f"Filtered to {rule_filter}: {len(ok)} trades")

    print(f"Processing {len(ok)} trades")

    # Filter exit times by interval
    exit_times = [t for t in ALL_EXIT_TIMES if t.minute % interval == 0]
    print(f"Testing {len(exit_times)} exit times ({interval}-min intervals)")
    print(f"Range: {exit_times[0].strftime('%-I:%M %p')} to {exit_times[-1].strftime('%-I:%M %p')}")

    client = OptionsClient()

    # Results: dict of {exit_time_str: {rule: [pnl_list]}}
    results = {t.strftime('%H:%M'): {} for t in exit_times}
    trade_count = 0
    skipped = 0

    for idx, (_, trade) in enumerate(ok.iterrows()):
        rule = trade['rule']
        option_ticker = trade['option_ticker']
        entry_price = trade['option_entry_price']
        num_contracts = int(trade['num_contracts'])
        premium_paid = trade['premium_paid']
        entry_dt = pd.to_datetime(trade['entry_time'], utc=True).tz_convert('America/New_York')
        trade_date = entry_dt.date()

        # Load bars for this option
        try:
            bars = client.get_options_bars(option_ticker, str(trade_date), str(trade_date))
        except Exception as e:
            skipped += 1
            continue

        if bars is None or bars.empty:
            skipped += 1
            continue

        trade_count += 1

        # Test each exit time
        for exit_t in exit_times:
            t_key = exit_t.strftime('%H:%M')

            result = compute_pnl_at_exit(
                bars, entry_dt, exit_t, entry_price, num_contracts, premium_paid
            )

            if result is not None:
                if rule not in results[t_key]:
                    results[t_key][rule] = []
                results[t_key][rule].append(result['pnl'])

        if (idx + 1) % 50 == 0 or idx == len(ok) - 1:
            print(f"  [{idx+1}/{len(ok)}] processed, {skipped} skipped")

    print(f"\nProcessed {trade_count} trades, skipped {skipped}")

    # Build summary
    rules = sorted(ok['rule'].unique())

    print(f"\n{'='*120}")
    print(f"EXIT TIME OPTIMIZATION RESULTS")
    print(f"{'='*120}")

    # Header
    hdr = f"{'Exit Time':<12}"
    short_names = {
        'GapLarge_First30min_SPY':'GapLrg',
        'HighVolWR_30min_SPY_filtered':'HiVol',
        'PriorDayStrong_AboveOR_QQQ_short':'QQQSht',
        'PriorDayStrong_AboveOR_SPY_short':'SPYSht',
        'PriorDayWeak_30min_QQQ':'QQQWk',
        'PriorDayWeak_30min_SPY_filtered':'SPYWk',
        'PriorDayWeak_50Hi_SPY_filtered':'50HiWk',
    }
    for rule in rules:
        sn = short_names.get(rule, rule[:6])
        hdr += f"{sn:>10}"
    hdr += f"{'TOTAL':>12}"
    print(hdr)
    print("-" * 120)

    # Per exit time
    summary_rows = []
    for exit_t in exit_times:
        t_key = exit_t.strftime('%H:%M')
        t_display = exit_t.strftime('%-I:%M %p')
        row = f"{t_display:<12}"
        total_pnl = 0
        row_data = {'exit_time': t_key}

        for rule in rules:
            pnl_list = results[t_key].get(rule, [])
            pnl_sum = sum(pnl_list)
            total_pnl += pnl_sum
            sn = short_names.get(rule, rule[:6])
            row += f"${pnl_sum:>+8,.0f}"
            row_data[rule] = pnl_sum

        row += f"  ${total_pnl:>+9,.0f}"
        row_data['total'] = total_pnl
        summary_rows.append(row_data)
        print(row)

    # Find best exit time per rule
    print(f"\n{'='*80}")
    print(f"OPTIMAL EXIT TIME PER STRATEGY")
    print(f"{'='*80}")

    for rule in rules:
        sn = short_names.get(rule, rule[:6])
        best_time = None
        best_pnl = -1e18
        for sr in summary_rows:
            pnl = sr.get(rule, 0)
            if pnl > best_pnl:
                best_pnl = pnl
                best_time = sr['exit_time']

        # Also find the current exit P&L for comparison
        current_pnl = ok[ok['rule'] == rule]['pnl'].sum()

        if best_time:
            bt = pd.Timestamp(f"2025-01-01 {best_time}").strftime('%-I:%M %p')
            improvement = best_pnl - current_pnl
            print(f"  {rule:<45} Best: {bt:<10} ${best_pnl:>+10,.0f}  (current: ${current_pnl:>+10,.0f}  change: ${improvement:>+9,.0f})")

    # Best overall
    best_total_row = max(summary_rows, key=lambda r: r['total'])
    bt = pd.Timestamp(f"2025-01-01 {best_total_row['exit_time']}").strftime('%-I:%M %p')
    current_total = ok['pnl'].sum()
    print(f"\n  {'PORTFOLIO':<45} Best: {bt:<10} ${best_total_row['total']:>+10,.0f}  (current: ${current_total:>+10,.0f})")

    # Save detailed results
    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(OUTPUT_DIR, "exit_optimization.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options exit time optimizer")
    parser.add_argument("--rule", type=str, help="Filter to single rule")
    parser.add_argument("--interval", type=int, default=5, help="Exit time interval in minutes (default: 5)")
    args = parser.parse_args()

    run_exit_optimization(rule_filter=args.rule, interval=args.interval)
