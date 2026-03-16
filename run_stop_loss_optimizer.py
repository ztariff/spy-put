#!/usr/bin/env python3
"""
Stop Loss Optimizer for Options Strategies
==========================================
Tests various stop-loss levels on option premium to see if cutting losers
early improves overall performance.

For each trade, reads the cached 5-min option bars and checks if the option
price ever drops below -(X)% of entry price during the holding period.
If triggered, exits at that bar instead of the planned exit.

Usage:
    python run_stop_loss_optimizer.py              # Full analysis
    python run_stop_loss_optimizer.py --rule HighVolWR_30min_SPY_filtered
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

SLIPPAGE_PCT = 0.01  # 1% per side (same as backtest)

# Stop loss levels to test (% of premium lost)
STOP_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
# 1.00 = no stop (let it ride to planned exit)


# ── Core Logic ───────────────────────────────────────────────────────────────

def compute_pnl_with_stop(bars_df, entry_time, planned_exit_time, entry_price,
                          num_contracts, premium_paid, stop_pct):
    """
    Given an option's 5-min bars, compute PnL with a stop loss.

    Parameters
    ----------
    bars_df : DataFrame with 5-min bars
    entry_time : pd.Timestamp (NY tz)
    planned_exit_time : pd.Timestamp (NY tz) — the original exit
    entry_price : float (option entry price per share)
    num_contracts : int
    premium_paid : float
    stop_pct : float (e.g. 0.50 means stop if option drops 50% from entry)

    Returns
    -------
    dict with pnl, exit_price, exit_reason, exit_bar_time, or None
    """
    trade_date = entry_time.date()

    # Parse timestamps
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

    # Filter to same day, between entry and planned exit
    day_bars = bars_df[bars_df['ts'].dt.date == trade_date]
    holding_bars = day_bars[(day_bars['ts'] > entry_time) & (day_bars['ts'] <= planned_exit_time)]

    if holding_bars.empty:
        return None

    # Adjusted entry price (with slippage)
    entry_adj = entry_price * (1 + SLIPPAGE_PCT)

    # Stop price: if option drops to this level, we exit
    # stop_pct=0.50 means we stop when option is worth 50% less → price = entry * (1 - 0.50)
    stop_price = entry_price * (1 - stop_pct)

    # No stop if stop_pct >= 1.0 (let it expire worthless)
    if stop_pct >= 1.0:
        stop_price = -1  # Never triggers

    # Walk through bars to check for stop
    exit_price = None
    exit_reason = 'planned_exit'
    exit_bar_time = None

    for _, bar in holding_bars.iterrows():
        bar_low = float(bar['low']) if 'low' in bar.index else float(bar['close'])

        # Check if low breaches stop
        if stop_price >= 0 and bar_low <= stop_price:
            # Stopped out — use stop price (or close if close < stop)
            # In practice you'd get filled around the stop price
            exit_price = min(float(bar['close']), stop_price)
            exit_price = max(exit_price, 0)
            exit_reason = f'stop_{int(stop_pct*100)}pct'
            exit_bar_time = str(bar['ts'])
            break

    # If no stop triggered, use the last bar (planned exit)
    if exit_price is None:
        last_bar = holding_bars.iloc[-1]
        exit_price = float(last_bar['close'])
        exit_bar_time = str(last_bar['ts'])

    # P&L calculation
    exit_adj = exit_price * (1 - SLIPPAGE_PCT)
    exit_adj = max(exit_adj, 0)
    pnl_per_share = exit_adj - entry_adj
    pnl = pnl_per_share * num_contracts * 100
    pnl = max(pnl, -premium_paid)  # Cap loss at premium

    return {
        'exit_price': exit_price,
        'pnl': pnl,
        'exit_reason': exit_reason,
        'exit_bar_time': exit_bar_time,
        'stopped': exit_reason != 'planned_exit',
    }


def run_stop_loss_optimization(rule_filter=None):
    """
    Main function: test all stop loss levels for each optimal trade.
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
    print(f"Testing stop levels: {[f'-{int(s*100)}%' for s in STOP_LEVELS if s < 1.0]} + no-stop")

    client = OptionsClient()

    # Results: {stop_pct: {rule: [pnl_list]}}
    results = {s: {} for s in STOP_LEVELS}
    # Also track stop trigger rates
    stop_counts = {s: {} for s in STOP_LEVELS}
    trade_count = 0
    skipped = 0

    for idx, (_, trade) in enumerate(ok.iterrows()):
        rule = trade['rule']
        option_ticker = trade['option_ticker']
        entry_price = trade['option_entry_price']
        num_contracts = int(trade['num_contracts'])
        premium_paid = trade['premium_paid']
        entry_dt = pd.to_datetime(trade['entry_time'], utc=True).tz_convert('America/New_York')
        exit_dt = pd.to_datetime(trade['exit_time'], utc=True).tz_convert('America/New_York')
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

        # Test each stop level
        for stop_pct in STOP_LEVELS:
            result = compute_pnl_with_stop(
                bars, entry_dt, exit_dt, entry_price,
                num_contracts, premium_paid, stop_pct
            )

            if result is not None:
                if rule not in results[stop_pct]:
                    results[stop_pct][rule] = []
                    stop_counts[stop_pct][rule] = {'stopped': 0, 'total': 0}

                results[stop_pct][rule].append(result['pnl'])
                stop_counts[stop_pct][rule]['total'] += 1
                if result['stopped']:
                    stop_counts[stop_pct][rule]['stopped'] += 1

        if (idx + 1) % 50 == 0 or idx == len(ok) - 1:
            print(f"  [{idx+1}/{len(ok)}] processed, {skipped} skipped")

    print(f"\nProcessed {trade_count} trades, skipped {skipped}")

    # Build summary
    rules = sorted(ok['rule'].unique())

    short_names = {
        'GapLarge_First30min_SPY':'GapLrg',
        'HighVolWR_30min_SPY_filtered':'HiVol',
        'PriorDayStrong_AboveOR_QQQ_short':'QQQSht',
        'PriorDayStrong_AboveOR_SPY_short':'SPYSht',
        'PriorDayWeak_30min_QQQ':'QQQWk',
        'PriorDayWeak_30min_SPY_filtered':'SPYWk',
        'PriorDayWeak_50Hi_SPY_filtered':'50HiWk',
    }

    print(f"\n{'='*140}")
    print(f"STOP LOSS OPTIMIZATION RESULTS — P&L by Stop Level")
    print(f"{'='*140}")

    # Header
    hdr = f"{'Stop Level':<14}"
    for rule in rules:
        sn = short_names.get(rule, rule[:6])
        hdr += f"{sn:>10}"
    hdr += f"{'TOTAL':>12}"
    print(hdr)
    print("-" * 140)

    summary_rows = []
    for stop_pct in STOP_LEVELS:
        label = f"-{int(stop_pct*100)}%" if stop_pct < 1.0 else "No Stop"
        row = f"{label:<14}"
        total_pnl = 0
        row_data = {'stop_pct': stop_pct, 'stop_label': label}

        for rule in rules:
            pnl_list = results[stop_pct].get(rule, [])
            pnl_sum = sum(pnl_list)
            total_pnl += pnl_sum
            row += f"${pnl_sum:>+8,.0f}"
            row_data[rule] = pnl_sum

        row += f"  ${total_pnl:>+9,.0f}"
        row_data['total'] = total_pnl
        summary_rows.append(row_data)
        print(row)

    # Stop trigger rate table
    print(f"\n{'='*140}")
    print(f"STOP TRIGGER RATE — % of Trades Stopped Out")
    print(f"{'='*140}")

    hdr2 = f"{'Stop Level':<14}"
    for rule in rules:
        sn = short_names.get(rule, rule[:6])
        hdr2 += f"{sn:>10}"
    hdr2 += f"{'AVG':>12}"
    print(hdr2)
    print("-" * 140)

    for stop_pct in STOP_LEVELS:
        if stop_pct >= 1.0:
            continue
        label = f"-{int(stop_pct*100)}%"
        row = f"{label:<14}"
        rates = []

        for rule in rules:
            sc = stop_counts[stop_pct].get(rule, {'stopped': 0, 'total': 1})
            rate = sc['stopped'] / sc['total'] * 100 if sc['total'] > 0 else 0
            row += f"{rate:>9.1f}%"
            rates.append(rate)

        avg_rate = np.mean(rates) if rates else 0
        row += f"{avg_rate:>11.1f}%"
        print(row)

    # Best stop per rule
    print(f"\n{'='*100}")
    print(f"OPTIMAL STOP LOSS PER STRATEGY")
    print(f"{'='*100}")

    current_total = 0
    for rule in rules:
        best_stop = None
        best_pnl = -1e18
        no_stop_pnl = 0

        for sr in summary_rows:
            pnl = sr.get(rule, 0)
            if sr['stop_pct'] >= 1.0:
                no_stop_pnl = pnl
            if pnl > best_pnl:
                best_pnl = pnl
                best_stop = sr['stop_label']

        improvement = best_pnl - no_stop_pnl
        current_total += no_stop_pnl
        sn = short_names.get(rule, rule[:6])
        flag = " ***" if improvement > 0 else ""
        print(f"  {rule:<45} Best: {best_stop:<10} ${best_pnl:>+10,.0f}  (no stop: ${no_stop_pnl:>+10,.0f}  delta: ${improvement:>+9,.0f}){flag}")

    # Best overall
    best_row = max(summary_rows, key=lambda r: r['total'])
    no_stop_row = [r for r in summary_rows if r['stop_pct'] >= 1.0][0]
    print(f"\n  {'PORTFOLIO':<45} Best: {best_row['stop_label']:<10} ${best_row['total']:>+10,.0f}  (no stop: ${no_stop_row['total']:>+10,.0f}  delta: ${best_row['total'] - no_stop_row['total']:>+9,.0f})")

    # Per-rule optimal (mix different stops per rule)
    optimal_total = 0
    print(f"\n  PER-RULE OPTIMAL MIX:")
    for rule in rules:
        best_pnl = -1e18
        best_label = "No Stop"
        for sr in summary_rows:
            pnl = sr.get(rule, 0)
            if pnl > best_pnl:
                best_pnl = pnl
                best_label = sr['stop_label']
        optimal_total += best_pnl
        sn = short_names.get(rule, rule[:6])
        print(f"    {sn:<8} → {best_label:<10} ${best_pnl:>+10,.0f}")

    print(f"    {'TOTAL':<8}            ${optimal_total:>+10,.0f}")

    # Save detailed results
    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(OUTPUT_DIR, "stop_loss_optimization.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options stop loss optimizer")
    parser.add_argument("--rule", type=str, help="Filter to single rule")
    args = parser.parse_args()

    run_stop_loss_optimization(rule_filter=args.rule)
