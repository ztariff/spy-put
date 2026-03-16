#!/usr/bin/env python3
"""
Opening Price Confirmation Entry
==================================
Wait for the first bar where:
  - LONG trades:  underlying closes ABOVE the 9:30 opening price
  - SHORT trades: underlying closes BELOW the 9:30 opening price

Uses 5-min equity bars. Re-prices options at the confirmed entry time.

Usage:
    python run_open_price_entry.py
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import time, datetime, timedelta
from collections import defaultdict

from src.config import OUTPUT_DIR, DATA_DIR
from src.options_client import OptionsClient


SLIPPAGE_PCT = 0.01

PREMIUM_BY_DELTA = {
    0.10: 10_000, 0.20: 20_000, 0.30: 30_000, 0.40: 40_000,
    0.50: 50_000, 0.60: 60_000, 0.70: 75_000, 0.80: 80_000, 0.90: 90_000,
}

OPTIMAL_EXITS = {
    "GapLarge_First30min_SPY":            time(14, 50),
    "HighVolWR_30min_SPY_filtered":       time(14, 45),
    "PriorDayStrong_AboveOR_QQQ_short":   time(10, 0),
    "PriorDayStrong_AboveOR_SPY_short":   time(10, 10),
    "PriorDayWeak_30min_QQQ":             time(15, 20),
    "PriorDayWeak_30min_SPY_filtered":    time(15, 5),
    "PriorDayWeak_50Hi_SPY_filtered":     time(15, 5),
}

MAX_WAITS = [15, 30, 60, 120, 999]

SHORT_NAMES = {
    'GapLarge_First30min_SPY': 'GapLarge',
    'HighVolWR_30min_SPY_filtered': 'HighVolWR',
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
    'PriorDayWeak_30min_QQQ': 'QQQ Weak',
    'PriorDayWeak_30min_SPY_filtered': 'SPY Weak',
    'PriorDayWeak_50Hi_SPY_filtered': '50Hi Weak',
}

RULE_TICKER = {
    'GapLarge_First30min_SPY': 'SPY',
    'HighVolWR_30min_SPY_filtered': 'SPY',
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY',
    'PriorDayWeak_30min_QQQ': 'QQQ',
    'PriorDayWeak_30min_SPY_filtered': 'SPY',
    'PriorDayWeak_50Hi_SPY_filtered': 'SPY',
}


def load_equity_bars():
    """Load 5-min bars, build lookup: (ticker, date) -> list of bar dicts + opening price."""
    equity = {}
    opens = {}  # (ticker, date) -> opening price at 9:30

    for ticker in ['SPY', 'QQQ']:
        path = os.path.join(DATA_DIR, '5m', f'{ticker}.parquet')
        print(f"  Loading {path}...")
        df = pd.read_parquet(path)

        if hasattr(df.index, 'tz') and df.index.tz is not None:
            idx = df.index.tz_convert('America/New_York')
        else:
            idx = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
        df.index = idx
        df['date'] = df.index.date

        day_groups = {}
        for date, group in df.groupby('date'):
            bars = []
            open_price = None
            for ts, row in group.iterrows():
                bar_time = ts.time()
                # The 9:30 bar's open is the session opening price
                if open_price is None and bar_time >= time(9, 30):
                    open_price = float(row['open'])
                bars.append({
                    'ts': ts,
                    'close': float(row['close']),
                    'open': float(row['open']),
                })
            day_groups[date] = {'bars': bars, 'open_price': open_price}

        equity[ticker] = day_groups
        print(f"    {len(df):,} bars, {len(day_groups)} days")

    return equity


def find_open_cross(equity, ticker, trade_date, direction, earliest_dt, latest_dt):
    """
    Find first 5-min bar where:
      - LONG:  close > 9:30 open price
      - SHORT: close < 9:30 open price
    Returns bar timestamp or None.
    """
    day_data = equity.get(ticker, {}).get(trade_date)
    if not day_data:
        return None

    open_price = day_data['open_price']
    if open_price is None:
        return None

    for bar in day_data['bars']:
        ts = bar['ts']
        if ts < earliest_dt:
            continue
        if ts > latest_dt:
            break

        close = bar['close']

        if direction == 'long' and close > open_price:
            return ts
        elif direction == 'short' and close < open_price:
            return ts

    return None


def get_option_price_at_time(client, option_ticker, trade_date, target_dt):
    """Get option price at a specific time from cached 5-min bars."""
    try:
        bars = client.get_options_bars(option_ticker, str(trade_date), str(trade_date))
    except Exception:
        return None

    if bars is None or bars.empty:
        return None

    if 'timestamp' in bars.columns:
        bars = bars.copy()
        bars['ts'] = pd.to_datetime(bars['timestamp'], utc=True).dt.tz_convert('America/New_York')
    elif hasattr(bars.index, 'tz') and bars.index.tz is not None:
        bars = bars.copy()
        bars['ts'] = bars.index.tz_convert('America/New_York')
    else:
        bars = bars.copy()
        bars['ts'] = pd.to_datetime(bars.index, utc=True).tz_convert('America/New_York')

    day_bars = bars[bars['ts'].dt.date == trade_date]

    after = day_bars[day_bars['ts'] >= target_dt]
    if not after.empty:
        return float(after.iloc[0]['close'])

    before = day_bars[day_bars['ts'] < target_dt]
    if not before.empty:
        return float(before.iloc[-1]['close'])

    return None


def get_exit_price(client, option_ticker, trade_date, exit_time_target, entry_dt):
    """Get option price at exit time."""
    try:
        bars = client.get_options_bars(option_ticker, str(trade_date), str(trade_date))
    except Exception:
        return None

    if bars is None or bars.empty:
        return None

    if 'timestamp' in bars.columns:
        bars = bars.copy()
        bars['ts'] = pd.to_datetime(bars['timestamp'], utc=True).dt.tz_convert('America/New_York')
    elif hasattr(bars.index, 'tz') and bars.index.tz is not None:
        bars = bars.copy()
        bars['ts'] = bars.index.tz_convert('America/New_York')
    else:
        bars = bars.copy()
        bars['ts'] = pd.to_datetime(bars.index, utc=True).tz_convert('America/New_York')

    exit_dt = pd.Timestamp(
        year=trade_date.year, month=trade_date.month, day=trade_date.day,
        hour=exit_time_target.hour, minute=exit_time_target.minute,
        tz='America/New_York'
    )

    if exit_dt <= entry_dt:
        return None

    day_bars = bars[bars['ts'].dt.date == trade_date]
    valid = day_bars[(day_bars['ts'] <= exit_dt) & (day_bars['ts'] > entry_dt)]

    if valid.empty:
        return None

    return float(valid.iloc[-1]['close'])


def run():
    input_path = os.path.join(OUTPUT_DIR, "options_regime_filtered.csv")
    print(f"Loading trades from {input_path}")
    trades = pd.read_csv(input_path)
    trades = trades[trades['status'] == 'ok'].copy()
    trades['entry_dt'] = pd.to_datetime(trades['entry_time'], utc=True).dt.tz_convert('America/New_York')
    trades['trade_date'] = trades['entry_dt'].dt.date
    trades = trades.sort_values('entry_dt').reset_index(drop=True)
    print(f"  {len(trades)} trades loaded")

    baseline_pnl = trades['pnl'].sum()
    baseline_n = len(trades)
    baseline_wr = (trades['pnl'] > 0).mean() * 100

    print(f"\nBaseline: {baseline_n} trades, ${baseline_pnl:+,.0f}, WR {baseline_wr:.1f}%")

    print("\nLoading equity bars...")
    equity = load_equity_bars()
    client = OptionsClient()

    rules = sorted(trades['rule'].unique())
    all_results = []

    for max_wait in MAX_WAITS:
        wait_label = f"{max_wait}min" if max_wait < 999 else "unlimited"
        print(f"\n{'='*110}")
        print(f"OPEN PRICE CROSS — Max wait: {wait_label}")
        print(f"  Longs: wait for first 5-min close ABOVE 9:30 open")
        print(f"  Shorts: wait for first 5-min close BELOW 9:30 open")
        print(f"{'='*110}")

        total_pnl = 0
        total_n = 0
        total_wins = 0
        skipped_no_cross = 0
        skipped_no_data = 0
        entered_immediately = 0
        avg_delay_min = []

        per_rule = defaultdict(lambda: {
            'pnl': 0, 'n': 0, 'wins': 0, 'skip': 0,
            'delays': [], 'immediate': 0
        })

        for idx, (_, trade) in enumerate(trades.iterrows()):
            rule = trade['rule']
            direction = trade['direction']
            ticker = RULE_TICKER[rule]
            entry_dt = trade['entry_dt']
            trade_date = trade['trade_date']
            option_ticker = trade['option_ticker']
            target_delta = float(trade['target_delta'])

            exit_t = OPTIMAL_EXITS.get(rule, time(14, 5))
            exit_dt = pd.Timestamp(
                year=trade_date.year, month=trade_date.month, day=trade_date.day,
                hour=exit_t.hour, minute=exit_t.minute,
                tz='America/New_York'
            )

            if max_wait < 999:
                deadline = entry_dt + timedelta(minutes=max_wait)
                latest = min(deadline, exit_dt - timedelta(minutes=5))
            else:
                latest = exit_dt - timedelta(minutes=5)

            if latest <= entry_dt:
                # Short strategies with tight exits — use original
                total_pnl += trade['pnl']
                total_n += 1
                if trade['pnl'] > 0:
                    total_wins += 1
                entered_immediately += 1
                per_rule[rule]['pnl'] += trade['pnl']
                per_rule[rule]['n'] += 1
                if trade['pnl'] > 0:
                    per_rule[rule]['wins'] += 1
                per_rule[rule]['immediate'] += 1
                continue

            cross_time = find_open_cross(equity, ticker, trade_date, direction, entry_dt, latest)

            if cross_time is None:
                skipped_no_cross += 1
                per_rule[rule]['skip'] += 1
                continue

            delay = (cross_time - entry_dt).total_seconds() / 60
            avg_delay_min.append(delay)
            per_rule[rule]['delays'].append(delay)

            if delay <= 0:
                entered_immediately += 1
                per_rule[rule]['immediate'] += 1

            # Re-price option
            new_entry_price = get_option_price_at_time(client, option_ticker, trade_date, cross_time)
            new_exit_price = get_exit_price(client, option_ticker, trade_date, exit_t, cross_time)

            if new_entry_price is None or new_entry_price <= 0 or new_exit_price is None:
                if delay <= 5:
                    total_pnl += trade['pnl']
                    total_n += 1
                    if trade['pnl'] > 0:
                        total_wins += 1
                    per_rule[rule]['pnl'] += trade['pnl']
                    per_rule[rule]['n'] += 1
                    if trade['pnl'] > 0:
                        per_rule[rule]['wins'] += 1
                else:
                    skipped_no_data += 1
                    per_rule[rule]['skip'] += 1
                continue

            budget = PREMIUM_BY_DELTA.get(target_delta, 50_000)
            cost_per = new_entry_price * 100
            if cost_per <= 0:
                skipped_no_data += 1
                per_rule[rule]['skip'] += 1
                continue

            num_contracts = max(1, min(500, int(budget / cost_per)))
            entry_adj = new_entry_price * (1 + SLIPPAGE_PCT)
            exit_adj = new_exit_price * (1 - SLIPPAGE_PCT)
            exit_adj = max(exit_adj, 0)

            if direction == 'long':
                pnl = (exit_adj - entry_adj) * num_contracts * 100
                premium = num_contracts * entry_adj * 100
                pnl = max(pnl, -premium)
            else:
                pnl = (entry_adj - exit_adj) * num_contracts * 100
                premium = num_contracts * entry_adj * 100

            total_pnl += pnl
            total_n += 1
            if pnl > 0:
                total_wins += 1
            per_rule[rule]['pnl'] += pnl
            per_rule[rule]['n'] += 1
            if pnl > 0:
                per_rule[rule]['wins'] += 1

            if (idx + 1) % 200 == 0:
                print(f"    [{idx+1}/{len(trades)}] processing...")

        wr = total_wins / total_n * 100 if total_n > 0 else 0
        avg_d = np.mean(avg_delay_min) if avg_delay_min else 0

        print(f"\n  Trades taken:   {total_n} / {baseline_n}")
        print(f"  Skipped (no cross): {skipped_no_cross}, (no data): {skipped_no_data}")
        print(f"  Entered immediately: {entered_immediately}")
        print(f"  Avg delay: {avg_d:.1f} min")
        print(f"  P&L:    ${total_pnl:>+12,.0f}  (baseline ${baseline_pnl:>+12,.0f})")
        print(f"  Change: ${total_pnl - baseline_pnl:>+12,.0f}  ({(total_pnl-baseline_pnl)/abs(baseline_pnl)*100:+.1f}%)")
        print(f"  WR: {wr:.1f}%  Avg/trade: ${total_pnl/total_n:+,.0f}")

        print(f"\n  {'Strategy':<16} {'Taken':>6} {'Skip':>6} {'Imm':>5} {'AvgDel':>7} {'P&L':>12} {'WR':>6} {'Base P&L':>12}")
        print(f"  {'-'*85}")
        for rule in rules:
            sn = SHORT_NAMES.get(rule, rule[:12])
            pr = per_rule[rule]
            r_wr = pr['wins']/pr['n']*100 if pr['n'] > 0 else 0
            r_avg_d = np.mean(pr['delays']) if pr['delays'] else 0
            base_pnl_r = trades[trades['rule'] == rule]['pnl'].sum()
            print(f"  {sn:<16} {pr['n']:>6} {pr['skip']:>6} {pr['immediate']:>5} {r_avg_d:>5.1f}m ${pr['pnl']:>+10,.0f} {r_wr:>5.1f}% ${base_pnl_r:>+10,.0f}")

        all_results.append({
            'max_wait': wait_label, 'trades': total_n,
            'pnl': total_pnl, 'wr': wr,
            'avg_pnl': total_pnl/total_n if total_n > 0 else 0,
            'avg_delay': avg_d, 'skipped': skipped_no_cross + skipped_no_data,
        })

    # Summary
    print(f"\n{'='*110}")
    print("SUMMARY: OPEN PRICE CROSS ENTRY")
    print(f"{'='*110}")
    print(f"\n  {'Max Wait':<12} {'Trades':>7} {'P&L':>14} {'vs Base':>12} {'WR':>7} {'Avg/Trade':>12} {'Avg Delay':>10}")
    print(f"  {'-'*80}")
    print(f"  {'Baseline':<12} {baseline_n:>7} ${baseline_pnl:>+12,.0f} {'':>12} {baseline_wr:>6.1f}% ${baseline_pnl/baseline_n:>+10,.0f} {'0.0m':>10}")

    for r in all_results:
        print(f"  {r['max_wait']:<12} {r['trades']:>7} ${r['pnl']:>+12,.0f} ${r['pnl']-baseline_pnl:>+10,.0f} {r['wr']:>6.1f}% ${r['avg_pnl']:>+10,.0f} {r['avg_delay']:>8.1f}m")

    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(OUTPUT_DIR, "open_price_entry_optimization.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    run()
