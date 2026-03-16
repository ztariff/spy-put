#!/usr/bin/env python3
"""
Entry Optimization: VWAP Confirmation + Entry Delay
=====================================================
Tests whether confirming entries with intraday price action improves results.

Three types of entry filters tested:

1. VWAP FILTER
   - For LONG trades:  only enter if price < VWAP (buying below fair value)
   - For SHORT trades: only enter if price > VWAP (selling above fair value)
   - Also tests the OPPOSITE: longs above VWAP (momentum), shorts below VWAP

2. ENTRY DELAY
   - Instead of entering at 9:31, wait 5/10/15/20/30 min
   - Re-prices the option at the delayed entry time from cached 5-min bars
   - Keeps the same optimal exit time

3. VWAP + DELAY COMBO
   - Wait N minutes, THEN check VWAP condition
   - Gives the market time to establish direction before entering

Reads:
  - output/options_regime_filtered.csv (current best portfolio)
  - data/5m/SPY.parquet, data/5m/QQQ.parquet (equity 5-min bars for VWAP)
  - data/options_cache/ (cached option 5-min bars for re-pricing)

Output:
  - output/entry_optimization.csv (summary of all tests)
  - output/options_entry_optimized.csv (trades with best entry method applied)

Usage:
    python run_entry_optimizer.py
"""
import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import time, datetime, timedelta
from collections import defaultdict

from src.config import OUTPUT_DIR, DATA_DIR
from src.options_client import OptionsClient


# ── Configuration ────────────────────────────────────────────────────────────

SLIPPAGE_PCT = 0.01

PREMIUM_BY_DELTA = {
    0.10: 10_000, 0.20: 20_000, 0.30: 30_000, 0.40: 40_000,
    0.50: 50_000, 0.60: 60_000, 0.70: 75_000, 0.80: 80_000, 0.90: 90_000,
}

# Current optimal exit times (from combined optimizer)
OPTIMAL_EXITS = {
    "GapLarge_First30min_SPY":            time(14, 50),
    "HighVolWR_30min_SPY_filtered":       time(14, 45),
    "PriorDayStrong_AboveOR_QQQ_short":   time(10, 0),
    "PriorDayStrong_AboveOR_SPY_short":   time(10, 10),
    "PriorDayWeak_30min_QQQ":             time(15, 20),
    "PriorDayWeak_30min_SPY_filtered":    time(15, 5),
    "PriorDayWeak_50Hi_SPY_filtered":     time(15, 5),
}

# Entry delays to test (minutes after original entry)
ENTRY_DELAYS = [0, 5, 10, 15, 20, 30]

SHORT_NAMES = {
    'GapLarge_First30min_SPY': 'GapLarge',
    'HighVolWR_30min_SPY_filtered': 'HighVolWR',
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
    'PriorDayWeak_30min_QQQ': 'QQQ Weak',
    'PriorDayWeak_30min_SPY_filtered': 'SPY Weak',
    'PriorDayWeak_50Hi_SPY_filtered': '50Hi Weak',
}

# Map rules to their underlying ticker for VWAP lookup
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
    """Load 5-min bars for SPY and QQQ, compute session VWAP."""
    equity = {}
    for ticker in ['SPY', 'QQQ']:
        path = os.path.join(DATA_DIR, '5m', f'{ticker}.parquet')
        print(f"  Loading {path}...")
        df = pd.read_parquet(path)

        # Ensure timezone-aware index
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            idx = df.index.tz_convert('America/New_York')
        else:
            idx = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York')
        df.index = idx

        # Add date column for grouping
        df['date'] = df.index.date

        # Compute session VWAP (cumulative volume-weighted price within each day)
        if 'volume' in df.columns and 'close' in df.columns:
            vol_price = df['close'] * df['volume']
            df['cum_vol_price'] = vol_price.groupby(df['date']).cumsum()
            df['cum_vol'] = df['volume'].groupby(df['date']).cumsum()
            df['session_vwap'] = df['cum_vol_price'] / df['cum_vol'].replace(0, np.nan)
        elif 'vwap' in df.columns:
            # Polygon provides bar-level VWAP; compute session VWAP from it
            df['session_vwap'] = df['vwap']  # approximation
        else:
            print(f"    WARNING: No volume or vwap column for {ticker}")
            df['session_vwap'] = df['close']

        equity[ticker] = df
        print(f"    {len(df):,} bars, {df['date'].nunique()} days, cols={list(df.columns)[:8]}")

    return equity


def get_vwap_at_time(equity, ticker, target_dt):
    """
    Get the session VWAP and close price at a specific time.
    Returns (close_price, session_vwap) or (None, None) if unavailable.
    """
    df = equity.get(ticker)
    if df is None:
        return None, None

    target_date = target_dt.date()
    day_bars = df[df['date'] == target_date]

    if day_bars.empty:
        return None, None

    # Find the bar at or just before target time
    valid = day_bars[day_bars.index <= target_dt]
    if valid.empty:
        return None, None

    bar = valid.iloc[-1]
    return float(bar['close']), float(bar['session_vwap'])


def reprice_entry(client, option_ticker, trade_date, new_entry_dt, original_entry_price):
    """
    Get the option price at a delayed entry time from cached 5-min bars.
    Returns new entry price or None if unavailable.
    """
    try:
        bars = client.get_options_bars(option_ticker, str(trade_date), str(trade_date))
    except Exception:
        return None

    if bars is None or bars.empty:
        return None

    # Parse timestamps
    if 'timestamp' in bars.columns:
        bars = bars.copy()
        bars['ts'] = pd.to_datetime(bars['timestamp'], utc=True).dt.tz_convert('America/New_York')
    elif hasattr(bars.index, 'tz') and bars.index.tz is not None:
        bars = bars.copy()
        bars['ts'] = bars.index.tz_convert('America/New_York')
    else:
        bars = bars.copy()
        bars['ts'] = pd.to_datetime(bars.index, utc=True).tz_convert('America/New_York')

    # Find bar at or just after new entry time
    day_bars = bars[bars['ts'].dt.date == trade_date]
    valid = day_bars[day_bars['ts'] >= new_entry_dt]

    if valid.empty:
        # Try bars at or before entry time
        valid = day_bars[day_bars['ts'] <= new_entry_dt]
        if valid.empty:
            return None
        return float(valid.iloc[-1]['close'])

    return float(valid.iloc[0]['open'])  # entry at open of the bar


def simulate_trade(entry_price, exit_price, target_delta, direction):
    """Compute P&L for a trade with delta-scaled sizing."""
    budget = PREMIUM_BY_DELTA.get(target_delta, 50_000)
    cost_per = entry_price * 100
    if cost_per <= 0:
        return None

    num_contracts = max(1, min(500, int(budget / cost_per)))

    entry_adj = entry_price * (1 + SLIPPAGE_PCT)
    exit_adj = exit_price * (1 - SLIPPAGE_PCT)
    exit_adj = max(exit_adj, 0)

    if direction == 'long':
        pnl = (exit_adj - entry_adj) * num_contracts * 100
        premium = num_contracts * entry_adj * 100
        pnl = max(pnl, -premium)
    else:
        # Short: sold at entry, buy back at exit
        pnl = (entry_adj - exit_adj) * num_contracts * 100
        premium = num_contracts * entry_adj * 100

    pnl_pct = pnl / premium if premium > 0 else 0
    return {
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'num_contracts': num_contracts,
        'premium_paid': premium,
        'entry_price': entry_price,
    }


def get_exit_price(client, option_ticker, trade_date, exit_time_target, entry_dt):
    """Get option price at exit time from cached bars."""
    try:
        bars = client.get_options_bars(option_ticker, str(trade_date), str(trade_date))
    except Exception:
        return None

    if bars is None or bars.empty:
        return None

    # Parse timestamps
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

    # Must exit after entry
    if exit_dt <= entry_dt:
        return None

    day_bars = bars[bars['ts'].dt.date == trade_date]
    valid = day_bars[(day_bars['ts'] <= exit_dt) & (day_bars['ts'] > entry_dt)]

    if valid.empty:
        return None

    return float(valid.iloc[-1]['close'])


def run_entry_optimizer():
    """Main: test entry delays and VWAP filters."""

    # Load trades
    input_path = os.path.join(OUTPUT_DIR, "options_regime_filtered.csv")
    print(f"Loading trades from {input_path}")
    trades = pd.read_csv(input_path)
    trades = trades[trades['status'] == 'ok'].copy()
    trades['entry_dt'] = pd.to_datetime(trades['entry_time'], utc=True).dt.tz_convert('America/New_York')
    trades['trade_date'] = trades['entry_dt'].dt.date
    trades = trades.sort_values('entry_dt').reset_index(drop=True)
    print(f"  {len(trades)} trades loaded")

    # Load equity bars for VWAP
    print("\nLoading equity bars for VWAP...")
    equity = load_equity_bars()

    # Initialize options client for re-pricing
    client = OptionsClient()

    rules = sorted(trades['rule'].unique())
    baseline_pnl = trades['pnl'].sum()
    baseline_wr = (trades['pnl'] > 0).mean()
    print(f"\nBaseline: {len(trades)} trades, ${baseline_pnl:+,.0f}, WR {baseline_wr:.1%}")

    # ── Test 1: VWAP Filter (no delay, just filter at original entry time) ──
    print(f"\n{'='*110}")
    print("TEST 1: VWAP FILTER (at original entry time)")
    print(f"{'='*110}")

    vwap_results = []

    for vwap_mode in ['confirming', 'opposing']:
        # confirming: longs below VWAP (buying cheap), shorts above VWAP (selling rich)
        # opposing:   longs above VWAP (momentum), shorts below VWAP (breakdown)
        kept_pnl = 0
        kept_n = 0
        kept_wins = 0
        skipped_pnl = 0
        skipped_n = 0
        no_data = 0

        per_rule = defaultdict(lambda: {'kept_pnl': 0, 'kept_n': 0, 'kept_wins': 0, 'skip_pnl': 0, 'skip_n': 0})

        for _, trade in trades.iterrows():
            rule = trade['rule']
            direction = trade['direction']
            ticker = RULE_TICKER[rule]
            entry_dt = trade['entry_dt']

            price, vwap = get_vwap_at_time(equity, ticker, entry_dt)

            if price is None or vwap is None or np.isnan(price) or np.isnan(vwap):
                # No VWAP data — keep trade as-is
                kept_pnl += trade['pnl']
                kept_n += 1
                if trade['pnl'] > 0:
                    kept_wins += 1
                per_rule[rule]['kept_pnl'] += trade['pnl']
                per_rule[rule]['kept_n'] += 1
                if trade['pnl'] > 0:
                    per_rule[rule]['kept_wins'] += 1
                no_data += 1
                continue

            if vwap_mode == 'confirming':
                # Longs: price < VWAP (buying below fair value)
                # Shorts: price > VWAP (selling above fair value)
                take = (direction == 'long' and price < vwap) or \
                       (direction == 'short' and price > vwap)
            else:
                # Longs: price > VWAP (momentum confirmation)
                # Shorts: price < VWAP (breakdown confirmation)
                take = (direction == 'long' and price > vwap) or \
                       (direction == 'short' and price < vwap)

            if take:
                kept_pnl += trade['pnl']
                kept_n += 1
                if trade['pnl'] > 0:
                    kept_wins += 1
                per_rule[rule]['kept_pnl'] += trade['pnl']
                per_rule[rule]['kept_n'] += 1
                if trade['pnl'] > 0:
                    per_rule[rule]['kept_wins'] += 1
            else:
                skipped_pnl += trade['pnl']
                skipped_n += 1
                per_rule[rule]['skip_pnl'] += trade['pnl']
                per_rule[rule]['skip_n'] += 1

        total = kept_n + skipped_n
        wr = kept_wins / kept_n * 100 if kept_n > 0 else 0
        print(f"\n  VWAP {vwap_mode.upper()}: Keep {kept_n}/{total} ({kept_n/total*100:.0f}%), "
              f"P&L ${kept_pnl:+,.0f} (vs ${baseline_pnl:+,.0f}), WR {wr:.1f}%, "
              f"Avg ${kept_pnl/kept_n:+,.0f}/trade (no-data: {no_data})")
        print(f"  Skipped trades had P&L: ${skipped_pnl:+,.0f} ({'loss avoided' if skipped_pnl < 0 else 'gains missed'})")

        print(f"\n  {'Strategy':<16} {'Kept':>6} {'Skip':>6} {'Kept P&L':>12} {'Skip P&L':>12} {'Kept WR':>8}")
        for rule in rules:
            sn = SHORT_NAMES.get(rule, rule[:12])
            pr = per_rule[rule]
            k_wr = pr['kept_wins']/pr['kept_n']*100 if pr['kept_n'] > 0 else 0
            print(f"  {sn:<16} {pr['kept_n']:>6} {pr['skip_n']:>6} ${pr['kept_pnl']:>+10,.0f} ${pr['skip_pnl']:>+10,.0f} {k_wr:>7.1f}%")

        vwap_results.append({
            'mode': vwap_mode,
            'kept': kept_n, 'total': total,
            'pnl': kept_pnl, 'wr': wr,
            'skipped_pnl': skipped_pnl,
        })

    # ── Test 2: Entry Delay (re-price entry, same exit) ─────────────────────
    print(f"\n{'='*110}")
    print("TEST 2: ENTRY DELAY (wait N minutes, re-price option)")
    print(f"{'='*110}")

    delay_results = []

    for delay_min in ENTRY_DELAYS:
        if delay_min == 0:
            # Baseline — just report current
            delay_results.append({
                'delay': 0, 'pnl': baseline_pnl,
                'trades': len(trades), 'wr': baseline_wr * 100,
                'avg_pnl': baseline_pnl / len(trades),
            })
            print(f"\n  Delay +0 min (baseline): {len(trades)} trades, ${baseline_pnl:+,.0f}")
            continue

        total_pnl = 0
        total_trades_n = 0
        total_wins = 0
        skipped = 0
        per_rule_delay = defaultdict(lambda: {'pnl': 0, 'n': 0, 'wins': 0})

        for idx, (_, trade) in enumerate(trades.iterrows()):
            rule = trade['rule']
            option_ticker = trade['option_ticker']
            entry_dt = trade['entry_dt']
            trade_date = trade['trade_date']
            direction = trade['direction']
            target_delta = float(trade['target_delta'])

            # New entry time
            new_entry_dt = entry_dt + timedelta(minutes=delay_min)

            # Check: new entry must be before exit time
            exit_t = OPTIMAL_EXITS.get(rule, time(14, 5))
            exit_dt = pd.Timestamp(
                year=trade_date.year, month=trade_date.month, day=trade_date.day,
                hour=exit_t.hour, minute=exit_t.minute,
                tz='America/New_York'
            )

            if new_entry_dt >= exit_dt:
                skipped += 1
                continue

            # Re-price entry
            new_entry_price = reprice_entry(client, option_ticker, trade_date, new_entry_dt,
                                           trade['option_entry_price'])

            if new_entry_price is None or new_entry_price <= 0:
                # Can't re-price, use original
                total_pnl += trade['pnl']
                total_trades_n += 1
                if trade['pnl'] > 0:
                    total_wins += 1
                per_rule_delay[rule]['pnl'] += trade['pnl']
                per_rule_delay[rule]['n'] += 1
                if trade['pnl'] > 0:
                    per_rule_delay[rule]['wins'] += 1
                continue

            # Get exit price (same exit time as before)
            exit_price = get_exit_price(client, option_ticker, trade_date, exit_t, new_entry_dt)

            if exit_price is None:
                # Can't get exit price, use original trade
                total_pnl += trade['pnl']
                total_trades_n += 1
                if trade['pnl'] > 0:
                    total_wins += 1
                per_rule_delay[rule]['pnl'] += trade['pnl']
                per_rule_delay[rule]['n'] += 1
                if trade['pnl'] > 0:
                    per_rule_delay[rule]['wins'] += 1
                continue

            # Simulate with new entry price
            result = simulate_trade(new_entry_price, exit_price, target_delta, direction)

            if result is None:
                total_pnl += trade['pnl']
                total_trades_n += 1
                if trade['pnl'] > 0:
                    total_wins += 1
                per_rule_delay[rule]['pnl'] += trade['pnl']
                per_rule_delay[rule]['n'] += 1
                if trade['pnl'] > 0:
                    per_rule_delay[rule]['wins'] += 1
                continue

            total_pnl += result['pnl']
            total_trades_n += 1
            if result['pnl'] > 0:
                total_wins += 1
            per_rule_delay[rule]['pnl'] += result['pnl']
            per_rule_delay[rule]['n'] += 1
            if result['pnl'] > 0:
                per_rule_delay[rule]['wins'] += 1

            if (idx + 1) % 200 == 0:
                print(f"    [{idx+1}/{len(trades)}] delay +{delay_min}min processing...")

        wr = total_wins / total_trades_n * 100 if total_trades_n > 0 else 0
        avg = total_pnl / total_trades_n if total_trades_n > 0 else 0
        print(f"\n  Delay +{delay_min} min: {total_trades_n} trades (skipped {skipped}), "
              f"${total_pnl:+,.0f} (vs ${baseline_pnl:+,.0f}), WR {wr:.1f}%, Avg ${avg:+,.0f}")

        print(f"  {'Strategy':<16} {'Trades':>7} {'P&L':>12} {'WR':>7}")
        for rule in rules:
            sn = SHORT_NAMES.get(rule, rule[:12])
            pr = per_rule_delay[rule]
            r_wr = pr['wins']/pr['n']*100 if pr['n'] > 0 else 0
            print(f"  {sn:<16} {pr['n']:>7} ${pr['pnl']:>+10,.0f} {r_wr:>6.1f}%")

        delay_results.append({
            'delay': delay_min, 'pnl': total_pnl,
            'trades': total_trades_n, 'wr': wr,
            'avg_pnl': avg,
        })

    # ── Test 3: VWAP + Delay Combo ──────────────────────────────────────────
    print(f"\n{'='*110}")
    print("TEST 3: VWAP CONFIRMING + ENTRY DELAY COMBO")
    print(f"{'='*110}")

    combo_results = []
    # Test: wait 10 min, then check VWAP confirming filter
    for delay_min in [5, 10, 15]:
        kept_pnl = 0
        kept_n = 0
        kept_wins = 0
        skipped_entry = 0
        skipped_vwap = 0

        for _, trade in trades.iterrows():
            rule = trade['rule']
            direction = trade['direction']
            ticker = RULE_TICKER[rule]
            entry_dt = trade['entry_dt']
            trade_date = trade['trade_date']
            option_ticker = trade['option_ticker']
            target_delta = float(trade['target_delta'])

            new_entry_dt = entry_dt + timedelta(minutes=delay_min)

            # Check: new entry must be before exit time
            exit_t = OPTIMAL_EXITS.get(rule, time(14, 5))
            exit_dt_ts = pd.Timestamp(
                year=trade_date.year, month=trade_date.month, day=trade_date.day,
                hour=exit_t.hour, minute=exit_t.minute,
                tz='America/New_York'
            )
            if new_entry_dt >= exit_dt_ts:
                skipped_entry += 1
                continue

            # Check VWAP at delayed entry time
            price, vwap = get_vwap_at_time(equity, ticker, new_entry_dt)

            if price is not None and vwap is not None and not np.isnan(price) and not np.isnan(vwap):
                # Confirming: longs below VWAP, shorts above VWAP
                if direction == 'long' and price >= vwap:
                    skipped_vwap += 1
                    continue
                if direction == 'short' and price <= vwap:
                    skipped_vwap += 1
                    continue

            # Re-price entry
            new_entry_price = reprice_entry(client, option_ticker, trade_date, new_entry_dt,
                                           trade['option_entry_price'])
            exit_price_new = get_exit_price(client, option_ticker, trade_date, exit_t, new_entry_dt)

            if new_entry_price and new_entry_price > 0 and exit_price_new is not None:
                result = simulate_trade(new_entry_price, exit_price_new, target_delta, direction)
                if result:
                    kept_pnl += result['pnl']
                    kept_n += 1
                    if result['pnl'] > 0:
                        kept_wins += 1
                    continue

            # Fallback to original trade
            kept_pnl += trade['pnl']
            kept_n += 1
            if trade['pnl'] > 0:
                kept_wins += 1

        wr = kept_wins / kept_n * 100 if kept_n > 0 else 0
        print(f"\n  VWAP Confirming + {delay_min}min delay: {kept_n} trades "
              f"(skipped {skipped_entry} timing + {skipped_vwap} VWAP), "
              f"${kept_pnl:+,.0f}, WR {wr:.1f}%")
        combo_results.append({
            'delay': delay_min, 'vwap': 'confirming',
            'trades': kept_n, 'pnl': kept_pnl, 'wr': wr,
        })

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("SUMMARY: ALL ENTRY METHODS")
    print(f"{'='*110}")
    print(f"\n  {'Method':<40} {'Trades':>7} {'P&L':>14} {'vs Base':>12} {'WR':>7} {'Avg/Trade':>12}")
    print(f"  {'-'*95}")
    print(f"  {'Baseline (current)':<40} {len(trades):>7} ${baseline_pnl:>+12,.0f} {'':>12} {baseline_wr*100:>6.1f}% ${baseline_pnl/len(trades):>+10,.0f}")

    for vr in vwap_results:
        label = f"VWAP {vr['mode']}"
        avg = vr['pnl']/vr['kept'] if vr['kept'] > 0 else 0
        print(f"  {label:<40} {vr['kept']:>7} ${vr['pnl']:>+12,.0f} ${vr['pnl']-baseline_pnl:>+10,.0f} {vr['wr']:>6.1f}% ${avg:>+10,.0f}")

    for dr in delay_results:
        if dr['delay'] == 0:
            continue
        label = f"Entry delay +{dr['delay']}min"
        print(f"  {label:<40} {dr['trades']:>7} ${dr['pnl']:>+12,.0f} ${dr['pnl']-baseline_pnl:>+10,.0f} {dr['wr']:>6.1f}% ${dr['avg_pnl']:>+10,.0f}")

    for cr in combo_results:
        label = f"VWAP confirm + {cr['delay']}min delay"
        avg = cr['pnl']/cr['trades'] if cr['trades'] > 0 else 0
        print(f"  {label:<40} {cr['trades']:>7} ${cr['pnl']:>+12,.0f} ${cr['pnl']-baseline_pnl:>+10,.0f} {cr['wr']:>6.1f}% ${avg:>+10,.0f}")

    # Save summary
    summary_rows = []
    summary_rows.append({'method': 'Baseline', 'trades': len(trades), 'pnl': baseline_pnl,
                         'wr': baseline_wr*100, 'avg_pnl': baseline_pnl/len(trades)})
    for vr in vwap_results:
        avg = vr['pnl']/vr['kept'] if vr['kept'] > 0 else 0
        summary_rows.append({'method': f'VWAP_{vr["mode"]}', 'trades': vr['kept'],
                            'pnl': vr['pnl'], 'wr': vr['wr'], 'avg_pnl': avg})
    for dr in delay_results:
        if dr['delay'] > 0:
            summary_rows.append({'method': f'Delay_{dr["delay"]}min', 'trades': dr['trades'],
                                'pnl': dr['pnl'], 'wr': dr['wr'], 'avg_pnl': dr['avg_pnl']})
    for cr in combo_results:
        avg = cr['pnl']/cr['trades'] if cr['trades'] > 0 else 0
        summary_rows.append({'method': f'VWAP_confirm_delay_{cr["delay"]}min', 'trades': cr['trades'],
                            'pnl': cr['pnl'], 'wr': cr['wr'], 'avg_pnl': avg})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "entry_optimization.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    run_entry_optimizer()
