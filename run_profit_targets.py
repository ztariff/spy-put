#!/usr/bin/env python3
"""
Profit Target & Trailing Stop Optimizer
========================================
Scans intraday 5-min option bars for each trade to simulate:
  1. Fixed profit targets (take profit at +X% on the option)
  2. Trailing stops (once up +X%, trail a stop at Y% below peak)
  3. Combined: profit target + trailing stop

Goal: improve win rate even if total P&L decreases.

Usage:
    python run_profit_targets.py
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import date, timedelta
from collections import defaultdict

from src.config import OUTPUT_DIR, DATA_DIR
from src.options_client import OptionsClient

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_CSV = os.path.join(OUTPUT_DIR, "options_regime_filtered.csv")

# Profit target levels to test (% gain on option price)
PROFIT_TARGETS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]

# Trailing stop configs: (activation_pct, trail_pct)
# e.g., (0.50, 0.30) = once option is up 50%, trail a stop 30% below peak
TRAILING_STOPS = [
    (0.25, 0.15),
    (0.50, 0.25),
    (0.50, 0.30),
    (0.75, 0.30),
    (1.00, 0.40),
    (1.00, 0.50),
    (1.50, 0.50),
]

# Combined: profit target + trailing stop
COMBINED = [
    # (profit_target, trail_activation, trail_pct)
    (1.00, 0.30, 0.20),   # Take profit at +100% OR trail after +30%
    (1.50, 0.50, 0.25),   # Take profit at +150% OR trail after +50%
    (2.00, 0.50, 0.30),   # Take profit at +200% OR trail after +50%
    (2.00, 0.75, 0.40),   # Take profit at +200% OR trail after +75%
]

SLIPPAGE_PCT = 0.01  # 1% per side


def load_trades():
    """Load regime-filtered trades."""
    df = pd.read_csv(INPUT_CSV)
    df = df[df['status'] == 'ok'].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    print(f"Loaded {len(df)} trades from {INPUT_CSV}")
    return df


def get_intraday_option_bars(client, option_ticker, trade_date, entry_time, exit_time):
    """
    Fetch 5-min option bars for the trade window.
    Returns DataFrame with close prices indexed by timestamp.
    """
    try:
        bars = client.get_options_bars(option_ticker, str(trade_date), str(trade_date),
                                        multiplier=5, timespan="minute")
    except Exception as e:
        return pd.DataFrame()

    if bars is None or bars.empty:
        return pd.DataFrame()

    # Filter to trade window
    bars = bars[(bars.index >= entry_time) & (bars.index <= exit_time)]
    return bars


def simulate_profit_target(bars, entry_price, profit_target_pct, direction):
    """
    Simulate a fixed profit target.
    Returns (exit_price, exit_time, hit) or None if not hit.
    """
    target_price = entry_price * (1 + profit_target_pct)

    for ts, row in bars.iterrows():
        # For long options (both calls and puts we bought), profit = price goes up
        if row['high'] >= target_price:
            return target_price, ts, True
        # Also check close in case high data is spotty
        if row['close'] >= target_price:
            return row['close'], ts, True

    return None, None, False


def simulate_trailing_stop(bars, entry_price, activation_pct, trail_pct, direction):
    """
    Simulate a trailing stop.
    Once option price rises by activation_pct, trail a stop at trail_pct below peak.
    Returns (exit_price, exit_time, activated) or None if not triggered.
    """
    activation_price = entry_price * (1 + activation_pct)
    activated = False
    peak_price = entry_price

    for ts, row in bars.iterrows():
        bar_high = row['high']
        bar_low = row['low']
        bar_close = row['close']

        # Update peak
        if bar_high > peak_price:
            peak_price = bar_high

        # Check activation
        if not activated and bar_high >= activation_price:
            activated = True
            peak_price = max(peak_price, bar_high)

        # Check trailing stop
        if activated:
            stop_price = peak_price * (1 - trail_pct)
            if bar_low <= stop_price:
                return stop_price, ts, True

    return None, None, activated


def simulate_combined(bars, entry_price, pt_pct, trail_act_pct, trail_pct, direction):
    """
    Combined: profit target + trailing stop.
    Exit at whichever triggers first.
    """
    target_price = entry_price * (1 + pt_pct)
    activation_price = entry_price * (1 + trail_act_pct)
    activated = False
    peak_price = entry_price

    for ts, row in bars.iterrows():
        bar_high = row['high']
        bar_low = row['low']

        # Check profit target first
        if bar_high >= target_price:
            return target_price, ts, "profit_target"

        # Update peak
        if bar_high > peak_price:
            peak_price = bar_high

        # Check trailing stop activation
        if not activated and bar_high >= activation_price:
            activated = True

        if activated:
            stop_price = peak_price * (1 - trail_pct)
            if bar_low <= stop_price:
                return stop_price, ts, "trailing_stop"

    return None, None, "time_exit"


def compute_trade_pnl(entry_price, exit_price, num_contracts, premium_paid):
    """Compute P&L with slippage."""
    entry_adj = entry_price * (1 + SLIPPAGE_PCT)
    exit_adj = exit_price * (1 - SLIPPAGE_PCT)
    exit_adj = max(exit_adj, 0)
    pnl = (exit_adj - entry_adj) * num_contracts * 100
    pnl = max(pnl, -premium_paid)
    return pnl


def compute_stats(pnls, trade_dates):
    """Compute portfolio stats from a list of P&Ls."""
    if not pnls:
        return {}

    pnls = np.array(pnls)
    n = len(pnls)
    wins = (pnls > 0).sum()
    wr = wins / n * 100

    # Daily aggregation for Sharpe
    daily = defaultdict(float)
    for pnl, dt in zip(pnls, trade_dates):
        daily[dt] += pnl
    daily_pnls = np.array(list(daily.values()))

    sharpe = daily_pnls.mean() / daily_pnls.std() * np.sqrt(252) if daily_pnls.std() > 0 else 0
    cum = np.cumsum(daily_pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = (cum - peak).min()

    win_days = (daily_pnls > 0).sum()

    return {
        'trades': n,
        'win_rate': wr,
        'total_pnl': pnls.sum(),
        'avg_pnl': pnls.mean(),
        'median_pnl': np.median(pnls),
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_days': win_days,
        'total_days': len(daily_pnls),
        'win_day_rate': win_days / len(daily_pnls) * 100,
        'profit_factor': pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else float('inf'),
    }


def run():
    trades = load_trades()
    client = OptionsClient()

    # ── Collect intraday bars for all trades ─────────────────────────────────
    print("\nFetching intraday option bars for all trades...")
    trade_bars = {}  # index -> DataFrame of 5m bars
    baseline_pnls = []
    baseline_dates = []

    for idx, row in trades.iterrows():
        option_ticker = row['option_ticker']
        trade_date = row['trade_date'].date() if hasattr(row['trade_date'], 'date') else row['trade_date']
        entry_time = row['entry_time']
        exit_time = row['exit_time']

        bars = get_intraday_option_bars(client, option_ticker, trade_date, entry_time, exit_time)
        trade_bars[idx] = bars
        baseline_pnls.append(row['pnl'])
        baseline_dates.append(trade_date)

        if (len(trade_bars)) % 100 == 0:
            print(f"  Fetched bars for {len(trade_bars)}/{len(trades)} trades...")

    print(f"  Done. {sum(1 for b in trade_bars.values() if not b.empty)}/{len(trades)} trades have intraday bars.")

    # ── Baseline stats ───────────────────────────────────────────────────────
    baseline = compute_stats(baseline_pnls, baseline_dates)
    print(f"\nBASELINE: {baseline['trades']} trades, WR {baseline['win_rate']:.1f}%, "
          f"P&L ${baseline['total_pnl']:,.0f}, Sharpe {baseline['sharpe']:.2f}, "
          f"Win Days {baseline['win_day_rate']:.1f}%")

    results = []
    results.append({
        'method': 'BASELINE (no target)',
        'param': '-',
        **baseline,
    })

    # ── Test Profit Targets ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TESTING PROFIT TARGETS")
    print("=" * 70)

    for pt_pct in PROFIT_TARGETS:
        pnls = []
        dates = []
        n_hit = 0

        for idx, row in trades.iterrows():
            bars = trade_bars[idx]
            entry_price = row['option_entry_price']
            num_contracts = row['num_contracts']
            premium_paid = row['premium_paid']
            trade_date = row['trade_date'].date() if hasattr(row['trade_date'], 'date') else row['trade_date']

            if bars.empty or len(bars) < 2:
                # No intraday data — use original exit
                pnls.append(row['pnl'])
                dates.append(trade_date)
                continue

            exit_px, exit_ts, hit = simulate_profit_target(
                bars, entry_price, pt_pct, row['direction']
            )

            if hit:
                n_hit += 1
                pnl = compute_trade_pnl(entry_price, exit_px, num_contracts, premium_paid)
                pnls.append(pnl)
            else:
                # Target not hit — use original exit
                pnls.append(row['pnl'])

            dates.append(trade_date)

        stats = compute_stats(pnls, dates)
        hit_rate = n_hit / len(trades) * 100
        label = f"PT +{pt_pct:.0%}"
        print(f"  {label:<16} WR {stats['win_rate']:5.1f}%  P&L ${stats['total_pnl']:>12,.0f}  "
              f"Sharpe {stats['sharpe']:5.2f}  WinDays {stats['win_day_rate']:5.1f}%  "
              f"Hit {hit_rate:5.1f}%  PF {stats['profit_factor']:.2f}")

        results.append({
            'method': f'Profit Target',
            'param': f'+{pt_pct:.0%}',
            'hit_rate': hit_rate,
            **stats,
        })

    # ── Test Trailing Stops ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TESTING TRAILING STOPS")
    print("=" * 70)

    for act_pct, trail_pct in TRAILING_STOPS:
        pnls = []
        dates = []
        n_activated = 0
        n_stopped = 0

        for idx, row in trades.iterrows():
            bars = trade_bars[idx]
            entry_price = row['option_entry_price']
            num_contracts = row['num_contracts']
            premium_paid = row['premium_paid']
            trade_date = row['trade_date'].date() if hasattr(row['trade_date'], 'date') else row['trade_date']

            if bars.empty or len(bars) < 2:
                pnls.append(row['pnl'])
                dates.append(trade_date)
                continue

            exit_px, exit_ts, activated = simulate_trailing_stop(
                bars, entry_price, act_pct, trail_pct, row['direction']
            )

            if exit_px is not None:
                n_stopped += 1
                pnl = compute_trade_pnl(entry_price, exit_px, num_contracts, premium_paid)
                pnls.append(pnl)
            else:
                if activated:
                    n_activated += 1
                pnls.append(row['pnl'])

            dates.append(trade_date)

        stats = compute_stats(pnls, dates)
        label = f"Trail +{act_pct:.0%}/{trail_pct:.0%}"
        print(f"  {label:<20} WR {stats['win_rate']:5.1f}%  P&L ${stats['total_pnl']:>12,.0f}  "
              f"Sharpe {stats['sharpe']:5.2f}  WinDays {stats['win_day_rate']:5.1f}%  "
              f"Stopped {n_stopped}  PF {stats['profit_factor']:.2f}")

        results.append({
            'method': f'Trailing Stop',
            'param': f'+{act_pct:.0%}/trail {trail_pct:.0%}',
            'n_stopped': n_stopped,
            **stats,
        })

    # ── Test Combined ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TESTING COMBINED (PROFIT TARGET + TRAILING STOP)")
    print("=" * 70)

    for pt_pct, trail_act, trail_pct in COMBINED:
        pnls = []
        dates = []
        n_pt = 0
        n_trail = 0

        for idx, row in trades.iterrows():
            bars = trade_bars[idx]
            entry_price = row['option_entry_price']
            num_contracts = row['num_contracts']
            premium_paid = row['premium_paid']
            trade_date = row['trade_date'].date() if hasattr(row['trade_date'], 'date') else row['trade_date']

            if bars.empty or len(bars) < 2:
                pnls.append(row['pnl'])
                dates.append(trade_date)
                continue

            exit_px, exit_ts, reason = simulate_combined(
                bars, entry_price, pt_pct, trail_act, trail_pct, row['direction']
            )

            if exit_px is not None:
                if reason == "profit_target":
                    n_pt += 1
                else:
                    n_trail += 1
                pnl = compute_trade_pnl(entry_price, exit_px, num_contracts, premium_paid)
                pnls.append(pnl)
            else:
                pnls.append(row['pnl'])

            dates.append(trade_date)

        stats = compute_stats(pnls, dates)
        label = f"PT+{pt_pct:.0%} Trail+{trail_act:.0%}/{trail_pct:.0%}"
        print(f"  {label:<30} WR {stats['win_rate']:5.1f}%  P&L ${stats['total_pnl']:>12,.0f}  "
              f"Sharpe {stats['sharpe']:5.2f}  WinDays {stats['win_day_rate']:5.1f}%  "
              f"PT {n_pt} Trail {n_trail}  PF {stats['profit_factor']:.2f}")

        results.append({
            'method': f'Combined',
            'param': f'PT+{pt_pct:.0%} Trail+{trail_act:.0%}/{trail_pct:.0%}',
            'n_pt_hit': n_pt,
            'n_trail_hit': n_trail,
            **stats,
        })

    # ── Per-strategy breakdown for best profit target ────────────────────────
    print("\n" + "=" * 70)
    print("PER-STRATEGY BREAKDOWN: PROFIT TARGET +50% vs BASELINE")
    print("=" * 70)

    pt_pct = 0.50
    for rule in sorted(trades['rule'].unique()):
        rule_trades = trades[trades['rule'] == rule]
        base_pnls = []
        pt_pnls = []

        for idx, row in rule_trades.iterrows():
            bars = trade_bars[idx]
            entry_price = row['option_entry_price']
            num_contracts = row['num_contracts']
            premium_paid = row['premium_paid']

            base_pnls.append(row['pnl'])

            if bars.empty or len(bars) < 2:
                pt_pnls.append(row['pnl'])
                continue

            exit_px, exit_ts, hit = simulate_profit_target(bars, entry_price, pt_pct, row['direction'])
            if hit:
                pnl = compute_trade_pnl(entry_price, exit_px, num_contracts, premium_paid)
                pt_pnls.append(pnl)
            else:
                pt_pnls.append(row['pnl'])

        base_wr = (np.array(base_pnls) > 0).sum() / len(base_pnls) * 100
        pt_wr = (np.array(pt_pnls) > 0).sum() / len(pt_pnls) * 100
        base_total = sum(base_pnls)
        pt_total = sum(pt_pnls)

        name = rule.replace('PriorDay', 'PD').replace('_filtered', '').replace('_30min', '').replace('_First30min', '')
        print(f"  {name:<35} Base WR {base_wr:5.1f}% → PT WR {pt_wr:5.1f}% "
              f"({pt_wr - base_wr:+.1f}%)  P&L ${base_total:>10,.0f} → ${pt_total:>10,.0f} "
              f"({pt_total - base_total:+,.0f})")

    # ── Save results ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "profit_target_optimization.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Sorted by Win Rate (descending)")
    print("=" * 70)
    print(f"  {'Method':<40} {'WR':>6} {'WinDay':>7} {'P&L':>13} {'Sharpe':>7} {'PF':>6}")
    print(f"  {'-'*80}")
    for r in sorted(results, key=lambda x: x.get('win_rate', 0), reverse=True):
        label = f"{r['method']} {r.get('param', '')}"
        print(f"  {label:<40} {r['win_rate']:5.1f}% {r['win_day_rate']:5.1f}%  "
              f"${r['total_pnl']:>11,.0f} {r['sharpe']:6.2f}  {r['profit_factor']:.2f}")


if __name__ == "__main__":
    run()
