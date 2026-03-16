#!/usr/bin/env python3
"""
Dynamic Position Sizing via Rolling Kelly Criterion
=====================================================
Re-sizes each trade using a rolling half-Kelly fraction computed from
that strategy's past N trades (no look-ahead bias).

Two modes:
  1. "Oracle" Kelly  – uses full-sample stats per strategy (look-ahead, upper bound)
  2. "Rolling" Kelly – uses only prior trades to size (realistic, no look-ahead)

For each trade, the premium budget = base_budget × kelly_multiplier, where:
  kelly_multiplier = (strategy half-Kelly weight) / (average half-Kelly weight)

This keeps total capital deployed roughly constant but redistributes toward
strategies with stronger edges.

Minimum budget floor: 50% of base budget (never size below half)
Maximum budget cap:   200% of base budget (never size above double)

Usage:
    python run_dynamic_sizing.py
"""
import os
import sys
import math
import numpy as np
import pandas as pd
from collections import defaultdict

from src.config import OUTPUT_DIR, DATA_DIR

# ── Configuration ────────────────────────────────────────────────────────────

# Base premium budgets (same as current system)
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

# Rolling window: number of past trades to compute Kelly from
ROLLING_WINDOW = 60  # minimum trades before applying Kelly
MIN_TRADES_FOR_KELLY = 20  # absolute minimum to compute any Kelly

# Sizing bounds (as multipliers on base budget)
MIN_MULTIPLIER = 0.50  # never size below 50% of base
MAX_MULTIPLIER = 2.00  # never size above 200% of base

SLIPPAGE_PCT = 0.01  # same as backtest

SHORT_NAMES = {
    'GapLarge_First30min_SPY': 'GapLarge',
    'HighVolWR_30min_SPY_filtered': 'HighVolWR',
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
    'PriorDayWeak_30min_QQQ': 'QQQ Weak',
    'PriorDayWeak_30min_SPY_filtered': 'SPY Weak',
    'PriorDayWeak_50Hi_SPY_filtered': '50Hi Weak',
}


def compute_kelly(pnl_pcts):
    """Compute half-Kelly fraction from a series of return-on-premium percentages."""
    if len(pnl_pcts) < MIN_TRADES_FOR_KELLY:
        return None  # not enough data

    wins = pnl_pcts[pnl_pcts > 0]
    losses = pnl_pcts[pnl_pcts <= 0]

    if len(wins) == 0 or len(losses) == 0:
        return None

    p = len(wins) / len(pnl_pcts)  # win rate
    q = 1 - p
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())

    if avg_loss == 0:
        return None

    b = avg_win / avg_loss  # win/loss ratio
    kelly = (b * p - q) / b

    if kelly <= 0:
        return 0.0  # no edge, size at minimum

    return kelly / 2  # half-Kelly for safety


def run_dynamic_sizing():
    """Apply rolling Kelly-based dynamic sizing to the portfolio."""

    # Load regime-filtered trades (our current best)
    input_path = os.path.join(OUTPUT_DIR, "options_regime_filtered.csv")
    print(f"Loading trades from {input_path}")
    df = pd.read_csv(input_path)
    df = df[df['status'] == 'ok'].copy()
    df['entry_dt'] = pd.to_datetime(df['entry_time'], utc=True)
    df = df.sort_values('entry_dt').reset_index(drop=True)
    print(f"  {len(df)} trades loaded")

    rules = sorted(df['rule'].unique())

    # ── Mode 1: Oracle Kelly (full sample, look-ahead) ──────────────────────
    print(f"\n{'='*100}")
    print("MODE 1: ORACLE KELLY (full-sample, look-ahead — upper bound)")
    print(f"{'='*100}")

    oracle_kelly = {}
    for rule in rules:
        sub = df[df['rule'] == rule]
        hk = compute_kelly(sub['pnl_pct'])
        oracle_kelly[rule] = hk if hk is not None else 0.0
        sn = SHORT_NAMES.get(rule, rule[:12])
        print(f"  {sn:<16} half-Kelly: {oracle_kelly[rule]:.1%}")

    # Compute multipliers: normalize so average multiplier = 1.0
    avg_hk = np.mean([v for v in oracle_kelly.values() if v > 0]) or 0.01
    oracle_mult = {}
    for rule in rules:
        raw = oracle_kelly[rule] / avg_hk if avg_hk > 0 else 1.0
        oracle_mult[rule] = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, raw))

    print(f"\n  Average half-Kelly: {avg_hk:.1%}")
    print(f"  Multipliers (min {MIN_MULTIPLIER}x, max {MAX_MULTIPLIER}x):")
    for rule in rules:
        sn = SHORT_NAMES.get(rule, rule[:12])
        print(f"    {sn:<16} {oracle_mult[rule]:.2f}x")

    # Apply oracle sizing
    oracle_results = apply_sizing(df, oracle_mult, "oracle")

    # ── Mode 2: Rolling Kelly (realistic, no look-ahead) ────────────────────
    print(f"\n{'='*100}")
    print(f"MODE 2: ROLLING KELLY (past {ROLLING_WINDOW} trades, no look-ahead)")
    print(f"{'='*100}")

    rolling_results = apply_rolling_sizing(df)

    # ── Comparison ──────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("COMPARISON")
    print(f"{'='*100}")

    base_pnl = df['pnl'].sum()
    oracle_pnl = oracle_results['pnl'].sum()
    rolling_pnl = rolling_results['pnl'].sum()

    print(f"\n  {'Method':<30} {'P&L':>14} {'vs Base':>12} {'Avg/Trade':>12} {'WR':>7}")
    print(f"  {'-'*80}")

    base_wr = (df['pnl'] > 0).mean() * 100
    oracle_wr = (oracle_results['pnl'] > 0).mean() * 100
    rolling_wr = (rolling_results['pnl'] > 0).mean() * 100

    print(f"  {'Fixed (current)':<30} ${base_pnl:>+12,.0f} {'':>12} ${base_pnl/len(df):>+10,.0f} {base_wr:>6.1f}%")
    print(f"  {'Oracle Kelly (look-ahead)':<30} ${oracle_pnl:>+12,.0f} ${oracle_pnl-base_pnl:>+10,.0f} ${oracle_pnl/len(oracle_results):>+10,.0f} {oracle_wr:>6.1f}%")
    print(f"  {'Rolling Kelly (realistic)':<30} ${rolling_pnl:>+12,.0f} ${rolling_pnl-base_pnl:>+10,.0f} ${rolling_pnl/len(rolling_results):>+10,.0f} {rolling_wr:>6.1f}%")

    # By year
    print(f"\n  BY YEAR:")
    df['year'] = df['entry_dt'].dt.year
    oracle_results['year'] = pd.to_datetime(oracle_results['entry_time'], utc=True).dt.year
    rolling_results['year'] = pd.to_datetime(rolling_results['entry_time'], utc=True).dt.year

    print(f"  {'Year':<6} {'Fixed':>12} {'Oracle':>12} {'Rolling':>12} {'Roll Δ':>12}")
    for yr in sorted(df['year'].unique()):
        bp = df[df['year'] == yr]['pnl'].sum()
        op = oracle_results[oracle_results['year'] == yr]['pnl'].sum()
        rp = rolling_results[rolling_results['year'] == yr]['pnl'].sum()
        print(f"  {yr:<6} ${bp:>+10,.0f} ${op:>+10,.0f} ${rp:>+10,.0f} ${rp-bp:>+10,.0f}")

    # By strategy
    print(f"\n  BY STRATEGY (Rolling Kelly):")
    print(f"  {'Strategy':<16} {'Fixed P&L':>12} {'Rolling P&L':>12} {'Change':>12} {'Avg Mult':>10}")
    for rule in rules:
        sn = SHORT_NAMES.get(rule, rule[:12])
        bp = df[df['rule'] == rule]['pnl'].sum()
        rr = rolling_results[rolling_results['rule'] == rule]
        rp = rr['pnl'].sum()
        avg_m = rr['kelly_mult'].mean() if 'kelly_mult' in rr.columns else 1.0
        print(f"  {sn:<16} ${bp:>+10,.0f} ${rp:>+10,.0f} ${rp-bp:>+10,.0f} {avg_m:>9.2f}x")

    # Total premium comparison
    base_premium = df['premium_paid'].sum()
    rolling_premium = rolling_results['premium_paid'].sum()
    print(f"\n  Total premium deployed: ${base_premium:>,.0f} (fixed) vs ${rolling_premium:>,.0f} (rolling) = {rolling_premium/base_premium:.1%} of base")

    # Save rolling results
    out_path = os.path.join(OUTPUT_DIR, "options_dynamic_sized.csv")
    rolling_results.to_csv(out_path, index=False)
    print(f"\n  Rolling Kelly results saved to {out_path}")


def apply_sizing(df, multipliers, label):
    """Apply fixed multipliers to each trade and recompute P&L."""
    results = df.copy()

    new_pnl = []
    new_contracts = []
    new_premium = []

    for _, row in results.iterrows():
        rule = row['rule']
        delta = row['target_delta']
        entry_price = row['option_entry_price']
        exit_price = row['option_exit_price']
        direction = row['direction']

        # New budget
        base_budget = PREMIUM_BY_DELTA.get(delta, 50_000)
        mult = multipliers.get(rule, 1.0)
        budget = base_budget * mult

        # Contracts
        cost_per = entry_price * 100 * (1 + SLIPPAGE_PCT)
        contracts = max(1, int(budget / cost_per)) if cost_per > 0 else 1
        premium = contracts * entry_price * 100 * (1 + SLIPPAGE_PCT)

        # P&L
        if direction == 'long':
            exit_val = contracts * exit_price * 100 * (1 - SLIPPAGE_PCT)
        else:
            exit_val = contracts * exit_price * 100 * (1 + SLIPPAGE_PCT)
            premium, exit_val = exit_val, premium  # short: sold at entry, bought at exit

        pnl = exit_val - premium if direction == 'long' else premium - exit_val

        # Actually, let's just scale linearly from current P&L for simplicity
        # (avoids re-implementing the full slippage logic which could diverge)
        pass

    # Simpler approach: scale P&L proportionally to contract count change
    for idx, row in results.iterrows():
        rule = row['rule']
        delta = row['target_delta']
        entry_price = row['option_entry_price']

        base_budget = PREMIUM_BY_DELTA.get(delta, 50_000)
        mult = multipliers.get(rule, 1.0)
        new_budget = base_budget * mult

        # Old contracts
        old_contracts = row['num_contracts']

        # New contracts
        cost_per = entry_price * 100 * (1 + SLIPPAGE_PCT)
        new_c = max(1, int(new_budget / cost_per)) if cost_per > 0 else 1

        # Scale P&L by contract ratio
        if old_contracts > 0:
            scale = new_c / old_contracts
        else:
            scale = 1.0

        new_pnl.append(row['pnl'] * scale)
        new_contracts.append(new_c)
        new_premium.append(row['premium_paid'] * scale)

    results['pnl'] = new_pnl
    results['num_contracts'] = new_contracts
    results['premium_paid'] = new_premium

    return results


def apply_rolling_sizing(df):
    """Apply rolling Kelly sizing: for each trade, compute Kelly from past trades only."""
    results = df.copy()

    # Track past trades per rule
    past_returns = defaultdict(list)  # rule -> list of pnl_pct values

    new_pnl = []
    new_contracts = []
    new_premium = []
    kelly_mults = []

    for idx, row in results.iterrows():
        rule = row['rule']
        delta = row['target_delta']
        entry_price = row['option_entry_price']

        base_budget = PREMIUM_BY_DELTA.get(delta, 50_000)

        # Compute Kelly from past trades for this rule
        past = past_returns[rule]

        if len(past) >= MIN_TRADES_FOR_KELLY:
            # Use last ROLLING_WINDOW trades
            window = past[-ROLLING_WINDOW:] if len(past) > ROLLING_WINDOW else past
            hk = compute_kelly(pd.Series(window))
        else:
            hk = None  # not enough data, use base sizing

        # Compute multiplier
        # We need the average Kelly across all rules to normalize
        # For rolling, we compute each rule's current Kelly and normalize
        all_kellys = {}
        for r in results['rule'].unique():
            p = past_returns[r]
            if len(p) >= MIN_TRADES_FOR_KELLY:
                w = p[-ROLLING_WINDOW:] if len(p) > ROLLING_WINDOW else p
                all_kellys[r] = compute_kelly(pd.Series(w))
            else:
                all_kellys[r] = None

        valid_kellys = [v for v in all_kellys.values() if v is not None and v > 0]

        if hk is not None and len(valid_kellys) > 0:
            avg_hk = np.mean(valid_kellys)
            raw_mult = hk / avg_hk if avg_hk > 0 else 1.0
            mult = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, raw_mult))
        else:
            mult = 1.0  # default to base sizing until we have enough data

        new_budget = base_budget * mult

        # Compute new contract count
        old_contracts = row['num_contracts']
        cost_per = entry_price * 100 * (1 + SLIPPAGE_PCT)
        new_c = max(1, int(new_budget / cost_per)) if cost_per > 0 else 1

        # Scale P&L by contract ratio
        if old_contracts > 0:
            scale = new_c / old_contracts
        else:
            scale = 1.0

        new_pnl.append(row['pnl'] * scale)
        new_contracts.append(new_c)
        new_premium.append(row['premium_paid'] * scale)
        kelly_mults.append(mult)

        # Add this trade's return to history (AFTER computing sizing — no look-ahead)
        past_returns[rule].append(row['pnl_pct'])

    results['pnl'] = new_pnl
    results['num_contracts'] = new_contracts
    results['premium_paid'] = new_premium
    results['kelly_mult'] = kelly_mults

    return results


if __name__ == "__main__":
    run_dynamic_sizing()
