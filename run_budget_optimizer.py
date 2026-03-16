#!/usr/bin/env python3
"""
Per-Strategy Budget Optimizer
==============================
Tests different premium budgets for each strategy independently,
then finds the combination that maximizes risk-adjusted P&L.

Currently:
  - D0.10 strategies (HighVolWR): $10K
  - D0.50 strategies (GapLarge, QQQ Weak, SPY Weak, 50Hi Weak): $50K
  - D0.70 strategies (QQQ Short, SPY Short): $75K

Tests budgets from $10K to $150K per strategy.
Since P&L scales linearly with budget (same option, more contracts),
we just scale the existing P&L by the budget ratio.

Also tests a few specific proposed allocations.

Usage:
    python run_budget_optimizer.py
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product

from src.config import OUTPUT_DIR


SHORT_NAMES = {
    'GapLarge_First30min_SPY': 'GapLarge',
    'HighVolWR_30min_SPY_filtered': 'HighVolWR',
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
    'PriorDayWeak_30min_QQQ': 'QQQ Weak',
    'PriorDayWeak_30min_SPY_filtered': 'SPY Weak',
    'PriorDayWeak_50Hi_SPY_filtered': '50Hi Weak',
}

CURRENT_BUDGETS = {
    'GapLarge_First30min_SPY': 50_000,
    'HighVolWR_30min_SPY_filtered': 10_000,
    'PriorDayStrong_AboveOR_QQQ_short': 75_000,
    'PriorDayStrong_AboveOR_SPY_short': 75_000,
    'PriorDayWeak_30min_QQQ': 50_000,
    'PriorDayWeak_30min_SPY_filtered': 50_000,
    'PriorDayWeak_50Hi_SPY_filtered': 50_000,
}


def compute_portfolio_stats(df, budget_map):
    """Given per-rule budgets, scale each trade's P&L and compute portfolio stats."""
    scaled_pnls = []
    scaled_prems = []

    for _, trade in df.iterrows():
        rule = trade['rule']
        current_budget = CURRENT_BUDGETS[rule]
        new_budget = budget_map.get(rule, current_budget)
        scale = new_budget / current_budget

        scaled_pnls.append(trade['pnl'] * scale)
        scaled_prems.append(trade['premium_paid'] * scale)

    pnl_series = pd.Series(scaled_pnls)
    total_pnl = pnl_series.sum()
    total_prem = sum(scaled_prems)
    n = len(pnl_series)
    wr = (pnl_series > 0).mean() * 100
    avg = pnl_series.mean()
    std = pnl_series.std()
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0

    # Max drawdown
    cum = pnl_series.cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()

    # Daily aggregation for daily sharpe
    daily_pnl = []
    date_groups = df.groupby('trade_date')
    for date, group in date_groups:
        day_pnl = 0
        for _, trade in group.iterrows():
            rule = trade['rule']
            scale = budget_map.get(rule, CURRENT_BUDGETS[rule]) / CURRENT_BUDGETS[rule]
            day_pnl += trade['pnl'] * scale
        daily_pnl.append(day_pnl)

    daily_series = pd.Series(daily_pnl)
    daily_sharpe = daily_series.mean() / daily_series.std() * np.sqrt(252) if daily_series.std() > 0 else 0

    return {
        'pnl': total_pnl,
        'premium': total_prem,
        'n': n,
        'wr': wr,
        'avg': avg,
        'sharpe': sharpe,
        'daily_sharpe': daily_sharpe,
        'max_dd': dd,
        'roi': total_pnl / total_prem * 100 if total_prem > 0 else 0,
    }


def run():
    input_path = os.path.join(OUTPUT_DIR, "options_regime_filtered.csv")
    print(f"Loading trades from {input_path}")
    df = pd.read_csv(input_path)
    df = df[df['status'] == 'ok'].copy()
    df['entry_dt'] = pd.to_datetime(df['entry_time'], utc=True)
    df['trade_date'] = df['entry_dt'].dt.date
    df['year'] = df['entry_dt'].dt.year
    df = df.sort_values('entry_dt').reset_index(drop=True)
    print(f"  {len(df)} trades loaded")

    rules = sorted(df['rule'].unique())

    # ── Current baseline ────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("CURRENT SIZING (baseline):")
    print(f"{'='*110}")
    baseline = compute_portfolio_stats(df, CURRENT_BUDGETS)
    print(f"  P&L: ${baseline['pnl']:>+12,.0f}  Sharpe: {baseline['sharpe']:.2f}  Daily Sharpe: {baseline['daily_sharpe']:.2f}")
    print(f"  MaxDD: ${baseline['max_dd']:>+10,.0f}  WR: {baseline['wr']:.1f}%  ROI: {baseline['roi']:.1f}%")
    for rule in rules:
        sn = SHORT_NAMES.get(rule, rule[:12])
        print(f"    {sn:<16} ${CURRENT_BUDGETS[rule]:>7,}")

    # ── Named proposals ─────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("PROPOSED ALLOCATIONS:")
    print(f"{'='*110}")

    proposals = {
        'Current': CURRENT_BUDGETS.copy(),

        'Bump HighVolWR 25K': {**CURRENT_BUDGETS, 'HighVolWR_30min_SPY_filtered': 25_000},
        'Bump HighVolWR 50K': {**CURRENT_BUDGETS, 'HighVolWR_30min_SPY_filtered': 50_000},
        'Bump HighVolWR 75K': {**CURRENT_BUDGETS, 'HighVolWR_30min_SPY_filtered': 75_000},

        'ROI-weighted': {
            # Scale budgets proportional to ROI: HighVolWR (128%) gets most, shorts (7%) get least
            'GapLarge_First30min_SPY': 50_000,         # 29% ROI, keep same
            'HighVolWR_30min_SPY_filtered': 75_000,    # 128% ROI, big bump
            'PriorDayStrong_AboveOR_QQQ_short': 50_000,  # 7% ROI, trim
            'PriorDayStrong_AboveOR_SPY_short': 50_000,  # 6% ROI, trim
            'PriorDayWeak_30min_QQQ': 25_000,          # 10% ROI, lowest, trim
            'PriorDayWeak_30min_SPY_filtered': 50_000, # 30% ROI, keep
            'PriorDayWeak_50Hi_SPY_filtered': 50_000,  # 30% ROI, keep
        },

        'Uniform 50K': {r: 50_000 for r in rules},
        'Uniform 75K': {r: 75_000 for r in rules},

        'Aggressive HighVolWR': {
            **CURRENT_BUDGETS,
            'HighVolWR_30min_SPY_filtered': 100_000,
        },

        'Boost high-ROI': {
            'GapLarge_First30min_SPY': 75_000,         # 29% ROI, bump
            'HighVolWR_30min_SPY_filtered': 75_000,    # 128% ROI, big bump
            'PriorDayStrong_AboveOR_QQQ_short': 75_000,  # keep
            'PriorDayStrong_AboveOR_SPY_short': 75_000,  # keep
            'PriorDayWeak_30min_QQQ': 50_000,          # keep
            'PriorDayWeak_30min_SPY_filtered': 75_000, # 30% ROI, bump
            'PriorDayWeak_50Hi_SPY_filtered': 75_000,  # 30% ROI, bump
        },

        'Conservative': {
            'GapLarge_First30min_SPY': 25_000,
            'HighVolWR_30min_SPY_filtered': 50_000,
            'PriorDayStrong_AboveOR_QQQ_short': 50_000,
            'PriorDayStrong_AboveOR_SPY_short': 50_000,
            'PriorDayWeak_30min_QQQ': 25_000,
            'PriorDayWeak_30min_SPY_filtered': 25_000,
            'PriorDayWeak_50Hi_SPY_filtered': 25_000,
        },
    }

    results = []
    header = "{:<25} {:>12} {:>12} {:>8} {:>8} {:>10} {:>6} {:>8}".format(
        'Allocation', 'P&L', 'vs Current', 'Sharpe', 'DlyShrp', 'MaxDD', 'WR', 'ROI')
    print(f"\n{header}")
    print('-' * 100)

    for name, budgets in proposals.items():
        stats = compute_portfolio_stats(df, budgets)
        delta = stats['pnl'] - baseline['pnl']
        row = "{:<25} ${:>+10,.0f} ${:>+10,.0f} {:>7.2f} {:>7.2f} ${:>+8,.0f} {:>5.1f}% {:>+6.1f}%".format(
            name, stats['pnl'], delta, stats['sharpe'], stats['daily_sharpe'],
            stats['max_dd'], stats['wr'], stats['roi'])
        print(row)
        results.append({'name': name, **stats, 'delta': delta})

    # ── Per-strategy breakdown for best proposal ────────────────────────────
    print(f"\n{'='*110}")
    print("PER-STRATEGY BREAKDOWN: Boost high-ROI vs Current")
    print(f"{'='*110}")

    for label, budgets in [('Current', CURRENT_BUDGETS), ('Boost high-ROI', proposals['Boost high-ROI'])]:
        print(f"\n  {label}:")
        total = 0
        for rule in rules:
            sn = SHORT_NAMES.get(rule, rule[:12])
            sub = df[df['rule'] == rule]
            scale = budgets[rule] / CURRENT_BUDGETS[rule]
            pnl = sub['pnl'].sum() * scale
            total += pnl
            cum = (sub['pnl'] * scale).cumsum()
            peak = cum.cummax()
            dd = (cum - peak).min()
            print("    {:<16} ${:>7,} budget  P&L ${:>+10,.0f}  MaxDD ${:>+8,.0f}".format(
                sn, budgets[rule], pnl, dd))
        print("    {:<16} {:>15} P&L ${:>+10,.0f}".format('TOTAL', '', total))

    # ── Year-by-year for top proposals ──────────────────────────────────────
    print(f"\n{'='*110}")
    print("YEAR-BY-YEAR COMPARISON:")
    print(f"{'='*110}")

    top_proposals = ['Current', 'Bump HighVolWR 50K', 'ROI-weighted', 'Boost high-ROI']
    years = sorted(df['year'].unique())

    header = "{:<25}".format("Allocation")
    for yr in years:
        header += " {:>10}".format(yr)
    print(header)
    print('-' * 100)

    for name in top_proposals:
        budgets = proposals[name]
        line = "{:<25}".format(name)
        for yr in years:
            yr_df = df[df['year'] == yr]
            yr_pnl = 0
            for _, trade in yr_df.iterrows():
                rule = trade['rule']
                scale = budgets[rule] / CURRENT_BUDGETS[rule]
                yr_pnl += trade['pnl'] * scale
            line += " ${:>+8,.0f}".format(yr_pnl)
        print(line)

    # Save results
    results_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "budget_optimization.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()
