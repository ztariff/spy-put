#!/usr/bin/env python3
"""
Combined Backtest: EqLoss $20K Sizing + PT 50% Exit
=====================================================
Re-runs the full portfolio with:
  1. EqLoss $20K sizing: each strategy sized so avg loss ≈ $20K
  2. +50% profit target: exit when option gains +50%, otherwise hold to normal exit
  3. 1.51x scale on top (to recover baseline P&L level)

This reads the regime-filtered trades and 5-min option bars from cache.
Outputs a new CSV with all trades re-priced and full stats.

Usage:
    python run_combined_backtest.py
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
PROFIT_TARGET_PCT = 0.50  # Exit at +50% on the option
SLIPPAGE_PCT = 0.01       # 1% per side

# Budgets as they exist in the CSV (what the backtest originally used)
CSV_BUDGETS = {
    'GapLarge_First30min_SPY': 50000,
    'HighVolWR_30min_SPY_filtered': 10000,
    'PriorDayStrong_AboveOR_QQQ_short': 75000,
    'PriorDayStrong_AboveOR_SPY_short': 75000,
    'PriorDayWeak_30min_QQQ': 50000,
    'PriorDayWeak_30min_SPY_filtered': 50000,
    'PriorDayWeak_50Hi_SPY_filtered': 50000,
}

# New EqLoss $20K budgets (computed from avg loss analysis)
# Each strategy sized so that avg loss per losing trade ≈ $20K
EQLOSS_BUDGETS = {
    'GapLarge_First30min_SPY': 32000,
    'HighVolWR_30min_SPY_filtered': 35000,
    'PriorDayStrong_AboveOR_QQQ_short': 104000,
    'PriorDayStrong_AboveOR_SPY_short': 99000,
    'PriorDayWeak_30min_QQQ': 26000,
    'PriorDayWeak_30min_SPY_filtered': 29000,
    'PriorDayWeak_50Hi_SPY_filtered': 48000,
}

NAME_MAP = {
    'GapLarge_First30min_SPY': 'GapLarge',
    'HighVolWR_30min_SPY_filtered': 'HighVolWR',
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
    'PriorDayWeak_30min_QQQ': 'QQQ Weak',
    'PriorDayWeak_30min_SPY_filtered': 'SPY Weak',
    'PriorDayWeak_50Hi_SPY_filtered': '50Hi Weak',
}


def load_trades():
    df = pd.read_csv(INPUT_CSV)
    df = df[df['status'] == 'ok'].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    print(f"Loaded {len(df)} trades from {INPUT_CSV}")
    return df


def get_intraday_option_bars(client, option_ticker, trade_date, entry_time, exit_time):
    try:
        bars = client.get_options_bars(option_ticker, str(trade_date), str(trade_date),
                                       multiplier=5, timespan="minute")
    except Exception:
        return pd.DataFrame()
    if bars is None or bars.empty:
        return pd.DataFrame()
    bars = bars[(bars.index >= entry_time) & (bars.index <= exit_time)]
    return bars


def simulate_profit_target(bars, entry_price, pt_pct):
    """Check if option reaches +pt_pct% at any point during the trade."""
    target_price = entry_price * (1 + pt_pct)
    for ts, row in bars.iterrows():
        if row['high'] >= target_price:
            return target_price, ts, True
        if row['close'] >= target_price:
            return row['close'], ts, True
    return None, None, False


def compute_pnl(entry_price, exit_price, num_contracts, premium_paid):
    entry_adj = entry_price * (1 + SLIPPAGE_PCT)
    exit_adj = exit_price * (1 - SLIPPAGE_PCT)
    exit_adj = max(exit_adj, 0)
    pnl = (exit_adj - entry_adj) * num_contracts * 100
    pnl = max(pnl, -premium_paid)
    return pnl


def compute_stats(pnls, dates, label=""):
    pnls = np.array(pnls)
    n = len(pnls)
    wins = (pnls > 0).sum()
    wr = wins / n * 100

    daily = defaultdict(float)
    for p, d in zip(pnls, dates):
        daily[d] += p
    daily_arr = np.array([daily[d] for d in sorted(daily.keys())])

    sharpe = daily_arr.mean() / daily_arr.std() * np.sqrt(252) if daily_arr.std() > 0 else 0
    cum = np.cumsum(daily_arr)
    peak = np.maximum.accumulate(cum)
    max_dd = (cum - peak).min()
    win_days = (daily_arr > 0).sum()
    worst_day = daily_arr.min()
    best_day = daily_arr.max()
    avg_day = daily_arr.mean()

    losers = pnls[pnls < 0]
    avg_loss = losers.mean() if len(losers) > 0 else 0
    max_loss = pnls.min()
    gross_win = pnls[pnls > 0].sum()
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    years = len(daily_arr) / 252
    calmar = (pnls.sum() / years) / abs(max_dd) if max_dd != 0 else float('inf')

    return {
        'trades': n, 'win_rate': wr, 'total_pnl': pnls.sum(),
        'avg_pnl': pnls.mean(), 'median_pnl': np.median(pnls),
        'sharpe': sharpe, 'max_dd': max_dd, 'worst_day': worst_day,
        'best_day': best_day, 'avg_day': avg_day,
        'win_days': win_days, 'total_days': len(daily_arr),
        'win_day_rate': win_days / len(daily_arr) * 100,
        'profit_factor': pf, 'avg_loss': avg_loss, 'max_loss': max_loss,
        'calmar': calmar,
    }


def run():
    trades = load_trades()
    client = OptionsClient()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Fetch intraday option bars for all trades (from cache)
    # ══════════════════════════════════════════════════════════════════════════
    print("\nStep 1: Fetching intraday option bars...")
    trade_bars = {}

    for idx, row in trades.iterrows():
        bars = get_intraday_option_bars(
            client, row['option_ticker'],
            row['trade_date'].date() if hasattr(row['trade_date'], 'date') else row['trade_date'],
            row['entry_time'], row['exit_time']
        )
        trade_bars[idx] = bars
        if len(trade_bars) % 200 == 0:
            print(f"  {len(trade_bars)}/{len(trades)} done...")

    bars_found = sum(1 for b in trade_bars.values() if not b.empty)
    print(f"  Done. {bars_found}/{len(trades)} trades have intraday bars.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Simulate 4 portfolio variants
    # ══════════════════════════════════════════════════════════════════════════

    variants = {
        'A) Original CSV sizing, no PT':       {'budgets': CSV_BUDGETS,    'pt': False, 'scale': 1.0},
        'B) EqLoss $20K sizing, no PT':        {'budgets': EQLOSS_BUDGETS, 'pt': False, 'scale': 1.0},
        'C) Original sizing + PT 50%':         {'budgets': CSV_BUDGETS,    'pt': True,  'scale': 1.0},
        'D) EqLoss $20K + PT 50%':             {'budgets': EQLOSS_BUDGETS, 'pt': True,  'scale': 1.0},
    }

    all_results = {}
    all_trade_details = {}  # For saving CSV

    for vname, vcfg in variants.items():
        print(f"\n{'='*70}")
        print(f"  {vname}")
        print(f"{'='*70}")

        budgets = vcfg['budgets']
        use_pt = vcfg['pt']
        pnls = []
        dates = []
        trade_rows = []
        n_pt_hit = 0
        per_rule_pnls = defaultdict(list)
        per_rule_losses = defaultdict(list)

        for idx, row in trades.iterrows():
            rule = row['rule']
            csv_budget = CSV_BUDGETS[rule]
            new_budget = budgets[rule]
            budget_scale = new_budget / csv_budget

            entry_price = row['option_entry_price']
            num_contracts = int(row['num_contracts'] * budget_scale)
            premium_paid = row['premium_paid'] * budget_scale

            # Determine exit price
            bars = trade_bars[idx]
            pt_hit = False
            pt_exit_price = None
            pt_exit_time = None

            if use_pt and not bars.empty and len(bars) >= 2:
                pt_exit_price, pt_exit_time, pt_hit = simulate_profit_target(
                    bars, entry_price, PROFIT_TARGET_PCT
                )

            if pt_hit:
                n_pt_hit += 1
                pnl = compute_pnl(entry_price, pt_exit_price, num_contracts, premium_paid)
                exit_reason = 'profit_target_50'
                actual_exit_price = pt_exit_price
                actual_exit_time = str(pt_exit_time)
            else:
                # Use original exit, scaled by budget
                pnl = row['pnl'] * budget_scale
                exit_reason = row['exit_reason']
                actual_exit_price = row['option_exit_price']
                actual_exit_time = str(row['exit_time'])

            pnls.append(pnl)
            td = row['trade_date']
            dates.append(td)
            per_rule_pnls[rule].append(pnl)
            if pnl < 0:
                per_rule_losses[rule].append(pnl)

            trade_rows.append({
                'rule': rule,
                'ticker': row['ticker'],
                'direction': row['direction'],
                'trade_date': str(td.date()) if hasattr(td, 'date') else str(td),
                'entry_time': str(row['entry_time']),
                'exit_time': actual_exit_time,
                'option_ticker': row['option_ticker'],
                'strike': row['strike'],
                'expiry_date': row['expiry_date'],
                'option_type': row['option_type'],
                'target_delta': row['target_delta'],
                'option_entry_price': entry_price,
                'option_exit_price': actual_exit_price,
                'num_contracts': num_contracts,
                'premium_paid': round(premium_paid, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl / premium_paid, 4) if premium_paid > 0 else 0,
                'exit_reason': exit_reason,
                'budget': new_budget,
            })

        stats = compute_stats(pnls, dates, vname)
        all_results[vname] = stats
        all_trade_details[vname] = trade_rows

        if use_pt:
            print(f"  Profit target hit: {n_pt_hit}/{len(trades)} ({n_pt_hit/len(trades)*100:.1f}%)")

        # Per-strategy summary
        print(f"\n  {'Strategy':<14} {'Budget':>8} {'N':>5} {'WR%':>6} {'AvgPnL':>9} {'TotPnL':>11} {'AvgLoss':>9} {'NLoss':>5}")
        print(f"  {'-'*75}")
        for rule in sorted(trades['rule'].unique()):
            rpnls = np.array(per_rule_pnls[rule])
            rlosses = np.array(per_rule_losses[rule]) if per_rule_losses[rule] else np.array([0])
            n = len(rpnls)
            wr = (rpnls > 0).sum() / n * 100
            avg = rpnls.mean()
            total = rpnls.sum()
            avg_l = rlosses.mean()
            n_l = len(per_rule_losses[rule])
            budget = budgets[rule]
            name = NAME_MAP[rule]
            print(f"  {name:<14} {budget:>8,} {n:>5} {wr:>5.1f}% {avg:>9,.0f} {total:>11,.0f} {avg_l:>9,.0f} {n_l:>5}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Also test with scale-up to match original P&L
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("STEP 3: SCALE-UP ANALYSIS")
    print("If we scale D) up to match A)'s P&L, what happens?")
    print("=" * 80)

    base_pnl = all_results["A) Original CSV sizing, no PT"]['total_pnl']
    d_pnl = all_results["D) EqLoss $20K + PT 50%"]['total_pnl']
    d_sharpe = all_results["D) EqLoss $20K + PT 50%"]['sharpe']
    d_dd = all_results["D) EqLoss $20K + PT 50%"]['max_dd']
    d_wr = all_results["D) EqLoss $20K + PT 50%"]['win_rate']
    d_wd = all_results["D) EqLoss $20K + PT 50%"]['win_day_rate']
    d_worst = all_results["D) EqLoss $20K + PT 50%"]['worst_day']

    if d_pnl > 0:
        scale_to_match = base_pnl / d_pnl
        print(f"\n  D) P&L: ${d_pnl:,.0f}")
        print(f"  A) P&L: ${base_pnl:,.0f}")
        print(f"  Scale needed: {scale_to_match:.2f}x")
        print(f"\n  Scaled D) to match A):")
        print(f"    P&L:      ${base_pnl:,.0f}")
        print(f"    Sharpe:   {d_sharpe:.2f} (unchanged — scale-invariant)")
        print(f"    MaxDD:    ${d_dd * scale_to_match:,.0f}")
        print(f"    WorstDay: ${d_worst * scale_to_match:,.0f}")
        print(f"    WinRate:  {d_wr:.1f}%")
        print(f"    WinDay:   {d_wd:.1f}%")

        # Scaled budgets
        print(f"\n  Scaled Budgets (EqLoss $20K × {scale_to_match:.2f}x):")
        for rule in sorted(EQLOSS_BUDGETS.keys()):
            name = NAME_MAP[rule]
            new_b = round(EQLOSS_BUDGETS[rule] * scale_to_match / 1000) * 1000
            print(f"    {name:<14} ${new_b:>8,}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Head-to-head comparison table
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 140)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 140)
    print(f"{'Variant':<35} {'P&L':>12} {'Sharpe':>7} {'MaxDD':>11} {'WorstDay':>10} {'WR%':>6} {'WinDay%':>8} {'AvgLoss':>10} {'PF':>6} {'Calmar':>7}")
    print("-" * 120)

    for vname, stats in all_results.items():
        print(f"{vname:<35} ${stats['total_pnl']:>11,.0f} {stats['sharpe']:>7.2f} "
              f"${stats['max_dd']:>10,.0f} ${stats['worst_day']:>9,.0f} "
              f"{stats['win_rate']:>5.1f}% {stats['win_day_rate']:>6.1f}%  "
              f"${stats['avg_loss']:>9,.0f} {stats['profit_factor']:>5.2f} {stats['calmar']:>7.1f}")

    # Add scaled D variant
    if d_pnl > 0:
        s = scale_to_match
        print(f"{'D-scaled to match A P&L':<35} ${base_pnl:>11,.0f} {d_sharpe:>7.2f} "
              f"${d_dd*s:>10,.0f} ${d_worst*s:>9,.0f} "
              f"{d_wr:>5.1f}% {d_wd:>6.1f}%  "
              f"${all_results['D) EqLoss $20K + PT 50%']['avg_loss']*s:>9,.0f} "
              f"{all_results['D) EqLoss $20K + PT 50%']['profit_factor']:>5.2f} "
              f"{all_results['D) EqLoss $20K + PT 50%']['calmar']:>7.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Year-by-year comparison A vs D
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 100)
    print("YEAR-BY-YEAR: A) Original vs D) EqLoss $20K + PT 50%")
    print("=" * 100)

    for vname in ["A) Original CSV sizing, no PT", "D) EqLoss $20K + PT 50%"]:
        trows = all_trade_details[vname]
        tdf = pd.DataFrame(trows)
        tdf['trade_date'] = pd.to_datetime(tdf['trade_date'])
        tdf['year'] = tdf['trade_date'].dt.year

        print(f"\n  {vname}:")
        print(f"  {'Year':<8} {'P&L':>12} {'Trades':>7} {'WR%':>6}")
        print(f"  {'-'*38}")
        for yr in sorted(tdf['year'].unique()):
            ydf = tdf[tdf['year'] == yr]
            ypnl = ydf['pnl'].sum()
            yn = len(ydf)
            ywr = (ydf['pnl'] > 0).sum() / yn * 100
            print(f"  {yr:<8} ${ypnl:>11,.0f} {yn:>7} {ywr:>5.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: Save best variant trades to CSV
    # ══════════════════════════════════════════════════════════════════════════
    best_key = "D) EqLoss $20K + PT 50%"
    out_df = pd.DataFrame(all_trade_details[best_key])
    out_path = os.path.join(OUTPUT_DIR, "options_eqloss20k_pt50.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n\nSaved {len(out_df)} trades to {out_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    run()
