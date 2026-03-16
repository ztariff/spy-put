#!/usr/bin/env python3
"""
Apply Regime Filters to Options Portfolio
==========================================
Only filters the 3 long strategies (QQQ Weak, SPY Weak, 50Hi Weak) using
"Below SMA20" — meaning we only take long trades when SPY closed below
its 20-day simple moving average the prior day.

Short strategies and GapLarge/HighVolWR are left unfiltered.

Usage:
    python run_regime_filter.py
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

from src.config import OUTPUT_DIR, DATA_DIR


# ── Configuration ────────────────────────────────────────────────────────────

# Rules to filter (long strategies only)
FILTERED_RULES = {
    "PriorDayWeak_30min_QQQ":             "Below SMA20",
    "PriorDayWeak_30min_SPY_filtered":    "Below SMA20",
    "PriorDayWeak_50Hi_SPY_filtered":     "Below SMA20",
}

# Rules left unfiltered
UNFILTERED_RULES = [
    "GapLarge_First30min_SPY",
    "HighVolWR_30min_SPY_filtered",
    "PriorDayStrong_AboveOR_QQQ_short",
    "PriorDayStrong_AboveOR_SPY_short",
]

SHORT_NAMES = {
    'GapLarge_First30min_SPY': 'GapLarge',
    'HighVolWR_30min_SPY_filtered': 'HighVolWR',
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
    'PriorDayWeak_30min_QQQ': 'QQQ Weak',
    'PriorDayWeak_30min_SPY_filtered': 'SPY Weak',
    'PriorDayWeak_50Hi_SPY_filtered': '50Hi Weak',
}


def run_regime_filter():
    """Apply regime filters and output new portfolio."""

    # Load combined optimal trades
    trades_path = os.path.join(OUTPUT_DIR, "options_combined_optimal.csv")
    print(f"Loading trades from {trades_path}")
    trades = pd.read_csv(trades_path)
    ok = trades[trades['status'] == 'ok'].copy()
    ok['trade_date'] = pd.to_datetime(ok['entry_time'], utc=True).dt.date
    print(f"  {len(ok)} OK trades")

    # Load SPY daily bars for SMA20
    spy_path = os.path.join(DATA_DIR, "1D", "SPY.parquet")
    spy = pd.read_parquet(spy_path)
    print(f"  SPY daily bars: {len(spy)}")

    # Compute SMA20 and "below SMA20" flag
    if 'timestamp' in spy.columns:
        spy['date'] = pd.to_datetime(spy['timestamp']).dt.date
    elif hasattr(spy.index, 'date'):
        spy['date'] = pd.to_datetime(spy.index).date
    else:
        spy['date'] = pd.to_datetime(spy.index).date

    spy = spy.sort_values('date').reset_index(drop=True)
    spy['sma20'] = spy['close'].rolling(20).mean()
    spy['below_sma20'] = (spy['close'] < spy['sma20']).astype(int)

    # Shift by 1: use prior day's close vs SMA20 (known at today's open)
    spy['prev_below_sma20'] = spy['below_sma20'].shift(1)

    # Build lookup: date -> prev_below_sma20
    sma_lookup = dict(zip(spy['date'], spy['prev_below_sma20']))

    # Apply filters
    kept = []
    skipped = []

    for _, trade in ok.iterrows():
        rule = trade['rule']
        td = trade['trade_date']

        if rule in FILTERED_RULES:
            # Check regime
            below = sma_lookup.get(td, np.nan)
            if below == 1:
                kept.append(trade)
            else:
                skipped.append(trade)
        else:
            # Unfiltered — keep all
            kept.append(trade)

    kept_df = pd.DataFrame(kept)
    skipped_df = pd.DataFrame(skipped) if skipped else pd.DataFrame()

    print(f"\n  Kept:    {len(kept_df)} trades")
    print(f"  Skipped: {len(skipped_df)} trades (long trades in uptrend)")

    # ── Summary ──────────────────────────────────────────────────────────
    rules = sorted(ok['rule'].unique())

    print(f"\n{'='*130}")
    print(f"REGIME-FILTERED PORTFOLIO: Longs only when SPY below SMA20")
    print(f"{'='*130}")

    hdr = f"{'Strategy':<18} {'Filter':<16} {'Trades':>7} {'Old Trades':>10} {'P&L':>12} {'Old P&L':>12} {'WR':>6} {'Old WR':>7} {'Avg P&L':>10}"
    print(hdr)
    print("-" * 130)

    total_new_pnl = 0
    total_old_pnl = 0
    total_new_n = 0
    total_old_n = 0
    total_new_wins = 0
    total_old_wins = 0

    for rule in rules:
        sn = SHORT_NAMES.get(rule, rule[:12])
        old_trades = ok[ok['rule'] == rule]
        new_trades = kept_df[kept_df['rule'] == rule]

        old_pnl = old_trades['pnl'].sum()
        new_pnl = new_trades['pnl'].sum()
        old_n = len(old_trades)
        new_n = len(new_trades)
        old_wr = (old_trades['pnl'] > 0).mean() * 100
        new_wr = (new_trades['pnl'] > 0).mean() * 100 if new_n > 0 else 0
        new_avg = new_pnl / new_n if new_n > 0 else 0

        filt = FILTERED_RULES.get(rule, "None")

        total_new_pnl += new_pnl
        total_old_pnl += old_pnl
        total_new_n += new_n
        total_old_n += old_n
        total_new_wins += (new_trades['pnl'] > 0).sum()
        total_old_wins += (old_trades['pnl'] > 0).sum()

        changed = "  ***" if rule in FILTERED_RULES else ""
        print(f"  {sn:<18} {filt:<16} {new_n:>5} {old_n:>10} ${new_pnl:>+10,.0f} ${old_pnl:>+10,.0f} {new_wr:>5.0f}% {old_wr:>5.0f}%  ${new_avg:>+8,.0f}{changed}")

    print("-" * 130)
    new_wr_total = total_new_wins / total_new_n * 100 if total_new_n > 0 else 0
    old_wr_total = total_old_wins / total_old_n * 100 if total_old_n > 0 else 0
    new_avg_total = total_new_pnl / total_new_n if total_new_n > 0 else 0
    print(f"  {'TOTAL':<18} {'':16} {total_new_n:>5} {total_old_n:>10} ${total_new_pnl:>+10,.0f} ${total_old_pnl:>+10,.0f} {new_wr_total:>5.0f}% {old_wr_total:>5.0f}%  ${new_avg_total:>+8,.0f}")

    print(f"\n  Change:  ${total_new_pnl - total_old_pnl:>+11,.0f}  ({(total_new_pnl - total_old_pnl)/abs(total_old_pnl)*100:+.1f}%)")
    print(f"  Trades removed: {total_old_n - total_new_n} ({(total_old_n - total_new_n)/total_old_n*100:.0f}%)")
    print(f"  Avg P&L/trade:  ${new_avg_total:>+8,.0f}  (was ${total_old_pnl/total_old_n:>+8,.0f})")

    # By year
    kept_df2 = kept_df.copy()
    kept_df2['year'] = pd.to_datetime(kept_df2['entry_time'], utc=True).dt.year
    ok2 = ok.copy()
    ok2['year'] = pd.to_datetime(ok2['entry_time'], utc=True).dt.year

    print(f"\n  BY YEAR:")
    print(f"  {'Year':<6} {'New Trades':>10} {'Old Trades':>10} {'New P&L':>14} {'Old P&L':>14} {'Change':>14}")
    for yr in sorted(kept_df2['year'].unique()):
        new_yr = kept_df2[kept_df2['year'] == yr]
        old_yr = ok2[ok2['year'] == yr]
        n_pnl = new_yr['pnl'].sum()
        o_pnl = old_yr['pnl'].sum()
        print(f"  {yr:<6} {len(new_yr):>10} {len(old_yr):>10} ${n_pnl:>+12,.0f} ${o_pnl:>+12,.0f} ${n_pnl-o_pnl:>+12,.0f}")

    # By long vs short
    print(f"\n  LONG vs SHORT:")
    for d in ['long', 'short']:
        new_d = kept_df[kept_df['direction'] == d]
        old_d = ok[ok['direction'] == d]
        n_pnl = new_d['pnl'].sum()
        o_pnl = old_d['pnl'].sum()
        n_wr = (new_d['pnl'] > 0).mean() * 100 if len(new_d) > 0 else 0
        o_wr = (old_d['pnl'] > 0).mean() * 100
        print(f"    {d.upper():<8} {len(new_d):>5} trades  ${n_pnl:>+10,.0f}  WR {n_wr:.0f}%   (was {len(old_d)} trades  ${o_pnl:>+10,.0f}  WR {o_wr:.0f}%)")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "options_regime_filtered.csv")
    kept_df.to_csv(out_path, index=False)
    print(f"\nFiltered trades saved to {out_path}")


if __name__ == "__main__":
    run_regime_filter()
