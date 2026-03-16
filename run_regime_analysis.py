#!/usr/bin/env python3
"""
Regime Filter Analysis for Options Strategies
==============================================
Tests whether filtering trades by market regime improves performance.

Regime indicators (derived from SPY/QQQ daily bars):
  1. Realized Vol (20-day) — proxy for VIX
  2. SPY trend: above/below 20-day SMA
  3. SPY trend: above/below 50-day SMA
  4. Prior day return magnitude (big move vs small move)
  5. 5-day momentum (trending vs mean-reverting)
  6. 20-day high/low range (compression vs expansion)
  7. Consecutive up/down days

For each strategy, tests each regime filter to find which improve P&L.

Usage:
    python run_regime_analysis.py
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

from src.config import OUTPUT_DIR, DATA_DIR

# ── Configuration ────────────────────────────────────────────────────────────

SHORT_NAMES = {
    'GapLarge_First30min_SPY': 'GapLarge',
    'HighVolWR_30min_SPY_filtered': 'HighVolWR',
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
    'PriorDayWeak_30min_QQQ': 'QQQ Weak',
    'PriorDayWeak_30min_SPY_filtered': 'SPY Weak',
    'PriorDayWeak_50Hi_SPY_filtered': '50Hi Weak',
}


def load_daily_bars():
    """Load SPY and QQQ daily bars."""
    bars = {}
    for ticker in ['SPY', 'QQQ']:
        path = os.path.join(DATA_DIR, "1D", f"{ticker}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
            elif hasattr(df.index, 'date'):
                df['date'] = pd.to_datetime(df.index).date
            else:
                df['date'] = pd.to_datetime(df.index).date
            bars[ticker] = df
            print(f"  {ticker}: {len(df)} daily bars")
    return bars


def compute_regime_features(bars):
    """
    Compute regime indicators for each trading day.
    Returns DataFrame indexed by date with regime columns.
    """
    spy = bars.get('SPY')
    if spy is None:
        print("ERROR: No SPY daily bars!")
        return pd.DataFrame()

    df = spy[['date', 'close', 'high', 'low', 'volume']].copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Returns
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['ret_20d'] = df['close'].pct_change(20)

    # Realized volatility (annualized, 20-day rolling)
    df['rvol_10d'] = df['ret_1d'].rolling(10).std() * np.sqrt(252) * 100
    df['rvol_20d'] = df['ret_1d'].rolling(20).std() * np.sqrt(252) * 100
    df['rvol_5d'] = df['ret_1d'].rolling(5).std() * np.sqrt(252) * 100

    # Moving averages
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    # Trend flags
    df['above_sma20'] = (df['close'] > df['sma_20']).astype(int)
    df['above_sma50'] = (df['close'] > df['sma_50']).astype(int)
    df['above_sma200'] = (df['close'] > df['sma_200']).astype(int)
    df['sma20_above_sma50'] = (df['sma_20'] > df['sma_50']).astype(int)

    # Prior day absolute return
    df['abs_ret_1d'] = df['ret_1d'].abs() * 100  # in %

    # Range compression: 20-day high/low range as % of price
    df['range_20d'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close'] * 100

    # Consecutive up/down days
    df['up_day'] = (df['ret_1d'] > 0).astype(int)
    consec = []
    streak = 0
    for _, row in df.iterrows():
        if pd.isna(row['up_day']):
            consec.append(0)
            continue
        if row['up_day'] == 1:
            streak = max(streak, 0) + 1
        else:
            streak = min(streak, 0) - 1
        consec.append(streak)
    df['consec_days'] = consec

    # Distance from 52-week (252-day) high
    df['high_252'] = df['high'].rolling(252, min_periods=50).max()
    df['dist_from_high'] = (df['close'] - df['high_252']) / df['high_252'] * 100

    # Volume regime
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20']

    # Shift indicators by 1 day so they represent "known at market open"
    # Only shift the derived indicator columns, not raw price data
    indicator_cols = [
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
        'rvol_10d', 'rvol_20d', 'rvol_5d',
        'above_sma20', 'above_sma50', 'above_sma200', 'sma20_above_sma50',
        'abs_ret_1d', 'range_20d', 'consec_days',
        'dist_from_high', 'vol_ratio',
    ]
    for c in indicator_cols:
        if c in df.columns:
            df[f'prev_{c}'] = df[c].shift(1)

    return df


def run_regime_analysis():
    """Test regime filters on each strategy."""

    # Load trades
    trades_path = os.path.join(OUTPUT_DIR, "options_combined_optimal.csv")
    print(f"Loading trades from {trades_path}")
    trades = pd.read_csv(trades_path)
    ok = trades[trades['status'] == 'ok'].copy()
    ok['trade_date'] = pd.to_datetime(ok['entry_time'], utc=True).dt.date
    print(f"  {len(ok)} OK trades")

    # Load daily bars and compute features
    print("\nLoading daily bars...")
    bars = load_daily_bars()
    print("\nComputing regime features...")
    regime = compute_regime_features(bars)
    regime_dates = dict(zip(regime['date'], range(len(regime))))

    # Merge regime features onto trades via date
    # Ensure date types match
    regime['date'] = pd.to_datetime(regime['date']).dt.date
    regime_lookup = {}
    for _, row in regime.iterrows():
        regime_lookup[row['date']] = row.to_dict()

    print(f"  Regime features for {len(regime_lookup)} dates")
    # Check match rate
    matched = sum(1 for _, t in ok.iterrows() if t['trade_date'] in regime_lookup)
    print(f"  Trades matched to regime: {matched}/{len(ok)}")

    # Define regime filters to test
    # Each filter: (name, column, condition_fn, description)
    # condition_fn takes the regime row and returns True/False
    import math

    def safe_get(r, key, default=0):
        """Get a value from dict, returning default if missing or NaN."""
        v = r.get(key, default)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return v

    filters = [
        # Volatility regimes
        ("RVol20 < 15",    lambda r: safe_get(r, 'prev_rvol_20d', 99) < 15,   "Low vol regime"),
        ("RVol20 15-25",   lambda r: 15 <= safe_get(r, 'prev_rvol_20d', 0) < 25, "Medium vol"),
        ("RVol20 > 25",    lambda r: safe_get(r, 'prev_rvol_20d', 0) >= 25,  "High vol regime"),
        ("RVol20 > 20",    lambda r: safe_get(r, 'prev_rvol_20d', 0) >= 20,  "Elevated vol"),
        ("RVol20 < 20",    lambda r: safe_get(r, 'prev_rvol_20d', 99) < 20,  "Calm vol"),
        ("RVol5 > 25",     lambda r: safe_get(r, 'prev_rvol_5d', 0) >= 25,   "Short-term vol spike"),
        ("RVol5 < 15",     lambda r: safe_get(r, 'prev_rvol_5d', 99) < 15,   "Short-term calm"),

        # Trend
        ("Above SMA20",    lambda r: safe_get(r, 'prev_above_sma20', 0) == 1, "Uptrend (20d)"),
        ("Below SMA20",    lambda r: safe_get(r, 'prev_above_sma20', 1) == 0, "Downtrend (20d)"),
        ("Above SMA50",    lambda r: safe_get(r, 'prev_above_sma50', 0) == 1, "Uptrend (50d)"),
        ("Below SMA50",    lambda r: safe_get(r, 'prev_above_sma50', 1) == 0, "Downtrend (50d)"),
        ("Above SMA200",   lambda r: safe_get(r, 'prev_above_sma200', 0) == 1, "Bull market"),
        ("Below SMA200",   lambda r: safe_get(r, 'prev_above_sma200', 1) == 0, "Bear market"),
        ("SMA20 > SMA50",  lambda r: safe_get(r, 'prev_sma20_above_sma50', 0) == 1, "Golden cross"),
        ("SMA20 < SMA50",  lambda r: safe_get(r, 'prev_sma20_above_sma50', 1) == 0, "Death cross"),

        # Prior day move
        ("PriorDay > +1%", lambda r: safe_get(r, 'prev_ret_1d', 0) > 0.01,  "Big up day prior"),
        ("PriorDay < -1%", lambda r: safe_get(r, 'prev_ret_1d', 0) < -0.01, "Big down day prior"),
        ("PriorDay > +0.5%", lambda r: safe_get(r, 'prev_ret_1d', 0) > 0.005, "Up day prior"),
        ("PriorDay < -0.5%", lambda r: safe_get(r, 'prev_ret_1d', 0) < -0.005, "Down day prior"),
        ("|PriorDay| > 1%", lambda r: abs(safe_get(r, 'prev_ret_1d', 0)) > 0.01, "Big move prior"),
        ("|PriorDay| < 0.5%", lambda r: abs(safe_get(r, 'prev_ret_1d', 0)) < 0.005, "Small move prior"),

        # Momentum
        ("5d Mom > +1%",   lambda r: safe_get(r, 'prev_ret_5d', 0) > 0.01,  "5-day uptrend"),
        ("5d Mom < -1%",   lambda r: safe_get(r, 'prev_ret_5d', 0) < -0.01, "5-day downtrend"),
        ("10d Mom > +2%",  lambda r: safe_get(r, 'prev_ret_10d', 0) > 0.02, "10-day rally"),
        ("10d Mom < -2%",  lambda r: safe_get(r, 'prev_ret_10d', 0) < -0.02, "10-day selloff"),
        ("20d Mom > +3%",  lambda r: safe_get(r, 'prev_ret_20d', 0) > 0.03, "Monthly rally"),
        ("20d Mom < -3%",  lambda r: safe_get(r, 'prev_ret_20d', 0) < -0.03, "Monthly selloff"),

        # Consecutive days
        ("3+ Up Days",     lambda r: safe_get(r, 'prev_consec_days', 0) >= 3,  "Extended up streak"),
        ("3+ Down Days",   lambda r: safe_get(r, 'prev_consec_days', 0) <= -3, "Extended down streak"),

        # Range
        ("Range20 > 8%",   lambda r: safe_get(r, 'prev_range_20d', 0) > 8,   "Wide range"),
        ("Range20 < 5%",   lambda r: safe_get(r, 'prev_range_20d', 99) < 5,  "Tight range"),

        # Distance from high
        ("Within 2% of High", lambda r: safe_get(r, 'prev_dist_from_high', -99) > -2, "Near highs"),
        ("> 5% from High",    lambda r: safe_get(r, 'prev_dist_from_high', 0) < -5,  "Correction"),
        ("> 10% from High",   lambda r: safe_get(r, 'prev_dist_from_high', 0) < -10, "Deep correction"),

        # Volume
        ("High Volume Day",  lambda r: safe_get(r, 'prev_vol_ratio', 0) > 1.3, "Above avg volume"),
        ("Low Volume Day",   lambda r: safe_get(r, 'prev_vol_ratio', 99) < 0.8, "Below avg volume"),
    ]

    # Tag each trade with regime
    trade_regimes = []
    for _, trade in ok.iterrows():
        td = trade['trade_date']
        regime_row = regime_lookup.get(td, {})
        trade_regimes.append(regime_row)

    ok = ok.reset_index(drop=True)

    rules = sorted(ok['rule'].unique())

    # ── Test each filter on each rule ──────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"REGIME FILTER ANALYSIS")
    print(f"{'='*140}")

    results = []

    for rule in rules:
        sn = SHORT_NAMES.get(rule, rule[:12])
        rule_mask = ok['rule'] == rule
        rule_trades = ok[rule_mask]
        base_pnl = rule_trades['pnl'].sum()
        base_n = len(rule_trades)
        base_wr = (rule_trades['pnl'] > 0).mean() * 100
        base_avg = base_pnl / base_n if base_n > 0 else 0

        print(f"\n{'─'*140}")
        print(f"  {sn}  |  Base: {base_n} trades  ${base_pnl:>+11,.0f}  WR {base_wr:.0f}%  Avg ${base_avg:>+,.0f}")
        print(f"  {'Filter':<22} {'Trades':>7} {'P&L':>12} {'WR':>6} {'Avg':>9} {'vs Base':>12} {'Notes'}")
        print(f"  {'─'*130}")

        rule_results = []
        for fname, ffunc, fdesc in filters:
            # Apply filter
            pass_mask = []
            for i in rule_trades.index:
                regime_row = trade_regimes[i] if i < len(trade_regimes) else {}
                try:
                    pass_mask.append(ffunc(regime_row))
                except:
                    pass_mask.append(False)

            filtered = rule_trades[pass_mask]
            if len(filtered) < 5:  # Skip if too few trades
                continue

            f_pnl = filtered['pnl'].sum()
            f_n = len(filtered)
            f_wr = (filtered['pnl'] > 0).mean() * 100
            f_avg = f_pnl / f_n if f_n > 0 else 0

            # Improvement: compare avg P&L per trade (fairer than total since count differs)
            avg_improvement = f_avg - base_avg
            pct_kept = f_n / base_n * 100

            # Flag notable improvements
            flag = ""
            if f_avg > base_avg * 1.5 and f_n >= 10:
                flag = "*** STRONG"
            elif f_avg > base_avg * 1.2 and f_n >= 10:
                flag = "** GOOD"

            rule_results.append({
                'filter': fname,
                'trades': f_n,
                'pnl': f_pnl,
                'wr': f_wr,
                'avg': f_avg,
                'improvement': avg_improvement,
                'pct_kept': pct_kept,
                'flag': flag,
            })

        # Sort by avg P&L improvement
        rule_results.sort(key=lambda x: x['avg'], reverse=True)

        for rr in rule_results[:20]:  # Top 20
            flag_str = f"  {rr['flag']}" if rr['flag'] else ""
            print(f"  {rr['filter']:<22} {rr['trades']:>7} ${rr['pnl']:>+10,.0f} {rr['wr']:>5.0f}% ${rr['avg']:>+8,.0f} ${rr['improvement']:>+10,.0f}  ({rr['pct_kept']:.0f}% kept){flag_str}")

        results.append({
            'rule': rule,
            'short_name': sn,
            'base_pnl': base_pnl,
            'base_n': base_n,
            'base_avg': base_avg,
            'filters': rule_results,
        })

    # ── Summary: Best filter per rule ──────────────────────────────────────
    print(f"\n{'='*140}")
    print(f"BEST REGIME FILTER PER STRATEGY (by avg P&L improvement, min 10 trades)")
    print(f"{'='*140}")

    improved_total = 0
    base_total = 0

    for res in results:
        sn = res['short_name']
        base = res['base_avg']
        base_total += res['base_pnl']

        # Best filter with at least 10 trades
        viable = [r for r in res['filters'] if r['trades'] >= 10]
        if viable:
            best = viable[0]  # Already sorted by avg desc
            # Estimate P&L if we only took filtered trades
            est_pnl = best['pnl']
            improved_total += est_pnl
            print(f"  {sn:<18} Best: {best['filter']:<22} {best['trades']:>4} trades  ${best['pnl']:>+10,.0f}  avg ${best['avg']:>+8,.0f}  (base avg: ${base:>+8,.0f})  WR {best['wr']:.0f}%")
        else:
            improved_total += res['base_pnl']
            print(f"  {sn:<18} No viable filter (< 10 trades pass)")

    print(f"\n  Base total P&L:     ${base_total:>+11,.0f}")
    print(f"  Filtered total P&L: ${improved_total:>+11,.0f}")
    print(f"  Note: Filtered total uses fewer trades — compare avg P&L per trade for fair comparison")

    # Save detailed results
    rows_out = []
    for res in results:
        for fr in res['filters']:
            rows_out.append({
                'rule': res['rule'],
                'short_name': res['short_name'],
                'filter': fr['filter'],
                'trades': fr['trades'],
                'pnl': fr['pnl'],
                'wr': fr['wr'],
                'avg_pnl': fr['avg'],
                'base_avg_pnl': res['base_avg'],
                'avg_improvement': fr['improvement'],
                'pct_trades_kept': fr['pct_kept'],
            })
    out_df = pd.DataFrame(rows_out)
    out_path = os.path.join(OUTPUT_DIR, "regime_analysis.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    run_regime_analysis()
