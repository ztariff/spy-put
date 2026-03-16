#!/usr/bin/env python3
"""
Apply recommended delta changes and produce an updated backtest CSV.
====================================================================
Changes applied:
  - QQQ Short:  δ0.70 → δ0.55  (maximizes P&L: +$1.88M)
  - SPY Short:  δ0.70 → δ0.70  (no change — δ0.65 difference is noise)
  - 50Hi Weak:  δ0.50 → δ0.65  (best P&L/Sharpe balance)
  - GapLarge:   δ0.50 → δ0.50  (no change — not swept)
  - SPY Weak:   δ0.50 → δ0.50  (no change — not swept)
  - HighVolWR:  hibernated
  - QQQ Weak:   hibernated

Method:
  1. Read options_931filter_1min.csv as the base (has entry_time, exit_time, etc.)
  2. For QQQ Short: substitute δ0.55 rows from delta_sweep_results.csv
  3. For 50Hi Weak: substitute δ0.65 rows from delta_sweep_results.csv
     - 2 dates missing from sweep → keep original δ0.50 data as fallback
  4. All other rows kept as-is from 1min CSV
  5. Output: options_updated_deltas.csv

Inputs:
  output/options_931filter_1min.csv
  output/delta_sweep_results.csv

Output:
  output/options_updated_deltas.csv
"""

import os
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR

BASE_CSV   = os.path.join(OUTPUT_DIR, "options_931filter_1min.csv")
SWEEP_CSV  = os.path.join(OUTPUT_DIR, "delta_sweep_results.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "options_updated_deltas.csv")

# Strategies to substitute and their new target deltas
SUBSTITUTIONS = {
    'PriorDayStrong_AboveOR_QQQ_short': 0.55,   # QQQ Short
    'PriorDayWeak_50Hi_SPY_filtered':   0.65,   # 50Hi Weak
}

# Human-readable names for summary
NAME_MAP = {
    'GapLarge_First30min_SPY':          'GapLarge',
    'HighVolWR_30min_SPY_filtered':     'HighVolWR',
    'PriorDayStrong_AboveOR_QQQ_short': 'QQQ Short',
    'PriorDayStrong_AboveOR_SPY_short': 'SPY Short',
    'PriorDayWeak_30min_QQQ':           'QQQ Weak',
    'PriorDayWeak_30min_SPY_filtered':  'SPY Weak',
    'PriorDayWeak_50Hi_SPY_filtered':   '50Hi Weak',
}


def main():
    # ── Load base 1-min CSV ──────────────────────────────────────────────
    print(f"Loading base: {BASE_CSV}")
    base = pd.read_csv(BASE_CSV)
    base['trade_date'] = base['trade_date'].astype(str).str[:10]
    print(f"  {len(base)} trades loaded")

    # ── Load delta sweep results ─────────────────────────────────────────
    print(f"Loading sweep: {SWEEP_CSV}")
    sweep = pd.read_csv(SWEEP_CSV)
    sweep['trade_date'] = sweep['trade_date'].astype(str).str[:10]
    print(f"  {len(sweep)} sweep rows loaded")

    # Rename sweep columns to match base CSV column names
    sweep = sweep.rename(columns={
        'entry_price': 'option_entry_price',
        'exit_price':  'option_exit_price',
        'contracts':   'num_contracts',
        'premium':     'premium_paid',
    })

    # ── Build output row-by-row ──────────────────────────────────────────
    # Index the base CSV by rule + trade_date for quick lookups
    base_idx = base.set_index(['rule', 'trade_date'])

    output_rows = []
    substituted = 0
    fallback    = 0
    kept        = 0

    for rule, new_delta in SUBSTITUTIONS.items():
        # Extract sweep rows for this rule at the new delta
        sweep_sub = sweep[
            (sweep['rule'] == rule) &
            (sweep['target_delta'] == new_delta)
        ].copy()

        sweep_dates = set(sweep_sub['trade_date'].tolist())

        # Get all base rows for this rule
        base_sub = base[base['rule'] == rule].copy()

        for _, brow in base_sub.iterrows():
            td = brow['trade_date']

            if td in sweep_dates:
                # Use sweep data — merge with base to get entry_time, exit_time, etc.
                srow = sweep_sub[sweep_sub['trade_date'] == td].iloc[0]

                new_row = brow.copy()
                # Update with sweep values
                new_row['option_ticker']       = srow['option_ticker']
                new_row['strike']              = srow['strike']
                new_row['target_delta']        = srow['target_delta']
                new_row['option_entry_price']  = srow['option_entry_price']
                new_row['option_exit_price']   = srow['option_exit_price']
                new_row['num_contracts']       = int(srow['num_contracts'])
                new_row['premium_paid']        = srow['premium_paid']
                new_row['pnl']                 = srow['pnl']
                new_row['pnl_pct']             = srow['pnl_pct']
                # entry_time, exit_time, exit_reason, budget, direction, ticker,
                # expiry_date, option_type — all kept from base row (unchanged)
                output_rows.append(new_row)
                substituted += 1
            else:
                # Missing from sweep — keep original base row as fallback
                print(f"  [FALLBACK] {NAME_MAP.get(rule, rule)} {td}: "
                      f"no δ{new_delta} data — keeping original δ{brow['target_delta']}")
                output_rows.append(brow)
                fallback += 1

    # ── Append all non-substituted rules unchanged ───────────────────────
    non_sub_rules = [r for r in base['rule'].unique() if r not in SUBSTITUTIONS]
    for rule in non_sub_rules:
        rule_rows = base[base['rule'] == rule]
        for _, row in rule_rows.iterrows():
            output_rows.append(row)
            kept += 1

    # ── Assemble final DataFrame ─────────────────────────────────────────
    out = pd.DataFrame(output_rows)
    out = out.sort_values(['trade_date', 'rule']).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    out['label'] = out['rule'].map(NAME_MAP)
    base['label'] = base['rule'].map(NAME_MAP)

    all_dates = pd.bdate_range(
        pd.to_datetime(out['trade_date']).min(),
        pd.to_datetime(out['trade_date']).max()
    )

    print(f"\n{'='*75}")
    print(f"DELTA CHANGE RESULTS  (1-min pricing)")
    print(f"{'='*75}")
    print(f"  Substituted: {substituted}  |  Fallbacks: {fallback}  |  Unchanged: {kept}")
    print()

    header = f"  {'Strategy':<14} {'δ old→new':>10}  {'N':>5}  {'Old WR':>7} {'New WR':>7}  {'Old P&L':>13} {'New P&L':>13}  {'Sharpe':>7}"
    print(header)
    print(f"  {'-'*85}")

    active = ['QQQ Short', 'SPY Short', '50Hi Weak', 'GapLarge', 'SPY Weak']
    delta_changes = {
        'QQQ Short': ('0.70', '0.55'),
        'SPY Short': ('0.70', '0.70'),
        '50Hi Weak': ('0.50', '0.65'),
        'GapLarge':  ('0.50', '0.50'),
        'SPY Weak':  ('0.50', '0.50'),
    }

    for strat in active:
        bo = base[base['label'] == strat]
        bn = out[out['label'] == strat]
        if len(bo) == 0:
            continue

        old_wr  = (bo['pnl'] > 0).mean() * 100
        new_wr  = (bn['pnl'] > 0).mean() * 100
        old_pnl = bo['pnl'].sum()
        new_pnl = bn['pnl'].sum()

        # Sharpe (new)
        daily = bn.groupby('trade_date')['pnl'].sum()
        daily_full = daily.reindex(
            pd.to_datetime(all_dates).strftime('%Y-%m-%d'),
            fill_value=0
        )
        mu  = daily_full.mean()
        sig = daily_full.std()
        sharpe = (mu / sig * np.sqrt(252)) if sig > 0 else 0

        old_d, new_d = delta_changes.get(strat, ('—', '—'))
        arrow = '→' if old_d != new_d else '  '
        delta_str = f"{old_d}{arrow}{new_d}" if old_d != new_d else f"  {old_d}  "

        print(f"  {strat:<14} {delta_str:>10}  {len(bn):>5}  {old_wr:>6.1f}% {new_wr:>6.1f}%  "
              f"${old_pnl:>12,.0f} ${new_pnl:>12,.0f}  {sharpe:>7.2f}")

    print(f"  {'-'*85}")

    # Active-only totals
    active_old = base[base['label'].isin(active)]
    active_new = out[out['label'].isin(active)]
    print(f"  {'ACTIVE TOTAL':<14} {'':>10}  {len(active_new):>5}  "
          f"{(active_old['pnl']>0).mean()*100:>6.1f}% {(active_new['pnl']>0).mean()*100:>6.1f}%  "
          f"${active_old['pnl'].sum():>12,.0f} ${active_new['pnl'].sum():>12,.0f}")

    print()

    # Portfolio-level stats (active strategies only)
    active_out   = out[out['label'].isin(active)]
    daily_all    = active_out.groupby('trade_date')['pnl'].sum()
    daily_full   = daily_all.reindex(
        pd.to_datetime(all_dates).strftime('%Y-%m-%d'), fill_value=0
    )
    cum          = daily_full.cumsum()
    maxdd        = (cum - cum.cummax()).min()
    win_days     = (daily_all > 0).sum()
    total_days   = len(daily_all)
    mu           = daily_full.mean()
    sig          = daily_full.std()
    sharpe_port  = (mu / sig * np.sqrt(252)) if sig > 0 else 0
    calmar       = (active_out['pnl'].sum() / abs(maxdd)) if maxdd != 0 else 0

    print(f"  Portfolio (active 5 strategies):")
    print(f"    Total P&L:    ${active_out['pnl'].sum():>12,.0f}")
    print(f"    Win Day Rate: {win_days}/{total_days} ({win_days/total_days*100:.0f}%)")
    print(f"    Avg/day:      ${daily_all.mean():>10,.0f}")
    print(f"    Max DrawDown: ${maxdd:>10,.0f}")
    print(f"    Sharpe:       {sharpe_port:.2f}")
    print(f"    Calmar:       {calmar:.2f}")

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
