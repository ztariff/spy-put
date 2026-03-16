#!/usr/bin/env python3
"""
Test different PT (profit target) levels with FULL EXIT (no trail).
All use 1-min bars, same V6 logic: exit 100% at PT or at 9:50 close.
"""
import os, json, csv, math
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
COMMISSION = 1.10
PT_LEVELS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

# ── Load data ──
csv_path = os.path.join(BASE, "options_2strat_v6_costs.csv")
cache_1m_path = os.path.join(BASE, "exit_sim_1min_cache.json")
recent_path = os.path.join(BASE, "recent_trades.json")

with open(cache_1m_path) as f:
    cache_1m = json.load(f)
recent_trades = []
if os.path.exists(recent_path):
    with open(recent_path) as f:
        recent_trades = json.load(f)

# Build recent bars lookup
recent_bars_lookup = {}
for rt in recent_trades:
    key = f"{rt.get('option_ticker','')}|{rt.get('trade_date','')}"
    if rt.get('bars_1m'):
        recent_bars_lookup[key] = rt['bars_1m']

# ── Helpers ──
def get_sorted_1m(option_ticker, trade_date):
    key = f"{option_ticker}|{trade_date}"
    bars_dict = cache_1m.get(key) or recent_bars_lookup.get(key)
    if not bars_dict:
        return None
    sorted_bars = []
    for t_str in sorted(bars_dict.keys()):
        hh, mm = int(t_str[:2]), int(t_str[3:5])
        m = (hh - 9) * 60 + (mm - 30)
        sorted_bars.append((m, t_str, bars_dict[t_str]))
    return sorted_bars

def sim_full_exit(sorted_bars, entry_px, n_contracts, pt_mult):
    """Full exit at PT or at 9:50 close. pt_mult is e.g. 1.50 for +50%."""
    pt_target = entry_px * pt_mult
    # Look for PT hit (skip 9:30 bar)
    pt_bar = None
    for m, t_str, bar in sorted_bars:
        if m < 1:
            continue
        if bar.get('h', bar.get('c', 0)) >= pt_target:
            if m <= 20:  # must be before or at 9:50
                pt_bar = (m, t_str, bar)
            break

    if pt_bar:
        m, t_str, bar = pt_bar
        exit_px = bar['c']
        pnl = (exit_px - entry_px) * n_contracts * 100 - n_contracts * COMMISSION
        return pnl, exit_px, 'pt', t_str

    # 9:50 exit
    b950 = None
    for m, t_str, bar in sorted_bars:
        if t_str[:5] == '09:50':
            b950 = (m, t_str, bar)
            break
    if not b950:
        for m, t_str, bar in sorted_bars:
            if m >= 20:
                b950 = (m, t_str, bar)
                break

    if b950:
        m, t_str, bar = b950
        exit_px = bar['c']
        pnl = (exit_px - entry_px) * n_contracts * 100 - n_contracts * COMMISSION
        return pnl, exit_px, 'time_950', t_str

    # Fallback
    if sorted_bars:
        _, t_str, bar = sorted_bars[-1]
        exit_px = bar['c']
        pnl = (exit_px - entry_px) * n_contracts * 100 - n_contracts * COMMISSION
        return pnl, exit_px, 'fallback', t_str

    return 0, entry_px, 'no_data', '??'

# ── Run simulation across all trades ──
print("=" * 80)
print("PT LEVEL COMPARISON — Full Exit (no trail) — 1-min bars")
print("=" * 80)

# Collect all trades
trades = []
with open(csv_path, newline='') as f:
    for r in csv.DictReader(f):
        trades.append({
            'date': r['trade_date'],
            'ticker': r['ticker'],
            'option_ticker': r['option_ticker'],
            'entry_px': float(r['option_entry_price']),
            'contracts': int(r['num_contracts']),
            'v6_recorded_pnl': float(r.get('v6_pnl') or r.get('pnl_after_costs', 0)),
        })

# Add recent trades
for rt in recent_trades:
    entry = rt.get('option_entry_price') or rt.get('entry_price')
    if not entry or entry <= 0:
        continue
    trades.append({
        'date': rt['trade_date'],
        'ticker': rt['ticker'],
        'option_ticker': rt.get('option_ticker', ''),
        'entry_px': entry,
        'contracts': rt.get('num_contracts', 0),
        'v6_recorded_pnl': None,
    })

# Results per PT level
results = {pt: {'daily_pnl': defaultdict(float), 'total': 0, 'pt_hits': 0, 'trades': 0, 'time_exits': 0}
           for pt in PT_LEVELS}

for t in trades:
    sorted_bars = get_sorted_1m(t['option_ticker'], t['date'])
    if not sorted_bars:
        continue

    for pt_pct in PT_LEVELS:
        pt_mult = 1.0 + pt_pct
        pnl, exit_px, reason, exit_time = sim_full_exit(
            sorted_bars, t['entry_px'], t['contracts'], pt_mult
        )
        r = results[pt_pct]
        r['daily_pnl'][t['date']] += pnl
        r['total'] += pnl
        r['trades'] += 1
        if reason == 'pt':
            r['pt_hits'] += 1
        elif reason == 'time_950':
            r['time_exits'] += 1

# ── Compute stats ──
print(f"\n{'PT Level':>10} {'Total P&L':>14} {'Sharpe':>8} {'Max DD':>12} {'Daily Std':>12} "
      f"{'Win%':>6} {'PT Hit%':>8} {'Avg Day':>12} {'Days':>6}")
print("-" * 100)

for pt_pct in PT_LEVELS:
    r = results[pt_pct]
    daily = sorted(r['daily_pnl'].items())
    daily_vals = [v for _, v in daily]

    if not daily_vals:
        continue

    total = sum(daily_vals)
    n_days = len(daily_vals)
    avg = total / n_days
    std = (sum((d - avg) ** 2 for d in daily_vals) / n_days) ** 0.5
    sharpe = (avg / std * math.sqrt(252)) if std > 0 else 0

    # Max drawdown
    cumul = 0
    peak = 0
    max_dd = 0
    for d in daily_vals:
        cumul += d
        peak = max(peak, cumul)
        max_dd = max(max_dd, peak - cumul)

    win_days = sum(1 for d in daily_vals if d >= 0)
    win_pct = win_days / n_days * 100
    pt_hit_pct = r['pt_hits'] / r['trades'] * 100 if r['trades'] > 0 else 0

    print(f"  +{pt_pct*100:.0f}%    ${total:>12,.0f}  {sharpe:>7.2f}  ${max_dd:>10,.0f}  ${std:>10,.0f}  "
          f"{win_pct:>5.0f}%  {pt_hit_pct:>6.1f}%  ${avg:>10,.0f}  {n_days:>5}")

# ── Also show V6 recorded for reference ──
v6_daily = defaultdict(float)
v6_count = 0
for t in trades:
    if t['v6_recorded_pnl'] is not None:
        v6_daily[t['date']] += t['v6_recorded_pnl']
        v6_count += 1

if v6_daily:
    daily_vals = [v for _, v in sorted(v6_daily.items())]
    total = sum(daily_vals)
    n_days = len(daily_vals)
    avg = total / n_days
    std = (sum((d - avg) ** 2 for d in daily_vals) / n_days) ** 0.5
    sharpe = (avg / std * math.sqrt(252)) if std > 0 else 0
    cumul = 0; peak = 0; max_dd = 0
    for d in daily_vals:
        cumul += d; peak = max(peak, cumul); max_dd = max(max_dd, peak - cumul)
    win_pct = sum(1 for d in daily_vals if d >= 0) / n_days * 100
    print(f"\n  V6 rec  ${total:>12,.0f}  {sharpe:>7.2f}  ${max_dd:>10,.0f}  ${std:>10,.0f}  "
          f"{win_pct:>5.0f}%     n/a   ${avg:>10,.0f}  {n_days:>5}")

# ── Monthly breakdown for each level ──
print("\n\n" + "=" * 80)
print("MONTHLY BREAKDOWN")
print("=" * 80)

months = sorted(set(d[:7] for pt in PT_LEVELS for d in results[pt]['daily_pnl']))

header = f"{'Month':>8}"
for pt in PT_LEVELS:
    header += f"  +{pt*100:.0f}%".rjust(12)
print(header)
print("-" * (8 + 12 * len(PT_LEVELS)))

for mo in months:
    row = f"{mo:>8}"
    for pt in PT_LEVELS:
        mo_pnl = sum(v for d, v in results[pt]['daily_pnl'].items() if d[:7] == mo)
        row += f"  ${mo_pnl:>9,.0f}"
    print(row)

# Totals
row = f"{'TOTAL':>8}"
for pt in PT_LEVELS:
    row += f"  ${results[pt]['total']:>9,.0f}"
print("-" * (8 + 12 * len(PT_LEVELS)))
print(row)
